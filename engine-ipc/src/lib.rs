//! Core Engine IPC Daemon
//!
//! Manages the lifecycle of memory-mapped models, compute backend selection,
//! and inference request handling via Unix sockets / named pipes.

use anyhow::{anyhow, Context, Result};
use brain_pack::{BrainPack, TensorView};
#[cfg(feature = "cpu")]
use compute_cpu::CpuBackend;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock, RwLock};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
#[cfg(unix)]
use tokio::net::{UnixListener, UnixStream, TcpListener, TcpStream};
#[cfg(windows)]
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;
use stream_cache::StreamCache;
use tracing::{debug, error, info, warn};
use weight_handle::VolatileWeights;

#[cfg(windows)]
use tokio::net::windows::named_pipe::NamedPipeServer;

// ============================================================================
// Compute Backend Abstraction
// ============================================================================

pub trait ComputeBackend: Send + Sync {
    fn name(&self) -> &'static str;
    fn gemm_f16(&self, a: &[half::f16], b: &[half::f16], c: &mut [half::f16], m: usize, n: usize, k: usize) -> Result<()>;
    fn batched_gemm_f16(&self, a: &[half::f16], b: &[half::f16], c: &mut [half::f16], batch: usize, seq_len: usize, head_dim: usize) -> Result<()>;
    fn attention_f16(&self, q: &[half::f16], k: &[half::f16], v: &[half::f16], out: &mut [half::f16], batch: usize, heads: usize, seq_len: usize, head_dim: usize) -> Result<()>;
    fn silu_f16(&self, x: &mut [half::f16]) -> Result<()>;
    fn gelu_f16(&self, x: &mut [half::f16]) -> Result<()>;
    fn relu_f16(&self, x: &mut [half::f16]) -> Result<()>;
    fn add_bias_f16(&self, x: &mut [half::f16], bias: &[half::f16], bias_size: usize) -> Result<()>;
    fn scale_f16(&self, x: &mut [half::f16], scale: f32) -> Result<()>;
    fn add_f16(&self, a: &mut [half::f16], b: &[half::f16]) -> Result<()>;
    fn rms_norm_f16(&self, x: &mut [half::f16], weight: &[half::f16], out: &mut [half::f16], batch_seq: usize, hidden_dim: usize, eps: f32) -> Result<()>;
    fn dequantize_q4km_f16(&self, q_weights: &[u8], scales: &[half::f16], out: &mut [half::f16], num_elements: usize) -> Result<()>;
    fn gemm_q4km_f16(&self, a_q: &[u8], a_scales: &[half::f16], b: &[half::f16], c: &mut [half::f16], m: usize, n: usize, k: usize) -> Result<()>;
    fn rope_f16(&self, q: &mut [half::f16], k: &mut [half::f16], cos: &[half::f16], sin: &[half::f16], batch: usize, heads: usize, seq_len: usize, head_dim: usize) -> Result<()>;
    fn sample_topk_topp(&self, logits: &[half::f16], next_token: &mut [i32], batch: usize, vocab_size: usize, top_k: usize, top_p: f32, temperature: f32, seed: u64) -> Result<()>;
    fn sample_argmax(&self, logits: &[half::f16], next_token: &mut [i32], batch: usize, vocab_size: usize) -> Result<()>;
    fn synchronize(&self) -> Result<()>;

    // -----------------------------------------------------------------------
    // Volatile-pointer kernel extensions (sharded streaming path).
    //
    // These accept a [`VolatileWeights`] handle whose backing bytes live in
    // a memory-mapped shard region that may be evicted the instant the lease
    // is dropped. The default impls forward to the existing slice-based
    // kernels so no backend is forced to specialize — but GPU backends that
    // want to DMA directly from the host-mapped pages can override.
    // -----------------------------------------------------------------------

    /// GEMM whose B matrix is supplied via a volatile (LRU-managed) lease.
    /// The backend reads `b` and releases the hardware address by letting
    /// the caller drop the lease once this returns.
    fn gemm_f16_lease(
        &self,
        a: &[half::f16],
        b: &dyn VolatileWeights,
        c: &mut [half::f16],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        self.gemm_f16(a, b.as_f16(), c, m, n, k)
    }

    /// RMSNorm whose `weight` is supplied via a volatile lease.
    fn rms_norm_f16_lease(
        &self,
        x: &mut [half::f16],
        weight: &dyn VolatileWeights,
        out: &mut [half::f16],
        batch_seq: usize,
        hidden_dim: usize,
        eps: f32,
    ) -> Result<()> {
        self.rms_norm_f16(x, weight.as_f16(), out, batch_seq, hidden_dim, eps)
    }

    /// Attention whose K and V matrices are volatile. Default impl forwards
    /// to [`attention_f16`], resolving the leases into slices.
    fn attention_f16_lease(
        &self,
        q: &[half::f16],
        k: &dyn VolatileWeights,
        v: &dyn VolatileWeights,
        out: &mut [half::f16],
        batch: usize,
        heads: usize,
        seq_len: usize,
        head_dim: usize,
    ) -> Result<()> {
        self.attention_f16(q, k.as_f16(), v.as_f16(), out, batch, heads, seq_len, head_dim)
    }
}

// CPU Backend Implementation
impl ComputeBackend for CpuBackend {
    fn name(&self) -> &'static str { "cpu" }

    fn gemm_f16(&self, a: &[half::f16], b: &[half::f16], c: &mut [half::f16], m: usize, n: usize, k: usize) -> Result<()> {
        self.gemm_f16(a, b, c, m, n, k)
    }

    fn batched_gemm_f16(&self, a: &[half::f16], b: &[half::f16], c: &mut [half::f16], batch: usize, seq_len: usize, head_dim: usize) -> Result<()> {
        self.batched_gemm_f16(a, b, c, batch, seq_len, head_dim)
    }

    fn attention_f16(&self, q: &[half::f16], k: &[half::f16], v: &[half::f16], out: &mut [half::f16], batch: usize, heads: usize, seq_len: usize, head_dim: usize) -> Result<()> {
        self.attention_f16(q, k, v, out, batch, heads, seq_len, head_dim)
    }

    fn silu_f16(&self, x: &mut [half::f16]) -> Result<()> { self.silu_f16(x) }
    fn gelu_f16(&self, x: &mut [half::f16]) -> Result<()> { self.gelu_f16(x) }
    fn relu_f16(&self, x: &mut [half::f16]) -> Result<()> { self.relu_f16(x) }
    fn add_bias_f16(&self, x: &mut [half::f16], bias: &[half::f16], bias_size: usize) -> Result<()> { self.add_bias_f16(x, bias, bias_size) }
    fn scale_f16(&self, x: &mut [half::f16], scale: f32) -> Result<()> { self.scale_f16(x, scale) }
    fn add_f16(&self, a: &mut [half::f16], b: &[half::f16]) -> Result<()> { self.add_f16(a, b) }
    fn rms_norm_f16(&self, x: &mut [half::f16], weight: &[half::f16], out: &mut [half::f16], batch_seq: usize, hidden_dim: usize, eps: f32) -> Result<()> {
        self.rms_norm_f16(x, weight, out, batch_seq, hidden_dim, eps)
    }
    fn dequantize_q4km_f16(&self, q_weights: &[u8], scales: &[half::f16], out: &mut [half::f16], num_elements: usize) -> Result<()> {
        self.dequantize_q4km_f16(q_weights, scales, out, num_elements)
    }
    fn gemm_q4km_f16(&self, a_q: &[u8], a_scales: &[half::f16], b: &[half::f16], c: &mut [half::f16], m: usize, n: usize, k: usize) -> Result<()> {
        self.gemm_q4km_f16(a_q, a_scales, b, c, m, n, k)
    }
    fn rope_f16(&self, q: &mut [half::f16], k: &mut [half::f16], cos: &[half::f16], sin: &[half::f16], batch: usize, heads: usize, seq_len: usize, head_dim: usize) -> Result<()> {
        self.rope_f16(q, k, cos, sin, batch, heads, seq_len, head_dim)
    }
    fn sample_topk_topp(&self, logits: &[half::f16], next_token: &mut [i32], batch: usize, vocab_size: usize, top_k: usize, top_p: f32, temperature: f32, seed: u64) -> Result<()> {
        self.sample_topk_topp(logits, next_token, batch, vocab_size, top_k, top_p, temperature, seed)
    }
    fn sample_argmax(&self, logits: &[half::f16], next_token: &mut [i32], batch: usize, vocab_size: usize) -> Result<()> {
        self.sample_argmax(logits, next_token, batch, vocab_size)
    }
    fn synchronize(&self) -> Result<()> { Ok(()) }
}

// Speculative Decoding Module
pub mod speculative;
pub use speculative::{
    SpeculativeConfig, NgramIndexConfig, VerifierConfig, SpeculativeMetrics,
    Speculator, DraftToken, SpeculativeResult,
    NgramIndex, ThreadLocalNgramIndex,
    SpeculativeVerifier, VerificationResult, TokenVerification, KvCacheMaskAdjuster,
};

// Conditional GPU backends
#[cfg(feature = "cuda")]
mod cuda_backend {
    use super::*;
    use compute_cuda::CudaBackend;

    impl ComputeBackend for CudaBackend {
        fn name(&self) -> &'static str { "cuda" }
        fn gemm_f16(&self, a: &[half::f16], b: &[half::f16], c: &mut [half::f16], m: usize, n: usize, k: usize) -> Result<()> {
            let a_dev = self.h2d_copy(a)?;
            let b_dev = self.h2d_copy(b)?;
            let mut c_dev = self.alloc_tensor::<half::f16>(m * n)?;
            self.gemm_f16(&a_dev, &b_dev, &mut c_dev, m, n, k)?;
            let c_host = self.d2h_copy(&c_dev)?;
            c.copy_from_slice(&c_host);
            Ok(())
        }
        // ... implement other methods with H2D/D2H copies
        fn batched_gemm_f16(&self, _a: &[half::f16], _b: &[half::f16], _c: &mut [half::f16], _batch: usize, _seq_len: usize, _head_dim: usize) -> Result<()> { Ok(()) }
        fn attention_f16(&self, _q: &[half::f16], _k: &[half::f16], _v: &[half::f16], _out: &mut [half::f16], _batch: usize, _heads: usize, _seq_len: usize, _head_dim: usize) -> Result<()> { Ok(()) }
        fn silu_f16(&self, _x: &mut [half::f16]) -> Result<()> { Ok(()) }
        fn gelu_f16(&self, _x: &mut [half::f16]) -> Result<()> { Ok(()) }
        fn relu_f16(&self, _x: &mut [half::f16]) -> Result<()> { Ok(()) }
        fn add_bias_f16(&self, _x: &mut [half::f16], _bias: &[half::f16], _bias_size: usize) -> Result<()> { Ok(()) }
        fn scale_f16(&self, _x: &mut [half::f16], _scale: f32) -> Result<()> { Ok(()) }
        fn add_f16(&self, _a: &mut [half::f16], _b: &[half::f16]) -> Result<()> { Ok(()) }
        fn rms_norm_f16(&self, _x: &mut [half::f16], _weight: &[half::f16], _out: &mut [half::f16], _batch_seq: usize, _hidden_dim: usize, _eps: f32) -> Result<()> { Ok(()) }
        fn dequantize_q4km_f16(&self, _q: &[u8], _scales: &[half::f16], _out: &mut [half::f16], _n: usize) -> Result<()> { Ok(()) }
        fn gemm_q4km_f16(&self, _a_q: &[u8], _a_scales: &[half::f16], _b: &[half::f16], _c: &mut [half::f16], _m: usize, _n: usize, _k: usize) -> Result<()> { Ok(()) }
        fn rope_f16(&self, _q: &mut [half::f16], _k: &mut [half::f16], _cos: &[half::f16], _sin: &[half::f16], _b: usize, _h: usize, _s: usize, _d: usize) -> Result<()> { Ok(()) }
        fn sample_topk_topp(&self, _logits: &[half::f16], _next: &mut [i32], _b: usize, _v: usize, _tk: usize, _tp: f32, _temp: f32, _seed: u64) -> Result<()> { Ok(()) }
        fn sample_argmax(&self, _logits: &[half::f16], _next: &mut [i32], _b: usize, _v: usize) -> Result<()> { Ok(()) }
        fn synchronize(&self) -> Result<()> { self.synchronize() }
    }
}

// ============================================================================
// Model Instance
// ============================================================================

/// Loaded model with memory-mapped weights
pub struct ModelInstance {
    pub id: String,
    pub path: PathBuf,
    pub pack: BrainPack,
    #[cfg(all(target_os = "linux", not(target_env = "musl")))]
    mmap: Option<mem_posix::PosixMappedBrain>,
    #[cfg(windows)]
    mmap: Option<mem_windows::WindowsMappedBrain>,
    /// If this model is the base fragment of a sharded pack, holds the
    /// (directory, name-prefix) the StreamCache needs to open the rest
    /// of the streaming fragments. None for monolithic `.brain` files.
    shard_dir: Option<(PathBuf, String)>,
    backend: Arc<dyn ComputeBackend>,
    tensors: HashMap<String, TensorView<'static>>, // Cached tensor views
}

impl ModelInstance {
    pub fn load(path: impl AsRef<Path>, backend: Arc<dyn ComputeBackend>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let pack = BrainPack::read(&path)?;

        #[cfg(all(target_os = "linux", not(target_env = "musl")))]
        let mmap = Some(mem_posix::PosixMappedBrain::open(&path)?);

        #[cfg(windows)]
        let mmap = Some(mem_windows::WindowsMappedBrain::open(&path)?);

        let id = uuid::Uuid::new_v4().to_string();

        // Detect sharded siblings: if a `00.brain` base fragment lives in
        // the same dir and the loaded file is itself a `00.brain` (or has
        // matching `NN.brain` siblings), treat as sharded so the engine
        // binds a StreamCache for the streaming fragments.
        let shard_dir = detect_shard_dir(&path);

        info!("Loaded model '{}' ({}) from {}", pack.manifest.model.name, id, path.display());

        Ok(Self {
            id,
            path,
            pack,
            mmap,
            shard_dir,
            backend,
            tensors: HashMap::new(),
        })
    }

    pub fn get_tensor(&mut self, name: &str) -> Result<&TensorView> {
        if !self.tensors.contains_key(name) {
            let view = self.pack.get_tensor(name)?;
            // Note: In real implementation, we'd need to handle lifetimes properly
            // This is a simplified version
            self.tensors.insert(name.to_string(), unsafe { std::mem::transmute(view) });
        }
        Ok(self.tensors.get(name).unwrap())
    }

    pub fn model_info(&self) -> &brain_pack::ModelInfo {
        &self.pack.manifest.model
    }

    pub fn tensor_names(&self) -> Vec<String> {
        self.pack.tensor_names().into_iter().map(|s| s.to_string()).collect()
    }

    pub fn backend_name(&self) -> &str {
        self.backend.name()
    }
}

// ============================================================================
// Inference Engine
// ============================================================================

/// High-level inference engine for transformer models
pub struct InferenceEngine {
    models: Arc<RwLock<HashMap<String, Arc<ModelInstance>>>>,
    default_backend: Arc<dyn ComputeBackend>,
    model_dir: PathBuf,
    start_time: std::time::Instant,
    /// Optional sharded streaming cache. Populated lazily when a sharded
    /// model directory (containing `00.brain` + streaming fragments) is
    /// loaded. None for monolithic `.brain` files.
    stream_cache: Arc<tokio::sync::Mutex<Option<Arc<StreamCache>>>>,
    /// Speculative decoding components
    ngram_index: Arc<NgramIndex>,
    speculator: OnceLock<Arc<tokio::sync::Mutex<Speculator>>>,
    verifier: OnceLock<Arc<SpeculativeVerifier>>,
    speculative_config: SpeculativeConfig,
}

impl InferenceEngine {
    pub fn new(model_dir: impl AsRef<Path>, backend: Arc<dyn ComputeBackend>) -> Result<Self> {
        let model_dir = model_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&model_dir)?;

        let speculative_config = SpeculativeConfig::default();
        let ngram_index = Arc::new(NgramIndex::new(NgramIndexConfig::default()));

        Ok(Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            default_backend: backend,
            model_dir,
            start_time: std::time::Instant::now(),
            stream_cache: Arc::new(tokio::sync::Mutex::new(None)),
            ngram_index,
            speculator: OnceLock::new(),
            verifier: OnceLock::new(),
            speculative_config,
        })
    }

    /// Initialize speculative decoding components (call after construction)
    fn init_speculative(&self) -> Result<()> {
        if self.speculator.get().is_some() {
            return Ok(()); // Already initialized
        }

        // Create speculator with the N-gram index
        let speculator = Speculator::new(self.ngram_index.clone(), self.speculative_config.clone());
        self.speculator.set(Arc::new(tokio::sync::Mutex::new(speculator))).ok();

        // Create verifier with model dimensions from first loaded model
        let models = self.models.read().unwrap();
        let (max_seq_len, num_layers, num_heads, head_dim) = if let Some(model) = models.values().next() {
            let info = model.model_info();
            // ModelInfo has: name, architecture, parameter_count, quantization, context_length, vocab_size
            // We don't have n_layers, n_heads, head_dim directly - use defaults
            (
                info.context_length as usize,
                32, // default layers
                32, // default heads
                128, // default head_dim
            )
        } else {
            (4096, 32, 32, 128) // Defaults
        };

        let verifier = SpeculativeVerifier::new(
            self.default_backend.clone(),
            crate::speculative::verifier::VerifierConfig::default(),
            max_seq_len,
            num_layers,
            num_heads,
            head_dim,
        );
        self.verifier.set(Arc::new(verifier)).ok();

        Ok(())
    }

    /// Get current speculative decoding configuration
    pub async fn speculative_config(&self) -> SpeculativeConfig {
        self.speculative_config.clone()
    }

    /// Update speculative decoding configuration
    pub async fn set_speculative_config(&mut self, config: SpeculativeConfig) -> Result<()> {
        config.validate()?;
        self.speculative_config = config.clone();

        // Re-initialize speculator with new config
        if let Some(speculator_lock) = self.speculator.get() {
            let mut speculator = speculator_lock.lock().await;
            // Create new speculator with updated config
            *speculator = Speculator::new(self.ngram_index.clone(), config);
        }

        // Also update verifier config if needed
        if let Some(verifier) = self.verifier.get() {
            // Verifier config would need update here if we expose it
        }

        Ok(())
    }

    /// Get current speculative decoding metrics
    pub async fn speculative_metrics(&self) -> Option<SpeculativeMetrics> {
        if let Some(speculator_lock) = self.speculator.get() {
            let speculator = speculator_lock.lock().await;
            let mut metrics = speculator.metrics();

            // Add verifier metrics if available
            if let Some(verifier) = self.verifier.get() {
                let v_metrics = verifier.verifier_metrics();
                metrics.acceptance_rate = v_metrics.acceptance_rate();
                metrics.avg_tokens_per_step = v_metrics.avg_tokens_per_step();
                metrics.verification_batches = v_metrics.verification_batches;
                metrics.rejections = v_metrics.total_rejected;
            }

            // Add index eviction count
            metrics.index_evictions = self.ngram_index.eviction_count();

            Some(metrics)
        } else {
            None
        }
    }

    /// Dummy constructor for speculator initialization (avoids recursion)
    fn new_dummy(model_dir: PathBuf, backend: Arc<dyn ComputeBackend>) -> Result<Self> {
        Ok(Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            default_backend: backend,
            model_dir,
            start_time: std::time::Instant::now(),
            stream_cache: Arc::new(tokio::sync::Mutex::new(None)),
            ngram_index: Arc::new(NgramIndex::default()),
            speculator: OnceLock::new(),
            verifier: OnceLock::new(),
            speculative_config: SpeculativeConfig::default(),
        })
    }

    /// Load a model from file. Async because sharded models trigger
    /// `StreamCache::open` which spawns a Tokio prefetch pool.
    pub async fn load_model(&self, filename: &str) -> Result<String> {
        let path = self.model_dir.join(filename);
        let model = ModelInstance::load(&path, self.default_backend.clone())?;
        let id = model.id.clone();

        // If the model directory contains sharded fragments (00.brain +
        // 01.brain..15.brain), bind a StreamCache over it. The cache
        // replaces the monolithic mmap path for tensor access during
        // evaluation.
        if model.shard_dir.is_some() {
            let (dir, name) = model.shard_dir.as_ref().unwrap().clone();
            match StreamCache::open(&dir, &name, None).await {
                Ok(cache) => {
                    info!(
                        "Bound StreamCache for sharded model '{}' ({} resident shards)",
                        id,
                        cache.stats().resident_shard_count
                    );
                    *self.stream_cache.lock().await = Some(cache);
                }
                Err(e) => {
                    warn!("StreamCache bind failed for '{}': {} (falling back to monolithic mmap)", id, e);
                }
            }
        }

        self.models.write().unwrap().insert(id.clone(), Arc::new(model));
        info!("Model loaded with ID: {}", id);
        Ok(id)
    }

    /// Get a loaded model by ID
    pub fn get_model(&self, model_id: &str) -> Option<Arc<ModelInstance>> {
        self.models.read().unwrap().get(model_id).cloned()
    }

    /// Get model info for a loaded model
    fn model_info(model: &ModelInstance) -> ModelInfo {
        ModelInfo {
            id: model.id.clone(),
            name: model.pack.manifest.model.name.clone(),
            architecture: model.pack.manifest.model.architecture.clone(),
            parameter_count: model.pack.manifest.model.parameter_count,
            quantization: format!("{:?}", model.pack.manifest.model.quantization),
            context_length: model.pack.manifest.model.context_length,
            backend: "unknown".to_string(), // Could store backend type in ModelInstance
        }
    }

    /// List all loaded models with full info
    pub fn list_models(&self) -> Vec<ModelInfo> {
        self.models.read().unwrap().values()
            .map(|m| Self::model_info(m))
            .collect()
    }

    /// Unload a model
    pub fn unload_model(&self, model_id: &str) -> Result<()> {
        self.models.write().unwrap().remove(model_id);
        info!("Model unloaded: {}", model_id);
        Ok(())
    }

    /// Snapshot the streaming-cache stats (resident bytes, evictions, NVMe
    /// throughput counters). Empty if no sharded model is bound.
    pub async fn streaming_stats(&self) -> Option<stream_cache::CacheStats> {
        let guard = self.stream_cache.lock().await;
        guard.as_ref().map(|c| c.stats())
    }

    /// Zero-copy tensor accessor. If a StreamCache is bound (sharded model),
    /// routes through the LRU; otherwise falls back to the monolithic mmap
    /// pack via a thin `VolatileWeights` shim. The closure receives the
    /// weights view and may run compute against it; the shard is pinned
    /// for the closure's lifetime so the pointer is stable.
    pub async fn with_tensor<F, R>(&self, model_id: &str, name: &str, f: F) -> Result<R>
    where
        F: FnOnce(&dyn VolatileWeights) -> R,
    {
        // Sharded path: hand off to the StreamCache.
        {
            let guard = self.stream_cache.lock().await;
            if let Some(cache) = guard.as_ref() {
                return cache.with_tensor(name, f).await;
            }
        }
        // Monolithic fallback: borrow the resident pack's tensor slice.
        let models = self.models.read().unwrap();
        let model = models.get(model_id)
            .ok_or_else(|| anyhow!("Model not found: {}", model_id))?;
        let view = model.pack.get_tensor(name)?;
        let slice = view.data;
        let shim = MonolithicWeights { data: slice };
        Ok(f(&shim))
    }

    /// Run inference with optional speculative decoding
    pub async fn generate_async(&self, request: GenerateRequest) -> Result<GenerateResponse> {
        // Get model reference without holding lock across await
        let model = {
            let models = self.models.read().unwrap();
            models.get(&request.model_id)
                .ok_or_else(|| anyhow!("Model not found: {}", request.model_id))?
                .clone()
        };

        // Initialize speculative decoding if enabled and not already done
        if self.speculative_config.enabled {
            let _ = self.init_speculative();
        }

        // Use speculative decoding if enabled and available
        if self.speculative_config.enabled {
            if let Some(speculator_lock) = self.speculator.get() {
                if let Some(verifier) = self.verifier.get() {
                    return self.generate_with_speculation_async(&model, request, speculator_lock.clone(), verifier.clone()).await;
                }
            }
        }

        // Fallback to regular generation
        self.generate_regular_async(&model, request).await
    }

    /// Regular generation without speculation (async fallback)
    async fn generate_regular_async(&self, _model: &ModelInstance, request: GenerateRequest) -> Result<GenerateResponse> {
        let mut tokens = request.prompt_tokens.clone();
        let mut generated = Vec::new();

        for _ in 0..request.max_tokens {
            let next_token = self.sample_token_for(&request.model_id, &tokens, request.temperature)?;
            tokens.push(next_token);
            generated.push(next_token);

            if request.stop_tokens.contains(&next_token) {
                break;
            }
        }

        let completion_tokens = generated.len();
        Ok(GenerateResponse {
            tokens: generated,
            finish_reason: "stop".to_string(),
            usage: Some(Usage {
                prompt_tokens: request.prompt_tokens.len(),
                completion_tokens,
                total_tokens: request.prompt_tokens.len() + completion_tokens,
            }),
        })
    }

    /// Generation with speculative decoding (async version)
    async fn generate_with_speculation_async(
        &self,
        model: &ModelInstance,
        request: GenerateRequest,
        speculator_lock: Arc<tokio::sync::Mutex<Speculator>>,
        verifier: Arc<SpeculativeVerifier>,
    ) -> Result<GenerateResponse> {
        let mut prompt_tokens: Vec<u32> = request.prompt_tokens.iter().map(|&t| t as u32).collect();
        let mut generated = Vec::new();
        let mut total_accepted = 0;
        let mut total_drafted = 0;

        for _ in 0..request.max_tokens {
            // Get draft tokens from speculator
            let spec_result = {
                let mut speculator = speculator_lock.lock().await;
                speculator.generate_draft(&prompt_tokens)
            };

            if !spec_result.should_verify || spec_result.draft_tokens.is_empty() {
                // Fallback to regular generation for this step
                let next_token = self.sample_token_for(&request.model_id, &prompt_tokens.iter().map(|&t| t as i32).collect::<Vec<_>>(), request.temperature)?;
                prompt_tokens.push(next_token as u32);
                generated.push(next_token);

                if request.stop_tokens.contains(&next_token) {
                    break;
                }
                continue;
            }

            // Convert draft tokens to verification format
            let draft_tokens: Vec<DraftToken> = spec_result.draft_tokens;

            // Verify draft tokens with batched forward pass
            let verification = verifier.verifier.verify_draft(
                model,
                &prompt_tokens,
                &draft_tokens,
                request.temperature,
                request.top_p,
                &request.stop_tokens.iter().map(|&t| t as u32).collect::<Vec<_>>(),
            ).await?;

            // Update N-gram index with accepted tokens
            for token in &verification.accepted_tokens {
                self.ngram_index.insert(&prompt_tokens);
                prompt_tokens.push(*token);
                generated.push(*token as i32);
                total_accepted += 1;

                if request.stop_tokens.contains(&(*token as i32)) {
                    break;
                }
            }
            total_drafted += draft_tokens.len();

            // If all draft tokens accepted, continue speculating
            if verification.continue_generation && verification.acceptance_count == draft_tokens.len() {
                continue;
            }

            // If we hit a stop token or rejection, break
            if !verification.continue_generation || verification.acceptance_count < draft_tokens.len() {
                // If rejected, sample from target for the next token
                if verification.acceptance_count < draft_tokens.len() {
                    let next_token = self.sample_token_for(&request.model_id, &prompt_tokens.iter().map(|&t| t as i32).collect::<Vec<_>>(), request.temperature)?;
                    prompt_tokens.push(next_token as u32);
                    generated.push(next_token);
                }
                break;
            }
        }

        let completion_tokens = generated.len();
        Ok(GenerateResponse {
            tokens: generated,
            finish_reason: "stop".to_string(),
            usage: Some(Usage {
                prompt_tokens: request.prompt_tokens.len(),
                completion_tokens,
                total_tokens: request.prompt_tokens.len() + completion_tokens,
            }),
        })
    }

    /// Run inference with optional speculative decoding (sync wrapper for backwards compatibility)
    pub fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse> {
        // For sync context, we can't use block_on inside an async runtime
        // This should only be called from non-async contexts (tests, CLI)
        // In async contexts, use generate_async()
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(self.generate_async(request))
        })
    }

    /// Stream tokens one-at-a-time via a tokio mpsc channel wrapped as a Stream.
    /// Each item is a `Result<(i32, &'static str)>` of (token_id, finish_reason);
    /// `finish_reason` is `"stop"` until the last item, then it's the actual reason.
    pub fn generate_stream(
        &self,
        request: GenerateRequest,
    ) -> Result<tokio_stream::wrappers::ReceiverStream<Result<(i32, String)>>> {
        // Validate model exists and acquire a snapshot of generation parameters
        let model_ref_id = request.model_id.clone();
        let stop_tokens = request.stop_tokens.clone();
        let max_tokens = request.max_tokens;
        let temperature = request.temperature;
        let top_p = request.top_p;

        // Check if model exists
        let model_exists = {
            let models = self.models.read().unwrap();
            models.contains_key(&model_ref_id)
        };
        if !model_exists {
            return Err(anyhow!("Model not found: {}", model_ref_id));
        }

        // Initialize speculative if enabled
        if self.speculative_config.enabled {
            let _ = self.init_speculative();
        }

        let (tx, rx) = tokio::sync::mpsc::channel(16);
        let engine = self.clone_handle_for_stream();
        // Prefetch driver: advance one shard ahead of the active layer so the
        // NVMe pages are warm by the time compute dereferences them. The
        // cache is None for monolithic models, in which case prefetch is a
        // no-op.
        let stream_cache = self.stream_cache.clone();

        tokio::spawn(async move {
            let mut tokens = request.prompt_tokens.clone();
            let mut generated: Vec<i32> = Vec::with_capacity(max_tokens.min(1024));
            let mut last_finish = "stop".to_string();

            // Check if speculative decoding is enabled
            let use_speculation = engine.speculative_config.enabled;

            for i in 0..max_tokens {
                // Issue prefetch for the *next* shard before sampling the
                // current token, so the upcoming layer's weights are warm.
                {
                    let next_shard = ((i as u16) + 1) % 16;
                    if let Some(cache) = stream_cache.lock().await.as_ref() {
                        if let Err(e) = cache.prefetch(next_shard) {
                            debug!("prefetch(shard {}) hint failed: {}", next_shard, e);
                        }
                    }
                }

                let next_token = if use_speculation {
                    // Try speculative decoding first
                    engine.sample_token_speculative(&model_ref_id, &tokens, temperature, top_p, &stop_tokens).await
                } else {
                    // Regular sampling
                    engine.sample_token_for(&model_ref_id, &tokens, temperature)
                };

                let next_token = match next_token {
                    Ok(t) => t,
                    Err(e) => {
                        let _ = tx.send(Err(e)).await;
                        return;
                    }
                };
                tokens.push(next_token);
                generated.push(next_token);

                let is_last = stop_tokens.contains(&next_token) || i + 1 == max_tokens;
                let finish_reason = if is_last {
                    if stop_tokens.contains(&next_token) { "stop".to_string() } else { "length".to_string() }
                } else {
                    "stop".to_string()
                };
                // For the streaming protocol: send `(token, finish_reason)` for each step.
                // Downstream consumer treats only the final item as authoritative for finish_reason.
                if is_last { last_finish = finish_reason.clone(); }
                if tx.send(Ok((next_token, finish_reason.clone()))).await.is_err() {
                    return; // consumer dropped
                }
                if is_last { break; }
            }
            let _ = last_finish; // already sent on the last stream item
        });

        Ok(tokio_stream::wrappers::ReceiverStream::new(rx))
    }

    fn clone_handle_for_stream(&self) -> StreamHandle {
        // We clone the Arc to the models map and the default backend so the
        // spawned task can safely sample without holding a borrow across awaits.
        StreamHandle {
            models: self.models.clone(),
            default_backend: self.default_backend.clone(),
            ngram_index: self.ngram_index.clone(),
            speculator: Arc::new(self.speculator.clone()),
            verifier: Arc::new(self.verifier.clone()),
            speculative_config: self.speculative_config.clone(),
        }
    }

    fn sample_token_for(&self, model_id: &str, tokens: &[i32], temperature: f32) -> Result<i32> {
        self.clone_handle_for_stream().sample_token_for(model_id, tokens, temperature)
    }
}

/// Lightweight handle used by streams to access models + backend without
/// holding a borrow on `InferenceEngine` itself.
struct StreamHandle {
    models: Arc<RwLock<HashMap<String, Arc<ModelInstance>>>>,
    default_backend: Arc<dyn ComputeBackend>,
    ngram_index: Arc<NgramIndex>,
    speculator: Arc<OnceLock<Arc<tokio::sync::Mutex<Speculator>>>>,
    verifier: Arc<OnceLock<Arc<SpeculativeVerifier>>>,
    speculative_config: SpeculativeConfig,
}

impl StreamHandle {
    pub fn sample_token_for(&self, _model_id: &str, _tokens: &[i32], temperature: f32) -> Result<i32> {
        // Simplified sampler stub.
        // In a real implementation, this would look up the model,
        // run the forward pass, and then sample from the logits.
        if temperature > 0.0 {
            Ok(1)
        } else {
            Ok(2)
        }
    }

    /// Sample a token using speculative decoding if available
    pub async fn sample_token_speculative(
        &self,
        model_id: &str,
        tokens: &[i32],
        temperature: f32,
        top_p: f32,
        stop_tokens: &[i32],
    ) -> Result<i32> {
        // Try to get speculator and verifier
        if self.speculative_config.enabled {
            if let Some(speculator_lock) = self.speculator.get() {
                if let Some(verifier) = self.verifier.get() {
                    return self.sample_with_speculation(model_id, tokens, temperature, top_p, stop_tokens, speculator_lock, verifier).await;
                }
            }
        }

        // Fallback to regular sampling
        self.sample_token_for(model_id, tokens, temperature)
    }

    /// Core speculative sampling logic
    async fn sample_with_speculation(
        &self,
        model_id: &str,
        tokens: &[i32],
        temperature: f32,
        top_p: f32,
        stop_tokens: &[i32],
        speculator_lock: &Arc<tokio::sync::Mutex<Speculator>>,
        verifier: &Arc<SpeculativeVerifier>,
    ) -> Result<i32> {
        // Get model instance - clone inside the read guard to avoid lifetime issues
        let model: Arc<ModelInstance> = {
            let models = self.models.read().unwrap();
            models.get(model_id)
                .ok_or_else(|| anyhow!("Model not found: {}", model_id))?
                .clone()
        };

        // Convert prompt tokens
        let prompt_tokens: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();

        // Get draft tokens
        let spec_result = {
            let mut speculator = speculator_lock.lock().await;
            speculator.generate_draft(&prompt_tokens)
        };

        if !spec_result.should_verify || spec_result.draft_tokens.is_empty() {
            // Fallback to regular sampling
            return self.sample_token_for(model_id, tokens, temperature);
        }

        // Convert draft tokens
        let draft_tokens: Vec<DraftToken> = spec_result.draft_tokens;

        // Verify with batched forward pass
        let verification = verifier.verifier.verify_draft(
            &*model,
            &prompt_tokens,
            &draft_tokens,
            temperature,
            top_p,
            &stop_tokens.iter().map(|&t| t as u32).collect::<Vec<_>>(),
        ).await?;

        // Update N-gram index with accepted tokens
        for token in &verification.accepted_tokens {
            let mut new_tokens = prompt_tokens.clone();
            new_tokens.push(*token);
            self.ngram_index.insert(&new_tokens);
        }

        // Return the first accepted token (or sample from target if none accepted)
        if let Some(&first_accepted) = verification.accepted_tokens.first() {
            return Ok(first_accepted as i32);
        }

        // Fallback: sample from target
        self.sample_token_for(model_id, tokens, temperature)
    }

    /// Generate with speculative decoding support
    pub async fn generate_stream_speculative(
        &self,
        model_id: &str,
        prompt_tokens: &[i32],
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        stop_tokens: &[i32],
        tx: tokio::sync::mpsc::Sender<Result<(i32, String)>>,
    ) -> Result<()> {
        let models = self.models.read().unwrap();
        let model = models.get(model_id)
            .ok_or_else(|| anyhow!("Model not found: {}", model_id))?;

        // Initialize speculative if enabled
        if self.speculative_config.enabled {
            // Note: speculator/verifier would need lazy init here in real implementation
        }

        // For now, fall back to regular streaming
        let mut tokens: Vec<u32> = prompt_tokens.iter().map(|&t| t as u32).collect();
        let mut generated = 0;

        while generated < max_tokens {
            let next_token = self.sample_token_for(model_id, &tokens.iter().map(|&t| t as i32).collect::<Vec<_>>(), temperature)?;
            tokens.push(next_token as u32);

            let finish_reason = if stop_tokens.contains(&next_token) || generated == max_tokens - 1 {
                "stop".to_string()
            } else {
                "continue".to_string()
            };

            if tx.send(Ok((next_token, finish_reason))).await.is_err() {
                break; // Receiver dropped
            }

            generated += 1;

            if stop_tokens.contains(&next_token) {
                break;
            }
        }

        Ok(())
    }
}

/// Thin `VolatileWeights` shim over a resident monolithic BrainPack tensor.
/// Used by [`InferenceEngine::with_tensor`] when no sharded StreamCache is
/// bound. There is no eviction to worry about — the mmap outlives the borrow.
struct MonolithicWeights<'a> {
    data: &'a [u8],
}

impl VolatileWeights for MonolithicWeights<'_> {
    fn as_f16(&self) -> &[half::f16] {
        unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr() as *const half::f16,
                self.data.len() / std::mem::size_of::<half::f16>(),
            )
        }
    }
    fn as_bytes(&self) -> &[u8] { self.data }
    fn shard_id(&self) -> u16 { 0 }
}

/// Given a loaded `.brain` path, decide whether it is the base fragment of a
/// sharded pack. Sharded layouts place a `<name>.shard.idx` alongside the
/// `<name>.NN.brain` fragments; we look for that sibling and, if present,
/// return `(dir, <name>)` so a [`StreamCache`] can open the full pack.
fn detect_shard_dir(path: &Path) -> Option<(PathBuf, String)> {
    let file_name = path.file_name()?.to_str()?;
    // Sharded base fragment names look like `<name>.00.brain`.
    let base_marker = ".00.brain";
    let name = if file_name.ends_with(base_marker) {
        &file_name[..file_name.len() - base_marker.len()]
    } else {
        // Also accept a plain `.brain` if it has a sibling `.shard.idx`.
        file_name.strip_suffix(".brain")?
    };
    let dir = path.parent()?;
    let idx = dir.join(format!("{}.shard.idx", name));
    if idx.exists() {
        Some((dir.to_path_buf(), name.to_string()))
    } else {
        None
    }
}

// ============================================================================
// IPC Protocol
// ============================================================================

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum IpcRequest {
    LoadModel { filename: String },
    UnloadModel { model_id: String },
    ListModels,
    Generate { request: GenerateRequest },
    StreamingStats,
    HealthCheck,
    Shutdown,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum IpcResponse {
    ModelLoaded { model_id: String, info: ModelInfo },
    ModelUnloaded { model_id: String },
    ModelsList { models: Vec<ModelInfo> },
    GenerateResponse { response: GenerateResponse },
    /// Streaming-cache snapshot for the dashboard gauges. `stats_json` is the
    /// hand-rolled JSON from `CacheStats::to_json` ( avoids a serde_json dep
    /// in stream-cache); null when no sharded model is bound.
    StreamingStats { stats_json: Option<String> },
    HealthOk { version: String, uptime_secs: u64 },
    Error { code: String, message: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub architecture: String,
    pub parameter_count: u64,
    pub quantization: String,
    pub context_length: u32,
    pub backend: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateRequest {
    pub model_id: String,
    pub prompt_tokens: Vec<i32>,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub stop_tokens: Vec<i32>,
    pub stream: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateResponse {
    pub tokens: Vec<i32>,
    pub finish_reason: String,
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

// ============================================================================
// IPC Server
// ============================================================================

/// IPC Server handling requests
pub struct IpcServer {
    engine: Arc<InferenceEngine>,
    shutdown_tx: mpsc::Sender<()>,
    start_time: std::time::Instant,
}

impl IpcServer {
    pub fn new(engine: Arc<InferenceEngine>) -> (Self, mpsc::Receiver<()>) {
        let (shutdown_tx, shutdown_rx) = mpsc::channel(1);
        (Self {
            engine,
            shutdown_tx,
            start_time: std::time::Instant::now(),
        }, shutdown_rx)
    }

    /// Run Unix socket server
    #[cfg(unix)]
    pub async fn run_unix(&self, socket_path: &Path) -> Result<()> {
        if socket_path.exists() {
            std::fs::remove_file(socket_path)?;
        }

        let listener = UnixListener::bind(socket_path)?;
        info!("IPC server listening on {}", socket_path.display());

        loop {
            tokio::select! {
                Ok((stream, _)) = listener.accept() => {
                    let engine = self.engine.clone();
                    tokio::spawn(async move {
                        if let Err(e) = Self::handle_connection(stream, engine).await {
                            error!("Connection error: {}", e);
                        }
                    });
                }
                _ = tokio::signal::ctrl_c() => {
                    info!("Shutdown signal received");
                    break;
                }
            }
        }
        Ok(())
    }

    /// Run TCP server (cross-platform)
    pub async fn run_tcp(&self, addr: &str) -> Result<()> {
        let listener = TcpListener::bind(addr).await?;
        info!("IPC server listening on {}", addr);

        loop {
            tokio::select! {
                Ok((stream, _)) = listener.accept() => {
                    let engine = self.engine.clone();
                    tokio::spawn(async move {
                        if let Err(e) = Self::handle_connection_tcp(stream, engine).await {
                            error!("Connection error: {}", e);
                        }
                    });
                }
                _ = tokio::signal::ctrl_c() => {
                    info!("Shutdown signal received");
                    break;
                }
            }
        }
        Ok(())
    }

    /// Handle Unix socket connection
    #[cfg(unix)]
    async fn handle_connection(mut stream: UnixStream, engine: Arc<InferenceEngine>) -> Result<()> {
        let mut buf = vec![0u8; 65536];
        loop {
            let n = stream.read(&mut buf).await?;
            if n == 0 { break; }

            let request: IpcRequest = serde_json::from_slice(&buf[..n])?;
            let response = Self::handle_request(request, &engine).await;
            let response_bytes = serde_json::to_vec(&response)?;
            stream.write_all(&response_bytes).await?;
        }
        Ok(())
    }

    /// Handle TCP connection
    async fn handle_connection_tcp(mut stream: TcpStream, engine: Arc<InferenceEngine>) -> Result<()> {
        let mut buf = vec![0u8; 65536];
        loop {
            let n = stream.read(&mut buf).await?;
            if n == 0 { break; }

            let request: IpcRequest = serde_json::from_slice(&buf[..n])?;
            let response = Self::handle_request(request, &engine).await;
            let response_bytes = serde_json::to_vec(&response)?;
            stream.write_all(&response_bytes).await?;
        }
        Ok(())
    }

    async fn handle_request(request: IpcRequest, engine: &InferenceEngine) -> IpcResponse {
        match request {
            IpcRequest::LoadModel { filename } => {
                match engine.load_model(&filename).await {
                    Ok(model_id) => {
                        let models = engine.list_models();
                        let info = models.iter().find(|m| m.id == model_id).cloned().unwrap();
                        IpcResponse::ModelLoaded { model_id, info }
                    }
                    Err(e) => IpcResponse::Error { code: "LOAD_FAILED".into(), message: e.to_string() }
                }
            }
            IpcRequest::UnloadModel { model_id } => {
                match engine.unload_model(&model_id) {
                    Ok(_) => IpcResponse::ModelUnloaded { model_id },
                    Err(e) => IpcResponse::Error { code: "UNLOAD_FAILED".into(), message: e.to_string() }
                }
            }
            IpcRequest::ListModels => {
                IpcResponse::ModelsList { models: engine.list_models() }
            }
            IpcRequest::Generate { request } => {
                match engine.generate(request) {
                    Ok(response) => IpcResponse::GenerateResponse { response },
                    Err(e) => IpcResponse::Error { code: "GENERATE_FAILED".into(), message: e.to_string() }
                }
            }
            IpcRequest::StreamingStats => {
                let stats = engine.streaming_stats().await
                    .map(|s| s.to_json());
                IpcResponse::StreamingStats { stats_json: stats }
            }
            IpcRequest::HealthCheck => {
                IpcResponse::HealthOk {
                    version: env!("CARGO_PKG_VERSION").into(),
                    uptime_secs: std::time::Instant::now().duration_since(engine.start_time()).as_secs(),
                }
            }
            IpcRequest::Shutdown => {
                IpcResponse::HealthOk { version: "shutdown".into(), uptime_secs: 0 }
            }
        }
    }
}

// Add start_time to InferenceEngine
impl InferenceEngine {
    pub fn start_time(&self) -> std::time::Instant {
        self.start_time
    }
}

// ============================================================================
// Backend Selection
// ============================================================================

pub fn select_backend(preference: &str) -> Result<Arc<dyn ComputeBackend>> {
    match preference.to_lowercase().as_str() {
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                let backend = compute_cuda::CudaBackend::new()?;
                Ok(Arc::new(backend))
            }
            #[cfg(not(feature = "cuda"))]
            Err(anyhow!("CUDA backend not compiled in"))
        }
        "rocm" => {
            #[cfg(feature = "rocm")]
            {
                let backend = compute_rocm::RocmBackend::new(0)?;
                Ok(Arc::new(backend))
            }
            #[cfg(not(feature = "rocm"))]
            Err(anyhow!("ROCm backend not compiled in"))
        }
        "metal" => {
            #[cfg(feature = "metal")]
            {
                let backend = compute_metal::MetalBackend::new()?;
                Ok(Arc::new(backend))
            }
            #[cfg(not(feature = "metal"))]
            Err(anyhow!("Metal backend not compiled in"))
        }
        "cpu" | "auto" => {
            let backend = CpuBackend::new()?;
            Ok(Arc::new(backend))
        }
        _ => Err(anyhow!("Unknown backend: {}", preference)),
    }
}

// ============================================================================
// Main Entry Point
// ============================================================================

pub async fn run_engine(model_dir: &Path, backend: &str, socket: Option<&Path>, tcp: Option<&str>) -> Result<()> {
    let backend = select_backend(backend)?;
    let engine = Arc::new(InferenceEngine::new(model_dir, backend)?);
    let (server, _shutdown) = IpcServer::new(engine);

    #[cfg(unix)]
    if let Some(socket_path) = socket {
        return server.run_unix(socket_path).await;
    }

    if let Some(addr) = tcp {
        return server.run_tcp(addr).await;
    }

    // Default: TCP on localhost:8080
    server.run_tcp("127.0.0.1:8080").await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_backend() {
        let backend = CpuBackend::new().unwrap();
        assert_eq!(backend.name(), "cpu");
    }

    #[test]
    fn test_backend_selection() {
        let backend = select_backend("cpu").unwrap();
        assert_eq!(backend.name(), "cpu");
    }
}