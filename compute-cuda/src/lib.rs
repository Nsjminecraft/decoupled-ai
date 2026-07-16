// compute-cuda/src/lib.rs
use anyhow::{anyhow, Result};
use cudarc::driver::{
    CudaDevice, CudaFunction, CudaSlice, DevicePtr, LaunchAsync, LaunchConfig,
    MemcpyDst, MemcpySrc, ValidAsZeroBits,
};
use half::f16;
use std::sync::Arc;
use tracing::{debug, info, warn};

// ============================================================================
// CUDA Context & Device Management
// ============================================================================

/// CUDA compute backend for NVIDIA GPUs
pub struct CudaBackend {
    device: Arc<CudaDevice>,
    kernels: KernelRegistry,
}

impl CudaBackend {
    /// Initialize CUDA backend on first available device
    pub fn new() -> Result<Self> {
        let device = CudaDevice::new(0)?;
        info!("CUDA device: {}", device.name()?);
        info!("Compute capability: {}.{}", device.capability().0, device.capability().1);

        let kernels = KernelRegistry::load(&device)?;

        Ok(Self { device, kernels })
    }

    /// Initialize on specific device ordinal
    pub fn new_on_device(device_ordinal: usize) -> Result<Self> {
        let device = CudaDevice::new(device_ordinal)?;
        info!("CUDA device {}: {}", device_ordinal, device.name()?);

        let kernels = KernelRegistry::load(&device)?;

        Ok(Self { device, kernels })
    }

    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    pub fn synchronize(&self) -> Result<()> {
        self.device.synchronize()?;
        Ok(())
    }

    /// Allocate device memory for tensor
    pub fn alloc_tensor<T: ValidAsZeroBits>(&self, len: usize) -> Result<CudaSlice<T>> {
        let slice = self.device.alloc_zeros::<T>(len)?;
        Ok(slice)
    }

    /// Copy host tensor to device
    pub fn h2d_copy<T: ValidAsZeroBits>(&self, host: &[T]) -> Result<CudaSlice<T>> {
        let slice = self.device.htod_copy(host)?;
        Ok(slice)
    }

    /// Copy device tensor to host
    pub fn d2h_copy<T: ValidAsZeroBits>(&self, device: &CudaSlice<T>) -> Result<Vec<T>> {
        let host = self.device.dtoh_sync_copy(device)?;
        Ok(host)
    }

    // ========================================================================
    // GEMM Operations (General Matrix Multiply)
    // ========================================================================

    /// Batched GEMM: C = A @ B^T (for attention)
    /// A: [batch, seq_len, head_dim]  B: [batch, seq_len, head_dim]  C: [batch, seq_len, seq_len]
    pub fn batched_gemm_f16(
        &self,
        a: &CudaSlice<f16>,  // [batch * seq_len * head_dim]
        b: &CudaSlice<f16>,  // [batch * seq_len * head_dim]
        c: &mut CudaSlice<f16>, // [batch * seq_len * seq_len]
        batch: usize,
        seq_len: usize,
        head_dim: usize,
    ) -> Result<()> {
        let kernel = self.kernels.get("batched_gemm_f16")?;

        let grid = (batch * seq_len * seq_len + 255) / 256;
        let block = 256;

        unsafe {
            kernel.launch(LaunchConfig {
                grid_dim: (grid as u32, 1, 1),
                block_dim: (block as u32, 1, 1),
                shared_mem_bytes: 0,
            }, (
                a, b, c,
                batch as i32, seq_len as i32, head_dim as i32,
            ))?;
        }
        Ok(())
    }

    /// GEMM: C = A @ B where A: [M, K], B: [K, N], C: [M, N]
    pub fn gemm_f16(
        &self,
        a: &CudaSlice<f16>,  // [M, K]
        b: &CudaSlice<f16>,  // [K, N]
        c: &mut CudaSlice<f16>, // [M, N]
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        // Use cuBLAS for optimal performance
        use cudarc::cublas::{Cublas, GemmOp};

        let cublas = Cublas::new(self.device.clone())?;

        // CUBLAS expects column-major, so we compute C^T = B^T @ A^T
        // which means we call cublasSgemm with transposed matrices
        let alpha = 1.0f32;
        let beta = 0.0f32;

        // For row-major A[m,k], B[k,n] -> C[m,n]
        // cuBLAS is column-major: C_col = A_col * B_col
        // So: C_row^T = (A_row * B_row)^T = B_row^T * A_row^T
        // Call: cublasSgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, m, k, alpha, B, n, A, k, beta, C, n)
        cublas.gemm(
            GemmOp::T, GemmOp::T,
            n as i32, m as i32, k as i32,
            &alpha,
            b, n as i32,
            a, k as i32,
            &beta,
            c, n as i32,
        )?;

        Ok(())
    }

    /// GEMM with FP32 accumulation for numerical stability
    pub fn gemm_f16_f32_accum(
        &self,
        a: &CudaSlice<f16>,  // [M, K]
        b: &CudaSlice<f16>,  // [K, N]
        c: &mut CudaSlice<f16>, // [M, N]
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        use cudarc::cublas::{Cublas, GemmOp};

        let cublas = Cublas::new(self.device.clone())?;

        // Use FP32 tensor cores via cublasGemmEx
        // For now use standard gemm - cublasGemmEx requires more setup
        let alpha = 1.0f32;
        let beta = 0.0f32;

        cublas.gemm(
            GemmOp::T, GemmOp::T,
            n as i32, m as i32, k as i32,
            &alpha,
            b, n as i32,
            a, k as i32,
            &beta,
            c, n as i32,
        )?;

        Ok(())
    }

    // ========================================================================
    // Attention Kernels
    // ========================================================================

    /// Flash Attention forward pass (simplified)
    /// Q: [batch, heads, seq_len, head_dim]
    /// K: [batch, heads, seq_len, head_dim]
    /// V: [batch, heads, seq_len, head_dim]
    /// O: [batch, heads, seq_len, head_dim]
    pub fn flash_attention_f16(
        &self,
        q: &CudaSlice<f16>,
        k: &CudaSlice<f16>,
        v: &CudaSlice<f16>,
        o: &mut CudaSlice<f16>,
        batch: usize,
        heads: usize,
        seq_len: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<()> {
        let kernel = self.kernels.get("flash_attention_f16")?;

        let block = 256;
        let grid = (batch * heads * seq_len + block - 1) / block;

        unsafe {
            kernel.launch(LaunchConfig {
                grid_dim: (grid as u32, 1, 1),
                block_dim: (block as u32, 1, 1),
                shared_mem_bytes: seq_len * head_dim * 2, // Q*K^T in shared mem
            }, (
                q, k, v, o,
                batch as i32, heads as i32, seq_len as i32, head_dim as i32,
                scale,
            ))?;
        }
        Ok(())
    }

    /// Standard scaled dot-product attention
    pub fn attention_f16(
        &self,
        q: &CudaSlice<f16>,   // [batch, heads, seq_len, head_dim]
        k: &CudaSlice<f16>,
        v: &CudaSlice<f16>,
        out: &mut CudaSlice<f16>, // [batch, heads, seq_len, head_dim]
        batch: usize,
        heads: usize,
        seq_len: usize,
        head_dim: usize,
    ) -> Result<()> {
        // Step 1: Q @ K^T -> scores [batch, heads, seq_len, seq_len]
        let scores_size = batch * heads * seq_len * seq_len;
        let mut scores = self.device.alloc_zeros::<f16>(scores_size)?;

        self.batched_gemm_f16(q, k, &mut scores, batch * heads, seq_len, head_dim)?;

        // Step 2: Scale scores
        let scale = (head_dim as f32).sqrt().recip();
        self.scale_f16(&mut scores, scale)?;

        // Step 3: Softmax
        self.softmax_f16(&mut scores, batch * heads, seq_len)?;

        // Step 4: scores @ V -> out
        self.batched_gemm_f16(&scores, v, out, batch * heads, seq_len, head_dim)?;

        Ok(())
    }

    // ========================================================================
    // Activation Functions
    // ========================================================================

    /// SiLU / Swish activation: x * sigmoid(x)
    pub fn silu_f16(&self, x: &mut CudaSlice<f16>, n: usize) -> Result<()> {
        let kernel = self.kernels.get("silu_f16")?;
        let grid = (n + 255) / 256;

        unsafe {
            kernel.launch(LaunchConfig {
                grid_dim: (grid as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            }, (x, n as i32))?;
        }
        Ok(())
    }

    /// GELU activation
    pub fn gelu_f16(&self, x: &mut CudaSlice<f16>, n: usize) -> Result<()> {
        let kernel = self.kernels.get("gelu_f16")?;
        let grid = (n + 255) / 256;

        unsafe {
            kernel.launch(LaunchConfig {
                grid_dim: (grid as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            }, (x, n as i32))?;
        }
        Ok(())
    }

    /// ReLU activation
    pub fn relu_f16(&self, x: &mut CudaSlice<f16>, n: usize) -> Result<()> {
        let kernel = self.kernels.get("relu_f16")?;
        let grid = (n + 255) / 256;

        unsafe {
            kernel.launch(LaunchConfig {
                grid_dim: (grid as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            }, (x, n as i32))?;
        }
        Ok(())
    }

    // ========================================================================
    // Element-wise Operations
    // ========================================================================

    /// Add bias: x += bias (broadcast over last dim)
    pub fn add_bias_f16(&self, x: &mut CudaSlice<f16>, bias: &CudaSlice<f16>, n: usize, bias_size: usize) -> Result<()> {
        let kernel = self.kernels.get("add_bias_f16")?;
        let grid = (n + 255) / 256;

        unsafe {
            kernel.launch(LaunchConfig {
                grid_dim: (grid as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            }, (x, bias, n as i32, bias_size as i32))?;
        }
        Ok(())
    }

    /// Scale tensor: x *= scale
    pub fn scale_f16(&self, x: &mut CudaSlice<f16>, scale: f32, n: usize) -> Result<()> {
        let kernel = self.kernels.get("scale_f16")?;
        let grid = (n + 255) / 256;

        unsafe {
            kernel.launch(LaunchConfig {
                grid_dim: (grid as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            }, (x, scale, n as i32))?;
        }
        Ok(())
    }

    /// Element-wise add: a += b
    pub fn add_f16(&self, a: &mut CudaSlice<f16>, b: &CudaSlice<f16>, n: usize) -> Result<()> {
        let kernel = self.kernels.get("add_f16")?;
        let grid = (n + 255) / 256;

        unsafe {
            kernel.launch(LaunchConfig {
                grid_dim: (grid as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            }, (a, b, n as i32))?;
        }
        Ok(())
    }

    /// RMSNorm: x = x / sqrt(mean(x^2) + eps) * weight
    pub fn rms_norm_f16(
        &self,
        x: &mut CudaSlice<f16>,
        weight: &CudaSlice<f16>,
        out: &mut CudaSlice<f16>,
        batch_seq: usize,
        hidden_dim: usize,
        eps: f32,
    ) -> Result<()> {
        let kernel = self.kernels.get("rms_norm_f16")?;
        let grid = (batch_seq + 255) / 256;

        unsafe {
            kernel.launch(LaunchConfig {
                grid_dim: (grid as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: hidden_dim * 2, // for reduction
            }, (x, weight, out, batch_seq as i32, hidden_dim as i32, eps))?;
        }
        Ok(())
    }

    // ========================================================================
    // Quantized Operations
    // ========================================================================

    /// Dequantize Q4_K_M to FP16
    pub fn dequantize_q4km_f16(
        &self,
        q_weights: &CudaSlice<u8>,
        scales: &CudaSlice<f16>,
        out: &mut CudaSlice<f16>,
        num_elements: usize,
    ) -> Result<()> {
        let kernel = self.kernels.get("dequantize_q4km_f16")?;
        let grid = (num_elements + 255) / 256;

        unsafe {
            kernel.launch(LaunchConfig {
                grid_dim: (grid as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            }, (q_weights, scales, out, num_elements as i32))?;
        }
        Ok(())
    }

    /// Quantized GEMM: Q4_K x FP16 -> FP16
    pub fn gemm_q4km_f16(
        &self,
        a_q: &CudaSlice<u8>,     // Q4_K_M quantized [M, K]
        a_scales: &CudaSlice<f16>,
        b: &CudaSlice<f16>,      // FP16 [K, N]
        c: &mut CudaSlice<f16>,  // Output [M, N]
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        let kernel = self.kernels.get("gemm_q4km_f16")?;
        let grid = (m * n + 255) / 256;

        unsafe {
            kernel.launch(LaunchConfig {
                grid_dim: (grid as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: k * 2, // K elements in shared mem
            }, (a_q, a_scales, b, c, m as i32, n as i32, k as i32))?;
        }
        Ok(())
    }

    // ========================================================================
    // RoPE (Rotary Positional Embeddings)
    // ========================================================================

    /// Apply RoPE to Q and K in-place
    pub fn rope_f16(
        &self,
        q: &mut CudaSlice<f16>,
        k: &mut CudaSlice<f16>,
        cos: &CudaSlice<f16>,
        sin: &CudaSlice<f16>,
        batch: usize,
        heads: usize,
        seq_len: usize,
        head_dim: usize,
    ) -> Result<()> {
        let kernel = self.kernels.get("rope_f16")?;
        let total = batch * heads * seq_len * (head_dim / 2);
        let grid = (total + 255) / 256;

        unsafe {
            kernel.launch(LaunchConfig {
                grid_dim: (grid as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            }, (q, k, cos, sin, batch as i32, heads as i32, seq_len as i32, head_dim as i32))?;
        }
        Ok(())
    }

    // ========================================================================
    // Sampling
    // ========================================================================

    /// Top-k top-p sampling from logits
    pub fn sample_topk_topp(
        &self,
        logits: &CudaSlice<f16>,
        next_token: &mut CudaSlice<i32>,
        batch: usize,
        vocab_size: usize,
        top_k: usize,
        top_p: f32,
        temperature: f32,
        seed: u64,
    ) -> Result<()> {
        let kernel = self.kernels.get("sample_topk_topp")?;
        let grid = (batch + 255) / 256;

        unsafe {
            kernel.launch(LaunchConfig {
                grid_dim: (grid as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: vocab_size * 2, // logits in shared mem
            }, (logits, next_token, batch as i32, vocab_size as i32, top_k as i32, top_p, temperature, seed))?;
        }
        Ok(())
    }

    /// Argmax sampling (greedy)
    pub fn sample_argmax(
        &self,
        logits: &CudaSlice<f16>,
        next_token: &mut CudaSlice<i32>,
        batch: usize,
        vocab_size: usize,
    ) -> Result<()> {
        let kernel = self.kernels.get("sample_argmax")?;
        let grid = (batch + 255) / 256;

        unsafe {
            kernel.launch(LaunchConfig {
                grid_dim: (grid as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            }, (logits, next_token, batch as i32, vocab_size as i32))?;
        }
        Ok(())
    }
}

// ============================================================================
// Kernel Registry
// ============================================================================

struct KernelRegistry {
    kernels: std::collections::HashMap<String, CudaFunction>,
}

impl KernelRegistry {
    fn load(device: &Arc<CudaDevice>) -> Result<Self> {
        let mut kernels = std::collections::HashMap::new();

        // Load PTX kernels
        let ptx = include_str!("kernels.ptx");
        let module = device.load_ptx(ptx, "kernels", &[])?;

        let kernel_names = [
            "batched_gemm_f16",
            "flash_attention_f16",
            "silu_f16",
            "gelu_f16",
            "relu_f16",
            "add_bias_f16",
            "scale_f16",
            "add_f16",
            "rms_norm_f16",
            "dequantize_q4km_f16",
            "gemm_q4km_f16",
            "rope_f16",
            "sample_topk_topp",
            "sample_argmax",
            "softmax_f16",
        ];

        for name in &kernel_names {
            let func = module.get_function(name)?;
            kernels.insert(name.to_string(), func);
        }

        info!("Loaded {} CUDA kernels", kernels.len());
        Ok(Self { kernels })
    }

    fn get(&self, name: &str) -> Result<&CudaFunction> {
        self.kernels.get(name).ok_or_else(|| anyhow!("Kernel '{}' not found", name))
    }
}

// ============================================================================
// CUDA Kernel Source (PTX will be compiled from this)
// ============================================================================

// Kernels are in compute-cuda/src/kernels.cu and compiled to PTX at build time
// See build.rs for compilation

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires CUDA device"]
    fn test_cuda_backend_init() {
        let backend = CudaBackend::new().unwrap();
        assert!(backend.synchronize().is_ok());
    }
}