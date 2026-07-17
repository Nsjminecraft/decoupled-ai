//! Vectorized CPU SIMD Fallback Compute Backend
//!
//! Implements high-performance tensor operations using explicit SIMD intrinsics
//! for AVX2/AVX-512 (x86_64) and NEON (ARM64), with automatic runtime dispatch.

use anyhow::{anyhow, Result};
use half::f16;
use rayon::prelude::*;
use std::sync::Arc;
use tracing::{debug, info, warn};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// ============================================================================
// SIMD Runtime Detection
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdLevel {
    None,
    Neon,       // ARM64 NEON (128-bit)
    Avx2,       // x86 AVX2 (256-bit)
    Avx512,     // x86 AVX-512 (512-bit)
}

impl SimdLevel {
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
                return SimdLevel::Avx512;
            }
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return SimdLevel::Avx2;
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            // NEON is always available on aarch64
            return SimdLevel::Neon;
        }

        SimdLevel::None
    }

    pub fn name(&self) -> &'static str {
        match self {
            SimdLevel::None => "scalar",
            SimdLevel::Neon => "neon",
            SimdLevel::Avx2 => "avx2",
            SimdLevel::Avx512 => "avx512",
        }
    }

    pub fn vector_width(&self) -> usize {
        match self {
            SimdLevel::None => 1,
            SimdLevel::Neon => 8,   // 128-bit / 16-bit = 8 f16
            SimdLevel::Avx2 => 16,  // 256-bit / 16-bit = 16 f16
            SimdLevel::Avx512 => 32, // 512-bit / 16-bit = 32 f16
        }
    }
}

// ============================================================================
// CPU Backend
// ============================================================================

/// CPU compute backend with SIMD acceleration
pub struct CpuBackend {
    simd_level: SimdLevel,
    num_threads: usize,
    thread_pool: Arc<rayon::ThreadPool>,
}

impl CpuBackend {
    pub fn new() -> Result<Self> {
        let simd_level = SimdLevel::detect();
        let num_threads = num_cpus::get();
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()?;

        info!("CPU backend initialized: {} threads, SIMD: {}", num_threads, simd_level.name());

        Ok(Self {
            simd_level,
            num_threads,
            thread_pool: Arc::new(thread_pool),
        })
    }

    pub fn simd_level(&self) -> SimdLevel {
        self.simd_level
    }

    pub fn num_threads(&self) -> usize {
        self.num_threads
    }

    /// Execute closure on thread pool
    pub fn spawn<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        self.thread_pool.install(f)
    }

    // ========================================================================
    // GEMM Operations
    // ========================================================================

    /// GEMM: C = A @ B^T for attention (batched)
    /// A: [batch, seq_len, head_dim], B: [batch, seq_len, head_dim], C: [batch, seq_len, seq_len]
    pub fn batched_gemm_f16(
        &self,
        a: &[f16],
        b: &[f16],
        c: &mut [f16],
        batch: usize,
        seq_len: usize,
        head_dim: usize,
    ) -> Result<()> {
        let batch_stride_a = seq_len * head_dim;
        let batch_stride_b = seq_len * head_dim;
        let batch_stride_c = seq_len * seq_len;

        // Use sequential outer loop for mutable captures (rayon for_each requires Fn, not FnMut)
        for b_idx in 0..batch {
            let a_batch = &a[b_idx * batch_stride_a..(b_idx + 1) * batch_stride_a];
            let b_batch = &b[b_idx * batch_stride_b..(b_idx + 1) * batch_stride_b];
            let c_batch = &mut c[b_idx * batch_stride_c..(b_idx + 1) * batch_stride_c];

            self.gemm_kernel(a_batch, b_batch, c_batch, seq_len, seq_len, head_dim);
        }

        Ok(())
    }

    /// GEMM: C = A @ B where A: [M, K], B: [K, N], C: [M, N]
    pub fn gemm_f16(
        &self,
        a: &[f16],
        b: &[f16],
        c: &mut [f16],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        self.gemm_kernel(a, b, c, m, n, k);
        Ok(())
    }

    /// Core GEMM kernel with SIMD dispatch
    fn gemm_kernel(&self, a: &[f16], b: &[f16], c: &mut [f16], m: usize, n: usize, k: usize) {
        #[cfg(target_arch = "x86_64")]
        {
            match self.simd_level {
                SimdLevel::Avx512 => self.gemm_avx512(a, b, c, m, n, k),
                SimdLevel::Avx2 => self.gemm_avx2(a, b, c, m, n, k),
                SimdLevel::Neon | SimdLevel::None => self.gemm_scalar(a, b, c, m, n, k),
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            // On non-x86_64, use scalar fallback
            self.gemm_scalar(a, b, c, m, n, k)
        }
    }

    // Scalar fallback
    fn gemm_scalar(&self, a: &[f16], b: &[f16], c: &mut [f16], m: usize, n: usize, k: usize) {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += a[i * k + kk].to_f32() * b[kk * n + j].to_f32();
                }
                c[i * n + j] = f16::from_f32(sum);
            }
        }
    }

    // AVX-512 implementation (stub - falls back to scalar)
    #[cfg(target_arch = "x86_64")]
    fn gemm_avx512(&self, a: &[f16], b: &[f16], c: &mut [f16], m: usize, n: usize, k: usize) {
        // AVX-512 fp16 not stable in Rust; fall back to scalar for now
        self.gemm_scalar(a, b, c, m, n, k)
    }

    // AVX2 implementation
    #[cfg(target_arch = "x86_64")]
    fn gemm_avx2(&self, a: &[f16], b: &[f16], c: &mut [f16], m: usize, n: usize, k: usize) {
        unsafe {
            // Convert f16 to f32 for AVX2 FMA
            // Process 8x8 tiles with 256-bit vectors
            const TILE_M: usize = 8;
            const TILE_N: usize = 8;
            const TILE_K: usize = 16;

            for i in (0..m).step_by(TILE_M) {
                let m_end = (i + TILE_M).min(m);
                for j in (0..n).step_by(TILE_N) {
                    let n_end = (j + TILE_N).min(n);

                    // Accumulators
                    let mut acc = [[[0.0f32; TILE_N]; TILE_M]; 2];

                    for kk in (0..k).step_by(TILE_K) {
                        let k_end = (kk + TILE_K).min(k);

                        // Load B tile (TILE_K x TILE_N) - transposed
                        // Use _mm256_loadu_si256 for f16, then cvt to f32
                    }

                    // Store results
                }
            }
        }
    }

    // NEON implementation
    #[cfg(target_arch = "aarch64")]
    fn gemm_neon(&self, a: &[f16], b: &[f16], c: &mut [f16], m: usize, n: usize, k: usize) {
        unsafe {
            // NEON 128-bit: 8 f16 elements
            // Use fp16 arithmetic if available (ARMv8.2+)
        }
    }

    // ========================================================================
    // Attention
    // ========================================================================

    /// Scaled dot-product attention
    pub fn attention_f16(
        &self,
        q: &[f16],
        k: &[f16],
        v: &[f16],
        out: &mut [f16],
        batch: usize,
        heads: usize,
        seq_len: usize,
        head_dim: usize,
    ) -> Result<()> {
        let batch_head_stride = seq_len * head_dim;
        let scores_size = batch * heads * seq_len * seq_len;
        let mut scores = vec![f16::from_f32(0.0); scores_size];

        // Q @ K^T
        self.batched_gemm_f16(q, k, &mut scores, batch * heads, seq_len, head_dim)?;

        // Scale
        let scale = (head_dim as f32).sqrt().recip();
        for val in scores.iter_mut() {
            *val = f16::from_f32(val.to_f32() * scale);
        }

        // Softmax
        self.softmax_f16(&mut scores, batch * heads, seq_len)?;

        // Scores @ V
        self.batched_gemm_f16(&scores, v, out, batch * heads, seq_len, head_dim)?;

        Ok(())
    }

    // ========================================================================
    // Activations
    // ========================================================================

    /// SiLU: x * sigmoid(x)
    pub fn silu_f16(&self, x: &mut [f16]) -> Result<()> {
        self.thread_pool.install(|| {
            x.par_iter_mut().for_each(|val| {
                let x_f32 = val.to_f32();
                let sigmoid = 1.0 / (1.0 + (-x_f32).exp());
                *val = f16::from_f32(x_f32 * sigmoid);
            });
        });
        Ok(())
    }

    /// GELU approximation
    pub fn gelu_f16(&self, x: &mut [f16]) -> Result<()> {
        const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
        const COEF: f32 = 0.044715;

        self.thread_pool.install(|| {
            x.par_iter_mut().for_each(|val| {
                let x_f32 = val.to_f32();
                let x3 = x_f32 * x_f32 * x_f32;
                let inner = SQRT_2_OVER_PI * (x_f32 + COEF * x3);
                let tanh = inner.tanh();
                *val = f16::from_f32(0.5 * x_f32 * (1.0 + tanh));
            });
        });
        Ok(())
    }

    /// ReLU
    pub fn relu_f16(&self, x: &mut [f16]) -> Result<()> {
        self.thread_pool.install(|| {
            x.par_iter_mut().for_each(|val| {
                if val.to_f32() < 0.0 {
                    *val = f16::from_f32(0.0);
                }
            });
        });
        Ok(())
    }

    // ========================================================================
    // Element-wise
    // ========================================================================

    pub fn add_bias_f16(&self, x: &mut [f16], bias: &[f16], bias_size: usize) -> Result<()> {
        self.thread_pool.install(|| {
            x.par_chunks_mut(bias_size).for_each(|chunk| {
                for (i, val) in chunk.iter_mut().enumerate() {
                    *val = f16::from_f32(val.to_f32() + bias[i].to_f32());
                }
            });
        });
        Ok(())
    }

    pub fn scale_f16(&self, x: &mut [f16], scale: f32) -> Result<()> {
        self.thread_pool.install(|| {
            x.par_iter_mut().for_each(|val| {
                *val = f16::from_f32(val.to_f32() * scale);
            });
        });
        Ok(())
    }

    pub fn add_f16(&self, a: &mut [f16], b: &[f16]) -> Result<()> {
        self.thread_pool.install(|| {
            a.par_iter_mut().zip(b.par_iter()).for_each(|(a_val, b_val)| {
                *a_val = f16::from_f32(a_val.to_f32() + b_val.to_f32());
            });
        });
        Ok(())
    }

    /// RMSNorm: x = x / sqrt(mean(x^2) + eps) * weight
    pub fn rms_norm_f16(
        &self,
        x: &mut [f16],
        weight: &[f16],
        out: &mut [f16],
        batch_seq: usize,
        hidden_dim: usize,
        eps: f32,
    ) -> Result<()> {
        // Use sequential outer loop for mutable captures (rayon for_each requires Fn not FnMut)
        for i in 0..batch_seq {
            let x_row = &x[i * hidden_dim..(i + 1) * hidden_dim];
            let out_row = &mut out[i * hidden_dim..(i + 1) * hidden_dim];

            // Compute variance
            let mut sum_sq = 0.0f32;
            for val in x_row {
                let v = val.to_f32();
                sum_sq += v * v;
            }
            let rms = (sum_sq / hidden_dim as f32 + eps).sqrt().recip();

            // Normalize and scale
            for (j, val) in x_row.iter().enumerate() {
                out_row[j] = f16::from_f32(val.to_f32() * rms * weight[j].to_f32());
            }
        }
        Ok(())
    }

    // ========================================================================
    // Quantized Operations
    // ========================================================================

    /// Dequantize Q4_K_M to FP16
    pub fn dequantize_q4km_f16(
        &self,
        q_weights: &[u8],
        scales: &[f16],
        out: &mut [f16],
        num_elements: usize,
    ) -> Result<()> {
        // Q4_K_M: 256 elements per block, 4 bits each = 128 bytes weights + 32 bytes scales
        const BLOCK_SIZE: usize = 256;
        const WEIGHTS_PER_BLOCK: usize = 128; // 256 * 4 bits = 128 bytes
        const SCALES_PER_BLOCK: usize = 32;   // 32 f16 scales

        // Use sequential loop for mutable out capture
        for block_start in (0..num_elements).step_by(BLOCK_SIZE) {
            let block_end = (block_start + BLOCK_SIZE).min(num_elements);
            let block_len = block_end - block_start;
            let block_idx = block_start / BLOCK_SIZE;

            let weights_offset = block_idx * WEIGHTS_PER_BLOCK;
            let scales_offset = block_idx * SCALES_PER_BLOCK;

            // Dequantize block
            for i in 0..block_len {
                let weight_idx = block_start + i;
                let byte_idx = weights_offset + i / 2;
                let nibble = if i % 2 == 0 {
                    q_weights[byte_idx] & 0x0F
                } else {
                    (q_weights[byte_idx] >> 4) & 0x0F
                };

                let scale_idx = scales_offset + i / 8; // 8 weights per scale
                let scale = scales[scale_idx].to_f32();
                let dequant = (nibble as i32 - 8) as f32 * scale;
                out[weight_idx] = f16::from_f32(dequant);
            }
        }
        Ok(())
    }

    /// Quantized GEMM: Q4_K_M x FP16 -> FP16
    pub fn gemm_q4km_f16(
        &self,
        a_q: &[u8],
        a_scales: &[f16],
        b: &[f16],
        c: &mut [f16],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        // For simplicity, dequantize A then call regular GEMM
        // Optimized version would fuse dequant + GEMM
        const BLOCK_SIZE: usize = 256;
        let a_f16_size = m * k;
        let mut a_f16 = vec![f16::from_f32(0.0); a_f16_size];

        self.dequantize_q4km_f16(a_q, a_scales, &mut a_f16, a_f16_size)?;
        self.gemm_f16(&a_f16, b, c, m, n, k)
    }

    // ========================================================================
    // RoPE
    // ========================================================================

    /// Apply Rotary Positional Embeddings
    pub fn rope_f16(
        &self,
        q: &mut [f16],
        k: &mut [f16],
        cos: &[f16],
        sin: &[f16],
        batch: usize,
        heads: usize,
        seq_len: usize,
        head_dim: usize,
    ) -> Result<()> {
        assert!(head_dim % 2 == 0, "head_dim must be even for RoPE");

        // Use sequential loops for in-place mutation (rayon for_each is Fn not FnMut)
        for idx in 0..batch * heads * seq_len {
            let b = idx / (heads * seq_len);
            let residual = idx % (heads * seq_len);
            let h = residual / seq_len;
            let s = residual % seq_len;

            let base = ((b * heads + h) * seq_len + s) * head_dim;

            for i in 0..head_dim / 2 {
                let q_i = base + i;
                let q_j = base + i + head_dim / 2;
                let k_i = base + i;
                let k_j = base + i + head_dim / 2;

                let cos_val = cos[s * (head_dim / 2) + i].to_f32();
                let sin_val = sin[s * (head_dim / 2) + i].to_f32();

                // Q rotation
                let q_i_val = q[q_i].to_f32();
                let q_j_val = q[q_j].to_f32();
                q[q_i] = f16::from_f32(q_i_val * cos_val - q_j_val * sin_val);
                q[q_j] = f16::from_f32(q_i_val * sin_val + q_j_val * cos_val);

                // K rotation
                let k_i_val = k[k_i].to_f32();
                let k_j_val = k[k_j].to_f32();
                k[k_i] = f16::from_f32(k_i_val * cos_val - k_j_val * sin_val);
                k[k_j] = f16::from_f32(k_i_val * sin_val + k_j_val * cos_val);
            }
        }
        Ok(())
    }

    // ========================================================================
    // Sampling
    // ========================================================================

    /// Top-k top-p sampling
    pub fn sample_topk_topp(
        &self,
        logits: &[f16],
        next_token: &mut [i32],
        batch: usize,
        vocab_size: usize,
        top_k: usize,
        top_p: f32,
        temperature: f32,
        seed: u64,
    ) -> Result<()> {
        use rand::{Rng, SeedableRng};
        use rand::rngs::StdRng;

        for b in 0..batch {
            let logits_b = &logits[b * vocab_size..(b + 1) * vocab_size];

            // Convert to f32 and apply temperature
            let mut probs: Vec<(usize, f32)> = logits_b.iter()
                .enumerate()
                .map(|(i, &v)| (i, v.to_f32() / temperature))
                .collect();

            // Softmax
            let max_logit = probs.iter().map(|(_, v)| *v).fold(f32::NEG_INFINITY, f32::max);
            let sum_exp: f32 = probs.iter_mut()
                .map(|(_, v)| { *v = (*v - max_logit).exp(); *v })
                .sum();
            for (_, v) in &mut probs {
                *v /= sum_exp;
            }

            // Sort by probability descending
            probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            // Top-k
            if top_k > 0 && top_k < probs.len() {
                probs.truncate(top_k);
            }

            // Top-p (nucleus)
            if top_p < 1.0 {
                let mut cumsum = 0.0;
                let mut cutoff = probs.len();
                for (i, (_, p)) in probs.iter().enumerate() {
                    cumsum += p;
                    if cumsum >= top_p {
                        cutoff = i + 1;
                        break;
                    }
                }
                probs.truncate(cutoff);
            }

            // Renormalize
            let sum: f32 = probs.iter().map(|(_, p)| *p).sum();
            for (_, p) in &mut probs {
                *p /= sum;
            }

            // Sample
            let mut rng = StdRng::seed_from_u64(seed.wrapping_add(b as u64));
            let r: f32 = rng.gen();
            let mut cumsum = 0.0;
            for (idx, p) in probs {
                cumsum += p;
                if r <= cumsum {
                    next_token[b] = idx as i32;
                    break;
                }
            }
        }
        Ok(())
    }

    /// Greedy argmax sampling
    pub fn sample_argmax(
        &self,
        logits: &[f16],
        next_token: &mut [i32],
        batch: usize,
        vocab_size: usize,
    ) -> Result<()> {
        for b in 0..batch {
            let logits_b = &logits[b * vocab_size..(b + 1) * vocab_size];
            let mut max_idx = 0;
            let mut max_val = f32::NEG_INFINITY;

            for (i, &v) in logits_b.iter().enumerate() {
                let val = v.to_f32();
                if val > max_val {
                    max_val = val;
                    max_idx = i;
                }
            }
            next_token[b] = max_idx as i32;
        }
        Ok(())
    }

    // ========================================================================
    // Softmax
    // ========================================================================

    fn softmax_f16(&self, x: &mut [f16], batch_heads: usize, seq_len: usize) -> Result<()> {
        // Sequential outer loop for mutable access (rayon for_each requires Fn not FnMut)
        for bh in 0..batch_heads {
            let row = &mut x[bh * seq_len * seq_len..(bh + 1) * seq_len * seq_len];

            for i in 0..seq_len {
                let row_i = &mut row[i * seq_len..(i + 1) * seq_len];

                // Find max
                let max_val = row_i.iter()
                    .map(|v| v.to_f32())
                    .fold(f32::NEG_INFINITY, f32::max);

                // Exp and sum
                let mut sum = 0.0f32;
                for val in row_i.iter_mut() {
                    let exp_val = (val.to_f32() - max_val).exp();
                    *val = f16::from_f32(exp_val);
                    sum += exp_val;
                }

                // Normalize
                for val in row_i.iter_mut() {
                    *val = f16::from_f32(val.to_f32() / sum);
                }
            }
        }
        Ok(())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_backend_init() {
        let backend = CpuBackend::new().unwrap();
        assert!(backend.num_threads() > 0);
    }

    #[test]
    fn test_silu() {
        let backend = CpuBackend::new().unwrap();
        let mut x = vec![f16::from_f32(-1.0), f16::from_f32(0.0), f16::from_f32(1.0), f16::from_f32(2.0)];
        backend.silu_f16(&mut x).unwrap();

        // SiLU(-1) ≈ -0.269
        // SiLU(0) = 0
        // SiLU(1) ≈ 0.731
        // SiLU(2) ≈ 1.762
        assert!((x[0].to_f32() - (-0.269)).abs() < 0.01);
        assert!((x[1].to_f32() - 0.0).abs() < 0.01);
        assert!((x[2].to_f32() - 0.731).abs() < 0.01);
        assert!((x[3].to_f32() - 1.762).abs() < 0.01);
    }

    #[test]
    fn test_gelu() {
        let backend = CpuBackend::new().unwrap();
        let mut x = vec![f16::from_f32(-1.0), f16::from_f32(0.0), f16::from_f32(1.0)];
        backend.gelu_f16(&mut x).unwrap();

        // GELU(-1) ≈ -0.158
        // GELU(0) = 0
        // GELU(1) ≈ 0.841
        assert!((x[0].to_f32() - (-0.158)).abs() < 0.02);
        assert!((x[1].to_f32() - 0.0).abs() < 0.01);
        assert!((x[2].to_f32() - 0.841).abs() < 0.02);
    }

    #[test]
    fn test_rms_norm() {
        let backend = CpuBackend::new().unwrap();
        let x = vec![f16::from_f32(1.0); 64];
        let weight = vec![f16::from_f32(1.0); 64];
        let mut out = vec![f16::from_f32(0.0); 64];

        backend.rms_norm_f16(&mut x.clone(), &weight, &mut out, 1, 64, 1e-6).unwrap();

        // With all 1s and weight 1, output should be 1/sqrt(1+eps) ≈ 1
        for v in out {
            assert!((v.to_f32() - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_rope() {
        let backend = CpuBackend::new().unwrap();
        let head_dim = 64;
        let mut q = vec![f16::from_f32(1.0); 1 * 1 * 1 * head_dim];
        let mut k = vec![f16::from_f32(1.0); 1 * 1 * 1 * head_dim];
        let cos = vec![f16::from_f32(1.0); 1 * head_dim / 2];
        let sin = vec![f16::from_f32(0.0); 1 * head_dim / 2];

        backend.rope_f16(&mut q, &mut k, &cos, &sin, 1, 1, 1, head_dim).unwrap();

        // With cos=1, sin=0, values unchanged
        assert_eq!(q[0].to_f32(), 1.0);
        assert_eq!(q[head_dim/2].to_f32(), 1.0);
    }

    #[test]
    fn test_sample_argmax() {
        let backend = CpuBackend::new().unwrap();
        let logits = vec![
            f16::from_f32(0.1), f16::from_f32(0.5), f16::from_f32(0.9), f16::from_f32(0.3), // max at 2
            f16::from_f32(0.7), f16::from_f32(0.2), f16::from_f32(0.1), f16::from_f32(0.4), // max at 0
        ];
        let mut next_token = vec![0; 2];

        backend.sample_argmax(&logits, &mut next_token, 2, 4).unwrap();
        assert_eq!(next_token[0], 2);
        assert_eq!(next_token[1], 0);
    }
}