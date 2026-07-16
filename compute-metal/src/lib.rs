//! Apple Silicon Metal Compute Backend
//!
//! This crate provides the Apple GPU acceleration layer using Metal Performance Shaders (MPS)
//! and the Accelerate framework to leverage Unified Memory Architecture (UMA).
//!
//! Note: Metal requires macOS 11+ and M-series chips for optimal performance.

use anyhow::{anyhow, Context, Result};
use half::f16;
use metal::{Device, CommandQueue, Buffer, MTLResourceOptions, MTLSize};
use objc::runtime::YES;
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Metal device properties
#[derive(Debug, Clone)]
pub struct MetalDevice {
    pub name: String,
    pub is_unified_memory: bool,
    pub max_threads_per_threadgroup: MTLSize,
    pub recommended_max_workgroup_size: usize,
}

/// Metal backend for Apple Silicon
pub struct MetalBackend {
    device: Device,
    command_queue: CommandQueue,
    device_info: MetalDevice,
    pipelines: MetalPipelineRegistry,
    buffers: BufferPool,
}

/// Compiled compute pipelines
struct MetalPipelineRegistry {
    gemm: metal::ComputePipelineState,
    batched_gemm: metal::ComputePipelineState,
    attention: metal::ComputePipelineState,
    silu: metal::ComputePipelineState,
    gelu: metal::ComputePipelineState,
    relu: metal::ComputePipelineState,
    add_bias: metal::ComputePipelineState,
    scale: metal::ComputePipelineState,
    rms_norm: metal::ComputePipelineState,
    dequant_q4km: metal::ComputePipelineState,
    gemm_q4km: metal::ComputePipelineState,
    rope: metal::ComputePipelineState,
    sample_topk_topp: metal::ComputePipelineState,
    sample_argmax: metal::ComputePipelineState,
}

/// Buffer pool for efficient allocation
struct BufferPool {
    device: Device,
    buffers: std::collections::VecDeque<Buffer>,
}

impl MetalBackend {
    /// Initialize Metal backend
    pub fn new() -> Result<Self> {
        // Get default Metal device
        let device = Device::system_default()
            .ok_or_else(|| anyhow!("No Metal device available"))?;

        let is_unified_memory = device.has_unified_memory();
        info!("Metal device: {} (unified memory: {})", device.name(), is_unified_memory);

        // Create command queue
        let command_queue = device.new_command_queue();

        // Get device properties
        let device_info = MetalDevice {
            name: device.name().to_string(),
            is_unified_memory,
            max_threads_per_threadgroup: device.max_threads_per_threadgroup(),
            recommended_max_workgroup_size: 256,
        };

        // Load pipelines
        let pipelines = MetalPipelineRegistry::load(&device)?;

        // Create buffer pool
        let buffers = BufferPool::new(device.clone());

        Ok(Self {
            device,
            command_queue,
            device_info,
            pipelines,
            buffers,
        })
    }

    pub fn device_info(&self) -> &MetalDevice {
        &self.device_info
    }

    /// Synchronize GPU
    pub fn synchronize(&self) -> Result<()> {
        let buffer = self.command_queue.new_command_buffer();
        buffer.commit();
        buffer.wait_until_completed();
        Ok(())
    }

    /// Allocate buffer (uses pool for reuse)
    pub fn alloc_buffer(&self, size: usize, options: MTLResourceOptions) -> Buffer {
        if let Some(buf) = self.buffers.get(size, options) {
            buf
        } else {
            self.device.new_buffer(size as u64, options)
        }
    }

    /// Allocate buffer from data
    pub fn buffer_from_slice<T>(&self, data: &[T], options: MTLResourceOptions) -> Buffer {
        let size = std::mem::size_of_val(data);
        let buffer = self.alloc_buffer(size, options);
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                buffer.contents() as *mut u8,
                size,
            );
        }
        buffer
    }

    // ========================================================================
    // GEMM Operations
    // ========================================================================

    /// GEMM using MPSMatrixMultiplication
    pub fn gemm_f16(
        &self,
        a: &Buffer,  // [M, K] row-major
        b: &Buffer,  // [K, N] row-major
        c: &mut Buffer, // [M, N] row-major
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        // Use MPSMatrixMultiplication for optimal performance
        // MPS expects column-major, so we transpose the operation:
        // C_row = A_row * B_row = (B_col * A_col)^T
        // We compute C_col = B_col * A_col using MPS, then transpose

        let buffer = self.command_queue.new_command_buffer();
        let encoder = buffer.compute_command_encoder();

        // For now, use custom kernel (MPS matrix multiply setup is complex)
        encoder.set_compute_pipeline_state(&self.pipelines.gemm);
        encoder.set_buffer(0, Some(a), 0);
        encoder.set_buffer(1, Some(b), 0);
        encoder.set_buffer(2, Some(c), 0);

        let threads = MTLSize::new(16, 16, 1);
        let grid = MTLSize::new(
            (n + 15) / 16,
            (m + 15) / 16,
            1,
        );

        encoder.dispatch_threadgroups(grid, threads);
        encoder.end_encoding();

        buffer.commit();
        buffer.wait_until_completed();
        Ok(())
    }

    /// Batched GEMM for attention
    pub fn batched_gemm_f16(
        &self,
        a: &Buffer,   // [batch, seq_len, head_dim]
        b: &Buffer,   // [batch, seq_len, head_dim]
        c: &mut Buffer, // [batch, seq_len, seq_len]
        batch: usize,
        seq_len: usize,
        head_dim: usize,
    ) -> Result<()> {
        let buffer = self.command_queue.new_command_buffer();
        let encoder = buffer.compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipelines.batched_gemm);
        encoder.set_buffer(0, Some(a), 0);
        encoder.set_buffer(1, Some(b), 0);
        encoder.set_buffer(2, Some(c), 0);
        encoder.set_bytes(3, std::mem::size_of::<u32>(), &batch as *const _ as *const _);
        encoder.set_bytes(4, std::mem::size_of::<u32>(), &seq_len as *const _ as *const _);
        encoder.set_bytes(5, std::mem::size_of::<u32>(), &head_dim as *const _ as *const _);

        let threads = MTLSize::new(16, 16, 1);
        let grid = MTLSize::new(
            (batch * seq_len * seq_len + 255) / 256,
            1,
            1,
        );

        encoder.dispatch_threads(grid, threads);
        encoder.end_encoding();

        buffer.commit();
        buffer.wait_until_completed();
        Ok(())
    }

    // ========================================================================
    // Attention
    // ========================================================================

    /// Scaled dot-product attention
    pub fn attention_f16(
        &self,
        q: &Buffer,   // [batch, heads, seq_len, head_dim]
        k: &Buffer,
        v: &Buffer,
        out: &mut Buffer,
        batch: usize,
        heads: usize,
        seq_len: usize,
        head_dim: usize,
    ) -> Result<()> {
        // Step 1: Q @ K^T -> scores
        let scores_size = batch * heads * seq_len * seq_len;
        let mut scores = self.alloc_buffer(
            scores_size * std::mem::size_of::<f16>(),
            MTLResourceOptions::StorageModeShared,
        );

        self.batched_gemm_f16(q, k, &mut scores, batch * heads, seq_len, head_dim)?;

        // Step 2: Scale
        let scale = (head_dim as f32).sqrt().recip();
        self.scale_f16(&mut scores, scale, scores_size)?;

        // Step 3: Softmax
        self.softmax_f16(&mut scores, batch * heads, seq_len)?;

        // Step 4: scores @ V
        self.batched_gemm_f16(&scores, v, out, batch * heads, seq_len, head_dim)?;

        Ok(())
    }

    // ========================================================================
    // Activations
    // ========================================================================

    pub fn silu_f16(&self, x: &mut Buffer, n: usize) -> Result<()> {
        let buffer = self.command_queue.new_command_buffer();
        let encoder = buffer.compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipelines.silu);
        encoder.set_buffer(0, Some(x), 0);
        encoder.set_bytes(1, std::mem::size_of::<u32>(), &n as *const _ as *const _);

        let threads = MTLSize::new(256, 1, 1);
        let grid = MTLSize::new((n + 255) / 256, 1, 1);

        encoder.dispatch_threads(grid, threads);
        encoder.end_encoding();

        buffer.commit();
        buffer.wait_until_completed();
        Ok(())
    }

    pub fn gelu_f16(&self, x: &mut Buffer, n: usize) -> Result<()> {
        let buffer = self.command_queue.new_command_buffer();
        let encoder = buffer.compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipelines.gelu);
        encoder.set_buffer(0, Some(x), 0);
        encoder.set_bytes(1, std::mem::size_of::<u32>(), &n as *const _ as *const _);

        let threads = MTLSize::new(256, 1, 1);
        let grid = MTLSize::new((n + 255) / 256, 1, 1);

        encoder.dispatch_threads(grid, threads);
        encoder.end_encoding();

        buffer.commit();
        buffer.wait_until_completed();
        Ok(())
    }

    pub fn relu_f16(&self, x: &mut Buffer, n: usize) -> Result<()> {
        let buffer = self.command_queue.new_command_buffer();
        let encoder = buffer.compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipelines.relu);
        encoder.set_buffer(0, Some(x), 0);
        encoder.set_bytes(1, std::mem::size_of::<u32>(), &n as *const _ as *const _);

        let threads = MTLSize::new(256, 1, 1);
        let grid = MTLSize::new((n + 255) / 256, 1, 1);

        encoder.dispatch_threads(grid, threads);
        encoder.end_encoding();

        buffer.commit();
        buffer.wait_until_completed();
        Ok(())
    }

    // ========================================================================
    // Element-wise
    // ========================================================================

    pub fn add_bias_f16(&self, x: &mut Buffer, bias: &Buffer, n: usize, bias_size: usize) -> Result<()> {
        let buffer = self.command_queue.new_command_buffer();
        let encoder = buffer.compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipelines.add_bias);
        encoder.set_buffer(0, Some(x), 0);
        encoder.set_buffer(1, Some(bias), 0);
        encoder.set_bytes(2, std::mem::size_of::<u32>(), &n as *const _ as *const _);
        encoder.set_bytes(3, std::mem::size_of::<u32>(), &bias_size as *const _ as *const _);

        let threads = MTLSize::new(256, 1, 1);
        let grid = MTLSize::new((n + 255) / 256, 1, 1);

        encoder.dispatch_threads(grid, threads);
        encoder.end_encoding();

        buffer.commit();
        buffer.wait_until_completed();
        Ok(())
    }

    pub fn scale_f16(&self, x: &mut Buffer, scale: f32, n: usize) -> Result<()> {
        let buffer = self.command_queue.new_command_buffer();
        let encoder = buffer.compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipelines.scale);
        encoder.set_buffer(0, Some(x), 0);
        encoder.set_bytes(1, std::mem::size_of::<f32>(), &scale as *const _ as *const _);
        encoder.set_bytes(2, std::mem::size_of::<u32>(), &n as *const _ as *const _);

        let threads = MTLSize::new(256, 1, 1);
        let grid = MTLSize::new((n + 255) / 256, 1, 1);

        encoder.dispatch_threads(grid, threads);
        encoder.end_encoding();

        buffer.commit();
        buffer.wait_until_completed();
        Ok(())
    }

    pub fn rms_norm_f16(
        &self,
        x: &mut Buffer,
        weight: &Buffer,
        out: &mut Buffer,
        batch_seq: usize,
        hidden_dim: usize,
        eps: f32,
    ) -> Result<()> {
        let buffer = self.command_queue.new_command_buffer();
        let encoder = buffer.compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipelines.rms_norm);
        encoder.set_buffer(0, Some(x), 0);
        encoder.set_buffer(1, Some(weight), 0);
        encoder.set_buffer(2, Some(out), 0);
        encoder.set_bytes(3, std::mem::size_of::<u32>(), &batch_seq as *const _ as *const _);
        encoder.set_bytes(4, std::mem::size_of::<u32>(), &hidden_dim as *const _ as *const _);
        encoder.set_bytes(5, std::mem::size_of::<f32>(), &eps as *const _ as *const _);

        let threads = MTLSize::new(256, 1, 1);
        let grid = MTLSize::new((batch_seq + 255) / 256, 1, 1);

        encoder.dispatch_threads(grid, threads);
        encoder.end_encoding();

        buffer.commit();
        buffer.wait_until_completed();
        Ok(())
    }

    // ========================================================================
    // Quantized Operations
    // ========================================================================

    pub fn dequantize_q4km_f16(
        &self,
        q_weights: &Buffer,
        scales: &Buffer,
        out: &mut Buffer,
        num_elements: usize,
    ) -> Result<()> {
        let buffer = self.command_queue.new_command_buffer();
        let encoder = buffer.compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipelines.dequant_q4km);
        encoder.set_buffer(0, Some(q_weights), 0);
        encoder.set_buffer(1, Some(scales), 0);
        encoder.set_buffer(2, Some(out), 0);
        encoder.set_bytes(3, std::mem::size_of::<u32>(), &num_elements as *const _ as *const _);

        let threads = MTLSize::new(256, 1, 1);
        let grid = MTLSize::new((num_elements + 255) / 256, 1, 1);

        encoder.dispatch_threads(grid, threads);
        encoder.end_encoding();

        buffer.commit();
        buffer.wait_until_completed();
        Ok(())
    }

    pub fn gemm_q4km_f16(
        &self,
        a_q: &Buffer,
        a_scales: &Buffer,
        b: &Buffer,
        c: &mut Buffer,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        let buffer = self.command_queue.new_command_buffer();
        let encoder = buffer.compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipelines.gemm_q4km);
        encoder.set_buffer(0, Some(a_q), 0);
        encoder.set_buffer(1, Some(a_scales), 0);
        encoder.set_buffer(2, Some(b), 0);
        encoder.set_buffer(3, Some(c), 0);
        encoder.set_bytes(4, std::mem::size_of::<u32>(), &m as *const _ as *const _);
        encoder.set_bytes(5, std::mem::size_of::<u32>(), &n as *const _ as *const _);
        encoder.set_bytes(6, std::mem::size_of::<u32>(), &k as *const _ as *const _);

        let threads = MTLSize::new(16, 16, 1);
        let grid = MTLSize::new((n + 15) / 16, (m + 15) / 16, 1);

        encoder.dispatch_threadgroups(grid, threads);
        encoder.end_encoding();

        buffer.commit();
        buffer.wait_until_completed();
        Ok(())
    }

    // ========================================================================
    // RoPE
    // ========================================================================

    pub fn rope_f16(
        &self,
        q: &mut Buffer,
        k: &mut Buffer,
        cos: &Buffer,
        sin: &Buffer,
        batch: usize,
        heads: usize,
        seq_len: usize,
        head_dim: usize,
    ) -> Result<()> {
        let buffer = self.command_queue.new_command_buffer();
        let encoder = buffer.compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipelines.rope);
        encoder.set_buffer(0, Some(q), 0);
        encoder.set_buffer(1, Some(k), 0);
        encoder.set_buffer(2, Some(cos), 0);
        encoder.set_buffer(3, Some(sin), 0);
        encoder.set_bytes(4, std::mem::size_of::<u32>(), &batch as *const _ as *const _);
        encoder.set_bytes(5, std::mem::size_of::<u32>(), &heads as *const _ as *const _);
        encoder.set_bytes(6, std::mem::size_of::<u32>(), &seq_len as *const _ as *const _);
        encoder.set_bytes(7, std::mem::size_of::<u32>(), &head_dim as *const _ as *const _);

        let total = batch * heads * seq_len * (head_dim / 2);
        let threads = MTLSize::new(256, 1, 1);
        let grid = MTLSize::new((total + 255) / 256, 1, 1);

        encoder.dispatch_threads(grid, threads);
        encoder.end_encoding();

        buffer.commit();
        buffer.wait_until_completed();
        Ok(())
    }

    // ========================================================================
    // Sampling
    // ========================================================================

    pub fn sample_topk_topp(
        &self,
        logits: &Buffer,
        next_token: &mut Buffer,
        batch: usize,
        vocab_size: usize,
        top_k: usize,
        top_p: f32,
        temperature: f32,
        seed: u64,
    ) -> Result<()> {
        let buffer = self.command_queue.new_command_buffer();
        let encoder = buffer.compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipelines.sample_topk_topp);
        encoder.set_buffer(0, Some(logits), 0);
        encoder.set_buffer(1, Some(next_token), 0);
        encoder.set_bytes(2, std::mem::size_of::<u32>(), &batch as *const _ as *const _);
        encoder.set_bytes(3, std::mem::size_of::<u32>(), &vocab_size as *const _ as *const _);
        encoder.set_bytes(4, std::mem::size_of::<u32>(), &top_k as *const _ as *const _);
        encoder.set_bytes(5, std::mem::size_of::<f32>(), &top_p as *const _ as *const _);
        encoder.set_bytes(6, std::mem::size_of::<f32>(), &temperature as *const _ as *const _);
        encoder.set_bytes(7, std::mem::size_of::<u64>(), &seed as *const _ as *const _);

        let threads = MTLSize::new(256, 1, 1);
        let grid = MTLSize::new((batch + 255) / 256, 1, 1);

        encoder.dispatch_threads(grid, threads);
        encoder.end_encoding();

        buffer.commit();
        buffer.wait_until_completed();
        Ok(())
    }

    pub fn sample_argmax(
        &self,
        logits: &Buffer,
        next_token: &mut Buffer,
        batch: usize,
        vocab_size: usize,
    ) -> Result<()> {
        let buffer = self.command_queue.new_command_buffer();
        let encoder = buffer.compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipelines.sample_argmax);
        encoder.set_buffer(0, Some(logits), 0);
        encoder.set_buffer(1, Some(next_token), 0);
        encoder.set_bytes(2, std::mem::size_of::<u32>(), &batch as *const _ as *const _);
        encoder.set_bytes(3, std::mem::size_of::<u32>(), &vocab_size as *const _ as *const _);

        let threads = MTLSize::new(256, 1, 1);
        let grid = MTLSize::new((batch + 255) / 256, 1, 1);

        encoder.dispatch_threads(grid, threads);
        encoder.end_encoding();

        buffer.commit();
        buffer.wait_until_completed();
        Ok(())
    }

    // ========================================================================
    // Softmax (internal)
    // ========================================================================

    fn softmax_f16(&self, x: &mut Buffer, batch_heads: usize, seq_len: usize) -> Result<()> {
        // Implement softmax kernel
        // For simplicity, this would be a separate pipeline
        // In practice, use MPS softmax or custom kernel
        unimplemented!("Softmax kernel not yet implemented")
    }
}

// ============================================================================
// Pipeline Registry
// ============================================================================

impl MetalPipelineRegistry {
    fn load(device: &Device) -> Result<Self> {
        let library = device.new_library_with_source(include_str!("kernels.metal"))?;

        macro_rules! pipeline {
            ($name:expr) => {
                library.get_function($name, None)
                    .and_then(|f| device.new_compute_pipeline_state_with_function(&f))
                    .with_context(|| format!("Failed to create pipeline: {}", $name))?
            };
        }

        Ok(Self {
            gemm: pipeline!("gemm_f16"),
            batched_gemm: pipeline!("batched_gemm_f16"),
            attention: pipeline!("attention_f16"),
            silu: pipeline!("silu_f16"),
            gelu: pipeline!("gelu_f16"),
            relu: pipeline!("relu_f16"),
            add_bias: pipeline!("add_bias_f16"),
            scale: pipeline!("scale_f16"),
            rms_norm: pipeline!("rms_norm_f16"),
            dequant_q4km: pipeline!("dequant_q4km_f16"),
            gemm_q4km: pipeline!("gemm_q4km_f16"),
            rope: pipeline!("rope_f16"),
            sample_topk_topp: pipeline!("sample_topk_topp"),
            sample_argmax: pipeline!("sample_argmax"),
        })
    }
}

// ============================================================================
// Buffer Pool
// ============================================================================

impl BufferPool {
    fn new(device: Device) -> Self {
        Self {
            device,
            buffers: std::collections::VecDeque::new(),
        }
    }

    fn get(&mut self, size: usize, options: MTLResourceOptions) -> Option<Buffer> {
        self.buffers.iter().position(|b| b.length() >= size as u64)
            .map(|i| self.buffers.remove(i).unwrap())
    }

    fn return_buffer(&mut self, buffer: Buffer) {
        if self.buffers.len() < 32 {
            self.buffers.push_back(buffer);
        }
    }
}

impl Drop for BufferPool {
    fn drop(&mut self) {
        // Buffers are automatically released
    }
}

// ============================================================================
// Metal Kernel Source
// ============================================================================

// Kernel source is in compute-metal/src/kernels.metal
// This is included at compile time via include_str!

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires macOS with Metal"]
    fn test_metal_init() {
        let backend = MetalBackend::new().unwrap();
        assert!(backend.synchronize().is_ok());
    }
}