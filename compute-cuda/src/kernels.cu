// CUDA Kernels for DeCoupled-AI
// Compiled to PTX with nvcc --ptx

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// GEMM Kernels
// ============================================================================

// Simple GEMM kernel for FP16
__global__ void gemm_f16_kernel(
    const half* __restrict__ A,  // [M, K]
    const half* __restrict__ B,  // [K, N]
    half* __restrict__ C,        // [M, N]
    int M, int N, int K,
    float alpha, float beta
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        float a = __half2float(A[row * K + k]);
        float b = __half2float(B[k * N + col]);
        sum += a * b;
    }

    C[row * N + col] = __float2half_rn(alpha * sum + beta * __half2float(C[row * N + col]));
}

// Batched GEMM for attention scores
__global__ void batched_gemm_f16_kernel(
    const half* __restrict__ A,  // [batch, seq_len, head_dim]
    const half* __restrict__ B,  // [batch, seq_len, head_dim]
    half* __restrict__ C,        // [batch, seq_len, seq_len]
    int batch, int seq_len, int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seq_len * seq_len;
    if (idx >= total) return;

    int b = idx / (seq_len * seq_len);
    int residual = idx % (seq_len * seq_len);
    int i = residual / seq_len;
    int j = residual % seq_len;

    float sum = 0.0f;
    const half* A_b = A + b * seq_len * head_dim;
    const half* B_b = B + b * seq_len * head_dim;

    for (int k = 0; k < head_dim; ++k) {
        sum += __half2float(A_b[i * head_dim + k]) * __half2float(B_b[j * head_dim + k]);
    }

    C[idx] = __float2half_rn(sum);
}

// ============================================================================
// Flash Attention Kernel (Simplified)
// ============================================================================

__global__ void flash_attention_f16_kernel(
    const half* __restrict__ Q,   // [batch, heads, seq_len, head_dim]
    const half* __restrict__ K,   // [batch, heads, seq_len, head_dim]
    const half* __restrict__ V,   // [batch, heads, seq_len, head_dim]
    half* __restrict__ O,         // [batch, heads, seq_len, head_dim]
    int batch, int heads, int seq_len, int head_dim,
    float scale
) {
    // Simplified - each thread handles one output element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * heads * seq_len * head_dim;
    if (idx >= total) return;

    // This is a placeholder - real flash attention is much more complex
    // requiring shared memory, warp-level primitives, etc.
    int b = idx / (heads * seq_len * head_dim);
    int residual = idx % (heads * seq_len * head_dim);
    int h = residual / (seq_len * head_dim);
    residual = residual % (seq_len * head_dim);
    int i = residual / head_dim;
    int d = residual % head_dim;

    // Compute Q[i] * K^T
    const half* Q_bh = Q + ((b * heads + h) * seq_len + i) * head_dim;
    const half* K_bh = K + (b * heads + h) * seq_len * head_dim;
    const half* V_bh = V + (b * heads + h) * seq_len * head_dim;
    half* O_bh = O + ((b * heads + h) * seq_len + i) * head_dim;

    float max_val = -INFINITY;
    float scores[1024]; // Max seq_len

    // Compute scores
    for (int j = 0; j < seq_len; ++j) {
        float sum = 0.0f;
        for (int k = 0; k < head_dim; ++k) {
            sum += __half2float(Q_bh[k]) * __half2float(K_bh[j * head_dim + k]);
        }
        scores[j] = sum * scale;
        if (scores[j] > max_val) max_val = scores[j];
    }

    // Softmax
    float sum_exp = 0.0f;
    for (int j = 0; j < seq_len; ++j) {
        scores[j] = expf(scores[j] - max_val);
        sum_exp += scores[j];
    }
    for (int j = 0; j < seq_len; ++j) {
        scores[j] /= sum_exp;
    }

    // Compute output
    float out_val = 0.0f;
    for (int j = 0; j < seq_len; ++j) {
        out_val += scores[j] * __half2float(V_bh[j * head_dim + d]);
    }

    O_bh[d] = __float2half_rn(out_val);
}

// ============================================================================
// Activation Kernels
// ============================================================================

// SiLU / Swish: x * sigmoid(x)
__global__ void silu_f16_kernel(half* x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float val = __half2float(x[idx]);
    float sigmoid = 1.0f / (1.0f + expf(-val));
    x[idx] = __float2half_rn(val * sigmoid);
}

// GELU
__global__ void gelu_f16_kernel(half* x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float val = __half2float(x[idx]);
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coef = 0.044715f;
    float x3 = val * val * val;
    float tanh_arg = sqrt_2_over_pi * (val + coef * x3);
    float gelu = 0.5f * val * (1.0f + tanhf(tanh_arg));
    x[idx] = __float2half_rn(gelu);
}

// ReLU
__global__ void relu_f16_kernel(half* x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float val = __half2float(x[idx]);
    x[idx] = __float2half_rn(fmaxf(0.0f, val));
}

// ============================================================================
// Element-wise Operations
// ============================================================================

// Add bias (broadcast over last dim)
__global__ void add_bias_f16_kernel(half* x, const half* bias, int n, int bias_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int bias_idx = idx % bias_size;
    float x_val = __half2float(x[idx]);
    float b_val = __half2float(bias[bias_idx]);
    x[idx] = __float2half_rn(x_val + b_val);
}

// Scale
__global__ void scale_f16_kernel(half* x, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    x[idx] = __float2half_rn(__half2float(x[idx]) * scale);
}

// Add
__global__ void add_f16_kernel(half* a, const half* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    a[idx] = __float2half_rn(__half2float(a[idx]) + __half2float(b[idx]));
}

// ============================================================================
// RMSNorm Kernel
// ============================================================================

__global__ void rms_norm_f16_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    half* __restrict__ output,
    int batch_seq, int hidden_dim, float eps
) {
    extern __shared__ float s_data[];

    int row = blockIdx.x;
    if (row >= batch_seq) return;

    const half* x = input + row * hidden_dim;
    half* out = output + row * hidden_dim;

    // Compute mean of squares (using warp reduce)
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        float val = __half2float(x[i]);
        sum_sq += val * val;
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
    }

    // First thread in warp has the sum
    if (threadIdx.x % 32 == 0) {
        s_data[threadIdx.x / 32] = sum_sq;
    }
    __syncthreads();

    // First warp reduces across warps
    if (threadIdx.x < 32) {
        float val = (threadIdx.x < (blockDim.x + 31) / 32) ? s_data[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        if (threadIdx.x == 0) {
            s_data[0] = val;
        }
    }
    __syncthreads();

    float mean_sq = s_data[0] / hidden_dim;
    float rms = rsqrtf(mean_sq + eps);

    // Normalize and scale
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        float val = __half2float(x[i]) * rms;
        val *= __half2float(weight[i]);
        out[i] = __float2half_rn(val);
    }
}

// ============================================================================
// Quantization Kernels
// ============================================================================

// Dequantize Q4_K_M to FP16
__global__ void dequantize_q4km_f16_kernel(
    const uint8_t* __restrict__ q_weights,
    const half* __restrict__ scales,
    half* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Q4_K_M: 256 elements per block
    // Each block has: 32 scales, 32 mins, 128 packed weights (4-bit), 32 high bits
    // Simplified: assume scales are per 8 elements
    int block_idx = idx / 256;
    int elem_idx = idx % 256;
    int scale_idx = block_idx * 32 + elem_idx / 8;

    // Extract 4-bit weight
    int byte_idx = block_idx * 256 + elem_idx / 2;
    uint8_t packed = q_weights[byte_idx];
    uint8_t nibble = (elem_idx % 2 == 0) ? (packed & 0x0F) : (packed >> 4);

    // Dequantize
    float scale = __half2float(scales[scale_idx]);
    float val = (nibble - 8) * scale; // Centered at 8
    output[idx] = __float2half_rn(val);
}

// Quantized GEMM: Q4_K_M x FP16 -> FP16
__global__ void gemm_q4km_f16_kernel(
    const uint8_t* __restrict__ A_q,     // [M, K] quantized
    const half* __restrict__ A_scales,   // [M, K/256 * 32]
    const half* __restrict__ B,          // [K, N] fp16
    half* __restrict__ C,                // [M, N]
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        // Dequantize A[row, k] on the fly
        int block_idx = k / 256;
        int elem_idx = k % 256;
        int scale_idx = row * (K / 256 * 32) + block_idx * 32 + elem_idx / 8;

        int byte_idx = row * K + block_idx * 256 + elem_idx / 2;
        uint8_t packed = A_q[byte_idx];
        uint8_t nibble = (elem_idx % 2 == 0) ? (packed & 0x0F) : (packed >> 4);

        float a_val = (nibble - 8) * __half2float(A_scales[scale_idx]);
        float b_val = __half2float(B[k * N + col]);
        sum += a_val * b_val;
    }

    C[row * N + col] = __float2half_rn(sum);
}

// ============================================================================
// RoPE Kernel
// ============================================================================

__global__ void rope_f16_kernel(
    half* __restrict__ Q,       // [batch, heads, seq_len, head_dim]
    half* __restrict__ K,       // [batch, heads, seq_len, head_dim]
    const half* __restrict__ cos, // [seq_len, head_dim/2]
    const half* __restrict__ sin, // [seq_len, head_dim/2]
    int batch, int heads, int seq_len, int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = head_dim / 2;
    int total = batch * heads * seq_len * half_dim;
    if (idx >= total) return;

    int b = idx / (heads * seq_len * half_dim);
    int residual = idx % (heads * seq_len * half_dim);
    int h = residual / (seq_len * half_dim);
    residual = residual % (seq_len * half_dim);
    int s = residual / half_dim;
    int d = residual % half_dim;

    int qk_base = ((b * heads + h) * seq_len + s) * head_dim;

    float q1 = __half2float(Q[qk_base + d]);
    float q2 = __half2float(Q[qk_base + d + half_dim]);
    float k1 = __half2float(K[qk_base + d]);
    float k2 = __half2float(K[qk_base + d + half_dim]);

    float cos_val = __half2float(cos[s * half_dim + d]);
    float sin_val = __half2float(sin[s * half_dim + d]);

    // Apply rotation
    Q[qk_base + d] = __float2half_rn(q1 * cos_val - q2 * sin_val);
    Q[qk_base + d + half_dim] = __float2half_rn(q1 * sin_val + q2 * cos_val);
    K[qk_base + d] = __float2half_rn(k1 * cos_val - k2 * sin_val);
    K[qk_base + d + half_dim] = __float2half_rn(k1 * sin_val + k2 * cos_val);
}

// ============================================================================
// Sampling Kernels
// ============================================================================

// Argmax sampling
__global__ void sample_argmax_kernel(
    const half* __restrict__ logits,  // [batch, vocab_size]
    int* __restrict__ output,         // [batch]
    int batch, int vocab_size
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) return;

    const half* logits_b = logits + b * vocab_size;

    float max_val = -INFINITY;
    int max_idx = 0;

    for (int i = 0; i < vocab_size; ++i) {
        float val = __half2float(logits_b[i]);
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }

    output[b] = max_idx;
}

// Top-k top-p sampling (simplified - each block handles one batch)
__global__ void sample_topk_topp_kernel(
    const half* __restrict__ logits,   // [batch, vocab_size]
    int* __restrict__ output,          // [batch]
    int batch, int vocab_size,
    int top_k, float top_p, float temperature,
    unsigned long long seed
) {
    int b = blockIdx.x;
    if (b >= batch) return;

    // This is a simplified version - real implementation would use
    // parallel sorting and prefix sum for top-k/top-p
    extern __shared__ float s_logits[];
    half* s_logits_h = (half*)s_logits;

    const half* logits_b = logits + b * vocab_size;

    // Load logits to shared memory and apply temperature
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        s_logits_h[i] = __float2half_rn(__half2float(logits_b[i]) / temperature);
    }
    __syncthreads();

    // Softmax (simplified - single thread does it)
    if (threadIdx.x == 0) {
        float max_val = -INFINITY;
        for (int i = 0; i < vocab_size; ++i) {
            float val = __half2float(s_logits_h[i]);
            if (val > max_val) max_val = val;
        }

        float sum_exp = 0.0f;
        for (int i = 0; i < vocab_size; ++i) {
            float val = expf(__half2float(s_logits_h[i]) - max_val);
            s_logits_h[i] = __float2half_rn(val);
            sum_exp += val;
        }

        for (int i = 0; i < vocab_size; ++i) {
            s_logits_h[i] = __float2half_rn(__half2float(s_logits_h[i]) / sum_exp);
        }

        // Top-k: find top k indices
        // For simplicity, just do argmax
        float max_p = 0.0f;
        int max_idx = 0;
        for (int i = 0; i < vocab_size; ++i) {
            float p = __half2float(s_logits_h[i]);
            if (p > max_p) {
                max_p = p;
                max_idx = i;
            }
        }

        output[b] = max_idx;
    }
}

// Softmax kernel
__global__ void softmax_f16_kernel(
    half* __restrict__ x,  // [batch_heads, seq_len, seq_len]
    int batch_heads, int seq_len
) {
    int bh = blockIdx.x;
    if (bh >= batch_heads) return;

    half* row = x + bh * seq_len * seq_len;

    // Each thread handles one row of the attention matrix
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        half* row_i = row + i * seq_len;

        // Find max
        float max_val = -INFINITY;
        for (int j = 0; j < seq_len; ++j) {
            float val = __half2float(row_i[j]);
            if (val > max_val) max_val = val;
        }

        // Exp and sum
        float sum_exp = 0.0f;
        for (int j = 0; j < seq_len; ++j) {
            float val = expf(__half2float(row_i[j]) - max_val);
            row_i[j] = __float2half_rn(val);
            sum_exp += val;
        }

        // Normalize
        for (int j = 0; j < seq_len; ++j) {
            row_i[j] = __float2half_rn(__half2float(row_i[j]) / sum_exp);
        }
    }
}

// ============================================================================
// Launch Functions (called from Rust)
// ============================================================================

extern "C" void launch_gemm_f16(
    const half* A, const half* B, half* C,
    int M, int N, int K, float alpha, float beta
) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    gemm_f16_kernel<<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
}

extern "C" void launch_batched_gemm_f16(
    const half* A, const half* B, half* C,
    int batch, int seq_len, int head_dim
) {
    int total = batch * seq_len * seq_len;
    int block = 256;
    int grid = (total + block - 1) / block;
    batched_gemm_f16_kernel<<<grid, block>>>(A, B, C, batch, seq_len, head_dim);
}

extern "C" void launch_flash_attention_f16(
    const half* Q, const half* K, const half* V, half* O,
    int batch, int heads, int seq_len, int head_dim, float scale
) {
    int total = batch * heads * seq_len * head_dim;
    int block = 256;
    int grid = (total + block - 1) / block;
    flash_attention_f16_kernel<<<grid, block>>>(Q, K, V, O, batch, heads, seq_len, head_dim, scale);
}

extern "C" void launch_silu_f16(half* x, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    silu_f16_kernel<<<grid, block>>>(x, n);
}

extern "C" void launch_gelu_f16(half* x, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    gelu_f16_kernel<<<grid, block>>>(x, n);
}

extern "C" void launch_relu_f16(half* x, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    relu_f16_kernel<<<grid, block>>>(x, n);
}

extern "C" void launch_add_bias_f16(half* x, const half* bias, int n, int bias_size) {
    int block = 256;
    int grid = (n + block - 1) / block;
    add_bias_f16_kernel<<<grid, block>>>(x, bias, n, bias_size);
}

extern "C" void launch_scale_f16(half* x, float scale, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    scale_f16_kernel<<<grid, block>>>(x, scale, n);
}

extern "C" void launch_add_f16(half* a, const half* b, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    add_f16_kernel<<<grid, block>>>(a, b, n);
}

extern "C" void launch_rms_norm_f16(
    const half* input, const half* weight, half* output,
    int batch_seq, int hidden_dim, float eps
) {
    int block = 256;
    int grid = batch_seq;
    size_t shared_mem = ((block + 31) / 32) * sizeof(float);
    rms_norm_f16_kernel<<<grid, block, shared_mem>>>(input, weight, output, batch_seq, hidden_dim, eps);
}

extern "C" void launch_dequantize_q4km_f16(
    const uint8_t* q_weights, const half* scales, half* output, int n
) {
    int block = 256;
    int grid = (n + block - 1) / block;
    dequantize_q4km_f16_kernel<<<grid, block>>>(q_weights, scales, output, n);
}

extern "C" void launch_gemm_q4km_f16(
    const uint8_t* A_q, const half* A_scales, const half* B, half* C,
    int M, int N, int K
) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    gemm_q4km_f16_kernel<<<grid, block>>>(A_q, A_scales, B, C, M, N, K);
}

extern "C" void launch_rope_f16(
    half* Q, half* K, const half* cos, const half* sin,
    int batch, int heads, int seq_len, int head_dim
) {
    int total = batch * heads * seq_len * (head_dim / 2);
    int block = 256;
    int grid = (total + block - 1) / block;
    rope_f16_kernel<<<grid, block>>>(Q, K, cos, sin, batch, heads, seq_len, head_dim);
}

extern "C" void launch_sample_argmax(
    const half* logits, int* output, int batch, int vocab_size
) {
    int block = 256;
    int grid = (batch + block - 1) / block;
    sample_argmax_kernel<<<grid, block>>>(logits, output, batch, vocab_size);
}

extern "C" void launch_sample_topk_topp(
    const half* logits, int* output, int batch, int vocab_size,
    int top_k, float top_p, float temperature, unsigned long long seed
) {
    int block = 256;
    int grid = batch;
    size_t shared_mem = vocab_size * sizeof(half);
    sample_topk_topp_kernel<<<grid, block, shared_mem>>>(logits, output, batch, vocab_size, top_k, top_p, temperature, seed);
}

extern "C" void launch_softmax_f16(half* x, int batch_heads, int seq_len) {
    int block = 256;
    int grid = batch_heads;
    softmax_f16_kernel<<<grid, block>>>(x, batch_heads, seq_len);
}