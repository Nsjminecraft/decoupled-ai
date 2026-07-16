// HIP Kernels for DeCoupled-AI (AMD ROCm)
// Compiled to HSACO via hipcc

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define WARP_SIZE 64

// ============================================================================
// GEMM Kernel
// ============================================================================

__global__ void gemm_f16_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    const int M, const int N, const int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
    }
    C[row * N + col] = __float2half(sum);
}

// ============================================================================
// Activation Kernels
// ============================================================================

__global__ void silu_f16_kernel(__half* x, const int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float val = __half2float(x[idx]);
    float sigmoid = 1.0f / (1.0f + expf(-val));
    x[idx] = __float2half(val * sigmoid);
}

__global__ void gelu_f16_kernel(__half* x, const int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float val = __half2float(x[idx]);
    float x3 = val * val * val;
    float tanh_arg = 0.7978845608028654f * (val + 0.044715f * x3);
    float gelu = 0.5f * val * (1.0f + tanhf(tanh_arg));
    x[idx] = __float2half(gelu);
}

// ============================================================================
// RMSNorm Kernel with warp-level Reduction
// ============================================================================

__global__ void rms_norm_f16_kernel(
    const __half* __restrict__ input,
    const __half* __restrict__ weight,
    __half* __restrict__ output,
    const int hidden_dim,
    const float eps
) {
    int row = blockIdx.x;
    if (row >= gridDim.x) return;

    const __half* x = input + row * hidden_dim;
    __half* out = output + row * hidden_dim;

    __shared__ float sum_sq_shared;
    if (threadIdx.x == 0) sum_sq_shared = 0.0f;
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        float val = __half2float(x[i]);
        local_sum += val * val;
    }

    // Warp-level reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down(local_sum, offset);
    }

    if (threadIdx.x % WARP_SIZE == 0) {
        atomicAdd(&sum_sq_shared, local_sum);
    }
    __syncthreads();

    float rms = rsqrtf(sum_sq_shared / float(hidden_dim) + eps);

    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        float val = __half2float(x[i]) * rms * __half2float(weight[i]);
        out[i] = __float2half(val);
    }
}

// ============================================================================
// RoPE Kernel
// ============================================================================

__global__ void rope_f16_kernel(
    __half* Q,
    __half* K,
    const __half* __restrict__ cos,
    const __half* __restrict__ sin,
    const int batch, const int heads, const int seq_len, const int head_dim
) {
    int half_dim = head_dim / 2;
    int total = batch * heads * seq_len * half_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int b = idx / (heads * seq_len * half_dim);
    int rem = idx % (heads * seq_len * half_dim);
    int h = rem / (seq_len * half_dim);
    rem = rem % (seq_len * half_dim);
    int s = rem / half_dim;
    int d = rem % half_dim;

    int qk_base = ((b * heads + h) * seq_len + s) * head_dim;
    float q1 = __half2float(Q[qk_base + d]);
    float q2 = __half2float(Q[qk_base + d + half_dim]);
    float k1 = __half2float(K[qk_base + d]);
    float k2 = __half2float(K[qk_base + d + half_dim]);
    float cos_val = __half2float(cos[s * half_dim + d]);
    float sin_val = __half2float(sin[s * half_dim + d]);

    Q[qk_base + d] = __float2half(q1 * cos_val - q2 * sin_val);
    Q[qk_base + d + half_dim] = __float2half(q1 * sin_val + q2 * cos_val);
    K[qk_base + d] = __float2half(k1 * cos_val - k2 * sin_val);
    K[qk_base + d + half_dim] = __float2half(k1 * sin_val + k2 * cos_val);
}

// ============================================================================
// Sampling Kernel
// ============================================================================

__global__ void sample_argmax_kernel(const __half* logits, int* output, int vocab_size) {
    int b = blockIdx.x;
    const __half* logits_b = logits + b * vocab_size;
    float max_val = -INFINITY;
    int max_idx = 0;
    for (int i = 0; i < vocab_size; ++i) {
        float v = __half2float(logits_b[i]);
        if (v > max_val) { max_val = v; max_idx = i; }
    }
    output[b] = max_idx;
}
