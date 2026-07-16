// Metal Shaders for DeCoupled-AI
// Compiled at runtime from this source

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// GEMM Kernels
// ============================================================================

// Simple GEMM: C = A @ B
kernel void gemm_f16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= N || gid.y >= M) return;

    float sum = 0.0;
    for (uint k = 0; k < K; ++k) {
        sum += float(A[gid.y * K + k]) * float(B[k * N + gid.x]);
    }
    C[gid.y * N + gid.x] = half(sum);
}

// Batched GEMM for attention
kernel void batched_gemm_f16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant uint& batch [[buffer(3)]],
    constant uint& seq_len [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    uint idx [[thread_position_in_grid]]
) {
    uint total = batch * seq_len * seq_len;
    if (idx >= total) return;

    uint b = idx / (seq_len * seq_len);
    uint residual = idx % (seq_len * seq_len);
    uint i = residual / seq_len;
    uint j = residual % seq_len;

    float sum = 0.0;
    device const half* A_b = A + b * seq_len * head_dim;
    device const half* B_b = B + b * seq_len * head_dim;

    for (uint k = 0; k < head_dim; ++k) {
        sum += float(A_b[i * head_dim + k]) * float(B_b[j * head_dim + k]);
    }

    C[idx] = half(sum);
}

// ============================================================================
// Activation Kernels
// ============================================================================

kernel void silu_f16(
    device half* x [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= n) return;
    float val = float(x[idx]);
    float sigmoid = 1.0 / (1.0 + exp(-val));
    x[idx] = half(val * sigmoid);
}

kernel void gelu_f16(
    device half* x [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= n) return;
    float val = float(x[idx]);
    float x3 = val * val * val;
    float tanh_arg = 0.7978845608028654 * (val + 0.044715 * x3);
    float gelu = 0.5 * val * (1.0 + tanh(tanh_arg));
    x[idx] = half(gelu);
}

kernel void relu_f16(
    device half* x [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= n) return;
    float val = float(x[idx]);
    x[idx] = half(max(0.0, val));
}

// ============================================================================
// Element-wise Operations
// ============================================================================

kernel void add_bias_f16(
    device half* x [[buffer(0)]],
    device const half* bias [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& bias_size [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= n) return;
    uint bias_idx = idx % bias_size;
    x[idx] = half(float(x[idx]) + float(bias[bias_idx]));
}

kernel void scale_f16(
    device half* x [[buffer(0)]],
    constant float& scale [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= n) return;
    x[idx] = half(float(x[idx]) * scale);
}

// ============================================================================
// RMSNorm Kernel
// ============================================================================

kernel void rms_norm_f16(
    device const half* input [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& batch_seq [[buffer(3)]],
    constant uint& hidden_dim [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= batch_seq) return;

    device const half* x = input + row * hidden_dim;
    device half* out = output + row * hidden_dim;

    // Compute mean of squares
    float sum_sq = 0.0;
    for (uint i = 0; i < hidden_dim; ++i) {
        float val = float(x[i]);
        sum_sq += val * val;
    }

    float rms = rsqrt(sum_sq / float(hidden_dim) + eps);

    // Normalize and scale
    for (uint i = 0; i < hidden_dim; ++i) {
        float val = float(x[i]) * rms * float(weight[i]);
        out[i] = half(val);
    }
}

// ============================================================================
// Attention Kernel (Simplified)
// ============================================================================

kernel void attention_f16(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device half* O [[buffer(3)]],
    constant uint& batch [[buffer(4)]],
    constant uint& heads [[buffer(5)]],
    constant uint& seq_len [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    uint idx [[thread_position_in_grid]]
) {
    uint total = batch * heads * seq_len * head_dim;
    if (idx >= total) return;

    uint b = idx / (heads * seq_len * head_dim);
    uint residual = idx % (heads * seq_len * head_dim);
    uint h = residual / (seq_len * head_dim);
    residual = residual % (seq_len * head_dim);
    uint i = residual / head_dim;
    uint d = residual % head_dim;

    device const half* Q_bh = Q + ((b * heads + h) * seq_len + i) * head_dim;
    device const half* K_bh = K + (b * heads + h) * seq_len * head_dim;
    device const half* V_bh = V + (b * heads + h) * seq_len * head_dim;
    device half* O_bh = O + ((b * heads + h) * seq_len + i) * head_dim;

    float scale = 1.0 / sqrt(float(head_dim));
    float max_val = -INFINITY;

    // Compute scores
    thread float scores[1024];
    for (uint j = 0; j < seq_len; ++j) {
        float sum = 0.0;
        for (uint k = 0; k < head_dim; ++k) {
            sum += float(Q_bh[k]) * float(K_bh[j * head_dim + k]);
        }
        scores[j] = sum * scale;
        if (scores[j] > max_val) max_val = scores[j];
    }

    // Softmax
    float sum_exp = 0.0;
    for (uint j = 0; j < seq_len; ++j) {
        scores[j] = exp(scores[j] - max_val);
        sum_exp += scores[j];
    }
    for (uint j = 0; j < seq_len; ++j) {
        scores[j] /= sum_exp;
    }

    // Compute output
    float out_val = 0.0;
    for (uint j = 0; j < seq_len; ++j) {
        out_val += scores[j] * float(V_bh[j * head_dim + d]);
    }

    O_bh[d] = half(out_val);
}

// ============================================================================
// Quantization Kernels
// ============================================================================

kernel void dequant_q4km_f16(
    device const uint8_t* q_weights [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= n) return;

    uint block_idx = idx / 256;
    uint elem_idx = idx % 256;
    uint scale_idx = block_idx * 32 + elem_idx / 8;

    uint byte_idx = block_idx * 256 + elem_idx / 2;
    uint8_t packed = q_weights[byte_idx];
    uint8_t nibble = (elem_idx % 2 == 0) ? (packed & 0x0F) : (packed >> 4);

    float scale = float(scales[scale_idx]);
    float val = float(nibble - 8) * scale;
    output[idx] = half(val);
}

kernel void gemm_q4km_f16(
    device const uint8_t* A_q [[buffer(0)]],
    device const half* A_scales [[buffer(1)]],
    device const half* B [[buffer(2)]],
    device half* C [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= N || gid.y >= M) return;

    float sum = 0.0;
    for (uint k = 0; k < K; ++k) {
        uint block_idx = k / 256;
        uint elem_idx = k % 256;
        uint scale_idx = gid.y * (K / 256 * 32) + block_idx * 32 + elem_idx / 8;

        uint byte_idx = gid.y * K + block_idx * 256 + elem_idx / 2;
        uint8_t packed = A_q[byte_idx];
        uint8_t nibble = (elem_idx % 2 == 0) ? (packed & 0x0F) : (packed >> 4);

        float a_val = float(nibble - 8) * float(A_scales[scale_idx]);
        float b_val = float(B[k * N + gid.x]);
        sum += a_val * b_val;
    }

    C[gid.y * N + gid.x] = half(sum);
}

// ============================================================================
// RoPE Kernel
// ============================================================================

kernel void rope_f16(
    device half* Q [[buffer(0)]],
    device half* K [[buffer(1)]],
    device const half* cos [[buffer(2)]],
    device const half* sin [[buffer(3)]],
    constant uint& batch [[buffer(4)]],
    constant uint& heads [[buffer(5)]],
    constant uint& seq_len [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    uint idx [[thread_position_in_grid]]
) {
    uint half_dim = head_dim / 2;
    uint total = batch * heads * seq_len * half_dim;
    if (idx >= total) return;

    uint b = idx / (heads * seq_len * half_dim);
    uint residual = idx % (heads * seq_len * half_dim);
    uint h = residual / (seq_len * half_dim);
    residual = residual % (seq_len * half_dim);
    uint s = residual / half_dim;
    uint d = residual % half_dim;

    uint qk_base = ((b * heads + h) * seq_len + s) * head_dim;

    float q1 = float(Q[qk_base + d]);
    float q2 = float(Q[qk_base + d + half_dim]);
    float k1 = float(K[qk_base + d]);
    float k2 = float(K[qk_base + d + half_dim]);

    float cos_val = float(cos[s * half_dim + d]);
    float sin_val = float(sin[s * half_dim + d]);

    Q[qk_base + d] = half(q1 * cos_val - q2 * sin_val);
    Q[qk_base + d + half_dim] = half(q1 * sin_val + q2 * cos_val);
    K[qk_base + d] = half(k1 * cos_val - k2 * sin_val);
    K[qk_base + d + half_dim] = half(k1 * sin_val + k2 * cos_val);
}

// ============================================================================
// Sampling Kernels
// ============================================================================

kernel void sample_argmax(
    device const half* logits [[buffer(0)]],
    device int* output [[buffer(1)]],
    constant uint& batch [[buffer(2)]],
    constant uint& vocab_size [[buffer(3)]],
    uint b [[thread_position_in_grid]]
) {
    if (b >= batch) return;

    device const half* logits_b = logits + b * vocab_size;

    float max_val = -INFINITY;
    int max_idx = 0;

    for (uint i = 0; i < vocab_size; ++i) {
        float val = float(logits_b[i]);
        if (val > max_val) {
            max_val = val;
            max_idx = int(i);
        }
    }

    output[b] = max_idx;
}

kernel void sample_topk_topp(
    device const half* logits [[buffer(0)]],
    device int* output [[buffer(1)]],
    constant uint& batch [[buffer(2)]],
    constant uint& vocab_size [[buffer(3)]],
    constant uint& top_k [[buffer(4)]],
    constant float& top_p [[buffer(5)]],
    constant float& temperature [[buffer(6)]],
    constant ulong& seed [[buffer(7)]],
    uint b [[thread_position_in_grid]]
) {
    if (b >= batch) return;

    // Simplified - just argmax with temperature
    device const half* logits_b = logits + b * vocab_size;

    float max_val = -INFINITY;
    int max_idx = 0;

    for (uint i = 0; i < vocab_size; ++i) {
        float val = float(logits_b[i]) / temperature;
        if (val > max_val) {
            max_val = val;
            max_idx = int(i);
        }
    }

    output[b] = max_idx;
}