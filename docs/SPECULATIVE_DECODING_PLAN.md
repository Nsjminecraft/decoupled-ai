# N-Gram Speculative Decoding Implementation Plan

## Overview
This document outlines the implementation of **Draftless N-Gram Speculative Decoding** for DeCoupled-AI as specified in the Architectural Directive (Phase 2).

### Key Properties
- **Zero VRAM overhead**: Runs entirely on CPU/host RAM
- **Parallel verification**: Batched forward pass for candidate tokens
- **Dynamic N-gram order**: 4-gram base with back-off to 3/2/1-gram
- **Confidence-gated**: Only speculate when probability exceeds threshold
- **KV-cache mask adjustment**: Maintains correct attention masks during speculative acceptance

---

## Architecture Integration Points

### Existing Components to Extend
| Component | Role in Speculative Decoding |
|-----------|------------------------------|
| `engine-ipc/src/lib.rs` | Core inference engine, `StreamHandle::sample_token_for`, `ComputeBackend` trait |
| `compute-cpu/src/lib.rs` | CPU backend with SIMD kernels, sampling, attention, GEMM |
| `server-backend/src/lib.rs` | Axum server, SSE/WS streaming, API endpoints |
| `api-openai/src/lib.rs` | OpenAI-compatible API types (chat completions, streaming) |
| `stream-cache/src/lib.rs` | Sharded weight access via `with_tensor` |

### New Components to Create
| Component | Purpose |
|-----------|---------|
| `engine-ipc/src/speculative.rs` | N-gram index, speculative engine, verification logic |
| `engine-ipc/src/speculative_config.rs` | Configuration structs for speculative decoding |
| `server-backend/src/speculative_handlers.rs` | API endpoints for speculative config/metrics |
| `frontend-ui/src/speculative_dashboard.js` | Real-time acceptance rate dashboard |

---

## Sub-Task Breakdown

### Task 1: N-Gram Sliding Window Hash Indexer
**File**: `engine-ipc/src/speculative/ngram_index.rs`

#### Design
```rust
/// N-gram index with rolling window hash (Rabin-Karp style)
/// Capacity: ~16MB (~4M entries for N=4 with 4-byte keys + 4-byte values)
pub struct NgramIndex {
    // Hash table: key = packed N-gram (u32), value = Vec<u32> (continuation tokens)
    table: hashbrown::HashMap<u32, SmallVec<[u32; 4]>>,
    max_order: usize,          // 4 (4-gram base)
    window_size: usize,        // Context window for N-gram extraction
    token_count: usize,        // Total tokens indexed
    config: NgramIndexConfig,
}
```

#### Key Operations
1. **`insert(context: &[u32])`** - Rolling window update on each generated token
2. **`query(context: &[u32], order: usize) -> Option<&[u32]>`** - Lookup continuations
3. **`backoff_query(context: &[u32]) -> Vec<u32>`** - Try N=4, then 3, 2, 1
4. **`pack_ngram(tokens: &[u32]) -> u32`** - Perfect hash for 4 tokens (vocab ≤ 65K) or FNV-1a

#### Performance Target
- Lookup latency: **< 50μs** (including back-off)
- Memory: **~16MB** (capped via LRU eviction on token overflow)
- Thread-safe for concurrent inference streams

---

### Task 2: Draft Generation Speculator Engine
**File**: `engine-ipc/src/speculative/speculator.rs`

#### Design
```rust
/// Speculative draft generator using N-gram index
pub struct Speculator {
    ngram_index: Arc<NgramIndex>,
    config: SpeculatorConfig,
    rng: StdRng,
}
```

#### Configuration
```rust
pub struct SpeculatorConfig {
    pub enabled: bool,
    pub max_draft_tokens: usize,      // Max speculative depth (e.g., 8)
    pub min_confidence: f32,          // Probability threshold (e.g., 0.65)
    pub ngram_order: usize,           // Base N (default 4)
    pub temperature: f32,             // Sampling temperature for draft
    pub top_k: usize,                 // Top-k for draft sampling
    pub top_p: f32,                   // Top-p for draft sampling
}
```

#### Algorithm
1. **Context preparation**: Take last `N-1` tokens from generation history
2. **N-gram query**: `backoff_query(context)` → candidate continuations
3. **Confidence check**: If `max_prob < min_confidence` → disable speculation for this step
4. **Draft generation**: 
   - For each draft step: sample from N-gram continuations
   - If N-gram empty → fall back to model logits (argmax)
   - Update rolling context with sampled token
5. **Return**: `Vec<i32>` of draft tokens (length ≤ `max_draft_tokens`)

#### Integration Point
- Called from `StreamHandle::sample_token_for` when `stream` is enabled
- Returns draft tokens + metadata for verification step

---

### Task 3: Target Verification & KV-Cache Mask Adjuster
**File**: `engine-ipc/src/speculative/verifier.rs`

#### Design
```rust
/// Batched verification engine
pub struct Verifier {
    backend: Arc<dyn ComputeBackend>,
    config: VerifierConfig,
}
```

#### Verification Algorithm
```
Input: prompt_tokens + draft_tokens[0..K]
Output: accepted_tokens[0..M] where M ≤ K

1. Batch forward pass:
   - Concatenate prompt + all K draft tokens → single sequence
   - Run ONE full forward pass through model
   - Get logits for each position (prompt_len .. prompt_len + K)

2. Parallel token verification:
   For i in 0..K:
       target_logits = logits[prompt_len + i]
       draft_token = draft_tokens[i]
       target_prob = softmax(target_logits)[draft_token]
       
       if target_prob >= acceptance_threshold:
           accept token
       else:
           reject at position i, sample from target_logits
           BREAK (subsequent tokens invalid)

3. KV-Cache Mask Adjustment:
   - Attention mask must cover only accepted tokens
   - For accepted tokens [0..M-1]: normal causal mask
   - For rejected token at M: mask blocks future positions
   - Next generation step starts from position M+1 (re-sampled token)

4. Return: (accepted_count, final_token, finish_reason)
```

#### Key Implementation Details
- **Batched forward pass**: Extend `ComputeBackend` with `forward_batch` that processes variable-length sequences
- **KV-cache**: Reuse existing KV-cache infrastructure; mask adjustment is a matter of tracking valid positions
- **Acceptance threshold**: Configurable (default 0.5 or adaptive based on entropy)

---

### Task 4: Server API & Performance Dashboard Update
**Files**: 
- `server-backend/src/speculative_config.rs`
- `server-backend/src/speculative_handlers.rs`
- `frontend-ui/assets/speculative_dashboard.js`

#### API Endpoints
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/speculative/config` | GET | Current speculative config |
| `/v1/speculative/config` | POST | Update config (enabled, depth, confidence, etc.) |
| `/v1/speculative/metrics` | GET | Real-time metrics (acceptance rate, tokens/pass, speedup) |
| `/v1/speculative/metrics/sse` | GET | SSE stream for live dashboard updates |

#### Metrics Structure
```rust
#[derive(Serialize)]
pub struct SpeculativeMetrics {
    pub enabled: bool,
    pub total_generation_steps: u64,
    pub total_draft_tokens: u64,
    pub total_accepted_tokens: u64,
    pub acceptance_rate: f32,       // accepted / draft
    pub avg_tokens_per_pass: f32,   // accepted per forward pass
    pub speedup_factor: f32,        // vs non-speculative baseline
    pub avg_verification_latency_ms: f32,
    pub draft_generation_latency_ms: f32,
}
```

#### Dashboard Features
- Real-time line chart: Acceptance rate over time
- Gauge: Current tokens/pass
- Counter: Total speedup factor
- Config panel: Toggle, depth, confidence threshold sliders
- SSE auto-reconnect on disconnect

---

### Task 5: Speculation Verification & Edge-Case Tests
**File**: `tests/src/speculative_tests.rs`

#### Test Categories

| Test | Description | Expected |
|------|-------------|----------|
| `identical_distribution_test` | Generate 10K tokens with/without speculation, compare distributions | KL-divergence < 0.01 |
| `high_repetition_acceptance` | Generate repetitive text (e.g., "the the the..."), measure acceptance | > 80% acceptance rate |
| `backoff_correctness` | Verify N-gram back-off (4→3→2→1) works with synthetic data | Correct fallback chain |
| `kv_cache_mask` | Verify attention mask matches accepted token count | No attention to rejected positions |
| `streaming_consistency` | SSE/WS streams match non-speculative output token-for-token (when acceptance=100%) | Identical output |
| `temperature_zero_greedy` | Speculation with temp=0 should match argmax exactly | Deterministic match |
| `long_context` | Test with context > 4K tokens | No OOM, correct indexing |
| `concurrent_streams` | Multiple simultaneous generations with speculation | Thread-safe, no cross-contamination |

#### Integration Test Flow
1. Download tiny model (e.g., `tiny-random-LlamaForCausalLM`)
2. Convert to sharded `.brain` format
3. Load model via server
4. Run each test with `cargo test --test speculative_tests`
5. Verify metrics endpoint reports correct values

---

## Implementation Sequence

### Phase 2A: Core Infrastructure (Week 1)
1. Create `engine-ipc/src/speculative/` module structure
2. Implement `NgramIndex` with hashbrown + rolling hash
3. Implement `Speculator` with back-off logic
4. Add unit tests for N-gram indexing and draft generation

### Phase 2B: Verification Engine (Week 1-2)
1. Extend `ComputeBackend` trait with batched forward pass
2. Implement `Verifier` with parallel token verification
3. Implement KV-cache mask adjustment
4. Integrate into `StreamHandle::sample_token_for` / `generate_stream`

### Phase 2C: Server Integration (Week 2)
1. Add `SpeculativeConfig` to `ServerConfig`
2. Create API endpoints in `speculative_handlers.rs`
3. Wire metrics collection in inference engine
4. Add SSE streaming for live metrics

### Phase 2D: Frontend Dashboard (Week 2-3)
1. Create `speculative_dashboard.js` with Chart.js
2. Add `/speculative` route in server
3. Connect SSE metrics to dashboard UI

### Phase 2E: Testing & Validation (Week 3)
1. Write comprehensive test suite in `tests/src/speculative_tests.rs`
2. Run distribution equivalence tests
3. Benchmark speedup on target hardware
4. Fix edge cases and tune defaults

---

## Configuration Defaults

```rust
impl Default for SpeculatorConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_draft_tokens: 8,
            min_confidence: 0.65,
            ngram_order: 4,
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
        }
    }
}

impl Default for VerifierConfig {
    fn default() -> Self {
        Self {
            acceptance_threshold: 0.5,
            max_batch_size: 16,
        }
    }
}
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Distribution shift | Continuous KL-divergence monitoring in tests; disable if drift detected |
| KV-cache corruption | Exhaustive mask tests; verify with attention visualization |
| Memory bloat | N-gram index capped at 16MB with LRU eviction |
| Latency regression | Profile verification vs. baseline; only enable when speedup > 1.1x |
| Concurrency bugs | Thread-local N-gram indices per stream; integration tests with concurrent requests |

---

## Success Criteria

1. **Functional**: All 5 sub-tasks implemented and integrated
2. **Correctness**: `identical_distribution_test` passes (KL < 0.01)
3. **Performance**: > 80% acceptance on repetitive text, > 1.3x speedup on typical prompts
4. **Observability**: Dashboard shows real-time metrics via SSE
5. **Stability**: No crashes in 10K token generation stress test
6. **Compatibility**: Existing non-speculative paths unchanged

---

*Plan Version: 1.0 | Created: 2026-07-15 | Phase 2 Start*