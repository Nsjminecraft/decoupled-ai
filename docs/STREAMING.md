# DeCoupled-AI Sharded Disk-Streaming & LRU Expert Caching Specification

## Overview

This document specifies the sharded disk-streaming architecture for DeCoupled-AI, enabling inference of large models (e.g., 30B+ parameters) on hardware with limited RAM (≤ 2 GB resident). The design splits a monolithic model into a static RAM-resident base fragment plus 15 sequential streaming shards, with an LRU cache driver managing shard residency and a Tokio-based async prefetch pool.

---

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DeCoupled-AI Engine                          │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌─────────────────┐    ┌──────────────────┐   │
│  │ Inference    │───▶│ StreamCache     │───▶│ Prefetch Pool    │   │
│  │ Engine       │    │ (LRU + Cache)   │    │ (Tokio blocking) │   │
│  └──────────────┘    └─────────────────┘    └──────────────────┘   │
│         ▲                    ▲                      ▲                │
│         │                    │                      │                │
│         ▼                    ▼                      ▼                │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                  ShardOverlay (Platform)                    │   │
│  │  ┌─────────────────────┐         ┌─────────────────────┐    │   │
│  │  │ mem-posix (mmap)    │  OR     │ mem-windows         │    │   │
│  │  │ madvise(WILLNEED)   │         │ MapViewOfFile       │    │   │
│  │  └─────────────────────┘         └─────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│         ▲                                              ▲            │
│         │                                              │            │
│         ▼                                              ▼            │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    NVMe Storage (.brain shards)              │   │
│  │  00.brain  01.brain  02.brain  ...  15.brain  model.shard.idx│   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Design Principles

1. **Static base fragment (`00.brain`)** — Embeddings, attention projections, and other frequently accessed weights stay permanently RAM-resident (~200-500 MB).
2. **15 streaming shards (`01.brain`…`15.brain`)** — Each ≤ 2 GiB, containing feed-forward/MLP weights for experts or layer groups.
3. **Shard index (`model.shard.idx`)** — Tiny (≈ 2 KB) in-RAM routing table mapping tensor name → shard ID.
4. **Strict LRU with pin counts** — Prevents eviction of in-use shards; max 2 GiB active resident.
5. **Async prefetch (Tokio blocking pool)** — Background thread pool issues `madvise(MADV_WILLNEED)` / `MapViewOfFile` ahead of compute.
6. **Zero-copy tensor access** — Closure pattern `with_tensor(|bytes| …)` returns `&[u8]` directly into mapped memory; no `Vec<u8>` allocation.

---

## 2. On-Disk Layout

### 2.1 Model Directory Structure

```
<model-dir>/
├── 00.brain              # Static base fragment (always resident)
├── 01.brain              # Streaming shard 1
├── 02.brain              # Streaming shard 2
...
├── 15.brain              # Streaming shard 15
├── <model>.shard.idx     # Shard index (SHARD.1\0 magic)
└── manifest.json         # Human-readable manifest (optional)
```

### 2.2 Shard File Format (`.brain`)

Each shard is a valid `.brain` container per `docs/SPEC.md`:
- Magic: `BRAIN.1\0` (8 bytes)
- JSON manifest (null-terminated)
- Tensor index map (64-byte entries)
- Raw weights array (64-byte aligned)

The base shard (`00.brain`) contains a full manifest. Streaming shards contain only the tensors they own; their manifests reference the base manifest for shared metadata.

### 2.3 Shard Index Format (`SHARD.1\0`)

```c
// File header (26 bytes)
Magic:       [8]  = "SHARD.1\0"
Version:     u16  = 1
NumShards:   u16  = 16        // 1 base + 15 streaming
BaseHdrSize: u64              // bytes of 00.brain header (for fast skip)
PerShardMax: u64  = 2 GiB     // enforced by packer

// Shard descriptors: 16 × 64 bytes = 1024 bytes
ShardDesc[0..15]:
    shard_id:       u16   // 0 = base, 1..15 = streaming
    version:        u16   // always 1
    _pad:           u32   // alignment to 8
    byte_length:    u64   // file size on disk
    tensor_count:   u64   // number of tensors in this shard
    first_tensor:   u64   // index into global tensor list
    reserved:       [4]u64

// Routing table: tensor name → shard_id
RouteCount:   u64
Routes[]:
    shard_id:   u16
    name_len:   u64
    name_bytes: [name_len]u8
```

Total index size ≈ 1.5–2 KB; stays permanently in RAM.

---

## 3. Core Crates

### 3.1 `brain-pack` — Sharded Packing CLI & Library

**New CLI flag:** `--shard` / `-s`

```bash
brain-pack pack --input ./model.safetensors --output ./my-model --shard
```

**Library API additions:**

```rust
// Sharded model writer
pub fn write_sharded(
    tensors: &[TensorInfo],
    manifest: &ModelManifest,
    out_dir: &Path,
    shard_max_bytes: u64,
    num_shards: u16,
) -> Result<ShardIndex>;

// Sharded model reader (stream-cache consumer)
pub struct ShardedModel {
    pub index: ShardIndex,
    pub base_header_bytes: Vec<u8>,  // 00.brain header for mmap
    pub shard_dir: PathBuf,
}

impl ShardedModel {
    pub fn open(dir: &Path) -> Result<Self>;
    pub fn locate_tensor(&self, name: &str) -> Option<(u16, TensorIndexEntry)>;  // (shard_id, entry)
    pub fn base_header(&self) -> &[u8];
    pub fn shard_path(&self, shard_id: u16) -> PathBuf;
}

// Zero-copy tensor accessors (for stream-cache / engine-ipc)
impl BrainPack {
    pub fn weights_ptr(&self) -> *const u8;
    pub fn weights_len(&self) -> usize;
    pub fn tensor_view(&self, entry: &TensorIndexEntry) -> &[u8];
    pub fn manifest_from_bytes(bytes: &[u8]) -> Result<ModelManifest>;
}
```

### 3.2 `weight-handle` — `VolatileWeights` Trait (Breaks Cycle)

A tiny crate (`weight-handle`) defining a single trait to decouple `engine-ipc` from `stream-cache`:

```rust
use half::f16;

pub trait VolatileWeights {
    fn as_f16(&self) -> &[f16];
    fn as_bytes(&self) -> &[u8];
    fn shard_id(&self) -> u16;
}

// Implemented by stream-cache for its lease types
```

### 3.3 `stream-cache` — Async Prefetch Pool + LRU Driver

**Public API:**

```rust
pub struct StreamCache {
    // ... internals
}

impl StreamCache {
    pub fn new(
        index: ShardIndex,
        shard_dir: PathBuf,
        max_resident_bytes: u64,  // 2 GiB default
        prefetch_depth: usize,     // how many shards ahead to prefetch
    ) -> Result<Self>;

    /// Zero-copy access to a tensor; loads shard if needed, pins for duration.
    pub fn with_tensor<F, R>(&self, tensor_name: &str, f: F) -> Result<R>
    where F: FnOnce(&[u8]) -> R;

    /// Background prefetch hint (non-blocking)
    pub fn prefetch(&self, tensor_name: &str);

    /// Stats for dashboard gauges
    pub fn stats(&self) -> CacheStats;
}

#[derive(Debug, Clone, Serialize)]
pub struct CacheStats {
    pub current_resident_bytes: u64,
    pub max_resident_bytes: u64,
    pub resident_shard_count: u32,
    pub total_bytes_pulled: u64,
    pub evictions: u64,
    pub prefetch_hits: u64,
}
```

**Internal components:**

| Component | Responsibility |
|-----------|----------------|
| `ShardOverlay` (platform) | mmap/MapViewOfFile + `madvise`/`MADV_WILLNEED` |
| `Lru` | `VecDeque<u16>` + pin counts; evicts tail when over budget |
| `PrefetchPool` | Tokio `spawn_blocking` workers issuing `madvise`/`PrefetchVirtualMemory` |
| `BaseWeights` | Zero-copy view into permanently mapped `00.brain` |
| `ShardLease<'a>` | RAII guard returned by `with_tensor`; holds pin + `&[u8]` slice |

**LRU Eviction Algorithm:**

```text
on with_tensor(tensor_name):
    1. Look up shard_id from index.routes
    2. If shard already resident:
           increment pin_count
           return slice to tensor bytes
    3. If resident_bytes + shard_size > MAX_RESIDENT:
           while resident_bytes + shard_size > MAX_RESIDENT:
               evict_lru_unpinned()  // pop_back where pin_count == 0
    4. Map shard via ShardOverlay (mmap / MapViewOfFile)
    5. madvise(MADV_WILLNEED | MADV_SEQUENTIAL) / PrefetchVirtualMemory
    6. Push shard_id to front of LRU deque; pin_count = 1
    7. Return slice into mapped region

on ShardLease drop:
    decrement pin_count for that shard_id
```

**Prefetch Strategy:**

- Called from `engine-ipc` at the start of each transformer layer.
- Looks up next layer's MLP shard; fires `prefetch()` → Tokio `spawn_blocking` → `madvise`/`PrefetchVirtualMemory`.
- Depth = 1–2 shards ahead (configurable).

### 3.4 `mem-posix` / `mem-windows` — Rolling Block Overlay Mappers

Both crates expose identical trait surface via `ShardOverlay`:

```rust
pub struct ShardOverlay {
    // platform-specific handles
}

impl ShardOverlay {
    /// Map a read-only view of `path` at `offset`..`offset+len`.
    pub fn map_shard(path: &Path, offset: u64, len: u64) -> Result<Self>;

    /// Raw pointer + length for zero-copy closure access (escapes lock guard lifetime).
    pub fn shard_ptr_len(&self) -> (*const u8, usize);

    /// Hint OS to pull pages (POSIX: madvise; Windows: PrefetchVirtualMemory).
    pub fn prefetch(&self, offset: u64, len: u64) -> Result<()>;

    /// Unmap on drop.
}
```

**POSIX (`mem-posix`):**
- `mmap` with `MAP_PRIVATE | MAP_POPULATE` (or `mmap` + `madvise(MADV_WILLNEED | MADV_SEQUENTIAL)`)
- `munmap` on drop
- `madvise(MADV_DONTNEED)` on eviction

**Windows (`mem-windows`):**
- `CreateFileMappingW` + `MapViewOfFileEx` with large-page support where available
- `UnmapViewOfFile` on drop
- `PrefetchVirtualMemory` for async prefetch
- `VirtualUnlock` / `SetFileIoOverlappedRange` hints on eviction

---

## 4. Engine Integration (`engine-ipc`)

### 4.1 `InferenceEngine` Changes

```rust
pub struct InferenceEngine {
    backend: Arc<dyn ComputeBackend>,
    base_model: BrainPack,           // 00.brain (always resident)
    stream_cache: Option<StreamCache>,  // None for monolithic models
    shard_dir: Option<PathBuf>,
    // ...
}

impl InferenceEngine {
    pub async fn load_model(&mut self, path: &str) -> Result<ModelInfo> {
        // Detect sharded model by presence of *.shard.idx
        if shard_index_exists(path) {
            let index = ShardIndex::open(path)?;
            self.stream_cache = Some(StreamCache::new(index, path, 2*GiB, 2)?);
            self.shard_dir = Some(path.into());
            // base_model loads 00.brain header only
        } else {
            // monolithic load (existing path)
        }
    }

    /// Zero-copy tensor access for compute backends
    pub fn with_tensor<F, R>(&self, name: &str, f: F) -> Result<R>
    where F: FnOnce(&[u8]) -> R {
        if let Some(cache) = &self.stream_cache {
            cache.with_tensor(name, f)
        } else {
            // monolithic: direct slice from base_model
            f(self.base_model.tensor_view(...))
        }
    }

    pub fn streaming_stats(&self) -> Option<CacheStats> {
        self.stream_cache.as_ref().map(|c| c.stats())
    }

    // generate() / generate_stream() now prefetch next-layer shards
    pub async fn generate(&mut self, req: GenerateRequest) -> Result<GenerateResponse> {
        // ... at each layer boundary:
        self.prefetch_next_layer_mlp()?;
        // ...
    }
}
```

### 4.2 `ComputeBackend` Trait Extensions (Default Impls)

```rust
use weight_handle::VolatileWeights;

pub trait ComputeBackend: Send + Sync {
    // Existing methods...
    
    /// Lease-based GEMM: backend receives &VolatileWeights for A/B/C.
    fn gemm_f16_lease(
        &self,
        a: &dyn VolatileWeights,
        b: &dyn VolatileWeights,
        c: &mut [f16],
        m: usize, n: usize, k: usize,
    ) -> Result<()> {
        // Default: copy to stack, call gemm_f16
        let a_vec = a.as_f16().to_vec();
        let b_vec = b.as_f16().to_vec();
        self.gemm_f16(&a_vec, &b_vec, c, m, n, k)
    }

    fn rms_norm_f16_lease(&self, w: &dyn VolatileWeights, x: &mut [f16], eps: f32) -> Result<()> { ... }
    fn attention_f16_lease(&self, qkv: &dyn VolatileWeights, ...) -> Result<()> { ... }
}
```

Backends override when they can consume raw pointers directly (Metal, CUDA, ROCm).

---

## 5. Frontend Dashboard (Settings Page)

**Endpoint:** `GET /v1/streaming/stats` → `CacheStats` JSON

**Polling:** Every 4 s via `setInterval` in `app.js`

**UI (Settings → Streaming Cache):**

| Gauge | Field | Format |
|-------|-------|--------|
| RAM Resident | `current_resident_bytes` / `max_resident_bytes` | Progress bar + "X.XX GiB / 2.00 GiB" |
| Streaming Shards Mapped | `resident_shard_count` | Integer (0–16) |
| NVMe Throughput (lifetime) | `total_bytes_pulled` | Human-readable bytes (e.g., "14.3 GiB") |
| Evictions | `evictions` | Counter |
| Prefetch Hits | `prefetch_hits` | Counter |

**Note:** Monolithic models (no `.shard.idx`) show "No sharded model loaded — monolithic .brain files show no streaming stats."

---

## 6. Server Endpoint

```rust
// GET /v1/streaming/stats
async fn streaming_stats(State(state): State<ServerState>) -> impl IntoResponse {
    match state.engine.streaming_stats() {
        Some(stats) => Json(stats).into_response(),
        None => (StatusCode::NOT_FOUND, Json(json!({"error": "No sharded model loaded"}))).into_response(),
    }
}
```

Response matches `CacheStats` (serde-serialized).

---

## 7. Packer CLI Usage

```bash
# Shard a model into 16 fragments (1 base + 15 streaming) with 2 GiB cap
brain-pack pack \
  --input ./llama-3-70b.safetensors \
  --output ./llama-3-70b-sharded \
  --shard \
  --shard-max-bytes 2147483648 \
  --num-shards 15

# Output directory structure:
llama-3-70b-sharded/
├── 00.brain              # ~400 MiB (embeddings, attention)
├── 01.brain .. 15.brain  # ~1.8 GiB each (MLP / experts)
├── llama-3-70b.shard.idx # ~2 KiB index
└── manifest.json         # optional human-readable
```

### Sharding Heuristics (Packer)

1. Sort tensors by layer order (embedding → layers → output).
2. Keep embeddings + all attention projections (`q/k/v/o`) in base shard.
3. Pack MLP / feed-forward / expert weights sequentially into streaming shards.
4. Enforce `SHARD_MAX_BYTES` per shard; if a single tensor exceeds, it gets its own shard (oversize warning).
5. Write `ShardIndex` with routing table.

---

## 8. Runtime Flow (Inference Step)

```
generate(token) 
    │
    ├─▶ Layer 0: Attention (base shard) → no cache miss
    │
    ├─▶ Layer 0: MLP (shard 01)
    │     ├─ stream_cache.with_tensor("layers.0.mlp.gate", ...) 
    │     │     ├─ shard 01 not resident → evict LRU if needed
    │     │     ├─ mmap 01.brain, madvise(WILLNEED|SEQUENTIAL)
    │     │     ├─ return &[u8] slice into mapped region
    │     │     └─ prefetch("layers.1.mlp.gate") → spawn_blocking(madvise)
    │     └─ compute_backend.gemm_f16_lease(lease_a, lease_b, ...)
    │
    ├─▶ Layer 1: Attention (base) …
    └─▶ Layer 1: MLP (shard 02) … (prefetched)
```

---

## 9. Configuration

| Parameter | Default | Env / CLI | Description |
|-----------|---------|-----------|-------------|
| `max_resident_bytes` | 2 GiB | `--stream-max-resident` / `STREAM_MAX_RESIDENT` | Hard cap on active shard memory |
| `prefetch_depth` | 2 | `--stream-prefetch-depth` | Shards to prefetch ahead |
| `shard_max_bytes` | 2 GiB | `--shard-max-bytes` (packer) | Packer enforcement |
| `num_shards` | 15 | `--num-shards` (packer) | Streaming fragment count |

---

## 10. Validation & Testing

### 10.1 Packer Round-Trip
```bash
brain-pack pack -i model.safetensors -o sharded --shard
brain-pack verify -i sharded
```

### 10.2 Cache Stress Test
```rust
#[test]
fn lru_eviction_under_pressure() {
    let cache = StreamCache::new(index, dir, 100 * MiB, 1)?;
    // Touch shards 1..5 sequentially, each 30 MiB
    for i in 1..=5 { cache.with_tensor(&format!("layer.{}.mlp", i), |_| {})?; }
    // Shard 1 should be evicted
    assert!(!cache.is_resident(1));
}
```

### 10.3 Zero-Copy Assertion
```rust
#[test]
fn zero_copy_slice() {
    let cache = StreamCache::new(...)?;
    let ptr_before = cache.with_tensor("t", |s| s.as_ptr())?;
    let ptr_after  = cache.with_tensor("t", |s| s.as_ptr())?;
    assert_eq!(ptr_before, ptr_after); // same mmap region
}
```

---

## 11. Cross-Platform Notes

| Feature | Linux/macOS (`mem-posix`) | Windows (`mem-windows`) |
|---------|---------------------------|-------------------------|
| Map API | `mmap` | `CreateFileMappingW` + `MapViewOfFileEx` |
| Prefetch | `madvise(MADV_WILLNEED \| MADV_SEQUENTIAL)` | `PrefetchVirtualMemory` |
| Evict hint | `madvise(MADV_DONTNEED)` | `VirtualUnlock` / discard |
| Large pages | `mmap(MAP_HUGETLB)` (opt-in) | `MapViewOfFileEx` with `FILE_MAP_LARGE_PAGES` |
| Async I/O | `tokio::fs` + `spawn_blocking` | Same (IOCP via Tokio) |

---

## 12. Future Extensions

| Feature | Status | Notes |
|---------|--------|-------|
| Expert parallelism (MoE) | Planned | Route tokens to expert shards; cache per-expert |
| P2P shard fetch (multi-node) | Research | gRPC + RDMA for distributed streaming |
| Compressed shards (ZSTD) | Planned | Transparent decompression in overlay |
| Telemetry export (Prometheus) | Planned | `/metrics` endpoint with streaming gauges |

---

## 13. References

- `docs/SPEC.md` — Monolithic `.brain` container format
- `brain-pack/src/lib.rs` — Sharding implementation
- `stream-cache/src/lib.rs` — LRU + prefetch driver
- `mem-posix/src/lib.rs` / `mem-windows/src/lib.rs` — Platform overlays
- `engine-ipc/src/lib.rs` — Engine integration
- `server-backend/src/lib.rs` — `/v1/streaming/stats` endpoint
- `frontend-ui/assets/app.js` — Dashboard polling

---

*Specification Version: 1.0 | DeCoupled-AI Project*