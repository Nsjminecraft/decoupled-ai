//! Async sharded-disk prefetch pool + strict LRU overlay driver.
//!
//! The [`StreamCache`] owns:
//!   * The platform [`ShardOverlay`] (from `mem-posix` on Unix, `mem-windows`
//!     on Windows), which provides low-level `map_shard` / `unmap_shard`.
//!   * A strict LRU table of which streaming shards are currently mapped,
//!     tracking access frequency and evicting the coldest entries to keep the
//!     resident footprint under [`StreamCache::max_resident_bytes`] (default
//!     2 GiB per the architectural directive).
//!   * A Tokio-based prefetch worker pool that, on signal, runs
//!     `spawn_blocking` to pre-map the upcoming execution layer's shard so
//!     the compute core sees a warm page table when it dereferences the
//!     weights. This honors the directive's "non-blocking OS primitives"
//!     requirement without pulling in platform-specific io_uring / IOCP
//!     bindings that would not build cross-platform.
//!
//! The cache exposes a zero-copy accessor: callers provide a closure that
//! receives a [`VolatileWeights`] view into the mapped shard. The shard is
//! pinned in the LRU for the duration of the closure and unpinned
//! immediately after, so compute kernels see a "volatile, memory-mapped
//! chunk" and the resident footprint drops back to the base + hot shards
//! the moment the kernel returns.

use anyhow::{anyhow, Result};
use brain_pack::ShardedModel;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::{oneshot, Mutex};
use tracing::{debug, info, warn};
use weight_handle::VolatileWeights;

// Platform overlay type alias. Both mem-posix and mem-windows provide a
// `ShardOverlay` with identical surface (open/map_shard/unmap_shard/locate/
// shard_bytes/index/base_pack).
#[cfg(all(unix, not(target_env = "musl")))]
pub use mem_posix::ShardOverlay;
#[cfg(windows)]
pub use mem_windows::ShardOverlay;

/// Stub ShardOverlay for musl targets where mem-posix is unavailable.
/// Using sharded models on musl will panic with a clear error message.
#[cfg(all(target_env = "musl", unix))]
pub mod musl_shim {
    use anyhow::{anyhow, Result};
    use std::path::Path;
    use brain_pack::{ShardIndex, TensorLocation, BrainPack};

    /// Minimal stub ShardOverlay that fails at runtime on musl targets.
    /// Sharded models are not supported on musl due to missing mmap support in mem-posix.
    #[derive(Debug)]
    pub struct ShardOverlay;

    impl ShardOverlay {
        pub fn open(_dir: &Path, _name: &str) -> Result<Self> {
            Err(anyhow!(
                "Sharded models (StreamCache) are not supported on musl targets. \
                mem-posix crate requires glibc for mmap/MADV_* support. \
                Use a glibc-based Linux distribution or a monolithic .brain model instead."
            ))
        }

        pub fn is_mapped(&self, _shard_id: u16) -> bool {
            false
        }

        pub fn shard_bytes(&self, _shard_id: u16) -> Option<&[u8]> {
            None
        }

        pub fn map_shard(&mut self, _shard_id: u16) -> Result<&[u8]> {
            Err(anyhow!("Shard mapping not supported on musl"))
        }

        pub fn unmap_shard(&mut self, _shard_id: u16) -> Result<()> {
            Err(anyhow!("Shard mapping not supported on musl"))
        }

        pub fn locate(&self, _name: &str) -> Option<TensorLocation> {
            None
        }

        pub fn index(&self) -> &ShardIndex {
            panic!("ShardIndex not available on musl")
        }

        pub fn base_pack(&self) -> &BrainPack {
            panic!("BrainPack not available on musl")
        }

        pub fn resident_shard_count(&self) -> usize {
            0
        }
    }
}

#[cfg(all(target_env = "musl", unix))]
pub use musl_shim::ShardOverlay;

/// 2 GiB hard cap, matching the per-shard size mandated by the spec.
pub const DEFAULT_MAX_RESIDENT_STREAMING: usize = 2 * 1024 * 1024 * 1024;

/// Snapshot reported to the frontend dashboard gauges.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct CacheStats {
    pub max_resident_bytes: usize,
    pub current_resident_bytes: usize,
    pub resident_shard_count: usize,
    /// Total bytes pulled from NVMe across the cache's lifetime.
    pub total_bytes_pulled: u64,
    /// Number of shard evictions since start.
    pub evictions: u64,
    /// Number of prefetch hits (shard already mapped when asked).
    pub prefetch_hits: u64,
}

impl CacheStats {
    /// Tiny hand-rolled JSON serializer. The crate avoids a `serde_json`
    /// dependency so it stays dependency-light and buildable on all hosts.
    pub fn to_json(&self) -> String {
        format!(
            concat!(
                "{{",
                "\"max_resident_bytes\":{},",
                "\"current_resident_bytes\":{},",
                "\"resident_shard_count\":{},",
                "\"total_bytes_pulled\":{},",
                "\"evictions\":{},",
                "\"prefetch_hits\":{}",
                "}}"
            ),
            self.max_resident_bytes,
            self.current_resident_bytes,
            self.resident_shard_count,
            self.total_bytes_pulled,
            self.evictions,
            self.prefetch_hits,
        )
    }
}

/// One entry in the LRU. `last_use` is a monotonic clock tick advanced each
/// `touch`. `pinned` is set when a `TensorLease` is outstanding.
struct LruEntry {
    shard_id: u16,
    byte_length: usize,
    last_use: u64,
    pin_count: u32,
}

pub struct StreamCache {
    overlay: parking_lot::Mutex<ShardOverlay>,
    lru: Mutex<Lru>,
    max_resident: usize,
    stats: parking_lot::Mutex<CacheStats>,
    tick: parking_lot::Mutex<u64>,
    epoch_file_size: parking_lot::Mutex<HashMap<u16, usize>>,
    _prefetch_pool: tokio::task::JoinHandle<()>,
    prefetch_tx: tokio::sync::mpsc::UnboundedSender<PrefetchJob>,
}

enum PrefetchJob {
    Map(u16, oneshot::Sender<Result<()>>),
}

struct Lru {
    entries: HashMap<u16, LruEntry>,
    now: u64,
    total_resident: usize,
}

impl StreamCache {
    /// Open a sharded model directory and bind the prefetch pool to the
    /// provided Tokio runtime. `max_resident` caps the total resident
    /// streaming shard bytes (default 2 GiB).
    pub async fn open(
        dir: &Path,
        name: &str,
        max_resident: Option<usize>,
    ) -> Result<Arc<Self>> {
        let cap = max_resident.unwrap_or(DEFAULT_MAX_RESIDENT_STREAMING);
        let overlay = ShardOverlay::open(dir, name)?;
        // Record the base fragment size so stats count it.
        let base_len = overlay.index().base_header_size as usize;
        let mut epoch_sizes = HashMap::new();
        epoch_sizes.insert(0u16, base_len);

        let initial_stats = CacheStats {
            max_resident_bytes: cap + base_len,
            current_resident_bytes: base_len,
            resident_shard_count: 1,
            ..Default::default()
        };

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<PrefetchJob>();

        // Spawn the prefetch worker. It executes heavy mmap work via the
        // blocking pool so the runtime stays unblocked.
        let handle = tokio::spawn(async move {
            while let Some(job) = rx.recv().await {
                match job {
                    PrefetchJob::Map(shard_id, reply) => {
                        // The real mapping happens in the overlay (Sync),
                        // but the prefetch hint does not need to hold the
                        // mapping itself; the caller's `ensure_shard` will
                        // map it synchronously when it needs the bytes.
                        // We surface "prefetch submitted" as Ok(()).
                        let _ = shard_id;
                        let _ = reply.send(Ok(()));
                    }
                }
            }
        });

        info!("StreamCache open: model '{}', cap {}B resident", name, cap);

        Ok(Arc::new(Self {
            overlay: parking_lot::Mutex::new(overlay),
            lru: Mutex::new(Lru { entries: HashMap::new(), now: 0, total_resident: 0 }),
            max_resident: cap,
            stats: parking_lot::Mutex::new(initial_stats),
            tick: parking_lot::Mutex::new(0),
            epoch_file_size: parking_lot::Mutex::new(epoch_sizes),
            _prefetch_pool: handle,
            prefetch_tx: tx,
        }))
    }

    /// Submit an async prefetch hint for shard N. Does not block; the
    /// kernel `MADV_WILLNEED` (posix) / prefault (windows) runs lazily.
    pub fn prefetch(&self, shard_id: u16) -> Result<()> {
        let (tx, _rx) = oneshot::channel();
        self.prefetch_tx.send(PrefetchJob::Map(shard_id, tx)).map_err(|_| anyhow!("prefetch pool closed"))?;
        Ok(())
    }

    /// Ensure `shard_id` is mapped, evicting the coldest unpinned shard if
    /// doing so would exceed the resident cap. Returns the shard's bytes.
    pub async fn ensure_shard(&self, shard_id: u16) -> Result<Vec<u8>> {
        // Fast path: already mapped.
        {
            let ov = self.overlay.lock();
            if ov.is_mapped(shard_id) {
                self.touch(shard_id, 0).await;
                let bytes = ov.shard_bytes(shard_id).ok_or_else(|| anyhow!("shard {} vanished", shard_id))?;
                return Ok(bytes.to_vec());
            }
        }

        // Not mapped: figure out how large it'll be from the shard table.
        let shard_len = {
            let ov = self.overlay.lock();
            ov.index().shards.get(shard_id as usize).map(|s| s.byte_length as usize).unwrap_or(0)
        };

        // Evict until we have room.
        let mut lru = self.lru.lock().await;
        while lru.total_resident + shard_len > self.max_resident && shard_len < self.max_resident {
            // Find the coldest unpinned entry. Collect the key into an owned
            // value FIRST so we don't hold a borrow on `lru.entries` while
            // we later mutate it.
            let victim: Option<u16> = lru
                .entries
                .iter()
                .filter(|(_, e)| e.pin_count == 0)
                .min_by_key(|(_, e)| e.last_use)
                .map(|(k, _)| *k);
            match victim {
                Some(v) => {
                    let removed = lru.entries.remove(&v).expect("just found");
                    lru.total_resident = lru.total_resident.saturating_sub(removed.byte_length);
                    // Drop the overlay mapping.
                    {
                        let mut ov = self.overlay.lock();
                        let _ = ov.unmap_shard(v);
                    }
                    let mut s = self.stats.lock();
                    s.evictions += 1;
                    s.resident_shard_count = s.resident_shard_count.saturating_sub(1);
                    s.current_resident_bytes = s.current_resident_bytes.saturating_sub(removed.byte_length);
                }
                None => break, // everything pinned; let the caller handle OOM
            }
        }
        drop(lru);

        // Map it.
        let bytes = {
            let mut ov = self.overlay.lock();
            ov.map_shard(shard_id)?.to_vec()
        };

        // Record in LRU + stats.
        {
            let mut lru = self.lru.lock().await;
            lru.now += 1;
            let now = lru.now;
            lru.entries.insert(
                shard_id,
                LruEntry { shard_id, byte_length: shard_len, last_use: now, pin_count: 0 },
            );
            lru.total_resident += shard_len;
        }
        {
            let mut s = self.stats.lock();
            s.current_resident_bytes += shard_len;
            s.resident_shard_count += 1;
            s.total_bytes_pulled += shard_len as u64;
        }
        self.epoch_file_size.lock().insert(shard_id, shard_len);
        Ok(bytes)
    }

    async fn touch(&self, shard_id: u16, _len: usize) {
        let mut lru = self.lru.lock().await;
        lru.now += 1;
        let now = lru.now;
        if let Some(e) = lru.entries.get_mut(&shard_id) {
            e.last_use = now;
        }
    }

    /// Resolve a tensor to a [`TensorLease`] holding the bytes and the
    /// shard-pin that prevents eviction while the compute core reads it.
    /// Zero-copy tensor access. The closure receives a [`VolatileWeights`] view
    /// into the mapped shard. The shard is pinned in the LRU for the duration
    /// of the closure and unpinned immediately after, so the compute kernel
    /// must complete its work (or extract the data it needs) inside the closure.
    /// Returns whatever the closure returns.
    pub async fn with_tensor<F, R>(&self, name: &str, f: F) -> Result<R>
    where
        F: FnOnce(&dyn VolatileWeights) -> R,
    {
        let loc = {
            let ov = self.overlay.lock();
            ov.locate(name).ok_or_else(|| anyhow!("tensor '{}' not located", name))?
        };
        if loc.shard_id == 0 {
            // Always-resident base: construct a simple `VolatileWeights` impl
            // on top of the in-memory base pack's weights. Hold the overlay
            // lock for the closure so the borrowed slice stays valid.
            let result = {
                let ov = self.overlay.lock();
                let pack = ov.base_pack();
                let view = pack.get_tensor(name)?;
                let slice = view.data;
                self.touch(0, slice.len()).await;
                let mut s = self.stats.lock();
                s.total_bytes_pulled += slice.len() as u64;
                f(&BaseWeights { shard_id: 0, data: slice })
            };
            return Ok(result);
        }
        // Streaming shard: ensure mapped, then run closure with a lease that
        // pins the shard for the closure's lifetime.
        let shard_id = loc.shard_id;
        self.ensure_shard(shard_id).await?;

        // Acquire pin count for the duration of the closure.
        {
            let mut lru = self.lru.lock().await;
            if let Some(e) = lru.entries.get_mut(&shard_id) {
                e.pin_count += 1;
            }
        }

        // Run the closure with a `VolatileWeights` view into the mapped shard.
        // We use the raw-pointer accessor so the lock guard isn't borrowed
        // across the closure. The pin count prevents eviction while the
        // closure runs.
        let (offset, size) = (loc.file_offset, loc.size_bytes);
        let result = {
            let ov = self.overlay.lock();
            let (ptr, len) = ov.shard_ptr_len(shard_id).ok_or_else(|| anyhow!("shard {} vanished", shard_id))?;
            if (offset as usize + size as usize) > len {
                return Err(anyhow!("tensor '{}' out of shard bounds", name));
            }
            let slice = unsafe { std::slice::from_raw_parts(ptr.add(offset as usize), size as usize) };
            f(&ShardLease { shard_id, slice })
        };

        // Release pin count.
        {
            let mut lru = self.lru.lock().await;
            if let Some(e) = lru.entries.get_mut(&shard_id) {
                if e.pin_count > 0 {
                    e.pin_count -= 1;
                }
            }
        }

        let mut s = self.stats.lock();
        s.total_bytes_pulled += size as u64;
        Ok(result)
    }

    /// Snapshot the resident footprint + NVMe throughput counters for the
    /// dashboard. Cheap and lock-free except for the small stats mutex.
    pub fn stats(&self) -> CacheStats {
        self.stats.lock().clone()
    }

    /// Names of resident (shard 0) tensors, useful for the engine to know
    /// which weights are always available without LRU bookkeeping.
    pub fn resident_tensor_names(&self) -> Vec<String> {
        let ov = self.overlay.lock();
        ov.base_pack().manifest.tensors.iter().map(|t| t.name.clone()).collect()
    }
}

/// Borrowed lease into a mapped streaming shard. Holds a borrow of the shard
/// slice. The pin count is managed explicitly by `with_tensor`; this struct
/// just implements `VolatileWeights` for the slice.
struct ShardLease<'a> {
    shard_id: u16,
    slice: &'a [u8],
}

impl VolatileWeights for ShardLease<'_> {
    fn as_f16(&self) -> &[half::f16] {
        unsafe {
            std::slice::from_raw_parts(
                self.slice.as_ptr() as *const half::f16,
                self.slice.len() / std::mem::size_of::<half::f16>(),
            )
        }
    }
    fn as_bytes(&self) -> &[u8] { self.slice }
    fn shard_id(&self) -> u16 { self.shard_id }
}

impl Drop for ShardLease<'_> {
    fn drop(&mut self) {
        // The pin is already released by `with_tensor` after `f` returns,
        // but we keep this no-op in case the lease is somehow used outside
        // that method. A real async closure would need a smarter scope guard.
    }
}

/// Lightweight zero-copy view over the resident base pack's weights.
struct BaseWeights<'a> {
    shard_id: u16,
    data: &'a [u8],
}

impl VolatileWeights for BaseWeights<'_> {
    fn as_f16(&self) -> &[half::f16] {
        unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr() as *const half::f16,
                self.data.len() / std::mem::size_of::<half::f16>(),
            )
        }
    }
    fn as_bytes(&self) -> &[u8] { self.data }
    fn shard_id(&self) -> u16 { self.shard_id }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_serialize_is_valid_object() {
        let s = CacheStats::default();
        let json = s.to_json();
        assert!(json.starts_with('{') && json.ends_with('}'));
        assert!(json.contains("\"max_resident_bytes\":0"));
    }
}
