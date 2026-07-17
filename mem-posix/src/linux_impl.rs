#[cfg(target_os = "linux")]
mod linux_impl {
    use anyhow::{anyhow, Context, Result};
    use brain_pack::{BrainPack, TensorIndexEntry, BRAIN_MAGIC, WEIGHTS_ALIGNMENT, INDEX_ENTRY_SIZE};
    use libc::{c_void, madvise, MADV_SEQUENTIAL, MADV_WILLNEED};
    use memmap2::{Mmap, MmapOptions};
    use std::fs::File;
    use std::os::unix::io::AsRawFd;
    use std::path::Path;
    use std::ptr;
    use std::sync::Arc;
    use tracing::{debug, info, warn};

    // ============================================================================
    // POSIX Memory Mapper (Linux)
    // ============================================================================

    // ---------------------------------------------------------------------------
    // Rolling Block Overlay Mapper
    // ---------------------------------------------------------------------------
    //
    // `ShardOverlay` is the POSIX implementation of the sharded disk-streaming
    // contract from the architectural upgrade. It:
    //   * mmap's the always-resident base fragment (00.brain) up-front and keeps
    //     it mapped for the lifetime of the model.
    //   * mmap/munmap's the 15 streaming fragments (`01.brain`..`15.brain`) on
    //     demand, applying `madvise(MADV_WILLNEED)` before the compute stage so
    //     the kernel begins prefaulting from the NVMe.
    //
    // It is the POSIX counterpart of `mem_windows::ShardOverlay`. The rolling
    // strategy (which shards stay mapped, which get evicted) is decided by the
    // LRU driver in the `stream-cache` crate. `ShardOverlay` only owns the
    // low-level (un)map + advise primitives a caller can use to enforce the
    // 2 GiB resident cap.

    pub struct ShardOverlay {
        model: brain_pack::ShardedModel,
        /// Currently mapped shard files, keyed by shard id. Shard 0 (base) is
        /// always present; streaming shards are added/removed by `map_shard` /
        /// `unmap_shard`.
        mapped: std::collections::HashMap<u16, Mmap>,
    }

    impl ShardOverlay {
        /// Open a sharded model and map the resident base fragment.
        pub fn open(dir: &Path, name: &str) -> Result<Self> {
            let model = brain_pack::ShardedModel::open(dir, name)?;
            let mut mapped = std::collections::HashMap::new();
            // Base fragment: memory-map read-only and advise the kernel.
            let file = File::open(&model.shard_paths[0])
                .with_context(|| format!("open base shard {}", model.shard_paths[0].display()))?;
            let mmap = unsafe { MmapOptions::new().map(&file)? };

            // Keep base fragment permanently mapped
            mapped.insert(0, mmap);

            Ok(Self { model, mapped })
        }

        /// Check if a shard is currently mapped.
        pub fn is_mapped(&self, shard_id: u16) -> bool {
            self.mapped.contains_key(&shard_id)
        }

        /// Get raw bytes of a mapped shard.
        pub fn shard_bytes(&self, shard_id: u16) -> Option<&[u8]> {
            self.mapped.get(&shard_id).map(|m| m.as_ref())
        }

        /// Map a streaming shard (id >= 1). Returns the byte slice.
        pub fn map_shard(&mut self, shard_id: u16) -> Result<&[u8]> {
            if shard_id == 0 {
                return Ok(self.mapped.get(&0).expect("base always mapped").as_ref());
            }
            if self.mapped.contains_key(&shard_id) {
                return Ok(self.mapped[&shard_id].as_ref());
            }
            let path = &self.model.shard_paths[shard_id as usize];
            let file = File::open(path)
                .with_context(|| format!("open shard {}", path.display()))?;
            let mmap = unsafe { MmapOptions::new().map(&file)? };
            // Advise the kernel we'll need these pages soon (prefetch)
            unsafe {
                madvise(mmap.as_ptr() as *mut c_void, mmap.len(), MADV_WILLNEED);
            }
            self.mapped.insert(shard_id, mmap);
            Ok(self.mapped[&shard_id].as_ref())
        }

        /// Unmap a streaming shard (id >= 1). Base shard (0) cannot be unmapped.
        pub fn unmap_shard(&mut self, shard_id: u16) -> Result<()> {
            if shard_id == 0 {
                return Err(anyhow!("base shard 0 cannot be unmapped"));
            }
            self.mapped.remove(&shard_id);
            Ok(())
        }

        /// Locate a tensor by name.
        pub fn locate(&self, name: &str) -> Option<brain_pack::TensorLocation> {
            self.model.locate(name)
        }

        /// Get reference to the shard index.
        pub fn index(&self) -> &brain_pack::ShardIndex {
            self.model.index()
        }

        /// Get reference to the base pack.
        pub fn base_pack(&self) -> &BrainPack {
            &self.model.base_pack
        }

        /// Number of currently resident shards.
        pub fn resident_shard_count(&self) -> usize {
            self.mapped.len()
        }

        /// Get raw pointer and length of a mapped shard without borrowing the
        /// lock guard. The caller must ensure the shard stays mapped (via the
        /// LRU pin count) while using the returned pointer.
        pub fn shard_ptr_len(&self, shard_id: u16) -> Option<(*const u8, usize)> {
            self.mapped.get(&shard_id).map(|m| (m.as_ptr(), m.len()))
        }
    }
}