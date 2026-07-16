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
// POSIX Memory Mapper
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
        unsafe {
            madvise(mmap.as_ptr() as *mut c_void, mmap.len(), MADV_SEQUENTIAL);
            madvise(mmap.as_ptr() as *mut c_void, mmap.len(), MADV_WILLNEED);
        }
        mapped.insert(0, mmap);
        Ok(Self { model, mapped })
    }

    /// Reference to the parsed shard index.
    pub fn index(&self) -> &brain_pack::ShardIndex {
        &self.model.index
    }

    /// The resident `BrainPack` (embeddings / attention / norms live here).
    pub fn base_pack(&self) -> &brain_pack::BrainPack {
        &self.model.base
    }

    /// Is `shard_id` currently mapped into this process?
    pub fn is_mapped(&self, shard_id: u16) -> bool {
        self.mapped.contains_key(&shard_id)
    }

    /// Map a streaming shard into RAM. Returns the shard's mapped bytes.
    /// Idempotent: re-mapping an already-mapped shard is a no-op.
    pub fn map_shard(&mut self, shard_id: u16) -> Result<&[u8]> {
        if shard_id == 0 {
            return Ok(self.mapped.get(&0).map(|m| m.as_slice()).unwrap_or(&[]));
        }
        if self.mapped.contains_key(&shard_id) {
            // Already mapped — touch so LRU ordering updates in the caller.
            return Ok(self.mapped.get(&shard_id).map(|m| m.as_slice()).unwrap_or(&[]));
        }
        let path = self.model.shard_paths.get(shard_id as usize)
            .ok_or_else(|| anyhow!("Unknown shard id {}", shard_id))?;
        let file = File::open(path)
            .with_context(|| format!("open shard {}: {}", shard_id, path.display()))?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        // Hint the kernel before compute: we will need these pages soon.
        unsafe {
            madvise(mmap.as_ptr() as *mut c_void, mmap.len(), MADV_WILLNEED);
        }
        // Stash the pointer before the borrow so we can return a slice.
        let ptr = mmap.as_ptr();
        let len = mmap.len();
        self.mapped.insert(shard_id, mmap);
        unsafe { Ok(std::slice::from_raw_parts(ptr, len)) }
    }

    /// Unmap a streaming shard, releasing its virtual pages back to the OS.
    /// Shard 0 (base) cannot be evicted and is a no-op.
    pub fn unmap_shard(&mut self, shard_id: u16) -> Result<()> {
        if shard_id == 0 {
            return Ok(());
        }
        if self.mapped.remove(&shard_id).is_some() {
            // Drop runs munmap via memmap2.
            debug!("unmapped shard {}", shard_id);
        }
        Ok(())
    }

    /// Borrow the bytes of a currently-mapped shard (panics if not mapped).
    pub fn shard_bytes(&self, shard_id: u16) -> Option<&[u8]> {
        self.mapped.get(&shard_id).map(|m| m.as_slice())
    }

    /// Get raw pointer and length of a mapped shard without borrowing the
    /// lock guard. The caller must ensure the shard stays mapped (via the
    /// LRU pin count) while using the returned pointer.
    pub fn shard_ptr_len(&self, shard_id: u16) -> Option<(*const u8, usize)> {
        self.mapped.get(&shard_id).map(|m| (m.as_ptr(), m.len()))
    }

    /// Locate a tensor's residence (shard + offset + size).
    pub fn locate(&self, name: &str) -> Option<brain_pack::TensorLocation> {
        self.model.locate(name)
    }

    /// Number of currently-mapped shards (including the base).
    pub fn resident_shard_count(&self) -> usize {
        self.mapped.len()
    }
}

/// Memory-mapped .brain file with zero-copy tensor access
pub struct PosixMappedBrain {
    mmap: Mmap,
    pack: BrainPack,
    file_size: usize,
}

impl PosixMappedBrain {
    /// Open and memory-map a .brain file
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)
            .with_context(|| format!("Failed to open {}", path.display()))?;

        let metadata = file.metadata()?;
        let file_size = metadata.len() as usize;

        // Memory map the entire file read-only
        let mmap = unsafe { MmapOptions::new().map(&file)? }
            .with_context(|| format!("Failed to mmap {}", path.display()))?;

        // Advise kernel on access pattern
        unsafe {
            madvise(mmap.as_ptr() as *mut c_void, mmap.len(), MADV_SEQUENTIAL);
            madvise(mmap.as_ptr() as *mut c_void, mmap.len(), MADV_WILLNEED);
        }

        // Parse the brain pack from the mapped memory
        let pack = BrainPack::from_bytes(&mmap)?;

        info!("Mapped {} ({} bytes, {} tensors)", path.display(), file_size, pack.manifest.tensors.len());

        Ok(Self {
            mmap,
            pack,
            file_size,
        })
    }

    /// Get the underlying BrainPack
    pub fn pack(&self) -> &BrainPack {
        &self.pack
    }

    /// Get a tensor view (zero-copy)
    pub fn tensor(&self, name: &str) -> Result<TensorView> {
        self.pack.tensor_view(name, &self.mmap)
    }

    /// Get all tensor names
    pub fn tensor_names(&self) -> Vec<String> {
        self.pack.manifest.tensors.iter().map(|t| t.name.clone()).collect()
    }

    /// Get file size
    pub fn file_size(&self) -> usize {
        self.file_size
    }

    /// Prefault pages for a tensor (touch pages to avoid page faults during inference)
    pub fn prefault_tensor(&self, name: &str) -> Result<()> {
        let tensor = self.tensor(name)?;
        let ptr = tensor.data.as_ptr() as *mut c_void;
        let len = tensor.data.len();

        // Touch each page
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        let mut offset = 0;
        while offset < len {
            unsafe {
                ptr::read_volatile(ptr.add(offset) as *const u8);
            }
            offset += page_size;
        }
        Ok(())
    }

    /// Prefault all weights
    pub fn prefault_all(&self) -> Result<()> {
        let weights_ptr = self.pack.weights_ptr(&self.mmap);
        let weights_len = self.pack.weights_len();
        let ptr = weights_ptr as *mut c_void;

        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        let mut offset = 0;
        while offset < weights_len {
            unsafe {
                ptr::read_volatile(ptr.add(offset) as *const u8);
            }
            offset += page_size;
        }
        Ok(())
    }
}

/// Zero-copy tensor view into mapped memory
#[derive(Debug, Clone)]
pub struct TensorView {
    pub name: String,
    pub data: &'static [u8],
    pub shape: Vec<u64>,
    pub dtype: brain_pack::DataType,
    pub quantization: Option<brain_pack::QuantizationParams>,
    pub strides: Vec<u64>,
}

impl TensorView {
    pub fn num_elements(&self) -> u64 {
        self.shape.iter().product()
    }

    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }

    pub fn is_contiguous(&self) -> bool {
        let mut expected_stride = 1;
        for &dim in self.shape.iter().rev() {
            if self.strides[self.shape.len() - 1 - &dim as usize] != expected_stride {
                return false;
            }
            expected_stride *= dim;
        }
        true
    }
}

// ============================================================================
// Memory Advice Helpers
// ============================================================================

/// Memory access pattern hints
pub enum MemoryAdvice {
    Sequential,
    Random,
    WillNeed,
    DontNeed,
    Free,
}

impl MemoryAdvice {
    pub fn apply(&self, ptr: *mut c_void, len: usize) -> Result<()> {
        let advice = match self {
            MemoryAdvice::Sequential => MADV_SEQUENTIAL,
            MemoryAdvice::Random => libc::MADV_RANDOM,
            MemoryAdvice::WillNeed => MADV_WILLNEED,
            MemoryAdvice::DontNeed => libc::MADV_DONTNEED,
            MemoryAdvice::Free => libc::MADV_FREE,
        };
        let ret = unsafe { madvise(ptr, len, advice) };
        if ret != 0 {
            return Err(anyhow!("madvise failed: {}", std::io::Error::last_os_error()));
        }
        Ok(())
    }
}

// ============================================================================
// Huge Pages Support (Linux)
// ============================================================================

#[cfg(target_os = "linux")]
pub mod huge_pages {
    use super::*;
    use std::fs::OpenOptions;
    use std::os::unix::io::AsRawFd;

    /// Map with huge pages (2MB or 1GB) - requires root or CAP_IPC_LOCK
    pub fn map_huge(path: &Path, huge_page_size: HugePageSize) -> Result<PosixMappedBrain> {
        let file = OpenOptions::new().read(true).open(path)?;
        let file_size = file.metadata()?.len() as usize;

        // Use mmap with MAP_HUGETLB flag
        let mmap = unsafe {
            let ptr = libc::mmap(
                ptr::null_mut(),
                file_size,
                libc::PROT_READ,
                libc::MAP_PRIVATE | libc::MAP_HUGETLB | (huge_page_size as i32),
                file.as_raw_fd(),
                0,
            );
            if ptr == libc::MAP_FAILED {
                return Err(anyhow!("huge page mmap failed: {}", std::io::Error::last_os_error()));
            }
            std::slice::from_raw_parts_mut(ptr as *mut u8, file_size)
        };

        let mmap = Mmap::from_raw_parts(mmap.as_ptr(), file_size, file)?;

        let pack = BrainPack::from_bytes(&mmap)?;
        Ok(PosixMappedBrain { mmap, pack, file_size })
    }

    #[repr(i32)]
    pub enum HugePageSize {
        Size2MB = 0x40000,   // MAP_HUGE_2MB
        Size1GB = 0x80000,   // MAP_HUGE_1GB
    }
}

// ============================================================================
// Shared Memory Support (for multi-process)
// ============================================================================

pub mod shared {
    use super::*;
    use std::os::unix::io::AsRawFd;

    /// Create a shared memory mapping that can be accessed by multiple processes
    pub fn create_shared(path: &Path, size: usize) -> Result<SharedMappedBrain> {
        // Create shared memory object
        let shm_name = format!("/decoupled-ai-{}", uuid::Uuid::new_v4().simple());
        let shm_fd = unsafe {
            libc::shm_open(
                shm_name.as_ptr() as *const i8,
                libc::O_CREAT | libc::O_RDWR,
                0o600,
            )
        };
        if shm_fd < 0 {
            return Err(anyhow!("shm_open failed: {}", std::io::Error::last_os_error()));
        }

        // Set size
        if unsafe { libc::ftruncate(shm_fd, size as i64) } != 0 {
            return Err(anyhow!("ftruncate failed: {}", std::io::Error::last_os_error()));
        }

        // Map it
        let ptr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                shm_fd,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(anyhow!("mmap failed: {}", std::io::Error::last_os_error()));
        }

        Ok(SharedMappedBrain {
            ptr: ptr as *mut u8,
            size,
            shm_fd,
            shm_name,
        })
    }

    pub struct SharedMappedBrain {
        ptr: *mut u8,
        size: usize,
        shm_fd: i32,
        shm_name: String,
    }

    impl SharedMappedBrain {
        pub fn as_slice(&self) -> &[u8] {
            unsafe { std::slice::from_raw_parts(self.ptr, self.size) }
        }

        pub fn as_mut_slice(&mut self) -> &mut [u8] {
            unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size) }
        }

        pub fn write_brain(&mut self, pack: &BrainPack) -> Result<()> {
            let data = pack.to_bytes()?;
            if data.len() > self.size {
                return Err(anyhow!("Shared memory too small"));
            }
            self.as_mut_slice()[..data.len()].copy_from_slice(&data);
            Ok(())
        }
    }

    impl Drop for SharedMappedBrain {
        fn drop(&mut self) {
            unsafe {
                libc::munmap(self.ptr as *mut c_void, self.size);
                libc::close(self.shm_fd);
                libc::shm_unlink(self.shm_name.as_ptr() as *const i8);
            }
        }
    }

    unsafe impl Send for SharedMappedBrain {}
    unsafe impl Sync for SharedMappedBrain {}
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use brain_pack::{BrainPackBuilder, ModelInfo, Metadata, TensorInfo, DataType, QuantizationScheme};
    use tempfile::tempdir;

    #[test]
    fn test_mmap_roundtrip() {
        let dir = tempdir().unwrap();
        let brain_path = dir.path().join("test.brain");

        // Create test brain file
        let model = ModelInfo {
            name: "test".to_string(),
            architecture: "test".to_string(),
            parameter_count: 100,
            quantization: "f16".to_string(),
            context_length: 2048,
            vocab_size: 32000,
        };

        let metadata = Metadata {
            created_epoch: 1234567890,
            created_by: "test".to_string(),
            checksum: String::new(),
            license: "test".to_string(),
            description: "test".to_string(),
        };

        let tensor_data: Vec<u8> = (0..100).map(|i| (i % 256) as u8).collect();

        let pack = BrainPackBuilder::new()
            .model(model)
            .metadata(metadata)
            .add_tensor(TensorInfo {
                name: "weight".to_string(),
                shape: vec![10, 10],
                dtype: DataType::F16,
                offset: 0,
                size_bytes: 0,
                quantization: None,
                quantization_type: QuantizationScheme::None,
            }, &tensor_data).unwrap()
            .build().unwrap();

        pack.write(&brain_path).unwrap();

        // Memory map it
        let mapped = PosixMappedBrain::open(&brain_path).unwrap();
        assert_eq!(mapped.pack.manifest.model.name, "test");
        assert_eq!(mapped.tensor_names().len(), 1);

        let tensor = mapped.tensor("weight").unwrap();
        assert_eq!(tensor.data.len(), 100);
    }
}