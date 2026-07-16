use anyhow::{anyhow, Result};
use brain_pack::{BrainPack, TensorView};
use std::path::Path;
use std::sync::Arc;
use tracing::{debug, info, warn};
use windows::core::PCWSTR;
use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::Storage::FileSystem::{
    CreateFileW, FILE_ATTRIBUTE_NORMAL, FILE_SHARE_READ, OPEN_EXISTING,
};
use windows::Win32::System::Memory::{
    CreateFileMappingW, MapViewOfFile, UnmapViewOfFile, FILE_MAP_READ, MEMORY_MAPPED_VIEW_ADDRESS,
    PAGE_READONLY,
};

// ============================================================================
// Windows Memory Mapper
// ============================================================================

/// Memory-mapped .brain file on Windows using Win32 API
pub struct WindowsMappedBrain {
    file_handle: HANDLE,
    mapping_handle: HANDLE,
    view_ptr: *mut u8,
    file_size: usize,
    pack: BrainPack,
    _marker: std::marker::PhantomData<Arc<()>>,
}

unsafe impl Send for WindowsMappedBrain {}
unsafe impl Sync for WindowsMappedBrain {}

impl WindowsMappedBrain {
    /// Open and memory-map a .brain file on Windows
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let wide_path = to_wide_string(
            path.to_str().ok_or_else(|| anyhow!("Invalid UTF-8 path: {}", path.display()))?,
        );

        // 1. Open file
        let file_handle = unsafe {
            CreateFileW(
                PCWSTR(wide_path.as_ptr()),
                0x80000000, // GENERIC_READ
                FILE_SHARE_READ,
                None,
                OPEN_EXISTING,
                FILE_ATTRIBUTE_NORMAL,
                None,
            )?
        };

        // 2. Get file size
        let file_size = {
            let mut file = std::fs::File::open(path)?;
            file.metadata()?.len() as usize
        };

        // 3. Create file mapping
        let mapping_handle = unsafe {
            CreateFileMappingW(
                file_handle,
                None,
                PAGE_READONLY,
                0,
                0,
                PCWSTR::null(),
            )?
        };
        if mapping_handle.is_invalid() {
            let _ = unsafe { CloseHandle(file_handle) };
            return Err(anyhow!("CreateFileMappingW returned invalid handle"));
        }

        // 4. Map view
        let view = unsafe {
            MapViewOfFile(
                mapping_handle,
                FILE_MAP_READ,
                0,
                0,
                0,
            )
        };
        if view.Value.is_null() {
            let _ = unsafe { CloseHandle(mapping_handle) };
            let _ = unsafe { CloseHandle(file_handle) };
            return Err(anyhow!("MapViewOfFile returned null"));
        }
        let view_ptr = view.Value as *mut u8;

        // 5. Parse .brain file from the mapped slice.
        // The mapped view's bytes are the on-disk .brain container, which
        // BrainPack::read_from_bytes expects verbatim.
        let bytes = unsafe { std::slice::from_raw_parts(view_ptr, file_size) };
        let pack = BrainPack::read_from_bytes(bytes)?;

        info!(
            "Memory-mapped .brain file: {} ({} bytes, {} tensors)",
            path.display(),
            file_size,
            pack.tensor_names().len()
        );

        Ok(Self {
            file_handle,
            mapping_handle,
            view_ptr,
            file_size,
            pack,
            _marker: std::marker::PhantomData,
        })
    }

    /// Borrow the parsed BrainPack.
    pub fn pack(&self) -> &BrainPack {
        &self.pack
    }

    /// Get a tensor view by name (zero-copy borrow of the mapped weights).
    pub fn get_tensor(&self, name: &str) -> Result<TensorView<'_>> {
        self.pack.get_tensor(name)
    }

    /// Names of all tensors in the pack.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.pack.tensor_names()
    }

    /// Raw mapped view pointer (unsafe to use directly unless caller
    /// understands the .brain layout).
    pub unsafe fn view_ptr(&self) -> *const u8 {
        self.view_ptr
    }

    pub fn file_size(&self) -> usize {
        self.file_size
    }
}

impl Drop for WindowsMappedBrain {
    fn drop(&mut self) {
        unsafe {
            let view = MEMORY_MAPPED_VIEW_ADDRESS { Value: self.view_ptr as *mut std::ffi::c_void };
            let _ = UnmapViewOfFile(view);
            let _ = CloseHandle(self.mapping_handle);
            let _ = CloseHandle(self.file_handle);
        }
        debug!("Unmapped .brain file ({} bytes)", self.file_size);
    }
}

// ---------------------------------------------------------------------------
// Rolling Block Overlay Mapper (Win32)
// ---------------------------------------------------------------------------
//
// `ShardOverlay` is the Windows counterpart of `mem_posix::ShardOverlay`.
// It maps the always-resident base fragment (`00.brain`) up-front and
// rolls streaming fragments (`01.brain`..`15.brain`) in and out via
// `MapViewOfFile` / `UnmapViewOfFile`, mirroring the POSIX mmadvise +
// mmap/munmap flow. Resident shard count is gated by the LRU driver in
// `stream-cache`, not here.

struct MappedShard {
    file_handle: HANDLE,
    mapping_handle: HANDLE,
    view_ptr: *mut u8,
    view_len: usize,
}

impl Drop for MappedShard {
    fn drop(&mut self) {
        unsafe {
            if !self.view_ptr.is_null() {
                let view = MEMORY_MAPPED_VIEW_ADDRESS { Value: self.view_ptr as *mut std::ffi::c_void };
                let _ = UnmapViewOfFile(view);
            }
            if !self.mapping_handle.is_invalid() {
                let _ = CloseHandle(self.mapping_handle);
            }
            if !self.file_handle.is_invalid() {
                let _ = CloseHandle(self.file_handle);
            }
        }
    }
}

unsafe impl Send for MappedShard {}

pub struct ShardOverlay {
    model: brain_pack::ShardedModel,
    mapped: std::collections::HashMap<u16, MappedShard>,
}

unsafe impl Send for ShardOverlay {}
unsafe impl Sync for ShardOverlay {}

impl ShardOverlay {
    /// Open a sharded model and map the resident base fragment.
    pub fn open(dir: &Path, name: &str) -> Result<Self> {
        let model = brain_pack::ShardedModel::open(dir, name)?;
        let mut mapped = std::collections::HashMap::new();
        // Always-resident base fragment (00.brain).
        let base = map_shard_file_readonly(&model.shard_paths[0])?;
        mapped.insert(0, base);
        Ok(Self { model, mapped })
    }

    pub fn index(&self) -> &brain_pack::ShardIndex {
        &self.model.index
    }

    pub fn base_pack(&self) -> &brain_pack::BrainPack {
        &self.model.base
    }

    pub fn is_mapped(&self, shard_id: u16) -> bool {
        self.mapped.contains_key(&shard_id)
    }

    /// Map a streaming shard's bytes into RAM. Idempotent.
    pub fn map_shard(&mut self, shard_id: u16) -> Result<&[u8]> {
        if shard_id == 0 {
            return Ok(self.shard_bytes(0).unwrap_or(&[]));
        }
        if self.mapped.contains_key(&shard_id) {
            return Ok(self.shard_bytes(shard_id).unwrap_or(&[]));
        }
        let path = self.model.shard_paths.get(shard_id as usize)
            .ok_or_else(|| anyhow!("Unknown shard id {}", shard_id))?;
        let ms = map_shard_file_readonly(path)?;
        let (ptr, len) = (ms.view_ptr, ms.view_len);
        self.mapped.insert(shard_id, ms);
        unsafe { Ok(std::slice::from_raw_parts(ptr, len)) }
    }

    /// Unmap a streaming shard. Base (id 0) is never evicted.
    pub fn unmap_shard(&mut self, shard_id: u16) -> Result<()> {
        if shard_id == 0 {
            return Ok(());
        }
        if self.mapped.remove(&shard_id).is_some() {
            debug!("unmapped shard {}", shard_id);
        }
        Ok(())
    }

    pub fn shard_bytes(&self, shard_id: u16) -> Option<&[u8]> {
        self.mapped.get(&shard_id).map(|ms| unsafe { std::slice::from_raw_parts(ms.view_ptr, ms.view_len) })
    }

    /// Get raw pointer and length of a mapped shard without borrowing the
    /// lock guard. The caller must ensure the shard stays mapped (via the
    /// LRU pin count) while using the returned pointer.
    pub fn shard_ptr_len(&self, shard_id: u16) -> Option<(*const u8, usize)> {
        self.mapped.get(&shard_id).map(|ms| (ms.view_ptr as *const u8, ms.view_len))
    }

    pub fn locate(&self, name: &str) -> Option<brain_pack::TensorLocation> {
        self.model.locate(name)
    }

    pub fn resident_shard_count(&self) -> usize {
        self.mapped.len()
    }
}

fn map_shard_file_readonly(path: &Path) -> Result<MappedShard> {
    let wide = to_wide_string(
        path.to_str().ok_or_else(|| anyhow!("Invalid UTF-8 path: {}", path.display()))?,
    );
    let file_handle = unsafe {
        CreateFileW(
            PCWSTR(wide.as_ptr()),
            0x80000000, // GENERIC_READ
            FILE_SHARE_READ,
            None,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL,
            None,
        )?
    };
    let file_size = std::fs::metadata(path)?.len() as usize;
    let mapping_handle = unsafe {
        CreateFileMappingW(file_handle, None, PAGE_READONLY, 0, 0, PCWSTR::null())?
    };
    if mapping_handle.is_invalid() {
        let _ = unsafe { CloseHandle(file_handle) };
        return Err(anyhow!("CreateFileMappingW returned invalid handle for {}", path.display()));
    }
    let view = unsafe { MapViewOfFile(mapping_handle, FILE_MAP_READ, 0, 0, 0) };
    if view.Value.is_null() {
        let _ = unsafe { CloseHandle(mapping_handle) };
        let _ = unsafe { CloseHandle(file_handle) };
        return Err(anyhow!("MapViewOfFile returned null for {}", path.display()));
    }
    Ok(MappedShard {
        file_handle,
        mapping_handle,
        view_ptr: view.Value as *mut u8,
        view_len: file_size,
    })
}

// ============================================================================
// Shared Memory (cross-process weight sharing)
// ============================================================================

/// Named shared memory mapping for cross-process weight sharing on Windows.
pub struct WindowsSharedMemory {
    mapping_handle: HANDLE,
    view_ptr: *mut u8,
    size: usize,
    _marker: std::marker::PhantomData<Arc<()>>,
}

unsafe impl Send for WindowsSharedMemory {}
unsafe impl Sync for WindowsSharedMemory {}

impl WindowsSharedMemory {
    /// Create a new named shared-memory section of the given size.
    pub fn create(name: &str, size: usize) -> Result<Self> {
        let wide_name = to_wide_string(name);
        let mapping_handle = unsafe {
            CreateFileMappingW(
                HANDLE::default(),
                None,
                PAGE_READONLY,
                (size >> 32) as u32,
                (size & 0xFFFF_FFFF) as u32,
                PCWSTR(wide_name.as_ptr()),
            )?
        };
        if mapping_handle.is_invalid() {
            return Err(anyhow!("CreateFileMappingW returned invalid handle"));
        }
        let view = unsafe {
            MapViewOfFile(
                mapping_handle,
                FILE_MAP_READ,
                0,
                0,
                0,
            )
        };
        if view.Value.is_null() {
            let _ = unsafe { CloseHandle(mapping_handle) };
            return Err(anyhow!("MapViewOfFile returned null"));
        }
        Ok(Self {
            mapping_handle,
            view_ptr: view.Value as *mut u8,
            size,
            _marker: std::marker::PhantomData,
        })
    }

    /// Open an existing named shared-memory section by name.
    pub fn open(name: &str, size: usize) -> Result<Self> {
        let wide_name = to_wide_string(name);
        let mapping_handle = unsafe {
            windows::Win32::System::Memory::OpenFileMappingW(
                FILE_MAP_READ.0,
                false,
                PCWSTR(wide_name.as_ptr()),
            )?
        };
        if mapping_handle.is_invalid() {
            return Err(anyhow!("OpenFileMappingW returned invalid handle for {}", name));
        }
        let view = unsafe {
            MapViewOfFile(
                mapping_handle,
                FILE_MAP_READ,
                0,
                0,
                0,
            )
        };
        if view.Value.is_null() {
            let _ = unsafe { CloseHandle(mapping_handle) };
            return Err(anyhow!("MapViewOfFile returned null"));
        }
        Ok(Self {
            mapping_handle,
            view_ptr: view.Value as *mut u8,
            size,
            _marker: std::marker::PhantomData,
        })
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.view_ptr, self.size) }
    }

    pub fn size(&self) -> usize {
        self.size
    }
}

impl Drop for WindowsSharedMemory {
    fn drop(&mut self) {
        unsafe {
            let view = MEMORY_MAPPED_VIEW_ADDRESS { Value: self.view_ptr as *mut std::ffi::c_void };
            let _ = UnmapViewOfFile(view);
            let _ = CloseHandle(self.mapping_handle);
        }
        debug!("Unmapped shared memory ({} bytes)", self.size);
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn to_wide_string(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0u16)).collect()
}

// ============================================================================
// Trait impl: MemoryBackend
// ============================================================================

pub trait WindowsMemoryOps {
    fn mapped_view_size(&self) -> usize;
    fn tensors(&self) -> Vec<&str>;
    fn weights_byte_len(&self) -> usize;
}

impl WindowsMemoryOps for WindowsMappedBrain {
    fn mapped_view_size(&self) -> usize {
        self.file_size
    }

    fn tensors(&self) -> Vec<&str> {
        self.tensor_names()
    }

    fn weights_byte_len(&self) -> usize {
        self.pack.weights.len()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_wide_string_terminates_with_nul() {
        let v = to_wide_string("abc");
        assert_eq!(v, vec![b'a' as u16, b'b' as u16, b'c' as u16, 0]);
    }
}
