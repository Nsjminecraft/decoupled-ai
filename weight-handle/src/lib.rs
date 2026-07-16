//! Volatile weight handle trait shared between the compute-core abstraction
//! (`engine-ipc`) and the sharded streaming cache (`stream-cache`).
//!
//! This crate exists purely to break what would otherwise be a dependency
//! cycle: `engine-ipc` declares kernels that accept volatile mapped chunks,
//! and `stream-cache` produces those chunks. Both depend on the trait here;
//! neither depends on the other via this contract.

/// Borrowed view onto a volatile, memory-mapped chunk of tensor data.
///
/// Implementors must guarantee that the bytes returned by [`as_bytes`] /
/// [`as_f16`] remain valid for as long as the `&self` reference is held. The
/// streaming LRU driver evicts the backing shard only after the lease is
/// dropped, so compute kernels should resolve the view, run, and let the
/// owning lease fall out of scope immediately.
pub trait VolatileWeights {
    /// Reinterpret the mapped bytes as a borrowed f16 slice. The caller is
    /// responsible for guaranteeing the underlying dtype is F16.
    fn as_f16(&self) -> &[half::f16];
    /// Raw byte view of the mapped region.
    fn as_bytes(&self) -> &[u8];
    /// Which streaming shard this lease belongs to (instrumentation).
    fn shard_id(&self) -> u16;
}
