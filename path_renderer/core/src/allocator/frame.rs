use header_buffer::allocator::scoped_allocator::{ScopedAllocator, ScopedAllocatorStorage};

use super::ChunkPool;
use super::{BumpAllocator, BumpAllocatorStorage};
use super::{ConcurrentBumpAllocator, ConcurrentBumpAllocatorStorage};

pub type LocalFrameVec<T> = allocator_api2::vec::Vec<T, BumpAllocator>;
pub type FrameVec<T> = allocator_api2::vec::Vec<T, ConcurrentBumpAllocator>;
pub type TmpVec<'l, T> = allocator_api2::vec::Vec<T, ScopedAllocator<'l>>;

/// Per-worker allocators.
pub struct WorkerAllocators {
    pub local: BumpAllocatorStorage,
    pub frame: ConcurrentBumpAllocatorStorage,
    pub tmp: ScopedAllocatorStorage,
}

impl WorkerAllocators {
    pub fn new(chunks: ChunkPool) -> WorkerAllocators {
        WorkerAllocators {
            local: BumpAllocatorStorage::new(chunks.clone()),
            frame: ConcurrentBumpAllocatorStorage::new(chunks.clone()),
            tmp: ScopedAllocatorStorage::new(chunks),
        }
    }
}
