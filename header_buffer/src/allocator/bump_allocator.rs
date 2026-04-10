use std::{
    alloc::Layout,
    cell::UnsafeCell,
    marker::PhantomPinned,
    pin::Pin,
    ptr::{self, NonNull},
};

use crate::allocator::{Allocator, AllocError};
use crate::allocator::chunk_pool::{CHUNK_ALIGNMENT, CHUNK_HEADER_SIZE, ChunkPool};

// TODO:
//  - Put the stats behind a feature flag (empty struct when the feature is not enabled)
//  - keep a pointer to the bump allocator storage in the chunk headers
//  - Keep a pointer to the chunk header instead of the storage in BumpAllocator to avoid an indirection.
//    - overall the idea is to only ever touch the storage when recording stats and when running out of
//      space in the current chunk.

#[derive(Copy, Clone, Debug, Default)]
pub struct Stats {
    /// Number of allocated memory chunks.
    pub chunks: u32,
    /// A measure of the sum of the utilization of each chunk.
    ///
    /// - When a chunk is full, it's utilization is 1.0.
    /// - for the current chunk, utilization is the fraction of used space
    ///   over the total size of the chunk.
    ///
    /// For example, a utilization of 2.25 means that 3 chunk are allocated
    /// and a quarter of the current chunk is used.
    pub chunk_utilization: f32,
    /// Total number of allocations.
    ///
    /// This does not take deallocations into account.
    pub allocations: u32,
    /// Total number of deallocations.
    pub deallocations: u32,
    /// Total number of reallocations.
    pub reallocations: u32,
    /// Number of reallocations that used the in-place fast path. These did
    /// not require copying data.
    pub in_place_reallocations: u32,
    /// Total number of bytes that were copied during reallocation.
    pub reallocated_bytes: usize,
    /// Total number of allocated bytes.
    ///
    /// This does not take deallocations into account.
    pub allocated_bytes: usize,
}

struct BumpAllocatorStorageImpl {
    current_chunk: Option<NonNull<Chunk>>,
    chunks: ChunkPool,
    allocation_count: i32,
    ref_count: i32,

    stats: Stats,
}

impl BumpAllocatorStorageImpl {
    fn new(chunks: ChunkPool) -> Self {
        BumpAllocatorStorageImpl {
            current_chunk: None,
            chunks,
            allocation_count: 0,
            ref_count: 0,
            stats: Stats::default(),
        }
    }

    fn get_stats(&mut self) -> Stats {
        let cur_utilization = self.current_chunk.map(|c| Chunk::utilization(c) - 1.0).unwrap_or(0.0);
        self.stats.chunk_utilization = self.stats.chunks as f32 + cur_utilization;
        self.stats
    }

    fn allocate_item(&mut self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.stats.allocations += 1;
        self.stats.allocated_bytes += layout.size();

        if let Some(chunk) = self.current_chunk {
            if let Ok(alloc) = Chunk::allocate_item(chunk, layout) {
                self.allocation_count += 1;
                return Ok(alloc);
            }
        }

        let chunk = self.alloc_chunk(layout.size())?;

        match Chunk::allocate_item(chunk, layout) {
            Ok(alloc) => {
                self.allocation_count += 1;
                    return Ok(alloc);
            }
            Err(_) => {
                return Err(AllocError);
            }
        }
    }

    fn deallocate_item(&mut self, ptr: NonNull<u8>, layout: Layout) {
        self.stats.deallocations += 1;

        // If we are deallocating an item them we allocated one and therefore
        // we must have a chunk.
        let current_chunk = self.current_chunk.unwrap();

        // If the allocation is in the current chunk, try to reclaim its memory,
        // otherwise it will be reclaimed at the end of the frame.
        if Chunk::contains_item(current_chunk, ptr) {
            unsafe { Chunk::deallocate_item(current_chunk, ptr, layout); }
        }

        // Either way, count this as deallocated.
        self.allocation_count -= 1;
        debug_assert!(self.allocation_count >= 0);
    }

    unsafe fn grow_item(&mut self, ptr: NonNull<u8>, old_layout: Layout, new_layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        debug_assert!(
            new_layout.size() >= old_layout.size(),
            "`new_layout.size()` must be greater than or equal to `old_layout.size()`"
        );

        self.stats.reallocations += 1;

        let current_chunk = self.current_chunk.unwrap();

        // If we can, attempt to grow the existing allocation, otherwise just create a new one
        // and copy. The original allocation's memory will be reclaimed at the end of the frame.
        if Chunk::contains_item(current_chunk, ptr) {
            if let Ok(alloc) = Chunk::grow_item(current_chunk, ptr, old_layout, new_layout) {
                self.stats.allocated_bytes += new_layout.size() - old_layout.size();
                self.stats.in_place_reallocations += 1;
                return Ok(alloc);
            }
        }

        let new_alloc = if let Ok(alloc) = Chunk::allocate_item(current_chunk, new_layout) {
            alloc
        } else {
            let chunk = self.alloc_chunk(new_layout.size())?;
            Chunk::allocate_item(chunk, new_layout).map_err(|_| AllocError)?
        };

        self.stats.allocated_bytes += new_layout.size();
        self.stats.reallocated_bytes += old_layout.size();

        unsafe {
            ptr::copy_nonoverlapping(ptr.as_ptr(), new_alloc.as_ptr().cast(), old_layout.size());
        }

        Ok(new_alloc)
    }

    unsafe fn shrink_item(&mut self, ptr: NonNull<u8>, old_layout: Layout, new_layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        debug_assert!(
            new_layout.size() <= old_layout.size(),
            "`new_layout.size()` must be smaller than or equal to `old_layout.size()`"
        );

        if let Some(chunk) = self.current_chunk {
            if Chunk::contains_item(chunk, ptr) {
                return unsafe { Ok(Chunk::shrink_item(chunk, ptr, old_layout, new_layout)) };
            }
        }
        // Can't actually shrink, so return the full range of the previous allocation.
        Ok(NonNull::slice_from_raw_parts(ptr, old_layout.size()))
    }

    fn alloc_chunk(&mut self, item_size: usize) -> Result<NonNull<Chunk>, AllocError> {
        let chunk = self.chunks.allocate_chunk(
            align(item_size, CHUNK_ALIGNMENT) + CHUNK_HEADER_SIZE,
            self.current_chunk,
        )?;

        self.current_chunk = Some(chunk);

        self.stats.chunks += 1;

        Ok(chunk)
    }
}

impl Drop for BumpAllocatorStorageImpl {
    fn drop(&mut self) {
        assert!(self.allocation_count == 0);
        unsafe {
            if let Some(chunk) = self.current_chunk {
                self.chunks.recycle_chunks(chunk, None);
            }
        }
    }
}


const _SANITY_CHECK: () = {
    assert!(CHUNK_HEADER_SIZE >= std::mem::size_of::<Chunk>());
};

pub(crate) struct Chunk {
    pub(crate) previous: Option<NonNull<Chunk>>,
    pub(crate) chunk_end: *mut u8,
    pub(crate) cursor: *mut u8,
    pub(crate) size: usize,
}

impl Chunk {
    pub fn allocate_item(this: NonNull<Chunk>, layout: Layout) -> Result<NonNull<[u8]>, ()> {
        debug_assert!(CHUNK_ALIGNMENT % layout.align() == 0);

        let size = align(layout.size(), CHUNK_ALIGNMENT);

        unsafe {
            let cursor = (*this.as_ptr()).cursor;
            let end = (*this.as_ptr()).chunk_end;
            let available_size = end.offset_from(cursor);

            if size as isize > available_size {
                return Err(());
            }

            let next = cursor.add(size);

            (*this.as_ptr()).cursor = next;

            let cursor = NonNull::new(cursor).unwrap();
            let suballocation: NonNull<[u8]> = NonNull::slice_from_raw_parts(cursor, size);

            Ok(suballocation)
        }
    }

    pub unsafe fn deallocate_item(this: NonNull<Chunk>, item: NonNull<u8>, layout: Layout) {
        debug_assert!(Chunk::contains_item(this, item));

        unsafe {
            let size = align(layout.size(), CHUNK_ALIGNMENT);
            let item_end = item.as_ptr().add(size);

            // If the item is the last allocation, then move the cursor back
            // to reuse its memory.
            if item_end == (*this.as_ptr()).cursor {
                (*this.as_ptr()).cursor = item.as_ptr();
            }
        }
    }

    pub unsafe fn grow_item(this: NonNull<Chunk>, item: NonNull<u8>, old_layout: Layout, new_layout: Layout) -> Result<NonNull<[u8]>, ()> {
        debug_assert!(Chunk::contains_item(this, item));

        let old_size = align(old_layout.size(), CHUNK_ALIGNMENT);
        let new_size = align(new_layout.size(), CHUNK_ALIGNMENT);
        let old_item_end = item.as_ptr().add(old_size);

        if old_item_end != (*this.as_ptr()).cursor {
            return Err(());
        }

        // The item is the last allocation. we can attempt to just move
        // the cursor if the new size fits.

        let chunk_end = (*this.as_ptr()).chunk_end;
        let available_size = chunk_end.offset_from(item.as_ptr());

        if new_size as isize > available_size {
            // Does not fit.
            return Err(());
        }

        let new_item_end = item.as_ptr().add(new_size);
        (*this.as_ptr()).cursor = new_item_end;

        Ok(NonNull::slice_from_raw_parts(item, new_size))
    }

    pub unsafe fn shrink_item(this: NonNull<Chunk>, item: NonNull<u8>, old_layout: Layout, new_layout: Layout) -> NonNull<[u8]> {
        debug_assert!(Chunk::contains_item(this, item));

        let old_size = align(old_layout.size(), CHUNK_ALIGNMENT);
        let new_size = align(new_layout.size(), CHUNK_ALIGNMENT);
        let old_item_end = item.as_ptr().add(old_size);

        // The item is the last allocation. we can attempt to just move
        // the cursor if the new size fits.

        if old_item_end == (*this.as_ptr()).cursor {
            let new_item_end = item.as_ptr().add(new_size);
            (*this.as_ptr()).cursor = new_item_end;
        }

        NonNull::slice_from_raw_parts(item, new_size)
    }

    pub fn contains_item(this: NonNull<Chunk>, item: NonNull<u8>) -> bool {
        unsafe {
            let start: *mut u8 = this.cast::<u8>().as_ptr().add(CHUNK_HEADER_SIZE);
            let end: *mut u8 = (*this.as_ptr()).chunk_end;
            let item = item.as_ptr();

            start <= item && item < end
        }
    }

    fn available_size(this: NonNull<Chunk>) -> usize {
        unsafe {
            let this = this.as_ptr();
            (*this).chunk_end.offset_from((*this).cursor) as usize
        }
    }

    pub(crate) fn utilization(this: NonNull<Chunk>) -> f32 {
        let size = unsafe { (*this.as_ptr()).size } as f32;
        (size - Chunk::available_size(this) as f32) / size
    }

    #[allow(unused)]
    pub(crate) unsafe fn poison_memory(this: NonNull<Chunk>) {
        unsafe {
            let start: *mut u32 = this.as_ptr().cast::<u32>().add(CHUNK_HEADER_SIZE / 4);
            let end: *const u32 = this.as_ref().chunk_end.cast();
            let len = end.offset_from(start) as usize;
            let slice = std::slice::from_raw_parts_mut(start, len);
            slice.fill(0xDEADBEEF);
        }
    }
}

fn align(val: usize, alignment: usize) -> usize {
    let rem = val % alignment;
    if rem == 0 {
        return val;
    }

    val + alignment - rem
}

/// Wrapper around [`BumpAllocatorMemory`] that allows interior mutability
/// and implements the [`Allocator`] trait via [`BumpAllocator`] handles.
///
/// This type is `!Unpin` and must be used behind `Pin<Box<...>>` to ensure
/// a stable address for the raw pointers held by [`BumpAllocator`] handles.
/// The [`new`](Self::new) constructor returns `Pin<Box<Self>>` directly.
///
/// This type is not `Sync`: it must not be shared across threads. Multiple
/// [`BumpAllocator`] handles (e.g. via `Clone` on containers that store the
/// allocator) are fine on a single thread because bump allocation is
/// inherently ordered.
///
/// # Panics
///
/// Dropping a `BumpAllocatorStorage` while [`BumpAllocator`] handles are
/// still alive will panic.
pub struct BumpAllocatorStorage {
    inner: UnsafeCell<BumpAllocatorStorageImpl>,
    _pin: PhantomPinned,
}

impl BumpAllocatorStorage {
    pub fn new(chunks: ChunkPool) -> Pin<Box<Self>> {
        Box::pin(BumpAllocatorStorage {
            inner: UnsafeCell::new(BumpAllocatorStorageImpl::new(chunks)),
            _pin: PhantomPinned,
        })
    }

    /// Obtain a [`BumpAllocator`] handle that implements [`Allocator`].
    ///
    /// The handle is reference-counted; cloning it increments the count and
    /// dropping it decrements the count. The storage must not be dropped
    /// while any handle is alive.
    pub fn allocator(self: &Pin<Box<Self>>) -> BumpAllocator {
        unsafe { (*self.inner.get()).ref_count += 1; }
        BumpAllocator {
            storage: &**self as *const Self as *mut Self,
        }
    }

    pub fn get_stats(&self) -> Stats {
        unsafe { self.inner().get_stats() }
    }

    /// The number of live [`BumpAllocator`] handles pointing to this storage.
    pub fn ref_count(&self) -> i32 {
        unsafe { (*self.inner.get()).ref_count }
    }

    /// Mutably access the underlying memory for stats, reset, etc.
    ///
    /// # Safety
    ///
    /// This obtains a mutable reference to the inner memory without moving
    /// the storage. The caller must ensure no [`BumpAllocator`] handles are
    /// concurrently accessing the memory.
    unsafe fn inner(&self) -> &mut BumpAllocatorStorageImpl {
        // SAFETY: we are not moving the BumpAllocatorStorage, only accessing
        // the inner UnsafeCell's content mutably.
        unsafe { &mut *self.inner.get() }
    }

}

impl Drop for BumpAllocatorStorage {
    fn drop(&mut self) {
        let ref_count = self.ref_count();
        assert!(
            ref_count == 0,
            "BumpAllocatorStorage dropped with {ref_count} outstanding BumpAllocator handle(s)",
        );
    }
}

/// A lightweight, cloneable allocator handle that implements [`Allocator`].
///
/// Created via [`BumpAllocatorStorage::allocator`]. Cloning increments a
/// reference count; dropping decrements it. The backing
/// [`BumpAllocatorStorage`] will panic on drop if any handles are still alive.
///
/// # Safety
///
/// Must only be used from a single thread. The backing
/// [`BumpAllocatorStorage`] must outlive all handles.
pub struct BumpAllocator {
    storage: *mut BumpAllocatorStorage,
}

impl BumpAllocator {
    #[inline]
    unsafe fn memory(&self) -> &mut BumpAllocatorStorageImpl {
        // SAFETY: the caller guarantees single-threaded access and that the
        // storage outlives this handle.
        unsafe { &mut *(*self.storage).inner.get() }
    }

    #[inline]
    fn add_ref(&self) {
        unsafe {
            (*self.storage).inner().ref_count += 1;
        }
    }

    // Returns true if this was the last reference.
    #[inline]
    fn release_ref(&self) -> bool {
        unsafe {
            let rc = &mut (*self.storage).inner().ref_count;
            *rc -= 1;
            *rc == 0
        }
    }
}

impl Clone for BumpAllocator {
    fn clone(&self) -> Self {
        self.add_ref();
        BumpAllocator { storage: self.storage }
    }
}

impl Drop for BumpAllocator {
    fn drop(&mut self) {
        self.release_ref();
    }
}

unsafe impl Allocator for BumpAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        unsafe {
            self.memory().allocate_item(layout)
        }
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        self.memory().deallocate_item(ptr, layout)
    }

    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        self.memory().grow_item(ptr, old_layout, new_layout)
    }

    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        self.memory().shrink_item(ptr, old_layout, new_layout)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vector;

    fn make_storage() -> Pin<Box<BumpAllocatorStorage>> {
        BumpAllocatorStorage::new(ChunkPool::new())
    }

    #[test]
    fn vec_push_and_read() {
        let storage = make_storage();
        let alloc = storage.allocator();

        let mut v: Vector<i32, BumpAllocator> = Vector::new_in(alloc);
        v.push(1);
        v.push(2);
        v.push(3);

        assert_eq!(v.len(), 3);
        assert_eq!(&v[..], &[1, 2, 3]);
    }

    #[test]
    fn vec_grow_triggers_realloc() {
        let storage = make_storage();
        let alloc = storage.allocator();

        let mut v: Vector<u32, BumpAllocator> = Vector::new_in(alloc);
        for i in 0..1000 {
            v.push(i);
        }

        assert_eq!(v.len(), 1000);
        for (i, val) in v.iter().enumerate() {
            assert_eq!(*val, i as u32);
        }
    }

    #[test]
    fn vec_with_capacity() {
        let storage = make_storage();
        let alloc = storage.allocator();

        let mut v: Vector<u64, BumpAllocator> = Vector::with_capacity_in(64, alloc);
        assert!(v.capacity() >= 64);

        for i in 0..64 {
            v.push(i);
        }
        assert_eq!(v.len(), 64);
        assert_eq!(v[0], 0);
        assert_eq!(v[63], 63);
    }

    #[test]
    fn vec_clone() {
        let storage = make_storage();
        let alloc = storage.allocator();

        let mut v1: Vector<i32, BumpAllocator> = Vector::new_in(alloc);
        v1.extend_from_slice(&[10, 20, 30]);

        let v2 = v1.clone();
        assert_eq!(v1, v2);
        assert_eq!(&v2[..], &[10, 20, 30]);
    }

    #[test]
    fn multiple_vecs_same_allocator() {
        let storage = make_storage();
        let alloc = storage.allocator();

        let mut v1: Vector<u8, BumpAllocator> = Vector::new_in(alloc.clone());
        let mut v2: Vector<u8, BumpAllocator> = Vector::new_in(alloc.clone());
        let mut v3: Vector<u8, BumpAllocator> = Vector::new_in(alloc);

        for i in 0..100u8 {
            v1.push(i);
            v2.push(i.wrapping_mul(2));
            v3.push(i.wrapping_mul(3));
        }

        assert_eq!(v1.len(), 100);
        assert_eq!(v2.len(), 100);
        assert_eq!(v3.len(), 100);
        assert_eq!(v1[50], 50);
        assert_eq!(v2[50], 100);
        assert_eq!(v3[50], 150);
    }

    #[test]
    fn vec_drop_decrements_allocation_count() {
        let storage = make_storage();

        {
            let alloc = storage.allocator();
            let mut v: Vector<i32, BumpAllocator> = Vector::new_in(alloc);
            v.extend_from_slice(&[1, 2, 3, 4, 5]);
        }

        // After the vec and allocator handle are dropped, allocation_count
        // should be back to zero.
        unsafe { assert_eq!(storage.inner().allocation_count, 0); }
    }

    #[test]
    fn ref_count_tracks_handles() {
        let storage = make_storage();
        assert_eq!(storage.ref_count(), 0);

        let a1 = storage.allocator();
        assert_eq!(storage.ref_count(), 1);

        let a2 = a1.clone();
        assert_eq!(storage.ref_count(), 2);

        drop(a1);
        assert_eq!(storage.ref_count(), 1);

        drop(a2);
        assert_eq!(storage.ref_count(), 0);
    }

    #[test]
    fn storage_drop_panics_with_live_handles() {
        let storage = make_storage();
        let alloc = storage.allocator();
        // TODO: instead of papering over the issue here, we should make it so
        // that dropping the storage actually leaks it before panicking if there
        // are external references, so that the external references don't touch
        // a deallocated storage during unwinding.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            drop(storage);
        }));
        assert!(result.is_err());
        // The storage has been freed (even though its Drop panicked), so we
        // must not let the handle's Drop access it.
        std::mem::forget(alloc);
    }

    #[test]
    fn vec_of_strings() {
        let storage = make_storage();
        let alloc = storage.allocator();

        let mut v: Vector<String, BumpAllocator> = Vector::new_in(alloc);
        v.push("hello".to_string());
        v.push("world".to_string());
        v.push("foo bar baz this is a longer string to avoid SSO".to_string());

        assert_eq!(v.len(), 3);
        assert_eq!(v[0], "hello");
        assert_eq!(v[2], "foo bar baz this is a longer string to avoid SSO");
    }
}
