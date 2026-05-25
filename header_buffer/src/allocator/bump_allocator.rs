use std::{
    alloc::Layout,
    cell::{Cell, UnsafeCell},
    ptr::{self, NonNull},
};

use crate::allocator::chunk_pool::{ChunkPool, CHUNK_ALIGNMENT};
use crate::allocator::{chunk_pool::DEFAULT_CHUNK_SIZE, AllocError, Allocator};

const CHUNK_HEADER_SIZE: usize = 64;

// TODO:
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
    current_chunk: NonNull<Chunk>,
    chunks: ChunkPool,

    #[cfg(feature = "stats")]
    stats: Stats,
}

impl BumpAllocatorStorageImpl {
    #[cfg(feature = "stats")]
    fn get_stats(&mut self) -> Stats {
        let cur_utilization = Chunk::utilization(self.current_chunk);
        self.stats.chunk_utilization = self.stats.chunks as f32 + cur_utilization - 1.0;
        self.stats
    }

    #[cfg(not(feature = "stats"))]
    fn get_stats(&mut self) -> Stats {
        Stats::default()
    }

    fn allocation_count(&self) -> i32 {
        let mut count = 0;
        let mut iter = Some(self.current_chunk);
        unsafe {
            while let Some(chunk) = iter {
                count += chunk.as_ref().allocation_count;
                iter = chunk.as_ref().previous;
            }
        }

        count
    }
}

impl Drop for BumpAllocatorStorageImpl {
    fn drop(&mut self) {
        assert!(self.allocation_count() == 0);
        unsafe {
            let mut iter = ChunkIterator {
                current: Some(self.current_chunk),
                stop: None,
            };
            self.chunks.recycle_chunks(&mut iter);
        }
    }
}

const _SANITY_CHECK: () = {
    assert!(CHUNK_HEADER_SIZE >= std::mem::size_of::<Chunk>());
};

struct Chunk {
    previous: Option<NonNull<Chunk>>,
    chunk_end: *mut u8,
    cursor: *mut u8,
    size: usize,
    allocator: NonNull<BumpAllocatorStorageCell>,
    // Due to how Allocaors can migrate to new chunks, the reference count for
    // a given chunk can go below zero. Only the sum of all reference counts
    // of a BumpAllocatorStorage is meaningful.
    ref_count: i32,
    allocation_count: i32,
}

impl Chunk {
    fn allocate_item(this: NonNull<Chunk>, layout: Layout) -> Result<NonNull<[u8]>, ()> {
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

    unsafe fn deallocate_item(this: NonNull<Chunk>, item: NonNull<u8>, layout: Layout) {
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

    unsafe fn grow_item(
        this: NonNull<Chunk>,
        item: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, ()> {
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

    unsafe fn shrink_item(
        this: NonNull<Chunk>,
        item: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> NonNull<[u8]> {
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

    fn contains_item(this: NonNull<Chunk>, item: NonNull<u8>) -> bool {
        unsafe {
            let start: *mut u8 = this.cast::<u8>().as_ptr().add(CHUNK_HEADER_SIZE);
            let end: *mut u8 = (*this.as_ptr()).chunk_end;
            let item = item.as_ptr();

            start <= item && item < end
        }
    }

    #[cfg(feature = "stats")]
    fn available_size(this: NonNull<Chunk>) -> usize {
        unsafe {
            let this = this.as_ptr();
            (*this).chunk_end.offset_from((*this).cursor) as usize
        }
    }

    #[cfg(feature = "stats")]
    fn utilization(this: NonNull<Chunk>) -> f32 {
        let size = unsafe { (*this.as_ptr()).size } as f32;
        (size - Chunk::available_size(this) as f32) / size
    }
}

pub struct ChunkIterator {
    current: Option<NonNull<Chunk>>,
    stop: Option<NonNull<Chunk>>,
}

impl Iterator for ChunkIterator {
    type Item = (NonNull<u8>, usize);
    fn next(&mut self) -> Option<(NonNull<u8>, usize)> {
        if self.current == self.stop {
            return None;
        }

        let Some(mut chunk) = self.current else {
            return None;
        };

        unsafe {
            let size = chunk.as_ref().size;
            self.current = chunk.as_mut().previous.take();
            Some((chunk.cast(), size))
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

type BumpAllocatorStorageCell = UnsafeCell<BumpAllocatorStorageImpl>;

/// Wrapper around [`BumpAllocatorMemory`] that allows interior mutability
/// and implements the [`Allocator`] trait via [`BumpAllocator`] handles.
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
    inner: NonNull<BumpAllocatorStorageCell>,
}

impl BumpAllocatorStorage {
    pub fn new(chunks: ChunkPool) -> Self {
        let (raw, size) = chunks.allocate_raw(DEFAULT_CHUNK_SIZE).unwrap();
        let chunk: NonNull<Chunk> = raw.cast();

        let cell = Box::new(UnsafeCell::new(BumpAllocatorStorageImpl {
            current_chunk: chunk,
            chunks,
            #[cfg(feature = "stats")]
            stats: Stats::default(),
        }));

        let ptr = Box::into_raw(cell);
        let ptr = NonNull::new(ptr).unwrap();
        let chunk_start: *mut u8 = chunk.cast().as_ptr();
        unsafe {
            let chunk_end = chunk_start.add(size);
            let cursor = chunk_start.add(CHUNK_HEADER_SIZE);
            ptr::write(
                chunk.as_ptr(),
                Chunk {
                    previous: None,
                    chunk_end,
                    cursor,
                    size,
                    allocator: ptr,
                    ref_count: 0,
                    allocation_count: 0,
                },
            );
        }

        BumpAllocatorStorage { inner: ptr }
    }

    /// Obtain a [`BumpAllocator`] handle that implements [`Allocator`].
    ///
    /// The handle is reference-counted; cloning it increments the count and
    /// dropping it decrements the count. The storage must not be dropped
    /// while any handle is alive.
    pub fn allocator(self: &Self) -> BumpAllocator {
        unsafe {
            let mut chunk = (*self.inner.as_ref().get()).current_chunk;
            chunk.as_mut().ref_count += 1;

            BumpAllocator {
                chunk: Cell::new(chunk),
            }
        }
    }

    pub fn get_stats(&self) -> Stats {
        unsafe { self.inner().get_stats() }
    }

    /// The number of live [`BumpAllocator`] handles pointing to this storage.
    pub fn ref_count(&self) -> i32 {
        unsafe {
            let mut count = 0;
            let mut iter = Some(self.inner().current_chunk);
            while let Some(chunk) = iter {
                count += chunk.as_ref().ref_count;
                iter = chunk.as_ref().previous;
            }

            count
        }
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
        unsafe { &mut *(self.inner.as_ref()).get() }
    }
}

impl Drop for BumpAllocatorStorage {
    fn drop(&mut self) {
        let ref_count = self.ref_count();
        assert!(
            ref_count == 0,
            "BumpAllocatorStorage dropped with {ref_count} outstanding BumpAllocator handle(s)",
        );
        unsafe {
            let _ = Box::from_raw(self.inner.as_ptr());
        }
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
    chunk: Cell<NonNull<Chunk>>,
}

impl BumpAllocator {
    #[inline]
    fn add_ref(&self) {
        unsafe {
            self.chunk.get().as_mut().ref_count += 1;
        }
    }

    // Returns true if this was the last reference.
    #[inline]
    fn release_ref(&self) {
        unsafe {
            self.chunk.get().as_mut().ref_count -= 1;
        }
    }

    fn allocate_item(
        chunk: &Cell<NonNull<Chunk>>,
        layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        #[cfg(feature = "stats")]
        {
            let storage_mut = unsafe { &mut *chunk.as_mut().allocator.as_ref().get() };
            //let storage_mut = unsafe { self.storage() };
            storage_mut.stats.allocations += 1;
            storage_mut.stats.allocated_bytes += layout.size();
        }

        if let Ok(alloc) = Chunk::allocate_item(chunk.get(), layout) {
            unsafe {
                chunk.get().as_mut().allocation_count += 1;
            }
            return Ok(alloc);
        }

        let storage = unsafe { (*chunk.get().as_ptr()).allocator };

        unsafe {
            let mut storage_chunk = (*storage.as_ref().get()).current_chunk;
            if storage_chunk != chunk.get() {
                if let Ok(alloc) = Chunk::allocate_item(storage_chunk, layout) {
                    storage_chunk.as_mut().allocation_count += 1;
                    chunk.set(storage_chunk);
                    return Ok(alloc);
                }
            }
        }

        let mut new_chunk = Self::alloc_chunk(storage, layout.size())?;
        chunk.set(new_chunk);

        match Chunk::allocate_item(new_chunk, layout) {
            Ok(alloc) => {
                unsafe {
                    new_chunk.as_mut().allocation_count += 1;
                }
                return Ok(alloc);
            }
            Err(_) => {
                return Err(AllocError);
            }
        }
    }

    fn deallocate_item(mut chunk: NonNull<Chunk>, ptr: NonNull<u8>, layout: Layout) {
        #[cfg(feature = "stats")]
        {
            let storage_mut = unsafe { &mut *chunk.as_mut().allocator.as_ref().get() };
            storage_mut.stats.deallocations += 1;
        }

        unsafe { chunk.as_mut().allocation_count -= 1 };
        // If the allocation is in the current chunk, try to reclaim its memory,
        // otherwise it will be reclaimed at the end of the frame.
        if Chunk::contains_item(chunk, ptr) {
            unsafe {
                Chunk::deallocate_item(chunk, ptr, layout);
            }
        }
    }

    unsafe fn grow_item(
        chunk: &Cell<NonNull<Chunk>>,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        debug_assert!(
            new_layout.size() >= old_layout.size(),
            "`new_layout.size()` must be greater than or equal to `old_layout.size()`"
        );

        #[cfg(feature = "stats")]
        let storage_mut = unsafe { &mut *chunk.as_ref().allocator.as_ref().get() };

        #[cfg(feature = "stats")]
        {
            storage_mut.stats.reallocations += 1;
        }

        // If we can, attempt to grow the existing allocation, otherwise just create a new one
        // and copy. The original allocation's memory will be reclaimed at the end of the frame.
        if Chunk::contains_item(chunk.get(), ptr) {
            if let Ok(alloc) = Chunk::grow_item(chunk.get(), ptr, old_layout, new_layout) {
                #[cfg(feature = "stats")]
                {
                    storage_mut.stats.allocated_bytes += new_layout.size() - old_layout.size();
                    storage_mut.stats.in_place_reallocations += 1;
                }
                return Ok(alloc);
            }
        }

        let new_alloc = if let Ok(alloc) = Chunk::allocate_item(chunk.get(), new_layout) {
            alloc
        } else {
            let storage = chunk.get().as_ref().allocator;
            let new_chunk = BumpAllocator::alloc_chunk(storage, new_layout.size())?;
            chunk.set(new_chunk);
            Chunk::allocate_item(new_chunk, new_layout).map_err(|_| AllocError)?
        };

        #[cfg(feature = "stats")]
        {
            storage_mut.stats.allocated_bytes += new_layout.size();
            storage_mut.stats.reallocated_bytes += old_layout.size();
        }

        unsafe {
            ptr::copy_nonoverlapping(ptr.as_ptr(), new_alloc.as_ptr().cast(), old_layout.size());
        }

        Ok(new_alloc)
    }

    unsafe fn shrink_item(
        chunk: NonNull<Chunk>,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        debug_assert!(
            new_layout.size() <= old_layout.size(),
            "`new_layout.size()` must be smaller than or equal to `old_layout.size()`"
        );

        if Chunk::contains_item(chunk, ptr) {
            return unsafe { Ok(Chunk::shrink_item(chunk, ptr, old_layout, new_layout)) };
        }

        // Can't actually shrink, so return the full range of the previous allocation.
        Ok(NonNull::slice_from_raw_parts(ptr, old_layout.size()))
    }

    fn alloc_chunk(
        storage: NonNull<BumpAllocatorStorageCell>,
        item_size: usize,
    ) -> Result<NonNull<Chunk>, AllocError> {
        let storage_mut = unsafe { &mut *storage.as_ref().get() };
        let min_chunk_size = align(item_size, CHUNK_ALIGNMENT) + CHUNK_HEADER_SIZE;

        let (raw, size) = storage_mut.chunks.allocate_raw(min_chunk_size)?;
        let chunk: NonNull<Chunk> = raw.cast();
        let chunk_start: *mut u8 = chunk.cast().as_ptr();
        unsafe {
            let chunk_end = chunk_start.add(size);
            let cursor = chunk_start.add(CHUNK_HEADER_SIZE);
            ptr::write(
                chunk.as_ptr(),
                Chunk {
                    previous: Some(storage_mut.current_chunk),
                    chunk_end,
                    cursor,
                    size,
                    allocator: storage,
                    ref_count: 0,
                    allocation_count: 0,
                },
            );
        }

        storage_mut.current_chunk = chunk;

        #[cfg(feature = "stats")]
        {
            storage_mut.stats.chunks += 1;
        }

        Ok(chunk)
    }
}

impl Clone for BumpAllocator {
    fn clone(&self) -> Self {
        self.add_ref();
        BumpAllocator {
            chunk: self.chunk.clone(),
        }
    }
}

impl Drop for BumpAllocator {
    fn drop(&mut self) {
        self.release_ref();
    }
}

unsafe impl Allocator for BumpAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        BumpAllocator::allocate_item(&self.chunk, layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        BumpAllocator::deallocate_item(self.chunk.get(), ptr, layout)
    }

    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        BumpAllocator::grow_item(&self.chunk, ptr, old_layout, new_layout)
    }

    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        BumpAllocator::shrink_item(self.chunk.get(), ptr, old_layout, new_layout)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vector;

    fn make_storage() -> BumpAllocatorStorage {
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
        unsafe {
            assert_eq!(storage.inner().allocation_count(), 0);
        }
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

    // Note: this test intentionally leaks memeory but there is no way to tell
    // miri that it's OK.
    #[test]
    fn storage_drop_panics_with_live_handles() {
        let storage = make_storage();
        let alloc = storage.allocator();
        let mut v = Vector::new_in(alloc);
        v.push(1u32);

        // This leaks the storage so that alloc does not point to dangling
        // memory.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // Panics because alloc is still alive.
            drop(storage);
        }));
        assert!(result.is_err());
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
