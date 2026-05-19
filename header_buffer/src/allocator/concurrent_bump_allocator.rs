//! Lock-free bump allocator for concurrent use.
//!
//! Allocation within a chunk is lock-free and on the fast path only touches
//! the current chunk's header, the storage struct is not read or written.
//!
//! Adding a new chunk acquires a short-lived mutex (rare: only when a chunk
//! is full). Deallocation only decrements an atomic counter.

use std::alloc::Layout;
use std::marker::PhantomPinned;
use std::ptr::{self, NonNull};
use std::sync::atomic::{AtomicI32, AtomicPtr, AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::allocator::{Allocator, AllocError};
use crate::allocator::chunk_pool::{CHUNK_ALIGNMENT, ChunkPool};

const CHUNK_HEADER_SIZE: usize = 64;

/// Chunk header written at the start of every raw allocation.
///
/// The data area begins `CHUNK_HEADER_SIZE` bytes from the chunk pointer.
///
/// The `next` field forms an intrusive singly-linked list of all live chunks,
/// with `ConcurrentBumpAllocatorStorageImpl::current_chunk` as the head.
/// Links are set at chunk-creation time under the slow-path mutex.
///
/// `ref_count` and `allocation_count` live on the chunk so the fast path
/// does not need to touch the storage struct. Per-chunk values are not
/// individually meaningful (a handle's chunk pointer can be retargeted on
/// the slow path without adjusting either counter, so a handle may decrement
/// a chunk it did not allocate on). The SUM across all chunks is invariant
/// and equals the live handle count and net outstanding allocations
/// respectively; storage-wide accessors and the drop assertion walk the
/// chunk list to obtain it.
struct AtomicChunk {
    /// Current byte offset into the data area. Advanced atomically.
    cursor: AtomicUsize,
    /// Size of the data area in bytes (does not include the header).
    capacity: usize,
    /// Total raw allocation size (header + data), stored for recycling.
    size: usize,
    /// Next (older) chunk in the intrusive list, or null if this is the oldest.
    next: AtomicPtr<AtomicChunk>,
    /// Back-pointer to the storage impl. The impl is heap-allocated and
    /// never moved, so this pointer is stable for the chunk's lifetime.
    /// Read only on the slow path.
    storage: NonNull<ConcurrentBumpAllocatorStorageImpl>,
    /// Per-chunk live handle count. Sum across all chunks == total live handles.
    ref_count: AtomicI32,
    /// Per-chunk net allocation count. Sum across all chunks == net
    /// outstanding allocations (allocs - deallocs).
    allocation_count: AtomicI32,
}

const _SANITY_CHECK: () = {
    assert!(CHUNK_HEADER_SIZE >= std::mem::size_of::<AtomicChunk>());
};

impl AtomicChunk {
    #[inline]
    fn data_start(this: NonNull<Self>) -> *mut u8 {
        // SAFETY: CHUNK_HEADER_SIZE is always within the chunk allocation.
        unsafe { this.cast::<u8>().as_ptr().add(CHUNK_HEADER_SIZE) }
    }

    /// Attempt to reserve `size` bytes (must be a multiple of `CHUNK_ALIGNMENT`).
    ///
    /// Lock-free: CAS loop on the atomic cursor. The caller must have loaded
    /// the chunk pointer with `Acquire` before calling, which establishes
    /// visibility of the header (including `capacity`).
    #[inline]
    fn try_allocate(this: NonNull<Self>, size: usize) -> Result<NonNull<[u8]>, ()> {
        let chunk = unsafe { this.as_ref() };
        let mut current = chunk.cursor.load(Ordering::Relaxed);
        loop {
            let next = current + size;
            if next > chunk.capacity {
                return Err(());
            }
            match chunk.cursor.compare_exchange_weak(
                current,
                next,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    let ptr = unsafe {
                        NonNull::new_unchecked(Self::data_start(this).add(current))
                    };
                    return Ok(NonNull::slice_from_raw_parts(ptr, size));
                }
                Err(actual) => current = actual,
            }
        }
    }
}

/// Acquire a chunk from `pool` with at least `min_size` bytes (header + data).
/// `next` is threaded into the new chunk's intrusive-list link. `storage` is
/// the chunk's back-pointer to its owning storage impl.
fn new_chunk(
    pool: &ChunkPool,
    min_size: usize,
    next: *mut AtomicChunk,
    storage: NonNull<ConcurrentBumpAllocatorStorageImpl>,
) -> Result<NonNull<AtomicChunk>, AllocError> {
    let (raw, total_size) = pool.allocate_raw(min_size)?;
    let chunk: NonNull<AtomicChunk> = raw.cast();
    unsafe {
        ptr::write(chunk.as_ptr(), AtomicChunk {
            cursor: AtomicUsize::new(0),
            capacity: total_size - CHUNK_HEADER_SIZE,
            size: total_size,
            next: AtomicPtr::new(next),
            storage,
            ref_count: AtomicI32::new(0),
            allocation_count: AtomicI32::new(0),
        });
    }
    Ok(chunk)
}

unsafe fn recycle_chunk(pool: &ChunkPool, chunk: NonNull<AtomicChunk>) {
    let size = unsafe { (*chunk.as_ptr()).size };
    unsafe { pool.recycle_raw(chunk.cast(), size); }
}

#[inline]
fn round_up(val: usize, align: usize) -> usize {
    let rem = val % align;
    if rem == 0 { val } else { val + align - rem }
}

/// Inner storage. Heap-allocated and never moved so chunks' back-pointers
/// remain valid. Wrapped by [`ConcurrentBumpAllocatorStorage`] which owns
/// it via a raw `NonNull` so that a panicking outer `Drop` does not
/// deallocate this struct.
struct ConcurrentBumpAllocatorStorageImpl {
    /// Head of the intrusive chunk list and target for the slow path's
    /// re-check. Updated under `install_lock`. The fast path does not read
    /// this field, handles cache their own chunk pointer.
    current_chunk: AtomicPtr<AtomicChunk>,
    /// Source and sink for chunk memory.
    chunks: ChunkPool,
    /// Serialises concurrent attempts to install a new chunk.
    install_lock: Mutex<()>,
    _pin: PhantomPinned,
}

impl ConcurrentBumpAllocatorStorageImpl {
    fn walk_chunks_sum(&self, mut f: impl FnMut(&AtomicChunk) -> i32) -> i32 {
        let mut sum = 0i32;
        let mut ptr = self.current_chunk.load(Ordering::Acquire);
        while let Some(chunk) = NonNull::new(ptr) {
            unsafe {
                sum = sum.wrapping_add(f(chunk.as_ref()));
                ptr = chunk.as_ref().next.load(Ordering::Acquire);
            }
        }
        sum
    }

    /// Sum `(allocation_count, ref_count)` over the chunk list using
    /// non-atomic accesses. Requires `&mut self`.
    fn sum_counters_mut(&mut self) -> (i32, i32) {
        let mut total_alloc = 0i32;
        let mut total_ref = 0i32;
        let mut ptr = *self.current_chunk.get_mut();
        while let Some(chunk) = NonNull::new(ptr) {
            unsafe {
                total_alloc = total_alloc.wrapping_add(*(*chunk.as_ptr()).allocation_count.get_mut());
                total_ref = total_ref.wrapping_add(*(*chunk.as_ptr()).ref_count.get_mut());
                ptr = *(*chunk.as_ptr()).next.get_mut();
            }
        }
        (total_alloc, total_ref)
    }

    /// Return every chunk to the pool and clear `current_chunk`. Requires
    /// `&mut self`.
    fn recycle_all_chunks(&mut self) {
        let mut ptr = *self.current_chunk.get_mut();
        while let Some(chunk) = NonNull::new(ptr) {
            unsafe {
                ptr = *(*chunk.as_ptr()).next.get_mut();
                recycle_chunk(&self.chunks, chunk);
            }
        }
        *self.current_chunk.get_mut() = ptr::null_mut();
    }
}

impl Drop for ConcurrentBumpAllocatorStorageImpl {
    fn drop(&mut self) {
        // Reachable only after the outer `ConcurrentBumpAllocatorStorage::drop`
        // has verified that no handles or allocations remain. Chunks must be
        // returned to the pool here so the pool can reuse them.
        self.recycle_all_chunks();
    }
}

/// Backing storage for [`ConcurrentBumpAllocator`] handles.
///
/// Internally wraps a heap allocation of a private impl struct; chunks
/// store back-pointers to that impl, so its address is stable.
///
/// `ConcurrentBumpAllocatorStorage` and `ConcurrentBumpAllocator` are
/// `Send + Sync`.
///
/// # Panics
///
/// Dropping a `ConcurrentBumpAllocatorStorage` while
/// [`ConcurrentBumpAllocator`] handles or outstanding allocations remain
/// will panic. If the panic is caught, the inner backing memory is *not*
/// deallocated, so any surviving handles continue to point at valid memory.
pub struct ConcurrentBumpAllocatorStorage {
    inner: NonNull<ConcurrentBumpAllocatorStorageImpl>,
}

// SAFETY: the inner impl is itself Send + Sync (atomics, Mutex, ChunkPool);
// ownership of the heap allocation is unique to this wrapper.
unsafe impl Send for ConcurrentBumpAllocatorStorage {}
unsafe impl Sync for ConcurrentBumpAllocatorStorage {}

impl ConcurrentBumpAllocatorStorage {
    pub fn new(chunks: ChunkPool) -> Self {
        let boxed = Box::new(ConcurrentBumpAllocatorStorageImpl {
            current_chunk: AtomicPtr::new(ptr::null_mut()),
            chunks,
            install_lock: Mutex::new(()),
            _pin: PhantomPinned,
        });

        let mut inner = NonNull::new(Box::into_raw(boxed)).unwrap();

        // Allocate the initial chunk so handles can always rely on a
        // non-null chunk pointer on the fast path.
        unsafe {
            let chunk = new_chunk(
                &inner.as_ref().chunks,
                CHUNK_HEADER_SIZE,
                ptr::null_mut(),
                inner,
            ).expect("failed to allocate initial chunk");
            inner.as_mut().current_chunk = AtomicPtr::new(chunk.as_ptr());
        }

        ConcurrentBumpAllocatorStorage { inner }
    }

    #[inline]
    fn storage(&self) -> &ConcurrentBumpAllocatorStorageImpl {
        unsafe { self.inner.as_ref() }
    }

    /// Obtain a [`ConcurrentBumpAllocator`] handle that implements [`Allocator`].
    ///
    /// Cloning the handle increments a per-chunk atomic reference count;
    /// dropping decrements it. The storage must not be dropped while any
    /// handle is alive.
    pub fn allocator(&self) -> ConcurrentBumpAllocator {
        let chunk = self.storage().current_chunk.load(Ordering::Acquire);
        debug_assert!(!chunk.is_null());
        unsafe { (*chunk).ref_count.fetch_add(1, Ordering::Relaxed); }
        ConcurrentBumpAllocator {
            chunk: AtomicPtr::new(chunk),
        }
    }

    /// The number of live [`ConcurrentBumpAllocator`] handles pointing to
    /// this storage.
    ///
    /// Computed by walking the intrusive chunk list and summing per-chunk
    /// reference counts.
    pub fn ref_count(&self) -> i32 {
        self.storage().walk_chunks_sum(|c| c.ref_count.load(Ordering::Relaxed))
    }

    /// Net outstanding allocations (allocs - deallocs) across all chunks.
    pub fn allocation_count(&self) -> i32 {
        self.storage().walk_chunks_sum(|c| c.allocation_count.load(Ordering::Relaxed))
    }

    /// Free all chunks and reset to an initial (one empty chunk) state.
    ///
    /// # Panics
    ///
    /// Panics if any allocations or handles are still outstanding.
    pub fn reset(&mut self) {
        let mut storage_ptr = self.inner;
        let storage = unsafe { storage_ptr.as_mut() };

        let (total_alloc, total_ref) = storage.sum_counters_mut();
        assert_eq!(total_ref, 0, "reset() called with {total_ref} live handle(s)");
        assert_eq!(total_alloc, 0, "reset() called with {total_alloc} outstanding allocation(s)");

        // TODO: instead of recycling all chunks and reallocating, just leave one.
        storage.recycle_all_chunks();

        // Reinstall a fresh empty chunk so subsequent allocator() calls
        // produce handles with non-null chunk pointers.
        let chunk = new_chunk(&storage.chunks, CHUNK_HEADER_SIZE, ptr::null_mut(), storage_ptr)
            .expect("failed to allocate initial chunk after reset");
        *storage.current_chunk.get_mut() = chunk.as_ptr();
    }
}

impl Drop for ConcurrentBumpAllocatorStorage {
    fn drop(&mut self) {
        // The assertions below run before `Box::from_raw` reclaims the inner
        // allocation. If either panics, the inner is leaked rather than
        // freed and any surviving handles continue to point at valid memory.
        let (total_alloc, total_ref) = unsafe { self.inner.as_mut().sum_counters_mut() };
        assert!(
            total_ref == 0,
            "ConcurrentBumpAllocatorStorage dropped with {total_ref} outstanding handle(s)",
        );
        assert_eq!(
            total_alloc, 0,
            "ConcurrentBumpAllocatorStorage dropped with {total_alloc} outstanding allocation(s)",
        );

        // Both assertions passed: hand the box back so its Drop recycles
        // every chunk and the allocation itself is freed.
        unsafe {
            let _ = Box::from_raw(self.inner.as_ptr());
        }
    }
}

/// A lightweight, cloneable allocator handle that implements [`Allocator`].
///
/// Created via [`ConcurrentBumpAllocatorStorage::allocator`]. The fast path
/// only touches the current chunk's header (a single chunk-pointer load plus
/// a CAS on the chunk's cursor and a fetch_add on its allocation_count)
/// the storage struct is not read or written.
///
/// Cloning increments a per-chunk atomic reference count; dropping decrements
/// it. The backing storage will panic on drop if any handles are still alive.
///
/// This type is `Send + Sync`; the same handle can be used from multiple
/// threads concurrently.
pub struct ConcurrentBumpAllocator {
    /// Most recently observed current chunk. May lag behind the storage's
    /// `current_chunk`, the slow path catches up by re-reading and storing.
    /// Non-null from construction through drop.
    chunk: AtomicPtr<AtomicChunk>,
}

// SAFETY: chunks remain valid for the lifetime of any handle (storage drop
// asserts ref_count == 0). All access to the chunk header is atomic.
unsafe impl Send for ConcurrentBumpAllocator {}
unsafe impl Sync for ConcurrentBumpAllocator {}

impl ConcurrentBumpAllocator {
    fn allocate_impl(
        chunk: &AtomicPtr<AtomicChunk>,
        layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        debug_assert!(
            CHUNK_ALIGNMENT % layout.align() == 0,
            "layout alignment {} is not supported (CHUNK_ALIGNMENT = {})",
            layout.align(), CHUNK_ALIGNMENT,
        );
        let size = round_up(layout.size(), CHUNK_ALIGNMENT);

        // Fast path: only the chunk header is touched.
        let raw = chunk.load(Ordering::Acquire);
        // SAFETY: a handle's chunk pointer is non-null from construction
        // through drop.
        let current = unsafe { NonNull::new_unchecked(raw) };
        if let Ok(alloc) = AtomicChunk::try_allocate(current, size) {
            unsafe { current.as_ref().allocation_count.fetch_add(1, Ordering::Relaxed); }
            return Ok(alloc);
        }

        Self::allocate_slow(chunk, current, size)
    }

    #[cold]
    fn allocate_slow(
        chunk: &AtomicPtr<AtomicChunk>,
        current: NonNull<AtomicChunk>,
        size: usize,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // Reach the storage impl through the chunk's back-pointer.
        let mut storage_ptr = unsafe { current.as_ref().storage };
        let storage = unsafe { storage_ptr.as_mut() };

        let _guard = storage.install_lock.lock().unwrap();

        // Re-check: another thread may have installed a new chunk while we
        // waited on the lock.
        let storage_head = storage.current_chunk.load(Ordering::Acquire);
        if storage_head != current.as_ptr() {
            if let Some(head) = NonNull::new(storage_head) {
                if let Ok(alloc) = AtomicChunk::try_allocate(head, size) {
                    unsafe { head.as_ref().allocation_count.fetch_add(1, Ordering::Relaxed); }
                    chunk.store(storage_head, Ordering::Release);
                    return Ok(alloc);
                }
            }
        }

        // Install a new chunk.
        let new = new_chunk(
            &storage.chunks,
            size + CHUNK_HEADER_SIZE,
            storage_head,
            storage_ptr,
        )?;
        storage.current_chunk.store(new.as_ptr(), Ordering::Release);

        let alloc = AtomicChunk::try_allocate(new, size)
            .expect("freshly allocated chunk must be large enough");
        unsafe { new.as_ref().allocation_count.fetch_add(1, Ordering::Relaxed); }
        chunk.store(new.as_ptr(), Ordering::Release);
        Ok(alloc)
    }

    fn deallocate_impl(chunk: &AtomicPtr<AtomicChunk>, _ptr: NonNull<u8>, _layout: Layout) {
        // Decrement the count on whichever chunk this handle currently
        // points at. The actual allocation may have been on a different
        // chunk; the storage-wide sum is what's checked.
        let current = chunk.load(Ordering::Relaxed);
        unsafe { (*current).allocation_count.fetch_sub(1, Ordering::Relaxed); }
    }

    unsafe fn grow_impl(
        chunk: &AtomicPtr<AtomicChunk>,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        debug_assert!(new_layout.size() >= old_layout.size());

        // In-place growth is unsafe concurrently: another thread may already
        // have claimed the bytes immediately after this allocation.
        let new_alloc = Self::allocate_impl(chunk, new_layout)?;
        unsafe {
            ptr::copy_nonoverlapping(
                ptr.as_ptr(),
                new_alloc.as_ptr().cast(),
                old_layout.size(),
            );
        }
        // grow replaces the old allocation: net change in the storage-wide
        // allocation count is zero.
        Self::deallocate_impl(chunk, ptr, old_layout);
        Ok(new_alloc)
    }

    unsafe fn shrink_impl(
        _chunk: &AtomicPtr<AtomicChunk>,
        ptr: NonNull<u8>,
        _old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        Ok(NonNull::slice_from_raw_parts(ptr, new_layout.size()))
    }
}

impl Clone for ConcurrentBumpAllocator {
    fn clone(&self) -> Self {
        let chunk = self.chunk.load(Ordering::Relaxed);
        unsafe { (*chunk).ref_count.fetch_add(1, Ordering::Relaxed); }
        ConcurrentBumpAllocator {
            chunk: AtomicPtr::new(chunk),
        }
    }
}

impl Drop for ConcurrentBumpAllocator {
    fn drop(&mut self) {
        let chunk = self.chunk.load(Ordering::Relaxed);
        unsafe { (*chunk).ref_count.fetch_sub(1, Ordering::Relaxed); }
    }
}

unsafe impl Allocator for ConcurrentBumpAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        Self::allocate_impl(&self.chunk, layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        Self::deallocate_impl(&self.chunk, ptr, layout)
    }

    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        Self::grow_impl(&self.chunk, ptr, old_layout, new_layout)
    }

    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        Self::shrink_impl(&self.chunk, ptr, old_layout, new_layout)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vector;

    fn make_storage() -> ConcurrentBumpAllocatorStorage {
        ConcurrentBumpAllocatorStorage::new(ChunkPool::new())
    }

    #[test]
    fn vec_push_and_read() {
        let storage = make_storage();
        let alloc = storage.allocator();

        let mut v: Vector<i32, ConcurrentBumpAllocator> = Vector::new_in(alloc);
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

        let mut v: Vector<u32, ConcurrentBumpAllocator> = Vector::new_in(alloc);
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

        let mut v: Vector<u64, ConcurrentBumpAllocator> = Vector::with_capacity_in(64, alloc);
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

        let mut v1: Vector<i32, ConcurrentBumpAllocator> = Vector::new_in(alloc);
        v1.extend_from_slice(&[10, 20, 30]);

        let v2 = v1.clone();
        assert_eq!(v1, v2);
        assert_eq!(&v2[..], &[10, 20, 30]);
    }

    #[test]
    fn multiple_vecs_same_allocator() {
        let storage = make_storage();
        let alloc = storage.allocator();

        let mut v1: Vector<u8, ConcurrentBumpAllocator> = Vector::new_in(alloc.clone());
        let mut v2: Vector<u8, ConcurrentBumpAllocator> = Vector::new_in(alloc.clone());
        let mut v3: Vector<u8, ConcurrentBumpAllocator> = Vector::new_in(alloc);

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
            let mut v: Vector<i32, ConcurrentBumpAllocator> = Vector::new_in(alloc);
            v.extend_from_slice(&[1, 2, 3, 4, 5]);
        }

        assert_eq!(storage.allocation_count(), 0);
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

    // Note: this test intentionally leaks memory but there is no way to tell
    // miri that it's OK.
    #[test]
    fn storage_drop_panics_with_live_handles() {
        let storage = make_storage();
        let alloc = storage.allocator();
        let mut v = Vector::new_in(alloc);
        v.push(1u32);

        // The panic in storage Drop leaves the inner backing memory live, so
        // the handle (and its Vector) can still be safely dropped afterwards.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            drop(storage);
        }));
        assert!(result.is_err());
    }

    #[test]
    fn vec_of_strings() {
        let storage = make_storage();
        let alloc = storage.allocator();

        let mut v: Vector<String, ConcurrentBumpAllocator> = Vector::new_in(alloc);
        v.push("hello".to_string());
        v.push("world".to_string());
        v.push("foo bar baz this is a longer string to avoid SSO".to_string());

        assert_eq!(v.len(), 3);
        assert_eq!(v[0], "hello");
        assert_eq!(v[2], "foo bar baz this is a longer string to avoid SSO");
    }

    #[test]
    fn concurrent_allocation() {
        let storage = make_storage();

        let handles: Vec<std::thread::JoinHandle<Vec<i32>>> = (0..4)
            .map(|thread_id| {
                let alloc = storage.allocator();
                std::thread::spawn(move || {
                    let mut v: Vector<i32, ConcurrentBumpAllocator> = Vector::new_in(alloc);
                    for i in 0..500 {
                        v.push(thread_id * 1000 + i);
                    }
                    v.to_vec()
                })
            })
            .collect();

        let results: Vec<Vec<i32>> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        for (thread_id, result) in results.iter().enumerate() {
            assert_eq!(result.len(), 500);
            for (i, val) in result.iter().enumerate() {
                assert_eq!(*val, thread_id as i32 * 1000 + i as i32);
            }
        }

        assert_eq!(storage.ref_count(), 0);
        assert_eq!(storage.allocation_count(), 0);
    }
}
