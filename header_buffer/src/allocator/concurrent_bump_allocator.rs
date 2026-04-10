//! Lock-free bump allocator for concurrent use.
//!
//! Allocation within a chunk is lock-free (CAS on an atomic cursor).
//! Adding a new chunk acquires a short-lived mutex (rare: only when a chunk is full).
//! Deallocation is lock-free.

use std::alloc::Layout;
use std::marker::PhantomPinned;
use std::pin::Pin;
use std::ptr::{self, NonNull};
use std::sync::atomic::{AtomicI32, AtomicPtr, AtomicU32, AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::allocator::{Allocator, AllocError};
use crate::allocator::chunk_pool::{CHUNK_ALIGNMENT, CHUNK_HEADER_SIZE, ChunkPool};

/// Chunk header written at the start of every raw allocation.
///
/// The data area begins `DATA_OFFSET` bytes from the chunk pointer.
/// `DATA_OFFSET` is `size_of::<AtomicChunk>()` rounded up to `CHUNK_ALIGNMENT`.
///
/// The `next` field forms an intrusive singly-linked list of all live chunks,
/// with `ConcurrentBumpAllocator::current_chunk` as the head.  Links are set
/// once at chunk-creation time (under the slow-path mutex) and read only in
/// `reset` (under `&mut self`), so no additional synchronisation is needed
/// beyond what the mutex and `&mut self` already provide.
struct AtomicChunk {
    /// Current byte offset into the data area. Advanced atomically by allocators.
    cursor: AtomicUsize,
    /// Size of the data area in bytes (does not include the header).
    capacity: usize,
    /// Total raw allocation size (header + data), stored for `Global::deallocate`.
    size: usize,
    /// Next (older) chunk in the intrusive list, or null if this is the oldest.
    next: AtomicPtr<AtomicChunk>,
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
    /// Lock-free: uses a CAS loop on the atomic cursor.  Spurious CAS failures
    /// (e.g. from `compare_exchange_weak` on ARM) cause an immediate retry.
    ///
    /// Memory ordering: the caller must have loaded the chunk pointer with
    /// `Acquire` before calling this, which establishes visibility of the header
    /// (including `capacity`).  The cursor CAS itself uses `Relaxed` — the
    /// uniqueness of the claimed slot follows from the atomicity of the CAS, not
    /// from any additional ordering.
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
/// The pool may round up to its preferred chunk size.
/// `next` is the previous list head, threaded into the new chunk's header.
fn new_chunk(pool: &ChunkPool, min_size: usize, next: *mut AtomicChunk) -> Result<NonNull<AtomicChunk>, AllocError> {
    let (raw, total_size) = pool.allocate_raw(min_size)?;
    let chunk: NonNull<AtomicChunk> = raw.cast();
    unsafe {
        ptr::write(chunk.as_ptr(), AtomicChunk {
            cursor: AtomicUsize::new(0),
            capacity: total_size - CHUNK_HEADER_SIZE,
            size: total_size,
            next: AtomicPtr::new(next),
        });
    }
    Ok(chunk)
}

unsafe fn recycle_chunk(pool: &ChunkPool, chunk: NonNull<AtomicChunk>) {
    let size = unsafe { (*chunk.as_ptr()).size };
    unsafe { pool.recycle_raw(chunk.cast(), size); }
}

/// A bump allocator that can be shared and used concurrently from multiple threads.
///
/// Each call to [`allocate`](Self::allocate) loads the current chunk pointer
/// (with `Acquire`) and then does a CAS loop on the chunk's atomic cursor to
/// claim a slot.  No lock is taken on the common path.
///
/// When the current chunk is full, a mutex is acquired to install a new one.
/// The double-check pattern ensures at most one new chunk is added per full
/// chunk, even under heavy contention.
///
/// [`deallocate`](Self::deallocate) only decrements an atomic counter.  Memory
/// is not reclaimed per-item; call [`reset`](Self::reset) to free all chunks at
/// once.
///
/// Growing always allocates a new region and copies — in-place extension is
/// unsafe because another thread may have claimed the bytes immediately after
/// any given slot.  Shrinking returns a smaller slice view without reclaiming
/// the tail (same reason).
pub struct ConcurrentBumpAllocatorMemory {
    /// Head of the intrusive chunk list and the target for fast-path allocation.
    /// Loaded lock-free on the fast path; updated under `install_lock`.
    current_chunk: AtomicPtr<AtomicChunk>,
    /// Net outstanding allocations (allocs − deallocs).
    /// Only used for the debug assertion in `reset` / `drop`.
    allocation_count: AtomicI32,
    /// Source and sink for chunk memory.  Shared with other allocators so that
    /// recycled chunks can be reused across frames / threads.
    /// `chunks.chunk_size` is the preferred total allocation size per chunk.
    chunks: ChunkPool,
    /// Serialises concurrent attempts to install a new chunk.  Holds no data:
    /// the chunk list is maintained through the intrusive `next` pointers.
    install_lock: Mutex<()>,
}

// SAFETY: all interior mutability is mediated by atomics or a Mutex.
unsafe impl Send for ConcurrentBumpAllocatorMemory {}
unsafe impl Sync for ConcurrentBumpAllocatorMemory {}

impl ConcurrentBumpAllocatorMemory {
    pub fn new(chunks: ChunkPool) -> Self {
        ConcurrentBumpAllocatorMemory {
            current_chunk: AtomicPtr::new(ptr::null_mut()),
            allocation_count: AtomicI32::new(0),
            chunks,
            install_lock: Mutex::new(()),
        }
    }

    /// Allocate memory for `layout`.
    ///
    /// Alignments greater than `CHUNK_ALIGNMENT` are not supported.
    pub fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        debug_assert!(
            CHUNK_ALIGNMENT % layout.align() == 0,
            "layout alignment {} is not supported (CHUNK_ALIGNMENT = {})",
            layout.align(), CHUNK_ALIGNMENT,
        );

        let size = round_up(layout.size(), CHUNK_ALIGNMENT);

        // Fast path: CAS on the current chunk's cursor, no lock taken.
        let raw = self.current_chunk.load(Ordering::Acquire);
        if let Some(chunk) = NonNull::new(raw) {
            if let Ok(alloc) = AtomicChunk::try_allocate(chunk, size) {
                self.allocation_count.fetch_add(1, Ordering::Relaxed);
                return Ok(alloc);
            }
        }

        self.allocate_slow(size)
    }

    /// Slow path: acquire the lock, maybe install a new chunk, then allocate.
    #[cold]
    fn allocate_slow(&self, size: usize) -> Result<NonNull<[u8]>, AllocError> {
        let _guard = self.install_lock.lock().unwrap();

        // Re-check: another thread may have installed a new chunk while we waited.
        let old_head = self.current_chunk.load(Ordering::Acquire);
        if let Some(chunk) = NonNull::new(old_head) {
            if let Ok(alloc) = AtomicChunk::try_allocate(chunk, size) {
                self.allocation_count.fetch_add(1, Ordering::Relaxed);
                return Ok(alloc);
            }
        }

        // Install a new chunk large enough for this request.
        // Pass `old_head` so the new chunk's `next` links it into the list.
        // The pool applies max(min_size, pool.chunk_size) internally.
        let chunk = new_chunk(&self.chunks, size + CHUNK_HEADER_SIZE, old_head)?;

        // Release: ensures header writes are visible to any thread that
        // subsequently loads current_chunk with Acquire.
        self.current_chunk.store(chunk.as_ptr(), Ordering::Release);

        let alloc = AtomicChunk::try_allocate(chunk, size)
            .expect("freshly allocated chunk must be large enough");

        self.allocation_count.fetch_add(1, Ordering::Relaxed);
        Ok(alloc)
    }

    /// Signal that `ptr` is no longer in use.
    ///
    /// The memory is not immediately reclaimed; call [`reset`](Self::reset) to
    /// free all chunks.
    pub fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {
        let prev = self.allocation_count.fetch_sub(1, Ordering::Relaxed);
        debug_assert!(prev > 0, "more deallocations than allocations");
    }

    /// Grow an allocation by allocating a new, larger region and copying.
    ///
    /// The old allocation is counted as freed; the returned pointer must be
    /// passed to `deallocate` (or `grow` / `shrink`) when done.
    pub fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        debug_assert!(new_layout.size() >= old_layout.size());

        // In-place growth is unsafe in a concurrent context because another thread
        // may have claimed the bytes immediately following the old allocation.
        let new_alloc = self.allocate(new_layout)?; // count +1
        unsafe {
            ptr::copy_nonoverlapping(
                ptr.as_ptr(),
                new_alloc.as_ptr().cast(),
                old_layout.size(),
            );
        }
        // grow replaces the old allocation: net allocation_count change is zero.
        self.allocation_count.fetch_sub(1, Ordering::Relaxed);
        Ok(new_alloc)
    }

    /// Return a smaller view of an allocation.
    ///
    /// The tail bytes are not reclaimed (see `deallocate`).
    pub fn shrink(
        &self,
        ptr: NonNull<u8>,
        _old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        Ok(NonNull::slice_from_raw_parts(ptr, new_layout.size()))
    }

    /// Free all chunks and reset the allocator to its initial (empty) state.
    ///
    /// Requires `&mut self` to guarantee no concurrent access is possible.
    ///
    /// # Panics
    ///
    /// Panics if any allocations are still outstanding.
    pub fn reset(&mut self) {
        assert_eq!(
            *self.allocation_count.get_mut(),
            0,
            "reset() called while allocations are still outstanding",
        );

        // Walk the intrusive list from head (newest) to tail (oldest).
        let mut ptr = *self.current_chunk.get_mut();
        while let Some(chunk) = NonNull::new(ptr) {
            // Read `next` before recycling — `recycle_chunk` hands the memory back.
            ptr = *unsafe { (*chunk.as_ptr()).next.get_mut() };
            unsafe { recycle_chunk(&self.chunks, chunk); }
        }
        *self.current_chunk.get_mut() = ptr::null_mut();
    }
}

impl Default for ConcurrentBumpAllocatorMemory {
    fn default() -> Self {
        Self::new(ChunkPool::new())
    }
}

impl Drop for ConcurrentBumpAllocatorMemory {
    fn drop(&mut self) {
        // Reuse reset() logic but keep the panic for outstanding allocations,
        // matching the behaviour of BumpAllocator::drop.
        self.reset();
    }
}

#[inline]
fn round_up(val: usize, align: usize) -> usize {
    let rem = val % align;
    if rem == 0 { val } else { val + align - rem }
}

/// Wrapper around [`ConcurrentBumpAllocatorMemory`] that provides
/// reference-counted [`ConcurrentBumpAllocator`] handles implementing
/// the [`Allocator`] trait.
///
/// This type is `!Unpin` and must be used behind `Pin<Box<...>>` to ensure
/// a stable address for the raw pointers held by [`ConcurrentBumpAllocator`]
/// handles. The [`new`](Self::new) constructor returns `Pin<Box<Self>>`
/// directly.
///
/// The storage and its handles are `Send + Sync` — they can be shared
/// across threads freely.
///
/// # Panics
///
/// Dropping a `ConcurrentBumpAllocatorStorage` while
/// [`ConcurrentBumpAllocator`] handles are still alive will panic.
pub struct ConcurrentBumpAllocatorStorage {
    inner: ConcurrentBumpAllocatorMemory,
    ref_count: AtomicU32,
    _pin: PhantomPinned,
}

impl ConcurrentBumpAllocatorStorage {
    pub fn new(chunks: ChunkPool) -> Pin<Box<Self>> {
        Box::pin(ConcurrentBumpAllocatorStorage {
            inner: ConcurrentBumpAllocatorMemory::new(chunks),
            ref_count: AtomicU32::new(0),
            _pin: PhantomPinned,
        })
    }

    /// Obtain a [`ConcurrentBumpAllocator`] handle that implements [`Allocator`].
    ///
    /// The handle is reference-counted; cloning it increments the count and
    /// dropping it decrements the count. The storage must not be dropped
    /// while any handle is alive.
    pub fn allocator(self: &Pin<Box<Self>>) -> ConcurrentBumpAllocator {
        self.ref_count.fetch_add(1, Ordering::Relaxed);
        ConcurrentBumpAllocator {
            storage: &**self as *const Self as *mut Self,
        }
    }

    /// The number of live [`ConcurrentBumpAllocator`] handles pointing to
    /// this storage.
    pub fn ref_count(&self) -> u32 {
        self.ref_count.load(Ordering::Relaxed)
    }

    /// Access the underlying memory.
    pub fn memory(&self) -> &ConcurrentBumpAllocatorMemory {
        &self.inner
    }

    /// Mutably access the underlying memory for reset, etc.
    ///
    /// # Safety
    ///
    /// This obtains a mutable reference to the inner memory without moving
    /// the storage. The caller must ensure no [`ConcurrentBumpAllocator`]
    /// handles are concurrently accessing the memory.
    pub fn memory_mut(self: &mut Pin<Box<Self>>) -> &mut ConcurrentBumpAllocatorMemory {
        // SAFETY: we are not moving the ConcurrentBumpAllocatorStorage,
        // only accessing the inner field mutably.
        unsafe { &mut self.as_mut().get_unchecked_mut().inner }
    }
}

impl Drop for ConcurrentBumpAllocatorStorage {
    fn drop(&mut self) {
        let count = self.ref_count.load(Ordering::Relaxed);
        assert!(
            count == 0,
            "ConcurrentBumpAllocatorStorage dropped with {count} outstanding \
             ConcurrentBumpAllocator handle(s)",
        );
    }
}

/// A lightweight, cloneable allocator handle that implements [`Allocator`].
///
/// Created via [`ConcurrentBumpAllocatorStorage::allocator`]. Cloning
/// increments an atomic reference count; dropping decrements it. The backing
/// storage will panic on drop if any handles are still alive.
///
/// This type is `Send + Sync` and can be used from multiple threads
/// concurrently.
pub struct ConcurrentBumpAllocator {
    storage: *mut ConcurrentBumpAllocatorStorage,
}

// SAFETY: The inner ConcurrentBumpAllocatorMemory is Send + Sync (all
// mutability mediated by atomics / Mutex), and the ref_count is atomic.
unsafe impl Send for ConcurrentBumpAllocator {}
unsafe impl Sync for ConcurrentBumpAllocator {}

impl ConcurrentBumpAllocator {
    #[inline]
    fn memory(&self) -> &ConcurrentBumpAllocatorMemory {
        unsafe { &(*self.storage).inner }
    }
}

impl Clone for ConcurrentBumpAllocator {
    fn clone(&self) -> Self {
        unsafe { (*self.storage).ref_count.fetch_add(1, Ordering::Relaxed); }
        ConcurrentBumpAllocator { storage: self.storage }
    }
}

impl Drop for ConcurrentBumpAllocator {
    fn drop(&mut self) {
        unsafe { (*self.storage).ref_count.fetch_sub(1, Ordering::Relaxed); }
    }
}

unsafe impl Allocator for ConcurrentBumpAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.memory().allocate(layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        self.memory().deallocate(ptr, layout)
    }

    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        self.memory().grow(ptr, old_layout, new_layout)
    }

    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        self.memory().shrink(ptr, old_layout, new_layout)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vector;

    fn make_storage() -> Pin<Box<ConcurrentBumpAllocatorStorage>> {
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

        assert_eq!(
            storage.memory().allocation_count.load(Ordering::Relaxed),
            0,
        );
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
        assert_eq!(
            storage.memory().allocation_count.load(Ordering::Relaxed),
            0,
        );
    }
}
