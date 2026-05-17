use std::sync::Mutex;
use std::{ptr::NonNull, sync::Arc};
use std::ptr;
use crate::allocator::{Allocator, Layout, AllocError, Global};

// TODO: 16 might be enough in practice.
pub(crate) const CHUNK_ALIGNMENT: usize = 32;
pub(crate) const DEFAULT_CHUNK_SIZE: usize = 128 * 1024;


#[derive(Clone)]
pub struct ChunkPool {
    /// Preferred total allocation size for pooled chunks.
    /// Chunks of a different size are freed immediately rather than recycled.
    pub(crate) chunk_size: usize,
    inner: Arc<Mutex<ChunkPoolImpl>>,
}

struct ChunkPoolImpl {
    first: Option<NonNull<AvailableChunk>>,
    count: i32,
}

struct AvailableChunk {
    next: Option<NonNull<AvailableChunk>>,
}

impl ChunkPool {
    pub fn new() -> Self {
        Self::with_chunk_size(DEFAULT_CHUNK_SIZE)
    }

    pub fn with_chunk_size(chunk_size: usize) -> Self {
        ChunkPool {
            chunk_size,
            inner: Arc::new(Mutex::new(ChunkPoolImpl {
                first: None,
                count: 0,
            })),
        }
    }

    /// Put the provided list of chunks into the pool.
    ///
    /// Chunks with size different from the default chunk size are deallocated
    /// immediately.
    ///
    /// # Safety
    ///
    /// Ownership of the provided chunks is transfered to the pool, nothing
    /// else can access them after this function runs.
    pub(crate) unsafe fn recycle_chunks(&self, chunks: &mut dyn Iterator<Item=(NonNull<u8>, usize)>) {
        let mut inner = self.inner.lock().unwrap();
        for (chunk, size) in chunks {
            #[cfg(debug_assertions)]
            poison_memory(chunk, size);

            if size != self.chunk_size {
                let layout = Layout::from_size_align(size, CHUNK_ALIGNMENT).unwrap();
                //println!(" - dealloc large chunk {:?}", chunk);
                Global.deallocate(chunk, layout);
                continue;
            }

            // Turn the chunk into a recycled chunk.
            let recycled: NonNull<AvailableChunk> = chunk.cast();

            // Insert into the recycled list.
            unsafe {
                ptr::write(recycled.as_ptr(), AvailableChunk {
                    next: inner.first,
                });
            }
            inner.first = Some(recycled);

            inner.count += 1;
        }
    }

    /// Acquire a raw memory block.
    ///
    /// The actual allocation size is `max(min_size, self.chunk_size)`.  A recycled
    /// block is returned when available and the chosen size equals `self.chunk_size`.
    /// No header is written; the caller is responsible for initialising the memory.
    ///
    /// Returns `(ptr, actual_size)`.
    pub(crate) fn allocate_raw(&self, min_size: usize) -> Result<(NonNull<u8>, usize), AllocError> {
        let size = min_size.max(self.chunk_size);
        if size == self.chunk_size {
            let mut inner = self.inner.lock().unwrap();
            if let Some(mut available) = inner.first {
                inner.first = unsafe { available.as_mut().next.take() };
                inner.count -= 1;
                debug_assert!(inner.count >= 0);
                return Ok((available.cast(), size));
            }
        }
        let layout = Layout::from_size_align(size, CHUNK_ALIGNMENT).map_err(|_| AllocError)?;
        Ok((Global.allocate(layout)?.cast(), size))
    }

    /// Return a raw memory block to the pool.
    ///
    /// Blocks whose `size != self.chunk_size` are freed immediately rather than recycled.
    ///
    /// # Safety
    ///
    /// `ptr` must have been obtained via `allocate_raw` and must no longer be in use.
    pub(crate) unsafe fn recycle_raw(&self, ptr: NonNull<u8>, size: usize) {
        if size != self.chunk_size {
            let layout = Layout::from_size_align(size, CHUNK_ALIGNMENT).unwrap();
            unsafe { Global.deallocate(ptr.cast(), layout); }
            return;
        }
        let recycled: NonNull<AvailableChunk> = ptr.cast();
        let mut inner = self.inner.lock().unwrap();
        unsafe { ptr::write(recycled.as_ptr(), AvailableChunk { next: inner.first }); }
        inner.first = Some(recycled);
        inner.count += 1;
    }

    /// Deallocate chunks until the pool contains at most `target` items, or
    /// `count` chunks have been deallocated.
    ///
    /// Returns `true` if the target number of chunks in the pool was reached,
    /// `false` if this method stopped before reaching the target.
    ///
    /// Purging chunks can be expensive so it is preferable to perform this
    /// operation outside of the critical path. Specifying a lower `count`
    /// allows the caller to split the work and spread it over time.
    #[inline(never)]
    pub fn purge_chunks(&self, target: u32, mut count: u32) -> bool {
        let mut inner = self.inner.lock().unwrap();
        assert!(inner.count >= 0);

        while inner.count as u32 > target {
            if count == 0 {
                return false;
            }

            unsafe {
                // First can't be None because inner.count > 0.
                let chunk = inner.first.unwrap();

                // Pop chunk off the list.
                inner.first = chunk.as_ref().next;

                // Deallocate chunk.
                let layout = Layout::from_size_align(
                    self.chunk_size,
                    CHUNK_ALIGNMENT
                ).unwrap();
                //println!(" - dealloc chunk {:?}", chunk);
                Global.deallocate(chunk.cast(), layout);
            }

            inner.count -= 1;
            count -= 1;
        }

        return true;
    }

    /// Deallocate all of the chunks.
    pub fn purge_all_chunks(&self) {
        self.purge_chunks(0, u32::MAX);
    }
}

impl Drop for ChunkPool {
    fn drop(&mut self) {
        self.purge_all_chunks();
    }
}

// SAFETY: ChunkPoolImpl contains NonNull pointers to heap-allocated chunks that
// it exclusively owns. All access is serialised by the Mutex in ChunkPool, so
// transferring ChunkPoolImpl between threads is safe.
unsafe impl Send for ChunkPoolImpl {}

#[allow(unused)]
pub(crate) unsafe fn poison_memory(chunk: NonNull<u8>, size: usize) {
    unsafe {
        let start: *mut u32 = chunk.add(core::mem::size_of::<AvailableChunk>()).as_ptr().cast::<u32>();
        let end: *const u32 = chunk.add(size).as_ptr().cast();
        let len = end.offset_from(start) as usize;
        let slice = std::slice::from_raw_parts_mut(start, len);
        slice.fill(0xDEADBEEF);
    }
}
