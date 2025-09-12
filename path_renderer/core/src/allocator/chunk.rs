use std::sync::{Arc, Mutex};
use std::ptr::{self, NonNull};

use crate::allocator::bump::ChunkHeader;

use super::{Allocator, Layout, AllocError, Global};

pub(crate) const CHUNK_ALIGNMENT: usize = 32;
pub(crate) const DEFAULT_CHUNK_SIZE: usize = 128 * 1024;

#[derive(Clone)]
pub struct ChunkPool {
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
        ChunkPool {
            inner: Arc::new(Mutex::new(ChunkPoolImpl {
                first: None,
                count: 0,
            })),
        }
    }

    /// Pop a chunk from the pool or allocate a new one.
    ///
    /// If the requested size is not equal to the default chunk size,
    /// a new chunk is allocated.
    pub(crate) fn allocate_chunk(&self, size: usize) -> Result<NonNull<ChunkHeader>, AllocError> {
        let chunk: Option<NonNull<AvailableChunk>> = if size == DEFAULT_CHUNK_SIZE {
            // Try to reuse a chunk.
            let mut inner = self.inner.lock().unwrap();
            let mut chunk = inner.first.take();
            inner.first = chunk.as_mut().and_then(|chunk| unsafe { chunk.as_mut().next.take() });

            if chunk.is_some() {
                inner.count -= 1;
                debug_assert!(inner.count >= 0);
            }

            chunk
        } else {
            // Always allocate a new chunk if it is not the standard size.
            None
        };

        let chunk: NonNull<ChunkHeader> = match chunk {
            Some(chunk) => chunk.cast(),
            None => {
                // Allocate a new one.
                let layout = match Layout::from_size_align(size, CHUNK_ALIGNMENT) {
                    Ok(layout) => layout,
                    Err(_) => {
                        return Err(AllocError);
                    }
                };

                let alloc = Global.allocate(layout)?;

                alloc.cast()
            }
        };
        println!(" + alloc chunk {:?}", chunk);

        let chunk_start: *mut u8 = chunk.cast().as_ptr();

        unsafe {
            let chunk_end = chunk_start.add(size);
            let cursor = chunk_start.add(CHUNK_ALIGNMENT);
            ptr::write(
                chunk.as_ptr(),
                ChunkHeader {
                    previous: None,
                    chunk_end,
                    cursor,
                    size,
                },
            );
        }

        Ok(chunk)
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
    pub(crate) unsafe fn recycle_chunks(&self, chunk: NonNull<ChunkHeader>, stop: Option<NonNull<ChunkHeader>>) {
        let mut inner = self.inner.lock().unwrap();
        let mut iter = Some(chunk);
        // Go through the provided linked list of chunks, and insert each
        // of them at the beginning of our linked list of recycled chunks.
        while let Some(mut chunk) = iter {
            if iter == stop {
                break;
            }
            // Advance the iterator.
            iter = unsafe { chunk.as_mut().previous.take() };

            unsafe {
                // Don't recycle chunks with a non-standard size.
                let size = chunk.as_ref().size;
                if size != DEFAULT_CHUNK_SIZE {
                    let layout = Layout::from_size_align(size, CHUNK_ALIGNMENT).unwrap();
                    println!(" - dealloc large chunk {:?}", chunk);
                    Global.deallocate(chunk.cast(), layout);
                    continue;
                }

                #[cfg(feature = "poison_frame_memory")]
                super::bump::poison_chunk(chunk);
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
            unsafe {
                // First can't be None because inner.count > 0.
                let chunk = inner.first.unwrap();

                // Pop chunk off the list.
                inner.first = chunk.as_ref().next;

                // Deallocate chunk.
                let layout = Layout::from_size_align(
                    DEFAULT_CHUNK_SIZE,
                    CHUNK_ALIGNMENT
                ).unwrap();
                println!(" - dealloc chunk {:?}", chunk);
                Global.deallocate(chunk.cast(), layout);
            }

            inner.count -= 1;
            count -= 1;

            if count == 0 {
                return false;
            }
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

unsafe impl Send for ChunkPoolImpl {}
