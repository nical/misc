use std::{
    alloc::Layout,
    ptr::{self, NonNull}, sync::atomic::AtomicPtr,
};

use super::AllocError;
use super::chunk::{CHUNK_ALIGNMENT, ChunkPool, DEFAULT_CHUNK_SIZE};

/// A simple bump allocator, sub-allocating from fixed size chunks that are provided
/// by a parent allocator.
///
/// If an allocation is larger than the chunk size, a chunk sufficiently large to contain
/// the allocation is added.
pub struct BumpAllocator {
    current_chunk: NonNull<ChunkHeader>,
    chunks: ChunkPool,
    live_alloc_count: i32,
    stats: Stats,
}

impl BumpAllocator {
    pub fn new(chunk_pool: ChunkPool) -> Self {
        let mut stats = Stats::default();
        let first_chunk = chunk_pool.allocate_chunk(DEFAULT_CHUNK_SIZE).unwrap();
        stats.chunks = 1;
        stats.reserved_bytes += DEFAULT_CHUNK_SIZE;

        BumpAllocator {
            current_chunk: first_chunk,
            chunks: chunk_pool,
            live_alloc_count: 0,
            stats,
        }
    }

    pub fn allocate(&mut self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.stats.allocations += 1;
        self.stats.allocated_bytes += layout.size();

        if let Ok(alloc) = allocate_item_impl(self.current_chunk, layout) {
            self.live_alloc_count += 1;
            return Ok(alloc);
        }

        self.alloc_chunk(layout.size())?;

        let alloc = allocate_item_impl(self.current_chunk, layout);

        if alloc.is_ok() {
            self.live_alloc_count += 1;
        }

        alloc
    }

    pub fn deallocate(&mut self, ptr: NonNull<u8>, layout: Layout) {
        self.stats.deallocations += 1;

        if ChunkHeader::contains_item(self.current_chunk, ptr) {
            unsafe {
                deallocate_item_impl(self.current_chunk, ptr, layout);
            }
        }

        self.live_alloc_count -= 1;
        debug_assert!(self.live_alloc_count >= 0);
    }

    pub unsafe fn grow(
        &mut self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        debug_assert!(
            new_layout.size() >= old_layout.size(),
            "`new_layout.size()` must be greater than or equal to `old_layout.size()`"
        );

        self.stats.reallocations += 1;

        if ChunkHeader::contains_item(self.current_chunk, ptr) {
            unsafe {
                if let Ok(alloc) = grow_item_impl(self.current_chunk, ptr, old_layout, new_layout) {
                    self.stats.in_place_reallocations += 1;
                    return Ok(alloc);
                }
            }
        }

        let new_alloc = if let Ok(alloc) = allocate_item_impl(self.current_chunk, new_layout) {
            alloc
        } else {
            self.alloc_chunk(new_layout.size())?;
            allocate_item_impl(self.current_chunk, new_layout).map_err(|_| AllocError)?
        };

        self.stats.reallocated_bytes += old_layout.size();

        unsafe {
            ptr::copy_nonoverlapping(ptr.as_ptr(), new_alloc.as_ptr().cast(), old_layout.size());
        }

        Ok(new_alloc)
    }

    pub unsafe fn shrink(
        &mut self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        debug_assert!(
            new_layout.size() <= old_layout.size(),
            "`new_layout.size()` must be smaller than or equal to `old_layout.size()`"
        );

        if ChunkHeader::contains_item(self.current_chunk, ptr) {
            return unsafe {
                Ok(shrink_item_impl(
                    self.current_chunk,
                    ptr,
                    old_layout,
                    new_layout,
                ))
            };
        }

        // Can't actually shrink, so return the full range of the previous allocation.
        Ok(NonNull::slice_from_raw_parts(ptr, old_layout.size()))
    }

    fn alloc_chunk(&mut self, item_size: usize) -> Result<(), AllocError> {
        let chunk_size =
            DEFAULT_CHUNK_SIZE.max(align(item_size, CHUNK_ALIGNMENT) + CHUNK_ALIGNMENT);
        self.stats.reserved_bytes += chunk_size;

        let chunk = self.chunks.allocate_chunk(chunk_size)?;

        unsafe {
            (*chunk.as_ptr()).previous = Some(self.current_chunk);
        }

        self.current_chunk = chunk;

        self.stats.chunks += 1;

        Ok(())
    }

    pub fn get_stats(&mut self) -> Stats {
        self.stats.chunk_utilization =
            self.stats.chunks as f32 - 1.0 + ChunkHeader::utilization(self.current_chunk);
        self.stats
    }

    pub(crate) fn save(&self) -> BumpAllocatorSavePoint {
        BumpAllocatorSavePoint {
            chunk: self.current_chunk,
            cursor: unsafe { self.current_chunk.as_ref() }.cursor,
        }
    }

    pub(crate) unsafe fn restore(&mut self, saved: BumpAllocatorSavePoint) {
        unsafe {
            self.chunks.recycle_chunks(self.current_chunk, Some(saved.chunk));

            self.current_chunk = saved.chunk;
            self.current_chunk.as_mut().cursor = saved.cursor;
        }
    }
}

impl Drop for BumpAllocator {
    fn drop(&mut self) {
        assert!(self.live_alloc_count == 0);
        unsafe {
            self.chunks.recycle_chunks(self.current_chunk, None);
        }
    }
}

fn allocate_item_impl(
    this: NonNull<ChunkHeader>,
    layout: Layout,
) -> Result<NonNull<[u8]>, AllocError> {
    debug_assert!(CHUNK_ALIGNMENT % layout.align() == 0);
    debug_assert!(layout.align() > 0);
    debug_assert!(layout.align().is_power_of_two());

    let size = align(layout.size(), CHUNK_ALIGNMENT);

    unsafe {
        let cursor = (*this.as_ptr()).cursor;
        let end = (*this.as_ptr()).chunk_end;
        let available_size = end.offset_from(cursor);

        if size as isize > available_size {
            return Err(AllocError);
        }

        let next = cursor.add(size);

        (*this.as_ptr()).cursor = next;

        let cursor = NonNull::new(cursor).unwrap();
        let suballocation: NonNull<[u8]> = NonNull::slice_from_raw_parts(cursor, size);

        Ok(suballocation)
    }
}

unsafe fn deallocate_item_impl(this: NonNull<ChunkHeader>, item: NonNull<u8>, layout: Layout) {
    debug_assert!(ChunkHeader::contains_item(this, item));

    unsafe {
        let size = align(layout.size(), CHUNK_ALIGNMENT);
        let item_end = item.as_ptr().add(size);

        // If the item is the last allocation, then move the cursor back
        // to reuse its memory.
        if item_end == (*this.as_ptr()).cursor {
            (*this.as_ptr()).cursor = item.as_ptr();
        }

        // Otherwise, deallocation is a no-op
    }
}

unsafe fn grow_item_impl(
    this: NonNull<ChunkHeader>,
    item: NonNull<u8>,
    old_layout: Layout,
    new_layout: Layout,
) -> Result<NonNull<[u8]>, ()> {
    unsafe {
        debug_assert!(ChunkHeader::contains_item(this, item));

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
}

unsafe fn shrink_item_impl(
    this: NonNull<ChunkHeader>,
    item: NonNull<u8>,
    old_layout: Layout,
    new_layout: Layout,
) -> NonNull<[u8]> {
    unsafe {
        debug_assert!(ChunkHeader::contains_item(this, item));

        let old_size = align(old_layout.size(), CHUNK_ALIGNMENT);
        let new_size = align(new_layout.size(), CHUNK_ALIGNMENT);
        let old_item_end = item.as_ptr().add(old_size);

        // The item is the last allocation. We can attempt to just move
        // the cursor if the new size fits.

        if old_item_end == (*this.as_ptr()).cursor {
            let new_item_end = item.as_ptr().add(new_size);
            (*this.as_ptr()).cursor = new_item_end;
        }

        NonNull::slice_from_raw_parts(item, new_size)
    }
}

/// A Contiguous buffer of memory holding multiple sub-allocaions.
pub(crate) struct MultiThreadChunkHeader {
    pub previous: Option<AtomicPtr<MultiThreadChunkHeader>>,
    /// Offset of the next allocation.
    pub cursor: AtomicPtr<u8>,
    /// Points to the first byte after the chunk's header.
    pub chunk_end: *mut u8,
    /// Size of the chunk in bytes.
    pub size: usize,
}

/// A Contiguous buffer of memory holding multiple sub-allocaions.
pub(crate) struct ChunkHeader {
    pub previous: Option<NonNull<ChunkHeader>>,
    /// Offset of the next allocation.
    pub cursor: *mut u8,
    /// Points to the first byte after the chunk's header.
    pub chunk_end: *mut u8,
    /// Size of the chunk in bytes.
    pub size: usize,
}

pub(crate) unsafe fn poison_chunk(this: NonNull<ChunkHeader>) {
    unsafe {
        let start: *mut u32 = this.as_ptr().add(1).cast();
        let end: *const u32 = this.as_ref().chunk_end.cast();
        let len = end.offset_from(start) as usize;
        let slice = std::slice::from_raw_parts_mut(start, len);
        slice.fill(0xDEADBEEF);
    }
}

impl ChunkHeader {
    pub fn contains_item(this: NonNull<ChunkHeader>, item: NonNull<u8>) -> bool {
        unsafe {
            let start: *mut u8 = this.cast::<u8>().as_ptr().add(CHUNK_ALIGNMENT);
            let end: *mut u8 = (*this.as_ptr()).chunk_end;
            let item = item.as_ptr();

            start <= item && item < end
        }
    }

    fn available_size(this: NonNull<ChunkHeader>) -> usize {
        unsafe {
            let this = this.as_ref();
            this.chunk_end.offset_from(this.cursor) as usize
        }
    }

    fn utilization(this: NonNull<ChunkHeader>) -> f32 {
        let size = unsafe { this.as_ref().size } as f32;
        (size - ChunkHeader::available_size(this) as f32) / size
    }
}

fn align(val: usize, alignment: usize) -> usize {
    let rem = val % alignment;
    if rem == 0 {
        return val;
    }

    val.checked_add(alignment).unwrap() - rem
}

#[derive(Copy, Clone, Debug, Default)]
pub struct Stats {
    pub chunks: u32,
    pub chunk_utilization: f32,
    pub allocations: u32,
    pub deallocations: u32,
    pub reallocations: u32,
    pub in_place_reallocations: u32,
    pub reallocated_bytes: usize,
    pub allocated_bytes: usize,
    pub reserved_bytes: usize,
}

#[derive(Clone)]
pub struct BumpAllocatorSavePoint {
    chunk: NonNull<ChunkHeader>,
    cursor: *mut u8,
}
