use std::{alloc::Layout, cell::RefCell, ptr::{self, NonNull}};

use crate::allocator::{AllocError, Allocator};

const CHUNK_ALIGNMENT: usize = 32;

pub struct Chunk {
    previous: Option<NonNull<Chunk>>,
    chunk_end: *mut u8,
    cursor: *mut u8,
    size: u32,
}

impl Chunk {
    pub fn previous(this: NonNull<Chunk>) -> Option<NonNull<Chunk>> {
        unsafe { (*this.as_ptr()).previous }
    }

    pub fn allocate_chunk(size: usize, previous: Option<NonNull<Chunk>>, allocator: &dyn Allocator) -> Result<NonNull<Self>, AllocError> {
        let layout = match Layout::from_size_align(size, CHUNK_ALIGNMENT) {
            Ok(layout) => layout,
            Err(_) => {
                return Err(AllocError);
            }
        };

        let alloc = allocator.allocate(layout)?;
        let chunk: NonNull<Chunk> = alloc.cast();
        let chunk_start: *mut u8 = alloc.cast().as_ptr();

        unsafe {
            let chunk_end = chunk_start.add(size);
            let cursor = chunk_start.add(CHUNK_ALIGNMENT);
            ptr::write(chunk.as_ptr(), Chunk {
                previous,
                chunk_end,
                cursor,
                size: size as u32,
            });
        }

        Ok(chunk)
    }

    pub fn deallocate_chunk(this: NonNull<Chunk>, allocator: &dyn Allocator) {
        let size = unsafe { (*this.as_ptr()).size } as usize;
        let layout = Layout::from_size_align(size, CHUNK_ALIGNMENT).unwrap();

        unsafe {
            allocator.deallocate(this.cast(), layout);
        }
    }

    pub fn allocate_item(this: NonNull<Chunk>, size: usize) -> Result<NonNull<[u8]>, ()> {
        let size = align(size, CHUNK_ALIGNMENT);

        unsafe {
            let cursor = (*this.as_ptr()).cursor;
            let end = (*this.as_ptr()).chunk_end;
            let next = cursor.add(size);

            if next > end {
                return Err(());
            }

            (*this.as_ptr()).cursor = next;

            let slice = std::slice::from_raw_parts_mut(cursor, size);
            let suballocation: NonNull<[u8]> = NonNull::new_unchecked(slice);

            Ok(suballocation)
        }
    }

    pub unsafe fn deallocate_item(this: NonNull<Chunk>, item: NonNull<u8>, size: usize) {
        debug_assert!(Self::contains_item(this, item));

        unsafe {
            let item_end = item.as_ptr().add(size);

            // If the item is the last allocation, then move the cursor back
            // to reuse its memory.
            if item_end == (*this.as_ptr()).cursor {
                (*this.as_ptr()).cursor = item.as_ptr();
            }
        }
    }

    pub fn contains_item(this: NonNull<Chunk>, item: NonNull<u8>) -> bool {
        unsafe {
            let start:*mut u8 = this.cast::<u8>().as_ptr().add(CHUNK_ALIGNMENT);
            let end: *mut u8 = (*this.as_ptr()).chunk_end;
            let item = item.as_ptr();

            start <= item && item < end
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

pub struct FrameAllocator<A> {
    current_chunk: Option<NonNull<u8>>,
    parent_allocator: A,
    allocation_count: RefCell<usize>,
}

impl<A: Allocator> FrameAllocator<A> {
    pub fn new_in(parent_allocator: A) -> Self {
        FrameAllocator {
            current_chunk: None,
            parent_allocator,
            allocation_count: RefCell::new(0),
        }
    }

    pub fn allocate_item(&self) -> Result<NonNull<u8>, AllocError> {
        todo!()
    }
}
