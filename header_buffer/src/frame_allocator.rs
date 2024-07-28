use std::{
    alloc::Layout,
    ptr::{self, NonNull},
};

use crate::allocator::{AllocError, Allocator};

const CHUNK_ALIGNMENT: usize = 32;
const DEFAULT_CHUNK_SIZE: usize = 4 * 1024 * 1024;

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

    pub fn allocate_chunk(
        size: usize,
        previous: Option<NonNull<Chunk>>,
        allocator: &dyn Allocator,
    ) -> Result<NonNull<Self>, AllocError> {
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
            ptr::write(
                chunk.as_ptr(),
                Chunk {
                    previous,
                    chunk_end,
                    cursor,
                    size: size as u32,
                },
            );
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

            let slice = std::slice::from_raw_parts_mut(cursor, size);
            let suballocation: NonNull<[u8]> = NonNull::new_unchecked(slice);

            Ok(suballocation)
        }
    }

    pub unsafe fn deallocate_item(this: NonNull<Chunk>, item: NonNull<u8>, layout: Layout) {
        debug_assert!(Self::contains_item(this, item));

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

    pub fn contains_item(this: NonNull<Chunk>, item: NonNull<u8>) -> bool {
        unsafe {
            let start: *mut u8 = this.cast::<u8>().as_ptr().add(CHUNK_ALIGNMENT);
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

pub struct BumpAllocator<A: Allocator> {
    current_chunk: NonNull<Chunk>,
    chunk_size: usize,
    allocation_count: i32,
    parent_allocator: A,
}

impl<A: Allocator> BumpAllocator<A> {
    pub fn new_in(parent_allocator: A) -> Self {
        Self::with_chunk_size_in(DEFAULT_CHUNK_SIZE, parent_allocator)
    }

    pub fn with_chunk_size_in(chunk_size: usize, parent_allocator: A) -> Self {
        BumpAllocator {
            current_chunk: Chunk::allocate_chunk(
                chunk_size,
                None,
                &parent_allocator
            ).unwrap(),
            chunk_size,
            parent_allocator,
            allocation_count: 0,
        }
    }

    pub fn allocate_item(&mut self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if let Ok(alloc) = Chunk::allocate_item(self.current_chunk, layout) {
            self.allocation_count += 1;
            return Ok(alloc);
        }

        self.alloc_chunk(layout.size())?;

        match Chunk::allocate_item(self.current_chunk, layout) {
            Ok(alloc) => {
                self.allocation_count += 1;
                return Ok(alloc);
            }
            Err(_) => {
                return Err(AllocError);
            }
        }
    }

    pub fn deallocate_item(&mut self, ptr: NonNull<u8>, layout: Layout) {
        if Chunk::contains_item(self.current_chunk, ptr) {
            unsafe { Chunk::deallocate_item(self.current_chunk, ptr, layout); }
        }

        self.allocation_count -= 1;
        debug_assert!(self.allocation_count >= 0);
    }

    fn alloc_chunk(&mut self, item_size: usize) -> Result<(), AllocError> {
        let chunk_size = self.chunk_size.max(item_size + CHUNK_ALIGNMENT);
        let chunk = Chunk::allocate_chunk(
            chunk_size,
            None,
            &self.parent_allocator
        )?;

        unsafe {
            (*chunk.as_ptr()).previous = Some(self.current_chunk);
        }
        self.current_chunk = chunk;

        Ok(())
    }
}

impl<A: Allocator> Drop for BumpAllocator<A> {
    fn drop(&mut self) {
        assert!(self.allocation_count == 0);
        let mut iter = Some(self.current_chunk);
        while let Some(chunk) = iter {
            iter = unsafe { (*chunk.as_ptr()).previous };
            Chunk::deallocate_chunk(chunk, &self.parent_allocator)
        }
    }
}

#[derive(Copy, Clone)]
pub struct BumpAllocatorRef<A: Allocator>(NonNull<BumpAllocator<A>>);

impl<A: Allocator> BumpAllocatorRef<A> {
    pub unsafe fn new(allocator: &mut BumpAllocator<A>) -> Self {
        BumpAllocatorRef(NonNull::new_unchecked(allocator as *mut _))
    }
}

unsafe impl<A: Allocator> Allocator for BumpAllocatorRef<A> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        unsafe {
            (*self.0.as_ptr()).allocate_item(layout)
        }
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        unsafe {
            (*self.0.as_ptr()).deallocate_item(ptr, layout);
        }
    }

    // TODO: grow/shrink
}

#[test]
fn bump_simple() {
    use crate::global::Global;
    use crate::vector;
    let mut bump = BumpAllocator::with_chunk_size_in(1024, Global);
    let bumpref = unsafe { BumpAllocatorRef::new(&mut bump) };

    let v1 = vector![[0i32; 4] in bumpref];
    let v2 = vector![[1i32; 64] in bumpref];
    let v3 = vector![[2i32; 1024] in bumpref];

    std::mem::drop(v1);
    std::mem::drop(v2);
    std::mem::drop(v3);
}
