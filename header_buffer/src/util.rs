use core::alloc::Layout;
use core::mem;
use core::ptr::{self, NonNull};

use crate::allocator::*;

pub type BufferSize = u32;

pub(crate) const fn header_size<Header, T>() -> usize {
    const fn max(a: usize, b: usize) -> usize {
        if a > b { a } else { b }
    }
    let s = mem::size_of::<Header>();
    if s == 0 {
        return 0;
    }

    let a = mem::align_of::<T>();
    let mut size = max(a, s);

    // Favor L1 cache line alignment for large structs.
    if mem::size_of::<T>() >= 64 {
        size = max(size, 64);
    }

    if size % a != 0 {
        size += a - (size % a);
    }

    size
}

pub fn header_buffer_layout<Header, T>(n: usize) -> Result<Layout, AllocError> {
    let size = mem::size_of::<T>().checked_mul(n).ok_or(AllocError)?;
    let align = mem::align_of::<Header>().max(mem::align_of::<T>());
    let align = if mem::size_of::<T>() < 64 {
        align
    } else {
        align.max(64)
    };
    let header_size = header_size::<Header, T>();

    Layout::from_size_align(header_size + size, align).map_err(|_| AllocError)
}

#[inline(never)]
pub fn allocate_in<H, T, A>(
    mut cap: usize,
    allocator: &A,
) -> Result<(NonNull<T>, usize), AllocError>
where
    A: Allocator,
{
    if cap > BufferSize::MAX as usize {
        return Err(alloc_error_cold());
    }

    let layout = header_buffer_layout::<H, T>(cap)?;
    let header_size = header_size::<H, T>();
    let t_size = mem::size_of::<T>();

    if (t_size == 0 || cap == 0) && header_size == 0 {
        return Ok((NonNull::dangling(), cap));
    }

    if cap == 0 {
        cap = 16;
    }

    let allocation = allocator.allocate(layout)?;

    let items_size = allocation.len() - header_size;
    let real_capacity = if t_size == 0 {
        cap
    } else {
        items_size / t_size
    };

    let items_ptr = unsafe {
        nnptr::add(allocation.cast::<u8>(), header_size).cast::<T>()
    };

    Ok((items_ptr, real_capacity))
}

#[inline(never)]
pub unsafe fn deallocate_in<H, T, A: Allocator>(ptr: NonNull<T>, cap: usize, allocator: &A) {
    let header_size = header_size::<H, T>();
    let t_size = mem::size_of::<T>();
    if (t_size == 0 || cap == 0) && header_size == 0 {
        return;
    }

    let allocation = nnptr::sub(ptr.cast::<u8>(), header_size);

    let layout = header_buffer_layout::<H, T>(cap).unwrap();
    allocator.deallocate(allocation.cast::<u8>(), layout);
}

#[cold]
pub unsafe fn reallocate_in<H, T, A: Allocator>(
    old_ptr: NonNull<T>,
    old_cap: usize,
    new_cap: usize,
    allocator: &A,
) -> Result<(NonNull<T>, usize), AllocError> {

    let header_size = header_size::<H, T>();
    let t_size = mem::size_of::<T>();

    let old_dangling = (t_size == 0 || old_cap == 0) && header_size == 0;
    let new_dangling = (t_size == 0 || new_cap == 0) && header_size == 0;

    if new_dangling {
        if !old_dangling {
            deallocate_in::<H, T, A>(old_ptr, old_cap, allocator);
        }
        return Ok((NonNull::dangling(), new_cap));
    }

    if old_dangling {
        return allocate_in::<H, T, A>(new_cap, allocator);
    }

    unsafe {
        let new_layout = header_buffer_layout::<H, T>(new_cap).unwrap();

        let new_alloc = if old_cap == 0 {
            allocator.allocate(new_layout)?
        } else {
            let old_layout = header_buffer_layout::<(), T>(old_cap).unwrap();
            let old_alloc = get_header::<H, T>(old_ptr).cast::<u8>();

            if new_layout.size() >= old_layout.size() {
                allocator.grow(old_alloc, old_layout, new_layout)
            } else {
                allocator.shrink(old_alloc, old_layout, new_layout)
            }?
        };

        let items_ptr = nnptr::add(new_alloc.cast::<u8>(), header_size).cast::<T>();

        Ok((items_ptr, new_cap))
    }
}

pub unsafe fn get_header<H, T>(ptr: NonNull<T>) -> NonNull<H> {
    let header_size = header_size::<H, T>();
    unsafe { nnptr::byte_sub(ptr, header_size).cast::<H>() }
}

#[cold]
pub fn alloc_error_cold() -> AllocError {
    AllocError
}

pub unsafe fn remove<T>(data: NonNull<T>, len: usize, index: usize) -> T {
    #[cold]
    #[inline(never)]
    #[track_caller]
    fn assert_failed(index: usize, len: usize) -> ! {
        panic!("removal index (is {index}) should be < len (is {len})");
    }

    if index >= len {
        assert_failed(index, len);
    }

    // infallible
    let ret;
    {
        // the place we are taking from.
        let ptr = nnptr::add(data, index);
        // copy it out, unsafely having a copy of the value on
        // the stack and in the vector at the same time.
        ret = nnptr::read(ptr);

        // Shift everything down to fill in that spot.
        nnptr::copy(nnptr::add(ptr, 1), ptr, len - index - 1);
    }

    ret
}

#[inline]
pub unsafe fn swap_remove<T>(data: NonNull<T>, len: usize, idx: usize) -> T {
    assert!(idx < len);

    let ptr = nnptr::add(data, idx);
    let item = nnptr::read(ptr);

    let last_idx = len - 1;
    if idx != last_idx {
        let last_ptr = nnptr::add(data, last_idx);
        nnptr::write(ptr, nnptr::read(last_ptr));
    }

    item
}

pub unsafe fn extend_from_slice_assuming_capacity<T: Clone>(data: NonNull<T>, inital_len: usize, slice: &[T])
where
    T: Clone,
{
    let mut ptr = nnptr::add(data, inital_len as usize);

    for item in slice {
        nnptr::write(ptr, item.clone());
        ptr = nnptr::add(ptr, 1)
    }
}

pub unsafe fn drop_header<H, T>(ptr: NonNull<T>) {
    let header = get_header::<H, T>(ptr);
    ptr::drop_in_place(header.as_ptr());
}

pub unsafe fn drop_items<T>(ptr: NonNull<T>, count: usize) {
    let mut ptr = ptr.as_ptr();
    for _ in 0..count {
        ptr::drop_in_place(ptr);
        ptr = ptr.add(1);
    }
}

pub fn grow_amortized(len: usize, additional: usize) -> Result<usize, AllocError> {
    let required = len.saturating_add(additional);
    let cap = len.saturating_add(len).max(required).max(8);

    const MAX: usize = BufferSize::MAX as usize;

    if cap > MAX {
        if required <= MAX {
            return Ok(required);
        }

        return Err(AllocError)
    }

    Ok(cap)
}

// Waiting for `non_null_convenience` to be stabilized.
pub mod nnptr {
    use std::ptr::{self, NonNull};

    #[inline(always)]
    pub unsafe fn read<T>(src: NonNull<T>) -> T {
        ptr::read(src.as_ptr())
    }

    #[inline(always)]
    pub unsafe fn write<T>(dst: NonNull<T>, val: T) {
        ptr::write(dst.as_ptr(), val)
    }

    #[inline(always)]
    pub unsafe fn copy<T>(src: NonNull<T>, dst: NonNull<T>, count: usize) {
        ptr::copy(src.as_ptr(), dst.as_ptr(), count)
    }

    #[inline(always)]
    pub unsafe fn add<T>(p: NonNull<T>, count: usize) -> NonNull<T> {
        NonNull::new_unchecked(p.as_ptr().add(count))
    }

    #[inline(always)]
    pub unsafe fn sub<T>(p: NonNull<T>, count: usize) -> NonNull<T> {
        let offset = p.as_ptr().sub(count);
        NonNull::new_unchecked(offset)
    }

    #[inline(always)]
    pub unsafe fn byte_add<T>(p: NonNull<T>, count: usize) -> NonNull<T> {
        let u8_ptr = p.as_ptr() as *mut u8;
        let offset_u8_ptr = u8_ptr.add(count) as *mut T;
        NonNull::new_unchecked(offset_u8_ptr)
    }

    #[inline(always)]
    pub unsafe fn byte_sub<T>(p: NonNull<T>, count: usize) -> NonNull<T> {
        let u8_ptr = p.as_ptr() as *mut u8;
        let offset_u8_ptr = u8_ptr.sub(count) as *mut T;
        NonNull::new_unchecked(offset_u8_ptr)
    }
}
