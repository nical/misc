use core::alloc::Layout;
use core::mem;
use core::ptr::NonNull;

use crate::allocator::*;

#[track_caller]
#[inline(always)]
#[cfg(debug_assertions)]
pub unsafe fn assume(v: bool) {
    if !v {
        core::unreachable!()
    }
}

#[track_caller]
#[inline(always)]
#[cfg(not(debug_assertions))]
pub unsafe fn assume(v: bool) {
    if !v {
        unsafe {
            core::hint::unreachable_unchecked();
        }
    }
}

#[inline(always)]
pub fn invalid_mut<T>(addr: usize) -> *mut T {
    #[allow(clippy::useless_transmute, clippy::transmutes_expressible_as_ptr_casts)]
    unsafe {
        core::mem::transmute(addr)
    }
}

pub(crate) const fn is_zst<T>() -> bool {
    mem::size_of::<T>() == 0
}

pub(crate) const fn header_size<Header, T>() -> usize {
    const fn max(a: usize, b: usize) -> usize {
        if a > b {
            a
        } else {
            b
        }
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

pub fn header_vector_layout<Header, T>(n: usize) -> Result<Layout, AllocError> {
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

pub unsafe fn get_header<H, T>(ptr: NonNull<T>) -> NonNull<H> {
    let header_size = header_size::<H, T>();
    unsafe { ptr.byte_sub(header_size).cast::<H>() }
}

//#[cold]
//pub fn alloc_error_cold() -> AllocError {
//    AllocError
//}

pub fn grow_amortized(len: usize, additional: usize) -> Result<usize, AllocError> {
    let required = len.saturating_add(additional);
    let cap = len.saturating_add(len).max(required).max(8);

    const MAX: usize = isize::MAX as usize;

    if cap > MAX {
        if required <= MAX {
            return Ok(required);
        }

        return Err(AllocError);
    }

    Ok(cap)
}
