
extern crate alloc as alloc_crate;

//pub mod alloc {
//    pub use allocator_api2::alloc::{AllocError, Allocator, Global};
//}

mod util;
pub mod allocator;
pub mod global;
pub mod vec;
pub mod header_vec;
pub mod buffer;
pub mod seg_vec;
pub mod frame_allocator;


#[track_caller]
#[inline(always)]
#[cfg(debug_assertions)]
unsafe fn assume(v: bool) {
    if !v {
        core::unreachable!()
    }
}

#[track_caller]
#[inline(always)]
#[cfg(not(debug_assertions))]
unsafe fn assume(v: bool) {
    if !v {
        unsafe {
            core::hint::unreachable_unchecked();
        }
    }
}

#[inline(always)]
fn addr<T>(x: *const T) -> usize {
    #[allow(clippy::useless_transmute, clippy::transmutes_expressible_as_ptr_casts)]
    unsafe {
        core::mem::transmute(x)
    }
}

#[inline(always)]
fn invalid_mut<T>(addr: usize) -> *mut T {
    #[allow(clippy::useless_transmute, clippy::transmutes_expressible_as_ptr_casts)]
    unsafe {
        core::mem::transmute(addr)
    }
}
