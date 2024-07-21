use crate::util;
use crate::util::nnptr;
use crate::allocator::*;
use std::marker::PhantomData;
use std::ptr::NonNull;

pub struct UnmanagedHeaderBuffer<H, T> {
    data: NonNull<T>,
    _marker: PhantomData<H>,
    cap: usize,
}

impl<T> UnmanagedHeaderBuffer<(), T> {
    /// Creates an empty buffer without doing any memory allocation.
    pub fn new() -> Self {
        unsafe { Self::dangling() }
    }
}

impl<H, T> UnmanagedHeaderBuffer<H, T> {
    /// Creates an empty pre-allocated vector with a given storage capacity.
    pub fn try_allocate_in<A: Allocator>(header: H, cap: usize, allocator: &A) -> Result<Self, AllocError> {
        let (data, cap) = util::allocate_in::<H, T, A>(cap, allocator)?;
        unsafe {
            let header_ptr = util::get_header::<H, T>(data);
            nnptr::write(header_ptr, header);
        }

        Ok(UnmanagedHeaderBuffer {
            data,
            _marker: PhantomData,
            cap,
        })
    }

    #[inline]
    pub unsafe fn dangling() -> Self {
        UnmanagedHeaderBuffer {
            data: NonNull::dangling(),
            _marker: PhantomData,
            cap: 0,
        }
    }

    /// Drops the header and deallocates this raw buffer, leaving it in its unallocated state.
    ///
    /// The elements in the array potion of the buffer are expected to be dropped by the caller.
    /// It is safe (no-op) to call `deallocate` on a vector that is already in its unallocated state.
    ///
    /// # Safety
    ///
    /// The provided allocator must be the one this raw buffer was created with.
    pub unsafe fn deallocate_in<A: Allocator>(&mut self, allocator: &A) {
        unsafe {
            util::drop_header::<H, T>(self.data);
            util::deallocate_in::<H, T, A>(self.data, self.cap, allocator);
        }

        self.data = NonNull::dangling();
    }

    pub unsafe fn try_reallocate_in<A: Allocator>(&mut self, new_cap: usize, allocator: &A) -> Result<(), AllocError> {
        let old_cap = self.cap;
        let (new_ptr, new_cap) = util::reallocate_in::<(), T, A>(self.data, old_cap, new_cap, allocator)?;
        self.data = new_ptr;
        self.cap = new_cap;

        Ok(())
    }

    #[inline]
    pub fn header(&self) -> &H {
        unsafe {
            util::get_header::<H, T>(self.data).as_ref()
        }
    }

    #[inline]
    pub fn header_mut(&mut self) -> &mut H {
        unsafe {
            util::get_header::<H, T>(self.data).as_mut()
        }
    }

    #[inline]
    pub fn header_ptr(&self) -> NonNull<H> {
        unsafe {
            util::get_header::<H, T>(self.data)
        }
    }

    #[inline]
    pub fn items_ptr(&self) -> NonNull<T> {
        self.data
    }

    #[inline]
    /// Returns the total number of elements the vector can hold without reallocating.
    pub fn capacity(&self) -> usize {
        self.cap
    }

    #[inline]
    pub unsafe fn as_slice(&self, len: usize) -> &[T] {
        core::slice::from_raw_parts(self.data.as_ptr(), len)
    }

    #[inline]
    pub unsafe fn as_mut_slice(&mut self, len: usize) -> &mut [T] {
        core::slice::from_raw_parts_mut(self.data.as_ptr(), len)
    }

    #[inline]
    pub unsafe fn write_item(&mut self, index: usize, val: T) {
        debug_assert!(index < self.cap);
        let dst = nnptr::add(self.data, index);
        nnptr::write(dst, val);
    }

    #[inline]
    pub unsafe fn read_item(&mut self, index: usize) -> T {
        debug_assert!(index < self.cap);
        let dst = nnptr::add(self.data, index);
        nnptr::read(dst)
    }
}
