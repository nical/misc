use std::marker::PhantomData;
use std::ptr;

use crate::allocator::{Allocator, AllocError};
use crate::unmanaged::{AllocInit, UnmanagedHeaderVector};

type Chunk<T> = UnmanagedHeaderVector<Header<T>, T>;

struct Header<T> {
    next: Chunk<T>,
}

pub struct UnmanagedSegmentedVector<T> {
    current: Chunk<T>,
    first: Chunk<T>,
    last_header: *mut Header<T>,
    len: usize,
}

impl<T> UnmanagedSegmentedVector<T> {
    pub fn new() -> Self {
        unsafe {
            UnmanagedSegmentedVector {
                current: Chunk::dangling(),
                first: Chunk::dangling(),
                last_header: ptr::null_mut(),
                len: 0,
            }
        }
    }

    pub fn try_with_capacity_in<A: Allocator>(cap: usize, allocator: &A) -> Result<Self, AllocError> {
        let chunk = Chunk::try_with_capacity_in(
            Header { next: unsafe { Chunk::dangling() } },
            cap,
            AllocInit::Uninit,
            &allocator
        )?;

        Ok(UnmanagedSegmentedVector {
            current: chunk,
            first: unsafe { Chunk::dangling() },
            last_header: ptr::null_mut(),
            len: 0,
        })
    }

    pub fn with_capacity_in<A: Allocator>(cap: usize, allocator: &A) -> Self {
        Self::try_with_capacity_in(cap, allocator).unwrap()
    }

    pub unsafe fn deallocate_in<A: Allocator>(&mut self, allocator: &A) {
        if self.last_header.is_null() {
            // TODO: Ugly workaround for using dangling vectors with headers
            // when they are empty.
            return;
        }

        let mut iter = self.first;
        loop {
            if iter.is_empty() {
                break;
            }
            let next = iter.header().next;
            unsafe {
                iter.deallocate_in(allocator);
            }
            iter = next;
        }

        self.current.deallocate_in(allocator);

        self.first = Chunk::dangling();
        self.last_header = ptr::null_mut();
        self.len = 0;
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub unsafe fn push<A: Allocator>(&mut self, val: T, allocator: &A) {
        if self.current.remaining_capacity() == 0 {
            self.add_chunk(allocator);
        }

        let _r = self.current.push_within_capacity(val);
        debug_assert!(_r.is_ok());
        self.len += 1;
    }

    unsafe fn add_chunk<A: Allocator>(&mut self, allocator: &A) {
        let cap = self.current.capacity() * 2;

        if self.last_header.is_null() {
            self.first = self.current;
        } else {
            (*self.last_header).next = self.current;
        }
        self.last_header = self.current.header_ptr().as_ptr();

        self.current = Chunk::<T>::with_capacity_in(
            Header { next: Chunk::dangling() }, 
            cap,
            AllocInit::Uninit,
            allocator,
        );
    }

    pub fn chunks(&self) -> Chunks<T> {
        Chunks {
            current: self.first,
            last: self.current,
            total_len: self.len,
            _lifetime: PhantomData,
        }
    }
}

pub struct Chunks<'a, T> {
    current: Chunk<T>,
    last: Chunk<T>,
    total_len: usize,
    _lifetime: PhantomData<&'a ()>,
}

impl<'a, T:'a> Iterator for Chunks<'a, T> {
    type Item = &'a[T];
    fn next(&mut self) -> Option<&'a[T]> {
        if self.current.is_empty() {
            if self.last.is_empty() {
                return None;
            }
            let chunk = self.last;
            let data = chunk.as_slice().as_ptr();
            let len = chunk.len();
            self.last = unsafe { Chunk::dangling() };
            return Some(unsafe { std::slice::from_raw_parts(data, len) });
        }

        let chunk = self.current;
        self.current = self.current.header().next;

        let data = chunk.as_slice().as_ptr();
        let len = chunk.len();
        return Some(unsafe { std::slice::from_raw_parts(data, len) });
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.total_len, Some(self.total_len))
    }
}


#[test]
fn seg_simple() {
    use crate::global::Global;

    let allocator = Global;
    let mut v = UnmanagedSegmentedVector::with_capacity_in(16, &allocator);
    for i in 0i32..200 {
        unsafe { v.push(i, &allocator) }
    }

    let mut idx = 0;
    for chunk in v.chunks() {
        for item in chunk {
            assert_eq!(*item, idx);
            idx += 1;
        }
    }

    assert_eq!(v.len(), 200);

    unsafe { v.deallocate_in(&allocator); }
}

#[test]
fn seg_empty() {
    use crate::global::Global;
    let allocator = Global;

    let mut empty: UnmanagedSegmentedVector<i32> = UnmanagedSegmentedVector::new();
    unsafe { empty.deallocate_in(&allocator); }
}
