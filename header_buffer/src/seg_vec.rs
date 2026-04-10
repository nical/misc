use std::marker::PhantomData;
use std::ptr::NonNull;

use crate::allocator::{Allocator, AllocError};
use crate::unmanaged::{AllocInit, UnmanagedHeaderVector};

// Basic push micro-benchmark against a resizable UnmnanagedHeaderVector:
//
// |           |     100 |   1000 |  10_000 |
// | --------- | ------- | ------ | ------- |
// | resizable | 337.8ns | 2.72us | 24.82us |
// | segmented | 330.0ns | 2.55us | 23.50us |
//
// This benchmark runs in ideal conditions for the default allocator (no threading/contention,
// no fragmentation) and everything is rather cache-friendly. Copying the data during the
// reallocation is taking at most 5% of the time for the resizable vector, which is unlike
// what I typically see in real workloads.
//
// In real world workloads the difference is likely to be bigger in cases where data copies
// are observed to be expensive.
// Note that since this container keeps multiple allocations, dropping the container potentially
// causes many deallocations which can potentially cause significant contention with some
// allocators.


type Chunk<T> = UnmanagedHeaderVector<Header<T>, T>;

struct Header<T> {
    next: NonNull<T>,
    prev: NonNull<T>,
    len: usize,
    cap: usize,
}

impl<T> Header<T> {
    fn dangling() -> Self {
        Header {
            next: NonNull::dangling(),
            prev: NonNull::dangling(),
            len: 0,
            cap: 0,
        }
    }

    unsafe fn chunk(&self, data: NonNull<T>) -> Chunk<T> {
        Chunk::from_raw_parts(data, self.len, self.cap)
    }
}

pub struct UnmanagedSegmentedVector<T> {
    current: Chunk<T>,
    first: NonNull<T>,
    len: usize,
}

impl<T> UnmanagedSegmentedVector<T> {
    pub fn new() -> Self {
        unsafe {
            UnmanagedSegmentedVector {
                current: Chunk::dangling(),
                first: NonNull::dangling(),
                len: 0,
            }
        }
    }

    pub fn try_with_capacity_in<A: Allocator>(cap: usize, allocator: &A) -> Result<Self, AllocError> {
        let chunk = Chunk::try_with_capacity_in(
            Header::dangling(),
            cap,
            AllocInit::Uninit,
            &allocator
        )?;

        Ok(UnmanagedSegmentedVector {
            current: chunk,
            first: NonNull::dangling(),
            len: 0,
        })
    }

    pub fn with_capacity_in<A: Allocator>(cap: usize, allocator: &A) -> Self {
        Self::try_with_capacity_in(cap, allocator).unwrap()
    }

    pub unsafe fn deallocate_in<A: Allocator>(&mut self, allocator: &A) {
        // Walk the linked list of retired chunks.
        if self.first != NonNull::dangling() {
            let mut data = self.first;
            loop {
                let header = &*Self::header_from_data(data);
                let next = header.next;
                let mut chunk = header.chunk(data);
                chunk.deallocate_in(allocator);
                if next == NonNull::dangling() {
                    break;
                }
                data = next;
            }
        }

        // Deallocate the current chunk if it's a real allocation.
        if self.current.capacity() > 0 {
            self.current.deallocate_in(allocator);
        }

        self.current = Chunk::dangling();
        self.first = NonNull::dangling();
        self.len = 0;
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub unsafe fn push<A: Allocator>(&mut self, val: T, allocator: &A) {
        if self.current.is_full() {
            self.add_chunk(allocator);
        }

        debug_assert!(!self.current.is_full());
        self.current.push_assuming_capacity(val);
        self.len += 1;
    }

    #[inline]
    unsafe fn header_from_data(data: NonNull<T>) -> *mut Header<T> {
        crate::util::get_header::<Header<T>, T>(data).as_ptr()
    }

    // Marking this inline(never) regresses the push benchmark in all cases.
    // Marking this cold is a significant regression (20%) when pushing 100 elements
    // and a small improvement (5.3%) when pushing 10_000 elements.
    unsafe fn add_chunk<A: Allocator>(&mut self, allocator: &A) {
        let cap = self.current.capacity().max(crate::MIN_CAPACITY) * 2;

        // Only link the current chunk into the retired list if it's a
        // real allocation (not dangling). A dangling chunk has no valid
        // header to read or link through.
        let prev = if self.current.capacity() > 0 {
            let (data, len, item_cap) = self.current.into_raw_parts();
            let header = &mut *self.current.header_ptr().as_ptr();
            header.len = len;
            header.cap = item_cap;
            header.next = NonNull::dangling();

            if self.first == NonNull::dangling() {
                // First retired chunk.
                self.first = data;
            } else {
                // Link the previous tail's next to the current chunk.
                (*Self::header_from_data(header.prev)).next = data;
            }
            data
        } else {
            NonNull::dangling()
        };

        self.current = Chunk::<T>::with_capacity_in(
            Header { next: NonNull::dangling(), prev, len: 0, cap: 0 },
            cap,
            AllocInit::Uninit,
            allocator,
        );
    }

    pub fn chunks(&self) -> Chunks<T> {
        Chunks {
            current: self.first,
            last: self.current,
            _lifetime: PhantomData,
        }
    }
}

pub struct Chunks<'a, T> {
    current: NonNull<T>,
    last: Chunk<T>,
    _lifetime: PhantomData<&'a ()>,
}

impl<'a, T:'a> Iterator for Chunks<'a, T> {
    type Item = &'a[T];
    fn next(&mut self) -> Option<&'a[T]> {
        if self.current == NonNull::dangling() {
            if self.last.is_empty() {
                return None;
            }
            let chunk = self.last;
            let data = chunk.as_slice().as_ptr();
            let len = chunk.len();
            self.last = unsafe { Chunk::dangling() };
            return Some(unsafe { std::slice::from_raw_parts(data, len) });
        }

        let data = self.current;
        let header = unsafe { &*UnmanagedSegmentedVector::header_from_data(data) };
        self.current = header.next;

        return Some(unsafe { std::slice::from_raw_parts(data.as_ptr(), header.len) });
    }
}


#[test]
fn seg_simple() {
    use crate::allocator::Global;

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
    use crate::allocator::Global;
    let allocator = Global;

    let mut empty: UnmanagedSegmentedVector<i32> = UnmanagedSegmentedVector::new();
    unsafe { empty.deallocate_in(&allocator); }
}

#[test]
fn push_from_empty() {
    let allocator = crate::allocator::Global;
    let mut v: UnmanagedSegmentedVector<u32> = UnmanagedSegmentedVector::new();
    for i in 0..200 {
        unsafe { v.push(i, &allocator); }
    }

    let mut idx = 0;
    for chunk in v.chunks() {
        for &item in chunk {
            assert_eq!(item, idx);
            idx += 1;
        }
    }
    assert_eq!(v.len(), 200);

    unsafe { v.deallocate_in(&allocator); }
}

#[test]
fn dealloc_single_chunk() {
    let allocator = crate::allocator::Global;
    let mut v: UnmanagedSegmentedVector<u32> = UnmanagedSegmentedVector::with_capacity_in(16, &allocator);
    // Push fewer items than capacity, or none at all
    unsafe { v.deallocate_in(&allocator); }
}

#[test]
fn iterate_empty_no_capacity() {
    let v: UnmanagedSegmentedVector<i32> = UnmanagedSegmentedVector::new();
    assert_eq!(v.len(), 0);
    assert_eq!(v.chunks().count(), 0);
}

#[test]
fn iterate_empty_with_capacity() {
    use crate::allocator::Global;
    let allocator = Global;
    let mut v: UnmanagedSegmentedVector<i32> = UnmanagedSegmentedVector::with_capacity_in(16, &allocator);
    assert_eq!(v.len(), 0);
    assert_eq!(v.chunks().count(), 0);
    unsafe { v.deallocate_in(&allocator); }
}

#[test]
fn single_element() {
    use crate::allocator::Global;
    let allocator = Global;
    let mut v: UnmanagedSegmentedVector<u32> = UnmanagedSegmentedVector::with_capacity_in(16, &allocator);
    unsafe { v.push(42, &allocator); }

    assert_eq!(v.len(), 1);
    let items: Vec<u32> = v.chunks().flat_map(|c| c.iter().copied()).collect();
    assert_eq!(items, vec![42]);

    unsafe { v.deallocate_in(&allocator); }
}

#[test]
fn partial_fill_no_retired_chunks() {
    use crate::allocator::Global;
    let allocator = Global;
    let mut v: UnmanagedSegmentedVector<u32> = UnmanagedSegmentedVector::with_capacity_in(32, &allocator);
    for i in 0..5 {
        unsafe { v.push(i, &allocator); }
    }

    assert_eq!(v.len(), 5);

    let chunks: Vec<&[u32]> = v.chunks().collect();
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0], &[0, 1, 2, 3, 4]);

    unsafe { v.deallocate_in(&allocator); }
}

#[test]
fn exactly_one_retired_chunk() {
    use crate::allocator::Global;
    let allocator = Global;
    // MIN_CAPACITY is 16, so the first chunk has capacity 16.
    let mut v: UnmanagedSegmentedVector<u32> = UnmanagedSegmentedVector::with_capacity_in(16, &allocator);

    // Fill the first chunk (16 items) + push one more to trigger add_chunk.
    for i in 0..17 {
        unsafe { v.push(i, &allocator); }
    }

    assert_eq!(v.len(), 17);

    let chunks: Vec<&[u32]> = v.chunks().collect();
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].len(), 16);
    assert_eq!(chunks[1], &[16]);

    let items: Vec<u32> = chunks.iter().flat_map(|c| c.iter().copied()).collect();
    let expected: Vec<u32> = (0..17).collect();
    assert_eq!(items, expected);

    unsafe { v.deallocate_in(&allocator); }
}

#[test]
fn len_at_various_points() {
    use crate::allocator::Global;
    let allocator = Global;
    let mut v: UnmanagedSegmentedVector<u32> = UnmanagedSegmentedVector::with_capacity_in(16, &allocator);

    assert_eq!(v.len(), 0);
    unsafe { v.push(1, &allocator); }
    assert_eq!(v.len(), 1);

    for i in 1..16 {
        unsafe { v.push(i, &allocator); }
    }
    assert_eq!(v.len(), 16);

    // Triggers add_chunk.
    unsafe { v.push(99, &allocator); }
    assert_eq!(v.len(), 17);

    for _ in 0..100 {
        unsafe { v.push(0, &allocator); }
    }
    assert_eq!(v.len(), 117);

    unsafe { v.deallocate_in(&allocator); }
}

#[test]
fn many_retired_chunks() {
    use crate::allocator::Global;
    let allocator = Global;
    // Start small to create many chunks: capacity 16, then 32, 64, 128 ...
    let mut v: UnmanagedSegmentedVector<u32> = UnmanagedSegmentedVector::with_capacity_in(16, &allocator);

    // Push enough to create at least 4 retired chunks (16 + 32 + 64 + 128 = 240).
    let n = 250u32;
    for i in 0..n {
        unsafe { v.push(i, &allocator); }
    }

    assert_eq!(v.len(), n as usize);

    let chunks: Vec<&[u32]> = v.chunks().collect();
    assert!(chunks.len() >= 5, "expected at least 5 chunks, got {}", chunks.len());

    let items: Vec<u32> = chunks.iter().flat_map(|c| c.iter().copied()).collect();
    let expected: Vec<u32> = (0..n).collect();
    assert_eq!(items, expected);

    unsafe { v.deallocate_in(&allocator); }
}

#[test]
fn dealloc_after_partial_use() {
    use crate::allocator::Global;
    let allocator = Global;
    let mut v: UnmanagedSegmentedVector<u32> = UnmanagedSegmentedVector::with_capacity_in(32, &allocator);
    for i in 0..10 {
        unsafe { v.push(i, &allocator); }
    }
    assert_eq!(v.len(), 10);
    // Deallocate without filling the chunk.
    unsafe { v.deallocate_in(&allocator); }
}

// Note: ZST elements are not supported. The dangling pointer used as
// sentinel for "no retired chunks" collides with the ZST data pointer.

#[test]
fn chunk_sizes_grow() {
    use crate::allocator::Global;
    let allocator = Global;
    let mut v: UnmanagedSegmentedVector<u32> = UnmanagedSegmentedVector::with_capacity_in(16, &allocator);

    // Push enough to create several retired chunks.
    // Chunk capacities: 16, 32, 64, 128. Total = 240.
    for i in 0u32..250 {
        unsafe { v.push(i, &allocator); }
    }

    let chunk_lens: Vec<usize> = v.chunks().map(|c| c.len()).collect();
    // Retired chunks should be full, so their lengths equal their capacities.
    // The last chunk (current) may be partially filled.
    // Expected retired chunk sizes: 16, 32, 64, 128
    assert_eq!(chunk_lens[0], 16);
    assert_eq!(chunk_lens[1], 32);
    assert_eq!(chunk_lens[2], 64);
    assert_eq!(chunk_lens[3], 128);
    // Remaining 10 items in the current chunk.
    assert_eq!(chunk_lens[4], 250 - 16 - 32 - 64 - 128);

    unsafe { v.deallocate_in(&allocator); }
}
