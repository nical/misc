use crate::allocator::{AllocError, Allocator};
use crate::unmanaged::{AllocInit, UnmanagedVector};

/// A segmented vector that grows by allocating new chunks instead of
/// reallocating and copying existing data.
///
/// The current (last) chunk is stored inline to avoid indirection on
/// push/pop. Previous chunks are kept in a separate vector.
///
/// Push and pop are O(1) amortized. Elements are not contiguous in memory;
/// iteration yields one slice per chunk.
///
/// # Safety
///
/// This is an *unmanaged* container. The caller is responsible for calling
/// `drop_and_deallocate_in` (or `deallocate_in` for non-Drop types) with
/// the correct allocator before the value is forgotten.
pub struct UnmanagedSegmentedVector<T> {
    /// Previous full chunks, in order.
    chunks: UnmanagedVector<UnmanagedVector<T>>,
    /// The current chunk we push into. Stored inline for fast access.
    current: UnmanagedVector<T>,
    len: usize,
}

impl<T> UnmanagedSegmentedVector<T> {
    pub fn new() -> Self {
        UnmanagedSegmentedVector {
            chunks: UnmanagedVector::new(),
            current: UnmanagedVector::new(),
            len: 0,
        }
    }

    pub fn try_with_capacity_in<A: Allocator>(cap: usize, allocator: &A) -> Result<Self, AllocError> {
        let current = UnmanagedVector::try_with_capacity_in((), cap, AllocInit::Uninit, allocator)?;

        Ok(UnmanagedSegmentedVector {
            chunks: UnmanagedVector::new(),
            current,
            len: 0,
        })
    }

    pub fn with_capacity_in<A: Allocator>(cap: usize, allocator: &A) -> Self {
        Self::try_with_capacity_in(cap, allocator).unwrap()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Appends an element to the back of the vector.
    ///
    /// # Safety
    ///
    /// The provided allocator must be the one this vector was created with.
    #[inline]
    pub unsafe fn push<A: Allocator>(&mut self, val: T, allocator: &A) {
        if self.current.capacity() == self.current.len() {
            self.grow(allocator);
        }

        unsafe {
            self.current.push_assuming_capacity(val);
        }
        self.len += 1;
    }

    #[cold]
    unsafe fn grow<A: Allocator>(&mut self, allocator: &A) {
        let cap = self.current.capacity().max(crate::MIN_CAPACITY) * 2;

        let new_chunk = UnmanagedVector::try_with_capacity_in((), cap, AllocInit::Uninit, allocator).unwrap();
        // Move the full current chunk into the previous list.
        let full_chunk = core::mem::replace(&mut self.current, new_chunk);
        if full_chunk.capacity() > 0 {
            self.chunks.push(full_chunk, allocator);
        }
    }

    /// Removes the last element and returns it, or `None` if empty.
    /// Deallocates chunks that become empty.
    ///
    /// # Safety
    ///
    /// The provided allocator must be the one this vector was created with.
    pub unsafe fn pop<A: Allocator>(&mut self, allocator: &A) -> Option<T> {
        if let Some(val) = self.current.pop() {
            self.len -= 1;
            return Some(val);
        }

        // Current chunk is empty — deallocate it and promote the previous one.
        if self.current.capacity() > 0 {
            self.current.deallocate_in(allocator);
        }

        self.current = self.chunks.pop().unwrap_or_else(UnmanagedVector::new);
        let val = self.current.pop()?;
        self.len -= 1;
        Some(val)
    }

    /// Deallocates all storage without dropping elements.
    ///
    /// Appropriate when `T` does not need drop (e.g. `Copy` types).
    ///
    /// # Safety
    ///
    /// The provided allocator must be the one this vector was created with.
    /// If `T` implements `Drop`, the destructors will not run — this leaks.
    pub unsafe fn deallocate_in<A: Allocator>(&mut self, allocator: &A) {
        for chunk in self.chunks.as_mut_slice() {
            chunk.deallocate_in(allocator);
        }
        self.chunks.deallocate_in(allocator);
        self.current.deallocate_in(allocator);
        self.len = 0;
    }

    /// Iterates over chunks, yielding each as a slice.
    pub fn chunks(&self) -> Chunks<'_, T> {
        Chunks {
            previous: self.chunks.as_slice().iter(),
            current: Some(&self.current),
        }
    }

    /// Iterates over chunks, yielding each as a mutable slice.
    pub fn chunks_mut(&mut self) -> ChunksMut<'_, T> {
        ChunksMut {
            previous: self.chunks.as_mut_slice().iter_mut(),
            current: Some(&mut self.current),
        }
    }

    /// Iterates over all elements in order.
    pub fn iter(&self) -> Iter<'_, T> {
        let mut chunks = self.chunks();
        let inner = chunks.next().unwrap_or(&[]);
        Iter { chunks, inner }
    }
}

/// Iterator over chunk slices.
pub struct Chunks<'a, T> {
    previous: core::slice::Iter<'a, UnmanagedVector<T>>,
    current: Option<&'a UnmanagedVector<T>>,
}

impl<'a, T> Iterator for Chunks<'a, T> {
    type Item = &'a [T];
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(chunk) = self.previous.next() {
            return Some(chunk.as_slice());
        }
        let current = self.current.take()?;
        if current.is_empty() {
            return None;
        }
        Some(current.as_slice())
    }
}

/// Iterator over mutable chunk slices.
pub struct ChunksMut<'a, T> {
    previous: core::slice::IterMut<'a, UnmanagedVector<T>>,
    current: Option<&'a mut UnmanagedVector<T>>,
}

impl<'a, T> Iterator for ChunksMut<'a, T> {
    type Item = &'a mut [T];
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(chunk) = self.previous.next() {
            return Some(chunk.as_mut_slice());
        }
        let current = self.current.take()?;
        if current.is_empty() {
            return None;
        }
        Some(current.as_mut_slice())
    }
}

/// Iterator over all elements in order.
pub struct Iter<'a, T> {
    chunks: Chunks<'a, T>,
    inner: &'a [T],
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some((first, rest)) = self.inner.split_first() {
                self.inner = rest;
                return Some(first);
            }
            self.inner = self.chunks.next()?;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allocator::Global;

    #[test]
    fn empty() {
        let allocator = Global;
        let mut v: UnmanagedSegmentedVector<i32> = UnmanagedSegmentedVector::new();
        assert_eq!(v.len(), 0);
        assert!(v.is_empty());
        assert_eq!(unsafe { v.pop(&allocator) }, None);
        assert_eq!(v.chunks().count(), 0);
        assert_eq!(v.iter().count(), 0);
        unsafe { v.deallocate_in(&allocator); }
    }

    #[test]
    fn push_and_iterate() {
        let allocator = Global;
        let mut v = UnmanagedSegmentedVector::with_capacity_in(16, &allocator);
        for i in 0i32..200 {
            unsafe { v.push(i, &allocator); }
        }

        assert_eq!(v.len(), 200);

        let collected: Vec<i32> = v.iter().copied().collect();
        let expected: Vec<i32> = (0..200).collect();
        assert_eq!(collected, expected);

        // Also verify via chunks
        let mut idx = 0i32;
        for chunk in v.chunks() {
            for &item in chunk {
                assert_eq!(item, idx);
                idx += 1;
            }
        }
        assert_eq!(idx, 200);

        unsafe { v.deallocate_in(&allocator); }
    }

    #[test]
    fn pop() {
        let allocator = Global;
        let mut v = UnmanagedSegmentedVector::with_capacity_in(4, &allocator);
        for i in 0i32..10 {
            unsafe { v.push(i, &allocator); }
        }

        for i in (0..10i32).rev() {
            assert_eq!(unsafe { v.pop(&allocator) }, Some(i));
        }
        assert_eq!(unsafe { v.pop(&allocator) }, None);
        assert_eq!(v.len(), 0);

        unsafe { v.deallocate_in(&allocator); }
    }

    #[test]
    fn pop_in_deallocates_chunks() {
        let allocator = Global;
        let mut v = UnmanagedSegmentedVector::with_capacity_in(4, &allocator);
        for i in 0i32..100 {
            unsafe { v.push(i, &allocator); }
        }

        while let Some(_) = unsafe { v.pop(&allocator) } {}
        assert_eq!(v.len(), 0);

        unsafe { v.deallocate_in(&allocator); }
    }

    #[test]
    fn single_chunk_deallocate() {
        let allocator = Global;
        let mut v: UnmanagedSegmentedVector<u32> = UnmanagedSegmentedVector::with_capacity_in(16, &allocator);
        unsafe { v.push(42, &allocator); }
        unsafe { v.deallocate_in(&allocator); }
    }

    #[test]
    fn with_capacity_no_push() {
        let allocator = Global;
        let mut v: UnmanagedSegmentedVector<u32> = UnmanagedSegmentedVector::with_capacity_in(16, &allocator);
        unsafe { v.deallocate_in(&allocator); }
    }

    #[test]
    fn drop_types() {
        let allocator = Global;
        let mut v: UnmanagedSegmentedVector<String> = UnmanagedSegmentedVector::with_capacity_in(4, &allocator);
        for i in 0..20 {
            unsafe { v.push(format!("item {i}"), &allocator); }
        }
        assert_eq!(v.len(), 20);
        unsafe { v.deallocate_in(&allocator); }
    }

    #[test]
    fn chunks_mut() {
        let allocator = Global;
        let mut v = UnmanagedSegmentedVector::with_capacity_in(4, &allocator);
        for i in 0i32..10 {
            unsafe { v.push(i, &allocator); }
        }

        for chunk in v.chunks_mut() {
            for item in chunk.iter_mut() {
                *item *= 2;
            }
        }

        let collected: Vec<i32> = v.iter().copied().collect();
        let expected: Vec<i32> = (0..10).map(|i| i * 2).collect();
        assert_eq!(collected, expected);

        unsafe { v.deallocate_in(&allocator); }
    }
}
