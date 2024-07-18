use crate::header_vec::RawHeaderVec;
use crate::allocator::{Allocator, AllocError};

pub struct RawVec<T> {
    inner: RawHeaderVec<(), T>,
}

impl<T> RawVec<T> {
    /// Creates an empty, unallocated raw vector.
    pub fn new() -> Self {
        RawVec { inner: RawHeaderVec::new() }
    }

    /// Creates an empty pre-allocated vector with a given storage capacity.
    ///
    /// Does not allocate memory if `cap` is zero.
    pub fn try_with_capacity_in<A: Allocator>(cap: usize, allocator: &A) -> Result<Self, AllocError> {
        if cap == 0 {
            return Ok(RawVec::new());
        }

        Ok(RawVec {
            inner: RawHeaderVec::try_with_capacity_in((), cap, allocator)?,
        })
    }

    pub fn try_from_slice<A: Allocator>(allocator: &A, data: &[T]) -> Result<Self, AllocError>
    where
        T: Clone,
    {
        if data.len() == 0 {
            return Ok(RawVec::new())
        }

        Ok(RawVec {
            inner: RawHeaderVec::try_from_slice((), data, allocator)?
        })
    }

    /// Tries to reserve at least enough space for `additional` extra items.
    ///
    /// # Safety
    ///
    /// The provided allocator must be the one this raw vector was created with.
    #[inline]
    pub unsafe fn try_reserve<A: Allocator>(&mut self, allocator: &A, additional: usize) -> Result<(), AllocError> {
        self.inner.try_reserve(additional, allocator)
    }

    /// Clears and deallocates this raw vector, leaving it in its unallocated state.
    ///
    /// It is safe (no-op) to call `deallocate` on a vector that is already in its unallocated state.
    ///
    /// # Safety
    ///
    /// The provided allocator must be the one this raw vector was created with.
    pub unsafe fn deallocate_in<A: Allocator>(&mut self, allocator: &A) {
        self.inner.deallocate_in(allocator);
    }

    #[inline]
    /// Returns `true` if the vector contains no elements.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    #[inline]
    /// Returns the number of elements in the vector, also referred to as its ‘length’.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    #[inline]
    /// Returns the total number of elements the vector can hold without reallocating.
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Returns number of elements that can be added without reallocating.
    #[inline]
    pub fn remaining_capacity(&self) -> usize {
        self.inner.remaining_capacity()
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.inner.as_slice()
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.inner.as_mut_slice()
    }

    /// Clears the vector, removing all values.
    pub fn clear(&mut self) {
        self.inner.clear()
    }

    /// Appends an element to the back of a collection.
    ///
    /// # Safety
    ///
    /// The provided allocator must be the one this raw vector was created with.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity exceeds `u32::MAX` bytes.
    #[inline]
    pub unsafe fn push<A: Allocator>(&mut self, val: T, allocator: &A) {
        self.inner.push(val, allocator)
    }

    /// Appends an element if there is sufficient spare capacity, otherwise an error is returned
    /// with the element.
    ///
    /// Unlike push this method will not reallocate when there’s insufficient capacity.
    /// The caller should use reserve or try_reserve to ensure that there is enough capacity.
    #[inline]
    pub fn push_within_capacity(&mut self, val: T) -> Result<(), T> {
        self.inner.push_within_capacity(val)
    }

    /// Removes the last element from the vector and returns it, or `None` if it is empty.
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        self.inner.pop()
    }

    /// Removes and returns the element at position `index` within the vector,
    /// shifting all elements after it to the left.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    ///
    pub fn remove(&mut self, index: usize) -> T {
        self.inner.remove(index)
    }

    /// Removes an element from the vector and returns it.
    ///
    /// The removed element is replaced by the last element of the vector.
    ///
    /// # Panics
    ///
    /// Panics if index is out of bounds.
    #[inline]
    pub fn swap_remove(&mut self, index: usize) -> T {
        self.inner.swap_remove(index)
    }

    /// Inserts an element at position `index` within the vector, shifting all
    /// elements after it to the right.
    ///
    /// # Panics
    ///
    /// Panics if `index > len`.
    pub unsafe fn insert<A: Allocator>(&mut self, index: usize, element: T, allocator: &A) {
       self.inner.insert(index, element, allocator)
    }

    /// Clones and appends the contents of the slice to the back of a collection.
    ///
    /// # Safety
    ///
    /// The provided allocator must be the one this raw vector was created with.
    pub unsafe fn extend_from_slice<A: Allocator>(&mut self, allocator: &A, slice: &[T])
    where
        T: Clone,
    {
        self.inner.extend_from_slice(slice, allocator)
    }
}
