use crate::buffer::UnmanagedHeaderBuffer;
use crate::util::{self, nnptr};
use crate::allocator::{Allocator, AllocError};

pub struct UnmanagedHeaderVec<H, T> {
    buffer: UnmanagedHeaderBuffer<H, T>,
    len: usize,
}

impl<T> UnmanagedHeaderVec<(), T> {
    pub fn new() -> Self {
        UnmanagedHeaderVec {
            buffer: UnmanagedHeaderBuffer::new(),
            len: 0,
        }
    }
}

impl<H, T> UnmanagedHeaderVec<H, T> {
    /// Creates an empty pre-allocated vector with a given storage capacity.
    pub fn try_with_capacity_in<A: Allocator>(header: H, cap: usize, allocator: &A) -> Result<Self, AllocError> {
        let buffer = UnmanagedHeaderBuffer::<H, T>::try_allocate_in(header, cap, allocator)?;

        Ok(UnmanagedHeaderVec {
            buffer,
            len: 0,
        })
    }

    pub fn try_from_slice<A: Allocator>(header: H, data: &[T], allocator: &A) -> Result<Self, AllocError>
    where
        T: Clone,
    {
        let mut v = Self::try_with_capacity_in(header, data.len(), allocator)?;
        unsafe {
            v.extend_from_slice(data, allocator);
        }

        Ok(v)
    }

    /// Tries to reserve at least enough space for `additional` extra items.
    ///
    /// # Safety
    ///
    /// The provided allocator must be the one this raw vector was created with.
    #[inline]
    pub unsafe fn try_reserve<A: Allocator>(&mut self, additional: usize, allocator: &A) -> Result<(), AllocError> {
        if self.remaining_capacity() < additional {
            self.try_realloc_additional(additional, allocator)?;
        }

        Ok(())
    }

    /// Clears and deallocates this raw vector, leaving it in its unallocated state.
    ///
    /// It is safe (no-op) to call `deallocate` on a vector that is already in its unallocated state.
    ///
    /// # Safety
    ///
    /// The provided allocator must be the one this raw vector was created with.
    pub unsafe fn deallocate_in<A: Allocator>(&mut self, allocator: &A) {
        unsafe {
            util::drop_items(self.buffer.header_ptr(), self.len);
            self.buffer.deallocate_in(allocator);
        }

        self.buffer = UnmanagedHeaderBuffer::dangling();
        self.len = 0;
    }

    // Note: Marking this #[inline(never)] is a pretty large regression in the push benchmark.
    #[cold]
    pub(crate) unsafe fn try_realloc_additional<A: Allocator>(&mut self, additional: usize, allocator: &A) -> Result<(), AllocError> {
        let new_cap = util::grow_amortized(self.len(), additional)?;
        if new_cap < self.len() {
            return Err(AllocError);
        }

        self.try_realloc_with_capacity(new_cap, allocator)
    }

    #[cold]
    pub(crate) unsafe fn try_realloc_with_capacity<A: Allocator>(&mut self, new_cap: usize, allocator: &A) -> Result<(), AllocError> {
        self.buffer.try_reallocate_in(new_cap, allocator)
    }

    #[inline]
    pub fn header(&self) -> &H {
        self.buffer.header()
    }

    #[inline]
    pub fn header_mut(&mut self) -> &mut H {
        self.buffer.header_mut()
    }

    #[inline]
    /// Returns `true` if the vector contains no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    /// Returns the number of elements in the vector, also referred to as its ‘length’.
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    /// Returns the total number of elements the vector can hold without reallocating.
    pub fn capacity(&self) -> usize {
        self.buffer.capacity()
    }

    /// Returns number of elements that can be added without reallocating.
    #[inline]
    pub fn remaining_capacity(&self) -> usize {
        self.capacity() - self.len
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe { self.buffer.as_slice(self.len()) }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { self.buffer.as_mut_slice(self.len()) }
    }

    /// Clears the vector, removing all values.
    pub fn clear(&mut self) {
        unsafe {
            util::drop_items(self.buffer.items_ptr(), self.len);
            self.len = 0;
        }
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
        if self.len == self.capacity() {
            self.try_realloc_additional(1, allocator).unwrap();
        }

        self.buffer.write_item(self.len, val);
        self.len += 1;
    }

    /// Appends an element if there is sufficient spare capacity, otherwise an error is returned
    /// with the element.
    ///
    /// Unlike push this method will not reallocate when there’s insufficient capacity.
    /// The caller should use reserve or try_reserve to ensure that there is enough capacity.
    #[inline]
    pub fn push_within_capacity(&mut self, val: T) -> Result<(), T> {
        if self.len == self.capacity() {
            return Err(val);
        }

        unsafe {
            self.buffer.write_item(self.len, val);
            self.len += 1;
        }

        Ok(())
    }

    /// Removes the last element from the vector and returns it, or `None` if it is empty.
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        unsafe {
            self.len -= 1;
            Some(self.buffer.read_item(self.len))
        }
    }

    /// Removes and returns the element at position `index` within the vector,
    /// shifting all elements after it to the left.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    ///
    pub fn remove(&mut self, index: usize) -> T {
        unsafe {
            let original_len = self.len;
            self.len -= 1;
            util::remove(self.buffer.items_ptr(), original_len, index)
        }
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
        unsafe {
            let original_len = self.len;
            self.len -= 1;
            util::swap_remove(self.buffer.items_ptr(), original_len, index)
        }
    }

    /// Inserts an element at position `index` within the vector, shifting all
    /// elements after it to the right.
    ///
    /// # Panics
    ///
    /// Panics if `index > len`.
    pub unsafe fn insert<A: Allocator>(&mut self, index: usize, element: T, allocator: &A) {
        #[cold]
        #[inline(never)]
        fn assert_failed(index: usize, len: usize) -> ! {
            panic!("insertion index (is {index}) should be <= len (is {len})");
        }

        unsafe {
            // space for the new element
            if self.len == self.capacity() {
                self.try_reserve(1, allocator).unwrap();
            }

            let len = self.len();

            // infallible
            // The spot to put the new value
            {
                let p = nnptr::add(self.buffer.items_ptr(), index);
                if index < len {
                    // Shift everything over to make space. (Duplicating the
                    // `index`th element into two consecutive places.)
                    nnptr::copy(p, nnptr::add(p, 1), len - index);
                } else if index == len {
                    // No elements need shifting.
                } else {
                    assert_failed(index, len);
                }
                // Write it in, overwriting the first copy of the `index`th
                // element.
                nnptr::write(p, element);
            }
            self.len += 1;
        }
    }

    /// Clones and appends the contents of the slice to the back of a collection.
    ///
    /// # Safety
    ///
    /// The provided allocator must be the one this raw vector was created with.
    pub unsafe fn extend_from_slice<A: Allocator>(&mut self, slice: &[T], allocator: &A)
    where
        T: Clone,
    {
        self.try_extend_from_slice(slice, allocator).unwrap();
    }

    pub unsafe fn try_extend_from_slice<A: Allocator>(&mut self, slice: &[T], allocator: &A) -> Result<(), AllocError>
    where
        T: Clone,
    {
        self.try_reserve(slice.len(), allocator)?;
        unsafe {
            util::extend_from_slice_assuming_capacity(self.buffer.items_ptr(), self.len, slice);
            self.len += slice.len();
        }

        Ok(())
    }

    pub unsafe fn extend_from_slice_within_capacity(&mut self, slice: &[T])
    where
        T: Clone,
    {
        let n = self.remaining_capacity().min(slice.len());
        unsafe {
            if n > 0 {
                util::extend_from_slice_assuming_capacity(self.buffer.items_ptr(), self.len, &slice[0..n]);
                self.len += n;
            }
        }
    }
}
