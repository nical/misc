extern crate alloc as alloc_crate;

//pub mod alloc {
//    pub use allocator_api2::alloc::{AllocError, Allocator, Global};
//}

pub mod allocator;
pub mod frame_allocator;
pub mod global;
pub mod seg_vec;
pub mod unmanaged;
mod util;
pub mod vec;

pub use crate::unmanaged::{UnmanagedVector, UnmanagedHeaderVector};
pub use crate::vec::Vector;

#[macro_export]
macro_rules! impl_vector_methods {
    () => {
        /// Returns `true` if the vector contains no elements.
        #[inline(always)]
        pub fn is_empty(&self) -> bool {
            self.inner.is_empty()
        }

        /// Returns the number of elements in the vector.
        #[inline(always)]
        pub fn len(&self) -> usize {
            self.inner.len()
        }

        /// Returns the total number of elements the vector can hold without reallocating.
        #[inline(always)]
        pub fn capacity(&self) -> usize {
            self.inner.capacity()
        }

        /// Forces the length of the vector to `new_len`.
        ///
        /// # Safety
        ///
        /// - `new_len` must be less than or equal to `capacity()`.
        /// - The elements at `old_len`..`new_len` must be initialized.
        #[inline(always)]
        pub unsafe fn set_len(&mut self, new_len: usize) {
            self.inner.set_len(new_len);
        }

        /// Returns a reference to the underlying allocator.
        #[inline(always)]
        pub fn allocator(&self) -> &A {
            &self.allocator
        }

        /// Extracts a slice containing the entire vector.
        #[inline(always)]
        pub fn as_slice(&self) -> &[T] {
            self.inner.as_slice()
        }

        /// Extracts a mutable slice containing the entire vector.
        #[inline(always)]
        pub fn as_mut_slice(&mut self) -> &mut [T] {
            self.inner.as_mut_slice()
        }

        /// Clears the vector, removing all values.
        pub fn clear(&mut self) {
            self.inner.clear()
        }

        // TODO: get and similar functions are more expressive in the standard library.

        #[inline(always)]
        pub fn get(&self, index: usize) -> Option<&T> {
            self.inner.get(index)
        }

        /// Returns a reference to an element or subslice, without doing bounds checking.
        ///
        /// # Safety
        ///
        /// Calling this method with an out-of-bounds index is undefined behavior even
        /// if the resulting reference is not used.
        #[inline(always)]
        pub unsafe fn get_unchecked(&self, index: usize) -> &T {
            self.inner.get_unchecked(index)
        }

        #[inline(always)]
        pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
            self.inner.get_mut(index)
        }

        #[inline(always)]
        pub unsafe fn get_unchecked_mut(&mut self, index: usize) -> &T {
            self.inner.get_unchecked_mut(index)
        }

        /// Appends an element to the back of a collection.
        #[inline(always)]
        pub fn push(&mut self, val: T) {
            unsafe { self.inner.push(val, &self.allocator) }
        }

        /// Appends an element if there is sufficient spare capacity, otherwise
        /// an error is returned with the element.
        #[inline(always)]
        pub fn push_within_capacity(&mut self, val: T) -> Result<(), T> {
            self.inner.push_within_capacity(val)
        }

        /// Removes the last element from a vector and returns it, or `None` if it is empty.
        #[inline(always)]
        pub fn pop(&mut self) -> Option<T> {
            self.inner.pop()
        }

        /// Removes and returns the element at position `index` within the vector,
        /// shifting all elements after it to the left.
        #[inline(always)]
        pub fn remove(&mut self, index: usize) -> T {
            self.inner.remove(index)
        }

        /// Removes an element from the vector and returns it.
        ///
        /// The removed element is replaced by the last element of the vector.
        #[inline(always)]
        pub fn swap_remove(&mut self, index: usize) -> T {
            self.inner.swap_remove(index)
        }

        /// Inserts an element at position index within the vector, shifting all
        /// elements after it to the right.
        #[inline(always)]
        pub fn insert(&mut self, index: usize, element: T) {
            unsafe {
                self.inner.insert(index, element, &self.allocator);
            }
        }

        /// Clones and appends all elements in a slice to the vector.
        pub fn extend_from_slice(&mut self, slice: &[T])
        where
            T: Clone,
        {
            unsafe { self.inner.extend_from_slice(slice, &self.allocator) }
        }

        /// Shrinks the capacity of the vector with a lower bound.
        pub fn shrink_to(&mut self, new_cap: usize) {
            unsafe {
                self.inner.shrink_to(new_cap, &self.allocator)
            }
        }

        /// Shrinks the capacity of the vector as much as possible.
        pub fn shrink_to_fit(&mut self) {
            unsafe {
                self.inner.shrink_to_fit(&self.allocator)
            }
        }
    };
}
