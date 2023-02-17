//! Shared and mutable vectors.
//!
//! This crate provides the following two types:
//! - `SharedVector<T>`, an immutable reference counted vector with a value-oriented API.
//! - `MutableVector<T>`, an unique vector type with an API similar to `std::Vec<T>`.
//!
//! Internally these two types share the same representation: Their value is a single pointer to a buffer containing
//! - A 16 bytes header (plus possible padding for alignment),
//! - the contiguous sequence of items of type `T`.
//!
//! This allows very cheap conversion between the two:
//! - shared to mutable: a new allocation is made only if there are other handles to the same buffer (the reference
//!   count is greather than one).
//! - mutable to shared: always free since mutable buffers are guarantted to be unique.
//!
//! # Use cases
//!
//! ## `Arc<Vec<T>>` without the indirection.
//!
//! A mutable vector can be be built using a Vec-style API, and then made immutable and reference counted for various
//! use case (easy multi-threading or simply shared ownership).
//!
//! Using the standard library one might be tempted to firs build a `Vec<T>` and share it via `Arc<Vec<T>>`. This is
//! a fine approach at the cost of an extra pointer indirection.
//! Another approach is to share it as an `Arc<[T]>` which removes the indirection at the cost of the need to copy
//! from the vector.
//!
//! Using this crate there is no extra indirection in the resulting shared vector or copy between the mutable and
//! shared versions.
//!
//! ```
//! use immutable::MutableVector;
//! let mut builder = MutableVector::with_capacity(16);
//! builder.push(1u32);
//! builder.push(2);
//! builder.push(3);
//! // Make it reference counted, no allocation.
//! let mut shared = builder.into_shared();
//! // We can now create new references
//! let shared_2 = shared.new_ref();
//! let shared_3 = shared.new_ref();
//! ```
//!
//! ## You like immutable data structures and value-oriented APIs.
//!
//! You keep telling everyone around you about the "value of values"? That's OK! Maybe you'll enjoy using the
//! `SharedVector` type.
//!
//! ```
//! use immutable::SharedVector;
//! let a = SharedVector::with_capacity(16)
//!     .push(1u32)
//!     .push(2)
//!     .push(3);
//!
//! // `new_ref` (you can also use `clone`) creates a second reference to the same buffer.
//! let b = a.new_ref();
//!
//! let a = a.push(4) // This push needs to allocate new storage because there multiple references.
//!     .push(5);     // This one does not.
//!
//! assert_eq!(a.as_slice(), &[1, 2, 3, 4, 5]);
//! assert_eq!(b.as_slice(), &[1, 2, 3]);
//! ```
//!
//! Note that `SharedVector` is *not* a RRB vector implementation.
//!
//! ## The slim value representation
//!
//! That's certainly niche but the representation being different than `std::Vec`'s that may be good or
//! bad for you depending on what you want to do with it.
//!
//! ```ascii
//!  +---+
//!  |   | SharedVector (8 bytes on 64bit systems)
//!  +---+
//!    |
//!    v
//!  +----------++----+----+----+----+----+----+----+----+
//!  |          ||    |    |    |    |    |    |    |    |
//!  +----------++----+----+----+----+----+----+----+----+
//!   \________/  \_____________________________________/
//!     Header                  Items
//!   (16 bytes)
//! ```
//!
//! Both `SharedVector` and `ImmutableVector` contain a single pointer, so they occupy a third of the space
//! of `Vec<T>` and half the space of `Box<[T]>`. Of course that comes at a price, most methods need to
//! have to read the header located at the beginning of the buffer. That may not matter if the most common
//! operation is to iterate over vector, but large numbers of random accesses will likely be slower.
//!
//! # Limitiations
//!
//! These vector types can hold at most `u32::MAX` elements.
//! ```

mod raw;
//pub mod store;
//pub mod chunked;
//pub mod value;

use std::mem;
use std::ops::{Index, IndexMut};

use raw::{AllocError, BufferSize, RawBuffer, HeaderBuffer};

// TODO: Value oriented vectors are cool for simple things but interact poorly with mutability of surrounding code
// or some types of APIs like `last_mut`. Maybe a compromise would be to make both vectors mutable by default, but
// also provide a `Value<V>` wrapper that people can opt into to get a subset of the API ina value-oriented way.


// TODO: if checking the atomic refcount is costly, we could set a bit in SharedVector when we know the handle
// to be unique to avoid that.

/// A heap allocated, reference counted, immutable contiguous buffer containing elements of type `T`.
///
/// Similar in principle to `Arc<[T]>`. It can be converted into a `MutableChumk<T>` for
/// free if there is only a single reference to the SharedVector alive.
pub struct SharedVector<T> {
    inner: RawBuffer<T>,
}

// TODO: MutableVector does not need to have a different representation from std::Vec other than allocating
// space for the header. The header has to be filled out when converting into a mutable vector but if perforance
// demands it, the length and capacity could be managed inline to avoid reading the header and to avoid the
// offset.

/// A heap allocated, mutable contiguous buffer containing elements of type `T`.
///
/// Similar in principle to a `Box<ArrayVec<T, Size>>` where size would be set at runtime.
/// It can be converted for free into an immutable `SharedVector<T>`.
pub struct MutableVector<T> {
    inner: RawBuffer<T>,
}

pub struct UniqueVector<T> {
    inner: HeaderBuffer<raw::Header, T>,
}


impl<T> SharedVector<T> {
    /// Creates an empty shared buffer without allocating memory.
    #[inline]
    pub fn new() -> Self {
        SharedVector {
            inner: RawBuffer::new_empty(),
        }
    }

    #[inline]
    pub fn with_capacity(cap: usize) -> Self {
        SharedVector {
            inner: RawBuffer::try_with_capacity(cap).unwrap(),
        }
    }

    #[inline]
    pub fn from_slice(data: &[T]) -> Self
    where
        T: Clone,
    {
        SharedVector {
            inner: RawBuffer::try_from_slice(data, None).unwrap(),
        }
    }

    /// Returns `true` if the vector contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns the number of elements in the vector, also referred to as its ‘length’.
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns the total number of elements the vector can hold without reallocating.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Returns number of elements that can be added without reallocating.
    #[inline]
    pub fn remaining_capacity(&self) -> usize {
        self.inner.remaining_capacity()
    }

    /// Creates a new reference without allocating.
    ///
    /// Equivalent to `Clone::clone`.
    #[inline]
    pub fn new_ref(&self) -> Self {
        SharedVector {
            inner: self.inner.new_ref(),
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.inner.as_slice()
    }

    /// Allocates a duplicate of this buffer (infallible).
    pub fn clone_buffer(&self) -> Self
    where
        T: Clone,
    {
        SharedVector {
            inner: self.inner.try_clone_buffer().unwrap(),
        }
    }

    /// Allocates a duplicate of this buffer (infallible).
    pub fn copy_buffer(&self) -> Self
    where
        T: Copy,
    {
        SharedVector {
            inner: self.inner.try_copy_buffer().unwrap(),
        }
    }

    /// Returns true if this is the only existing handle to the buffer.
    #[inline]
    pub fn can_mutate(&self) -> bool {
        let b = self.inner.can_mutate();
        assert!(b);
        b
    }

    /// Converts this SharedVector into an immutable one, allocating a new one if there are other references.
    #[inline]
    pub fn into_mut(mut self) -> MutableVector<T>
    where
        T: Clone,
    {
        self.ensure_mutable();
        MutableVector {
            inner: self.inner,
        }
    }

    /// Converts this shared buffer into a mutable one if it is the only reference to its data.
    ///
    /// Never allocates.
    #[inline]
    pub fn try_into_mut(self) -> Option<Self> {
        if self.can_mutate() {
            Some(self)
        } else {
            None
        }
    }

    #[inline]
    pub fn first(&self) -> Option<&T> {
        unsafe { self.inner.first().map(|ptr| &*ptr) }
    }

    #[inline]
    pub fn last(&self) -> Option<&T> {
        unsafe { self.inner.last().map(|ptr| &*ptr) }
    }

    #[inline]
    pub fn first_mut(&mut self) -> Option<&mut T> where T: Clone {
        self.ensure_mutable();
        unsafe { self.inner.first().map(|ptr| &mut *ptr) }
    }

    #[inline]
    pub fn last_mut(&mut self) -> Option<&mut T>  where T: Clone {
        self.ensure_mutable();
        unsafe { self.inner.last().map(|ptr| &mut *ptr) }
    }

    pub fn push(&mut self, val: T)
    where
        T: Clone,
    {
        self.ensure_capacity(1);
        unsafe {
            self.inner.push(val);
        }
    }

    pub fn pop(&mut self) -> Option<T>
    where
        T: Clone,
    {
        self.ensure_mutable();
        unsafe { self.inner.pop() }
    }

    pub fn push_slice(&mut self, data: &[T])
    where
        T: Clone,
    {
        self.ensure_capacity(data.len());
        unsafe {
            self.inner.try_push_slice(data).unwrap();
        }
    }

    pub fn extend(&mut self, data: impl IntoIterator<Item = T>)
    where
        T: Clone,
    {
        let mut iter = data.into_iter();
        let (min, max) = iter.size_hint();
        self.ensure_capacity(max.unwrap_or(min));
        unsafe {
            self.inner.try_extend(&mut iter).unwrap();
        }
    }

    pub fn clear(&mut self) {
        if self.can_mutate() {
            unsafe { self.inner.clear(); }
        }

        *self = Self::with_capacity(self.capacity());
    }

    /// Returns true if the two shapred buffers point to the same underlying storage.
    pub fn ptr_eq(&self, other: &Self) -> bool {
        self.inner.ptr_eq(&other.inner)
    }

    #[inline]
    fn ensure_mutable(&mut self)
    where
        T: Clone,
    {
        if !self.can_mutate() {
            self.inner = self.inner.try_clone_buffer().unwrap();
        }
    }

    /// Returns a buffer that can be safely mutated and has enough extra capacity to
    /// add `additional` more items.
    #[inline]
    pub fn ensure_capacity(&mut self, additional: usize)
    where
        T: Clone,
    {
        let can_mutate = self.can_mutate();
        let off_capacity = self.remaining_capacity() <= additional;

        if !can_mutate || off_capacity {
            // Hopefully the least common case.
            self.realloc(can_mutate, additional);
        }
    }

    #[cold]
    fn realloc(&mut self, can_mutate: bool, additional: usize)
    where
        T: Clone,
    {
        let new_cap = grow_amortized(self.len(), additional);

        if can_mutate {
            // The buffer is not large enough, we'll have to create a new one, however we
            // know that we have the only reference to it so we'll move the data with
            // a simple memcpy instead of cloning it.
            unsafe {
                let dst = raw::copy_buffer(self.inner.header, Some(new_cap as BufferSize)).unwrap();
                self.inner.set_len(0);

                self.inner = dst;
            }
        }

        // The slowest path, we pay for both the new allocation and the need to clone
        // each item one by one.
        let new_cap = grow_amortized(self.len(), additional);
        self.inner = RawBuffer::try_from_slice(self.as_slice(), Some(new_cap)).unwrap();
    }

    /// Returns the concatenation of two vectors.
    pub fn concatenate(mut self, mut other: Self) -> Self
    where
        T: Clone,
    {
        self.ensure_capacity(other.len());

        unsafe {
            if other.can_mutate() {
                // Fast path: memcpy
                other.inner.move_data(&mut self.inner);
            } else {
                // Slow path, clone each item.
                self.inner.try_push_slice(other.as_slice()).unwrap();
            }
        }

        self
    }
}

unsafe impl<T: Sync> Send for SharedVector<T> {}

impl<T> Clone for SharedVector<T> {
    fn clone(&self) -> Self {
        self.new_ref()
    }
}

impl<T: PartialEq<T>> PartialEq<SharedVector<T>> for SharedVector<T> {
    fn eq(&self, other: &Self) -> bool {
        self.ptr_eq(other) || self.as_slice() == other.as_slice()
    }
}

impl<T: PartialEq<T>> PartialEq<&[T]> for SharedVector<T> {
    fn eq(&self, other: &&[T]) -> bool {
        self.as_slice() == *other
    }
}

impl<T> AsRef<[T]> for SharedVector<T> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> Default for SharedVector<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, T> IntoIterator for &'a SharedVector<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> std::slice::Iter<'a, T> {
        self.as_slice().iter()
    }
}

impl<T, I> Index<I> for SharedVector<T>
where
    I: std::slice::SliceIndex<[T]>,
{
    type Output = <I as std::slice::SliceIndex<[T]>>::Output;
    fn index(&self, index: I) -> &Self::Output {
        self.as_slice().index(index)
    }
}

impl<T> MutableVector<T> {
    /// Allocates a mutable buffer with a default capacity of 16.
    pub fn new() -> Self {
        Self::with_capacity(16)
    }

    pub fn with_capacity(cap: usize) -> Self {
        MutableVector {
            inner: RawBuffer::try_with_capacity(cap).unwrap(),
        }
    }

    pub fn from_slice(data: &[T]) -> Self
    where
        T: Clone,
    {
        MutableVector {
            inner: RawBuffer::try_from_slice(data, None).unwrap(),
        }
    }

    /// Returns `true` if the vector contains no elements.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns the number of elements in the vector, also referred to as its ‘length’.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns the total number of elements the vector can hold without reallocating.
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Returns number of elements that can be added without reallocating.
    #[inline]
    fn remaining_capacity(&self) -> usize {
        self.capacity() - self.len()
    }

    /// Make this SharedVector immutable.
    ///
    /// This operation is cheap, the underlying storage does not not need
    /// to be reallocated.
    #[inline]
    pub fn into_shared(self) -> SharedVector<T> {
        SharedVector { inner: self.inner }
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.inner.as_slice()
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // Safe because this type guarantees mutability.
        unsafe { self.inner.as_mut_slice() }
    }

    #[inline]
    pub fn first(&self) -> Option<&T> {
        unsafe { self.inner.first().map(|ptr| &*ptr) }
    }

    #[inline]
    pub fn last(&self) -> Option<&T> {
        unsafe { self.inner.last().map(|ptr| &*ptr) }
    }

    #[inline]
    pub fn first_mut(&mut self) -> Option<&mut T> {
        unsafe { self.inner.first().map(|ptr| &mut *ptr) }
    }

    #[inline]
    pub fn last_mut(&mut self) -> Option<&mut T> {
        unsafe { self.inner.last().map(|ptr| &mut *ptr) }
    }

    #[inline]
    pub fn push(&mut self, val: T) {
        self.ensure_capacity(1);

        // Safe because this type guarantees mutability.
        unsafe {
            self.inner.push(val);
        }
    }

    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        // Safe because this type guarantees mutability.
        unsafe { self.inner.pop() }
    }

    #[inline]
    pub fn push_slice(&mut self, data: &[T])
    where
        T: Clone,
    {
        self.ensure_capacity(data.len());

        // Safe because this type guarantees mutability.
        unsafe {
            self.inner.try_push_slice(data).unwrap();
        }
    }

    pub fn extend(&mut self, data: impl IntoIterator<Item = T>) {
        let mut iter = data.into_iter();
        let (min, max) = iter.size_hint();
        self.ensure_capacity(max.unwrap_or(min));
        // Safe because this type guarantees mutability.
        unsafe {
            self.inner.try_extend(&mut iter).unwrap();
        }
    }

    #[inline]
    pub fn clear(&mut self) {
        unsafe {
            self.inner.clear();
        }
    }

    pub fn clone_buffer(&self) -> Self
    where
        T: Clone,
    {
        MutableVector {
            inner: self.inner.try_clone_buffer().unwrap(),
        }
    }

    pub fn clone_buffer_with_capacity(&self, cap: BufferSize) -> Self
    where
        T: Clone,
    {
        MutableVector {
            inner: self.inner.clone_buffer_with_capacity(cap),
        }
    }

    fn try_realloc(&mut self, new_cap: usize) -> Result<(), AllocError> {
        let new_cap = grow_amortized(self.len(), new_cap);

        if new_cap < self.len() {
            return Err(AllocError::CapacityOverflow);
        }

        let mut dst = RawBuffer::try_with_capacity(new_cap)?;

        unsafe { self.inner.move_data(&mut dst) };

        mem::swap(&mut self.inner, &mut dst);

        Ok(())
    }

    // Note: Marking this #[inline(never)] is a pretty large regression in the push benchmark.
    #[cold]
    fn realloc(&mut self, new_cap: usize) {
        self.try_realloc(new_cap).unwrap();
    }

    #[inline]
    fn ensure_capacity(&mut self, additional: usize) {
        if self.remaining_capacity() < additional {
            self.realloc(additional)
        }
    }
}

impl<T: Clone> Clone for MutableVector<T> {
    fn clone(&self) -> Self {
        self.clone_buffer()
    }
}

impl<T: PartialEq<T>> PartialEq<MutableVector<T>> for MutableVector<T> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: PartialEq<T>> PartialEq<&[T]> for MutableVector<T> {
    fn eq(&self, other: &&[T]) -> bool {
        self.as_slice() == *other
    }
}

impl<T> AsRef<[T]> for MutableVector<T> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> AsMut<[T]> for MutableVector<T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T> Default for MutableVector<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, T> IntoIterator for &'a MutableVector<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> std::slice::Iter<'a, T> {
        self.as_slice().iter()
    }
}

impl<T, I> Index<I> for MutableVector<T>
where
    I: std::slice::SliceIndex<[T]>,
{
    type Output = <I as std::slice::SliceIndex<[T]>>::Output;
    fn index(&self, index: I) -> &Self::Output {
        self.as_slice().index(index)
    }
}

impl<T, I> IndexMut<I> for MutableVector<T>
where
    I: std::slice::SliceIndex<[T]>,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        self.as_mut_slice().index_mut(index)
    }
}



impl<T> UniqueVector<T> {
    /// Allocates a mutable buffer with a default capacity of 16.
    pub fn new() -> Self {
        Self::with_capacity(16)
    }

    pub fn with_capacity(cap: usize) -> Self {
        UniqueVector {
            inner: HeaderBuffer::try_with_capacity(cap).unwrap(),
        }
    }

    pub fn from_slice(data: &[T]) -> Self
    where
        T: Clone,
    {
        UniqueVector {
            inner: HeaderBuffer::try_from_slice(data, None).unwrap(),
        }
    }

    /// Returns `true` if the vector contains no elements.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns the number of elements in the vector, also referred to as its ‘length’.
    pub fn len(&self) -> usize {
        self.inner.len() as usize
    }

    /// Returns the total number of elements the vector can hold without reallocating.
    pub fn capacity(&self) -> usize {
        self.inner.capacity() as usize
    }

    /// Returns number of elements that can be added without reallocating.
    #[inline]
    fn remaining_capacity(&self) -> usize {
        self.inner.remaining_capacity() as usize
    }

    // /// Make this SharedVector immutable.
    // ///
    // /// This operation is cheap, the underlying storage does not not need
    // /// to be reallocated.
    // #[inline]
    // pub fn into_shared(self) -> SharedVector<T> {
    //     SharedVector { inner: self.inner }
    // }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.inner.as_slice()
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // Safe because this type guarantees mutability.
        unsafe { self.inner.as_mut_slice() }
    }

    #[inline]
    pub fn first(&self) -> Option<&T> {
        unsafe { self.inner.first().map(|ptr| &*ptr) }
    }

    #[inline]
    pub fn last(&self) -> Option<&T> {
        unsafe { self.inner.last().map(|ptr| &*ptr) }
    }

    #[inline]
    pub fn first_mut(&mut self) -> Option<&mut T> {
        unsafe { self.inner.first().map(|ptr| &mut *ptr) }
    }

    #[inline]
    pub fn last_mut(&mut self) -> Option<&mut T> {
        unsafe { self.inner.last().map(|ptr| &mut *ptr) }
    }

    #[inline]
    pub fn push(&mut self, val: T) {
        self.ensure_capacity(1);

        // Safe because this type guarantees mutability.
        unsafe {
            self.inner.push(val);
        }
    }

    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        // Safe because this type guarantees mutability.
        unsafe { self.inner.pop() }
    }

    #[inline]
    pub fn push_slice(&mut self, data: &[T])
    where
        T: Clone,
    {
        self.ensure_capacity(data.len());

        // Safe because this type guarantees mutability.
        unsafe {
            self.inner.try_push_slice(data).unwrap();
        }
    }

    pub fn extend(&mut self, data: impl IntoIterator<Item = T>) {
        let mut iter = data.into_iter();
        let (min, max) = iter.size_hint();
        self.ensure_capacity(max.unwrap_or(min));
        // Safe because this type guarantees mutability.
        unsafe {
            self.inner.try_extend(&mut iter).unwrap();
        }
    }

    #[inline]
    pub fn clear(&mut self) {
        unsafe {
            self.inner.clear();
        }
    }

    pub fn clone_buffer(&self) -> Self
    where
        T: Clone,
    {
        UniqueVector {
            inner: self.inner.try_clone_buffer(None).unwrap(),
        }
    }

    pub fn clone_buffer_with_capacity(&self, cap: BufferSize) -> Self
    where
        T: Clone,
    {
        UniqueVector {
            inner: self.inner.try_clone_buffer(Some(cap)).unwrap(),
        }
    }

    fn try_realloc(&mut self, new_cap: usize) -> Result<(), AllocError> {
        let new_cap = grow_amortized(self.len(), new_cap);

        if new_cap < self.len() {
            return Err(AllocError::CapacityOverflow);
        }

        let mut dst = HeaderBuffer::try_with_capacity(new_cap)?;

        unsafe { self.inner.move_data(&mut dst) };

        mem::swap(&mut self.inner, &mut dst);

        Ok(())
    }

    // Note: Marking this #[inline(never)] is a pretty large regression in the push benchmark.
    #[cold]
    fn realloc(&mut self, new_cap: usize) {
        self.try_realloc(new_cap).unwrap();
    }

    #[inline]
    fn ensure_capacity(&mut self, additional: usize) {
        if self.remaining_capacity() < additional {
            self.realloc(additional)
        }
    }
}




fn grow_amortized(len: usize, additional: usize) -> usize {
    let required = len.saturating_add(additional);
    let cap = len.saturating_add(len).max(required).min(8);

    const MAX: usize = BufferSize::MAX as usize;

    if cap > MAX {
        if required <= MAX {
            return required;
        }

        panic!("Required allocation size is too large");
    }

    cap
}

// In order to give us a chance to catch leaks and double-frees, test with values that implement drop.
#[cfg(test)]
fn num(val: u32) -> Box<u32> {
    Box::new(val)
}

#[test]
fn basic() {
    let mut a = MutableVector::with_capacity(256);

    a.push(num(0));
    a.push(num(1));
    a.push(num(2));

    let a = a.into_shared();

    assert_eq!(a.len(), 3);

    assert_eq!(a.as_slice(), &[num(0), num(1), num(2)]);

    assert!(a.can_mutate());

    let b = MutableVector::from_slice(&[num(0), num(1), num(2), num(3), num(4)]);

    assert_eq!(b.as_slice(), &[num(0), num(1), num(2), num(3), num(4)]);

    let c = a.clone_buffer();
    assert!(!c.ptr_eq(&a));

    let a2 = a.new_ref();
    assert!(a2.ptr_eq(&a));
    assert!(!a.can_mutate());
    assert!(!a2.can_mutate());

    mem::drop(a2);

    assert!(a.can_mutate());

    let _ = c.clone_buffer();
    let _ = b.clone_buffer();

    let mut d = MutableVector::with_capacity(64);
    d.push_slice(&[num(0), num(1), num(2)]);
    d.push_slice(&[]);
    d.push_slice(&[num(3), num(4)]);

    assert_eq!(d.as_slice(), &[num(0), num(1), num(2), num(3), num(4)]);
}

#[test]
fn value_oriented() {
    let mut a = SharedVector::with_capacity(64);
    a.push(num(1));
    a.push(num(2));

    let mut b = a.new_ref();
    b.push(num(4));

    a.push(num(3));

    assert_eq!(a.as_slice(), &[num(1), num(2), num(3)]);
    assert_eq!(b.as_slice(), &[num(1), num(2), num(4)]);

    let popped = a.pop();
    assert_eq!(a.as_slice(), &[num(1), num(2)]);
    assert_eq!(popped, Some(num(3)));

    let mut b2 = b.new_ref();
    let popped = b2.pop();
    assert_eq!(b2.as_slice(), &[num(1), num(2)]);
    assert_eq!(popped, Some(num(4)));

    let c = a.concatenate(b2);
    assert_eq!(c.as_slice(), &[num(1), num(2), num(1), num(2)]);
}

#[test]
fn empty_buffer() {
    let a: SharedVector<u32> = SharedVector::new();
    assert!(!a.can_mutate());
    {
        let b: SharedVector<u32> = SharedVector::new();
        assert!(!b.can_mutate());
        assert!(a.ptr_eq(&b));
    }

    assert!(!a.can_mutate());

    let _: SharedVector<()> = SharedVector::new();
    let _: SharedVector<()> = SharedVector::new();
}

#[test]
#[rustfmt::skip]
fn grow() {
    let mut a = MutableVector::with_capacity(0);

    a.push(num(1));
    a.push(num(2));
    a.push(num(3));

    a.push_slice(&[num(4), num(5), num(6), num(7), num(8), num(9), num(10), num(12), num(12), num(13), num(14), num(15), num(16), num(17), num(18)]);

    assert_eq!(
        a.as_slice(),
        &[num(1), num(2), num(3), num(4), num(5), num(6), num(7), num(8), num(9), num(10), num(12), num(12), num(13), num(14), num(15), num(16), num(17), num(18)]
    );

    let mut b = SharedVector::new();
    b.push(num(1));
    b.push(num(2));
    b.push(num(3));

    assert_eq!(b.as_slice(), &[num(1), num(2), num(3)]);
}

#[test]
fn unique() {
    let mut a: UniqueVector<u32> = UniqueVector::with_capacity(64);
    a.push(1);
    a.push(2);
    a.push(3);
}