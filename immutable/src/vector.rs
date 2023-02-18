use std::mem;
use std::ops::{Index, IndexMut};

use crate::raw::{self, AllocError, BufferSize, HeaderBuffer, BufferHeader};

pub trait ReferenceCount {
    type Header: BufferHeader;
}

pub struct DefaultRefCount;
pub struct AtomicRefCount;

impl ReferenceCount for DefaultRefCount { type Header = raw::Header; }
impl ReferenceCount for AtomicRefCount { type Header = raw::AtomicHeader; }

pub type AtomicSharedVector<T> = RefCountedVector<T, AtomicRefCount>;
pub type SharedVector<T> = RefCountedVector<T, DefaultRefCount>;

/// A heap allocated, reference counted, immutable contiguous buffer containing elements of type `T`.
///
/// Similar in principle to `Arc<[T]>`. It can be converted into a `UniqueVector<T>` for
/// free if there is only a single reference to the RefCountedVector alive.
pub struct RefCountedVector<T, R: ReferenceCount = DefaultRefCount> {
    inner: HeaderBuffer<R::Header, T>,
}

/// A heap allocated, mutable contiguous buffer containing elements of type `T`.
///
/// Similar in principle to a `Box<ArrayVec<T, Size>>` where size would be set at runtime.
/// It can be converted for free into an immutable `SharedVector<T>` or `AtomicSharedVector<T>`.
///
/// Unique and shared vectors have similar functionality. UniqueVector's main advantage is that
/// it does not need to check unicity at runtime since it is guaranteed at the type level.
pub struct UniqueVector<T> {
    inner: HeaderBuffer<raw::Header, T>,
}

impl<T, R: ReferenceCount> RefCountedVector<T, R> {
    /// Creates an empty shared buffer without allocating memory.
    #[inline]
    pub fn new() -> Self {
        RefCountedVector {
            inner: HeaderBuffer::new_empty().unwrap(),
        }
    }

    #[inline]
    pub fn with_capacity(cap: usize) -> Self {
        RefCountedVector {
            inner: HeaderBuffer::try_with_capacity(cap).unwrap(),
        }
    }

    #[inline]
    pub fn from_slice(data: &[T]) -> Self
    where
        T: Clone,
    {
        RefCountedVector {
            inner: HeaderBuffer::try_from_slice(data, None).unwrap(),
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
        self.inner.len() as usize
    }

    /// Returns the total number of elements the vector can hold without reallocating.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.inner.capacity() as usize
    }

    /// Returns number of elements that can be added without reallocating.
    #[inline]
    pub fn remaining_capacity(&self) -> usize {
        self.inner.remaining_capacity() as usize
    }

    /// Creates a new reference without allocating.
    ///
    /// Equivalent to `Clone::clone`.
    #[inline]
    pub fn new_ref(&self) -> Self {
        RefCountedVector {
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
        RefCountedVector {
            inner: self.inner.try_clone_buffer(None).unwrap(),
        }
    }

    /// Allocates a duplicate of this buffer (infallible).
    pub fn copy_buffer(&self) -> Self
    where
        T: Copy,
    {
        RefCountedVector {
            inner: self.inner.try_copy_buffer(None).unwrap(),
        }
    }

    /// Returns true if this is the only existing handle to the buffer.
    #[inline]
    pub fn is_unique(&self) -> bool {
        self.inner.is_unique()
    }

    /// Converts this RefCountedVector into an immutable one, allocating a new one if there are other references.
    #[inline]
    pub fn into_unique(mut self) -> UniqueVector<T>
    where
        T: Clone,
    {
        self.ensure_mutable();
        UniqueVector {
            inner: unsafe { mem::transmute(self.inner) },
        }
    }

    /// Converts this shared buffer into a mutable one if it is the only reference to its data.
    ///
    /// Never allocates.
    #[inline]
    pub fn try_into_mut(self) -> Option<Self> {
        if self.is_unique() {
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
        self.reserve(1);
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
        self.reserve(data.len());
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
        self.reserve(max.unwrap_or(min));
        unsafe {
            self.inner.try_extend(&mut iter).unwrap();
        }
    }

    pub fn clear(&mut self) {
        if self.is_unique() {
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
        if !self.is_unique() {
            self.inner = self.inner.try_clone_buffer(None).unwrap();
        }
    }

    /// Returns a buffer that can be safely mutated and has enough extra capacity to
    /// add `additional` more items.
    #[inline]
    pub fn reserve(&mut self, additional: usize)
    where
        T: Clone,
    {
        let is_unique = self.is_unique();
        let off_capacity = self.remaining_capacity() <= additional;

        if !is_unique || off_capacity {
            // Hopefully the least common case.
            self.realloc(is_unique, additional);
        }
    }

    #[cold]
    fn realloc(&mut self, is_unique: bool, additional: usize)
    where
        T: Clone,
    {
        let new_cap = grow_amortized(self.len(), additional);

        if is_unique {
            // The buffer is not large enough, we'll have to create a new one, however we
            // know that we have the only reference to it so we'll move the data with
            // a simple memcpy instead of cloning it.
            unsafe {
                let mut dst = Self::with_capacity(new_cap);
                let len = self.len();
                if len > 0 {
                    std::ptr::copy_nonoverlapping(
                        self.inner.data_ptr(),
                        dst.inner.data_ptr(),
                        len as usize,
                    );
                    dst.inner.set_len(len as BufferSize);
                    self.inner.set_len(0);
                }

                self.inner = dst.inner;
            }
        }

        // The slowest path, we pay for both the new allocation and the need to clone
        // each item one by one.
        let new_cap = grow_amortized(self.len(), additional);
        self.inner = HeaderBuffer::try_from_slice(self.as_slice(), Some(new_cap)).unwrap();
    }

    /// Returns the concatenation of two vectors.
    pub fn concatenate(mut self, mut other: Self) -> Self
    where
        T: Clone,
    {
        self.reserve(other.len());

        unsafe {
            if other.is_unique() {
                // Fast path: memcpy
                other.inner.move_data(&mut self.inner);
            } else {
                // Slow path, clone each item.
                self.inner.try_push_slice(other.as_slice()).unwrap();
            }
        }

        self
    }

    pub fn ref_count(&self) -> i32 {
        self.inner.ref_count()
    }
}

unsafe impl<T: Sync, R: ReferenceCount> Send for RefCountedVector<T, R> {}

impl<T, R: ReferenceCount> Clone for RefCountedVector<T, R> {
    fn clone(&self) -> Self {
        self.new_ref()
    }
}

impl<T: PartialEq<T>, R: ReferenceCount> PartialEq<RefCountedVector<T, R>> for RefCountedVector<T, R> {
    fn eq(&self, other: &Self) -> bool {
        self.ptr_eq(other) || self.as_slice() == other.as_slice()
    }
}

impl<T: PartialEq<T>, R: ReferenceCount> PartialEq<&[T]> for RefCountedVector<T, R> {
    fn eq(&self, other: &&[T]) -> bool {
        self.as_slice() == *other
    }
}

impl<T, R: ReferenceCount> AsRef<[T]> for RefCountedVector<T, R> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T, R: ReferenceCount> Default for RefCountedVector<T, R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, T, R: ReferenceCount> IntoIterator for &'a RefCountedVector<T, R> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> std::slice::Iter<'a, T> {
        self.as_slice().iter()
    }
}

impl<T, R: ReferenceCount, I> Index<I> for RefCountedVector<T, R>
where
    I: std::slice::SliceIndex<[T]>,
{
    type Output = <I as std::slice::SliceIndex<[T]>>::Output;
    fn index(&self, index: I) -> &Self::Output {
        self.as_slice().index(index)
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

    /// Make this vector immutable.
    ///
    /// This operation is cheap, the underlying storage does not not need
    /// to be reallocated.
    #[inline]
    pub fn into_shared(self) -> SharedVector<T> {
        SharedVector { inner: unsafe { self.inner.cast_header() } }
    }

    /// Make this vector immutable.
    ///
    /// This operation is cheap, the underlying storage does not not need
    /// to be reallocated.
    #[inline]
    pub fn into_shared_atomic(self) -> AtomicSharedVector<T> {
        AtomicSharedVector { inner: unsafe { self.inner.cast_header() } }
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
        self.reserve(1);

        unsafe {
            self.inner.push(val);
        }
    }

    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        unsafe { self.inner.pop() }
    }

    #[inline]
    pub fn push_slice(&mut self, data: &[T])
    where
        T: Clone,
    {
        self.reserve(data.len());

        unsafe {
            self.inner.try_push_slice(data).unwrap();
        }
    }

    pub fn extend(&mut self, data: impl IntoIterator<Item = T>) {
        let mut iter = data.into_iter();
        let (min, max) = iter.size_hint();
        self.reserve(max.unwrap_or(min));
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

    /// Allocate a clone of this buffer.
    pub fn clone_buffer(&self) -> Self
    where
        T: Clone,
    {
        UniqueVector {
            inner: self.inner.try_clone_buffer(None).unwrap(),
        }
    }

    /// Allocate a clone of this buffer with a different capacity
    ///
    /// The capacity must be at least as large as the buffer's length.
    pub fn clone_buffer_with_capacity(&self, cap: BufferSize) -> Self
    where
        T: Clone,
    {
        UniqueVector {
            inner: self.inner.try_clone_buffer(Some(cap)).unwrap(),
        }
    }

    fn try_realloc(&mut self, additional: usize) -> Result<(), AllocError> {
        let new_cap = grow_amortized(self.len(), additional);
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
    fn realloc(&mut self, additional: usize) {
        self.try_realloc(additional).unwrap();
    }

    #[inline]
    fn reserve(&mut self, additional: usize) {
        if self.remaining_capacity() < additional {
            self.realloc(additional);
        }
    }
}

impl<T: Clone> Clone for UniqueVector<T> {
    fn clone(&self) -> Self {
        self.clone_buffer()
    }
}

impl<T: PartialEq<T>> PartialEq<UniqueVector<T>> for UniqueVector<T> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: PartialEq<T>> PartialEq<&[T]> for UniqueVector<T> {
    fn eq(&self, other: &&[T]) -> bool {
        self.as_slice() == *other
    }
}

impl<T> AsRef<[T]> for UniqueVector<T> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> AsMut<[T]> for UniqueVector<T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T> Default for UniqueVector<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, T> IntoIterator for &'a UniqueVector<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> std::slice::Iter<'a, T> {
        self.as_slice().iter()
    }
}

impl<T, I> Index<I> for UniqueVector<T>
where
    I: std::slice::SliceIndex<[T]>,
{
    type Output = <I as std::slice::SliceIndex<[T]>>::Output;
    fn index(&self, index: I) -> &Self::Output {
        self.as_slice().index(index)
    }
}

impl<T, I> IndexMut<I> for UniqueVector<T>
where
    I: std::slice::SliceIndex<[T]>,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        self.as_mut_slice().index_mut(index)
    }
}

fn grow_amortized(len: usize, additional: usize) -> usize {
    let required = len.saturating_add(additional);
    let cap = len.saturating_add(len).max(required).max(8);

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
fn basic1() {
    let mut a = UniqueVector::with_capacity(256);

    a.push(num(0));
    a.push(num(1));
    a.push(num(2));

    let a = a.into_shared();

    assert_eq!(a.len(), 3);

    assert_eq!(a.as_slice(), &[num(0), num(1), num(2)]);

    assert!(a.is_unique());

    let b = UniqueVector::from_slice(&[num(0), num(1), num(2), num(3), num(4)]);

    assert_eq!(b.as_slice(), &[num(0), num(1), num(2), num(3), num(4)]);

    let c = a.clone_buffer();
    assert!(!c.ptr_eq(&a));

    let a2 = a.new_ref();
    assert!(a2.ptr_eq(&a));
    assert!(!a.is_unique());
    assert!(!a2.is_unique());

    mem::drop(a2);

    assert!(a.is_unique());

    let _ = c.clone_buffer();
    let _ = b.clone_buffer();

    let mut d = UniqueVector::with_capacity(64);
    d.push_slice(&[num(0), num(1), num(2)]);
    d.push_slice(&[]);
    d.push_slice(&[num(3), num(4)]);

    assert_eq!(d.as_slice(), &[num(0), num(1), num(2), num(3), num(4)]);
}

#[test]
fn basic2() {
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
    // TODO: The behavior is different for SharedVector because it does not use a global header.
    let a: AtomicSharedVector<u32> = AtomicSharedVector::new();
    assert!(!a.is_unique());
    {
        let b: AtomicSharedVector<u32> = AtomicSharedVector::new();
        assert!(!b.is_unique());
        assert!(a.ptr_eq(&b));
    }

    assert!(!a.is_unique());

    let _: SharedVector<()> = SharedVector::new();
    let _: SharedVector<()> = SharedVector::new();
}

#[test]
#[rustfmt::skip]
fn grow() {
    let mut a = UniqueVector::with_capacity(0);

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
