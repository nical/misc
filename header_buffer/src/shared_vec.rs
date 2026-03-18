use core::sync::atomic::{AtomicI32, Ordering::*};
use core::cell::UnsafeCell;
use core::{fmt, ops};
use core::marker::PhantomData;
use core::ptr::NonNull;

use crate::alloc::{Allocator, AllocError};

use crate::global::Global;
use crate::unmanaged::AllocInit;
use crate::util::{self, grow_amortized, is_zst};
use crate::{UnmanagedHeaderVector, header_vec::HeaderVector};

pub trait RefCount {
    unsafe fn add_ref(&self);
    unsafe fn release_ref(&self) -> bool;
    fn new(count: i32) -> Self;
    fn get(&self) -> i32;
}

pub struct DefaultRefCount(UnsafeCell<i32>);
pub struct AtomicRefCount(AtomicI32);

impl RefCount for AtomicRefCount {
    #[inline]
    unsafe fn add_ref(&self) {
        // Relaxed ordering is OK since the presence of the existing reference
        // prevents threads from deleting the buffer.
        self.0.fetch_add(1, Relaxed);
    }

    #[inline]
    unsafe fn release_ref(&self) -> bool {
        self.0.fetch_sub(1, Release) == 1
    }

    #[inline]
    fn new(val: i32) -> Self {
        AtomicRefCount(AtomicI32::new(val))
    }

    #[inline]
    fn get(&self) -> i32 {
        self.0.load(Relaxed)
    }
}

impl RefCount for DefaultRefCount {
    #[inline]
    unsafe fn add_ref(&self) {
        *self.0.get() += 1;
    }

    #[inline]
    unsafe fn release_ref(&self) -> bool {
        let count = self.0.get();
        *count -= 1;
        *count == 0
    }

    #[inline]
    fn new(val: i32) -> Self {
        DefaultRefCount(UnsafeCell::new(val))
    }

    #[inline]
    fn get(&self) -> i32 {
        unsafe { *self.0.get() }
    }
}

#[repr(C)]
pub struct Header<Data, R, A> {
    len: u32,
    cap: u32,
    ref_count: R,
    allocator: Option<A>,
    data: Data,
}

pub type SharedHeaderVector<H, T, A = Global> = RefCountedHeaderVector<H, T, DefaultRefCount, A>;
pub type SharedVector<T, A = Global> = SharedHeaderVector<(), T, A>;
pub type AtomicSharedHeaderVector<H, T, A = Global> = RefCountedHeaderVector<H, T, AtomicRefCount, A>;
pub type AtomicSharedVector<T, A = Global> = AtomicSharedHeaderVector<(), T, A>;

unsafe impl<H: Send + Sync, T: Send + Sync, A: Allocator + Send> Send for AtomicSharedHeaderVector<H, T, A> {}
unsafe impl<H: Send + Sync, T: Send + Sync, A: Allocator + Sync> Sync for AtomicSharedHeaderVector<H, T, A> {}
unsafe impl<H: Send, T: Send, A: Allocator + Send> Send for ShareableHeaderVector<H, T, A> {}
unsafe impl<H: Sync, T: Sync, A: Allocator + Sync> Sync for ShareableHeaderVector<H, T, A> {}

pub struct RefCountedHeaderVector<H, T, R: RefCount, A: Allocator> {
    data: NonNull<T>,
    _marker: PhantomData<(H, R, A)>,
}

impl<H, T, R: RefCount, A: Allocator> RefCountedHeaderVector<H, T, R, A> {
    pub fn try_with_capacity_in(header: H, cap: usize, allocator: A) -> Result<Self, AllocError> {
        let mut unmanaged = UnmanagedHeaderVector::try_with_capacity_in(
            Header {
                len: 0,
                cap: 0,
                ref_count: R::new(1),
                allocator: None,
                data: header,
            },
            cap,
            AllocInit::Uninit,
            &allocator
        )?;

        let real_cap = unmanaged.capacity();
        unmanaged.header_mut().cap = real_cap as u32;
        unmanaged.header_mut().allocator = Some(allocator);
        let (data, _, _ ) = unmanaged.into_raw_parts();

        Ok(RefCountedHeaderVector {
            data,
            _marker: PhantomData,
        })
    }

    #[inline]
    pub fn new_ref(&self) -> Self {
        unsafe {
            self.internal_header().ref_count.add_ref();
        }

        RefCountedHeaderVector {
            data: self.data,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.internal_header().len == 0
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.internal_header().len as usize
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.internal_header().cap as usize
    }

    #[inline]
    pub fn remaining_capacity(&self) -> usize {
        self.capacity() - self.len()
    }

    #[inline]
    pub fn header(&self) -> &H {
        &self.internal_header().data
    }

    #[inline]
    pub fn allocator(&self) -> &A {
        &self.internal_header().allocator.as_ref().unwrap()
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe { core::slice::from_raw_parts(self.data.as_ptr(), self.len()) }
    }

    #[inline]
    fn internal_header_ptr(&self) -> NonNull<Header<H, R, A>> {
        unsafe { util::get_header(self.data) }
    }

    #[inline]
    fn internal_header(&self) -> &Header<H, R, A> {
        unsafe {
            self.internal_header_ptr().as_ref()
        }
    }

    #[inline]
    pub fn reference_count(&self) -> i32 {
        self.internal_header().ref_count.get()
    }

    #[inline]
    pub fn is_unique(&self) -> bool {
        self.reference_count() == 1
    }

    #[inline]
    unsafe fn as_unmanaged(&self) -> UnmanagedHeaderVector<Header<H, DefaultRefCount, A>, T> {
        let len = self.internal_header().len as usize;
        let cap = self.internal_header().cap as usize;
        UnmanagedHeaderVector::from_raw_parts(self.data, len, cap)
    }

    #[inline]
    pub fn ptr_eq(&self, other: &Self) -> bool {
        self.data == other.data
    }

    pub fn try_into_unique(self) -> Result<ShareableHeaderVector<H, T, A>, Self> {
        if self.reference_count() != 1 {
            return Err(self);
        }

        unsafe {
            let Some(allocator) = self.internal_header_ptr().as_mut().allocator.take() else {
                return Err(self);
            };

            let unmanaged = self.as_unmanaged();

            // Don't run Drop!
            core::mem::forget(self);

            Ok(ShareableHeaderVector {
                inner: HeaderVector::from_unmanaged(unmanaged, allocator),
            })
        }
    }

    #[inline(always)]
    unsafe fn item_ptr(&self, index: usize) -> NonNull<T> {
        self.data.add(index)
    }

    #[inline(always)]
    unsafe fn write_item(&mut self, index: usize, val: T) {
        let dst = self.item_ptr(index);
        dst.write(val);
    }

    #[inline(always)]
    unsafe fn read_item(&self, index: usize) -> T {
        let dst = self.item_ptr(index);
        dst.read()
    }
}

impl<H: Clone, T: Clone, R: RefCount, A: Allocator + Clone> RefCountedHeaderVector<H, T, R, A> {
    /// Ensures this shared vector uniquely owns its storage, allocating a new copy
    /// If there are other references to it.
    ///
    /// In principle this is mostly useful internally to provide safe mutable methods
    /// as it does not observaly affect most of the shared vector behavior, however
    /// it has a few niche use cases, for example to provoke copies earlier for more
    /// predictable performance or in some unsafe endeavors.
    #[inline]
    pub fn ensure_unique(&mut self) {
        if !self.is_unique() {
            self.try_realloc_with_capacity(false, self.capacity()).unwrap()
        }
    }

    /// Appends an element to the back of a collection.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity exceeds `u32::MAX` bytes.
    pub fn push(&mut self, val: T) {
        self.reserve(1);
        unsafe {
            self.write_item(self.len(), val);
            self.internal_header_ptr().as_mut().len += 1;
        }
    }

    /// Removes the last element from the vector and returns it, or `None` if it is empty.
    pub fn pop(&mut self) -> Option<T> {
        let len = self.len();
        if len == 0 {
            return None;
        }

        self.ensure_unique();

        unsafe {
            self.internal_header_ptr().as_mut().len -= 1;
            Some(self.read_item(len - 1))
        }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut[T] {
        self.ensure_unique();
        unsafe { core::slice::from_raw_parts_mut(self.data.as_ptr(), self.len()) }
    }

    #[cold]
    fn try_realloc_additional(
        &mut self,
        is_unique: bool,
        enough_capacity: bool,
        additional: usize,
    ) -> Result<(), AllocError> {
        let new_cap = if enough_capacity {
            self.capacity()
        } else {
            grow_amortized(self.len(), additional).unwrap()
        };

        self.try_realloc_with_capacity(is_unique, new_cap)
    }

    #[cold]
    fn try_realloc_with_capacity(
        &mut self,
        is_unique: bool,
        new_cap: usize,
    ) -> Result<(), AllocError> {
        let mut new_vector = Self::try_with_capacity_in(
            self.header().clone(),
            new_cap,
            self.allocator().clone(),
        )?;

        if self.len() > 0 {
            unsafe {
                if is_unique {
                    // The buffer is not large enough, we'll have to create a new one, however we
                    // know that we have the only reference to it so we'll move the data with
                    // a simple memcpy instead of cloning it.

                    let len = self.len();
                    let old_ptr = self.data.as_ptr();
                    let new_ptr = new_vector.data.as_ptr();
                    self.internal_header_ptr().as_mut().len = 0;
                    core::ptr::copy_nonoverlapping(old_ptr, new_ptr, len);
                    new_vector.internal_header_ptr().as_mut().len = len as u32;
                } else {
                    new_vector.extend_from_slice_assuming_capacity(&self.as_slice());
                }
            }
        }

        *self = new_vector;
        Ok(())
    }

    /// Clones and appends the contents of the slice to the back of a collection.
    pub fn extend_from_slice(&mut self, slice: &[T]) {
        self.reserve(slice.len());
        unsafe {
            self.extend_from_slice_assuming_capacity(slice);
        }
    }


    unsafe fn extend_from_slice_assuming_capacity(&mut self, slice: &[T]) {
        debug_assert!(self.remaining_capacity() >= slice.len());
        self.as_unmanaged().extend_from_slice_assuming_capacity(slice);
        self.internal_header_ptr().as_mut().len += slice.len() as u32;
    }

    /// Ensures the vector can be safely mutated and has enough extra capacity to
    /// add `additional` more items.
    ///
    /// This will allocate new storage for the vector if the vector is not unique or if
    /// the capacity is not sufficient to accomodate `self.len() + additional` items.
    /// The vector may reserve more space to speculatively avoid frequent reallocations.
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        let is_unique = self.is_unique();
        let enough_capacity = self.remaining_capacity() >= additional;

        if !is_unique || !enough_capacity {
            // Hopefully the least common case.
            self.try_realloc_additional(is_unique, enough_capacity, additional)
                .unwrap();
        }
    }
}

impl<H, T, R: RefCount> RefCountedHeaderVector<H, T, R, Global> {
    pub fn try_with_capacity(header: H, cap: usize) -> Result<Self, AllocError> {
        Self::try_with_capacity_in(header, cap, Global)
    }

    pub fn with_capacity(header: H, cap: usize) -> Self {
        Self::try_with_capacity_in(header, cap, Global).unwrap()
    }
}


impl<H, T, R, A, I> core::ops::Index<I> for RefCountedHeaderVector<H, T, R, A>
where
    R: RefCount,
    A: Allocator,
    I: core::slice::SliceIndex<[T]>,
{
    type Output = <I as core::slice::SliceIndex<[T]>>::Output;
    fn index(&self, index: I) -> &Self::Output {
        self.as_slice().index(index)
    }
}

impl<H, T, R, A, I> core::ops::IndexMut<I> for RefCountedHeaderVector<H, T, R, A>
where
    H: Clone,
    T: Clone,
    R: RefCount,
    A: Allocator + Clone,
    I: core::slice::SliceIndex<[T]>,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        self.as_mut_slice().index_mut(index)
    }
}

impl<H, T, R, A> ops::Deref for RefCountedHeaderVector<H, T, R, A>
where
    R: RefCount,
    A: Allocator,
{
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<H, T, R, A> ops::DerefMut for RefCountedHeaderVector<H, T, R, A>
where
    H: Clone,
    T: Clone,
    R: RefCount,
    A: Allocator + Clone,
{
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<H, T, R: RefCount, A: Allocator> AsRef<[T]> for RefCountedHeaderVector<H, T, R, A> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<H, T, R, A> Drop for RefCountedHeaderVector<H, T, R, A>
where
    R: RefCount,
    A: Allocator,
{
    fn drop(&mut self) {
        unsafe {
            if self.internal_header().ref_count.release_ref() {
                // See the implementation of std Arc for the need to use this fence. Note that
                // we only need it for the atomic reference counted version but I don't expect
                // this to make a measurable difference.
                core::sync::atomic::fence(Acquire);

                if let Some(allocator) = self.internal_header_ptr().as_mut().allocator.take() {
                    self.as_unmanaged().deallocate_in(&allocator);
                }
            }
        }
    }
}

impl<H, T, R: RefCount, A: Allocator> Clone for RefCountedHeaderVector<H, T, R, A> {
    fn clone(&self) -> Self {
        self.new_ref()
    }
}

impl<H, T, R, A> fmt::Debug for RefCountedHeaderVector<H, T, R, A>
where
    H: fmt::Debug,
    T: fmt::Debug,
    R: RefCount,
    A: Allocator,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        if is_zst::<H>() {
            self.as_slice().fmt(f)
        } else {
            write!(f, "{:?}:{:?}", self.header(), self.as_slice())
        }
    }
}

impl<H, T, A> From<ShareableHeaderVector<H, T, A>> for SharedHeaderVector<H, T, A>
where
    A: Allocator,
{
    fn from(vec: ShareableHeaderVector<H, T, A>) -> Self {
        vec.into_shared()
    }
}

impl<H, T, A> From<ShareableHeaderVector<H, T, A>> for AtomicSharedHeaderVector<H, T, A>
where
    A: Allocator,
{
    fn from(vec: ShareableHeaderVector<H, T, A>) -> Self {
        vec.into_shared_atomic()
    }
}


// TODO: ShareableVector does not have a way to represent empty
// vectors without allocations (because from the point of view
// of the underlying HeaderBuffer it has a header, even if the
// header will only be used if the vector is converted into a
// reference counted one).

/// A uniquely owned vector that can be converted into a reference
/// couned one without reallocating.
pub struct ShareableHeaderVector<H, T, A: Allocator> {
    inner: HeaderVector<Header<H, DefaultRefCount, A>, T, A>,
}

impl<H, T> ShareableHeaderVector<H, T, Global> {
    pub fn try_with_capacity(header: H, cap: usize) -> Result<Self, AllocError> {
        Self::try_with_capacity_in(header, cap, Global)
    }

    pub fn with_capacity(header: H, cap: usize) -> Self {
        Self::try_with_capacity_in(header, cap, Global).unwrap()
    }
}

impl<H, T, A: Allocator> ShareableHeaderVector<H, T, A> {
    pub fn try_with_capacity_in(header: H, cap: usize, allocator: A) -> Result<Self, AllocError> {
        Ok(ShareableHeaderVector {
            inner: HeaderVector::try_with_capacity_in(
                Header {
                    len: 0,
                    cap: 0,
                    ref_count: DefaultRefCount::new(-1),
                    allocator: None,
                    data: header,
                },
                cap,
                allocator
            )?
        })
    }

    pub fn with_capacity_in(header: H, cap: usize, allocator: A) -> Self {
        Self::try_with_capacity_in(header, cap, allocator).unwrap()
    }

    pub fn into_shared(self) -> SharedHeaderVector<H, T, A> {
        let (mut unmanaged, allocator) = self.inner.into_unmanaged();
        let len = unmanaged.len() as u32;
        let cap = unmanaged.capacity() as u32;
        let header = unmanaged.header_mut();

        header.len = len;
        header.cap = cap;
        header.ref_count = DefaultRefCount::new(1);
        header.allocator = Some(allocator);

        let (data, _, _) = unmanaged.into_raw_parts();

        SharedHeaderVector {
            data,
            _marker: PhantomData
        }
    }

    pub fn into_shared_atomic(self) -> AtomicSharedHeaderVector<H, T, A> {
        let (unmanaged, allocator) = self.inner.into_unmanaged();

        // Cast the non-atomic reference count into an atomic one.
        let mut unmanaged: UnmanagedHeaderVector<Header<H, AtomicRefCount, A>, T> = unsafe {
            core::mem::transmute(unmanaged)
        };

        let len = unmanaged.len() as u32;
        let cap = unmanaged.capacity() as u32;
        let header = unmanaged.header_mut();

        header.len = len;
        header.cap = cap;
        header.ref_count = AtomicRefCount::new(1);
        header.allocator = Some(allocator);

        let (data, _, _) = unmanaged.into_raw_parts();

        AtomicSharedHeaderVector {
            data,
            _marker: PhantomData
        }
    }

    #[inline]
    pub fn header(&self) -> &H {
        &self.inner.header().data
    }

    #[inline]
    pub fn header_mut(&mut self) -> &mut H {
        &mut self.inner.header_mut().data
    }

    #[inline(always)]
    pub fn push(&mut self, val: T) {
        self.inner.push(val);
    }

    #[inline]
    pub fn extend_from_slice(&mut self, slice: &[T])
    where T: Clone
    {
        self.inner.extend_from_slice(slice);
    }

    crate::impl_vector_methods!();
}

impl<H, T, A, I> core::ops::Index<I> for ShareableHeaderVector<H, T, A>
where
    A: Allocator,
    I: core::slice::SliceIndex<[T]>,
{
    type Output = <I as core::slice::SliceIndex<[T]>>::Output;
    fn index(&self, index: I) -> &Self::Output {
        self.as_slice().index(index)
    }
}

impl<H, T, A, I> core::ops::IndexMut<I> for ShareableHeaderVector<H, T, A>
where
    A: Allocator,
    I: core::slice::SliceIndex<[T]>,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        self.as_mut_slice().index_mut(index)
    }
}

impl<H, T, A> ops::Deref for ShareableHeaderVector<H, T, A>
where
    A: Allocator,
{
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<H, T, A> ops::DerefMut for ShareableHeaderVector<H, T, A>
where
    A: Allocator,
{
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<H, T, A: Allocator> AsRef<[T]> for ShareableHeaderVector<H, T, A> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<H, T, A: Allocator> Clone for ShareableHeaderVector<H, T, A> {
    fn clone(&self) -> Self {
        todo!()
    }
}

impl<H, T, A> fmt::Debug for ShareableHeaderVector<H, T, A>
where
    H: fmt::Debug,
    T: fmt::Debug,
    A: Allocator,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        if is_zst::<H>() {
            self.as_slice().fmt(f)
        } else {
            write!(f, "{:?}:{:?}", self.header(), self.as_slice())
        }
    }
}


#[test]
#[rustfmt::skip]
fn grow() {
    fn num(val: i32) -> Box<i32> { Box::new(val) }

    let mut a = ShareableHeaderVector::with_capacity(num(-1), 0);

    a.push(num(1));
    a.push(num(2));
    a.push(num(3));

    a.extend_from_slice(&[num(4), num(5), num(6), num(7), num(8), num(9), num(10), num(12), num(12), num(13), num(14), num(15), num(16), num(17), num(18)]);

    assert_eq!(
        a.as_slice(),
        &[num(1), num(2), num(3), num(4), num(5), num(6), num(7), num(8), num(9), num(10), num(12), num(12), num(13), num(14), num(15), num(16), num(17), num(18)]
    );

    let mut b = SharedVector::with_capacity((), 0);
    b.push(num(1));
    b.push(num(2));
    b.push(num(3));

    assert_eq!(b.as_slice(), &[num(1), num(2), num(3)]);

    let mut b = AtomicSharedVector::with_capacity((), 0);
    b.push(num(1));
    b.push(num(2));
    b.push(num(3));

    assert_eq!(b.as_slice(), &[num(1), num(2), num(3)]);
}

#[test]
fn ensure_unique_empty() {
    let mut v: SharedVector<u32> = SharedVector::with_capacity((), 0);
    v.ensure_unique();
}
