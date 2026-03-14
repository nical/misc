use core::fmt::Debug;
use core::mem;
use core::ops::{Deref, DerefMut, Index, IndexMut};

use crate::allocator::{AllocError, Allocator};
use crate::global::Global;
use crate::unmanaged::{AllocInit, UnmanagedHeaderVector};

pub struct HeaderVector<H, T, A: Allocator = Global> {
    inner: UnmanagedHeaderVector<H, T>,
    allocator: A,
}

impl<H, T> HeaderVector<H, T, Global> {
    /// Constructs a new, empty Vec<T> with at least the specified capacity.
    #[inline(always)]
    pub fn with_capacity(header: H, cap: usize) -> HeaderVector<H, T, Global> {
        Self::with_capacity_in(header, cap, Global)
    }
}

impl<H, T, A: Allocator> HeaderVector<H, T, A> {
    #[inline(always)]
    /// Constructs a new, empty vector with at least the specified capacity,
    /// usig the provided allocator.
    pub fn with_capacity_in(header: H, cap: usize, allocator: A) -> Self {
        HeaderVector {
            inner: UnmanagedHeaderVector::with_capacity_in(header, cap, AllocInit::Uninit, &allocator),
            allocator,
        }
    }

    /// Constructs a new, empty vector with at least the specified capacity,
    /// using the provided allocator.
    pub fn try_with_capacity_in(header: H, cap: usize, allocator: A) -> Result<Self, AllocError> {
        let inner = UnmanagedHeaderVector::try_with_capacity_in(header, cap, AllocInit::Uninit, &allocator)?;
        Ok(HeaderVector { inner, allocator })
    }

    #[inline]
    pub fn header(&self) -> &H {
        self.inner.header()
    }

    #[inline]
    pub fn header_mut(&mut self) -> &mut H {
        self.inner.header_mut()
    }

    crate::impl_vector_methods!();
    crate::impl_vector_methods_with_allocator!();

    #[inline(always)]
    pub fn into_unmanaged(self) -> (UnmanagedHeaderVector<H, T>, A) {
        unsafe {
            let inner = core::ptr::read(&self.inner);
            let allocator = core::ptr::read(&self.allocator);
            mem::forget(self);

            (inner, allocator)
        }
    }

    #[inline(always)]
    pub fn from_unmanaged(unmanaged: UnmanagedHeaderVector<H, T>, allocator: A) -> Self {
        HeaderVector {
            inner: unmanaged,
            allocator,
        }
    }

    pub fn clone_in<A2: Allocator>(&self, allocator: A2) -> HeaderVector<H, T, A2>
    where
        H: Clone,
        T: Clone,
    {
        HeaderVector {
            inner: self.inner.clone_in(&allocator, self.capacity()),
            allocator,
        }
    }

    pub fn try_reallocate_in<A2: Allocator>(mut self, new_cap: usize, new_allocator: A2) -> Result<HeaderVector<H, T, A2>, AllocError> {
        unsafe {
            self.inner.try_realloc_in_new_allocator(new_cap, &self.allocator, &new_allocator)?;
        }

        let inner = self.inner;
        mem::forget(self);

        Ok(HeaderVector {
            inner,
            allocator: new_allocator,
        })
    }
}

impl<H, T, A: Allocator> Drop for HeaderVector<H, T, A> {
    fn drop(&mut self) {
        unsafe { self.inner.deallocate_in(&self.allocator) }
    }
}

impl<T: Clone, A: Allocator + Clone> Clone for HeaderVector<T, A> {
    fn clone(&self) -> Self {
        HeaderVector {
            inner: self.inner.clone_in(&self.allocator, self.capacity()),
            allocator: self.allocator.clone(),
        }
    }
}

impl<H: PartialEq<H>, T: PartialEq<T>, A1: Allocator, A2: Allocator> PartialEq<HeaderVector<H, T, A2>> for HeaderVector<H, T, A1> {
    fn eq(&self, other: &HeaderVector<H, T, A2>) -> bool {
        self.header() == other.header() && self.as_slice() == other.as_slice()
    }
}

impl<H, T, A: Allocator> AsRef<[T]> for HeaderVector<H, T, A> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<H, T, A: Allocator> AsMut<[T]> for HeaderVector<H, T, A> {
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<'a, H, T, A: Allocator> IntoIterator for &'a HeaderVector<H, T, A> {
    type Item = &'a T;
    type IntoIter = core::slice::Iter<'a, T>;
    fn into_iter(self) -> core::slice::Iter<'a, T> {
        self.as_slice().iter()
    }
}

impl<'a, H, T, A: Allocator> IntoIterator for &'a mut HeaderVector<H, T, A> {
    type Item = &'a mut T;
    type IntoIter = core::slice::IterMut<'a, T>;
    fn into_iter(self) -> core::slice::IterMut<'a, T> {
        self.as_mut_slice().iter_mut()
    }
}

impl<H, T, A: Allocator, I> Index<I> for HeaderVector<H, T, A>
where
    I: core::slice::SliceIndex<[T]>,
{
    type Output = <I as core::slice::SliceIndex<[T]>>::Output;
    fn index(&self, index: I) -> &Self::Output {
        self.as_slice().index(index)
    }
}

impl<H, T, A: Allocator, I> IndexMut<I> for HeaderVector<H, T, A>
where
    I: core::slice::SliceIndex<[T]>,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        self.as_mut_slice().index_mut(index)
    }
}

impl<H, T, A: Allocator> Deref for HeaderVector<H, T, A> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<H, T, A: Allocator> DerefMut for HeaderVector<H, T, A> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<H: Debug, T: Debug, A: Allocator> Debug for HeaderVector<H, T, A> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> Result<(), core::fmt::Error> {
        self.inner.fmt(f)
    }
}

#[test]
fn basic_unique() {
    fn num(val: i32) -> Box<i32> {
        Box::new(val)
    }

    let mut a = HeaderVector::with_capacity(num(-1), 256);

    a.push(num(0));
    a.push(num(1));
    a.push(num(2));

    assert_eq!(a.len(), 3);

    assert_eq!(a.as_slice(), &[num(0), num(1), num(2)]);

    let mut d = HeaderVector::with_capacity(num(-1), 64);
    d.extend_from_slice(&[num(0), num(1), num(2)]);
    d.extend_from_slice(&[]);
    d.extend_from_slice(&[num(3), num(4)]);

    assert_eq!(d.as_slice(), &[num(0), num(1), num(2), num(3), num(4)]);
}
