use core::fmt::Debug;
use core::mem;
use core::ops::{Deref, DerefMut, Index, IndexMut};

use crate::allocator::{AllocError, Allocator};
use crate::global::Global;
use crate::unmanaged::UnmanagedVector;

pub struct Vector<T, A: Allocator = Global> {
    inner: UnmanagedVector<T>,
    allocator: A,
}

impl<T> Vector<T, Global> {
    /// Constructs a new, empty vector.
    ///
    /// The vector will not allocate until elements are pushed onto it.
    #[inline(always)]
    pub fn new() -> Self {
        Vector::new_in(Global)
    }

    /// Constructs a new, empty Vec<T> with at least the specified capacity.
    #[inline(always)]
    pub fn with_capacity(cap: usize) -> Vector<T, Global> {
        Self::with_capacity_in(cap, Global)
    }

    /// Constructs a vector from a slice.
    #[inline(always)]
    pub fn from_slice(slice: &[T]) -> Self
    where
        T: Clone,
    {
        let mut v = Self::with_capacity(slice.len());
        v.extend_from_slice(slice);

        v
    }
}

impl<T, A: Allocator> Vector<T, A> {
    #[inline(always)]
    /// Constructs a new, empty vector using the provided allocator.
    ///
    /// The vector will not allocate until elements are pushed onto it.
    pub fn new_in(allocator: A) -> Self {
        Vector {
            inner: UnmanagedVector::new(),
            allocator,
        }
    }

    /// Constructs a new, empty vector with at least the specified capacity,
    /// usig the provided allocator.
    pub fn with_capacity_in(cap: usize, allocator: A) -> Self {
        Self::try_with_capacity_in(cap, allocator).unwrap()
    }

    /// Constructs a new, empty vector with at least the specified capacity,
    /// using the provided allocator.
    pub fn try_with_capacity_in(cap: usize, allocator: A) -> Result<Self, AllocError> {
        let inner = UnmanagedVector::try_with_capacity_in((), cap, crate::unmanaged::AllocInit::Uninit, &allocator)?;
        Ok(Vector { inner, allocator })
    }

    /// Constructs a vector from the elements of a slice, using the provided
    /// allocator.
    pub fn from_slice_in(slice: &[T], allocator: A) -> Self
    where
        T: Clone,
    {
        let mut v = Self::with_capacity_in(slice.len(), allocator);
        v.extend_from_slice(slice);

        v
    }

    pub fn from_elem_in(elem: T, n: usize, allocator: A) -> Self
    where
        T: Clone,
    {
        let mut v = Self::with_capacity_in(n, allocator);

        // TODO: std::Vec has optimizations here, in particular for zeroes.
        // Unsure how to do them without specialization.
        for _ in 0..n {
            v.push(elem.clone())
        }

        v
    }

    crate::impl_vector_methods!();

    #[inline(always)]
    pub fn into_unmanaged(self) -> (UnmanagedVector<T>, A) {
        unsafe {
            let inner = core::ptr::read(&self.inner);
            let allocator = core::ptr::read(&self.allocator);
            mem::forget(self);

            (inner, allocator)
        }
    }

    #[inline(always)]
    pub fn from_unmanaged(unmanaged: UnmanagedVector<T>, allocator: A) -> Self {
        Vector {
            inner: unmanaged,
            allocator,
        }
    }

    pub fn clone_in<A2: Allocator>(&self, allocator: A2) -> Vector<T, A2>
    where
        T: Clone,
    {
        Vector {
            inner: self.inner.clone_in(&allocator, self.capacity()),
            allocator,
        }
    }
}

impl<T, A: Allocator> Drop for Vector<T, A> {
    fn drop(&mut self) {
        unsafe { self.inner.deallocate_in(&self.allocator) }
        println!("Vector::Drop");
    }
}

impl<T: Clone, A: Allocator + Clone> Clone for Vector<T, A> {
    fn clone(&self) -> Self {
        Vector {
            inner: self.inner.clone_in(&self.allocator, self.capacity()),
            allocator: self.allocator.clone(),
        }
    }
}

impl<T: PartialEq<T>, A1: Allocator, A2: Allocator> PartialEq<Vector<T, A2>> for Vector<T, A1> {
    fn eq(&self, other: &Vector<T, A2>) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: PartialEq<T>, A: Allocator> PartialEq<&[T]> for Vector<T, A> {
    fn eq(&self, other: &&[T]) -> bool {
        self.as_slice() == *other
    }
}

impl<T, A: Allocator> AsRef<[T]> for Vector<T, A> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T, A: Allocator> AsMut<[T]> for Vector<T, A> {
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T> Default for Vector<T, Global> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, T, A: Allocator> IntoIterator for &'a Vector<T, A> {
    type Item = &'a T;
    type IntoIter = core::slice::Iter<'a, T>;
    fn into_iter(self) -> core::slice::Iter<'a, T> {
        self.as_slice().iter()
    }
}

impl<'a, T, A: Allocator> IntoIterator for &'a mut Vector<T, A> {
    type Item = &'a mut T;
    type IntoIter = core::slice::IterMut<'a, T>;
    fn into_iter(self) -> core::slice::IterMut<'a, T> {
        self.as_mut_slice().iter_mut()
    }
}

impl<T, A: Allocator, I> Index<I> for Vector<T, A>
where
    I: core::slice::SliceIndex<[T]>,
{
    type Output = <I as core::slice::SliceIndex<[T]>>::Output;
    fn index(&self, index: I) -> &Self::Output {
        self.as_slice().index(index)
    }
}

impl<T, A: Allocator, I> IndexMut<I> for Vector<T, A>
where
    I: core::slice::SliceIndex<[T]>,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        self.as_mut_slice().index_mut(index)
    }
}

impl<T, A: Allocator> Deref for Vector<T, A> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T, A: Allocator> DerefMut for Vector<T, A> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T: Debug, A: Allocator> Debug for Vector<T, A> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> Result<(), core::fmt::Error> {
        self.as_slice().fmt(f)
    }
}

#[macro_export]
macro_rules! vector {
    (@one@ $x:expr) => (1usize);
    ($elem:expr; $n:expr) => ({
        $crate::Vector::from_elem($elem, $n)
    });
    ($($x:expr),*$(,)*) => ({
        let count = 0usize $(+ $crate::vector!(@one@ $x))*;
        let mut vec = $crate::Vector::with_capacity(count);
        $(vec.push($x);)*
        vec
    });
    ([$x:expr;$n:expr] in $allocator:expr) => ({
        $crate::Vector::from_elem_in($x, $n, $allocator)
    });
    ([$($x:expr),*$(,)*] in $allocator:expr) => ({
        let count = 0usize $(+ $crate::vector!(@one@ $x))*;
        let mut vec = $crate::Vector::with_capacity_in(count, $allocator);
        $(vec.push($x);)*
        vec
    });
}

#[test]
fn basic_unique() {
    fn num(val: u32) -> Box<u32> {
        Box::new(val)
    }

    let mut a = Vector::with_capacity(256);

    a.push(num(0));
    a.push(num(1));
    a.push(num(2));

    assert_eq!(a.len(), 3);

    assert_eq!(a.as_slice(), &[num(0), num(1), num(2)]);

    let b = Vector::from_slice(&[num(0), num(1), num(2), num(3), num(4)]);

    assert_eq!(b.as_slice(), &[num(0), num(1), num(2), num(3), num(4)]);

    let mut d = Vector::with_capacity(64);
    d.extend_from_slice(&[num(0), num(1), num(2)]);
    d.extend_from_slice(&[]);
    d.extend_from_slice(&[num(3), num(4)]);

    assert_eq!(d.as_slice(), &[num(0), num(1), num(2), num(3), num(4)]);
}

#[test]
fn shrink() {
    let mut v: Vector<u32> = Vector::with_capacity(32);
    v.shrink_to(8);
}

#[test]
fn zst() {
    let mut v = Vector::new();
    v.push(());
    v.push(());
    v.push(());
    v.push(());

    assert_eq!(v.len(), 4);
}

#[test]
fn dyn_allocator() {
    let allocator: &dyn Allocator = &Global;
    let mut v = crate::vector!([1u32, 2, 3] in allocator);

    v.push(4);

    assert_eq!(&v[..], &[1, 2, 3, 4]);
}

#[test]
fn borrowed_dyn_alloc() {
    struct DataStructure<'a> {
        data: Vector<u32, &'a dyn Allocator>,
    }

    impl DataStructure<'static> {
        fn new() -> DataStructure<'static> {
            DataStructure {
                data: Vector::new_in(&Global as &'static dyn Allocator),
            }
        }
    }

    impl<'a> DataStructure<'a> {
        fn new_in(allocator: &'a dyn Allocator) -> DataStructure<'a> {
            DataStructure {
                data: Vector::new_in(allocator),
            }
        }

        fn push(&mut self, val: u32) {
            self.data.push(val);
        }
    }

    let mut ds1 = DataStructure::new();
    ds1.push(1);

    let alloc = Global;
    let mut ds2 = DataStructure::new_in(&alloc);
    ds2.push(2);
}
