use std::ops::{Index, IndexMut};
use std::sync::atomic::{AtomicI32, Ordering};
use std::marker::PhantomData;
use std::alloc::{self, Layout};
use std::ptr::{self, NonNull};
use std::mem;

type BufferSize = u32;

struct RawBuffer<T> {
    ptr: NonNull<Header>,
    _marker: PhantomData<T>,
}

/// A heap allocated, reference counted, immutable contiguous buffer containing elements of type `T`.
///
/// Similar in principle to `Arc<[T]>`. It can be converted into a `MutableChumk<T>` for
/// free if there is only a single reference to the SharedBuffer alive.
pub struct SharedBuffer<T> {
    inner: RawBuffer<T>,
}

/// A heap allocated, mutable contiguous buffer containing elements of type `T`.
///
/// Similar in principle to a `Box<ArrayVec<T, Size>>` where size would be set at runtime.
/// It can be converted for free into an immutable `SharedBuffer<T>`.
pub struct MutableBuffer<T> {
    inner: RawBuffer<T>,
}

/// Some information stored at the beginning of every buffer.
struct Header {
    ref_count: AtomicI32,
    cap: BufferSize,
    len: BufferSize,
    _pad: u32,
}

// A global empty header so that we can create empty shared buffers with allocating memory.
static GLOBAL_EMPTY_BUFFER: Header = Header {
    // The initial reference count is 1 so that it never gets to zero.
    // this is important in order to ensure that the global empty buffer
    // is never considered mutable (any live handle will contribute at least one reference
    // meaning the ref_count should always be observably more than 1 if a RawBuffer points to it.)
    ref_count: AtomicI32::new(1),
    cap: 0,
    len: 0,
    _pad: 0
};

impl<T> RawBuffer<T> {
    fn new_empty() -> Self {
        GLOBAL_EMPTY_BUFFER.ref_count.fetch_add(1, Ordering::SeqCst);

        let global = unsafe { mem::transmute(&GLOBAL_EMPTY_BUFFER) };

        let buffer = RawBuffer {
            ptr: NonNull::new(global).unwrap(),
            _marker: PhantomData,
        };

        buffer
    }

    fn try_with_capacity(cap: usize) -> Result<RawBuffer<T>, AllocError> {
        unsafe {
            if cap > BufferSize::MAX as usize {
                return Err(AllocError::CapacityOverflow);
            }

            let layout = buffer_layout::<T>(cap)?;
            let alloc: NonNull<Header> = NonNull::new(alloc::alloc(layout))
                .ok_or(AllocError::Allocator { layout })?
                .cast();
        
            ptr::write(alloc.as_ptr(), Header {
                ref_count: AtomicI32::new(1),
                cap: cap as BufferSize,
                len: 0,
                _pad: 0,
            });

            Ok(RawBuffer {
                ptr: alloc,
                _marker: PhantomData
            })
        }
    }

    fn try_from_slice(data: &[T], cap: Option<usize>) -> Result<RawBuffer<T>, AllocError>  where T: Clone {
        let len = data.len();
        let cap = cap.map(|cap| cap.max(len)).unwrap_or(len);

        if cap > BufferSize::MAX as usize {
            return Err(AllocError::CapacityOverflow);
        }

        let mut buffer = Self::try_with_capacity(cap)?;

        unsafe {
            buffer.ptr.as_mut().len = len as BufferSize;

            let mut ptr = buffer.data_ptr();

            for item in data {
                ptr::write(ptr, item.clone());
                ptr = ptr.add(1)
            }
        }
        
        Ok(buffer)
    }

    #[inline]
    fn len(&self) -> usize {
        self.header().len as usize
    }

    #[inline]
    fn capacity(&self) -> usize {
        self.header().cap as usize
    }

    #[inline]
    fn remaining_capacity(&self) -> usize {
        self.capacity() - self.len()
    }

    /// Allocates a duplicate of this SharedBuffer with a different capacity (infallible).
    ///
    /// Resturns an error if the capacity is lower than this SharedBuffer's length.
    fn clone_buffer_with_capacity(&self, cap: BufferSize) -> Self where T: Clone {
        unsafe { clone_buffer::<T>(self.ptr, Some(cap)).unwrap() }
    }

    /// Allocates a duplicate of this SharedBuffer (fallible).
    fn try_clone_buffer(&self) -> Result<Self, AllocError> where T: Clone {
        unsafe { clone_buffer::<T>(self.ptr, None) }
    }

    /// Allocates a duplicate of this SharedBuffer (fallible).
    fn try_copy_buffer(&self) -> Result<Self, AllocError> where T: Copy {
        unsafe { copy_buffer::<T>(self.ptr, None) }
    }
    
    #[inline]
    fn can_mutate(&self) -> bool {
        self.header().ref_count.load(Ordering::SeqCst) == 1
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe {
            std::slice::from_raw_parts(self.data_ptr(), self.header().len as usize)
        }
    }

    #[inline]
    pub fn new_ref(&self) -> Self {
        unsafe { self.ptr.as_ref().ref_count.fetch_add(1, Ordering::SeqCst); }
        RawBuffer { ptr: self.ptr, _marker: PhantomData }
    }

    #[inline]
    fn data_ptr(&self) -> *mut T {
        unsafe {
            (self.ptr.as_ptr() as *mut u8).offset(header_size::<T>() as isize) as *mut T
        }
    }

    #[inline]
    fn header(&self) -> &Header {
        unsafe { self.ptr.as_ref() }
    }

    #[inline]
    pub fn ptr_eq(&self, other: &Self) -> bool {
        self.ptr == other.ptr
    }
}

// SAFETY: all of the following methods require the buffer to be safely mutable. In other
// words, there is a single reference to the buffer (can_mutate() returned true).
impl<T> RawBuffer<T> {
    unsafe fn reserve_one(&mut self) -> Option<NonNull<T>> {
        unsafe {
            let header = self.ptr.as_mut();
            if header.len >= header.cap {
                return None;
            }

            let ptr = self.data_ptr().offset(header.len as isize);
            header.len += 1;

            NonNull::new(ptr)
        }
    }

    unsafe fn try_push(&mut self, val: T) -> Result<(), AllocError> {
        if let Some(ptr) = self.reserve_one() {
            unsafe {
                ptr::write(ptr.as_ptr(), val);
            }
    
            return Ok(());
        }

        return Err(AllocError::CapacityOverflow);
    }

    unsafe fn push(&mut self, val: T) {
        let ptr = self.reserve_one().unwrap();
        unsafe {
            ptr::write(ptr.as_ptr(), val);
        }
    }

    unsafe fn pop(&mut self) -> Option<T> {
        let len = self.len();
        if len == 0 {
            return None;
        }

        unsafe {
            let popped = ptr::read(self.data_ptr().add(len - 1));
            self.ptr.as_mut().len -= 1;

            Some(popped)
        }
    }

    unsafe fn try_push_slice(&mut self, data: &[T]) -> Result<(), AllocError> where T: Clone {
        if data.len() > self.capacity() - self.len() {
            return Err(AllocError::CapacityOverflow);
        }

        unsafe {
            let offset = self.len();
            self.ptr.as_mut().len += data.len() as BufferSize;

            let mut ptr = self.data_ptr().add(offset);

            for item in data {
                ptr::write(ptr, item.clone());
                ptr = ptr.add(1)
            }
        }

        Ok(())
    }

    unsafe fn try_extend(&mut self, iter: &mut impl Iterator<Item = T>) -> Result<(), AllocError> {
        let (min_len, _upper_bound) = iter.size_hint();
        if min_len > self.capacity() - self.len() {
            return Err(AllocError::CapacityOverflow);
        }

        if min_len > 0 {
            self.extend_n(iter, min_len as BufferSize);
        }

        for item in iter {
            self.try_push(item)?;
        }

        Ok(())
    }

    unsafe fn extend_n(&mut self, iter: &mut impl Iterator<Item = T>, n: BufferSize) {
        let offset = self.len();

        let mut ptr = self.data_ptr().add(offset);
        let mut count = 0;
        for item in iter {
            if count == n {
                break;
            }
            ptr::write(ptr, item);
            ptr = ptr.add(1);
            count += 1;
        }

        self.ptr.as_mut().len += count;
    }

    unsafe fn clear(&mut self) {
        unsafe {
            clear::<T>(self.ptr);
        }
    }


    #[inline]
    unsafe fn as_mut_slice(&mut self) -> &mut[T] {
        unsafe {
            std::slice::from_raw_parts_mut(self.data_ptr(), self.header().len as usize)
        }
    }

    unsafe fn move_data(&mut self, dst_buffer: &mut Self) {
        debug_assert!(dst_buffer.remaining_capacity() >= self.len());

        let len = self.header().len;
        if len > 0 {
            unsafe {
                let dst_header = dst_buffer.ptr.as_mut();
    
                let src = data_ptr(self.ptr);
                let dst = data_ptr::<T>(dst_buffer.ptr).add(dst_header.len as usize);

                dst_header.len = len;
                self.ptr.as_mut().len = 0;

                ptr::copy_nonoverlapping(src, dst, len as usize);
            }    
        }
    }
}

impl<T> Clone for RawBuffer<T> {
    fn clone(&self) -> Self {
        self.new_ref()
    }
}

impl<T> Drop for RawBuffer<T> {
    fn drop(&mut self) {
        unsafe {
            release::<T>(self.ptr);
        }
    }
}


impl<T> SharedBuffer<T> {
    /// Creates an empty shared buffer without allocating memory.
    #[inline]
    pub fn new() -> Self {
        SharedBuffer { inner: RawBuffer::new_empty() }
    }

    #[inline]
    pub fn with_capacity(cap: usize) -> Self {
        SharedBuffer { inner: RawBuffer::try_with_capacity(cap).unwrap() }
    }

    #[inline]
    pub fn from_slice(data: &[T]) -> Self where T: Clone {
        SharedBuffer { inner: RawBuffer::try_from_slice(data, None).unwrap() }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.len() == 0
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    #[inline]
    pub fn remaining_capacity(&self) -> usize {
        self.inner.remaining_capacity()
    }

    #[inline]
    pub fn new_ref(&self) -> Self {
        SharedBuffer { inner: self.inner.new_ref() }
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.inner.as_slice()
    }

    /// Allocates a duplicate of this buffer (infallible).
    pub fn clone_buffer(&self) -> Self where T: Clone {
        SharedBuffer { inner: self.inner.try_clone_buffer().unwrap() }
    }

    /// Allocates a duplicate of this buffer (infallible).
    pub fn copy_buffer(&self) -> Self where T: Copy {
        SharedBuffer { inner: self.inner.try_copy_buffer().unwrap() }
    }

    /// Returns true if this is the only existing handle to the buffer.
    #[inline]
    pub fn can_mutate(&self) -> bool {
        self.inner.can_mutate()
    }

    #[inline]
    fn make_mutable(self) -> Self where T: Clone {
        if self.can_mutate() {
            self
        } else {
            self.clone_buffer()
        }
    }

    /// Converts this SharedBuffer into an immutable one, allocating a new one if there are other references.
    #[inline]
    pub fn into_mut(self) -> MutableBuffer<T> where T: Clone {
        MutableBuffer { inner: self.make_mutable().inner }
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

    pub fn push(self, val: T) -> Self where T: Clone {
        let mut this = self.ensure_capacity(1);
        unsafe { this.inner.push(val); }
        
        this
    }

    pub fn pop(self) -> (Self, Option<T>) where T: Clone {
        let mut this = self.make_mutable();
        let popped = unsafe { this.inner.pop() };

        (this, popped)
    }

    pub fn push_slice(self, data: &[T]) -> Self where T: Clone {
        let mut this = self.ensure_capacity(data.len());
        unsafe { this.inner.try_push_slice(data).unwrap(); }
        
        this
    }

    pub fn extend(self, data: impl IntoIterator<Item = T>) -> Self where T: Clone {
        let mut iter = data.into_iter();
        let (min, max) = iter.size_hint();
        let mut this = self.ensure_capacity(max.unwrap_or(min));
        unsafe { this.inner.try_extend(&mut iter).unwrap(); }
        
        this
    }

    /// Returns true if the two shapred buffers point to the same underlying storage.
    pub fn ptr_eq(&self, other: &Self) -> bool {
        self.inner.ptr_eq(&other.inner)
    }

    /// Returns a buffer that can be safely mutated and has enough extra capacity to
    /// add `cap` more items.
    fn ensure_capacity(self, cap: usize) -> Self where T: Clone {
        if self.remaining_capacity() > cap && self.can_mutate() {
            return self;
        }

        // TODO: if the buffer is mutable (unique), we could avoid cloning
        // the data and make a copy instead.

        // TODO: grow the buffer exponentially.
        let new_cap = self.len() + cap;
        SharedBuffer { inner: RawBuffer::try_from_slice(self.as_slice(), Some(new_cap)).unwrap() }
    }
}

unsafe impl<T: Sync> Send for SharedBuffer<T> {}

impl<T: PartialEq<T>> PartialEq<SharedBuffer<T>> for SharedBuffer<T> {
    fn eq(&self, other: &Self) -> bool {
        self.ptr_eq(other) || self.as_slice() == other.as_slice()
    }
}

impl<T: PartialEq<T>> PartialEq<&[T]> for SharedBuffer<T> {
    fn eq(&self, other: &&[T]) -> bool {
        self.as_slice() == *other
    }
}

impl<T> AsRef<[T]> for SharedBuffer<T> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> Default for SharedBuffer<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, T> IntoIterator for &'a SharedBuffer<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> std::slice::Iter<'a, T> {
        self.as_slice().iter()
    }
}

impl<T, I> Index<I> for SharedBuffer<T> where I: std::slice::SliceIndex<[T]> {
    type Output = <I as std::slice::SliceIndex<[T]>>::Output;
    fn index(&self, index: I) -> &Self::Output {
        self.as_slice().index(index)
    }
}

impl<T> MutableBuffer<T> {
    /// Allocates a mutable buffer with a default capacity of 16.
    pub fn new() -> Self {
        Self::with_capacity(16)
    }

    pub fn with_capacity(cap: usize) -> Self {
        MutableBuffer { inner: RawBuffer::try_with_capacity(cap).unwrap() }
    }

    pub fn from_slice(data: &[T]) -> Self where T: Clone {
        MutableBuffer { inner: RawBuffer::try_from_slice(data, None).unwrap() }
    }

    pub fn is_empty(&self) -> bool {
        self.inner.len() == 0
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    #[inline]
    fn remaining_capacity(&self) -> usize {
        self.capacity() - self.len()
    }

    /// Make this SharedBuffer immutable.
    ///
    /// This operation is cheap, the underlying storage does not not need
    /// to be reallocated.
    #[inline]
    pub fn into_immutable(self) -> SharedBuffer<T> {
        SharedBuffer { inner: self.inner }
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
    pub fn push(&mut self, val: T) {
        self.ensure_capacity(1);

        // Safe because this type guarantees mutability.
        unsafe { self.inner.push(val); }
    }

    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        // Safe because this type guarantees mutability.
        unsafe { self.inner.pop() }
    }

    #[inline]
    pub fn push_slice(&mut self, data: &[T]) where T: Clone {
        self.ensure_capacity(data.len());

        // Safe because this type guarantees mutability.
        unsafe { self.inner.try_push_slice(data). unwrap(); }
    }

    pub fn extend(&mut self, data: impl IntoIterator<Item = T>) {
        let mut iter = data.into_iter();
        let (min, max) = iter.size_hint();
        self.ensure_capacity(max.unwrap_or(min));
        // Safe because this type guarantees mutability.
        unsafe { self.inner.try_extend(&mut iter).unwrap(); }
    }

    #[inline]
    pub fn clear(mut self) -> Self {
        unsafe { self.inner.clear(); }

        self
    }

    pub fn clone_buffer(&self) -> Self where T: Clone {
        MutableBuffer { inner: self.inner.try_clone_buffer().unwrap() }
    }

    pub fn clone_buffer_with_capacity(&self, cap: BufferSize) -> Self where T: Clone {
        MutableBuffer { inner: self.inner.clone_buffer_with_capacity(cap) }
    }

    fn try_realloc(&mut self, new_cap: usize) -> Result<(), AllocError> {
        if new_cap < self.len() {
            return Err(AllocError::CapacityOverflow);
        }

        let mut dst = RawBuffer::try_with_capacity(new_cap)?;

        unsafe { self.inner.move_data(&mut dst) };

        mem::swap(&mut self.inner, &mut dst);

        Ok(())
    }

    fn realloc(&mut self, new_cap: usize) {
        self.try_realloc(new_cap).unwrap();
    }

    fn ensure_capacity(&mut self, cap: usize) {
        if self.remaining_capacity() < cap {
            let new_cap = self.len() + cap;
            self.realloc(new_cap)
        }
    }
}

impl<T: Clone> Clone for MutableBuffer<T> {
    fn clone(&self) -> Self {
        self.clone_buffer()
    }
}

impl<T: PartialEq<T>> PartialEq<MutableBuffer<T>> for MutableBuffer<T> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: PartialEq<T>> PartialEq<&[T]> for MutableBuffer<T> {
    fn eq(&self, other: &&[T]) -> bool {
        self.as_slice() == *other
    }
}

impl<T> AsRef<[T]> for MutableBuffer<T> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> AsMut<[T]> for MutableBuffer<T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T> Default for MutableBuffer<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, T> IntoIterator for &'a MutableBuffer<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> std::slice::Iter<'a, T> {
        self.as_slice().iter()
    }
}

impl<T, I> Index<I> for MutableBuffer<T> where I: std::slice::SliceIndex<[T]> {
    type Output = <I as std::slice::SliceIndex<[T]>>::Output;
    fn index(&self, index: I) -> &Self::Output {
        self.as_slice().index(index)
    }
}

impl<T, I> IndexMut<I> for MutableBuffer<T> where I: std::slice::SliceIndex<[T]> {
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        self.as_mut_slice().index_mut(index)
    }
}




unsafe fn release<T>(ptr: NonNull<Header>) -> bool {
    if ptr.as_ref().ref_count.fetch_sub(1, Ordering::SeqCst) == 1 {
        dealloc::<T>(ptr);
        return true;
    }

    false
}

unsafe fn data_ptr<T>(header: NonNull<Header>) -> *mut T {
    (header.as_ptr() as *mut u8).offset(header_size::<T>() as isize) as *mut T
}

// SAFETY: T must implement Copy or the source buffer must forget its content.
unsafe fn copy_buffer<T>(header: NonNull<Header>, new_cap: Option<BufferSize>) -> Result<RawBuffer<T>, AllocError> {
    let cap = new_cap.unwrap_or(header.as_ref().cap);
    let len = header.as_ref().len;

    if len > cap {
        return Err(AllocError::CapacityOverflow);
    }

    let mut clone = RawBuffer::try_with_capacity(cap as usize)?;

    if len > 0 {
        std::ptr::copy_nonoverlapping(data_ptr::<T>(header), clone.data_ptr(), len as usize);
        clone.ptr.as_mut().len = len;
    }

    Ok(clone)
}

unsafe fn clone_buffer<T>(header: NonNull<Header>, new_cap: Option<BufferSize>) -> Result<RawBuffer<T>, AllocError>
    where T: Clone
{
    let cap = new_cap.unwrap_or(header.as_ref().cap);
    let len = header.as_ref().len;

    if len > cap {
        return Err(AllocError::CapacityOverflow);
    }

    let mut clone = RawBuffer::try_with_capacity(cap as usize)?;

    unsafe {
        let mut src = data_ptr::<T>(header);
        let mut dst = clone.data_ptr();
        for _ in 0..len {
            ptr::write(dst, (*src).clone());
            src = src.add(1);
            dst = dst.add(1);
        }    
    }

    clone.ptr.as_mut().len = len;

    Ok(clone)
}

const fn header_size<T>() -> usize {
    let align = mem::align_of::<T>();
    let size = mem::size_of::<Header>();

    if align > 0 {
        ((size + align - 1) / align) * align
    } else {
        size
    }
}

fn buffer_layout<T>(n: usize) -> Result<Layout, AllocError> {
    let size = mem::size_of::<T>()
        .checked_mul(n)
        .ok_or(AllocError::CapacityOverflow)?;
    let align = mem::align_of::<Header>().max(mem::align_of::<T>());
    let header_size = mem::size_of::<Header>().max(mem::align_of::<T>());

    Layout::from_size_align(size + header_size, align).map_err(|_| AllocError::CapacityOverflow)
}

unsafe fn clear<T>(ptr: NonNull<Header>) {
    let len = ptr.as_ref().len;

    let mut item = data_ptr::<T>(ptr);
    for _ in 0..len {
        std::ptr::drop_in_place(item);
        item = item.add(1);
    }
}

unsafe fn dealloc<T>(ptr: NonNull<Header>) {
    clear::<T>(ptr);

    let cap = ptr.as_ref().cap as usize;
    let layout = buffer_layout::<T>(cap).unwrap();

    alloc::dealloc(ptr.as_ptr() as *mut u8, layout);
}

/// Error type for APIs with fallible heap allocation
#[derive(Debug)]
pub enum AllocError {
    /// Overflow `usize::MAX` or other error during size computation
    CapacityOverflow,
    /// The allocator return an error
    Allocator {
        /// The layout that was passed to the allocator
        layout: Layout,
    },
}

// In order to give us a chance to catch leaks and double-frees, test with values that implement drop.
#[cfg(test)]
fn num(val: u32) -> Box<u32> { Box::new(val) }

#[test]
fn basic() {
    let mut a = MutableBuffer::with_capacity(256);

    a.push(num(0));
    a.push(num(1));
    a.push(num(2));

    let a = a.into_immutable();

    assert_eq!(a.len(), 3);

    assert_eq!(a.as_slice(), &[num(0), num(1), num(2)]);

    assert!(a.can_mutate());

    let b = MutableBuffer::from_slice(&[num(0), num(1), num(2), num(3), num(4)]);

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

    let mut d = MutableBuffer::with_capacity(64);
    d.push_slice(&[num(0), num(1), num(2)]);
    d.push_slice(&[]);
    d.push_slice(&[num(3), num(4)]);

    assert_eq!(d.as_slice(), &[num(0), num(1), num(2), num(3), num(4)]);
}

#[test]
fn value_oriented() {
    let a = SharedBuffer::with_capacity(64)
        .push(num(1))
        .push(num(2));

    let b = a.new_ref().push(num(4));

    let a = a.push(num(3));

    assert_eq!(a.as_slice(), &[num(1), num(2), num(3)]);
    assert_eq!(b.as_slice(), &[num(1), num(2), num(4)]);

    let (a, popped) = a.pop();
    assert_eq!(a.as_slice(), &[num(1), num(2)]);
    assert_eq!(popped, Some(num(3)));

    let (b, popped) = b.new_ref().pop();
    assert_eq!(b.as_slice(), &[num(1), num(2)]);
    assert_eq!(popped, Some(num(4)));
}

#[test]
fn empty_buffer() {
    let a: SharedBuffer<u32> = SharedBuffer::new();
    assert!(!a.can_mutate());
    {
        let b: SharedBuffer<u32> = SharedBuffer::new();
        assert!(!b.can_mutate());
        assert!(a.ptr_eq(&b));    
    }

    assert!(!a.can_mutate());

    let _: SharedBuffer<()> = SharedBuffer::new();
    let _: SharedBuffer<()> = SharedBuffer::new();
}

#[test]
fn grow() {
    let mut a = MutableBuffer::with_capacity(0);

    a.push(num(1));
    a.push(num(2));
    a.push(num(3));

    a.push_slice(&[num(4), num(5), num(6), num(7), num(8), num(9), num(10), num(12), num(12), num(13), num(14), num(15), num(16), num(17), num(18)]);

    assert_eq!(a.as_slice(), &[num(1), num(2), num(3), num(4), num(5), num(6), num(7), num(8), num(9), num(10), num(12), num(12), num(13), num(14), num(15), num(16), num(17), num(18)]);

    let b = SharedBuffer::new()
        .push(num(1))
        .push(num(2))
        .push(num(3));
    
    assert_eq!(b.as_slice(), &[num(1), num(2), num(3)]);
}

#[test]
fn buffer_layout_alignemnt() {
    type B = Box<u32>;
    let layout = buffer_layout::<B>(2).unwrap();
    assert_eq!(layout.align(), mem::size_of::<B>())
}
