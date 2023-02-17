use std::alloc::{self, Layout};
use std::marker::PhantomData;
use std::mem;
use std::ptr::{self, NonNull};
use std::sync::atomic::{AtomicI32, Ordering};

pub type BufferSize = u32;

pub struct RawBuffer<T> {
    pub header: NonNull<Header>,
    _marker: PhantomData<T>,
}

/// Some information stored at the beginning of every buffer.
pub struct Header {
    // TODO: It would be great to make the types generic over whether or not the reference count is atomic.
    pub ref_count: AtomicI32,
    pub cap: BufferSize,
    pub len: BufferSize,
    pub _pad: u32,
}

impl BufferHeader for Header {
    fn with_capacity(cap: BufferSize) -> Self {
        Header {
            ref_count: AtomicI32::new(1),
            cap,
            len: 0,
            _pad: 0,
        }
    }
    #[inline]
    unsafe fn add_ref(this: NonNull<Self>) {
        (*this.as_ptr()).ref_count.fetch_add(1, Ordering::SeqCst);
    }
    #[inline]
    unsafe fn release_ref(this:  NonNull<Self>) -> bool {
        (*this.as_ptr()).ref_count.fetch_sub(1, Ordering::SeqCst) == 1
    }
    #[inline]
    unsafe fn len(this:  NonNull<Self>) -> BufferSize {
        (*this.as_ptr()).len
    }
    #[inline]
    unsafe fn set_len(this:  NonNull<Self>, val: BufferSize) {
        (*this.as_ptr()).len = val;
    }
    #[inline]
    unsafe fn capacity(this:  NonNull<Self>) -> BufferSize {
        (*this.as_ptr()).cap
    }
    #[inline]
    unsafe fn ref_count(this: NonNull<Self>) -> i32 {
        (*this.as_ptr()).ref_count.load(Ordering::SeqCst)
    }
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
    _pad: 0,
};

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

impl<T> RawBuffer<T> {
    pub fn new_empty() -> Self {
        GLOBAL_EMPTY_BUFFER.ref_count.fetch_add(1, Ordering::SeqCst);

        let global = &GLOBAL_EMPTY_BUFFER as *const Header as *mut Header;

        RawBuffer {
            header: NonNull::new(global).unwrap(),
            _marker: PhantomData,
        }
    }

    #[inline(never)]
    pub fn try_with_capacity(cap: usize) -> Result<RawBuffer<T>, AllocError> {
        unsafe {
            if cap > BufferSize::MAX as usize {
                return Err(capacity_error());
            }

            let layout = buffer_layout::<Header, T>(cap)?;
            let alloc: NonNull<Header> = NonNull::new(alloc::alloc(layout))
                .ok_or(AllocError::Allocator { layout })?
                .cast();

            ptr::write(
                alloc.as_ptr(),
                Header {
                    ref_count: AtomicI32::new(1),
                    cap: cap as BufferSize,
                    len: 0,
                    _pad: 0,
                },
            );

            Ok(RawBuffer {
                header: alloc,
                _marker: PhantomData,
            })
        }
    }

    pub fn try_from_slice(data: &[T], cap: Option<usize>) -> Result<RawBuffer<T>, AllocError>
    where
        T: Clone,
    {
        let len = data.len();
        let cap = cap.map(|cap| cap.max(len)).unwrap_or(len);

        if cap > BufferSize::MAX as usize {
            return Err(capacity_error());
        }

        let mut buffer = Self::try_with_capacity(cap)?;

        unsafe {
            buffer.header.as_mut().len = len as BufferSize;

            let mut ptr = buffer.data_ptr();

            for item in data {
                ptr::write(ptr, item.clone());
                ptr = ptr.add(1)
            }
        }

        Ok(buffer)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.header().len as usize
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.header().len == 0
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.header().cap as usize
    }

    #[inline]
    pub fn remaining_capacity(&self) -> usize {
        let header = self.header();
        (header.cap - header.len) as usize
    }

    /// Allocates a duplicate of this SharedBuffer with a different capacity (infallible).
    ///
    /// Resturns an error if the capacity is lower than this SharedBuffer's length.
    pub fn clone_buffer_with_capacity(&self, cap: BufferSize) -> Self
    where
        T: Clone,
    {
        unsafe { clone_buffer::<Header, T>(self.header, Some(cap)).unwrap() }
    }

    /// Allocates a duplicate of this SharedBuffer (fallible).
    pub fn try_clone_buffer(&self) -> Result<Self, AllocError>
    where
        T: Clone,
    {
        unsafe { clone_buffer::<Header, T>(self.header, None) }
    }

    /// Allocates a duplicate of this SharedBuffer (fallible).
    pub fn try_copy_buffer(&self) -> Result<Self, AllocError>
    where
        T: Copy,
    {
        unsafe { copy_buffer::<T>(self.header, None) }
    }

    #[inline]
    pub fn can_mutate(&self) -> bool {
        self.header().ref_count.load(Ordering::SeqCst) == 1
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.data_ptr(), self.header().len as usize) }
    }

    #[inline]
    pub fn new_ref(&self) -> Self {
        unsafe {
            self.header
                .as_ref()
                .ref_count
                .fetch_add(1, Ordering::SeqCst);
        }
        RawBuffer {
            header: self.header,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn data_ptr(&self) -> *mut T {
        unsafe { (self.header.as_ptr() as *mut u8).add(header_size::<Header, T>()) as *mut T }
    }

    #[inline]
    pub fn header(&self) -> &Header {
        unsafe { self.header.as_ref() }
    }

    #[inline]
    pub fn ptr_eq(&self, other: &Self) -> bool {
        self.header == other.header
    }
}

// SAFETY: All of the following methods require the buffer to be safely mutable. In other
// words, there is a single reference to the buffer (can_mutate() returned true).
impl<T> RawBuffer<T> {
    #[inline]
    pub unsafe fn set_len(&mut self, new_len: BufferSize) {
        self.header.as_mut().len = new_len;
    }

    pub unsafe fn try_push(&mut self, val: T) -> Result<(), AllocError> {
        let header = self.header.as_mut();
        if header.len >= header.cap {
            return Err(capacity_error());
        }

        let address = self.data_ptr().offset(header.len as isize);
        header.len += 1;

        ptr::write(address, val);

        Ok(())
    }

    // SAFETY: The capacity MUST be ensured beforehand.
    // The inline annotation really helps here.
    #[inline]
    pub unsafe fn push(&mut self, val: T) {
        let header = self.header.as_mut();
        let address = self.data_ptr().add(header.len as usize);
        header.len += 1;

        ptr::write(address, val);
    }

    #[inline]
    pub unsafe fn pop(&mut self) -> Option<T> {
        let len = self.len();
        if len == 0 {
            return None;
        }

        let popped = ptr::read(self.data_ptr().add(len - 1));
        self.header.as_mut().len -= 1;

        Some(popped)
    }

    #[inline]
    pub unsafe fn first(&self) -> Option<*mut T> {
        let len = self.len();
        if len == 0 {
            return None;
        }

        Some(self.data_ptr())    
    }

    #[inline]
    pub unsafe fn last(&self) -> Option<*mut T> {
        let len = self.len();
        if len == 0 {
            return None;
        }

        unsafe{
            Some(self.data_ptr().add(len - 1))    
        }
    }

    pub unsafe fn try_push_slice(&mut self, data: &[T]) -> Result<(), AllocError>
    where
        T: Clone,
    {
        if data.len() > self.capacity() - self.len() {
            return Err(capacity_error());
        }

        let offset = self.len();
        self.header.as_mut().len += data.len() as BufferSize;

        let mut ptr = self.data_ptr().add(offset);

        for item in data {
            ptr::write(ptr, item.clone());
            ptr = ptr.add(1)
        }

        Ok(())
    }

    pub unsafe fn try_extend(
        &mut self,
        iter: &mut impl Iterator<Item = T>,
    ) -> Result<(), AllocError> {
        let (min_len, _upper_bound) = iter.size_hint();
        if min_len > self.capacity() - self.len() {
            return Err(capacity_error());
        }

        if min_len > 0 {
            self.extend_n(iter, min_len as BufferSize);
        }

        for item in iter {
            self.try_push(item)?;
        }

        Ok(())
    }

    pub unsafe fn extend_n(&mut self, iter: &mut impl Iterator<Item = T>, n: BufferSize) {
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

        self.header.as_mut().len += count;
    }

    pub unsafe fn clear(&mut self) {
        unsafe {
            clear::<Header, T>(self.header);
        }
    }

    #[inline]
    pub unsafe fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.data_ptr(), self.header().len as usize) }
    }

    pub unsafe fn move_data(&mut self, dst_buffer: &mut Self) {
        debug_assert!(dst_buffer.remaining_capacity() >= self.len());

        let len = self.header().len;
        if len > 0 {
            unsafe {
                let dst_header = dst_buffer.header.as_mut();

                let src = data_ptr(self.header);
                let dst = data_ptr::<Header, T>(dst_buffer.header).add(dst_header.len as usize);

                dst_header.len += len;
                self.set_len(0);

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
            release::<T>(self.header);
        }
    }
}

pub unsafe fn release<T>(ptr: NonNull<Header>) -> bool {
    if ptr.as_ref().ref_count.fetch_sub(1, Ordering::SeqCst) == 1 {
        dealloc::<Header, T>(ptr);
        return true;
    }

    false
}

#[inline]
pub unsafe fn data_ptr<Header, T>(header: NonNull<Header>) -> *mut T {
    (header.as_ptr() as *mut u8).add(header_size::<Header, T>()) as *mut T
}

// SAFETY: T must implement Copy or the source buffer must forget its content.
pub unsafe fn copy_buffer<T>(
    header: NonNull<Header>,
    new_cap: Option<BufferSize>,
) -> Result<RawBuffer<T>, AllocError> {
    let cap = new_cap.unwrap_or(header.as_ref().cap);
    let len = header.as_ref().len;

    if len > cap {
        return Err(capacity_error());
    }

    let mut clone = RawBuffer::try_with_capacity(cap as usize)?;

    if len > 0 {
        std::ptr::copy_nonoverlapping(data_ptr::<Header, T>(header), clone.data_ptr(), len as usize);
        clone.set_len(len);
    }

    Ok(clone)
}

pub unsafe fn clone_buffer<Header: BufferHeader, T>(
    header: NonNull<Header>,
    new_cap: Option<BufferSize>,
) -> Result<RawBuffer<T>, AllocError>
where
    T: Clone,
{
    let cap = if let Some(cap) = new_cap {
        cap
    } else {
        Header::capacity(header)
    };

    let len = Header::len(header);

    if len > cap {
        return Err(capacity_error());
    }

    let mut clone = RawBuffer::try_with_capacity(cap as usize)?;

    unsafe {
        let mut src = data_ptr::<Header, T>(header);
        let mut dst = clone.data_ptr();
        for _ in 0..len {
            ptr::write(dst, (*src).clone());
            src = src.add(1);
            dst = dst.add(1);
        }
    }

    clone.set_len(len);

    Ok(clone)
}

const fn header_size<Header, T>() -> usize {
    let align = mem::align_of::<T>();
    let size = mem::size_of::<Header>();

    if align > 0 {
        ((size + align - 1) / align) * align
    } else {
        size
    }
}

fn buffer_layout<Header, T>(n: usize) -> Result<Layout, AllocError> {
    let size = mem::size_of::<T>()
        .checked_mul(n)
        .ok_or(AllocError::CapacityOverflow)?;
    let align = mem::align_of::<Header>().max(mem::align_of::<T>());
    let header_size = mem::size_of::<Header>().max(mem::align_of::<T>());

    Layout::from_size_align(size + header_size, align).map_err(|_| AllocError::CapacityOverflow)
}

pub unsafe fn clear<Header: BufferHeader, T>(ptr: NonNull<Header>) {
    let len = Header::len(ptr);

    let mut item = data_ptr::<Header, T>(ptr);
    for _ in 0..len {
        std::ptr::drop_in_place(item);
        item = item.add(1);
    }
}

pub unsafe fn dealloc<Header: BufferHeader, T>(ptr: NonNull<Header>) {
    clear::<Header, T>(ptr);

    let cap = Header::capacity(ptr) as usize;
    let layout = buffer_layout::<Header, T>(cap).unwrap();

    alloc::dealloc(ptr.as_ptr() as *mut u8, layout);
}

#[test]
fn buffer_layout_alignemnt() {
    type B = Box<u32>;
    let layout = buffer_layout::<Header, B>(2).unwrap();
    assert_eq!(layout.align(), mem::size_of::<B>())
}

#[cold]
fn capacity_error() -> AllocError {
    AllocError::CapacityOverflow
}


pub trait BufferHeader {
    fn with_capacity(cap: BufferSize) -> Self;
    unsafe fn add_ref(this: NonNull<Self>);
    unsafe fn release_ref(this:  NonNull<Self>) -> bool;
    unsafe fn len(this:  NonNull<Self>) -> BufferSize;
    unsafe fn set_len(this:  NonNull<Self>, val: BufferSize);
    unsafe fn capacity(this:  NonNull<Self>) -> BufferSize;
    unsafe fn ref_count(this: NonNull<Self>) -> i32;
}

pub struct HeaderBuffer<H: BufferHeader, T> {
    pub header: NonNull<H>,
    _marker: PhantomData<T>,
}

impl<H: BufferHeader, T> HeaderBuffer<H, T> {
    #[inline(never)]
    pub fn try_with_capacity(cap: usize) -> Result<Self, AllocError> {
        unsafe {
            if cap > BufferSize::MAX as usize {
                return Err(capacity_error());
            }

            let layout = buffer_layout::<Header, T>(cap)?;
            let alloc: NonNull<H> = NonNull::new(alloc::alloc(layout))
                .ok_or(AllocError::Allocator { layout })?
                .cast();

            let header = H::with_capacity(cap as BufferSize);
            ptr::write(
                alloc.as_ptr(),
                header,
            );

            Ok(HeaderBuffer {
                header: alloc,
                _marker: PhantomData,
            })
        }
    }

    pub fn try_from_slice(data: &[T], cap: Option<usize>) -> Result<Self, AllocError>
    where
        T: Clone,
    {
        let len = data.len();
        let cap = cap.map(|cap| cap.max(len)).unwrap_or(len);

        if cap > BufferSize::MAX as usize {
            return Err(capacity_error());
        }

        let buffer = Self::try_with_capacity(cap)?;

        unsafe {
            H::set_len(buffer.header, len as BufferSize);

            let mut ptr = buffer.data_ptr();

            for item in data {
                ptr::write(ptr, item.clone());
                ptr = ptr.add(1)
            }
        }

        Ok(buffer)
    }

    #[inline]
    pub fn len(&self) -> BufferSize {
        unsafe { H::len(self.header) }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        unsafe { H::len(self.header) == 0 }
    }

    #[inline]
    pub fn capacity(&self) -> BufferSize {
        unsafe { H::capacity(self.header) }
    }

    #[inline]
    pub fn remaining_capacity(&self) -> BufferSize {
        unsafe { (H::capacity(self.header) - H::len(self.header)) }
    }

    /// Allocates a duplicate of this SharedBuffer (fallible).
    pub fn try_clone_buffer(&self, new_cap: Option<BufferSize>) -> Result<Self, AllocError>
    where
        T: Clone,
    {
        unsafe {
            let len = H::len(self.header);
            let cap = if let Some(cap) = new_cap {
                cap
            } else {
                H::capacity(self.header)
            };

            if len > cap {
                return Err(capacity_error());
            }

            let mut clone = HeaderBuffer::try_with_capacity(cap as usize)?;

            unsafe {
                let mut src = self.data_ptr();
                let mut dst = clone.data_ptr();
                for _ in 0..len {
                    ptr::write(dst, (*src).clone());
                    src = src.add(1);
                    dst = dst.add(1);
                }
            }
        
            clone.set_len(len);
        
            Ok(clone)    
        }
    }

    /// Allocates a duplicate of this SharedBuffer (fallible).
    pub fn try_copy_buffer(&self, new_cap: Option<BufferSize>) -> Result<Self, AllocError>
    where
        T: Copy,
    {
        unsafe {
            let len = H::len(self.header);
            let cap = if let Some(cap) = new_cap {
                cap
            } else {
                H::capacity(self.header)
            };

            if len > cap {
                return Err(capacity_error());
            }
        
            let mut clone = HeaderBuffer::try_with_capacity(cap as usize)?;
        
            if len > 0 {
                std::ptr::copy_nonoverlapping(self.data_ptr(), clone.data_ptr(), len as usize);
                clone.set_len(len);
            }
        
            Ok(clone)    
        }
    }

    #[inline]
    pub fn can_mutate(&self) -> bool {
        unsafe { H::ref_count(self.header) == 1 }
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.data_ptr(), H::len(self.header) as usize) }
    }

    #[inline]
    pub fn new_ref(&self) -> Self {
        unsafe {
            H::add_ref(self.header);
        }
        HeaderBuffer {
            header: self.header,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn data_ptr(&self) -> *mut T {
        unsafe { (self.header.as_ptr() as *mut u8).add(header_size::<H, T>()) as *mut T }
    }

    #[inline]
    pub fn header(&self) -> &H {
        unsafe { self.header.as_ref() }
    }

    #[inline]
    pub fn ptr_eq(&self, other: &Self) -> bool {
        self.header == other.header
    }
}

// SAFETY: All of the following methods require the buffer to be safely mutable. In other
// words, there is a single reference to the buffer (can_mutate() returned true).
impl<H: BufferHeader, T> HeaderBuffer<H, T> {
    #[inline]
    pub unsafe fn set_len(&mut self, new_len: BufferSize) {
        H::set_len(self.header, new_len);
    }

    pub unsafe fn try_push(&mut self, val: T) -> Result<(), AllocError> {
        let len = H::len(self.header);
        if len >= H::capacity(self.header) {
            return Err(capacity_error());
        }

        let address = self.data_ptr().offset(H::len(self.header) as isize);
        H::set_len(self.header, len + 1);

        ptr::write(address, val);

        Ok(())
    }

    // SAFETY: The capacity MUST be ensured beforehand.
    // The inline annotation really helps here.
    #[inline]
    pub unsafe fn push(&mut self, val: T) {
        let len = H::len(self.header);
        H::set_len(self.header, len + 1);

        let address = self.data_ptr().add(len as usize);
        ptr::write(address, val);
    }

    #[inline]
    pub unsafe fn pop(&mut self) -> Option<T> {
        let len = H::len(self.header);
        if len == 0 {
            return None;
        }

        let new_len = len - 1;
        let popped = ptr::read(self.data_ptr().add(new_len as usize));
        H::set_len(self.header, new_len);

        Some(popped)
    }

    #[inline]
    pub unsafe fn first(&self) -> Option<*mut T> {
        let len = self.len();
        if len == 0 {
            return None;
        }

        Some(self.data_ptr())    
    }

    #[inline]
    pub unsafe fn last(&self) -> Option<*mut T> {
        let len = self.len() as usize;
        if len == 0 {
            return None;
        }

        unsafe{
            Some(self.data_ptr().add(len - 1))    
        }
    }

    pub unsafe fn try_push_slice(&mut self, data: &[T]) -> Result<(), AllocError>
    where
        T: Clone,
    {
        if data.len() > self.remaining_capacity() as usize {
            return Err(capacity_error());
        }

        let inital_len = H::len(self.header);
        H::set_len(self.header, inital_len + data.len() as BufferSize);

        let mut ptr = self.data_ptr().add(inital_len as usize);

        for item in data {
            ptr::write(ptr, item.clone());
            ptr = ptr.add(1)
        }

        Ok(())
    }

    pub unsafe fn try_extend(
        &mut self,
        iter: &mut impl Iterator<Item = T>,
    ) -> Result<(), AllocError> {
        let (min_len, _upper_bound) = iter.size_hint();
        if min_len > self.remaining_capacity() as usize {
            return Err(capacity_error());
        }

        if min_len > 0 {
            self.extend_n(iter, min_len as BufferSize);
        }

        for item in iter {
            self.try_push(item)?;
        }

        Ok(())
    }

    pub unsafe fn extend_n(&mut self, iter: &mut impl Iterator<Item = T>, n: BufferSize) {
        let initial_len = H::len(self.header);

        let mut ptr = self.data_ptr().add(initial_len as usize);
        let mut count = 0;
        for item in iter {
            if count == n {
                break;
            }
            ptr::write(ptr, item);
            ptr = ptr.add(1);
            count += 1;
        }

        H::set_len(self.header, initial_len + count);
    }

    pub unsafe fn clear(&mut self) {
        unsafe {
            clear::<H, T>(self.header);
        }
    }

    #[inline]
    pub unsafe fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.data_ptr(), self.len() as usize) }
    }

    pub unsafe fn move_data(&mut self, dst_buffer: &mut Self) {
        debug_assert!(dst_buffer.remaining_capacity() >= self.len());

        let len = H::len(self.header);
        if len > 0 {
            unsafe {
                let src = self.data_ptr();
                let dst = dst_buffer.data_ptr().add(dst_buffer.len() as usize);

                let inital_dst_len = H::len(dst_buffer.header);
                H::set_len(dst_buffer.header, inital_dst_len + len);
                self.set_len(0);

                ptr::copy_nonoverlapping(src, dst, len as usize);
            }
        }
    }    
}

impl<H: BufferHeader, T> Drop for HeaderBuffer<H, T> {
    fn drop(&mut self) {
        unsafe {
            if H::release_ref(self.header) {
                dealloc::<H, T>(self.header);
            }
        }
    }
}

