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

    pub fn try_with_capacity(cap: usize) -> Result<RawBuffer<T>, AllocError> {
        unsafe {
            if cap > BufferSize::MAX as usize {
                return Err(capacity_error());
            }

            let layout = buffer_layout::<T>(cap)?;
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
    pub fn capacity(&self) -> usize {
        self.header().cap as usize
    }

    #[inline]
    pub fn remaining_capacity(&self) -> usize {
        self.capacity() - self.len()
    }

    /// Allocates a duplicate of this SharedBuffer with a different capacity (infallible).
    ///
    /// Resturns an error if the capacity is lower than this SharedBuffer's length.
    pub fn clone_buffer_with_capacity(&self, cap: BufferSize) -> Self
    where
        T: Clone,
    {
        unsafe { clone_buffer::<T>(self.header, Some(cap)).unwrap() }
    }

    /// Allocates a duplicate of this SharedBuffer (fallible).
    pub fn try_clone_buffer(&self) -> Result<Self, AllocError>
    where
        T: Clone,
    {
        unsafe { clone_buffer::<T>(self.header, None) }
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
        unsafe { (self.header.as_ptr() as *mut u8).add(header_size::<T>()) as *mut T }
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

// SAFETY: all of the following methods require the buffer to be safely mutable. In other
// words, there is a single reference to the buffer (can_mutate() returned true).
impl<T> RawBuffer<T> {
    pub unsafe fn set_len(&mut self, new_len: BufferSize) {
        self.header.as_mut().len = new_len;
    }

    pub unsafe fn reserve_one(&mut self) -> Option<NonNull<T>> {
        unsafe {
            let header = self.header.as_mut();
            if header.len >= header.cap {
                return None;
            }

            let ptr = self.data_ptr().offset(header.len as isize);
            header.len += 1;

            NonNull::new(ptr)
        }
    }

    pub unsafe fn try_push(&mut self, val: T) -> Result<(), AllocError> {
        if let Some(ptr) = self.reserve_one() {
            unsafe {
                ptr::write(ptr.as_ptr(), val);
            }

            return Ok(());
        }

        Err(capacity_error())
    }

    pub unsafe fn push(&mut self, val: T) {
        let ptr = self.reserve_one().unwrap();
        unsafe {
            ptr::write(ptr.as_ptr(), val);
        }
    }

    pub unsafe fn pop(&mut self) -> Option<T> {
        let len = self.len();
        if len == 0 {
            return None;
        }

        unsafe {
            let popped = ptr::read(self.data_ptr().add(len - 1));
            self.header.as_mut().len -= 1;

            Some(popped)
        }
    }

    pub unsafe fn try_push_slice(&mut self, data: &[T]) -> Result<(), AllocError>
    where
        T: Clone,
    {
        if data.len() > self.capacity() - self.len() {
            return Err(capacity_error());
        }

        unsafe {
            let offset = self.len();
            self.header.as_mut().len += data.len() as BufferSize;

            let mut ptr = self.data_ptr().add(offset);

            for item in data {
                ptr::write(ptr, item.clone());
                ptr = ptr.add(1)
            }
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
            clear::<T>(self.header);
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
                let dst = data_ptr::<T>(dst_buffer.header).add(dst_header.len as usize);

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
        dealloc::<T>(ptr);
        return true;
    }

    false
}

pub unsafe fn data_ptr<T>(header: NonNull<Header>) -> *mut T {
    (header.as_ptr() as *mut u8).add(header_size::<T>()) as *mut T
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
        std::ptr::copy_nonoverlapping(data_ptr::<T>(header), clone.data_ptr(), len as usize);
        clone.set_len(len);
    }

    Ok(clone)
}

pub unsafe fn clone_buffer<T>(
    header: NonNull<Header>,
    new_cap: Option<BufferSize>,
) -> Result<RawBuffer<T>, AllocError>
where
    T: Clone,
{
    let cap = new_cap.unwrap_or(header.as_ref().cap);
    let len = header.as_ref().len;

    if len > cap {
        return Err(capacity_error());
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

    clone.set_len(len);

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

pub unsafe fn clear<T>(ptr: NonNull<Header>) {
    let len = ptr.as_ref().len;

    let mut item = data_ptr::<T>(ptr);
    for _ in 0..len {
        std::ptr::drop_in_place(item);
        item = item.add(1);
    }
}

pub unsafe fn dealloc<T>(ptr: NonNull<Header>) {
    clear::<T>(ptr);

    let cap = ptr.as_ref().cap as usize;
    let layout = buffer_layout::<T>(cap).unwrap();

    alloc::dealloc(ptr.as_ptr() as *mut u8, layout);
}

#[test]
fn buffer_layout_alignemnt() {
    type B = Box<u32>;
    let layout = buffer_layout::<B>(2).unwrap();
    assert_eq!(layout.align(), mem::size_of::<B>())
}

#[cold]
fn capacity_error() -> AllocError {
    AllocError::CapacityOverflow
}
