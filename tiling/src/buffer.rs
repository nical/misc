
use std::ops::{Range, Index, IndexMut};

pub struct Buffer<T> {
    ptr: *mut T,
    len: u32,
    cap: u32,
    idx: u32,
}


impl<T> Buffer<T> {
    #[inline]
    pub fn empty() -> Self {
        Buffer {
            ptr: std::ptr::null_mut(),
            len: 0,
            cap: 0,
            idx: std::u32::MAX,
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len()) }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len()) }
    }

    #[inline]
    pub fn get(&self, idx: usize) -> Option<&T> {
        if idx < self.len() {
            return None;
        }

        unsafe {
            Some(&*self.ptr.offset(idx as isize))
        }
    }

    #[inline]
    pub fn len(&self) -> usize { self.len as usize }

    #[inline]
    pub fn capacity(&self) -> usize { self.cap as usize }

    #[inline]
    pub fn remaining_capacity(&self) -> usize { (self.cap - self.len) as usize }

    #[inline]
    pub fn is_full(&self) -> bool { self.remaining_capacity() == 0 }

    #[inline]
    pub fn index(&self) -> u32 { self.idx }

    #[inline]
    pub fn push(&mut self, val: T) {
        assert!(self.len < self.cap);
        unsafe {
            self.push_unchecked(val);
        }
    }

    #[inline]
    pub unsafe fn push_unchecked(&mut self, val: T) {
        std::ptr::write(self.ptr.offset(self.len as isize), val);
        self.len += 1;
    }

    #[inline]
    pub unsafe fn set_len(&mut self, len: usize) {
        let len = len as u32;
        assert!(len <= self.cap, "length ({:?}) < capacity ({:?})", len, self.cap);

        self.len = len;
    }

    #[inline]
    pub fn as_ptr(&self) -> *const T { self.ptr }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T { self.ptr }

    pub fn clear(&mut self) {
        unsafe {
            let mut vec = Vec::from_raw_parts(self.ptr, self.len(), self.capacity());
            vec.clear();
            std::mem::forget(vec);
        }
    }

    fn forget(&mut self) {
        self.len = 0;
        self.cap = 0;
        self.ptr = std::ptr::null_mut();
    }
}

unsafe impl<T: Send> Send for Buffer<T> {}
unsafe impl<T: Sync> Sync for Buffer<T> {}

impl<T> Drop for Buffer<T> {
    fn drop(&mut self) {
        assert!(self.cap == 0, "Leaked buffer");
    }
}

impl<T> AsRef<[T]> for Buffer<T> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> AsMut<[T]> for Buffer<T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T> Index<usize> for Buffer<T> {
    type Output = T;
    fn index(&self, idx: usize) -> &T {
        assert!(idx < self.len());
        unsafe {
            &*self.ptr.offset(idx as isize)
        }
    }
}

impl<T> IndexMut<usize> for Buffer<T> {
    fn index_mut(&mut self, idx: usize) -> &mut T {
        assert!(idx < self.len());
        unsafe {
            &mut *self.ptr.offset(idx as isize)
        }
    }
}

impl<T> Index<Range<usize>> for Buffer<T> {
    type Output = [T];
    fn index(&self, range: Range<usize>) -> &[T] {
        &self.as_slice()[range]
    }
}

impl<T> IndexMut<Range<usize>> for Buffer<T> {
    fn index_mut(&mut self, range: Range<usize>) -> &mut [T] {
        &mut self.as_mut_slice()[range]
    }
}

impl<T> Index<Range<u32>> for Buffer<T> {
    type Output = [T];
    fn index(&self, range: Range<u32>) -> &[T] {
        let range = (range.start as usize) .. (range.end as usize);
        &self.as_slice()[range]
    }
}

impl<T> IndexMut<Range<u32>> for Buffer<T> {
    fn index_mut(&mut self, range: Range<u32>) -> &mut [T] {
        let range = (range.start as usize) .. (range.end as usize);
        &mut self.as_mut_slice()[range]
    }
}

pub struct BufferPool<T> {
    available_buffers: Vec<Buffer<T>>,
    size: u32,
}

impl<T> BufferPool<T> {
    pub fn new(size: u32) -> Self {
        BufferPool {
            available_buffers: Vec::new(),
            size,
        }
    }

    fn allocate_buffer(&mut self) -> Buffer<T> {
        let mut alloc = Vec::with_capacity(self.size as usize);
        let ptr = alloc.as_mut_ptr();
        let cap = alloc.capacity() as u32;
        let idx = self.available_buffers.len() as u32;

        let buffer = Buffer {
            ptr,
            cap,
            len: 0,
            idx,
        };

        std::mem::forget(alloc);

        buffer
    }

    fn deallocate_buffer(&mut self, mut buffer: Buffer<T>) {
        unsafe {
            let _ = Vec::from_raw_parts(buffer.ptr, buffer.len(), buffer.capacity());
        }
        buffer.forget();
    }

    pub fn get_buffer(&mut self) -> Buffer<T> {
        if let Some(buffer) = self.available_buffers.pop() {
            return buffer;
        }

        self.allocate_buffer()
    }

    pub fn return_buffer(&mut self, buffer: Buffer<T>) {
        self.available_buffers.push(buffer);
    }
}

impl<T> Drop for BufferPool<T> {
    fn drop(&mut self) {
        while let Some(buffer) = self.available_buffers.pop() {
            self.deallocate_buffer(buffer);
        }
    }
}

struct GpuBuffer {
    handle: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    slice: Option<wgpu::BufferSlice<'static>>,
    view: Option<wgpu::BufferViewMut<'static>>,
}

pub struct UniformBufferPool<T> {
    available_buffers: Vec<Box<GpuBuffer>>,
    used_buffers: Vec<Box<GpuBuffer>>,
    cap: u32,
    device: *const wgpu::Device,
    bind_group_layout: *const wgpu::BindGroupLayout,
    _marker: std::marker::PhantomData<T>,
}

impl<T> UniformBufferPool<T> {
    pub fn new(buffer_size: u32, device: *const wgpu::Device, bind_group_layout: *const wgpu::BindGroupLayout) -> Self {
        UniformBufferPool {
            available_buffers: Vec::new(),
            used_buffers: Vec::new(),
            cap: buffer_size,
            device,
            bind_group_layout,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn create_similar(&self) -> Self {
        UniformBufferPool {
            available_buffers: Vec::new(),
            used_buffers: Vec::new(),
            cap: self.cap,
            device: self.device,
            bind_group_layout: self.bind_group_layout,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn reset(&mut self) {
        self.deallocate_buffers();
    }

    pub fn get_buffer(&mut self) -> Buffer<T> {

        if self.available_buffers.is_empty() {
            unsafe {
                let handle = (*self.device).create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Staging buffer"),
                    size: self.cap as u64,
                    usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: true,
                });

                let bind_group = (*self.device).create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &*self.bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(handle.as_entire_buffer_binding()),
                        },
                    ],
                });

                let mut buffer = Box::new(GpuBuffer {
                    handle,
                    bind_group,
                    slice: None,
                    view: None,
                });

                // Transmute away the lifetime of the slice and view. We guarantee that the handle won't move or be deleted
                // before they are dropped.
                buffer.slice = Some(std::mem::transmute(buffer.handle.slice(..)));
                buffer.view = Some(std::mem::transmute(buffer.slice.as_mut().unwrap().get_mapped_range_mut()));

                self.available_buffers.push(buffer);
            }
        }

        let mut buffer = self.available_buffers.pop().unwrap();
        let view = buffer.view.as_mut().unwrap();

        let ptr = unsafe { std::mem::transmute(view.as_mut_ptr()) };
        let idx = self.used_buffers.len() as u32;

        self.used_buffers.push(buffer);

        Buffer {
            ptr,
            cap: self.cap,
            len: 0,
            idx,
        }
    }

    pub fn return_buffer(&mut self, mut buffer: Buffer<T>) -> &wgpu::Buffer {
        assert!(buffer.capacity() > 0);

        let idx = buffer.idx as usize;

        buffer.forget();

        let gpu_buffer = &mut self.used_buffers[idx];
        gpu_buffer.view = None;
        gpu_buffer.slice = None;
        gpu_buffer.handle.unmap();

        &gpu_buffer.handle
    }

    pub fn get_bind_group(&self, idx: u32) -> &wgpu::BindGroup {
        &self.used_buffers[idx as usize].bind_group
    }

    fn deallocate_buffers(&mut self) {
        for mut buffer in self.available_buffers.drain(..) {
            buffer.slice = None;
            buffer.handle.destroy();
        }

        for mut buffer in self.used_buffers.drain(..) {
            buffer.slice = None;
            buffer.handle.destroy();
        }
    }
}

impl<T> Drop for UniformBufferPool<T> {
    fn drop(&mut self) {
        self.deallocate_buffers();
    }
}


#[test]
fn buffer_pool() {
    let mut pool: BufferPool<u32> = BufferPool::new(2048);

    let mut b0 = pool.get_buffer();
    let b1 = pool.get_buffer();

    assert_eq!(b0.as_slice(), &[]);
    assert_eq!(b1.as_slice(), &[]);

    assert_eq!(b0.len(), 0);
    assert_eq!(b0.capacity(), 2048);
    assert_eq!(b0.remaining_capacity(), 2048);

    b0.push(0);
    b0.push(1);
    b0.push(2);

    assert_eq!(b0[0], 0);
    assert_eq!(b0[1], 1);
    assert_eq!(b0[2], 2);

    assert_eq!(b0.len(), 3);
    assert_eq!(b0.capacity(), 2048);
    assert_eq!(b0.remaining_capacity(), 2048 - 3);

    pool.return_buffer(b0);
    pool.return_buffer(b1);
}

