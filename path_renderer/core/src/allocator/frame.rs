use std::cell::RefCell;
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicI32, Ordering};

use crate::allocator::chunk::ChunkPool;

use super::{Allocator, AllocError, Layout, Global};
use super::bump::BumpAllocator;

pub type TlFframeVec<'frame, T> = allocator_api2::vec::Vec<T, &'frame ThreadLocalFrameAllocator>;

// TODO: multi-threading
// - ref-counted: one allocator per thread or a shared bump allocator?
// - lifetime'd FrameAllocator: How to create one per worker?

pub struct FrameAllocators {
    pub local: ThreadLocalFrameAllocator,
    //pub shared: SharedFrameAllocator,
    // pub tmp: TmpAllocator,
}

impl FrameAllocators {
    pub fn new(chunks: ChunkPool) -> FrameAllocators {
        FrameAllocators {
            local: ThreadLocalFrameAllocator::new(chunks.clone()),
            //shared: SharedFrameAllocator::new(chunks),
        }
    }
}

// TODO: should this have a lifetime?
pub struct ThreadLocalFrameAllocator {
    bump: RefCell<BumpAllocator>,
    _no_send_sync: PhantomData<*const ()>,
}

impl ThreadLocalFrameAllocator {
    fn new(chunks: ChunkPool) -> Self {
        ThreadLocalFrameAllocator {
            bump: RefCell::new(BumpAllocator::new(chunks)),
            _no_send_sync: PhantomData,
        }
    }

    pub fn new_vec<T>(&self) -> TlFframeVec<T> {
        TlFframeVec::new_in(self)
    }

    pub fn new_vec_with_capacity<T>(&self, cap: usize) -> TlFframeVec<T> {
        TlFframeVec::with_capacity_in(cap, self)
    }

    pub fn push_scope<'me>(
        &'me self,
        scope: impl FnOnce(&ThreadLocalFrameAllocator),
    ) {
        let saved = self.bump.borrow().save();

        scope(self);

        unsafe {
            self.bump.borrow_mut().restore(saved);
        }
    }
}

unsafe impl<'l> Allocator for &'l ThreadLocalFrameAllocator {
    #[inline(never)]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.bump.borrow_mut().allocate(layout)
    }

    #[inline(never)]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        self.bump.borrow_mut().deallocate(ptr, layout)
    }

    #[inline(never)]
    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout
    ) -> Result<NonNull<[u8]>, AllocError> {
        unsafe {
            self.bump.borrow_mut().grow(ptr, old_layout, new_layout)
        }
    }

    #[inline(never)]
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout
    ) -> Result<NonNull<[u8]>, AllocError> {
        unsafe {
            self.bump.borrow_mut().shrink(ptr, old_layout, new_layout)
        }
    }
}

struct FrameBumpAllocatorImpl {
    bump: BumpAllocator,
    refs: AtomicI32,
}

// TODO: this leaks a chunk.
pub struct SharedFrameAllocator {
    allocator: *mut FrameBumpAllocatorImpl,
}

impl SharedFrameAllocator {
    pub fn new(chunks: ChunkPool) -> Self {
        let layout = Layout::new::<FrameBumpAllocatorImpl>();

        let uninit_u8 = Global.allocate(layout).unwrap();

        unsafe {
            let allocator: NonNull<FrameBumpAllocatorImpl> = uninit_u8.cast();
            allocator.as_ptr().write(FrameBumpAllocatorImpl {
                bump: BumpAllocator::new(chunks.clone()),
                refs: AtomicI32::new(1),
            });

            SharedFrameAllocator {
                allocator: allocator.as_ptr(),
            }
        }
    }
}

unsafe impl Allocator for SharedFrameAllocator {
    #[inline(never)]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if self.allocator.is_null() {
            unimplemented!()
        }

        unsafe {
            (*self.allocator).bump.allocate(layout)
        }
    }

    #[inline(never)]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        if self.allocator.is_null() {
            unimplemented!()
        }

        unsafe {
            (*self.allocator).bump.deallocate(ptr, layout)
        }
    }

    #[inline(never)]
    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout
    ) -> Result<NonNull<[u8]>, AllocError> {
        if self.allocator.is_null() {
            unimplemented!()
        }

        unsafe {
            (*self.allocator).bump.grow(ptr, old_layout, new_layout)
        }
    }

    #[inline(never)]
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout
    ) -> Result<NonNull<[u8]>, AllocError> {
        if self.allocator.is_null() {
            unimplemented!()
        }

        unsafe {
            (*self.allocator).bump.shrink(ptr, old_layout, new_layout)
        }
    }
}

impl Clone for SharedFrameAllocator {
    fn clone(&self) -> SharedFrameAllocator {
        unsafe {
            if let Some(allocator) = self.allocator.as_ref() {
                allocator.refs.fetch_add(1, Ordering::Relaxed);
            }
        }
        SharedFrameAllocator { allocator: self.allocator }
    }
}

impl Drop for SharedFrameAllocator {
    fn drop(&mut self) {
        unsafe {
            if let Some(allocator) = self.allocator.as_ref() {
                allocator.refs.fetch_sub(1, Ordering::Relaxed);
            }
        }
    }
}

#[test]
fn frame_allocator_01() {
    let allocators = FrameAllocators::new(ChunkPool::new());

    let mut v1 = allocators.local.new_vec();
    for i in 0..256u32 {
        v1.push(i);
    }


    let mut v2 = allocators.local.new_vec_with_capacity(8);

    for i in 0..256u32 {
        v2.push(i);
    }
}

#[test]
fn stacked_allocator_01() {
    let allocators = FrameAllocators::new(ChunkPool::new());
    const N: usize = 256;

    let mut offset: usize = 0;
    let mut v0 = allocators.local.new_vec_with_capacity(16);
    for i in offset..(offset + N) {
        v0.push(i);
    }
    let o0 = offset;
    offset += N;
    allocators.local.push_scope(|allocator| {
        let mut v1 = allocator.new_vec_with_capacity(16);
        for i in offset..(offset + N) {
            v1.push(i);
        }
        let o1 = offset;
        offset += N;

        allocator.push_scope(|allocator| {
            let mut v2 = allocator.new_vec_with_capacity(16);
            let mut v3 = allocator.new_vec_with_capacity(16);
            for i in offset..(offset + N) {
                v2.push(i);
            }
            let o2 = offset;
            offset += N;
            for i in offset..(offset + N) {
                v3.push(i);
            }
            let o3 = offset;
            offset += N;

            for i in 0..N {
                assert_eq!(v2[i], o2 + i);
                assert_eq!(v3[i], o3 + i);
            }
        });

        for i in offset..(offset+N) {
            v1.push(i);
        }
        let o1b = offset;
        offset += N;

        let mut v4 = allocator.new_vec_with_capacity(16);
        for i in offset..(offset + N) {
            v4.push(i);
        }
        let o4 = offset;
        offset += N;

        for i in 0..(N as usize) {
            assert_eq!(v1[i], o1 + i);
            assert_eq!(v1[i + N], o1b + i);
            assert_eq!(v4[i], o4 + i);
        }
    });

    let mut v5 = allocators.local.new_vec_with_capacity(16);
    for i in offset..(offset + N) {
        v5.push(i);
    }

    for i in 0..N {
        assert_eq!(v0[i], o0 + i);
        assert_eq!(v5[i], offset + i);
    }
}
