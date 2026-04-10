use crate::allocator::{AllocError, Allocator};
use crate::util::{self, is_zst};
use core::marker::PhantomData;
use core::mem;
use core::ptr::{self, NonNull};
use core::fmt::Debug;

// Note: switching len and cap to use u32 causes a 10% regression on the push
// benchmark.

//            |      100 |    1000 |    10 000 |
// +----------+----------+---------+-----------+
// | umanaged | 328.38ns | 2.682us |  24.429us |
// | slim     | 349.26ns | 2.576us |  23.437us |

pub type UnmanagedVector<T> = HeaderVectorBuffer<(), T>;

pub enum AllocInit {
    Uninit,
    Zeroed
}

#[derive(Copy, Clone)]
pub struct VectorMetadata {
    pub len: usize,
    pub cap: usize,
}

impl VectorMetadata {
    #[inline]
    /// Returns `true` if the vector contains no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    /// Returns the number of elements in the vector, also referred to as its ‘length’.
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub unsafe fn set_len(&mut self, new_len: usize) {
        debug_assert!(new_len <= self.capacity());

        self.len = new_len;
    }

    #[inline]
    /// Returns the total number of elements the vector can hold without reallocating.
    pub fn capacity(&self) -> usize {
        self.cap
    }

    /// Returns number of elements that can be added without reallocating.
    #[inline]
    pub fn remaining_capacity(&self) -> usize {
        self.capacity() - self.len()
    }

}

pub struct HeaderVectorBuffer<H, T> {
    data: NonNull<T>,
    _marker: PhantomData<H>,
}

impl<T> HeaderVectorBuffer<(), T> {
    pub fn new() -> (Self, VectorMetadata) {
        (
            HeaderVectorBuffer {
                data: NonNull::dangling(),
                _marker: PhantomData,
            },
            VectorMetadata {
                len: 0,
                cap: 0,
            },
        )
    }
}

impl<H, T> HeaderVectorBuffer<H, T> {
    #[inline(always)]
    fn should_be_dangling(cap: usize) -> bool {
        util::is_zst::<H>() && (util::is_zst::<T>() || cap == 0)
    }

    /// Creates an empty pre-allocated vector with a given storage size.
    ///
    /// The size is the total allocated buffer size including space for
    /// the header if any.
    #[inline(never)]
    pub fn try_with_buffer_size_in<A: Allocator>(
        header: H,
        size: usize,
        init: AllocInit,
        allocator: &A,
    ) -> Result<(Self, VectorMetadata), AllocError> {
        if Self::should_be_dangling(size) {
            if is_zst::<H>() {
                // In case the header has a Drop implementation.
                mem::forget(header);
            }
            return Ok(unsafe { Self::dangling() });
        }

        let layout = util::header_vector_layout::<H, T>(1)?;
        let header_size = util::header_size::<H, T>();
        let t_size = mem::size_of::<T>();
        if size < header_size + t_size {
            return Err(AllocError);
        }

        let layout = crate::allocator::Layout::from_size_align(size, layout.align()).unwrap();

        let allocation = match init {
            AllocInit::Uninit => allocator.allocate(layout),
            AllocInit::Zeroed => allocator.allocate_zeroed(layout),
        }?;

        let items_size = allocation.len() - header_size;
        let cap = if t_size > 0 {
            items_size / t_size
        } else {
            isize::MAX as usize
        };

        let data = unsafe { allocation.cast::<u8>().add(header_size).cast::<T>() };

        if is_zst::<H>() {
            // In case the header has a Drop implementation.
            mem::forget(header);
        } else {
            unsafe {
                let header_ptr = util::get_header::<H, T>(data);
                header_ptr.write(header);
            }
        }

        Ok((
            HeaderVectorBuffer {
                data,
                _marker: PhantomData,
            },
            VectorMetadata {
                len: 0,
                cap
            },
        ))
    }

    /// Creates an empty pre-allocated vector with a given storage size.
    #[inline(never)]
    pub fn with_buffer_size_in<A: Allocator>(
        header: H,
        size: usize,
        init: AllocInit,
        allocator: &A,
    ) -> (Self, VectorMetadata) {
        Self::try_with_buffer_size_in(header, size, init, allocator).unwrap()
    }

    /// Creates an empty pre-allocated vector with a given storage capacity.
    #[inline(never)]
    pub fn try_with_capacity_in<A: Allocator>(
        header: H,
        mut cap: usize,
        init: AllocInit,
        allocator: &A,
    ) -> Result<(Self, VectorMetadata), AllocError> {
        cap = cap.max(crate::MIN_CAPACITY);
        let header_size = util::header_size::<H, T>();
        let size = header_size + cap * mem::size_of::<T>();
        Self::try_with_buffer_size_in(header, size, init, allocator)
    }

    pub fn with_capacity_in<A: Allocator>(
        header: H,
        cap: usize,
        init: AllocInit,
        allocator: &A,
    ) -> (Self, VectorMetadata) {
        Self::try_with_capacity_in(header, cap, init, allocator).unwrap()
    }

    #[inline]
    pub unsafe fn dangling() -> (Self, VectorMetadata) {
        let cap = if util::is_zst::<T>() {
            isize::MAX as usize
        } else {
            0
        };

        (
            HeaderVectorBuffer {
                data: NonNull::dangling(),
                _marker: PhantomData,
            },
            VectorMetadata {
                len: 0,
                cap,
            }
        )
    }

    pub fn try_from_slice<A: Allocator>(
        header: H,
        data: &[T],
        allocator: &A,
    ) -> Result<(Self, VectorMetadata), AllocError>
    where
        T: Clone,
    {
        let (mut v, mut md) = Self::try_with_capacity_in(header, data.len(), AllocInit::Uninit, allocator)?;
        unsafe {
            v.extend_from_slice(&mut md, data, allocator);
        }

        Ok((v, md))
    }

    /// Tries to reserve at least enough space for `additional` extra items.
    ///
    /// # Safety
    ///
    /// The provided allocator must be the one this raw vector was created with.
    #[inline]
    pub unsafe fn try_reserve<A: Allocator>(
        &mut self,
        metadata: &mut VectorMetadata,
        additional: usize,
        allocator: &A,
    ) -> Result<(), AllocError> {
        if metadata.remaining_capacity() < additional {
            self.try_realloc_additional(metadata, additional, allocator)?;
        }

        Ok(())
    }

    /// Clears and deallocates this raw vector, leaving it in its unallocated state.
    ///
    /// It is safe (no-op) to call `deallocate` on a vector that is already in its unallocated state.
    ///
    /// # Safety
    ///
    /// The provided allocator must be the one this raw vector was created with.
    pub unsafe fn deallocate_in<A: Allocator>(&mut self, metadata: &mut VectorMetadata, allocator: &A) {
        self.clear(metadata);

        unsafe {
            core::ptr::drop_in_place(self.header_ptr().as_ptr());

            if Self::should_be_dangling(metadata.cap) {
                return;
            }

            let allocation = self.header_ptr().cast::<u8>();

            let layout = util::header_vector_layout::<H, T>(metadata.cap).unwrap();
            allocator.deallocate(allocation.cast::<u8>(), layout);
        }

        let (new_self, new_md) = Self::dangling();
        *self = new_self;
        *metadata = new_md;
    }

    #[inline]
    pub fn ptr_eq(&self, other: &Self) -> bool {
        self.data == other.data
    }

    pub fn clone_in<A: Allocator>(&self, metadata: &VectorMetadata, allocator: &A, new_cap: usize) -> (Self, VectorMetadata)
    where
        H: Clone,
        T: Clone,
    {
        self.try_clone_in(metadata, allocator, new_cap).unwrap()
    }

    pub fn try_clone_in<A: Allocator>(
        &self,
        metadata: &VectorMetadata,
        allocator: &A,
        mut new_cap: usize,
    ) -> Result<(Self, VectorMetadata), AllocError>
    where
        H: Clone,
        T: Clone,
    {
        new_cap = new_cap.max(metadata.len);
        unsafe {
            if Self::should_be_dangling(new_cap) {
                return Ok(HeaderVectorBuffer::dangling());
            }

            let header_size = util::header_size::<H, T>();
            let t_size = mem::size_of::<T>();
            let old_empty = (t_size == 0 || metadata.len == 0) && header_size == 0;

            if old_empty {
                return HeaderVectorBuffer::try_with_capacity_in(
                    self.header().clone(),
                    new_cap,
                    AllocInit::Uninit,
                    allocator,
                );
            }

            let new_layout = util::header_vector_layout::<H, T>(new_cap).unwrap();

            let new_alloc = allocator.allocate(new_layout)?;
            let new_items_ptr = new_alloc.cast::<u8>().add(header_size).cast::<T>();

            if !is_zst::<H>() {
                let old_header = util::get_header::<H, T>(self.data);
                let new_header = util::get_header::<H, T>(new_items_ptr);
                core::ptr::write(new_header.as_ptr(), old_header.as_ref().clone());
            }

            if !is_zst::<T>() {
                let mut src = self.data.as_ptr();
                let mut dst = new_items_ptr.as_ptr();
                for _ in 0..metadata.len {
                    core::ptr::write(dst, (*src).clone());
                    src = src.add(1);
                    dst = dst.add(1);
                }
            }

            Ok((
                HeaderVectorBuffer {
                    data: new_items_ptr,
                    _marker: PhantomData,
                },
                VectorMetadata {
                    len: metadata.len,
                    cap: new_cap,
                },
            ))
        }
    }

    // Note: Marking this #[inline(never)] is a pretty large regression in the push benchmark.
    #[cold]
    pub(crate) unsafe fn try_realloc_additional<A: Allocator>(
        &mut self,
        metadata: &mut VectorMetadata,
        additional: usize,
        allocator: &A,
    ) -> Result<(), AllocError> {
        let new_cap = util::grow_amortized(metadata.len, additional)?;
        if new_cap < metadata.len {
            return Err(AllocError);
        }

        self.try_realloc_with_capacity(metadata, new_cap, allocator)
    }

    #[cold]
    pub(crate) unsafe fn try_realloc_with_capacity<A: Allocator>(
        &mut self,
        metadata: &mut VectorMetadata,
        new_cap: usize,
        allocator: &A,
    ) -> Result<(), AllocError> {
        let old_cap = metadata.cap;

        let old_dangling = Self::should_be_dangling(old_cap);
        let new_dangling = Self::should_be_dangling(new_cap);

        if new_dangling && !old_dangling {
            self.deallocate_in(metadata, allocator);
            return Ok(());
        }

        if old_dangling {
            // According to https://doc.rust-lang.org/nomicon/vec/vec-zsts.html
            // reading a ZST from NonNull::dangling is fine.
            let header = self.header_ptr().read();
            let (new_self, new_metadata) = HeaderVectorBuffer::try_with_capacity_in(header, new_cap, AllocInit::Uninit, allocator)?;
            *self = new_self;
            *metadata = new_metadata;
            return Ok(());
        }

        unsafe {
            let new_layout = util::header_vector_layout::<H, T>(new_cap).unwrap();

            let new_alloc = if old_cap == 0 {
                allocator.allocate(new_layout)?
            } else {
                let old_layout = util::header_vector_layout::<H, T>(old_cap).unwrap();
                let old_alloc = self.alloc_ptr();

                if new_layout.size() >= old_layout.size() {
                    allocator.grow(old_alloc, old_layout, new_layout)
                } else {
                    allocator.shrink(old_alloc, old_layout, new_layout)
                }?
            };

            let header_size = util::header_size::<H, T>();
            self.data = new_alloc.cast::<u8>().add(header_size).cast::<T>();
            metadata.cap = new_cap;
        }

        Ok(())
    }

    /// Attempts to move this vector into another allocator.
    ///
    /// All of the data is copied over into a new allocation in `new_allocator` after which
    /// the old allocation is deallocated from `old_allocator`.
    ///
    /// If this method succeeds, `new_allocator` becomes the allocator currently used by this
    /// vector.
    ///
    /// If the old and new allocator are the same, this method works but is likely less
    /// efficient than `try_realloc_with_capacity`.
    ///
    /// # Safety
    ///
    /// The provided `old_allocator` must be the one currently used by this vector.
    ///
    /// # Error
    ///
    /// If reallocation fails:
    ///  - The vector remains in its current state, still associated to the old allocator.
    ///  - An allocation error is returned.
    #[cold]
    pub unsafe fn try_realloc_in_new_allocator<OldAllocator, NewAllocator>(
        &mut self,
        metadata: &mut VectorMetadata,
        new_cap: usize,
        old_allocator: &OldAllocator,
        new_allocator: &NewAllocator,
    ) -> Result<(), AllocError>
    where
        OldAllocator: Allocator,
        NewAllocator: Allocator,
    {
        if Self::should_be_dangling(new_cap) {
            self.deallocate_in(metadata, old_allocator);
            return Ok(());
        }

        let new_layout = util::header_vector_layout::<H, T>(new_cap)?;
        let new_buffer = new_allocator.allocate(new_layout)?.cast::<u8>();

        let old_len = metadata.len;
        if old_len > new_cap {
            self.truncate(metadata, new_cap);
        }

        let old_cap = metadata.cap;
        let header_size = util::header_size::<H, T>();

        let old_buffer = if !Self::should_be_dangling(old_cap) {
            let old_buffer = self.alloc_ptr();
            let copy_size = header_size + old_len.min(new_cap) * mem::size_of::<T>();
            if copy_size > 0 {
                ptr::copy_nonoverlapping(
                    old_buffer.as_ptr(),
                    new_buffer.as_ptr(),
                    copy_size,
                );
            }

            Some(old_buffer)
        } else {
            None
        };

        self.data = new_buffer.add(header_size).cast::<T>();
        metadata.cap = new_cap;
        metadata.len = metadata.len.min(new_cap);

        if let Some(old_alloc) = old_buffer {
            let old_layout = util::header_vector_layout::<H, T>(old_cap).unwrap();
            old_allocator.deallocate(old_alloc, old_layout);
        }

        Ok(())
    }

    pub unsafe fn try_shrink_to<A: Allocator>(
        &mut self,
        metadata: &mut VectorMetadata,
        new_cap: usize,
        allocator: &A,
    ) -> Result<(), AllocError> {
        let new_cap = new_cap.max(metadata.len);
        if metadata.cap <= new_cap {
            return Ok(());
        }

        self.try_realloc_with_capacity(metadata, new_cap, allocator)
    }

    pub unsafe fn shrink_to<A: Allocator>(&mut self, metadata: &mut VectorMetadata, new_cap: usize, allocator: &A) {
        self.try_shrink_to(metadata, new_cap, allocator).unwrap()
    }

    pub unsafe fn shrink_to_fit<A: Allocator>(&mut self, metadata: &mut VectorMetadata, allocator: &A) {
        self.try_shrink_to(metadata, metadata.len, allocator).unwrap()
    }

    pub fn truncate(&mut self, metadata: &mut VectorMetadata, new_len: usize) {
        if metadata.len <= new_len {
            return;
        }

        unsafe {
            let ptr = self.data.as_ptr().add(new_len);
            let elems = core::ptr::slice_from_raw_parts_mut(ptr, metadata.len - new_len);
            core::ptr::drop_in_place(elems);
        }

        metadata.len = new_len;
    }

    #[inline(always)]
    pub fn header(&self) -> &H {
        unsafe { self.header_ptr().as_ref() }
    }

    #[inline(always)]
    pub fn header_mut(&mut self) -> &mut H {
        unsafe { self.header_ptr().as_mut() }
    }

    #[inline(always)]
    pub(crate) fn header_ptr(&self) -> NonNull<H> {
        unsafe { util::get_header::<H, T>(self.data) }
    }

    #[inline(always)]
    pub(crate) fn alloc_ptr(&self) -> NonNull<u8> {
        unsafe { util::get_header::<H, T>(self.data).cast::<u8>() }
    }

    #[inline]
    pub fn as_slice(&self, metadata: &VectorMetadata) -> &[T] {
        unsafe { core::slice::from_raw_parts(self.data.as_ptr(), metadata.len) }
    }

    #[inline]
    pub fn as_mut_slice(&mut self, metadata: &VectorMetadata) -> &mut [T] {
        unsafe { core::slice::from_raw_parts_mut(self.data.as_ptr(), metadata.len) }
    }

    /// Clears the vector, removing all values.
    pub fn clear(&mut self, metadata: &mut VectorMetadata) {
        let elems: *mut [T] = self.as_mut_slice(metadata);
        unsafe {
            metadata.len = 0;
            core::ptr::drop_in_place(elems);
        }
    }

    #[inline(always)]
    unsafe fn item_ptr(&self, index: usize) -> NonNull<T> {
        self.data.add(index)
    }

    #[inline(always)]
    unsafe fn write_item(&mut self, metadata: &VectorMetadata, index: usize, val: T) {
        debug_assert!(index < metadata.cap);
        let dst = self.item_ptr(index);
        dst.write(val);
    }

    #[inline(always)]
    unsafe fn read_item(&self, metadata: &VectorMetadata, index: usize) -> T {
        debug_assert!(index < metadata.cap);
        let dst = self.item_ptr(index);
        dst.read()
    }

    /// Appends an element to the back of a collection.
    ///
    /// # Safety
    ///
    /// The provided allocator must be the one this raw vector was created with.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity exceeds `u32::MAX` bytes.
    #[inline(always)]
    pub unsafe fn push<A: Allocator>(&mut self, metadata: &mut VectorMetadata, val: T, allocator: &A) {
        // Inform codegen that the length does not change across rtry_realloc_additional.
        let len = metadata.len;

        if len == metadata.cap {
            self.try_realloc_additional(metadata, 1, allocator).unwrap();
        }

        self.write_item(metadata, len, val);
        metadata.len += 1;
    }

    /// Appends an element if there is sufficient spare capacity, otherwise an error is returned
    /// with the element.
    ///
    /// Unlike push this method will not reallocate when there’s insufficient capacity.
    /// The caller should use reserve or try_reserve to ensure that there is enough capacity.
    #[inline(always)]
    pub fn push_within_capacity(&mut self, metadata: &mut VectorMetadata, val: T) -> Result<(), T> {
        if metadata.len == metadata.cap {
            return Err(val);
        }

        unsafe {
            self.push_assuming_capacity(metadata, val);
        }

        Ok(())
    }

    #[inline(always)]
    pub unsafe fn push_assuming_capacity(&mut self, metadata: &mut VectorMetadata, val: T) {
        debug_assert!(metadata.len < metadata.cap);

        unsafe {
            self.write_item(metadata, metadata.len, val);
        }
        metadata.len += 1;
    }

    /// Removes the last element from the vector and returns it, or `None` if it is empty.
    #[inline(always)]
    pub fn pop(&mut self, metadata: &mut VectorMetadata) -> Option<T> {
        if metadata.len == 0 {
            return None;
        }

        metadata.len -= 1;
        unsafe { Some(self.read_item(metadata, metadata.len)) }
    }

    #[inline(always)]
    pub fn get(&self, metadata: &VectorMetadata, index: usize) -> Option<&T> {
        if index >= metadata.len {
            return None;
        }

        unsafe { Some(self.get_unchecked(index)) }
    }

    #[inline(always)]
    pub unsafe fn get_unchecked(&self, index: usize) -> &T {
        unsafe { self.item_ptr(index).as_ref() }
    }

    #[inline(always)]
    pub fn get_mut(&mut self, metadata: &mut VectorMetadata, index: usize) -> Option<&mut T> {
        if index >= metadata.len {
            return None;
        }

        unsafe { Some(self.get_unchecked_mut(index)) }
    }

    #[inline(always)]
    pub unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T {
        self.item_ptr(index).as_mut()
    }

    /// Removes and returns the element at position `index` within the vector,
    /// shifting all elements after it to the left.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    ///
    pub fn remove(&mut self, metadata: &mut VectorMetadata, index: usize) -> T {
        unsafe {
            #[cold]
            #[inline(never)]
            #[track_caller]
            fn assert_failed(index: usize, len: usize) -> ! {
                panic!("remove: index {index} should be < len {len}.");
            }

            if index >= metadata.len {
                assert_failed(index, metadata.len);
            }

            // infallible
            let ret;
            {
                // the place we are taking from.
                let ptr = self.item_ptr(index);
                // copy it out, unsafely having a copy of the value on
                // the stack and in the vector at the same time.
                ret = ptr.read();

                // Shift everything down to fill in that spot.
                ptr.add(1).copy_to(ptr, metadata.len - index - 1);
            }

            metadata.len -= 1;

            ret
        }
    }

    /// Removes an element from the vector and returns it.
    ///
    /// The removed element is replaced by the last element of the vector.
    ///
    /// # Panics
    ///
    /// Panics if index is out of bounds.
    #[inline]
    pub fn swap_remove(&mut self, metadata: &mut VectorMetadata, index: usize) -> T {
        unsafe {
            assert!(index < metadata.len);

            let ptr = self.item_ptr(index);
            let item = ptr.read();

            let last_idx = metadata.len - 1;
            if index != last_idx {
                let last_ptr = self.item_ptr(last_idx);
                ptr.write(last_ptr.read());
            }

            metadata.len -= 1;

            item
        }
    }

    /// Inserts an element at position `index` within the vector, shifting all
    /// elements after it to the right.
    ///
    /// # Panics
    ///
    /// Panics if `index > len`.
    pub unsafe fn insert<A: Allocator>(&mut self, metadata: &mut VectorMetadata, index: usize, element: T, allocator: &A) {
        #[cold]
        #[inline(never)]
        fn assert_failed(index: usize, len: usize) -> ! {
            panic!("insertion index (is {index}) should be <= len (is {len})");
        }

        unsafe {
            // Space for the new element
            if metadata.len == metadata.cap {
                self.try_reserve(metadata, 1, allocator).unwrap();
            }

            let len = metadata.len;

            // Infallible
            // The spot to put the new value
            {
                let p = self.item_ptr(index);
                if index < len {
                    // Shift everything over to make space. (Duplicating the
                    // `index`th element into two consecutive places.)
                    p.copy_to(p.add(1), len - index);
                } else if index == len {
                    // No elements need shifting.
                } else {
                    assert_failed(index, len);
                }
                // Write it in, overwriting the first copy of the `index`th
                // element.
                p.write(element);
            }
            metadata.len += 1;
        }
    }

    /// Clones and appends the contents of the slice to the back of a collection.
    ///
    /// # Safety
    ///
    /// The provided allocator must be the one this raw vector was created with.
    pub unsafe fn extend_from_slice<A: Allocator>(&mut self, metadata: &mut VectorMetadata, slice: &[T], allocator: &A)
    where
        T: Clone,
    {
        self.try_extend_from_slice(metadata, slice, allocator).unwrap();
    }

    pub unsafe fn try_extend_from_slice<A: Allocator>(
        &mut self,
        metadata: &mut VectorMetadata,
        slice: &[T],
        allocator: &A,
    ) -> Result<(), AllocError>
    where
        T: Clone,
    {
        self.try_reserve(metadata, slice.len(), allocator)?;
        self.extend_from_slice_assuming_capacity(metadata, slice);

        Ok(())
    }

    pub unsafe fn extend_from_slice_within_capacity(&mut self, metadata: &mut VectorMetadata, slice: &[T])
    where
        T: Clone,
    {
        let n = metadata.remaining_capacity().min(slice.len());
        if n > 0 {
            self.extend_from_slice_assuming_capacity(metadata, &slice[..n]);
        }
    }

    pub fn extend_from_slice_assuming_capacity(&mut self, metadata: &mut VectorMetadata, slice: &[T])
    where
        T: Clone,
    {
        assert!(metadata.cap - metadata.len >= slice.len());
        unsafe {
            let mut ptr = self.item_ptr(metadata.len);

            for item in slice {
                ptr.write(item.clone());
                ptr = ptr.add(1)
            }
        }
        metadata.len += slice.len();
    }

    pub fn into_raw_parts(self, metadata: VectorMetadata) -> (NonNull<T>, usize, usize) {
        (
            self.data,
            metadata.len,
            metadata.cap,
        )
    }

    pub unsafe fn from_raw_parts(data: NonNull<T>, len: usize, cap: usize) -> (Self, VectorMetadata) {
        (
            HeaderVectorBuffer { data, _marker: PhantomData },
            VectorMetadata { len, cap },
        )
    }
}

impl<H, T> Copy for HeaderVectorBuffer<H, T> {}

impl<H, T> Clone for HeaderVectorBuffer<H, T> {
    fn clone(&self) -> Self {
        *self
    }
}

pub type UnmanagedVector2<T> = UnmanagedHeaderVector2<(), T>;

pub struct UnmanagedHeaderVector2<H, T> {
    buf: HeaderVectorBuffer<H, T>,
    metadata: VectorMetadata,
}

impl<T> UnmanagedHeaderVector2<(), T> {
    #[inline]
    pub fn new() -> Self {
        let (buf, metadata) = HeaderVectorBuffer::new();
        UnmanagedHeaderVector2 { buf, metadata }
    }

    #[inline]
    pub fn take(&mut self) -> Self {
        core::mem::replace(self, Self::new())
    }
}

impl<H, T> UnmanagedHeaderVector2<H, T> {
    #[inline]
    pub fn try_with_buffer_size_in<A: Allocator>(
        header: H,
        size: usize,
        init: AllocInit,
        allocator: &A,
    ) -> Result<Self, AllocError> {
        let (buf, metadata) = HeaderVectorBuffer::try_with_buffer_size_in(header, size, init, allocator)?;
        Ok(UnmanagedHeaderVector2 { buf, metadata })
    }

    #[inline]
    pub fn with_buffer_size_in<A: Allocator>(
        header: H,
        size: usize,
        init: AllocInit,
        allocator: &A,
    ) -> Self {
        Self::try_with_buffer_size_in(header, size, init, allocator).unwrap()
    }

    #[inline]
    pub fn try_with_capacity_in<A: Allocator>(
        header: H,
        cap: usize,
        init: AllocInit,
        allocator: &A,
    ) -> Result<Self, AllocError> {
        let (buf, metadata) = HeaderVectorBuffer::try_with_capacity_in(header, cap, init, allocator)?;
        Ok(UnmanagedHeaderVector2 { buf, metadata })
    }

    #[inline]
    pub fn with_capacity_in<A: Allocator>(
        header: H,
        cap: usize,
        init: AllocInit,
        allocator: &A,
    ) -> Self {
        Self::try_with_capacity_in(header, cap, init, allocator).unwrap()
    }

    #[inline]
    pub fn try_from_slice<A: Allocator>(
        header: H,
        data: &[T],
        allocator: &A,
    ) -> Result<Self, AllocError>
    where
        T: Clone,
    {
        let (buf, metadata) = HeaderVectorBuffer::try_from_slice(header, data, allocator)?;
        Ok(UnmanagedHeaderVector2 { buf, metadata })
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.metadata.is_empty()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.metadata.len()
    }

    #[inline]
    pub unsafe fn set_len(&mut self, new_len: usize) {
        self.metadata.set_len(new_len);
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.metadata.capacity()
    }

    #[inline]
    pub fn remaining_capacity(&self) -> usize {
        self.metadata.remaining_capacity()
    }

    #[inline]
    pub unsafe fn try_reserve<A: Allocator>(
        &mut self,
        additional: usize,
        allocator: &A,
    ) -> Result<(), AllocError> {
        self.buf.try_reserve(&mut self.metadata, additional, allocator)
    }

    #[inline]
    pub unsafe fn deallocate_in<A: Allocator>(&mut self, allocator: &A) {
        self.buf.deallocate_in(&mut self.metadata, allocator);
    }

    #[inline]
    pub fn ptr_eq(&self, other: &Self) -> bool {
        self.buf.ptr_eq(&other.buf)
    }

    #[inline]
    pub fn clone_in<A: Allocator>(&self, allocator: &A, new_cap: usize) -> Self
    where
        H: Clone,
        T: Clone,
    {
        let (buf, metadata) = self.buf.clone_in(&self.metadata, allocator, new_cap);
        UnmanagedHeaderVector2 { buf, metadata }
    }

    #[inline]
    pub fn try_clone_in<A: Allocator>(
        &self,
        allocator: &A,
        new_cap: usize,
    ) -> Result<Self, AllocError>
    where
        H: Clone,
        T: Clone,
    {
        let (buf, metadata) = self.buf.try_clone_in(&self.metadata, allocator, new_cap)?;
        Ok(UnmanagedHeaderVector2 { buf, metadata })
    }

    #[inline]
    pub unsafe fn try_realloc_in_new_allocator<OldAllocator, NewAllocator>(
        &mut self,
        new_cap: usize,
        old_allocator: &OldAllocator,
        new_allocator: &NewAllocator,
    ) -> Result<(), AllocError>
    where
        OldAllocator: Allocator,
        NewAllocator: Allocator,
    {
        self.buf.try_realloc_in_new_allocator(&mut self.metadata, new_cap, old_allocator, new_allocator)
    }

    #[inline]
    pub unsafe fn try_shrink_to<A: Allocator>(
        &mut self,
        new_cap: usize,
        allocator: &A,
    ) -> Result<(), AllocError> {
        self.buf.try_shrink_to(&mut self.metadata, new_cap, allocator)
    }

    #[inline]
    pub unsafe fn shrink_to<A: Allocator>(&mut self, new_cap: usize, allocator: &A) {
        self.buf.shrink_to(&mut self.metadata, new_cap, allocator);
    }

    #[inline]
    pub unsafe fn shrink_to_fit<A: Allocator>(&mut self, allocator: &A) {
        self.buf.shrink_to_fit(&mut self.metadata, allocator);
    }

    #[inline]
    pub fn truncate(&mut self, new_len: usize) {
        self.buf.truncate(&mut self.metadata, new_len);
    }

    #[inline]
    pub fn header(&self) -> &H {
        self.buf.header()
    }

    #[inline]
    pub fn header_mut(&mut self) -> &mut H {
        self.buf.header_mut()
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.buf.as_slice(&self.metadata)
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.buf.as_mut_slice(&self.metadata)
    }

    #[inline]
    pub fn clear(&mut self) {
        self.buf.clear(&mut self.metadata);
    }

    #[inline]
    pub unsafe fn push<A: Allocator>(&mut self, val: T, allocator: &A) {
        self.buf.push(&mut self.metadata, val, allocator);
    }

    #[inline]
    pub fn push_within_capacity(&mut self, val: T) -> Result<(), T> {
        self.buf.push_within_capacity(&mut self.metadata, val)
    }

    #[inline]
    pub unsafe fn push_assuming_capacity(&mut self, val: T) {
        self.buf.push_assuming_capacity(&mut self.metadata, val);
    }

    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        self.buf.pop(&mut self.metadata)
    }

    #[inline]
    pub fn get(&self, index: usize) -> Option<&T> {
        self.buf.get(&self.metadata, index)
    }

    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> &T {
        self.buf.get_unchecked(index)
    }

    #[inline]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.buf.get_mut(&mut self.metadata, index)
    }

    #[inline]
    pub unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T {
        self.buf.get_unchecked_mut(index)
    }

    #[inline]
    pub fn remove(&mut self, index: usize) -> T {
        self.buf.remove(&mut self.metadata, index)
    }

    #[inline]
    pub fn swap_remove(&mut self, index: usize) -> T {
        self.buf.swap_remove(&mut self.metadata, index)
    }

    #[inline]
    pub unsafe fn insert<A: Allocator>(&mut self, index: usize, element: T, allocator: &A) {
        self.buf.insert(&mut self.metadata, index, element, allocator);
    }

    #[inline]
    pub unsafe fn extend_from_slice<A: Allocator>(&mut self, slice: &[T], allocator: &A)
    where
        T: Clone,
    {
        self.buf.extend_from_slice(&mut self.metadata, slice, allocator);
    }

    #[inline]
    pub unsafe fn try_extend_from_slice<A: Allocator>(
        &mut self,
        slice: &[T],
        allocator: &A,
    ) -> Result<(), AllocError>
    where
        T: Clone,
    {
        self.buf.try_extend_from_slice(&mut self.metadata, slice, allocator)
    }

    #[inline]
    pub unsafe fn extend_from_slice_within_capacity(&mut self, slice: &[T])
    where
        T: Clone,
    {
        self.buf.extend_from_slice_within_capacity(&mut self.metadata, slice);
    }

    #[inline]
    pub fn extend_from_slice_assuming_capacity(&mut self, slice: &[T])
    where
        T: Clone,
    {
        self.buf.extend_from_slice_assuming_capacity(&mut self.metadata, slice);
    }

    #[inline]
    pub fn into_raw_parts(self) -> (NonNull<T>, usize, usize) {
        self.buf.into_raw_parts(self.metadata)
    }

    #[inline]
    pub unsafe fn from_raw_parts(data: NonNull<T>, len: usize, cap: usize) -> Self {
        let (buf, metadata) = HeaderVectorBuffer::from_raw_parts(data, len, cap);
        UnmanagedHeaderVector2 { buf, metadata }
    }
}

impl<H, T> Copy for UnmanagedHeaderVector2<H, T> {}

impl<H, T> Clone for UnmanagedHeaderVector2<H, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<H: PartialEq, T: PartialEq> PartialEq for UnmanagedHeaderVector2<H, T> {
    fn eq(&self, other: &Self) -> bool {
        self.header() == other.header() && self.as_slice() == other.as_slice()
    }
}

impl<T: PartialEq> PartialEq<&[T]> for UnmanagedHeaderVector2<(), T> {
    fn eq(&self, other: &&[T]) -> bool {
        self.as_slice() == *other
    }
}

impl<H, T> AsRef<[T]> for UnmanagedHeaderVector2<H, T> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<H, T> AsMut<[T]> for UnmanagedHeaderVector2<H, T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<H, T, I> core::ops::Index<I> for UnmanagedHeaderVector2<H, T>
where
    I: core::slice::SliceIndex<[T]>,
{
    type Output = <I as core::slice::SliceIndex<[T]>>::Output;
    fn index(&self, index: I) -> &Self::Output {
        self.as_slice().index(index)
    }
}

impl<H, T, I> core::ops::IndexMut<I> for UnmanagedHeaderVector2<H, T>
where
    I: core::slice::SliceIndex<[T]>,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        self.as_mut_slice().index_mut(index)
    }
}

impl<H, T> core::ops::Deref for UnmanagedHeaderVector2<H, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<H, T> core::ops::DerefMut for UnmanagedHeaderVector2<H, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<H: Debug, T: Debug> Debug for UnmanagedHeaderVector2<H, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> Result<(), core::fmt::Error> {
        if is_zst::<H>() {
            self.as_slice().fmt(f)
        } else {
            write!(f, "{:?}:{:?}", self.header(), self.as_slice())
        }
    }
}

impl<T> Default for UnmanagedHeaderVector2<(), T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<H: Debug, T: Debug> HeaderVectorBuffer<H, T> {
    pub fn fmt_debug(&self, metadata: &VectorMetadata, f: &mut core::fmt::Formatter<'_>) -> Result<(), core::fmt::Error> {
        if is_zst::<H>() {
            self.as_slice(metadata).fmt(f)
        } else {
            write!(f, "{:?}:{:?}", self.header(), self.as_slice(metadata))
        }
    }
}

#[test]
fn realloc_zst_header() {
    use std::sync::atomic::{AtomicI32, Ordering::SeqCst};

    static S_DROP_COUNT: AtomicI32 = AtomicI32::new(0);

    struct Foo;
    impl Drop for Foo {
        fn drop(&mut self) {
            S_DROP_COUNT.fetch_add(1, SeqCst);
        }
    }

    assert!(is_zst::<Foo>());

    let allocator = crate::allocator::Global;
    let (mut v, mut md) = HeaderVectorBuffer::with_capacity_in(Foo, 0, AllocInit::Uninit, &allocator);
    for i in 0u32..512 {
        unsafe {
            v.push(&mut md, i, &allocator);
        }
    }

    unsafe { v.deallocate_in(&mut md, &allocator); }
    assert_eq!(S_DROP_COUNT.load(SeqCst), 1);
}

#[test]
fn realloc_zst_items() {
    use std::sync::atomic::{AtomicI32, Ordering::SeqCst};

    static S_DROP_COUNT: AtomicI32 = AtomicI32::new(0);

    struct Foo;
    impl Drop for Foo {
        fn drop(&mut self) {
            S_DROP_COUNT.fetch_add(1, SeqCst);
        }
    }

    let allocator = crate::allocator::Global;
    let (mut v, mut md) = HeaderVectorBuffer::with_capacity_in((), 0, AllocInit::Uninit, &allocator);
    for _ in 0u32..512 {
        unsafe {
            v.push(&mut md, Foo, &allocator);
        }
    }

    unsafe { v.deallocate_in(&mut md, &allocator); }
    assert_eq!(S_DROP_COUNT.load(SeqCst), 512);
}

#[test]
fn header_capacity_respects_requested_minimum() {
    let allocator = crate::allocator::Global;
    let (mut v, mut md): (HeaderVectorBuffer<u64, u32>, VectorMetadata) =
        HeaderVectorBuffer::with_capacity_in(7u64, 4, AllocInit::Uninit, &allocator);

    assert_eq!(*v.header(), 7);
    assert!(md.capacity() >= crate::MIN_CAPACITY);

    unsafe { v.deallocate_in(&mut md, &allocator); }
}

#[test]
fn buffer_size_includes_header_space() {
    let allocator = crate::allocator::Global;
    let header_size = util::header_size::<u64, u32>();
    let size = header_size + 4 * mem::size_of::<u32>();
    let (mut v, mut md) = HeaderVectorBuffer::with_buffer_size_in(9u64, size, AllocInit::Uninit, &allocator);

    assert_eq!(*v.header(), 9);
    assert_eq!(md.capacity(), 4);

    for i in 0..4u32 {
        let result = v.push_within_capacity(&mut md, i);
        assert!(result.is_ok());
    }
    assert_eq!(v.push_within_capacity(&mut md, 4), Err(4));

    unsafe { v.deallocate_in(&mut md, &allocator); }
}

#[test]
fn reallocate_in_new_allocator_preserves_header_and_prefix() {
    let old_allocator = crate::allocator::Global;
    let new_allocator = crate::allocator::Global;
    let (mut v, mut md) = HeaderVectorBuffer::with_capacity_in(11u32, 8, AllocInit::Uninit, &old_allocator);

    unsafe {
        v.extend_from_slice(&mut md, &[1u32, 2, 3, 4, 5, 6], &old_allocator);
        v.try_realloc_in_new_allocator(&mut md, 4, &old_allocator, &new_allocator)
            .unwrap();
    }

    assert_eq!(*v.header(), 11);
    assert_eq!(md.len(), 4);
    assert_eq!(md.capacity(), 4);
    assert_eq!(v.as_slice(&md), &[1, 2, 3, 4]);

    unsafe { v.deallocate_in(&mut md, &new_allocator); }
}

#[test]
fn insert_remove_swap_remove_and_pop_work() {
    let allocator = crate::allocator::Global;
    let (mut v, mut md) = HeaderVectorBuffer::with_capacity_in(5u32, 8, AllocInit::Uninit, &allocator);

    unsafe {
        v.extend_from_slice(&mut md, &[1u32, 2, 3], &allocator);
        v.insert(&mut md, 1, 9, &allocator);
    }
    assert_eq!(*v.header(), 5);
    assert_eq!(v.as_slice(&md), &[1, 9, 2, 3]);

    assert_eq!(v.remove(&mut md, 2), 2);
    assert_eq!(v.as_slice(&md), &[1, 9, 3]);

    assert_eq!(v.swap_remove(&mut md, 0), 1);
    assert_eq!(md.len(), 2);
    assert_eq!(v.as_slice(&md), &[3, 9]);

    assert_eq!(v.pop(&mut md), Some(9));
    assert_eq!(v.pop(&mut md), Some(3));
    assert_eq!(v.pop(&mut md), None);

    unsafe { v.deallocate_in(&mut md, &allocator); }
}

#[test]
fn extend_variants_respect_capacity() {
    let allocator = crate::allocator::Global;
    let header_size = util::header_size::<u32, u32>();
    let size = header_size + 4 * mem::size_of::<u32>();
    let (mut v, mut md) = HeaderVectorBuffer::with_buffer_size_in(3u32, size, AllocInit::Uninit, &allocator);

    unsafe {
        v.push_assuming_capacity(&mut md, 1);
    }
    assert_eq!(v.push_within_capacity(&mut md, 2), Ok(()));

    unsafe {
        v.extend_from_slice_within_capacity(&mut md, &[3, 4, 5]);
    }
    assert_eq!(v.as_slice(&md), &[1, 2, 3, 4]);
    assert_eq!(md.remaining_capacity(), 0);

    unsafe {
        assert!(v.try_extend_from_slice(&mut md, &[6], &allocator).is_ok());
    }
    assert_eq!(v.as_slice(&md), &[1, 2, 3, 4, 6]);
    assert_eq!(*v.header(), 3);

    unsafe { v.deallocate_in(&mut md, &allocator); }
}

#[test]
fn clone_and_shrink_preserve_header_and_contents() {
    let allocator = crate::allocator::Global;
    let (mut v, mut md) = HeaderVectorBuffer::with_capacity_in(13u32, 16, AllocInit::Uninit, &allocator);

    unsafe {
        v.extend_from_slice(&mut md, &[1u32, 2, 3, 4, 5], &allocator);
    }

    let (mut clone, mut clone_md) = v.clone_in(&md, &allocator, 32);
    assert_eq!(*clone.header(), 13);
    assert_eq!(clone.as_slice(&clone_md), &[1, 2, 3, 4, 5]);
    assert!(clone_md.capacity() >= 32);

    clone.header_mut().clone_from(&99);
    clone.as_mut_slice(&clone_md)[0] = 42;

    assert_eq!(*v.header(), 13);
    assert_eq!(v.as_slice(&md), &[1, 2, 3, 4, 5]);

    unsafe {
        v.shrink_to_fit(&mut md, &allocator);
    }
    assert_eq!(*v.header(), 13);
    assert_eq!(md.capacity(), md.len());
    assert_eq!(v.as_slice(&md), &[1, 2, 3, 4, 5]);

    unsafe {
        clone.deallocate_in(&mut clone_md, &allocator);
        v.deallocate_in(&mut md, &allocator);
    }
}

#[test]
fn raw_parts_roundtrip_preserves_state() {
    let allocator = crate::allocator::Global;
    let (mut v, mut md) = HeaderVectorBuffer::with_capacity_in(21u32, 8, AllocInit::Uninit, &allocator);

    unsafe {
        v.extend_from_slice(&mut md, &[7u32, 8, 9], &allocator);
    }

    let cap = md.capacity();
    let (data, len, cap2) = v.into_raw_parts(md);
    assert_eq!(cap2, cap);

    let (mut v, mut md) = unsafe { HeaderVectorBuffer::<u32, u32>::from_raw_parts(data, len, cap2) };
    assert_eq!(*v.header(), 21);
    assert_eq!(v.as_slice(&md), &[7, 8, 9]);
    assert_eq!(md.capacity(), cap);

    unsafe { v.deallocate_in(&mut md, &allocator); }
}

#[test]
fn clear() {
    use std::rc::Rc;
    let rc = Rc::new(());
    let allocator = crate::alloc::Global;

    let (mut v, mut md) = UnmanagedVector::new();
    unsafe {
        v.push(&mut md, rc.clone(), &allocator);
        v.push(&mut md, rc.clone(), &allocator);
        v.push(&mut md, rc.clone(), &allocator);
        v.push(&mut md, rc.clone(), &allocator);
        assert_eq!(Rc::strong_count(&rc), 5);

        v.clear(&mut md);
        assert_eq!(Rc::strong_count(&rc), 1);

        v.deallocate_in(&mut md, &allocator);
    }
}

#[test]
fn truncate() {
    use std::rc::Rc;
    let rc = Rc::new(());
    let allocator = crate::alloc::Global;

    let (mut v, mut md) = UnmanagedVector::new();
    unsafe {
        v.push(&mut md, rc.clone(), &allocator);
        v.push(&mut md, rc.clone(), &allocator);
        v.push(&mut md, rc.clone(), &allocator);
        v.push(&mut md, rc.clone(), &allocator);
        assert_eq!(Rc::strong_count(&rc), 5);

        v.truncate(&mut md, 2);
        assert_eq!(Rc::strong_count(&rc), 3);

        v.deallocate_in(&mut md, &allocator);
    }
}
