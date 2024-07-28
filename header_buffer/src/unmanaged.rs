use crate::allocator::{AllocError, Allocator};
use crate::util::{self, is_zst, nnptr};
use core::marker::PhantomData;
use core::mem;
use core::ptr::NonNull;

pub type UnmanagedVector<T> = UnmanagedHeaderVector<(), T>;

pub enum AllocInit {
    Uninit,
    Zeroed
}

pub struct UnmanagedHeaderVector<H, T> {
    data: NonNull<T>,
    len: usize,
    cap: usize,
    _marker: PhantomData<H>,
}

impl<T> UnmanagedHeaderVector<(), T> {
    pub fn new() -> Self {
        UnmanagedHeaderVector {
            data: NonNull::dangling(),
            len: 0,
            cap: 0,
            _marker: PhantomData,
        }
    }
}

impl<H, T> UnmanagedHeaderVector<H, T> {
    #[inline(always)]
    fn should_be_dangling(cap: usize) -> bool {
        util::is_zst::<H>() && (util::is_zst::<T>() || cap == 0)
    }

    /// Creates an empty pre-allocated vector with a given storage size.
    #[inline(never)]
    pub fn try_with_buffer_size_in<A: Allocator>(
        header: H,
        size: usize,
        init: AllocInit,
        allocator: &A,
    ) -> Result<Self, AllocError> {
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

        if size < header_size + t_size {
            return Err(AllocError);
        }

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

        let data = unsafe { nnptr::add(allocation.cast::<u8>(), header_size).cast::<T>() };

        if is_zst::<H>() {
            // In case the header has a Drop implementation.
            mem::forget(header);
        } else {
            unsafe {
                let header_ptr = util::get_header::<H, T>(data);
                nnptr::write(header_ptr, header);
            }
        }

        Ok(UnmanagedHeaderVector {
            data,
            len: 0,
            cap,
            _marker: PhantomData,
        })
    }

    /// Creates an empty pre-allocated vector with a given storage size.
    #[inline(never)]
    pub fn with_buffer_size_in<A: Allocator>(
        header: H,
        size: usize,
        init: AllocInit,
        allocator: &A,
    ) -> Self {
        Self::try_with_buffer_size_in(header, size, init, allocator).unwrap()
    }
    
    /// Creates an empty pre-allocated vector with a given storage capacity.
    #[inline(never)]
    pub fn try_with_capacity_in<A: Allocator>(
        header: H,
        mut cap: usize,
        init: AllocInit,
        allocator: &A,
    ) -> Result<Self, AllocError> {
        cap = cap.min(16);
        let size = cap * mem::size_of::<T>();
        Self::try_with_buffer_size_in(header, size, init, allocator)
    }

    pub fn with_capacity_in<A: Allocator>(
        header: H,
        cap: usize,
        init: AllocInit,
        allocator: &A,
    ) -> Self {
        Self::try_with_capacity_in(header, cap, init, allocator).unwrap()
    }

    #[inline]
    pub unsafe fn dangling() -> Self {
        let cap = if util::is_zst::<T>() {
            isize::MAX as usize
        } else {
            0
        };

        UnmanagedHeaderVector {
            data: NonNull::dangling(),
            len: 0,
            cap,
            _marker: PhantomData,
        }
    }

    pub fn try_from_slice<A: Allocator>(
        header: H,
        data: &[T],
        allocator: &A,
    ) -> Result<Self, AllocError>
    where
        T: Clone,
    {
        let mut v = Self::try_with_capacity_in(header, data.len(), AllocInit::Uninit, allocator)?;
        unsafe {
            v.extend_from_slice(data, allocator);
        }

        Ok(v)
    }

    /// Tries to reserve at least enough space for `additional` extra items.
    ///
    /// # Safety
    ///
    /// The provided allocator must be the one this raw vector was created with.
    #[inline]
    pub unsafe fn try_reserve<A: Allocator>(
        &mut self,
        additional: usize,
        allocator: &A,
    ) -> Result<(), AllocError> {
        if self.remaining_capacity() < additional {
            self.try_realloc_additional(additional, allocator)?;
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
    pub unsafe fn deallocate_in<A: Allocator>(&mut self, allocator: &A) {
        self.clear();

        unsafe {
            core::ptr::drop_in_place(self.header_ptr().as_ptr());

            if Self::should_be_dangling(self.cap) {
                return;
            }

            let allocation = self.header_ptr().cast::<u8>();

            let layout = util::header_vector_layout::<H, T>(self.cap).unwrap();
            allocator.deallocate(allocation.cast::<u8>(), layout);
        }

        *self = Self::dangling();
    }

    #[inline]
    pub fn ptr_eq(&self, other: &Self) -> bool {
        self.data == other.data
    }

    pub fn clone_in<A: Allocator>(&self, allocator: &A, new_cap: usize) -> Self
    where
        H: Clone,
        T: Clone,
    {
        self.try_clone_in(allocator, new_cap).unwrap()
    }

    pub fn try_clone_in<A: Allocator>(
        &self,
        allocator: &A,
        mut new_cap: usize,
    ) -> Result<Self, AllocError>
    where
        H: Clone,
        T: Clone,
    {
        new_cap = new_cap.max(self.len);
        unsafe {
            if Self::should_be_dangling(new_cap) {
                return Ok(UnmanagedHeaderVector::dangling());
            }

            let header_size = util::header_size::<H, T>();
            let t_size = mem::size_of::<T>();
            let old_empty = (t_size == 0 || self.len == 0) && header_size == 0;

            if old_empty {
                return UnmanagedHeaderVector::try_with_capacity_in(
                    self.header().clone(),
                    new_cap,
                    AllocInit::Uninit,
                    allocator,
                );
            }

            let new_layout = util::header_vector_layout::<H, T>(new_cap).unwrap();

            let new_alloc = allocator.allocate(new_layout)?;
            let new_items_ptr = nnptr::add(new_alloc.cast::<u8>(), header_size).cast::<T>();

            if !is_zst::<H>() {
                let old_header = util::get_header::<H, T>(self.data);
                let new_header = util::get_header::<H, T>(new_items_ptr);
                core::ptr::write(new_header.as_ptr(), old_header.as_ref().clone());
            }

            if !is_zst::<T>() {
                let mut src = self.data.as_ptr();
                let mut dst = new_items_ptr.as_ptr();
                for _ in 0..self.len {
                    core::ptr::write(dst, (*src).clone());
                    src = src.add(1);
                    dst = dst.add(1);
                }
            }

            Ok(UnmanagedHeaderVector {
                data: new_items_ptr,
                len: self.len,
                cap: new_cap,
                _marker: PhantomData,
            })
        }
    }

    // Note: Marking this #[inline(never)] is a pretty large regression in the push benchmark.
    #[cold]
    pub(crate) unsafe fn try_realloc_additional<A: Allocator>(
        &mut self,
        additional: usize,
        allocator: &A,
    ) -> Result<(), AllocError> {
        let new_cap = util::grow_amortized(self.len(), additional)?;
        if new_cap < self.len() {
            return Err(AllocError);
        }

        self.try_realloc_with_capacity(new_cap, allocator)
    }

    #[cold]
    pub(crate) unsafe fn try_realloc_with_capacity<A: Allocator>(
        &mut self,
        new_cap: usize,
        allocator: &A,
    ) -> Result<(), AllocError> {
        let old_cap = self.cap;

        let old_dangling = Self::should_be_dangling(old_cap);
        let new_dangling = Self::should_be_dangling(new_cap);

        if new_dangling && !old_dangling {
            self.deallocate_in(allocator);
            return Ok(());
        }

        if old_dangling {
            // According to https://doc.rust-lang.org/nomicon/vec/vec-zsts.html
            // reading a ZST from NonNull::dangling is fine.
            let header = nnptr::read(self.header_ptr());
            *self = UnmanagedHeaderVector::try_with_capacity_in(header, new_cap, AllocInit::Uninit, allocator)?;
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
            self.data = nnptr::add(new_alloc.cast::<u8>(), header_size).cast::<T>();
            self.cap = new_cap;
        }

        Ok(())
    }

    pub unsafe fn try_shrink_to<A: Allocator>(
        &mut self,
        new_cap: usize,
        allocator: &A,
    ) -> Result<(), AllocError> {
        let new_cap = new_cap.max(self.len);
        if self.cap <= new_cap {
            return Ok(());
        }

        self.try_realloc_with_capacity(new_cap, allocator)
    }

    pub unsafe fn shrink_to<A: Allocator>(&mut self, new_cap: usize, allocator: &A) {
        self.try_shrink_to(new_cap, allocator).unwrap()
    }

    pub unsafe fn shrink_to_fit<A: Allocator>(&mut self, allocator: &A) {
        self.try_shrink_to(self.len, allocator).unwrap()
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
        self.capacity() - self.len
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe { core::slice::from_raw_parts(self.data.as_ptr(), self.len) }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { core::slice::from_raw_parts_mut(self.data.as_ptr(), self.len) }
    }

    /// Clears the vector, removing all values.
    pub fn clear(&mut self) {
        let elems: *mut [T] = self.as_mut_slice();
        unsafe {
            self.len = 0;
            core::ptr::drop_in_place(elems);
        }
    }

    #[inline(always)]
    unsafe fn item_ptr(&self, index: usize) -> NonNull<T> {
        nnptr::add(self.data, index)
    }

    #[inline(always)]
    unsafe fn write_item(&mut self, index: usize, val: T) {
        debug_assert!(index < self.cap);
        let dst = self.item_ptr(index);
        nnptr::write(dst, val);
    }

    #[inline(always)]
    unsafe fn read_item(&self, index: usize) -> T {
        debug_assert!(index < self.cap);
        let dst = self.item_ptr(index);
        nnptr::read(dst)
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
    pub unsafe fn push<A: Allocator>(&mut self, val: T, allocator: &A) {
        // Inform codegen that the length does not change across rtry_realloc_additional.
        let len = self.len;

        if len == self.capacity() {
            self.try_realloc_additional(1, allocator).unwrap();
        }

        self.write_item(len, val);
        self.len += 1;
    }

    /// Appends an element if there is sufficient spare capacity, otherwise an error is returned
    /// with the element.
    ///
    /// Unlike push this method will not reallocate when there’s insufficient capacity.
    /// The caller should use reserve or try_reserve to ensure that there is enough capacity.
    #[inline(always)]
    pub fn push_within_capacity(&mut self, val: T) -> Result<(), T> {
        if self.len == self.capacity() {
            return Err(val);
        }

        unsafe {
            self.write_item(self.len, val);
        }
        self.len += 1;

        Ok(())
    }

    /// Removes the last element from the vector and returns it, or `None` if it is empty.
    #[inline(always)]
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }

        self.len -= 1;
        unsafe { Some(self.read_item(self.len)) }
    }

    #[inline(always)]
    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.len {
            return None;
        }

        unsafe { Some(self.get_unchecked(index)) }
    }

    #[inline(always)]
    pub unsafe fn get_unchecked(&self, index: usize) -> &T {
        unsafe { self.item_ptr(index).as_ref() }
    }

    #[inline(always)]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index >= self.len {
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
    pub fn remove(&mut self, index: usize) -> T {
        unsafe {
            #[cold]
            #[inline(never)]
            #[track_caller]
            fn assert_failed(index: usize, len: usize) -> ! {
                panic!("remove: index {index} should be < len {len}.");
            }

            if index >= self.len {
                assert_failed(index, self.len);
            }

            // infallible
            let ret;
            {
                // the place we are taking from.
                let ptr = self.item_ptr(index);
                // copy it out, unsafely having a copy of the value on
                // the stack and in the vector at the same time.
                ret = nnptr::read(ptr);

                // Shift everything down to fill in that spot.
                nnptr::copy(nnptr::add(ptr, 1), ptr, self.len - index - 1);
            }

            self.len -= 1;

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
    pub fn swap_remove(&mut self, index: usize) -> T {
        unsafe {
            assert!(index < self.len);

            let ptr = self.item_ptr(index);
            let item = nnptr::read(ptr);

            let last_idx = self.len - 1;
            if index != last_idx {
                let last_ptr = self.item_ptr(last_idx);
                nnptr::write(ptr, nnptr::read(last_ptr));
            }

            self.len -= 1;

            item
        }
    }

    /// Inserts an element at position `index` within the vector, shifting all
    /// elements after it to the right.
    ///
    /// # Panics
    ///
    /// Panics if `index > len`.
    pub unsafe fn insert<A: Allocator>(&mut self, index: usize, element: T, allocator: &A) {
        #[cold]
        #[inline(never)]
        fn assert_failed(index: usize, len: usize) -> ! {
            panic!("insertion index (is {index}) should be <= len (is {len})");
        }

        unsafe {
            // Space for the new element
            if self.len == self.capacity() {
                self.try_reserve(1, allocator).unwrap();
            }

            let len = self.len();

            // Infallible
            // The spot to put the new value
            {
                let p = self.item_ptr(index);
                if index < len {
                    // Shift everything over to make space. (Duplicating the
                    // `index`th element into two consecutive places.)
                    nnptr::copy(p, nnptr::add(p, 1), len - index);
                } else if index == len {
                    // No elements need shifting.
                } else {
                    assert_failed(index, len);
                }
                // Write it in, overwriting the first copy of the `index`th
                // element.
                nnptr::write(p, element);
            }
            self.len += 1;
        }
    }

    /// Clones and appends the contents of the slice to the back of a collection.
    ///
    /// # Safety
    ///
    /// The provided allocator must be the one this raw vector was created with.
    pub unsafe fn extend_from_slice<A: Allocator>(&mut self, slice: &[T], allocator: &A)
    where
        T: Clone,
    {
        self.try_extend_from_slice(slice, allocator).unwrap();
    }

    pub unsafe fn try_extend_from_slice<A: Allocator>(
        &mut self,
        slice: &[T],
        allocator: &A,
    ) -> Result<(), AllocError>
    where
        T: Clone,
    {
        self.try_reserve(slice.len(), allocator)?;
        self.extend_from_slice_assuming_capacity(slice);

        Ok(())
    }

    pub unsafe fn extend_from_slice_within_capacity(&mut self, slice: &[T])
    where
        T: Clone,
    {
        let n = self.remaining_capacity().min(slice.len());
        if n > 0 {
            self.extend_from_slice_assuming_capacity(&slice[..n]);
        }
    }

    pub fn extend_from_slice_assuming_capacity(&mut self, slice: &[T])
    where
        T: Clone,
    {
        assert!(self.cap - self.len >= slice.len());
        unsafe {
            let mut ptr = self.item_ptr(self.len);

            for item in slice {
                nnptr::write(ptr, item.clone());
                ptr = nnptr::add(ptr, 1)
            }
        }
        self.len += slice.len();
    }
}

impl<H, T> Copy for UnmanagedHeaderVector<H, T> {}

impl<H, T> Clone for UnmanagedHeaderVector<H, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<H: PartialEq<H>, T: PartialEq<T>> PartialEq<UnmanagedHeaderVector<H, T>>
    for UnmanagedHeaderVector<H, T>
{
    fn eq(&self, other: &Self) -> bool {
        self.header() == other.header() && self.as_slice() == other.as_slice()
    }
}

impl<T: PartialEq<T>> PartialEq<&[T]> for UnmanagedHeaderVector<(), T> {
    fn eq(&self, other: &&[T]) -> bool {
        self.as_slice() == *other
    }
}

impl<H, T> AsRef<[T]> for UnmanagedHeaderVector<H, T> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<H, T> AsMut<[T]> for UnmanagedHeaderVector<H, T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
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

    let allocator = crate::global::Global;
    let mut v = UnmanagedHeaderVector::with_capacity_in(Foo, 0, AllocInit::Uninit, &allocator);
    for i in 0u32..512 {
        unsafe {
            v.push(i, &allocator);
        }
    }

    unsafe { v.deallocate_in(&allocator); }
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

    let allocator = crate::global::Global;
    let mut v = UnmanagedHeaderVector::with_capacity_in((), 0, AllocInit::Uninit, &allocator);
    for _ in 0u32..512 {
        unsafe {
            v.push(Foo, &allocator);
        }
    }

    unsafe { v.deallocate_in(&allocator); }
    assert_eq!(S_DROP_COUNT.load(SeqCst), 512);
}
