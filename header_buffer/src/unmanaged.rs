use crate::allocator::{AllocError, Allocator};
use crate::util::{self, is_zst};
use core::marker::PhantomData;
use core::mem;
use core::ptr::{self, NonNull};
use core::ops::{Index, IndexMut, Deref, DerefMut};
use core::fmt::Debug;

// Note: switch len and cap to use causes a 10% regression on the push
// benchmark.

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

    pub fn take(&mut self) -> Self {
        std::mem::replace(self, Self::new())
    }
}

impl<H, T> UnmanagedHeaderVector<H, T> {
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
            let new_items_ptr = new_alloc.cast::<u8>().add(header_size).cast::<T>();

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
            let header = self.header_ptr().read();
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
            self.data = new_alloc.cast::<u8>().add(header_size).cast::<T>();
            self.cap = new_cap;
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
        new_cap: usize,
        old_allocator: &OldAllocator,
        new_allocator: &NewAllocator,
    ) -> Result<(), AllocError>
    where
        OldAllocator: Allocator,
        NewAllocator: Allocator,
    {
        if Self::should_be_dangling(new_cap) {
            self.deallocate_in(old_allocator);
            return Ok(());
        }

        let new_layout = util::header_vector_layout::<H, T>(new_cap)?;
        let new_buffer = new_allocator.allocate(new_layout)?.cast::<u8>();

        let old_len = self.len();
        if old_len > new_cap {
            self.truncate(new_cap);
        }

        let old_cap = self.capacity();
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
        self.cap = new_cap;
        self.len = self.len.min(new_cap);

        if let Some(old_alloc) = old_buffer {
            let old_layout = util::header_vector_layout::<H, T>(old_cap).unwrap();
            old_allocator.deallocate(old_alloc, old_layout);
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

    pub fn truncate(&mut self, new_len: usize) {
        if self.len() <= new_len {
            return;
        }

        unsafe {
            let elems: *mut [T] = &mut self[new_len..];
            core::ptr::drop_in_place(elems);
        }

        self.len = new_len;
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
        self.capacity() - self.len()
    }

    #[inline(always)]
    pub fn is_full(&self) -> bool {
        self.len == self.cap
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
        self.data.add(index)
    }

    #[inline(always)]
    unsafe fn write_item(&mut self, index: usize, val: T) {
        debug_assert!(index < self.cap);
        let dst = self.item_ptr(index);
        dst.write(val);
    }

    #[inline(always)]
    unsafe fn read_item(&self, index: usize) -> T {
        debug_assert!(index < self.cap);
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
            self.push_assuming_capacity(val);
        }

        Ok(())
    }

    #[inline(always)]
    pub unsafe fn push_assuming_capacity(&mut self, val: T) {
        debug_assert!(self.len < self.capacity());

        unsafe {
            self.write_item(self.len, val);
        }
        self.len += 1;
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
                ret = ptr.read();

                // Shift everything down to fill in that spot.
                ptr.add(1).copy_to(ptr, self.len - index - 1);
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
            let item = ptr.read();

            let last_idx = self.len - 1;
            if index != last_idx {
                let last_ptr = self.item_ptr(last_idx);
                ptr.write(last_ptr.read());
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
                ptr.write(item.clone());
                ptr = ptr.add(1)
            }
        }
        self.len += slice.len();
    }

    pub fn into_raw_parts(self) -> (NonNull<T>, usize, usize) {
        (
            self.data,
            self.len,
            self.cap,
        )
    }

    pub unsafe fn from_raw_parts(data: NonNull<T>, len: usize, cap: usize) -> Self {
        UnmanagedHeaderVector { data, len, cap, _marker: PhantomData }
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

impl<H, T, I> Index<I> for UnmanagedHeaderVector<H, T>
where
    I: core::slice::SliceIndex<[T]>,
{
    type Output = <I as core::slice::SliceIndex<[T]>>::Output;
    fn index(&self, index: I) -> &Self::Output {
        self.as_slice().index(index)
    }
}

impl<H, T, I> IndexMut<I> for UnmanagedHeaderVector<H, T>
where
    I: core::slice::SliceIndex<[T]>,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        self.as_mut_slice().index_mut(index)
    }
}

impl<H, T> Deref for UnmanagedHeaderVector<H, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<H, T> DerefMut for UnmanagedHeaderVector<H, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<H: Debug, T: Debug> Debug for UnmanagedHeaderVector<H, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> Result<(), core::fmt::Error> {
        if is_zst::<H>() {
            self.as_slice().fmt(f)
        } else {
            write!(f, "{:?}:{:?}", self.header(), self.as_slice())
        }
    }
}

impl<T> Default for UnmanagedHeaderVector<(), T> {
    fn default() -> Self {
        UnmanagedHeaderVector::new()
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

#[test]
fn header_capacity_respects_requested_minimum() {
    let allocator = crate::global::Global;
    let mut v: UnmanagedHeaderVector<u64, u32> =
        UnmanagedHeaderVector::with_capacity_in(7u64, 4, AllocInit::Uninit, &allocator);

    assert_eq!(*v.header(), 7);
    assert!(v.capacity() >= crate::MIN_CAPACITY);

    unsafe { v.deallocate_in(&allocator); }
}

#[test]
fn buffer_size_includes_header_space() {
    let allocator = crate::global::Global;
    let header_size = util::header_size::<u64, u32>();
    let size = header_size + 4 * mem::size_of::<u32>();
    let mut v = UnmanagedHeaderVector::with_buffer_size_in(9u64, size, AllocInit::Uninit, &allocator);

    assert_eq!(*v.header(), 9);
    assert_eq!(v.capacity(), 4);

    for i in 0..4u32 {
        let result = v.push_within_capacity(i);
        assert!(result.is_ok());
    }
    assert_eq!(v.push_within_capacity(4), Err(4));

    unsafe { v.deallocate_in(&allocator); }
}

#[test]
fn reallocate_in_new_allocator_preserves_header_and_prefix() {
    let old_allocator = crate::global::Global;
    let new_allocator = crate::global::Global;
    let mut v = UnmanagedHeaderVector::with_capacity_in(11u32, 8, AllocInit::Uninit, &old_allocator);

    unsafe {
        v.extend_from_slice(&[1u32, 2, 3, 4, 5, 6], &old_allocator);
        v.try_realloc_in_new_allocator(4, &old_allocator, &new_allocator)
            .unwrap();
    }

    assert_eq!(*v.header(), 11);
    assert_eq!(v.len(), 4);
    assert_eq!(v.capacity(), 4);
    assert_eq!(v.as_slice(), &[1, 2, 3, 4]);

    unsafe { v.deallocate_in(&new_allocator); }
}

#[test]
fn insert_remove_swap_remove_and_pop_work() {
    let allocator = crate::global::Global;
    let mut v = UnmanagedHeaderVector::with_capacity_in(5u32, 8, AllocInit::Uninit, &allocator);

    unsafe {
        v.extend_from_slice(&[1u32, 2, 3], &allocator);
        v.insert(1, 9, &allocator);
    }
    assert_eq!(*v.header(), 5);
    assert_eq!(v.as_slice(), &[1, 9, 2, 3]);

    assert_eq!(v.remove(2), 2);
    assert_eq!(v.as_slice(), &[1, 9, 3]);

    assert_eq!(v.swap_remove(0), 1);
    assert_eq!(v.len(), 2);
    assert_eq!(v.as_slice(), &[3, 9]);

    assert_eq!(v.pop(), Some(9));
    assert_eq!(v.pop(), Some(3));
    assert_eq!(v.pop(), None);

    unsafe { v.deallocate_in(&allocator); }
}

#[test]
fn extend_variants_respect_capacity() {
    let allocator = crate::global::Global;
    let header_size = util::header_size::<u32, u32>();
    let size = header_size + 4 * mem::size_of::<u32>();
    let mut v = UnmanagedHeaderVector::with_buffer_size_in(3u32, size, AllocInit::Uninit, &allocator);

    unsafe {
        v.push_assuming_capacity(1);
    }
    assert_eq!(v.push_within_capacity(2), Ok(()));

    unsafe {
        v.extend_from_slice_within_capacity(&[3, 4, 5]);
    }
    assert_eq!(v.as_slice(), &[1, 2, 3, 4]);
    assert_eq!(v.remaining_capacity(), 0);

    unsafe {
        assert!(v.try_extend_from_slice(&[6], &allocator).is_ok());
    }
    assert_eq!(v.as_slice(), &[1, 2, 3, 4, 6]);
    assert_eq!(*v.header(), 3);

    unsafe { v.deallocate_in(&allocator); }
}

#[test]
fn clone_and_shrink_preserve_header_and_contents() {
    let allocator = crate::global::Global;
    let mut v = UnmanagedHeaderVector::with_capacity_in(13u32, 16, AllocInit::Uninit, &allocator);

    unsafe {
        v.extend_from_slice(&[1u32, 2, 3, 4, 5], &allocator);
    }

    let mut clone = v.clone_in(&allocator, 32);
    assert_eq!(*clone.header(), 13);
    assert_eq!(clone.as_slice(), &[1, 2, 3, 4, 5]);
    assert!(clone.capacity() >= 32);

    clone.header_mut().clone_from(&99);
    clone.as_mut_slice()[0] = 42;

    assert_eq!(*v.header(), 13);
    assert_eq!(v.as_slice(), &[1, 2, 3, 4, 5]);

    unsafe {
        v.shrink_to_fit(&allocator);
    }
    assert_eq!(*v.header(), 13);
    assert_eq!(v.capacity(), v.len());
    assert_eq!(v.as_slice(), &[1, 2, 3, 4, 5]);

    unsafe {
        clone.deallocate_in(&allocator);
        v.deallocate_in(&allocator);
    }
}

#[test]
fn raw_parts_roundtrip_preserves_state() {
    let allocator = crate::global::Global;
    let mut v = UnmanagedHeaderVector::with_capacity_in(21u32, 8, AllocInit::Uninit, &allocator);

    unsafe {
        v.extend_from_slice(&[7u32, 8, 9], &allocator);
    }

    let cap = v.capacity();
    let (data, len, cap2) = v.into_raw_parts();
    assert_eq!(cap2, cap);

    let mut v = unsafe { UnmanagedHeaderVector::<u32, u32>::from_raw_parts(data, len, cap2) };
    assert_eq!(*v.header(), 21);
    assert_eq!(v.as_slice(), &[7, 8, 9]);
    assert_eq!(v.capacity(), cap);

    unsafe { v.deallocate_in(&allocator); }
}

#[test]
fn clear() {
    use std::rc::Rc;
    let rc = Rc::new(());
    let allocator = crate::alloc::Global;

    let mut v = UnmanagedVector::new();
    unsafe {
        v.push(rc.clone(), &allocator);
        v.push(rc.clone(), &allocator);
        v.push(rc.clone(), &allocator);
        v.push(rc.clone(), &allocator);
        assert_eq!(Rc::strong_count(&rc), 5);

        v.clear();
        assert_eq!(Rc::strong_count(&rc), 1);

        v.deallocate_in(&allocator);
    }
}

#[test]
fn truncate() {
    use std::rc::Rc;
    let rc = Rc::new(());
    let allocator = crate::alloc::Global;

    let mut v = UnmanagedVector::new();
    unsafe {
        v.push(rc.clone(), &allocator);
        v.push(rc.clone(), &allocator);
        v.push(rc.clone(), &allocator);
        v.push(rc.clone(), &allocator);
        assert_eq!(Rc::strong_count(&rc), 5);

        v.truncate(2);
        assert_eq!(Rc::strong_count(&rc), 3);

        v.deallocate_in(&allocator);
    }
}

#[test]
fn test_bench_unmanaged() {
    const CAP: usize = 16;
    const N: usize = 100;
    type Item = [u32; 8];
    fn val(i: usize) -> Item {
        [i as u32; 8]
    }

    unsafe {
        let allocator = crate::alloc::Global;
        let mut v = UnmanagedVector::with_capacity_in((), CAP, AllocInit::Uninit, &allocator);
        for i in 0..N {
            v.push(val(i), &allocator);
        }
        v.deallocate_in(&allocator);
    }
}

#[test]
fn test_bench_std() {
    const CAP: usize = 16;
    const N: usize = 100;
    type Item = [u32; 8];
    fn val(i: usize) -> Item {
        [i as u32; 8]
    }

    let mut v = Vec::with_capacity(CAP);
    for i in 0..N {
        v.push(val(i));
    }
}
