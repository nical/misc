//! Handles are references to ongoing work.
//!
//! They consist of:
//!  - A strong reference to a reference counted heap alocation to maintain the job's data alive,
//!  - A pointer to an Event to be able to wait until the job is done,
//!  - And optionally a pointer to where the output of the job will be written, if any.
//!
//! Typically, the heap allocation will contain the job's callback, then event, the output
//! and whatever else the job needs to do its work. The job will also hold a self-reference
//! that goes away at the end of the work. This way the job data isn't deleted until all
//! handles are gone and the job is done.

use crate::{Context, Event};
use crate::helpers::TaskDependency;
use crate::sync::{AtomicI32, Ordering};
use std::cell::UnsafeCell;

/// This implements reference counting by hand so that the job can release its own refcount, which is hard to do with `Arc`.
pub trait RefCounted {
    unsafe fn add_ref(&self);
    unsafe fn release_ref(&self);
}

pub struct InlineRefCounted<T> {
    ref_count: AtomicI32,
    payload: T,
}

impl<T> InlineRefCounted<T> {
    pub fn inner(&self) -> &T {
        &self.payload
    }
}

impl<T> RefCounted for InlineRefCounted<T> {
    unsafe fn add_ref(&self) {
        self.ref_count.fetch_add(1, Ordering::Acquire);
    }

    unsafe fn release_ref(&self) {
        let ref_count = self.ref_count.fetch_sub(1, Ordering::Release) - 1;
        debug_assert!(ref_count >= 0);
        if ref_count == 0 {
            let this : *mut Self = std::mem::transmute(self);
            let _ = Box::from_raw(this);
        }
    }
}

pub struct RefPtr<T> {
    ptr: *mut InlineRefCounted<T>
}

impl<T: 'static> RefPtr<T> {
    pub fn new(payload: T) -> Self {
        let ptr = Box::new(InlineRefCounted {
            ref_count: AtomicI32::new(2),
            payload,
        });

        let ptr = Box::into_raw(ptr);

        RefPtr { ptr }
    }

    pub fn into_any(self) -> AnyRefPtr {
        AnyRefPtr::already_reffed(self.ptr as *mut dyn RefCounted)
    }

    pub unsafe fn mut_payload_unchecked(this: &Self) -> *mut T {
        &mut (*this.ptr).payload
    }

    pub fn as_raw(this: &Self) -> *const InlineRefCounted<T> {
        (*this).ptr
    }
}

impl<T> std::ops::Deref for InlineRefCounted<T> {
    type Target = T;
    fn deref(&self) -> &T { &self.payload }
}

impl<T> std::ops::Deref for RefPtr<T> {
    type Target = InlineRefCounted<T>;
    fn deref(&self) -> &InlineRefCounted<T> { unsafe { &*self.ptr } }
}

/// A strong reference to any `RefCounted` object.
pub struct AnyRefPtr {
    ptr: *mut dyn RefCounted,
}

impl AnyRefPtr {
    pub fn already_reffed(ptr: *mut dyn RefCounted) -> Self {
        assert!(!ptr.is_null());
        AnyRefPtr { ptr }
    }
}

impl Clone for AnyRefPtr {
    fn clone(&self) -> Self {
        unsafe {
            (*self.ptr).add_ref();
        }

        AnyRefPtr::already_reffed(self.ptr)
    }
}

impl Drop for AnyRefPtr {
    fn drop(&mut self) {
        unsafe {
            (*self.ptr).release_ref()
        }
    }
}

/// An unsynchronized, internally mutable slot where data can be placed, for example
/// to store the output or input of a job.
pub struct DataSlot<T> {
    cell: UnsafeCell<Option<T>>,
}

impl<T> DataSlot<T> {
    /// Create a data slot.
    #[inline]
    pub fn new() -> Self {
        DataSlot {
            cell: UnsafeCell::new(None)
        }
    }

    #[inline]
    /// Create an already-set data slot.
    pub fn from(data: T) -> Self {
        DataSlot {
            cell: UnsafeCell::new(Some(data))
        }
    }

    /// Place data in the slot.
    /// 
    /// Safety:
    ///  - `set` must be called at most once.
    ///  - `set` must not be called if the slot was created with `from`.
    #[inline]
    pub unsafe fn set(&self, payload: T) {
        debug_assert!((*self.cell.get()).is_none());
        (*self.cell.get()) = Some(payload);
    }

    /// Move the data out of the slot.
    ///
    /// Safety:
    ///  - `take` must be called at most once, *after* the
    ///    slot is set (either after `set` was called or if the slot
    ///    was created with `from`).
    ///  - `take` and `get_ref` must not be called concurrently.
    #[inline]
    pub unsafe fn take(&self) -> T {
        (*self.cell.get()).take().unwrap()
    }

    /// Get a reference on the data.
    ///
    /// Safety:
    ///  - This must be called *after* the slot is set (either
    ///    after `set` was called or if the slot was created with
    ///    `from`).
    ///  - `take` and `get_ref` must not be called concurrently.
    pub unsafe fn get_ref(&self) -> &T {
        (*self.cell.get()).as_ref().unwrap()
    }
}

/// A non-clonable handle which owns the result.
pub struct OwnedHandle<Output> {
    // Maintains the task's data alive.
    job_data: AnyRefPtr,
    event: *const Event,
    output: *const DataSlot<Output>,
}

impl<Output> OwnedHandle<Output> {
    pub unsafe fn new(
        job_data: AnyRefPtr,
        event: *const Event,
        output: *const DataSlot<Output>,
    ) -> Self {
        OwnedHandle { job_data, event, output }
    }

    pub fn wait(&self, ctx: &mut Context) {
        unsafe {
            (*self.event).wait(ctx);
            debug_assert!((*self.event).is_signaled());
        }
    }

    pub fn resolve(self, ctx: &mut Context) -> Output {
        self.wait(ctx);
        unsafe {
            (*self.output).take()
        }
    }

    pub fn poll(&self) -> bool {
        unsafe {
            (*self.event).is_signaled()
        }
    }

    pub fn shared(self) -> SharedHandle<Output> {
        SharedHandle { inner: self }
    }

    pub fn handle(&self) -> Handle {
        Handle {
            _job_data: self.job_data.clone(),
            event: self.event
        }
    }
}

// TODO: Need some way to implement TaskDependency<Output = &T> for SharedHandle<T>
// But the lifetime makes it hard to express.

/// A clonable handle that can only borrow the result.
pub struct SharedHandle<Output> {
    inner: OwnedHandle<Output>
}

impl<Output> SharedHandle<Output> {
    pub unsafe fn new(
        job_data: AnyRefPtr,
        event: *const Event,
        output: *mut DataSlot<Output>,
    ) -> Self {
        SharedHandle {
            inner: OwnedHandle::new(job_data, event, output)
        }
    }

    pub fn wait(&self, ctx: &mut Context) -> &Output {
        unsafe {
            (*self.inner.event).wait(ctx);
            (*self.inner.output).get_ref()
        }
    }

    pub fn poll(&self) -> bool {
        self.inner.poll()
    }

    pub fn handle(&self) -> Handle {
        self.inner.handle()
    }
}

impl<Output> Clone for SharedHandle<Output> {
    fn clone(&self) -> Self {
        SharedHandle {
            inner: OwnedHandle {
                job_data: self.inner.job_data.clone(),
                event: self.inner.event,
                output: self.inner.output,
            }
        }
    }
}

#[derive(Clone)]
/// A handle that doesn't know about the output of the task, but can
/// be used to wait or as a dependency.
pub struct Handle {
    _job_data: AnyRefPtr,
    event: *const Event,
}

impl Handle {
    pub unsafe fn new(
        job_data: AnyRefPtr,
        event: *const Event,
    ) -> Self {
        Handle { _job_data: job_data, event }
    }

    pub fn wait(&self, ctx: &mut Context) {
        unsafe {
            (*self.event).wait(ctx);
        }
    }

    pub fn poll(&self) -> bool {
        unsafe {
            (*self.event).is_signaled()
        }
    }
}

impl<T> TaskDependency for OwnedHandle<T> {
    type Output = T;
    fn get_output(&self) -> T {
        unsafe {
            debug_assert!((*self.event).is_signaled());
            (*self.output).take()
        }
    }

    fn get_event(&self) -> Option<&Event> {
        unsafe { Some(&*self.event) }
    }
}

impl TaskDependency for Handle {
    type Output = ();
    fn get_output(&self) -> () { () }
    fn get_event(&self) -> Option<&Event> {
        unsafe { Some(&*self.event) }
    }
}

impl TaskDependency for () {
    type Output = ();
    fn get_output(&self) -> () { () }
    fn get_event(&self) -> Option<&Event> { None }
}

impl<T> TaskDependency for DataSlot<T> {
    type Output = T;
    fn get_output(&self) -> T {
        unsafe { self.take() }
    }
    fn get_event(&self) -> Option<&Event> {
        None
    }
}