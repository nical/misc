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

use crate::{Context, Event, Priority};
use crate::core::job::{Job, JobRef};
use crate::helpers::TaskDependency;
use crate::sync::{AtomicI32, Ordering};
use std::cell::UnsafeCell;
use std::ops::Range;

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

    pub unsafe fn as_job_ref(&self, priority: Priority) -> JobRef where T: Job {
        JobRef::new(self).with_priority(priority)
    }
}

// TODO: In retrospect, implementing Job automatically is probably a mistake
// because it works very poorly with splitable jobs which need to release only
// when the event is signaled.
impl<T: Job> Job for InlineRefCounted<T> {
    unsafe fn execute(this: *const Self, ctx: &mut Context, range: Range<u32>) {
        T::execute((*this).inner() as *const _, ctx, range);
        (*this).release_ref();
    }
}

impl<T> RefCounted for InlineRefCounted<T> {
    unsafe fn add_ref(&self) {
        self.ref_count.fetch_add(1, Ordering::Acquire);
    }

    unsafe fn release_ref(&self) {
        if self.ref_count.fetch_add(-1, Ordering::Release) - 1 == 0 {
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

pub struct OutputSlot<T> {
    cell: UnsafeCell<Option<T>>,
}

impl<T> OutputSlot<T> {
    #[inline]
    pub fn new() -> Self {
        OutputSlot {
            cell: UnsafeCell::new(None)
        }
    }

    #[inline]
    pub unsafe fn set(&self, payload: T) {
        debug_assert!((*self.cell.get()).is_none());
        (*self.cell.get()) = Some(payload);
    }

    #[inline]
    pub unsafe fn take(&self) -> T {
        (*self.cell.get()).take().unwrap()
    }

    pub unsafe fn get_ref(&self) -> &T {
        (*self.cell.get()).as_ref().unwrap()
    }
}

/// A non-clonable handle which owns the result.
pub struct OwnedHandle<Output> {
    // Maintains the task's data alive.
    job_data: AnyRefPtr,
    event: *const Event,
    output: *const OutputSlot<Output>,
}

impl<Output> OwnedHandle<Output> {
    pub unsafe fn new(
        job_data: AnyRefPtr,
        event: *const Event,
        output: *const OutputSlot<Output>,
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
        output: *mut OutputSlot<Output>,
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
