use std::ops::Range;
use std::cell::UnsafeCell;
use std::mem;
use crate::context::Context;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Priority {
    High,
    Low,
}

// TODO: move into job.rs
impl Priority {
    pub(crate) fn index(&self) -> usize {
        match self {
            Priority::High => 0,
            Priority::Low => 1,
        }
    }
}


/// A `Job` is used to advertise work for other threads that they may
/// want to steal. In accordance with time honored tradition, jobs are
/// arranged in a deque, so that thieves can take from the top of the
/// deque while the main worker manages the bottom of the deque. This
/// deque is managed by the `thread_pool` module.
pub trait Job {
    /// Unsafe: this may be called from a different thread than the one
    /// which scheduled the job, so the implementer must ensure the
    /// appropriate traits are met, whether `Send`, `Sync`, or both.
    unsafe fn execute(this: *const Self, ctx: &mut Context, range: Range<u32>);
}

/// Effectively a Job trait object. Each JobRef **must** be executed
/// exactly once, or else data may leak.
///
/// Internally, we store the job's data in a `*const ()` pointer.  The
/// true type is something like `*const StackJob<...>`, but we hide
/// it. We also carry the "execute fn" from the `Job` trait.
///
/// The interesting parts of this type are taken from Rayon.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct JobRef {
    // The "deconstructed trait object" part, taken directly from rayon.
    pointer: *const (),
    execute_fn: unsafe fn(*const (), *const (), Range<u32>),

    // Optional start/end parameters to allow splitting a job that operates
    // over a range of items.
    start: u32,
    end: u32,
    split_thresold: u32,
    priority: Priority,
}

unsafe impl Send for JobRef {}
unsafe impl Sync for JobRef {}

impl JobRef {
    /// Unsafe: caller asserts that `data` will remain valid until the
    /// job is executed.
    pub unsafe fn new<T>(data: *const T) -> JobRef
    where
        T: Job,
    {
        Self::with_range(data, 0..1, 1)
    }

    pub unsafe fn with_range<T>(data: *const T, range: Range<u32>, split_thresold: u32) -> JobRef
    where
        T: Job,
    {
        debug_assert!(range.start < range.end);
        let fn_ptr: unsafe fn(*const T, &mut Context, range: Range<u32>) = <T as Job>::execute;
        // erase types:
        JobRef {
            pointer: data as *const (),
            execute_fn: mem::transmute(fn_ptr),
            start: range.start,
            end: range.end,
            split_thresold: split_thresold.max(1),
            priority: Priority::High,
        }
    }

    #[inline]
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    #[inline]
    pub fn priority(&self) -> Priority { self.priority }

    #[inline]
    pub unsafe fn execute(&self, ctx: &mut Context) {
        (self.execute_fn)(self.pointer, mem::transmute(ctx), self.start..self.end)
    }

    #[inline]
    pub fn split(&mut self) -> Option<Self> {
        if self.end - self.start <= self.split_thresold {
            return None;
        }

        let split = (self.start + self.end) / 2;

        let end = self.end;
        self.end = split;

        Some(JobRef {
            pointer: self.pointer,
            execute_fn: self.execute_fn,
            start: split,
            end,
            split_thresold: self.split_thresold,
            priority: self.priority,
        })
    }
}

/// Represents a job stored in the heap. Used to implement
/// `scope`. Unlike `StackJob`, when executed, `HeapJob` simply
/// invokes a closure, which then triggers the appropriate logic to
/// signal that the job executed.
///
/// (Probably `StackJob` should be refactored in a similar fashion.)
pub struct HeapJob<BODY>
where
    BODY: FnOnce(&mut Context) + Send,
{
    job: UnsafeCell<Option<BODY>>,
}

impl<F> HeapJob<F>
where
    F: FnOnce(&mut Context) + Send,
{
    pub fn new(func: F) -> Self {
        HeapJob {
            job: UnsafeCell::new(Some(func)),
        }
    }

    pub unsafe fn new_ref(func: F) -> JobRef {
        Box::new(Self::new(func)).as_job_ref()
    }

    /// Creates a `JobRef` from this job -- note that this hides all
    /// lifetimes, so it is up to you to ensure that this JobRef
    /// doesn't outlive any data that it closes over.
    pub unsafe fn as_job_ref(self: Box<Self>) -> JobRef {
        let this: *const Self = mem::transmute(self);
        JobRef::new(this)
    }
}

impl<BODY> Job for HeapJob<BODY>
where
    BODY: FnOnce(&mut Context) + Send,
{
    unsafe fn execute(this: *const Self, ctx: &mut Context, _range: Range<u32>) {
        let this: Box<Self> = mem::transmute(this);
        let job = (*this.job.get()).take().unwrap();
        job(ctx);
    }
}

struct AbortIfPanic;

impl Drop for AbortIfPanic {
    fn drop(&mut self) {
        eprintln!("unexpected panic; aborting");
        ::std::process::abort();
    }
}

/*

/// A job that will be owned by a stack slot. This means that when it
/// executes it need not free any heap data, the cleanup occurs when
/// the stack frame is later popped.  The function parameter indicates
/// `true` if the job was stolen -- executed on a different thread.
pub struct StackJob<F, R>
where
    F: FnOnce(&mut Context) -> R + Send,
    R: Send,
{
    func: UnsafeCell<Option<F>>,
}

impl<F, R> StackJob<F, R>
where
    F: FnOnce(&mut Context) -> R + Send,
    R: Send,
{
    pub fn new(func: F) -> StackJob<F, R> {
        StackJob {
            func: UnsafeCell::new(Some(func)),
            //result: UnsafeCell::new(JobResult::None),
        }
    }

    pub unsafe fn as_job_ref(&self) -> JobRef {
        JobRef::new(self)
    }
}

impl<F, R> Job for StackJob<F, R>
where
    F: FnOnce(&mut Context) -> R + Send,
    R: Send,
{
    unsafe fn execute(this: *const Self, ctx: &mut Context, _range: Range<u32>) {
        let this = &*this;
        let abort = AbortIfPanic;
        let func = (*this.func.get()).take().unwrap();

        // TODO: in order to support storing the result here we have to integrate
        // the synchronization at this level rather than in the callback otherwise
        // there is nothing keeping the result slot alive for us to write into it.

        //(*this.result.get()) = JobResult::Ok(func(ctx));
        func(ctx);

        mem::forget(abort);
    }
}

pub(super) enum JobResult<T> {
    None,
    Ok(T),
    Panic(Box<dyn Any + Send>),
}
impl<T> JobResult<T> {
    /// Convert the `JobResult` for a job that has finished (and hence
    /// its JobResult is populated) into its return value.
    ///
    /// NB. This will panic if the job panicked.
    pub(super) fn into_return_value(self) -> T {
        match self {
            JobResult::None => unreachable!(),
            JobResult::Ok(x) => x,
            JobResult::Panic(x) => resume_unwinding(x),
        }
    }
}
pub(super) fn resume_unwinding(payload: Box<dyn Any + Send>) -> ! {
    panic::resume_unwind(payload)
}

*/
