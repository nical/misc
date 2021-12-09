use std::sync::{Mutex, Condvar};
use std::sync::atomic::{Ordering, AtomicI32, AtomicPtr};

use crossbeam_utils::Backoff;

use crate::Context;
use crate::job::JobRef;
use crate::thread_pool::ThreadPoolId;

// for debugging.
static SYNC_ID: AtomicI32 = AtomicI32::new(12345);

const STATE_DEFAULT: i32 = 0;
const STATE_SIGNALING: i32 = 1;
const STATE_SIGNALED: i32 = 2;

pub struct SyncPoint {
    // The number of unresolved dependency.
    deps: AtomicI32,
    // Whether the dependencies has been met AND it is safe to deallocate the syncpoint.
    // We can't simply use deps, because we need to keep the object alive for a little
    // bit after deps reach zero.
    //
    // No read or write to the sync point is safe after state is set to STATE_SIGNALED,
    // except for the threads that owns the sync point (the one that calls wait).
    state: AtomicI32,
    // A list of jobs to schedule when the dependencies are met.
    waiting_jobs: AtomicLinkedList<JobRef>,
    // As a last resort, a condition variable and its mutex to wait on if we couldn't
    // keep busy while the dependencies are being processed.
    mutex: Mutex<()>,
    cond: Condvar,

    thread_pool_id: ThreadPoolId,
    // An ID for debugging
    id: i32,
}

impl SyncPoint {
    pub fn new(deps: u32, pool_id: ThreadPoolId) -> Self {
        let state = if deps == 0 {
            STATE_SIGNALED
        } else {
            STATE_DEFAULT
        };

        SyncPoint {
            deps: AtomicI32::new(deps as i32),
            waiting_jobs: AtomicLinkedList::new(),
            state: AtomicI32::new(state),
            mutex: Mutex::new(()),
            cond: Condvar::new(),
            thread_pool_id: pool_id,
            id: SYNC_ID.fetch_add(1, Ordering::Relaxed),
        }
    }

    pub fn reset(&mut self, deps: u32, thread_pool_id: ThreadPoolId) {
        assert!(self.state.load(Ordering::Acquire) == STATE_SIGNALED);

        let state = if deps == 0 { STATE_SIGNALED } else { STATE_DEFAULT };
        self.thread_pool_id = thread_pool_id;
        self.state.store(state, Ordering::Release);
        self.deps.store(deps as i32, Ordering::Release);
    }

    pub fn signal_one(&self, ctx: &mut Context) -> bool {
        self.signal(ctx, 1)
    }

    pub fn signal(&self, ctx: &mut Context, n: u32) -> bool {
        debug_assert!(!self.is_signaled(), "already signaled {:?}:{:?}", self as *const _, self.id); // TODO: this fails
        debug_assert!(self.deps.load(Ordering::SeqCst) >= 1);
        //assert_eq!(self.thread_pool_id, ctx.thread_pool_id());

        profiling::scope!("signal");
        let n = n as i32;
        let deps = self.deps.fetch_add(-n, Ordering::Relaxed) - n;

        if deps > 0 {
            // After reading deps, it isn't guaranteed that self is valid except for the
            // one thread which signaled the last dependency (the one thread not taking this
            // branch).
            return false;
        }

        debug_assert!(deps == 0, "signaled too many time");

        self.state.store(STATE_SIGNALING, Ordering::SeqCst);

        // Executing the first job ourselves avoids the overhead of going
        // through the job queue.
        // TODO: this can create an unbounded recursion.
        // We could track the recursion level in Context and decide whether to
        // execute the first job ourselves based on that.
        let mut first = None;
        self.waiting_jobs.pop_all(&mut |job| {
            if first.is_none() {
                first = Some(job);
            } else {
                ctx.schedule_job(job)
            }
        });

        {
            std::mem::drop(self.mutex.lock().unwrap());

            self.cond.notify_all();
        }

        // It is important to mark this atomic boolean after setting the event.
        // when waiting we can only assume that the wait is over when this atomic
        // is set to true waiting on the event is not sufficient. This is because
        // we have to make sure this store can safely happen.
        // If we'd do the store before setting the event, then setting the event
        // would not be safe because the waiting thread might have continued from
        // an early-out on the state check. The waiting thread is responsible
        // for keeping the sync point alive state has been set to true.
        self.state.store(STATE_SIGNALED, Ordering::Release);

        // After the state store above, self isn't guaranteed to be valid.

        if let Some(job) = first {
            unsafe {
                job.execute(ctx);
            }
        }

        true
    }

    #[inline]
    fn has_unresolved_dependencies(&self) -> bool {
        self.deps.load(Ordering::Acquire) > 0
    }

    #[inline]
    pub fn is_signaled(&self) -> bool {
        self.state.load(Ordering::Acquire) == STATE_SIGNALED
    }

    #[allow(unused)]
    pub(crate) fn log(&self, msg: &str) {
        println!("sync {:?}:{} {}", self as *const _, self.id, msg);
    }

    #[allow(unused)]
    pub(crate) fn then(&self, ctx: &mut Context, job: JobRef) {
        if self.is_signaled() {
            ctx.schedule_job(job);
            return;
        }

        self.waiting_jobs.push(job);

        // This can be called concurrently with `signal`, its possible for `deps` to be read here
        // before decrementing it in `signal` but submitting the jobs happens before `push`.
        // This sequence means the sync point ends up signaled with a job sitting in the waiting list.
        // To prevent that we check `deps` a second time and submit again if it has reached zero in
        // the mean time.
        if !self.has_unresolved_dependencies() {
            self.waiting_jobs.pop_all(&mut |job| {
                ctx.schedule_job(job);
            });
        }
    }


    pub fn try_wait(&self, ctx: &mut Context) -> bool {
        profiling::scope!("steal jobs");
        loop {
            if self.is_signaled() {
                // Fast path: the sync point's dependencies were all met before
                // we had to block on the condvar.

                ctx.stats.fast_wait += 1;
                return true;
            }

            // Steal a job and execute it. If we are lucky our dependencies will
            // be met by the time we run out of useful thin to do.
            if !ctx.keep_busy() {
                return false;
            }
        }
    }

    /// Wait until all dependencies of this synchronization points are met, and until
    /// it is safe to destroy the sync point (no other threads are going to read or write
    /// into it)
    pub fn wait(&self, ctx: &mut Context) {
        profiling::scope!("wait");

        //assert_eq!(self.thread_pool_id, ctx.thread_pool_id());

        // TODO: would it be possible to block on the worker thread's condition variable if there is
        // one instead of always blocking on the sync point's? That would allow the worker to resume
        // working if there is new work.

        {
            if self.try_wait(ctx) {
                return;
            }
        }

        // Slower path: using the condition variable.

        ctx.stats.cond_wait += 1;

        {
            profiling::scope!("wait(condvar)");
            let mut stolen = None;
            'outer: loop {
                if let Some(job) = stolen.take() {
                    unsafe {
                        ctx.execute_job(job);
                    }
                }

                let mut guard = self.mutex.lock().unwrap();
                while self.state.load(Ordering::Acquire) == STATE_DEFAULT {
                    guard = self.cond.wait(guard).unwrap();
                }

                break 'outer;
            }
        }

        // We have to spin until state has been stored to ensure that it is safe
        // for the signaling thread to do the store operation.
        // TODO: would it be better to spin in the destructor?

        let backoff = Backoff::new();

        for i in 0..200 {
            if self.state.load(Ordering::Acquire) == STATE_SIGNALED {
                if i != 0 {
                    ctx.stats.cond_wait_spin += 1;
                    ctx.stats.spinned += i;
                }
                return;
            }
            backoff.spin();
        }

        // The majority of the time we only check state once. If we are unlucky we
        // can end up spinning for a longer time, so get back to trying to steal some jobs.
        let mut i = 200;
        while self.state.load(Ordering::Acquire) != STATE_SIGNALED {
            ctx.keep_busy();
            i += 1;
        }

        ctx.stats.cond_wait_spin += 1;
        ctx.stats.spinned += i;

        //println!("spinned {} times", i);
    }

    /// Block the current thread until all of the sync point's dependencies are met.
    ///
    /// This does not attempt to steal/execute jobs.
    ///
    /// In general this should be avoided in favor of `Context::wait`.
    /// Only call this if you need to wait but don't have a `Context` available.
    pub(crate) fn wait_no_context(&self) {
        if self.is_signaled() {
            return;
        }

        profiling::scope!("wait");

        let mut guard = self.mutex.lock().unwrap();

        while self.state.load(Ordering::SeqCst) == STATE_DEFAULT {
            guard = self.cond.wait(guard).unwrap();
        }

        // Ensure state has been stored

        let backoff = Backoff::new();
        while self.state.load(Ordering::Acquire) != STATE_SIGNALED {
            backoff.spin();
        }
    }

    pub fn unsafe_ref(&self) -> SyncPointRef {
        SyncPointRef { sync: self }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct SyncPointRef {
    sync: *const SyncPoint,
}

unsafe impl Send for SyncPointRef {}
unsafe impl Sync for SyncPointRef {}

impl SyncPointRef {
    pub unsafe fn signal(&self, ctx: &mut Context, n: u32) -> bool {
        (*self.sync).signal(ctx, n)
    }
}

impl Drop for SyncPoint {
    fn drop(&mut self) {
        //debug_assert_eq!(self.deps.load(Ordering::Acquire), 0);
        //debug_assert!(self.state.load(Ordering::Acquire) == STATE_SIGNALED);
    }
}

unsafe impl Sync for SyncPoint {}
unsafe impl Send for SyncPoint {}


pub struct AtomicLinkedList<T> {
    first: AtomicPtr<Node<T>>,
}

struct Node<T> {
    payload: Option<T>,
    next: AtomicPtr<Node<T>>,
}

impl<T> AtomicLinkedList<T> {
    pub fn new() -> Self {
        AtomicLinkedList {
            first: AtomicPtr::new(std::ptr::null_mut()),
        }
    }

    pub fn push(&self, payload: T) {
        let node = Box::into_raw(Box::new(Node {
            payload: Some(payload),
            next: AtomicPtr::new(std::ptr::null_mut()),
        }));

        unsafe {
            loop {
                let first = self.first.load(Ordering::Acquire);
                (*node).next.store(first, Ordering::Release);

                if self.first.compare_exchange(first, node, Ordering::SeqCst, Ordering::Relaxed).is_ok() {
                    break;
                }
            }
        }
    }

    pub fn pop_all(&self, cb: &mut dyn FnMut(T)) {
        // First atomically swap out the first node.
        let mut node;
        loop {
            node = self.first.load(Ordering::Acquire);
            let res = self.first.compare_exchange(node, std::ptr::null_mut(), Ordering::SeqCst, Ordering::Relaxed);
            if res.is_ok() {
                break;
            }
        }

        // Now that we have exclusive access to the nodes, we can execute the callback.
        while !node.is_null() {
            unsafe {
                if let Some(payload) = (*node).payload.take() {
                    cb(payload);
                }

                let next = (*node).next.load(Ordering::Relaxed);
                {
                    let _ = Box::from_raw(node);
                }

                node = next;
            }
        }
    }
}

impl<T> Drop for AtomicLinkedList<T> {
    fn drop(&mut self) {
        self.pop_all(&mut |_| {
            panic!("Leaked job !");
        });
    }
}
