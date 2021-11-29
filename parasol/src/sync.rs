use std::sync::{Arc, Mutex, Condvar};
use std::sync::atomic::{Ordering, AtomicI32, AtomicPtr};

use crate::Context;
use crate::job::{HeapJob, JobRef};

pub struct SyncPoint {
    deps: AtomicI32,
    waiting_jobs: AtomicLinkedList<JobRef>,
}

impl SyncPoint {
    pub fn new(deps: u32) -> Self {
        //println!("sync point {:?}", deps);
        SyncPoint {
            deps: AtomicI32::new(deps as i32),
            waiting_jobs: AtomicLinkedList::new(),
        }
    }

    pub fn signal(&self, ctx: &mut Context) -> bool {
        //println!("signal");
        profiling::scope!("signal");

        let prev = self.deps.fetch_add(-1, Ordering::SeqCst);

        if prev > 1 {
            return false;
        }

        assert!(prev == 1, "signaled too many time");

        let mut first = None;
        self.waiting_jobs.pop_all(&mut |job| {
            if first.is_none() {
                first = Some(job);
            } else {
                ctx.schedule_job(job)
            }
        });

        if let Some(job) = first {
            // TODO: this can create an unbounded recursion.
            unsafe {
                job.execute(ctx);
            }
        }

        true
    }

    pub fn has_unresolved_dependencies(&self) -> bool {
        self.deps.load(Ordering::SeqCst) > 0
    }

    pub(crate) fn then(&self, ctx: &mut Context, job: JobRef) {
        let deps = self.deps.load(Ordering::SeqCst);

        if deps <= 0 {
            debug_assert_eq!(deps, 0);
            ctx.schedule_job(job);
        }

        self.waiting_jobs.push(job);

        // This can be called concurrently with `signal`, its possible for `deps` to be read here
        // before decrementing it in `signal` but submitting the jobs happens before `push`.
        // This sequence means the sync point ends up signaled with a job sitting in the waiting list.
        // To prevent that we check `deps` a second time and submit again if it has reached zero in
        // the mean time.
        let deps = self.deps.load(Ordering::SeqCst);

        if deps <= 0 {
            debug_assert_eq!(deps, 0);
            self.waiting_jobs.pop_all(&mut |job| {
                ctx.schedule_job(job);
            });
        }
    }
}

unsafe impl Sync for SyncPoint {}
unsafe impl Send for SyncPoint {}

pub struct Event {
    mutex: Mutex<bool>,
    cond: Condvar,
}

impl Event {
    pub fn new() -> Self {
        Event {
            mutex: Mutex::new(false),
            cond: Condvar::new(),
        }
    }

    pub fn set(&self) {
        profiling::scope!("event:set");
        let mut is_set = self.mutex.lock().unwrap();
        *is_set = true;
        self.cond.notify_all();
    }

    pub fn peek(&self) -> bool {
        let is_set = self.mutex.lock().unwrap();
        *is_set
    }

    pub fn wait(&self) {
        let mut is_set = self.mutex.lock().unwrap();
        if *is_set {
            return;
        }

        {
            profiling::scope!("event:wait");

            while !*is_set {
                is_set = self.cond.wait(is_set).unwrap();
            }
        }
    }

    pub fn set_job(event: Arc<Event>) -> JobRef {
        unsafe {
            Box::new(HeapJob::new(move |_| { event.set() })).as_job_ref()
        }
    }
}

unsafe impl Send for Event {}
unsafe impl Sync for Event {}

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
                let first = self.first.load(Ordering::SeqCst);
                (*node).next.store(first, Ordering::SeqCst);

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
            node = self.first.load(Ordering::SeqCst);
            if self.first.compare_exchange(node, std::ptr::null_mut(), Ordering::SeqCst, Ordering::SeqCst).is_ok() {
                break;
            }
        }

        // Now that we have exclusive access to the nodes, we can execute the callback.
        while !node.is_null() {
            unsafe {                
                let next = (*node).next.load(Ordering::Relaxed);

                if let Some(payload) = (*node).payload.take() {
                    cb(payload);
                }

                let _ = Box::from_raw(node);

                node = next;
            }
        }
    }
}

impl<T> Drop for AtomicLinkedList<T> {
    fn drop(&mut self) {
        self.pop_all(&mut |_| {});
    }
}

