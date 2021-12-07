use std::sync::{Mutex, Arc};

use crossbeam_deque::{Steal, Worker as WorkerQueue};

use crate::core::Shared;
use crate::job::{HeapJob, JobRef, Priority};
use crate::sync::SyncPoint;
use crate::array::{ForEach, new_for_each};
use crate::thread_pool::{ThreadPool, ThreadPoolId};


#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ContextId(pub(crate) u32);

impl ContextId {
    pub fn index(&self) -> usize { self.0 as usize }
}

pub struct Context {
    id: u32,
    is_worker: bool,
    num_contexts: u8,
    queues: [WorkerQueue<JobRef>; 2],

    pub(crate) shared: Arc<Shared>,
    pub(crate) stats: Stats,
}

impl Context {
    pub(crate) fn new_worker(id: u32, num_contexts: u32, queues: [WorkerQueue<JobRef>; 2], shared: Arc<Shared>) -> Self {
        Context {
            id,
            is_worker: true,
            num_contexts: num_contexts as u8,
            queues,
            shared,
            stats: Stats::new(),
        }
    }

    pub(crate) fn new(id: u32, num_contexts: u32, queues: [WorkerQueue<JobRef>; 2], shared: Arc<Shared>) -> Self {
        Context {
            id,
            is_worker: false,
            num_contexts: num_contexts as u8,
            queues,
            shared,
            stats: Stats::new(),
        }
    }

    pub fn id(&self) -> ContextId { ContextId(self.id) }

    pub fn thread_pool_id(&self) -> ThreadPoolId {
        self.shared.id
    }

    pub fn is_worker_thread(&self) -> bool {
        self.is_worker
    }

    pub fn num_worker_threads(&self) -> u32 { self.shared.num_workers() }

    /// Returns the total number of contexts, including worker threads.
    pub fn num_contexts(&self) -> u32 { self.num_contexts as u32 }

    /// Returns a reference to this context's thread pool.
    pub fn thread_pool(&self) -> ThreadPool {
        ThreadPool {
            shared: self.shared.clone(),
        }
    }

    pub fn wait(&mut self, sync: &SyncPoint) {
        sync.wait(self);
    }

    pub fn schedule_one<F>(&mut self, job: F, priority: Priority) where F: FnOnce(&mut Context) + Send {
        unsafe {
            self.schedule_job(HeapJob::new_ref(job).with_priority(priority));
        }
    }

    pub(crate) fn schedule_job(&mut self, job: JobRef) {
        profiling::scope!("schedule_job");

        self.enqueue_job(job);

        self.wake(1);
    }

    pub(crate) fn enqueue_job(&mut self, job: JobRef) {
        self.queues[job.priority().index()].push(job);
    }


    #[inline]
    pub fn for_each<'a, 'c, Item: Send>(&'c mut self, items: &'a mut [Item]) -> ForEach<'a, 'static, 'static, 'c, Item, (), (), ()> {
        new_for_each(self, items)
    }

    pub fn keep_busy(&mut self) -> bool {
        if let Some(job) = self.fetch_job(false) {
            unsafe {
                self.execute_job(job);
            }
            return true;
        }

        false
    }

    pub(crate) fn index(&self) -> usize { self.id as usize }

    pub(crate) fn fetch_job(&mut self, batch: bool) -> Option<JobRef> {
        for queue in &self.queues {
            if let Some(job) = queue.pop() {
                return Some(job);
            }
        }

        self.steal(batch)
    }

    pub(crate) fn fetch_local_job(&mut self) -> Option<JobRef> {
        for queue in &self.queues {
            if let Some(job) = queue.pop() {
                return Some(job);
            }
        }

        None
    }

    /// Attempt to steal a job.
    pub(crate) fn steal(&mut self, batch: bool) -> Option<JobRef> {
        let stealers = &self.shared.stealers[..];
        let len = stealers.len();
        let start = if self.is_worker {
            self.shared.sleep.get_waker_hint(self.index())
        } else {
            self.index()
        };

        'stealers: for i in 0..len {
            let idx = (start + i) % len;
            if idx == self.index() {
                continue;
            }
            for priority in 0..2 {
                for _ in 0..50 {
                    let stealer = &stealers[idx][priority];
                    let steal = if batch {
                        stealer.steal_batch_and_pop(&self.queues[priority])
                    } else {
                        stealer.steal()
                    };

                    match steal {
                        Steal::Success(job) => {
                            // We'll try to steal from here again next time.
                            if self.is_worker {
                                self.shared.sleep.set_waker_hint(self.index(), i);
                            }
                            return Some(job);
                        }
                        Steal::Empty => {
                            continue 'stealers;
                        }
                        Steal::Retry => {}
                    }
                }
            }
        }

        None
    }

    pub(crate) unsafe fn execute_job(&mut self, mut job: JobRef) {
        if let Some(next) = job.split() {
            self.enqueue_job(next);
            self.wake(1);
        }

        job.execute(self);
        self.stats.jobs_executed += 1;
    }

    /// Wake up to n worker threads (stop when they are all awake).
    ///
    /// This function is fairly expensive when it causes a thread to
    /// wake up (most of the time is spent dealing with the condition
    /// variable).
    /// However it is fairly cheap if all workers are already awake.
    pub(crate) fn wake(&mut self, n: u32) {
        self.shared.sleep.wake(n, self.id());
    }

    pub(crate) fn queues_are_empty(&self) -> bool {
        self.queues[0].is_empty() && self.queues[1].is_empty()
    }
}

// We don't store the context itself when recycling it to avoid a reference cycle
// with the shared struct.
struct InactiveContext {
    id: u32,
    is_worker: bool,
    num_contexts: u8,
    queues: [WorkerQueue<JobRef>; 2],
}


pub(crate) struct ContextPool {
    contexts: Mutex<Vec<InactiveContext>>,
}

impl ContextPool {
    pub fn with_capacity(cap: usize) -> ContextPool {
        ContextPool {
            contexts: Mutex::new(Vec::with_capacity(cap))
        }
    }

    pub fn pop(shared: Arc<Shared>) -> Option<Context> {
        let mut contexts = shared.context_pool.contexts.lock().unwrap();
        let shared = shared.clone();
        contexts.pop().map(|ctx| Context {
            id: ctx.id,
            is_worker: ctx.is_worker,
            num_contexts: ctx.num_contexts,
            queues: ctx.queues,
            shared,
            stats: Stats::new(),
        })
    }

    pub fn recycle(&self, ctx: Context) {
        let mut contexts = self.contexts.lock().unwrap();
        contexts.push(InactiveContext {
            id: ctx.id,
            is_worker: ctx.is_worker,
            num_contexts: ctx.num_contexts,
            queues: ctx.queues,
        });
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Stats {
    /// number of jobs executed.
    pub jobs_executed: u64,
    /// How many times waiting on a sync event didn't involve waiting on a condvar.
    pub fast_wait: u64,
    /// How many times waiting on a sync event involved waiting on a condvar.
    pub cond_wait: u64,
    /// number of spinned iterations
    pub spinned: u64,
    /// Number of times we spinning was necessary after waiting for a condvar.
    pub cond_wait_spin: u64,
}

impl Stats {
    pub fn new() -> Self {
        Stats {
            jobs_executed: 0,
            fast_wait: 0,
            cond_wait: 0,
            spinned: 0,
            cond_wait_spin: 0,
        }
    }
}

