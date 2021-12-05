//! An experimental parallel job scheduler with the goal of doing better than rayon
//! specifically in the types of workloads we have in Firefox.
//!
//! What we want:
//! - Allow running jobs outside of the thread pool.
//! - Avoid blocking the thread that submits the work if possible.
//! - No implicit global thread pool.
//! - Ways to safely manage per-worker data.
//! - Avoid hoarding CPU resources in worker threads that don't have work to execute (this
//!   is at the cost of higher latency).
//! - No need to scale to a very large number of threads. We prefer to have something that
//!   runs efficiently on up to 8 threads and not need to scale well above that.
//!
//! Part of the code was copied from rayon (JobRef, HeapJob and StackJob).

mod job;
mod sync;
mod for_each_mut;
mod graph;
mod util;
mod thread_pool;

use std::sync::{Mutex, Condvar, Arc};
use std::sync::atomic::{Ordering, AtomicBool, AtomicU32};

use crossbeam_deque::{Stealer, Steal, Worker as WorkerQueue};
use crossbeam_utils::CachePadded;
use job::*;

pub use sync::SyncPoint;
pub use for_each_mut::{ForEachMut};
pub use thread_pool::{ThreadPool, ThreadPoolId, ThreadPoolBuilder, ShutdownHandle};
use thread_pool::{WorkerHook, SleepState};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ContextId(pub(crate) u32);

impl ContextId {
    pub fn index(&self) -> usize { self.0 as usize }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Priority {
    High,
    Low,
}

impl Priority {
    pub(crate) fn index(&self) -> usize {
        match self {
            Priority::High => 0,
            Priority::Low => 1,
        }
    }
}

/// Data accessible by all contexts from any thread.
struct Shared {
    /// Number of dedicated worker threads.
    num_workers: u32,
    /// Number of contexts. Always greater than the number of workers.
    num_contexts: u32,

    /// Atomic bitfield. Setting the Nth bit to one means the Nth worker thread is sleepy.
    sleepy_workers: AtomicU32,

    stealers: Vec<CachePadded<[Stealer<JobRef>; 2]>>,
    sleep_states: Vec<CachePadded<SleepState>>,

    start_handler: Option<Box<dyn WorkerHook>>,
    exit_handler: Option<Box<dyn WorkerHook>>,

    is_shutting_down: AtomicBool,
    shutdown_mutex: Mutex<u32>,
    shutdown_cond: Condvar,

    contexts: Mutex<Vec<InactiveContext>>,

    id: ThreadPoolId,
}

impl Shared {
    fn pop_context(this: &Arc<Shared>) -> Option<Context> {
        if this.is_shutting_down.load(Ordering::Acquire) {
            return None;
        }

        let shared = this.clone();
        let mut contexts = this.contexts.lock().unwrap();
        contexts.pop().map(|ctx| Context {
            id: ctx.id,
            is_worker: ctx.is_worker,
            num_contexts: ctx.num_contexts,
            queues: ctx.queues,
            shared,
            stats: Stats::new(),
        })
    }

    fn recycle_context(this: &Arc<Shared>, ctx: Context) {
        let mut contexts = this.contexts.lock().unwrap();
        contexts.push(InactiveContext {
            id: ctx.id,
            is_worker: ctx.is_worker,
            num_contexts: ctx.num_contexts,
            queues: ctx.queues,
        })
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
    fn new() -> Self {
        Stats {
            jobs_executed: 0,
            fast_wait: 0,
            cond_wait: 0,
            spinned: 0,
            cond_wait_spin: 0,
        }
    }
}

struct InactiveContext {
    id: u32,
    is_worker: bool,
    num_contexts: u8,
    queues: [WorkerQueue<JobRef>; 2],
}

pub struct Context {
    id: u32,
    is_worker: bool,
    num_contexts: u8,

    queues: [WorkerQueue<JobRef>; 2],

    shared: Arc<Shared>,

    pub(crate) stats: Stats,
}

impl Context {

    pub fn id(&self) -> ContextId { ContextId(self.id) }

    pub(crate) fn index(&self) -> usize { self.id as usize }

    pub fn thread_pool_id(&self) -> ThreadPoolId {
        self.shared.id
    }

    pub fn is_worker_thread(&self) -> bool {
        self.is_worker
    }

    pub fn num_worker_threads(&self) -> u32 { self.shared.num_workers }

    /// Returns the total number of contexts, including worker threads.
    pub fn num_contexts(&self) -> u32 { self.shared.num_contexts }

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
    pub fn for_each_mut<'a, 'c, Item: Send>(&'c mut self, items: &'a mut [Item]) -> ForEachMut<'a, 'static, 'c, Item, (), (), ()> {
        ForEachMut {
            items,
            context_data: None,
            function: (),
            filter: (),
            group_size: 1,
            ctx: self,
            priority: Priority::High,
        }
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

    pub(crate) fn fetch_job(&mut self, batch: bool) -> Option<JobRef> {
        for queue in &self.queues {
            if let Some(job) = queue.pop() {
                return Some(job);
            }
        }

        self.steal(batch)
    }

    /// Attempt to steal a job.
    pub(crate) fn steal(&mut self, batch: bool) -> Option<JobRef> {
        let stealers = &self.shared.stealers[..];
        let len = stealers.len();
        let start = if self.is_worker {
            let sleep_state = &self.shared.sleep_states[self.index()];
            sleep_state.next_target.load(Ordering::Relaxed) as usize
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
                            //println!("worker:{} stole a job from {}", self.id(), i);
                            // We'll try to steal from here again next time.
                            if self.is_worker {
                                let sleep_state = &self.shared.sleep_states[self.index()];
                                sleep_state.next_target.store(i as u32, Ordering::Release);
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
    fn wake(&mut self, mut n: u32) {
        while n > 0 {
            //profiling::scope!("wake workers");
            let mut sleepy_bits = self.shared.sleepy_workers.load(Ordering::Acquire);

            profiling::scope!(&format!("wake {} workers ({:b})", n, sleepy_bits));

            if sleepy_bits == 0 {
                // Everyone is already awake.
                return;
            }

            for i in 0..self.shared.num_workers {
                let bit = 1 << i;
                if sleepy_bits & bit == 0 {
                    continue;
                }

                let prev = self.shared.sleepy_workers.fetch_and(!bit, Ordering::Release);
                if prev & bit == 0 {
                    // Someone else woke the thread up before we got to it.
                    // A good time to refresh our view of the sleep thread bits.
                    sleepy_bits = self.shared.sleepy_workers.load(Ordering::Acquire);

                    if sleepy_bits == 0 {
                        return;
                    }

                    continue;
                }

                let sleep_state = &self.shared.sleep_states[i as usize];
                sleep_state.next_target.store(self.id, Ordering::Relaxed);

                profiling::scope!("unpark");
                sleep_state.unparker.unpark();

                n -= 1;
                break;
            }
        }
    }
}

pub struct ExclusiveCheck<T> {
    lock: AtomicBool,
    tag: T
}

impl<T: std::fmt::Debug> ExclusiveCheck<T> {
    pub fn new() -> Self where T: Default {
        ExclusiveCheck {
            lock: AtomicBool::new(false),
            tag: Default::default(),
        }
    }

    pub fn with_tag(tag: T) -> Self {
        ExclusiveCheck {
            lock: AtomicBool::new(false),
            tag,
        }
    }

    pub fn begin(&self) {
        let res = self.lock.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed);
        assert!(res.is_ok(), "Exclusive check failed (begin): {:?}", self.tag);
    }

    pub fn end(&self) {
        let res = self.lock.compare_exchange(true, false, Ordering::Release, Ordering::Relaxed);
        assert!(res.is_ok(), "Exclusive check failed (end): {:?}", self.tag);
    }
}

#[test]
fn test_simple_workload() {
    static INITIALIZED_WORKERS: AtomicU32 = AtomicU32::new(0);
    static SHUTDOWN_WORKERS: AtomicU32 = AtomicU32::new(0);

    let pool = ThreadPool::builder()
        .with_worker_threads(3)
        .with_start_handler(|_id| { INITIALIZED_WORKERS.fetch_add(1, Ordering::SeqCst); })
        .with_exit_handler(|_id| { SHUTDOWN_WORKERS.fetch_add(1, Ordering::SeqCst); })
        .build();

    let mut ctx = pool.pop_context().unwrap();

    for _ in 0..300 {
        let input = &mut [0i32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let worker_data = &mut [0i32, 0, 0, 0];

        ctx.for_each_mut(input)
            .with_context_data(worker_data)
            .run(|ctx, item, wd| {
                let _v: i32 = *item;
                *wd += 1;
                *item *= 2;
                //println!(" * worker {:} : {:?} * 2 = {:?}", ctx.id(), _v, item);

                for i in 0..10 {
                    let priority = if i % 2 == 0 { Priority::High } else { Priority::Low };
                    let nested_input = &mut [0i32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
                    ctx.for_each_mut(nested_input)
                        .with_priority(priority)
                        .run(|_, item, _| { *item += 1; });
                    for item in nested_input {
                        assert_eq!(*item, 1);
                    }

                    for j in 0..100 {
                        let priority = if j % 2 == 0 { Priority::High } else { Priority::Low };
                        let nested_input = &mut [0i32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
                        ctx.for_each_mut(nested_input)
                            .with_priority(priority)
                            .run(|_, item, _| { *item += 1; });
                        for item in nested_input {
                            let item = *item;
                            assert_eq!(item, 1);
                        }
                    }

                }
            });

        assert_eq!(input, &[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]);
    }

    for _i in 0..300 {
        //println!(" - {:?}", _i);
        let input = &mut [0i32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

        let handle = ctx.for_each_mut(input).run_async(|ctx, val, wd| {

            for _ in 0..10 {
                let nested_input = &mut [0i32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

                let handle = ctx.for_each_mut(nested_input).run_async(|_, item, _| { *item += 1; });
                handle.wait(ctx);

                for item in nested_input {
                    assert_eq!(*item, 1);
                }
            }

            *val *= 2;
            *wd = ();
        });

        handle.wait(&mut ctx);

        assert_eq!(input, &[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]);
    }

    let handle = pool.shut_down();
    handle.wait();

    assert_eq!(INITIALIZED_WORKERS.load(Ordering::SeqCst), 3);
    assert_eq!(SHUTDOWN_WORKERS.load(Ordering::SeqCst), 3);
}

#[test]
fn test_few_items() {
    let pool = ThreadPool::builder().with_worker_threads(3).build();
    let mut ctx = pool.pop_context().unwrap();
    for _ in 0..100 {
        for n in 0..8 {
            let mut input = vec![0i32; n];

            ctx.for_each_mut(&mut input).run(|_, item, _| {
                *item += 1;
            });

            let handle = ctx.for_each_mut(&mut input).run_async(|_, item, _| {
                *item += 1;
            });

            handle.wait(&mut ctx);

            for val in &input {
                assert_eq!(*val, 2);
            }
        }
    }
}

#[test]
fn exclu_check_01() {
    let lock = ExclusiveCheck::with_tag(());

    lock.begin();
    lock.end();

    lock.begin();
    lock.end();

    lock.begin();
    lock.end();
}

#[test]
#[should_panic]
fn exclu_check_02() {
    let lock = ExclusiveCheck::with_tag(());

    lock.begin();
    lock.begin();

    lock.end();
    lock.end();
}

#[test]
fn test_shutdown() {
    static INITIALIZED_WORKERS: AtomicU32 = AtomicU32::new(0);
    static SHUTDOWN_WORKERS: AtomicU32 = AtomicU32::new(0);

    for _ in 0..100 {
        for num_threads in 1..32 {
            INITIALIZED_WORKERS.store(0, Ordering::SeqCst);
            SHUTDOWN_WORKERS.store(0, Ordering::SeqCst);

            let pool = ThreadPool::builder()
                .with_worker_threads(num_threads)
                .with_start_handler(|_id| { INITIALIZED_WORKERS.fetch_add(1, Ordering::SeqCst); })
                .with_exit_handler(|_id| { SHUTDOWN_WORKERS.fetch_add(1, Ordering::SeqCst); })
                .build();

            let handle = pool.shut_down();
            handle.wait();

            assert_eq!(INITIALIZED_WORKERS.load(Ordering::SeqCst), num_threads);
            assert_eq!(SHUTDOWN_WORKERS.load(Ordering::SeqCst), num_threads);
        }
    }
}
