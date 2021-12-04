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

/// Data accessible by all contexts from any thread.
struct Shared {
    /// Number of dedicated worker threads.
    num_workers: u32,
    /// Number of contexts. Always greater than the number of workers.
    num_contexts: u32,

    /// Atomic bitfield. Setting the Nth bit to one means the Nth worker thread is sleepy.
    sleepy_workers: AtomicU32,

    stealers: Vec<CachePadded<Stealer<JobRef>>>,
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
            queue: ctx.queue,
            shared,
            stats: Stats::new(),
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
    queue: WorkerQueue<JobRef>,
}

pub struct Context {
    id: u32,
    is_worker: bool,
    num_contexts: u8,

    queue: WorkerQueue<JobRef>,
    shared: Arc<Shared>,

    pub(crate) stats: Stats,
}

impl Context {
    pub fn id(&self) -> u32 { self.id }

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

    pub fn schedule_one<F>(&mut self, job: F) where F: FnOnce(&mut Context) + Send {
        unsafe {
            self.schedule_job(HeapJob::new_ref(job));
        }
    }

    pub(crate) fn schedule_job(&mut self, job: JobRef) {
        profiling::scope!("schedule_job");
        //println!("worker:{} schedule a job", self.id());

        self.queue.push(job);
        self.wake_n(1);
    }

    pub fn enqueue_job(&mut self, job: JobRef) {
        self.queue.push(job);
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
        }
    }

    pub fn keep_busy(&mut self) -> bool {
        if let Some(job) = self.try_fetch_one_job() {
            unsafe {
                self.execute_job(job);
            }
            return true;
        }

        false
    }

    pub(crate) fn try_fetch_one_job(&mut self) -> Option<JobRef> {
        if let Some(job) = self.queue.pop() {
            return Some(job);
        }

        let stealers = &self.shared.stealers[..];
        let len = stealers.len();
        let start = if self.is_worker {
            let sleep_state = &self.shared.sleep_states[self.id() as usize];
            sleep_state.next_target.load(Ordering::Relaxed) as usize
        } else {
            (self.id() as usize + 1) % len
        };

        'stealers: for i in 0..len {
            let idx = (start + i) % len;
            if idx == self.id() as usize {
                continue;
            }
            for _ in 0..50 {
                match stealers[idx].steal() {
                    Steal::Success(job) => {
                        //println!("worker:{} stole a job from {}", self.id(), i);
                        // We'll try to steal from here again next time.
                        if self.is_worker {
                            let sleep_state = &self.shared.sleep_states[self.id() as usize];
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

        None
    }

    pub(crate) unsafe fn execute_job(&mut self, job: JobRef) {
        job.execute(self);
        self.stats.jobs_executed += 1;
    }

    fn wake_n(&mut self, mut n: u32) {
        profiling::scope!("wake workers");
        while n > 0 {
            let mut sleepy_bits = self.shared.sleepy_workers.load(Ordering::Acquire);
            //println!("worker:{} wake({}) sleepy bits {:b}", self.id(), n, sleepy_bits);

            if sleepy_bits == 0 {
                // Everyone is already awake.
                //println!("worker:{} woke no threads", self.id());
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
                        //println!("worker:{} woke no threads", self.id());
                        return;
                    }

                    continue;
                }

                let sleep_state = &self.shared.sleep_states[i as usize];
                sleep_state.next_target.store(self.id, Ordering::Relaxed);

                sleep_state.unparker.unpark();

                //println!("worker:{} woke {}", self.id(), i);

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

    let mut pool = ThreadPool::builder()
        .with_worker_threads(3)
        .with_start_handler(|_id| { INITIALIZED_WORKERS.fetch_add(1, Ordering::SeqCst); })
        .with_exit_handler(|_id| { SHUTDOWN_WORKERS.fetch_add(1, Ordering::SeqCst); })
        .build();

    let mut ctx = pool.pop_context().unwrap();

    for _ in 0..500 {
        let input = &mut [0i32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let worker_data = &mut [0i32, 0, 0, 0];

        ctx.for_each_mut(input)
            .with_context_data(worker_data)
            .run(|ctx, item, wd| {
                let _v: i32 = *item;
                *wd += 1;
                *item *= 2;
                //println!(" * worker {:} : {:?} * 2 = {:?}", ctx.id(), _v, item);

                for _ in 0..10 {
                    let nested_input = &mut [0i32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
                    ctx.for_each_mut(nested_input).run(|_, item, _| { *item += 1; });
                    for item in nested_input {
                        assert_eq!(*item, 1);
                    }

                    for _ in 0..100 {
                        let nested_input = &mut [0i32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
                        ctx.for_each_mut(nested_input).run(|_, item, _| { *item += 1; });
                        for item in nested_input {
                            let item = *item;
                            assert_eq!(item, 1);
                        }
                    }

                }
            });

        assert_eq!(input, &[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]);
    }

    for _i in 0..500 {
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
    let mut pool = ThreadPool::builder().with_worker_threads(3).build();
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
