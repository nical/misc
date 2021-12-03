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

use std::sync::{Mutex, Condvar, Arc};
use std::sync::atomic::{Ordering, AtomicBool, AtomicU32};

use crossbeam_deque::{Stealer, Steal, Worker as WorkerQueue};
use crossbeam_utils::{CachePadded};
use job::*;

pub use sync::SyncPoint;
pub use for_each_mut::{ForEachMut};

// TODO: proper worker thread shutdown.

pub struct ThreadPool {
    ctx: Context,
}

struct Shared {
    /// Number of dedicated worker threads.
    num_workers: u32,
    /// Number of contexts. Always greater than the number of workers.
    num_contexts: u32,

    /// Atomic bitfield. Setting the Nth bit to one means the Nth worker thread is sleepy.
    sleepy_workers: AtomicU32,

    stealers: Vec<CachePadded<Stealer<JobRef>>>,
    sleep_states: Vec<CachePadded<SleepState>>,
}

struct SleepState {
    mutex: Mutex<bool>,
    cond: Condvar,
    // The index of the context this one will start searching at next time it tries to steal.
    // Can be used as hint of which context last woke this worker.
    // Since it only guides a heuristic, it doesn't need to be perfectly accurate.
    next_target: AtomicU32,
}

impl ThreadPool {
    pub fn new(mut num_threads: usize) -> Self {
        num_threads = num_threads.min(32);

        let num_threads = num_threads.min(127);
        let num_contexts = num_threads + 1;

        let mut stealers = Vec::with_capacity(num_contexts);
        let mut queues = Vec::with_capacity(num_threads);
        for _ in 0..(num_threads + 1) {
            let w = WorkerQueue::new_fifo();
            stealers.push(CachePadded::new(w.stealer()));
            queues.push(Some(w));
        }

        let mut sleep_states = Vec::with_capacity(num_threads);
        for i in 0..num_threads {
            sleep_states.push(CachePadded::new(SleepState {
                mutex: Mutex::new(false),
                cond: Condvar::new(),
                next_target: AtomicU32::new(((i + 1) % num_contexts) as u32),
            }));
        }

        let sleepy_worker_bits = (1 << (num_threads as u32)) - 1;

        let shared = Arc::new(Shared {
            num_workers: num_threads as u32,
            num_contexts: num_contexts as u32,
            sleepy_workers: AtomicU32::new(sleepy_worker_bits),
            stealers,
            sleep_states,
        });

        for i in 0..num_threads {
            let mut worker = Context {
                id: i as u32,
                is_worker: true,
                num_contexts: num_contexts as u8,

                queue: queues[i].take().unwrap(),
                shared: Arc::clone(&shared),

                stats: Stats::new(),
            };

            let _ = std::thread::spawn(move || {
                profiling::register_thread!("Worker");

                worker.run_worker();
            });
        }

        ThreadPool {
            ctx: Context {
                id: num_threads as u32,
                is_worker: false,
                num_contexts: num_contexts as u8,

                queue: queues.last_mut().unwrap().take().unwrap(),
                shared,

                stats: Stats::new(),
            }
        }
    }

    pub fn context(&mut self) -> &mut Context { &mut self.ctx }
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

pub struct Context {
    id: u32,
    is_worker: bool,
    num_contexts: u8,

    queue: WorkerQueue<JobRef>,
    shared: Arc<Shared>,

    pub stats: Stats,
}

impl Context {
    pub(crate) fn schedule_job(&mut self, job: JobRef) {
        profiling::scope!("schedule_job");
        //println!("worker:{} schedule a job", self.id());

        self.queue.push(job);
        self.wake_n(1);
    }

    pub fn dispatch_one<F>(&mut self, job: F) where F: FnOnce(&mut Context) + Send {
        unsafe {
            self.schedule_job(HeapJob::new_ref(job));
        }
    }

    /// similar to dispatch_one but does not attempt to wake worker threads.
    pub fn enqueue_heap_job<F>(&mut self, job: F) where F: FnOnce(&mut Context) + Send {
        unsafe {
            self.queue.push(HeapJob::new_ref(job));
        }
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
            group_size: 1, // TODO
            ctx: self,
        }
    }

    pub fn id(&self) -> u32 { self.id }

    pub fn num_worker_threads(&self) -> u32 { self.shared.num_workers }

    pub fn num_context(&self) -> u32 { self.shared.num_contexts }

    fn run_worker(&mut self) {
        let shared = Arc::clone(&self.shared);
        let stealers = &shared.stealers[..];
        loop {
            // First see if we have work to do in our own queue.

            //println!("worker:{} begin pop", self.id());
            while let Some(job) = self.queue.pop() {
                unsafe {
                    self.execute_job(job);
                }
            }
            //println!("worker:{} end pop", self.id());

            // See if there is work we can steal from other contexts.

            let len = stealers.len();
            let mut stolen = None;
            let sleep_state = &shared.sleep_states[self.id() as usize];
            let start = sleep_state.next_target.load(Ordering::Relaxed) as usize;
            // Reset the next_target hint to a default value which is the next context.
            sleep_state.next_target.store((self.id() + 1) % (len as u32), Ordering::Release);

            'stealers: for i in 0..len {
                let idx = (start + i) % len;
                for _ in 0..50 {
                    match stealers[idx].steal_batch_and_pop(&self.queue) {
                        Steal::Success(job) => {
                            //println!("worker:{} stole jobs from {}", self.id(), i);
                            stolen = Some(job);
                            break 'stealers;
                        }
                        Steal::Empty => {
                            continue 'stealers;
                        }
                        Steal::Retry => {}
                    }
                }
            }

            if let Some(job) = stolen.take() {
                unsafe {
                    self.execute_job(job);
                    continue;
                }
            }

            //println!("worker:{} end theft", self.id());

            // Couldn't find work to do in our or another context's queue, so
            // it's sleepy time.

            self.sleep();
        }
    }

    pub fn wait(&mut self, sync: &SyncPoint) {
        sync.wait(self);
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

    fn sleep(&mut self) {
        let sleepy_bit = 1 << self.id();
        let _b = self.shared.sleepy_workers.fetch_or(sleepy_bit, Ordering::Release);

        //println!("worker:{} goes to sleep (sleepy bits {:b})", self.id(), _b | sleepy_bit);
        debug_assert!(self.queue.is_empty());

        let sleep_state = &self.shared.sleep_states[self.id() as usize];

        {
            let mut wake_up = sleep_state.mutex.lock().unwrap();
            while !*wake_up {
                wake_up = sleep_state.cond.wait(wake_up).unwrap();
            }
            // Reset for next time.
            *wake_up = false;
        }
        //println!("worker:{} wakes up", self.id());
    }

    pub fn wake_n(&mut self, mut n: u32) {
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
                let mut wake_up = sleep_state.mutex.lock().unwrap();
                *wake_up = true;
                sleep_state.cond.notify_one();

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
    let mut pool = ThreadPool::new(3);
    for _ in 0..100 {
        let input = &mut [0i32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let worker_data = &mut [0i32, 0, 0, 0];

        pool.ctx.for_each_mut(input)
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

    for _i in 0..2000 {
        //println!(" - {:?}", _i);
        let input = &mut [0i32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

        let handle = pool.ctx.for_each_mut(input).run_async(|ctx, val, wd| {

            for _ in 0..10 {
                let nested_input = &mut [0i32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
                ctx.for_each_mut(nested_input).run(|_, item, _| { *item += 1; });
                for item in nested_input {
                    assert_eq!(*item, 1);
                }
            }

            *val *= 2;
            *wd = ();
        });

        handle.wait(&mut pool.ctx);

        assert_eq!(input, &[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]);
    }
}

#[test]
fn test_few_items() {
    let mut pool = ThreadPool::new(3);
    for _ in 0..100 {
        for n in 0..8 {
            let mut input = vec![0i32; n];

            pool.ctx.for_each_mut(&mut input).run(|_, item, _| {
                *item += 1;
            });

            let handle = pool.ctx.for_each_mut(&mut input).run_async(|_, item, _| {
                *item += 1;
            });

            handle.wait(&mut pool.ctx);

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

