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

use std::sync::atomic::{Ordering, AtomicBool};

use crossbeam_channel::{Sender, Receiver, unbounded};
use job::*;

pub use sync::SyncPoint;
pub use for_each_mut::{ForEachMut};

// TODO: proper worker thread shutdown.

pub struct ThreadPool {
    ctx: Context,
}

impl ThreadPool {
    pub fn new(num_threads: usize) -> Self {
        let (tx, rx) = unbounded();

        let num_threads = num_threads.min(127);
        let num_contexts = num_threads as u8 + 1;

        for i in 0..num_threads {
            let mut worker = Context {
                tx: tx.clone(),
                rx: rx.clone(),
                id: i as u32,
                num_contexts,

                stats: Stats::new(),
            };

            let _ = std::thread::spawn(move || {
                profiling::register_thread!("Worker");

                worker.run_worker();
            });
        }

        ThreadPool {
            ctx: Context {
                tx,
                rx,
                id: num_threads as u32,
                num_contexts,

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
    tx: Sender<JobRef>,
    rx: Receiver<JobRef>,
    id: u32,
    num_contexts: u8,

    pub stats: Stats,
}

impl Context {
    pub(crate) fn schedule_job(&mut self, job: JobRef) {
        profiling::scope!("schedule_job");
        self.tx.send(job).unwrap();
    }

    #[inline]
    pub fn for_each_mut<'a, 'c, Item: Send>(&'c mut self, items: &'a mut [Item]) -> ForEachMut<'a, 'static, 'c, Item, (), ()> {
        ForEachMut {
            items,
            context_data: None,
            function: (),
            group_size: 5, // TODO
            ctx: self,
        }
    }

    pub fn id(&self) -> u32 { self.id }

    fn run_worker(&mut self) {
        while let Ok(job) = self.rx.recv() {
            unsafe {
                job.execute(self);
                self.stats.jobs_executed += 1;
            }
        }
    }

    pub fn dispatch_one<F>(&mut self, job: F) where F: FnOnce(&mut Context) + Send {
        unsafe {
            self.schedule_job(HeapJob::new_ref(job));
        }
    }

    pub fn wait(&mut self, sync: &SyncPoint) {
        sync.wait(self);
    }

    pub fn try_steal_one(&mut self) -> bool {
        if let Ok(job) = self.rx.try_recv() {
            unsafe {
                self.execute_job(job);
                return true;
            }
        }

        false
    }

    pub(crate) unsafe fn execute_job(&mut self, job: JobRef) {
        job.execute(self);
        self.stats.jobs_executed += 1;
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
        let res = self.lock.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst);
        assert!(res.is_ok(), "Exclusive check failed (begin): {:?}", self.tag);
    }

    pub fn end(&self) {
        let res = self.lock.compare_exchange(true, false, Ordering::SeqCst, Ordering::SeqCst);
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
                let v: i32 = *item;
                *wd += 1;
                *item *= 2;
                //println!(" * worker {:} : {:?} * 2 = {:?}", ctx.id(), v, item);

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

