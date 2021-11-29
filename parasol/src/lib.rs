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

pub use sync::{SyncPoint, Event};
pub use for_each_mut::{ForEachMut};

// TODO: proper worker thread shutdown.

pub struct ThreadPool {
    ctx: Context,
}

impl ThreadPool {
    pub fn new(num_threads: usize) -> Self {
        let (tx, rx) = unbounded();

        let num_threads = num_threads.min(127);
        let worker_count = num_threads as u8 + 1;

        for i in 0..num_threads {
            let mut worker = Context {
                tx: tx.clone(),
                rx: rx.clone(),
                id: i as u32,
                worker_count,
                job_idx: None
            };

            let _ = std::thread::spawn(move || {
                profiling::register_thread!("Worker");

                worker.run();
            });
        }

        ThreadPool {
            ctx: Context {
                tx,
                rx,
                id: num_threads as u32,
                worker_count,
                job_idx: None,
            }
        }
    }

    pub fn context(&mut self) -> &mut Context { &mut self.ctx }
}

pub struct Context {
    tx: Sender<JobRef>,
    rx: Receiver<JobRef>,
    id: u32,
    job_idx: Option<u32>,
    worker_count: u8,
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
            worker_data: None,
            function: (),
            group_size: 5, // TODO
            ctx: self,
        }
    }

    pub fn id(&self) -> u32 { self.id }

    #[allow(unused)]
    fn job_index(&self) -> Option<u32> { self.job_idx }

    pub fn run(&mut self) {
        while let Ok(job) = self.rx.recv() {
            unsafe {
                job.execute(self);
            }
        }
    }

    pub fn try_run_one(&mut self) -> bool {
        if let Ok(job) = self.rx.try_recv() {
            unsafe {
                job.execute(self);
                return true;
            }
        }

        false
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

    for _ in 0..2000 {
        let input = &mut [0i32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let worker_data = &mut [0i32, 0, 0, 0];

        pool.ctx.for_each_mut(input)
            .with_worker_data(worker_data)
            .run(|ctx, item, wd| {
                let v: i32 = *item;
                *wd += 1;
                *item *= 2;
                println!(" * worker {:} job {:?} : {:?} * 2 = {:?}", ctx.id(), ctx.job_index(), v, item);
                assert_eq!(ctx.job_index(), Some(v as u32));
            });

        assert_eq!(input, &[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]);
    }

    for _ in 0..2000 {
        let input = &mut [0i32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

        pool.ctx.for_each_mut(input).run(|ctx, val, wd| {
            let v: i32 = *val;
            *val *= 2;
            *wd = ();
            println!(" * worker {:} job {:?} : {:?} * 2 = {:?}", ctx.id(), ctx.job_index(), v, val);
            assert_eq!(ctx.job_index(), Some(v as u32));
        });

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

