use std::sync::Arc;
use std::sync::atomic::{Ordering, AtomicU32};

use crossbeam_deque::{Stealer, Worker as WorkerQueue};
use crossbeam_utils::{CachePadded, sync::{Parker, Unparker}};

use crate::job::JobRef;
use crate::thread_pool::{ThreadPool, ThreadPoolBuilder, ThreadPoolId};
use crate::context::{Context, ContextId, ContextPool};
use crate::shutdown::Shutdown;

static NEXT_THREADPOOL_ID: AtomicU32 = AtomicU32::new(0);

/// Data accessible by all contexts from any thread.
pub(crate) struct Shared {
    /// Number of dedicated worker threads.
    num_workers: u32,
    /// Number of contexts. Always greater than the number of workers.
    num_contexts: u32,

    pub stealers: Vec<CachePadded<[Stealer<JobRef>; 2]>>,

    pub sleep: Sleep,

    pub context_pool: ContextPool,

    pub id: ThreadPoolId,

    pub shutdown: Shutdown,

    handlers: ThreadPoolHooks,
}

pub(crate) fn init(params: ThreadPoolBuilder) -> ThreadPool {
    let num_threads = params.num_threads as usize;
    let num_contexts = num_threads + params.num_contexts as usize;

    let mut stealers = Vec::with_capacity(num_contexts);
    let mut queues = Vec::with_capacity(num_threads);
    for _ in 0..(num_threads + 1) {
        let hp = WorkerQueue::new_fifo();
        let lp = WorkerQueue::new_fifo();
        stealers.push(CachePadded::new([
            hp.stealer(),
            lp.stealer(),
        ]));
        queues.push(Some([hp, lp]));
    }

    let (sleep, mut parkers) = Sleep::new(num_threads, num_contexts);

    let shared = Arc::new(Shared {
        num_workers: num_threads as u32,
        num_contexts: num_contexts as u32,

        stealers,

        sleep,

        handlers: ThreadPoolHooks {
            start: params.start_handler,
            exit: params.exit_handler,
        },

        shutdown: Shutdown::new(num_threads as u32),

        id: ThreadPoolId(NEXT_THREADPOOL_ID.fetch_add(1, Ordering::Relaxed)),

        context_pool: ContextPool::with_capacity(params.num_contexts as usize),
    });

    for i in 0..num_threads {
        let mut worker = Worker {
            ctx: Context::new_worker(
                i as u32, num_contexts as u32,
                queues[i].take().unwrap(),
                shared.clone()
            ),
            parker: parkers[i].take().unwrap(),
        };

        let mut builder = std::thread::Builder::new()
            .name((params.name_handler)(i as u32));

        if let Some(stack_size) = params.stack_size {
            builder = builder.stack_size(stack_size);
        }

        let _ = builder.spawn(move || {
            profiling::register_thread!("Worker");

            worker.run();

        }).unwrap();
    }

    for i in num_threads..num_contexts {
        shared.context_pool.recycle(Context::new(
            i as u32,
            num_contexts as u32,
            queues[i].take().unwrap(),
            shared.clone(),
        ));
    }

    ThreadPool { shared }
}

impl Shared {
    pub fn num_workers(&self) -> u32 { self.num_workers }

    pub fn num_contexts(&self) -> u32 { self.num_contexts }
}

pub(crate) struct SleepState {
    pub(crate) unparker: Unparker,
    // The index of the context this one will start searching at next time it tries to steal.
    // Can be used as hint of which context last woke this worker.
    // Since it only guides a heuristic, it doesn't need to be perfectly accurate.
    pub(crate) next_target: AtomicU32,
}

pub(crate) struct Sleep {
    /// Atomic bitfield. Setting the Nth bit to one means the Nth worker thread is sleepy.
    sleepy_workers: AtomicU32,
    sleep_states: Vec<CachePadded<SleepState>>,
}

impl Sleep {
    fn new(num_threads: usize, num_contexts: usize) -> (Self, Vec<Option<Parker>>) {
        let mut parkers = Vec::with_capacity(num_threads);
        let mut sleep_states = Vec::with_capacity(num_threads);

        for i in 0..num_threads {
            let parker = Parker::new();
            sleep_states.push(CachePadded::new(SleepState {
                unparker: parker.unparker().clone(),
                next_target: AtomicU32::new(((i + 1) % num_contexts) as u32),
            }));
            parkers.push(Some(parker));
        }

        let sleepy_worker_bits = (1 << (num_threads as u32)) - 1;


        (
            Sleep {
                sleepy_workers: AtomicU32::new(sleepy_worker_bits),
                sleep_states,
            },
            parkers,
        )
    }

    /// Wake up to n worker threads (stop when they are all awake).
    ///
    /// This function is fairly expensive when it causes a thread to
    /// wake up (most of the time is spent dealing with the condition
    /// variable).
    /// However it is fairly cheap if all workers are already awake.
    pub fn wake(&self, mut n: u32, waker: ContextId) {
        while n > 0 {
            //profiling::scope!("wake workers");
            let mut sleepy_bits = self.sleepy_workers.load(Ordering::Acquire);

            profiling::scope!(&format!("wake {} workers ({:b})", n, sleepy_bits));

            if sleepy_bits == 0 {
                // Everyone is already awake.
                return;
            }

            for i in 0..(self.sleep_states.len() as u32) {
                let bit = 1 << i;
                if sleepy_bits & bit == 0 {
                    continue;
                }

                let prev = self.sleepy_workers.fetch_and(!bit, Ordering::Release);
                if prev & bit == 0 {
                    // Someone else woke the thread up before we got to it.
                    // A good time to refresh our view of the sleep thread bits.
                    sleepy_bits = self.sleepy_workers.load(Ordering::Acquire);

                    if sleepy_bits == 0 {
                        return;
                    }

                    continue;
                }

                let sleep_state = &self.sleep_states[i as usize];
                sleep_state.next_target.store(waker.0, Ordering::Relaxed);

                profiling::scope!("unpark");
                sleep_state.unparker.unpark();

                n -= 1;
                break;
            }
        }
    }

    pub fn mark_sleepy(&self, worker: u32) -> u32 {
        let sleepy_bit = 1 << worker;
        self.sleepy_workers.fetch_or(sleepy_bit, Ordering::SeqCst) | sleepy_bit
    }

    pub fn get_waker_hint(&self, worker_index: usize) -> usize {
        self.sleep_states[worker_index]
            .next_target
            .load(Ordering::Relaxed) as usize
    }

    pub fn set_waker_hint(&self, worker_index: usize, waker: usize) {
        self.sleep_states[worker_index]
            .next_target
            .store(waker as u32, Ordering::Release);
    }

    pub fn wake_all(&self) {
        for state in &self.sleep_states {
            state.unparker.unpark();
        }
    }
}

struct Worker {
    ctx: Context,
    parker: Parker,
}

impl Worker {
    fn run(&mut self) {
        let ctx = &mut self.ctx;
        let shared = Arc::clone(&ctx.shared);

        if let Some(handler) = &shared.handlers.start {
            handler.run(ctx.id().0);
        }

        loop {
            // First see if we have work to do in our own queues.
            while let Some(job) = ctx.fetch_local_job() {
                unsafe {
                    ctx.execute_job(job);
                }
            }

            // See if there is work we can steal from other contexts.
            if let Some(job) = ctx.steal(true) {
                unsafe {
                    ctx.execute_job(job);
                }
                continue;
            }

            debug_assert!(ctx.queues_are_empty());

            if shared.shutdown.is_shutting_down() {
                break;
            }

            // Couldn't find work to do in our or another context's queue, so
            // it's sleepy time.

            let _sleepy_bits = shared.sleep.mark_sleepy(ctx.id().0);

            self.parker.park();
        }

        // Shutdown phase.

        if let Some(handler) = &ctx.shared.handlers.exit {
            handler.run(ctx.id().0);
        }

        shared.shutdown.worker_has_shut_down();
    }
}

pub(crate) struct ThreadPoolHooks {
    start: Option<Box<dyn WorkerHook>>,
    exit: Option<Box<dyn WorkerHook>>,
}

pub trait WorkerHook: Send + Sync {
    fn run(&self, worker_id: u32);
}

impl<F> WorkerHook for F where F: Fn(u32) + Send + Sync + 'static {
    fn run(&self, worker_id: u32) { self(worker_id) }
}
