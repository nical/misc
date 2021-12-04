use std::sync::{Mutex, Condvar, Arc};
use std::sync::atomic::{Ordering, AtomicBool, AtomicU32};

use crossbeam_deque::{Worker as WorkerQueue, Steal };
use crossbeam_utils::{CachePadded, sync::{Parker, Unparker}};

use crate::{Context, InactiveContext, Shared, Stats};

static NEXT_THREADPOOL_ID: AtomicU32 = AtomicU32::new(0);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ThreadPoolId(u32);

/// A reference to a thread pool.
#[derive(Clone)]
pub struct ThreadPool {
    pub(crate) shared: Arc<Shared>,
}

impl ThreadPool {
    pub fn builder() -> ThreadPoolBuilder {
        ThreadPoolBuilder {
            num_threads: 3,
            num_contexts: 1,
            start_handler: None,
            exit_handler: None,
        }
    }

    pub fn shut_down(&self) -> ShutdownHandle {
        let shared = self.shared.clone();
        shared.is_shutting_down.store(true, Ordering::SeqCst);

        for state in &shared.sleep_states {
            state.unparker.unpark();
        }

        ShutdownHandle { shared }
    }

    pub fn pop_context(&self) -> Option<Context> {
        Shared::pop_context(&self.shared)
    }

    pub fn id(&self) -> ThreadPoolId {
        self.shared.id
    }

    pub fn num_worker_threads(&self) -> u32 { self.shared.num_workers }

    pub fn num_contexts(&self) -> u32 { self.shared.num_contexts }

}

pub struct ThreadPoolBuilder {
    num_threads: u32,
    num_contexts: u32,
    start_handler: Option<Box<dyn WorkerHook>>,
    exit_handler: Option<Box<dyn WorkerHook>>,
}

impl ThreadPoolBuilder {
    // TODO: stack size, thread names

    pub fn with_start_handler<F>(self, handler: F) -> Self
    where F: Fn(u32) + Send + Sync + 'static
    {
        ThreadPoolBuilder {
            num_threads: self.num_threads,
            num_contexts: self.num_contexts,
            start_handler: Some(Box::new(handler)),
            exit_handler: self.exit_handler,
        }
    }

    pub fn with_exit_handler<F>(self, handler: F) -> Self
    where F: Fn(u32) + Send + Sync + 'static
    {
        ThreadPoolBuilder {
            num_threads: self.num_threads,
            num_contexts: self.num_contexts,
            start_handler: self.start_handler,
            exit_handler: Some(Box::new(handler)),
        }
    }

    pub fn with_worker_threads(mut self, num_threads: u32) -> Self {
        self.num_threads = num_threads.max(1).min(31);

        self
    }

    pub fn with_contexts(mut self, num_contexts: u32) -> Self {
        self.num_contexts = num_contexts.max(1).min(127);

        self
    }

    pub fn build(self) -> ThreadPool {
        let num_threads = self.num_threads as usize;
        let num_contexts = num_threads + self.num_contexts as usize;

        let mut stealers = Vec::with_capacity(num_contexts);
        let mut queues = Vec::with_capacity(num_threads);
        for _ in 0..(num_threads + 1) {
            let w = WorkerQueue::new_fifo();
            stealers.push(CachePadded::new(w.stealer()));
            queues.push(Some(w));
        }
        let mut parkers = Vec::new();
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

        let mut contexts = Vec::with_capacity(self.num_contexts as usize);
        for i in num_threads..num_contexts {
            contexts.push(InactiveContext {
                id: i as u32,
                is_worker: false,
                num_contexts: num_contexts as u8,

                queue: queues[i].take().unwrap(),
            });
        }


        let shared = Arc::new(Shared {
            num_workers: num_threads as u32,
            num_contexts: num_contexts as u32,
            sleepy_workers: AtomicU32::new(sleepy_worker_bits),
            stealers,
            sleep_states,
            start_handler: self.start_handler,
            exit_handler: self.exit_handler,

            is_shutting_down: AtomicBool::new(false),
            shutdown_mutex: Mutex::new(num_threads as u32),
            shutdown_cond: Condvar::new(),

            id: ThreadPoolId(NEXT_THREADPOOL_ID.fetch_add(1, Ordering::Relaxed)),

            contexts: Mutex::new(contexts),
        });

        for i in 0..num_threads {
            let mut worker = Worker {
                ctx: Context {
                    id: i as u32,
                    is_worker: true,
                    num_contexts: num_contexts as u8,

                    queue: queues[i].take().unwrap(),
                    shared: Arc::clone(&shared),

                    stats: Stats::new(),
                },
                parker: parkers[i].take().unwrap(),
            };

            let _ = std::thread::spawn(move || {
                profiling::register_thread!("Worker");

                if let Some(handler) = &worker.ctx.shared.start_handler {
                    handler.run(worker.ctx.id());
                }

                worker.run();

                if let Some(handler) = &worker.ctx.shared.exit_handler {
                    handler.run(worker.ctx.id());
                }

                let shared = worker.ctx.shared.clone();
                let mut num_workers = shared.shutdown_mutex.lock().unwrap();
                *num_workers -= 1;
                if *num_workers == 0 {
                    shared.shutdown_cond.notify_all();
                }
            });
        }

        ThreadPool { shared }
    }
}

pub(crate) struct SleepState {
    pub(crate) unparker: Unparker,
    // The index of the context this one will start searching at next time it tries to steal.
    // Can be used as hint of which context last woke this worker.
    // Since it only guides a heuristic, it doesn't need to be perfectly accurate.
    pub(crate) next_target: AtomicU32,
}

struct Worker {
    ctx: Context,
    parker: Parker,
}

impl Worker {
    fn run(&mut self) {
        let ctx = &mut self.ctx;
        let shared = Arc::clone(&ctx.shared);
        let stealers = &shared.stealers[..];
        loop {
            // First see if we have work to do in our own queue.
            while let Some(job) = ctx.queue.pop() {
                unsafe {
                    ctx.execute_job(job);
                }
            }

            // See if there is work we can steal from other contexts.

            let len = stealers.len();
            let mut stolen = None;
            let sleep_state = &shared.sleep_states[ctx.id() as usize];
            let start = sleep_state.next_target.load(Ordering::Relaxed) as usize;
            // Reset the next_target hint to a default value which is the next context.
            sleep_state.next_target.store((ctx.id() + 1) % (len as u32), Ordering::Release);

            'stealers: for i in 0..len {
                let idx = (start + i) % len;
                for _ in 0..50 {
                    match stealers[idx].steal_batch_and_pop(&ctx.queue) {
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
                    ctx.execute_job(job);
                    continue;
                }
            }

            if shared.is_shutting_down.load(Ordering::SeqCst) {
                if ctx.queue.is_empty() {
                    return;
                } else {
                    continue;
                }
            }

            // Couldn't find work to do in our or another context's queue, so
            // it's sleepy time.

            let sleepy_bit = 1 << ctx.id();

            let _b = ctx.shared.sleepy_workers.fetch_or(sleepy_bit, Ordering::Release);
            //println!("worker:{} goes to sleep (sleepy bits {:b})", self.id(), _b | sleepy_bit);

            debug_assert!(ctx.queue.is_empty());

            self.parker.park();
        }
    }
}

pub struct ShutdownHandle {
    shared: Arc<Shared>
}

impl ShutdownHandle {
    pub fn wait(self) {
        let mut num_workers = self.shared.shutdown_mutex.lock().unwrap();
        while *num_workers > 0 {
            num_workers = self.shared.shutdown_cond.wait(num_workers).unwrap();
        }

        self.shared.is_shutting_down.store(false, Ordering::Release);
    }
}

pub trait WorkerHook: Send + Sync {
    fn run(&self, worker_id: u32);
}

impl<F> WorkerHook for F where F: Fn(u32) + Send + Sync + 'static {
    fn run(&self, worker_id: u32) { self(worker_id) }
}
