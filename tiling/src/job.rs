use std::any::Any;
use std::cell::UnsafeCell;
use std::mem;

pub(super) enum JobResult<T> {
    None,
    Ok(T),
    Panic(Box<dyn Any + Send>),
}

/// A `Job` is used to advertise work for other threads that they may
/// want to steal. In accordance with time honored tradition, jobs are
/// arranged in a deque, so that thieves can take from the top of the
/// deque while the main worker manages the bottom of the deque. This
/// deque is managed by the `thread_pool` module.
pub(super) trait Job {
    /// Unsafe: this may be called from a different thread than the one
    /// which scheduled the job, so the implementer must ensure the
    /// appropriate traits are met, whether `Send`, `Sync`, or both.
    unsafe fn execute(this: *const Self, worker: &mut Worker);
}

/// Effectively a Job trait object. Each JobRef **must** be executed
/// exactly once, or else data may leak.
///
/// Internally, we store the job's data in a `*const ()` pointer.  The
/// true type is something like `*const StackJob<...>`, but we hide
/// it. We also carry the "execute fn" from the `Job` trait.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct JobRef {
    pointer: *const (),
    execute_fn: unsafe fn(*const (), *const ()),
}

unsafe impl Send for JobRef {}
unsafe impl Sync for JobRef {}

impl JobRef {
    /// Unsafe: caller asserts that `data` will remain valid until the
    /// job is executed.
    pub(super) unsafe fn new<T>(data: *const T) -> JobRef
    where
        T: Job,
    {
        let fn_ptr: unsafe fn(*const T, &mut Worker) = <T as Job>::execute;

        // erase types:
        JobRef {
            pointer: data as *const (),
            execute_fn: mem::transmute(fn_ptr),
        }
    }

    #[inline]
    pub(super) unsafe fn execute(&self, worker: &mut Worker) {
        (self.execute_fn)(self.pointer, mem::transmute(worker))
    }
}

/// A job that will be owned by a stack slot. This means that when it
/// executes it need not free any heap data, the cleanup occurs when
/// the stack frame is later popped.  The function parameter indicates
/// `true` if the job was stolen -- executed on a different thread.
pub(super) struct StackJob<F, R>
where
    F: FnOnce(&mut Worker) -> R + Send,
    R: Send,
{
    func: UnsafeCell<Option<F>>,
    //result: UnsafeCell<JobResult<R>>,
}

impl<F, R> StackJob<F, R>
where
    F: FnOnce(&mut Worker) -> R + Send,
    R: Send,
{
    pub(super) fn new(func: F) -> StackJob<F, R> {
        StackJob {
            func: UnsafeCell::new(Some(func)),
            //result: UnsafeCell::new(JobResult::None),
        }
    }

    pub(super) unsafe fn as_job_ref(&self) -> JobRef {
        JobRef::new(self)
    }

    //pub(super) unsafe fn into_result(self) -> R {
    //    self.result.into_inner().into_return_value()
    //}
}

impl<F, R> Job for StackJob<F, R>
where
    F: FnOnce(&mut Worker) -> R + Send,
    R: Send,
{
    unsafe fn execute(this: *const Self, worker: &mut Worker) {
        let this = &*this;
        let abort = AbortIfPanic;
        let func = (*this.func.get()).take().unwrap();

        // TODO: in order to support storing the result here we have to integrate
        // the synchronization at this level rather than in the callback otherwise
        // there is nothing keeping the result slot alive for us to write into it.

        //(*this.result.get()) = JobResult::Ok(func(worker));
        func(worker);

        mem::forget(abort);
    }
}

/// Represents a job stored in the heap. Used to implement
/// `scope`. Unlike `StackJob`, when executed, `HeapJob` simply
/// invokes a closure, which then triggers the appropriate logic to
/// signal that the job executed.
///
/// (Probably `StackJob` should be refactored in a similar fashion.)
pub(super) struct HeapJob<BODY>
where
    BODY: FnOnce(&mut Worker) + Send,
{
    job: UnsafeCell<Option<BODY>>,
}

impl<F> HeapJob<F>
where
    F: FnOnce(&mut Worker) + Send,
{
    pub fn new(func: F) -> Self {
        HeapJob {
            job: UnsafeCell::new(Some(func)),
        }
    }

    pub unsafe fn new_ref(func: F) -> JobRef {
        Box::new(Self::new(func)).as_job_ref()
    }

    /// Creates a `JobRef` from this job -- note that this hides all
    /// lifetimes, so it is up to you to ensure that this JobRef
    /// doesn't outlive any data that it closes over.
    pub unsafe fn as_job_ref(self: Box<Self>) -> JobRef {
        let this: *const Self = mem::transmute(self);
        JobRef::new(this)
    }
}

impl<BODY> Job for HeapJob<BODY>
where
    BODY: FnOnce(&mut Worker) + Send,
{
    unsafe fn execute(this: *const Self, worker: &mut Worker) {
        let this: Box<Self> = mem::transmute(this);
        let job = (*this.job.get()).take().unwrap();
        job(worker);
    }
}

impl<T> JobResult<T> {
    /// Convert the `JobResult` for a job that has finished (and hence
    /// its JobResult is populated) into its return value.
    ///
    /// NB. This will panic if the job panicked.
    pub(super) fn into_return_value(self) -> T {
        match self {
            JobResult::None => unreachable!(),
            JobResult::Ok(x) => x,
            JobResult::Panic(x) => resume_unwinding(x),
        }
    }
}



use std::panic::{self, AssertUnwindSafe};
use std::thread;

pub(super) fn resume_unwinding(payload: Box<dyn Any + Send>) -> ! {
    panic::resume_unwind(payload)
}

pub(super) struct AbortIfPanic;

impl Drop for AbortIfPanic {
    fn drop(&mut self) {
        eprintln!("unexpected panic; aborting");
        ::std::process::abort();
    }
}


use std::sync::atomic::{Ordering, AtomicI32, AtomicPtr, AtomicBool};
use std::sync::{Arc, Mutex, Condvar};
use std::marker::PhantomData;
use std::alloc::Layout;

use crossbeam_channel::{Sender, Receiver, unbounded};

struct SyncPtr<T: ?Sized>(*const T);
unsafe impl<T> Send for SyncPtr<T> {}
unsafe impl<T> Sync for SyncPtr<T> {}
impl<T> Copy for SyncPtr<T> {}
impl<T> Clone for SyncPtr<T> { fn clone(&self) -> Self { *self } }
impl<T: ?Sized> SyncPtr<T> {
    unsafe fn get(&self) -> &T { &(*self.0) }
}

struct SyncPtrMut<T: ?Sized>(*mut T);
unsafe impl<T> Send for SyncPtrMut<T> {}
unsafe impl<T> Sync for SyncPtrMut<T> {}
impl<T> Copy for SyncPtrMut<T> {}
impl<T> Clone for SyncPtrMut<T> { fn clone(&self) -> Self { *self } }
impl<T: ?Sized> SyncPtrMut<T> {
    unsafe fn get(&self) -> &mut T { &mut(*self.0) }
}

pub struct ThreadPool {
    worker: Worker,
}

impl ThreadPool {
    pub fn new(num_threads: usize) -> Self {
        let (tx, rx) = unbounded();

        for i in 0..num_threads {
            let mut worker = Worker { tx: tx.clone(), rx: rx.clone(), id: i as u32, job_idx: 0 };
            let _ = std::thread::spawn(move || {
                profiling::register_thread!("Worker");

                worker.run();
            });
        }

        ThreadPool { worker: Worker { tx, rx, id: num_threads as u32, job_idx: 0 } }
    }

    pub fn worker(&mut self) -> &mut Worker { &mut self.worker }
}

pub struct Worker {
    tx: Sender<JobRef>,
    rx: Receiver<JobRef>,
    id: u32,
    job_idx: u32,
}

impl Worker {
    fn schedule_job(&mut self, job: JobRef) {
        profiling::scope!("schedule_job");
        self.tx.send(job).unwrap();
    }

    pub fn id(&self) -> u32 { self.id }

    fn job_index(&self) -> u32 { self.job_idx }

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

    pub fn then(&mut self, sync: &SyncPoint, job: JobRef) {
        sync.then(job, self);
    }

    // pub fn for_each_async<I>(&mut self, input: Vec<I>, cb: F) -> HeapWorkload<I, O>
    // pub fn for_each_async_stack<I>(&mut self, input: &[I], output: &[O], cb: F) -> StackWorkload<I, O, 'scope>
    // pub fn for_each_sync<I>(&mut self, input: &[I], output: &mut[O], cb: F);

    // let scope = worker.scope();
    // let 
    unsafe fn for_each_heap_impl<Input, Output, W, F>(
        &mut self,
        count: usize,
        input: *const Input,
        output: *mut Output,
        worker_data: Option<*mut W>,
        sync: *const SyncPoint,
        cb: &F)
    where
        F: Fn(&mut Worker, &Input, Option<&mut W>) -> Output + Sync + Send,
        Input: Sync + Send,
        Output: Sized,
    {
        // I'm not entirely sure why I have to do this.
        // If I don't wrap the unsafe pointers, the compiler tries to capture
        // the &*T instead of *T.
        let mut sync_ptr = SyncPtr(sync);
        let mut in_ptr = SyncPtr(input);
        let mut out_ptr = SyncPtrMut(output);
        let mut wd_ptr = worker_data.map(|wd| SyncPtrMut(wd));

        for i in 0..(count as isize) {
            unsafe {
                let job = HeapJob::new_ref(
                    move |worker| {
                        let idx = worker.id() as isize;
                        *out_ptr.0.offset(i) = cb(
                            worker,
                            &*in_ptr.0.offset(i),
                            wd_ptr.map(|wd| &mut*wd.0.offset(idx)),
                        );
                        sync_ptr.get().signal(worker);
                    }
                );
                self.schedule_job(job);
            }
        }
    }

    pub fn for_each<Input, Output, W, F>(
        &mut self,
        input: &[Input],
        output: &mut [Output],
        mut worker_data: Option<&mut [W]>,
        cb: F,
    )
    where
        F: Fn(&mut Worker, &Input, Option<&mut W>) -> Output + Sync + Send,
        Input: Sync + Send,
        Output: Sized,
    {
        assert_eq!(input.len(), output.len());
        unsafe {
            let in_ptr = input.as_ptr();
            let out_ptr = output.as_mut_ptr();
            let wd_ptr = worker_data.map(|wd| wd.as_mut_ptr());
            let sync = SyncPoint::new(input.len() as u32);
            let event = Event::new();

            // It is important hat the StackJob exist on the stack until we wait.
            let event_job = StackJob::new(|_| event.set());

            self.then(&sync, event_job.as_job_ref());

            self.for_each_heap_impl(input.len(), in_ptr, out_ptr, wd_ptr, &sync, &cb);

            event.wait();
        }
    }


    pub fn for_each_mut<Input, Output, W, F>(
        &mut self,
        input: &mut [Input],
        output: &mut [Output],
        mut worker_data: Option<&mut [W]>,
        cb: F,
    )
    where
        F: Fn(&mut Worker, &mut Input, Option<&mut W>) -> Output + Sync + Send,
        Input: Sync + Send,
        Output: Sized,
    {
        profiling::scope!("for_each_mut");
        assert!(input.len() == output.len() || mem::size_of::<Output>() == 0);
        unsafe {
            fn div_ceil(a: usize, b: usize) -> usize {
                let d = a / b;
                let r = a % b;

                if (r > 0 && b > 0) {
                    d + 1
                } else {
                    d
                }
            }

            let n = input.len();
            let num_parallel = (n * 4) / 5;
            let first_parallel = n - num_parallel;

            let chunk_size = 5;
            let num_chunks = div_ceil(num_parallel, chunk_size);

            let in_ptr = input.as_mut_ptr();
            let out_ptr = output.as_mut_ptr();
            let wd_ptr = worker_data.as_mut().map(|arr| arr.as_mut_ptr());
            let sync = SyncPoint::new(num_chunks as u32 + 1);
            let event = Event::new();

            //println!(" -- {:?} jobs, {:?} parallel {:?} chunks, chunksize {:?} ", n, num_parallel, num_chunks, chunk_size);

            // It is important hat the StackJob exist on the stack until we wait.
            let event_job = StackJob::new(|_| event.set());

            self.then(&sync, event_job.as_job_ref());

            //panic!();

            //println!("jobs: {:?}, num chunks: {:?} ", n, num_chunks);

            self.for_each_mut_impl(
                input.len(),
                in_ptr,
                out_ptr,
                wd_ptr,
                &sync,
                first_parallel,
                num_chunks,
                chunk_size,
                &cb,
            );

            {
            profiling::scope!("mt:job group");
            //println!("start main thread {:?}", 0..num_parallel);
            for i in 0..first_parallel {
                self.job_idx = i as u32;
                profiling::scope!("mt:job");
                let result = cb(
                    self,
                    &mut input[i],
                    worker_data
                        .as_mut()
                        .map(|arr| &mut arr[self.id() as usize]),
                );

                if output.len() > 0 {
                    output[i] = result;
                }
            }
            }

            //println!("end main thread jobs");

            sync.signal(self);

            {
                profiling::scope!("steal jobs");
                while !event.peek() {
                    if let Ok(job) = self.rx.try_recv() {
                        job.execute(self);
                    } else {
                        break
                    }
                }
            }

            event.wait();
        }
    }

    unsafe fn for_each_mut_impl<Input, Output, W, F>(
        &mut self,
        count: usize,
        input: *mut Input,
        output: *mut Output,
        worker_data: Option<*mut W>,
        sync: *const SyncPoint,
        first_parallel: usize,
        num_chunks: usize,
        chunk_size: usize,
        cb: &F)
    where
        F: Fn(&mut Worker, &mut Input, Option<&mut W>) -> Output + Sync + Send,
        Input: Sync + Send,
        Output: Sized,
    {
        //println!("------- {:?}", 0..first_parallel);
        // I'm not entirely sure why I have to do this.
        // If I don't wrap the unsafe pointers, the compiler tries to capture
        // the &*T instead of *T.
        let mut sync_ptr = SyncPtr(sync);
        let mut in_ptr = SyncPtrMut(input);
        let mut out_ptr = SyncPtrMut(output);
        let mut wd_ptr = worker_data.map(|wd| SyncPtrMut(wd));

        assert!(num_chunks * chunk_size >= (count - first_parallel), "num_chunks: {} * chunk_size: {} >= count: {}", num_chunks, chunk_size, count - first_parallel);

        unsafe {
            let fork_job = HeapJob::new_ref(move |worker| {
                profiling::scope!("schedule jobs");
                for chunk_idx in 0..num_chunks {
                    let start = first_parallel + chunk_size * chunk_idx;
                    let job = HeapJob::new_ref(
                        move |worker| {
                            profiling::scope!("job group");

                            let wroker_idx = worker.id() as isize;
                            let end = (start + chunk_size).min(count);
                            assert!(end > start);
                            //println!(" -- (w{:?}) chunk {:?}", worker.id(), start..end);
                            for job_idx in start..end {
                                profiling::scope!("worker:job");
                                worker.job_idx = job_idx as u32;
                                let job_idx = job_idx as isize;
                                //println!("  -- (w{:?}) job {:?}", worker.id(), job_idx);
                                let result = cb(
                                    worker,
                                    &mut *in_ptr.0.offset(job_idx),
                                    wd_ptr.map(|p| &mut *p.0.offset(wroker_idx)),
                                );

                                if mem::size_of::<Output>() > 0 {
                                    *out_ptr.0.offset(job_idx) = result;
                                }
                            }
                            sync_ptr.get().signal(worker);
                        }
                    );
                    worker.schedule_job(job);
                }
            });
            self.schedule_job(fork_job);
        }
    }
}

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

    pub fn signal(&self, worker: &mut Worker) -> bool {
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
                worker.schedule_job(job)
            }
        });

        if let Some(job) = first {
            // TODO: this can create an unbounded recursion.
            unsafe {
                job.execute(worker);
            }
        }

        true
    }

    pub fn has_unresolved_dependencies(&self) -> bool {
        self.deps.load(Ordering::SeqCst) > 0
    }

    fn then(&self, job: JobRef, worker: &mut Worker) {
        let deps = self.deps.load(Ordering::SeqCst);

        if deps <= 0 {
            debug_assert_eq!(deps, 0);
            worker.schedule_job(job);
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
                worker.schedule_job(job);
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
            //println!("no wait");
            return;
        }

        //println!("slow wait");

        {
            profiling::scope!("event:wait");

            while !*is_set {
                is_set = self.cond.wait(is_set).unwrap();
            }
        }
    }

    fn set_job(event: Arc<Event>) -> JobRef {
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
        let mut node = Box::into_raw(Box::new(Node {
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
        let mut node = std::ptr::null_mut();
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

pub fn no_worker_data() -> Option<&'static mut [()]> { None }
pub fn no_output() -> &'static mut [()] { &mut[] }

#[test]
fn test_simple_workload() {
    let mut pool = ThreadPool::new(3);

    let input = &[0i32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    let output = &mut [0i32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

    pool.worker.for_each(input, output, no_worker_data(), |worker, val, _| {
        println!(" - worker {:} executing job", worker.id());
        val * 2
    });

    assert_eq!(output, &[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]);

    for _ in 0..20000 {
        let input = &mut [0i32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let worker_data = &mut [0i32, 0, 0, 0];

        pool.worker.for_each_mut(input, no_output(), Some(worker_data), |worker, val, wd| {
            let v: i32 = *val;
            *wd.unwrap() += 1;
            *val *= 2;
            println!(" * worker {:} job {:?} : {:?} * 2 = {:?}", worker.id(), worker.job_index(), v, val);
            assert_eq!(v, worker.job_index() as i32);
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
