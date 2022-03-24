use crate::core::event::{Event, EventRef};
use crate::core::job::{JobRef, Job, Priority};
use crate::Context;
use crate::helpers::{Parameters, ContextDataRef, OwnedParameters, owned_parameters};
use crate::ThreadPoolId;
use crate::sync::Arc;

use std::mem;
use std::ops::{Range, Deref, DerefMut};
use std::marker::PhantomData;

pub struct Args<'l, Item, ContextData, ImmutableData> {
    pub item: &'l mut Item,
    pub item_index: u32,
    pub context_data: &'l mut ContextData,
    pub immutable_data: &'l ImmutableData,
}

impl<'l, Item, ContextData, ImmutableData> Deref for Args<'l, Item, ContextData, ImmutableData> {
    type Target = Item;
    fn deref(&self) -> &Item { self.item }
}

impl<'l, Item, ContextData, ImmutableData> DerefMut for Args<'l, Item, ContextData, ImmutableData> {
    fn deref_mut(&mut self) -> &mut Item { self.item }
}

pub struct ForEach<'a, 'b, 'g, 'c, Item, ContextData, ImmutableData, Func> {
    pub items: &'a mut [Item],
    pub inner: Parameters<'c, 'b, 'g, ContextData, ImmutableData>,
    pub function: Func,
    pub range: Range<u32>,
    pub group_size: u32,
    pub parallel: Ratio,
}

pub(crate) fn new_for_each<'i, 'cd, 'id, 'c, Item, ContextData, ImmutableData>(
    inner: Parameters<'c, 'cd, 'id, ContextData, ImmutableData>,
    items: &'i mut [Item],
) -> ForEach<'i, 'cd, 'id, 'c, Item, ContextData, ImmutableData, ()> {
    ForEach {
        range: 0..items.len() as u32,
        items,
        inner,
        function: (),
        group_size: 1,
        parallel: Ratio::DEFAULT,
    }
}

impl<'a, 'b, 'g, 'c, Item, ContextData, ImmutableData, F> ForEach<'a, 'b, 'g, 'c, Item, ContextData, ImmutableData, F>
{
    /// Specify some per-context data that can be mutably accessed by the run function.
    ///
    /// This can be useful to store and reuse some scratch buffers and avoid memory allocations in the
    /// run function.
    ///
    /// The length of the slice must be at least equal to the number of worker threads plus one.
    ///
    /// For best performance make sure the size of the data is a multiple of L1 cache line size (see `CachePadded`).
    #[inline]
    pub fn with_context_data<'w, CtxData: Send>(self, context_data: &'w mut [CtxData]) -> ForEach<'a, 'w, 'g, 'c, Item, CtxData, ImmutableData, F> {
        ForEach {
            items: self.items,
            inner: self.inner.with_context_data(context_data),
            function: self.function,
            range: self.range,
            group_size: self.group_size,
            parallel: self.parallel,
        }
    }

    #[inline]
    pub fn with_immutable_data<'i, Data>(self, immutable_data: &'i Data) -> ForEach<'a, 'b, 'i, 'c, Item, ContextData, Data, F> {
        ForEach {
            items: self.items,
            inner: self.inner.with_immutable_data(immutable_data),
            function: self.function,
            range: self.range,
            group_size: self.group_size,
            parallel: self.parallel,
        }
    }

    /// Restrict processing to a range of the data.
    pub fn with_range(mut self, range: Range<u32>) -> Self {
        assert!(range.end >= range.start);
        assert!(range.end <= self.items.len() as u32);
        self.range = range;

        self
    }

    /// Specify the number below which the scheduler doesn't attempt to split the workload.
    #[inline]
    pub fn with_group_size(mut self, group_size: u32) -> Self {
        self.group_size = group_size.max(1).min(self.items.len() as u32);

        self
    }

    /// Specify the priority of this workload.
    #[inline]
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.inner = self.inner.with_priority(priority);

        self
    }

    // TODO: "parallel ratio" isn't a great name.

    /// Specify the proportion of items that we want to expose to worker threads.
    ///
    /// The remaining items will be processed on this thread.
    #[inline]
    pub fn with_parallel_ratio(mut self, ratio: Ratio) -> Self {
        self.parallel = ratio;

        self
    }

    /// Run this workload with the help of worker threads.
    ///
    /// This function returns after the workload has completed.
    #[inline]
    pub fn run<Func>(self, function: Func)
    where
        Func: Fn(&mut Context, Args<Item, ContextData, ImmutableData>) + Sync + Send,
        Item: Sync + Send,
    {
        if !self.items.is_empty() {
            for_each(self.apply(function));
        }
    }

    /// Run this workload asynchronously on the worker threads
    ///
    /// Returns an object to wait on.
    #[inline]
    pub fn run_async<Func>(self, function: Func) -> JoinHandle<'a, 'b>
    where
        Func: Fn(&mut Context, Args<Item, ContextData, ImmutableData>) + Sync + Send + 'static,
        Item: Sync + Send + 'static,
        ContextData: 'static,
        ImmutableData: 'static,
    {
        for_each_async(self.with_parallel_ratio(Ratio::ONE).apply(function))
    }

    #[inline]
    fn apply<Func>(self, function: Func) -> ForEach<'a, 'b, 'g, 'c, Item, ContextData, ImmutableData, Func>
    where
        Func: Fn(&mut Context, Args<Item, ContextData, ImmutableData>) + Sync + Send,
    {
        ForEach {
            items: self.items,
            inner: self.inner,
            function,
            range: self.range,
            group_size: self.group_size,
            parallel: self.parallel,
        }
    }

    fn dispatch_parameters(&self) -> DispatchParameters {
        DispatchParameters::new(self.inner.context(), self.range.clone(), self.group_size, self.parallel)
    }
}

fn for_each<Item, ContextData, ImmutableData, F>(mut params: ForEach<Item, ContextData, ImmutableData, F>)
where
    F: Fn(&mut Context, Args<Item, ContextData, ImmutableData>) + Sync + Send,
    Item: Sync + Send,
{
    profiling::scope!("for_each");

    unsafe {
        let first_item = params.range.start;
        let parallel = params.dispatch_parameters();

        let event = Event::new(params.range.end - params.range.start, params.inner.context().thread_pool_id());

        // Once we start converting this into jobrefs, it MUST NOT move or be destroyed until
        // we are done waiting on the event.
        let job_data: ArrayJob<Item, ContextData, ImmutableData, F> = ArrayJob::new(
            params.items,
            params.group_size,
            ContextDataRef::from_ref(&mut params.inner),
            params.function,
            &event,
        );

        let priority = params.inner.priority();
        let ctx = params.inner.context_mut();

        // This makes `parallel.range` items available for the thread pool to steal from us.
        // and wakes some workers up if need be.
        // The items from `params.range.start` to `parallel.range.start` are reserved. we will
        // execute them last without exposing them to the worker threads.
        for_each_dispatch(ctx, &job_data, &parallel, priority);

        // Now the interesting bits: We kept around a range of items that for this thread to
        // execute, that workers threads cannot steal. The goal here is to make it as likely
        // as possible for this thread to be the last one finishing a job from this workload.
        // The reason is that if we run out of jobs to execute we have no other choice but to
        // block the thread on a condition variable, and that's bad for two reasons:
        //  - It's rather expensive.
        //  - There's extra latency on top of that from not necessarily being re-scheduled by the
        //    OS as soon as the condvar is signaled.

        // Pull work from our queue and execute it.
        while !event.is_signaled() && ctx.keep_busy() {}

        // Execute the reserved batch of items that we didn't make available to worker threads.
        // Doing this removes some potential for parallelism, but greatly increase the likelihood
        // of not having to block the thread so it usually a win.
        if parallel.range.start > first_item {
            profiling::scope!("mt:job group");
            let (context_data, immutable_data) = job_data.data.get(ctx);
            for item_index in first_item..parallel.range.start {
                let item = &mut params.items[item_index as usize];
                profiling::scope!("mt:job");
                let args = Args {
                    item,
                    item_index,
                    context_data,
                    immutable_data,
                };

                (job_data.function)(ctx, args);
            }

            event.signal(ctx, parallel.range.start - first_item);
        }

        // Hopefully by now all items have been processed. If so, wait will return quickly,
        // otherwise we'll have to block on a condition variable.
        ctx.wait(&event);
    }
}

unsafe fn for_each_dispatch<Item, ContextData, ImmutableData, F>(
    ctx: &mut Context,
    job_data: &ArrayJob<Item, ContextData, ImmutableData, F>,
    dispatch: &DispatchParameters,
    priority: Priority,
)
where
    F: Fn(&mut Context, Args<Item, ContextData, ImmutableData>) + Send,
{
    let mut actual_group_count = dispatch.group_count;
    for group_idx in 0..dispatch.group_count {
        let start = dispatch.range.start + dispatch.initial_group_size * group_idx;
        let end = (start + dispatch.initial_group_size).min(dispatch.range.end);

        // TODO: handle this in DispatchParameters::new
        if start >= end {
            actual_group_count -= 1;
            continue;
        }

        ctx.enqueue_job(job_data.as_job_ref(start..end).with_priority(priority));
    }

    // Waking up worker threads is the expensive part.
    // There's a balancing act between waking more threads now (which means they probably will get
    // to pick work up sooner but we spend time waking them up) or waking fewer of them.
    // Right now we wake at most half of the workers. Workers themselves will wake other workers
    // up if they have enough work.
    ctx.wake(actual_group_count.min((ctx.num_worker_threads() + 1) / 2));
}

fn for_each_async<'a, 'b, Item, ContextData, ImmutableData, F>(mut params: ForEach<Item, ContextData, ImmutableData, F>) -> JoinHandle<'a, 'b>
where
    F: Fn(&mut Context, Args<Item, ContextData, ImmutableData>) + Sync + Send + 'static,
    Item: Sync + Send + 'static,
    ContextData: 'static,
    ImmutableData: 'static,
{
    profiling::scope!("for_each_async");

    unsafe {
        let parallel = params.dispatch_parameters();

        let event = Box::new(Event::new(params.range.end - params.range.start, params.inner.context().thread_pool_id()));

        let data = Box::new(ArrayJob::new(
            params.items,
            params.group_size,
            ContextDataRef::from_ref(&mut params.inner),
            params.function,
            &event,
        ));

        let priority = params.inner.priority();

        for_each_dispatch(
            params.inner.context_mut(),
            &data,
            &parallel,
            priority,
        );

        JoinHandle { event, data: data, _marker: PhantomData }
    }
}

#[derive(Debug)]
struct DispatchParameters {
    range: Range<u32>,
    group_count: u32,
    initial_group_size: u32,
}

impl DispatchParameters {
    fn new(ctx: &Context, item_range: Range<u32>, group_size: u32, mut parallel_ratio: Ratio) -> Self {
        if parallel_ratio.dividend == 0 {
            parallel_ratio.dividend = 1;
        }
        if parallel_ratio.divisor < parallel_ratio.dividend {
            parallel_ratio.divisor = parallel_ratio.dividend;
        }

        let n = item_range.end - item_range.start;

        let num_parallel = (parallel_ratio.dividend as u32 * n) / parallel_ratio.divisor as u32;
        let first_parallel = item_range.start + n - num_parallel;
        let group_count = div_ceil(num_parallel, group_size).min(ctx.num_worker_threads() * 2);
        let initial_group_size = if group_count == 0 { 0 } else { div_ceil(num_parallel, group_count) };

        DispatchParameters {
            range: first_parallel..item_range.end,
            group_count,
            initial_group_size,
        }
    }
}

/// A job that represents an array-like workload, for example updating a slice of items in parallel.
///
/// Once the workload starts, the object must NOT move or be dropped until the workload completes.
/// Typically this structure leaves on the stack if we know the workload to end within this stack
/// frame (see `for_each`) or on the heap otherwise.
struct ArrayJob<Item, ContextData, ImmutableData, Func> {
    items: *mut Item,
    data: ContextDataRef<ContextData, ImmutableData>,
    function: Func,
    event: EventRef,
    range: Range<u32>,
    split_thresold: u32,
}

impl<Item, ContextData, ImmutableData, Func> Job for ArrayJob<Item, ContextData, ImmutableData, Func>
where
    Func: Fn(&mut Context, Args<Item, ContextData, ImmutableData>) + Send,
{
    unsafe fn execute(this: *const Self, ctx: &mut Context, range: Range<u32>) {
        let this: &Self = mem::transmute(this);
        let n = range.end - range.start;

        assert!(range.start >= this.range.start && range.end <= this.range.end);

        let (context_data, immutable_data) = this.data.get(ctx);

        for item_idx in range {
            // SAFETY: The situation for the item pointer is the same as with context_data.
            // The pointer can be null, but when it is the case, the type is always ().
            let item = &mut *this.items.wrapping_offset(item_idx as isize);
            profiling::scope!("job");
            let args = Args {
                item,
                item_index: item_idx,
                context_data,
                immutable_data,
            };

            (this.function)(ctx, args);
        }

        this.event.signal(ctx, n);
    }
}

impl<Item, ContextData, ImmutableData, Func> ArrayJob<Item, ContextData, ImmutableData, Func>
where
    Func: Fn(&mut Context, Args<Item, ContextData, ImmutableData>) + Send,
{
    pub unsafe fn new(
        items: &mut[Item],
        split_thresold: u32,
        data: ContextDataRef<ContextData, ImmutableData>,
        function: Func,
        event: &Event,
    ) -> Self {
        ArrayJob {
            items: items.as_mut_ptr(),
            data,
            function,
            event: event.unsafe_ref(),
            range: 0..(items.len() as u32),
            split_thresold,
        }
    }

    pub unsafe fn as_job_ref(&self, range: Range<u32>) -> JobRef {
        assert!(range.start >= self.range.start);
        assert!(range.end <= self.range.end);
        JobRef::with_range(self, range, self.split_thresold)
    }
}

pub struct JoinHandle<'a, 'b> {
    event: Box<Event>,
    #[allow(unused)]
    data: Box<dyn std::any::Any>,
    _marker: PhantomData<(&'a (), &'b ())>,
}

impl<'a, 'b> JoinHandle<'a, 'b> {
    pub fn wait(self, ctx: &mut Context) {
        ctx.wait(&self.event);
    }
}

impl<'a, 'b> Drop for JoinHandle<'a, 'b> {
    fn drop(&mut self) {
        self.event.wait_no_context();
    }
}

pub struct Workload<Item, ContextData, ImmutableData> {
    pub items: Vec<Item>,
    parameters: OwnedParameters<ContextData, ImmutableData>,
    range: Range<u32>,
    group_size: u32,
    event: Box<Event>,
    ext: Option<Box<WorkloadExtension<Item>>>,
}

pub struct RunningWorkload<Item, ContextData, ImmutableData> {
    items: Vec<Item>,
    parameters: OwnedParameters<ContextData, ImmutableData>,
    range: Range<u32>,
    group_size: u32,
    #[allow(unused)]
    data: Box<dyn std::any::Any>,
    event: Option<Box<Event>>,
    ext: Option<Box<WorkloadExtension<Item>>>,
}

pub fn range_workload(range: Range<u32>) -> Workload<(), (), ()> {
    workload(vec![(); range.end as usize]).with_range(range)
}

pub fn workload<Item>(items: Vec<Item>) -> Workload<Item, (), ()> {
    Workload {
        range: 0..(items.len() as u32),
        items,
        parameters: owned_parameters(),
        group_size: 1,
        event: Box::new(Event::new(0, ThreadPoolId(std::u32::MAX))),
        ext: None,
    }
}

impl<Item, ContextData, ImmutableData> Workload<Item, ContextData, ImmutableData> {
    #[inline]
    pub fn with_context_data<CtxData>(self, ctx_data: Vec<CtxData>) -> Workload<Item, CtxData, ImmutableData> {
        Workload {
            items: self.items,
            parameters: self.parameters.with_context_data(ctx_data),
            range: self.range,
            group_size: self.group_size,
            event: self.event,
            ext: self.ext,
        }
    }

    #[inline]
    pub fn with_immutable_data<Data>(self, immutable_data: Arc<Data>) -> Workload<Item, ContextData, Data> {
        Workload {
            items: self.items,
            parameters: self.parameters.with_immutable_data(immutable_data),
            range: self.range,
            group_size: self.group_size,
            event: self.event,
            ext: self.ext,
        }
    }

    #[inline]
    pub fn with_range(mut self, range: Range<u32>) -> Self {
        assert!(range.end >= range.start);
        assert!(range.end <= self.items.len() as u32);
        self.range = range;

        self
    }

    #[inline]
    pub fn with_group_size(mut self, group_size: u32) -> Self {
        self.group_size = group_size;

        self
    }

    #[inline]
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.parameters = self.parameters.with_priority(priority);

        self
    }

    pub fn context_data(&mut self) -> &mut [ContextData] {
        self.parameters.context_data()
    }

    pub fn submit<F>(mut self, ctx: &mut Context, function: F) -> RunningWorkload<Item, ContextData, ImmutableData>
    where
        F: Fn(&mut Context, Args<Item, ContextData, ImmutableData>) + Sync + Send + 'static,
        ContextData: 'static,
        ImmutableData: 'static,
        Item: 'static,
    {
        unsafe {
            let dispatch = DispatchParameters::new(ctx, self.range.clone(), self.group_size, Ratio::ONE);

            self.event.reset(self.range.end - self.range.start, ctx.thread_pool_id());

            let data = Box::new(ArrayJob::new(
                &mut self.items[..],
                self.group_size,
                ContextDataRef::from_owned(&mut self.parameters, ctx),
                function,
                &self.event,
            ));

            for_each_dispatch(
                ctx,
                &data,
                &dispatch,
                self.parameters.priority(),
            );

            RunningWorkload {
                items: self.items,
                parameters: self.parameters,
                range: self.range,
                group_size: self.group_size,
                data,
                event: Some(self.event),
                ext: None,
            }
        }
    }

    pub fn pop_items(&mut self) -> Option<Vec<Item>> {
        if !self.items.is_empty() {
            return Some(std::mem::take(&mut self.items));
        }

        if let Some(mut ext) = self.ext.take() {
            self.ext = ext.ext;
            return Some(std::mem::take(&mut ext.items));
        }

        None
    }
}


/// A somewhat inelegant way to model a dynamically growing workload. An example use case is
/// glyph rasterization in webrender where we send batches of glyphs throughout the frame
/// building process and gather all of the results at some point towards the end.
struct WorkloadExtension<Item> {
    items: Vec<Item>,
    event: Event,
    data: Option<Box<dyn std::any::Any>>,
    ext: Option<Box<WorkloadExtension<Item>>>,
}

impl<Item, ContextData, ImmutableData> RunningWorkload<Item, ContextData, ImmutableData> {
    pub fn wait(mut self, ctx: &mut Context) -> Workload<Item, ContextData, ImmutableData> {
        self.event.as_ref().unwrap().wait(ctx);

        let mut ext_ref = &self.ext;
        while let Some(ext) = ext_ref {
            ext.event.wait(ctx);
            ext_ref = &ext.ext;
        }

        Workload {
            items: std::mem::take(&mut self.items),
            parameters: self.parameters.take(),
            range: self.range.clone(),
            group_size: self.group_size,
            event: self.event.take().unwrap(),
            ext: self.ext.take(),
        }
    }

    /// Start preparing additional work to add to the running workload.
    ///
    /// The extra items will have access to the same context and immutable data as the original
    /// running workload.
    ///
    /// `WorkloadExt::submit` must be called for the work to happen.
    pub fn extend(&mut self, items: Vec<Item>) -> WorkloadExt<Item, ContextData, ImmutableData> {
        WorkloadExt {
            range: 0..(items.len() as u32),
            items,
            workload: self,
        }
    }
}

impl<Item, ContextData, ImmutableData> Drop for RunningWorkload<Item, ContextData, ImmutableData> {
    fn drop(&mut self) {
        if let Some(event) = &self.event {
            event.wait_no_context();
        }
    }
}

pub struct WorkloadExt<'l, Item, ContextData, ImmutableData> {
    workload: &'l mut RunningWorkload<Item, ContextData, ImmutableData>,
    items: Vec<Item>,
    range: Range<u32>,
}

impl<'l, Item, ContextData, ImmutableData> WorkloadExt<'l, Item, ContextData, ImmutableData> {
    pub fn with_range(mut self, range: Range<u32>) -> Self {
        assert!(range.start <= range.end);
        assert!(range.end <= self.items.len() as u32);
        self.range = range;

        self
    }

    pub fn submit<F>(self, ctx: &mut Context, function: F)
    where
        F: Fn(&mut Context, Args<Item, ContextData, ImmutableData>) + Sync + Send + 'static,
        ContextData: 'static,
        ImmutableData: 'static,
        Item: 'static,
    {
        unsafe {
            let count = self.range.end - self.range.start;
            let dispatch = DispatchParameters::new(ctx, self.range, self.workload.group_size, Ratio::ONE);

            let mut ext = Box::new(WorkloadExtension {
                items: self.items,
                data: None,
                event: Event::new(count, ctx.thread_pool_id()),
                ext: None,
            });

            let data = Box::new(ArrayJob::new(
                &mut ext.items[..],
                self.workload.group_size,
                ContextDataRef::from_owned(&mut self.workload.parameters, ctx),
                function,
                &ext.event,
            ));

            for_each_dispatch(
                ctx,
                &data,
                &dispatch,
                self.workload.parameters.priority(),
            );

            ext.data = Some(data);

            // Enqueue at the end of the list, mainly so that we will conveniently
            // dequeue in FIFO order and be less likely to hit unfinished work.
            let mut next_ref = &mut self.workload.ext;
            while let Some(node) = next_ref {
                next_ref = &mut node.ext;
            }

            *next_ref = Some(ext);
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Ratio {
    pub dividend: u8,
    pub divisor: u8,
}

impl Ratio {
    pub const DEFAULT: Ratio = Ratio { dividend: 4, divisor: 5 };
    pub const ONE: Ratio = Ratio { dividend: 1, divisor: 1 };
    pub const HALF: Ratio = Ratio { dividend: 1, divisor: 2 };
    pub const THREE_QUARTERS: Ratio = Ratio { dividend: 3, divisor: 4 };
    pub const TWO_THIRDS: Ratio = Ratio { dividend: 2, divisor: 3 };
    pub const ONE_THIRD: Ratio = Ratio { dividend: 1, divisor: 3 };
}

fn div_ceil(a: u32, b: u32) -> u32 {
    let d = a / b;
    let r = a % b;
    if r > 0 && b > 0 { d + 1 } else { d }
}

#[test]
fn test_workload() {
    use crate::ThreadPool;

    let pool = ThreadPool::builder().with_worker_threads(3).build();

    let mut ctx = pool.pop_context().unwrap();

    let mut workload_1 = workload(vec![0u32; 8192]).with_group_size(16);
    let mut workload_2 = workload(vec![0u32; 8192]).with_group_size(16);
    let mut workload_3 = workload(vec![0u32; 8192]).with_group_size(16);

    for i in 0..3000 {
        let handle_1 = workload_1.submit(&mut ctx, |_, mut item| { *item += 1; });
        let handle_2 = workload_2.submit(&mut ctx, |_, mut item| { *item += 1; });
        let handle_3 = workload_3.submit(&mut ctx, |_, mut item| { *item += 1; });
        workload_1 = handle_1.wait(&mut ctx);
        workload_2 = handle_2.wait(&mut ctx);
        workload_3 = handle_3.wait(&mut ctx);
        for item in &workload_1.items {
            assert_eq!(*item, i + 1);
        }
        for item in &workload_2.items {
            assert_eq!(*item, i + 1);
        }
        for item in &workload_3.items {
            assert_eq!(*item, i + 1);
        }
    }

    pool.shut_down().wait();
}

#[test]
fn test_range_workload() {
    use crate::ThreadPool;

    let pool = ThreadPool::builder().with_worker_threads(3).build();

    let mut ctx = pool.pop_context().unwrap();

    let w = range_workload(20..100)
        .with_immutable_data(Arc::new(1234u32))
        .submit(&mut ctx, |_, args| {
            assert!(args.item_index >= 20, "Error: {} should be in 20..100", args.item_index);
            assert!(args.item_index < 100, "Error: {} should be in 20..100", args.item_index);
            assert_eq!(args.immutable_data, &1234);
        });

    w.wait(&mut ctx);

    pool.shut_down().wait();
}

#[test]
fn test_simple_for_each() {
    use crate::ThreadPool;
    use std::sync::atomic::{AtomicU32, Ordering};
    static INITIALIZED_WORKERS: AtomicU32 = AtomicU32::new(0);
    static SHUTDOWN_WORKERS: AtomicU32 = AtomicU32::new(0);

    let pool = ThreadPool::builder()
        .with_worker_threads(3)
        .with_contexts(1)
        .with_start_handler(|_id| { INITIALIZED_WORKERS.fetch_add(1, Ordering::SeqCst); })
        .with_exit_handler(|_id| { SHUTDOWN_WORKERS.fetch_add(1, Ordering::SeqCst); })
        .build();

    let mut ctx = pool.pop_context().unwrap();

    for _ in 0..300 {
        let input = &mut [0i32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let worker_data = &mut [0i32, 0, 0, 0];

        ctx.for_each(input)
            .with_context_data(worker_data)
            .run(|ctx, args| {
                let _v: i32 = *args;
                *args.context_data += 1;
                *args.item *= 2;
                //println!(" * worker {:} : {:?} * 2 = {:?}", ctx.id(), _v, item);

                for i in 0..10 {
                    let priority = if i % 2 == 0 { Priority::High } else { Priority::Low };
                    let nested_input = &mut [0i32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
                    ctx.with_priority(priority)
                        .for_each(nested_input)
                        .with_range(3..14)
                        .run(|_, mut args| { *args += 1; });

                    for item in &nested_input[0..3] {
                        assert_eq!(*item, 0);
                    }
                    for item in &nested_input[3..14] {
                        assert_eq!(*item, 1);
                    }
                    for item in &nested_input[14..16] {
                        assert_eq!(*item, 0);
                    }

                    for j in 0..100 {
                        let priority = if j % 2 == 0 { Priority::High } else { Priority::Low };
                        let nested_input = &mut [0i32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
                        ctx.for_each(nested_input)
                            .with_priority(priority)
                            .run(|_, mut item| { *item += 1; });
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

        let handle = ctx.for_each(input).run_async(|ctx, mut args| {

            for _ in 0..10 {
                let nested_input = &mut [0i32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

                let handle = ctx.for_each(nested_input).run_async(|_, mut item| { *item += 1; });
                handle.wait(ctx);

                for item in nested_input {
                    assert_eq!(*item, 1);
                }
            }

            *args *= 2;
            *args.context_data = ();
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
    use crate::ThreadPool;

    let pool = ThreadPool::builder().with_worker_threads(3).build();
    let mut ctx = pool.pop_context().unwrap();
    for _ in 0..100 {
        for n in 0..8 {
            let mut input = vec![0i32; n];

            ctx.for_each(&mut input).run(|_, mut item| { *item += 1; });

            let handle = ctx.for_each(&mut input).run_async(|_, mut item| { *item += 1; });

            handle.wait(&mut ctx);

            for val in &input {
                assert_eq!(*val, 2);
            }
        }
    }
}

#[test]
fn test_extend() {
    use crate::ThreadPool;

    let pool = ThreadPool::builder()
        .with_worker_threads(3)
        .with_contexts(1)
        .build();

    let mut ctx = pool.pop_context().unwrap();

    for _ in 0..100 {
        let mut workload = workload(vec![0u32; 16])
            .submit(&mut ctx, |_, mut args| *args += 1);

        for _ in 0..10 {
            workload.extend(vec![0u32; 16]).submit(&mut ctx, |_, mut args| *args += 1);
        }

        let mut workload = workload.wait(&mut ctx);

        for _ in 0..11 {
            assert_eq!(workload.pop_items(), Some(vec![1u32; 16]));
        }

        assert_eq!(workload.pop_items(), None);
    }

    pool.shut_down().wait();
}

#[cfg(loom)]
#[test]
fn test_loom_for_each() {
    // TODO: this test deadlocks during shutdown, likely because crossbeam_utils::Parker
    // isn't part of loom's visible model. When shutdown starts we first store a shutdown
    // flag (SeqCst) and unpark the worker threads. When executing with loom we can see the
    // worker unparking but they don't see the flag (they should).
    // I've tried to compile crossbeam utils with the crossbeam_loom cfg but the build fails
    // all over the place.

    loom::model(move || {
        let pool = crate::ThreadPool::builder()
            .with_worker_threads(2)
            .with_contexts(1)
            .build();

        let mut ctx = pool.pop_context().unwrap();

        let input = &mut [0i32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let ctx_data = &mut [0i32, 0, 0];

        ctx.for_each(input)
            .with_context_data(ctx_data)
            .run(|_, args| {
                *args.context_data += 1;
                *args.item *= 2;
            });

        assert_eq!(input, &[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]);
        assert_eq!(ctx_data[0] + ctx_data[1] + ctx_data[2], 16);

        pool.recycle_context(ctx);
    });
}
