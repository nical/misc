use crate::{Context, Priority};
use crate::sync::{SyncPoint, SyncPointRef};

use crate::job::{JobRef, Job};

use std::mem;
use std::ops::{Range, Deref, DerefMut};
use std::marker::PhantomData;
use std::sync::Arc;

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
    pub(crate) items: &'a mut [Item],
    pub(crate) context_data: Option<&'b mut [ContextData]>,
    pub(crate) immutable_data: Option<&'g ImmutableData>,
    pub(crate) function: Func,
    pub(crate) range: Range<u32>,
    pub(crate) group_size: u32,
    pub(crate) priority: Priority,
    pub(crate) ctx: &'c mut Context,
}

pub fn new_for_each<'a, 'c, Item: Send>(ctx: &'c mut Context, items: &'a mut [Item]) -> ForEach<'a, 'static, 'static, 'c, Item, (), (), ()> {
    ForEach {
        range: 0..items.len() as u32,
        items,
        context_data: None,
        immutable_data: None,
        function: (),
        group_size: 1,
        ctx,
        priority: Priority::High,
    }
}

impl<'a, 'b, 'g, 'c, Item, ContextData, ImmutableData, F> ForEach<'a, 'b, 'g, 'c, Item, ContextData, ImmutableData, F>
{
    /// Specify some per-context data that can be mutably accessed by the run function.
    ///
    /// This can be useful to store and reuse some scratch buffers and avoid memory allocations in the
    /// run function.
    ///
    /// The length of the slice must be at least equal to the number of contexts.
    #[inline]
    pub fn with_context_data<'w, CtxData: Send>(self, context_data: &'w mut [CtxData]) -> ForEach<'a, 'w, 'g, 'c, Item, CtxData, ImmutableData, F> {
        assert!(
            context_data.len() >= self.ctx.num_contexts as usize,
            "Got {:?} context items, need at least {:?}",
            context_data.len(), self.ctx.num_contexts,
        );

        ForEach {
            items: self.items,
            context_data: Some(context_data),
            immutable_data: self.immutable_data,
            function: self.function,
            range: self.range,
            group_size: self.group_size,
            priority: self.priority,
            ctx: self.ctx
        }
    }

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
        self.priority = priority;

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
        for_each_async(self.apply(function))
    }

    #[inline]
    fn apply<Func>(self, function: Func) -> ForEach<'a, 'b, 'g, 'c, Item, ContextData, ImmutableData, Func>
    where
        Func: Fn(&mut Context, Args<Item, ContextData, ImmutableData>) + Sync + Send,
    {
        ForEach {
            items: self.items,
            context_data: self.context_data,
            immutable_data: self.immutable_data,
            function,
            range: self.range,
            group_size: self.group_size,
            priority: self.priority,
            ctx: self.ctx,
        }
    }

    fn dispatch_parameters(&self, is_async: bool) -> DispatchParameters {
        DispatchParameters::new(self.ctx, self.range.clone(), self.group_size, is_async)
    }
}

fn for_each<Item, ContextData, ImmutableData, F>(params: ForEach<Item, ContextData, ImmutableData, F>)
where
    F: Fn(&mut Context, Args<Item, ContextData, ImmutableData>) + Sync + Send,
    Item: Sync + Send,
{
    profiling::scope!("for_each");

    unsafe {
        let first_item = params.range.start;
        let parallel = params.dispatch_parameters(false);

        let sync = SyncPoint::new(params.range.end - params.range.start);

        let ctx = params.ctx;

        // Once we start converting this into jobrefs, it MUST NOT move or be destroyed until
        // we are done waiting on the sync point.
        let job_data: MutSliceJob<Item, ContextData, ImmutableData, F> = MutSliceJob::new(
            params.items,
            params.group_size,
            params.context_data,
            params.immutable_data,
            params.function,
            &sync,
        );

        for_each_dispatch(
            ctx,
            &job_data,
            &parallel,
            params.priority,
        );

        {
            profiling::scope!("mt:job group");
            for i in first_item..parallel.range.start {
                let item = &mut params.items[i as usize];
                profiling::scope!("mt:job");
                let args = Args {
                    item,
                    item_index: i,
                    context_data: &mut *job_data.ctx_data.offset(ctx.index() as isize),
                    immutable_data: &*job_data.immutable_data,
                };

                (job_data.function)(ctx, args);
            }

            sync.signal(ctx, parallel.range.start - first_item);
        }

        ctx.wait(&sync);
    }
}

unsafe fn for_each_dispatch<Item, ContextData, ImmutableData, F>(
    ctx: &mut Context,
    job_data: &MutSliceJob<Item, ContextData, ImmutableData, F>,
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
    ctx.wake(actual_group_count.min(ctx.num_worker_threads()));
}

fn for_each_async<'a, 'b, Item, ContextData, ImmutableData, F>(params: ForEach<Item, ContextData, ImmutableData, F>) -> JoinHandle<'a, 'b>
where
    F: Fn(&mut Context, Args<Item, ContextData, ImmutableData>) + Sync + Send + 'static,
    Item: Sync + Send + 'static,
    ContextData: 'static,
    ImmutableData: 'static,
{
    profiling::scope!("for_each_async");

    unsafe {
        let parallel = params.dispatch_parameters(true);
        let ctx = params.ctx;

        let sync = Box::new(SyncPoint::new(params.items.len() as u32));
        let data = Box::new(MutSliceJob::new(
            params.items,
            params.group_size,
            params.context_data,
            params.immutable_data,
            params.function,
            &sync,
        ));

        for_each_dispatch(
            ctx,
            &data,
            &parallel,
            params.priority,
        );

        JoinHandle { sync, data: data, _marker: PhantomData }
    }
}

#[derive(Debug)]
struct DispatchParameters {
    range: Range<u32>,
    group_count: u32,
    initial_group_size: u32,
}

impl DispatchParameters {
    fn new(ctx: &Context, item_range: Range<u32>, group_size: u32, is_async: bool) -> Self {
        let n = item_range.end - item_range.start;

        let num_parallel = if is_async { n } else { (2 * n) / 3 };
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

pub(crate) struct MutSliceJob<Item, ContextData, ImmutableData, Func> {
    items: *mut Item,
    ctx_data: *mut ContextData,
    immutable_data: *const ImmutableData,
    function: Func,
    sync: SyncPointRef,
    range: Range<u32>,
    split_thresold: u32,
}

impl<Item, ContextData, ImmutableData, Func> Job for MutSliceJob<Item, ContextData, ImmutableData, Func>
where
    Func: Fn(&mut Context, Args<Item, ContextData, ImmutableData>) + Send,
{
    unsafe fn execute(this: *const Self, ctx: &mut Context, range: Range<u32>) {
        let this: &Self = mem::transmute(this);
        let n = range.end - range.start;

        for item_idx in range {
            let item = &mut *this.items.offset(item_idx as isize);
            profiling::scope!("job");
            let args = Args {
                item,
                item_index: item_idx,
                context_data: &mut *this.ctx_data.offset(ctx.index() as isize),
                immutable_data: &*this.immutable_data,
            };

            (this.function)(ctx, args);
        }

        this.sync.signal(ctx, n);
    }
}

impl<Item, ContextData, ImmutableData, Func> MutSliceJob<Item, ContextData, ImmutableData, Func>
where
    Func: Fn(&mut Context, Args<Item, ContextData, ImmutableData>) + Send,
{
    pub unsafe fn new(
        items: &mut[Item],
        split_thresold: u32,
        mut ctx_data: Option<&mut [ContextData]>,
        immutable_data: Option<&ImmutableData>,
        function: Func,
        sync: &SyncPoint,
    ) -> Self {
        MutSliceJob {
            items: items.as_mut_ptr(),
            ctx_data: ctx_data.as_mut().map_or(std::ptr::null_mut(), |arr| arr.as_mut_ptr()),
            immutable_data: immutable_data.map_or(std::ptr::null_mut(), |ptr| ptr),
            function,
            sync: sync.unsafe_ref(),
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
    sync: Box<SyncPoint>,
    #[allow(unused)]
    data: Box<dyn std::any::Any>,
    _marker: PhantomData<(&'a (), &'b ())>,
}

impl<'a, 'b> JoinHandle<'a, 'b> {
    pub fn wait(self, ctx: &mut Context) {
        ctx.wait(&self.sync);
    }
}

impl<'a, 'b> Drop for JoinHandle<'a, 'b> {
    fn drop(&mut self) {
        self.sync.wait_no_context();
    }
}

pub struct Workload<Item, ContextData, ImmutableData> {
    pub items: Vec<Item>,
    pub context_data: Option<Vec<ContextData>>,
    pub immutable_data: Option<Arc<ImmutableData>>,
    range: Range<u32>,
    group_size: u32,
    priority: Priority,
    sync: Box<SyncPoint>,
}

pub struct RunningWorkload<Item, ContextData, ImmutableData> {
    items: Vec<Item>,
    context_data: Option<Vec<ContextData>>,
    immutable_data: Option<Arc<ImmutableData>>,
    range: Range<u32>,
    group_size: u32,
    priority: Priority,
    #[allow(unused)]
    data: Box<dyn std::any::Any>,
    sync: Option<Box<SyncPoint>>,
}

pub fn range_workload(range: Range<u32>) -> Workload<(), (), ()> {
    workload(vec![(); range.end as usize]).with_range(range)
}

pub fn workload<Item>(items: Vec<Item>) -> Workload<Item, (), ()> {
    Workload {
        range: 0..(items.len() as u32),
        items,
        context_data: None,
        immutable_data: None,
        group_size: 1,
        priority: Priority::High,
        sync: Box::new(SyncPoint::new(0)),
    }
}

impl<Item, ContextData, ImmutableData> Workload<Item, ContextData, ImmutableData> {
    pub fn with_context_data<CtxData>(self, ctx_data: Vec<CtxData>) -> Workload<Item, CtxData, ImmutableData> {
        Workload {
            items: self.items,
            context_data: Some(ctx_data),
            immutable_data: self.immutable_data,
            range: self.range,
            group_size: self.group_size,
            priority: self.priority,
            sync: self.sync,
        }
    }

    pub fn with_immutable_data<Data>(self, immutable_data: Arc<Data>) -> Workload<Item, ContextData, Data> {
        Workload {
            items: self.items,
            context_data: self.context_data,
            immutable_data: Some(immutable_data),
            range: self.range,
            group_size: self.group_size,
            priority: self.priority,
            sync: self.sync,
        }
    }

    pub fn with_range(mut self, range: Range<u32>) -> Self {
        assert!(range.end >= range.start);
        assert!(range.end <= self.items.len() as u32);
        self.range = range;

        self
    }

    pub fn with_group_size(mut self, group_size: u32) -> Self {
        self.group_size = group_size;

        self
    }

    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;

        self
    }

    pub fn submit<F>(mut self, ctx: &mut Context, function: F) -> RunningWorkload<Item, ContextData, ImmutableData>
    where
        F: Fn(&mut Context, Args<Item, ContextData, ImmutableData>) + Sync + Send + 'static,
        ContextData: 'static,
        ImmutableData: 'static,
        Item: 'static,
    {
        unsafe {
            let dispatch = DispatchParameters::new(ctx, self.range.clone(), self.group_size, true);

            self.sync.reset(self.range.end - self.range.start);

            let immutable_data: Option<&ImmutableData> = self.immutable_data
                .as_ref()
                .map(|boxed| boxed.deref());

            let data = Box::new(MutSliceJob::new(
                &mut self.items[..],
                self.group_size,
                self.context_data.as_mut().map(|vec| &mut vec[..]),
                immutable_data,
                function,
                &self.sync,
            ));

            for_each_dispatch(
                ctx,
                &data,
                &dispatch,
                self.priority,
            );

            RunningWorkload {
                items: self.items,
                context_data: self.context_data,
                immutable_data: self.immutable_data,
                range: self.range,
                group_size: self.group_size,
                priority: self.priority,
                data,
                sync: Some(self.sync),
            }
        }
    }
}

impl<Item, ContextData, ImmutableData> RunningWorkload<Item, ContextData, ImmutableData> {
    pub fn wait(mut self, ctx: &mut Context) -> Workload<Item, ContextData, ImmutableData> {
        self.sync.as_ref().unwrap().wait(ctx);

        Workload {
            items: std::mem::take(&mut self.items),
            context_data: self.context_data.take(),
            immutable_data: self.immutable_data.take(),
            range: self.range.clone(),
            group_size: self.group_size,
            priority: self.priority,
            sync: self.sync.take().unwrap(),
        }
    }
}

impl<Item, ContextData, ImmutableData> Drop for RunningWorkload<Item, ContextData, ImmutableData> {
    fn drop(&mut self) {
        if let Some(sync) = &self.sync {
            sync.wait_no_context();
        }
    }
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
