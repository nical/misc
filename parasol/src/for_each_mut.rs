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

pub trait Filter<Item>: Sync + 'static {
    fn is_empty(&self) -> bool;
    fn filter(&self, item: &Item) -> bool;
}

pub struct CallbackFilter<Cb>(Cb);

impl<Item> Filter<Item> for () {
    fn is_empty(&self) -> bool { true }
    fn filter(&self, _: &Item) -> bool { true }
}

impl<Item, F: Fn(&Item) -> bool + Sync + 'static> Filter<Item> for CallbackFilter<F> {
    fn is_empty(&self) -> bool { false }
    fn filter(&self, item: &Item) -> bool { self.0(item) }
}

pub struct ForEachMut<'a, 'b, 'g, 'c, Item, ContextData, ImmutableData, Func, Filtr> {
    pub(crate) items: &'a mut [Item],
    pub(crate) context_data: Option<&'b mut [ContextData]>,
    pub(crate) immutable_data: Option<&'g ImmutableData>,
    pub(crate) function: Func,
    pub(crate) filter: Filtr,
    pub(crate) range: Range<u32>,
    pub(crate) group_size: u32,
    pub(crate) priority: Priority,
    pub(crate) ctx: &'c mut Context,
}

pub fn new_for_each_mut<'a, 'c, Item: Send>(ctx: &'c mut Context, items: &'a mut [Item]) -> ForEachMut<'a, 'static, 'static, 'c, Item, (), (), (), ()> {
    ForEachMut {
        range: 0..items.len() as u32,
        items,
        context_data: None,
        immutable_data: None,
        function: (),
        filter: (),
        group_size: 1,
        ctx,
        priority: Priority::High,
    }
}

impl<'a, 'b, 'g, 'c, Item, ContextData, ImmutableData, F, Filtr> ForEachMut<'a, 'b, 'g, 'c, Item, ContextData, ImmutableData, F, Filtr>
{
    /// Specify some per-context data that can be mutably accessed by the run function.
    ///
    /// This can be useful to store and reuse some scratch buffers and avoid memory allocations in the
    /// run function.
    ///
    /// The length of the slice must be at least equal to the number of contexts.
    #[inline]
    pub fn with_context_data<'w, CtxData: Send>(self, context_data: &'w mut [CtxData]) -> ForEachMut<'a, 'w, 'g, 'c, Item, CtxData, ImmutableData, F, Filtr> {
        assert!(
            context_data.len() >= self.ctx.num_contexts as usize,
            "Got {:?} context items, need at least {:?}",
            context_data.len(), self.ctx.num_contexts,
        );

        ForEachMut {
            items: self.items,
            context_data: Some(context_data),
            immutable_data: self.immutable_data,
            function: self.function,
            filter: self.filter,
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

    /// Pass a callback that can be used to filter out items before processing them in parallel.
    ///
    /// This can be faster than an early-out in the run function because it allows the scheduler
    /// to discard chunks of work before scheduling them.
    #[inline]
    pub fn filter<FilterFn>(self, filter: FilterFn) -> ForEachMut<'a, 'b, 'g, 'c, Item, ContextData, ImmutableData, F, CallbackFilter<FilterFn>>
    where FilterFn: Fn(&Item) -> bool + Sync,
    {
        ForEachMut {
            items: self.items,
            context_data: self.context_data,
            immutable_data: self.immutable_data,
            function: self.function,
            filter: CallbackFilter(filter),
            range: self.range,
            group_size: self.group_size,
            priority: self.priority,
            ctx: self.ctx
        }
    }

    /// Run this workload with the help of worker threads.
    ///
    /// This function returns after the workload has completed.
    #[inline]
    pub fn run<Func>(self, function: Func)
    where
        Func: Fn(&mut Context, Args<Item, ContextData, ImmutableData>) + Sync + Send,
        Item: Sync + Send,
        Filtr: Filter<Item>,
    {
        if !self.items.is_empty() {
            for_each_mut(self.apply(function));
        }
    }

    /// Run this workload asynchronously on the worker threads
    ///
    /// Returns an object to wait on.
    #[inline]
    pub fn run_async<Func>(self, function: Func) -> JoinHandle<'a, 'b>
    where
        Func: Fn(&mut Context, Args<Item, ContextData, ImmutableData>) + Sync + Send + 'static,
        Filtr: Filter<Item>,
        Item: Sync + Send + 'static,
        ContextData: 'static,
        ImmutableData: 'static,
    {
        for_each_mut_async(self.apply(function))
    }

    #[inline]
    fn apply<Func>(self, function: Func) -> ForEachMut<'a, 'b, 'g, 'c, Item, ContextData, ImmutableData, Func, Filtr>
    where
        Func: Fn(&mut Context, Args<Item, ContextData, ImmutableData>) + Sync + Send,
    {
        ForEachMut {
            items: self.items,
            context_data: self.context_data,
            immutable_data: self.immutable_data,
            function,
            filter: self.filter,
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

fn for_each_mut<Item, ContextData, ImmutableData, F, Filtr>(params: ForEachMut<Item, ContextData, ImmutableData, F, Filtr>)
where
    F: Fn(&mut Context, Args<Item, ContextData, ImmutableData>) + Sync + Send,
    Filtr: Filter<Item>,
    Item: Sync + Send,
{
    profiling::scope!("for_each_mut");

    unsafe {
        let first_item = params.range.start;
        let parallel = params.dispatch_parameters(false);

        let sync = SyncPoint::new(params.range.end - params.range.start);

        let ctx = params.ctx;

        // Once we start converting this into jobrefs, it MUST NOT move or be destroyed until
        // we are done waiting on the sync point.
        let job_data: MutSliceJob<Item, ContextData, ImmutableData, F, Filtr> = MutSliceJob::new(
            params.items,
            params.group_size,
            params.context_data,
            params.immutable_data,
            params.function,
            params.filter,
            &sync,
        );

        for_each_mut_dispatch(
            ctx,
            &job_data,
            &parallel,
            &params.items,
            &sync,
            params.priority,
        );

        {
            profiling::scope!("mt:job group");
            for i in first_item..parallel.range.start {
                let item = &mut params.items[i as usize];
                if job_data.filter.filter(item) {
                    profiling::scope!("mt:job");
                    let args = Args {
                        item,
                        item_index: i,
                        context_data: &mut *job_data.ctx_data.offset(ctx.index() as isize),
                        immutable_data: &*job_data.immutable_data,
                    };

                    (job_data.function)(ctx, args);
                }
            }

            sync.signal(ctx, parallel.range.start - first_item);
        }

        ctx.wait(&sync);
    }
}

unsafe fn for_each_mut_dispatch<Item, ContextData, ImmutableData, F, Filtr>(
    ctx: &mut Context,
    job_data: &MutSliceJob<Item, ContextData, ImmutableData, F, Filtr>,
    dispatch: &DispatchParameters,
    items: &[Item],
    sync: &SyncPoint,
    priority: Priority,
)
where
    F: Fn(&mut Context, Args<Item, ContextData, ImmutableData>) + Send,
    Filtr: Filter<Item>,
{
    let mut skipped_items = 0;
    let mut actual_group_count = dispatch.group_count;
    for group_idx in 0..dispatch.group_count {
        let mut start = dispatch.range.start + dispatch.initial_group_size * group_idx;
        let mut end = (start + dispatch.initial_group_size).min(dispatch.range.end);

        if !job_data.filter.is_empty() {
            while start < end && !job_data.filter.filter(&items[start as usize]) {
                skipped_items += 1;
                start += 1;
            }

            while start < end && !job_data.filter.filter(&items[end as usize - 1]) {
                skipped_items += 1;
                end -= 1;
            }
        }

        // If all items in the group are filtered out, skip scheduling it
        // entirely
        if start >= end {
            actual_group_count -= 1;
            continue;
        }

        ctx.enqueue_job(job_data.as_job_ref(start..end).with_priority(priority));
    }

    if skipped_items > 0 {
        sync.signal(ctx, skipped_items);
    }

    // Waking up worker threads is the expensive part.
    ctx.wake(actual_group_count.min(ctx.num_worker_threads()));
}

fn for_each_mut_async<'a, 'b, Item, ContextData, ImmutableData, F, Filtr>(params: ForEachMut<Item, ContextData, ImmutableData, F, Filtr>) -> JoinHandle<'a, 'b>
where
    F: Fn(&mut Context, Args<Item, ContextData, ImmutableData>) + Sync + Send + 'static,
    Filtr: Filter<Item> + 'static,
    Item: Sync + Send + 'static,
    ContextData: 'static,
    ImmutableData: 'static,
{
    profiling::scope!("for_each_mut_async");

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
            params.filter,
            &sync,
        ));

        for_each_mut_dispatch(
            ctx,
            &data,
            &parallel,
            &params.items,
            &sync,
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

pub(crate) struct MutSliceJob<Item, ContextData, ImmutableData, Func, Filtr> {
    items: *mut Item,
    ctx_data: *mut ContextData,
    immutable_data: *const ImmutableData,
    function: Func,
    filter: Filtr,
    sync: SyncPointRef,
    range: Range<u32>,
    split_thresold: u32,
}

impl<Item, ContextData, ImmutableData, Func, Filtr> Job for MutSliceJob<Item, ContextData, ImmutableData, Func, Filtr>
where
    Func: Fn(&mut Context, Args<Item, ContextData, ImmutableData>) + Send,
    Filtr: Filter<Item>,
{
    unsafe fn execute(this: *const Self, ctx: &mut Context, range: Range<u32>) {
        let this: &Self = mem::transmute(this);
        let n = range.end - range.start;

        for item_idx in range {
            let item = &mut *this.items.offset(item_idx as isize);
            if this.filter.filter(item) {
                profiling::scope!("job");
                let args = Args {
                    item,
                    item_index: item_idx,
                    context_data: &mut *this.ctx_data.offset(ctx.index() as isize),
                    immutable_data: &*this.immutable_data,
                };

                (this.function)(ctx, args);
            }
        }

        this.sync.signal(ctx, n);
    }
}

impl<Item, ContextData, ImmutableData, Func, Filtr> MutSliceJob<Item, ContextData, ImmutableData, Func, Filtr>
where
    Func: Fn(&mut Context, Args<Item, ContextData, ImmutableData>) + Send,
    Filtr: Filter<Item>,
{
    pub unsafe fn new(
        items: &mut[Item],
        split_thresold: u32,
        mut ctx_data: Option<&mut [ContextData]>,
        immutable_data: Option<&ImmutableData>,
        function: Func,
        filter: Filtr,
        sync: &SyncPoint,
    ) -> Self {
        MutSliceJob {
            items: items.as_mut_ptr(),
            ctx_data: ctx_data.as_mut().map_or(std::ptr::null_mut(), |arr| arr.as_mut_ptr()),
            immutable_data: immutable_data.map_or(std::ptr::null_mut(), |ptr| ptr),
            function,
            filter,
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
                (),
                &self.sync,
            ));

            for_each_mut_dispatch(
                ctx,
                &data,
                &dispatch,
                &self.items,
                &self.sync,
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