use crate::{Context};
use crate::sync::{SyncPoint, SyncPointRef};

use crate::job::{JobRef, Job};

use std::mem;
use std::ops::Range;
use std::marker::PhantomData;

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

pub struct ForEachMut<'a, 'b, 'c, Item, ContextData, Func, Filtr> {
    pub(crate) items: &'a mut [Item],
    pub(crate) context_data: Option<&'b mut [ContextData]>,
    pub(crate) function: Func,
    pub(crate) filter: Filtr,
    pub(crate) group_size: usize,
    pub(crate) ctx: &'c mut Context,
}

impl<'a, 'b, 'c, Item, ContextData, F, Filtr> ForEachMut<'a, 'b, 'c, Item, ContextData, F, Filtr>
{
    /// Specify some per-context data that can be mutably accessed by the run function.
    ///
    /// This can be useful to store and reuse some scratch buffers and avoid memory allocations in the
    /// run function.
    ///
    /// The length of the slice must be at least equal to the number of contexts.
    #[inline]
    pub fn with_context_data<'w, CtxData: Send>(self, context_data: &'w mut [CtxData]) -> ForEachMut<'a, 'w, 'c, Item, CtxData, F, Filtr> {
        assert!(
            context_data.len() >= self.ctx.num_contexts as usize,
            "Got {:?} context items, need at least {:?}",
            context_data.len(), self.ctx.num_contexts,
        );

        ForEachMut {
            items: self.items,
            context_data: Some(context_data),
            function: self.function,
            filter: self.filter,
            group_size: self.group_size,
            ctx: self.ctx
        }
    }

    /// Specify the number below which the scheduler doesn't attempt to split the workload.
    #[inline]
    pub fn with_group_size(mut self, group_size: usize) -> Self {
        self.group_size = group_size;

        self
    }

    /// Pass a callback that can be used to filter out items before processing them in parallel.
    ///
    /// This can be faster than an early-out in the run function because it allows the scheduler
    /// to discard chunks of work before scheduling them.
    #[inline]
    pub fn filter<FilterFn>(self, filter: FilterFn) -> ForEachMut<'a, 'b, 'c, Item, ContextData, F, CallbackFilter<FilterFn>>
    where FilterFn: Fn(&Item) -> bool + Sync,
    {
        ForEachMut {
            items: self.items,
            context_data: self.context_data,
            function: self.function,
            filter: CallbackFilter(filter),
            group_size: self.group_size,
            ctx: self.ctx
        }
    }

    /// Run this workload with the help of worker threads.
    ///
    /// This function returns after the workload has completed.
    #[inline]
    pub fn run<Func>(self, function: Func)
    where
        Func: Fn(&mut Context, &mut Item, &mut ContextData) + Sync + Send,
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
        Func: Fn(&mut Context, &mut Item, &mut ContextData) + Sync + Send + 'static,
        Filtr: Filter<Item>,
        Item: Sync + Send + 'static,
        ContextData: 'static,
    {
        for_each_mut_async(self.apply(function))
    }

    #[inline]
    fn apply<Func>(self, function: Func) -> ForEachMut<'a, 'b, 'c, Item, ContextData, Func, Filtr>
    where
        Func: Fn(&mut Context, &mut Item, &mut ContextData) + Sync + Send,
    {
        ForEachMut {
            items: self.items,
            context_data: self.context_data,
            function,
            filter: self.filter,
            group_size: self.group_size,
            ctx: self.ctx,
        }
    }

    fn dispatch_parameters(&self, is_async: bool) -> DispatchParameters {
        let n = self.items.len() as u32;

        let num_parallel = if is_async { n } else { (2 * n) / 3 };
        let first_parallel = n - num_parallel;
        let group_size = self.group_size.min(self.items.len()) as u32;
        let group_count = if group_size == 0 { 0 } else { div_ceil(num_parallel, group_size) };

        DispatchParameters {
            item_count: n,
            group_count,
            group_size,
            first_parallel,
        }
    }
}

fn for_each_mut<Item, ContextData, F, Filtr>(params: ForEachMut<Item, ContextData, F, Filtr>)
where
    F: Fn(&mut Context, &mut Item, &mut ContextData) + Sync + Send,
    Filtr: Filter<Item>,
    Item: Sync + Send,
{
    profiling::scope!("for_each_mut");

    unsafe {
        let dispatch = params.dispatch_parameters(false);

        let sync = SyncPoint::new(dispatch.group_count);

        let ctx = params.ctx;

        // Once we start converting this into jobrefs, it MUST NOT move or be destroyed until
        // we are done waiting on the sync point.
        let job_data: MutSliceJob<Item, ContextData, F, Filtr> = MutSliceJob::new(
            params.items,
            params.context_data,
            params.function,
            params.filter,
            &sync,
        );

        for_each_mut_dispatch(
            ctx,
            &job_data,
            &dispatch,
            &params.items,
            &sync,
        );

        {
            profiling::scope!("mt:job group");
            for i in 0..dispatch.first_parallel {
                profiling::scope!("mt:job");
                (job_data.function)(
                    ctx,
                    &mut params.items[i as usize],
                    &mut *job_data.ctx_data.offset(ctx.id() as isize),
                );
            }
        }

        ctx.wait(&sync);
    }
}

unsafe fn for_each_mut_dispatch<Item, ContextData, F, Filtr>(
    ctx: &mut Context,
    job_data: &MutSliceJob<Item, ContextData, F, Filtr>,
    dispatch: &DispatchParameters,
    items: &[Item],
    sync: &SyncPoint,
)
where
    F: Fn(&mut Context, &mut Item, &mut ContextData) + Send,
    Filtr: Filter<Item>,
{
    let mut actual_group_count = dispatch.group_count;
    for group_idx in 0..dispatch.group_count {
        let mut start = dispatch.first_parallel + dispatch.group_size * group_idx;
        let mut end = (start + dispatch.group_size).min(dispatch.item_count);
        debug_assert!(start < end);

        if !job_data.filter.is_empty() {
            while start < end && !job_data.filter.filter(&items[end as usize]) {
                start += 1;
            }

            while start < end && !job_data.filter.filter(&items[start as usize - 1]) {
                end -= 1;
            }

            // If all items in the group are filtered out, skip scheduling it
            // entirely
            if start == end {
                sync.signal(ctx);
                actual_group_count -= 1;
                continue;
            }
        }

        ctx.enqueue_job(job_data.as_job_ref(start..end));
    }

    // Waking up worker threads is the expensive part.
    ctx.wake_n(actual_group_count.min(ctx.num_worker_threads()));
}

fn for_each_mut_async<'a, 'b, Item, ContextData, F, Filtr>(params: ForEachMut<Item, ContextData, F, Filtr>) -> JoinHandle<'a, 'b>
where
    F: Fn(&mut Context, &mut Item, &mut ContextData) + Sync + Send + 'static,
    Filtr: Filter<Item> + 'static,
    Item: Sync + Send + 'static,
    ContextData: 'static,
{
    profiling::scope!("for_each_mut_async");

    unsafe {
        let dispatch = params.dispatch_parameters(true);
        let ctx = params.ctx;

        let sync = Box::new(SyncPoint::new(dispatch.group_count));
        let data = Box::new(MutSliceJob::new(
            params.items,
            params.context_data,
            params.function,
            params.filter,
            &sync,
        ));

        for_each_mut_dispatch(
            ctx,
            &data,
            &dispatch,
            &params.items,
            &sync,
        );

        JoinHandle { sync, data: data, _marker: PhantomData }
    }
}

#[derive(Debug)]
struct DispatchParameters {
    item_count: u32,
    group_count: u32,
    group_size: u32,
    first_parallel: u32,
}

pub(crate) struct MutSliceJob<Item, ContextData, Func, Filtr> {
    items: *mut Item,
    ctx_data: *mut ContextData,
    function: Func,
    filter: Filtr,
    sync: SyncPointRef,
    range: Range<u32>,
}

impl<Item, ContextData, Func, Filtr> Job for MutSliceJob<Item, ContextData, Func, Filtr>
where
    Func: Fn(&mut Context, &mut Item, &mut ContextData) + Send,
    Filtr: Filter<Item>,
{
    unsafe fn execute(this: *const Self, ctx: &mut Context, range: Range<u32>) {
        let this: &Self = mem::transmute(this);

        for item_idx in range {
            let item = &mut *this.items.offset(item_idx as isize);
            if this.filter.filter(item) {
                (this.function)(ctx, item, &mut *this.ctx_data.offset(ctx.id() as isize));
            }
        }

        this.sync.signal(ctx);
    }
}

impl<Item, ContextData, Func, Filtr> MutSliceJob<Item, ContextData, Func, Filtr>
where
    Func: Fn(&mut Context, &mut Item, &mut ContextData) + Send,
    Filtr: Filter<Item>,
{
    pub unsafe fn new(
        items: &mut[Item],
        mut ctx_data: Option<&mut [ContextData]>,
        function: Func,
        filter: Filtr,
        sync: &SyncPoint,
    ) -> Self {
        MutSliceJob {
            items: items.as_mut_ptr(),
            ctx_data: ctx_data.as_mut().map_or(std::ptr::null_mut(), |arr| arr.as_mut_ptr()),
            function,
            filter,
            sync: sync.unsafe_ref(),
            range: 0..(items.len() as u32),
        }
    }

    pub unsafe fn as_job_ref(&self, range: Range<u32>) -> JobRef {
        assert!(range.start >= self.range.start);
        assert!(range.end <= self.range.end);
        JobRef::with_range(self, range)
    }
}

trait Foo {}
impl<T> Foo for T {}

pub struct JoinHandle<'a, 'b> {
    sync: Box<SyncPoint>,
    #[allow(unused)]
    data: Box<dyn Foo>,
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

fn div_ceil(a: u32, b: u32) -> u32 {
    let d = a / b;
    let r = a % b;
    if r > 0 && b > 0 { d + 1 } else { d }
}
