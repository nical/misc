use crate::{Context};
use crate::util::SyncPtrMut;
use crate::sync::{SyncPoint, SyncPointRef};

use smallvec::SmallVec;
use crate::job::MutSliceJob;

use std::marker::PhantomData;

pub trait Filter<Item>: Sync {
    fn is_empty(&self) -> bool;
    fn filter(&self, item: &Item) -> bool;
}

pub struct CallbackFilter<Cb>(Cb);

impl<Item> Filter<Item> for () {
    fn is_empty(&self) -> bool { true }
    fn filter(&self, _: &Item) -> bool { true }
}

impl<Item, F: Fn(&Item) -> bool + Sync> Filter<Item> for CallbackFilter<F> {
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
    pub fn with_context_data<'w, W: Send>(self, context_data: &'w mut [W]) -> ForEachMut<'a, 'w, 'c, Item, W, F, Filtr> {
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
        Func: Fn(&mut Context, &mut Item, &mut ContextData) + Sync + Send,
        Item: Sync + Send,
        Filtr: Filter<Item>,
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

    unsafe fn unsafe_pointers(&mut self) -> (SyncPtrMut<Item>, SyncPtrMut<ContextData>) {
        // Note: the safety of how we manipulate wd_ptr relies on that `ForEachMut::context_data` can only
        // be null if its type is `()` due to how `ForEachMut` is constructed. As a result we can safely
        // mess with the pointer when it is null because () makes sure we'll never actually read or write
        // the pointed memory.
        let in_ptr = self.items.as_mut_ptr();
        let wd_ptr = self.context_data.as_mut().map_or(std::ptr::null_mut(), |arr| arr.as_mut_ptr());

        (SyncPtrMut(in_ptr), SyncPtrMut(wd_ptr))
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

/*
fn for_each_mut<Item, W, F, Filtr>(mut params: ForEachMut<Item, W, F, Filtr>)
where
    F: Fn(&mut Context, &mut Item, &mut W) + Sync + Send,
    Filtr: Filter<Item>,
    Item: Sync + Send,
{
    profiling::scope!("for_each_mut");

    unsafe {
        let (in_ptr, wd_ptr) = params.unsafe_pointers();
        let dispatch = params.dispatch_parameters(false);

        let sync = SyncPoint::new(dispatch.group_count);

        let ctx = params.ctx;

        for_each_mut_dispatch(
            ctx,
            &dispatch,
            in_ptr,
            wd_ptr,
            sync.unsafe_ref(),
            &params.function,
            &params.filter,
        );

        {
            profiling::scope!("mt:job group");
            for i in 0..dispatch.first_parallel {
                profiling::scope!("mt:job");
                (params.function)(
                    ctx,
                    &mut params.items[i as usize],
                    &mut *wd_ptr.offset(ctx.id() as isize),
                );
            }
        }

        //println!("end main thread jobs");

        ctx.wait(&sync);
    }
}
*/

fn for_each_mut<Item, W, F, Filtr>(mut params: ForEachMut<Item, W, F, Filtr>)
where
    F: Fn(&mut Context, &mut Item, &mut W) + Sync + Send,
    Filtr: Filter<Item>,
    Item: Sync + Send,
{
    profiling::scope!("for_each_mut");

    unsafe {
        let (_, cd_ptr) = params.unsafe_pointers();
        let dispatch = params.dispatch_parameters(false);

        let sync = SyncPoint::new(dispatch.group_count);

        let ctx = params.ctx;

        let mut jobs: SmallVec<[MutSliceJob<Item, W, F, Filtr>; 64]> = SmallVec::with_capacity(dispatch.group_count as usize);

        let mut actual_group_count = dispatch.group_count;
        for group_idx in 0..dispatch.group_count {
            let mut start = dispatch.first_parallel + dispatch.group_size * group_idx;
            let mut end = (start + dispatch.group_size).min(dispatch.item_count);
            debug_assert!(start < end);

            if !params.filter.is_empty() {
                while start < end && !params.filter.filter(&params.items[end as usize]) {
                    start += 1;
                }

                while start < end && !params.filter.filter(&params.items[start as usize - 1]) {
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

            jobs.push(MutSliceJob {
                items: (&mut params.items[start as usize .. end as usize]) as *mut _,
                ctx_data: cd_ptr.0,
                run_fn: (&params.function) as *const _,
                filter: (&params.filter) as *const _,
                sync: sync.unsafe_ref(),
            });
        }

        // Once we start converting the jobs into jobrefs, they MUST NOT move or be destroyed until
        // we are done waiting on the sync point.
        for job in &jobs {
            ctx.enqueue_job(job.as_job_ref());
        }

        // Waking up worker threads is the expensive part.
        ctx.wake_n(actual_group_count.min(ctx.num_worker_threads()));

        {
            profiling::scope!("mt:job group");
            for i in 0..dispatch.first_parallel {
                profiling::scope!("mt:job");
                (params.function)(
                    ctx,
                    &mut params.items[i as usize],
                    &mut *cd_ptr.offset(ctx.id() as isize),
                );
            }
        }

        ctx.wait(&sync);
    }
}


fn for_each_mut_async<'a, 'b, Item, W, F, Filtr>(mut params: ForEachMut<Item, W, F, Filtr>) -> JoinHandle<'a, 'b>
where
    F: Fn(&mut Context, &mut Item, &mut W) + Sync + Send,
    Filtr: Filter<Item>,
    Item: Sync + Send,
{
    profiling::scope!("for_each_mut_async");

    unsafe {
        let (in_ptr, wd_ptr) = params.unsafe_pointers();
        let dispatch = params.dispatch_parameters(true);

        let sync = Box::new(SyncPoint::new(dispatch.group_count));

        for_each_mut_dispatch(
            params.ctx,
            &dispatch,
            in_ptr,
            wd_ptr,
            sync.unsafe_ref(),
            &params.function,
            &params.filter,
        );

        JoinHandle { sync, _marker: PhantomData }
    }
}

/// Dispatch the work that will be done in parallel.
unsafe fn for_each_mut_dispatch<Item, Output, W, F, Filtr>(
    ctx: &mut Context,
    dispatch: &DispatchParameters,
    in_ptr: SyncPtrMut<Item>,
    wd_ptr: SyncPtrMut<W>,
    sync: SyncPointRef,
    cb: &F,
    filter: &Filtr,
)
where
    F: Fn(&mut Context, &mut Item, &mut W) -> Output + Sync + Send,
    Filtr: Filter<Item>,
    Item: Sync + Send,
    Output: Sized,
{
    //println!("------- {:?}", 0..first_parallel);

    if dispatch.group_count == 0 {
        return;
    }

    // Make sure the closure captures the values directly instead of taking the address
    // of the parameters struct.
    let DispatchParameters {
        item_count,
        group_count,
        group_size,
        first_parallel,
    } = *dispatch;

    assert!(group_count * group_size >= (item_count - first_parallel));
    let mut actual_group_count = group_count;

    profiling::scope!("schedule jobs");
    for chunk_idx in 0..group_count {
        let mut start = first_parallel + group_size * chunk_idx;
        let mut end = (start + group_size).min(item_count);

        // Apply the filter (if any) to reduce the size of the item group.
        if !filter.is_empty() {
            while start < end && !filter.filter(in_ptr.offset(start as isize)) {
                start += 1;
            }

            while start < end && !filter.filter(in_ptr.offset(end as isize - 1)) {
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

        ctx.enqueue_heap_job(move |ctx| {
            profiling::scope!("job group");

            assert!(end > start);
            let wroker_idx = ctx.id() as isize;
            let context_data = wd_ptr.offset(wroker_idx);

            //println!(" -- (w{:?}) chunk {:?}", ctx.id(), start..end);
            for job_idx in start..end {
                profiling::scope!("worker:job");
                let job_idx = job_idx as isize;
                //println!("  -- (w{:?}) job {:?}", ctx.id(), job_idx);

                let item = in_ptr.offset(job_idx);
                if filter.filter(item) {
                    cb(ctx, item, context_data);
                }
            }
            sync.signal(ctx);
        });
    }

    // Waking threads up is rather expensive we populate the queue first
    ctx.wake_n(actual_group_count.min(ctx.num_worker_threads()));
}

#[derive(Debug)]
struct DispatchParameters {
    item_count: u32,
    group_count: u32,
    group_size: u32,
    first_parallel: u32,
}

pub struct JoinHandle<'a, 'b> {
    sync: Box<SyncPoint>,
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
