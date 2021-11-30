use crate::{Context};
use crate::util::SyncPtrMut;
use crate::sync::{SyncPoint, SyncPointRef};

use std::marker::PhantomData;

pub struct ForEachMut<'a, 'b, 'c, Item, ContextData, F> {
    pub(crate) items: &'a mut [Item],
    pub(crate) context_data: Option<&'b mut [ContextData]>,
    pub(crate) function: F,
    pub(crate) group_size: usize,
    pub(crate) ctx: &'c mut Context,
}

impl<'a, 'b, 'c, Item, ContextData, F> ForEachMut<'a, 'b, 'c, Item, ContextData, F> 
{
    #[inline]
    pub fn with_context_data<'w, W: Send>(self, context_data: &'w mut [W]) -> ForEachMut<'a, 'w, 'c, Item, W, F> {
        assert!(
            context_data.len() >= self.ctx.num_contexts as usize,
            "Got {:?} texture items, need at least {:?}",
            context_data.len(), self.ctx.num_contexts,
        );

        ForEachMut {
            items: self.items,
            context_data: Some(context_data),
            function: self.function,
            group_size: self.group_size,
            ctx: self.ctx
        }
    }

    #[inline]
    pub fn with_group_size(mut self, group_size: usize) -> Self {
        self.group_size = group_size;

        self
    }

    /// Run this workload with the help of worker threads.
    ///
    /// This function returns after the workload has completed.
    #[inline]
    pub fn run<Func>(self, function: Func)
    where
        Func: Fn(&mut Context, &mut Item, &mut ContextData) + Sync + Send,
        Item: Sync + Send,
    {
        for_each_mut(self.apply(function));
    }

    /// Run this workload asynchronously on the worker threads
    ///
    /// Returns an object to wait on.
    #[inline]
    pub fn run_async<Func>(self, function: Func) -> JoinHandle<'a, 'b>
    where
        Func: Fn(&mut Context, &mut Item, &mut ContextData) + Sync + Send,
        Item: Sync + Send,
    {
        for_each_mut_async(self.apply(function))
    }

    #[inline]
    fn apply<Func>(self, function: Func) -> ForEachMut<'a, 'b, 'c, Item, ContextData, Func>
    where
        Func: Fn(&mut Context, &mut Item, &mut ContextData) + Sync + Send
    {
        ForEachMut {
            items: self.items,
            context_data: self.context_data,
            function,
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

        let num_parallel = if is_async { n } else { (n * 2) / 3 };
        let first_parallel = n - num_parallel;
        let group_size = self.group_size.min(self.items.len()) as u32;
        let group_count = div_ceil(num_parallel, group_size);

        DispatchParameters {
            item_count: n,
            group_count,
            group_size,
            first_parallel,
        }
    }
}

fn for_each_mut<Input, W, F>(mut params: ForEachMut<Input, W, F>)
where
    F: Fn(&mut Context, &mut Input, &mut W) + Sync + Send,
    Input: Sync + Send,
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

fn for_each_mut_async<'a, 'b, Input, W, F>(mut params: ForEachMut<Input, W, F>) -> JoinHandle<'a, 'b>
where
    F: Fn(&mut Context, &mut Input, &mut W) + Sync + Send,
    Input: Sync + Send,
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
        );

        JoinHandle { sync, _marker: PhantomData }
    }
}

/// Dispatch the work that will be done in parallel.
unsafe fn for_each_mut_dispatch<Input, Output, W, F>(
    ctx: &mut Context,
    dispatch: &DispatchParameters,
    in_ptr: SyncPtrMut<Input>,
    wd_ptr: SyncPtrMut<W>,
    sync: SyncPointRef,
    cb: &F)
where
    F: Fn(&mut Context, &mut Input, &mut W) -> Output + Sync + Send,
    Input: Sync + Send,
    Output: Sized,
{
    //println!("------- {:?}", 0..first_parallel);

    // Make sure the closure captures the values directly instead of taking the address
    // of the parameters struct.
    let DispatchParameters {
        item_count,
        group_count,
        group_size,
        first_parallel,
    } = *dispatch;

    assert!(group_count * group_size >= (item_count - first_parallel));

    // There's currently quite a bit of overhead from pushing the job groups one by one
    // on the submitter thread so we first submit a single job that will submit the
    // other ones while the initial thread starts to work.
    ctx.dispatch_one(move |ctx| {
        profiling::scope!("schedule jobs");
        for chunk_idx in 0..group_count {
            let start = first_parallel + group_size * chunk_idx;
            ctx.dispatch_one(move |ctx| {
                profiling::scope!("job group");

                let end = (start + group_size).min(item_count);
                assert!(end > start);
                let wroker_idx = ctx.id() as isize;
                let context_data = wd_ptr.offset(wroker_idx);

                //println!(" -- (w{:?}) chunk {:?}", ctx.id(), start..end);
                for job_idx in start..end {
                    profiling::scope!("worker:job");
                    let job_idx = job_idx as isize;
                    //println!("  -- (w{:?}) job {:?}", ctx.id(), job_idx);

                    cb(
                        ctx,
                        in_ptr.offset(job_idx),
                        context_data,
                    );
                }
                sync.signal(ctx);
            });
        }
    });
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
