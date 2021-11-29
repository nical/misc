use crate::{Context};
use crate::util::{SyncPtr, SyncPtrMut};
use crate::sync::{SyncPoint, Event};
use crate::job::{HeapJob, StackJob};


pub struct ForEachMut<'a, 'b, 'c, Item, ContextData, F> {
    pub(crate) items: &'a mut [Item],
    pub(crate) worker_data: Option<&'b mut [ContextData]>,
    pub(crate) function: F,
    pub(crate) group_size: usize,
    pub(crate) ctx: &'c mut Context,
}

impl<'a, 'b, 'c, Item, ContextData, F> ForEachMut<'a, 'b, 'c, Item, ContextData, F> 
{
    #[inline]
    pub fn with_worker_data<'w, W: Send>(self, worker_data: &'w mut [W]) -> ForEachMut<'a, 'w, 'c, Item, W, F> {
        ForEachMut {
            items: self.items,
            worker_data: Some(worker_data),
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

    #[inline]
    fn apply<Func>(self, function: Func) -> ForEachMut<'a, 'b, 'c, Item, ContextData, Func>
    where
        Func: Fn(&mut Context, &mut Item, &mut ContextData) + Sync + Send
    {
        ForEachMut {
            items: self.items,
            worker_data: self.worker_data,
            function,
            group_size: self.group_size,
            ctx: self.ctx,
        }
    }

    #[inline]
    pub fn run<Func>(self, function: Func)
    where
        Func: Fn(&mut Context, &mut Item, &mut ContextData) + Sync + Send,
        Item: Sync + Send,
    {
        for_each_mut(self.apply(function));
    }
}

fn for_each_mut<Input, W, F>(mut params: ForEachMut<Input, W, F>)
where
    F: Fn(&mut Context, &mut Input, &mut W) + Sync + Send,
    Input: Sync + Send,
{
    profiling::scope!("for_each_mut");

    let mut ctx = params.ctx;

    if let Some(wd) = &params.worker_data {
        assert!(
            wd.len() >= ctx.worker_count as usize,
            "Got {:?} worker data, need at least {:?}",
            wd.len(), ctx.worker_count,
        );
    }

    unsafe {
        let n = params.items.len();
        let num_parallel = (n * 4) / 5;
        let first_parallel = n - num_parallel;

        let group_size = params.group_size.min(params.items.len());
        let num_chunks = div_ceil(num_parallel, group_size);

        let in_ptr = params.items.as_mut_ptr();
        let wd_ptr = params.worker_data.as_mut().map_or(std::ptr::null_mut(), |arr| arr.as_mut_ptr());
        let sync = SyncPoint::new(num_chunks as u32 + 1);
        let event = Event::new();

        //println!(" -- {:?} jobs, {:?} parallel {:?} chunks, chunksize {:?} ", n, num_parallel, num_chunks, group_size);

        // It is important hat the StackJob exist on the stack until we wait.
        let event_job = StackJob::new(|_| event.set());

        sync.then(ctx, event_job.as_job_ref());

        //panic!();

        //println!("jobs: {:?}, num chunks: {:?} ", n, num_chunks);

        for_each_mut_impl(
            ctx,
            params.items.len(),
            in_ptr,
            wd_ptr,
            &sync,
            first_parallel,
            num_chunks,
            group_size,
            &params.function,
        );

        {
            profiling::scope!("mt:job group");
            //println!("start main thread {:?}", 0..num_parallel);
            for i in 0..first_parallel {
                ctx.job_idx = Some(i as u32);
                profiling::scope!("mt:job");
                (params.function)(
                    ctx,
                    &mut params.items[i],
                    &mut *wd_ptr.offset(ctx.id() as isize),
                );
            }
            ctx.job_idx = None;
        }

        //println!("end main thread jobs");

        sync.signal(ctx);

        {
            profiling::scope!("steal jobs");
            while !event.peek() {
                if let Ok(job) = ctx.rx.try_recv() {
                    job.execute(ctx);
                } else {
                    break
                }
            }
        }

        event.wait();
    }
}

unsafe fn for_each_mut_impl<Input, Output, W, F>(
    ctx: &mut Context,
    count: usize,
    input: *mut Input,
    worker_data: *mut W,
    sync: *const SyncPoint,
    first_parallel: usize,
    num_chunks: usize,
    group_size: usize,
    cb: &F)
where
    F: Fn(&mut Context, &mut Input, &mut W) -> Output + Sync + Send,
    Input: Sync + Send,
    Output: Sized,
{
    //println!("------- {:?}", 0..first_parallel);

    // I'm not entirely sure why I have to do this.
    // If I don't wrap the unsafe pointers, the compiler tries to capture
    // the &*T instead of *T.
    let sync_ptr = SyncPtr(sync);
    let in_ptr = SyncPtrMut(input);
    let wd_ptr = SyncPtrMut(worker_data);

    assert!(num_chunks * group_size >= (count - first_parallel), "num_chunks: {} * group_size: {} >= count: {}", num_chunks, group_size, count - first_parallel);

    let fork_job = HeapJob::new_ref(move |worker| {
        profiling::scope!("schedule jobs");
        for chunk_idx in 0..num_chunks {
            let start = first_parallel + group_size * chunk_idx;
            let job = HeapJob::new_ref(
                move |worker| {
                    profiling::scope!("job group");

                    let end = (start + group_size).min(count);
                    assert!(end > start);
                    let wroker_idx = worker.id() as isize;
                    let worker_data = wd_ptr.offset(wroker_idx);

                    //println!(" -- (w{:?}) chunk {:?}", worker.id(), start..end);
                    for job_idx in start..end {
                        profiling::scope!("worker:job");
                        worker.job_idx = Some(job_idx as u32);
                        let job_idx = job_idx as isize;
                        //println!("  -- (w{:?}) job {:?}", worker.id(), job_idx);
                        
                        cb(
                            worker,
                            in_ptr.offset(job_idx),
                            worker_data,
                        );
                    }
                    worker.job_idx = None;
                    sync_ptr.get().signal(worker);
                }
            );
            worker.schedule_job(job);
        }
    });
    ctx.schedule_job(fork_job);
}

fn div_ceil(a: usize, b: usize) -> usize {
    let d = a / b;
    let r = a % b;
    if r > 0 && b > 0 { d + 1 } else { d }
}
