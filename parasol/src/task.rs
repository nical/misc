use crate::core::event::{Event};
use crate::core::job::{Job, Priority, JobRef};
use crate::Context;
use crate::helpers::*;
use crate::handle::*;
use crate::sync::Arc;

use std::mem;
use std::ops::{Range};

pub struct Args<'l, Input, ContextData, ImmutableData> {
    pub input: Input,
    pub context_data: &'l mut ContextData,
    pub immutable_data: &'l ImmutableData,
}

pub struct Task<'l, ContextData, ImmutableData, Dependency, Function> {
    parameters: OwnedParameters<ContextData, ImmutableData>,
    input: Dependency,
    function: Function,
    ctx: &'l mut Context,
}

#[inline]
pub fn new_task(ctx: &mut Context) -> Task<(), (), (), ()> {
    Task { parameters: owned_parameters(), input: (), function: (), ctx, }
}

impl<Output, ContextData, ImmutableData, Dependency, Function> Job for InlineRefCounted<TaskJobData<Output, ContextData, ImmutableData, Dependency, Function>>
where
    Dependency: TaskDependency,
    Function: Fn(&mut Context, Args<Dependency::Output, ContextData, ImmutableData>) -> Output + Send,
{
    unsafe fn execute(this: *const Self, ctx: &mut Context, range: Range<u32>) {
        TaskJobData::execute((*this).inner() as *const _, ctx, range);
        (*this).release_ref();
    }
}

impl<'l, ContextData, ImmutableData, Dependency, Function> Task<'l, ContextData, ImmutableData, Dependency, Function> {
    #[inline]
    pub fn with_context_data<T>(self, data: HeapContextData<T>) -> Task<'l, T, ImmutableData, Dependency, Function> {
        Task {
            parameters: self.parameters.with_context_data(data),
            function: self.function,
            input: self.input,
            ctx: self.ctx,
        }
    }

    #[inline]
    pub fn with_immutable_data<T>(self, data: Arc<T>) -> Task<'l, ContextData, T, Dependency, Function> {
        Task {
            parameters: self.parameters.with_immutable_data(data),
            function: self.function,
            input: self.input,
            ctx: self.ctx,
        }
    }

    #[inline]
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.parameters = self.parameters.with_priority(priority);

        self
    }

    #[inline]
    pub fn with_input<Dep: TaskDependency>(self, input: Dep) -> Task<'l, ContextData, ImmutableData, Dep, Function> {
        Task {
            parameters: self.parameters,
            function: self.function,
            input,
            ctx: self.ctx,
        }
    }

    #[inline]
    pub fn with_input_data<T>(self, data: T) -> Task<'l, ContextData, ImmutableData, DataSlot<T>, Function> {
        let input = DataSlot::new();
        unsafe {
            input.set(data);
        }
        Task {
            parameters: self.parameters,
            function: self.function,
            input,
            ctx: self.ctx,
        }
    }

    #[inline]
    pub fn after(self, handle: Handle) -> Task<'l, ContextData, ImmutableData, Handle, Function> {
        Task {
            parameters: self.parameters,
            function: self.function,
            input: handle,
            ctx: self.ctx,
        }
    }

    #[inline]
    fn with_function<F, Output>(self, function: F) -> Task<'l, ContextData, ImmutableData, Dependency, F>
    where
        Dependency: TaskDependency,
        F: Fn(&mut Context, Args<Dependency::Output, ContextData, ImmutableData>) -> Output + Send,
        Output: Send,
    {
        Task {
            parameters: self.parameters,
            function,
            input: self.input,
            ctx: self.ctx,
        }
    }

    #[inline]
    pub fn run<F, Output>(self, function: F) -> OwnedHandle<Output>
    where
        Dependency: TaskDependency + 'static,
        F: Fn(&mut Context, Args<Dependency::Output, ContextData, ImmutableData>) -> Output + Send + 'static,
        Output: Send + 'static,
        ContextData: 'static,
        ImmutableData: 'static,
    {
        let mut task = self.with_function(function);
        let priority = task.parameters.priority();
        unsafe {
            let ctx = task.ctx;

            let task_job: RefPtr<TaskJobData<Output, ContextData, ImmutableData, Dependency, F>> = RefPtr::new(
                TaskJobData {
                    function: task.function,
                    input: task.input,
                    data: ConcurrentDataRef::from_owned(&mut task.parameters, ctx),
                    parameters: task.parameters,
                    output: DataSlot::new(),
                    event: Event::new(1, ctx.thread_pool_id()),
                }
            );

            // Add a self-reference that will be removed after executing the job.
            task_job.add_ref();

            let event: *const Event = &task_job.event;
            let output: *mut DataSlot<Output> = mem::transmute(&task_job.output);

            let job_ref = JobRef::new(task_job.inner()).with_priority(priority);

            if let Some(evt) = task_job.input.get_event() {
                evt.then(ctx, job_ref);
            } else {
                ctx.schedule_job(job_ref);
            }

            OwnedHandle::new(
                task_job.into_any(),
                event,
                output,
            )
        }
    }

}

struct TaskJobData<Output, ContextData, ImmutableData, Dependency, F> {
    function: F,
    input: Dependency,
    #[allow(dead_code)] // Not really dead code, needed to keep the strong references alive.
    parameters: OwnedParameters<ContextData, ImmutableData>,
    data: ConcurrentDataRef<ContextData, ImmutableData>,
    output: DataSlot<Output>,
    event: Event,
}

impl<Output, ContextData, ImmutableData, Dependency, Func> Job for TaskJobData<Output, ContextData, ImmutableData, Dependency, Func>
where
    Dependency: TaskDependency,
    Func: Fn(&mut Context, Args<Dependency::Output, ContextData, ImmutableData>) -> Output + Send,
{
    unsafe fn execute(this: *const Self, ctx: &mut Context, _range: Range<u32>) {
        let (context_data, immutable_data) = (*this).data.get(ctx);
        (*this).output.set(((*this).function)(ctx, Args {
            input: (*this).input.get_output(),
            context_data,
            immutable_data,
        }));
        (*this).event.signal(ctx, 1);
    }
}

#[test]
fn simple_task() {
    use crate::sync::{AtomicI32, Ordering};
    use crate::ThreadPool;
    let pool = ThreadPool::builder()
        .with_worker_threads(3)
        .with_contexts(1)
        .build();

    let mut ctx = pool.pop_context().unwrap();

    let mut handles: Vec<OwnedHandle<u32>> = Vec::new();
    for _ in 0..100_000 {
        handles.push(
            ctx.task().run(|_ctx, _args| { 1u32 + 1 })
        );
    }

    for handle in handles {
        assert_eq!(handle.resolve(&mut ctx), 2);
    }

    // Task t1 produces an output which becomes which is the input of task t2.
    let input: f32 = 1.0;
    let t1 = ctx.task().with_input_data(input).run(|_, args| { args.input + 1.0 });
    let t2 = ctx.task().with_input(t1).run(|_, args| {
        args.input as u32 + 1
    });
    assert_eq!(t2.resolve(&mut ctx), 3);

    for _ in 0..100_000 {
        ctx.task().run(|_ctx, _args| { 1u32 + 1 });
    }

    let counter = Arc::new(AtomicI32::new(0));
    let c3 = counter.clone();
    let t3 = ctx.task().run(move|_, _| { c3.fetch_add(1, Ordering::Release)});
    let t3_done = t3.handle();

    let c = counter.clone();
    ctx.task().after(t3_done.clone()).run(move |_,_| { assert_eq!(c.load(Ordering::Acquire), 1); });
    let c = counter.clone();
    ctx.task().after(t3_done.clone()).run(move |_,_| { assert_eq!(c.load(Ordering::Acquire), 1); });
    let c = counter.clone();
    ctx.task().after(t3_done.clone()).run(move |_,_| { assert_eq!(c.load(Ordering::Acquire), 1); });

    pool.shut_down().wait();
}
