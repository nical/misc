use std::{marker::PhantomData, ptr::NonNull, sync::Arc};

pub struct Workers {
    thread_pool: Arc<rayon::ThreadPool>,
}

impl Workers {
    pub fn new(num_workers: usize) -> Self {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_workers)
            .thread_name(|idx| format!("Worker#{idx}"))
            .use_current_thread()
            .build()
            .unwrap();

        Workers {
            thread_pool: Arc::new(pool),
        }
    }

    pub fn ctx<'a>(&'a self) -> Context<'a, ()> {
        Context {
            ctx_data: NonNull::dangling(),
            _marker: PhantomData,
        }
    }

    pub fn ctx_with<'a, CtxData>(&'a self, data: &'a mut[CtxData]) -> Context<'a, CtxData>
    where CtxData: Send
    {
        assert!(data.len() >= rayon::current_num_threads() + 1);
        Context {
            ctx_data: NonNull::new(data.as_mut_ptr()).unwrap(),
            _marker: PhantomData,
        }
    }

    pub fn broadcast<OP, R>(&self, op: OP) -> Vec<R>
    where
        OP: Fn(&mut Context<()>) -> R + Sync,
        R: Send,
    {
        self.thread_pool.broadcast(
            |_| {
                let anchor = ();
                let ctx_data = CtxDataPtr { ptr: NonNull::dangling() };
                op(&mut Context::worker(ctx_data, &anchor))
            }
        )
    }
}

/// A temporary wrapper for the pointer whihc implements send.
#[derive(Clone)]
struct CtxDataPtr<T> {
    ptr: NonNull<T>,
}

unsafe impl<T: Send> Send for CtxDataPtr<T> {}

pub struct Context<'a, CtxData> {
    /// # Safety
    ///
    /// NonNull *must* be either:
    ///  - Pointing to a contiguous array that is at least N+1 large where N
    ///    is the number of workers.
    ///  - A dangling pointer if and only if CtxData is the unit type `()`.
    ///
    /// If ctx_data is a dangling unit pointer, accesses (at an offset) won't
    /// produce any actual reads or writes, so it is safe.
    ctx_data: NonNull<CtxData>,
    _marker: PhantomData<&'a ()>,
}

impl<'a, CtxData> Context<'a, CtxData> {
    fn worker(ctx_data: CtxDataPtr<CtxData>, _anchor: &'a ()) -> Self {
        Context {
            ctx_data: ctx_data.ptr,
            _marker: PhantomData,
        }
    }

    pub fn index(&self) -> usize {
        rayon::current_thread_index().unwrap()
    }

    pub fn data(&mut self) -> &mut CtxData {
        let idx = rayon::current_thread_index().unwrap();
        unsafe { self.ctx_data.offset(idx as isize).as_mut() }
    }

    pub fn join<A, B, RA, RB>(&self, oper_a: A, oper_b: B) -> (RA, RB)
    where
        A: FnOnce(&mut Context<CtxData>) -> RA + Send,
        B: FnOnce(&mut Context<CtxData>) -> RB + Send,
        RA: Send,
        RB: Send,
        CtxData: Send
    {
        let ctx_data_a = CtxDataPtr { ptr: self.ctx_data };
        let ctx_data_b = CtxDataPtr { ptr: self.ctx_data };
        rayon::join(
            move || {
                let anchor = ();
                oper_a(&mut Context::worker(ctx_data_a, &anchor))
            },
            move || {
                let anchor = ();
                oper_b(&mut Context::worker(ctx_data_b, &anchor))
            },
        )
    }

    pub fn with_data<'b, Data>(&mut self, data: &'b mut[Data]) -> Context<'b, Data> {
        assert!(data.len() >= rayon::current_num_threads() + 1);
        Context {
            ctx_data: NonNull::new(data.as_mut_ptr()).unwrap(),
            _marker: PhantomData,
        }
    }
}
