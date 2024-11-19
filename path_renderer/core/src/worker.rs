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
            slice_split_threshold: 2,
            _marker: PhantomData,
        }
    }

    pub fn ctx_with<'a, CtxData>(&'a self, data: &'a mut[CtxData]) -> Context<'a, CtxData>
    where CtxData: Send
    {
        assert!(data.len() >= rayon::current_num_threads());
        Context {
            ctx_data: NonNull::new(data.as_mut_ptr()).unwrap(),
            slice_split_threshold: 2,
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
                op(&mut Context::worker(ctx_data, 2, &anchor))
            }
        )
    }

    pub fn num_workers(&self) -> usize {
        self.thread_pool.current_num_threads()
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
    slice_split_threshold: usize,
    _marker: PhantomData<&'a ()>,
}

impl<'a, CtxData: Send> Context<'a, CtxData> {
    fn worker(ctx_data: CtxDataPtr<CtxData>, split: usize, _anchor: &'a ()) -> Self {
        Context {
            ctx_data: ctx_data.ptr,
            slice_split_threshold: split,
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
        let split = self.slice_split_threshold;
        let ctx_data_a = CtxDataPtr { ptr: self.ctx_data };
        let ctx_data_b = CtxDataPtr { ptr: self.ctx_data };
        rayon::join(
            move || {
                let anchor = ();
                oper_a(&mut Context::worker(ctx_data_a, split, &anchor))
            },
            move || {
                let anchor = ();
                oper_b(&mut Context::worker(ctx_data_b, split, &anchor))
            },
        )
    }

    pub fn for_each<I: Sync, F>(
        &mut self,
        mut iter: impl Iterator<Item = I>,
        op: &F,
    )
    where
        F: Fn(&mut Context<CtxData>, &I) + Send + Sync,
    {
        const BUFFER_SIZE: usize = 1024 * 4; // 4Kb
        // Process items by batch of n items.
        // This is not great because we wait for each batch to be done before
        // starting the next.
        let n = (BUFFER_SIZE / std::mem::size_of::<I>()).max(1);
        let mut buffer = Vec::with_capacity(n);

        'outer: loop {
            for _ in 0..n {
                if let Some(item) = iter.next() {
                    buffer.push(item);
                } else {
                    break 'outer;
                }
            }

            self.slice_for_each(&buffer, op);

            buffer.clear();
        }

        if !buffer.is_empty() {
            self.slice_for_each(&buffer, op);
        }
    }

    pub fn slice_for_each<I, F>(
        &mut self,
        slice: &[I],
        op: &F
    )
    where
        I: Sync,
        F: Fn(&mut Context<CtxData>, &I) + Send + Sync
    {
        self.slice_split_threshold = (slice.len() / rayon::current_num_threads()).max(2);
        self.slice_for_each_impl(slice, op);
    }

    #[inline]
    fn slice_for_each_impl<I, F>(
        &mut self,
        slice: &[I],
        op: &F
    )
    where
        I: Sync,
        F: Fn(&mut Context<CtxData>, &I) + Send + Sync
    {
        //println!("ctx #{} for each n={}", self.index(), slice.len());
        if slice.len() <= self.slice_split_threshold {
            //println!("ctx #{} job exec n={}", self.index(), slice.len());
            for item in slice {
                op(self, item)
            }
            return;
        }
        let split = slice.len() / 2;
        let left = &slice[..split];
        let right = &slice[split..];
        self.join(
            move |ctx| ctx.slice_for_each_impl(left, op),
            move |ctx| ctx.slice_for_each_impl(right, op),
        );
    }

    pub fn with_data<'b, Data>(&mut self, data: &'b mut[Data]) -> Context<'b, Data> {
        assert!(data.len() >= rayon::current_num_threads() + 1);
        Context {
            ctx_data: NonNull::new(data.as_mut_ptr()).unwrap(),
            slice_split_threshold: self.slice_split_threshold,
            _marker: PhantomData,
        }
    }
}

#[test]
fn par_iter_simple() {
    use std::sync::atomic::{Ordering, AtomicU32};
    let workers = Workers::new(8);

    let mut input = Vec::new();
    let mut expected_result = 0;

    let mut ctx_data = vec![0; 9];

    for i in 0..4096u32 {
        input.push(i);
        expected_result += i;
    }

    let result = AtomicU32::new(0);

    workers.ctx_with(&mut ctx_data).for_each(
        input.into_iter(),
        &|ctx, idx| {
            *ctx.data() += idx;
            result.fetch_add(*idx, Ordering::Relaxed);
        }
    );

    let ctx_data_sum: u32 = ctx_data.into_iter().sum();
    assert_eq!(result.load(Ordering::SeqCst), expected_result);
    assert_eq!(ctx_data_sum, expected_result);
}
