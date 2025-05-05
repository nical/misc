#![allow(private_bounds)]

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
            ctx_data: (),
            slice_split_threshold: 2,
            _marker: PhantomData,
        }
    }

    pub fn ctx_with<'a, CtxData>(&'a self, data: &'a mut[CtxData]) -> Context<'a, (CtxData,)>
    where CtxData: Send
    {
        assert!(data.len() >= rayon::current_num_threads());
        Context {
            ctx_data: SendPtr { ptr: NonNull::new(data.as_mut_ptr()).unwrap() },
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
                op(&mut Context::worker((), 2, &anchor))
            }
        )
    }

    pub fn num_workers(&self) -> usize {
        self.thread_pool.current_num_threads()
    }
}

/// A temporary wrapper for the pointer which implements send.
pub(crate) struct SendPtr<T> {
    ptr: NonNull<T>,
}

impl<T> SendPtr<T> {
    pub fn from_ptr(p: *mut T) -> Self {
        Self { ptr: NonNull::new(p).unwrap() }
    }

    pub fn ptr(&self) -> *mut T { self.ptr.as_ptr() }

    #[allow(unused)]
    pub unsafe fn as_mut(&mut self) -> &mut T { unsafe { self.ptr.as_mut() } }
}

unsafe impl<T: Send> Send for SendPtr<T> {}

impl<T> Clone for SendPtr<T> {
    fn clone(&self) -> Self {
        SendPtr { ptr: self.ptr }
    }
}

/// A temporary wrapper for the pointer which implements send.
pub(crate) struct SendConstPtr<T> {
    ptr: *const T,
}

impl<T> SendConstPtr<T> {
    pub fn new(ptr: *const T) -> Self {
        Self { ptr }
    }
    pub fn ptr(&self) -> *const T { self.ptr }
}

unsafe impl<T: Send> Send for SendConstPtr<T> {}

impl<T> Clone for SendConstPtr<T> {
    fn clone(&self) -> Self {
        SendConstPtr { ptr: self.ptr }
    }
}

pub struct Context<'a, CtxData: WorkerData> {
    /// # Safety
    ///
    /// NonNull *must* be either:
    ///  - Pointing to a contiguous array that is at least N+1 large where N
    ///    is the number of workers.
    ///  - A dangling pointer if and only if CtxData is the unit type `()`.
    ///
    /// If ctx_data is a dangling unit pointer, accesses (at an offset) won't
    /// produce any actual reads or writes, so it is safe.
    ctx_data: CtxData::Ptr,
    slice_split_threshold: usize,
    _marker: PhantomData<&'a ()>,
}

impl<'a, CtxData: WorkerData> Context<'a, CtxData> {
    fn worker(ctx_data: CtxData::Ptr, split: usize, _anchor: &'a ()) -> Self {
        Context {
            ctx_data,
            slice_split_threshold: split,
            _marker: PhantomData,
        }
    }

    pub fn index(&self) -> usize {
        rayon::current_thread_index().unwrap()
    }

    pub fn num_workers(&self) -> usize {
        rayon::current_num_threads()
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
        let ctx_data_a = self.ctx_data.clone();
        let ctx_data_b = self.ctx_data.clone();
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

    pub fn for_each<I: Send, F>(
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

            self.slice_for_each(&buffer, &|ctx, slice, _| {
                for item in slice {
                    op(ctx, item)
                }
            });

            buffer.clear();
        }

        if !buffer.is_empty() {
            self.slice_for_each(&buffer, &|ctx, slice, _| {
                for item in slice {
                    op(ctx, item)
                }
            });
        }
    }

    pub fn for_each_mut<'b, I: Sync, F>(
        &mut self,
        mut iter: impl Iterator<Item = &'b mut I>,
        op: &'b F,
    )
    where
        I: Send + 'b,
        F: Fn(&mut Context<CtxData>, &mut I) + Send + Sync,
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

            self.slice_for_each_mut(&mut buffer, &|ctx, slice| {
                for item in slice {
                    op(ctx, item)
                }
            });

            buffer.clear();
        }

        if !buffer.is_empty() {
            self.slice_for_each_mut(&mut buffer, &|ctx, slice| {
                for item in slice {
                    op(ctx, item)
                }
            });
        }
    }

    pub fn slice_for_each<I, F>(
        &mut self,
        slice: &[I],
        op: &F
    )
    where
        I: Send,
        F: Fn(&mut Context<CtxData>, &[I], u32) + Send + Sync
    {
        self.slice_split_threshold = (slice.len() / rayon::current_num_threads()).max(2);
        self.slice_for_each_impl(0, slice, op);
    }

    #[inline]
    fn slice_for_each_impl<I, F>(
        &mut self,
        index: u32,
        slice: &[I],
        op: &F
    )
    where
        I: Send,
        F: Fn(&mut Context<CtxData>, &[I], u32) + Send + Sync
    {
        //println!("ctx #{} for each n={}", self.index(), slice.len());
        if slice.len() <= self.slice_split_threshold {
            //println!("ctx #{} job exec n={}", self.index(), slice.len());
            op(self, slice, index);
            return;
        }

        unsafe {
            let base_ptr = slice.as_ptr();
            let left = SendConstPtr::new(base_ptr);
            let left_len = slice.len() / 2;
            let right = SendConstPtr::new(base_ptr.add(left_len));
            let right_len = slice.len() - left_len;
            self.join(
                move |ctx| {
                    let slice = std::slice::from_raw_parts(left.ptr(), left_len);
                    ctx.slice_for_each_impl(index, slice, op)
                },
                move |ctx| {
                    let right_idx = index + left_len as u32;
                    let slice = std::slice::from_raw_parts(right.ptr(), right_len);
                    ctx.slice_for_each_impl(right_idx, slice, op)
                },
            );
        }
    }

    pub fn slice_for_each_mut<I, F>(
        &mut self,
        slice: &mut [I],
        op: &F
    )
    where
        I: Send,
        F: Fn(&mut Context<CtxData>, &mut [I]) + Send + Sync
    {
        self.slice_split_threshold = (slice.len() / rayon::current_num_threads()).max(2);
        self.slice_for_each_mut_impl(slice, op);
    }

    #[inline]
    fn slice_for_each_mut_impl<I, F>(
        &mut self,
        slice: &mut [I],
        op: &F
    )
    where
        I: Send,
        F: Fn(&mut Context<CtxData>, &mut [I]) + Send + Sync
    {
        //println!("ctx #{} for each n={}", self.index(), slice.len());
        if slice.len() <= self.slice_split_threshold {
            //println!("ctx #{} job exec n={}", self.index(), slice.len());
            op(self, slice);
            return;
        }
        unsafe {
            let base_ptr = slice.as_mut_ptr();
            let left = SendPtr::from_ptr(base_ptr);
            let left_len = slice.len() / 2;
            let right = SendPtr::from_ptr(base_ptr.add(left_len));
            let right_len = slice.len() - left_len;
            self.join(
                move |ctx| {
                    let slice = std::slice::from_raw_parts_mut(left.ptr(), left_len);
                    ctx.slice_for_each_mut_impl(slice, op)
                },
                move |ctx| {
                    let slice = std::slice::from_raw_parts_mut(right.ptr(), right_len);
                    ctx.slice_for_each_mut_impl(slice, op)
                },
            );
        }
    }
}

impl<'a> Context<'a, ()> {
    pub fn data(&mut self) -> () { () }

    pub fn with_data<'b, Data: Send>(&mut self, data: &'b mut[Data]) -> Context<'b, (Data,)> {
        assert!(data.len() >= rayon::current_num_threads());
        Context {
            ctx_data: SendPtr { ptr: NonNull::new(data.as_mut_ptr()).unwrap() },
            slice_split_threshold: self.slice_split_threshold,
            _marker: PhantomData,
        }
    }
}

impl<'a, D1: Send> Context<'a, (D1,)> {
    pub fn data(&mut self) -> &mut D1 {
        let idx = rayon::current_thread_index().unwrap() as isize;
        unsafe { self.ctx_data.ptr.offset(idx).as_mut() }
    }

    pub fn with_data<'b, D2: Send>(&mut self, data: &'b mut[D2]) -> Context<'b, (D1, D2)> {
        assert!(data.len() >= rayon::current_num_threads());
        Context {
            ctx_data: (
                self.ctx_data.clone(),
                SendPtr { ptr: NonNull::new(data.as_mut_ptr()).unwrap() }
            ),
            slice_split_threshold: self.slice_split_threshold,
            _marker: PhantomData,
        }
    }
}

impl<'a, D1: Send, D2: Send> Context<'a, (D1, D2)> {
    pub fn data(&mut self) -> (&mut D1, &mut D2) {
        let idx = rayon::current_thread_index().unwrap() as isize;
        unsafe {
            (
                self.ctx_data.0.ptr.offset(idx).as_mut(),
                self.ctx_data.1.ptr.offset(idx).as_mut(),
            )
        }
    }

    pub fn with_data<'b, D3: Send>(&mut self, data: &'b mut[D3]) -> Context<'b, (D1, D2, D3)> {
        assert!(data.len() >= rayon::current_num_threads() + 1);
        Context {
            ctx_data: (
                self.ctx_data.0.clone(),
                self.ctx_data.1.clone(),
                SendPtr { ptr: NonNull::new(data.as_mut_ptr()).unwrap() }
            ),
            slice_split_threshold: self.slice_split_threshold,
            _marker: PhantomData,
        }
    }
}

impl<'a, D1: Send, D2: Send, D3: Send> Context<'a, (D1, D2, D3)> {
    pub fn data(&mut self) -> (&mut D1, &mut D2, &mut D3) {
        let idx = rayon::current_thread_index().unwrap() as isize;
        unsafe {
            (
                self.ctx_data.0.ptr.offset(idx).as_mut(),
                self.ctx_data.1.ptr.offset(idx).as_mut(),
                self.ctx_data.2.ptr.offset(idx).as_mut(),
            )
        }
    }
}

pub(crate) trait WorkerData: Send {
    type Ptr: Send + Clone;
}

impl WorkerData for () {
    type Ptr = ();
}

impl<A: Send> WorkerData for (A,) {
    type Ptr = SendPtr<A>;
}

impl<A: Send, B: Send> WorkerData for (A, B) {
    type Ptr = (SendPtr<A>, SendPtr<B>);
}

impl<A: Send, B: Send, C: Send> WorkerData for (A, B, C) {
    type Ptr = (SendPtr<A>, SendPtr<B>, SendPtr<C>);
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
