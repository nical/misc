use bitflags::bitflags;
use std::collections::VecDeque;

use crate::render_pass::RenderPassContext;
use crate::render_task::{RenderTaskAdress, RenderTask};
use crate::units::{SurfaceIntPoint, SurfaceIntRect, SurfaceVector};
use crate::worker::SendPtr;
use crate::RenderPassConfig;

pub type Rect = crate::units::SurfaceRect;
pub type IntRect = crate::units::SurfaceIntRect;

pub type SurfaceIndex = u16;
pub type RendererId = u16;
pub type BatchIndex = u32;
pub type BatchKey = u64;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct BatchId {
    pub index: BatchIndex,
    pub renderer: RendererId,
    pub surface: SurfaceIndex,
    pub order_independent: bool,
    pub scissor: Option<ScissorRect>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ScissorRect {
    pub x: u16,
    pub y: u16,
    pub w: u16,
    pub h: u16,
}

impl ScissorRect {
    pub fn from_surface_rect(r: &IntRect) -> Self {
        ScissorRect {
            x: r.min.x as u16,
            y: r.min.y as u16,
            w: r.width() as u16,
            h: r.height() as u16,
        }
    }
}

bitflags! {
    #[derive(Copy, Clone, Debug, PartialEq)]
    pub struct BatchFlags: u8 {
        const ORDER_INDEPENDENT = 1;
        /// Hint to do extra work toward finding the earliest batch condidate
        /// instead of stopping at the most recent one.
        const EARLIEST_CANDIDATE = 2;
        const NO_OVERLAP = 4;
        /// Whether the item being added to the batch does not handle render
        /// task clips in the shader and may need a scissor rect to avoid
        /// overflowing the render task.
        const NEED_SCISSOR_RECT = 8;
    }
}

struct Batch {
    renderer: RendererId,
    index: BatchIndex,
    key: BatchKey,
    rects: BatchRects,
    scissor: Option<IntRect>,
}

struct BatchRects {
    batch: Rect,
    items: Vec<Rect>,
}

impl BatchRects {
    fn new(rect: &Rect) -> Self {
        BatchRects {
            batch: *rect,
            items: Vec::new(),
        }
    }

    fn add_rect(&mut self, rect: &Rect) {
        let union = self.batch.union(rect);

        if !self.items.is_empty() {
            self.items.push(*rect);
        } else if self.batch.area() + rect.area() > union.area() {
            self.items.reserve(16);
            self.items.push(self.batch);
            self.items.push(*rect);
        }

        self.batch = union;
    }

    fn intersects(&mut self, rect: &Rect) -> bool {
        if !self.batch.intersects(rect) {
            return false;
        }

        if self.items.is_empty() {
            true
        } else {
            self.items.iter().any(|item| item.intersects(rect))
        }
    }
}

pub struct OrderedBatcher {
    lookback: VecDeque<Batch>,
    batches: Vec<BatchId>,
    pass_idx: SurfaceIndex,
}

impl OrderedBatcher {
    fn find_compatible_batch(
        &mut self,
        renderer: RendererId,
        key: &BatchKey,
        rect: &Rect,
        scissor: Option<&IntRect>,
        flags: BatchFlags,
    ) -> Option<BatchIndex> {
        let selected = if flags.contains(BatchFlags::NO_OVERLAP) {
            self.find_no_overlap(renderer, key, rect, flags)
        } else {
            self.find(renderer, key, rect, scissor, flags)
        };

        if let Some(idx) = selected {
            let batch = &mut self.lookback[idx];
            batch.rects.add_rect(rect);
            return Some(batch.index);
        }

        return None;
    }

    pub fn add_batch(
        &mut self,
        renderer: RendererId,
        batch_index: BatchIndex,
        key: &BatchKey,
        rect: &Rect,
        scissor: Option<&IntRect>,
        _flags: BatchFlags,
    ) {
        if self.lookback.capacity() == self.lookback.len() {
            self.lookback.pop_front();
        }
        self.lookback.push_back(Batch {
            renderer,
            index: batch_index,
            key: *key,
            rects: BatchRects::new(rect),
            scissor: scissor.cloned(),
        });
        self.batches.push(BatchId {
            renderer,
            index: batch_index,
            surface: self.pass_idx,
            order_independent: false,
            scissor: scissor.map(|r| ScissorRect::from_surface_rect(r)),
        });
    }

    pub fn set_render_pass(&mut self, pass_idx: u16) {
        self.pass_idx = pass_idx;
        self.lookback.clear();
    }

    pub fn begin(&mut self) {
        self.lookback.clear();
        self.batches.clear();
        self.pass_idx = 0;
    }

    pub fn batches(&self) -> &[BatchId] {
        &self.batches
    }

    pub fn new() -> Self {
        Self::with_lookback(64)
    }

    pub fn with_lookback(lookback_limit: usize) -> Self {
        OrderedBatcher {
            lookback: VecDeque::with_capacity(lookback_limit),
            batches: Vec::new(),
            pass_idx: 0,
        }
    }

    fn find(
        &mut self,
        renderer: RendererId,
        key: &BatchKey,
        rect: &Rect,
        scissor: Option<&IntRect>,
        flags: BatchFlags,
    ) -> Option<usize> {
        let mut selected = None;
        let first_candidate = !flags.contains(BatchFlags::EARLIEST_CANDIDATE);
        for (offset, batch) in self.lookback.iter_mut().enumerate().rev() {
            if batch.renderer == renderer && *key == batch.key && scissor == batch.scissor.as_ref() {
                selected = Some(offset);
                if first_candidate {
                    break;
                }
            }

            if batch.rects.intersects(rect) {
                break;
            }
        }

        selected
    }

    fn find_no_overlap(
        &mut self,
        renderer: RendererId,
        key: &BatchKey,
        rect: &Rect,
        flags: BatchFlags,
    ) -> Option<usize> {
        let mut selected = None;
        let first_candidate = !flags.contains(BatchFlags::EARLIEST_CANDIDATE);
        for (offset, batch) in self.lookback.iter_mut().enumerate().rev() {
            if batch.rects.intersects(rect) {
                break;
            }

            if batch.renderer == renderer && *key == batch.key {
                selected = Some(offset);
                if first_candidate {
                    break;
                }
            }
        }

        selected
    }
}

struct OrderIndependentBatch {
    renderer: RendererId,
    index: BatchIndex,
    key: BatchKey,
    rect: Rect,
}

pub struct OrderIndependentBatcher {
    candidates: Vec<OrderIndependentBatch>,
    batches: Vec<BatchId>,
    splits: Vec<usize>,
    pass_idx: SurfaceIndex,
}

impl OrderIndependentBatcher {
    pub fn find_compatible_batch(
        &mut self,
        renderer: RendererId,
        key: &BatchKey,
        rect: &Rect,
        flags: BatchFlags,
    ) -> Option<BatchIndex> {
        if flags.contains(BatchFlags::NO_OVERLAP) {
            return self.find_no_overlap(renderer, key, rect);
        }

        for batch in self.candidates.iter_mut().rev() {
            if batch.renderer == renderer && *key == batch.key {
                batch.rect = batch.rect.union(rect);
                return Some(batch.index);
            }
        }

        return None;
    }

    pub fn add_batch(
        &mut self,
        renderer: RendererId,
        index: BatchIndex,
        key: &BatchKey,
        rect: &Rect,
        scissor: Option<&IntRect>,
        _flags: BatchFlags,
    ) {
        self.candidates.push(OrderIndependentBatch {
            renderer,
            index,
            key: *key,
            rect: *rect,
        });
        self.batches.push(BatchId {
            renderer,
            index,
            surface: self.pass_idx,
            order_independent: true,
            scissor: scissor.map(|r| ScissorRect::from_surface_rect(r)),
        });
    }

    pub fn set_render_pass(&mut self, pass_idx: SurfaceIndex) {
        self.candidates.clear();
        self.splits.push(self.batches.len());
        self.pass_idx = pass_idx;
    }

    pub fn begin(&mut self) {
        self.candidates.clear();
        self.batches.clear();
        self.splits.clear();
        self.pass_idx = 0;
    }

    pub fn batches(&self) -> &[BatchId] {
        &self.batches
    }

    pub fn new() -> Self {
        OrderIndependentBatcher {
            batches: Vec::new(),
            candidates: Vec::new(),
            splits: Vec::new(),
            pass_idx: 0,
        }
    }

    fn find_no_overlap(
        &mut self,
        renderer: RendererId,
        key: &BatchKey,
        rect: &Rect,
    ) -> Option<BatchIndex> {
        for batch in self.candidates.iter_mut().rev() {
            if batch.renderer == renderer && *key == batch.key && !batch.rect.intersects(rect) {
                return Some(batch.index);
            }
        }

        return None;
    }
}

pub struct Batcher {
    ordered: OrderedBatcher,
    order_independent: OrderIndependentBatcher,
    view_port: Rect,
    view: RenderTask,
    // True if the current render task does not cover the entire
    // render target. In this case we may need to insert scissor
    // rects to ensure that drawing commands don't overlfow the
    // bounds of the render task.
    may_need_scissor: bool,
}

impl Batcher {
    pub fn find_compatible_batch(
        &mut self,
        renderer: RendererId,
        key: &BatchKey,
        rect: &Rect,
        flags: BatchFlags,
    ) -> Option<BatchIndex> {
        let scissor = if flags.contains(BatchFlags::NEED_SCISSOR_RECT) {
            Some(&self.view.target_rect)
        } else {
            None
        };
        if flags.contains(BatchFlags::ORDER_INDEPENDENT) {
            self.order_independent
                .find_compatible_batch(renderer, key, rect, flags)
        } else {
            self.ordered
                .find_compatible_batch(renderer, key, rect, scissor, flags)
        }
    }

    pub fn add_batch(
        &mut self,
        renderer: RendererId,
        index: BatchIndex,
        key: &BatchKey,
        rect: &Rect,
        flags: BatchFlags,
    ) {
        let scissor = if flags.contains(BatchFlags::NEED_SCISSOR_RECT) {
            Some(&self.view.target_rect)
        } else {
            None
        };
        if flags.contains(BatchFlags::ORDER_INDEPENDENT) {
            self.order_independent
                .add_batch(renderer, index, key, rect, scissor, flags);
        } else {
            self.ordered.add_batch(renderer, index, key, rect, scissor, flags);
        }
    }

    pub fn find_or_add_batch(
        &mut self,
        renderer: RendererId,
        or_add_index: BatchIndex,
        key: &BatchKey,
        rect: &Rect,
        flags: BatchFlags,
    ) -> BatchIndex {
        if let Some(idx) = self.find_compatible_batch(renderer, key, rect, flags) {
            return idx;
        }
        self.add_batch(renderer, or_add_index, key, rect, flags);

        or_add_index
    }

    pub fn set_render_pass(&mut self, pass_idx: SurfaceIndex, batches: &mut Vec<BatchId>) {
        self.flush_batches(batches);

        self.ordered.set_render_pass(pass_idx);
        self.order_independent.set_render_pass(pass_idx);
    }

    pub fn begin(&mut self, render_task: &RenderTask) {
        self.ordered.begin();
        self.order_independent.begin();
        self.view = *render_task;
        //let r = render_task.bounds.inflate(-20, -20);
        //self.view_port = r.to_f32();
        //self.view.bounds = r;
    }

    pub fn set_render_task(&mut self, render_task: &RenderTask) {
        self.view_port = render_task.bounds;
        self.view = *render_task;
        self.may_need_scissor = self.view.bounds.to_i32() != self.view.target_rect;
    }

    pub fn finish(&mut self, batches: &mut Vec<BatchId>) {
        self.flush_batches(batches);
    }

    pub fn new() -> Self {
        use std::f32::{MAX, MIN};
        use crate::units::point;
        let max_rect = SurfaceIntRect {
            min: SurfaceIntPoint::new(i32::MIN, i32::MIN),
            max: SurfaceIntPoint::new(i32::MAX, i32::MAX),
        };
        Batcher {
            ordered: OrderedBatcher::new(),
            order_independent: OrderIndependentBatcher::new(),
            view_port: Rect {
                min: point(MIN, MIN),
                max: point(MAX, MAX),
            },
            view: RenderTask {
                bounds: max_rect.to_f32(),
                target_rect: max_rect,
                offset: SurfaceVector::new(0.0, 0.0),
                gpu_address: RenderTaskAdress::NONE,
            },
            may_need_scissor: false,
        }
    }

    fn flush_batches(&mut self, batches: &mut Vec<BatchId>) {
        let count = self.order_independent.batches.len() + self.ordered.batches.len();
        batches.reserve(count);

        batches.extend_from_slice(&self.order_independent.batches);
        self.order_independent.batches.clear();

        batches.extend_from_slice(&self.ordered.batches);
        self.ordered.batches.clear();
    }
}

#[repr(transparent)]
pub struct BatchRef<'l, T, I> {
    inner: &'l mut (Vec<T>, RenderPassConfig, I),
}

impl <'l, T, I> BatchRef<'l, T, I> {
    #[inline]
    pub fn push(&mut self, val: T) {
        self.inner.0.push(val);
    }

    #[inline]
    pub fn render_pass_config(&mut self) -> RenderPassConfig {
        self.inner.1
    }

    #[inline]
    pub fn batch_data(&mut self) -> &mut I {
        &mut self.inner.2
    }
}

/// A helper class for storing batches in a renderer.
pub struct BatchList<T, I> {
    // TODO: try something more efficient than Vec<Vec<T>>
    batches: Vec<(Vec<T>, RenderPassConfig, I)>,
    renderer: RendererId,
}

impl<T, I> BatchList<T, I> {
    pub fn new(renderer: RendererId) -> Self {
        BatchList {
            batches: Vec::new(),
            renderer,
        }
    }

    pub fn add(
        &mut self,
        ctx: &mut RenderPassContext,
        batch_key: &BatchKey,
        aabb: &Rect,
        mut flags: BatchFlags,
        or_add: &mut impl FnMut() -> I,
        then: &mut impl FnMut(BatchRef<T, I>, &RenderTask),
    ) {
        let mut skip_scissor = !ctx.batcher.may_need_scissor;
        skip_scissor = skip_scissor ||flags.contains(BatchFlags::NEED_SCISSOR_RECT)
            && ctx.batcher.view.bounds.contains_box(&aabb);
        if skip_scissor {
            flags.remove(BatchFlags::NEED_SCISSOR_RECT);
        }

        // TODO: using the intersection with the viewport isn't quite correct in
        // the case of stencil and cover because while the cover geometry can be
        // clipped exactly, the stencil geometry may extend out of it.
        let clipped_aabb = aabb.intersection_unchecked(&ctx.batcher.view_port)
            .translate(ctx.batcher.view.offset);

        if clipped_aabb.is_empty() {
            return;
        }

        let new_batch_index = self.batches.len() as u32;
        let batch_index =
            ctx.batcher.find_or_add_batch(self.renderer, new_batch_index, batch_key, &aabb, flags);

        if batch_index == new_batch_index {
            self.batches.push((Vec::with_capacity(32), ctx.config, or_add()));
        }

        let b = &mut self.batches[batch_index as usize];

        then(BatchRef { inner: b }, &ctx.batcher.view);
    }

    pub fn get(&self, index: BatchIndex) -> (&[T], &RenderPassConfig, &I) {
        let b = &self.batches[index as usize];
        (&b.0, &b.1, &b.2)
    }

    pub fn get_mut(&mut self, index: BatchIndex) -> (&mut Vec<T>, &RenderPassConfig, &mut I) {
        let b = &mut self.batches[index as usize];
        (&mut b.0, &b.1, &mut b.2)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&Vec<T>, &RenderPassConfig, &I)> {
        self.batches.iter().map(|b| (&b.0, &b.1, &b.2))
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&mut Vec<T>, &RenderPassConfig, &mut I)> {
        self.batches.iter_mut().map(|b| (&mut b.0, &b.1, &mut b.2))
    }

    /// # Safety
    ///
    /// batch ids in the render pass must appear at most once.
    #[allow(private_bounds)]
    pub unsafe fn par_iter_mut<D, Op>(
        &mut self,
        ctx: &mut crate::worker::Context<D>,
        pass: &crate::render_pass::BuiltRenderPass,
        renderer_id: RendererId,
        op: &Op,
    )
    where
        D: crate::worker::WorkerData,
        I: Send + Sync,
        T: Send,
        Op: Fn(&mut crate::worker::Context<D>, BatchId, &[T], &RenderPassConfig, &mut I) + Send + Sync,
    {
        let batches_ptr = self.batches.as_mut_ptr();
        let batches = pass
            .batches()
            .iter()
            .filter(|batch| batch.renderer == renderer_id)
            .map(|id| unsafe {
                (*id, SendPtr::from_ptr(batches_ptr.add(id.index as usize)))
            });

        ctx.for_each(batches, &|ctx, (id, batch)| {
            let batch = unsafe { batch.ptr().as_mut().unwrap() };
            op(ctx, *id,  &batch.0, &batch.1, &mut batch.2)
        })
    }

    pub fn is_empty(&self) -> bool {
        self.batches.is_empty()
    }

    pub fn batch_count(&self) -> usize {
        self.batches.len()
    }

    pub fn clear(&mut self) {
        self.batches.clear();
    }

    pub fn take(&mut self) -> Self {
        BatchList {
            batches: std::mem::take(&mut self.batches),
            renderer: self.renderer,
        }
    }
}
