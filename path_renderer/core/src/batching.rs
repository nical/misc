use bitflags::bitflags;
use std::collections::VecDeque;

use crate::{context::RenderPassContext, SurfacePassConfig};

pub type Rect = crate::units::SurfaceRect;

pub type SurfaceIndex = u16;
pub type RendererId = u16;
pub type BatchIndex = u32;
pub type BatchKey = u64;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct BatchId {
    pub renderer: RendererId,
    pub index: BatchIndex,
    pub surface: SurfaceIndex,
}

bitflags! {
    #[derive(Copy, Clone, Debug, PartialEq)]
    pub struct BatchFlags: u8 {
        const ORDER_INDEPENDENT = 1;
        /// Hint to do extra work toward finding the earliest batch condidate
        /// instead of stopping at the most recent one.
        const EARLIEST_CANDIDATE = 2;
        const NO_OVERLAP = 4;
    }
}

struct Batch {
    renderer: RendererId,
    index: BatchIndex,
    key: BatchKey,
    rects: BatchRects,
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
    pub fn find_compatible_batch(
        &mut self,
        renderer: RendererId,
        key: &BatchKey,
        rect: &Rect,
        flags: BatchFlags,
    ) -> Option<BatchIndex> {
        let selected = if flags.contains(BatchFlags::NO_OVERLAP) {
            self.find_no_overlap(renderer, key, rect, flags)
        } else {
            self.find(renderer, key, rect, flags)
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
        });
        self.batches.push(BatchId {
            renderer,
            index: batch_index,
            surface: self.pass_idx,
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
        flags: BatchFlags,
    ) -> Option<usize> {
        let mut selected = None;
        let first_candidate = !flags.contains(BatchFlags::EARLIEST_CANDIDATE);
        for (offset, batch) in self.lookback.iter_mut().enumerate().rev() {
            if batch.renderer == renderer && *key == batch.key {
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

    pub fn finish(&mut self) {}
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

    pub fn finish(&mut self) {
        // Reverse the batches to increase the likelihood of opaque primitives
        // being rendered in front-to-back order.
        // Don't reorder batches across split points, though.
        let mut start = 0;
        for split in &self.splits {
            self.batches[start..*split].reverse();
            start = *split;
        }
        let end = self.batches.len();
        self.batches[start..end].reverse();
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
}

impl Batcher {
    pub fn find_compatible_batch(
        &mut self,
        renderer: RendererId,
        key: &BatchKey,
        rect: &Rect,
        flags: BatchFlags,
    ) -> Option<BatchIndex> {
        if flags.contains(BatchFlags::ORDER_INDEPENDENT) {
            self.order_independent
                .find_compatible_batch(renderer, key, rect, flags)
        } else {
            self.ordered
                .find_compatible_batch(renderer, key, rect, flags)
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
        if flags.contains(BatchFlags::ORDER_INDEPENDENT) {
            self.order_independent
                .add_batch(renderer, index, key, rect, flags);
        } else {
            self.ordered.add_batch(renderer, index, key, rect, flags);
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

    pub fn begin(&mut self) {
        self.ordered.begin();
        self.order_independent.begin();
    }

    pub fn finish(&mut self, batches: &mut Vec<BatchId>) {
        self.flush_batches(batches);

        self.ordered.finish();
        self.order_independent.finish();
    }

    pub fn new() -> Self {
        Batcher {
            ordered: OrderedBatcher::new(),
            order_independent: OrderIndependentBatcher::new(),
        }
    }

    fn flush_batches(&mut self, batches: &mut Vec<BatchId>) {
        let count = self.order_independent.batches.len() + self.ordered.batches.len();
        batches.reserve(count);

        self.order_independent.batches.reverse();
        batches.extend_from_slice(&self.order_independent.batches);
        self.order_independent.batches.clear();

        batches.extend_from_slice(&self.ordered.batches);
        self.ordered.batches.clear();
    }
}

pub struct BatchRef<'l, T, I> {
    inner: &'l mut (Vec<T>, SurfacePassConfig, I),
}

impl <'l, T, I> BatchRef<'l, T, I> {
    #[inline]
    pub fn push(&mut self, val: T) {
        self.inner.0.push(val);
    }

    #[inline]
    pub fn surface(&mut self) -> SurfacePassConfig {
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
    batches: Vec<(Vec<T>, SurfacePassConfig, I)>,
    renderer: RendererId,
}

impl<T, I> BatchList<T, I> {
    pub fn new(renderer: RendererId) -> Self {
        BatchList {
            batches: Vec::new(),
            renderer,
        }
    }

    pub fn find_or_add_batch(
        &mut self,
        ctx: &mut RenderPassContext,
        batch_key: &BatchKey,
        aabb: &Rect,
        flags: BatchFlags,
        or_add: &mut impl FnMut() -> I,
    ) -> BatchRef<T, I> {
        let new_batch_index = self.batches.len() as u32;
        let batch_index =
            ctx.batcher.find_or_add_batch(self.renderer, new_batch_index, batch_key, aabb, flags);

        if batch_index == new_batch_index {
            self.batches.push((Vec::with_capacity(32), ctx.surface, or_add()));
        }

        let b = &mut self.batches[batch_index as usize];

        BatchRef { inner: b }
    }

    pub fn get(&self, index: BatchIndex) -> (&[T], &SurfacePassConfig, &I) {
        let b = &self.batches[index as usize];
        (&b.0, &b.1, &b.2)
    }

    pub fn get_mut(&mut self, index: BatchIndex) -> (&mut Vec<T>, &SurfacePassConfig, &mut I) {
        let b = &mut self.batches[index as usize];
        (&mut b.0, &b.1, &mut b.2)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&Vec<T>, &SurfacePassConfig, &I)> {
        self.batches.iter().map(|b| (&b.0, &b.1, &b.2))
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&mut Vec<T>, &SurfacePassConfig, &mut I)> {
        self.batches.iter_mut().map(|b| (&mut b.0, &b.1, &mut b.2))
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
