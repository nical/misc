use std::collections::VecDeque;
use bitflags::bitflags;

pub type Rect = lyon::math::Box2D;

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


pub trait Batcher {
    fn begin(&mut self) {}
    fn finish(&mut self) {}

    fn find_compatible_batch(
        &mut self,
        renderer: RendererId,
        key: &BatchKey,
        rect: &Rect,
        flags: BatchFlags,
    ) -> Option<BatchIndex>;

    fn add_batch(
        &mut self,
        renderer: RendererId,
        batch_index: BatchIndex,
        key: &BatchKey,
        rect: &Rect,
        flags: BatchFlags,
    );

    fn find_or_add_batch(
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

    /// Batches are not merged and reordered across split points.
    fn set_render_pass(&mut self, pass_idx: SurfaceIndex);

    fn batches(&self) -> &[BatchId];
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

impl Batcher for OrderedBatcher {
    fn find_compatible_batch(
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

    fn add_batch(
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

    fn set_render_pass(&mut self, pass_idx: u16) {
        self.pass_idx = pass_idx;
        self.lookback.clear();
    }


    fn begin(&mut self) {
        self.lookback.clear();
        self.batches.clear();
        self.pass_idx = 0;
    }

    fn batches(&self) -> &[BatchId] {
        &self.batches
    }
}

impl OrderedBatcher {
    pub fn new() -> Self {
        Self::with_lookback(32)
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

    pub fn batches(&self) -> &[BatchId] {
        &self.batches
    }
}


pub struct BasicOrderedBatcher {
    batches: Vec<BatchId>,
    prev: Option<(RendererId, BatchIndex, BatchKey)>,
    pass_idx: SurfaceIndex,
}

impl Batcher for BasicOrderedBatcher {
    fn find_compatible_batch(
        &mut self,
        renderer: RendererId,
        key: &BatchKey,
        _rect: &Rect,
        _flags: BatchFlags,
    ) -> Option<BatchIndex> {
        if let Some((prev_renderer, index, prev_key)) = self.prev {
            if renderer == prev_renderer && *key == prev_key {
                return Some(index);
            }
        }

        None
    }

    fn add_batch(
        &mut self,
        renderer: RendererId,
        index: BatchIndex,
        key: &BatchKey,
        _rect: &Rect,
        _flags: BatchFlags,
    ) {
        self.prev = Some((renderer, index, *key));
        self.batches.push(BatchId { renderer, index, surface: self.pass_idx });
    }

    fn set_render_pass(&mut self, pass_idx: SurfaceIndex) {
        self.prev = None;
        self.pass_idx = pass_idx;
    }

    fn begin(&mut self) {
        self.batches.clear();
        self.prev = None;
        self.pass_idx = 0;
    }

    fn batches(&self) -> &[BatchId] {
        &self.batches
    }
}

impl BasicOrderedBatcher {
    pub fn new() -> Self {
        BasicOrderedBatcher {
            batches: Vec::new(),
            prev: None,
            pass_idx: 0,
        }
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

impl Batcher for OrderIndependentBatcher {
    fn find_compatible_batch(
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

    fn add_batch(
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
        self.batches.push(BatchId { renderer, index, surface: self.pass_idx });
    }

    fn set_render_pass(&mut self, pass_idx: SurfaceIndex) {
        self.candidates.clear();
        self.splits.push(self.batches.len());
        self.pass_idx = pass_idx;
    }

    fn begin(&mut self) {
        self.candidates.clear();
        self.batches.clear();
        self.splits.clear();
        self.pass_idx = 0;
    }

    fn finish(&mut self) {
        // Reverse the batches to increase the likelihood of opaque primitives
        // being rendered in front-to-back order.
        // Don´t reorder batches across split points, though.
        let mut start = 0;
        for split in &self.splits {
            self.batches[start..*split].reverse();
            start = *split;
        }
        let end = self.batches.len();
        self.batches[start..end].reverse();
    }

    fn batches(&self) -> &[BatchId] {
        &self.batches
    }
}

impl OrderIndependentBatcher {
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
            if batch.renderer == renderer
            && *key == batch.key
            && !batch.rect.intersects(rect) {
                return Some(batch.index);
            }
        }

        return None;
    }

}

pub struct DefaultBatcher {
    ordered: OrderedBatcher,
    order_independent: OrderIndependentBatcher,
}

impl Batcher for DefaultBatcher {
    fn find_compatible_batch(
        &mut self,
        renderer: RendererId,
        key: &BatchKey,
        rect: &Rect,
        flags: BatchFlags,
    ) -> Option<BatchIndex> {
        if flags.contains(BatchFlags::ORDER_INDEPENDENT) {
            self.order_independent.find_compatible_batch(renderer, key, rect, flags)
        } else {
            self.ordered.find_compatible_batch(renderer, key, rect, flags)
        }
    }

    fn add_batch(
        &mut self,
        renderer: RendererId,
        index: BatchIndex,
        key: &BatchKey,
        rect: &Rect,
        flags: BatchFlags,
    ) {
        if flags.contains(BatchFlags::ORDER_INDEPENDENT) {
            self.order_independent.add_batch(renderer, index, key, rect, flags);
        } else {
            self.ordered.add_batch(renderer, index, key, rect, flags);
        }
    }

    fn set_render_pass(&mut self, pass_idx: SurfaceIndex) {
        self.ordered.set_render_pass(pass_idx);
        self.order_independent.set_render_pass(pass_idx);
    }

    fn begin(&mut self) {
        self.ordered.begin();
        self.order_independent.begin();
    }

    fn finish(&mut self) {
        self.ordered.finish();
        self.order_independent.finish();
        // Put all batches in the same vector.
        self.order_independent.batches.extend_from_slice(&self.ordered.batches);
    }

    fn batches(&self) -> &[BatchId] {
        &self.order_independent.batches
    }
}

impl DefaultBatcher {
    pub fn new() -> Self {
        DefaultBatcher {
            ordered: OrderedBatcher::new(),
            order_independent: OrderIndependentBatcher::new(),
        }
    }
}

