use super::{BatchingConfig, Rect, Stats, BatchId, Batcher, BatchIndex, SystemId};

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
    batches: Vec<BatchId>,
    rects: Vec<BatchRects>,
    max_lookback: usize,
    hit_lookback_limit: u32,
}

impl OrderedBatcher {
    pub fn new(config: &BatchingConfig) -> Self {
        OrderedBatcher {
            batches: Vec::new(),
            rects: Vec::new(),
            max_lookback: config.max_lookback,
            hit_lookback_limit: 0,
        }
    }

    pub fn clear(&mut self) {
        self.batches.clear();
        self.rects.clear();
        self.hit_lookback_limit = 0;
    }

    pub fn upudate_stats(&self, stats: &mut Stats) {
        stats.hit_lookback_limit += self.hit_lookback_limit;
    }

    pub fn batches(&self) -> &[BatchId] { &self.batches }
}

impl Batcher for OrderedBatcher {
    fn add_to_existing_batch(
        &mut self,
        system_id: SystemId,
        rect: &Rect,
        callback: &mut dyn FnMut(BatchIndex) -> bool,
    ) -> bool {
        let mut intersected = false;
        for (batch_index, batch) in self.batches.iter_mut().enumerate().rev().take(self.max_lookback) {
            if batch.system == system_id && callback(batch.index) {
                self.rects[batch_index].add_rect(rect);
                return true;
            }

            if self.rects[batch_index].intersects(rect) {
                intersected = true;
                break;
            }
        }

        if !intersected && self.batches.len() > self.max_lookback {
            self.hit_lookback_limit += 1;
        }

        return false;
    }

    fn add_batch(
        &mut self,
        batch_id: BatchId,
        rect: &Rect,
    ) {
        self.batches.push(batch_id);
        self.rects.push(BatchRects::new(rect));
    }
}


pub struct BasicOrderedBatcher {
    batches: Vec<BatchId>,
    prev_system: Option<SystemId>,
    hit_lookback_limit: u32,
}

impl BasicOrderedBatcher {
    pub fn new() -> Self {
        BasicOrderedBatcher {
            batches: Vec::new(),
            prev_system: None,
            hit_lookback_limit: 0,
        }
    }

    pub fn clear(&mut self) {
        self.batches.clear();
        self.prev_system = None;
        self.hit_lookback_limit = 0;
    }

    pub fn upudate_stats(&self, stats: &mut Stats) {
        stats.hit_lookback_limit += self.hit_lookback_limit;
    }
}

impl Batcher for BasicOrderedBatcher {
    fn add_to_existing_batch(
        &mut self,
        system_id: SystemId,
        _rect: &Rect,
        callback: &mut dyn FnMut(BatchIndex) -> bool,
    ) -> bool {
        if self.prev_system != Some(system_id) {
            self.hit_lookback_limit += 1;
            return false;
        }

        if callback(self.batches.last().unwrap().index) {
            return true;
        }

        false
    }

    fn add_batch(
        &mut self,
        batch_id: BatchId,
        _rect: &Rect,
    ) {
        self.batches.push(batch_id);
    }
}


    /*
    pub fn optimize(&mut self, config: &BatchingConfig) {
        if self.batches.len() < config.ideal_batch_count {
            return;
        }

        let mut merge_candidates = Vec::new();

        for batch_index in 0..(self.batches.len() - 1) {
            let a = &self.batches[batch_index];
            let b = &self.batches[batch_index + 1];

            let cost = a.cost + b.cost;
            if a.batch.can_merge(&b.batch) && cost < config.max_merge_cost {
                merge_candidates.push((batch_index, cost));
            }
        }

        merge_candidates.sort_by(|a, b| { a.1.partial_cmp(&b.1).unwrap() });            

        let num_merges = self.batches.len() + 1 - config.ideal_batch_count;
        merge_candidates.truncate(num_merges);

        for &(batch_index, _) in &merge_candidates {
            // we may have merged the batch into its previous one. If so the batch will be empty
            // and we'll want to merge with the previous batch instead.
            let src_index = batch_index + 1;
            let mut dest_index = batch_index;
            while dest_index > 0 && self.batches[dest_index].batch.num_instances() == 0 {
                dest_index -= 1;
            }

            let (src_batch, dest_batch) = get_both_mut(&mut self.batches, src_index, dest_index);

            if src_batch.batch.merge(&mut dest_batch.batch) {
                dest_batch.cost += src_batch.cost;
                src_batch.cost = 0.0;
            }
        }
    }
pub fn get_both_mut<T>(v: &mut [T], a: usize, b: usize) -> (&mut T, &mut T) {
    assert!(a != b);
    assert!(a < v.len());
    assert!(b < v.len());

    unsafe {
        (
            &mut *v.as_mut_ptr().add(a),
            &mut *v.as_mut_ptr().add(b),
        )
    }
}
    */



