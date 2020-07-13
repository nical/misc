use crate::{Batch, BatchingConfig, Rect, Stats};

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

/// A list of batches that preserve the ordering of overlapping primitives. 
pub struct OrderedBatchList<B> {
    batches: Vec<OrderedBatch<B>>,
    rects: Vec<BatchRects>,
    max_lookback: usize,
    hit_lookback_limit: u32,
}

struct OrderedBatch<B> {
    batch: B,
    cost: f32,
}

impl<B: Batch> OrderedBatchList<B> {
    pub fn new(config: &BatchingConfig) -> Self {
        OrderedBatchList {
            batches: Vec::new(),
            rects: Vec::new(),
            max_lookback: config.max_lookback,
            hit_lookback_limit: 0,
        }
    }

    pub fn add_instance(&mut self, key: &B::Key, instance: B::Instance, rect: &Rect) {
        self.add_instances(key, &[instance], rect);
    }

    pub fn add_instances(&mut self, key: &B::Key, instances: &[B::Instance], rect: &Rect) {
        let mut intersected = false;
        for (batch_index, batch) in self.batches.iter_mut().enumerate().rev().take(self.max_lookback) {
            if batch.batch.add_instances(key, instances, rect) {
                self.rects[batch_index].add_rect(rect);
                batch.cost += rect.area();
                return;
            }

            if self.rects[batch_index].intersects(rect) {
                intersected = true;
                break;
            }
        }

        if !intersected && self.batches.len() > self.max_lookback {
            self.hit_lookback_limit += 1;
        }

        self.batches.push(OrderedBatch {
            batch: Batch::new(key, instances, rect),
            cost: rect.area(),
        });
        self.rects.push(BatchRects::new(rect));
    }

    pub fn add_batch(&mut self, key: &B::Key, instances: &[B::Instance], rect: &Rect) {
        self.batches.push(OrderedBatch {
            batch: Batch::new(key, instances, rect),
            cost: rect.area(),
        });
        self.rects.push(BatchRects::new(rect));
    }

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

    pub fn stats(&self) -> Stats {
        Stats {
            hit_lookback_limit: self.hit_lookback_limit,
            num_batches: self.batches.len() as u32,
            num_instances: self.batches.iter().fold(
                0,
                |count, batch| count + batch.batch.num_instances() as u32,
            ),
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
