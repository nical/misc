use crate::{Batch, BatchingConfig, BatchKey, Rect};

/// A list of batches that preserve the ordering of overlapping primitives. 
pub struct OrderedBatchList<Key, Instance> {
    batches: Vec<Batch<Key, Instance>>,
    item_rects: Vec<Vec<Rect>>,
    max_lookback: usize,
}

impl<Key: BatchKey, Instance: Clone> OrderedBatchList<Key, Instance> {
    pub fn new(config: &BatchingConfig) -> Self {
        OrderedBatchList {
            batches: Vec::new(),
            item_rects: Vec::new(),
            max_lookback: config.max_lookback,
        }
    }

    pub fn add_instance(&mut self, key: &Key, instance: Instance, rect: &Rect) {
        let selected_batch_index = self.select_batch(key, rect);

        self.item_rects[selected_batch_index].push(*rect);
        self.batches[selected_batch_index].instances.push(instance);
    }

    pub fn add_instances(&mut self, key: &Key, instances: &[Instance], rect: &Rect) {
        let selected_batch_index = self.select_batch(key, rect);

        let batch = &mut self.batches[selected_batch_index];
        batch.instances.extend_from_slice(instances);
        batch.cost += rect.area();
        batch.key.combine(&key);

        self.item_rects[selected_batch_index].push(*rect);
    }

    fn select_batch(&mut self, key: &Key, rect: &Rect) -> usize {
        let mut selected_batch_index = None;
        'outer: for (batch_index, batch) in self.batches.iter().enumerate().rev().take(self.max_lookback) {
            if key.should_add_to_batch(&batch.key) {
                selected_batch_index = Some(batch_index);
                break;
            }
            for item_rect in &self.item_rects[batch_index] {
                if item_rect.intersects(&rect) {
                    break 'outer;
                }
            }
        }

        selected_batch_index.unwrap_or_else(|| {
            let index = self.batches.len();
            self.batches.push(Batch::with_key(key.clone()));
            self.item_rects.push(Vec::new());

            index
        })
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
            if Key::can_merge_batches(&a.key, &b.key) && cost < config.max_merge_cost {
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
            while dest_index > 0 && self.batches[dest_index].instances.is_empty() {
                dest_index -= 1;
            }

            let mut src_batch = std::mem::replace(
                &mut self.batches[src_index],
                Batch::with_key(Key::invalid()),
            );
            let dest_batch = &mut self.batches[dest_index];

            dest_batch.instances.append(&mut src_batch.instances);
            dest_batch.cost += src_batch.cost;
            dest_batch.key.combine(&src_batch.key);
        }
    }
}

