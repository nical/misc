use crate::{Rect, BatchingConfig, BatchType, Batcher, SystemId, BatchIndex, BatchId, Stats};

pub struct BatchStore<B: BatchType> {
    batch_type: B,
    id: SystemId,
    batches: Vec<StoredBatch<B::Key, B::Instance>>,
    default_batch_capacity: usize,
    max_merge_cost: f32,
}

struct StoredBatch<Key, Instance> {
    key: Key,
    instances: Vec<Instance>,
    area: f32,
}

impl<B: BatchType> BatchStore<B> {
    pub fn new(batch_type: B, config: &BatchingConfig, id: SystemId) -> Self {
        BatchStore {
            batch_type,
            id,
            batches: Vec::new(),
            default_batch_capacity: 128,
            max_merge_cost: config.max_merge_cost,
        }
    }

    pub fn add_instance(
        &mut self,
        batcher: &mut impl Batcher,
        key: &B::Key,
        instance: B::Instance,
        rect: &Rect
    ) {
        self.add_instances(batcher, key, &[instance], rect);
    }

    pub fn add_instances(
        &mut self,
        batcher: &mut impl Batcher,
        key: &B::Key,
        instances: &[B::Instance],
        rect: &Rect,
    ) {
        let added = batcher.add_to_existing_batch(self.id(), rect, &mut|index| {
            let batch = &mut self.batches[index as usize];

            if !self.batch_type.is_compatible(&batch.key, key) {
                return false;
            }
    
            let new_key = self.batch_type.combine_keys(&batch.key, key);
            let current_cost = self.batch_type.cost(&batch.key);
            let new_cost = self.batch_type.cost(&new_key);
            if (new_cost - current_cost) * batch.area > self.max_merge_cost {
                return false;
            }
    
            batch.instances.extend_from_slice(instances);
            batch.area += rect.area();
            batch.key = new_key;
    
            true    
        });

        if !added {
            let index = self.begin_batch(key, instances, rect);
            batcher.add_batch( BatchId { system: self.id(), index }, rect);
        }
    }

    fn begin_batch(&mut self, key: &B::Key, instances: &[B::Instance], rect: &Rect) -> BatchIndex {
        let idx = self.batches.len() as BatchIndex;

        let mut batch = Vec::with_capacity(self.default_batch_capacity.max(instances.len()));
        batch.extend_from_slice(instances);

        self.batches.push(StoredBatch {
            key: key.clone(),
            instances: batch,
            area: rect.area(),
        });

        idx
    }

    pub fn clear(&mut self) {
        self.batches.clear();
    }

    pub fn get_instances(&self, index: BatchIndex) -> &[B::Instance] {
        &self.batches[index as usize].instances[..]
    }

    pub fn id(&self) -> SystemId { self.id }

    pub fn update_stats(&self, stats: &mut Stats) {
        stats.num_batches += self.batches.len() as u32;
        for batch in &self.batches {
            stats.num_instances += batch.instances.len() as u32;
        }
    }
}
