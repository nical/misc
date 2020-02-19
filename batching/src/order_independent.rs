use crate::{Batch, BatchingConfig, BatchKey, Rect};

/// A list of batches that don't preserve ordering.
///
/// Typically usedful for fully opaque primitives when the depth-buffer is used for
/// occlusion culling.
pub struct OrderIndependentBatchList<Key, Instance> {
    batches: Vec<Batch<Key, Instance>>,
}

impl<Key: BatchKey, Instance: Clone> OrderIndependentBatchList<Key, Instance> {
    pub fn new(_config: &BatchingConfig) -> Self {
        OrderIndependentBatchList {
            batches: Vec::new(),
        }
    }

    pub fn add_instance(&mut self, key: &Key, instance: Instance, rect: &Rect) {
        let selected_batch_index = self.select_batch(key, rect);

        self.batches[selected_batch_index].instances.push(instance);
    }

    pub fn add_instances(&mut self, key: &Key, instances: &[Instance], rect: &Rect) {
        let selected_batch_index = self.select_batch(key, rect);

        let batch = &mut self.batches[selected_batch_index];
        batch.instances.extend_from_slice(instances);
        batch.cost += rect.area();
        batch.key.combine(&key);
    }

    fn select_batch(&mut self, key: &Key, rect: &Rect) -> usize {
        let mut selected_batch_index = None;
        'outer: for (batch_index, batch) in self.batches.iter().enumerate().rev() {
            if key.should_add_to_batch(&batch.key) {
                selected_batch_index = Some(batch_index);
                break;
            }
        }

        selected_batch_index.unwrap_or_else(|| {
            let index = self.batches.len();
            self.batches.push(Batch::with_key(key.clone()));

            index
        })
    }

    pub fn optimize(&mut self, _config: &BatchingConfig) {

    }
}
