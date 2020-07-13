use crate::{Batch, BatchingConfig, Rect};

/// A list of batches that don't preserve ordering.
///
/// Typically usedful for fully opaque primitives when the depth-buffer is used for
/// occlusion culling.
pub struct OrderIndependentBatchList<Batch> {
    batches: Vec<Batch>,
}

impl<B: Batch> OrderIndependentBatchList<B> {
    pub fn new(_config: &BatchingConfig) -> Self {
        OrderIndependentBatchList {
            batches: Vec::new(),
        }
    }

    pub fn add_instance(&mut self, key: &B::Key, instance: B::Instance, rect: &Rect) {
        self.add_instances(key, &[instance], rect);
    }

    pub fn add_instances(&mut self, key: &B::Key, instances: &[B::Instance], rect: &Rect) {
        for batch in self.batches.iter_mut().rev() {
            if batch.add_instances(key, instances, rect) {
                return;
            }
        }

        self.batches.push(Batch::new(key, instances, rect));
    }

    pub fn optimize(&mut self, _config: &BatchingConfig) {

    }
}
