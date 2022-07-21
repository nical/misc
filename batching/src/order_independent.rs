use crate::{BatchingConfig, Rect, BatchId, Batcher, SystemId, BatchIndex};

/// A list of batches that don't preserve ordering.
///
/// Typically usedful for fully opaque primitives when the depth-buffer is used for
/// occlusion culling.
pub struct OrderIndependentBatcher {
    batches: Vec<BatchId>,
    // TODO: record batches with large occluders and reorder them to render them first.
}

impl OrderIndependentBatcher {
    pub fn new(_config: &BatchingConfig) -> Self {
        OrderIndependentBatcher {
            batches: Vec::new(),
        }
    }
}

impl Batcher for OrderIndependentBatcher {
    fn add_to_existing_batch(
        &mut self,
        system_id: SystemId,
        _rect: &Rect,
        callback: &mut impl FnMut(BatchIndex) -> bool,
    ) -> bool {
        for batch in &mut self.batches {
            if batch.system == system_id && callback(batch.index) {
                return true;
            }
        }

        return false;
    }

    fn add_batch(
        &mut self,
        batch: BatchId,
        _rect: &Rect,
    ) {
        self.batches.push(batch);
    }
}
