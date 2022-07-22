use crate::system::RenderingSystem;
//use crate::types::{Primitive, BatchId, units::Rect};

pub struct RendererOptions {

}

pub struct Renderer {
    systems: Vec<Box<dyn RenderingSystem>>,
}

impl Renderer {
    pub fn new(_options: &RendererOptions, systems: Vec<Box<dyn RenderingSystem>>) -> Self {
        Renderer {
            systems,
        }
    }

    pub fn systems(&self) -> &[Box<dyn RenderingSystem>] {
        &self.systems
    }
}

/*
fn generate_batches(
    batcher: &mut dyn crate::batching::Batcher,
    systems: &mut [Box<dyn RenderingSystem>],
    primitives: &[Primitive],
    target_rects: &[Rect],
) {
    for (primitive, rect) in primitives.iter().zip(target_rects.iter()) {
        systems[primitive.system as usize].batch_primitive(batcher, primitive.index, rect);
    }
}

fn generate_render_pass_draw_calls(
    systems: &mut [Box<dyn RenderingSystem>],
    ctx: &mut DrawCallsContext,
    batches: &[BatchId],
) {
    for batch in batches {
        systems[batch.system as usize].generate_draw_call(batch.index, ctx);
    }
}
*/