use crate::batching::Batcher;
use crate::types::units::Rect;
use crate::types::*;

pub struct TransformTree;

pub struct PrepareContext<'l> {
    pub transforms: &'l TransformTree,
}

pub struct ProcessContext<'l> {
    pub transforms: &'l TransformTree,
}

pub struct DrawCallsContext<'l> {
    pub device: &'l mut wgpu::Device,
    pub encoder: &'l mut wgpu::CommandEncoder,
}

pub trait RenderingSystem: Send {
    fn begin_frame(&mut self);
    fn end_frame(&mut self);
    fn prepare(&mut self, ctx: &mut PrepareContext);
    fn process(&mut self, ctx: &mut ProcessContext);
    fn batch_primitive(&mut self, batcher: &mut dyn Batcher, prim_index: PrimitiveIndex, rect: &Rect);
    fn generate_draw_call(&self, batch: BatchIndex, ctx: &mut DrawCallsContext);
}
