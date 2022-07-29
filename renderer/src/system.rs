use crate::batching::Batcher;
use crate::types::units::Rect;
use crate::types::*;

pub struct PrepareContext<'l> {
    pub par: &'l mut parasol::Context,
}

pub struct ProcessContext<'l> {
    pub par: &'l mut parasol::Context,
}

pub struct DrawCallsContext<'l> {
    pub par: &'l mut parasol::Context,
    pub device: &'l wgpu::Device,
    pub encoder: &'l mut wgpu::CommandEncoder,
}

pub trait RenderingSystem: Send + Sync {
    fn begin_frame(&mut self, par: &mut parasol::Context, device: &wgpu::Device);
    fn end_frame(&mut self, par: &mut parasol::Context, device: &wgpu::Device);
    fn prepare(&mut self, ctx: &mut PrepareContext);
    fn process(&mut self, ctx: &mut ProcessContext);
    fn batch_primitive(&mut self, batcher: &mut dyn Batcher, prim_index: PrimitiveIndex, rect: &Rect);
    fn generate_draw_call(&self, batch: BatchIndex, ctx: &mut DrawCallsContext);
}
