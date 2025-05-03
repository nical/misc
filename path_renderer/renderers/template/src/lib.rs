#![allow(exported_private_dependencies)]

pub extern crate core;

mod renderer;

pub use renderer::*;
use core::batching::RendererId;
use core::wgpu;
use core::gpu::shader::Shaders;

pub struct Template {
}

impl Template {
    pub fn new(_device: &wgpu::Device, _shaders: &Shaders) -> Self {
        Template {}
    }

    pub fn new_renderer(&self, renderer_id: RendererId) -> TemplateRenderer {
        TemplateRenderer::new(renderer_id)
    }
}
