//use core::gpu::PipelineDefaults;
use core::wgpu;
use core::{
    gpu::shader::Shaders,
    resources::RendererResources,
};

pub struct TemplateGpuResources {
}

impl TemplateGpuResources {
    pub fn new(_device: &wgpu::Device, _shaders: &mut Shaders) -> Self {
        TemplateGpuResources {
        }
    }
}

impl RendererResources for TemplateGpuResources {
    fn name(&self) -> &'static str {
        "TemplateGpuResources"
    }

    fn begin_frame(&mut self) {}

    fn begin_rendering(&mut self, _encoder: &mut wgpu::CommandEncoder) {}

    fn end_frame(&mut self) {}
}
