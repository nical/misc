pub mod gpu_store;
pub mod shader;
pub mod storage_buffer;

pub use gpu_store::*;
pub use shader::Shaders;

pub use wgslp::preprocessor::{Preprocessor, Source, SourceError};

use self::shader::{DepthMode, StencilMode, SurfaceConfig};

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct GpuTargetDescriptor {
    pub width: f32,
    pub height: f32,
    pub inv_width: f32,
    pub inv_height: f32,
}

impl GpuTargetDescriptor {
    pub fn new(w: u32, h: u32) -> Self {
        let width = w as f32;
        let height = h as f32;
        let inv_width = 1.0 / width;
        let inv_height = 1.0 / height;
        GpuTargetDescriptor {
            width,
            height,
            inv_width,
            inv_height,
        }
    }
}

unsafe impl bytemuck::Pod for GpuTargetDescriptor {}
unsafe impl bytemuck::Zeroable for GpuTargetDescriptor {}

#[derive(Clone)]
pub struct VertexBuilder {
    location: u32,
    offset: u64,
    attributes: Vec<wgpu::VertexAttribute>,
    step_mode: wgpu::VertexStepMode,
}

impl VertexBuilder {
    pub fn new(step_mode: wgpu::VertexStepMode) -> Self {
        VertexBuilder {
            location: 0,
            offset: 0,
            attributes: Vec::with_capacity(16),
            step_mode,
        }
    }

    pub fn from_slice(step_mode: wgpu::VertexStepMode, formats: &[wgpu::VertexFormat]) -> Self {
        let mut attributes = VertexBuilder::new(step_mode);
        for format in formats {
            attributes.push(*format);
        }

        attributes
    }

    pub fn push(&mut self, format: wgpu::VertexFormat) {
        self.attributes.push(wgpu::VertexAttribute {
            format,
            offset: self.offset,
            shader_location: self.location,
        });
        self.offset += format.size();
        self.location += 1;
    }

    pub fn get(&self) -> &[wgpu::VertexAttribute] {
        &self.attributes
    }

    pub fn clear(&mut self) {
        self.location = 0;
        self.offset = 0;
        self.attributes.clear();
    }

    pub fn buffer_layout(&self) -> wgpu::VertexBufferLayout {
        wgpu::VertexBufferLayout {
            array_stride: self.offset,
            step_mode: self.step_mode,
            attributes: &self.attributes,
        }
    }
}

// TODO: manage the number of shader configurations so that it remains reasonable
// while not constraining renderers too much.
// for example we don't want depth and stencil in the tile atlas passes
// but we should allow the tiling renderer to work in a render pass that has depth
// and/or stencil enabled (ignoring them).
pub struct PipelineDefaults {
    color_format: wgpu::TextureFormat,
    mask_format: wgpu::TextureFormat,
    depth_buffer: bool,
    stencil_buffer: bool,
    msaa_samples: u32,
}

impl PipelineDefaults {
    pub fn new() -> Self {
        PipelineDefaults {
            color_format: wgpu::TextureFormat::Bgra8Unorm,
            mask_format: wgpu::TextureFormat::R8Unorm,
            depth_buffer: true,
            stencil_buffer: true,
            msaa_samples: 4,
        }
    }

    pub fn primitive_state(&self) -> wgpu::PrimitiveState {
        wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            polygon_mode: wgpu::PolygonMode::Fill,
            front_face: wgpu::FrontFace::Ccw,
            strip_index_format: None,
            cull_mode: None,
            unclipped_depth: false,
            conservative: false,
        }
    }

    pub fn color_target_state(&self) -> Option<wgpu::ColorTargetState> {
        Some(wgpu::ColorTargetState {
            format: self.color_format,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        })
    }

    pub fn color_target_state_no_blend(&self) -> Option<wgpu::ColorTargetState> {
        Some(wgpu::ColorTargetState {
            format: self.color_format,
            blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
            write_mask: wgpu::ColorWrites::ALL,
        })
    }

    pub fn alpha_target_state(&self) -> Option<wgpu::ColorTargetState> {
        Some(wgpu::ColorTargetState {
            format: self.mask_format,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        })
    }

    pub fn depth_stencil_format(&self) -> Option<wgpu::TextureFormat> {
        match (self.depth_buffer, self.stencil_buffer) {
            (false, false) => None,
            (true, false) => Some(wgpu::TextureFormat::Depth32Float),
            (false, true) => Some(wgpu::TextureFormat::Stencil8),
            (true, true) => Some(wgpu::TextureFormat::Depth24PlusStencil8),
        }
    }

    pub fn color_format(&self) -> wgpu::TextureFormat {
        self.color_format
    }

    pub fn mask_format(&self) -> wgpu::TextureFormat {
        self.mask_format
    }

    pub fn msaa_format(&self) -> wgpu::TextureFormat {
        self.color_format
    }

    pub fn msaa_sample_count(&self) -> u32 {
        self.msaa_samples
    }

    pub fn surface_config(&self, msaa: bool) -> SurfaceConfig {
        SurfaceConfig {
            msaa,
            depth: if self.depth_buffer {
                DepthMode::Ignore
            } else {
                DepthMode::None
            },
            stencil: if self.stencil_buffer {
                StencilMode::Ignore
            } else {
                StencilMode::None
            },
        }
    }
}
