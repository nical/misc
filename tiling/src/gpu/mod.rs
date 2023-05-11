pub mod atlas_uploader;
pub mod gpu_store;
pub mod storage_buffer;

pub use gpu_store::*;

pub use wgslp::preprocessor::{Preprocessor, Source, SourceError};
use std::{collections::HashMap};

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
        let width = w as  f32;
        let height = h as f32;
        let inv_width = 1.0 / width;
        let inv_height = 1.0 / height;
        GpuTargetDescriptor { width, height, inv_width, inv_height }
    }
}

unsafe impl bytemuck::Pod for GpuTargetDescriptor {}
unsafe impl bytemuck::Zeroable for GpuTargetDescriptor {}

impl GpuTargetDescriptor {
    pub fn create_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("target descriptor"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<GpuTargetDescriptor>() as u64),
                    },
                    count: None,
                },
            ],
        })
    }
}

pub struct ShaderSources {
    pub source_library: HashMap<String, Source>,
    pub preprocessor: Preprocessor,
}

impl ShaderSources {
    pub fn new() -> Self {
        let mut library = HashMap::default();

        library.insert("rect".into(), include_str!("../../shaders/lib/rect.wgsl").into());
        library.insert("tiling".into(), include_str!("../../shaders/lib/tiling.wgsl").into());
        library.insert("gpu_store".into(), include_str!("../../shaders/lib/gpu_store.wgsl").into());
        library.insert("render_target".into(), include_str!("../../shaders/lib/render_target.wgsl").into());
        library.insert("mask::fill".into(), include_str!("../../shaders/lib/mask/fill.wgsl").into());
        library.insert("mask::circle".into(), include_str!("../../shaders/lib/mask/circle.wgsl").into());
        library.insert("pattern::color".into(), include_str!("../../shaders/lib/pattern/color.wgsl").into());

        ShaderSources {
            source_library: library,
            preprocessor: Preprocessor::new()
        }
    }

    pub fn preprocess(&mut self, name: &str, src: &str, defines: &[&str]) -> Result<String, SourceError> {
        self.preprocessor.reset_defines();
        for define in defines {
            self.preprocessor.define(define);
        }
        self.preprocessor.preprocess(name, src, &mut self.source_library)
    }

    pub fn create_shader_module(
        &mut self,
        device: &wgpu::Device,
        name: &str,
        src: &str,
        defines: &[&str]
    ) -> wgpu::ShaderModule {
        let src = self.preprocess(name, src, defines).unwrap();

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });

        module
    }

    pub fn define(&mut self, name: &str, content: &str) {
        self.source_library.insert(name.into(), content.into());
    }
}

pub struct VertexBuilder {
    location: u32,
    offset: u64,
    attributes: Vec<wgpu::VertexAttribute>,
    step_mode: wgpu::VertexStepMode,
}

impl VertexBuilder {
    pub fn new(step_mode: wgpu::VertexStepMode) -> Self {
        VertexBuilder { location: 0, offset: 0, attributes: Vec::with_capacity(16), step_mode }
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
            shader_location: self.location
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

pub struct PipelineDefaults {
    target_states: [Option<wgpu::ColorTargetState>; 3]
}

impl PipelineDefaults {
    pub fn new() -> Self {
        PipelineDefaults {
            target_states: [
                Some(wgpu::ColorTargetState {
                    format: Self::color_format(),
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                Some(wgpu::ColorTargetState {
                    format: Self::color_format(),
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                Some(wgpu::ColorTargetState {
                    format: Self::mask_format(),
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
            ]
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

    pub fn color_target_state(&self) -> &[Option<wgpu::ColorTargetState>] {
        let idx = 1;
        &self.target_states[idx..idx+1]
    }

    pub fn color_target_state_no_blend(&self) -> &[Option<wgpu::ColorTargetState>] {
        let idx = 0;
        &self.target_states[idx..idx+1]
    }

    pub fn alpha_target_state(&self) -> &[Option<wgpu::ColorTargetState>] {
        let idx = 2;
        &self.target_states[idx..idx+1]
    }

    pub fn depth_format() -> wgpu::TextureFormat {
        wgpu::TextureFormat::Depth32Float
    }

    pub fn color_format() -> wgpu::TextureFormat {
        wgpu::TextureFormat::Bgra8UnormSrgb
    }

    pub fn mask_format() -> wgpu::TextureFormat {
        wgpu::TextureFormat::R8Unorm
    }

    pub fn msaa_format() -> wgpu::TextureFormat {
        Self::color_format()
    }

    pub fn msaa_sample_count() -> u32 {
        4
    }
}