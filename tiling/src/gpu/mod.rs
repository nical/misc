pub mod atlas_uploader;
pub mod gpu_store;
pub mod storage_buffer;

pub use gpu_store::*;

pub use wgslp::preprocessor::{Preprocessor, Source, SourceError};
use std::{collections::HashMap};

use crate::stencil;

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

pub struct PipelineSet {
    pipelines: [Option<wgpu::RenderPipeline>; 8],
    params: Box<PipelineParams>,
}

pub struct PipelineKey {
    pub msaa: bool,
    pub depth: bool,
    pub stencil: bool,
}

impl PipelineKey {
    fn idx(&self) -> usize {
        return if self.msaa { 1 } else { 0 }
            + if self.depth { 2 } else { 0 }
            + if self.stencil { 4 } else { 0 }
    }
}

pub struct PipelineParams {
    label: &'static str,
    shader_module: wgpu::ShaderModule,
    pipeline_layout: wgpu::PipelineLayout,
    vertex_layout: VertexBuilder,
    output_format: wgpu::TextureFormat,
    alpha_blend: bool,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum StencilMode {
    EvenOdd,
    NonZero,
    None,
}

impl PipelineSet {
    pub fn new(
        label: &'static str,
        layout_descriptor: &wgpu::PipelineLayoutDescriptor,
        shader_module: wgpu::ShaderModule,
        vertex_layout: VertexBuilder,
        output_format: wgpu::TextureFormat,
        device: &wgpu::Device,
        alpha_blend: bool,
    ) -> Self {
        let pipeline_layout = device.create_pipeline_layout(layout_descriptor);

        PipelineSet {
            params: Box::new(PipelineParams {
                label,
                shader_module,
                pipeline_layout,
                vertex_layout,
                output_format,
                alpha_blend,
            }),
            pipelines: [None, None, None, None, None, None, None, None],
        }
    }

    pub fn build(&mut self, device: &wgpu::Device, key: &PipelineKey) {
        if self.pipelines[key.idx()].is_some() {
            return;
        }

        let defaults = PipelineDefaults::new();

        let targets = if self.params.alpha_blend {
            defaults.color_target_state()
        } else {
            defaults.color_target_state_no_blend()
        };

        let multisample = if key.msaa {
            wgpu::MultisampleState {
                count: PipelineDefaults::msaa_sample_count(),
                .. wgpu::MultisampleState::default()
            }
        } else {
            wgpu::MultisampleState::default()
        };

        let depth_stencil = if key.depth || key.stencil {
            Some(wgpu::DepthStencilState {
                depth_write_enabled: !self.params.alpha_blend,
                depth_compare: wgpu::CompareFunction::GreaterEqual,
                stencil: if key.stencil {
                    let face_state = wgpu::StencilFaceState {
                        compare: wgpu::CompareFunction::NotEqual,
                        fail_op: wgpu::StencilOperation::Keep,
                        depth_fail_op: wgpu::StencilOperation::Keep,
                        pass_op: wgpu::StencilOperation::Keep,
                    };
                    wgpu::StencilState {
                        front: face_state,
                        back: face_state,
                        read_mask: !1, // even-odd
                        write_mask: 0xFFFFFFFF,
                    }
                } else {
                    wgpu::StencilState::default()
                },
                bias: wgpu::DepthBiasState::default(),
                format: PipelineDefaults::depth_format(),
            })
        } else {
            None
        };

        let label = format!("{}{}{}{})",
            self.params.label,
            if key.msaa { "|msaa" } else { "" },
            if key.depth { "|depth" } else { "" },
            if key.stencil { "|stencil" } else { "" }
        );

        let descriptor = wgpu::RenderPipelineDescriptor {
            label: Some(&label),
            layout: Some(&self.params.pipeline_layout),
            vertex: wgpu::VertexState {
                module: &self.params.shader_module,
                entry_point: "vs_main",
                buffers: &[self.params.vertex_layout.buffer_layout()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &self.params.shader_module,
                entry_point: "fs_main",
                targets,
            }),
            primitive: PipelineDefaults::primitive_state(),
            depth_stencil,
            multiview: None,
            multisample,
        };

        let pipeline = device.create_render_pipeline(&descriptor);

        self.pipelines[key.idx()] = Some(pipeline);
    }

    pub fn try_get(&self, key: &PipelineKey) -> Option<&wgpu::RenderPipeline> {
        self.pipelines[key.idx()].as_ref()
    }

    pub fn get(&mut self, device: &wgpu::Device, key: &PipelineKey) -> &wgpu::RenderPipeline {
        self.build(device, key);
        self.try_get(key).unwrap()
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

    pub fn primitive_state() -> wgpu::PrimitiveState {
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
        wgpu::TextureFormat::Depth24PlusStencil8
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