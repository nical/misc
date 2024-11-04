use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt::Write;

use wgslp::preprocessor::{Preprocessor, Source, SourceError};

use super::VertexBuilder;
use crate::{gpu::PipelineDefaults, context::{SurfaceDrawConfig, StencilMode, DepthMode, SurfaceKind}};

#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ShaderPatternId(u16);
impl ShaderPatternId {
    pub fn from_index(idx: usize) -> Self {
        debug_assert!(idx < (u16::MAX - 1) as usize);
        ShaderPatternId(idx as u16)
    }
    pub fn index(self) -> usize {
        self.0 as usize
    }
    pub fn get(self) -> u16 {
        self.0
    }
    pub fn is_none(self) -> bool {
        self.0 == u16::MAX
    }
    pub const NONE: Self = ShaderPatternId(u16::MAX);
    fn map<Out>(self, cb: impl Fn(Self) -> Out) -> Option<Out> {
        if self.is_none() {
            return None;
        }

        Some(cb(self))
    }
}

#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct BaseShaderId(u16);
impl BaseShaderId {
    pub fn from_index(idx: usize) -> Self {
        debug_assert!(idx < (u16::MAX - 1) as usize);
        BaseShaderId(idx as u16)
    }
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

// Note: The pipeline layout key hash relies on BindGroupLayoutId using
// 16 bits.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct BindGroupLayoutId(pub(crate) u16);

impl BindGroupLayoutId {
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }

    #[inline]
    fn from_index(idx: usize) -> Self {
        BindGroupLayoutId(idx as u16)
    }
}

pub struct CommonBindGroupLayouts {
    pub target_and_gpu_store: BindGroupLayoutId,
    pub color_texture: BindGroupLayoutId,
    pub alpha_texture: BindGroupLayoutId,
}

fn init_common_layouts(layouts: &mut Vec<BindGroupLayout>, device: &wgpu::Device) -> CommonBindGroupLayouts {
    assert!(layouts.is_empty());

    let target_desc_buffer_size = std::mem::size_of::<crate::gpu::RenderPassDescriptor>() as u64;
    layouts.push(BindGroupLayout::new(
        device,
        "target and gpu store".into(),
        vec![
            Binding {
                name: "render_target".into(),
                struct_type: "RenderTarget".into(),
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(target_desc_buffer_size),
                },
            },
            Binding {
                name: "gpu_store_texture".into(),
                struct_type: "f32".into(),
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
            },
            Binding {
                name: "default_sampler".into(),
                struct_type: String::new(),
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            },
        ],
    ));

    layouts.push(BindGroupLayout::new(
        device,
        "color texture".into(),
        vec![Binding {
            name: "src_color_texture".into(),
            struct_type: "f32".into(),
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
        }],
    ));

    layouts.push(BindGroupLayout::new(
        device,
        "alpha texture".into(),
        vec![Binding {
            name: "src_alpha_texture".into(),
            struct_type: "f32".into(),
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
        }],
    ));

    CommonBindGroupLayouts {
        target_and_gpu_store: BindGroupLayoutId(0),
        color_texture: BindGroupLayoutId(1),
        alpha_texture: BindGroupLayoutId(2),
    }
}

pub struct Shaders {
    sources: ShaderSources,

    patterns: Vec<PatternDescriptor>,
    base_shaders: Vec<BaseShaderDescriptor>,
    module_cache: HashMap<ModuleKey, wgpu::ShaderModule>,
    bind_group_layouts: Vec<BindGroupLayout>,
    pipeline_layouts: HashMap<u64, wgpu::PipelineLayout>,
    // TODO: storing base bindings here and letting the common resources set is not great.
    // TODO: name clash with base shader's bindings. Rename into global bindings?
    // pub base_bindings: Option<BindGroupLayoutId>,
    pub defaults: PipelineDefaults,
    pub common_bind_group_layouts: CommonBindGroupLayouts,
}

impl Shaders {
    pub fn new(device: &wgpu::Device) -> Self {
        let mut bind_group_layouts = Vec::new();
        let common_bind_group_layouts = init_common_layouts(&mut bind_group_layouts, device);

        Shaders {
            sources: ShaderSources::new(),

            patterns: Vec::new(),
            base_shaders: Vec::new(),
            module_cache: HashMap::new(),
            bind_group_layouts,
            pipeline_layouts: HashMap::new(),
            defaults: PipelineDefaults::new(),
            common_bind_group_layouts,
        }
    }

    pub fn register_bind_group_layout(&mut self, bgl: BindGroupLayout) -> BindGroupLayoutId {
        let id = BindGroupLayoutId::from_index(self.bind_group_layouts.len());
        self.bind_group_layouts.push(bgl);

        id
    }

    pub fn get_base_bind_group_layout(&self) -> &BindGroupLayout {
        &self.bind_group_layouts[self.common_bind_group_layouts.target_and_gpu_store.index()]
    }

    pub fn get_bind_group_layout(&self, id: BindGroupLayoutId) -> &BindGroupLayout {
        &self.bind_group_layouts[id.index()]
    }

    pub fn find_bind_group_layout(&self, name: &str) -> Option<BindGroupLayoutId> {
        for (idx, item) in self.bind_group_layouts.iter().enumerate() {
            if item.name == name {
                return Some(BindGroupLayoutId::from_index(idx));
            }
        }

        None
    }

    pub fn register_library(&mut self, name: &str, source: Source) {
        self.sources.source_library.insert(name.into(), source);
    }

    pub fn register_pattern(&mut self, pattern: PatternDescriptor) -> ShaderPatternId {
        self.sources
            .source_library
            .insert(pattern.name.clone().into(), pattern.source.clone());
        let id = ShaderPatternId::from_index(self.patterns.len());
        self.patterns.push(pattern);

        id
    }

    pub fn register_base_shader(&mut self, base: BaseShaderDescriptor) -> BaseShaderId {
        self.sources
            .source_library
            .insert(base.name.clone().into(), base.source.clone());
        let id = BaseShaderId::from_index(self.base_shaders.len());
        self.base_shaders.push(base);

        id
    }

    pub fn find_base_shader(&self, name: &str) -> Option<BaseShaderId> {
        for (idx, pat) in self.base_shaders.iter().enumerate() {
            if &*pat.name == name {
                return Some(BaseShaderId::from_index(idx));
            }
        }

        None
    }

    pub fn find_pattern(&self, name: &str) -> ShaderPatternId {
        for (idx, pat) in self.patterns.iter().enumerate() {
            if &*pat.name == name {
                return ShaderPatternId::from_index(idx);
            }
        }

        ShaderPatternId::NONE
    }

    pub fn num_patterns(&self) -> usize {
        self.patterns.len()
    }

    pub fn create_shader_module(
        &mut self,
        device: &wgpu::Device,
        name: &str,
        src: &str,
        defines: &[&str],
    ) -> wgpu::ShaderModule {
        self.sources
            .create_shader_module(device, name, src, defines)
    }

    pub fn print_pipeline_variant(
        &mut self,
        pipeline_id: BaseShaderId,
        pattern_id: ShaderPatternId,
    ) {
        let base = &self.base_shaders[pipeline_id.index()];
        let pattern = pattern_id.map(|p| &self.patterns[p.index()]);

        let base_bindings = &self.bind_group_layouts[self.common_bind_group_layouts.target_and_gpu_store.index()];
        let geom_bindings = base
            .bindings
            .map(|id| &self.bind_group_layouts[id.index()]);
        let pattern_bindings = pattern
            .map(|desc| desc.bindings)
            .flatten()
            .map(|id| &self.bind_group_layouts[id.index()]);

        let src = generate_shader_source(
            base,
            pattern,
            base_bindings,
            geom_bindings,
            pattern_bindings,
        );

        self.sources.preprocessor.reset_defines();
        for define in &base.shader_defines {
            self.sources.preprocessor.define(define);
        }

        let src = self
            .sources
            .preprocessor
            .preprocess("generated module", &src, &mut self.sources.source_library)
            .unwrap();

        println!("{src}");
    }

    fn generate_pipeline_variant(
        &mut self,
        device: &wgpu::Device,
        pipeline_id: BaseShaderId,
        pattern_id: ShaderPatternId,
        blend_mode: BlendMode,
        surface: &SurfaceDrawConfig,
    ) -> wgpu::RenderPipeline {
        let base = &self.base_shaders[pipeline_id.index()];
        let pattern = pattern_id.map(|p| &self.patterns[p.index()]);

        let module_key = ModuleKey {
            base: pipeline_id,
            pattern: pattern_id,
            defines: base.shader_defines.clone(),
        };

        let module = self.module_cache.entry(module_key).or_insert_with(|| {
            let base_bindings = &self.bind_group_layouts[self.common_bind_group_layouts.target_and_gpu_store.index()];
            let geom_bindings = base
                .bindings
                .map(|id| &self.bind_group_layouts[id.index()]);
            let pattern_bindings = pattern
                .map(|desc| desc.bindings)
                .flatten()
                .map(|id| &self.bind_group_layouts[id.index()]);

            let src = generate_shader_source(
                base,
                pattern,
                base_bindings,
                geom_bindings,
                pattern_bindings,
            );

            //println!("--------------- pipeline {} mask {:?}. pattern {:?}", params.label, mask.map(|m| &m.name), pattern.map(|p| &p.name));
            //println!("{src}");
            //println!("----");
            self.sources.preprocessor.reset_defines();
            for define in &base.shader_defines {
                self.sources.preprocessor.define(define);
            }

            let src = self
                .sources
                .preprocessor
                .preprocess("generated module", &src, &mut self.sources.source_library)
                .unwrap();

            //println!("==================\n{src}\n=================");
            let label = format!("{}|{}", base.name, pattern.map(|p| &p.name[..]).unwrap_or(""));
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            })
        });

        // Increment the ids by 1 so that id zero isn't equal to any id.
        let mut key = 0u64;
        let mut shift = 0;
        if let Some(id) = base.bindings {
            key |= (id.0 as u64 + 1) << shift;
            shift += 16;
        }

        if let Some(pattern) = pattern {
            if let Some(id) = pattern.bindings {
                key |= (id.0 as u64 + 1) << shift;
            }
        }

        let layout = self.pipeline_layouts.entry(key).or_insert_with(|| {
            let mut layouts = Vec::new();
            let base_id = self.common_bind_group_layouts.target_and_gpu_store;
            layouts.push(&self.bind_group_layouts[base_id.index()].handle);

            if let Some(id) = base.bindings {
                layouts.push(&self.bind_group_layouts[id.index()].handle);
            }

            if let Some(pattern) = pattern {
                if let Some(id) = pattern.bindings {
                    layouts.push(&self.bind_group_layouts[id.index()].handle);
                }
            }

            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &layouts[..],
                push_constant_ranges: &[],
            })
        });

        let mut color_targets = [None, None, None];
        for (idx, kind) in surface.color_attachments().iter().enumerate() {
            let format = match kind {
                SurfaceKind::None => None,
                SurfaceKind::Color => Some(self.defaults.color_format()),
                SurfaceKind::Alpha => Some(self.defaults.mask_format()),
                SurfaceKind::HdrColor => Some(self.defaults.hdr_color_format()),
                SurfaceKind::HdrAlpha => Some(self.defaults.hdr_alpha_format()),
            };
            color_targets[idx] = format.map(|format| {
                wgpu::ColorTargetState {
                    format,
                    // TODO: using the blend mode on all attachments is probably
                    // not the right thing to do.
                    blend: blend_state(blend_mode),
                    write_mask: wgpu::ColorWrites::ALL,
                }
            });
        }
        let targets = &color_targets[0..surface.num_color_attachments()];

        let multisample = if surface.msaa {
            wgpu::MultisampleState {
                count: self.defaults.msaa_sample_count(),
                ..wgpu::MultisampleState::default()
            }
        } else {
            wgpu::MultisampleState::default()
        };

        let depth_stencil =
            if (surface.depth, surface.stencil) != (DepthMode::None, StencilMode::None) {
                let stencil = match surface.stencil {
                    StencilMode::None | StencilMode::Ignore => wgpu::StencilState::default(),
                    StencilMode::NonZero => {
                        let face_state = wgpu::StencilFaceState {
                            compare: wgpu::CompareFunction::NotEqual,
                            // reset the stencil buffer.
                            fail_op: wgpu::StencilOperation::Replace,
                            depth_fail_op: wgpu::StencilOperation::Replace,
                            pass_op: wgpu::StencilOperation::Replace,
                        };
                        wgpu::StencilState {
                            front: face_state,
                            back: face_state,
                            read_mask: 0xFFFFFFFF,
                            write_mask: 0xFFFFFFFF,
                        }
                    }
                    StencilMode::EvenOdd => {
                        let face_state = wgpu::StencilFaceState {
                            compare: wgpu::CompareFunction::NotEqual,
                            // reset the stencil buffer.
                            fail_op: wgpu::StencilOperation::Replace,
                            depth_fail_op: wgpu::StencilOperation::Replace,
                            pass_op: wgpu::StencilOperation::Replace,
                        };
                        wgpu::StencilState {
                            front: face_state,
                            back: face_state,
                            read_mask: 1,
                            write_mask: 0xFFFFFFFF,
                        }
                    }
                };
                let depth_write_enabled =
                    surface.depth == DepthMode::Enabled && blend_mode == BlendMode::None;
                let depth_compare = if surface.depth == DepthMode::Enabled {
                    wgpu::CompareFunction::Greater
                } else {
                    wgpu::CompareFunction::Always
                };

                Some(wgpu::DepthStencilState {
                    depth_write_enabled,
                    depth_compare,
                    stencil,
                    bias: wgpu::DepthBiasState::default(),
                    format: self.defaults.depth_stencil_format().unwrap(),
                })
            } else {
                None
            };

        let label = format!(
            "{}|{}{}{}{}",
            base.name,
            pattern.map(|p| &p.name[..]).unwrap_or(""),
            match surface.depth {
                DepthMode::Enabled => "|depth(enabled)",
                DepthMode::Ignore => "|depth(ignore)",
                DepthMode::None => "",
            },
            match surface.stencil {
                StencilMode::NonZero => "|stencil(nonzero)",
                StencilMode::EvenOdd => "|stencil(evenodd)",
                StencilMode::Ignore => "|stencil(ignore)",
                StencilMode::None => "",
            },
            if surface.msaa { "|msaa" } else { "" },
        );

        let mut vertices = VertexBuilder::new(wgpu::VertexStepMode::Vertex);
        let mut instances = VertexBuilder::new(wgpu::VertexStepMode::Instance);

        let base = &self.base_shaders[pipeline_id.index()];
        for vtx in &base.vertex_attributes {
            vertices.push(vtx.to_wgpu());
        }
        for inst in &base.instance_attributes {
            instances.push(inst.to_wgpu());
        }

        let mut vertex_buffers = Vec::new();
        if !base.vertex_attributes.is_empty() {
            vertex_buffers.push(vertices.buffer_layout());
        }
        if !base.instance_attributes.is_empty() {
            vertex_buffers.push(instances.buffer_layout());
        }

        let descriptor = wgpu::RenderPipelineDescriptor {
            label: Some(&label),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module,
                entry_point: Some("vs_main"),
                buffers: &vertex_buffers,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module,
                entry_point: Some("fs_main"),
                targets,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: base.primitive,
            depth_stencil,
            multiview: None,
            multisample,
            cache: None,
        };

        device.create_render_pipeline(&descriptor)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ShaderIndex(u32);

impl ShaderIndex {
    #[inline]
    pub fn index(&self) -> usize {
        self.0 as usize
    }
}

struct ShaderSources {
    pub source_library: HashMap<String, Source>,
    pub preprocessor: Preprocessor,
}

impl ShaderSources {
    pub fn new() -> Self {
        let mut library = HashMap::default();

        library.insert(
            "rect".into(),
            include_str!("../../shaders/lib/rect.wgsl").into(),
        );
        library.insert(
            "z_index".into(),
            include_str!("../../shaders/lib/z_index.wgsl").into(),
        );
        library.insert(
            "tiling".into(),
            include_str!("../../shaders/lib/tiling.wgsl").into(),
        );
        library.insert(
            "gpu_store".into(),
            include_str!("../../shaders/lib/gpu_store.wgsl").into(),
        );
        library.insert(
            "render_target".into(),
            include_str!("../../shaders/lib/render_target.wgsl").into(),
        );
        library.insert(
            "mask::circle".into(),
            include_str!("../../shaders/lib/mask/circle.wgsl").into(),
        );
        library.insert(
            "pattern::color".into(),
            include_str!("../../shaders/lib/pattern/color.wgsl").into(),
        );

        ShaderSources {
            source_library: library,
            preprocessor: Preprocessor::new(),
        }
    }

    pub fn preprocess(
        &mut self,
        name: &str,
        src: &str,
        defines: &[&str],
    ) -> Result<String, SourceError> {
        self.preprocessor.reset_defines();
        for define in defines {
            self.preprocessor.define(define);
        }
        self.preprocessor
            .preprocess(name, src, &mut self.source_library)
    }

    pub fn create_shader_module(
        &mut self,
        device: &wgpu::Device,
        name: &str,
        src: &str,
        defines: &[&str],
    ) -> wgpu::ShaderModule {
        let src = self.preprocess(name, src, defines).unwrap();

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });

        module
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum WgslType {
    Float32,
    Uint32,
    Sint32,
    Bool,
    Float32x2,
    Float32x3,
    Float32x4,
    Uint32x2,
    Uint32x3,
    Uint32x4,
    Sint32x2,
    Sint32x3,
    Sint32x4,
}

impl WgslType {
    fn as_str(self) -> &'static str {
        match self {
            WgslType::Float32 => "f32",
            WgslType::Uint32 => "u32",
            WgslType::Sint32 => "i32",
            WgslType::Bool => "bool",
            WgslType::Float32x2 => "vec2<f32>",
            WgslType::Float32x3 => "vec3<f32>",
            WgslType::Float32x4 => "vec4<f32>",
            WgslType::Uint32x2 => "vec2<u32>",
            WgslType::Uint32x3 => "vec3<u32>",
            WgslType::Uint32x4 => "vec4<u32>",
            WgslType::Sint32x2 => "vec2<i32>",
            WgslType::Sint32x3 => "vec3<i32>",
            WgslType::Sint32x4 => "vec4<i32>",
        }
    }
}

#[derive(Clone)]
pub struct Varying {
    pub name: Cow<'static, str>,
    pub kind: WgslType,
    pub interpolated: bool,
}

impl Varying {
    #[inline]
    pub fn float32(name: &'static str) -> Self {
        Varying {
            name: name.into(),
            kind: WgslType::Float32,
            interpolated: true,
        }
    }
    #[inline]
    pub fn float32x2(name: &'static str) -> Self {
        Varying {
            name: name.into(),
            kind: WgslType::Float32x2,
            interpolated: true,
        }
    }
    #[inline]
    pub fn float32x3(name: &'static str) -> Self {
        Varying {
            name: name.into(),
            kind: WgslType::Float32x3,
            interpolated: true,
        }
    }
    #[inline]
    pub fn float32x4(name: &'static str) -> Self {
        Varying {
            name: name.into(),
            kind: WgslType::Float32x4,
            interpolated: true,
        }
    }
    #[inline]
    pub fn uint32(name: &'static str) -> Self {
        Varying {
            name: name.into(),
            kind: WgslType::Uint32,
            interpolated: false,
        }
    }
    #[inline]
    pub fn uint32x2(name: &'static str) -> Self {
        Varying {
            name: name.into(),
            kind: WgslType::Uint32x2,
            interpolated: false,
        }
    }
    #[inline]
    pub fn uint32x3(name: &'static str) -> Self {
        Varying {
            name: name.into(),
            kind: WgslType::Uint32x3,
            interpolated: false,
        }
    }
    #[inline]
    pub fn uint32x4(name: &'static str) -> Self {
        Varying {
            name: name.into(),
            kind: WgslType::Uint32x4,
            interpolated: false,
        }
    }
    #[inline]
    pub fn sint32(name: &'static str) -> Self {
        Varying {
            name: name.into(),
            kind: WgslType::Sint32,
            interpolated: false,
        }
    }
    #[inline]
    pub fn sint32x2(name: &'static str) -> Self {
        Varying {
            name: name.into(),
            kind: WgslType::Sint32x2,
            interpolated: false,
        }
    }
    #[inline]
    pub fn sint32x3(name: &'static str) -> Self {
        Varying {
            name: name.into(),
            kind: WgslType::Sint32x3,
            interpolated: false,
        }
    }
    #[inline]
    pub fn sint32x4(name: &'static str) -> Self {
        Varying {
            name: name.into(),
            kind: WgslType::Sint32x4,
            interpolated: false,
        }
    }
    #[inline]
    pub fn with_interpolation(mut self, interpolated: bool) -> Self {
        self.interpolated = interpolated;
        self
    }
    #[inline]
    pub fn interpolated(self) -> Self {
        self.with_interpolation(true)
    }
    #[inline]
    pub fn flat(self) -> Self {
        self.with_interpolation(false)
    }
}

#[derive(Clone)]
pub struct VertexAtribute {
    pub name: Cow<'static, str>,
    pub kind: WgslType,
}

impl VertexAtribute {
    #[inline]
    pub fn float32(name: &'static str) -> Self {
        VertexAtribute {
            name: name.into(),
            kind: WgslType::Float32,
        }
    }
    #[inline]
    pub fn float32x2(name: &'static str) -> Self {
        VertexAtribute {
            name: name.into(),
            kind: WgslType::Float32x2,
        }
    }
    #[inline]
    pub fn float32x3(name: &'static str) -> Self {
        VertexAtribute {
            name: name.into(),
            kind: WgslType::Float32x3,
        }
    }
    #[inline]
    pub fn float32x4(name: &'static str) -> Self {
        VertexAtribute {
            name: name.into(),
            kind: WgslType::Float32x4,
        }
    }
    #[inline]
    pub fn uint32(name: &'static str) -> Self {
        VertexAtribute {
            name: name.into(),
            kind: WgslType::Uint32,
        }
    }
    #[inline]
    pub fn uint32x2(name: &'static str) -> Self {
        VertexAtribute {
            name: name.into(),
            kind: WgslType::Uint32x2,
        }
    }
    #[inline]
    pub fn uint32x3(name: &'static str) -> Self {
        VertexAtribute {
            name: name.into(),
            kind: WgslType::Uint32x3,
        }
    }
    #[inline]
    pub fn uint32x4(name: &'static str) -> Self {
        VertexAtribute {
            name: name.into(),
            kind: WgslType::Uint32x4,
        }
    }
    #[inline]
    pub fn int32(name: &'static str) -> Self {
        VertexAtribute {
            name: name.into(),
            kind: WgslType::Sint32,
        }
    }
    #[inline]
    pub fn int32x2(name: &'static str) -> Self {
        VertexAtribute {
            name: name.into(),
            kind: WgslType::Sint32x2,
        }
    }
    #[inline]
    pub fn int32x3(name: &'static str) -> Self {
        VertexAtribute {
            name: name.into(),
            kind: WgslType::Sint32x3,
        }
    }
    #[inline]
    pub fn int32x4(name: &'static str) -> Self {
        VertexAtribute {
            name: name.into(),
            kind: WgslType::Sint32x4,
        }
    }

    pub fn to_wgpu(&self) -> wgpu::VertexFormat {
        match self.kind {
            WgslType::Float32 => wgpu::VertexFormat::Float32,
            WgslType::Sint32 => wgpu::VertexFormat::Sint32,
            WgslType::Uint32 => wgpu::VertexFormat::Uint32,
            WgslType::Float32x2 => wgpu::VertexFormat::Float32x2,
            WgslType::Sint32x2 => wgpu::VertexFormat::Sint32x2,
            WgslType::Uint32x2 => wgpu::VertexFormat::Uint32x2,
            WgslType::Float32x3 => wgpu::VertexFormat::Float32x3,
            WgslType::Sint32x3 => wgpu::VertexFormat::Sint32x3,
            WgslType::Uint32x3 => wgpu::VertexFormat::Uint32x3,
            WgslType::Float32x4 => wgpu::VertexFormat::Float32x4,
            WgslType::Sint32x4 => wgpu::VertexFormat::Sint32x4,
            WgslType::Uint32x4 => wgpu::VertexFormat::Uint32x4,
            _ => unimplemented!(),
        }
    }
}

pub struct BindGroupLayout {
    pub name: String,
    pub handle: wgpu::BindGroupLayout,
    pub entries: Vec<Binding>,
}

impl BindGroupLayout {
    pub fn new(device: &wgpu::Device, name: String, entries: Vec<Binding>) -> Self {
        let mut bg = Vec::with_capacity(entries.len());
        for (idx, entry) in entries.iter().enumerate() {
            bg.push(wgpu::BindGroupLayoutEntry {
                binding: idx as u32,
                ty: entry.ty,
                visibility: entry.visibility,
                count: None,
            });
        }

        let handle = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &bg,
            label: Some(name.as_str()),
        });

        BindGroupLayout {
            name,
            handle,
            entries,
        }
    }
}

pub struct Binding {
    pub name: String,
    pub struct_type: String,
    pub ty: wgpu::BindingType,
    pub visibility: wgpu::ShaderStages,
}

impl Binding {
    pub fn uniform_buffer(name: &str, struct_type: &str) -> Self {
        Binding {
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            // TODO: min binding size.
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            name: name.into(),
            struct_type: struct_type.into(),
        }
    }

    pub fn storage_buffer(name: &str, struct_type: &str) -> Self {
        Binding {
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            // TODO: min binding size.
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            name: name.into(),
            struct_type: struct_type.into(),
        }
    }
}

impl BindGroupLayout {
    fn generate_shader_source(&self, index: u32, source: &mut String) {
        let mut binding = 0;
        for entry in &self.entries {
            write!(source, "@group({index}) @binding({binding}) ").unwrap();
            let name = &entry.name;
            let struct_ty = &entry.struct_type;
            match entry.ty {
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    ..
                } => {
                    writeln!(source, "var<uniform> {name}: {struct_ty};").unwrap();
                }
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { .. },
                    ..
                } => {
                    writeln!(source, "var<storage> {name}: {struct_ty};").unwrap();
                }
                wgpu::BindingType::Sampler(..) => {
                    writeln!(source, "var {name}: sampler;").unwrap();
                }
                wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    ..
                } => {
                    writeln!(source, "var {name}: texture_depth_2d;").unwrap();
                }
                wgpu::BindingType::Texture {
                    sample_type,
                    view_dimension,
                    ..
                } => {
                    let dim = match view_dimension {
                        wgpu::TextureViewDimension::D1 => "1",
                        wgpu::TextureViewDimension::D2 => "2",
                        wgpu::TextureViewDimension::D3 => "3",
                        _ => unimplemented!(),
                    };
                    let sample_type = match sample_type {
                        wgpu::TextureSampleType::Float { .. } => "f32",
                        wgpu::TextureSampleType::Sint { .. } => "i32",
                        wgpu::TextureSampleType::Uint { .. } => "u32",
                        _ => "error",
                    };
                    writeln!(source, "var {name}: texture_{dim}d<{sample_type}>;").unwrap();
                }
                wgpu::BindingType::StorageTexture { .. } => {
                    todo!();
                }
                wgpu::BindingType::AccelerationStructure => {
                    unimplemented!();
                }
            }

            binding += 1;
        }
    }
}

pub struct PatternDescriptor {
    pub name: Cow<'static, str>,
    pub source: Source,
    pub varyings: Vec<Varying>,
    pub bindings: Option<BindGroupLayoutId>,
}

#[derive(Clone)]
pub struct BaseShaderDescriptor {
    pub name: Cow<'static, str>,
    pub primitive: wgpu::PrimitiveState,
    pub source: Source,
    pub vertex_attributes: Vec<VertexAtribute>,
    pub instance_attributes: Vec<VertexAtribute>,
    pub varyings: Vec<Varying>,
    pub bindings: Option<BindGroupLayoutId>,
    pub shader_defines: Vec<&'static str>,
}

pub fn generate_shader_source(
    base: &BaseShaderDescriptor,
    pattern: Option<&PatternDescriptor>,
    _base_bindings: &BindGroupLayout,
    geom_bindings: Option<&BindGroupLayout>,
    pattern_bindings: Option<&BindGroupLayout>,
) -> String {
    let mut source = String::new();

    let mut group_index = 0;

    // TODO right now the base bindings are in the imported sources. Generate them
    // once the shaders have moved to this system.
    //base_bindings.generate_shader_source(group_index, &mut source);
    writeln!(
        source,
        "@group(0) @binding(2) var default_sampler: sampler;"
    )
    .unwrap();
    group_index += 1;

    if let Some(bindings) = geom_bindings {
        bindings.generate_shader_source(group_index, &mut source);
        group_index += 1;
    }
    if let Some(bindings) = pattern_bindings {
        bindings.generate_shader_source(group_index, &mut source);
    }

    writeln!(source, "struct BaseVertex {{").unwrap();
    writeln!(source, "    position: vec4<f32>,").unwrap();
    writeln!(source, "    pattern_position: vec2<f32>,").unwrap();
    writeln!(source, "    pattern_data: u32,").unwrap();
    for varying in &base.varyings {
        //println!("=====    {}: {},", varying.name, varying.kind.as_str());
        writeln!(source, "    {}: {},", varying.name, varying.kind.as_str()).unwrap();
    }
    writeln!(source, "}}").unwrap();

    writeln!(source, "#import {}", base.name).unwrap();
    writeln!(source, "").unwrap();

    if let Some(pattern) = pattern {
        writeln!(source, "struct Pattern {{").unwrap();
        for varying in &pattern.varyings {
            writeln!(source, "    {}: {},", varying.name, varying.kind.as_str()).unwrap();
        }
        writeln!(source, "}}").unwrap();
        writeln!(source, "#import {}", pattern.name).unwrap();
    }

    writeln!(source, "").unwrap();

    // vertex

    writeln!(source, "struct VertexOutput {{").unwrap();
    writeln!(source, "    @builtin(position) position: vec4<f32>,").unwrap();
    let mut idx = 0;
    for varying in &base.varyings {
        let interpolate = if varying.interpolated {
            "perspective"
        } else {
            "flat"
        };
        writeln!(
            source,
            "    @location({idx}) @interpolate({interpolate}) geom_{}: {},",
            varying.name,
            varying.kind.as_str()
        )
        .unwrap();
        idx += 1;
    }
    if let Some(pattern) = pattern {
        for varying in &pattern.varyings {
            let interpolate = if varying.interpolated {
                "perspective"
            } else {
                "flat"
            };
            writeln!(
                source,
                "    @location({idx}) @interpolate({interpolate}) pat_{}: {},",
                varying.name,
                varying.kind.as_str()
            )
            .unwrap();
            idx += 1;
        }
    }
    writeln!(source, "}}").unwrap();
    writeln!(source, "").unwrap();

    writeln!(source, "@vertex fn vs_main(").unwrap();
    writeln!(source, "    @builtin(vertex_index) vertex_index: u32,").unwrap();
    let mut attr_location = 0;
    for attrib in &base.vertex_attributes {
        writeln!(
            source,
            "    @location({attr_location}) vtx_{}: {},",
            attrib.name,
            attrib.kind.as_str()
        )
        .unwrap();
        attr_location += 1;
    }
    for attrib in &base.instance_attributes {
        writeln!(
            source,
            "    @location({attr_location}) inst_{}: {},",
            attrib.name,
            attrib.kind.as_str()
        )
        .unwrap();
        attr_location += 1;
    }
    writeln!(source, ") -> VertexOutput {{").unwrap();

    writeln!(source, "    var vertex = base_vertex(").unwrap();
    writeln!(source, "        vertex_index,").unwrap();
    for attrib in &base.vertex_attributes {
        writeln!(source, "        vtx_{},", attrib.name).unwrap();
    }
    for attrib in &base.instance_attributes {
        writeln!(source, "        inst_{},", attrib.name).unwrap();
    }
    writeln!(source, "    );").unwrap();

    if pattern.is_some() {
        writeln!(
            source,
            "    var pattern = pattern_vertex(vertex.pattern_position, vertex.pattern_data);"
        )
        .unwrap();
    }

    writeln!(source, "    return VertexOutput(").unwrap();
    writeln!(source, "        vertex.position,").unwrap();
    for varying in &base.varyings {
        writeln!(source, "        vertex.{},", varying.name).unwrap();
    }
    if let Some(pattern) = pattern {
        for varying in &pattern.varyings {
            writeln!(source, "        pattern.{},", varying.name).unwrap();
        }
    }
    writeln!(source, "    );").unwrap();
    writeln!(source, "}}").unwrap();
    writeln!(source, "").unwrap();

    // fragment

    writeln!(source, "@fragment fn fs_main(").unwrap();
    let mut idx = 0;
    for varying in &base.varyings {
        let interpolate = if varying.interpolated {
            "perspective"
        } else {
            "flat"
        };
        writeln!(
            source,
            "    @location({idx}) @interpolate({interpolate}) geom_{}: {},",
            varying.name,
            varying.kind.as_str()
        )
        .unwrap();
        idx += 1;
    }
    if let Some(pattern) = pattern {
        for varying in &pattern.varyings {
            let interpolate = if varying.interpolated {
                "perspective"
            } else {
                "flat"
            };
            writeln!(
                source,
                "    @location({idx}) @interpolate({interpolate}) pat_{}: {},",
                varying.name,
                varying.kind.as_str()
            )
            .unwrap();
            idx += 1;
        }
    }
    writeln!(source, ") -> @location(0) vec4<f32> {{").unwrap();
    writeln!(source, "    var color = vec4<f32>(1.0);").unwrap();

    writeln!(source, "    color.a *= base_fragment(").unwrap();
    for varying in &base.varyings {
        writeln!(source, "        geom_{},", varying.name).unwrap();
    }
    writeln!(source, "    );").unwrap();

    if let Some(pattern) = pattern {
        writeln!(source, "    color *= pattern_fragment(Pattern(").unwrap();
        for varying in &pattern.varyings {
            writeln!(source, "        pat_{},", varying.name).unwrap();
        }
        writeln!(source, "    ));").unwrap();
    }

    writeln!(source, "    // Premultiply").unwrap();
    writeln!(source, "    color.r *= color.a;").unwrap();
    writeln!(source, "    color.g *= color.a;").unwrap();
    writeln!(source, "    color.b *= color.a;").unwrap();

    writeln!(source, "    return color;").unwrap();

    writeln!(source, "}}").unwrap();

    source
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ModuleKey {
    pub base: BaseShaderId,
    pub pattern: ShaderPatternId,
    pub defines: Vec<&'static str>,
}

// If this grows to use up more than 4 bits, BatchKey must be adjusted.
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BlendMode {
    /// Overwrite the destination.
    None,
    /// The default blend mode for alpha blending.
    PremultipliedAlpha,
    Add,
    Subtract,
    Multiply,
    Screen,
    Lighter,
    Exclusion,
    /// Discards pixels of the destination where the source alpha is zero.
    ClipIn,
    /// Discards pixels of the destination where the source alpha is one.
    ClipOut,
}

impl BlendMode {
    pub fn with_alpha(self, alpha: bool) -> Self {
        match (self, alpha) {
            (BlendMode::None, true) => BlendMode::PremultipliedAlpha,
            (BlendMode::PremultipliedAlpha, false) => BlendMode::None,
            (mode, _) => mode,
        }
    }
}

fn blend_state(blend_mode: BlendMode) -> Option<wgpu::BlendState> {
    match blend_mode {
        BlendMode::None => None,
        BlendMode::PremultipliedAlpha => {
            Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING)
        }
        BlendMode::Add => {
            Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::SrcAlpha,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::Zero,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                }
            })
        }
        BlendMode::Subtract => {
            Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::SrcAlpha,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Subtract,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::Zero,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                }
            })
        }
        BlendMode::Multiply => {
            Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::Zero,
                    dst_factor: wgpu::BlendFactor::Src,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::SrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                }
            })
        }
        BlendMode::Screen => {
            Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::OneMinusSrc,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                }
            })
        }
        BlendMode::Lighter => {
            Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::SrcAlpha,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::Zero,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                }
            })
        }
        BlendMode::Exclusion => {
            Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::OneMinusDst,
                    dst_factor: wgpu::BlendFactor::OneMinusSrc,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                }
            })
        }
        BlendMode::ClipIn => {
            Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::Zero,
                    dst_factor: wgpu::BlendFactor::SrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Min,
                }
            })
        }
        BlendMode::ClipOut => {
            Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::Zero,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Min,
                }
            })
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct RenderPipelineKey(u64);
pub type RenderPipelineIndex = crate::cache::Index<RenderPipelineKey>;
pub type RenderPipelines = crate::cache::Registry<RenderPipelineKey, wgpu::RenderPipeline>;
pub type PrepareRenderPipelines = crate::cache::Prepare<RenderPipelineKey>;
pub struct RenderPipelineBuilder<'l>(pub &'l wgpu::Device, pub &'l mut Shaders);

impl RenderPipelineKey {
    pub fn new(
        base: BaseShaderId,
        pattern: ShaderPatternId,
        blend_mode: BlendMode,
        surf: SurfaceDrawConfig,
    ) -> Self {
        Self(
            surf.hash() as u64
            | ((base.0 as u64) << 16)
            | ((pattern.get() as u64) << 32)
            | ((blend_mode as u64) << 48)
        )
    }

    pub fn unpack(&self) -> (BaseShaderId, ShaderPatternId, BlendMode, SurfaceDrawConfig) {
        let base = BaseShaderId((self.0 >> 16) as u16);
        let pattern = ShaderPatternId((self.0 >> 32) as u16);
        let surf = SurfaceDrawConfig::from_hash(self.0 as u16);
        let blend: BlendMode = unsafe { std::mem::transmute((self.0 >> 48) as u8) };
        (base, pattern, blend, surf)
    }
}

impl<'l> crate::cache::Build<RenderPipelineKey, wgpu::RenderPipeline>
    for RenderPipelineBuilder<'l>
{
    fn build(&mut self, key: RenderPipelineKey) -> wgpu::RenderPipeline {
        let (pipeline, pattern, blend, surface) = key.unpack();
        self.1
            .generate_pipeline_variant(self.0, pipeline, pattern, blend, &surface)
    }

    fn finish(&mut self) {}
}
