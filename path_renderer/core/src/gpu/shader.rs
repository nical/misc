use std::{borrow::Cow};
use std::collections::HashMap;
use std::fmt::Write;

use lyon::path::FillRule;
use wgslp::preprocessor::{SourceError, Preprocessor, Source};

use super::VertexBuilder;
use crate::gpu::PipelineDefaults;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ShaderPatternId(u16);
impl ShaderPatternId {
    pub fn from_index(idx: usize) -> Self {
        debug_assert!(idx < (u16::MAX - 1) as usize);
        ShaderPatternId(idx as u16)
    }
    pub fn index(self) -> usize { self.0 as usize }
    pub fn get(self) -> u16 { self.0 }
    pub fn is_none(self) -> bool { self.0 == u16::MAX }
    pub const NONE: Self = ShaderPatternId(u16::MAX);
    fn map<Out>(self, cb: impl Fn(Self)->Out) -> Option<Out> {
        if self.is_none() {
            return None;
        }

        Some(cb(self))
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ShaderMaskId(u16);
impl ShaderMaskId {
    pub fn from_index(idx: usize) -> Self {
        debug_assert!(idx < (u16::MAX - 1) as usize);
        ShaderMaskId(idx as u16)
    }
    pub fn index(self) -> usize { self.0 as usize }
    pub fn get(self) -> u16 { self.0 }
    pub fn is_none(self) -> bool { self.0 == u16::MAX }
    pub const NONE: Self = ShaderMaskId(u16::MAX);
    fn map<Out>(self, cb: impl Fn(Self)->Out) -> Option<Out> {
        if self.is_none() {
            return None;
        }

        Some(cb(self))
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ShaderGeometryId(u16);
impl ShaderGeometryId {
    pub fn from_index(idx: usize) -> Self {
        debug_assert!(idx < (u16::MAX - 1) as usize);
        ShaderGeometryId(idx as u16)
    }
    pub fn index(self) -> usize { self.0 as usize }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct GeneratedPipelineId(u16);
impl GeneratedPipelineId {
    fn from_index(idx: usize) -> Self {
        debug_assert!(idx < (u16::MAX - 1) as usize);
        GeneratedPipelineId(idx as u16)
    }
    fn index(self) -> usize { self.0 as usize }
}

pub type BindGroupLayoutId = u16;

pub struct Shaders {
    sources: ShaderSources,

    pipelines: HashMap<u64, wgpu::RenderPipeline>,
    params: Vec<PipelineDescriptor>,

    patterns: Vec<PatternDescriptor>,
    masks: Vec<MaskDescriptor>,
    geometries: Vec<GeometryDescriptor>,
    module_cache: HashMap<ModuleKey, wgpu::ShaderModule>,
    bind_group_layouts: Vec<BindGroupLayout>,
    pipeline_layouts: HashMap<u64, wgpu::PipelineLayout>,
    // TODO: storing base bindings here and letting the common resources set is not great.
    pub base_bindings: Option<BindGroupLayoutId>,
    pub defaults: PipelineDefaults,
}

impl Shaders {
    pub fn new() -> Self {
        Shaders {
            sources: ShaderSources::new(),

            pipelines: HashMap::with_capacity(256),
            params: Vec::with_capacity(128),
            patterns: Vec::new(),
            masks: Vec::new(),
            geometries: Vec::new(),
            module_cache: HashMap::new(),
            bind_group_layouts: Vec::new(),
            pipeline_layouts: HashMap::new(),
            // TODO: this is hacky: base bindings is initialized by the common gpu resources.
            base_bindings: None,
            defaults: PipelineDefaults::new(),
        }
    }

    pub fn register_bind_group_layout(&mut self, bgl: BindGroupLayout) -> BindGroupLayoutId {
        let id = self.bind_group_layouts.len() as BindGroupLayoutId;
        self.bind_group_layouts.push(bgl);

        id
    }

    pub fn get_base_bind_group_layout(&self) -> &BindGroupLayout {
        &self.bind_group_layouts[self.base_bindings.unwrap() as usize]
    }

    pub fn get_bind_group_layout(&self, id: BindGroupLayoutId) -> &BindGroupLayout {
        &self.bind_group_layouts[id as usize]
    }

    pub fn find_bind_group_layout(&self, name: &str) -> Option<BindGroupLayoutId> {
        for (idx, item) in self.bind_group_layouts.iter().enumerate() {
            if item.name == name {
                return Some(idx as BindGroupLayoutId);
            }
        }

        None
    }

    pub fn register_library(&mut self, name: &str, source: Source) {
        self.sources.source_library.insert(name.into(), source);
    }

    pub fn register_pattern(&mut self, pattern: PatternDescriptor) -> ShaderPatternId {
        self.sources.source_library.insert(pattern.name.clone().into(), pattern.source.clone());
        let id = ShaderPatternId::from_index(self.patterns.len());
        self.patterns.push(pattern);

        id
    }

    pub fn register_mask(&mut self, mask: MaskDescriptor) -> ShaderMaskId {
        self.sources.source_library.insert(mask.name.clone().into(), mask.source.clone());
        let id = ShaderMaskId::from_index(self.masks.len());
        self.masks.push(mask);

        id
    }

    pub fn register_geometry(&mut self, geometry: GeometryDescriptor) -> ShaderGeometryId {
        self.sources.source_library.insert(geometry.name.clone().into(), geometry.source.clone());
        let id = ShaderGeometryId::from_index(self.geometries.len());
        self.geometries.push(geometry);

        id
    }

    pub fn find_geometry(&self, name: &str) -> Option<ShaderGeometryId> {
        for (idx, pat) in self.geometries.iter().enumerate() {
            if &*pat.name == name {
                return Some(ShaderGeometryId::from_index(idx));
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

    pub fn find_mask(&self, name: &str) -> ShaderMaskId {
        for (idx, mask) in self.masks.iter().enumerate() {
            if &*mask.name == name {
                return ShaderMaskId::from_index(idx);
            }
        }

        ShaderMaskId::NONE
    }

    pub fn num_patterns(&self) -> usize {
        self.patterns.len()
    }

    pub fn set_base_bindings(&mut self, id: BindGroupLayoutId) {
        self.base_bindings = Some(id);
    }

    pub fn create_shader_module(
        &mut self,
        device: &wgpu::Device,
        name: &str,
        src: &str,
        defines: &[&str]
    ) -> wgpu::ShaderModule {
        self.sources.create_shader_module(device, name, src, defines)
    }

    pub fn register_pipeline(&mut self, params: PipelineDescriptor) -> GeneratedPipelineId {
        let id = GeneratedPipelineId::from_index(self.params.len());
        self.params.push(params);

        id
    }

    pub fn prepare_pipeline(
        &mut self,
        device: &wgpu::Device,
        pipeline_id: GeneratedPipelineId,
        pattern: ShaderPatternId,
        surface: SurfaceConfig,
    ) {
        let key = Self::key(pipeline_id, pattern, surface);

        if self.pipelines.contains_key(&key) {
            return;
        }

        let pipeline = self.generate_pipeline_variant(device, pipeline_id, pattern, &surface);
        self.pipelines.insert(key, pipeline);
    }

    fn generate_pipeline_variant(
        &mut self,
        device: &wgpu::Device,
        pipeline_id: GeneratedPipelineId,
        pattern_id: ShaderPatternId,
        surface: &SurfaceConfig,
    ) -> wgpu::RenderPipeline {

        let params = &self.params[pipeline_id.index()];

        let geometry = &self.geometries[params.geometry.index()];
        let mask = params.mask.map(|m| &self.masks[m.index()]);
        let pattern = pattern_id.map(|p| &self.patterns[p.index()]);

        let module_key = ModuleKey {
            geometry: params.geometry,
            mask: params.mask,
            pattern: pattern_id,
            user_flags: params.user_flags,
            defines: params.shader_defines.clone(),
        };

        let module = self.module_cache.entry(module_key).or_insert_with(|| {
            let base_bindings = self.base_bindings.map(|id| &self.bind_group_layouts[id as usize]);
            let geom_bindings = geometry.bindings.map(|id| &self.bind_group_layouts[id as usize]);
            let mask_bindings = mask.map(|desc| desc.bindings).flatten().map(|id| &self.bind_group_layouts[id as usize]);
            let pattern_bindings = pattern.map(|desc| desc.bindings).flatten().map(|id| &self.bind_group_layouts[id as usize]);

            let src = generate_shader_source(geometry, mask, pattern, base_bindings, geom_bindings, mask_bindings, pattern_bindings);

            //println!("--------------- pipeline {} mask {:?}. pattern {:?}", params.label, mask.map(|m| &m.name), pattern.map(|p| &p.name));
            //println!("{src}");
            //println!("----");
            self.sources.preprocessor.reset_defines();
            for define in &params.shader_defines {
                self.sources.preprocessor.define(define);
            }

            let src = self.sources.preprocessor.preprocess("generated module", &src, &mut self.sources.source_library).unwrap();

            //println!("==================\n{src}\n=================");

            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&geometry.name),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            })
        });

        // Increment the ids by 1 so that id zero isn't equal to no id.
        let mut key = 0u64;
        let mut shift = 0;
        if let Some(id) = geometry.bindings {
            key |= (id as u64 + 1) << shift;
            shift += 16;
        }

        if let Some(mask) = mask {
            if let Some(id) = mask.bindings {
                key |= (id as u64 + 1) << shift;
                shift += 16;
            }
        }

        if let Some(pattern) = pattern {
            if let Some(id) = pattern.bindings {
                key |= (id as u64 + 1) << shift;
            }
        }


        let layout = self.pipeline_layouts.entry(key).or_insert_with(|| {
            let mut layouts = Vec::new();
            layouts.push(&self.bind_group_layouts[self.base_bindings.unwrap() as usize].handle);

            if let Some(id) = geometry.bindings {
                layouts.push(&self.bind_group_layouts[id as usize].handle);
            }

            if let Some(mask) = mask {
                if let Some(id) = mask.bindings {
                    layouts.push(&self.bind_group_layouts[id as usize].handle);
                }
            }

            if let Some(pattern) = pattern {
                if let Some(id) = pattern.bindings {
                    layouts.push(&self.bind_group_layouts[id as usize].handle);
                }
            }

            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &layouts[..],
                push_constant_ranges: &[],
            })
        });

        let color_target = &[Some(wgpu::ColorTargetState {
            format: match params.output {
                OutputType::Color => self.defaults.color_format(),
                OutputType::Alpha => self.defaults.mask_format(),
            },
            blend: match params.blend {
                BlendMode::None => None,
                BlendMode::PremultipliedAlpha => Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                _ => todo!(),
            },
            write_mask: wgpu::ColorWrites::ALL,
        })];

        let multisample = if surface.msaa {
            wgpu::MultisampleState {
                count: self.defaults.msaa_sample_count(),
                .. wgpu::MultisampleState::default()
            }
        } else {
            wgpu::MultisampleState::default()
        };

        let depth_stencil = if (surface.depth, surface.stencil) != (DepthMode::None, StencilMode::None) {
            let stencil = match surface.stencil {
                StencilMode::None | StencilMode::Ignore => wgpu::StencilState::default(),
                StencilMode::NonZero => {
                    let face_state = wgpu::StencilFaceState {
                        compare: wgpu::CompareFunction::Equal,
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
            let depth_write_enabled = surface.depth == DepthMode::Enabled && params.blend == BlendMode::None;
            let depth_compare = if surface.depth == DepthMode::Enabled { wgpu::CompareFunction::Greater } else { wgpu::CompareFunction::Always };

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

        let label = format!("{}|{}{}{}{}",
            params.label,
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

        let geom = &self.geometries[params.geometry.index()];
        for vtx in &geom.vertex_attributes {
            vertices.push(vtx.to_wgpu());
        }
        for inst in &geom.instance_attributes {
            instances.push(inst.to_wgpu());
        }

        let mut vertex_buffers = Vec::new();
        if !geom.vertex_attributes.is_empty() {
            vertex_buffers.push(vertices.buffer_layout());
        }
        if !geom.instance_attributes.is_empty() {
            vertex_buffers.push(instances.buffer_layout());
        }

        let descriptor = wgpu::RenderPipelineDescriptor {
            label: Some(&label),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module,
                entry_point: "vs_main",
                buffers: &vertex_buffers,
            },
            fragment: Some(wgpu::FragmentState {
                module,
                entry_point: "fs_main",
                targets: color_target,
            }),
            primitive: self.defaults.primitive_state(),
            depth_stencil,
            multiview: None,
            multisample,
        };

        device.create_render_pipeline(&descriptor)
    }


    pub fn try_get(
        &self,
        pipeline_id: GeneratedPipelineId,
        pattern: ShaderPatternId,
        surface: SurfaceConfig,
    ) -> Option<&wgpu::RenderPipeline> {
        let result = self.pipelines.get(&Self::key(pipeline_id, pattern, surface));
        if result.is_none() {
            println!("missing pipeline id={pipeline_id:?} pattern={pattern:?} surface={surface:?}");
        }
        //println!("- using pipeline {} with pattern {:?}", self.params[pipeline_id.index()].label, pattern);
        result
    }

    fn key(pipeline_id: GeneratedPipelineId, pattern: ShaderPatternId, variant: SurfaceConfig) -> u64 {
        variant.as_u8() as u64 | ((pipeline_id.0 as u64) << 16) | (pattern.get() as u64) << 32
    }

}

struct ShaderSources {
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
        library.insert("mask::circle".into(), include_str!("../../shaders/lib/mask/circle.wgsl").into());
        library.insert("pattern::color".into(), include_str!("../../shaders/lib/pattern/color.wgsl").into());

        ShaderSources {
            source_library: library,
            preprocessor: Preprocessor::new(),
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

pub struct Varying {
    pub name: Cow<'static, str>,
    pub kind: WgslType,
    pub interpolated: bool,
}

impl Varying {
    #[inline] pub fn float32(name: &'static str) -> Self {
        Varying { name: name.into(), kind: WgslType::Float32, interpolated: true }
    }
    #[inline] pub fn float32x2(name: &'static str) -> Self {
        Varying { name: name.into(), kind: WgslType::Float32x2, interpolated: true }
    }
    #[inline] pub fn float32x3(name: &'static str) -> Self {
        Varying { name: name.into(), kind: WgslType::Float32x3, interpolated: true }
    }
    #[inline] pub fn float32x4(name: &'static str) -> Self {
        Varying { name: name.into(), kind: WgslType::Float32x4, interpolated: true }
    }
    #[inline] pub fn uint32(name: &'static str) -> Self {
        Varying { name: name.into(), kind: WgslType::Uint32, interpolated: true }
    }
    #[inline] pub fn uint32x2(name: &'static str) -> Self {
        Varying { name: name.into(), kind: WgslType::Uint32x2, interpolated: true }
    }
    #[inline] pub fn uint32x3(name: &'static str) -> Self {
        Varying { name: name.into(), kind: WgslType::Uint32x3, interpolated: true }
    }
    #[inline] pub fn uint32x4(name: &'static str) -> Self {
        Varying { name: name.into(), kind: WgslType::Uint32x4, interpolated: true }
    }
    #[inline] pub fn sint32(name: &'static str) -> Self {
        Varying { name: name.into(), kind: WgslType::Sint32, interpolated: true }
    }
    #[inline] pub fn sint32x2(name: &'static str) -> Self {
        Varying { name: name.into(), kind: WgslType::Sint32x2, interpolated: true }
    }
    #[inline] pub fn sint32x3(name: &'static str) -> Self {
        Varying { name: name.into(), kind: WgslType::Sint32x3, interpolated: true }
    }
    #[inline] pub fn sint32x4(name: &'static str) -> Self {
        Varying { name: name.into(), kind: WgslType::Sint32x4, interpolated: true }
    }
    #[inline] pub fn with_interpolation(mut self, interpolated: bool) -> Self {
        self.interpolated = interpolated;
        self
    }
    #[inline] pub fn interpolated(self) -> Self {
        self.with_interpolation(true)
    }
    #[inline] pub fn flat(self) -> Self {
        self.with_interpolation(false)
    }
}

pub struct VertexAtribute {
    pub name: Cow<'static, str>,
    pub kind: WgslType,
}

impl VertexAtribute {
    #[inline] pub fn float32(name: &'static str) -> Self {
        VertexAtribute { name: name.into(), kind: WgslType::Float32 }
    }
    #[inline] pub fn float32x2(name: &'static str) -> Self {
        VertexAtribute { name: name.into(), kind: WgslType::Float32x2 }
    }
    #[inline]  pub fn float32x3(name: &'static str) -> Self {
        VertexAtribute { name: name.into(), kind: WgslType::Float32x3 }
    }
    #[inline] pub fn float32x4(name: &'static str) -> Self {
        VertexAtribute { name: name.into(), kind: WgslType::Float32x4 }
    }
    #[inline] pub fn uint32(name: &'static str) -> Self {
        VertexAtribute { name: name.into(), kind: WgslType::Uint32 }
    }
    #[inline] pub fn uint32x2(name: &'static str) -> Self {
        VertexAtribute { name: name.into(), kind: WgslType::Uint32x2 }
    }
    #[inline] pub fn uint32x3(name: &'static str) -> Self {
        VertexAtribute { name: name.into(), kind: WgslType::Uint32x3 }
    }
    #[inline] pub fn uint32x4(name: &'static str) -> Self {
        VertexAtribute { name: name.into(), kind: WgslType::Uint32x4 }
    }
    #[inline] pub fn int32(name: &'static str) -> Self {
        VertexAtribute { name: name.into(), kind: WgslType::Sint32 }
    }
    #[inline] pub fn int32x2(name: &'static str) -> Self {
        VertexAtribute { name: name.into(), kind: WgslType::Sint32x2 }
    }
    #[inline] pub fn int32x3(name: &'static str) -> Self {
        VertexAtribute { name: name.into(), kind: WgslType::Sint32x3 }
    }
    #[inline] pub fn int32x4(name: &'static str) -> Self {
        VertexAtribute { name: name.into(), kind: WgslType::Sint32x4 }
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
            _ => unimplemented!()
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
            label: Some(name.as_str())
        });

        BindGroupLayout { name, handle, entries }
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
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
            name: name.into(),
            struct_type: struct_type.into()
        }
    }

    pub fn storage_buffer(name: &str, struct_type: &str) -> Self {
        Binding {
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            // TODO: min binding size.
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
            name: name.into(),
            struct_type: struct_type.into()
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
                wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, .. } => {
                    writeln!(source, "var<uniform> {name}: {struct_ty};").unwrap();
                }
                wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { .. }, .. } => {
                    writeln!(source, "var<storage> {name}: {struct_ty};").unwrap();
                }
                wgpu::BindingType::Sampler(..) => {
                    writeln!(source, "var {name}: sampler;").unwrap();
                }
                wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth, .. } => {
                    writeln!(source, "var {name}: texture_depth_2d;").unwrap();
                }
                wgpu::BindingType::Texture { sample_type, view_dimension, .. } => {
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

pub struct MaskDescriptor {
    pub name: Cow<'static, str>,
    pub source: Source,
    pub varyings: Vec<Varying>,
    pub bindings: Option<BindGroupLayoutId>,
}

pub struct GeometryDescriptor {
    pub name: Cow<'static, str>,
    pub source: Source,
    pub vertex_attributes: Vec<VertexAtribute>,
    pub instance_attributes: Vec<VertexAtribute>,
    pub varyings: Vec<Varying>,
    pub bindings: Option<BindGroupLayoutId>,
}

pub fn generate_shader_source(
    geometry: &GeometryDescriptor,
    mask: Option<&MaskDescriptor>,
    pattern: Option<&PatternDescriptor>,
    base_bindings: Option<&BindGroupLayout>,
    geom_bindings: Option<&BindGroupLayout>,
    mask_bindings: Option<&BindGroupLayout>,
    pattern_bindings: Option<&BindGroupLayout>,
) -> String {

    let mut source = String::new();

    let mut group_index = 0;
    if let Some(_bindings) = base_bindings {
        // TODO right now the base bindings are in the imported sources. Generate them
        // once the shaders have moved to this system.
        //bindings.generate_shader_source(group_index, &mut source);
        writeln!(source, "@group(0) @binding(2) var default_sampler: sampler;").unwrap();
        group_index += 1;
    }
    if let Some(bindings) = geom_bindings {
        bindings.generate_shader_source(group_index, &mut source);
        group_index += 1;
    }
    if let Some(bindings) = mask_bindings {
        bindings.generate_shader_source(group_index, &mut source);
        group_index += 1;
    }
    if let Some(bindings) = pattern_bindings {
        bindings.generate_shader_source(group_index, &mut source);
    }

    writeln!(source, "struct Geometry {{").unwrap();
    writeln!(source, "    position: vec4<f32>,").unwrap();
    writeln!(source, "    pattern_position: vec2<f32>,").unwrap();
    writeln!(source, "    pattern_data: u32,").unwrap();
    writeln!(source, "    mask_position: vec2<f32>,").unwrap();
    writeln!(source, "    mask_data: u32,").unwrap();
    for varying in &geometry.varyings {
        //println!("=====    {}: {},", varying.name, varying.kind.as_str());
        writeln!(source, "    {}: {},", varying.name, varying.kind.as_str()).unwrap();
    }
    writeln!(source, "}}").unwrap();

    writeln!(source, "#import {}", geometry.name).unwrap();
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

    if let Some(mask) = mask {
        writeln!(source, "struct Mask {{").unwrap();
        for varying in &mask.varyings {
            writeln!(source, "    {}: {},", varying.name, varying.kind.as_str()).unwrap();
        }
        writeln!(source, "}}").unwrap();
        writeln!(source, "#import {}", mask.name).unwrap();
    }

    // vertex

    writeln!(source, "struct VertexOutput {{").unwrap();
    writeln!(source, "    @builtin(position) position: vec4<f32>,").unwrap();
    let mut idx = 0;
    for varying in &geometry.varyings {
        let interpolate = if varying.interpolated { "perspective" } else { "flat" };
        writeln!(source, "    @location({idx}) @interpolate({interpolate}) geom_{}: {},", varying.name, varying.kind.as_str()).unwrap();
        idx += 1;
    }
    if let Some(pattern) = pattern {
        for varying in &pattern.varyings {
            let interpolate = if varying.interpolated { "perspective" } else { "flat" };
            writeln!(source, "    @location({idx}) @interpolate({interpolate}) pat_{}: {},", varying.name, varying.kind.as_str()).unwrap();
            idx += 1;
        }
    }
    if let Some(mask) = mask {
        for varying in &mask.varyings {
            let interpolate = if varying.interpolated { "perspective" } else { "flat" };
            writeln!(source, "    @location({idx}) @interpolate({interpolate}) mask_{}: {},", varying.name, varying.kind.as_str()).unwrap();
            idx += 1;
        }
    }
    writeln!(source, "}}").unwrap();
    writeln!(source, "").unwrap();

    writeln!(source, "@vertex fn vs_main(").unwrap();
    writeln!(source, "    @builtin(vertex_index) vertex_index: u32,").unwrap();
    let mut attr_location = 0;
    for attrib in &geometry.vertex_attributes {
        writeln!(source, "    @location({attr_location}) vtx_{}: {},", attrib.name, attrib.kind.as_str()).unwrap();
        attr_location += 1;
    }
    for attrib in &geometry.instance_attributes {
        writeln!(source, "    @location({attr_location}) inst_{}: {},", attrib.name, attrib.kind.as_str()).unwrap();
        attr_location += 1;
    }
    writeln!(source, ") -> VertexOutput {{").unwrap();

    writeln!(source, "    var vertex = geometry_vertex(").unwrap();
    writeln!(source, "        vertex_index,").unwrap();
    for attrib in &geometry.vertex_attributes {
        writeln!(source, "        vtx_{},", attrib.name).unwrap();
    }
    for attrib in &geometry.instance_attributes {
        writeln!(source, "        inst_{},", attrib.name).unwrap();
    }
    writeln!(source, "    );").unwrap();

    if pattern.is_some() {
        writeln!(source, "    var pattern = pattern_vertex(vertex.pattern_position, vertex.pattern_data);").unwrap();
    }
    if mask.is_some() {
        writeln!(source, "    var mask = mask_vertex(vertex.mask_position, vertex.mask_data);").unwrap();
    }

    writeln!(source, "    return VertexOutput(").unwrap();
    writeln!(source, "        vertex.position,").unwrap();
    for varying in &geometry.varyings {
        writeln!(source, "        vertex.{},", varying.name).unwrap();
    }
    if let Some(pattern) = pattern {
        for varying in &pattern.varyings {
            writeln!(source, "        pattern.{},", varying.name).unwrap();
        }
    }
    if let Some(mask) = mask {
        for varying in &mask.varyings {
            writeln!(source, "        mask.{},", varying.name).unwrap();
        }
    }
    writeln!(source, "    );").unwrap();
    writeln!(source, "}}").unwrap();
    writeln!(source, "").unwrap();

    // fragment

    writeln!(source, "@fragment fn fs_main(").unwrap();
    let mut idx = 0;
    for varying in &geometry.varyings {
        let interpolate = if varying.interpolated { "perspective" } else { "flat" };
        writeln!(source, "    @location({idx}) @interpolate({interpolate}) geom_{}: {},", varying.name, varying.kind.as_str()).unwrap();
        idx += 1;
    }
    if let Some(pattern) = pattern {
        for varying in &pattern.varyings {
            let interpolate = if varying.interpolated { "perspective" } else { "flat" };
            writeln!(source, "    @location({idx}) @interpolate({interpolate}) pat_{}: {},", varying.name, varying.kind.as_str()).unwrap();
            idx += 1;
        }
    }
    if let Some(mask) = mask {
        for varying in &mask.varyings {
            let interpolate = if varying.interpolated { "perspective" } else { "flat" };
            writeln!(source, "    @location({idx}) @interpolate({interpolate}) mask_{}: {},", varying.name, varying.kind.as_str()).unwrap();
            idx += 1;
        }
    }
    writeln!(source, ") -> @location(0) vec4<f32> {{").unwrap();
    writeln!(source, "    var color = vec4<f32>(1.0);").unwrap();

    writeln!(source, "    color *= geometry_fragment(").unwrap();
    for varying in &geometry.varyings {
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

    if let Some(mask) = mask {
        writeln!(source, "    color.a *= mask_fragment(Mask(").unwrap();
        for varying in &mask.varyings {
            writeln!(source, "        mask_{},", varying.name).unwrap();
        }
        writeln!(source, "    ));").unwrap();
    }

    // TODO: premultiply the alpha only for the corresponding blend mode?
    // That would make the premultiplication part of the module key.
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
    pub geometry: ShaderGeometryId,
    pub pattern: ShaderPatternId,
    pub mask: ShaderMaskId,
    pub user_flags: u8,
    pub defines: Vec<&'static str>,
}

pub struct PipelineDescriptor {
    pub label: &'static str,
    pub geometry: ShaderGeometryId,
    pub mask: ShaderMaskId,
    pub user_flags: u8,
    pub output: OutputType,
    pub blend: BlendMode,
    pub shader_defines: Vec<&'static str>,
}


#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum OutputType {
    Color,
    Alpha,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BlendMode {
    None,
    PremultipliedAlpha,
    Add,
    Subtract,
    Min,
    Max,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum StencilMode {
    EvenOdd,
    NonZero,
    Ignore,
    None,
}

impl From<FillRule> for StencilMode {
    fn from(fill_rule: FillRule) -> Self {
        match fill_rule {
            FillRule::EvenOdd => { StencilMode::EvenOdd }
            FillRule::NonZero => { StencilMode::NonZero }
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DepthMode {
    Enabled,
    Ignore,
    None,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SurfaceConfig {
    pub msaa: bool,
    pub depth: DepthMode,
    pub stencil: StencilMode,
}

impl SurfaceConfig {
    pub(crate) fn as_u8(&self) -> u8 {
        return if self.msaa { 1 } else { 0 }
            + match self.depth {
                DepthMode::Enabled => 2,
                DepthMode::Ignore => 4,
                DepthMode::None => 0,
            }
            + match self.stencil {
                StencilMode::EvenOdd => 8,
                StencilMode::NonZero => 16,
                StencilMode::Ignore => 32,
                StencilMode::None => 0,
            };
    }
}

impl Default for SurfaceConfig {
    fn default() -> Self {
        SurfaceConfig { msaa: false, depth: DepthMode::None, stencil: StencilMode::None }
    }
}


#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Depth {
    None,
    Read,
    ReadWrite,
}
