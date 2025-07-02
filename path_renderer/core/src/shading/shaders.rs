use std::collections::HashMap;

use super::generator::generate_shader_source;
use super::preprocessor::{Preprocessor, Source, SourceError};
use super::{
    BindGroupLayout, BindGroupLayoutId, Binding, BlendMode, DepthMode, GeometryDescriptor,
    GeometryId, PatternDescriptor, PipelineDefaults, ShaderPatternId, StencilMode,
    SurfaceDrawConfig, SurfaceKind, VertexBuilder,
};

/// Generates and manages render pipelines
///
/// Renderers are not required to use this and can use wgpu to build render
/// pipeliens directly. However, `Shaders` provides a very convenient way to
/// decouple renderers from patterns.
///
/// A generated render pipeline is described by:
/// - A geometry (for example: tiling, mesh, rectangle, etc.).
/// - A pattern (for example: solid color, gradient, etc.).
/// - A blend mode.
/// - A surface configuration which defines the output surface format,
///   msaa, depth and stencil parameters.
///
/// See also `RenderPipelineKey`.
///
/// This allows renderers to register geometries that automatically work with
/// various patterns, surface configurations and blend modes without extra work.
pub struct Shaders {
    sources: ShaderSources,

    patterns: Vec<PatternDescriptor>,
    geometries: Vec<GeometryDescriptor>,
    module_cache: HashMap<ModuleKey, wgpu::ShaderModule>,
    bind_group_layouts: Vec<BindGroupLayout>,
    pipeline_layouts: HashMap<u64, wgpu::PipelineLayout>,
    pub defaults: PipelineDefaults,
    pub common_bind_group_layouts: CommonBindGroupLayouts,
}

pub struct CommonBindGroupLayouts {
    pub target_and_gpu_buffer: BindGroupLayoutId,
    pub color_texture: BindGroupLayoutId,
    pub alpha_texture: BindGroupLayoutId,
}

impl Shaders {
    pub fn new(device: &wgpu::Device) -> Self {
        let mut bind_group_layouts = Vec::new();
        let common_bind_group_layouts = init_common_layouts(&mut bind_group_layouts, device);

        Shaders {
            sources: ShaderSources::new(),

            patterns: Vec::new(),
            geometries: Vec::new(),
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

    /// Registers the source for a piece of shader code that can be imported by
    /// another piece of shader code using the `#import` directive.
    pub fn register_library(&mut self, name: &str, source: Source) {
        self.sources.source_library.insert(name.into(), source);
    }

    /// Register a pattern.
    ///
    /// The pattern source must contain the following two functions:
    /// ```wgsl
    /// fn pattern_vertex(pattern_pos: vec2<f32>, pattern_handle: u32) -> Pattern
    /// fn pattern_fragment(pattern: Pattern) -> vec4<f32>
    /// ```
    ///
    /// Typically, `pattern_vertex`, executed in the vertex shader, reads custom
    /// pattern parameters from buffers and/or textures and encodes them in the
    /// returned `Pattern` which is provided to the `pattern_fragment` in the
    /// fragment shader which returns the final color of the pattern for the
    /// provided pixel.
    ///
    /// Here is an example of a simple gradient pattern:
    ///
    /// ```wgsl
    ///    // The global gpu buffers are provided in the base bindings, so
    ///    // patterns do not need to include them in their own bindings.
    ///    #import gpu_buffer
    ///
    ///    struct Gradient {
    ///        p0: vec2<f32>,
    ///        p1: vec2<f32>,
    ///        color0: vec4<f32>,
    ///        color1: vec4<f32>,
    ///    };
    ///
    ///    fn fetch_gradient(address: u32) -> Gradient {
    ///        var raw = f32_gpu_buffer_fetch_3(address);
    ///        var gradient: Gradient;
    ///        gradient.p0 = raw.data0.xy;
    ///        gradient.p1 = raw.data0.zw;
    ///        gradient.color0 = raw.data1;
    ///        gradient.color1 = raw.data2;
    ///
    ///        return gradient;
    ///    }
    ///
    ///    // Entry point of the pattern in the vertex shader.
    ///    fn pattern_vertex(pattern_pos: vec2<f32>, pattern_handle: u32) -> Pattern {
    ///        var gradient = fetch_gradient(pattern_handle);
    ///        var dir = gradient.p1 - gradient.p0;
    ///        dir = dir / dot(dir, dir);
    ///        var offset = dot(gradient.p0, dir);
    ///
    ///        // The `Pattern` struct is generated based on this list of `Varrying`s
    ///        // provided in the PatternDescriptor.
    ///        return Pattern(
    ///            pattern_pos,
    ///            gradient.color0,
    ///            gradient.color1,
    ///            vec3<f32>(dir, offset),
    ///        );
    ///    }
    ///
    ///    // Entry point of the pattern in the fragment shader.
    ///    fn pattern_fragment(pattern: Pattern) -> vec4<f32> {
    ///        var d = clamp(dot(pattern.position, pattern.dir_offset.xy) - pattern.dir_offset.z, 0.0, 1.0);
    ///        return mix(pattern.color0, pattern.color1, d);
    ///    }
    /// ```
    pub fn register_pattern(&mut self, pattern: PatternDescriptor) -> ShaderPatternId {
        self.sources
            .source_library
            .insert(pattern.name.clone().into(), pattern.source.clone());
        let id = ShaderPatternId::from_index(self.patterns.len());
        self.patterns.push(pattern);

        id
    }

    /// Register a geometry
    ///
    /// The source must contain the following two functions:
    /// ```wgsl
    /// fn geometry_vertex(vertex_index: u32, rect: vec4<f32>, z_index: u32, pattern: u32, mask: u32, transform_flags: u32) -> GeometryVertex
    /// fn geometry_fragment(/*varyings...*/) -> f32 {
    /// ```
    ///
    /// The `GeometryVertex` struct is generated based on the list of `Varying`s provided in
    /// the `GeometryDescriptor` and three mandatory members.
    /// The first three members of `GeometryVertex` are always:
    /// - `position: vec2f`: the device-space position of the vertex.
    /// - `local_position: vec2f`:  the position in local space of the vertex, affecting
    ///   how the pattern is rendered.
    /// - `pattern: u32`: The pattern id provided as input.
    /// Then, the varying parameters in the same order as in the varying list.
    ///
    /// The parameters of the `geometry_fragment` are the varyings provided in the
    /// `GeometryDescriptor`.
    pub fn register_geometry(&mut self, geometry: GeometryDescriptor) -> GeometryId {
        self.sources
            .source_library
            .insert(geometry.name.clone().into(), geometry.source.clone());
        let id = GeometryId::from_index(self.geometries.len());
        self.geometries.push(geometry);

        id
    }

    /// Look up a geometry id by name.
    pub fn find_geometry(&self, name: &str) -> Option<GeometryId> {
        for (idx, pat) in self.geometries.iter().enumerate() {
            if &*pat.name == name {
                return Some(GeometryId::from_index(idx));
            }
        }

        None
    }

    /// Look up a pattern id by name.
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

    /// Create a `wgpu::ShaderModule` directly without using the geometry/pattern
    /// infrastructure.
    ///
    /// The difference with creating a module directly from the `wgpu::Device` is that
    /// the shader source is pre-processed, allowing it to use `#import` and static
    /// `#if` directives.
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

    fn generate_pipeline_variant_src(
        &mut self,
        pipeline_id: GeometryId,
        pattern_id: ShaderPatternId,
    ) -> String {
        let geometry = &self.geometries[pipeline_id.index()];
        let pattern = pattern_id.map(|p| &self.patterns[p.index()]);

        let base_bindings =
            &self.bind_group_layouts[self.common_bind_group_layouts.target_and_gpu_buffer.index()];
        let geometry_bindings = geometry
            .bindings
            .map(|id| &self.bind_group_layouts[id.index()]);
        let pattern_bindings = pattern
            .map(|desc| desc.bindings)
            .flatten()
            .map(|id| &self.bind_group_layouts[id.index()]);

        let src = generate_shader_source(
            geometry,
            pattern,
            base_bindings,
            geometry_bindings,
            pattern_bindings,
        );

        self.sources.preprocessor.reset_defines();
        for define in &geometry.shader_defines {
            self.sources.preprocessor.define(define);
        }

        let src = self
            .sources
            .preprocessor
            .preprocess("generated module", &src, &mut self.sources.source_library)
            .unwrap();

        src
    }

    /// Print to stdout the generated source for a specific combination of geometry
    /// and pattern.
    pub fn print_pipeline_variant(&mut self, geometry: GeometryId, pattern: ShaderPatternId) {
        let src = self.generate_pipeline_variant_src(geometry, pattern);
        println!("{src}");
    }

    pub(crate) fn generate_pipeline_variant(
        &mut self,
        device: &wgpu::Device,
        geometry_id: GeometryId,
        pattern_id: ShaderPatternId,
        blend_mode: BlendMode,
        surface: &SurfaceDrawConfig,
    ) -> wgpu::RenderPipeline {
        let geometry = &self.geometries[geometry_id.index()];
        let pattern = pattern_id.map(|p| &self.patterns[p.index()]);

        let module_key = ModuleKey {
            geometry: geometry_id,
            pattern: pattern_id,
            defines: geometry.shader_defines.clone(),
        };

        let module = self.module_cache.entry(module_key).or_insert_with(|| {
            let base_bindings = &self.bind_group_layouts
                [self.common_bind_group_layouts.target_and_gpu_buffer.index()];
            let geometry_bindings = geometry
                .bindings
                .map(|id| &self.bind_group_layouts[id.index()]);
            let pattern_bindings = pattern
                .map(|desc| desc.bindings)
                .flatten()
                .map(|id| &self.bind_group_layouts[id.index()]);

            let src = generate_shader_source(
                geometry,
                pattern,
                base_bindings,
                geometry_bindings,
                pattern_bindings,
            );

            //println!("--------------- pipeline {} mask {:?}. pattern {:?}", params.label, mask.map(|m| &m.name), pattern.map(|p| &p.name));
            //println!("{src}");
            //println!("----");
            self.sources.preprocessor.reset_defines();
            for define in &geometry.shader_defines {
                self.sources.preprocessor.define(define);
            }

            let src = self
                .sources
                .preprocessor
                .preprocess("generated module", &src, &mut self.sources.source_library)
                .unwrap();

            //println!("==================\n{src}\n=================");
            let label = format!(
                "{}|{}",
                geometry.name,
                pattern.map(|p| &p.name[..]).unwrap_or("")
            );
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            })
        });

        // Increment the ids by 1 so that id zero isn't equal to any id.
        let mut key = 0u64;
        let mut shift = 0;
        if let Some(id) = geometry.bindings {
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
            let geometry_id = self.common_bind_group_layouts.target_and_gpu_buffer;
            layouts.push(&self.bind_group_layouts[geometry_id.index()].handle);

            if let Some(id) = geometry.bindings {
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
            geometry.name,
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

        let geometry = &self.geometries[geometry_id.index()];
        for vtx in &geometry.vertex_attributes {
            vertices.push(vtx.to_wgpu());
        }
        for inst in &geometry.instance_attributes {
            instances.push(inst.to_wgpu());
        }

        let mut vertex_buffers = Vec::new();
        if !geometry.vertex_attributes.is_empty() {
            vertex_buffers.push(vertices.buffer_layout());
        }
        if !geometry.instance_attributes.is_empty() {
            vertex_buffers.push(instances.buffer_layout());
        }

        let descriptor = wgpu::RenderPipelineDescriptor {
            label: Some(&label),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module,
                entry_point: Some("vs_main"),
                buffers: &vertex_buffers,
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: geometry.constants.as_slice(),
                    zero_initialize_workgroup_memory: false,
                },
            },
            fragment: Some(wgpu::FragmentState {
                module,
                entry_point: Some("fs_main"),
                targets,
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: geometry.constants.as_slice(),
                    zero_initialize_workgroup_memory: false,
                },
            }),
            primitive: geometry.primitive,
            depth_stencil,
            multiview: None,
            multisample,
            cache: None,
        };

        device.create_render_pipeline(&descriptor)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ModuleKey {
    geometry: GeometryId,
    pattern: ShaderPatternId,
    defines: Vec<&'static str>,
}

fn blend_state(blend_mode: BlendMode) -> Option<wgpu::BlendState> {
    match blend_mode {
        BlendMode::None => None,
        BlendMode::PremultipliedAlpha => Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
        BlendMode::Add => Some(wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        }),
        BlendMode::Subtract => Some(wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Subtract,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        }),
        BlendMode::Multiply => Some(wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::Src,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::SrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
        }),
        BlendMode::Screen => Some(wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrc,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
        }),
        BlendMode::Lighter => Some(wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        }),
        BlendMode::Exclusion => Some(wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::OneMinusDst,
                dst_factor: wgpu::BlendFactor::OneMinusSrc,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
        }),
        BlendMode::ClipIn => Some(wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::SrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Min,
            },
        }),
        BlendMode::ClipOut => Some(wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Min,
            },
        }),
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
            "gpu_buffer".into(),
            include_str!("../../shaders/lib/gpu_buffer.wgsl").into(),
        );
        library.insert(
            "render_task".into(),
            include_str!("../../shaders/lib/render_task.wgsl").into(),
        );
        library.insert(
            "trigonometry".into(),
            include_str!("../../shaders/lib/trigonometry.wgsl").into(),
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

fn init_common_layouts(
    layouts: &mut Vec<BindGroupLayout>,
    device: &wgpu::Device,
) -> CommonBindGroupLayouts {
    assert!(layouts.is_empty());

    layouts.push(BindGroupLayout::new(
        device,
        "target and gpu buffers".into(),
        vec![
            Binding {
                name: "f32_gpu_buffer_texture".into(),
                struct_type: "f32".into(),
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
            },
            Binding {
                name: "u32_gpu_buffer_texture".into(),
                struct_type: "u32".into(),
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Uint,
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
        target_and_gpu_buffer: BindGroupLayoutId(0),
        color_texture: BindGroupLayoutId(1),
        alpha_texture: BindGroupLayoutId(2),
    }
}
