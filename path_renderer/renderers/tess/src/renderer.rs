use lyon::{
    tessellation::{FillTessellator, FillOptions, VertexBuffers, FillVertexConstructor, BuffersBuilder},
    path::traits::PathIterator, geom::euclid::vec2
};
use core::{
    canvas::{RendererId, Shape, RendererCommandIndex, Canvas, RecordedShape, CanvasRenderer, RenderPasses, SubPass, ZIndex, RenderPassState, TransformId, DrawHelper},
    resources::{ResourcesHandle, GpuResources, CommonGpuResources},
    gpu::{
        shader::{StencilMode, SurfaceConfig, DepthMode, ShaderPatternId},
        DynBufferRange, Shaders
    },
    pattern::{BuiltPattern, BindingsId}, usize_range,
    bytemuck,
    wgpu, BindingResolver,
};
use std::ops::Range;

use super::MeshGpuResources;

pub const PATTERN_KIND_COLOR: u32 = 0;
pub const PATTERN_KIND_SIMPLE_LINEAR_GRADIENT: u32 = 1;

pub struct Fill {
    pub shape: RecordedShape,
    pub pattern: BuiltPattern,
    pub transform: TransformId,
    pub z_index: ZIndex,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Vertex {
    pub x: f32,
    pub y: f32,
    pub z_index: u32,
    pub pattern: u32,
}

unsafe impl bytemuck::Zeroable for Vertex {}
unsafe impl bytemuck::Pod for Vertex {}

struct VertexCtor {
    pattern: u32, z_index: u32,
}

impl FillVertexConstructor<Vertex> for VertexCtor {
    fn new_vertex(&mut self, vertex: lyon::lyon_tessellation::FillVertex) -> Vertex {
        let (x, y) = vertex.position().to_tuple();
        Vertex { x, y, z_index: self.z_index, pattern: self.pattern }
    }
}

struct MeshSubPass {
    draws: Range<u32>,
    z_index: ZIndex,
}

struct Draw {
    indices: Range<u32>,
    pattern: ShaderPatternId,
    pattern_inputs: BindingsId,
    // TODO: blend mode.
    opaque: bool,
}

pub struct MeshRenderer {
    renderer_id: RendererId,
    common_resources: ResourcesHandle<CommonGpuResources>,
    resources: ResourcesHandle<MeshGpuResources>,
    tessellator: FillTessellator,
    geometry: VertexBuffers<Vertex, u32>,
    tolerenace: f32,
    enable_opaque_pass: bool,
    enable_msaa: bool,

    commands: Vec<Fill>,
    draws: Vec<Draw>,
    render_passes: Vec<MeshSubPass>,
    vbo_range: Option<DynBufferRange>,
    ibo_range: Option<DynBufferRange>,
}

impl MeshRenderer {
    pub fn new(renderer_id: RendererId, common_resources: ResourcesHandle<CommonGpuResources>, resources: ResourcesHandle<MeshGpuResources>) -> Self {
        MeshRenderer {
            renderer_id,
            common_resources,
            resources,
            tessellator: FillTessellator::new(),
            geometry: VertexBuffers::new(),
            tolerenace: 0.25,
            enable_opaque_pass: false,
            enable_msaa: false,

            commands: Vec::new(),
            draws: Vec::new(),
            render_passes: Vec::new(),
            vbo_range: None,
            ibo_range: None,
        }
    }

    pub fn begin_frame(&mut self, canvas: &Canvas) {
        self.commands.clear();
        self.draws.clear();
        self.render_passes.clear();
        self.geometry.vertices.clear();
        self.geometry.indices.clear();
        self.enable_opaque_pass = canvas.surface.opaque_pass();
        self.enable_msaa = canvas.surface.msaa();
        self.vbo_range = None;
        self.ibo_range = None;
    }

    pub fn fill<S: Shape>(&mut self, canvas: &mut Canvas, shape: S, pattern: BuiltPattern) {
        let transform = canvas.transforms.current();
        let z_index = canvas.z_indices.push();
        let index = self.commands.len() as RendererCommandIndex;
        self.commands.push(Fill {
            shape: shape.to_command(),
            pattern,
            transform,
            z_index,
        });
        canvas.commands.push(self.renderer_id, index);
    }

    pub fn prepare(&mut self, canvas: &Canvas) {
        if self.commands.is_empty() {
            return;
        }

        let commands = std::mem::take(&mut self.commands);
        for range in canvas.commands.with_renderer(self.renderer_id) {
            let range = range.start as usize .. range.end as usize;
            let z_index = commands[range.start].z_index;

            let draw_start = self.draws.len() as u32;
            let mut key = commands.first().as_ref().unwrap().pattern.batch_key();

            // Opaque pass.
            let mut geom_start = self.geometry.indices.len() as u32;
            if self.enable_opaque_pass {
                for fill in commands[range.clone()].iter().rev().filter(|fill| fill.pattern.is_opaque) {
                    if key != fill.pattern.batch_key() {
                        let end = self.geometry.indices.len() as u32;
                        if end > geom_start {
                            self.draws.push(Draw {
                                indices: geom_start..end,
                                pattern: key.0,
                                pattern_inputs: key.1,
                                opaque: true,
                            });
                        }
                        geom_start = end;
                        key = fill.pattern.batch_key();
                    }
                    self.prepare_fill(fill, canvas);
                }
            }

            let end = self.geometry.indices.len() as u32;
            if end > geom_start {
                self.draws.push(Draw {
                    indices: geom_start..end,
                    pattern: key.0,
                    pattern_inputs: key.1,
                    opaque: true,
                });
            }
            geom_start = end;

            // Blended pass.
            let enable_opaque_pass = self.enable_opaque_pass;
            for fill in commands[range.clone()].iter().filter(|fill| !enable_opaque_pass || !fill.pattern.is_opaque) {
                if key != fill.pattern.batch_key() {
                    let end = self.geometry.indices.len() as u32;
                    if end > geom_start {
                        self.draws.push(Draw {
                            indices: geom_start..end,
                            pattern: key.0,
                            pattern_inputs: key.1,
                            opaque: false,
                        });
                    }
                    geom_start = end;
                    key = fill.pattern.batch_key();
                }
                self.prepare_fill(fill, canvas);
            }

            let end = self.geometry.indices.len() as u32;
            if end > geom_start {
                self.draws.push(Draw {
                    indices: geom_start..end,
                    pattern: key.0,
                    pattern_inputs: key.1,
                    opaque: false,
                });
            }

            let draws = draw_start .. self.draws.len() as u32;

            if !draws.is_empty() {
                // TODO: emmit draws per pattern.
                self.render_passes.push(MeshSubPass { draws, z_index });
            }
        }

        self.commands = commands;
    }

    fn prepare_fill(&mut self, fill: &Fill, canvas: &Canvas) {
        let transform = canvas.transforms.get(fill.transform);

        match &fill.shape {
            RecordedShape::Path(shape) => {
                let options = FillOptions::tolerance(self.tolerenace)
                    .with_fill_rule(shape.fill_rule);

                // TODO: some way to simplify/discard offscreen geometry, would probably be best
                // done in the tessellator itself.
                self.tessellator.tessellate(
                    shape.path.iter().transformed(transform),
                    &options,
                    &mut BuffersBuilder::new(
                        &mut self.geometry,
                        VertexCtor {
                            z_index: fill.z_index,
                            pattern: fill.pattern.data,
                        },
                    )
                ).unwrap();
            }
            RecordedShape::Circle(circle) => {
                let options = FillOptions::tolerance(self.tolerenace);
                self.tessellator.tessellate_circle(
                    transform.transform_point(circle.center),
                    // TODO: that's not quite right if the transform has more than scale+offset
                    transform.transform_vector(vec2(circle.radius, 0.0)).length(),
                    &options,
                    &mut BuffersBuilder::new(
                        &mut self.geometry,
                        VertexCtor {
                            z_index: fill.z_index,
                            pattern: fill.pattern.data,
                        },
                    )
                ).unwrap();
            }
            _ => {
                todo!()
            }
        }
    }

    pub fn upload(&mut self,
        resources: &mut GpuResources,
        shaders: &mut Shaders,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) {
        let res = &resources[self.resources];
        let opaque_pipeline = res.opaque_pipeline;
        let alpha_pipeline = res.alpha_pipeline;

        let res = &mut resources[self.common_resources];
        self.vbo_range = res.vertices.upload(device, bytemuck::cast_slice(&self.geometry.vertices));
        self.ibo_range = res.indices.upload(device, bytemuck::cast_slice(&self.geometry.indices));

        let surface = SurfaceConfig {
            msaa: self.enable_msaa,
            depth: if self.enable_opaque_pass { DepthMode::Enabled } else { DepthMode::None },
            stencil: StencilMode::None,
        };

        let mut prev_pattern = None;
        for draw in &self.draws {
            if Some(draw.pattern) == prev_pattern {
                return;
            }
            shaders.prepare_pipeline(device, opaque_pipeline, draw.pattern, surface);
            shaders.prepare_pipeline(device, alpha_pipeline, draw.pattern, surface);
            prev_pattern = Some(draw.pattern);
        }
    }
}

impl CanvasRenderer for MeshRenderer {
    fn add_render_passes(&mut self, render_passes: &mut RenderPasses) {
        for (idx, pass) in self.render_passes.iter().enumerate() {
            render_passes.push(SubPass {
                renderer_id: self.renderer_id,
                internal_index: idx as u32,
                require_pre_pass: false,
                z_index: pass.z_index,
                use_depth: self.enable_opaque_pass,
                use_stencil: false,
                use_msaa: self.enable_msaa,
            });
        }
    }

    fn render<'pass, 'resources: 'pass>(
        &self,
        index: u32,
        surface_info: &RenderPassState,
        shaders: &'resources Shaders,
        resources: &'resources GpuResources,
        bindings: &'resources dyn BindingResolver,
        render_pass: &mut wgpu::RenderPass<'pass>,
    ) {
        let common_resources = &resources[self.common_resources];
        let mesh_resources = &resources[self.resources];

        let pass = &self.render_passes[index as usize];

        render_pass.set_bind_group(0, &common_resources.main_target_and_gpu_store_bind_group, &[]);

        render_pass.set_index_buffer(common_resources.indices.get_buffer_slice(self.ibo_range.as_ref().unwrap()), wgpu::IndexFormat::Uint32);
        render_pass.set_vertex_buffer(0, common_resources.vertices.get_buffer_slice(self.vbo_range.as_ref().unwrap()));

        let surface = surface_info.surface_config(self.enable_opaque_pass, None);

        let mut helper = DrawHelper::new();

        for draw in &self.draws[usize_range(pass.draws.clone())] {
            let pipeline = shaders.try_get(
                if draw.opaque { mesh_resources.opaque_pipeline } else { mesh_resources.alpha_pipeline },
                draw.pattern,
                surface
            ).unwrap();

            helper.resolve_and_bind(1, draw.pattern_inputs, bindings, render_pass);

            render_pass.set_pipeline(pipeline);
            render_pass.draw_indexed(draw.indices.clone(), 0, 0..1);
        }
    }
}

