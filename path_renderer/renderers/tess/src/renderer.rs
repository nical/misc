use lyon::{
    tessellation::{FillTessellator, FillOptions, VertexBuffers, FillVertexConstructor, BuffersBuilder},
    path::traits::PathIterator, geom::euclid::vec2
};
use core::{
    canvas::{RendererId, Canvas, Shape, CanvasRenderer, ZIndex, RenderPassState, TransformId, DrawHelper, SurfaceState},
    resources::{ResourcesHandle, GpuResources, CommonGpuResources},
    gpu::{
        shader::{ShaderPatternId},
        DynBufferRange, Shaders
    },
    pattern::{BuiltPattern, BindingsId}, usize_range,
    bytemuck,
    wgpu, BindingResolver, batching::BatchFlags,
};
use std::{ops::Range, collections::HashMap};

use super::MeshGpuResources;

pub const PATTERN_KIND_COLOR: u32 = 0;
pub const PATTERN_KIND_SIMPLE_LINEAR_GRADIENT: u32 = 1;

struct Batch {
    commands: Vec<Fill>,
    draws: Range<u32>,
    surface: SurfaceState,
}

pub struct Fill {
    pub shape: Shape,
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
    enable_msaa: bool,

    batches: Vec<Batch>,
    draws: Vec<Draw>,
    vbo_range: Option<DynBufferRange>,
    ibo_range: Option<DynBufferRange>,
    shaders: HashMap<(bool, ShaderPatternId, SurfaceState), Option<u32>>,
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
            enable_msaa: false,

            draws: Vec::new(),
            batches: Vec::new(),
            vbo_range: None,
            ibo_range: None,
            shaders: HashMap::new(),
        }
    }

    pub fn begin_frame(&mut self, canvas: &Canvas) {
        self.draws.clear();
        self.batches.clear();
        self.geometry.vertices.clear();
        self.geometry.indices.clear();
        self.enable_msaa = canvas.surface.msaa();
        self.vbo_range = None;
        self.ibo_range = None;
    }

    pub fn fill<S: Into<Shape>>(&mut self, canvas: &mut Canvas, shape: S, pattern: BuiltPattern) {
        let transform = canvas.transforms.current();
        let z_index = canvas.z_indices.push();
        let new_batch_index = self.batches.len() as u32;

        let shape = shape.into();
        let aabb = shape.aabb();

        let batch_index = canvas.batcher.find_or_add_batch(
            self.renderer_id,
            new_batch_index,
            &pattern.batch_key(),
            &aabb,
            BatchFlags::empty(),
        );

        if batch_index == new_batch_index {
            self.batches.push(Batch {
                commands: Vec::with_capacity(32),
                draws: 0..0,
                surface: canvas.surface.current_state()
            });
        }

        let batch = &mut self.batches[batch_index as usize];
        batch.commands.push(Fill { shape, pattern, transform, z_index });
    }

    pub fn prepare(&mut self, canvas: &Canvas) {
        if self.batches.is_empty() {
            return;
        }

        let id = self.renderer_id;
        let mut batches = std::mem::take(&mut self.batches);
        for batch_id in canvas.batcher.batches()
            .iter()
            .filter(|batch| batch.renderer == id) {

            let batch = &mut batches[batch_id.index as usize];

            let surface = batch.surface;

            let draw_start = self.draws.len() as u32;
            let mut key = batch.commands.first().as_ref().unwrap().pattern.shader_and_bindings();

            // Opaque pass.
            let mut geom_start = self.geometry.indices.len() as u32;
            if surface.depth {
                for fill in batch.commands.iter().rev().filter(|fill| fill.pattern.is_opaque) {
                    if key != fill.pattern.shader_and_bindings() {
                        let end = self.geometry.indices.len() as u32;
                        if end > geom_start {
                            let state = canvas.surface.state(batch_id.surface);
                            self.shaders.entry((true, key.0, state)).or_insert(None);
                            self.draws.push(Draw {
                                indices: geom_start..end,
                                pattern: key.0,
                                pattern_inputs: key.1,
                                opaque: true,
                            });
                        }
                        geom_start = end;
                        key = fill.pattern.shader_and_bindings();
                    }
                    self.prepare_fill(fill, canvas);
                }
            }

            let end = self.geometry.indices.len() as u32;
            if end > geom_start {
                let state = canvas.surface.state(batch_id.surface);
                self.shaders.entry((true, key.0, state)).or_insert(None);
                self.draws.push(Draw {
                    indices: geom_start..end,
                    pattern: key.0,
                    pattern_inputs: key.1,
                    opaque: true,
                });
            }
            geom_start = end;

            // Blended pass.
            for fill in batch.commands.iter().filter(|fill| !surface.depth || !fill.pattern.is_opaque) {
                if key != fill.pattern.shader_and_bindings() {
                    let end = self.geometry.indices.len() as u32;
                    if end > geom_start {
                        let state = canvas.surface.state(batch_id.surface);
                        self.shaders.entry((false, key.0, state)).or_insert(None);
                        self.draws.push(Draw {
                            indices: geom_start..end,
                            pattern: key.0,
                            pattern_inputs: key.1,
                            opaque: false,
                        });
                    }
                    geom_start = end;
                    key = fill.pattern.shader_and_bindings();
                }
                self.prepare_fill(fill, canvas);
            }

            let end = self.geometry.indices.len() as u32;
            if end > geom_start {
                let state = canvas.surface.state(batch_id.surface);
                self.shaders.entry((false, key.0, state)).or_insert(None);
                self.draws.push(Draw {
                    indices: geom_start..end,
                    pattern: key.0,
                    pattern_inputs: key.1,
                    opaque: false,
                });
            }

            let draws = draw_start .. self.draws.len() as u32;
            batch.draws = draws;
        }

        self.batches = batches;
    }

    fn prepare_fill(&mut self, fill: &Fill, canvas: &Canvas) {
        let transform = canvas.transforms.get(fill.transform);

        match &fill.shape {
            Shape::Path(shape) => {
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
            Shape::Circle(circle) => {
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

        for (&(opaque, pattern, surface), shader_id) in &mut self.shaders {
            if shader_id.is_none() {
                let surface = surface.surface_config(true, None);
                let id = if opaque { opaque_pipeline } else { alpha_pipeline };
                shaders.prepare_pipeline(device, id, pattern, surface);
            }
        }
    }
}

impl CanvasRenderer for MeshRenderer {
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

        let batch = &self.batches[index as usize];

        render_pass.set_bind_group(0, &common_resources.main_target_and_gpu_store_bind_group, &[]);

        render_pass.set_index_buffer(common_resources.indices.get_buffer_slice(self.ibo_range.as_ref().unwrap()), wgpu::IndexFormat::Uint32);
        render_pass.set_vertex_buffer(0, common_resources.vertices.get_buffer_slice(self.vbo_range.as_ref().unwrap()));

        let surface = surface_info.surface_config(true, None);

        let mut helper = DrawHelper::new();

        for draw in &self.draws[usize_range(batch.draws.clone())] {
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

