use lyon::{
    tessellation::{FillTessellator, FillOptions, VertexBuffers, FillVertexConstructor, BuffersBuilder},
    path::traits::PathIterator, geom::euclid::vec2
};
use core::{
    shape::{PathShape, Circle},
    canvas::{RendererId, Canvas, CanvasRenderer, ZIndex, RenderPassState, DrawHelper, SurfaceState, SubPass},
    resources::{ResourcesHandle, GpuResources, CommonGpuResources},
    gpu::{
        shader::{ShaderPatternId},
        DynBufferRange, Shaders
    },
    pattern::{BuiltPattern, BindingsId}, usize_range,
    bytemuck,
    wgpu, BindingResolver, batching::{BatchFlags, BatchList}, transform::TransformId, units::{LocalPoint, LocalRect, point},
};
use std::{ops::Range, collections::HashMap};

use super::MeshGpuResources;

pub const PATTERN_KIND_COLOR: u32 = 0;
pub const PATTERN_KIND_SIMPLE_LINEAR_GRADIENT: u32 = 1;

pub struct TessellatedMesh {
    pub vertices: Vec<LocalPoint>,
    pub indices: Vec<u32>,
    pub aabb: LocalRect,
}

struct BatchInfo {
    draws: Range<u32>,
    surface: SurfaceState,
}

enum Shape {
    Path(PathShape),
    Rect(LocalRect),
    Circle(Circle),
    Mesh(TessellatedMesh),
}

impl Shape {
    pub fn aabb(&self) -> LocalRect {
        match self {
            // TODO: return the correct aabb for inverted shapes.
            Shape::Path(shape) => *shape.path.aabb(),
            Shape::Rect(rect) => *rect,
            Shape::Circle(circle) => circle.aabb(),
            Shape::Mesh(mesh) => mesh.aabb
        }
    }
}

struct Fill {
    shape: Shape,
    pattern: BuiltPattern,
    transform: TransformId,
    z_index: ZIndex,
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

    batches: BatchList<Fill, BatchInfo>,
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
            batches: BatchList::new(renderer_id),
            vbo_range: None,
            ibo_range: None,
            shaders: HashMap::new(),
        }
    }

    pub fn supports_surface(&self, _surface: SurfaceState) -> bool {
        true
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

    pub fn fill_path<P: Into<PathShape>>(&mut self, canvas: &mut Canvas, path: P, pattern: BuiltPattern) {
        self.fill_shape(canvas, Shape::Path(path.into()), pattern);
    }

    pub fn fill_rect(&mut self, canvas: &mut Canvas, rect: LocalRect, pattern: BuiltPattern) {
        self.fill_shape(canvas, Shape::Rect(rect), pattern);
    }

    pub fn fill_circle(&mut self, canvas: &mut Canvas, circle: Circle, pattern: BuiltPattern) {
        self.fill_shape(canvas, Shape::Circle(circle), pattern);
    }

    pub fn fill_mesh(&mut self, canvas: &mut Canvas, mesh: TessellatedMesh, pattern: BuiltPattern) {
        self.fill_shape(canvas, Shape::Mesh(mesh), pattern);
    }

    fn fill_shape(&mut self, canvas: &mut Canvas, shape: Shape, pattern: BuiltPattern) {
        let transform = canvas.transforms.current_id();
        let z_index = canvas.z_indices.push();

        let aabb = canvas.transforms.get_current().matrix().outer_transformed_box(&shape.aabb());

        let (commands, info) = self.batches.find_or_add_batch(
            &mut canvas.batcher,
            &pattern.batch_key(),
            &aabb,
            BatchFlags::empty(),
            &mut || BatchInfo {
                draws: 0..0,
                surface: canvas.surface.current_state(),
            },
        );
        info.surface = canvas.surface.current_state();
        commands.push(Fill { shape, pattern, transform, z_index });
    }

    pub fn prepare(&mut self, canvas: &Canvas) {
        if self.batches.is_empty() {
            return;
        }

        let id = self.renderer_id;
        let mut batches = self.batches.take();
        for batch_id in canvas.batcher.batches()
            .iter()
            .filter(|batch| batch.renderer == id) {

            let (commands, info) = &mut batches.get_mut(batch_id.index);

            let surface = info.surface;

            let draw_start = self.draws.len() as u32;
            let mut key = commands.first().as_ref().unwrap().pattern.shader_and_bindings();

            // Opaque pass.
            let mut geom_start = self.geometry.indices.len() as u32;
            if surface.depth {
                for fill in commands.iter().rev().filter(|fill| fill.pattern.is_opaque) {
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
            for fill in commands.iter().filter(|fill| !surface.depth || !fill.pattern.is_opaque) {
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
            info.draws = draws;
        }

        self.batches = batches;
    }

    fn prepare_fill(&mut self, fill: &Fill, canvas: &Canvas) {
        let transform = canvas.transforms.get(fill.transform).matrix();
        let z_index = fill.z_index;
        let pattern = fill.pattern.data;

        match &fill.shape {
            Shape::Path(shape) => {
                let transform = transform.to_untyped();
                let options = FillOptions::tolerance(self.tolerenace)
                    .with_fill_rule(shape.fill_rule);

                // TODO: some way to simplify/discard offscreen geometry, would probably be best
                // done in the tessellator itself.
                self.tessellator.tessellate(
                    shape.path.iter().transformed(&transform),
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
                    transform.transform_point(circle.center).cast_unit(),
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
            Shape::Mesh(mesh) => {
                let vtx_offset = self.geometry.vertices.len() as u32;
                for vertex in &mesh.vertices {
                    let pos = transform.transform_point(*vertex);
                    self.geometry.vertices.push(Vertex {
                        x: pos.x,
                        y: pos.y,
                        z_index: fill.z_index,
                        pattern: fill.pattern.data,
                    });
                }
                for index in &mesh.indices {
                    self.geometry.indices.push(*index + vtx_offset);
                }
            }
            Shape::Rect(rect) => {
                let vtx_offset = self.geometry.vertices.len() as u32;
                let a = transform.transform_point(rect.min);
                let b = transform.transform_point(point(rect.max.x, rect.min.y));
                let c = transform.transform_point(rect.max);
                let d = transform.transform_point(point(rect.min.x, rect.max.y));

                self.geometry.vertices.push(Vertex { x: a.x, y: a.y, z_index, pattern });
                self.geometry.vertices.push(Vertex { x: b.x, y: b.y, z_index, pattern });
                self.geometry.vertices.push(Vertex { x: c.x, y: c.y, z_index, pattern });
                self.geometry.vertices.push(Vertex { x: d.x, y: d.y, z_index, pattern });
                self.geometry.indices.push(vtx_offset);
                self.geometry.indices.push(vtx_offset + 1);
                self.geometry.indices.push(vtx_offset + 2);
                self.geometry.indices.push(vtx_offset);
                self.geometry.indices.push(vtx_offset + 2);
                self.geometry.indices.push(vtx_offset + 3);
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
        sub_passes: &[SubPass],
        surface_info: &RenderPassState,
        shaders: &'resources Shaders,
        resources: &'resources GpuResources,
        bindings: &'resources dyn BindingResolver,
        render_pass: &mut wgpu::RenderPass<'pass>,
    ) {
        let common_resources = &resources[self.common_resources];
        let mesh_resources = &resources[self.resources];

        render_pass.set_bind_group(0, &common_resources.main_target_and_gpu_store_bind_group, &[]);
        render_pass.set_index_buffer(common_resources.indices.get_buffer_slice(self.ibo_range.as_ref().unwrap()), wgpu::IndexFormat::Uint32);
        render_pass.set_vertex_buffer(0, common_resources.vertices.get_buffer_slice(self.vbo_range.as_ref().unwrap()));

        let surface = surface_info.surface_config(true, None);

        let mut helper = DrawHelper::new();

        for sub_pass in sub_passes {
            let (_, batch_info) = self.batches.get(sub_pass.internal_index);
            for draw in &self.draws[usize_range(batch_info.draws.clone())] {
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
}

