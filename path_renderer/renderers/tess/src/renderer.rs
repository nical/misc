use core::{
    batching::{BatchFlags, BatchList, BatchId},
    bytemuck,
    context::{
        Renderer, Context, DrawHelper, RenderContext, RenderPassState, RendererId,
        SurfacePassConfig, ZIndex, FillPath,
    },
    gpu::{
        shader::{
            BaseShaderId, PrepareRenderPipelines, RenderPipelineIndex, RenderPipelineKey, BlendMode,
        },
        DynBufferRange,
    },
    path::Path,
    pattern::{BindingsId, BuiltPattern},
    resources::{CommonGpuResources, GpuResources, ResourcesHandle},
    shape::{Circle, FilledPath},
    transform::{TransformId, Transforms},
    units::{point, LocalPoint, LocalRect},
    usize_range, wgpu,
};
use lyon::{
    geom::euclid::vec2,
    lyon_tessellation::{StrokeOptions, StrokeTessellator, StrokeVertexConstructor},
    path::traits::PathIterator,
    tessellation::{
        BuffersBuilder, FillOptions, FillTessellator, FillVertexConstructor, VertexBuffers,
    },
};
use std::ops::Range;

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
    surface: SurfacePassConfig,
    blend_mode: BlendMode,
}

enum Shape {
    Path(FilledPath),
    Rect(LocalRect),
    Circle(Circle),
    Mesh(TessellatedMesh),
    StrokePath(Path, f32),
}

impl Shape {
    pub fn aabb(&self) -> LocalRect {
        match self {
            // TODO: return the correct aabb for inverted shapes.
            Shape::Path(shape) => *shape.path.aabb(),
            Shape::Rect(rect) => *rect,
            Shape::Circle(circle) => circle.aabb(),
            Shape::Mesh(mesh) => mesh.aabb,
            Shape::StrokePath(path, width) => path.aabb().inflate(*width, *width),
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
    pattern: u32,
    z_index: u32,
}

impl FillVertexConstructor<Vertex> for VertexCtor {
    fn new_vertex(&mut self, vertex: lyon::lyon_tessellation::FillVertex) -> Vertex {
        let (x, y) = vertex.position().to_tuple();
        Vertex {
            x,
            y,
            z_index: self.z_index,
            pattern: self.pattern,
        }
    }
}
impl StrokeVertexConstructor<Vertex> for VertexCtor {
    fn new_vertex(&mut self, vertex: lyon::lyon_tessellation::StrokeVertex) -> Vertex {
        let (x, y) = vertex.position().to_tuple();
        Vertex {
            x,
            y,
            z_index: self.z_index,
            pattern: self.pattern,
        }
    }
}

struct Draw {
    indices: Range<u32>,
    pattern_inputs: BindingsId,
    pipeline_idx: RenderPipelineIndex,
}

pub struct MeshRenderer {
    renderer_id: RendererId,
    common_resources: ResourcesHandle<CommonGpuResources>,
    _resources: ResourcesHandle<MeshGpuResources>,
    tessellator: FillTessellator,
    geometry: VertexBuffers<Vertex, u32>,
    tolerenace: f32,
    enable_msaa: bool,

    batches: BatchList<Fill, BatchInfo>,
    draws: Vec<Draw>,
    vbo_range: Option<DynBufferRange>,
    ibo_range: Option<DynBufferRange>,
    base_shader: BaseShaderId,
}

impl MeshRenderer {
    pub fn new(
        renderer_id: RendererId,
        common_resources: ResourcesHandle<CommonGpuResources>,
        resources: ResourcesHandle<MeshGpuResources>,
        res: &MeshGpuResources,
    ) -> Self {
        MeshRenderer {
            renderer_id,
            common_resources,
            _resources: resources,
            tessellator: FillTessellator::new(),
            geometry: VertexBuffers::new(),
            tolerenace: 0.25,
            enable_msaa: false,

            draws: Vec::new(),
            batches: BatchList::new(renderer_id),
            vbo_range: None,
            ibo_range: None,
            base_shader: res.base_shader,
        }
    }

    pub fn supports_surface(&self, _surface: SurfacePassConfig) -> bool {
        true
    }

    pub fn begin_frame(&mut self, ctx: &Context) {
        self.draws.clear();
        self.batches.clear();
        self.geometry.vertices.clear();
        self.geometry.indices.clear();
        self.enable_msaa = ctx.surface.msaa();
        self.vbo_range = None;
        self.ibo_range = None;
    }

    pub fn fill_path<P: Into<FilledPath>>(
        &mut self,
        ctx: &mut Context,
        transforms: &Transforms,
        path: P,
        pattern: BuiltPattern,
    ) {
        self.fill_shape(ctx, transforms, Shape::Path(path.into()), pattern);
    }

    pub fn stroke_path(
        &mut self,
        ctx: &mut Context,
        transforms: &Transforms,
        path: Path,
        width: f32,
        pattern: BuiltPattern,
    ) {
        self.fill_shape(ctx, transforms, Shape::StrokePath(path, width), pattern);
    }

    pub fn fill_rect(&mut self, ctx: &mut Context, transforms: &Transforms, rect: LocalRect, pattern: BuiltPattern) {
        self.fill_shape(ctx, transforms, Shape::Rect(rect), pattern);
    }

    pub fn fill_circle(&mut self, ctx: &mut Context, transforms: &Transforms, circle: Circle, pattern: BuiltPattern) {
        self.fill_shape(ctx, transforms, Shape::Circle(circle), pattern);
    }

    pub fn fill_mesh(
        &mut self,
        ctx: &mut Context,
        transforms: &Transforms,
        mesh: TessellatedMesh,
        pattern: BuiltPattern,
    ) {
        self.fill_shape(ctx, transforms, Shape::Mesh(mesh), pattern);
    }

    fn fill_shape(&mut self, ctx: &mut Context, transforms: &Transforms, shape: Shape, pattern: BuiltPattern) {
        let transform = transforms.current_id();
        let z_index = ctx.z_indices.push();

        let aabb = transforms
            .get_current()
            .matrix()
            .outer_transformed_box(&shape.aabb());

        let mut batch_flags = BatchFlags::empty();
        if pattern.is_opaque && ctx.surface.current_config().depth {
            batch_flags |= BatchFlags::ORDER_INDEPENDENT;
        }

        let (commands, info) = self.batches.find_or_add_batch(
            &mut ctx.batcher,
            &pattern.batch_key(),
            &aabb,
            batch_flags,
            &mut || BatchInfo {
                draws: 0..0,
                surface: ctx.surface.current_config(),
                blend_mode: pattern.blend_mode,
            },
        );
        info.surface = ctx.surface.current_config();
        commands.push(Fill {
            shape,
            pattern,
            transform,
            z_index,
        });
    }

    pub fn prepare(&mut self, ctx: &Context, transforms: &Transforms, shaders: &mut PrepareRenderPipelines) {
        if self.batches.is_empty() {
            return;
        }

        let id = self.renderer_id;
        let mut batches = self.batches.take();
        for batch_id in ctx
            .batcher
            .batches()
            .iter()
            .filter(|batch| batch.renderer == id)
        {
            let (commands, info) = &mut batches.get_mut(batch_id.index);

            let surface = info.surface;

            let draw_start = self.draws.len() as u32;
            let mut key = commands
                .first()
                .as_ref()
                .unwrap()
                .pattern
                .shader_and_bindings();

            // Opaque pass.
            let mut geom_start = self.geometry.indices.len() as u32;
            if surface.depth {
                for fill in commands.iter().rev().filter(|fill| fill.pattern.is_opaque) {
                    if key != fill.pattern.shader_and_bindings() {
                        let end = self.geometry.indices.len() as u32;
                        if end > geom_start {
                            self.draws.push(Draw {
                                indices: geom_start..end,
                                pattern_inputs: key.1,
                                pipeline_idx: shaders.prepare(RenderPipelineKey::new(
                                    self.base_shader,
                                    key.0,
                                    info.blend_mode,
                                    surface.draw_config(true, None),
                                )),
                            });
                        }
                        geom_start = end;
                        key = fill.pattern.shader_and_bindings();
                    }
                    self.prepare_fill(fill, transforms);
                }
            }

            let end = self.geometry.indices.len() as u32;
            if end > geom_start {
                self.draws.push(Draw {
                    indices: geom_start..end,
                    pattern_inputs: key.1,
                    pipeline_idx: shaders.prepare(RenderPipelineKey::new(
                        self.base_shader,
                        key.0,
                        info.blend_mode,
                        surface.draw_config(true, None),
                    )),
                });
            }
            geom_start = end;

            // Blended pass.
            for fill in commands
                .iter()
                .filter(|fill| !surface.depth || !fill.pattern.is_opaque)
            {
                if key != fill.pattern.shader_and_bindings() {
                    let end = self.geometry.indices.len() as u32;
                    if end > geom_start {
                        self.draws.push(Draw {
                            indices: geom_start..end,
                            pattern_inputs: key.1,
                            pipeline_idx: shaders.prepare(RenderPipelineKey::new(
                                self.base_shader,
                                key.0,
                                info.blend_mode,
                                surface.draw_config(true, None),
                            )),
                        });
                    }
                    geom_start = end;
                    key = fill.pattern.shader_and_bindings();
                }
                self.prepare_fill(fill, transforms);
            }

            let end = self.geometry.indices.len() as u32;
            if end > geom_start {
                self.draws.push(Draw {
                    indices: geom_start..end,
                    pattern_inputs: key.1,
                    pipeline_idx: shaders.prepare(RenderPipelineKey::new(
                        self.base_shader,
                        key.0,
                        info.blend_mode,
                        surface.draw_config(true, None),
                    )),
                });
            }

            let draws = draw_start..self.draws.len() as u32;
            info.draws = draws;
        }

        self.batches = batches;
    }

    fn prepare_fill(&mut self, fill: &Fill, transforms: &Transforms) {
        let transform = transforms.get(fill.transform).matrix();
        let z_index = fill.z_index;
        let pattern = fill.pattern.data;

        match &fill.shape {
            Shape::Path(shape) => {
                let transform = transform.to_untyped();
                let options =
                    FillOptions::tolerance(self.tolerenace).with_fill_rule(shape.fill_rule);

                // TODO: some way to simplify/discard offscreen geometry, would probably be best
                // done in the tessellator itself.
                self.tessellator
                    .tessellate(
                        shape.path.iter().transformed(&transform),
                        &options,
                        &mut BuffersBuilder::new(
                            &mut self.geometry,
                            VertexCtor {
                                z_index: fill.z_index,
                                pattern: fill.pattern.data,
                            },
                        ),
                    )
                    .unwrap();
            }
            Shape::Circle(circle) => {
                let options = FillOptions::tolerance(self.tolerenace);
                self.tessellator
                    .tessellate_circle(
                        transform.transform_point(circle.center).cast_unit(),
                        // TODO: that's not quite right if the transform has more than scale+offset
                        transform
                            .transform_vector(vec2(circle.radius, 0.0))
                            .length(),
                        &options,
                        &mut BuffersBuilder::new(
                            &mut self.geometry,
                            VertexCtor {
                                z_index: fill.z_index,
                                pattern: fill.pattern.data,
                            },
                        ),
                    )
                    .unwrap();
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

                self.geometry.vertices.push(Vertex {
                    x: a.x,
                    y: a.y,
                    z_index,
                    pattern,
                });
                self.geometry.vertices.push(Vertex {
                    x: b.x,
                    y: b.y,
                    z_index,
                    pattern,
                });
                self.geometry.vertices.push(Vertex {
                    x: c.x,
                    y: c.y,
                    z_index,
                    pattern,
                });
                self.geometry.vertices.push(Vertex {
                    x: d.x,
                    y: d.y,
                    z_index,
                    pattern,
                });
                self.geometry.indices.push(vtx_offset);
                self.geometry.indices.push(vtx_offset + 1);
                self.geometry.indices.push(vtx_offset + 2);
                self.geometry.indices.push(vtx_offset);
                self.geometry.indices.push(vtx_offset + 2);
                self.geometry.indices.push(vtx_offset + 3);
            }
            Shape::StrokePath(path, width) => {
                let transform = transform.to_untyped();
                let options = StrokeOptions::tolerance(self.tolerenace).with_line_width(*width);

                // TODO: some way to simplify/discard offscreen geometry, would probably be best
                // done in the tessellator itself.
                StrokeTessellator::new()
                    .tessellate(
                        path.iter().transformed(&transform),
                        &options,
                        &mut BuffersBuilder::new(
                            &mut self.geometry,
                            VertexCtor {
                                z_index: fill.z_index,
                                pattern: fill.pattern.data,
                            },
                        ),
                    )
                    .unwrap();
            }
        }
    }

    pub fn upload(
        &mut self,
        resources: &mut GpuResources,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) {
        let res = &mut resources[self.common_resources];
        self.vbo_range = res
            .vertices
            .upload(device, bytemuck::cast_slice(&self.geometry.vertices));
        self.ibo_range = res
            .indices
            .upload(device, bytemuck::cast_slice(&self.geometry.indices));
    }
}

impl Renderer for MeshRenderer {
    fn render<'pass, 'resources: 'pass>(
        &self,
        batches: &[BatchId],
        _surface_info: &RenderPassState,
        ctx: RenderContext<'resources>,
        render_pass: &mut wgpu::RenderPass<'pass>,
    ) {
        let common_resources = &ctx.resources[self.common_resources];

        render_pass.set_bind_group(
            0,
            &common_resources.main_target_and_gpu_store_bind_group,
            &[],
        );
        render_pass.set_index_buffer(
            common_resources
                .indices
                .get_buffer_slice(self.ibo_range.as_ref().unwrap()),
            wgpu::IndexFormat::Uint32,
        );
        render_pass.set_vertex_buffer(
            0,
            common_resources
                .vertices
                .get_buffer_slice(self.vbo_range.as_ref().unwrap()),
        );

        let mut helper = DrawHelper::new();

        for batch_id in batches {
            let (_, batch_info) = self.batches.get(batch_id.index);
            for draw in &self.draws[usize_range(batch_info.draws.clone())] {
                let pipeline = ctx.render_pipelines.get(draw.pipeline_idx).unwrap();

                helper.resolve_and_bind(1, draw.pattern_inputs, ctx.bindings, render_pass);

                render_pass.set_pipeline(pipeline);
                render_pass.draw_indexed(draw.indices.clone(), 0, 0..1);
            }
        }
    }
}

impl FillPath for MeshRenderer {
    fn fill_path(
        &mut self,
        ctx: &mut Context,
        transforms: &Transforms,
        path: FilledPath,
        pattern: BuiltPattern,
    ) {
        self.fill_shape(ctx, transforms, Shape::Path(path), pattern);
    }
}
