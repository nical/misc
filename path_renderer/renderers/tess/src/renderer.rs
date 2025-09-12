use core::{
    bytemuck, path::Path, pattern::BuiltPattern, render_pass::{BuiltRenderPass, RenderCommandId}, render_task::RenderTaskHandle, shape::{Circle, FilledPath}, wgpu, BindingsId, PrepareContext
};
use core::transform::{TransformId, Transforms};
use core::units::{point, LocalPoint, LocalRect};
use core::gpu::{GpuBufferWriter, GpuStreamWriter, StreamId};
use core::batching::{BatchFlags, BatchList};
use core::render_pass::{RenderPassContext, RendererId, RenderPassConfig, ZIndex};
use core::shading::{GeometryId, BlendMode, RenderPipelineIndex, RenderPipelineKey};
use core::utils::{DrawHelper, usize_range};

use lyon::{
    geom::euclid::vec2,
    lyon_tessellation::{StrokeOptions, StrokeTessellator},
    path::traits::PathIterator,
    tessellation::{
        FillGeometryBuilder, FillOptions, FillTessellator, StrokeGeometryBuilder, VertexId
    },
};
use std::{ops::Range, sync::Arc};

pub const PATTERN_KIND_COLOR: u32 = 0;
pub const PATTERN_KIND_SIMPLE_LINEAR_GRADIENT: u32 = 1;

pub struct TessellatedMesh {
    pub vertices: Vec<LocalPoint>,
    pub indices: Vec<u32>,
    pub aabb: LocalRect,
}

struct BatchInfo {
    draws: Range<u32>,
    blend_mode: BlendMode,
}

#[derive(Clone)]
enum Shape {
    Path(FilledPath),
    Rect(LocalRect),
    Circle(Circle),
    Mesh(Arc<TessellatedMesh>),
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
    render_task: RenderTaskHandle,
}

pub struct PrimitiveInfo {
    pub z_index: u32,
    pub pattern: u32,
    pub opacity: f32,
    pub render_task: RenderTaskHandle,
}

pub type EncodedPrimitiveInfo = [u32; 4];

impl PrimitiveInfo {
    pub fn encode(&self) -> EncodedPrimitiveInfo {
        [
            self.z_index,
            self.pattern,
            (self.opacity.max(0.0).min(1.0) * 65535.0) as u32,
            self.render_task.to_u32(),
        ]
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Vertex {
    pub x: f32,
    pub y: f32,
    pub prim_address: u32,
    pub _pad: f32,
}

unsafe impl bytemuck::Zeroable for Vertex {}
unsafe impl bytemuck::Pod for Vertex {}

struct GeomBuilder<'a, 'b, 'c> {
    indices: &'a mut GpuStreamWriter<'b>,
    vertices: &'a mut GpuBufferWriter<'c>,

    prim_address: u32,
}

impl<'a, 'b, 'c> lyon::tessellation::GeometryBuilder for GeomBuilder<'a ,'b, 'c> {
    fn add_triangle(&mut self, a: VertexId, b: VertexId, c: VertexId) {
        self.indices.push(a.0);
        self.indices.push(b.0);
        self.indices.push(c.0);
    }
}

impl<'a, 'b, 'c> FillGeometryBuilder for GeomBuilder<'a, 'b, 'c> {
    fn add_fill_vertex(
        &mut self,
        vertex: lyon::tessellation::FillVertex<'_>,
    ) -> Result<VertexId, lyon::tessellation::GeometryBuilderError> {
        let (x, y) = vertex.position().to_tuple();
        let handle = self.vertices.push(Vertex {
            x,
            y,
            prim_address: self.prim_address,
            _pad: 0.0,
        });

        Ok(VertexId(handle.to_u32()))
    }
}

impl<'a, 'b, 'c> StrokeGeometryBuilder for GeomBuilder<'a, 'b, 'c> {
    fn add_stroke_vertex(
        &mut self,
        vertex: lyon::tessellation::StrokeVertex,
    ) -> Result<VertexId, lyon::tessellation::GeometryBuilderError> {
        let (x, y) = vertex.position().to_tuple();
        let handle = self.vertices.push(Vertex {
            x,
            y,
            prim_address: self.prim_address,
            _pad: 0.0,
        });

        Ok(VertexId(handle.to_u32()))
    }
}

struct Draw {
    indices: Range<u32>,
    pattern_inputs: BindingsId,
    pipeline_idx: RenderPipelineIndex,
}

pub struct MeshRenderer {
    renderer_id: RendererId,
    tessellator: FillTessellator,
    pub tolerance: f32,

    batches: BatchList<Fill, BatchInfo>,
    draws: Vec<Draw>,
    indices: Option<StreamId>,
    geometry: GeometryId,
}

impl MeshRenderer {
    pub(crate) fn new(
        renderer_id: RendererId,
        geometry: GeometryId,
    ) -> Self {
        MeshRenderer {
            renderer_id,
            tessellator: FillTessellator::new(),
            tolerance: 0.25,

            draws: Vec::new(),
            batches: BatchList::new(renderer_id),
            indices: None,
            geometry,
        }
    }

    pub fn supports_surface(&self, _surface: RenderPassConfig) -> bool {
        true
    }

    pub fn begin_frame(&mut self) {
        self.draws.clear();
        self.batches.clear();
        self.indices = None;
    }

    pub fn fill_path<P: Into<FilledPath>>(
        &mut self,
        ctx: &mut RenderPassContext,
        transforms: &Transforms,
        path: P,
        pattern: BuiltPattern,
    ) {
        self.fill_shape(ctx, transforms, Shape::Path(path.into()), pattern);
    }

    pub fn stroke_path(
        &mut self,
        ctx: &mut RenderPassContext,
        transforms: &Transforms,
        path: Path,
        width: f32,
        pattern: BuiltPattern,
    ) {
        self.fill_shape(ctx, transforms, Shape::StrokePath(path, width), pattern);
    }

    pub fn fill_rect(&mut self, ctx: &mut RenderPassContext, transforms: &Transforms, rect: LocalRect, pattern: BuiltPattern) {
        self.fill_shape(ctx, transforms, Shape::Rect(rect), pattern);
    }

    pub fn fill_circle(&mut self, ctx: &mut RenderPassContext, transforms: &Transforms, circle: Circle, pattern: BuiltPattern) {
        self.fill_shape(ctx, transforms, Shape::Circle(circle), pattern);
    }

    pub fn fill_mesh(
        &mut self,
        ctx: &mut RenderPassContext,
        transforms: &Transforms,
        mesh: Arc<TessellatedMesh>,
        pattern: BuiltPattern,
    ) {
        self.fill_shape(ctx, transforms, Shape::Mesh(mesh), pattern);
    }

    fn fill_shape(&mut self, ctx: &mut RenderPassContext, transforms: &Transforms, shape: Shape, pattern: BuiltPattern) {
        let transform = transforms.current_id();
        let z_index = ctx.z_indices.push();

        let aabb = transforms
            .get_current()
            .matrix()
            .outer_transformed_box(&shape.aabb());

        let mut batch_flags = BatchFlags::empty();
        if pattern.is_opaque && ctx.config.depth {
            batch_flags |= BatchFlags::ORDER_INDEPENDENT;
        }

        self.batches.add(
            ctx,
            &pattern.batch_key(),
            &aabb,
            batch_flags,
            &mut || BatchInfo {
                draws: 0..0,
                blend_mode: pattern.blend_mode,
            },
            &mut |mut batch, task| {
                batch.push(Fill {
                    shape: shape.clone(),
                    pattern,
                    transform,
                    z_index,
                    render_task: task.handle,
                });
            }
        );
    }

    pub fn prepare_impl(&mut self, ctx: &mut PrepareContext, passes: &[BuiltRenderPass]) {
        if self.batches.is_empty() {
            return;
        }

        let transforms = &ctx.transforms;
        let worker_data = &mut ctx.workers.data();
        let shaders = &mut worker_data.pipelines;
        let indices = &mut worker_data.indices;

        let mut vertices = worker_data.vertices.write_items::<Vertex>();
        let mut prim_buffer = worker_data.u32_buffer.write_items::<EncodedPrimitiveInfo>();

        let idx_stream = indices.next_stream_id();
        let mut indices = indices.write(idx_stream, 0);
        self.indices = Some(idx_stream);

        let id = self.renderer_id;
        let mut batches = self.batches.take();

        for pass in passes {
            for batch_id in pass
                .batches()
                .iter()
                .filter(|batch| batch.renderer == id)
            {
                let (commands, surface, info) = &mut batches.get_mut(batch_id.index);

                let draw_start = self.draws.len() as u32;
                let mut key = commands
                    .first()
                    .as_ref()
                    .unwrap()
                    .pattern
                    .shader_and_bindings();

                // Opaque pass.
                let mut geom_start = indices.pushed_bytes() / 4;
                if surface.depth {
                    for fill in commands.iter().rev().filter(|fill| fill.pattern.is_opaque) {
                        if key != fill.pattern.shader_and_bindings() {
                            let end = indices.pushed_bytes() / 4;
                            if end > geom_start {
                                self.draws.push(Draw {
                                    indices: geom_start..end,
                                    pattern_inputs: key.1,
                                    pipeline_idx: shaders.prepare(RenderPipelineKey::new(
                                        self.geometry,
                                        key.0,
                                        info.blend_mode,
                                        surface.draw_config(true, None),
                                    )),
                                });
                            }
                            geom_start = end;
                            key = fill.pattern.shader_and_bindings();
                        }
                        self.prepare_fill(fill, transforms, &mut vertices, &mut indices, &mut prim_buffer);
                    }
                }

                let end = indices.pushed_bytes() / 4;
                if end > geom_start {
                    self.draws.push(Draw {
                        indices: geom_start..end,
                        pattern_inputs: key.1,
                        pipeline_idx: shaders.prepare(RenderPipelineKey::new(
                            self.geometry,
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
                        let end = indices.pushed_bytes() / 4;
                        if end > geom_start {
                            self.draws.push(Draw {
                                indices: geom_start..end,
                                pattern_inputs: key.1,
                                pipeline_idx: shaders.prepare(RenderPipelineKey::new(
                                    self.geometry,
                                    key.0,
                                    info.blend_mode,
                                    surface.draw_config(true, None),
                                )),
                            });
                        }
                        geom_start = end;
                        key = fill.pattern.shader_and_bindings();
                    }
                    self.prepare_fill(fill, transforms, &mut vertices, &mut indices, &mut prim_buffer);
                }

                let end = indices.pushed_bytes() / 4;
                if end > geom_start {
                    self.draws.push(Draw {
                        indices: geom_start..end,
                        pattern_inputs: key.1,
                        pipeline_idx: shaders.prepare(RenderPipelineKey::new(
                            self.geometry,
                            key.0,
                            info.blend_mode,
                            surface.draw_config(true, None),
                        )),
                    });
                }

                let draws = draw_start..self.draws.len() as u32;
                info.draws = draws;
            }
        }

        self.batches = batches;
    }

    fn prepare_fill(&mut self, fill: &Fill, transforms: &Transforms, vertices: &mut GpuBufferWriter, indices: &mut GpuStreamWriter, prim_buffer: &mut GpuBufferWriter) {
        let transform = transforms.get(fill.transform).matrix();

        let prim_address = prim_buffer.push(PrimitiveInfo {
            z_index: fill.z_index,
            pattern: fill.pattern.data,
            opacity: 1.0, // TODO,
            render_task: fill.render_task,
        }.encode()).to_u32();


        match &fill.shape {
            Shape::Path(shape) => {
                let transform = transform.to_untyped();
                let options =
                    FillOptions::tolerance(self.tolerance).with_fill_rule(shape.fill_rule);

                // TODO: some way to simplify/discard offscreen geometry, would probably be best
                // done in the tessellator itself.
                self.tessellator
                    .tessellate(
                        shape.path.iter().transformed(&transform),
                        &options,
                        &mut GeomBuilder {
                            vertices,
                            indices,
                            prim_address,
                        }
                    )
                    .unwrap();
            }
            Shape::Circle(circle) => {
                let options = FillOptions::tolerance(self.tolerance);
                self.tessellator
                    .tessellate_circle(
                        transform.transform_point(circle.center).cast_unit(),
                        // TODO: that's not quite right if the transform has more than scale+offset
                        transform
                            .transform_vector(vec2(circle.radius, 0.0))
                            .length(),
                        &options,
                        &mut GeomBuilder {
                            vertices,
                            indices,
                            prim_address,
                        }
                    )
                    .unwrap();
            }
            Shape::Mesh(_mesh) => {
                todo!();
                //let vtx_offset =
                //for vertex in &mesh.vertices {
                //    let pos = transform.transform_point(*vertex);
                //    vertices.push(&[Vertex {
                //        x: pos.x,
                //        y: pos.y,
                //        z_index: fill.z_index,
                //        pattern: fill.pattern.data,
                //    }]);
                //}
                //for index in &mesh.indices {
                //    indices.push_u32(&[*index + vtx_offset]);
                //}
            }
            Shape::Rect(rect) => {
                let a = transform.transform_point(rect.min);
                let b = transform.transform_point(point(rect.max.x, rect.min.y));
                let c = transform.transform_point(rect.max);
                let d = transform.transform_point(point(rect.min.x, rect.max.y));

                let a = vertices.push(Vertex { x: a.x, y: a.y, prim_address: prim_address, _pad: 0.0 }).to_u32();
                let b = vertices.push(Vertex { x: b.x, y: b.y, prim_address: prim_address, _pad: 0.0 }).to_u32();
                let c = vertices.push(Vertex { x: c.x, y: c.y, prim_address: prim_address, _pad: 0.0 }).to_u32();
                let d = vertices.push(Vertex { x: d.x, y: d.y, prim_address: prim_address, _pad: 0.0 }).to_u32();

                indices.push(a);
                indices.push(b);
                indices.push(c);
                indices.push(a);
                indices.push(c);
                indices.push(d);
            }
            Shape::StrokePath(path, width) => {
                let transform = transform.to_untyped();
                let options = StrokeOptions::tolerance(self.tolerance).with_line_width(*width);

                // TODO: some way to simplify/discard offscreen geometry, would probably be best
                // done in the tessellator itself.
                StrokeTessellator::new()
                    .tessellate(
                        path.iter().transformed(&transform),
                        &options,
                        &mut GeomBuilder {
                            vertices,
                            indices,
                            prim_address,
                        },
                    )
                    .unwrap();
            }
        }
    }
}

impl core::Renderer for MeshRenderer {
    fn prepare(&mut self, ctx: &mut PrepareContext, passes: &[BuiltRenderPass]) {
        self.prepare_impl(ctx, passes);
    }

    fn render<'pass, 'resources: 'pass, 'tmp>(
        &self,
        commands: &[RenderCommandId],
        _surface_info: &RenderPassConfig,
        ctx: core::RenderContext<'resources, 'tmp>,
        render_pass: &mut wgpu::RenderPass<'pass>,
    ) {
        let Some(idx_buffer) = ctx
            .resources
            .common
            .indices
            .resolve_buffer_slice(self.indices) else { return; };

        render_pass.set_index_buffer(idx_buffer, wgpu::IndexFormat::Uint32);
        render_pass.set_vertex_buffer(
            0,
            ctx.resources.common.vertices.as_buffer().unwrap().slice(..),
        );

        let mut helper = DrawHelper::new();

        for batch_id in commands {
            let (_, _, batch_info) = self.batches.get(batch_id.index);
            for draw in &self.draws[usize_range(batch_info.draws.clone())] {
                let pipeline = ctx.render_pipelines.get(draw.pipeline_idx).unwrap();

                helper.resolve_and_bind(1, draw.pattern_inputs, ctx.bindings, render_pass);

                render_pass.set_pipeline(pipeline);
                render_pass.draw_indexed(draw.indices.clone(), 0, 0..1);
                ctx.stats.draw_calls += 1;
            }
        }
    }
}

impl core::FillPath for MeshRenderer {
    fn fill_path(
        &mut self,
        ctx: &mut RenderPassContext,
        transforms: &Transforms,
        path: FilledPath,
        pattern: BuiltPattern,
    ) {
        self.fill_shape(ctx, transforms, Shape::Path(path), pattern);
    }
}
