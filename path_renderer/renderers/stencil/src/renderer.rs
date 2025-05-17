use std::ops::Range;
use std::sync::Arc;

use lyon::{
    tessellation::{FillGeometryBuilder, VertexId, FillOptions, FillTessellator},
    geom::{CubicBezierSegment, QuadraticBezierSegment},
    path::{PathEvent, PathSlice},
};

use crate::resources::StencilAndCoverResources;

use core::wgpu;
use core::gpu::{GpuStoreWriter, GpuStreamWriter, StreamId};
use core::{
    PrepareContext, BindingsId, StencilMode, SurfacePassConfig,
    bytemuck,
};
use core::units::{point, LocalRect, LocalToSurfaceTransform, Point, SurfaceRect};
use core::shading::{BlendMode, ShaderPatternId, GeometryId, RenderPipelineIndex, RenderPipelineKey};
use core::batching::{BatchId, BatchFlags, BatchList};
use core::shape::{Circle, FilledPath};
use core::context::{RenderPassContext, RendererId, ZIndex};
use core::pattern::BuiltPattern;
use core::transform::{Transforms, TransformId};
use core::utils::DrawHelper;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct StencilVertex {
    pub x: f32,
    pub y: f32,
}

impl StencilVertex {
    pub fn from_point(p: Point) -> Self {
        StencilVertex { x: p.x, y: p.y }
    }
}

unsafe impl bytemuck::Pod for StencilVertex {}
unsafe impl bytemuck::Zeroable for StencilVertex {}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct CoverVertex {
    pub x: f32,
    pub y: f32,
    pub z_index: u32,
    pub pattern: u32,
}

struct GeomBuilder<'a, 'b, 'c> {
    indices: &'a mut GpuStreamWriter<'b>,
    vertices: &'a mut GpuStoreWriter<'c>,

    pattern: u32,
    z_index: u32,
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
        let handle = self.vertices.push(CoverVertex {
            x,
            y,
            z_index: self.z_index,
            pattern: self.pattern,
        });

        Ok(VertexId(handle.to_u32()))
    }
}

unsafe impl bytemuck::Pod for CoverVertex {}
unsafe impl bytemuck::Zeroable for CoverVertex {}

#[derive(Clone)]
enum Shape {
    Path(FilledPath),
    Rect(LocalRect),
    Circle(Circle),
}

impl Shape {
    pub fn aabb(&self) -> LocalRect {
        match self {
            // TODO: return the correct aabb for inverted shapes.
            Shape::Path(shape) => *shape.path.aabb(),
            Shape::Rect(rect) => *rect,
            Shape::Circle(circle) => circle.aabb(),
            //Shape::Mesh(mesh) => mesh.aabb
        }
    }
}

pub enum Draw {
    Stencil {
        stream_id: Option<StreamId>,
        indices: Range<u32>,
    },
    Cover {
        stream_id: Option<StreamId>,
        indices: Range<u32>,
        pattern_inputs: BindingsId,
        pipeline_idx: RenderPipelineIndex,
    },
}


pub struct BatchInfo {
    draws: Range<usize>,
    worker_index: Option<u8>,
    pattern_shader: ShaderPatternId,
    pattern_bindings: BindingsId,
    stencil_mode: StencilMode,
    blend_mode: BlendMode,
}

struct Fill {
    shape: Shape,
    pattern: BuiltPattern,
    transform: TransformId,
    z_index: ZIndex,
}

struct Geometry<'a, 'b> {
    vertices: GpuStoreWriter<'a>,
    indices: GpuStreamWriter<'b>,
}

#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub stencil_batches: u32,
    pub cover_batches: u32,
    pub vertices: u32,
}

impl std::ops::AddAssign<Self> for Stats {
    fn add_assign(&mut self, rhs: Self) {
        self.stencil_batches += rhs.stencil_batches;
        self.cover_batches += rhs.cover_batches;
        self.vertices += rhs.vertices;
    }
}

pub struct StencilAndCoverRenderer {
    commands: Vec<Fill>,
    renderer_id: RendererId,
    stencil_vertices: Option<StreamId>,
    stencil_indices: Option<StreamId>,
    cover_indices: Option<StreamId>,
    draws: Vec<Draw>,
    parallel_draws: Vec<Vec<Draw>>,
    batches: BatchList<Fill, BatchInfo>,
    cover_pipeline: GeometryId,
    pub stats: Stats,
    pub tolerance: f32,
    pub parallel: bool,
    shared: Arc<StencilAndCoverResources>,
}

impl StencilAndCoverRenderer {
    pub(crate) fn new(
        shared: Arc<StencilAndCoverResources>,
        renderer_id: RendererId,
    ) -> Self {
        StencilAndCoverRenderer {
            commands: Vec::new(),
            renderer_id,
            stencil_vertices: None,
            stencil_indices: None,
            cover_indices: None,
            draws: Vec::new(),
            parallel_draws: Vec::new(),
            batches: BatchList::new(renderer_id),
            cover_pipeline: shared.cover_geometry,
            stats: Stats {
                stencil_batches: 0,
                cover_batches: 0,
                vertices: 0,
            },
            tolerance: 0.25,
            parallel: false,
            shared,
        }
    }

    pub fn supports_surface(&self, surface: SurfacePassConfig) -> bool {
        surface.stencil
    }

    pub fn begin_frame(&mut self) {
        self.commands.clear();
        self.draws.clear();
        self.batches.clear();
        self.stencil_vertices = None;
        self.stencil_indices = None;
        self.cover_indices = None;
        self.stats = Default::default();
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

    pub fn fill_rect(&mut self, ctx: &mut RenderPassContext, transforms: &Transforms, rect: &LocalRect, pattern: BuiltPattern) {
        self.fill_shape(ctx, transforms, Shape::Rect(*rect), pattern);
    }

    pub fn fill_circle(&mut self, ctx: &mut RenderPassContext, transforms: &Transforms, circle: Circle, pattern: BuiltPattern) {
        self.fill_shape(ctx, transforms, Shape::Circle(circle), pattern);
    }

    fn fill_shape(&mut self, ctx: &mut RenderPassContext, transforms: &Transforms, shape: Shape, pattern: BuiltPattern) {
        debug_assert!(self.supports_surface(ctx.surface));

        let aabb = transforms
            .get_current()
            .matrix()
            .outer_transformed_box(&shape.aabb());

        let stencil_mode: StencilMode = match &shape {
            Shape::Path(shape) => shape.fill_rule.into(),
            Shape::Rect(_) => StencilMode::Ignore,
            Shape::Circle(_) => StencilMode::Ignore,
            //Shape::Canvas => StencilMode::Ignore,
        };

        let stencil_key = match stencil_mode {
            StencilMode::None => 0,
            StencilMode::EvenOdd => 1,
            StencilMode::NonZero => 2,
            StencilMode::Ignore => 3,
        };

        let z_index = ctx.z_indices.push();
        self.batches.add(
            ctx,
            &(pattern.batch_key() | (stencil_key << 32)),
            &aabb,
            BatchFlags::NO_OVERLAP | BatchFlags::EARLIEST_CANDIDATE,
            &mut || BatchInfo {
                draws: 0..0,
                worker_index: None,
                pattern_shader: pattern.shader,
                pattern_bindings: pattern.bindings,
                stencil_mode,
                blend_mode: pattern.blend_mode,
            },
            &mut |mut batch| {
                batch.push(Fill {
                    shape: shape.clone(),
                    pattern,
                    transform: transforms.current_id(),
                    z_index,
                });
            }
        );
    }

    pub fn prepare_single_thread(&mut self, ctx: &mut PrepareContext) {
        if self.batches.is_empty() {
            return;
        }

        let transforms = &ctx.transforms;
        let worker_data = &mut ctx.workers.data();
        let shaders = &mut worker_data.pipelines;
        let vertices = &mut worker_data.vertices;
        let indices = &mut worker_data.indices;

        let stencil_idx_stream = indices.next_stream_id();
        let cover_idx_stream = indices.next_stream_id();
        let mut stencil = Geometry {
            vertices: vertices.write_items::<StencilVertex>(),
            indices: indices.write(stencil_idx_stream, 0),
        };
        let mut cover = Geometry {
            vertices: vertices.write_items::<CoverVertex>(),
            indices: indices.write(cover_idx_stream, 0),
        };

        self.stencil_indices = Some(stencil_idx_stream);
        self.cover_indices = Some(cover_idx_stream);

        let id = self.renderer_id;
        for batch_id in ctx.pass
            .batches()
            .iter()
            .filter(|batch| batch.renderer == id)
        {
            let (commands, surface, batch_info) = self.batches.get_mut(batch_id.index);

            let draws_start = self.draws.len();

            let stencil_idx_start = stencil.indices.pushed_bytes() / 4;
            let cover_idx_start = cover.indices.pushed_bytes() / 4;

            for fill in commands.iter() {
                Self::prepare_fill(transforms, fill, &mut stencil, &mut cover, self.tolerance);
            }

            let stencil_idx_end = stencil.indices.pushed_bytes() / 4;
            let cover_idx_end = cover.indices.pushed_bytes() / 4;

            if stencil_idx_end > stencil_idx_start {
                self.draws.push(Draw::Stencil {
                    stream_id: None,
                    indices: stencil_idx_start..stencil_idx_end,
                });
                self.stats.stencil_batches += 1;
            }

            // Flush the previous cover batch if needed.
            if cover_idx_end > cover_idx_start {
                let surface = surface.draw_config(true, None).with_stencil(batch_info.stencil_mode);
                let pipeline_idx =
                    shaders.prepare(RenderPipelineKey::new(self.cover_pipeline, batch_info.pattern_shader, batch_info.blend_mode, surface));
                self.draws.push(Draw::Cover {
                    stream_id: None,
                    indices: cover_idx_start..cover_idx_end,
                    pattern_inputs: batch_info.pattern_bindings,
                    pipeline_idx,
                });
                self.stats.cover_batches += 1;
            }

            let draws_end = self.draws.len();
            batch_info.draws = draws_start..draws_end;
        }

        self.stats.vertices = self.stats.vertices.max(stencil.vertices.pushed_items());
    }

    pub fn prepare_parallel(&mut self, prep_ctx: &mut PrepareContext) {
        if self.batches.is_empty() {
            return;
        }

        struct StencilWorkerData {
            draws: Vec<Draw>,
            stats: Stats,
        }

        let mut worker_data = Vec::new();
        let num_workers = prep_ctx.workers.num_workers();
        for _ in 0..num_workers {
            let mut draws = self.parallel_draws.pop().unwrap_or_else(|| {
                Vec::with_capacity(64)
            });
            draws.clear();
            worker_data.push(StencilWorkerData {
                draws,
                stats: Stats::default(),
            });
        }

        unsafe {
            self.batches.par_iter_mut(
                &mut prep_ctx.workers.with_data(&mut worker_data),
                prep_ctx.pass,
                self.renderer_id,
                &|workers, _batch, commands, surface, batch_info| {
                    let transforms = &prep_ctx.transforms;
                    let num_stencil_indices;
                    let num_cover_indices;
                    let num_stencil_vertices;
                    let stencil_stream;
                    let cover_stream;
                    {
                        let indices = &workers.const_data().0.indices;
                        let vertices = &workers.const_data().0.vertices;

                        stencil_stream = indices.next_stream_id();
                        cover_stream = indices.next_stream_id();

                        let mut stencil = Geometry {
                            indices: indices.write(stencil_stream, 0),
                            vertices: vertices.write_items::<StencilVertex>(),
                        };
                        let mut cover = Geometry {
                            indices: indices.write(cover_stream, 0),
                            vertices: vertices.write_items::<CoverVertex>(),
                        };

                        for fill in commands.iter() {
                            Self::prepare_fill(transforms, fill, &mut stencil, &mut cover, self.tolerance);
                        }

                        num_stencil_indices = stencil.indices.pushed_items::<u32>();
                        num_cover_indices = cover.indices.pushed_items::<u32>();
                        num_stencil_vertices = stencil.vertices.pushed_items();
                    }

                    let worker_idx = workers.index() as u8;
                    let (worker_data, stencil_data) = &mut workers.data();
                    let draws = &mut stencil_data.draws;
                    let stats = &mut stencil_data.stats;

                    stats.vertices += num_stencil_vertices;

                    let draws_start = draws.len();

                    if num_stencil_indices > 0 {
                        draws.push(Draw::Stencil {
                            stream_id: Some(stencil_stream),
                            indices: 0..num_stencil_indices,
                        });
                        stats.stencil_batches += 1;
                    }

                    if num_cover_indices > 0 {
                        let surface = surface.draw_config(true, None).with_stencil(batch_info.stencil_mode);
                        let pipeline_idx = worker_data.pipelines.prepare(RenderPipelineKey::new(
                            self.cover_pipeline,
                            batch_info.pattern_shader,
                            batch_info.blend_mode,
                            surface
                        ));

                        draws.push(Draw::Cover {
                            stream_id: Some(cover_stream),
                            indices: 0..num_cover_indices,
                            pattern_inputs: batch_info.pattern_bindings,
                            pipeline_idx,
                        });
                        stats.cover_batches += 1;
                    }

                    let draws_end = draws.len();
                    batch_info.draws = draws_start..draws_end;
                    batch_info.worker_index = Some(worker_idx);
                }
            );
        }
        self.parallel_draws.clear();
        for wd in worker_data {
            self.parallel_draws.push(wd.draws);
            self.stats += wd.stats;
        }
    }

    fn prepare_fill(transforms: &Transforms, fill: &Fill, stencil: &mut Geometry, cover: &mut Geometry, tolerance: f32) {

        let transform = transforms.get(fill.transform);
        let local_aabb = fill.shape.aabb();
        let transformed_aabb = transform.matrix().outer_transformed_box(&local_aabb);

        match &fill.shape {
            Shape::Path(shape) => {
                generate_stencil_geometry(
                    shape.path.as_slice(),
                    transform.matrix(),
                    tolerance,
                    &transformed_aabb,
                    stencil,
                );
            }
            _ => {
                // No stenciling to do.
            }
        }

        match &fill.shape {
            Shape::Circle(circle) => {
                if let Some(t) = transform.as_scale_offset() {
                    if (t.scale.x - t.scale.y).abs() > 0.001 {
                        todo!();
                    }
                    FillTessellator::new()
                        .tessellate_circle(
                            t.transform_point(circle.center).cast_unit(),
                            circle.radius * t.scale.x,
                            &FillOptions::tolerance(tolerance),
                            &mut GeomBuilder {
                                vertices: &mut cover.vertices,
                                indices: &mut cover.indices,
                                z_index: fill.z_index,
                                pattern: fill.pattern.data,
                            }
                        )
                        .unwrap();
                }
            }
            _ => {
                generate_cover_geometry(
                    &local_aabb,
                    transform.matrix(),
                    fill,
                    cover,
                );
            }
        }
    }
}

fn generate_stencil_geometry(
    path: PathSlice,
    transform: &LocalToSurfaceTransform,
    tolerance: f32,
    aabb: &SurfaceRect,
    stencil: &mut Geometry,
) {
    let transform = &transform.to_untyped();

    fn vertex(vertices: &mut GpuStoreWriter, p: Point) -> u32 {
        let handle = vertices.push(StencilVertex::from_point(p));
        handle.to_u32()
    }

    fn triangle(indices: &mut GpuStreamWriter, a: u32, b: u32, c: u32) {
        indices.push(a);
        indices.push(b);
        indices.push(c);
    }

    // Use the center of the bounding box as the pivot point.
    let pivot = vertex(&mut stencil.vertices, aabb.center().cast_unit());

    for evt in path.iter() {
        match evt {
            PathEvent::Begin { .. } => {}
            PathEvent::End { last, first, .. } => {
                let last = transform.transform_point(last);
                let first = transform.transform_point(first);
                if skip_edge(&aabb, last, first) {
                    continue;
                }

                let a = vertex(&mut stencil.vertices, last);
                let b = vertex(&mut stencil.vertices, first);
                triangle(&mut stencil.indices, pivot, a, b);
            }
            PathEvent::Line { from, to } => {
                let from = transform.transform_point(from);
                let to = transform.transform_point(to);
                if skip_edge(&aabb, from, to) {
                    continue;
                }

                let a = vertex(&mut stencil.vertices, from);
                let b = vertex(&mut stencil.vertices, to);
                triangle(&mut stencil.indices, pivot, a, b);
            }
            PathEvent::Quadratic { from, ctrl, to } => {
                let from = transform.transform_point(from);
                let ctrl = transform.transform_point(ctrl);
                let to = transform.transform_point(to);
                let max_x = from.x.max(ctrl.x).max(to.x);
                let max_y = from.y.max(ctrl.y).max(to.y);
                if max_x < aabb.min.x || max_y < aabb.min.y {
                    continue;
                }

                let a = vertex(&mut stencil.vertices, from);
                let b = vertex(&mut stencil.vertices, to);

                triangle(&mut stencil.indices, pivot, a, b);

                let seg_aabb = SurfaceRect {
                    min: point(from.x.min(ctrl.x).min(to.x), from.y.min(ctrl.y).min(to.y)),
                    max: point(max_x, max_y),
                };

                if seg_aabb.intersects(&aabb) {
                    let mut prev = a;
                    QuadraticBezierSegment { from, ctrl, to }.for_each_flattened(
                        tolerance,
                        &mut |seg| {
                            let next = vertex(&mut stencil.vertices, seg.to);
                            if prev != a {
                                triangle(&mut stencil.indices, a, prev, next);

                            }
                            prev = next;
                        },
                    );
                }
            }
            PathEvent::Cubic {
                from,
                ctrl1,
                ctrl2,
                to,
            } => {
                let from = transform.transform_point(from);
                let ctrl1 = transform.transform_point(ctrl1);
                let ctrl2 = transform.transform_point(ctrl2);
                let to = transform.transform_point(to);
                let max_x = from.x.max(ctrl1.x).max(ctrl2.x).max(to.x);
                let max_y = from.y.max(ctrl1.y).max(ctrl2.y).max(to.y);
                if max_x < aabb.min.x || max_y < aabb.min.y {
                    continue;
                }

                let seg_aabb = SurfaceRect {
                    min: point(
                        from.x.min(ctrl1.x).min(ctrl2.x).min(to.x),
                        from.y.min(ctrl1.y).min(ctrl2.y).min(to.y),
                    ),
                    max: point(max_x, max_y),
                };

                if seg_aabb.intersects(&aabb) {
                    CubicBezierSegment {
                        from,
                        ctrl1,
                        ctrl2,
                        to,
                    }
                    .for_each_quadratic_bezier(tolerance, &mut |quad| {
                        let a = vertex(&mut stencil.vertices, quad.from);
                        let b = vertex(&mut stencil.vertices, quad.to);

                        triangle(&mut stencil.indices, pivot, a, b);
                        let mut prev = a;
                        quad.for_each_flattened(tolerance, &mut |seg| {
                            let next = vertex(&mut stencil.vertices, seg.to);
                            if prev != a {
                                triangle(&mut stencil.indices, a, prev, next);
                            }
                            prev = next;
                        });
                    });
                }
            }
        }
    }
}

fn generate_cover_geometry(
    aabb: &LocalRect,
    transform: &LocalToSurfaceTransform,
    fill: &Fill,
    cover: &mut Geometry,
) {
    let a = transform.transform_point(aabb.min);
    let b = transform.transform_point(point(aabb.max.x, aabb.min.y));
    let c = transform.transform_point(aabb.max);
    let d = transform.transform_point(point(aabb.min.x, aabb.max.y));

    let z_index = fill.z_index;
    let pattern = fill.pattern.data;
    let a = cover.vertices.push(CoverVertex { x: a.x, y: a.y, z_index, pattern });
    let b = cover.vertices.push(CoverVertex { x: b.x, y: b.y, z_index, pattern });
    let c = cover.vertices.push(CoverVertex { x: c.x, y: c.y, z_index, pattern });
    let d = cover.vertices.push(CoverVertex { x: d.x, y: d.y, z_index, pattern });

    cover.indices.push(a);
    cover.indices.push(b);
    cover.indices.push(c);
    cover.indices.push(a);
    cover.indices.push(c);
    cover.indices.push(d);
}

impl core::Renderer for StencilAndCoverRenderer {
    fn prepare(&mut self, ctx: &mut PrepareContext) {
        // TODO: measure and adjust the batch count threshold.
        if self.parallel && self.batches.batch_count() > 16 {
            self.prepare_parallel(ctx);
        } else {
            self.prepare_single_thread(ctx);
        }
    }

    fn render<'pass, 'resources: 'pass, 'tmp>(
        &self,
        batches: &[BatchId],
        surface_info: &SurfacePassConfig,
        ctx: core::RenderContext<'resources, 'tmp>,
        render_pass: &mut wgpu::RenderPass<'pass>,
    ) {

        let stencil_idx_buffer = ctx.resources.common.indices.resolve_buffer_slice(self.stencil_indices);
        let cover_idx_buffer = ctx.resources.common.indices.resolve_buffer_slice(self.cover_indices);

        let mut helper = DrawHelper::new();

        render_pass.set_stencil_reference(128);

        render_pass.set_vertex_buffer(
            0,
            ctx.resources.common.vertices.as_buffer().unwrap().slice(..)
        );

        for batch_id in batches {
            let (_, _, batch) = self.batches.get(batch_id.index);

            let draws = if let Some(worker_idx) = batch.worker_index {
                self.parallel_draws[worker_idx as usize].as_slice()
            } else {
                self.draws.as_slice()
            };

            for draw in &draws[batch.draws.clone()] {
                match draw {
                    &Draw::Stencil { stream_id, ref indices } => {
                        let idx_buffer = stencil_idx_buffer.or_else(||{
                            ctx.resources.common.indices.resolve_buffer_slice(stream_id)
                        }).unwrap();
                        let pipeline = if surface_info.msaa {
                            &self.shared.msaa_stencil_pipeline
                        } else {
                            &self.shared.stencil_pipeline
                        };
                        render_pass.set_index_buffer(idx_buffer, wgpu::IndexFormat::Uint32);
                        render_pass.set_pipeline(pipeline);
                        render_pass.draw_indexed(indices.clone(), 0, 0..1);
                        ctx.stats.draw_calls += 1;
                    }
                    &Draw::Cover {
                        stream_id,
                        ref indices,
                        pattern_inputs,
                        pipeline_idx,
                    } => {
                        let idx_buffer = cover_idx_buffer.or_else(||{
                            ctx.resources.common.indices.resolve_buffer_slice(stream_id)
                        }).unwrap();
                        let pipeline = ctx.render_pipelines.get(pipeline_idx).unwrap();

                        helper.resolve_and_bind(1, pattern_inputs, ctx.bindings, render_pass);

                        render_pass.set_index_buffer(idx_buffer, wgpu::IndexFormat::Uint32);
                        render_pass.set_pipeline(pipeline);
                        render_pass.draw_indexed(indices.clone(), 0, 0..1);
                        ctx.stats.draw_calls += 1;
                    }
                }
            }
        }
    }
}

fn skip_edge(rect: &SurfaceRect, from: Point, to: Point) -> bool {
    from.x < rect.min.x && to.x < rect.min.x || from.y < rect.min.y && to.y < rect.min.y
}

impl core::FillPath for StencilAndCoverRenderer {
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
