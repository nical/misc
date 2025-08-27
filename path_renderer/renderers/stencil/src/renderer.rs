use std::ops::Range;
use std::sync::Arc;

use lyon::{
    geom::{CubicBezierSegment, QuadraticBezierSegment}, path::{PathEvent, PathSlice}, tessellation::{FillGeometryBuilder, FillOptions, FillTessellator, VertexId}
};
use tess::EncodedPrimitiveInfo;

use crate::resources::StencilAndCoverResources;

use core::{render_pass::{BuiltRenderPass, RenderCommandId}, units::SurfacePoint, wgpu};
use core::gpu::{GpuBufferWriter, GpuStreamWriter, StreamId};
use core::{
    PrepareContext, BindingsId, StencilMode, RenderPassConfig,
    bytemuck,
};
use core::units::{point, LocalRect, LocalToSurfaceTransform, Point, SurfaceRect};
use core::shading::{BlendMode, ShaderPatternId, GeometryId, RenderPipelineIndex, RenderPipelineKey};
use core::batching::{BatchFlags};
use core::shape::{Circle, FilledPath};
use core::render_pass::{RenderPassContext, RendererId, ZIndex};
use core::render_task::RenderTaskInfo;
use core::pattern::BuiltPattern;
use core::transform::{Transforms, TransformId};
use core::utils::DrawHelper;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct StencilVertex {
    pub x: f32,
    pub y: f32,
    pub path: u32,
    pub _pad: f32,
}

impl StencilVertex {
    pub fn new(p: Point, addr: u32) -> Self {
        StencilVertex { x: p.x, y: p.y, path: addr, _pad: 0.0 }
    }
}

unsafe impl bytemuck::Pod for StencilVertex {}
unsafe impl bytemuck::Zeroable for StencilVertex {}

pub type CoverVertex = tess::Vertex;

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
        let handle = self.vertices.push(CoverVertex {
            x,
            y,
            prim_address: self.prim_address,
            _pad: 0.0,
        });

        Ok(VertexId(handle.to_u32()))
    }
}

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
    task: RenderTaskInfo,
    pattern: BuiltPattern,
    transform: TransformId,
    z_index: ZIndex,
}

struct Geometry<'a, 'b> {
    vertices: GpuBufferWriter<'a>,
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
    cover_geometry: GeometryId,
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
            cover_geometry: shared.cover_geometry,
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

    pub fn supports_surface(&self, surface: RenderPassConfig) -> bool {
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
        debug_assert!(self.supports_surface(ctx.config));

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
            &mut |mut batch, view| {
                batch.push(Fill {
                    shape: shape.clone(),
                    task: *view,
                    pattern,
                    transform: transforms.current_id(),
                    z_index,
                });
            }
        );
    }

    pub fn prepare_single_thread(&mut self, ctx: &mut PrepareContext, pass: &BuiltRenderPass) {
        if self.batches.is_empty() {
            return;
        }

        let transforms = &ctx.transforms;
        let worker_data = &mut ctx.workers.data();
        let shaders = &mut worker_data.pipelines;
        let vertices = &mut worker_data.vertices;
        let indices = &mut worker_data.indices;
        let mut prim_buffer = worker_data.u32_buffer.write_items::<EncodedPrimitiveInfo>();

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
        for batch in pass
            .batches()
            .iter()
            .filter(|batch| batch.renderer_id() == id)
        {

            let (commands, surface, batch_info) = self.batches.get_mut(batch_id.index);

            let draws_start = self.draws.len();

            let stencil_idx_start = stencil.indices.pushed_bytes() / 4;
            let cover_idx_start = cover.indices.pushed_bytes() / 4;

            for fill in batch.items.iter() {
                Self::prepare_fill(transforms, fill, &mut stencil, &mut cover, &mut prim_buffer, self.tolerance);
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
                    shaders.prepare(RenderPipelineKey::new(self.cover_geometry, batch_info.pattern_shader, batch_info.blend_mode, surface));
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

    pub fn prepare_parallel(&mut self, prep_ctx: &mut PrepareContext, pass: &BuiltRenderPass) {
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
                pass,
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
                        let mut prim_buffer = workers.const_data().0.u32_buffer.write_items::<EncodedPrimitiveInfo>();

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
                            Self::prepare_fill(transforms, fill, &mut stencil, &mut cover, &mut prim_buffer, self.tolerance);
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
                            self.cover_geometry,
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

    fn prepare_fill(transforms: &Transforms, fill: &Fill, stencil: &mut Geometry, cover: &mut Geometry, prim_buffer: &mut GpuBufferWriter, tolerance: f32) {

        let transform = transforms.get(fill.transform);
        let local_aabb = fill.shape.aabb();
        let transformed_aabb = transform.matrix()
            .outer_transformed_box(&local_aabb)
            .intersection_unchecked(&fill.task.bounds.to_f32());

        if transformed_aabb.is_empty() {
            // Note: In theory this should have been skipped during batching.
            return;
        }

        let prim_address = prim_buffer.push(tess::PrimitiveInfo {
            z_index: fill.z_index,
            pattern: fill.pattern.data,
            opacity: 1.0, // TODO,
            render_task: fill.task.handle,
        }.encode()).to_u32();

        match &fill.shape {
            Shape::Path(shape) => {
                generate_stencil_geometry(
                    shape.path.as_slice(),
                    prim_address,
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
                                prim_address,
                            }
                        )
                        .unwrap();
                }
            }
            _ => {
                let clip = fill.task.bounds.to_f32();
                generate_cover_geometry(
                    &local_aabb,
                    transform.matrix(),
                    &clip,
                    prim_address,
                    cover,
                );
            }
        }
    }
}

fn generate_stencil_geometry(
    path: PathSlice,
    prim: u32,
    transform: &LocalToSurfaceTransform,
    tolerance: f32,
    aabb: &SurfaceRect,
    stencil: &mut Geometry,
) {
    let transform = &transform.to_untyped();

    fn vertex(vertices: &mut GpuBufferWriter, p: Point, prim_addr: u32) -> u32 {
        let handle = vertices.push(StencilVertex::new(p, prim_addr));
        handle.to_u32()
    }

    fn triangle(indices: &mut GpuStreamWriter, a: u32, b: u32, c: u32) {
        indices.push(a);
        indices.push(b);
        indices.push(c);
    }

    // Use the center of the bounding box as the pivot point.
    let pivot = vertex(&mut stencil.vertices, aabb.center().cast_unit(), prim);

    let mut skipped = None;
    for evt in path.iter() {
        match evt {
            PathEvent::Begin { .. } => {
                debug_assert!(skipped.is_none());
            }
            PathEvent::End { last, first, .. } => {
                let last = transform.transform_point(last);
                let first = transform.transform_point(first);

                if let Some(prev) = skipped {
                    let a = vertex(&mut stencil.vertices, prev, prim);
                    let b = vertex(&mut stencil.vertices, last, prim);
                    triangle(&mut stencil.indices, pivot, a, b);
                    skipped = None;
                }

                let a = vertex(&mut stencil.vertices, last, prim);
                let b = vertex(&mut stencil.vertices, first, prim);
                triangle(&mut stencil.indices, pivot, a, b);
            }
            PathEvent::Line { from, to } => {
                let from = transform.transform_point(from);
                let to = transform.transform_point(to);
                if skip_edge(&aabb, from, to) {
                    if skipped.is_none() {
                        skipped = Some(from);
                    }
                    continue;
                }

                if let Some(prev) = skipped {
                    let a = vertex(&mut stencil.vertices, prev, prim);
                    let b = vertex(&mut stencil.vertices, from, prim);
                    triangle(&mut stencil.indices, pivot, a, b);
                    skipped = None;
                }

                let a = vertex(&mut stencil.vertices, from, prim);
                let b = vertex(&mut stencil.vertices, to, prim);
                triangle(&mut stencil.indices, pivot, a, b);
            }
            PathEvent::Quadratic { from, ctrl, to } => {
                let from = transform.transform_point(from);
                let ctrl = transform.transform_point(ctrl);
                let to = transform.transform_point(to);

                let seg_aabb = SurfaceRect {
                    min: point(from.x.min(ctrl.x).min(to.x), from.y.min(ctrl.y).min(to.y)),
                    max: point(from.x.max(ctrl.x).max(to.x), from.y.max(ctrl.y).max(to.y)),
                };

                if seg_aabb.intersects(&aabb) {
                    if let Some(prev) = skipped {
                        let a = vertex(&mut stencil.vertices, prev, prim);
                        let b = vertex(&mut stencil.vertices, from, prim);
                        triangle(&mut stencil.indices, pivot, a, b);
                        skipped = None;
                    }

                    let a = vertex(&mut stencil.vertices, from, prim);
                    let b = vertex(&mut stencil.vertices, to, prim);
                    triangle(&mut stencil.indices, pivot, a, b);

                    let mut prev = a;
                    QuadraticBezierSegment { from, ctrl, to }.for_each_flattened(
                        tolerance,
                        &mut |seg| {
                            let next = vertex(&mut stencil.vertices, seg.to, prim);
                            if prev != a {
                                triangle(&mut stencil.indices, a, prev, next);

                            }
                            prev = next;
                        },
                    );
                } else if (seg_aabb.min.y < aabb.min.y) & (seg_aabb.max.y > aabb.min.y)
                        | (seg_aabb.min.y < aabb.max.y) & (seg_aabb.max.y > aabb.max.y) {
                    if let Some(prev) = skipped {
                        let a = vertex(&mut stencil.vertices, prev, prim);
                        let b = vertex(&mut stencil.vertices, from, prim);
                        triangle(&mut stencil.indices, pivot, a, b);
                        skipped = None;
                    }
                    let a = vertex(&mut stencil.vertices, from, prim);
                    let b = vertex(&mut stencil.vertices, to, prim);
                    triangle(&mut stencil.indices, pivot, a, b);
                } else if skipped.is_none() {
                    skipped = Some(from);
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

                let seg_aabb = SurfaceRect {
                    min: point(
                        from.x.min(ctrl1.x).min(ctrl2.x).min(to.x),
                        from.y.min(ctrl1.y).min(ctrl2.y).min(to.y),
                    ),
                    max: point(
                        from.x.max(ctrl1.x).max(ctrl2.x).max(to.x),
                        from.y.max(ctrl1.y).max(ctrl2.y).max(to.y),
                    ),
                };

                if seg_aabb.intersects(&aabb) {
                    if let Some(prev) = skipped {
                        let a = vertex(&mut stencil.vertices, prev, prim);
                        let b = vertex(&mut stencil.vertices, from, prim);
                        triangle(&mut stencil.indices, pivot, a, b);
                        skipped = None;
                    }

                    CubicBezierSegment {
                        from,
                        ctrl1,
                        ctrl2,
                        to,
                    }
                    .for_each_quadratic_bezier(tolerance, &mut |quad| {
                        let a = vertex(&mut stencil.vertices, quad.from, prim);
                        let b = vertex(&mut stencil.vertices, quad.to, prim);

                        triangle(&mut stencil.indices, pivot, a, b);
                        let mut prev = a;
                        quad.for_each_flattened(tolerance, &mut |seg| {
                            let next = vertex(&mut stencil.vertices, seg.to, prim);
                            if prev != a {
                                triangle(&mut stencil.indices, a, prev, next);
                            }
                            prev = next;
                        });
                    });
                } else if (seg_aabb.min.y < aabb.min.y) & (seg_aabb.max.y > aabb.min.y)
                        | (seg_aabb.min.y < aabb.max.y) & (seg_aabb.max.y > aabb.max.y) {
                    if let Some(prev) = skipped {
                        let a = vertex(&mut stencil.vertices, prev, prim);
                        let b = vertex(&mut stencil.vertices, from, prim);
                        triangle(&mut stencil.indices, pivot, a, b);
                        skipped = None;
                    }
                    let a = vertex(&mut stencil.vertices, from, prim);
                    let b = vertex(&mut stencil.vertices, to, prim);
                    triangle(&mut stencil.indices, pivot, a, b);
                } else if skipped.is_none() {
                    skipped = Some(from);
                }
            }
        }
    }
}

fn generate_cover_geometry(
    aabb: &LocalRect,
    transform: &LocalToSurfaceTransform,
    clip: &SurfaceRect,
    prim_address: u32,
    cover: &mut Geometry,
) {
    // TODO: The clip could be applied in the verex shader.
    let r = transform.outer_transformed_box(&aabb).intersection_unchecked(clip);
    let a = r.min;
    let b = SurfacePoint::new(r.max.x, r.min.y);
    let c = r.max;
    let d = SurfacePoint::new(r.min.x, r.max.y);

    let a = cover.vertices.push(CoverVertex { x: a.x, y: a.y, prim_address: prim_address, _pad: 0.0 });
    let b = cover.vertices.push(CoverVertex { x: b.x, y: b.y, prim_address: prim_address, _pad: 0.0 });
    let c = cover.vertices.push(CoverVertex { x: c.x, y: c.y, prim_address: prim_address, _pad: 0.0 });
    let d = cover.vertices.push(CoverVertex { x: d.x, y: d.y, prim_address: prim_address, _pad: 0.0 });

    cover.indices.push(a);
    cover.indices.push(b);
    cover.indices.push(c);
    cover.indices.push(a);
    cover.indices.push(c);
    cover.indices.push(d);
}

impl core::Renderer for StencilAndCoverRenderer {
    fn prepare_pass(&mut self, ctx: &mut PrepareContext, pass: &BuiltRenderPass) {
        // TODO: measure and adjust the batch count threshold.
        if self.parallel && self.batches.batch_count() > 16 {
            self.prepare_parallel(ctx, pass);
        } else {
            self.prepare_single_thread(ctx, pass);
        }
    }

    fn render<'pass, 'resources: 'pass, 'tmp>(
        &self,
        batches: &[RenderCommandId],
        surface_info: &RenderPassConfig,
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
    ((from.y < rect.min.y) & (to.y < rect.min.y))
        | ((from.y > rect.max.y) & (to.y > rect.max.y))
    //(from.x < rect.min.x && to.x < rect.min.x) || (from.y < rect.min.y && to.y < rect.min.y)
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
