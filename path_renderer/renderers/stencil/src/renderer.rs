use std::ops::Range;
use std::sync::Arc;

use lyon::{
    geom::{CubicBezierSegment, QuadraticBezierSegment},
    lyon_tessellation::{
        BuffersBuilder, FillOptions, FillTessellator, FillVertexConstructor, VertexBuffers,
    },
    path::{PathEvent, PathSlice},
};

use crate::resources::StencilAndCoverResources;

use core::{batching::BatchId, context::{BuiltRenderPass, RenderPassContext}, gpu::shader::{BlendMode, ShaderPatternId}, resources::GpuResources, transform::Transforms, PrepareContext, UploadContext};
use core::wgpu;
use core::{
    bytemuck,
    gpu::shader::{
        BaseShaderId, PrepareRenderPipelines, RenderPipelineIndex, RenderPipelineKey,
    },
    shape::{Circle, FilledPath},
};
use core::{
    BindingsId, StencilMode, SurfacePassConfig,
    batching::{BatchFlags, BatchList},
    context::{DrawHelper, RendererId, ZIndex},
    gpu::DynBufferRange,
    pattern::BuiltPattern,
    transform::TransformId,
    units::{point, LocalRect, LocalToSurfaceTransform, Point, SurfaceRect},
};

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

struct VertexCtor {
    pattern: u32,
    z_index: u32,
}

impl FillVertexConstructor<CoverVertex> for VertexCtor {
    fn new_vertex(&mut self, vertex: lyon::lyon_tessellation::FillVertex) -> CoverVertex {
        let (x, y) = vertex.position().to_tuple();
        CoverVertex {
            x,
            y,
            z_index: self.z_index,
            pattern: self.pattern,
        }
    }
}

unsafe impl bytemuck::Pod for CoverVertex {}
unsafe impl bytemuck::Zeroable for CoverVertex {}

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
        indices: Range<u32>,
    },
    Cover {
        indices: Range<u32>,
        pattern_inputs: BindingsId,
        pipeline_idx: RenderPipelineIndex,
    },
}


pub struct BatchInfo {
    draws: Range<usize>,
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

#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub commands: u32,
    pub stencil_batches: u32,
    pub cover_batches: u32,
    pub vertices: u32,
}

pub struct StencilAndCoverRenderer {
    commands: Vec<Fill>,
    renderer_id: RendererId,
    stencil_geometry: VertexBuffers<StencilVertex, u32>,
    cover_geometry: VertexBuffers<CoverVertex, u32>,
    draws: Vec<Draw>,
    batches: BatchList<Fill, BatchInfo>,
    vbo_range: Option<DynBufferRange>,
    ibo_range: Option<DynBufferRange>,
    cover_vbo_range: Option<DynBufferRange>,
    cover_ibo_range: Option<DynBufferRange>,
    cover_pipeline: BaseShaderId,
    pub stats: Stats,
    pub tolerance: f32,
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
            stencil_geometry: VertexBuffers::new(),
            cover_geometry: VertexBuffers::new(),
            draws: Vec::new(),
            batches: BatchList::new(renderer_id),
            vbo_range: None,
            ibo_range: None,
            cover_vbo_range: None,
            cover_ibo_range: None,
            cover_pipeline: shared.cover_base_shader,
            stats: Stats {
                commands: 0,
                stencil_batches: 0,
                cover_batches: 0,
                vertices: 0,
            },
            tolerance: 0.25,
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
        self.stencil_geometry.vertices.clear();
        self.stencil_geometry.indices.clear();
        self.cover_geometry.vertices.clear();
        self.cover_geometry.indices.clear();
        self.vbo_range = None;
        self.ibo_range = None;
        self.cover_vbo_range = None;
        self.cover_ibo_range = None;
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

        self.batches
            .find_or_add_batch(
                ctx,
                &(pattern.batch_key() | (stencil_key << 32)),
                &aabb,
                BatchFlags::NO_OVERLAP | BatchFlags::EARLIEST_CANDIDATE,
                &mut || BatchInfo {
                    draws: 0..0,
                    pattern_shader: pattern.shader,
                    pattern_bindings: pattern.bindings,
                    stencil_mode,
                    blend_mode: pattern.blend_mode,
                },
            ).push(Fill {
                shape,
                pattern,
                transform: transforms.current_id(),
                z_index: ctx.z_indices.push(),
            });
    }

    pub fn prepare(&mut self, pass: &BuiltRenderPass, transforms: &Transforms, shaders: &mut PrepareRenderPipelines) {
        let mut batches = self.batches.take();
        let id = self.renderer_id;
        for batch_id in pass
            .batches()
            .iter()
            .filter(|batch| batch.renderer == id)
        {
            let (commands, surface, batch_info) = batches.get_mut(batch_id.index);

            let draws_start = self.draws.len();

            let stencil_idx_start = self.stencil_geometry.indices.len() as u32;
            let cover_idx_start = self.cover_geometry.indices.len() as u32;

            for fill in commands.iter() {
                self.prepare_fill(transforms, fill);
            }

            let stencil_idx_end = self.stencil_geometry.indices.len() as u32;
            let cover_idx_end = self.cover_geometry.indices.len() as u32;

            if stencil_idx_end > stencil_idx_start {
                self.draws.push(Draw::Stencil { indices: stencil_idx_start..stencil_idx_end });
                self.stats.stencil_batches += 1;
            }

            // Flush the previous cover batch if needed.
            if cover_idx_end > cover_idx_start {
                let surface = surface.draw_config(true, None).with_stencil(batch_info.stencil_mode);
                let pipeline_idx =
                    shaders.prepare(RenderPipelineKey::new(self.cover_pipeline, batch_info.pattern_shader, batch_info.blend_mode, surface));
                self.draws.push(Draw::Cover {
                    indices: cover_idx_start..cover_idx_end,
                    pattern_inputs: batch_info.pattern_bindings,
                    pipeline_idx,
                });
                self.stats.cover_batches += 1;
            }

            let draws_end = self.draws.len();
            batch_info.draws = draws_start..draws_end;
        }

        self.batches = batches;
        self.stats.vertices = self.stats.vertices.max(self.stencil_geometry.vertices.len() as u32);
    }

    fn prepare_fill(&mut self, transforms: &Transforms, fill: &Fill) {

        let transform = transforms.get(fill.transform);
        let local_aabb = fill.shape.aabb();
        let transformed_aabb = transform.matrix().outer_transformed_box(&local_aabb);

        match &fill.shape {
            Shape::Path(shape) => {
                generate_stencil_geometry(
                    shape.path.as_slice(),
                    transform.matrix(),
                    self.tolerance,
                    &transformed_aabb,
                    &mut self.stencil_geometry,
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
                            &FillOptions::tolerance(self.tolerance),
                            &mut BuffersBuilder::new(
                                &mut self.cover_geometry,
                                VertexCtor {
                                    z_index: fill.z_index,
                                    pattern: fill.pattern.data,
                                },
                            ),
                        )
                        .unwrap();
                }
            }
            _ => {
                generate_cover_geometry(
                    &local_aabb,
                    transform.matrix(),
                    fill,
                    &mut self.cover_geometry,
                );
            }
        }
    }

    pub fn upload(&mut self, resources: &mut GpuResources, device: &wgpu::Device) {
        self.vbo_range = resources.common.vertices.upload(
            device,
            bytemuck::cast_slice(&self.stencil_geometry.vertices),
        );
        self.ibo_range = resources.common
            .indices
            .upload(device, bytemuck::cast_slice(&self.stencil_geometry.indices));
        self.cover_vbo_range = resources.common
            .vertices
            .upload(device, bytemuck::cast_slice(&self.cover_geometry.vertices));
        self.cover_ibo_range = resources.common
            .indices
            .upload(device, bytemuck::cast_slice(&self.cover_geometry.indices));
    }
}

pub fn generate_stencil_geometry(
    path: PathSlice,
    transform: &LocalToSurfaceTransform,
    tolerance: f32,
    aabb: &SurfaceRect,
    stencil_geometry: &mut VertexBuffers<StencilVertex, u32>,
) {
    let transform = &transform.to_untyped();
    let vertices = &mut stencil_geometry.vertices;
    let indices = &mut stencil_geometry.indices;

    fn vertex(vertices: &mut Vec<StencilVertex>, p: Point) -> u32 {
        let idx = vertices.len() as u32;
        vertices.push(StencilVertex::from_point(p));
        idx
    }

    fn triangle(indices: &mut Vec<u32>, a: u32, b: u32, c: u32) {
        indices.push(a);
        indices.push(b);
        indices.push(c)
    }

    // Use the center of the bounding box as the pivot point.
    let pivot = vertex(vertices, aabb.center().cast_unit());

    for evt in path.iter() {
        match evt {
            PathEvent::Begin { .. } => {}
            PathEvent::End { last, first, .. } => {
                let last = transform.transform_point(last);
                let first = transform.transform_point(first);
                if skip_edge(&aabb, last, first) {
                    continue;
                }

                let a = vertex(vertices, last);
                let b = vertex(vertices, first);
                triangle(indices, pivot, a, b);
            }
            PathEvent::Line { from, to } => {
                let from = transform.transform_point(from);
                let to = transform.transform_point(to);
                if skip_edge(&aabb, from, to) {
                    continue;
                }

                let a = vertex(vertices, from);
                let b = vertex(vertices, to);
                triangle(indices, pivot, a, b);
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

                let a = vertex(vertices, from);
                let b = vertex(vertices, to);

                triangle(indices, pivot, a, b);

                let seg_aabb = SurfaceRect {
                    min: point(from.x.min(ctrl.x).min(to.x), from.y.min(ctrl.y).min(to.y)),
                    max: point(max_x, max_y),
                };

                if seg_aabb.intersects(&aabb) {
                    let mut prev = a;
                    QuadraticBezierSegment { from, ctrl, to }.for_each_flattened(
                        tolerance,
                        &mut |seg| {
                            let next = vertex(vertices, seg.to);
                            if prev != a {
                                triangle(indices, a, prev, next);
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
                        let a = vertex(vertices, quad.from);
                        let b = vertex(vertices, quad.to);

                        triangle(indices, pivot, a, b);
                        let mut prev = a;
                        quad.for_each_flattened(tolerance, &mut |seg| {
                            let next = vertex(vertices, seg.to);
                            if prev != a {
                                triangle(indices, a, prev, next);
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
    geometry: &mut VertexBuffers<CoverVertex, u32>,
) {
    let a = transform.transform_point(aabb.min);
    let b = transform.transform_point(point(aabb.max.x, aabb.min.y));
    let c = transform.transform_point(aabb.max);
    let d = transform.transform_point(point(aabb.min.x, aabb.max.y));

    let z_index = fill.z_index;
    let pattern = fill.pattern.data;
    let offset = geometry.vertices.len() as u32;
    geometry.vertices.push(CoverVertex {
        x: a.x,
        y: a.y,
        z_index,
        pattern,
    });
    geometry.vertices.push(CoverVertex {
        x: b.x,
        y: b.y,
        z_index,
        pattern,
    });
    geometry.vertices.push(CoverVertex {
        x: c.x,
        y: c.y,
        z_index,
        pattern,
    });
    geometry.vertices.push(CoverVertex {
        x: d.x,
        y: d.y,
        z_index,
        pattern,
    });
    geometry.indices.push(offset);
    geometry.indices.push(offset + 1);
    geometry.indices.push(offset + 2);
    geometry.indices.push(offset);
    geometry.indices.push(offset + 2);
    geometry.indices.push(offset + 3);
}

impl core::Renderer for StencilAndCoverRenderer {
    fn prepare(&mut self, ctx: &mut PrepareContext) {
        self.prepare(ctx.pass, ctx.transforms, ctx.pipelines);
    }

    fn upload(&mut self, ctx: &mut UploadContext) {
        self.upload(ctx.resources, ctx.wgpu.device);
    }

    fn render<'pass, 'resources: 'pass>(
        &self,
        batches: &[BatchId],
        surface_info: &SurfacePassConfig,
        ctx: core::RenderContext<'resources>,
        render_pass: &mut wgpu::RenderPass<'pass>,
    ) {
        let mut helper = DrawHelper::new();

        render_pass.set_stencil_reference(128);

        for batch_id in batches {
            let (_, _, batch) = self.batches.get(batch_id.index);

            for draw in &self.draws[batch.draws.clone()] {
                match draw {
                    Draw::Stencil { indices } => {
                        // Stencil
                        let pipeline = if surface_info.msaa {
                            &self.shared.msaa_stencil_pipeline
                        } else {
                            &self.shared.stencil_pipeline
                        };
                        // TODO: switching the index and vertex buffers here has
                        // a fair amount of validation overhead in wgpu. It may be
                        // better to merge the sencil and cover buffers into a
                        // single pair to avoid rebinding.
                        render_pass.set_index_buffer(
                            ctx.resources.common
                                .indices
                                .get_buffer_slice(self.ibo_range.as_ref().unwrap()),
                            wgpu::IndexFormat::Uint32,
                        );
                        render_pass.set_vertex_buffer(
                            0,
                            ctx.resources.common
                                .vertices
                                .get_buffer_slice(self.vbo_range.as_ref().unwrap()),
                        );
                        render_pass.set_pipeline(pipeline);
                        render_pass.draw_indexed(indices.clone(), 0, 0..1);
                    }
                    &Draw::Cover {
                        ref indices,
                        pattern_inputs,
                        pipeline_idx,
                    } => {
                        // Cover
                        helper.resolve_and_bind(1, pattern_inputs, ctx.bindings, render_pass);

                        render_pass.set_index_buffer(
                            ctx.resources.common
                                .indices
                                .get_buffer_slice(self.cover_ibo_range.as_ref().unwrap()),
                            wgpu::IndexFormat::Uint32,
                        );
                        render_pass.set_vertex_buffer(
                            0,
                            ctx.resources.common
                                .vertices
                                .get_buffer_slice(self.cover_vbo_range.as_ref().unwrap()),
                        );

                        let pipeline = ctx.render_pipelines.get(pipeline_idx).unwrap();
                        render_pass.set_pipeline(pipeline);

                        render_pass.draw_indexed(indices.clone(), 0, 0..1);
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
