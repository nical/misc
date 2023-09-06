use std::{collections::HashMap, ops::Range};

use lyon::{
    geom::{arrayvec::ArrayVec, CubicBezierSegment, QuadraticBezierSegment},
    lyon_tessellation::{VertexBuffers, FillOptions, FillVertexConstructor, FillTessellator, BuffersBuilder},
    path::{traits::PathIterator, PathEvent, PathSlice},
};

use super::StencilAndCoverResources;
use core::{bytemuck, shape::{PathShape, Circle}, batching::SurfaceIndex};
use core::resources::{CommonGpuResources, GpuResources, ResourcesHandle};
use core::wgpu;
use core::{
    batching::{BatchFlags, BatchList},
    canvas::{
        Context, CanvasRenderer, DrawHelper, RenderPassState, RendererId, SubPass,
        SurfaceFeatures, ZIndex,
    },
    gpu::{
        shader::{DepthMode, ShaderPatternId, StencilMode, SurfaceConfig},
        DynBufferRange, Shaders,
    },
    pattern::{BindingsId, BuiltPattern},
    transform::TransformId,
    units::{point, LocalRect, LocalToSurfaceTransform, Point, SurfaceRect},
    BindingResolver,
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
    pattern: u32, z_index: u32,
}

impl FillVertexConstructor<CoverVertex> for VertexCtor {
    fn new_vertex(&mut self, vertex: lyon::lyon_tessellation::FillVertex) -> CoverVertex {
        let (x, y) = vertex.position().to_tuple();
        CoverVertex { x, y, z_index: self.z_index, pattern: self.pattern }
    }
}

unsafe impl bytemuck::Pod for CoverVertex {}
unsafe impl bytemuck::Zeroable for CoverVertex {}

enum Shape {
    Path(PathShape),
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
        stencil_mode: StencilMode,
        opaque: bool,
        pattern: ShaderPatternId,
        pattern_inputs: BindingsId,
    },
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
}

type BatchRects = ArrayVec<SurfaceRect, 16>;

struct BatchHelper {
    rects: BatchRects,
    cover_idx_start: u32,
    stencil_idx_start: u32,
    prev_pattern: Option<(ShaderPatternId, BindingsId, StencilMode, bool)>,
}

pub struct StencilAndCoverRenderer {
    commands: Vec<Fill>,
    renderer_id: RendererId,
    resources: ResourcesHandle<StencilAndCoverResources>,
    common_resources: ResourcesHandle<CommonGpuResources>,
    stencil_geometry: VertexBuffers<StencilVertex, u32>,
    cover_geometry: VertexBuffers<CoverVertex, u32>,
    draws: Vec<Draw>,
    batches: BatchList<Fill, Range<usize>>,
    vbo_range: Option<DynBufferRange>,
    ibo_range: Option<DynBufferRange>,
    cover_vbo_range: Option<DynBufferRange>,
    cover_ibo_range: Option<DynBufferRange>,
    enable_msaa: bool,
    opaque_pass: bool,
    shaders: HashMap<(bool, ShaderPatternId, StencilMode, SurfaceFeatures), Option<u32>>,
    pub stats: Stats,
}

impl StencilAndCoverRenderer {
    pub fn new(
        renderer_id: RendererId,
        common_resources: ResourcesHandle<CommonGpuResources>,
        resources: ResourcesHandle<StencilAndCoverResources>,
    ) -> Self {
        StencilAndCoverRenderer {
            commands: Vec::new(),
            renderer_id,
            resources,
            common_resources,
            stencil_geometry: VertexBuffers::new(),
            cover_geometry: VertexBuffers::new(),
            draws: Vec::new(),
            batches: BatchList::new(renderer_id),
            vbo_range: None,
            ibo_range: None,
            cover_vbo_range: None,
            cover_ibo_range: None,
            enable_msaa: false,
            opaque_pass: false,
            shaders: HashMap::new(),
            stats: Stats {
                commands: 0,
                stencil_batches: 0,
                cover_batches: 0,
            },
        }
    }

    pub fn supports_surface(&self, surface: SurfaceFeatures) -> bool {
        surface.stencil
    }

    pub fn begin_frame(&mut self, canvas: &Context) {
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
        self.enable_msaa = canvas.surface.msaa();
        self.opaque_pass = canvas.surface.opaque_pass();
        self.stats = Stats::default();
    }

    pub fn fill_path<P: Into<PathShape>>(&mut self, canvas: &mut Context, path: P, pattern: BuiltPattern) {
        self.fill_shape(canvas, Shape::Path(path.into()), pattern);
    }

    pub fn fill_rect(&mut self, canvas: &mut Context, rect: &LocalRect, pattern: BuiltPattern) {
        self.fill_shape(canvas, Shape::Rect(*rect), pattern);
    }

    pub fn fill_circle(&mut self, canvas: &mut Context, circle: Circle, pattern: BuiltPattern) {
        self.fill_shape(canvas, Shape::Circle(circle), pattern);
    }

    fn fill_shape(&mut self, canvas: &mut Context, shape: Shape, pattern: BuiltPattern) {
        debug_assert!(self.supports_surface(canvas.surface.current_features()));

        let aabb = canvas
            .transforms
            .get_current()
            .matrix()
            .outer_transformed_box(&shape.aabb());

        self.batches
            .find_or_add_batch(
                &mut canvas.batcher,
                &0,
                &aabb,
                BatchFlags::empty(),
                &mut Default::default,
            )
            .0
            .push(Fill {
                shape,
                pattern,
                transform: canvas.transforms.current_id(),
                z_index: canvas.z_indices.push(),
            });
    }

    pub fn prepare(&mut self, canvas: &Context) {
        let mut batching = BatchHelper {
            rects: ArrayVec::new(),
            prev_pattern: None,
            stencil_idx_start: 0,
            cover_idx_start: 0,
        };

        let mut batches = self.batches.take();
        let id = self.renderer_id;
        for batch_id in canvas
            .batcher
            .batches()
            .iter()
            .filter(|batch| batch.renderer == id)
        {
            let (commands, draws) = batches.get_mut(batch_id.index);

            let draws_start = self.draws.len();

            for (fill_idx, fill) in commands.iter().enumerate() {
                let is_last = fill_idx == commands.len() - 1;
                self.prepare_fill(canvas, fill, &mut batching, is_last, batch_id.surface);
            }

            let draws_end = self.draws.len();
            *draws = draws_start..draws_end;
        }

        self.batches = batches;
    }

    fn prepare_fill(
        &mut self,
        canvas: &Context,
        fill: &Fill,
        batch: &mut BatchHelper,
        is_last: bool,
        surface_idx: SurfaceIndex,
    ) {
        let transform = canvas.transforms.get(fill.transform);
        let opaque = fill.pattern.is_opaque;

        let local_aabb = fill.shape.aabb();
        let stencil_mode: StencilMode =  match &fill.shape {
            Shape::Path(shape) => shape.fill_rule.into(),
            Shape::Rect(_) => StencilMode::Ignore,
            Shape::Circle(_) => StencilMode::Ignore,
            //Shape::Canvas => StencilMode::Ignore,
        };

        let transformed_aabb = transform.matrix().outer_transformed_box(&local_aabb);

        let batch_key = (
            fill.pattern.shader,
            fill.pattern.bindings,
            stencil_mode,
            opaque,
        );

        let stencil_idx_end = self.stencil_geometry.indices.len() as u32;
        let cover_idx_end = self.cover_geometry.indices.len() as u32;

        let new_stencil_batch = stencil_idx_end > batch.stencil_idx_start
            && (batch.rects.capacity() == 0
                || intersects_batch_rects(&transformed_aabb, &batch.rects));
        let new_cover_batch = cover_idx_end > batch.cover_idx_start
            && (new_stencil_batch || batch.prev_pattern != Some(batch_key));

        if new_stencil_batch {
            self.draws.push(Draw::Stencil {
                indices: batch.stencil_idx_start..stencil_idx_end,
            });
            batch.stencil_idx_start = stencil_idx_end;
            batch.rects.clear();
            self.stats.stencil_batches += 1;
        }
        batch.rects.push(transformed_aabb);

        if new_cover_batch {
            self.draws.push(Draw::Cover {
                indices: batch.cover_idx_start..cover_idx_end,
                stencil_mode,
                opaque,
                pattern: fill.pattern.shader,
                pattern_inputs: fill.pattern.bindings,
            });
            batch.cover_idx_start = cover_idx_end;
            batch.prev_pattern = Some(batch_key);
            self.stats.cover_batches += 1;
        }

        match &fill.shape {
            Shape::Path(shape) => {
                generate_stencil_geometry(
                    shape.path.as_slice(),
                    transform.matrix(),
                    canvas.params.tolerance,
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
                    FillTessellator::new().tessellate_circle(
                        t.transform_point(circle.center).cast_unit(),
                        circle.radius * t.scale.x,
                        &FillOptions::tolerance(canvas.params.tolerance),
                        &mut BuffersBuilder::new(
                            &mut self.cover_geometry,
                            VertexCtor {
                                z_index: fill.z_index,
                                pattern: fill.pattern.data,
                            },
                        )
                    ).unwrap();    
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

        if is_last {
            let stencil_idx_end = self.stencil_geometry.indices.len() as u32;
            if stencil_idx_end > batch.stencil_idx_start {
                self.draws.push(Draw::Stencil {
                    indices: batch.stencil_idx_start..stencil_idx_end,
                });
                self.stats.stencil_batches += 1;
            }

            let cover_idx_end = self.cover_geometry.indices.len() as u32;
            if cover_idx_end > batch.cover_idx_start {
                self.draws.push(Draw::Cover {
                    indices: batch.cover_idx_start..cover_idx_end,
                    stencil_mode,
                    opaque,
                    pattern: fill.pattern.shader,
                    pattern_inputs: fill.pattern.bindings,
                });
                self.stats.cover_batches += 1;
            }
        }

        if is_last || new_cover_batch {
            let state = canvas.surface.features(surface_idx);
            self.shaders
                .entry((opaque, fill.pattern.shader, stencil_mode, state))
                .or_insert(None);
        }
    }

    pub fn upload(
        &mut self,
        resources: &mut GpuResources,
        shaders: &mut Shaders,
        device: &wgpu::Device,
    ) {
        let stencil_res = &resources[self.resources];
        let opaque_pipeline = stencil_res.opaque_cover_pipeline;
        let alpha_pipeline = stencil_res.alpha_cover_pipeline;

        let res = &mut resources[self.common_resources];
        self.vbo_range = res.vertices.upload(
            device,
            bytemuck::cast_slice(&self.stencil_geometry.vertices),
        );
        self.ibo_range = res
            .indices
            .upload(device, bytemuck::cast_slice(&self.stencil_geometry.indices));
        self.cover_vbo_range = res
            .vertices
            .upload(device, bytemuck::cast_slice(&self.cover_geometry.vertices));
        self.cover_ibo_range = res
            .indices
            .upload(device, bytemuck::cast_slice(&self.cover_geometry.indices));

        for (&(opaque, pattern, stencil, surface), shader_id) in &mut self.shaders {
            if shader_id.is_none() {
                let surface = SurfaceConfig {
                    msaa: surface.msaa,
                    depth: if surface.depth {
                        DepthMode::Ignore // TODO
                    } else {
                        DepthMode::None
                    },
                    stencil,
                };

                let id = if opaque {
                    opaque_pipeline
                } else {
                    alpha_pipeline
                };
                shaders.prepare_pipeline(device, id, pattern, surface);
            }
        }
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

    for evt in path.iter().transformed(transform) {
        match evt {
            PathEvent::Begin { .. } => {}
            PathEvent::End { last, first, .. } => {
                if skip_edge(&aabb, last, first) {
                    continue;
                }

                let a = vertex(vertices, last);
                let b = vertex(vertices, first);
                triangle(indices, pivot, a, b);
            }
            PathEvent::Line { from, to } => {
                if skip_edge(&aabb, from, to) {
                    continue;
                }

                let a = vertex(vertices, from);
                let b = vertex(vertices, to);
                triangle(indices, pivot, a, b);
            }
            PathEvent::Quadratic { from, ctrl, to } => {
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
    geometry.vertices.push(CoverVertex { x: a.x, y: a.y, z_index, pattern });
    geometry.vertices.push(CoverVertex { x: b.x, y: b.y, z_index, pattern });
    geometry.vertices.push(CoverVertex { x: c.x, y: c.y, z_index, pattern });
    geometry.vertices.push(CoverVertex { x: d.x, y: d.y, z_index, pattern });
    geometry.indices.push(offset);
    geometry.indices.push(offset + 1);
    geometry.indices.push(offset + 2);
    geometry.indices.push(offset);
    geometry.indices.push(offset + 2);
    geometry.indices.push(offset + 3);
}

impl CanvasRenderer for StencilAndCoverRenderer {
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
        let stencil_resources = &resources[self.resources];

        let mut helper = DrawHelper::new();

        render_pass.set_bind_group(
            0,
            &common_resources.main_target_and_gpu_store_bind_group,
            &[],
        );
        render_pass.set_stencil_reference(128);

        for sub_pass in sub_passes {
            let (_, draws) = self.batches.get(sub_pass.internal_index);

            for draw in &self.draws[draws.clone()] {
                match draw {
                    Draw::Stencil { indices } => {
                        // Stencil
                        let pipeline = if surface_info.surface.msaa {
                            &stencil_resources.msaa_stencil_pipeline
                        } else {
                            &stencil_resources.stencil_pipeline
                        };
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
                        render_pass.set_pipeline(pipeline);
                        render_pass.draw_indexed(indices.clone(), 0, 0..1);
                    }
                    &Draw::Cover {
                        ref indices,
                        stencil_mode,
                        opaque,
                        pattern,
                        pattern_inputs,
                    } => {
                        // Cover
                        let surface = SurfaceConfig {
                            msaa: surface_info.surface.msaa,
                            depth: if surface_info.surface.depth { DepthMode::Ignore } else { DepthMode::None },
                            stencil: stencil_mode,
                        };

                        let pipeline_id = if opaque {
                            stencil_resources.opaque_cover_pipeline
                        } else {
                            stencil_resources.alpha_cover_pipeline
                        };

                        helper.resolve_and_bind(1, pattern_inputs, bindings, render_pass);

                        // TODO: Take advantage of the fact that we tend to query the same pipeline multiple times in a row.
                        let pipeline = shaders.try_get(pipeline_id, pattern, surface).unwrap();

                        render_pass.set_index_buffer(
                            common_resources
                                .indices
                                .get_buffer_slice(self.cover_ibo_range.as_ref().unwrap()),
                            wgpu::IndexFormat::Uint32,
                        );
                        render_pass.set_vertex_buffer(
                            0,
                            common_resources
                                .vertices
                                .get_buffer_slice(self.cover_vbo_range.as_ref().unwrap()),
                        );
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

fn intersects_batch_rects(new_rect: &SurfaceRect, batch_rects: &[SurfaceRect]) -> bool {
    for rect in batch_rects {
        if new_rect.intersects(rect) {
            return true;
        }
    }

    return false;
}
