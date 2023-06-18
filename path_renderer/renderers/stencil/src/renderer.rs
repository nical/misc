use std::ops::Range;

use lyon::{path::{PathSlice, PathEvent, traits::PathIterator, FillRule}, math::{Box2D, Point, Transform, point}, geom::{QuadraticBezierSegment, CubicBezierSegment, arrayvec::ArrayVec}, lyon_tessellation::VertexBuffers};

use core::{canvas::{Canvas, Shape,  RendererCommandIndex, RendererId, RecordedShape, CanvasRenderer, SubPass, ZIndex, RenderPasses, RenderPassState, TransformId, DrawHelper, SurfaceState}, gpu::{DynBufferRange, shader::{SurfaceConfig, StencilMode, DepthMode, ShaderPatternId}, Shaders}, pattern::{BuiltPattern, BindingsId}, BindingResolver};
use core::resources::{GpuResources, ResourcesHandle, CommonGpuResources};
use super::StencilAndCoverResources;
use core::bytemuck;
use core::wgpu;


#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct StencilVertex {
    pub x: f32, pub y: f32 
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

unsafe impl bytemuck::Pod for CoverVertex {}
unsafe impl bytemuck::Zeroable for CoverVertex {}

pub enum Draw {
    Stencil { indices: Range<u32> },
    Cover {
        indices: Range<u32>,
        fill_rule: FillRule,
        opaque: bool,
        pattern: ShaderPatternId,
        pattern_inputs: BindingsId,
    }
}

pub struct Fill {
    pub shape: RecordedShape,
    pub pattern: BuiltPattern,
    pub transform: TransformId,
    pub z_index: ZIndex,
}

struct Pass {
    draws: Range<usize>,
    z_index: ZIndex,
    surface: SurfaceState,
}

#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub commands: u32,
    pub stencil_batches: u32,
    pub cover_batches: u32,
}

pub struct StencilAndCoverRenderer {
    commands: Vec<Fill>,
    renderer_id: RendererId,
    resources: ResourcesHandle<StencilAndCoverResources>,
    common_resources: ResourcesHandle<CommonGpuResources>,
    stencil_geometry: VertexBuffers<StencilVertex, u32>,
    cover_geometry: VertexBuffers<CoverVertex, u32>,
    draws: Vec<Draw>,
    passes: Vec<Pass>,
    vbo_range: Option<DynBufferRange>,
    ibo_range: Option<DynBufferRange>,
    cover_vbo_range: Option<DynBufferRange>,
    cover_ibo_range: Option<DynBufferRange>,
    enable_msaa: bool,
    opaque_pass: bool,
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
            passes: Vec::new(),
            vbo_range: None,
            ibo_range: None,
            cover_vbo_range: None,
            cover_ibo_range: None,
            enable_msaa: false,
            opaque_pass: false,
            stats: Stats {
                commands: 0,
                stencil_batches: 0,
                cover_batches: 0,
            }
        }
    }

    pub fn begin_frame(&mut self, canvas: &Canvas) {
        self.commands.clear();
        self.draws.clear();
        self.passes.clear();
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
        let mut batch_rects: ArrayVec<Box2D, 16> = ArrayVec::new();
        let mut prev_pattern = None;
        let mut stencil_idx_start = 0;
        let mut cover_idx_start = 0;

        let commands = std::mem::take(&mut self.commands);
        self.stats.commands += commands.len() as u32;
        for range in canvas.commands.with_renderer(self.renderer_id) {
            let surface = range.surface;
            let range = range.commands.start as usize .. range.commands.end as usize;

            let draws_start = self.draws.len();

            for (fill_idx, fill) in commands[range.clone()].iter().enumerate() {
                let is_last = fill_idx == range.end;
                let transform = canvas.transforms.get(fill.transform);
                let opaque = fill.pattern.is_opaque;

                let (local_aabb, fill_rule) = match &fill.shape {
                    RecordedShape::Path(shape) => (
                        lyon::algorithms::aabb::fast_bounding_box(&shape.path.as_slice()),
                        shape.fill_rule
                    ),
                    RecordedShape::Rect(rect) => (*rect, FillRule::NonZero),
                    RecordedShape::Circle(circle) => (circle.aabb(), FillRule::NonZero),
                    RecordedShape::Canvas => (
                        // TODO: that's the transformed aabb!
                        Box2D::from_size(canvas.surface.size().to_f32()),
                        FillRule::NonZero
                    ),
                };

                let transformed_aabb = transform.outer_transformed_box(&local_aabb);

                let batch_key = (fill.pattern.shader, fill.pattern.bindings, fill_rule, opaque);

                let stencil_idx_end = self.stencil_geometry.indices.len() as u32;
                let cover_idx_end = self.cover_geometry.indices.len() as u32;

                let new_stencil_batch = stencil_idx_end > stencil_idx_start
                    && (batch_rects.capacity() == 0 || intersects_batch_rects(&transformed_aabb, &batch_rects));
                let new_cover_batch = cover_idx_end > cover_idx_start
                    && (new_stencil_batch || prev_pattern != Some(batch_key));

                if new_stencil_batch {
                    self.draws.push(Draw::Stencil { indices: stencil_idx_start..stencil_idx_end });
                    stencil_idx_start = stencil_idx_end;
                    batch_rects.clear();
                    self.stats.stencil_batches += 1;
                }
                batch_rects.push(transformed_aabb);

                if new_cover_batch {
                    self.draws.push(Draw::Cover {
                        indices: cover_idx_start..cover_idx_end,
                        fill_rule: fill_rule,
                        opaque,
                        pattern: fill.pattern.shader,
                        pattern_inputs: fill.pattern.bindings,
                    });
                    cover_idx_start = cover_idx_end;
                    prev_pattern = Some(batch_key);
                    self.stats.cover_batches += 1;
                }

                match &fill.shape {
                    RecordedShape::Path(shape) => {
                        generate_stencil_geometry(
                            shape.path.as_slice(),
                            transform,
                            canvas.params.tolerance,
                            &transformed_aabb,
                            &mut self.stencil_geometry,
                        );
                    }
                    _ => {
                        todo!()
                    }
                }

                generate_cover_geometry(
                    &local_aabb,
                    transform,
                    fill,
                    &mut self.cover_geometry
                );

                if is_last {
                    let stencil_idx_end = self.stencil_geometry.indices.len() as u32;
                    if stencil_idx_end > stencil_idx_start {
                        self.draws.push(Draw::Stencil { indices: stencil_idx_start..stencil_idx_end });
                        self.stats.stencil_batches += 1;
                    }

                    let cover_idx_end = self.cover_geometry.indices.len() as u32;
                    if cover_idx_end > cover_idx_start {
                        self.draws.push(Draw::Cover {
                            indices: cover_idx_start..cover_idx_end,
                            fill_rule: fill_rule,
                            opaque,
                            pattern: fill.pattern.shader,
                            pattern_inputs: fill.pattern.bindings,
                        });
                        self.stats.cover_batches += 1;
                    }
                }
            }

            let pass_z_index = commands[range.start].z_index;
            let draws_end = self.draws.len();
            if draws_start < draws_end {
                self.passes.push(Pass {
                    draws: draws_start..draws_end,
                    z_index: pass_z_index,
                    surface,
                });
            }
        }
    }

    pub fn upload(&mut self, resources: &mut GpuResources, shaders: &mut Shaders, device: &wgpu::Device) {
        let stencil_res = &resources[self.resources];
        let opaque_pipeline = stencil_res.opaque_cover_pipeline;
        let alpha_pipeline = stencil_res.alpha_cover_pipeline;

        let res = &mut resources[self.common_resources];
        self.vbo_range = res.vertices.upload(device, bytemuck::cast_slice(&self.stencil_geometry.vertices));
        self.ibo_range = res.indices.upload(device, bytemuck::cast_slice(&self.stencil_geometry.indices));
        self.cover_vbo_range = res.vertices.upload(device, bytemuck::cast_slice(&self.cover_geometry.vertices));
        self.cover_ibo_range = res.indices.upload(device, bytemuck::cast_slice(&self.cover_geometry.indices));

        for pass in &self.passes {
            for draw in &self.draws[pass.draws.clone()] {
                if let &Draw::Cover { fill_rule, opaque, pattern, .. } = draw {
                    let surface = SurfaceConfig {
                        msaa: pass.surface.msaa,
                        // TODO: take advantage of the opaque pass.
                        depth: if pass.surface.depth { DepthMode::Ignore } else { DepthMode::None },
                        stencil: match fill_rule {
                            FillRule::EvenOdd => StencilMode::EvenOdd,
                            FillRule::NonZero => StencilMode::NonZero,
                        },
                    };
    
                    let id = if opaque { opaque_pipeline } else { alpha_pipeline };
                    shaders.prepare_pipeline(device, id, pattern, surface);    
                }
            }
        }
    }
}

pub fn generate_stencil_geometry(
    path: PathSlice,
    transform: &Transform,
    tolerance: f32,
    aabb: &Box2D,
    stencil_geometry: &mut VertexBuffers<StencilVertex, u32>,
) {
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

    let pivot = vertex(vertices, aabb.min);

    for evt in path.iter().transformed(transform) {
        match evt {
            PathEvent::Begin { .. } => {

            }
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

                let aabb = Box2D {
                    min: point(
                        from.x.min(ctrl.x).min(to.x),
                        from.y.min(ctrl.y).min(to.y),
                    ),
                    max: point(max_x, max_y),
                };

                if aabb.intersects(&aabb) {
                    let mut prev = a;
                    QuadraticBezierSegment { from, ctrl, to }.for_each_flattened(tolerance, &mut |seg| {
                        let next = vertex(vertices, seg.to);
                        if prev != a {
                            triangle(indices, a, prev, next);
                        }
                        prev = next;
                    });
                }
            }
            PathEvent::Cubic { from, ctrl1, ctrl2, to } => {
                let max_x = from.x.max(ctrl1.x).max(ctrl2.x).max(to.x);
                let max_y = from.y.max(ctrl1.y).max(ctrl2.y).max(to.y);
                if max_x < aabb.min.x || max_y < aabb.min.y {
                    continue;
                }

                let aabb = Box2D {
                    min: point(
                        from.x.min(ctrl1.x).min(ctrl2.x).min(to.x),
                        from.y.min(ctrl1.y).min(ctrl2.y).min(to.y),
                    ),
                    max: point(max_x, max_y),
                };

                if aabb.intersects(&aabb) {
                    CubicBezierSegment { from, ctrl1, ctrl2, to}.for_each_quadratic_bezier(tolerance, &mut |quad| {
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
    aabb: &Box2D,
    transform: &Transform,
    fill: &Fill,
    geometry: &mut VertexBuffers<CoverVertex, u32>,
) {
    let a = transform.transform_point(aabb.min);
    let b = transform.transform_point(point(aabb.max.x, aabb.min.y));
    let c = transform.transform_point(aabb.max);
    let d = transform.transform_point(point(aabb.min.x, aabb.max.y));

    let z_index = fill.z_index;
    let offset = geometry.vertices.len() as u32;
    geometry.vertices.push(CoverVertex { x: a.x, y: a.y, z_index, pattern: fill.pattern.data });
    geometry.vertices.push(CoverVertex { x: b.x, y: b.y, z_index, pattern: fill.pattern.data });
    geometry.vertices.push(CoverVertex { x: c.x, y: c.y, z_index, pattern: fill.pattern.data });
    geometry.vertices.push(CoverVertex { x: d.x, y: d.y, z_index, pattern: fill.pattern.data });
    geometry.indices.push(offset);
    geometry.indices.push(offset + 1);
    geometry.indices.push(offset + 2);
    geometry.indices.push(offset);
    geometry.indices.push(offset + 2);
    geometry.indices.push(offset + 3);
}

impl CanvasRenderer for StencilAndCoverRenderer {
    fn add_render_passes(&mut self, render_passes: &mut RenderPasses) {
        for (idx, pass) in self.passes.iter().enumerate() {
            render_passes.push(SubPass {
                renderer_id: self.renderer_id,
                internal_index: idx as u32,
                require_pre_pass: false,
                z_index: pass.z_index,
                surface: SurfaceState {
                    stencil: true,
                    .. pass.surface
                },
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
        let stencil_resources = &resources[self.resources];

        let pass = &self.passes[index as usize];

        render_pass.set_bind_group(0, &common_resources.main_target_and_gpu_store_bind_group, &[]);

        render_pass.set_stencil_reference(128);

        let mut helper = DrawHelper::new();

        for draw in &self.draws[pass.draws.clone()] {
            match draw {
                Draw::Stencil { indices } => {
                    // Stencil
                    let pipeline = if surface_info.surface.msaa {
                        &stencil_resources.msaa_stencil_pipeline
                    } else {
                        &stencil_resources.stencil_pipeline
                    };
                    render_pass.set_index_buffer(common_resources.indices.get_buffer_slice(self.ibo_range.as_ref().unwrap()), wgpu::IndexFormat::Uint32);
                    render_pass.set_vertex_buffer(0, common_resources.vertices.get_buffer_slice(self.vbo_range.as_ref().unwrap()));
                    render_pass.set_pipeline(pipeline);
                    render_pass.draw_indexed(indices.clone(), 0, 0..1);
                }
                &Draw::Cover { ref indices, fill_rule, opaque, pattern, pattern_inputs } => {
                    // Cover
                    let surface = surface_info.surface_config(false, Some(fill_rule));

                    let pipeline_id = if opaque {
                        stencil_resources.opaque_cover_pipeline
                    } else {
                        stencil_resources.alpha_cover_pipeline
                    };

                    helper.resolve_and_bind(1, pattern_inputs, bindings, render_pass);

                    // TODO: Take advantage of the fact that we tend to query the same pipeline multiple times in a row.
                    let pipeline = shaders.try_get(pipeline_id, pattern, surface).unwrap();

                    render_pass.set_index_buffer(common_resources.indices.get_buffer_slice(self.cover_ibo_range.as_ref().unwrap()), wgpu::IndexFormat::Uint32);
                    render_pass.set_vertex_buffer(0, common_resources.vertices.get_buffer_slice(self.cover_vbo_range.as_ref().unwrap()));
                    render_pass.set_pipeline(pipeline);
                    render_pass.draw_indexed(indices.clone(), 0, 0..1);
                }
            }
        }
    }
}

fn skip_edge(rect: &Box2D, from: Point, to: Point) -> bool {
    from.x < rect.min.x && to.x < rect.min.x 
        || from.y < rect.min.y && to.y < rect.min.y
}

fn intersects_batch_rects(new_rect: &Box2D, batch_rects: &[Box2D]) -> bool {
    for rect in batch_rects {
        if new_rect.intersects(rect) {
            return true;
        }
    }

    return false;
}