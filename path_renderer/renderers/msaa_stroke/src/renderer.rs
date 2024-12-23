use core::{
    batching::{BatchFlags, BatchId, BatchList}, bytemuck, context::{
        BuiltRenderPass, DrawHelper, RenderPassContext, RendererId, SurfacePassConfig, ZIndex
    }, gpu::{
        shader::{
            BaseShaderId, BindGroupLayoutId, BlendMode, PrepareRenderPipelines, RenderPipelineIndex, RenderPipelineKey
        }, storage_buffer::{StorageBuffer, StorageKind}, DynBufferRange, Shaders
    }, pattern::BuiltPattern, resources::GpuResources, shape::FilledPath, transform::{TransformId, Transforms}, units::LocalRect, usize_range, wgpu, BindingsId, Point, PrepareContext, UploadContext
};
use lyon::{
    geom::{QuadraticBezierSegment, CubicBezierSegment, LineSegment},
    path::{PathSlice, PathEvent},
};
use std::ops::Range;

pub const PATTERN_KIND_COLOR: u32 = 0;
pub const PATTERN_KIND_SIMPLE_LINEAR_GRADIENT: u32 = 1;

struct BatchInfo {
    draws: Range<u32>,
    blend_mode: BlendMode,
}

enum Shape {
    Path(FilledPath, f32),
    //Rect(LocalRect, f32),
    //Circle(Circle, f32),
}

impl Shape {
    pub fn aabb(&self) -> LocalRect {
        match self {
            // TODO: return the correct aabb for inverted shapes.
            Shape::Path(shape, r) => shape.path.aabb().inflate(*r, *r),
            //Shape::Rect(rect) => *rect,
            //Shape::Circle(circle) => circle.aabb(),
            //Shape::Mesh(mesh) => mesh.aabb,
            //Shape::StrokePath(path, width) => path.aabb().inflate(*width, *width),
        }
    }
}

struct Stroke {
    transform: TransformId,
    shape: Shape,
    pattern: BuiltPattern,
    z_index: ZIndex,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PathData {
    pub transform: [f32; 6],
    pub width: f32,
    pub pad0: f32,
    pub pattern: u32,
    pub z_index: u32,
    pub pad1: u32,
    pub pad2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct CurveInstance {
    pub from: Point,
    pub ctrl1: Point,
    pub ctrl2: Point,
    pub to: Point,
    pub prev_ctrl: Point,
    pub path_index: u32,
    pub segment_counts: u32,
}

unsafe impl bytemuck::Zeroable for CurveInstance {}
unsafe impl bytemuck::Pod for CurveInstance {}
unsafe impl bytemuck::Zeroable for PathData {}
unsafe impl bytemuck::Pod for PathData {}

struct Draw {
    segment_count: u32,
    instances: Range<u32>,
    pattern_inputs: BindingsId,
    pipeline_idx: RenderPipelineIndex,
}

pub struct MsaaStrokeRenderer {
    renderer_id: RendererId,
    curves: Vec<CurveInstance>,
    path_data: Vec<PathData>,
    pub tolerance: f32,

    batches: BatchList<Stroke, BatchInfo>,
    draws: Vec<Draw>,
    instance_range: Option<DynBufferRange>,
    ibo_range: Option<DynBufferRange>,
    base_shader: BaseShaderId,

    paths: StorageBuffer,
    geom_bind_group: Option<wgpu::BindGroup>,
    geom_bind_group_layout: BindGroupLayoutId,
}

impl MsaaStrokeRenderer {
    pub(crate) fn new(
        device: &wgpu::Device,
        renderer_id: RendererId,
        base_shader: BaseShaderId,
        geom_bind_group_layout: BindGroupLayoutId,
    ) -> Self {
        MsaaStrokeRenderer {
            renderer_id,
            curves: Vec::new(),
            path_data: Vec::new(),
            tolerance: 0.25,

            draws: Vec::new(),
            batches: BatchList::new(renderer_id),
            instance_range: None,
            ibo_range: None,
            base_shader,

            paths: StorageBuffer::new::<PathData>(device, "stroke path data", 4096 * 16, StorageKind::Buffer),
            geom_bind_group: None,
            geom_bind_group_layout,
        }
    }

    pub fn supports_surface(&self, _surface: SurfacePassConfig) -> bool {
        true
    }

    pub fn begin_frame(&mut self) {
        self.draws.clear();
        self.batches.clear();
        self.curves.clear();
        self.path_data.clear();
        self.instance_range = None;
        self.ibo_range = None;
        self.paths.begin_frame();
    }

    pub fn stroke_path<P: Into<FilledPath>>(
        &mut self,
        ctx: &mut RenderPassContext,
        transforms: &Transforms,
        path: P,
        pattern: BuiltPattern,
        width: f32,
    ) {
        self.stroke_shape(ctx, transforms, Shape::Path(path.into(), width), pattern);
    }

//    pub fn stroke_rect(&mut self, ctx: &mut Context, rect: LocalRect, pattern: BuiltPattern) {
//        self.stroke_shape(ctx, Shape::Rect(rect), pattern);
//    }
//
//    pub fn stroke_circle(&mut self, ctx: &mut Context, circle: Circle, pattern: BuiltPattern) {
//        self.stroke_shape(ctx, Shape::Circle(circle), pattern);
//    }

    fn stroke_shape(&mut self, ctx: &mut RenderPassContext, transforms: &Transforms, shape: Shape, pattern: BuiltPattern) {
        let transform = transforms.current_id();
        let z_index = ctx.z_indices.push();

        let aabb = transforms
            .get_current()
            .matrix()
            .outer_transformed_box(&shape.aabb());

        let mut batch_flags = BatchFlags::empty();
        if pattern.is_opaque && ctx.surface.depth {
            batch_flags |= BatchFlags::ORDER_INDEPENDENT;
        }
        self.batches.find_or_add_batch(
            ctx,
            &pattern.batch_key(),
            &aabb,
            batch_flags,
            &mut || BatchInfo {
                draws: 0..0,
                blend_mode: pattern.blend_mode,
            },
        ).push(Stroke {
            shape,
            pattern,
            transform,
            z_index,
        });
    }

    pub fn prepare(&mut self, ctx: &BuiltRenderPass, transforms: &Transforms, shaders: &mut PrepareRenderPipelines) {
        if self.batches.is_empty() {
            return;
        }

        let id = self.renderer_id;
        let mut batches = self.batches.take();
        for batch_id in ctx
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

            let mut geom_start = self.curves.len() as u32;

            let mut max_segments_per_instance = 1;
            for stroke in commands.iter() {
                if key != stroke.pattern.shader_and_bindings() {
                    let end = self.curves.len() as u32;
                    if end > geom_start {
                        self.draws.push(Draw {
                            segment_count: max_segments_per_instance,
                            instances: geom_start..end,
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
                    key = stroke.pattern.shader_and_bindings();
                    max_segments_per_instance = 1;
                }
                self.prepare_stroke(stroke, transforms, &mut max_segments_per_instance);
            }

            let end = self.curves.len() as u32;
            if end > geom_start {
                self.draws.push(Draw {
                    segment_count: max_segments_per_instance, // TODO
                    instances: geom_start..end,
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

    fn prepare_stroke(&mut self, shape: &Stroke, transforms: &Transforms, max_segments_per_instance: &mut u32) {
        let transform = transforms.get(shape.transform).matrix();
        let z_index = shape.z_index;
        let pattern = shape.pattern.data;

        let &Shape::Path(.., width) = &shape.shape;

        let path_idx = self.path_data.len() as u32;
        match &shape.shape {
            Shape::Path(shape, ..) => {
                let scale = f32::max(transform.m11, transform.m22);
                let tolerance = self.tolerance / scale;
                write_curves(shape.path.as_slice(), path_idx, tolerance, &mut self.curves, max_segments_per_instance);
            }
        }

        self.path_data.push(PathData {
            transform: transform.to_array(),
            width,
            pad0: 0.0,
            pattern,
            z_index,
            pad1: 0,
            pad2: 0,
        });
    }

    pub fn upload(
        &mut self,
        resources: &mut GpuResources,
        shaders: &Shaders,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        self.instance_range = resources.common
            .vertices
            .upload(device, bytemuck::cast_slice(&self.curves));

        if !self.path_data.is_empty() {
            self.paths.bump_allocator().push(self.path_data.len());
            // TODO: this should be a
            if self.paths.ensure_allocated(device) {
                self.geom_bind_group = None;
            }

            self.geom_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("msaa stroker geom"),
                layout: &shaders.get_bind_group_layout(self.geom_bind_group_layout).handle,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.paths.binding_resource(),
                    },
                ]
            }));

            self.paths.upload_bytes(0, bytemuck::cast_slice(&self.path_data), queue);
        }
    }
}

impl core::Renderer for MsaaStrokeRenderer {
    fn prepare(&mut self, ctx: &mut PrepareContext) {
        self.prepare(ctx.pass, ctx.transforms, ctx.pipelines);
    }

    fn upload(&mut self, ctx: &mut UploadContext) {
        self.upload(ctx.resources, ctx.shaders, ctx.wgpu.device, ctx.wgpu.queue);
    }

    fn render<'pass, 'resources: 'pass>(
        &self,
        batches: &[BatchId],
        _surface_info: &SurfacePassConfig,
        ctx: core::RenderContext<'resources>,
        render_pass: &mut wgpu::RenderPass<'pass>,
    ) {
        render_pass.set_bind_group(
            1,
            self.geom_bind_group.as_ref().unwrap(),
            &[],
        );

        render_pass.set_vertex_buffer(
            0,
            ctx.resources.common
                .vertices
                .get_buffer_slice(self.instance_range.as_ref().unwrap()),
        );

        let mut helper = DrawHelper::new();
        for batch_id in batches {
            let (_, _, batch_info) = self.batches.get(batch_id.index);
            for draw in &self.draws[usize_range(batch_info.draws.clone())] {
                let pipeline = ctx.render_pipelines.get(draw.pipeline_idx).unwrap();

                helper.resolve_and_bind(2, draw.pattern_inputs, ctx.bindings, render_pass);

                render_pass.set_pipeline(pipeline);
                let vertex_count = draw.segment_count * 2 + 2;
                render_pass.draw(0..vertex_count, draw.instances.clone());
            }
        }
    }
}

fn write_curves(path: PathSlice, path_index: u32, tolerance: f32, curves: &mut Vec<CurveInstance>, max_segments_per_instance: &mut u32) {
    // TODO: cull curves that are outside of the view.

    let mut builder = InstanceBuilder::new(curves, path_index, tolerance);
    for evt in path {
        match evt {
            PathEvent::Begin { at } => {
                builder.begin(at);
            }
            PathEvent::End { last, first, close } => {
                builder.end(last, first, close);
            }
            PathEvent::Line { from, to } => {
                builder.line(&LineSegment { from, to });
            }
            PathEvent::Quadratic { from, ctrl, to } => {
                builder.quadratic_bezier(&QuadraticBezierSegment { from, ctrl, to });
            }
            PathEvent::Cubic { from, ctrl1, ctrl2, to } => {
                builder.cubic_bezier(&CubicBezierSegment { from, ctrl1, ctrl2, to });
            }
        }
    }

    *max_segments_per_instance = u32::max(*max_segments_per_instance, builder.max_segments);
}

const MAX_SEGMENTS_PER_INSTANCE: u32 = 32;
pub struct InstanceBuilder<'l> {
    prev_ctrl: Point,
    path_index: u32,
    output: &'l mut Vec<CurveInstance>,
    max_segments: u32,
    tolerance: f32,
    is_first: bool,
}

impl<'l> InstanceBuilder<'l> {
    fn new(output: &'l mut Vec<CurveInstance>, path_index: u32, tolerance: f32) -> Self {
        InstanceBuilder {
            prev_ctrl: Point::new(0.0, 0.0),
            path_index,
            output,
            max_segments: 0,
            tolerance,
            is_first: true,
        }
    }

    fn begin(&mut self, at: Point) {
        self.is_first = true;
        self.prev_ctrl = at;
    }

    fn end(&mut self, last: Point, first: Point, close: bool) {
        if close {
            self.line(&LineSegment { from: last, to: first });
            // TODO last to first join
            return;
        }

        // TODO: caps
    }

    fn line(&mut self, line: &LineSegment<f32>) {
        let join_segments = self.compute_segments_per_join(line.from, line.to);
        self.output.push(CurveInstance {
            from: line.from,
            ctrl1: line.from,
            ctrl2: line.to,
            to: line.to,
            prev_ctrl: self.prev_ctrl,
            path_index: self.path_index,
            segment_counts: join_segments | (1 << 16),
        });
        self.prev_ctrl = line.from;
    }

    fn cubic_bezier(&mut self, curve: &CubicBezierSegment<f32>) {
        let join_segments = self.compute_segments_per_join(curve.from, curve.ctrl1);
        // TODO: cusps
        self.push_curve_no_cusp(curve, join_segments);
    }

    fn quadratic_bezier(&mut self, curve: &QuadraticBezierSegment<f32>) {
        let join_segments = self.compute_segments_per_join(curve.from, curve.ctrl);
        // TODO: cusps
        let curve = &curve.to_cubic();
        self.push_curve_no_cusp(curve, join_segments);
    }

    fn compute_segments_per_join(&self, _join: Point, _next: Point) -> u32 {
        if self.is_first {
            return 0;
        }

        // TODO
        2
    }

    fn push_curve_no_cusp(&mut self, curve: &CubicBezierSegment<f32>, mut join_segments: u32) {
        let mut remainging_curve_segments = num_segments_cagd(curve, self.tolerance); // TODO
        debug_assert!(join_segments < MAX_SEGMENTS_PER_INSTANCE);
        let mut t0 = 0.0;
        while remainging_curve_segments > 0 {
            let current_curve_segments = (MAX_SEGMENTS_PER_INSTANCE - join_segments).min(remainging_curve_segments);

            let t1 = current_curve_segments as f32 / remainging_curve_segments as f32;

            let subcurve = curve.split_range(t0..t1);

            self.output.push(CurveInstance {
                from: subcurve.from,
                ctrl1: subcurve.ctrl1,
                ctrl2: subcurve.ctrl2,
                to: subcurve.to,
                prev_ctrl: self.prev_ctrl,
                path_index: self.path_index,
                segment_counts: join_segments | (current_curve_segments << 16),
            });

            self.prev_ctrl = subcurve.ctrl2;

            self.max_segments = u32::max(self.max_segments, current_curve_segments + join_segments + 1);
            remainging_curve_segments -= current_curve_segments;
            join_segments = 0;
            t0 = t1;
        }
    }
}

/// Computes the number of line segments required to build a flattened approximation
/// of the curve with segments placed at regular `t` intervals.
pub fn num_segments_cagd(curve: &CubicBezierSegment<f32>, tolerance: f32) -> u32 {
    let from = curve.from.to_vector();
    let ctrl1 = curve.ctrl1.to_vector();
    let ctrl2 = curve.ctrl2.to_vector();
    let to = curve.to.to_vector();
    let l = (from - ctrl1 * 2.0 + to).max(ctrl1 - ctrl2 * 2.0 + to) * 6.0;
    let num_steps = f32::sqrt(l.length() / (8.0 * tolerance)).ceil() as u32;

    u32::max(num_steps, 1)
}
