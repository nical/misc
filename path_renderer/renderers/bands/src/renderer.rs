use crate::resources::Instance;
use core::batching::{BatchFlags, BatchList};
use core::geom::{CubicBezierSegment, QuadraticBezierSegment};
use core::gpu::{GpuBufferResources, GpuBufferWriter, GpuStreamWriter, StreamId, TransferOps, UploadStats};
use core::path::{FillRule, PathEvent};
use core::pattern::BuiltPattern;
use core::render_pass::{
    BuiltRenderPass, RenderCommandId, RenderPassConfig, RenderPassContext, RendererId, ZIndex,
};
use core::render_task::{RenderTask, RenderTaskAdress};
use core::shading::{BindGroupLayoutId, GeometryId, RenderPipelineIndex, RenderPipelineKey};
use core::shape::FilledPath;
use core::transform::{Transform, TransformId, Transforms};
use core::units::*;
use core::utils::DrawHelper;
use core::{wgpu, PrepareContext, UploadContext};

use std::ops::Range;

const BAND_HEIGHT: f32 = 8.0;

struct BatchInfo {
    instances: Range<u32>,
    pattern: BuiltPattern,
    pipeline_idx: Option<RenderPipelineIndex>,
}

struct Fill {
    path: FilledPath,
    transform: TransformId,
    z_index: ZIndex,
    pattern: BuiltPattern,
    task: RenderTask,
}

struct Band {
    min_x: f32,
    max_x: f32,
    indices: Vec<u16>,
}

pub struct BandsBuilder {
    bands: Vec<Band>,
    // For each path, edges are pushed in this temporary vector and
    // then copied over to the gpu buffer. Band indices store offsets
    // relative to the start of the current path's edges; they are
    // adjusted to absolute texture positions in end_path.
    edges: Vec<[f32; 4]>,

    viewport: SurfaceIntRect,
    scissor: SurfaceRect,
    min_band_y: f32,
}

impl BandsBuilder {
    pub fn new() -> Self {
        BandsBuilder {
            bands: Vec::new(),
            edges: Vec::new(),
            viewport: SurfaceIntRect::zero(),
            scissor: SurfaceRect::zero(),
            min_band_y: f32::MAX,
        }
    }

    pub fn begin_frame(&mut self) {
        for band in &mut self.bands {
            band.indices.clear();
        }
        self.edges.clear();
    }

    pub fn begin_target(&mut self, mut viewport: SurfaceIntRect) {
        viewport.min.x = viewport.min.x.max(0);
        viewport.min.y = viewport.min.y.max(0);
        viewport.max.x = viewport.max.x.max(0);
        viewport.max.y = viewport.max.y.max(0);
        self.viewport = viewport;
        self.scissor = viewport.cast();

        self.min_band_y = (self.scissor.min.y / BAND_HEIGHT).floor() * BAND_HEIGHT;
    }

    pub fn set_scissor(&mut self, i32_scissor: SurfaceIntRect) {
        let scissor: SurfaceRect = i32_scissor.cast();
        self.scissor = scissor.intersection_unchecked(&self.viewport.to_f32());

        self.min_band_y = (scissor.min.y / BAND_HEIGHT).floor() * BAND_HEIGHT;
    }

    pub fn begin_path(&mut self) {
        for band in &mut self.bands {
            band.min_x = f32::MAX;
            band.max_x = f32::MIN;
        }

        let n_bands = ((self.scissor.max.y / BAND_HEIGHT).ceil() - (self.scissor.min.y / BAND_HEIGHT).floor()) as usize + 1;
        while self.bands.len() < n_bands {
            self.bands.push(Band {
                indices: Vec::new(),
                min_x: f32::MAX,
                max_x: f32::MIN,
            });
        }
    }

    pub fn end_sub_path(&mut self, start: Point) {
        // Make sure that the last segment that was pushed has a `to` endpoint.
        self.edges.push([
            start.x,
            start.y,
            start.x,
            start.y,
        ]);
    }

    pub fn end_path(
        &mut self,
        curve_writer: &mut GpuBufferWriter,
        index_writer: &mut GpuBufferWriter,
        instances: &mut GpuStreamWriter,
        pattern: &BuiltPattern,
        z_index: u32,
        render_task: RenderTaskAdress,
        fill_rule: u32,
    ) {
        // Push the edges and get the actual base offset in the texture.
        // Using the returned address (rather than a manually-tracked counter)
        // is critical: GpuBufferWriter may skip to a new texture row when
        // the current row is full, leaving a gap that a manual counter would miss.
        let edge_base = curve_writer.push_slice(&self.edges).to_u32();
        self.edges.clear();

        let mut min_y = self.min_band_y;
        for band in &mut self.bands {
            let max_y = min_y + BAND_HEIGHT;
            let index_count = band.indices.len() as u32;

            if index_count == 0 {
                min_y = max_y;
                continue;
            }

            let start = index_writer.push_slice(&band.indices);
            band.indices.clear();

            let instance = Instance {
                rect: SurfaceRect {
                    min: SurfacePoint::new(
                        band.min_x,
                        min_y,
                    ),
                    max: SurfacePoint::new(
                        band.max_x,
                        max_y,
                    ),
                },
                pattern: pattern.data,
                z_index,
                render_task: render_task.to_u32(),
                curve_start: start.to_u32(),
                curve_count: index_count,
                fill_rule,
                path_start: edge_base,
                _padding: 0,
            };

            instances.push(instance);
            min_y = max_y;
        }
    }

    pub fn add_segment(&mut self, segment: &QuadraticBezierSegment<f32>) {
        let edge_offset = self.edges.len() as u16; // relative to this path's base
        self.edges.push([
            segment.from.x,
            segment.from.y,
            segment.ctrl.x,
            segment.ctrl.y,
        ]);
        // segment.to is pushed by the next segment.

        let min_x = segment.from.x.min(segment.to.x);
        let min_y = segment.from.y.min(segment.to.y).min(segment.ctrl.y);
        let max_y = segment.from.y.max(segment.to.y).max(segment.ctrl.y);

        if (max_y < self.scissor.min.y) | (min_y > self.scissor.max.y) {
            return;
        }

        let right_side = min_x > self.scissor.max.x;
        let min_x = min_x.max(self.scissor.min.x);
        let max_x = segment.from.x.max(segment.to.x).min(self.scissor.max.x);

        const INV_BAND_HEIGHT: f32 = 1.0 / BAND_HEIGHT;
        let min_idx = ((min_y - self.min_band_y) * INV_BAND_HEIGHT) as usize;
        let max_idx = (((max_y - self.min_band_y) * INV_BAND_HEIGHT + 1.0) as usize).min(self.bands.len());

        for band in &mut self.bands[min_idx .. max_idx] {
            // If the edge is right of the scissor edge, we don't need to
            // consider it in the shader, but we must still make sure that the
            // the band extends all the way to the right side of the scissor
            // rect.
            band.max_x = band.max_x.max(max_x);
            if !right_side {
                band.min_x = band.min_x.min(min_x);
                band.indices.push(edge_offset);
            }
        }
    }
}

pub struct BandsRenderer {
    #[allow(unused)]
    renderer_id: RendererId,
    geometry: GeometryId,
    batches: BatchList<Fill, BatchInfo>,
    instances: Option<StreamId>,

    builder: BandsBuilder,

    pub(crate) curve_store: GpuBufferResources,
    pub(crate) index_store: GpuBufferResources,
    pub(crate) curve_transfer_ops: Vec<TransferOps>,
    pub(crate) index_transfer_ops: Vec<TransferOps>,
    pub(crate) bind_group: wgpu::BindGroup,
    pub(crate) bind_group_layout: BindGroupLayoutId,
    curve_epoch: u32,
    index_epoch: u32,
}

impl BandsRenderer {
    pub(crate) fn new(
        renderer_id: RendererId,
        geometry: GeometryId,
        bind_group_layout: BindGroupLayoutId,
        curve_store: GpuBufferResources,
        index_store: GpuBufferResources,
        bind_group: wgpu::BindGroup,
    ) -> Self {
        let curve_epoch = curve_store.epoch();
        let index_epoch = index_store.epoch();
        BandsRenderer {
            renderer_id,
            geometry,
            batches: BatchList::new(renderer_id),
            instances: None,
            builder: BandsBuilder::new(),
            curve_store,
            index_store,
            curve_transfer_ops: Vec::new(),
            index_transfer_ops: Vec::new(),
            bind_group,
            bind_group_layout,
            curve_epoch,
            index_epoch,
        }
    }

    pub fn begin_frame(&mut self) {
        self.batches.clear();
        self.instances = None;
        self.curve_transfer_ops.clear();
        self.index_transfer_ops.clear();
        self.builder.begin_frame();
    }

    pub fn supports_surface(&self, _surface: RenderPassConfig) -> bool {
        true
    }

    pub fn fill_path(
        &mut self,
        ctx: &mut RenderPassContext,
        transform: &Transform,
        path: &FilledPath,
        pattern: BuiltPattern,
    ) {
        let z_index = ctx.z_indices.push();
        let transform_id = transform.id();
        let aabb = transform.matrix().outer_transformed_box(path.path.aabb());

        self.batches.add(
            ctx,
            &pattern.batch_key(),
            &aabb,
            BatchFlags::empty(),
            &mut || BatchInfo {
                instances: 0..0,
                pattern,
                pipeline_idx: None,
            },
            &mut |mut batch, task| {
                batch.push(Fill {
                    path: path.clone(),
                    transform: transform_id,
                    z_index,
                    pattern,
                    task: *task,
                });
            },
        );
    }

    fn prepare_fill(
        builder: &mut BandsBuilder,
        curve_writer: &mut GpuBufferWriter,
        index_writer: &mut GpuBufferWriter,
        instances: &mut GpuStreamWriter,
        fill: &Fill,
        transforms: &Transforms,
    ) {
        let matrix: &core::geom::Transform<f32> = unsafe {
            std::mem::transmute(transforms.get(fill.transform).matrix())
        };

        builder.begin_path();

        for event in fill.path.path.iter() {
            match event {
                PathEvent::Begin { .. } => {}
                PathEvent::Line { from, to } => {
                    let from = matrix.transform_point(from);
                    let to = matrix.transform_point(to);
                    builder.add_segment(&QuadraticBezierSegment { from, ctrl: from, to });
                }
                PathEvent::Quadratic { from, ctrl, to, .. } => {
                    let seg = QuadraticBezierSegment { from, ctrl, to }.transformed(matrix);
                    builder.add_segment(&seg);
                }
                PathEvent::Cubic { from, ctrl1, ctrl2, to } => {
                    let cubic = CubicBezierSegment { from, ctrl1, ctrl2, to }.transformed(matrix);
                    cubic.for_each_quadratic_bezier(0.2, &mut |quad| {
                        builder.add_segment(&quad);
                    });
                }
                PathEvent::End { last, first, .. } => {
                    let last = matrix.transform_point(last);
                    let first = matrix.transform_point(first);
                    if last != first {
                        builder.add_segment(&QuadraticBezierSegment {
                            from: last,
                            ctrl: last,
                            to: first,
                        });
                    }
                    builder.end_sub_path(first);
                }
            }
        }

        let fill_rule = match fill.path.fill_rule {
            FillRule::EvenOdd => 0u32,
            FillRule::NonZero => 1u32,
        };

        builder.end_path(
            curve_writer,
            index_writer,
            instances,
            &fill.pattern,
            fill.z_index,
            fill.task.gpu_address,
            fill_rule,
        );
    }

    pub fn upload(&mut self, ctx: &mut UploadContext) -> UploadStats {
        if self.curve_transfer_ops.is_empty() {
            return UploadStats::default();
        }

        let staging_buffers = ctx.resources.common.staging_buffers.lock().unwrap();

        let mut stats = self.curve_store.upload(
            &self.curve_transfer_ops,
            &*staging_buffers,
            ctx.wgpu.device,
            ctx.wgpu.encoder,
        );

        stats = stats + self.index_store.upload(
            &self.index_transfer_ops,
            &*staging_buffers,
            ctx.wgpu.device,
            ctx.wgpu.encoder,
        );

        if (self.curve_epoch != self.curve_store.epoch()) | (self.index_epoch != self.index_store.epoch()) {
            self.bind_group = ctx.wgpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bands::curves"),
                layout: &ctx
                    .shaders
                    .get_bind_group_layout(self.bind_group_layout)
                    .handle,
                entries: &[
                    self.curve_store.as_bind_group_entry(0).unwrap(),
                    self.index_store.as_bind_group_entry(1).unwrap(),
                ],
            });
            self.curve_epoch = self.curve_store.epoch();
            self.index_epoch = self.index_store.epoch();
        }

        stats
    }
}

impl core::Renderer for BandsRenderer {
    fn name(&self) -> &'static str { "bands" }

    fn prepare(&mut self, ctx: &mut PrepareContext, _passes: &[BuiltRenderPass]) {
        if self.batches.is_empty() {
            return;
        }

        let worker_data = &mut ctx.workers.data();
        let shaders = &mut worker_data.pipelines;
        let stream = worker_data.instances.next_stream_id();
        let mut instances = worker_data.instances.write(stream, 0);
        self.instances = Some(stream);

        let mut curve_store = self.curve_store.begin_frame(ctx.staging_buffers.clone());
        let mut curve_writer = curve_store.write_items::<[f32; 4]>();
        let mut index_store = self.index_store.begin_frame(ctx.staging_buffers.clone());
        let mut index_writer = index_store.write_items::<u16>();

        let mut task_rect = None;
        for (fills, surface, batch) in self.batches.iter_mut() {
            let start = instances.pushed_items::<Instance>();

            for fill in fills.iter() {
                if Some(fill.task.target_rect) != task_rect {
                    self.builder.begin_target(fill.task.target_rect);
                    task_rect = Some(fill.task.target_rect);
                }

                Self::prepare_fill(
                    &mut self.builder,
                    &mut curve_writer,
                    &mut index_writer,
                    &mut instances,
                    fill,
                    ctx.transforms,
                );
            }

            let end = instances.pushed_items::<Instance>();
            batch.instances = start..end;
            //println!("bands instances {:?}", batch.instances);

            let idx = shaders.prepare(RenderPipelineKey::new(
                self.geometry,
                batch.pattern.shader,
                batch.pattern.blend_mode.with_alpha(true),
                surface.draw_config(true, None),
            ));
            batch.pipeline_idx = Some(idx);
        }

        let has_curve_data = curve_writer.pushed_bytes() > 0;
        drop(curve_writer);
        drop(index_writer);
        if has_curve_data {
            self.curve_transfer_ops = vec![curve_store.finish()];
            self.index_transfer_ops = vec![index_store.finish()];
        }
    }

    fn upload(&mut self, ctx: &mut UploadContext) -> UploadStats {
        self.upload(ctx)
    }

    fn render<'pass, 'resources: 'pass, 'tmp>(
        &self,
        commands: &[RenderCommandId],
        _surface_info: &RenderPassConfig,
        ctx: core::RenderContext<'resources, 'tmp>,
        render_pass: &mut wgpu::RenderPass<'pass>,
    ) {
        let common = &ctx.resources.common;
        let Some(instance_buffer) = common.instances.resolve_buffer_slice(self.instances) else {
            return;
        };

        let mut helper = DrawHelper::new();

        render_pass.set_index_buffer(common.quad_ibo.slice(..), wgpu::IndexFormat::Uint16);
        render_pass.set_vertex_buffer(0, instance_buffer);
        render_pass.set_bind_group(1, &self.bind_group, &[]);

        for batch_id in commands {
            let (_, _, batch) = self.batches.get(batch_id.index);
            let pipeline = ctx.render_pipelines.get(batch.pipeline_idx.unwrap()).unwrap();

            helper.resolve_and_bind(2, batch.pattern.bindings, ctx.bindings, render_pass);

            render_pass.set_pipeline(pipeline);
            render_pass.draw_indexed(0..6, 0, batch.instances.clone());
            ctx.stats.draw_calls += 1;
        }
    }
}
