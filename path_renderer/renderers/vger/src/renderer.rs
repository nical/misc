use crate::path_scanner::PathScanner;
use crate::resources::Instance;
use core::batching::{BatchFlags, BatchList};
use core::geom::{CubicBezierSegment, QuadraticBezierSegment};
use core::gpu::{GpuBufferAddress, GpuBufferResources, GpuBufferWriter, GpuStreamWriter, StreamId, TransferOps, UploadStats};
use core::path::{FillRule, PathEvent};
use core::pattern::BuiltPattern;
use core::render_pass::{
    BuiltRenderPass, RenderCommandId, RenderPassConfig, RenderPassContext, RendererId, ZIndex,
};
use core::render_task::RenderTask;
use core::shading::{BindGroupLayoutId, GeometryId, RenderPipelineIndex, RenderPipelineKey};
use core::shape::FilledPath;
use core::transform::{Transform, TransformId, Transforms};
use core::units::{point, SurfaceRect};
use core::utils::DrawHelper;
use core::{wgpu, PrepareContext, UploadContext};

use std::ops::Range;

fn pt(p: core::geom::Point<f32>) -> [f32; 2] {
    [p.x, p.y]
}

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

pub struct VgerRenderer {
    #[allow(unused)]
    renderer_id: RendererId,
    geometry: GeometryId,
    batches: BatchList<Fill, BatchInfo>,
    instances: Option<StreamId>,

    scanner: PathScanner,
    band_curves: Vec<[f32; 2]>,

    pub(crate) curve_store: GpuBufferResources,
    pub(crate) curve_transfer_ops: Vec<TransferOps>,
    pub(crate) bind_group: wgpu::BindGroup,
    pub(crate) bind_group_layout: BindGroupLayoutId,
    curve_epoch: u32,
}

impl VgerRenderer {
    pub(crate) fn new(
        renderer_id: RendererId,
        geometry: GeometryId,
        bind_group_layout: BindGroupLayoutId,
        curve_store: GpuBufferResources,
        bind_group: wgpu::BindGroup,
    ) -> Self {
        let curve_epoch = curve_store.epoch();
        VgerRenderer {
            renderer_id,
            geometry,
            batches: BatchList::new(renderer_id),
            instances: None,
            scanner: PathScanner::new(),
            band_curves: Vec::new(),
            curve_store,
            curve_transfer_ops: Vec::new(),
            bind_group,
            bind_group_layout,
            curve_epoch,
        }
    }

    pub fn begin_frame(&mut self) {
        self.batches.clear();
        self.instances = None;
        self.curve_transfer_ops.clear();
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
        scanner: &mut PathScanner,
        band_curves: &mut Vec<[f32; 2]>,
        curve_writer: &mut GpuBufferWriter,
        instances: &mut GpuStreamWriter,
        fill: &Fill,
        transforms: &Transforms,
    ) {
        let matrix: &core::geom::Transform<f32> =
            unsafe { std::mem::transmute(transforms.get(fill.transform).matrix()) };

        let fill_rule = match fill.path.fill_rule {
            FillRule::EvenOdd => 0u32,
            FillRule::NonZero => 1u32,
        };

        scanner.clear();
        for event in fill.path.path.iter() {
            match event {
                PathEvent::Begin { .. } => {}
                PathEvent::Line { from, to } => {
                    let from = matrix.transform_point(from);
                    let to = matrix.transform_point(to);
                    let mid = from.lerp(to, 0.5);
                    scanner.segments.push(
                        crate::path_scanner::PathSegment::new(pt(from), pt(mid), pt(to)),
                    );
                }
                PathEvent::Quadratic { from, ctrl, to, .. } => {
                    let seg = QuadraticBezierSegment { from, ctrl, to }.transformed(matrix);
                    seg.for_each_y_monotonic_range(&mut |range| {
                        let sub = seg.split_range(range);
                        scanner.segments.push(
                            crate::path_scanner::PathSegment::new(
                                pt(sub.from), pt(sub.ctrl), pt(sub.to),
                            ),
                        );
                    });
                }
                PathEvent::Cubic { from, ctrl1, ctrl2, to } => {
                    let cubic = CubicBezierSegment { from, ctrl1, ctrl2, to }.transformed(matrix);
                    cubic.for_each_quadratic_bezier(0.2, &mut |quad| {
                        quad.for_each_y_monotonic_range(&mut |range| {
                            let sub = quad.split_range(range);
                            scanner.segments.push(
                                crate::path_scanner::PathSegment::new(
                                    pt(sub.from), pt(sub.ctrl), pt(sub.to),
                                ),
                            );
                        });
                    });
                }
                PathEvent::End { last, first, .. } => {
                    if last != first {
                        let last = matrix.transform_point(last);
                        let first = matrix.transform_point(first);
                        let mid = last.lerp(first, 0.5);
                        scanner.segments.push(
                            crate::path_scanner::PathSegment::new(
                                pt(last), pt(mid), pt(first),
                            ),
                        );
                    }
                }
            }
        }

        if scanner.segments.is_empty() {
            return;
        }

        scanner.init();

        while scanner.next() {
            band_curves.clear();
            let mut count = 0u32;
            let mut x_min = f32::MAX;
            let mut x_max = f32::MIN;

            let mut index = scanner.first;
            while let Some(seg_idx) = index {
                let seg = &scanner.segments[seg_idx];
                for i in 0..3 {
                    x_min = x_min.min(seg.curve[i][0]);
                    x_max = x_max.max(seg.curve[i][0]);
                }
                band_curves.push(seg.curve[0]);
                band_curves.push(seg.curve[1]);
                band_curves.push(seg.curve[2]);
                count += 1;
                index = seg.next;
            }

            if count == 0 {
                continue;
            }

            let curve_start: GpuBufferAddress = curve_writer.push_slice(band_curves);

            let band_rect = SurfaceRect {
                min: point(x_min, scanner.interval.a),
                max: point(x_max, scanner.interval.b),
            };

            instances.push(Instance {
                local_rect: band_rect.cast_unit(),
                z_index: fill.z_index,
                pattern: fill.pattern.data,
                render_task: fill.task.gpu_address.to_u32(),
                curve_start: curve_start.to_u32(),
                curve_count: count,
                fill_rule,
                _padding: [0; 2],
            });
        }
    }

    pub fn upload(&mut self, ctx: &mut UploadContext) -> UploadStats {
        if self.curve_transfer_ops.is_empty() {
            return UploadStats::default();
        }

        let staging_buffers = ctx.resources.common.staging_buffers.lock().unwrap();

        let stats = self.curve_store.upload(
            &self.curve_transfer_ops,
            &*staging_buffers,
            ctx.wgpu.device,
            ctx.wgpu.encoder,
        );

        if self.curve_epoch != self.curve_store.epoch() {
            self.bind_group = ctx.wgpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("vger::curves"),
                layout: &ctx
                    .shaders
                    .get_bind_group_layout(self.bind_group_layout)
                    .handle,
                entries: &[self.curve_store.as_bind_group_entry(0).unwrap()],
            });
            self.curve_epoch = self.curve_store.epoch();
        }

        stats
    }
}

impl core::Renderer for VgerRenderer {
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
        let mut curve_writer = curve_store.write_items::<[f32; 2]>();

        for (fills, surface, batch) in self.batches.iter_mut() {
            let start = instances.pushed_items::<Instance>();

            for fill in fills.iter() {
                Self::prepare_fill(
                    &mut self.scanner,
                    &mut self.band_curves,
                    &mut curve_writer,
                    &mut instances,
                    fill,
                    ctx.transforms,
                );
            }

            let end = instances.pushed_items::<Instance>();
            batch.instances = start..end;

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
        if has_curve_data {
            self.curve_transfer_ops = vec![curve_store.finish()];
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
