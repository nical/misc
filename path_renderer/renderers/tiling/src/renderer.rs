use core::batching::{BatchFlags, BatchId, BatchList};
use core::render_pass::{BuiltRenderPass, RenderCommandId, RenderCommands, RenderPassConfig, RenderPassContext, RendererId, ZIndex};
use core::shading::{
    GeometryId, BlendMode, PrepareRenderPipelines, RenderPipelineIndex, RenderPipelineKey,
};
use core::gpu::{GpuBuffer, StreamId, TransferOps, UploadStats};
use core::pattern::BuiltPattern;
use core::shape::FilledPath;
use core::render_task::RenderTaskInfo;
use core::transform::{Transform, TransformId, Transforms};
use core::units::{point, LocalRect, SurfaceIntRect, SurfaceRect};
use core::{wgpu, PrepareContext, UploadContext};
use core::utils::DrawHelper;

use crate::occlusion::OcclusionBuffer;
use crate::tiler::{EncodedTileInstance, Tiler, TilerOutput};
use crate::{FillOptions, Occlusion, RendererOptions, TileGpuResources};

use std::ops::Range;
use std::sync::atomic::{AtomicU32, Ordering};

pub(crate) struct BatchInfo {
    pattern: BuiltPattern,
    opaque_draw: Option<Draw>,
    masked_draw: Option<Draw>,
    blend_mode: BlendMode,
}

struct Draw {
    stream_id: Option<StreamId>,
    tiles: Range<u32>,
    pipeline: RenderPipelineIndex,
}

#[derive(Clone)]
enum Shape {
    Path(FilledPath),
    Surface,
}

impl Shape {
    pub fn aabb(&self) -> Option<LocalRect> {
        match self {
            // TODO: return the correct aabb for inverted shapes.
            Shape::Path(shape) => Some(*shape.path.aabb()),
            Shape::Surface => None,
        }
    }
}

pub(crate) struct Fill {
    shape: Shape,
    task: RenderTaskInfo,
    pattern: BuiltPattern,
    transform: TransformId,
    z_index: ZIndex,
}

pub struct TileRenderer {
    pub(crate) renderer_id: RendererId,
    pub tolerance: f32,
    pub(crate) occlusion: Occlusion,
    pub(crate) no_opaque_batches: bool,
    pub(crate) back_to_front: bool,
    pub(crate) tiler: Tiler,

    pub(crate) batches: BatchList<Fill, BatchInfo>,
    pub(crate) geometry: GeometryId,
    pub(crate) mask_instances: Option<StreamId>,
    pub(crate) opaque_instances: Option<StreamId>,

    pub path_transfer_ops: Vec<TransferOps>,
    pub edge_transfer_ops: Vec<TransferOps>,

    pub(crate) resources: TileGpuResources,
    pub parallel: bool,
}

impl TileRenderer {
    pub fn begin_frame(&mut self) {
        self.batches.clear();
        self.mask_instances = None;
        self.opaque_instances = None;
        self.path_transfer_ops.clear();
        self.edge_transfer_ops.clear();
    }

    pub fn fill_path<P: Into<FilledPath>>(
        &mut self,
        ctx: &mut RenderPassContext,
        transform: &Transform,
        path: P,
        pattern: BuiltPattern,
    ) {
        self.fill_shape(ctx, transform, Shape::Path(path.into()), pattern);
    }

    pub fn fill_surface(
        &mut self,
        ctx: &mut RenderPassContext,
        transform: &Transform,
        pattern: BuiltPattern,
    ) {
        self.fill_shape(ctx, transform, Shape::Surface, pattern);
    }

    fn fill_shape(
        &mut self,
        ctx: &mut RenderPassContext,
        transform: &Transform,
        shape: Shape,
        mut pattern: BuiltPattern,
    ) {
        let transform_id = transform.id();
        let z_index = ctx.z_indices.push();

        let aabb = if let Some(aabb) = shape.aabb() {
            transform
                .matrix()
                .outer_transformed_box(&aabb)
        } else {
            use std::f32::{MAX, MIN};
            SurfaceRect {
                min: point(MIN, MIN),
                max: point(MAX, MAX),
            }
        };

        if self.no_opaque_batches {
            pattern.is_opaque = false;
        }

        self.batches.add(ctx, &pattern.batch_key(), &aabb, BatchFlags::empty(),
            &mut || BatchInfo {
                pattern,
                opaque_draw: None,
                masked_draw: None,
                blend_mode: pattern.blend_mode,
            },
            &mut |mut batch, task| {
                batch.push(Fill {
                    shape: shape.clone(),
                    task: *task,
                    pattern,
                    transform: transform_id,
                    z_index,
                });
            }
        )
    }

    pub fn set_options(&mut self, options: &RendererOptions) {
        self.tolerance = options.tolerance;
        self.occlusion = options.occlusion;
        self.no_opaque_batches = options.no_opaque_batches;
    }

    pub fn prepare_single_thread(&mut self, ctx: &mut PrepareContext, passes: &[BuiltRenderPass]) {
        if self.batches.is_empty() {
            return;
        }

        let transforms = &ctx.transforms;
        let worker_data = &mut ctx.workers.data();
        let shaders = &mut worker_data.pipelines;
        let instances = &mut worker_data.instances;

        let mut edge_store = self.resources.edges.begin_frame(ctx.staging_buffers.clone());
        let mut path_store = self.resources.paths.begin_frame(ctx.staging_buffers.clone());

        let opaque_stream = instances.next_stream_id();
        let mut tiles = TilerOutput {
            opaque_tiles: instances.write(opaque_stream, 0),
            mask_tiles: Vec::new(),
            paths: path_store.write(),
            edges: edge_store.write(),
            occlusion: OcclusionBuffer::disabled(),
        };

        let mut batches = self.batches.take();

        for pass in passes {
            if self.occlusion.gpu {
                debug_assert!(pass.config().depth);
            }

            let size = pass.surface_size();
            self.tiler.begin_target(SurfaceIntRect::from_size(size));
            if self.occlusion.cpu {
                tiles.occlusion.init(size.width as u32, size.height as u32);
            } else {
                tiles.occlusion.disable();
            }

            let id = self.renderer_id;
            if self.back_to_front {
                for batch_id in pass
                    .batches()
                    .iter()
                    .rev()
                    .filter(|batch| batch.renderer == id)
                {
                    self.prepare_batch(*batch_id, transforms, &mut tiles, shaders, &mut batches);
                }
            } else {
                for batch_id in pass.batches().iter().filter(|batch| batch.renderer == id) {
                    self.prepare_batch(*batch_id, transforms, &mut tiles, shaders, &mut batches);
                }
            }

            if tiles.mask_tiles.len() > 0 {
                let masked_stream = instances.next_stream_id();
                let mut mask_tile_writer = instances.write(masked_stream, 0);
                mask_tile_writer.push_slice(&tiles.mask_tiles);
                self.mask_instances = Some(masked_stream);
            }

            if tiles.opaque_tiles.pushed_bytes() > 0 {
                self.opaque_instances = Some(opaque_stream);
            }
        }

        std::mem::drop(tiles);

        self.edge_transfer_ops = vec![edge_store.finish()];
        self.path_transfer_ops = vec![path_store.finish()];

        self.batches = batches;
    }

    pub fn prepare_parallel(&mut self, prep_ctx: &mut PrepareContext, passes: &[BuiltRenderPass]) {
        struct WorkerData {
            tiler: Tiler,
            edges: GpuBuffer,
            paths: GpuBuffer,
        }

        let edge_store = self.resources.edges.begin_frame(prep_ctx.staging_buffers.clone());
        let path_store = self.resources.paths.begin_frame(prep_ctx.staging_buffers.clone());

        let num_workers = prep_ctx.workers.num_workers();
        let mut worker_data = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            worker_data.push(WorkerData {
                tiler: Tiler::new(),
                edges: edge_store.clone(),
                paths: path_store.clone(),
            });
        }

        for pass in passes {
            let size = pass.surface_size();
            for wd in &mut worker_data {
                wd.tiler.begin_target(SurfaceIntRect::from_size(size));
            }

            unsafe {
                self.batches.par_iter_mut(
                    &mut prep_ctx.workers.with_data(&mut worker_data),
                    pass,
                    self.renderer_id,
                    &|ctx, _batch, commands, surface, info| {
                        let added_opaque_tiles: AtomicU32 = AtomicU32::new(0);
                        let added_mask_tiles: AtomicU32 = AtomicU32::new(0);
                        let opaque_stream = ctx.data().0.instances.next_stream_id();
                        let masked_stream = ctx.data().0.instances.next_stream_id();

                        ctx.slice_for_each(&commands, &|ctx, fills, fill_idx| {
                            let (worker_data, tiler_data) = ctx.data();

                            let opaque_tiles = worker_data.instances.write(opaque_stream, fill_idx);
                            let mut tiles = TilerOutput {
                                opaque_tiles,
                                mask_tiles: Vec::new(),
                                paths: tiler_data.paths.write(),
                                edges: tiler_data.edges.write(),
                                occlusion: OcclusionBuffer::disabled(),
                            };

                            let sort_key = fills[0].z_index;

                            for fill in fills {
                                Self::prepare_fill(
                                    &mut tiler_data.tiler,
                                    &mut tiles,
                                    fill,
                                    prep_ctx.transforms,
                                    self.tolerance,
                                );
                            }
                            let opaque_tiles_count = tiles.opaque_tiles.pushed_items::<EncodedTileInstance>();
                            let mask_tiles_count = tiles.mask_tiles.len() as u32;

                            if opaque_tiles_count > 0 {
                                added_opaque_tiles.fetch_add(opaque_tiles_count, Ordering::Relaxed);
                            }
                            if mask_tiles_count > 0 {
                                added_mask_tiles.fetch_add(mask_tiles_count, Ordering::Relaxed);
                                // TODO: if processing front-to-back, reverse the instances.
                                let mut mask_tiles = worker_data.instances.write(masked_stream, sort_key);
                                mask_tiles.push_slice(&tiles.mask_tiles);
                            }
                        });

                        let draw_config = surface.draw_config(true, None);

                        let (core_data, _tiler_data) = ctx.data();

                        let opaque_tile_count = added_opaque_tiles.load(Ordering::Relaxed);
                        if opaque_tile_count > 0 {
                            info.opaque_draw = Some(Draw {
                                stream_id: Some(opaque_stream),
                                tiles: 0..opaque_tile_count,
                                pipeline: core_data.pipelines.prepare(RenderPipelineKey::new(
                                    self.geometry,
                                    info.pattern.shader,
                                    BlendMode::Overwrite,
                                    draw_config,
                                )),
                            });
                        }

                        let mask_tile_count = added_mask_tiles.load(Ordering::Relaxed);
                        if mask_tile_count > 0 {
                            info.masked_draw = Some(Draw {
                                stream_id: Some(masked_stream),
                                tiles: 0..mask_tile_count,
                                pipeline: core_data.pipelines.prepare(RenderPipelineKey::new(
                                    self.geometry,
                                    info.pattern.shader,
                                    info.blend_mode.with_alpha(true),
                                    draw_config,
                                )),
                            });
                        }
                    },
                );
            } // unsafe
        }

        self.edge_transfer_ops.clear();
        self.path_transfer_ops.clear();
        for worker in &mut worker_data {
            self.edge_transfer_ops.push(worker.edges.finish());
            self.path_transfer_ops.push(worker.paths.finish());
        }
    }

    fn prepare_batch(
        &mut self,
        batch_id: BatchId,
        transforms: &Transforms,
        tiles: &mut TilerOutput,
        shaders: &mut PrepareRenderPipelines,
        batches: &mut BatchList<Fill, BatchInfo>,
    ) {
        let (commands, surface, info) = &mut batches.get_mut(batch_id.index);

        let opaque_tiles_start = tiles.opaque_tiles.pushed_items::<EncodedTileInstance>();
        let mask_tiles_start = tiles.mask_tiles.len() as u32;

        if self.back_to_front {
            for fill in commands.iter().rev() {
                Self::prepare_fill(
                    &mut self.tiler,
                    tiles,
                    fill,
                    transforms,
                    self.tolerance,
                );
            }
        } else {
            for fill in commands.iter() {
                Self::prepare_fill(
                    &mut self.tiler,
                    tiles,
                    fill,
                    transforms,
                    self.tolerance,
                );
            }
        }

        let opaque_tiles_end = tiles.opaque_tiles.pushed_items::<EncodedTileInstance>();
        let mask_tiles_end = tiles.mask_tiles.len() as u32;

        let draw_config = surface.draw_config(true, None);
        if opaque_tiles_end > opaque_tiles_start {
            info.opaque_draw = Some(Draw {
                stream_id: None,
                tiles: opaque_tiles_start..opaque_tiles_end,
                pipeline: shaders.prepare(RenderPipelineKey::new(
                    self.geometry,
                    info.pattern.shader,
                    BlendMode::Overwrite,
                    draw_config,
                )),
            })
        }

        if mask_tiles_end > mask_tiles_start {
            if self.back_to_front {
                // TODO: we can do better: instead of reversing and then copying, we can
                // copy each path's tile range into the writer in the right (reversed) order
                // since we don't rely on the ordering of the instances within a path.
                tiles.mask_tiles[mask_tiles_start as usize..mask_tiles_end as usize].reverse();
            }
            info.masked_draw = Some(Draw {
                stream_id: None,
                tiles: mask_tiles_start..mask_tiles_end,
                pipeline: shaders.prepare(RenderPipelineKey::new(
                    self.geometry,
                    info.pattern.shader,
                    info.blend_mode.with_alpha(true),
                    draw_config,
                )),
            })
        }
    }

    fn prepare_fill(
        tiler: &mut Tiler,
        tiles: &mut TilerOutput,
        fill: &Fill,
        transforms: &Transforms,
        tolerance: f32,
    ) {
        tiler.set_scissor_rect(fill.task.bounds.to_i32());
        let transform = transforms.get(fill.transform).matrix();

        match &fill.shape {
            Shape::Path(shape) => {
                let options = FillOptions::new()
                    .with_transform(Some(transform))
                    .with_fill_rule(shape.fill_rule)
                    .with_z_index(fill.z_index)
                    .with_tolerance(tolerance)
                    .with_inverted(shape.inverted)
                    .with_render_task(fill.task.gpu_address);
                tiler.fill_path(shape.path.iter(), &options, &fill.pattern, tiles);
            }
            Shape::Surface => {
                let opacity = 1.0; // TODO
                tiler.fill_surface(&fill.pattern, opacity, fill.z_index, fill.task.gpu_address, tiles);
            }
        }
    }

    pub fn upload(&mut self, ctx: &mut UploadContext) -> UploadStats {
        let mut stats = UploadStats::default();
        if self.path_transfer_ops.is_empty() && self.edge_transfer_ops.is_empty() {
            return stats;
        }

        let staging_buffers = ctx.resources.common.staging_buffers.lock().unwrap();

        stats += self.resources.paths.upload(
            &self.path_transfer_ops,
            &*staging_buffers,
            ctx.wgpu.device,
            ctx.wgpu.encoder
        );

        stats += self.resources.edges.upload(
            &self.edge_transfer_ops,
            &*staging_buffers,
            ctx.wgpu.device,
            ctx.wgpu.encoder
        );

        if self.resources.edges_epoch != self.resources.edges.epoch()
            || self.resources.paths_epoch != self.resources.paths.epoch() {
            self.resources.bind_group = ctx.wgpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tiling2::geom"),
                layout: &ctx.shaders.get_bind_group_layout(self.resources.bind_group_layout).handle,
                entries: &[
                    self.resources.paths.as_bind_group_entry(0).unwrap(),
                    self.resources.edges.as_bind_group_entry(1).unwrap(),
                ],
            });
            self.resources.edges_epoch = self.resources.edges.epoch();
            self.resources.paths_epoch = self.resources.paths.epoch();
        }

        stats
    }
}

const CMD_FLAGS_OPAQUE: u8 = 1;
const CMD_FLAGS_MASKED: u8 = 2;

impl core::Renderer for TileRenderer {
    fn prepare(&mut self, ctx: &mut PrepareContext, passes: &[BuiltRenderPass]) {
        if self.parallel {
            self.prepare_parallel(ctx, passes);
        } else {
            self.prepare_single_thread(ctx, passes);
        }
    }

    fn add_render_commands(&self, batch_id: BatchId, commands: &mut RenderCommands) {
        let mut cmd: RenderCommandId = batch_id.into();
        if self.occlusion.gpu {
            let mut opaque_cmd = cmd;
            let mut masked_cmd = cmd;
            opaque_cmd.flags = CMD_FLAGS_OPAQUE;
            masked_cmd.flags = CMD_FLAGS_MASKED;
            commands.push_order_independent(opaque_cmd);
            commands.push(masked_cmd);
        } else {
            cmd.flags = CMD_FLAGS_OPAQUE | CMD_FLAGS_MASKED;
            commands.push(cmd);
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
        let common_resources = &ctx.resources.common;

        // TODO: previous version was grouping the mask and opaque instances
        // in a single instance range which avoided the need to re-bind.
        let opaque_buffer = common_resources.instances.resolve_buffer_slice(self.opaque_instances);
        let mask_buffer = common_resources.instances.resolve_buffer_slice(self.mask_instances);

        render_pass.set_index_buffer(
            common_resources.quad_ibo.slice(..),
            wgpu::IndexFormat::Uint16,
        );

        render_pass.set_bind_group(1, &self.resources.bind_group, &[]);

        let mut helper = DrawHelper::new();

        for cmd in commands {
            let (_, _, batch) = self.batches.get(cmd.index);
            helper.resolve_and_bind(2, batch.pattern.bindings, ctx.bindings, render_pass);

            if (cmd.flags & CMD_FLAGS_OPAQUE) != 0 {
                if let Some(opaque) = &batch.opaque_draw {
                    let opaque_buffer = opaque_buffer.or_else(&|| {
                        common_resources.instances.resolve_buffer_slice(opaque.stream_id)
                    });

                    let query = ctx.gpu_profiler.begin_query("opaque tiles", render_pass);

                    render_pass.set_vertex_buffer(0, opaque_buffer.unwrap());

                    let pipeline = ctx.render_pipelines.get(opaque.pipeline).unwrap();
                    let instances = opaque.tiles.clone();

                    render_pass.set_pipeline(pipeline);
                    render_pass.draw_indexed(0..6, 0, instances);
                    ctx.stats.draw_calls += 1;

                    ctx.gpu_profiler.end_query(render_pass, query);
                }
            }

            if (cmd.flags & CMD_FLAGS_MASKED) != 0 {
                if let Some(masked) = &batch.masked_draw {
                    let mask_buffer = mask_buffer.or_else(&|| {
                        common_resources.instances.resolve_buffer_slice(masked.stream_id)
                    });

                    let query = ctx.gpu_profiler.begin_query("masked tiles", render_pass);

                    render_pass.set_vertex_buffer(0, mask_buffer.unwrap());

                    let pipeline = ctx.render_pipelines.get(masked.pipeline).unwrap();
                    let instances = masked.tiles.clone();

                    render_pass.set_pipeline(pipeline);
                    render_pass.draw_indexed(0..6, 0, instances);
                    ctx.stats.draw_calls += 1;

                    ctx.gpu_profiler.end_query(render_pass, query);
                }
            }
        }
    }
}
