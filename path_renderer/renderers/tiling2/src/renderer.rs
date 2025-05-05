use core::batching::{BatchFlags, BatchId, BatchList};
use core::context::{
    DrawHelper, RenderPassContext, RendererId, SurfacePassConfig, ZIndex,
};
use core::gpu::shader::{
    BaseShaderId, BlendMode, PrepareRenderPipelines, RenderPipelineIndex, RenderPipelineKey,
};
use core::gpu::{StreamId, TransferOps};
use core::pattern::BuiltPattern;
use core::shape::FilledPath;
use core::transform::{TransformId, Transforms};
use core::units::{point, LocalRect, SurfaceIntRect, SurfaceRect};
use core::{wgpu, PrepareContext, UploadContext};

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
    pub(crate) base_shader: BaseShaderId,
    pub(crate) mask_instances: Option<StreamId>,
    pub(crate) opaque_instances: Option<StreamId>,

    pub path_transfer_ops: Vec<TransferOps>,
    pub edge_transfer_ops: Vec<TransferOps>,

    pub(crate) resources: TileGpuResources,
}

impl TileRenderer {
    pub fn supports_surface(&self, surface: SurfacePassConfig) -> bool {
        if self.occlusion.gpu && !surface.depth {
            return false;
        }

        true
    }

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
        transforms: &Transforms,
        path: P,
        pattern: BuiltPattern,
    ) {
        self.fill_shape(ctx, transforms, Shape::Path(path.into()), pattern);
    }

    pub fn fill_surface(
        &mut self,
        ctx: &mut RenderPassContext,
        transforms: &Transforms,
        pattern: BuiltPattern,
    ) {
        self.fill_shape(ctx, transforms, Shape::Surface, pattern);
    }

    fn fill_shape(
        &mut self,
        ctx: &mut RenderPassContext,
        transforms: &Transforms,
        shape: Shape,
        mut pattern: BuiltPattern,
    ) {
        let transform = transforms.current_id();
        let z_index = ctx.z_indices.push();

        let aabb = if let Some(aabb) = shape.aabb() {
            transforms
                .get_current()
                .matrix()
                .outer_transformed_box(&aabb)
        } else {
            use std::f32::{MAX, MIN};
            SurfaceRect {
                min: point(MIN, MIN),
                max: point(MAX, MAX),
            }
        };

        let batch_flags = if self.back_to_front || self.no_opaque_batches {
            BatchFlags::empty()
        } else {
            // Each shape has potententially a draw call for the masked tiles and
            // one for the opaque interriors, so letting them overlap would break
            // ordering.
            BatchFlags::NO_OVERLAP | BatchFlags::EARLIEST_CANDIDATE
        };

        if self.no_opaque_batches {
            pattern.is_opaque = false;
        }

        self.batches
            .find_or_add_batch(ctx, &pattern.batch_key(), &aabb, batch_flags, &mut || {
                BatchInfo {
                    pattern,
                    opaque_draw: None,
                    masked_draw: None,
                    blend_mode: pattern.blend_mode,
                }
            })
            .push(Fill {
                shape,
                pattern,
                transform,
                z_index,
            });
    }

    pub fn set_options(&mut self, options: &RendererOptions) {
        self.tolerance = options.tolerance;
        self.occlusion = options.occlusion;
        self.no_opaque_batches = options.no_opaque_batches;
    }

    pub fn prepare_impl(&mut self, ctx: &mut PrepareContext) {
        if self.batches.is_empty() {
            return;
        }

        let pass = &ctx.pass;
        let transforms = &ctx.transforms;
        let worker_data = &mut ctx.workers.data();
        let shaders = &mut worker_data.pipelines;
        let instances = &mut worker_data.instances;

        let mut edge_store = self.resources.edges.begin_frame(ctx.staging_buffers.clone());
        let mut path_store = self.resources.paths.begin_frame(ctx.staging_buffers.clone());
        let mut edges = edge_store.write();
        let mut paths = path_store.write();

        if self.occlusion.gpu {
            debug_assert!(pass.surface().depth);
        }

        let opaque_stream = instances.next_stream_id();
        let mut tiles = TilerOutput {
            opaque_tiles: instances.write(opaque_stream, 0),
            mask_tiles: Vec::new(),
            paths: Vec::new(),
            edges: Vec::new(),
            occlusion: OcclusionBuffer::disabled(),
        };

        let size = pass.surface_size();
        self.tiler.begin_target(SurfaceIntRect::from_size(size));
        if self.occlusion.cpu {
            tiles.occlusion.init(size.width as u32, size.height as u32);
        } else {
            tiles.occlusion.disable();
        }

        let id = self.renderer_id;
        let mut batches = self.batches.take();
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

        if !tiles.edges.is_empty() {
            edges.push_slice(&tiles.edges);
            std::mem::drop(edges);
            self.edge_transfer_ops = vec![edge_store.finish()];
        }

        if !tiles.paths.is_empty() {
            paths.push_slice(&tiles.paths);
            std::mem::drop(paths);
            self.path_transfer_ops = vec![path_store.finish()];
        }

        self.batches = batches;
    }

    pub fn prepare_parallel(&mut self, prep_ctx: &mut PrepareContext) {
        struct WorkerData {
            tiler: Tiler,
        }

        let counter = std::sync::atomic::AtomicI32::new(0);

        let size = prep_ctx.pass.surface_size();
        let mut worker_data = Vec::new();
        for _ in 0..prep_ctx.workers.num_workers() {
            let mut tiler = Tiler::new();
            tiler.begin_target(SurfaceIntRect::from_size(size));

            worker_data.push(WorkerData {
                tiler,
            });
        }

        unsafe {
            self.batches.par_iter_mut(
                &mut prep_ctx.workers.with_data(&mut worker_data),
                prep_ctx.pass,
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
                            paths: Vec::new(),
                            edges: Vec::new(),
                            occlusion: OcclusionBuffer::disabled(),
                        };

                        counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        for fill in fills {
                            Self::prepare_fill_impl(
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
                            let mut mask_tiles = worker_data.instances.write(masked_stream, fill_idx);
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
                                self.base_shader,
                                info.pattern.shader,
                                BlendMode::None,
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
                                self.base_shader,
                                info.pattern.shader,
                                info.blend_mode.with_alpha(true),
                                draw_config,
                            )),
                        });
                    }
                },
            );
        } // unsafe

        println!("tiled using {:?} jobs", counter);
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
                self.prepare_fill(tiles, fill, transforms);
            }
        } else {
            for fill in commands.iter() {
                self.prepare_fill(tiles, fill, transforms);
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
                    self.base_shader,
                    info.pattern.shader,
                    BlendMode::None,
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
                    self.base_shader,
                    info.pattern.shader,
                    info.blend_mode.with_alpha(true),
                    draw_config,
                )),
            })
        }
    }

    fn prepare_fill(&mut self, tiles: &mut TilerOutput, fill: &Fill, transforms: &Transforms) {
        Self::prepare_fill_impl(
            &mut self.tiler,
            tiles,
            fill,
            transforms,
            self.tolerance,
        );
    }

    fn prepare_fill_impl(
        tiler: &mut Tiler,
        tiles: &mut TilerOutput,
        fill: &Fill,
        transforms: &Transforms,
        tolerance: f32,
    ) {
        let transform = transforms.get(fill.transform).matrix();

        match &fill.shape {
            Shape::Path(shape) => {
                let options = FillOptions::new()
                    .with_transform(Some(transform))
                    .with_fill_rule(shape.fill_rule)
                    .with_z_index(fill.z_index)
                    .with_tolerance(tolerance)
                    .with_inverted(shape.inverted);
                tiler.fill_path(shape.path.iter(), &options, &fill.pattern, tiles);
            }
            Shape::Surface => {
                let opacity = 1.0; // TODO
                tiler.fill_surface(&fill.pattern, opacity, fill.z_index, tiles);
            }
        }
    }

    pub fn upload(&mut self, ctx: &mut UploadContext) {
        if self.path_transfer_ops.is_empty() && self.edge_transfer_ops.is_empty() {
            return;
        }

        //println!("tiling2 stats:\n\tedges: {}({}kb)\n\t{} opaque tiles\n\t{} alpha tiles\n\t",
        //    self.tiles.edges.len(),
        //    (self.tiles.edges.len() * 4) as f32 / 1000.0,
        //    self.tiles.opaque_tiles.len(),
        //    self.tiles.mask_tiles.len(),
        //);

        let staging_buffers = ctx.resources.common.staging_buffers.lock().unwrap();

        self.resources.paths.upload(
            &self.path_transfer_ops,
            &*staging_buffers,
            ctx.wgpu.device,
            ctx.wgpu.encoder
        );

        self.resources.edges.upload(
            &self.edge_transfer_ops,
            &*staging_buffers,
            ctx.wgpu.device,
            ctx.wgpu.encoder
        );

        // TODO: check whether the paths and edges texture changed and if
        // so, re-generate the bind group!
    }
}

impl core::Renderer for TileRenderer {
    fn prepare(&mut self, ctx: &mut PrepareContext) {
        // TODO: Don't run it twice!
        self.prepare_impl(ctx);
        //self.prepare_parallel(ctx);
    }

    fn upload(&mut self, ctx: &mut UploadContext) {
        self.upload(ctx);
    }

    fn render<'pass, 'resources: 'pass>(
        &self,
        batches: &[BatchId],
        _surface_info: &SurfacePassConfig,
        ctx: core::RenderContext<'resources>,
        render_pass: &mut wgpu::RenderPass<'pass>,
    ) {
        if self.mask_instances.is_none() && self.opaque_instances.is_none() {
            return;
        }

        let common_resources = &ctx.resources.common;

        // TODO: previous version was grouping the mask and opaque instances
        // in a single instance range which made avoided the need to re-bind.
        let opaque_buffer = common_resources.instances.resolve_buffer_slice(self.opaque_instances);
        let mask_buffer = common_resources.instances.resolve_buffer_slice(self.mask_instances);

        render_pass.set_index_buffer(
            common_resources.quad_ibo.slice(..),
            wgpu::IndexFormat::Uint16,
        );

        render_pass.set_bind_group(1, &self.resources.bind_group, &[]);

        let mut helper = DrawHelper::new();

        for batch_id in batches {
            let (_, _, batch) = self.batches.get(batch_id.index);
            helper.resolve_and_bind(2, batch.pattern.bindings, ctx.bindings, render_pass);

            if let Some(opaque) = &batch.opaque_draw {
                let opaque_buffer = opaque_buffer.or_else(&|| {
                    common_resources.instances.resolve_buffer_slice(opaque.stream_id)
                });
                render_pass.set_vertex_buffer(0, opaque_buffer.unwrap());

                let pipeline = ctx.render_pipelines.get(opaque.pipeline).unwrap();
                let instances = opaque.tiles.clone();

                render_pass.set_pipeline(pipeline);
                render_pass.draw_indexed(0..6, 0, instances);
            }

            if let Some(masked) = &batch.masked_draw {
                let mask_buffer = mask_buffer.or_else(&|| {
                    common_resources.instances.resolve_buffer_slice(masked.stream_id)
                });
                render_pass.set_vertex_buffer(0, mask_buffer.unwrap());

                let pipeline = ctx.render_pipelines.get(masked.pipeline).unwrap();
                let instances = masked.tiles.clone();

                render_pass.set_pipeline(pipeline);
                render_pass.draw_indexed(0..6, 0, instances);
            }
        }
    }
}

impl core::FillPath for TileRenderer {
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
