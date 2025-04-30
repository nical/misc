use core::batching::{BatchFlags, BatchId, BatchIndex, BatchList, SurfaceIndex};
use core::context::{
    BuiltRenderPass, DrawHelper, RenderPassContext, RendererId, SurfacePassConfig, ZIndex,
};
use core::gpu::shader::{
    BaseShaderId, BlendMode, PrepareRenderPipelines, RenderPipelineIndex, RenderPipelineKey,
};
use core::gpu::StreamId;
use core::pattern::BuiltPattern;
use core::resources::GpuResources;
use core::shape::FilledPath;
use core::transform::{TransformId, Transforms};
use core::units::{point, LocalRect, SurfaceIntRect, SurfaceRect};
use core::{bytemuck, wgpu, PrepareContext, PrepareWorkerContext, UploadContext};

use crate::tiler::{EncodedPathInfo, Tiler, TilerOutput};
use crate::{FillOptions, Occlusion, RendererOptions, TileGpuResources};

use std::ops::Range;
use std::sync::atomic::{AtomicBool, Ordering};

pub(crate) struct BatchInfo {
    pattern: BuiltPattern,
    opaque_draw: Option<Draw>,
    masked_draw: Option<Draw>,
    blend_mode: BlendMode,
}

struct Draw {
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
    pub(crate) tiles: TilerOutput,

    pub(crate) batches: BatchList<Fill, BatchInfo>,
    pub(crate) base_shader: BaseShaderId,
    pub(crate) mask_instances: Option<StreamId>,
    pub(crate) opaque_instances: Option<StreamId>,

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

        if self.batches.is_empty() {
            return;
        }

        if self.occlusion.gpu {
            debug_assert!(pass.surface().depth);
        }

        let size = pass.surface_size();
        self.tiler.begin_target(SurfaceIntRect::from_size(size));
        if self.occlusion.cpu {
            self.tiles
                .occlusion
                .init(size.width as u32, size.height as u32);
        } else {
            self.tiles.occlusion.disable();
        }
        self.tiles.clear();

        let id = self.renderer_id;
        let mut batches = self.batches.take();
        if self.back_to_front {
            for batch_id in pass
                .batches()
                .iter()
                .rev()
                .filter(|batch| batch.renderer == id)
            {
                self.prepare_batch(*batch_id, transforms, shaders, &mut batches);
            }
        } else {
            for batch_id in pass.batches().iter().filter(|batch| batch.renderer == id) {
                self.prepare_batch(*batch_id, transforms, shaders, &mut batches);
            }
        }

        if !self.tiles.mask_tiles.is_empty() {
            let mask_stream = instances.next_stream_id();
            let mut writer = instances.write(mask_stream, 0);
            writer.push_bytes(bytemuck::cast_slice(&self.tiles.mask_tiles));
            self.mask_instances = Some(mask_stream);
        }
        if !self.tiles.opaque_tiles.is_empty() {
            let opaque_stream = instances.next_stream_id();
            let mut writer = instances.write(opaque_stream, 0);
            writer.push_bytes(bytemuck::cast_slice(&self.tiles.opaque_tiles));
            self.opaque_instances = Some(opaque_stream);
        }

        self.batches = batches;
    }

    pub fn prepare_parallel(
        &mut self,
        pass: &BuiltRenderPass,
        transforms: &Transforms,
        workers: &mut PrepareWorkerContext,
    ) {
        #[derive(Clone, Debug)]
        struct Item {
            instances: Range<u32>,
            batch_index: BatchIndex,
            surface_index: SurfaceIndex,
            worker_index: u8,
        }
        struct WorkerData {
            tiler: Tiler,
            tiles: TilerOutput,
            opaque_items: Vec<Item>,
            masked_items: Vec<Item>,
        }

        let counter = std::sync::atomic::AtomicI32::new(0);

        let size = pass.surface_size();
        let mut worker_data = Vec::new();
        for _ in 0..workers.num_workers() {
            let mut tiler = Tiler::new();
            tiler.begin_target(SurfaceIntRect::from_size(size));

            worker_data.push(WorkerData {
                tiler,
                tiles: TilerOutput::new(),
                opaque_items: Vec::with_capacity(2048),
                masked_items: Vec::with_capacity(2048),
            });
        }

        unsafe {
            self.batches.par_iter_mut(
                &mut workers.with_data(&mut worker_data),
                pass,
                self.renderer_id,
                &|ctx, batch, commands, surface, info| {
                    let added_opaque_tiles: AtomicBool = AtomicBool::new(false);
                    let added_mask_tiles: AtomicBool = AtomicBool::new(false);
                    ctx.slice_for_each(&commands, &|ctx, fills| {
                        let worker_index = ctx.index() as u8;
                        let (_, tiler_data) = ctx.data();
                        let opaque_tiles_start = tiler_data.tiles.opaque_tiles.len() as u32;
                        let mask_tiles_start = tiler_data.tiles.mask_tiles.len() as u32;
                        counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        for fill in fills {
                            Self::prepare_fill_impl(
                                &mut tiler_data.tiler,
                                &mut tiler_data.tiles,
                                fill,
                                transforms,
                                self.tolerance,
                            );
                        }
                        let opaque_tiles_end = tiler_data.tiles.opaque_tiles.len() as u32;
                        let mask_tiles_end = tiler_data.tiles.mask_tiles.len() as u32;

                        fn add_item(
                            items: &mut Vec<Item>,
                            range: Range<u32>,
                            batch_index: BatchIndex,
                            surface_index: SurfaceIndex,
                            worker_index: u8,
                        ) {
                            if let Some(last) = items.last_mut() {
                                if last.batch_index == batch_index
                                    && last.instances.end == range.start
                                {
                                    last.instances.end = range.end;
                                    return;
                                }
                            }

                            items.push(Item {
                                instances: range.clone(),
                                batch_index,
                                surface_index,
                                worker_index,
                            });
                        }

                        if opaque_tiles_end > opaque_tiles_start {
                            added_opaque_tiles.store(true, Ordering::Relaxed);
                            add_item(
                                &mut tiler_data.opaque_items,
                                opaque_tiles_start..opaque_tiles_end,
                                batch.index,
                                batch.surface,
                                worker_index,
                            );
                            // TODO
                        }
                        if mask_tiles_end > mask_tiles_start {
                            added_mask_tiles.store(true, Ordering::Relaxed);
                            add_item(
                                &mut tiler_data.masked_items,
                                mask_tiles_start..mask_tiles_end,
                                batch.index,
                                batch.surface,
                                worker_index,
                            );
                            // TODO
                        }
                    });

                    let draw_config = surface.draw_config(true, None);

                    let (core_data, _tiler_data) = ctx.data();

                    if added_opaque_tiles.load(Ordering::Relaxed) {
                        let _pipeline = core_data.pipelines.prepare(RenderPipelineKey::new(
                            self.base_shader,
                            info.pattern.shader,
                            BlendMode::None,
                            draw_config,
                        ));

                        // TODO
                    }

                    if added_mask_tiles.load(Ordering::Relaxed) {
                        let _pipleine = core_data.pipelines.prepare(RenderPipelineKey::new(
                            self.base_shader,
                            info.pattern.shader,
                            info.blend_mode.with_alpha(true),
                            draw_config,
                        ));

                        // TODO
                    }
                },
            );
        } // unsafe

        println!("tiled using {:?} jobs", counter);

        let mut num_opaque_items = 0;
        let mut num_masked_items = 0;
        for worker in &worker_data {
            num_opaque_items += worker.opaque_items.len();
            num_masked_items += worker.masked_items.len();
            println!(
                " - {:?}/{:?} opaque, {:?}/{:?} masked",
                worker.tiles.opaque_tiles.len(),
                worker.opaque_items.len(),
                worker.tiles.mask_tiles.len(),
                worker.masked_items.len()
            )
        }

        let mut ordered_opaque_items = Vec::with_capacity(num_opaque_items);
        let mut ordered_mask_items = Vec::with_capacity(num_masked_items);

        for worker in &worker_data {
            ordered_opaque_items.extend_from_slice(&worker.opaque_items);
            ordered_mask_items.extend_from_slice(&worker.masked_items);
        }

        // TODO: sort the ordered items
    }

    fn prepare_batch(
        &mut self,
        batch_id: BatchId,
        transforms: &Transforms,
        shaders: &mut PrepareRenderPipelines,
        batches: &mut BatchList<Fill, BatchInfo>,
    ) {
        let (commands, surface, info) = &mut batches.get_mut(batch_id.index);

        let opaque_tiles_start = self.tiles.opaque_tiles.len() as u32;
        let mask_tiles_start = self.tiles.mask_tiles.len() as u32;

        if self.back_to_front {
            for fill in commands.iter().rev() {
                self.prepare_fill(fill, transforms);
            }
        } else {
            for fill in commands.iter() {
                self.prepare_fill(fill, transforms);
            }
        }

        let opaque_tiles_end = self.tiles.opaque_tiles.len() as u32;
        let mask_tiles_end = self.tiles.mask_tiles.len() as u32;

        let draw_config = surface.draw_config(true, None);
        if opaque_tiles_end > opaque_tiles_start {
            info.opaque_draw = Some(Draw {
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
                self.tiles.mask_tiles[mask_tiles_start as usize..mask_tiles_end as usize].reverse();
            }
            info.masked_draw = Some(Draw {
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

    fn prepare_fill(&mut self, fill: &Fill, transforms: &Transforms) {
        Self::prepare_fill_impl(
            &mut self.tiler,
            &mut self.tiles,
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

    pub fn upload(
        &mut self,
        resources: &mut GpuResources,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        if self.tiles.is_empty() {
            return;
        }

        //println!("tiling2 stats:\n\tedges: {}({}kb)\n\t{} opaque tiles\n\t{} alpha tiles\n\t",
        //    self.tiles.edges.len(),
        //    (self.tiles.edges.len() * 4) as f32 / 1000.0,
        //    self.tiles.opaque_tiles.len(),
        //    self.tiles.mask_tiles.len(),
        //);

        let edges_per_row = 1024;
        let rem = edges_per_row - self.tiles.edges.len() % edges_per_row;
        if rem != edges_per_row {
            self.tiles.edges.reserve(rem);
            for _ in 0..rem {
                self.tiles.edges.push(0);
            }
        }

        let rows = (self.tiles.edges.len() / edges_per_row) as u32;
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.resources.edge_texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&self.tiles.edges),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(edges_per_row as u32 * 4),
                rows_per_image: Some(rows),
            },
            wgpu::Extent3d {
                width: edges_per_row as u32,
                height: rows,
                depth_or_array_layers: 1,
            },
        );

        let paths_per_row = 256; // 512 texels with 2 texels per path.
        let rem = paths_per_row - self.tiles.paths.len() % paths_per_row;
        if rem != paths_per_row {
            self.tiles.paths.reserve(rem);
            for _ in 0..rem {
                self.tiles.paths.push([0; 8]);
            }
        }

        let rows = (self.tiles.paths.len() / paths_per_row) as u32;
        let size_of_pathinfo = std::mem::size_of::<EncodedPathInfo>() as u32;
        let size_of_texel = 16;
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.resources.path_texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&self.tiles.paths),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(paths_per_row as u32 * size_of_pathinfo),
                rows_per_image: Some(rows),
            },
            wgpu::Extent3d {
                width: paths_per_row as u32 * (size_of_pathinfo / size_of_texel),
                height: rows,
                depth_or_array_layers: 1,
            },
        );
    }
}

impl core::Renderer for TileRenderer {
    fn prepare(&mut self, ctx: &mut PrepareContext) {
        // TODO: Don't run it twice!
        self.prepare_impl(ctx);
        //self.prepare_parallel(ctx.pass, ctx.transforms, &mut ctx.workers);
    }

    fn upload(&mut self, ctx: &mut UploadContext) {
        self.upload(ctx.resources, ctx.wgpu.device, ctx.wgpu.queue);
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

        render_pass.set_index_buffer(
            common_resources.quad_ibo.slice(..),
            wgpu::IndexFormat::Uint16,
        );

        // TODO: previous version was grouping the mask and opaque instances
        // in a single instance range which made avoided the need to re-bind.
        let mask_instances = self
            .mask_instances
            .and_then(|id| common_resources.instances.resolve(id));

        let opaque_instances = self
            .opaque_instances
            .and_then(|id| common_resources.instances.resolve(id));

        render_pass.set_bind_group(1, &self.resources.bind_group, &[]);

        let mut helper = DrawHelper::new();

        for batch_id in batches {
            let (_, _, batch) = self.batches.get(batch_id.index);
            helper.resolve_and_bind(2, batch.pattern.bindings, ctx.bindings, render_pass);

            if let Some(opaque) = &batch.opaque_draw {
                let (opaque_buffer, opaque_byte_range) = opaque_instances.as_ref().unwrap();
                render_pass.set_vertex_buffer(
                    0,
                    opaque_buffer
                        .slice(opaque_byte_range.start as u64..opaque_byte_range.end as u64),
                );

                let pipeline = ctx.render_pipelines.get(opaque.pipeline).unwrap();
                let instances = opaque.tiles.clone();

                render_pass.set_pipeline(pipeline);
                render_pass.draw_indexed(0..6, 0, instances);
            }
            if let Some(masked) = &batch.masked_draw {
                let (mask_buffer, mask_byte_range) = mask_instances.as_ref().unwrap();
                render_pass.set_vertex_buffer(
                    0,
                    mask_buffer.slice(mask_byte_range.start as u64..mask_byte_range.end as u64),
                );

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
