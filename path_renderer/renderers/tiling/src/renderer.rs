use super::{encoder::TileEncoder, mask::MaskEncoder, FillOptions, Stats, TilingGpuResources};
use crate::{encoder::SRC_COLOR_ATLAS_BINDING, TiledOcclusionBuffer, Tiler, TilerConfig, TILE_SIZE};
use core::{bytemuck, SurfaceKind};
use core::context::{SurfaceDrawConfig, FillPath};
use core::gpu::shader::{RenderPipelineIndex, GeneratedPipelineId, ShaderPatternId};
use core::wgpu;
use core::{
    batching::{BatchFlags, BatchId, BatchList},
    context::{
        CanvasRenderer, Context, DrawHelper, RenderContext, RenderPassState, RenderPasses,
        RendererId, SubPass, SurfacePassConfig, ZIndex,
    },
    gpu::shader::{PrepareRenderPipelines, RenderPipelineKey, RenderPipelines},
    pattern::BuiltPattern,
    resources::{CommonGpuResources, GpuResources, ResourcesHandle},
    shape::{Circle, FilledPath},
    transform::TransformId,
    u32_range,
    units::{point, LocalRect},
    BindingResolver,
};
use lyon::geom::euclid::Size2D;
use pattern_texture::TextureRenderer;
use std::ops::Range;

struct Fill {
    shape: Shape,
    pattern: BuiltPattern,
    transform: TransformId,
    z_index: ZIndex,
}

// TODO: the enum prevents other types of shapes from being added externally.
pub enum Shape {
    Path(FilledPath),
    Rect(LocalRect),
    Circle(Circle),
    Surface,
}

impl Shape {
    pub fn aabb(&self) -> LocalRect {
        match self {
            Shape::Path(shape) => shape.aabb(),
            Shape::Rect(rect) => *rect,
            Shape::Circle(circle) => circle.aabb(),
            Shape::Surface => LocalRect {
                min: point(std::f32::MIN, std::f32::MIN),
                max: point(std::f32::MAX, std::f32::MAX),
            },
        }
    }
}

struct BatchInfo {
    passes: Range<u32>,
    surface: SurfacePassConfig,
}

pub struct TileRenderer {
    pub encoder: TileEncoder,
    pub tiler: Tiler,
    pub occlusion_mask: TiledOcclusionBuffer,
    batches: BatchList<Fill, BatchInfo>,
    renderer_id: RendererId,
    common_resources: ResourcesHandle<CommonGpuResources>,
    resources: ResourcesHandle<TilingGpuResources>,
    tolerance: f32,
    masks: TilingMasks,
    current_mask_atlas: u32,
    current_color_atlas: u32,
    atlas_batch_pipelines: Vec<RenderPipelineIndex>,
    opaque_batch_pipelines: Vec<RenderPipelineIndex>,
    alpha_batch_pipelines: Vec<RenderPipelineIndex>,
    opaque_prerendered_pipelines: Vec<RenderPipelineIndex>,
    opaque_pipeline: GeneratedPipelineId,
    masked_pipeline: GeneratedPipelineId,
    texture_load_pattern: ShaderPatternId,
}

struct TilingMasks {
    circle_masks: MaskEncoder,
    rectangle_masks: MaskEncoder,
}

impl TileRenderer {
    pub fn new(
        renderer_id: RendererId,
        common_resources_id: ResourcesHandle<CommonGpuResources>,
        resources_id: ResourcesHandle<TilingGpuResources>,
        res: &TilingGpuResources,
        textures: &TextureRenderer,
        config: &TilerConfig,
        texture_load: &TextureRenderer,
    ) -> Self {
        let size = config.view_box.size().to_u32();
        let tiles_x = (size.width + crate::TILE_SIZE - 1) / crate::TILE_SIZE;
        let tiles_y = (size.height + crate::TILE_SIZE - 1) / crate::TILE_SIZE;
        TileRenderer {
            batches: BatchList::new(renderer_id),
            renderer_id,
            common_resources: common_resources_id,
            resources: resources_id,
            encoder: TileEncoder::new(config, texture_load, 8), // TODO number of patterns
            tiler: Tiler::new(config),
            tolerance: config.tolerance,
            occlusion_mask: TiledOcclusionBuffer::new(tiles_x, tiles_y),
            masks: TilingMasks {
                circle_masks: MaskEncoder::new(),
                rectangle_masks: MaskEncoder::new(),
            },
            current_color_atlas: std::u32::MAX,
            current_mask_atlas: std::u32::MAX,
            atlas_batch_pipelines: Vec::new(),
            opaque_batch_pipelines: Vec::new(),
            alpha_batch_pipelines: Vec::new(),
            opaque_prerendered_pipelines: Vec::new(),
            opaque_pipeline: res.opaque_pipeline,
            masked_pipeline: res.masked_pipeline,
            texture_load_pattern: textures.load_pattern_id()
        }
    }

    pub fn supports_surface(&self, surface: SurfacePassConfig) -> bool {
        surface.kind == SurfaceKind::Color
    }

    pub fn begin_frame(&mut self, ctx: &Context) {
        let size = ctx.surface.size();
        self.tolerance = ctx.params.tolerance;
        self.tiler.init(&size.to_f32().cast_unit().into());
        let tiles = (size.to_u32() + Size2D::new(TILE_SIZE - 1, TILE_SIZE - 1)) / TILE_SIZE;
        self.occlusion_mask.init(tiles.width, tiles.height);

        self.batches.clear();
        self.encoder.reset();
        self.masks.circle_masks.reset();
        self.masks.rectangle_masks.reset();
        self.occlusion_mask.clear();
        self.current_color_atlas = std::u32::MAX;
        self.current_mask_atlas = std::u32::MAX;
        self.atlas_batch_pipelines.clear();
        self.opaque_batch_pipelines.clear();
        self.alpha_batch_pipelines.clear();
        self.opaque_prerendered_pipelines.clear();
    }

    pub fn fill_path<P: Into<FilledPath>>(
        &mut self,
        ctx: &mut Context,
        shape: P,
        pattern: BuiltPattern,
    ) {
        self.fill_shape(ctx, Shape::Path(shape.into()), pattern);
    }

    pub fn fill_rect(&mut self, ctx: &mut Context, rect: LocalRect, pattern: BuiltPattern) {
        self.fill_shape(ctx, Shape::Rect(rect), pattern);
    }

    pub fn fill_circle(&mut self, ctx: &mut Context, circle: Circle, pattern: BuiltPattern) {
        self.fill_shape(ctx, Shape::Circle(circle), pattern);
    }

    pub fn fill_surface(&mut self, ctx: &mut Context, pattern: BuiltPattern) {
        self.fill_shape(ctx, Shape::Surface, pattern);
    }

    fn fill_shape(&mut self, ctx: &mut Context, shape: Shape, pattern: BuiltPattern) {
        debug_assert!(self.supports_surface(ctx.surface.current_config()));

        let aabb = ctx
            .transforms
            .get_current()
            .matrix()
            .outer_transformed_box(&shape.aabb());

        self.batches
            .find_or_add_batch(
                &mut ctx.batcher,
                &pattern.batch_key(),
                &aabb,
                BatchFlags::empty(),
                &mut || BatchInfo { passes: 0..0, surface: ctx.surface.current_config() },
            )
            .0
            .push(Fill {
                shape: shape.into(),
                pattern,
                transform: ctx.transforms.current_id(),
                z_index: ctx.z_indices.push(),
            });
    }

    pub fn prepare(
        &mut self,
        ctx: &Context,
        shaders: &mut PrepareRenderPipelines,
        device: &wgpu::Device,
    ) {
        if self.batches.is_empty() {
            return;
        }

        // Process paths back to front in order to let the occlusion culling logic do its magic.
        let id = self.renderer_id;
        let mut batches = self.batches.take();
        for batch_id in ctx
            .batcher
            .batches()
            .iter()
            .rev()
            .filter(|batch| batch.renderer == id)
        {
            let passes_start = self.encoder.render_passes.len();
            let (commands, info) = batches.get_mut(batch_id.index);
            let surface = info.surface.draw_config(false, None);
            for fill in commands.iter().rev() {
                self.prepare_fill(fill, &surface, ctx, device);
            }
            self.encoder.split_sub_pass();
            let passes_end = self.encoder.render_passes.len();
            let passes = passes_start..passes_end;
            self.encoder.render_passes[passes.clone()].reverse();
            info.passes = u32_range(passes);
        }

        self.batches = batches;

        self.masks.circle_masks.finish();
        self.masks.rectangle_masks.finish();

        let reversed = true;
        self.encoder.finish(reversed);

        // TODO: Here we go through batches and prepare the shaders. It would make
        // more sense to do that while producing the batches, however it is a bit
        // inconvenient to have to carry around the PreparePipelines everywhere in
        // the tile encoder.
        for batch in &self.encoder.atlas_pattern_batches {
            self.atlas_batch_pipelines.push(
                shaders.prepare(RenderPipelineKey::new(
                    self.opaque_pipeline,
                    batch.pattern,
                    SurfaceDrawConfig::color()
                ))
            );
        }

        for batch in &self.encoder.opaque_batches {
            self.opaque_batch_pipelines.push(
                shaders.prepare(RenderPipelineKey::new(
                    self.opaque_pipeline,
                    batch.pattern,
                    batch.surface,
                ))
            );
        }

        for batch in &self.encoder.alpha_batches {
            self.alpha_batch_pipelines.push(
                shaders.prepare(RenderPipelineKey::new(
                    self.masked_pipeline,
                    batch.pattern,
                    batch.surface,
                ))
            );
        }

        for pass in &self.encoder.render_passes {
            self.opaque_prerendered_pipelines.push(
                shaders.prepare(RenderPipelineKey::new(
                    self.opaque_pipeline,
                    self.texture_load_pattern,
                    pass.surface,
                ))
            )
        }
    }

    fn prepare_fill(&mut self, fill: &Fill, surface: &SurfaceDrawConfig, ctx: &Context, device: &wgpu::Device) {
        let transform = if fill.transform != TransformId::NONE {
            Some(ctx.transforms.get(fill.transform).matrix().to_untyped())
        } else {
            None
        };

        let prerender = fill.pattern.favor_prerendering;

        self.encoder.current_z_index = fill.z_index;
        self.encoder.current_surface = *surface;
        match &fill.shape {
            Shape::Path(shape) => {
                let options = FillOptions::new()
                    .with_transform(transform.as_ref())
                    .with_fill_rule(shape.fill_rule)
                    .with_prerendered_pattern(prerender)
                    .with_tolerance(self.tolerance)
                    .with_inverted(shape.inverted);
                self.tiler.fill_path(
                    shape.path.iter(),
                    &options,
                    &fill.pattern,
                    &mut self.occlusion_mask,
                    &mut self.encoder,
                    device,
                );
            }
            Shape::Circle(circle) => {
                let options = FillOptions::new()
                    .with_transform(transform.as_ref())
                    .with_prerendered_pattern(prerender)
                    .with_tolerance(self.tolerance)
                    .with_inverted(circle.inverted);
                crate::mask::circle::fill_circle(
                    circle.center,
                    circle.radius,
                    &options,
                    &fill.pattern,
                    &mut self.occlusion_mask,
                    &mut self.tiler,
                    &mut self.encoder,
                    &mut self.masks.circle_masks,
                    device,
                )
            }
            Shape::Rect(rect) => {
                let options = FillOptions::new()
                    .with_transform(transform.as_ref())
                    .with_prerendered_pattern(prerender)
                    .with_tolerance(self.tolerance);
                crate::mask::rect::fill_rect(
                    &rect.to_untyped(),
                    &options,
                    &fill.pattern,
                    &mut self.occlusion_mask,
                    &mut self.tiler,
                    &mut self.encoder,
                    &mut self.masks.rectangle_masks,
                    device,
                )
            }
            Shape::Surface => {
                self.tiler
                    .fill_surface(&fill.pattern, &mut self.occlusion_mask, &mut self.encoder);
            }
        }
    }

    pub fn upload(
        &mut self,
        resources: &mut GpuResources,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        let tile_resources = &mut resources[self.resources];

        tile_resources
            .edges
            .bump_allocator()
            .push(self.encoder.edges.len());
        // TODO: this should come after reserving the memory from potentially multiple tilers
        // however it currently needs to run between the line above and the one below.
        tile_resources.allocate(device);
        // TODO: hard coded offset implies a single tiler can use the edge buffer
        tile_resources
            .edges
            .upload_bytes(0, bytemuck::cast_slice(&self.encoder.edges), &queue);

        let common_resources = &mut resources[self.common_resources];

        self.encoder.upload(&mut common_resources.vertices, &device);
        self.masks
            .circle_masks
            .upload(&mut common_resources.vertices, &device);
        self.masks
            .rectangle_masks
            .upload(&mut common_resources.vertices, &device);

        self.encoder.mask_uploader.unmap();
        self.encoder
            .mask_uploader
            .upload_vertices(device, &mut common_resources.vertices);
    }

    pub fn update_stats(&self, stats: &mut Stats) {
        self.tiler.update_stats(stats);
        self.encoder.update_stats(stats);
        self.masks.circle_masks.update_stats(stats);
        self.masks.rectangle_masks.update_stats(stats);
    }

    pub fn render_color_atlas(
        &self,
        color_atlas_index: u32,
        render_pipelines: &RenderPipelines,
        common_resources: &CommonGpuResources,
        resources: &TilingGpuResources,
        bindings: &dyn BindingResolver,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Color tile atlas"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &resources.src_color_texture_view,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                resolve_target: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        pass.set_index_buffer(
            common_resources.quad_ibo.slice(..),
            wgpu::IndexFormat::Uint16,
        );
        pass.set_bind_group(
            0,
            &resources.color_atlas_target_and_gpu_store_bind_group,
            &[],
        );

        let batch_range = self.encoder.color_atlas_passes[color_atlas_index as usize].clone();
        let batches = &self.encoder.atlas_pattern_batches[batch_range.clone()];
        let pipelines = &self.atlas_batch_pipelines[batch_range];
        for (batch, pipeline_idx) in batches.iter().zip(pipelines) {
            let pattern = &self.encoder.patterns[batch.pattern.index()];
            if let Some(buffer_range) = &pattern.prerendered_vbo_range {
                let pipeline = render_pipelines.get(*pipeline_idx).unwrap();

                // We could try to avoid redundant bindings.
                if batch.pattern_inputs.is_some() {
                    if let Some(group) = bindings.resolve(batch.pattern_inputs) {
                        pass.set_bind_group(1, group, &[]);
                    }
                }

                pass.set_vertex_buffer(0, common_resources.vertices.get_buffer_slice(buffer_range));
                pass.set_pipeline(pipeline);
                pass.draw_indexed(0..6, 0, batch.tiles.clone());
            }
        }
    }

    pub fn render_mask_atlas(
        &self,
        mask_atlas_index: u32,
        common_resources: &CommonGpuResources,
        resources: &TilingGpuResources,
        _bindings: &dyn BindingResolver,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Mask atlas"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &resources.mask_texture_view,
                    ops: wgpu::Operations {
                        // TODO: Clear is actually quite expensive if a large portion of the atlas
                        // is not used. Could decide between Clear or Load depending on how
                        // much of the atlas we are going to render.
                        // We currently rely on clearing with white so that mask tile (0, 0) is entirely
                        // filled. It could be done by rendering the tile instead.
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: wgpu::StoreOp::Store,
                    },
                    resolve_target: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_index_buffer(
                common_resources.quad_ibo.slice(..),
                wgpu::IndexFormat::Uint16,
            );
            pass.set_bind_group(0, &resources.masks_bind_group, &[]);

            // TODO: Make it possible to register more mask types without hard-coding them.
            if let Some((buffer_range, instances)) = self
                .encoder
                .fill_masks
                .buffer_and_instance_ranges(mask_atlas_index)
            {
                pass.set_pipeline(&resources.masks.fill_pipeline);
                pass.set_vertex_buffer(0, common_resources.vertices.get_buffer_slice(buffer_range));
                pass.draw_indexed(0..6, 0, instances);
            }

            if let Some((buffer_range, instances)) = self
                .masks
                .circle_masks
                .buffer_and_instance_ranges(mask_atlas_index)
            {
                pass.set_pipeline(&resources.masks.circle_pipeline);
                pass.set_vertex_buffer(0, common_resources.vertices.get_buffer_slice(buffer_range));
                pass.draw_indexed(0..6, 0, instances);
            }

            if let Some((buffer_range, instances)) = self
                .masks
                .rectangle_masks
                .buffer_and_instance_ranges(mask_atlas_index)
            {
                pass.set_pipeline(&resources.masks.rect_pipeline);
                pass.set_vertex_buffer(0, common_resources.vertices.get_buffer_slice(buffer_range));
                pass.draw_indexed(0..6, 0, instances);
            }
        }

        // TODO: split this in two so that the copy shader can be in the previous pass.
        resources.mask_upload_copies.update_target(
            encoder,
            &common_resources.quad_ibo,
            &common_resources.vertices,
            &self.encoder.mask_uploader,
            mask_atlas_index,
            &resources.mask_texture_view,
            &resources.mask_atlas_target_and_gpu_store_bind_group,
        );
    }

    pub fn render_pass<'pass, 'resources: 'pass>(
        &self,
        sub_passes: &[SubPass],
        render_pipelines: &'resources RenderPipelines,
        common_resources: &'resources CommonGpuResources,
        resources: &'resources TilingGpuResources,
        bindings: &'resources dyn BindingResolver,
        pass: &mut wgpu::RenderPass<'pass>,
    ) {
        pass.set_index_buffer(
            common_resources.quad_ibo.slice(..),
            wgpu::IndexFormat::Uint16,
        );
        pass.set_bind_group(
            0,
            &common_resources.main_target_and_gpu_store_bind_group,
            &[],
        );

        let mut helper = DrawHelper::new();

        for sub_pass in sub_passes {
            let render_pass = &self.encoder.render_passes[sub_pass.internal_index as usize];

            if !render_pass.opaque_batches.is_empty() {
                let batches = &self.encoder.opaque_batches[render_pass.opaque_batches.clone()];
                let pipelines = &self.opaque_batch_pipelines[render_pass.opaque_batches.clone()];
                for (batch, pipeline_idx) in batches.iter().zip(pipelines) {
                    if let Some(range) = &self
                        .encoder
                        .get_opaque_batch_vertices(batch.pattern.index())
                    {
                        let pipeline = render_pipelines.get(*pipeline_idx).unwrap();

                        helper.resolve_and_bind(1, batch.pattern_inputs, bindings, pass);

                        pass.set_vertex_buffer(
                            0,
                            common_resources.vertices.get_buffer_slice(range),
                        );
                        pass.set_pipeline(pipeline);
                        pass.draw_indexed(0..6, 0, batch.tiles.clone());
                    }
                }
            }

            if !render_pass.opaque_prerendered_tiles.is_empty() {
                let pipeline = render_pipelines
                    .look_up(RenderPipelineKey::new(
                        resources.opaque_pipeline,
                        resources.texture.load_pattern_id(),
                        SurfaceDrawConfig::default(),
                    ))
                    .unwrap();

                let range = &self
                    .encoder
                    .ranges
                    .opaque_prerendered_tiles
                    .as_ref()
                    .unwrap();
                pass.set_vertex_buffer(0, common_resources.vertices.get_buffer_slice(range));

                pass.set_pipeline(pipeline);
                pass.set_bind_group(1, &resources.src_color_bind_group, &[]);
                pass.draw_indexed(0..6, 0, render_pass.opaque_prerendered_tiles.clone());

                helper.reset_binding(1);
            }

            if let Some(range) = &self.encoder.ranges.alpha_tiles {
                pass.set_vertex_buffer(0, common_resources.vertices.get_buffer_slice(range));
                pass.set_bind_group(1, &resources.mask_texture_bind_group, &[]);
                helper.reset_binding(1);

                let batches = &self.encoder.alpha_batches[render_pass.alpha_batches.clone()];
                let pipelines = &self.alpha_batch_pipelines[render_pass.alpha_batches.clone()];
                for (batch, pipeline_idx) in batches.iter().zip(pipelines) {
                    let pipeline = render_pipelines.get(*pipeline_idx).unwrap();

                    if batch.pattern_inputs == SRC_COLOR_ATLAS_BINDING {
                        helper.bind(
                            2,
                            batch.pattern_inputs,
                            &resources.src_color_bind_group,
                            pass,
                        );
                    } else {
                        helper.resolve_and_bind(2, batch.pattern_inputs, bindings, pass);
                    }

                    pass.set_pipeline(pipeline);
                    pass.draw_indexed(0..6, 0, batch.tiles.clone());
                }
            }
        }
    }
}

impl CanvasRenderer for TileRenderer {
    fn add_render_passes(&mut self, batch_id: BatchId, render_passes: &mut RenderPasses) {
        let (_, info) = self.batches.get(batch_id.index);
        for pass_idx in info.passes.clone() {
            let pass = &mut self.encoder.render_passes[pass_idx as usize];

            pass.color_pre_pass = self.current_color_atlas != pass.color_atlas_index;
            self.current_color_atlas = pass.color_atlas_index;

            pass.mask_pre_pass = self.current_mask_atlas != pass.mask_atlas_index;
            self.current_mask_atlas = pass.mask_atlas_index;
            render_passes.push(SubPass {
                renderer_id: self.renderer_id,
                internal_index: pass_idx,
                require_pre_pass: pass.mask_pre_pass || pass.color_pre_pass,
                surface: batch_id.surface,
            });
        }
    }

    fn render_pre_pass(&self, index: u32, ctx: RenderContext, encoder: &mut wgpu::CommandEncoder) {
        let pass = &self.encoder.render_passes[index as usize];
        if pass.color_pre_pass {
            self.render_color_atlas(
                pass.color_atlas_index,
                ctx.render_pipelines,
                &ctx.resources[self.common_resources],
                &ctx.resources[self.resources],
                ctx.bindings,
                encoder,
            )
        }
        if pass.mask_pre_pass {
            self.render_mask_atlas(
                pass.mask_atlas_index,
                &ctx.resources[self.common_resources],
                &ctx.resources[self.resources],
                ctx.bindings,
                encoder,
            )
        }
    }

    fn render<'pass, 'resources: 'pass>(
        &self,
        sub_passes: &[SubPass],
        _surface_info: &RenderPassState,
        ctx: RenderContext<'resources>,
        render_pass: &mut wgpu::RenderPass<'pass>,
    ) {
        self.render_pass(
            sub_passes,
            ctx.render_pipelines,
            &ctx.resources[self.common_resources],
            &ctx.resources[self.resources],
            ctx.bindings,
            render_pass,
        );
    }
}

impl FillPath for TileRenderer {
    fn fill_path(
        &mut self,
        ctx: &mut Context,
        path: FilledPath,
        pattern: BuiltPattern,
    ) {
        self.fill_shape(ctx, Shape::Path(path), pattern);
    }
}
