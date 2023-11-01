use lyon::geom::euclid::Size2D;
use core::{
    units::{LocalRect, point},
    shape::{PathShape, Circle},
    canvas::{Context, RenderPasses, SubPass, CanvasRenderer, RendererId, RenderPassState, DrawHelper, SurfaceFeatures, ZIndex, RenderContext},
    resources::{GpuResources, ResourcesHandle, CommonGpuResources},
    pattern::BuiltPattern,
    gpu::{shader::{SurfaceConfig, PrepareRenderPipelines, RenderPipelines, RenderPipelineKey}, Shaders}, BindingResolver, batching::{BatchId, BatchFlags, BatchList}, u32_range, transform::TransformId,
};
use crate::{Tiler, TilerConfig, TILE_SIZE, TileMask, encoder::SRC_COLOR_ATLAS_BINDING};
use super::{encoder::TileEncoder, TilingGpuResources, mask::MaskEncoder, FillOptions, Stats};
use pattern_texture::TextureRenderer;
use core::wgpu;
use core::bytemuck;
use std::ops::Range;


struct Fill {
    shape: Shape,
    pattern: BuiltPattern,
    transform: TransformId,
    z_index: ZIndex,
}

// TODO: the enum prevents other types of shapes from being added externally.
pub enum Shape {
    Path(PathShape),
    Rect(LocalRect),
    Circle(Circle),
    Canvas,
}

impl Shape {
    pub fn aabb(&self) -> LocalRect {
        match self {
            Shape::Path(shape) => shape.aabb(),
            Shape::Rect(rect) => *rect,
            Shape::Circle(circle) => circle.aabb(),
            Shape::Canvas => LocalRect {
                min: point(std::f32::MIN, std::f32::MIN),
                max: point(std::f32::MAX, std::f32::MAX),
            }
        }
    }
}

pub struct TileRenderer {
    pub encoder: TileEncoder,
    pub tiler: Tiler,
    pub occlusion_mask: TileMask,
    batches: BatchList<Fill, Range<u32>>,
    renderer_id: RendererId,
    common_resources: ResourcesHandle<CommonGpuResources>,
    resources: ResourcesHandle<TilingGpuResources>,
    tolerance: f32,
    masks: TilingMasks,
    current_mask_atlas: u32,
    current_color_atlas: u32,
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
            occlusion_mask: TileMask::new(tiles_x, tiles_y),
            masks: TilingMasks {
                circle_masks: MaskEncoder::new(),
                rectangle_masks: MaskEncoder::new(),
            },
            current_color_atlas: std::u32::MAX,
            current_mask_atlas: std::u32::MAX,
        }
    }

    pub fn supports_surface(&self, surface: SurfaceFeatures) -> bool {
        surface == SurfaceFeatures { depth: false, stencil: false, msaa: false }
    }

    pub fn begin_frame(&mut self, canvas: &Context) {
        let size = canvas.surface.size();
        self.tiler.init(&size.to_f32().cast_unit().into());
        let tiles = (size.to_u32() + Size2D::new(TILE_SIZE-1, TILE_SIZE-1)) / TILE_SIZE;
        self.occlusion_mask.init(tiles.width, tiles.height);

        self.batches.clear();
        self.encoder.reset();
        self.masks.circle_masks.reset();
        self.masks.rectangle_masks.reset();
        self.occlusion_mask.clear();
        self.current_color_atlas = std::u32::MAX;
        self.current_mask_atlas = std::u32::MAX;
    }

    pub fn fill_path<P: Into<PathShape>>(&mut self, canvas: &mut Context, shape: P, pattern: BuiltPattern) {
        self.fill_shape(canvas, Shape::Path(shape.into()), pattern);
    }

    pub fn fill_rect(&mut self, canvas: &mut Context, rect: LocalRect, pattern: BuiltPattern) {
        self.fill_shape(canvas, Shape::Rect(rect), pattern);
    }

    pub fn fill_circle(&mut self, canvas: &mut Context, circle: Circle, pattern: BuiltPattern) {
        self.fill_shape(canvas, Shape::Circle(circle), pattern);
    }

    pub fn fill_canvas(&mut self, canvas: &mut Context, pattern: BuiltPattern) {
        self.fill_shape(canvas, Shape::Canvas, pattern);
    }

    fn fill_shape(&mut self, canvas: &mut Context, shape: Shape, pattern: BuiltPattern) {
        debug_assert!(self.supports_surface(canvas.surface.current_features()));

        let aabb = canvas.transforms.get_current().matrix().outer_transformed_box(&shape.aabb());

        self.batches.find_or_add_batch(
            &mut canvas.batcher,
            &pattern.batch_key(),
            &aabb,
            BatchFlags::empty(),
            &mut || Default::default(),
        ).0.push(Fill {
            shape: shape.into(),
            pattern,
            transform: canvas.transforms.current_id(),
            z_index: canvas.z_indices.push(),
        });
    }

    pub fn prepare(&mut self, canvas: &Context, shaders: &mut PrepareRenderPipelines, device: &wgpu::Device) {
        if self.batches.is_empty() {
            return;
        }

        // Process paths back to front in order to let the occlusion culling logic do its magic.
        let id = self.renderer_id;
        let mut batches = self.batches.take();
        for batch_id in canvas.batcher.batches()
            .iter()
            .rev()
            .filter(|batch| batch.renderer == id) {
            let passes_start = self.encoder.render_passes.len();
            let (commands, info) = batches.get_mut(batch_id.index);
            for fill in commands.iter().rev() {
                self.prepare_fill(fill, canvas, device);
            }
            self.encoder.split_sub_pass();
            let passes_end = self.encoder.render_passes.len();
            let passes = passes_start..passes_end;
            self.encoder.render_passes[passes.clone()].reverse();
            *info = u32_range(passes);
        }

        self.batches = batches;

        self.masks.circle_masks.finish();
        self.masks.rectangle_masks.finish();

        let reversed = true;
        self.encoder.finish(reversed);
    }

    fn prepare_fill(&mut self, fill: &Fill, canvas: &Context, device: &wgpu::Device) {
        let transform = if fill.transform != TransformId::NONE {
            Some(canvas.transforms.get(fill.transform).matrix().to_untyped())
        } else {
            None
        };

        let prerender = fill.pattern.favor_prerendering;

        self.encoder.current_z_index = fill.z_index;
         match &fill.shape {
            Shape::Path(shape) => {
                let options = FillOptions::new()
                    .with_transform(transform.as_ref())
                    .with_fill_rule(shape.fill_rule)
                    .with_prerendered_pattern(prerender)
                    .with_tolerance(self.tolerance)
                    .with_inverted(shape.inverted);
                self.tiler.fill_path(shape.path.iter(), &options, &fill.pattern, &mut self.occlusion_mask, &mut self.encoder, device);
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
            Shape::Canvas => {
                self.tiler.fill_canvas(&fill.pattern, &mut self.occlusion_mask, &mut self.encoder);
            }
        }
    }

    pub fn upload(&mut self,
        resources: &mut GpuResources,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        let tile_resources = &mut resources[self.resources];

        tile_resources.edges.bump_allocator().push(self.encoder.edges.len());
        // TODO: this should come after reserving the memory from potentially multiple tilers
        // however it currently needs to run between the line above and the one below.
        tile_resources.allocate(device);
        // TODO: hard coded offset implies a single tiler can use the edge buffer
        tile_resources.edges.upload_bytes(0, bytemuck::cast_slice(&self.encoder.edges), &queue);

        let common_resources = &mut resources[self.common_resources];

        self.encoder.upload(&mut common_resources.vertices, &device);
        self.masks.circle_masks.upload(&mut common_resources.vertices, &device);
        self.masks.rectangle_masks.upload(&mut common_resources.vertices, &device);

        self.encoder.mask_uploader.unmap();
        self.encoder.mask_uploader.upload_vertices(device, &mut common_resources.vertices);
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

        pass.set_index_buffer(common_resources.quad_ibo.slice(..), wgpu::IndexFormat::Uint16);
        pass.set_bind_group(0, &resources.color_atlas_target_and_gpu_store_bind_group, &[]);

        for batch in self.encoder.get_color_atlas_batches(color_atlas_index) {
            let pattern = &self.encoder.patterns[batch.pattern.index()];
            if let Some(buffer_range) = &pattern.prerendered_vbo_range {
                let pipeline = render_pipelines.look_up(RenderPipelineKey::new(
                    resources.opaque_pipeline,
                    batch.pattern,
                    SurfaceConfig::default(),
                )).unwrap();

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

            pass.set_index_buffer(common_resources.quad_ibo.slice(..), wgpu::IndexFormat::Uint16);
            pass.set_bind_group(0, &resources.masks_bind_group, &[]);

            // TODO: Make it possible to register more mask types without hard-coding them.
            if let Some((buffer_range, instances)) = self.encoder.fill_masks.buffer_and_instance_ranges(mask_atlas_index) {
                pass.set_pipeline(&resources.masks.fill_pipeline);
                pass.set_vertex_buffer(0, common_resources.vertices.get_buffer_slice(buffer_range));
                pass.draw_indexed(0..6, 0, instances);
            }

            if let Some((buffer_range, instances)) = self.masks.circle_masks.buffer_and_instance_ranges(mask_atlas_index) {
                pass.set_pipeline(&resources.masks.circle_pipeline);
                pass.set_vertex_buffer(0, common_resources.vertices.get_buffer_slice(buffer_range));
                pass.draw_indexed(0..6, 0, instances);
            }

            if let Some((buffer_range, instances)) = self.masks.rectangle_masks.buffer_and_instance_ranges(mask_atlas_index) {
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
        pass.set_index_buffer(common_resources.quad_ibo.slice(..), wgpu::IndexFormat::Uint16);
        pass.set_bind_group(0, &common_resources.main_target_and_gpu_store_bind_group, &[]);

        let mut helper = DrawHelper::new();

        for sub_pass in sub_passes {
            let render_pass = &self.encoder.render_passes[sub_pass.internal_index as usize];

            if !render_pass.opaque_batches.is_empty() {
                for batch in &self.encoder.opaque_batches[render_pass.opaque_batches.clone()] {
                    if let Some(range) = &self.encoder.get_opaque_batch_vertices(batch.pattern.index()) {
                        let pipeline = render_pipelines.look_up(RenderPipelineKey::new(
                            resources.opaque_pipeline,
                            batch.pattern,
                            SurfaceConfig::default(),
                        )).unwrap();

                        helper.resolve_and_bind(1, batch.pattern_inputs, bindings, pass);

                        pass.set_vertex_buffer(0, common_resources.vertices.get_buffer_slice(range));
                        pass.set_pipeline(pipeline);
                        pass.draw_indexed(0..6, 0, batch.tiles.clone());
                    }
                }
            }

            if !render_pass.opaque_prerendered_tiles.is_empty() {
                let pipeline = render_pipelines.look_up(RenderPipelineKey::new(
                    resources.opaque_pipeline,
                    resources.texture.load_pattern_id(),
                    SurfaceConfig::default(),
                )).unwrap();

                let range = &self.encoder.ranges.opaque_prerendered_tiles.as_ref().unwrap();
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

                for batch in &self.encoder.alpha_batches[render_pass.alpha_batches.clone()] {
                    let pipeline = render_pipelines.look_up(RenderPipelineKey::new(
                        resources.masked_pipeline,
                        batch.pattern,
                        SurfaceConfig::default(),
                    )).unwrap();

                    if batch.pattern_inputs == SRC_COLOR_ATLAS_BINDING {
                        helper.bind(2, batch.pattern_inputs, &resources.src_color_bind_group, pass);
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
        let (_, passes) = self.batches.get(batch_id.index);
        for pass_idx in passes.clone() {
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

    fn render_pre_pass(
        &self,
        index: u32,
        _shaders: &Shaders,
        render_pipelines: &RenderPipelines,
        resources: &GpuResources,
        bindings: &dyn BindingResolver,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let pass = &self.encoder.render_passes[index as usize];
        if pass.color_pre_pass {
            self.render_color_atlas(pass.color_atlas_index, render_pipelines, &resources[self.common_resources], &resources[self.resources], bindings, encoder)
        }
        if pass.mask_pre_pass {
            self.render_mask_atlas(pass.mask_atlas_index, &resources[self.common_resources], &resources[self.resources], bindings, encoder)
        }
    }

    fn render<'pass, 'resources: 'pass>(
        &self,
        sub_passes: &[SubPass],
        _surface_info: &RenderPassState,
        ctx: RenderContext<'resources>,
        render_pass: &mut wgpu::RenderPass<'pass>,
    ) {
        self.render_pass(sub_passes, ctx.render_pipelines, &ctx.resources[self.common_resources], &ctx.resources[self.resources], ctx.bindings, render_pass);
    }
}
