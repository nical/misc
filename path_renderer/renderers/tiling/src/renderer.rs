use lyon::{geom::euclid::Size2D};
use core::{
    canvas::{Fill, Shape, RendererCommandIndex, Canvas, RecordedShape, RenderPasses, SubPass, CanvasRenderer, RendererId, RenderPassState, DrawHelper},
    resources::{GpuResources, ResourcesHandle, CommonGpuResources},
    pattern::{BuiltPattern},
    gpu::{shader::SurfaceConfig, Shaders}, BindingResolver,
};
use crate::{Tiler, TilerConfig, TILE_SIZE, TileMask, encoder::SRC_COLOR_ATLAS_BINDING};
use super::{encoder::TileEncoder, TilingGpuResources, mask::MaskEncoder, FillOptions, Stats};
use pattern_texture::TextureRenderer;
use core::wgpu;
use core::bytemuck;

pub struct TileRenderer {
    pub encoder: TileEncoder,
    pub tiler: Tiler,
    pub occlusion_mask: TileMask,
    commands: Vec<Fill>,
    renderer_id: RendererId,
    common_resources: ResourcesHandle<CommonGpuResources>,
    resources: ResourcesHandle<TilingGpuResources>,
    tolerance: f32,
    masks: TilingMasks,
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
            commands: Vec::new(),
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
        }
    }

    pub fn begin_frame(&mut self, canvas: &Canvas) {
        let size = canvas.surface.size();
        self.tiler.init(&size.to_f32().into());
        let tiles = (size + Size2D::new(TILE_SIZE-1, TILE_SIZE-1)) / TILE_SIZE;
        self.occlusion_mask.init(tiles.width, tiles.height);

        self.commands.clear();
        self.encoder.reset();
        self.masks.circle_masks.reset();
        self.masks.rectangle_masks.reset();
        self.occlusion_mask.clear();
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

    pub fn prepare(&mut self, canvas: &Canvas, device: &wgpu::Device) {
        let commands = std::mem::take(&mut self.commands);
        // Process paths back to front in order to let the occlusion culling logic do its magic.
        for range in canvas.commands.with_renderer_rev(self.renderer_id) {
            let range = range.start as usize .. range.end as usize;
            for fill in commands[range].iter().rev() {
                self.prepare_fill(fill, canvas, device);
            }

            self.encoder.split_sub_pass();
        }

        self.masks.circle_masks.finish();
        self.masks.rectangle_masks.finish();

        let reversed = true;
        self.encoder.finish(reversed);

        self.commands = commands;
    }

    pub fn prepare_fill(&mut self, fill: &Fill, canvas: &Canvas, device: &wgpu::Device) {
        let transform = if fill.transform != 0 {
            Some(canvas.transforms.get(fill.transform))
        } else {
            None
        };

        let prerender = fill.pattern.favor_prerendering;

        self.encoder.current_z_index = fill.z_index;
         match &fill.shape {
            RecordedShape::Path(shape) => {
                let options = FillOptions::new()
                    .with_transform(transform)
                    .with_fill_rule(shape.fill_rule)
                    .with_prerendered_pattern(prerender)
                    .with_tolerance(self.tolerance)
                    .with_inverted(shape.inverted);
                self.tiler.fill_path(shape.path.iter(), &options, &fill.pattern, &mut self.occlusion_mask, &mut self.encoder, device);
            }
            RecordedShape::Circle(circle) => {
                let options = FillOptions::new()
                    .with_transform(transform)
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
            RecordedShape::Rect(rect) => {
                let options = FillOptions::new()
                    .with_transform(transform)
                    .with_prerendered_pattern(prerender)
                    .with_tolerance(self.tolerance);
                crate::mask::rect::fill_rect(
                    rect,
                    &options,
                    &fill.pattern,
                    &mut self.occlusion_mask,
                    &mut self.tiler,
                    &mut self.encoder,
                    &mut self.masks.rectangle_masks,
                    device,
                )
            }
            RecordedShape::Canvas => {
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

        tile_resources.edges.bump_allocator().push(self.tiler.edges.len());
        // TODO: this should come after reserving the memory from potentially multiple tilers
        // however it currently needs to run between the line above and the one below.
        tile_resources.allocate(device);
        // TODO: hard coded offset implies a single tiler can use the edge buffer
        tile_resources.edges.upload_bytes(0, bytemuck::cast_slice(&self.tiler.edges), &queue);

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
        shaders: &Shaders,
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
                    store: true,
                },
                resolve_target: None,
            })],
            depth_stencil_attachment: None,
        });

        pass.set_index_buffer(common_resources.quad_ibo.slice(..), wgpu::IndexFormat::Uint16);
        pass.set_bind_group(0, &resources.color_atlas_target_and_gpu_store_bind_group, &[]);

        for batch in self.encoder.get_color_atlas_batches(color_atlas_index) {
            let pattern = &self.encoder.patterns[batch.pattern.index()];
            if let Some(buffer_range) = &pattern.prerendered_vbo_range {
                let pipeline = shaders.try_get(
                    resources.opaque_pipeline,
                    batch.pattern,
                    SurfaceConfig::default(),
                ).unwrap();
    
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
                        store: true,
                    },
                    resolve_target: None,
                })],
                depth_stencil_attachment: None,
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
        pass_idx: usize,
        shaders: &'resources Shaders,
        common_resources: &'resources CommonGpuResources,
        resources: &'resources TilingGpuResources,
        bindings: &'resources dyn BindingResolver,
        pass: &mut wgpu::RenderPass<'pass>,
    ) {
        let render_pass = &self.encoder.render_passes[pass_idx];

        pass.set_index_buffer(common_resources.quad_ibo.slice(..), wgpu::IndexFormat::Uint16);
        pass.set_bind_group(0, &common_resources.main_target_and_gpu_store_bind_group, &[]);

        let mut helper = DrawHelper::new();

        if !render_pass.opaque_batches.is_empty() {
            for batch in &self.encoder.opaque_batches[render_pass.opaque_batches.clone()] {
                if let Some(range) = &self.encoder.get_opaque_batch_vertices(batch.pattern.index()) {
                    let pipeline = shaders.try_get(
                        resources.opaque_pipeline,
                        batch.pattern,
                        SurfaceConfig::default(),
                    ).unwrap();

                    helper.resolve_and_bind(1, batch.pattern_inputs, bindings, pass);

                    pass.set_vertex_buffer(0, common_resources.vertices.get_buffer_slice(range));
                    // TODO: optional hook to bind extra data
                    pass.set_pipeline(pipeline);
                    pass.draw_indexed(0..6, 0, batch.tiles.clone());
                }
            }
        }

        if !render_pass.opaque_image_tiles.is_empty() {
            let pipeline = shaders.try_get(
                resources.opaque_pipeline,
                resources.texture.load_pattern_id(),
                SurfaceConfig::default(),
            ).unwrap();

            let range = &self.encoder.ranges.opaque_image_tiles.as_ref().unwrap();
            pass.set_vertex_buffer(0, common_resources.vertices.get_buffer_slice(range));

            pass.set_pipeline(pipeline);
            pass.set_bind_group(1, &resources.src_color_bind_group, &[]);
            pass.draw_indexed(0..6, 0, render_pass.opaque_image_tiles.clone());

            helper.reset_binding(1);
        }


        if let Some(range) = &self.encoder.ranges.alpha_tiles {
            pass.set_vertex_buffer(0, common_resources.vertices.get_buffer_slice(range));
            pass.set_bind_group(1, &resources.mask_texture_bind_group, &[]);
            helper.reset_binding(1);

            for batch in &self.encoder.alpha_batches[render_pass.alpha_batches.clone()] {
                let pipeline = shaders.try_get(
                    resources.masked_pipeline,
                    batch.pattern,
                    SurfaceConfig::default(),
                ).unwrap();

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

impl CanvasRenderer for TileRenderer {
    fn add_render_passes(&mut self, render_passes: &mut RenderPasses) {
        for (idx, pass) in self.encoder.render_passes.iter().enumerate() {
            render_passes.push(SubPass {
                renderer_id: self.renderer_id,
                internal_index: idx as u32,
                require_pre_pass: pass.color_pre_pass || pass.mask_pre_pass,
                z_index: pass.z_index,
                use_depth: false,
                use_stencil: false,
                use_msaa: false,
            });
        }
    }

    fn render_pre_pass(
        &self,
        index: u32,
        shaders: &Shaders,
        resources: &GpuResources,
        bindings: &dyn BindingResolver,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let pass = &self.encoder.render_passes[index as usize];
        if pass.color_pre_pass {
            self.render_color_atlas(pass.color_atlas_index, shaders, &resources[self.common_resources], &resources[self.resources], bindings, encoder)
        }
        if pass.mask_pre_pass {
            self.render_mask_atlas(pass.mask_atlas_index, &resources[self.common_resources], &resources[self.resources], bindings, encoder)
        }
    }

    fn render<'pass, 'resources: 'pass>(
        &self,
        index: u32,
        _surface_info: &RenderPassState,
        shaders: &'resources Shaders,
        resources: &'resources GpuResources,
        bindings: &'resources dyn BindingResolver,
        render_pass: &mut wgpu::RenderPass<'pass>,
    ) {
        self.render_pass(index as usize, shaders, &resources[self.common_resources], &resources[self.resources], bindings, render_pass);
    }
}
