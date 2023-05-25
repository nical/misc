use lyon::{math::vector, geom::euclid::Size2D};
use crate::{Tiler, canvas::{Fill, ResourcesHandle, Shape, Pattern, RendererCommandIndex, Canvas, RecordedPattern, RecordedShape, GpuResources, RenderPasses, SubPass, CanvasRenderer, RendererId, CommonGpuResources}, TileMask, pattern::{solid_color::{SolidColorBuilder, SolidColor}, simple_gradient::{SimpleGradientBuilder, SimpleGradient}, checkerboard::{CheckerboardPatternBuilder, CheckerboardPattern, add_checkerboard}}, gpu::GpuStore, Color, TilerConfig, TILE_SIZE};
use super::{encoder::TileEncoder, TilingGpuResources, mask::MaskEncoder, TilerPattern, FillOptions, Stats};

pub struct TileRenderer {
    pub encoder: TileEncoder,
    pub tiler: Tiler,
    pub occlusion_mask: TileMask,
    commands: Vec<Fill>,
    renderer_id: RendererId,
    common_resources: ResourcesHandle<CommonGpuResources>,
    resources: ResourcesHandle<TilingGpuResources>,
    tolerance: f32,
    patterns: TilingPatterns,
    masks: TilingMasks,
}

struct TilingPatterns {
    solid_color: SolidColorBuilder,
    gradient: SimpleGradientBuilder,
    checkerboard: CheckerboardPatternBuilder,
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
    ) -> Self {
        let size = config.view_box.size().to_u32();
        let tiles_x = (size.width + crate::TILE_SIZE - 1) / crate::TILE_SIZE;
        let tiles_y = (size.height + crate::TILE_SIZE - 1) / crate::TILE_SIZE;
        TileRenderer {
            commands: Vec::new(),
            renderer_id,
            common_resources: common_resources_id,
            resources: resources_id,
            encoder: TileEncoder::new(config, 3),
            tiler: Tiler::new(config),
            tolerance: config.tolerance,
            occlusion_mask: TileMask::new(tiles_x, tiles_y),
            patterns: TilingPatterns {
                solid_color: SolidColorBuilder::new(SolidColor::new(Color::BLACK), 0),
                gradient: SimpleGradientBuilder::new(SimpleGradient::new(), 1),
                checkerboard: CheckerboardPatternBuilder::new(CheckerboardPattern::new(), 2)
            },
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

    pub fn fill<S: Shape, P: Pattern>(&mut self, canvas: &mut Canvas, shape: S, pattern: P) {
        let transform = canvas.transforms.current();
        let z_index = canvas.z_indices.push();
        let index = self.commands.len() as RendererCommandIndex;
        self.commands.push(Fill {
            shape: shape.to_command(),
            pattern: pattern.to_command(),
            transform,
            z_index,
        });
        canvas.commands.push(self.renderer_id, index);
    }

    pub fn prepare(&mut self, canvas: &Canvas, gpu_store: &mut GpuStore, device: &wgpu::Device) {
        let commands = std::mem::take(&mut self.commands);
        // Process paths back to front in order to let the occlusion culling logic do its magic.
        for range in canvas.commands.with_renderer_rev(self.renderer_id) {
            let range = range.start as usize .. range.end as usize;
            for fill in commands[range].iter().rev() {
                self.prepare_fill(fill, canvas, gpu_store, device);
            }

            self.encoder.split_sub_pass();
        }

        self.masks.circle_masks.finish();
        self.masks.rectangle_masks.finish();

        let reversed = true;
        self.encoder.finish(reversed);

        self.commands = commands;
    }

    pub fn prepare_fill(&mut self, fill: &Fill, canvas: &Canvas, gpu_store: &mut GpuStore, device: &wgpu::Device) {
        let transform = if fill.transform != 0 {
            Some(canvas.transforms.get(fill.transform))
        } else {
            None
        };

        let mut prerender = false;
        let pattern: &mut dyn TilerPattern = match fill.pattern {
            RecordedPattern::Color(color) => {
                self.patterns.solid_color.set(SolidColor::new(color));
                &mut self.patterns.solid_color
            },
            RecordedPattern::Checkerboard(checkerboard) => {
                prerender = true;
                let mut checkerboard = checkerboard;
                if let Some(transform) = transform {
                    checkerboard.offset = transform.transform_point(checkerboard.offset);
                    checkerboard.scale = transform.transform_vector(vector(0.0, checkerboard.scale)).y
                }
                self.patterns.checkerboard.set(add_checkerboard(gpu_store, &checkerboard));
                &mut self.patterns.checkerboard
            }
            RecordedPattern::Gradient(gradient) => {
                let mut gradient = gradient;
                if let Some(transform) = transform {
                    gradient.from = transform.transform_point(gradient.from);
                    gradient.to = transform.transform_point(gradient.to);
                }

                self.patterns.gradient.set(gradient.write_gpu_data(gpu_store));
                &mut self.patterns.gradient
            }
            _ => { unimplemented!() }
        };

        self.encoder.current_z_index = fill.z_index;
         match &fill.shape {
            RecordedShape::Path(shape) => {
                let options = FillOptions::new()
                    .with_transform(transform)
                    .with_fill_rule(shape.fill_rule)
                    .with_prerendered_pattern(prerender)
                    .with_tolerance(self.tolerance)
                    .with_inverted(shape.inverted);
                self.tiler.fill_path(shape.path.iter(), &options, pattern, &mut self.occlusion_mask, &mut self.encoder, device);
            }
            RecordedShape::Circle(circle) => {
                let options = FillOptions::new()
                    .with_transform(transform)
                    .with_prerendered_pattern(prerender)
                    .with_tolerance(self.tolerance)
                    .with_inverted(circle.inverted);
                    crate::tiling::mask::circle::fill_circle(
                        circle.center,
                        circle.radius,
                        &options,
                        pattern,
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
                crate::tiling::mask::rect::fill_rect(
                    rect,
                    &options,
                    pattern,
                    &mut self.occlusion_mask,
                    &mut self.tiler,
                    &mut self.encoder,
                    &mut self.masks.rectangle_masks,
                    device,
                )
            }
            RecordedShape::Canvas => {
                self.tiler.fill_canvas(pattern, &mut self.occlusion_mask, &mut self.encoder);
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
        common_resources: &CommonGpuResources,
        resources: &TilingGpuResources,
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
            let pattern = &self.encoder.patterns[batch.pattern];
            if let Some(buffer_range) = &pattern.prerendered_vbo_range {
                pass.set_vertex_buffer(0, common_resources.vertices.get_buffer_slice(buffer_range));
                pass.set_pipeline(&resources.patterns[batch.pattern].opaque);
                pass.draw_indexed(0..6, 0, batch.tiles.clone());
            }
        }
    }

    pub fn render_mask_atlas(
        &self,
        mask_atlas_index: u32,
        common_resources: &CommonGpuResources,
        resources: &TilingGpuResources,
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
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
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
        common_resources: &'resources CommonGpuResources,
        resources: &'resources TilingGpuResources,
        pass: &mut wgpu::RenderPass<'pass>,
    ) {
        let render_pass = &self.encoder.render_passes[pass_idx];

        pass.set_index_buffer(common_resources.quad_ibo.slice(..), wgpu::IndexFormat::Uint16);
        pass.set_bind_group(0, &common_resources.main_target_and_gpu_store_bind_group, &[]);

        if !render_pass.opaque_batches.is_empty() {
            for batch in &self.encoder.opaque_batches[render_pass.opaque_batches.clone()] {
                if let Some(range) = &self.encoder.get_opaque_batch_vertices(batch.pattern) {
                    pass.set_vertex_buffer(0, common_resources.vertices.get_buffer_slice(range));
                    // TODO: optional hook to bind extra data
                    pass.set_pipeline(&resources.patterns[batch.pattern].opaque);
                    pass.draw_indexed(0..6, 0, batch.tiles.clone());
                }
            }
        }

        if !render_pass.opaque_image_tiles.is_empty() {
            let range = &self.encoder.ranges.opaque_image_tiles.as_ref().unwrap();
            pass.set_vertex_buffer(0, common_resources.vertices.get_buffer_slice(range));

            pass.set_pipeline(&resources.opaque_image_pipeline);
            pass.set_bind_group(1, &resources.mask_texture_bind_group, &[]);
            pass.set_bind_group(2, &resources.src_color_bind_group, &[]);
            pass.draw_indexed(0..6, 0, render_pass.opaque_image_tiles.clone());
        }

        if let Some(range) = &self.encoder.ranges.alpha_tiles {
            pass.set_vertex_buffer(0, common_resources.vertices.get_buffer_slice(range));
            pass.set_bind_group(1, &resources.mask_texture_bind_group, &[]);
    
            for batch in &self.encoder.alpha_batches[render_pass.alpha_batches.clone()] {
                match batch.pattern {
                    crate::tiling::TILED_IMAGE_PATTERN => {
                        pass.set_pipeline(&resources.masked_image_pipeline);
                        pass.set_bind_group(2, &resources.src_color_bind_group, &[]);
                    }
                    _ => {
                        pass.set_pipeline(&resources.patterns[batch.pattern].masked);
                    }
                }
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
        resources: &GpuResources,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let pass = &self.encoder.render_passes[index as usize];
        if pass.color_pre_pass {
            self.render_color_atlas(pass.color_atlas_index, &resources[self.common_resources], &resources[self.resources], encoder)
        }
        if pass.mask_pre_pass {
            self.render_mask_atlas(pass.mask_atlas_index, &resources[self.common_resources], &resources[self.resources], encoder)
        }
    }

    fn render<'pass, 'resources: 'pass>(
        &self,
        index: u32,
        resources: &'resources GpuResources,
        render_pass: &mut wgpu::RenderPass<'pass>,
    ) {
        self.render_pass(index as usize, &resources[self.common_resources], &resources[self.resources], render_pass);
    }
}
