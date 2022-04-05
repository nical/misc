/// A lot of the boilerplate for the interaction between the tiling code a wgpu moved into one place.
///
/// It's not very good, it can only serves the purpose of the prototype, it's just so that the code isn't all in main().

use crate::gpu::masked_tiles::MaskUploadCopies;
use crate::gpu::masked_tiles::Masks;
use crate::gpu::masked_tiles::MaskedTiles;
use crate::gpu::masked_tiles::MaskUploader;
use crate::gpu::masked_tiles::Mask;
use crate::gpu::solid_tiles::SolidTiles;

use crate::gpu_raster_encoder::{GpuRasterEncoder, MaskPass};
use lyon::math::vector;

use wgpu::util::DeviceExt;

pub struct TileRenderer {
    mask_upload_copies: MaskUploadCopies,
    masks: Masks,
    masked_tiles: MaskedTiles,
    solid_tiles: SolidTiles,

    mask_passes: Vec<MaskPass>,
    num_masked_tiles: u32,
    num_solid_tiles: u32,
    tile_size: u32,
    tile_atlas_size: u32,

    mask_texture: wgpu::Texture,
    mask_texture_view: wgpu::TextureView,

    masked_tiles_bind_group: wgpu::BindGroup,
    masks_bind_group: wgpu::BindGroup,
    edges_ssbo: wgpu::Buffer,

    quad_ibo: wgpu::Buffer,
    solid_tiles_vbo: wgpu::Buffer,
    masked_tiles_vbo: wgpu::Buffer,
    masks_vbo: wgpu::Buffer,
    mask_params_ubo: wgpu::Buffer,
}


impl TileRenderer {
    pub fn new(device: &wgpu::Device, tile_size: u32, tile_atlas_size: u32, globals_bind_group_layout: &wgpu::BindGroupLayout) -> Self {
        let mask_upload_copies = MaskUploadCopies::new(&device, &globals_bind_group_layout);
        let masks = Masks::new(&device);
        let masked_tiles = MaskedTiles::new(&device, &globals_bind_group_layout);
        let solid_tiles = SolidTiles::new(&device, &globals_bind_group_layout);

        let quad_indices = [0u16, 1, 2, 0, 2, 3];
        let quad_ibo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Quad indices"),
            contents: bytemuck::cast_slice(&quad_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let mask_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Mask atlas"),
            dimension: wgpu::TextureDimension::D2,
            size: wgpu::Extent3d {
                width: tile_atlas_size,
                height: tile_atlas_size,
                depth_or_array_layers: 1,
            },
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            mip_level_count: 1,
            sample_count: 1,
        });

        let mask_texture_view = mask_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let masked_tiles_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("masked tiles"),
            layout: &masked_tiles.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&mask_texture_view),
                },
            ],
        });

        let solid_tiles_vbo = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("solid tiles"),
            size: 4096 * 64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });


        let masked_tiles_vbo = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("masked tiles"),
            size: 4096 * 64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let masks_vbo = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("masks"),
            size: 4096 * 64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let edges_ssbo = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("edges"),
            size: 4096 * 256,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mask_params_ubo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Mask params"),
            contents: bytemuck::cast_slice(&[
                crate::gpu::masked_tiles::MaskParams {
                    tile_size: tile_size as f32,
                    inv_atlas_width: 1.0 / (tile_atlas_size as f32),
                    masks_per_row: tile_atlas_size / (tile_size as u32),
                }
            ]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let masks_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Masks"),
            layout: &masks.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(mask_params_ubo.as_entire_buffer_binding())
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(edges_ssbo.as_entire_buffer_binding()),
                },
            ],
        });

        TileRenderer {
            mask_upload_copies,
            masks,
            masked_tiles,
            solid_tiles,

            mask_passes: Vec::new(),
            num_masked_tiles: 0,
            num_solid_tiles: 0,
            tile_size,
            tile_atlas_size,

            quad_ibo,
            mask_texture,
            mask_texture_view,
            masked_tiles_bind_group,
            masks_bind_group,
            edges_ssbo,
            solid_tiles_vbo,
            masked_tiles_vbo,
            masks_vbo,
            mask_params_ubo,
        }
    }

    pub fn update(
        &mut self,
        builder: &mut GpuRasterEncoder,
        b0: &mut GpuRasterEncoder,
        b1: &mut GpuRasterEncoder,
        b2: &mut GpuRasterEncoder,
        parallel: bool,
    ) {

        if parallel {
            let mut gpu_masks = Vec::with_capacity(builder.gpu_masks.len() + b0.gpu_masks.len() + b1.gpu_masks.len() + b2.gpu_masks.len());
            // Merge worker data into the main builder. TODO: It's not necessary, we should upload directly
            // off of each worker's data.
            builder.solid_tiles.reserve(b0.solid_tiles.len() + b1.solid_tiles.len() + b2.solid_tiles.len());
            builder.solid_tiles.extend_from_slice(&b0.solid_tiles);
            builder.solid_tiles.extend_from_slice(&b1.solid_tiles);
            builder.solid_tiles.extend_from_slice(&b2.solid_tiles);

            let mut gpu_mask_range_start = 0;
            for atlas_index in 0..(builder.num_mask_atlases() as usize) {
                let mut edge_offset = 0;

                let mut mask_tiles_range = std::u32::MAX .. 0;

                for builder in &[&builder, &b0, &b1, &b2] {
                    for pass in &builder.mask_passes {
                        if pass.atlas_index != atlas_index as u32 {
                            continue;
                        }

                        mask_tiles_range.start = mask_tiles_range.start.min(pass.masked_tiles.start);
                        mask_tiles_range.end = mask_tiles_range.end.max(pass.masked_tiles.end);

                        for mask in &builder.gpu_masks[pass.gpu_masks.start as usize .. (pass.gpu_masks.end as usize)] {
                            gpu_masks.push(Mask {
                                edges: (mask.edges.0 + edge_offset, mask.edges.1 + edge_offset),
                                ..*mask
                            });
                        }

                        edge_offset += builder.line_edges.len() as u32;
                    }
                }
                let gpu_mask_range_end = gpu_masks.len() as u32;
                self.mask_passes.push(MaskPass {
                    gpu_masks: gpu_mask_range_start .. gpu_mask_range_end,
                    masked_tiles: mask_tiles_range,
                    atlas_index: atlas_index as u32,
                });
                gpu_mask_range_start = gpu_mask_range_end;
            }

            let mut offset = builder.line_edges.len() as u32;
            for mask in &b0.gpu_masks {
                builder.gpu_masks.push(Mask {
                    edges: (mask.edges.0 + offset, mask.edges.1 + offset),
                    ..*mask
                });
            }
            offset += b0.line_edges.len() as u32;
            for mask in &b1.gpu_masks {
                builder.gpu_masks.push(Mask {
                    edges: (mask.edges.0 + offset, mask.edges.1 + offset),
                    ..*mask
                });
            }
            offset += b1.line_edges.len() as u32;
            for mask in &b2.gpu_masks {
                builder.gpu_masks.push(Mask {
                    edges: (mask.edges.0 + offset, mask.edges.1 + offset),
                    ..*mask
                });
            }

            // TODO: the parallel code path is missing the equivalent code for quad_edges here.
            builder.line_edges.reserve(b0.line_edges.len() + b1.line_edges.len() + b2.line_edges.len());
            builder.line_edges.extend_from_slice(&b0.line_edges);
            builder.line_edges.extend_from_slice(&b1.line_edges);
            builder.line_edges.extend_from_slice(&b2.line_edges);

            for i in 0..16 {
                builder.edge_distributions[i] += b0.edge_distributions[i]
                    + b1.edge_distributions[i]
                    + b2.edge_distributions[i];
            }

            builder.gpu_masks = gpu_masks;
        } else {
            std::mem::swap(&mut self.mask_passes, &mut builder.mask_passes);
        }

        self.num_masked_tiles = builder.mask_tiles.len() as u32;
    }

    // TODO: this should be part of update
    pub fn begin_frame(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        builder: &GpuRasterEncoder,
        target_width: f32,
        target_height: f32,
    ) {
        self.num_solid_tiles = builder.solid_tiles.len() as u32;

        queue.write_buffer(
            &self.solid_tiles_vbo,
            0,
            bytemuck::cast_slice(&builder.solid_tiles),
        );

        queue.write_buffer(
            &self.masked_tiles_vbo,
            0,
            bytemuck::cast_slice(&builder.mask_tiles),
        );

        queue.write_buffer(
            &self.masks_vbo,
            0,
            bytemuck::cast_slice(&builder.gpu_masks),
        );

        let edges = if !builder.line_edges.is_empty() {
            bytemuck::cast_slice(&builder.line_edges)
        } else {
            bytemuck::cast_slice(&builder.quad_edges)
        };
        queue.write_buffer(&self.edges_ssbo, 0, edges);


    }

    pub fn render(
        &mut self,
        device: &wgpu::Device,
        target: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
        globals_bind_group: &wgpu::BindGroup,
        mask_uploaders: &mut[&mut MaskUploader],
    ) {
        let masked_tiles_in_first_pass = {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Mask atlas"),
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view: &self.mask_texture_view,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: true,
                    },
                    resolve_target: None,
                }],
                depth_stencil_attachment: None,
            });

            let masked_tiles_in_first_pass = {

                let mask_range = self.mask_passes.last().unwrap();

                pass.set_pipeline(&self.masks.line_evenodd_pipeline);
                pass.set_bind_group(0, &self.masks_bind_group, &[]);
                pass.set_index_buffer(self.quad_ibo.slice(..), wgpu::IndexFormat::Uint16);
                pass.set_vertex_buffer(0, self.masks_vbo.slice(..));
                pass.draw_indexed(0..6, 0, mask_range.gpu_masks.clone());

                (self.num_masked_tiles - mask_range.masked_tiles.end) .. (self.num_masked_tiles - mask_range.masked_tiles.start)
            };

/*
            if let Some(quad_masks_bind_group) = &quad_masks_bind_group {
                pass.set_pipeline(&masks.quad_evenodd_pipeline);
                pass.set_bind_group(0, &quad_masks_bind_group, &[]);
                pass.set_index_buffer(quad_ibo.slice(..), wgpu::IndexFormat::Uint16);
                pass.set_vertex_buffer(0, masks_vbo.slice(..));
                pass.draw_indexed(0..6, 0, 0..num_gpu_masks);
            }
*/

            for uploader in mask_uploaders.iter_mut() {
                uploader.upload(
                    &device,
                    &mut pass,
                    globals_bind_group,
                    &self.mask_upload_copies.pipeline,
                    &self.quad_ibo,
                    (self.mask_passes.len().max(1) - 1) as u32, // Last index or zero.
                );
            }

            masked_tiles_in_first_pass
        };
        {
            let bg_color = wgpu::Color { r: 0.8, g: 0.8, b: 0.8, a: 1.0 };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Color target"),
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view: target,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(bg_color),
                        store: true,
                    },
                    resolve_target: None,
                }],
                depth_stencil_attachment: None,
            });

            pass.set_index_buffer(self.quad_ibo.slice(..), wgpu::IndexFormat::Uint16);

            pass.set_bind_group(0, globals_bind_group, &[]);

            pass.set_pipeline(&self.solid_tiles.pipeline);
            pass.set_vertex_buffer(0, self.solid_tiles_vbo.slice(..));
            pass.draw_indexed(0..6, 0, 0..self.num_solid_tiles);

            pass.set_pipeline(&self.masked_tiles.pipeline);
            pass.set_bind_group(1, &self.masked_tiles_bind_group, &[]);
            pass.set_vertex_buffer(0, self.masked_tiles_vbo.slice(..));
            pass.draw_indexed(0..6, 0, masked_tiles_in_first_pass);
        }

        if self.mask_passes.len() > 1 {
            for pass_idx in (0..self.mask_passes.len()).rev().skip(1) {
                let mask_ranges = &self.mask_passes[pass_idx];
                {
                    let mut mask_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Mask atlas"),
                        color_attachments: &[wgpu::RenderPassColorAttachment {
                            view: &self.mask_texture_view,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: true,
                            },
                            resolve_target: None,
                        }],
                        depth_stencil_attachment: None,
                    });

                    mask_pass.set_pipeline(&self.masks.line_evenodd_pipeline);
                    mask_pass.set_bind_group(0, &self.masks_bind_group, &[]);
                    mask_pass.set_index_buffer(self.quad_ibo.slice(..), wgpu::IndexFormat::Uint16);
                    mask_pass.set_vertex_buffer(0, self.masks_vbo.slice(..));
                    mask_pass.draw_indexed(0..6, 0, mask_ranges.gpu_masks.clone());

                    for uploader in mask_uploaders.iter_mut() {
                        uploader.upload(
                            &device,
                            &mut mask_pass,
                            &globals_bind_group,
                            &self.mask_upload_copies.pipeline,
                            &self.quad_ibo,
                            pass_idx as u32,
                        );
                    }
                }

                {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Color target"),
                        color_attachments: &[wgpu::RenderPassColorAttachment {
                            view: target,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: true,
                            },
                            resolve_target: None,
                        }],
                        depth_stencil_attachment: None,
                    });

                    // The masked tile's buffer content was reversed for proper z-order earlier.
                    let mask_tiles_range = (self.num_masked_tiles - mask_ranges.masked_tiles.end) .. (self.num_masked_tiles - mask_ranges.masked_tiles.start);

                    pass.set_pipeline(&self.masked_tiles.pipeline);
                    pass.set_bind_group(0, globals_bind_group, &[]);
                    pass.set_bind_group(1, &self.masked_tiles_bind_group, &[]);
                    pass.set_index_buffer(self.quad_ibo.slice(..), wgpu::IndexFormat::Uint16);
                    pass.set_vertex_buffer(0, self.masked_tiles_vbo.slice(..));
                    pass.draw_indexed(0..6, 0, mask_tiles_range);
                }
            }
        }
    }
}