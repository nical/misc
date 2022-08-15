/// A lot of the boilerplate for the interaction between the tiling code a wgpu moved into one place.
///
/// It's not very good, it can only serves the purpose of the prototype, it's just so that the code isn't all in main().

use crate::gpu::masked_tiles::MaskUploadCopies;
use crate::gpu::masked_tiles::Masks;
use crate::gpu::masked_tiles::MaskedTiles;
use crate::gpu::masked_tiles::MaskUploader;
use crate::gpu::masked_tiles::Mask;
use crate::gpu::ShaderSources;

use crate::tile_encoder::{TileEncoder, MaskPass, AlphaBatch};

use wgpu::util::DeviceExt;

pub struct TileRenderer {
    mask_upload_copies: MaskUploadCopies,
    masks: Masks,
    masked_tiles: MaskedTiles,

    // TODO: move this state out into per-target data, or use TileEncoder directly.
    mask_passes: Vec<MaskPass>,
    batches: Vec<AlphaBatch>,
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

        let mut shaders = ShaderSources::new();

        let mask_upload_copies = MaskUploadCopies::new(&device, &globals_bind_group_layout);
        let masks = Masks::new(&device, &mut shaders);
        let masked_tiles = MaskedTiles::new(&device, &globals_bind_group_layout, &mut shaders);

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

            mask_passes: Vec::new(),
            batches: Vec::new(),
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
        builder: &mut TileEncoder,
    ) {
        std::mem::swap(&mut self.batches, &mut builder.batches);
        std::mem::swap(&mut self.mask_passes, &mut builder.mask_passes);
        self.num_masked_tiles = builder.masked_tiles.len() as u32;
    }

    // TODO: this should be part of update
    pub fn begin_frame(
        &mut self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        builder: &TileEncoder,
        _target_width: f32,
        _target_height: f32,
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
            bytemuck::cast_slice(&builder.masked_tiles),
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
        // TODO: the logic here is soooo hacky and fragile.

        let first_mask_pass = &self.mask_passes[0];

        let need_mask_pass = !first_mask_pass.gpu_masks.is_empty()
            || mask_uploaders.iter().any(|up| up.needs_upload());

        if need_mask_pass {
            // Clear the the mask passes with white so that tile 0 is a fully opaque mask.
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Mask atlas"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.mask_texture_view,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: true,
                    },
                    resolve_target: None,
                })],
                depth_stencil_attachment: None,
            });

            if !first_mask_pass.gpu_masks.is_empty() {
                pass.set_pipeline(&self.masks.line_pipeline);
                pass.set_bind_group(0, &self.masks_bind_group, &[]);
                pass.set_index_buffer(self.quad_ibo.slice(..), wgpu::IndexFormat::Uint16);
                pass.set_vertex_buffer(0, self.masks_vbo.slice(..));
                pass.draw_indexed(0..6, 0, first_mask_pass.gpu_masks.clone());
            }

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
        }

        {
            let bg_color = wgpu::Color { r: 0.8, g: 0.8, b: 0.8, a: 1.0 };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Color target"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(bg_color),
                        store: true,
                    },
                    resolve_target: None,
                })],
                depth_stencil_attachment: None,
            });

            pass.set_index_buffer(self.quad_ibo.slice(..), wgpu::IndexFormat::Uint16);
            pass.set_bind_group(0, globals_bind_group, &[]);
            if self.num_solid_tiles > 0 {    
                pass.set_pipeline(&self.masked_tiles.opaque_solid_pipeline);
                pass.set_vertex_buffer(0, self.solid_tiles_vbo.slice(..));
                pass.draw_indexed(0..6, 0, 0..self.num_solid_tiles);
            }

            if !first_mask_pass.batches.is_empty() {
                pass.set_vertex_buffer(0, self.masked_tiles_vbo.slice(..));
                for batch in &self.batches[first_mask_pass.batches.clone()] {
                    pass.set_pipeline(&self.masked_tiles.masked_solid_pipeline);
                    pass.set_bind_group(1, &self.masked_tiles_bind_group, &[]);
                    pass.draw_indexed(0..6, 0, batch.tiles.clone());
                }
            }
        }

        if self.mask_passes.len() > 1 {
            for pass_idx in 1..self.mask_passes.len() {
                let mask_ranges = &self.mask_passes[pass_idx];
                {
                    let mut mask_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Mask atlas"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &self.mask_texture_view,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                                store: true,
                            },
                            resolve_target: None,
                        })],
                        depth_stencil_attachment: None,
                    });

                    mask_pass.set_pipeline(&self.masks.line_pipeline);
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
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: target,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: true,
                            },
                            resolve_target: None,
                        })],
                        depth_stencil_attachment: None,
                    });

                    pass.set_index_buffer(self.quad_ibo.slice(..), wgpu::IndexFormat::Uint16);
                    pass.set_vertex_buffer(0, self.masked_tiles_vbo.slice(..));
                    for batch in &self.batches[mask_ranges.batches.clone()] {
                        pass.set_pipeline(&self.masked_tiles.masked_solid_pipeline);
                        pass.set_bind_group(0, globals_bind_group, &[]);
                        pass.set_bind_group(1, &self.masked_tiles_bind_group, &[]);
                        pass.draw_indexed(0..6, 0, batch.tiles.clone());
                    }
                }
            }
        }
    }
}