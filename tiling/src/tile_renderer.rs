/// A lot of the boilerplate for the interaction between the tiling code a wgpu moved into one place.
///
/// It's not very good, it can only serves the purpose of the prototype, it's just so that the code isn't all in main().

use crate::gpu::{ShaderSources, GpuTileAtlasDescriptor, VertexBuilder};
use crate::gpu::mask_uploader::{MaskUploadCopies, MaskUploader};
use crate::tile_encoder::{TileEncoder, MaskPass, AlphaBatch};
use crate::checkerboard_pattern::{CheckerboardRenderer, CheckerboardPatternBuilder};

use lyon::geom::Box2D;

use wgpu::util::DeviceExt;

/*

When rendering the tiger at 1800x1800 px, according to renderdoc on Intel UHD Graphics 620 (KBL GT2):
 - rasterizing the masks takes ~2ms
 - rendering into the color target takes ~0.8ms
  - ~0.28ms opaque tiles
  - ~0.48ms alpha tiles

*/

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct TileInstance {
    pub tile_id: u32,
    pub mask: u32,
    pub pattern_data: u32,
    pub width: u32,
}

unsafe impl bytemuck::Pod for TileInstance {}
unsafe impl bytemuck::Zeroable for TileInstance {}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Mask {
    pub edges: (u32, u32),
    pub mask_id: u32,
    pub fill_rule: u16,
    pub backdrop: i16,
}

unsafe impl bytemuck::Pod for Mask {}
unsafe impl bytemuck::Zeroable for Mask {}

pub struct TileRenderer {
    mask_upload_copies: MaskUploadCopies,
    masks: Masks,
    checkerboard_pattern: CheckerboardRenderer,

    pub masked_solid_pipeline: wgpu::RenderPipeline,
    pub masked_image_pipeline: wgpu::RenderPipeline,
    pub opaque_solid_pipeline: wgpu::RenderPipeline,
    pub opaque_image_pipeline: wgpu::RenderPipeline,
    pub mask_texture_bind_group_layout: wgpu::BindGroupLayout,

    // TODO: move this state out into per-target data, or use TileEncoder directly.
    mask_passes: Vec<MaskPass>,
    batches: Vec<AlphaBatch>,
    num_opaque_solid_tiles: u32,
    num_opaque_image_tiles: u32,
    num_checkerboard_tiles: u32,

    //mask_texture: wgpu::Texture,
    mask_texture_view: wgpu::TextureView,
    //src_color_texture: wgpu::Texture,
    src_color_texture_view: wgpu::TextureView,

    mask_texture_bind_group: wgpu::BindGroup,
    src_color_bind_group: wgpu::BindGroup,
    masks_bind_group: wgpu::BindGroup,
    mask_atlas_desc_bind_group: wgpu::BindGroup,
    edges_ssbo: wgpu::Buffer,

    quad_ibo: wgpu::Buffer,
    opaque_tiles_vbo: wgpu::Buffer,
    alpha_tiles_vbo: wgpu::Buffer,
    masks_vbo: wgpu::Buffer,
    mask_params_ubo: wgpu::Buffer,
}


impl TileRenderer {
    pub fn new(device: &wgpu::Device, tile_size: u32, tile_atlas_size: u32, globals_bind_group_layout: &wgpu::BindGroupLayout) -> Self {

        let mut shaders = ShaderSources::new();

        let atlas_desc_buffer_size = std::mem::size_of::<GpuTileAtlasDescriptor>() as u64;
        let mask_atlas_desc_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Tile mask atlas"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(atlas_desc_buffer_size),
                    },
                    count: None,
                },
            ],
        });

        let mask_params_ubo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Mask params"),
            contents: bytemuck::cast_slice(&[
                GpuTileAtlasDescriptor::new(tile_atlas_size, tile_atlas_size, tile_size)
            ]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let mask_upload_copies = MaskUploadCopies::new(&device, &globals_bind_group_layout);
        let masks = Masks::new(&device, &mut shaders);
        let checkerboard_pattern = CheckerboardRenderer::new(device, &mut shaders, &mask_atlas_desc_bind_group_layout);

        let src = include_str!("./../shaders/tile.wgsl");
        let masked_solid_module = shaders.create_shader_module(device, "masked_tile_solid", src, &["TILED_MASK", "SOLID_PATTERN"]);
        let masked_img_module = shaders.create_shader_module(device, "masked_tile_image", src, &["TILED_MASK", "TILED_IMAGE_PATTERN",]);
        let opaque_solid_module = shaders.create_shader_module(device, "masked_tile_solid", src, &["SOLID_PATTERN"]);
        let opaque_img_module = shaders.create_shader_module(device, "masked_tile_image", src, &["TILED_IMAGE_PATTERN",]);

        let mask_texture_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Tile mask atlas"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let src_color_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Tile src color atlas"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let mut attributes = VertexBuilder::new();
        attributes.push(wgpu::VertexFormat::Uint32);
        attributes.push(wgpu::VertexFormat::Uint32);
        attributes.push(wgpu::VertexFormat::Uint32);
        attributes.push(wgpu::VertexFormat::Uint32);
        attributes.push(wgpu::VertexFormat::Float32);

        let attributes_with_opacity = attributes.get();

        let attributes_no_opacity = &attributes_with_opacity[..4];
        let vertex_buffer_layouts = &[wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<TileInstance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: attributes_no_opacity,
        }];
        let alpha_color_target_states = &[
            Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Bgra8UnormSrgb,
                blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            }),
        ];
        let opaque_color_target_states = &[
            Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Bgra8UnormSrgb,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }),
        ];

        let primitive = wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            polygon_mode: wgpu::PolygonMode::Fill,
            front_face: wgpu::FrontFace::Ccw,
            strip_index_format: None,
            cull_mode: None,
            unclipped_depth: false,
            conservative: false,
        };

        let multisample = wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        };

        let opaque_solid_tile_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Opaque solid tiles"),
            bind_group_layouts: &[&globals_bind_group_layout],
            push_constant_ranges: &[],
        });
        let masked_solid_tile_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Masked solid tiles"),
            bind_group_layouts: &[&globals_bind_group_layout, &mask_texture_bind_group_layout],
            push_constant_ranges: &[],
        });
        let masked_img_tile_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Masked image tiles"),
            bind_group_layouts: &[
                &globals_bind_group_layout,
                &mask_texture_bind_group_layout,
                &src_color_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        let masked_solid_tile_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
            label: Some("Masked solid tiles"),
            layout: Some(&masked_solid_tile_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &masked_solid_module,
                entry_point: "vs_main",
                buffers: vertex_buffer_layouts,
            },
            fragment: Some(wgpu::FragmentState {
                module: &masked_solid_module,
                entry_point: "fs_main",
                targets: alpha_color_target_states
            }),
            primitive,
            depth_stencil: None,
            multiview: None,
            multisample: wgpu::MultisampleState::default(),
        };

        let masked_image_tile_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
            label: Some("Masked image tiles"),
            layout: Some(&masked_img_tile_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &masked_img_module,
                entry_point: "vs_main",
                buffers: vertex_buffer_layouts,
            },
            fragment: Some(wgpu::FragmentState {
                module: &masked_img_module,
                entry_point: "fs_main",
                targets: alpha_color_target_states
            }),
            primitive,
            depth_stencil: None,
            multiview: None,
            multisample,
        };

        let opaque_solid_tile_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
            label: Some("Opaque solid tiles"),
            layout: Some(&opaque_solid_tile_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &opaque_solid_module,
                entry_point: "vs_main",
                buffers: vertex_buffer_layouts,
            },
            fragment: Some(wgpu::FragmentState {
                module: &opaque_solid_module,
                entry_point: "fs_main",
                targets: opaque_color_target_states
            }),
            primitive,
            depth_stencil: None,
            multiview: None,
            multisample,
        };

        let opaque_image_tile_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
            label: Some("Opaque image tiles"),
            layout: Some(&masked_img_tile_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &opaque_img_module,
                entry_point: "vs_main",
                buffers: vertex_buffer_layouts,
            },
            fragment: Some(wgpu::FragmentState {
                module: &opaque_img_module,
                entry_point: "fs_main",
                targets: opaque_color_target_states
            }),
            primitive,
            depth_stencil: None,
            multiview: None,
            multisample,
        };

        let masked_solid_pipeline = device.create_render_pipeline(&masked_solid_tile_pipeline_descriptor);
        let masked_image_pipeline = device.create_render_pipeline(&masked_image_tile_pipeline_descriptor);
        let opaque_solid_pipeline = device.create_render_pipeline(&opaque_solid_tile_pipeline_descriptor);
        let opaque_image_pipeline = device.create_render_pipeline(&opaque_image_tile_pipeline_descriptor);

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

        let src_color_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Color tile atlas"),
            dimension: wgpu::TextureDimension::D2,
            size: wgpu::Extent3d {
                width: tile_atlas_size,
                height: tile_atlas_size,
                depth_or_array_layers: 1,
            },
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            mip_level_count: 1,
            sample_count: 1,
        });

        let src_color_texture_view = src_color_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mask_texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("alpha tiles"),
            layout: &mask_texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&mask_texture_view),
                },
            ],
        });

        let src_color_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Color tiles"),
            layout: &&src_color_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&src_color_texture_view),
                },
            ],
        });

        let opaque_tiles_vbo = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("solid tiles"),
            size: 4096 * 64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let alpha_tiles_vbo = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("alpha tiles"),
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

        let mask_atlas_desc_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Mask atlas descriptor"),
            layout: &mask_atlas_desc_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(mask_params_ubo.as_entire_buffer_binding())
                },
            ],
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
            checkerboard_pattern,

            opaque_solid_pipeline,
            opaque_image_pipeline,
            masked_solid_pipeline,
            masked_image_pipeline,
            mask_texture_bind_group_layout,

            mask_passes: Vec::new(),
            batches: Vec::new(),
            num_opaque_solid_tiles: 0,
            num_opaque_image_tiles: 0,
            num_checkerboard_tiles: 0,

            quad_ibo,
            //mask_texture,
            mask_texture_view,
            src_color_texture_view,
            mask_texture_bind_group,
            src_color_bind_group,
            masks_bind_group,
            mask_atlas_desc_bind_group,
            edges_ssbo,
            opaque_tiles_vbo,
            alpha_tiles_vbo,
            masks_vbo,
            mask_params_ubo,
        }
    }

    pub fn update(
        &mut self,
        encoder: &mut TileEncoder,
    ) {
        std::mem::swap(&mut self.batches, &mut encoder.batches);
        std::mem::swap(&mut self.mask_passes, &mut encoder.mask_passes);
    }

    // TODO: this should be part of update
    pub fn begin_frame(
        &mut self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        tile_encoder: &TileEncoder,
        checkerboard: &CheckerboardPatternBuilder,
        _target_width: f32,
        _target_height: f32,
    ) {

        self.num_opaque_solid_tiles = tile_encoder.opaque_solid_tiles.len() as u32;
        self.num_opaque_image_tiles = tile_encoder.opaque_image_tiles.len() as u32;
        self.num_checkerboard_tiles = checkerboard.tiles().len() as u32;

        queue.write_buffer(
            &self.opaque_tiles_vbo,
            0,
            bytemuck::cast_slice(&tile_encoder.opaque_solid_tiles),
        );
        queue.write_buffer(
            &self.opaque_tiles_vbo,
            tile_encoder.opaque_solid_tiles.len() as u64 * std::mem::size_of::<TileInstance>() as u64,
            bytemuck::cast_slice(&tile_encoder.opaque_image_tiles),
        );

        queue.write_buffer(
            &self.alpha_tiles_vbo,
            0,
            bytemuck::cast_slice(&tile_encoder.alpha_tiles),
        );

        queue.write_buffer(
            &self.masks_vbo,
            0,
            bytemuck::cast_slice(&tile_encoder.gpu_masks),
        );

        if !checkerboard.tiles().is_empty() {
            queue.write_buffer(
                &self.checkerboard_pattern.vbo,
                0,
                bytemuck::cast_slice(checkerboard.tiles()),
            );
        }

        let edges = if !tile_encoder.line_edges.is_empty() {
            bytemuck::cast_slice(&tile_encoder.line_edges)
        } else {
            bytemuck::cast_slice(&tile_encoder.quad_edges)
        };
        queue.write_buffer(&self.edges_ssbo, 0, edges);
    }

    pub fn render(
        &mut self,
        device: &wgpu::Device,
        target: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
        globals_bind_group: &wgpu::BindGroup,
        mask_uploader: &mut MaskUploader,
    ) {
        // TODO: the logic here is soooo hacky and fragile.

        let need_color_pattern_pass = self.num_checkerboard_tiles > 0;

        if need_color_pattern_pass {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Color tile atlas"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.src_color_texture_view,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: true,
                    },
                    resolve_target: None,
                })],
                depth_stencil_attachment: None,
            });

            if self.num_checkerboard_tiles > 0 {
                pass.set_pipeline(&self.checkerboard_pattern.pipeline);
                pass.set_bind_group(0, &self.mask_atlas_desc_bind_group, &[]);
                pass.set_index_buffer(self.quad_ibo.slice(..), wgpu::IndexFormat::Uint16);
                pass.set_vertex_buffer(0, self.checkerboard_pattern.vbo.slice(..));
                pass.draw_indexed(0..6, 0, 0..self.num_checkerboard_tiles);
            }
        }

        let first_mask_pass = &self.mask_passes[0];

        let need_mask_pass = !first_mask_pass.gpu_masks.is_empty()
            || mask_uploader.needs_upload();

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

            mask_uploader.upload(
                &device,
                &mut pass,
                globals_bind_group,
                &self.mask_upload_copies.pipeline,
                &self.quad_ibo,
                (self.mask_passes.len().max(1) - 1) as u32, // Last index or zero.
            );
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
            if self.num_opaque_solid_tiles > 0 {
                pass.set_pipeline(&self.opaque_solid_pipeline);
                pass.set_vertex_buffer(0, self.opaque_tiles_vbo.slice(..));
                pass.draw_indexed(0..6, 0, 0..self.num_opaque_solid_tiles);
            }
            if self.num_opaque_image_tiles > 0 {
                let first = self.num_opaque_solid_tiles;
                pass.set_pipeline(&self.opaque_image_pipeline);
                pass.set_vertex_buffer(0, self.opaque_tiles_vbo.slice(..));
                pass.set_bind_group(1, &self.mask_texture_bind_group, &[]);
                pass.set_bind_group(2, &self.src_color_bind_group, &[]);
                pass.draw_indexed(0..6, 0, first..(first + self.num_opaque_image_tiles));
            }

            if !first_mask_pass.batches.is_empty() {
                pass.set_vertex_buffer(0, self.alpha_tiles_vbo.slice(..));
                for batch in &self.batches[first_mask_pass.batches.clone()] {
                    match batch.batch_kind {
                        0 => {
                            pass.set_pipeline(&self.masked_solid_pipeline);
                            pass.set_bind_group(1, &self.mask_texture_bind_group, &[]);
                        }
                        1 => {
                            pass.set_pipeline(&self.masked_image_pipeline);
                            pass.set_bind_group(1, &self.mask_texture_bind_group, &[]);
                            pass.set_bind_group(2, &self.src_color_bind_group, &[]);
                        }
                        _ => {
                            unimplemented!()
                        }
                    }
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

                    mask_uploader.upload(
                        &device,
                        &mut mask_pass,
                        &globals_bind_group,
                        &self.mask_upload_copies.pipeline,
                        &self.quad_ibo,
                        pass_idx as u32,
                    );
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
                    pass.set_vertex_buffer(0, self.alpha_tiles_vbo.slice(..));
                    for batch in &self.batches[mask_ranges.batches.clone()] {
                        pass.set_pipeline(&self.masked_solid_pipeline);
                        pass.set_bind_group(0, globals_bind_group, &[]);
                        pass.set_bind_group(1, &self.mask_texture_bind_group, &[]);
                        pass.draw_indexed(0..6, 0, batch.tiles.clone());
                    }
                }
            }
        }
    }
}

pub struct Masks {
    pub line_pipeline: wgpu::RenderPipeline,
    pub quad_pipeline: wgpu::RenderPipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl Masks {
    pub fn new(device: &wgpu::Device, shaders: &mut ShaderSources) -> Self {
        create_mask_pipeline(device, shaders)
    }
}

fn create_mask_pipeline(device: &wgpu::Device, shaders: &mut ShaderSources) -> Masks {
    let quad_src = include_str!("./../shaders/mask_fill_quads.wgsl");
    let lin_src = include_str!("./../shaders/mask_fill.wgsl");
    let lin_module = &shaders.create_shader_module(device, "Mask fill linear", lin_src, &[]);
    let quad_module = &shaders.create_shader_module(device, "Mask fill quad", quad_src, &[]);

    let mask_globals_buffer_size = std::mem::size_of::<GpuTileAtlasDescriptor>() as u64;

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Mask"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(mask_globals_buffer_size),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(32),
                },
                count: None,
            },
        ],
    });

    let tile_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Masked tiles"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let line_tile_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
        label: Some("Tile mask linear"),
        layout: Some(&tile_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &lin_module,
            entry_point: "vs_main",
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<Mask>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[
                    wgpu::VertexAttribute {
                        offset: 0,
                        format: wgpu::VertexFormat::Uint32x2,
                        shader_location: 0,
                    },
                    wgpu::VertexAttribute {
                        offset: 8,
                        format: wgpu::VertexFormat::Uint32,
                        shader_location: 1,
                    },
                    wgpu::VertexAttribute {
                        offset: 12,
                        format: wgpu::VertexFormat::Uint32,
                        shader_location: 2,
                    },
                ],
            }],
        },
        fragment: Some(wgpu::FragmentState {
            module: &lin_module,
            entry_point: "fs_main",
            targets: &[
                Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
            ],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            polygon_mode: wgpu::PolygonMode::Fill,
            front_face: wgpu::FrontFace::Ccw,
            strip_index_format: None,
            cull_mode: None,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: None,
        multiview: None,
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
    };

    let quad_tile_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
        label: Some("Tile mask quad"),
        layout: Some(&tile_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &quad_module,
            entry_point: "vs_main",
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<Mask>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[
                    wgpu::VertexAttribute {
                        offset: 0,
                        format: wgpu::VertexFormat::Uint32x2,
                        shader_location: 0,
                    },
                    wgpu::VertexAttribute {
                        offset: 8,
                        format: wgpu::VertexFormat::Uint32,
                        shader_location: 1,
                    },
                    wgpu::VertexAttribute {
                        offset: 12,
                        format: wgpu::VertexFormat::Uint32,
                        shader_location: 2,
                    },
                ],
            }],
        },
        fragment: Some(wgpu::FragmentState {
            module: &quad_module,
            entry_point: "fs_main",
            targets: &[
                Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
            ],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            polygon_mode: wgpu::PolygonMode::Fill,
            front_face: wgpu::FrontFace::Ccw,
            strip_index_format: None,
            cull_mode: None,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: None,
        multiview: None,
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
    };

    let line_pipeline = device.create_render_pipeline(&line_tile_pipeline_descriptor);
    let quad_pipeline = device.create_render_pipeline(&quad_tile_pipeline_descriptor);

    Masks {
        line_pipeline,
        quad_pipeline,
        bind_group_layout,
    }
}
