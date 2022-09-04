use std::num::NonZeroU32;

use crate::{TilePosition, PatternData};
/// A lot of the boilerplate for the interaction between the tiling code a wgpu moved into one place.
///
/// It's not very good, it can only serves the purpose of the prototype, it's just so that the code isn't all in main().

use crate::gpu::{ShaderSources, GpuTargetDescriptor, VertexBuilder, PipelineDefaults};
use crate::gpu::mask_uploader::{MaskUploadCopies};
use crate::gpu_store::GpuStore;
use crate::tile_encoder::{TileEncoder, BufferRange, LineEdge};

use wgpu::TextureAspect;
use wgpu::util::DeviceExt;

/*

When rendering the tiger at 1800x1800 px, according to renderdoc on Intel UHD Graphics 620 (KBL GT2):
 - rasterizing the masks takes ~2ms
 - rendering into the color target takes ~0.8ms
  - ~0.28ms opaque tiles
  - ~0.48ms alpha tiles

*/

pub trait PatternRenderer {
    fn begin_frame(&mut self);
    fn render<'a, 'b: 'a>(&'b self, pass_idx: u32, pass: &mut wgpu::RenderPass<'a>);
    fn has_content(&self, pass_idx: u32) -> bool;
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct TileInstance {
    pub position: TilePosition,
    pub mask: TilePosition,
    pub pattern_data: PatternData,
}

unsafe impl bytemuck::Pod for TileInstance {}
unsafe impl bytemuck::Zeroable for TileInstance {}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Mask {
    pub edges: (u32, u32),
    pub tile: TilePosition,
    pub fill_rule: u16,
    pub backdrop: i16,
}

unsafe impl bytemuck::Pod for Mask {}
unsafe impl bytemuck::Zeroable for Mask {}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct CircleMask {
    pub tile: TilePosition,
    pub radius: f32,
    pub center: [f32; 2],
}

unsafe impl bytemuck::Pod for CircleMask {}
unsafe impl bytemuck::Zeroable for CircleMask {}

pub struct TileRenderer {
    mask_upload_copies: MaskUploadCopies,
    masks: Masks,

    pub masked_solid_pipeline: wgpu::RenderPipeline,
    pub masked_image_pipeline: wgpu::RenderPipeline,
    pub opaque_solid_pipeline: wgpu::RenderPipeline,
    pub opaque_image_pipeline: wgpu::RenderPipeline,
    pub mask_texture_bind_group_layout: wgpu::BindGroupLayout,

    pub mask_atlas_desc_bind_group_layout: wgpu::BindGroupLayout,

    //mask_texture: wgpu::Texture,
    mask_texture_view: wgpu::TextureView,
    //src_color_texture: wgpu::Texture,
    src_color_texture_view: wgpu::TextureView,

    mask_texture_bind_group: wgpu::BindGroup,
    src_color_bind_group: wgpu::BindGroup,
    masks_bind_group: wgpu::BindGroup,
    mask_atlas_desc_bind_group: wgpu::BindGroup,

    quad_ibo: wgpu::Buffer,
    pub tiles_vbo: BumpAllocatedBuffer,
    pub masks_vbo: BumpAllocatedBuffer,
    pub circles_vbo: BumpAllocatedBuffer,
    // TODO: some way to account for a per-TileEncoder offset or range.
    pub edges_ssbo: BumpAllocatedBuffer,
    mask_params_ubo: wgpu::Buffer,
}


impl TileRenderer {
    pub fn new(device: &wgpu::Device, shaders: &mut ShaderSources, tile_size: u32, tile_atlas_size: u32, globals_bind_group_layout: &wgpu::BindGroupLayout, gpu_store: &GpuStore) -> Self {

        let atlas_desc_buffer_size = std::mem::size_of::<GpuTargetDescriptor>() as u64;
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
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                }
            ],
        });

        let mask_params_ubo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Mask params"),
            contents: bytemuck::cast_slice(&[
                GpuTargetDescriptor::new(tile_atlas_size, tile_atlas_size)
            ]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let mask_upload_copies = MaskUploadCopies::new(&device, &globals_bind_group_layout);
        let masks = Masks::new(&device, shaders);

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

        let defaults = PipelineDefaults::new();
        let attributes = VertexBuilder::from_slice(&[wgpu::VertexFormat::Uint32x4]);

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

        let primitive = defaults.primitive_state();
        let multisample = wgpu::MultisampleState::default();

        let masked_solid_tile_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
            label: Some("Masked solid tiles"),
            layout: Some(&masked_solid_tile_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &masked_solid_module,
                entry_point: "vs_main",
                buffers: &[attributes.buffer_layout()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &masked_solid_module,
                entry_point: "fs_main",
                targets: defaults.color_target_state(),
            }),
            primitive,
            depth_stencil: None,
            multiview: None,
            multisample,
        };

        let masked_image_tile_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
            label: Some("Masked image tiles"),
            layout: Some(&masked_img_tile_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &masked_img_module,
                entry_point: "vs_main",
                buffers: &[attributes.buffer_layout()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &masked_img_module,
                entry_point: "fs_main",
                targets: defaults.color_target_state()
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
                buffers: &[attributes.buffer_layout()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &opaque_solid_module,
                entry_point: "fs_main",
                targets: defaults.color_target_state_no_blend(),
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
                buffers: &[attributes.buffer_layout()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &opaque_img_module,
                entry_point: "fs_main",
                targets: defaults.color_target_state_no_blend(),
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

        let tiles_vbo = BumpAllocatedBuffer::new::<TileInstance>(
            device,
            "tiles",
            4096 * 16,
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        );

        let masks_vbo = BumpAllocatedBuffer::new::<Mask>(
            device,
            "masks",
            4096 * 16,
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        );

        let circles_vbo = BumpAllocatedBuffer::new::<Mask>(
            device,
            "circles",
            4096 * 16,
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        );

        let edges_ssbo = BumpAllocatedBuffer::new::<LineEdge>(
            device,
            "edges",
            4096 * 256,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );

        let mask_atlas_desc_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Mask atlas descriptor"),
            layout: &mask_atlas_desc_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(mask_params_ubo.as_entire_buffer_binding())
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&gpu_store.texture().create_view(&wgpu::TextureViewDescriptor {
                        label: Some("gpu store"),
                        format: Some(wgpu::TextureFormat::Rgba32Float),
                        dimension: Some(wgpu::TextureViewDimension::D2),
                        aspect: TextureAspect::All,
                        base_mip_level: 0,
                        base_array_layer: 0,
                        mip_level_count: NonZeroU32::new(1),
                        array_layer_count: NonZeroU32::new(1),
                    }))
                }
            ],
        });

        let masks_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Masks"),
            layout: &masks.mask_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(mask_params_ubo.as_entire_buffer_binding())
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(edges_ssbo.buffer.as_entire_buffer_binding()),
                },
            ],
        });

        TileRenderer {
            mask_upload_copies,
            masks,

            opaque_solid_pipeline,
            opaque_image_pipeline,
            masked_solid_pipeline,
            masked_image_pipeline,
            mask_texture_bind_group_layout,

            mask_atlas_desc_bind_group_layout,
            quad_ibo,
            //mask_texture,
            mask_texture_view,
            src_color_texture_view,
            mask_texture_bind_group,
            src_color_bind_group,
            masks_bind_group,
            mask_atlas_desc_bind_group,
            edges_ssbo,
            tiles_vbo,
            masks_vbo,
            circles_vbo,
            mask_params_ubo,
        }
    }

    pub fn begin_frame(&mut self) {
        self.tiles_vbo.begin_frame();
        self.masks_vbo.begin_frame();
        self.circles_vbo.begin_frame();
        self.edges_ssbo.begin_frame();
    }

    pub fn allocate(&mut self, device: &wgpu::Device) {
        self.tiles_vbo.ensure_allocated(device);
        self.masks_vbo.ensure_allocated(device);
        self.circles_vbo.ensure_allocated(device);
        if self.edges_ssbo.ensure_allocated(device) {
            // reallocated the buffer, need to re-create the bind group.
            self.masks_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Masks"),
                layout: &self.masks.mask_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(self.mask_params_ubo.as_entire_buffer_binding())
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(self.edges_ssbo.buffer.as_entire_buffer_binding()),
                    },
                ],
            });
        }
    }

    pub fn render(
        &mut self,
        tile_encoder: &mut TileEncoder,
        patterns: &[&dyn PatternRenderer],
        device: &wgpu::Device,
        target: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
        globals_bind_group: &wgpu::BindGroup,
    ) {
        // TODO: the logic here is soooo hacky and fragile.

        let first_mask_pass = &tile_encoder.mask_passes[0];

        let mut src_color_atlas = first_mask_pass.color_atlas_index;
        let need_color_pattern_pass =  patterns.iter().any(|p| p.has_content(src_color_atlas));

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

            pass.set_bind_group(0, &self.mask_atlas_desc_bind_group, &[]);
            pass.set_index_buffer(self.quad_ibo.slice(..), wgpu::IndexFormat::Uint16);

            for pattern in patterns {
                pattern.render(src_color_atlas, &mut pass);
            }
        }

        let need_mask_pass = !first_mask_pass.gpu_masks.is_empty()
            || !first_mask_pass.circle_masks.is_empty()
            || tile_encoder.mask_uploader.needs_upload();

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
                debug_assert_eq!(tile_encoder.ranges.edges.byte_offset::<u8>(), 0, "Not implemented yet");
                pass.set_pipeline(&self.masks.line_pipeline);
                pass.set_bind_group(0, &self.masks_bind_group, &[]);
                pass.set_index_buffer(self.quad_ibo.slice(..), wgpu::IndexFormat::Uint16);
                pass.set_vertex_buffer(0, self.masks_vbo.buffer.slice(..));
                pass.draw_indexed(0..6, 0, first_mask_pass.gpu_masks.clone());
            }

            if !first_mask_pass.circle_masks.is_empty() {
                pass.set_pipeline(&self.masks.circle_pipeline);
                pass.set_bind_group(0, &self.masks_bind_group, &[]);
                pass.set_index_buffer(self.quad_ibo.slice(..), wgpu::IndexFormat::Uint16);
                pass.set_vertex_buffer(0, self.circles_vbo.buffer.slice(..));
                pass.draw_indexed(0..6, 0, first_mask_pass.circle_masks.clone());
            }

            tile_encoder.mask_uploader.upload(
                &device,
                &mut pass,
                globals_bind_group,
                &self.mask_upload_copies.pipeline,
                &self.quad_ibo,
                (tile_encoder.mask_passes.len().max(1) - 1) as u32, // Last index or zero.
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

            if !tile_encoder.ranges.opaque_solid_tiles.is_empty() {
                pass.set_pipeline(&self.opaque_solid_pipeline);
                pass.set_vertex_buffer(0, self.tiles_vbo.buffer.slice(..));
                pass.draw_indexed(0..6, 0, tile_encoder.ranges.opaque_solid_tiles.to_u32());
            }
            if !tile_encoder.ranges.opaque_image_tiles.is_empty() {
                pass.set_pipeline(&self.opaque_image_pipeline);
                pass.set_vertex_buffer(0, self.tiles_vbo.buffer.slice(..));
                pass.set_bind_group(1, &self.mask_texture_bind_group, &[]);
                pass.set_bind_group(2, &self.src_color_bind_group, &[]);
                pass.draw_indexed(0..6, 0, tile_encoder.ranges.opaque_image_tiles.to_u32());
            }

            if !first_mask_pass.batches.is_empty() {
                pass.set_vertex_buffer(0, self.tiles_vbo.buffer.slice(tile_encoder.ranges.alpha_tiles.byte_range::<TileInstance>()));
                pass.set_bind_group(1, &self.mask_texture_bind_group, &[]);
                for batch in &tile_encoder.batches[first_mask_pass.batches.clone()] {
                    match batch.batch_kind {
                        0 => {
                            pass.set_pipeline(&self.masked_solid_pipeline);
                        }
                        1 => {
                            pass.set_pipeline(&self.masked_image_pipeline);
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

        if tile_encoder.mask_passes.len() > 1 {
            for pass_idx in 1..tile_encoder.mask_passes.len() {
                let mask_ranges = &tile_encoder.mask_passes[pass_idx];

                if src_color_atlas != mask_ranges.color_atlas_index {
                    src_color_atlas = mask_ranges.color_atlas_index;
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
        
                    pass.set_index_buffer(self.quad_ibo.slice(..), wgpu::IndexFormat::Uint16);
                    pass.set_bind_group(0, &self.mask_atlas_desc_bind_group, &[]);

                    for pattern in patterns {
                        pattern.render(src_color_atlas, &mut pass);
                    }
                }
        
                {
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

                    pass.set_index_buffer(self.quad_ibo.slice(..), wgpu::IndexFormat::Uint16);

                    if !mask_ranges.gpu_masks.is_empty() {
                        pass.set_pipeline(&self.masks.line_pipeline);
                        pass.set_bind_group(0, &self.masks_bind_group, &[]);
                        pass.set_vertex_buffer(0, self.masks_vbo.buffer.slice(..));
                        pass.draw_indexed(0..6, 0, mask_ranges.gpu_masks.clone());
                    }

                    if !mask_ranges.circle_masks.is_empty() {
                        pass.set_pipeline(&self.masks.circle_pipeline);
                        pass.set_bind_group(0, &self.masks_bind_group, &[]);
                        pass.set_vertex_buffer(0, self.circles_vbo.buffer.slice(..));
                        pass.draw_indexed(0..6, 0, mask_ranges.circle_masks.clone());
                    }

                    tile_encoder.mask_uploader.upload(
                        &device,
                        &mut pass,
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
                    pass.set_vertex_buffer(0, self.tiles_vbo.buffer.slice(tile_encoder.ranges.alpha_tiles.byte_range::<TileInstance>()));

                    pass.set_bind_group(0, globals_bind_group, &[]);

                    for batch in &tile_encoder.batches[mask_ranges.batches.clone()] {
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
        }
    }
}

pub struct Masks {
    pub line_pipeline: wgpu::RenderPipeline,
    pub quad_pipeline: wgpu::RenderPipeline,
    pub circle_pipeline: wgpu::RenderPipeline,
    pub mask_bind_group_layout: wgpu::BindGroupLayout,
}

impl Masks {
    pub fn new(device: &wgpu::Device, shaders: &mut ShaderSources) -> Self {
        create_mask_pipeline(device, shaders)
    }
}

fn create_mask_pipeline(device: &wgpu::Device, shaders: &mut ShaderSources) -> Masks {
    let quad_src = include_str!("../shaders/mask_fill_quads.wgsl");
    let lin_src = include_str!("../shaders/mask_fill.wgsl");
    let circle_src = include_str!("../shaders/mask_circle.wgsl");
    let lin_module = &shaders.create_shader_module(device, "Mask fill linear", lin_src, &[]);
    let quad_module = &shaders.create_shader_module(device, "Mask fill quad", quad_src, &[]);
    let circle_module = &shaders.create_shader_module(device, "Circle mask", circle_src, &[]);

    let mask_globals_buffer_size = std::mem::size_of::<GpuTargetDescriptor>() as u64;

    let mask_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        bind_group_layouts: &[&mask_bind_group_layout],
        push_constant_ranges: &[],
    });

    let defaults = PipelineDefaults::new();
    let mut attributes = VertexBuilder::new();
    attributes.push(wgpu::VertexFormat::Uint32x2);
    attributes.push(wgpu::VertexFormat::Uint32);
    attributes.push(wgpu::VertexFormat::Uint32);

    let line_tile_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
        label: Some("Tile mask linear"),
        layout: Some(&tile_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &lin_module,
            entry_point: "vs_main",
            buffers: &[attributes.buffer_layout()],
        },
        fragment: Some(wgpu::FragmentState {
            module: &lin_module,
            entry_point: "fs_main",
            targets: defaults.alpha_target_state(),
        }),
        primitive: defaults.primitive_state(),
        depth_stencil: None,
        multiview: None,
        multisample: wgpu::MultisampleState::default(),
    };

    let quad_tile_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
        label: Some("Tile mask quad"),
        layout: Some(&tile_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &quad_module,
            entry_point: "vs_main",
            buffers: &[attributes.buffer_layout()],
        },
        fragment: Some(wgpu::FragmentState {
            module: &quad_module,
            entry_point: "fs_main",
            targets: defaults.alpha_target_state(),
        }),
        primitive: defaults.primitive_state(),
        depth_stencil: None,
        multiview: None,
        multisample: wgpu::MultisampleState::default(),
    };

    let line_pipeline = device.create_render_pipeline(&line_tile_pipeline_descriptor);
    let quad_pipeline = device.create_render_pipeline(&quad_tile_pipeline_descriptor);

    let mut attributes = VertexBuilder::new();
    attributes.push(wgpu::VertexFormat::Uint32);
    attributes.push(wgpu::VertexFormat::Float32);
    attributes.push(wgpu::VertexFormat::Float32x2);

    let circle_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
        label: Some("Circle mask"),
        layout: Some(&tile_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &circle_module,
            entry_point: "vs_main",
            buffers: &[attributes.buffer_layout()],
        },
        fragment: Some(wgpu::FragmentState {
            module: &circle_module,
            entry_point: "fs_main",
            targets: defaults.alpha_target_state(),
        }),
        primitive: defaults.primitive_state(),
        depth_stencil: None,
        multiview: None,
        multisample: wgpu::MultisampleState::default(),
    };

    let circle_pipeline = device.create_render_pipeline(&circle_pipeline_descriptor);

    Masks {
        line_pipeline,
        quad_pipeline,
        circle_pipeline,
        mask_bind_group_layout,
    }
}

pub struct BumpAllocatedBuffer {
    pub label: &'static str,
    pub usage: wgpu::BufferUsages,
    pub buffer: wgpu::Buffer,
    pub allocator: BufferBumpAllocator,
    pub allocated_size: u32,
    pub size_per_element: u32,
}

impl BumpAllocatedBuffer {
    pub fn new<Ty>(device: &wgpu::Device, label: &'static str, byte_size: u32, usage: wgpu::BufferUsages) -> Self {
        let size_per_element = std::mem::size_of::<Ty>() as u32;
        BumpAllocatedBuffer {
            label,
            usage,
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: byte_size as u64,
                usage,
                mapped_at_creation: false,
            }),
            allocator: BufferBumpAllocator::new(),
            allocated_size: byte_size,
            size_per_element,
        }
    }

    pub fn begin_frame(&mut self) {
        self.allocator.clear();
    }

    pub fn ensure_allocated(&mut self, device: &wgpu::Device) -> bool {
        if self.allocator.len() * self.size_per_element <= self.allocated_size {
            return false;
        }

        let p = self.allocated_size;
        let s = self.allocator.len() * self.size_per_element;
        let multiple = 4096 * 4;
        self.allocated_size = (s + (multiple - 1)) & !(multiple - 1);
        println!("reallocate {:?} from {} to {} ({})", self.label, p, self.allocated_size, s);

        self.buffer.destroy();
        self.buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(self.label),
            size: self.allocated_size as u64,
            usage: self.usage,
            mapped_at_creation: false,
        });

        true
    }

    pub fn buffer(&self) -> &wgpu::Buffer { &self.buffer }
}

pub struct BufferBumpAllocator {
    cursor: u32,
}

impl BufferBumpAllocator {
    pub fn new() -> Self {
        BufferBumpAllocator { cursor: 0 }
    }

    pub fn push(&mut self, n: usize) -> BufferRange {
        let range = BufferRange(self.cursor, self.cursor + n as u32);
        self.cursor = range.1;

        range
    }

    pub fn len(&self) -> u32 {
        self.cursor
    }

    pub fn clear(&mut self) {
        self.cursor = 0;
    }
}
