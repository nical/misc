use lyon::geom::Box2D;

const STAGING_BUFFER_SIZE: u32 = 65536;

use crate::gpu::ShaderSources;
use crate::buffer::{Buffer, UniformBufferPool};
use std::ops::Range;

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
    pub rect: Box2D<f32>,
    pub mask: u32,
    pub color: u32,
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

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct MaskParams {
    pub tile_size: f32,
    pub inv_atlas_width: f32,
    pub masks_per_row: u32,
}

unsafe impl bytemuck::Pod for MaskParams {}
unsafe impl bytemuck::Zeroable for MaskParams {}

pub struct MaskedTiles {
    pub masked_solid_pipeline: wgpu::RenderPipeline,
    pub masked_image_pipeline: wgpu::RenderPipeline,
    pub opaque_solid_pipeline: wgpu::RenderPipeline,
    pub opaque_image_pipeline: wgpu::RenderPipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl MaskedTiles {
    pub fn new(device: &wgpu::Device, globals_bg_layout: &wgpu::BindGroupLayout, shaders: &mut ShaderSources) -> Self {
        create_tile_pipeline(device, globals_bg_layout, shaders)
    }
}

fn create_tile_pipeline(device: &wgpu::Device, globals_bg_layout: &wgpu::BindGroupLayout, shaders: &mut ShaderSources) -> MaskedTiles {
    let src = include_str!("./../../shaders/tile.wgsl");
    let masked_solid_module = shaders.create_shader_module(device, "masked_tile_solid", src, &["TILED_MASK", "SOLID_PATTERN"]);
    let masked_img_module = shaders.create_shader_module(device, "masked_tile_image", src, &["TILED_MASK", "TILED_IMAGE_PATTERN",]);
    let opaque_solid_module = shaders.create_shader_module(device, "masked_tile_solid", src, &["SOLID_PATTERN"]);
    let opaque_img_module = shaders.create_shader_module(device, "masked_tile_image", src, &["TILED_IMAGE_PATTERN",]);

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

    let attributes_with_opacity = &[
        wgpu::VertexAttribute {
            offset: 0,
            format: wgpu::VertexFormat::Float32x4,
            shader_location: 0,
        },
        wgpu::VertexAttribute {
            offset: 16,
            format: wgpu::VertexFormat::Uint32,
            shader_location: 1,
        },
        wgpu::VertexAttribute {
            offset: 20,
            format: wgpu::VertexFormat::Uint32,
            shader_location: 2,
        },
        // opacity
        wgpu::VertexAttribute {
            offset: 24,
            format: wgpu::VertexFormat::Float32,
            shader_location: 3,
        },
    ];

    let attributes_no_opacity = &attributes_with_opacity[..3];
    let vertex_buffer_layouts = &[wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<TileInstance>() as u64,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: attributes_no_opacity,
    }];
    let alpha_color_target_states = &[
        Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Bgra8Unorm,
            blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
            write_mask: wgpu::ColorWrites::ALL,
        }),
    ];
    let opaque_color_target_states = &[
        Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Bgra8Unorm,
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
        bind_group_layouts: &[&globals_bg_layout],
        push_constant_ranges: &[],
    });
    let masked_solid_tile_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Masked solid tiles"),
        bind_group_layouts: &[&globals_bg_layout, &bind_group_layout],
        push_constant_ranges: &[],
    });
    let masked_img_tile_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Masked image tiles"),
        bind_group_layouts: &[
            &globals_bg_layout,
            &bind_group_layout,
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
        multisample,
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

    MaskedTiles {
        masked_solid_pipeline,
        masked_image_pipeline,
        opaque_solid_pipeline,
        opaque_image_pipeline,
        bind_group_layout,
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
    let vs = include_str!("./../../shaders/mask_fill.vs.wgsl");
    let lin_fs = include_str!("./../../shaders/mask_fill_lin.fs.wgsl");
    let quad_fs = include_str!("./../../shaders/mask_fill_quad.fs.wgsl");
    let vs_module = &shaders.create_shader_module(device, "Mask vs", vs, &[]);
    let lin_fs_module = &shaders.create_shader_module(device, "Mask fill linear fs", lin_fs, &[]);
    let quad_fs_module = &shaders.create_shader_module(device, "Mask fill quad fs", quad_fs, &[]);

    let mask_globals_buffer_size = std::mem::size_of::<MaskParams>() as u64;

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
            module: &vs_module,
            entry_point: "main",
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
            module: &lin_fs_module,
            entry_point: "main",
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
            module: &vs_module,
            entry_point: "main",
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
            module: &quad_fs_module,
            entry_point: "main",
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

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct CpuMask {
    pub mask_id: u32,
    pub byte_offset: u32,
}

unsafe impl bytemuck::Pod for CpuMask {}
unsafe impl bytemuck::Zeroable for CpuMask {}


pub struct MaskUploadCopies {
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl MaskUploadCopies {
    pub fn new(device: &wgpu::Device, globals_bg_layout: &wgpu::BindGroupLayout) -> Self {
        create_mask_upload_pipeline(device, globals_bg_layout)
    }
}

fn create_mask_upload_pipeline(device: &wgpu::Device, globals_bg_layout: &wgpu::BindGroupLayout) -> MaskUploadCopies {
    let vs_module = &device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Mask upload copy vs"),
        source: wgpu::ShaderSource::Wgsl(include_str!("./../../shaders/mask_upload_copy.vs.wgsl").into()),
    });
    let fs_module = &device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Mask upload copy fs"),
        source: wgpu::ShaderSource::Wgsl(include_str!("./../../shaders/mask_upload_copy.fs.wgsl").into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Mask upload copy"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(65536),
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Mask upload copy"),
        bind_group_layouts: &[&globals_bg_layout, &bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline_descriptor = wgpu::RenderPipelineDescriptor {
        label: Some("Mask upload copy"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &vs_module,
            entry_point: "main",
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<CpuMask>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[
                    wgpu::VertexAttribute {
                        offset: 0,
                        format: wgpu::VertexFormat::Uint32,
                        shader_location: 0,
                    },
                    wgpu::VertexAttribute {
                        offset: 4,
                        format: wgpu::VertexFormat::Uint32,
                        shader_location: 1,
                    },
                ],
            }],
        },
        fragment: Some(wgpu::FragmentState {
            module: &fs_module,
            entry_point: "main",
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

    let pipeline = device.create_render_pipeline(&pipeline_descriptor);

    MaskUploadCopies {
        pipeline,
        bind_group_layout,
    }
}

struct MaskUploadBatch {
    instances: Range<u32>,
    src_index: u32,
    dst_atlas: u32,
}


pub struct MaskUploader {
    pool: UniformBufferPool<u8>,

    batches: Vec<MaskUploadBatch>,
    copy_instances: Vec<CpuMask>,
    copy_instance_buffer: Option<wgpu::Buffer>,

    current_atlas: u32,
    current_instance_start: u32,

    pub current_mask_buffer: Buffer<u8>,

    masks_per_atlas: u32,
}

unsafe impl Send for MaskUploader {}

impl MaskUploader {
    pub fn new(device: *const wgpu::Device, bind_group_layout: *const wgpu::BindGroupLayout, atlas_size: u32) -> Self {
        MaskUploader {
            pool: UniformBufferPool::new(
                STAGING_BUFFER_SIZE,
                device,
                bind_group_layout,
            ),
            batches: Vec::new(),
            copy_instances: Vec::new(),
            copy_instance_buffer: None,
            current_atlas: 0,
            current_instance_start: 0,
            current_mask_buffer: Buffer::empty(),
            masks_per_atlas: (atlas_size * atlas_size) / (16 * 16),
        }
    }

    pub fn create_similar(&self) -> Self {
        MaskUploader {
            // TODO: we should use the same pool instead.
            pool: self.pool.create_similar(),
            batches: Vec::new(),
            copy_instances: Vec::new(),
            copy_instance_buffer: None,
            current_atlas: 0,
            current_instance_start: 0,
            current_mask_buffer: Buffer::empty(),
            masks_per_atlas: self.masks_per_atlas,
        }
    }

    pub fn reset(&mut self) {
        if self.current_mask_buffer.capacity() > 0 {
            self.pool.return_buffer(std::mem::replace(&mut self.current_mask_buffer, Buffer::empty()));
        }
        self.pool.reset();
        self.batches.clear();
        self.copy_instances.clear();
        self.current_atlas = 0;
        self.current_instance_start = 0;
    }

    pub fn new_mask(&mut self, id: u32) -> Range<usize> {
        const TILE_SIZE: usize = 16;

        let atlas_index = id / self.masks_per_atlas;

        if atlas_index != self.current_atlas || self.current_mask_buffer.remaining_capacity() < TILE_SIZE * TILE_SIZE {
            let instance_end = self.copy_instances.len() as u32;
            if instance_end > self.current_instance_start {
                self.batches.push(MaskUploadBatch {
                    instances: self.current_instance_start .. instance_end,
                    src_index: self.current_mask_buffer.index(),
                    dst_atlas: self.current_atlas,
                });

                self.current_instance_start = instance_end;
            }

            if (self.current_mask_buffer.remaining_capacity() as u32) < STAGING_BUFFER_SIZE / 2 {
                let buf = std::mem::replace(&mut self.current_mask_buffer, self.pool.get_buffer());

                if buf.capacity() > 0 {
                    self.pool.return_buffer(buf);
                }
            }

            self.current_atlas = atlas_index;
        }

        let start = self.current_mask_buffer.len();
        let end = start + (TILE_SIZE * TILE_SIZE);

        self.copy_instances.push(CpuMask {
            mask_id: id,
            byte_offset: start as u32,
        });

        start..end
    }

    pub fn needs_upload(&self) -> bool {
        !self.copy_instances.is_empty()
    }

    pub fn upload<'a, 'c, 'b: 'a>(
        &'b mut self,
        device: &'b wgpu::Device,
        pass: &'c mut wgpu::RenderPass<'a>,
        globals_bind_group: &'b wgpu::BindGroup,
        pipeline: &'b wgpu::RenderPipeline,
        quad_ibo: &'b wgpu::Buffer,
        mask_pass_index: u32,
    ) -> bool {
        use wgpu::util::DeviceExt;

        let instance_end = self.copy_instances.len() as u32;
        if instance_end > self.current_instance_start {
            self.batches.push(MaskUploadBatch {
                instances: self.current_instance_start .. instance_end,
                src_index: self.current_mask_buffer.index(),
                dst_atlas: self.current_atlas,
            });

            self.current_instance_start = instance_end;
        }

        if self.current_mask_buffer.capacity() > 0 {
            self.pool.return_buffer(std::mem::replace(&mut self.current_mask_buffer, Buffer::empty()));
        }

        if self.batches.is_empty() {
            return false;
        }

        if self.copy_instance_buffer.is_none() {
            self.copy_instance_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Mask copy instances"),
                contents: bytemuck::cast_slice(&self.copy_instances),
                usage: wgpu::BufferUsages::VERTEX,
            }));
        }

        let instances = self.copy_instance_buffer.as_ref().unwrap();

        let mut batches_start = 0;
        while batches_start < self.batches.len() {
            let current_atlas = self.batches[batches_start].dst_atlas;
            if current_atlas != mask_pass_index {
                batches_start += 1;
                continue;
            }

            pass.set_pipeline(pipeline);
            pass.set_index_buffer(quad_ibo.slice(..), wgpu::IndexFormat::Uint16);
            pass.set_vertex_buffer(0, instances.slice(..));
            pass.set_bind_group(0, globals_bind_group, &[]);

            let mut idx = batches_start;
            while idx < self.batches.len() && self.batches[idx].dst_atlas == current_atlas {
                let batch = &self.batches[idx];
                let bind_group = &self.pool.get_bind_group(batch.src_index);

                pass.set_bind_group(1, bind_group, &[]);
                pass.draw_indexed(0..6, 0, batch.instances.clone());

                idx += 1;
            }

            batches_start = idx;
        }

        //self.batches.clear();

        true
    }

    pub fn copy_instances(&self) -> &[CpuMask] {
        &self.copy_instances
    }
}

