use crate::{gpu::{
    atlas_uploader::{MaskUploadCopies},
    ShaderSources, GpuTargetDescriptor, VertexBuilder, PipelineDefaults, storage_buffer::*,
}, custom_pattern::TilePipelines, canvas::CommonGpuResources};
use crate::tiling::{
    encoder::{LineEdge},
    TilePosition, PatternData
};
use crate::canvas::RendererResources;

use wgpu::util::DeviceExt;

use super::{PatternIndex};

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct TileInstance {
    pub position: TilePosition,
    pub mask: TilePosition,
    pub pattern_position: TilePosition,
    pub pattern_data: PatternData,
}

unsafe impl bytemuck::Pod for TileInstance {}
unsafe impl bytemuck::Zeroable for TileInstance {}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct MaskedTileInstance {
    pub tile: TileInstance,
    pub mask: [u32; 4],
}

unsafe impl bytemuck::Pod for MaskedTileInstance {}
unsafe impl bytemuck::Zeroable for MaskedTileInstance {}

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


pub struct TilingGpuResources {
    pub mask_upload_copies: MaskUploadCopies,
    pub masks: Masks,

    // TODO: masked/opaque tiled images are similar to other patterns, maybe they
    // could be implemented as patterns.
    pub masked_image_pipeline: wgpu::RenderPipeline,
    pub opaque_image_pipeline: wgpu::RenderPipeline,
    pub mask_texture_bind_group_layout: wgpu::BindGroupLayout,
    pub mask_texture_view: wgpu::TextureView,
    pub src_color_texture_view: wgpu::TextureView,

    pub mask_texture_bind_group: wgpu::BindGroup,
    pub src_color_bind_group: wgpu::BindGroup,
    pub masks_bind_group: wgpu::BindGroup,
    pub mask_atlas_target_and_gpu_store_bind_group: wgpu::BindGroup,
    pub color_atlas_target_and_gpu_store_bind_group: wgpu::BindGroup,

    pub edges: StorageBuffer,
    pub mask_params_ubo: wgpu::Buffer,
    pub patterns: Vec<TilePipelines>,
}

impl TilingGpuResources {
    pub fn new(
        common: &CommonGpuResources,
        device: &wgpu::Device,
        shaders: &mut ShaderSources,
        mask_atlas_size: u32,
        color_atlas_size: u32,
    ) -> Self {
        let edges = StorageBuffer::new::<LineEdge>(
            device,
            "edges",
            4096 * 256,
            StorageKind::Buffer,
        );

        let mask_params_ubo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Mask params"),
            contents: bytemuck::cast_slice(&[
                GpuTargetDescriptor::new(mask_atlas_size, mask_atlas_size)
            ]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let color_tiles_params_ubo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Mask params"),
            contents: bytemuck::cast_slice(&[
                GpuTargetDescriptor::new(color_atlas_size, color_atlas_size)
            ]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let mask_upload_copies = MaskUploadCopies::new(&device, shaders, &common.target_and_gpu_store_layout, crate::BYTES_PER_MASK as u32 * 2048);
        let masks = Masks::new(&device, shaders, &edges);

        let src = include_str!("./../../shaders/tile.wgsl");
        let masked_img_module = shaders.create_shader_module(device, "masked_tile_image", src, &["TILED_MASK", "TILED_IMAGE_PATTERN",]);
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
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: edges.binding_type(),
                    count: None,
                }
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
        let opaque_attributes = VertexBuilder::from_slice(wgpu::VertexStepMode::Instance, &[wgpu::VertexFormat::Uint32x4]);
        let masked_attributes = VertexBuilder::from_slice(
            wgpu::VertexStepMode::Instance,
            &[
                wgpu::VertexFormat::Uint32x4,
                wgpu::VertexFormat::Uint32x4,
            ],
        );

        let masked_img_tile_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Masked image tiles"),
            bind_group_layouts: &[
                &common.target_and_gpu_store_layout,
                &mask_texture_bind_group_layout,
                &src_color_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        let primitive = PipelineDefaults::primitive_state();
        let multisample = wgpu::MultisampleState::default();

        let masked_image_tile_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
            label: Some("Masked image tiles"),
            layout: Some(&masked_img_tile_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &masked_img_module,
                entry_point: "vs_main",
                buffers: &[masked_attributes.buffer_layout()],
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

        let opaque_image_tile_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
            label: Some("Opaque image tiles"),
            layout: Some(&masked_img_tile_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &opaque_img_module,
                entry_point: "vs_main",
                buffers: &[opaque_attributes.buffer_layout()],
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

        let masked_image_pipeline = device.create_render_pipeline(&masked_image_tile_pipeline_descriptor);
        let opaque_image_pipeline = device.create_render_pipeline(&opaque_image_tile_pipeline_descriptor);

        let mask_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Mask atlas"),
            dimension: wgpu::TextureDimension::D2,
            size: wgpu::Extent3d {
                width: mask_atlas_size,
                height: mask_atlas_size,
                depth_or_array_layers: 1,
            },
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            mip_level_count: 1,
            sample_count: 1,
            view_formats: &[wgpu::TextureFormat::R8Unorm],
        });

        let mask_texture_view = mask_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let src_color_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Color tile atlas"),
            dimension: wgpu::TextureDimension::D2,
            size: wgpu::Extent3d {
                width: color_atlas_size,
                height: color_atlas_size,
                depth_or_array_layers: 1,
            },
            format: PipelineDefaults::color_format(),
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            mip_level_count: 1,
            sample_count: 1,
            view_formats: &[PipelineDefaults::color_format()],
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
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: edges.binding_resource()
                }
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

        let mask_atlas_target_and_gpu_store_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Atlas target & gpu store"),
            layout: &common.target_and_gpu_store_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(mask_params_ubo.as_entire_buffer_binding())
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&common.gpu_store_view)
                }
            ],
        });

        let color_atlas_target_and_gpu_store_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Atlas target & gpu store"),
            layout: &common.target_and_gpu_store_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(color_tiles_params_ubo.as_entire_buffer_binding())
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&common.gpu_store_view)
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
                    resource: edges.binding_resource(),
                },
            ],
        });

        TilingGpuResources {
            mask_upload_copies,
            masks,

            opaque_image_pipeline,
            masked_image_pipeline,
            mask_texture_bind_group_layout,

            mask_texture_view,
            src_color_texture_view,
            mask_texture_bind_group,
            src_color_bind_group,
            masks_bind_group,
            mask_atlas_target_and_gpu_store_bind_group,
            color_atlas_target_and_gpu_store_bind_group,
            edges,
            mask_params_ubo,
            patterns: Vec::new(),
        }
    }

    pub fn register_pattern(&mut self, pipelines: TilePipelines) -> PatternIndex {
        let idx = self.patterns.len() as PatternIndex;
        self.patterns.push(pipelines);
        idx
    }

    pub fn begin_frame(&mut self) {
        self.edges.begin_frame();
    }

    pub fn end_frame(&mut self) {
    }

    pub fn allocate(&mut self, device: &wgpu::Device) {
        if self.edges.ensure_allocated(device) {
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
                        resource: self.edges.binding_resource(),
                    },
                ],
            });
        }
    }
}

impl RendererResources for TilingGpuResources {
    fn name(&self) -> &'static str { "TilingGpuResources" }

    fn begin_frame(&mut self) {
        TilingGpuResources::begin_frame(self)
    }

    fn end_frame(&mut self) {
        TilingGpuResources::end_frame(self);
    }
}

pub struct Masks {
    pub fill_pipeline: wgpu::RenderPipeline,
    pub circle_pipeline: wgpu::RenderPipeline,
    pub rect_pipeline: wgpu::RenderPipeline,
    pub mask_bind_group_layout: wgpu::BindGroupLayout,
}

impl Masks {
    pub fn new(device: &wgpu::Device, shaders: &mut ShaderSources, edges: &StorageBuffer) -> Self {
        create_mask_pipeline(device, shaders, edges)
    }
}

fn create_mask_pipeline(device: &wgpu::Device, shaders: &mut ShaderSources, edges: &StorageBuffer) -> Masks {
    let fill_src = include_str!("../../shaders/mask_fill.wgsl");
    let circle_src = include_str!("../../shaders/mask_circle.wgsl");
    let rect_src = include_str!("../../shaders/mask_rect.wgsl");
    let circle_module = &shaders.create_shader_module(device, "Circle mask", circle_src, &[]);
    let rect_module = &shaders.create_shader_module(device, "Rectangle mask", rect_src, &[]);

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
                ty: edges.binding_type(),
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
    let mut attributes = VertexBuilder::new(wgpu::VertexStepMode::Instance);
    attributes.push(wgpu::VertexFormat::Uint32x2);
    attributes.push(wgpu::VertexFormat::Uint32);
    attributes.push(wgpu::VertexFormat::Uint32);

    let features: &[&str] = if edges.texture().is_some() {
        &["EDGE_TEXTURE"]
    } else {
        &[]
    };
    let fill_module = shaders.create_shader_module(device, "Mask fill linear", fill_src, features);

    let fill_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
        label: Some("Fill mask"),
        layout: Some(&tile_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &fill_module,
            entry_point: "vs_main",
            buffers: &[attributes.buffer_layout()],
        },
        fragment: Some(wgpu::FragmentState {
            module: &fill_module,
            entry_point: "fs_main",
            targets: defaults.alpha_target_state(),
        }),
        primitive: PipelineDefaults::primitive_state(),
        depth_stencil: None,
        multiview: None,
        multisample: wgpu::MultisampleState::default(),
    };

    let fill_pipeline = device.create_render_pipeline(&fill_pipeline_descriptor);

    let mut attributes = VertexBuilder::new(wgpu::VertexStepMode::Instance);
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
        primitive: PipelineDefaults::primitive_state(),
        depth_stencil: None,
        multiview: None,
        multisample: wgpu::MultisampleState::default(),
    };

    let circle_pipeline = device.create_render_pipeline(&circle_pipeline_descriptor);

    let mut attributes = VertexBuilder::new(wgpu::VertexStepMode::Instance);
    attributes.push(wgpu::VertexFormat::Uint32x4);

    let rect_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
        label: Some("Rectangle mask"),
        layout: Some(&tile_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &rect_module,
            entry_point: "vs_main",
            buffers: &[attributes.buffer_layout()],
        },
        fragment: Some(wgpu::FragmentState {
            module: &rect_module,
            entry_point: "fs_main",
            targets: defaults.alpha_target_state(),
        }),
        primitive: PipelineDefaults::primitive_state(),
        depth_stencil: None,
        multiview: None,
        multisample: wgpu::MultisampleState::default(),
    };

    let rect_pipeline = device.create_render_pipeline(&rect_pipeline_descriptor);

    Masks {
        fill_pipeline,
        circle_pipeline,
        rect_pipeline,
        mask_bind_group_layout,
    }
}
