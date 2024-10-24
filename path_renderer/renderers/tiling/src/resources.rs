use crate::{atlas_uploader::MaskUploadCopies, encoder::LineEdge, PatternData, TilePosition};
use core::bytemuck;
use core::gpu::PipelineDefaults;
use core::wgpu;
use core::wgpu::util::DeviceExt;
use core::{
    gpu::{
        shader::{
            BindGroupLayout, BindGroupLayoutId, Binding, BaseShaderId,
            BaseShaderDescriptor,
            Varying, VertexAtribute,
        },
        storage_buffer::*,
        RenderPassDescriptor, Shaders, VertexBuilder,
    },
    resources::{CommonGpuResources, RendererResources},
};
use pattern_texture::TextureRenderer;

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

    pub mask_texture_bind_group_layout: BindGroupLayoutId,
    pub mask_texture_view: wgpu::TextureView,
    pub src_color_texture_view: wgpu::TextureView,

    pub mask_texture_bind_group: wgpu::BindGroup,
    pub src_color_bind_group: wgpu::BindGroup,
    pub masks_bind_group: wgpu::BindGroup,
    pub mask_atlas_target_and_gpu_store_bind_group: wgpu::BindGroup,
    pub color_atlas_target_and_gpu_store_bind_group: wgpu::BindGroup,

    pub edges: StorageBuffer,
    pub mask_params_ubo: wgpu::Buffer,

    pub opaque_pipeline: BaseShaderId,
    pub masked_pipeline: BaseShaderId,
    pub texture: TextureRenderer,
}

const BASE_MASKED_SHADER_SRC: &'static str = "
#import render_target
#import tiling
#import rect

fn base_vertex(vertex_index: u32, instance_data: vec4<u32>) -> BaseVertex {
    var uv = rect_get_uv(vertex_index);
    var tile = tiling_decode_instance(instance_data, uv);
    var target_position = canvas_to_target(tile.position);

    // TODO: z_index
    return BaseVertex(
        target_position,
        tile.pattern_position,
        tile.pattern_data,
        tile.mask_position,
    );
}

fn base_fragment(mask_uv: vec2f) -> f32 {
    var uv = vec2<i32>(i32(mask_uv.x), i32(mask_uv.y));
    return textureLoad(mask_atlas_texture, uv, 0).r;
}
";

const BASE_OPAQUE_SHADER_SRC: &'static str = "
#import render_target
#import tiling
#import rect

fn base_vertex(vertex_index: u32, instance_data: vec4<u32>) -> BaseVertex {
    var uv = rect_get_uv(vertex_index);
    var tile = tiling_decode_instance(instance_data, uv);
    var target_position = canvas_to_target(tile.position);

    // TODO: z_index
    return BaseVertex(
        target_position,
        tile.pattern_position,
        tile.pattern_data,
    );
}

fn base_fragment() -> f32 { return 1.0; }
";


impl TilingGpuResources {
    pub fn new(
        common: &mut CommonGpuResources,
        device: &wgpu::Device,
        shaders: &mut Shaders,
        texture: &TextureRenderer,
        mask_atlas_size: u32,
        color_atlas_size: u32,
        use_ssaa4: bool,
    ) -> Self {
        shaders.register_library(
            "mask::fill".into(),
            include_str!("../shaders/fill.wgsl").into(),
        );

        let mask_atlas_bind_group_layout =
            shaders.register_bind_group_layout(BindGroupLayout::new(
                device,
                "tile_mask_atlas".into(),
                vec![Binding {
                    name: "mask_atlas_texture".into(),
                    struct_type: "f32".into(),
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                }],
            ));

        let masked_pipeline = shaders.register_base_shader(BaseShaderDescriptor {
            name: "masked_tile".into(),
            source: BASE_MASKED_SHADER_SRC.into(),
            vertex_attributes: Vec::new(),
            instance_attributes: vec![VertexAtribute::uint32x4("instance")],
            varyings: vec![Varying::float32x2("mask_uv").interpolated()],
            bindings: Some(mask_atlas_bind_group_layout),
            primitive: PipelineDefaults::primitive_state(),
            shader_defines: Vec::new(),
        });

        let opaque_pipeline = shaders.register_base_shader(BaseShaderDescriptor {
            name: "opaque_tile".into(),
            source: BASE_OPAQUE_SHADER_SRC.into(),
            vertex_attributes: Vec::new(),
            instance_attributes: vec![VertexAtribute::uint32x4("instance")],
            varyings: Vec::new(),
            bindings: None,
            primitive: PipelineDefaults::primitive_state(),
            shader_defines: Vec::new(),
        });

        let edges =
            StorageBuffer::new::<LineEdge>(device, "edges", 4096 * 256, StorageKind::Buffer);

        let mask_params_ubo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Mask params"),
            contents: bytemuck::cast_slice(&[RenderPassDescriptor::new(
                mask_atlas_size,
                mask_atlas_size,
            )]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let color_tiles_params_ubo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Mask params"),
            contents: bytemuck::cast_slice(&[RenderPassDescriptor::new(
                color_atlas_size,
                color_atlas_size,
            )]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let mask_upload_copies =
            MaskUploadCopies::new(&device, shaders, crate::BYTES_PER_MASK as u32 * 2048);
        let masks = Masks::new(&device, shaders, &edges, use_ssaa4);

        let target_and_gpu_store_layout = &shaders.get_base_bind_group_layout().handle;

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

        let mask_texture_view = mask_texture.create_view(&Default::default());

        let src_color_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Color tile atlas"),
            dimension: wgpu::TextureDimension::D2,
            size: wgpu::Extent3d {
                width: color_atlas_size,
                height: color_atlas_size,
                depth_or_array_layers: 1,
            },
            format: shaders.defaults.color_format(),
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            mip_level_count: 1,
            sample_count: 1,
            view_formats: &[shaders.defaults.color_format()],
        });

        let src_color_texture_view =
            src_color_texture.create_view(&Default::default());

        let mask_texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("alpha tiles"),
            layout: &shaders
                .get_bind_group_layout(mask_atlas_bind_group_layout)
                .handle,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&mask_texture_view),
            }],
        });

        let src_color_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Color tiles"),
            layout: &shaders
                .get_bind_group_layout(texture.bind_group_layout())
                .handle,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&src_color_texture_view),
            }],
        });

        let mask_atlas_target_and_gpu_store_bind_group =
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Atlas target & gpu store"),
                layout: target_and_gpu_store_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(
                            mask_params_ubo.as_entire_buffer_binding(),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&common.gpu_store.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&common.default_sampler),
                    },
                ],
            });

        let color_atlas_target_and_gpu_store_bind_group =
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Atlas target & gpu store"),
                layout: target_and_gpu_store_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(
                            color_tiles_params_ubo.as_entire_buffer_binding(),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&common.gpu_store.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&common.default_sampler),
                    },
                ],
            });

        let masks_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Masks"),
            layout: &masks.mask_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(
                        mask_params_ubo.as_entire_buffer_binding(),
                    ),
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

            mask_texture_bind_group_layout: mask_atlas_bind_group_layout,

            mask_texture_view,
            src_color_texture_view,
            mask_texture_bind_group,
            src_color_bind_group,
            masks_bind_group,
            mask_atlas_target_and_gpu_store_bind_group,
            color_atlas_target_and_gpu_store_bind_group,
            edges,
            mask_params_ubo,

            opaque_pipeline,
            masked_pipeline,
            texture: texture.clone(),
        }
    }

    pub fn begin_frame(&mut self) {
        self.edges.begin_frame();
    }

    pub fn end_frame(&mut self) {}

    pub fn allocate(&mut self, device: &wgpu::Device) {
        if self.edges.ensure_allocated(device) {
            // reallocated the buffer, need to re-create the bind group.
            self.masks_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Masks"),
                layout: &self.masks.mask_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(
                            self.mask_params_ubo.as_entire_buffer_binding(),
                        ),
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
    fn name(&self) -> &'static str {
        "TilingGpuResources"
    }

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
    pub fn new(
        device: &wgpu::Device,
        shaders: &mut Shaders,
        edges: &StorageBuffer,
        use_ssaa: bool,
    ) -> Self {
        create_mask_pipeline(device, shaders, edges, use_ssaa)
    }
}

fn create_mask_pipeline(
    device: &wgpu::Device,
    shaders: &mut Shaders,
    edges: &StorageBuffer,
    use_ssaa4: bool,
) -> Masks {
    let fill_src = include_str!("../shaders/mask_fill.wgsl");
    let circle_src = include_str!("../shaders/mask_circle.wgsl");
    let rect_src = include_str!("../shaders/mask_rect.wgsl");
    let circle_module = &shaders.create_shader_module(device, "Circle mask", circle_src, &[]);
    let rect_module = &shaders.create_shader_module(device, "Rectangle mask", rect_src, &[]);

    let mask_globals_buffer_size = std::mem::size_of::<RenderPassDescriptor>() as u64;

    let mask_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

    let mut attributes = VertexBuilder::new(wgpu::VertexStepMode::Instance);
    attributes.push(wgpu::VertexFormat::Uint32x2);
    attributes.push(wgpu::VertexFormat::Uint32);
    attributes.push(wgpu::VertexFormat::Uint32);

    let mut fill_features = Vec::new();
    if edges.texture().is_some() {
        fill_features.push("EDGE_TEXTURE");
    }
    if use_ssaa4 {
        fill_features.push("FILL_SSAA4");
    }

    let fill_module = shaders.create_shader_module(device, "Mask fill", fill_src, &fill_features);

    let targets = &[shaders.defaults.alpha_target_state()];
    let fill_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
        label: Some("Fill mask"),
        layout: Some(&tile_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &fill_module,
            entry_point: "vs_main",
            buffers: &[attributes.buffer_layout()],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &fill_module,
            entry_point: "fs_main",
            targets,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: PipelineDefaults::primitive_state(),
        depth_stencil: None,
        multiview: None,
        multisample: wgpu::MultisampleState::default(),
        cache: None,
    };

    let fill_pipeline = device.create_render_pipeline(&fill_pipeline_descriptor);

    let mut attributes = VertexBuilder::new(wgpu::VertexStepMode::Instance);
    attributes.push(wgpu::VertexFormat::Uint32);
    attributes.push(wgpu::VertexFormat::Float32);
    attributes.push(wgpu::VertexFormat::Float32x2);

    let alpha_target = &[shaders.defaults.alpha_target_state()];
    let circle_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
        label: Some("Circle mask"),
        layout: Some(&tile_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &circle_module,
            entry_point: "vs_main",
            buffers: &[attributes.buffer_layout()],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &circle_module,
            entry_point: "fs_main",
            targets: alpha_target,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: PipelineDefaults::primitive_state(),
        depth_stencil: None,
        multiview: None,
        multisample: wgpu::MultisampleState::default(),
        cache: None,
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
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &rect_module,
            entry_point: "fs_main",
            targets: alpha_target,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: PipelineDefaults::primitive_state(),
        depth_stencil: None,
        multiview: None,
        multisample: wgpu::MultisampleState::default(),
        cache: None,
    };

    let rect_pipeline = device.create_render_pipeline(&rect_pipeline_descriptor);

    Masks {
        fill_pipeline,
        circle_pipeline,
        rect_pipeline,
        mask_bind_group_layout,
    }
}
