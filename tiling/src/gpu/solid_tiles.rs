use lyon::geom::Box2D;
use super::GpuGlobals;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct TileInstance {
    pub rect: Box2D<f32>,
    pub color: u32,
}

unsafe impl bytemuck::Pod for TileInstance {}
unsafe impl bytemuck::Zeroable for TileInstance {}

pub struct SolidTiles {
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl SolidTiles {
    pub fn new(device: &wgpu::Device) -> Self {
        create_pipeline(device)
    }
}

fn create_pipeline(device: &wgpu::Device) -> SolidTiles {
    let vs_module = &device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: Some("Solid tiles vs"),
        source: wgpu::ShaderSource::Wgsl(include_str!("./../../shaders/solid_tile.vs.wgsl").into()),
    });
    let fs_module = &device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: Some("Solid tiles fs"),
        source: wgpu::ShaderSource::Wgsl(include_str!("./../../shaders/solid_tile.fs.wgsl").into()),
    });

    let globals_buffer_byte_size = std::mem::size_of::<GpuGlobals>() as u64;

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Solid tiles"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(globals_buffer_byte_size),
                },
                count: None,
            },
        ],
    });

    let tile_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Solid tiles"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let tile_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
        label: Some("Solid tiles"),
        layout: Some(&tile_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &vs_module,
            entry_point: "main",
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<TileInstance>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[
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
                ],
            }],
        },
        fragment: Some(wgpu::FragmentState {
            module: &fs_module,
            entry_point: "main",
            targets: &[
                wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Bgra8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                },
            ],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            polygon_mode: wgpu::PolygonMode::Fill,
            front_face: wgpu::FrontFace::Ccw,
            strip_index_format: None,
            cull_mode: None,
            clamp_depth: false,
            conservative: false,
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
    };

    let pipeline = device.create_render_pipeline(&tile_pipeline_descriptor);    

    SolidTiles {
        pipeline,
        bind_group_layout,
    }
}
