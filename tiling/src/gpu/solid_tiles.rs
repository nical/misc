use lyon::geom::Box2D;

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
}

impl SolidTiles {
    pub fn new(device: &wgpu::Device, globals_bind_group_layout: &wgpu::BindGroupLayout) -> Self {
        create_pipeline(device, globals_bind_group_layout)
    }
}

fn create_pipeline(device: &wgpu::Device, globals_bind_group_layout: &wgpu::BindGroupLayout) -> SolidTiles {
    let vs_module = &device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: Some("Solid tiles vs"),
        source: wgpu::ShaderSource::Wgsl(include_str!("./../../shaders/solid_tile.vs.wgsl").into()),
    });
    let fs_module = &device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: Some("Solid tiles fs"),
        source: wgpu::ShaderSource::Wgsl(include_str!("./../../shaders/solid_tile.fs.wgsl").into()),
    });

    let tile_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Solid tiles"),
        bind_group_layouts: &[&globals_bind_group_layout],
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

    let pipeline = device.create_render_pipeline(&tile_pipeline_descriptor);    

    SolidTiles {
        pipeline,
    }
}
