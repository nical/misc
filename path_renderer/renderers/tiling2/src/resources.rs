use core::gpu::PipelineDefaults;
use core::gpu::shader::{BaseShaderId, VertexAtribute, BaseShaderDescriptor, Binding, BindGroupLayout, Varying};
//use core::gpu::PipelineDefaults;
use core::wgpu;
use core::{
    gpu::shader::Shaders,
    resources::RendererResources,
};

pub struct TileGpuResources {
    pub(crate) base_shader: BaseShaderId,
    pub(crate) edge_texture: wgpu::Texture,
    pub(crate) path_texture: wgpu::Texture,
    pub(crate) bind_group: wgpu::BindGroup,
}

impl TileGpuResources {
    pub fn new(device: &wgpu::Device, shaders: &mut Shaders) -> Self {

        let bind_group_layout = shaders.register_bind_group_layout(BindGroupLayout::new(
            device,
            "tiling2::layout".into(),
            vec![
                Binding {
                    name: "path_texture".into(),
                    struct_type: "u32".into(),
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                },
                Binding {
                    name: "edge_texture".into(),
                    struct_type: "f32".into(),
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                }
            ],
        ));

        let base_shader = shaders.register_base_shader(BaseShaderDescriptor {
            name: "geometry::tile2".into(),
            source: include_str!("../shaders/tile.wgsl").into(),
            vertex_attributes: Vec::new(),
            instance_attributes: vec![VertexAtribute::uint32x4("instance")],
            varyings: vec![
                Varying::float32x2("uv"),
                Varying::uint32x2("edges"),
                Varying::sint32("backdrop"),
                Varying::uint32("fill_rule"),
                Varying::float32("opacity"),
            ],
            bindings: Some(bind_group_layout),
            primitive: PipelineDefaults::primitive_state(),
            shader_defines: Vec::new(),
        });

        let default_edge_texture_size = 1024;
        let default_path_texture_size = 512;

        let edge_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tiling2::edges"),
            dimension: wgpu::TextureDimension::D2,
            size: wgpu::Extent3d {
                width: default_edge_texture_size,
                height: default_edge_texture_size,
                depth_or_array_layers: 1,
            },
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            mip_level_count: 1,
            sample_count: 1,
            view_formats: &[],
        });
        let edge_texture_view = edge_texture.create_view(&wgpu::TextureViewDescriptor::default());


        let path_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tiling2::paths"),
            dimension: wgpu::TextureDimension::D2,
            size: wgpu::Extent3d {
                width: default_path_texture_size,
                height: default_path_texture_size,
                depth_or_array_layers: 1,
            },
            format: wgpu::TextureFormat::Rgba32Uint,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            mip_level_count: 1,
            sample_count: 1,
            view_formats: &[],
        });
        let path_texture_view = path_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tiling2::geom"),
            layout: &shaders.get_bind_group_layout(bind_group_layout).handle,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&path_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&edge_texture_view),
                },
            ],
        });

        TileGpuResources {
            base_shader,
            edge_texture,
            path_texture,
            bind_group,
        }
    }
}

impl RendererResources for TileGpuResources {
    fn name(&self) -> &'static str {
        "TileGpuResources2"
    }

    fn begin_frame(&mut self) {}

    fn begin_rendering(&mut self, _encoder: &mut wgpu::CommandEncoder) {}

    fn end_frame(&mut self) {}
}
