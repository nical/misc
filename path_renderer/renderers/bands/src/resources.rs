use core::batching::RendererId;
use core::bytemuck;
use core::gpu::{GpuBufferDescriptor, GpuBufferResources};
use core::shading::{
    BindGroupLayout, BindGroupLayoutId, Binding, GeometryDescriptor, GeometryId,
    PipelineDefaults, Shaders, Varying, VertexAtribute,
};
use core::units::*;
use core::wgpu;

use crate::{AaMode, BandsOptions, BandsRenderer};

/// Per-instance data sent to the GPU.
///
/// Each instance corresponds to one horizontal band of a filled path
/// (or a simple rectangle).
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Instance {
    /// Bounding rectangle of the band in surface space.
    pub rect: SurfaceRect,
    pub z_index: u32,
    pub pattern: u32,
    pub render_task: u32,
    /// Index of the first control vertex (in units of vec2<f32>) in the CV buffer.
    pub curve_start: u32,
    /// Number of quadratic segments in this band.
    pub curve_count: u32,
    /// Fill rule: 0 = even-odd, 1 = non-zero.
    pub fill_rule: u32,
    /// Offset of the first edge in the path. The edge indices are
    /// expressed relative to this offset.
    pub path_start: u32,
    pub _padding: u32,
}

unsafe impl bytemuck::Zeroable for Instance {}
unsafe impl bytemuck::Pod for Instance {}

pub struct Bands {
    pub(crate) geometry: GeometryId,
    pub(crate) bind_group_layout: BindGroupLayoutId,
}

impl Bands {
    pub fn new(device: &wgpu::Device, shaders: &mut Shaders, descriptor: &BandsOptions) -> Self {
        let bind_group_layout = shaders.register_bind_group_layout(BindGroupLayout::new(
            device,
            "bands::curvesTilingOptions_layout".into(),
            vec![
                Binding {
                    name: "curves".into(),
                    struct_type: "vec4f".into(),
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                },
                Binding {
                    name: "curve_indices".into(),
                    struct_type: "u16".into(),
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                }
            ],
        ));

        let mut shader_defines = Vec::new();
        match descriptor.antialiasing {
            AaMode::Ssaa4 => {
                shader_defines.push("BANDS_SSAA4");
            }
            AaMode::AreaCoverage => {}
        }

        let geometry = shaders.register_geometry(GeometryDescriptor {
            name: "geometry::bands".into(),
            source: BANDS_SRC.into(),
            vertex_attributes: Vec::new(),
            instance_attributes: vec![
                VertexAtribute::float32x4("rect"),
                VertexAtribute::uint32("z_index"),
                VertexAtribute::uint32("pattern"),
                VertexAtribute::uint32("render_task"),
                VertexAtribute::uint32("curve_start"),
                VertexAtribute::uint32("curve_count"),
                VertexAtribute::uint32("fill_rule"),
                VertexAtribute::uint32("path_start"),
                VertexAtribute::uint32("padding"),
            ],
            varyings: vec![
                Varying::float32x2("local_pos"),
                Varying::uint32x4("quad_data"),
            ],
            bindings: Some(bind_group_layout),
            primitive: PipelineDefaults::primitive_state(),
            shader_defines,
            constants: Vec::new(),
        });

        Bands {
            geometry,
            bind_group_layout,
        }
    }

    pub fn new_renderer(
        &self,
        device: &wgpu::Device,
        shaders: &Shaders,
        renderer_id: RendererId,
    ) -> BandsRenderer {
        const DATA_TEXTURE_WIDTH:u32 = 4096;

        let curve_desc = GpuBufferDescriptor::Texture {
            format: wgpu::TextureFormat::Rgba32Float,
            alignment: 16,
            width: DATA_TEXTURE_WIDTH,
            label: Some("bands::curves"),
        };
        let mut curve_store = GpuBufferResources::new(&curve_desc);
        curve_store.allocate(1024 * 8, device);

        let index_desc = GpuBufferDescriptor::Texture {
            format: wgpu::TextureFormat::R16Uint,
            alignment: 4,
            width: DATA_TEXTURE_WIDTH,
            label: Some("bands::indices"),
        };
        let mut index_store = GpuBufferResources::new(&index_desc);
        index_store.allocate(1024 * 8, device);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bands"),
            layout: &shaders.get_bind_group_layout(self.bind_group_layout).handle,
            entries: &[
                curve_store.as_bind_group_entry(0).unwrap(),
                index_store.as_bind_group_entry(1).unwrap(),
            ],
        });

        BandsRenderer::new(
            renderer_id,
            self.geometry,
            self.bind_group_layout,
            curve_store,
            index_store,
            bind_group,
        )
    }
}

const BANDS_SRC: &str = include_str!("../shaders/bands.wgsl");
