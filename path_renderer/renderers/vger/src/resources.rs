use core::batching::RendererId;
use core::bytemuck;
use core::gpu::{GpuBufferDescriptor, GpuBufferResources};
use core::shading::{
    BindGroupLayout, BindGroupLayoutId, Binding, GeometryDescriptor, GeometryId,
    PipelineDefaults, Shaders, Varying, VertexAtribute,
};
use core::units::LocalRect;
use core::wgpu;

use crate::VgerRenderer;

/// Per-instance data sent to the GPU.
///
/// Each instance corresponds to one horizontal band of a filled path
/// (or a simple rectangle).
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Instance {
    /// Bounding rectangle of the band in surface space.
    pub local_rect: LocalRect,
    pub z_index: u32,
    pub pattern: u32,
    pub render_task: u32,
    /// Index of the first control vertex (in units of vec2<f32>) in the CV buffer.
    pub curve_start: u32,
    /// Number of quadratic segments in this band.
    pub curve_count: u32,
    /// Fill rule: 0 = even-odd, 1 = non-zero.
    pub fill_rule: u32,
    pub _padding: [u32; 2],
}

unsafe impl bytemuck::Zeroable for Instance {}
unsafe impl bytemuck::Pod for Instance {}

pub struct Vger {
    pub(crate) geometry: GeometryId,
    pub(crate) bind_group_layout: BindGroupLayoutId,
}

impl Vger {
    pub fn new(device: &wgpu::Device, shaders: &mut Shaders) -> Self {
        let bind_group_layout = shaders.register_bind_group_layout(BindGroupLayout::new(
            device,
            "vger::curves_layout".into(),
            vec![Binding::storage_buffer("curves", "array<vec2<f32>>")],
        ));

        let geometry = shaders.register_geometry(GeometryDescriptor {
            name: "geometry::vger".into(),
            source: VGER_SRC.into(),
            vertex_attributes: Vec::new(),
            instance_attributes: vec![
                VertexAtribute::float32x4("rect"),
                VertexAtribute::uint32("z_index"),
                VertexAtribute::uint32("pattern"),
                VertexAtribute::uint32("render_task"),
                VertexAtribute::uint32("curve_start"),
                VertexAtribute::uint32("curve_count"),
                VertexAtribute::uint32("fill_rule"),
                VertexAtribute::uint32x2("padding"),
            ],
            varyings: vec![
                Varying::float32x2("local_pos"),
                Varying::uint32("seg_start"),
                Varying::uint32("seg_count"),
                Varying::uint32("winding_rule"),
            ],
            bindings: Some(bind_group_layout),
            primitive: PipelineDefaults::primitive_state(),
            shader_defines: Vec::new(),
            constants: Vec::new(),
        });

        Vger {
            geometry,
            bind_group_layout,
        }
    }

    pub fn new_renderer(
        &self,
        device: &wgpu::Device,
        shaders: &Shaders,
        renderer_id: RendererId,
    ) -> VgerRenderer {
        let curve_desc = GpuBufferDescriptor::Buffers {
            usages: wgpu::BufferUsages::STORAGE,
            default_alignment: 8, // vec2<f32> = 8 bytes
            min_size: 1024 * 8,
            label: Some("vger::curves"),
        };
        let mut curve_store = GpuBufferResources::new(&curve_desc);
        curve_store.allocate(1024 * 8, device);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vger::curves"),
            layout: &shaders.get_bind_group_layout(self.bind_group_layout).handle,
            entries: &[curve_store.as_bind_group_entry(0).unwrap()],
        });

        VgerRenderer::new(
            renderer_id,
            self.geometry,
            self.bind_group_layout,
            curve_store,
            bind_group,
        )
    }
}

const VGER_SRC: &str = include_str!("../shaders/vger.wgsl");
