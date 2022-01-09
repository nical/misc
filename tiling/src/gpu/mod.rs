use lyon::geom::Vector;
pub mod advanced_tiles;
pub mod masked_tiles;
pub mod solid_tiles;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct GpuGlobals {
    pub resolution: Vector<f32>,
    pub tile_size: u32,
    pub tile_atlas_size: u32,
}

unsafe impl bytemuck::Pod for GpuGlobals {}
unsafe impl bytemuck::Zeroable for GpuGlobals {}

impl GpuGlobals {
    pub fn create_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Globals"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<GpuGlobals>() as u64),
                    },
                    count: None,
                },
            ],
        })
    }
}
