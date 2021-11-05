use lyon::geom::Vector;
pub mod advanced_tiles;
pub mod masked_tiles;
pub mod solid_tiles;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct GpuGlobals {
    pub resolution: Vector<f32>,
}

unsafe impl bytemuck::Pod for GpuGlobals {}
unsafe impl bytemuck::Zeroable for GpuGlobals {}
