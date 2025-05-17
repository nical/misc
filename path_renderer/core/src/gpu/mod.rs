pub mod gpu_store;
pub mod stream;
pub mod storage_buffer;
pub mod staging_buffers;

pub use gpu_store::*;
pub use stream::*;
pub use staging_buffers::*;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct RenderPassDescriptor {
    pub width: f32,
    pub height: f32,
    pub inv_width: f32,
    pub inv_height: f32,
}

impl RenderPassDescriptor {
    pub fn new(w: u32, h: u32) -> Self {
        let width = w as f32;
        let height = h as f32;
        let inv_width = 1.0 / width;
        let inv_height = 1.0 / height;
        RenderPassDescriptor {
            width,
            height,
            inv_width,
            inv_height,
        }
    }
}

unsafe impl bytemuck::Pod for RenderPassDescriptor {}
unsafe impl bytemuck::Zeroable for RenderPassDescriptor {}
