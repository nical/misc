pub mod tiler;
pub mod gpu_raster_encoder;
pub mod advanced_raster_encoder;
pub mod cpu_rasterizer;
pub mod z_buffer;
pub mod load_svg;
pub mod gpu;
pub mod buffer;
pub mod flatten_simd;

pub use tiler::*;
pub use z_buffer::*;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Color {
    pub fn to_u32(&self) -> u32 {
        (self.r as u32) << 24
        | (self.g as u32) << 16
        | (self.b as u32) << 8
        | self.a as u32
    }
}
