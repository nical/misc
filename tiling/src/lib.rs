pub mod canvas;
pub mod tiler;
pub mod tile_encoder;
pub mod advanced_raster_encoder;
pub mod cpu_rasterizer;
pub mod occlusion;
pub mod load_svg;
pub mod gpu;
pub mod buffer;
//pub mod flatten_simd;
pub mod tile_renderer;

pub use tiler::*;
pub use occlusion::*;

type TexelRect = euclid::default::Box2D<u16>;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Color {
    pub fn is_opaque(&self) -> bool {
        self.a == 255
    }

    pub fn to_u32(&self) -> u32 {
        (self.r as u32) << 24
        | (self.g as u32) << 16
        | (self.b as u32) << 8
        | self.a as u32
    }
}
