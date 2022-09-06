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
pub mod checkerboard_pattern;
pub mod custom_pattern;
pub mod simple_gradient;
pub mod solid_color;
pub mod gpu_store;

pub use tiler::*;
pub use occlusion::*;

pub use lyon::path::math::{Point, point, Vector, vector};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Color {
    pub const RED: Self = Color { r: 255, g: 0, b: 0, a: 255 };
    pub const GREEN: Self = Color { r: 0, g: 255, b: 0, a: 255 };
    pub const BLUE: Self = Color { r: 0, g: 0, b: 255, a: 255 };
    pub const BLACK: Self = Color { r: 0, g: 0, b: 0, a: 255 };
    pub const WHITE: Self = Color { r: 255, g: 255, b: 255, a: 255 };

    pub fn is_opaque(&self) -> bool {
        self.a == 255
    }

    pub fn to_u32(&self) -> u32 {
        (self.r as u32) << 24
        | (self.g as u32) << 16
        | (self.b as u32) << 8
        | self.a as u32
    }

    pub fn to_f32(&self) -> [f32; 4] {
        [
            self.r as f32 / 255.0,
            self.g as f32 / 255.0,
            self.b as f32 / 255.0,
            self.a as f32 / 255.0,
        ]
    }

    pub fn linear_to_srgb(r: u8, g: u8, b: u8, a:u8) -> Self {
        fn f(linear: f32) -> f32 {
            if linear <= 0.0031308 { linear * 12.92}  else { 1.055 * linear.powf(1.0 / 2.4) - 0.055 }
        }
        let r = (f(r as f32 / 255.0) * 255.0) as u8;
        let g = (f(g as f32 / 255.0) * 255.0) as u8;
        let b = (f(b as f32 / 255.0) * 255.0) as u8;

        Color { r, g, b, a }
    }

    pub fn srgb_to_linear(r: u8, g: u8, b: u8, a:u8) -> Self {
        fn f(srgb: f32) -> f32 {
            if srgb <= 0.04045 { srgb * 12.92}  else { ((srgb + 0.055) / 1.055).powf(2.4) }
        }
        let r = (f(r as f32 / 255.0) * 255.0) as u8;
        let g = (f(g as f32 / 255.0) * 255.0) as u8;
        let b = (f(b as f32 / 255.0) * 255.0) as u8;

        Color { r, g, b, a }
    }
}
