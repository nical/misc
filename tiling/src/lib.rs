pub mod canvas;
pub mod load_svg;
pub mod gpu;
pub mod buffer;
//pub mod flatten_simd;
pub mod pattern;
pub mod custom_pattern;
pub mod tiling;
pub mod tess;

pub use tiling::tiler::*;
pub use tiling::occlusion::*;

pub use lyon::path::math::{Point, point, Vector, vector};

// To experiment with other tile sizes, also change the corresponding
// constants in shaders/lib/tiling.wgsl.
pub const TILE_SIZE: u32 = 16;
pub const TILE_SIZE_F32: f32 = TILE_SIZE as f32;
pub const BYTES_PER_MASK: usize = (TILE_SIZE * TILE_SIZE) as usize;
pub const BYTES_PER_RGBA_TILE: usize = (TILE_SIZE * TILE_SIZE) as usize * 4;


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

    pub fn is_opaque(self) -> bool {
        self.a == 255
    }

    pub fn to_u32(self) -> u32 {
        (self.r as u32) << 24
        | (self.g as u32) << 16
        | (self.b as u32) << 8
        | self.a as u32
    }

    pub fn to_f32(self) -> [f32; 4] {
        [
            self.r as f32 / 255.0,
            self.g as f32 / 255.0,
            self.b as f32 / 255.0,
            self.a as f32 / 255.0,
        ]
    }

    pub fn to_wgpu(self) -> wgpu::Color {
        wgpu::Color {
            r: self.r as f64 / 255.0,
            g: self.g as f64 / 255.0,
            b: self.b as f64 / 255.0,
            a: self.a as f64 / 255.0,
        }
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

use std::ops::Range;
#[inline]
pub(crate) fn u32_range(r: Range<usize>) -> Range<u32> {
    r.start as u32 .. r.end as u32
}

#[inline]
pub(crate) fn usize_range(r: Range<u32>) -> Range<usize> {
    r.start as usize .. r.end as usize
}
