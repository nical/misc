pub mod buffer;
pub mod context;
pub mod gpu;
//pub mod flatten_simd;
pub mod batching;
pub mod cache;
pub mod path;
pub mod pattern;
pub mod resources;
pub mod shape;
pub mod stroke;
pub mod transform;

pub use lyon::path::math::{point, vector, Point, Vector};

pub use bitflags;
pub use bytemuck;
pub use lyon::geom;
use pattern::BindingsId;
pub use wgpu;

pub mod units {
    use lyon::geom::euclid::{self, Box2D, Point2D, Size2D, Vector2D};

    #[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
    pub struct LocalSpace;

    #[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
    pub struct SurfaceSpace;

    pub type Rect = euclid::default::Box2D<f32>;
    pub type Point = euclid::default::Point2D<f32>;
    pub type Vector = euclid::default::Vector2D<f32>;
    pub type Size = euclid::default::Size2D<f32>;

    pub type SurfaceRect = Box2D<f32, SurfaceSpace>;
    pub type SurfacePoint = Point2D<f32, SurfaceSpace>;
    pub type SurfaceVector = Vector2D<f32, SurfaceSpace>;
    pub type SurfaceSize = Size2D<f32, SurfaceSpace>;
    pub type SurfaceIntSize = Size2D<i32, SurfaceSpace>;
    pub type SurfaceIntRect = Box2D<i32, SurfaceSpace>;

    pub type LocalRect = Box2D<f32, LocalSpace>;
    pub type LocalPoint = Point2D<f32, LocalSpace>;
    pub type LocalVector = Vector2D<f32, LocalSpace>;
    pub type LocalSize = Size2D<f32, LocalSpace>;

    pub type LocalToSurfaceTransform = euclid::Transform2D<f32, LocalSpace, SurfaceSpace>;
    pub type LocalTransform = euclid::Transform2D<f32, LocalSpace, LocalSpace>;

    pub use euclid::point2 as point;
    pub use euclid::vec2 as vector;
}

pub use crate::context::{
    SurfaceDrawConfig, SurfacePassConfig, SurfaceKind, StencilMode, DepthMode,
};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Color {
    pub const RED: Self = Color {
        r: 255,
        g: 0,
        b: 0,
        a: 255,
    };
    pub const GREEN: Self = Color {
        r: 0,
        g: 255,
        b: 0,
        a: 255,
    };
    pub const BLUE: Self = Color {
        r: 0,
        g: 0,
        b: 255,
        a: 255,
    };
    pub const BLACK: Self = Color {
        r: 0,
        g: 0,
        b: 0,
        a: 255,
    };
    pub const WHITE: Self = Color {
        r: 255,
        g: 255,
        b: 255,
        a: 255,
    };

    pub fn is_opaque(self) -> bool {
        self.a == 255
    }

    pub fn to_u32(self) -> u32 {
        (self.r as u32) << 24 | (self.g as u32) << 16 | (self.b as u32) << 8 | self.a as u32
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

    pub fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Color { r, g, b, a }
    }

    pub fn linear_to_srgb(r: u8, g: u8, b: u8, a: u8) -> Self {
        fn f(linear: f32) -> f32 {
            if linear <= 0.0031308 {
                linear * 12.92
            } else {
                1.055 * linear.powf(1.0 / 2.4) - 0.055
            }
        }
        let r = (f(r as f32 / 255.0) * 255.0) as u8;
        let g = (f(g as f32 / 255.0) * 255.0) as u8;
        let b = (f(b as f32 / 255.0) * 255.0) as u8;

        Color { r, g, b, a }
    }

    pub fn srgb_to_linear(r: u8, g: u8, b: u8, a: u8) -> Self {
        fn f(srgb: f32) -> f32 {
            if srgb <= 0.04045 {
                srgb * 12.92
            } else {
                ((srgb + 0.055) / 1.055).powf(2.4)
            }
        }
        let r = (f(r as f32 / 255.0) * 255.0) as u8;
        let g = (f(g as f32 / 255.0) * 255.0) as u8;
        let b = (f(b as f32 / 255.0) * 255.0) as u8;

        Color { r, g, b, a }
    }
}

use std::ops::Range;
#[inline]
pub fn u32_range(r: Range<usize>) -> Range<u32> {
    r.start as u32..r.end as u32
}

#[inline]
pub fn usize_range(r: Range<u32>) -> Range<usize> {
    r.start as usize..r.end as usize
}

pub trait BindingResolver {
    fn resolve(&self, id: BindingsId) -> Option<&wgpu::BindGroup>;
}

impl BindingResolver for () {
    fn resolve(&self, _: BindingsId) -> Option<&wgpu::BindGroup> {
        None
    }
}
