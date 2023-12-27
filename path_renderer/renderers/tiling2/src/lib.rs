mod renderer;
mod resources;
pub mod tiler;
pub mod occlusion;
mod flatten;

use core::units::{LocalSpace, SurfaceSpace};

pub use renderer::*;
pub use resources::*;

pub use lyon::lyon_tessellation::FillRule;

pub type Transform = lyon::geom::euclid::Transform2D<f32, LocalSpace, SurfaceSpace>;


pub struct FillOptions<'l> {
    pub fill_rule: FillRule,
    pub inverted: bool,
    pub z_index: u32,
    pub tolerance: f32,
    pub transform: Option<&'l Transform>,
    pub opacity: f32,
}

impl<'l> FillOptions<'l> {
    pub fn new() -> FillOptions<'static> {
        FillOptions {
            fill_rule: FillRule::EvenOdd,
            inverted: false,
            z_index: 0,
            tolerance: 0.25,
            transform: None,
            opacity: 1.0,
        }
    }

    pub fn transformed<'a>(transform: &'a Transform) -> FillOptions<'a> {
        FillOptions {
            fill_rule: FillRule::EvenOdd,
            inverted: false,
            z_index: 0,
            tolerance: 0.25,
            transform: Some(transform),
            opacity: 1.0,
        }
    }

    pub fn with_transform<'a>(self, transform: Option<&'a Transform>) -> FillOptions<'a>
    where
        'l: 'a,
    {
        FillOptions {
            fill_rule: self.fill_rule,
            inverted: false,
            z_index: self.z_index,
            tolerance: self.tolerance,
            transform,
            opacity: self.opacity,
        }
    }

    pub fn with_fill_rule(mut self, fill_rule: FillRule) -> Self {
        self.fill_rule = fill_rule;
        self
    }

    pub fn with_tolerance(mut self, tolerance: f32) -> Self {
        self.tolerance = tolerance;
        self
    }

    pub fn with_inverted(mut self, inverted: bool) -> Self {
        self.inverted = inverted;
        self
    }

    pub fn with_z_index(mut self, z_index: u32) -> Self {
        self.z_index = z_index;
        self
    }

    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity;
        self
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TilePosition(u32);

impl TilePosition {
    const MASK: u32 = 0x3FF;
    pub const ZERO: Self = TilePosition(0);
    pub const INVALID: Self = TilePosition(std::u32::MAX);

    pub fn extended(x: u32, y: u32, extend: u32) -> Self {
        debug_assert!(x <= Self::MASK);
        debug_assert!(y <= Self::MASK);
        debug_assert!(extend <= Self::MASK);

        TilePosition(extend << 20 | x << 10 | y)
    }

    pub fn new(x: u32, y: u32) -> Self {
        debug_assert!(x <= Self::MASK);
        debug_assert!(y <= Self::MASK);

        TilePosition(x << 10 | y)
    }

    pub fn extend(&mut self) {
        self.0 += 1 << 20;
    }

    pub fn with_flag(mut self) -> Self {
        self.add_flag();
        self
    }
    pub fn to_u32(&self) -> u32 {
        self.0
    }
    pub fn x(&self) -> u32 {
        (self.0 >> 10) & Self::MASK
    }
    pub fn y(&self) -> u32 {
        (self.0) & Self::MASK
    }
    pub fn extension(&self) -> u32 {
        (self.0 >> 20) & Self::MASK
    }

    // TODO: we have two unused bits and we use one of them to store
    // whether a tile in an indirection buffer is opaque. That's not
    // great.
    pub fn flag(&self) -> bool {
        self.0 & 1 << 31 != 0
    }
    pub fn add_flag(&mut self) {
        self.0 |= 1 << 31
    }
}

#[test]
fn tile_position() {
    let mut p0 = TilePosition::new(1, 2);
    assert_eq!(p0.x(), 1);
    assert_eq!(p0.y(), 2);
    assert_eq!(p0.extension(), 0);

    p0.extend();

    assert_eq!(p0.x(), 1);
    assert_eq!(p0.y(), 2);
    assert_eq!(p0.extension(), 1);

    p0.extend();

    assert_eq!(p0.x(), 1);
    assert_eq!(p0.y(), 2);
    assert_eq!(p0.extension(), 2);
}

