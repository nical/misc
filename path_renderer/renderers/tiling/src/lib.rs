pub mod mask;
pub mod tiler;
//pub mod tiler2;
pub mod atlas_uploader;
pub mod cpu_rasterizer;
pub mod encoder;
pub mod occlusion;
pub mod renderer;
pub mod resources;

use core::pattern::BuiltPattern;

pub use occlusion::*;
pub use renderer::*;
pub use resources::*;
pub use tiler::*;

pub type PatternData = u32;
pub type AtlasIndex = u32;

// To experiment with other tile sizes, also change the corresponding
// constants in shaders/lib/tiling.wgsl.
pub const TILE_SIZE: u32 = 16;
pub const TILE_SIZE_F32: f32 = TILE_SIZE as f32;
pub const BYTES_PER_MASK: usize = (TILE_SIZE * TILE_SIZE) as usize;
pub const BYTES_PER_RGBA_TILE: usize = (TILE_SIZE * TILE_SIZE) as usize * 4;

/*

When rendering the tiger at 1800x1800 px, according to renderdoc on Intel UHD Graphics 620 (KBL GT2):
 - rasterizing the masks takes ~2ms
 - rendering into the color target takes ~0.8ms
  - ~0.28ms opaque tiles
  - ~0.48ms alpha tiles

*/

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

pub fn tile_visibility(pat: &BuiltPattern) -> TileVisibility {
    if pat.is_opaque {
        TileVisibility::Opaque
    } else {
        TileVisibility::Alpha
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TileVisibility {
    Opaque,
    Alpha,
    Empty,
}

impl TileVisibility {
    pub fn is_empty(self) -> bool {
        self == TileVisibility::Empty
    }
    pub fn is_opaque(self) -> bool {
        self == TileVisibility::Opaque
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct Stats {
    pub opaque_tiles: usize,
    pub alpha_tiles: usize,
    pub prerendered_tiles: usize,
    pub gpu_mask_tiles: usize,
    pub cpu_mask_tiles: usize,
    pub edges: usize,
    pub render_passes: usize,
    pub batches: usize,
}

impl Stats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        *self = Stats::new();
    }

    pub fn tiles_bytes(&self) -> usize {
        (self.opaque_tiles + self.gpu_mask_tiles) * std::mem::size_of::<TileInstance>()
            + (self.alpha_tiles + self.prerendered_tiles) * std::mem::size_of::<TileInstance>()
    }

    pub fn edges_bytes(&self) -> usize {
        self.edges * std::mem::size_of::<TileInstance>()
    }

    pub fn cpu_masks_bytes(&self) -> usize {
        self.cpu_mask_tiles * 16 * 16 // TODO other tile sizes
    }

    pub fn uploaded_bytes(&self) -> usize {
        self.tiles_bytes() + self.edges_bytes() + self.cpu_masks_bytes()
    }
}

pub struct FillOptions<'l> {
    pub fill_rule: FillRule,
    pub inverted: bool,
    pub tolerance: f32,
    pub merge_tiles: bool,
    pub prerender_pattern: bool,
    pub transform: Option<&'l Transform2D<f32>>,
}

impl<'l> FillOptions<'l> {
    pub fn new() -> FillOptions<'static> {
        FillOptions {
            fill_rule: FillRule::EvenOdd,
            inverted: false,
            tolerance: 0.1,
            merge_tiles: true,
            prerender_pattern: false,
            transform: None,
        }
    }

    pub fn transformed<'a>(transform: &'a Transform2D<f32>) -> FillOptions<'a> {
        FillOptions {
            fill_rule: FillRule::EvenOdd,
            inverted: false,
            tolerance: 0.1,
            merge_tiles: true,
            prerender_pattern: false,
            transform: Some(transform),
        }
    }

    pub fn with_transform<'a>(self, transform: Option<&'a Transform2D<f32>>) -> FillOptions<'a>
    where
        'l: 'a,
    {
        FillOptions {
            fill_rule: self.fill_rule,
            inverted: false,
            tolerance: self.tolerance,
            merge_tiles: self.merge_tiles,
            prerender_pattern: self.prerender_pattern,
            transform,
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

    pub fn with_merged_tiles(mut self, merge_tiles: bool) -> Self {
        self.merge_tiles = merge_tiles;
        self
    }

    pub fn with_prerendered_pattern(mut self, prerender: bool) -> Self {
        self.prerender_pattern = prerender;
        self
    }

    pub fn inverted(mut self) -> Self {
        self.inverted = true;
        self
    }

    pub fn with_inverted(mut self, inverted: bool) -> Self {
        self.inverted = inverted;
        self
    }
}
