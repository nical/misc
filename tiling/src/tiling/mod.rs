pub mod tiler;
pub mod encoder;
pub mod occlusion;
pub mod tile_renderer;
pub mod cpu_rasterizer;

pub use tiler::*;
pub use tile_renderer::*;
pub use occlusion::*;

use std::ops::Range;

use lyon::geom::euclid::default::{Transform2D, Size2D};
use lyon::path::FillRule;

pub type PatternData = u32;
pub type AtlasIndex = u32;
pub type PatternIndex = usize;
pub const TILED_IMAGE_PATTERN: PatternIndex = 10000; // TODO


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
    pub fn to_u32(&self) -> u32 { self.0 }
    pub fn x(&self) -> u32 { (self.0 >> 10) & Self::MASK }
    pub fn y(&self) -> u32 { (self.0) & Self::MASK }
    pub fn extension(&self) -> u32 { (self.0 >> 20) & Self::MASK }

    // TODO: we have two unused bits and we use one of them to store
    // whether a tile in an indirection buffer is opaque. That's not
    // great.
    pub fn flag(&self) -> bool { self.0 & 1 << 31 != 0 }
    pub fn add_flag(&mut self) { self.0 |= 1 << 31 }
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

pub struct TileAllocator {
    pub next_id: u32,
    pub current_atlas: AtlasIndex,
    pub tiles_per_row: u32,
    pub tiles_per_atlas: u32,
}

impl TileAllocator {
    pub fn new(w: u32, h: u32) -> Self {
        TileAllocator {
            next_id: 1,
            current_atlas: 0,
            tiles_per_row: w,
            tiles_per_atlas: w * h,
        }
    }

    pub fn reset(&mut self) {
        self.next_id = 1;
        self.current_atlas = 0;
    }

    pub fn allocate(&mut self) -> (TilePosition, AtlasIndex) {
        let id = self.next_id;
        self.next_id += 1;

        let mut id2 = id % self.tiles_per_atlas;

        if id2 == id {
            // Common path.
            let pos = TilePosition::new(
                id % self.tiles_per_row,
                id / self.tiles_per_row,
            );
            return (pos, self.current_atlas)
        }

        if id2 == 0 {
            // Tile zero is reserved.
            id2 += 1;
        }

        self.next_id = id2 + 1;

        self.current_atlas += 1;
        let pos = TilePosition::new(
            id2 % self.tiles_per_row,
            id2 / self.tiles_per_row,
        );

        (pos, self.current_atlas)
    }

    pub fn finish_atlas(&mut self) {
        self.current_atlas += 1;
        self.next_id = 1;
    }

    pub fn width(&self) -> u32 { self.tiles_per_row }

    pub fn height(&self) -> u32 { self.tiles_per_atlas / self.tiles_per_row }

    pub fn current_atlas(&self) -> u32 { self.current_atlas }

    pub fn is_nearly_full(&self) -> bool {
        (self.next_id * 100) / self.tiles_per_atlas > 70
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct BufferRange(pub u32, pub u32);
impl BufferRange {
    pub fn start(&self) -> u32 { self.0 }
    pub fn is_empty(&self) -> bool { self.0 >= self.1 }
    pub fn to_u32(&self) -> Range<u32> { self.0 .. self.1 }
    pub fn byte_range<Ty>(&self) -> Range<u64> {
        let s = std::mem::size_of::<Ty>() as u64;
        self.0 as u64 * s .. self.1 as u64 * s
    }
    pub fn byte_offset<Ty>(&self) -> u64 {
        self.0 as u64 * std::mem::size_of::<Ty>() as u64
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TileVisibility {
    Opaque,
    Alpha,
    Empty,
}

impl TileVisibility {
    pub fn is_empty(self) -> bool { self == TileVisibility::Empty }
    pub fn is_opaque(self) -> bool { self == TileVisibility::Opaque }
}

// Note: the statefullness with set_tile(x, y) followed by tile-specific getters
// is unfortunate, the reason for it at the moment is the need to query whether
// a tile is empty or fully opaque before culling, while we want to only request
// tile if we really need to, that is after culling.
pub trait TilerPattern {
    fn index(&self) -> PatternIndex;

    fn is_entirely_opaque(&self) -> bool { false }
    fn tile_data(&mut self, x: u32, y: u32) -> PatternData;
    fn tile_visibility(&self, _x: u32, _y: u32) -> TileVisibility { TileVisibility::Alpha }
    fn can_stretch_horizontally(&self) -> bool { false }
}

pub struct TiledSourcePattern {
    indirection_buffer: IndirectionBuffer,
}

impl TiledSourcePattern {
    pub fn new(indirection_buffer: IndirectionBuffer) -> Self {
        TiledSourcePattern {
            indirection_buffer
        }
    }
}

impl TilerPattern for TiledSourcePattern {
    fn index(&self) -> PatternIndex { TILED_IMAGE_PATTERN }

    fn tile_visibility(&self, x: u32, y: u32) -> TileVisibility {
        match self.indirection_buffer.get(x, y) {
            Some((_, true)) => TileVisibility::Opaque,
            Some((_, false)) => TileVisibility::Alpha,
            None => TileVisibility::Empty,
        }
    }

    fn tile_data(&mut self, x: u32, y: u32) -> PatternData {
        self.indirection_buffer.get(x, y).unwrap().0.to_u32()
    }
}

pub struct TiledOutput {
    pub indirection_buffer: IndirectionBuffer,
    pub tile_allocator: TileAllocator,
}

pub fn get_output_tile(output: &mut Option<&mut TiledOutput>, x: u32, y: u32, opaque: bool) -> TilePosition {
    // Decide where to draw the tile.
    // Two configurations: Either we render in a plain target in which case the position
    // corresponds to the actual coordinates of the tile's content, or we are rendering
    // to a tiled intermediate target in which case the destination is linearly allocated.
    // The indirection buffer is used to determine whether an aallocation was already made
    // for this position.
    if let Some(output) = output {
        let tile = output.indirection_buffer.get_mut(x, y);
        if *tile == TilePosition::INVALID {
            *tile = output.tile_allocator.allocate().0;
        }
        if opaque {
            tile.add_flag()
        }
    
        return *tile
    }
    
    TilePosition::new(x, y)
}

pub struct IndirectionBuffer {
    data: Vec<TilePosition>,
    size: Size2D<u32>,
}

impl IndirectionBuffer {
    pub fn new(size: Size2D<u32>) -> Self {
        IndirectionBuffer {
            data: vec![TilePosition::INVALID; size.area() as usize],
            size
        }
    }

    pub fn row(&self, row: u32) -> &[TilePosition] {
        debug_assert!(row < self.size.height);
        let start = (row * self.size.width) as usize;
        let end = start + self.size.width as usize;

        &self.data[start..end]
    }

    pub fn row_mut(&mut self, row: u32) -> &mut[TilePosition] {
        debug_assert!(row < self.size.height);
        let start = (row * self.size.width) as usize;
        let end = start + self.size.width as usize;

        &mut self.data[start..end]
    }

    pub fn reset(&mut self) {
        self.data.fill(TilePosition::INVALID);
    }

    pub fn get(&self, x: u32, y: u32) -> Option<(TilePosition, bool)> {
        let idx = (y * self.size.width + x) as usize;
        let data = self.data[idx];
        if data == TilePosition::INVALID {
            return None;
        }

        Some((data, data.flag()))
    }

    pub fn get_mut(&mut self, x: u32, y: u32) -> &mut TilePosition {
        let idx = (y * self.size.width + x) as usize;
        &mut self.data[idx]
    }
}

#[derive(Copy, Clone, Debug)]
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
        Stats {
            opaque_tiles: 0,
            alpha_tiles: 0,
            prerendered_tiles: 0,
            gpu_mask_tiles: 0,
            cpu_mask_tiles: 0,
            edges: 0,
            render_passes: 0,
            batches: 0,
        }
    }

    pub fn clear(&mut self) {
        *self = Stats::new();
    }

    pub fn tiles_bytes(&self) -> usize {
        (self.opaque_tiles + self.alpha_tiles + self.prerendered_tiles + self.gpu_mask_tiles) * std::mem::size_of::<TileInstance>()
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
    where 'l: 'a
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

