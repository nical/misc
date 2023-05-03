// TODO: remove the contents of this file? support for it was removed from the tile
// encoder (but we may get similar functionality back in later).

use lyon::geom::euclid::default::Size2D;

use super::{TilerPattern, PatternIndex, TileVisibility, PatternData, TILED_IMAGE_PATTERN, TilePosition};

pub struct TiledOutput {
    pub indirection_buffer: IndirectionBuffer,
    pub tile_allocator: super::encoder::TileAllocator,
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

