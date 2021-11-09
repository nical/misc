use crate::tiler::*;
use crate::cpu_rasterizer::*;
use lyon::geom::euclid::point2;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct SolidTile {
    pub position: [f32; 2],
    pub path_id: u16,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct MaskTile {
    pub position: [f32; 2],
    pub path_id: u16,
    pub mask_idx: u16,
}

pub struct RasterEncoder<'l> {
    pub mask_buffer: Vec<u8>,
    pub solid_tiles: Vec<SolidTile>,
    pub mask_tiles: Vec<MaskTile>,
    pub next_tile_index: u16,
    pub z_buffer: &'l mut ZBuffer,
    pub z_index: u16,
}

impl<'l> RasterEncoder<'l> {
    pub fn new(z_buffer: &'l mut ZBuffer) -> Self {
        RasterEncoder {
            mask_buffer: Vec::with_capacity(TILE_SIZE * TILE_SIZE * 5000),
            solid_tiles: Vec::with_capacity(2000),
            mask_tiles: Vec::with_capacity(5000),
            next_tile_index: 0,
            z_buffer,
            z_index: 0,
        }
    }

    pub fn reset(&mut self) {
        self.solid_tiles.clear();
        self.mask_tiles.clear();
        self.mask_buffer.clear();
        self.next_tile_index = 0;
    }
}

impl<'l> TileEncoder for RasterEncoder<'l> {
    fn encode_tile(&mut self, tile: &TileInfo, active_edges: &[ActiveEdge], left: &SideEdgeTracker) {

        let mut solid = false;
        if active_edges.is_empty() {
            if tile.backdrop_winding % 2 != 0 {
                solid = true;
            } else {
                // Empty tile.
                return;
            }
        }

        if !self.z_buffer.test(tile.x, tile.y, self.z_index, solid) {
            // Culled by a solid tile.
            return;
        }

        if solid {
            self.solid_tiles.push(SolidTile {
                position: [tile.x as f32, tile.y as f32],
                path_id: self.z_index,
            });
            return;
        }

        let mask_idx = self.next_tile_index;
        self.next_tile_index += 1;
        self.mask_tiles.push(MaskTile {
            position: [tile.x as f32, tile.y as f32],
            path_id: self.z_index,
            mask_idx,
        });

        let mut accum = [0.0; TILE_SIZE * TILE_SIZE];
        let mut backdrops = [tile.backdrop_winding as f32; TILE_SIZE];

        let tile_offset = lyon::path::math::vector(tile.outer_rect.min.x, tile.outer_rect.min.y);
        for edge in active_edges {
            let edge = edge.clip_horizontally(tile.outer_rect.min.x .. tile.outer_rect.max.x);

            let from = (edge.from - tile_offset).clamp(point2(0.0, 0.0), point2(16.0, 16.0));
            let to = (edge.to - tile_offset).clamp(point2(0.0, 0.0), point2(16.0, 16.0));

            draw_line(from, to, &mut accum, &mut backdrops);
        }

        let mask_offset = self.mask_buffer.len();
        self.mask_buffer.reserve(TILE_SIZE * TILE_SIZE);
        unsafe {
            // Unfortunately it's measurably faster to leave the bytes uninitialized,
            // we are going to overwrite them anyway.
            self.mask_buffer.set_len(mask_offset + TILE_SIZE * TILE_SIZE);
        }

        accumulate_even_odd(
            &accum,
            &backdrops,
            &mut self.mask_buffer[mask_offset .. mask_offset + TILE_SIZE * TILE_SIZE],
        );
    }
}
