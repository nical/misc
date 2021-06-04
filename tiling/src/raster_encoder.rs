use crate::*;
use rasterizer2::*;
use euclid::point2;

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

/// An example encoder that encodes tiles in a similar format as pathfinder.
pub struct WrEncoder<'l> {
    pub mask_buffer: Vec<u8>,
    pub solid_tiles: Vec<SolidTile>,
    pub mask_tiles: Vec<MaskTile>,
    pub next_tile_index: u16,
    pub z_buffer: &'l mut ZBuffer,
    pub z_index: u16,
}

impl<'l> WrEncoder<'l> {
    pub fn new(z_buffer: &'l mut ZBuffer) -> Self {
        WrEncoder {
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

impl<'l> TileEncoder for WrEncoder<'l> {
    fn encode_tile(&mut self, tile: &TileInfo, active_edges: &[ActiveEdge]) {

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

        let tile_w = tile.outer_rect.max.x - tile.outer_rect.min.x;
        let offset = lyon_path::geom::math::vector(tile.outer_rect.min.x, tile.outer_rect.min.y);
        for edge in active_edges {
            let edge = edge.clip_horizontally(tile.outer_rect.min.x .. tile.outer_rect.max.x);

            let mut from = (edge.from - offset).clamp(point2(0.0, 0.0), point2(16.0, 16.0));
            let mut to = (edge.to - offset).clamp(point2(0.0, 0.0), point2(16.0, 16.0));
            //let ctrl = (edge.ctrl - offset);

            draw_line(from, to, &mut accum, &mut backdrops);
        }

        let mut offset = self.mask_buffer.len();
        self.mask_buffer.reserve(TILE_SIZE * TILE_SIZE);
        //for i in 0..(TILE_SIZE * TILE_SIZE) {
        //    self.mask_buffer.push(0);
        //}
        unsafe {
            self.mask_buffer.set_len(offset + TILE_SIZE * TILE_SIZE);
        }

        accumulate_even_odd(
            &accum,
            &backdrops,
            &mut self.mask_buffer[offset .. offset + TILE_SIZE * TILE_SIZE],
        );
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FillObjectPrimitive {
    pub px: LineSegmentU4,
    pub subpx: LineSegmentU8,
    pub tile_x: i16,
    pub tile_y: i16,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct LineSegmentU4 {
    pub from: u8,
    pub to: u8,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct LineSegmentU8 {
    pub from_x: u8,
    pub from_y: u8,
    pub to_x: u8,
    pub to_y: u8,
}

impl FillObjectPrimitive {
    pub fn decode(&self) -> LineSegment<f32> {
        LineSegment {
            from: point(
                (self.px.from & 0b1111) as f32 + self.subpx.from_x as f32 / 255.0,
                (self.px.from >> 4) as f32 + self.subpx.from_y as f32 / 255.0,
            ),
            to: point(
                (self.px.to & 0b1111) as f32 + self.subpx.to_x as f32 / 255.0,
                (self.px.to >> 4) as f32 + self.subpx.to_y as f32 / 255.0,
            ),
        }
    }

    pub fn encode(from: Point, to: Point, tile_x: i16, tile_y: i16) -> Self {
        let from_qx = (from.x * 256.0) as u32;
        let from_qy = (from.y * 256.0) as u32;
        let subpx_from_x = from_qx as u8;
        let subpx_from_y = from_qy as u8;
        let from_px_x = (from_qx >> 8) as u8 & 0b1111;
        let from_px_y = (from_qy >> 4) as u8 & 0b11110000;
        let px_from = from_px_x | from_px_y;

        let to_qx = (to.x * 256.0) as u32;
        let to_qy = (to.y * 256.0) as u32;
        let subpx_to_x = to_qx as u8;
        let subpx_to_y = to_qy as u8;
        let to_px_x = (to_qx >> 8) as u8 & 0b1111;
        let to_px_y = (to_qy >> 4) as u8 & 0b11110000;
        let px_to = to_px_x | to_px_y;

        FillObjectPrimitive {
            px: LineSegmentU4 {
                from: px_from,
                to: px_to,
            },
            subpx: LineSegmentU8 {
                from_x: subpx_from_x,
                from_y: subpx_from_y,
                to_x: subpx_to_x,
                to_y: subpx_to_y,
            },
            tile_x,
            tile_y,
        }
    }
}

#[test]
fn encode_decode() {
    use euclid::approxeq::ApproxEq;

    let encoded = FillObjectPrimitive::encode(
        point(9.1, 2.2),
        point(3.5, 10.9),
        0,
        0,
    );

    let decoded = encoded.decode();

    assert!(decoded.from.approx_eq_eps(&point(9.1, 2.2), &point(0.05, 0.05)));
    assert!(decoded.to.approx_eq_eps(&point(3.5, 10.9), &point(0.05, 0.05)));
}