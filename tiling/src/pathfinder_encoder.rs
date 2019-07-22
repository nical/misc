use crate::*;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct EdgeInstance {
    pub from: [f32; 2],
    pub ctrl: [f32; 2],
    pub to: [f32; 2],
    pub tile_index: u16
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct SolidTile {
    pub position: [f32; 2],
    pub path_id: u16,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct AlphaTile {
    pub position: [f32; 2],
    pub tile_index: u16,
    pub path_id: u16,
}

pub struct PathfinderLikeEncoder<'l> {
    pub edges: Vec<EdgeInstance>,
    pub solid_tiles: Vec<SolidTile>,
    pub alpha_tiles: Vec<AlphaTile>,
    pub next_tile_index: u16,
    pub z_buffer: &'l mut ZBuffer,
}

impl<'l> TileEncoder for PathfinderLikeEncoder<'l> {
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

        if !self.z_buffer.test(tile.x, tile.y, tile.path_id, solid) {
            // Culled by a solid tile.
            return;
        }

        if solid {
            self.solid_tiles.push(SolidTile {
                position: [tile.x as f32, tile.y as f32],
                path_id: tile.path_id,
            });
            return;
        }

        let tile_index = self.next_tile_index;
        self.next_tile_index += 1;
        self.alpha_tiles.push(AlphaTile {
            position: [tile.x as f32, tile.y as f32],
            tile_index,
            path_id: tile.path_id,
        });

        let tile_w = tile.outer_rect.max.x - tile.outer_rect.min.x;
        let offset = lyon_path::geom::math::vector(-tile.outer_rect.min.x, 0.0);
        for edge in active_edges {
            let edge = edge.clip_horizontally(tile.outer_rect.min.x .. tile.outer_rect.max.x);

            let from = (edge.from - offset).to_array();
            let to = (edge.to - offset).to_array();
            let ctrl = (edge.ctrl - offset).to_array();

            self.edges.push(EdgeInstance {
                from,
                to,
                ctrl,
                tile_index,
            });

            if edge.from.y == 0.0 {
                let (from, to) = if (from[0] < to[0]) ^ (edge.winding < 0) {
                    (from[0], tile_w)
                } else {
                    (tile_w, from[0])
                };

                self.edges.push(EdgeInstance {
                    from: [from, 0.0],
                    to: [to, 0.0],
                    ctrl: [to, 0.0],
                    tile_index,
                });
            }
        }
    }
}
