use lyon::{math::{Point, point, vector}, geom::Box2D, path::Winding};

use crate::{tiling::{TilePosition, encoder::{TileEncoder, as_scale_offset}, TileVisibility, FillOptions}, TileMask, Tiler, pattern::BuiltPattern};
use crate::tiling::tiler::affected_range;

use super::MaskEncoder;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct CircleMask {
    pub tile: TilePosition,
    pub radius: f32,
    pub center: [f32; 2],
}

unsafe impl bytemuck::Pod for CircleMask {}
unsafe impl bytemuck::Zeroable for CircleMask {}

const TILE_SIZE: f32 = crate::TILE_SIZE as f32;

pub fn add_cricle_mask(tile_encoder: &mut TileEncoder, circle_masks: &mut MaskEncoder, center: Point, radius: f32, inverted: bool) -> TilePosition {
    let (mut tile, atlas_index) = tile_encoder.allocate_mask_tile();
    if inverted {
        tile.add_flag();
    }
    circle_masks.prerender_mask(atlas_index, CircleMask {
        tile, radius,
        center: center.to_array()
    });

    tile
}

pub fn fill_circle(
    mut center: Point,
    mut radius: f32,
    options: &FillOptions,
    pattern: &BuiltPattern,
    tile_mask: &mut TileMask,
    tiler: &mut Tiler,
    encoder: &mut TileEncoder,
    circle_masks: &mut MaskEncoder,
    device: &wgpu::Device,
) {
    //tiler.set_fill_rule(options.fill_rule, options.inverted);
    encoder.prerender_pattern = options.prerender_pattern;

    let mut simple = true;
    if let Some(transform) = &options.transform {
        simple = false;
        if let Some((scale, offset)) = as_scale_offset(transform) {
            if (scale.x - scale.y).abs() < 0.01 {
                center = center * scale.x + offset;
                radius = radius * scale.x;
                simple = true;
            }
        }
    }

    if simple {
        fill_transformed_circle(
            center,
            radius,
            tiler.scissor(),
            options.inverted,
            pattern,
            tile_mask,
            encoder,
            circle_masks,
        );
    } else {
        let mut path = lyon::path::Path::builder();
        path.add_circle(center, radius, Winding::Positive);
        let path = path.build();

        tiler.fill_path(
            path.iter(),
            options,
            pattern,
            tile_mask,
            encoder,
            device,
        );
    }
}

fn fill_transformed_circle(
    center: Point,
    radius: f32,
    scissor: &Box2D<f32>,
    inverted: bool,
    pattern: &BuiltPattern,
    tile_mask: &mut TileMask,
    encoder: &mut TileEncoder,
    circle_masks: &mut MaskEncoder,
) {
    let mut y_min = center.y - radius;
    let mut y_max = center.y + radius;
    let mut x_min = center.x - radius;
    let mut x_max = center.x + radius;
    if inverted {
        x_min = scissor.min.x;
        y_min = scissor.min.y;
        x_max = scissor.max.x;
        y_max = scissor.max.y;
    }
    let (row_start, row_end) = affected_range(y_min, y_max, scissor.min.y, scissor.max.y);
    let (column_start, column_end) = affected_range(
        x_min, x_max,
        scissor.min.x,
        scissor.max.x,
    );
    let row_start = row_start as u32;
    let row_end = row_end as u32;
    let column_start = column_start as u32;
    let column_end = column_end as u32;
    let tile_radius = std::f32::consts::SQRT_2 * 0.5 * TILE_SIZE;
    encoder.begin_path(pattern);

    for tile_y in row_start..row_end {
        let mut tile_mask = tile_mask.row(tile_y);

        let mut tile_center = point(
            (column_start as f32 + 0.5) * TILE_SIZE,
            (tile_y as f32 + 0.5) * TILE_SIZE,
        );

        let mut tile_x = column_start;
        while tile_x < column_end {
            let d = (tile_center - center).length();
            if d - tile_radius < radius {
                break;
            }

            tile_center.x += TILE_SIZE;
            tile_x += 1;
        }
        if inverted && tile_x > column_start {
            encoder.span(column_start..tile_x, tile_y, &mut tile_mask, pattern);
        }

        let mut full = false;
        while tile_x < column_end {
            let d = (tile_center - center).length();

            full = d + tile_radius < radius;
            let outside = d - tile_radius > radius;
            if full || outside {
                break;
            }

            let tx = tile_x;

            tile_center.x += TILE_SIZE;
            tile_x += 1;

            let tile_vis = pattern.tile_visibility(tx, tile_y);

            if tile_vis == TileVisibility::Empty {
                continue;
            }

            let opaque = false;
            if !tile_mask.test(tx as u32, opaque) {
                continue;
            }

            let tile_offset = vector(tx as f32, tile_y as f32) * TILE_SIZE;
            let center = center - tile_offset;
            let mask_id = add_cricle_mask(encoder, circle_masks, center, radius, inverted);

            let tile_position = TilePosition::new(tx, tile_y);

            encoder.add_tile(pattern, opaque, tile_position, mask_id);
        }

        if full {
            let first_full_tile = tile_x;
            while tile_x < column_end {
                tile_center.x += TILE_SIZE;
                tile_x += 1;

                let d = (tile_center - center).length();

                let full = d + tile_radius < radius;
                if !full {
                    break;
                }
            }

            if !inverted  {
                assert!(first_full_tile < tile_x);
                let range = first_full_tile..tile_x;
                encoder.span(range, tile_y, &mut tile_mask, pattern);
            }
        }

        while tile_x < column_end {
            let d = (tile_center - center).length();

            if d - tile_radius > radius {
                break;
            }

            let tx = tile_x;

            tile_center.x += TILE_SIZE;
            tile_x += 1;

            let tile_vis = pattern.tile_visibility(tx, tile_y);
            if tile_vis == TileVisibility::Empty {
                continue;
            }

            let opaque = false;
            if !tile_mask.test(tx as u32, opaque) {
                continue;
            }

            let tile_offset = vector(tx as f32, tile_y as f32) * TILE_SIZE;
            let center = center - tile_offset;
            let mask_id = add_cricle_mask(encoder, circle_masks, center, radius, inverted);

            let tile_position = TilePosition::new(tx, tile_y);

            encoder.add_tile(pattern, opaque, tile_position, mask_id);
        }

        if inverted && tile_x < column_end {
            encoder.span(tile_x..column_end, tile_y, &mut tile_mask, pattern);
        }
    }
}
