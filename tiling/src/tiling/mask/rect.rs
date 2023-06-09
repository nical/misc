use lyon::{geom::Box2D, math::{vector, point}};

use crate::{tiling::{TilePosition, encoder::{TileEncoder, as_scale_offset}, FillOptions}, TileMask, affected_range, TILE_SIZE_F32, Tiler, pattern::BuiltPattern};

use super::MaskEncoder;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct RectangleMask {
    pub tile: TilePosition,
    pub invert: u32,
    pub rect: [u32; 2],
}

unsafe impl bytemuck::Pod for RectangleMask {}
unsafe impl bytemuck::Zeroable for RectangleMask {}

pub fn fill_rect(
    rect: &Box2D<f32>,
    options: &FillOptions,
    pattern: &BuiltPattern,
    tile_mask: &mut TileMask,
    tiler: &mut Tiler,
    encoder: &mut TileEncoder,
    rect_encoder: &mut MaskEncoder,
    device: &wgpu::Device,
) {
    let mut transformed_rect = *rect;
    let mut simple = true;
    if let Some(transform) = &options.transform {
        simple = false;
        if as_scale_offset(transform).is_some() {
            transformed_rect = transform.outer_transformed_box(rect);
            simple = true;
        }
    }

    if simple {
        fill_axis_aligned_rect(
            &transformed_rect,
            options.inverted,
            tiler.scissor(),
            pattern,
            tile_mask,
            encoder,
            rect_encoder,
        );
        return;
    }

    let mut path = lyon::path::Path::builder();
    path.begin(rect.min);
    path.line_to(point(rect.max.x, rect.min.y));
    path.line_to(rect.max);
    path.line_to(point(rect.min.x, rect.max.y));
    path.end(true);
    let path = path.build();

    tiler.fill_path(path.iter(), options, pattern, tile_mask, encoder, device);    
}


pub fn fill_axis_aligned_rect(
    rect: &Box2D<f32>,
    inverted: bool,
    scissor: &Box2D<f32>,
    pattern: &BuiltPattern,
    tile_mask: &mut TileMask,
    encoder: &mut TileEncoder,
    rect_encoder: &mut MaskEncoder,
) {
    // TODO: Lots of common code for the top/middle/bottom rows that could be shared.
    // TODO: This probably doesn't work with inverted fills.
    let rect = match rect.intersection(scissor) {
        Some(r) => r,
        None => {
            if inverted {
                *scissor
            } else {
                return;
            }
        }
    };

    let (row_start, row_end) = affected_range(rect.min.y, rect.max.y, scissor.min.y, scissor.max.y);
    let (column_start, column_end) = affected_range(
        rect.min.x, rect.max.x,
        scissor.min.x,
        scissor.max.x,
    );
    let row_start = row_start as u32;
    let row_end = row_end as u32;
    let column_start = column_start as u32;
    let column_end = column_end as u32;

    let rect_start_tile = (rect.min.x / TILE_SIZE_F32) as u32;
    let rect_end_tile = (rect.max.x / TILE_SIZE_F32) as u32;

    encoder.begin_path(pattern);

    let single_column = column_start + 1 == column_end;
    let need_left_masks = rect.min.x > 0.0;
    let need_right_masks = rect_end_tile < column_end && (!single_column || !need_left_masks);
    let need_top_row = rect.min.y > 0.0;
    let need_bottom_row = row_start + 1 < row_end;

    fn local_tile_rect(rect: &Box2D<f32>, tx: u32, ty: u32) -> Box2D<f32> {
        let ts = TILE_SIZE_F32;
        let offset = -vector(tx as f32 * ts, ty as f32 * ts);
        rect.translate(offset)
    }

    if inverted {
        let columns = column_start as u32 .. column_end as u32;
        encoder.fill_rows(0..row_start, columns, pattern, tile_mask);
    }

    let mut tile_y = row_start;
    if need_top_row {
        // Top of the rect
        let mut tile_mask = tile_mask.row(tile_y);

        if inverted && column_start < rect_start_tile {
            let range = column_start..rect_start_tile.min(column_end);
            encoder.span(range, tile_y, &mut tile_mask, pattern);
        }

        if need_left_masks && tile_mask.test(rect_start_tile, false) {
            let local_rect = local_tile_rect(&rect, rect_start_tile, tile_y);
            let tl_mask_tile = add_rectangle_mask(encoder, rect_encoder, &local_rect, inverted);
            let opaque = false;
            encoder.add_tile(pattern, opaque, TilePosition::new(rect_start_tile, tile_y), tl_mask_tile);
        }

        if rect_end_tile > rect_start_tile + 1 {
            let local_rect = local_tile_rect(&rect, rect_start_tile + 1, tile_y);
            let tr_mask_tile = add_rectangle_mask(encoder, rect_encoder, &local_rect, inverted);

            for x in rect_start_tile + 1 .. rect_end_tile {
                if tile_mask.test(x, false) {
                    encoder.add_tile(pattern, false, TilePosition::new(x, tile_y), tr_mask_tile);
                }
            }
        }

        if need_right_masks {
            let local_rect = local_tile_rect(&rect, rect_end_tile, tile_y);
            let tr_mask_tile = add_rectangle_mask(encoder, rect_encoder, &local_rect, inverted);
            encoder.add_tile(pattern, false, TilePosition::new(rect_end_tile, tile_y), tr_mask_tile);
        }

        tile_y += 1
    }

    let mut left_mask = None;
    let mut right_mask = None;
    while tile_y < row_end - 1 {
        let mut tile_mask = tile_mask.row(tile_y);

        if need_left_masks && left_mask.is_none() {
            let local_rect = local_tile_rect(&rect, rect_start_tile, tile_y);
            left_mask = Some(add_rectangle_mask(encoder, rect_encoder, &local_rect, inverted));
        }
        if need_right_masks && right_mask.is_none() {
            let local_rect = local_tile_rect(&rect, rect_end_tile, tile_y);
            right_mask = Some(add_rectangle_mask(encoder, rect_encoder, &local_rect, inverted));
        }

        if inverted && column_start < rect_start_tile {
            let range = column_start..rect_start_tile.min(column_end);
            encoder.span(range, tile_y, &mut tile_mask, pattern);
        }

        if let Some(mask) = left_mask {
            if tile_mask.test(rect_start_tile, false) {
                encoder.add_tile(pattern, false, TilePosition::new(rect_start_tile, tile_y), mask);
            }
        }

        if !inverted && rect_start_tile + 1 < rect_end_tile {
            let range = rect_start_tile + 1 .. rect_end_tile;
            encoder.span(range, tile_y, &mut tile_mask, pattern);
        }

        if let Some(mask) = right_mask {
            if tile_mask.test(rect_end_tile, false) {
                encoder.add_tile(pattern, false, TilePosition::new(rect_end_tile, tile_y), mask);
            }
        }

        if !inverted && rect_end_tile + 1 < column_end {
            let range = rect_end_tile + 1 .. column_end;
            encoder.span(range, tile_y, &mut tile_mask, pattern);
        }

        tile_y += 1;
    }

    if need_bottom_row {
        // Bottom of the rect
        let mut tile_mask = tile_mask.row(tile_y);

        if inverted && column_start < rect_start_tile {
            let range = column_start..rect_start_tile.min(column_end);
            encoder.span(range, tile_y, &mut tile_mask, pattern);
        }

        if need_left_masks && tile_mask.test(rect_start_tile, false) {
            let local_rect = local_tile_rect(&rect, rect_start_tile, tile_y);
            let tl_mask_tile = add_rectangle_mask(encoder, rect_encoder, &local_rect, inverted);
            let opaque = false;
            encoder.add_tile(pattern, opaque, TilePosition::new(rect_start_tile, tile_y), tl_mask_tile);
        }

        if rect_end_tile > rect_start_tile + 1 {
            let local_rect = local_tile_rect(&rect, rect_start_tile + 1, tile_y);
            let tr_mask_tile = add_rectangle_mask(encoder, rect_encoder, &local_rect, inverted);

            for x in rect_start_tile + 1 .. rect_end_tile {
                if tile_mask.test(x, false) {
                    encoder.add_tile(pattern, false, TilePosition::new(x, tile_y), tr_mask_tile);
                }
            }
        }

        if need_right_masks {
            let local_rect = local_tile_rect(&rect, rect_end_tile, tile_y);
            let tr_mask_tile = add_rectangle_mask(encoder, rect_encoder, &local_rect, inverted);
            encoder.add_tile(pattern, false, TilePosition::new(rect_end_tile, tile_y), tr_mask_tile);
        }

        tile_y += 1;
    }

    if inverted {
        let columns = column_start as u32 .. column_end as u32;
        encoder.fill_rows(tile_y..row_end as u32, columns, pattern, tile_mask);
    }
}

pub fn add_rectangle_mask(tile_encoder: &mut TileEncoder, rect_encoder: &mut MaskEncoder, rect: &Box2D<f32>, inverted: bool) -> TilePosition {
    let (tile, atlas_index) = tile_encoder.allocate_mask_tile();

    // TODO: handle inverted?

    let zero = point(0.0, 0.0);
    let one = point(1.0, 1.0);
    let min = ((rect.min / crate::TILE_SIZE_F32).clamp(zero, one) * std::u16::MAX as f32).to_u32();
    let max = ((rect.max / crate::TILE_SIZE_F32).clamp(zero, one) * std::u16::MAX as f32).to_u32();
    rect_encoder.prerender_mask(atlas_index, RectangleMask {
        tile,
        invert: if inverted { 1 } else { 0 },
        rect: [
            min.x << 16 | min.y,
            max.x << 16 | max.y,
        ]
    });

    tile
}

