use std::f32;

use lyon::geom::QuadraticBezierSegment;
use lyon::geom::euclid::default::*;
#[cfg(test)]
use lyon::geom::euclid::point2;

/*
use lyon::path::FillRule;
use aliasable::boxed::AliasableBox;
use parasol::{Context, Event};

pub struct TileRasterizer {
    tiles: AliasableBox<[TileRasterJobTile]>,
    edges: AliasableBox<[QuadraticBezierSegment<f32>]>,
    tile_range: Range<usize>,
    edge_range: Range<usize>,
    next_staging_buffer_offset: u32,
    rasterized_tiles: Vec<(u32, u32)>, // staging buffer offset, tile offset

    // TODO: make this a mapped wgpu buffer
    staging_buffer: AliasableBox<[u8]>,

    event: AliasableBox<Event>,
}

impl TileRasterizer {
    pub fn new(tile_buffer_size: usize, edge_buffer_size: usize, staging_buffer_size: usize, ctx: &Context) -> Self {
        let edge = QuadraticBezierSegment {
            from: point(0.0, 0.0),
            ctrl: point(0.0, 0.0),
            to: point(0.0, 0.0),
        };
        let tile = TileRasterJobTile {
            edges: 0..0,
            output: 0,
            backdrop: 0,
            fill_rule: FillRule::EvenOdd,
        };
        let edges = AliasableBox::from_unique(vec![edge; edge_buffer_size].into_boxed_slice());
        let tiles = AliasableBox::from_unique(vec![tile; tile_buffer_size].into_boxed_slice());
        let staging_buffer = AliasableBox::from_unique(vec![0; staging_buffer_size].into_boxed_slice());

        TileRasterizer {
            tiles,
            edges,
            tile_range: 0..0,
            edge_range: 0..0,
            next_staging_buffer_offset: 0,
            rasterized_tiles: Vec::with_capacity(512),
            staging_buffer,
            event: Event::new_boxed(Event::MAX_DEPENDECIES, ctx.thread_pool_id()),
        }
    }

    pub fn push_edge(&mut self, edge: &QuadraticBezierSegment<f32>) -> bool {
        if self.edge_range.end == self.edges.len() {
            return false;
        }

        self.edges[self.edge_range.end] = *edge;
        self.edge_range.end += 1;

        true
    }

    pub fn push_tile(&mut self, backdrop: i16, fill_rule: FillRule, tile_id: u32) -> bool {
        if self.tile_range.end == self.tiles.len() {
            return false;
        }

        let staging_buffer_offset = self.next_staging_buffer_offset;
        self.next_staging_buffer_offset += 1;

        self.tiles[self.tile_range.end] = TileRasterJobTile {
            edges: self.edge_range.start as u32 .. self.edge_range.end as u32,
            backdrop,
            fill_rule,
            output: staging_buffer_offset,
        };

        self.tile_range.end += 1;
        self.edge_range.start = self.edge_range.end;

        self.rasterized_tiles.push((staging_buffer_offset, tile_id));

        true
    }

    pub fn schedule_work(&mut self, ctx: &mut Context) {
        unimplemented!()
    }
}

pub fn rasterize_tile(
    edges: &[QuadraticBezierSegment<f32>],
    backdrop: i16,
    fill_rule: FillRule,
    tolerance: f32,
    output: &mut[u8]
) {
    let mut accum = [0.0; crate::BYTES_PER_MASK];
    let mut backdrops = [backdrop as f32; 32];

    for edge in edges {
        if edge.from.x < -0.5 && edge.from.y != -0.5 {
            add_backdrop(edge.from.y, 1.0, &mut backdrops[0..TILE_SIZE as usize]);
        }

        if edge.to.x < -0.5 && edge.to.y != -0.5 {
            add_backdrop(edge.to.y, -1.0, &mut backdrops[0..TILE_SIZE as usize]);
        }

        let is_line = edge.ctrl.x.is_nan();
        if is_line {
            draw_line(edge.from, edge.to, &mut accum);
        } else {
            draw_curve(edge.from, edge.ctrl, edge.to, tolerance, &mut accum);
        }
    }

    let accumulate = match fill_rule {
        FillRule::EvenOdd => accumulate_even_odd,
        FillRule::NonZero => accumulate_non_zero,
    };

    accumulate(
        &accum,
        &backdrops,
        output,
    );
}

pub struct TileRasterJob {
    pub edges: *const QuadraticBezierSegment<f32>,
    pub num_edges: u32,
    pub tiles: *const TileRasterJobTile,
    pub num_tiles: u32,
    pub tolerance: f32,
    pub output: *mut u8,
}

#[derive(Clone, Debug)]
pub struct TileRasterJobTile {
    pub edges: Range<u32>,
    pub output: u32,
    pub backdrop: i16,
    pub fill_rule: FillRule,
}

pub unsafe fn exec_tile_job(this: *const TileRasterJob) {
    let tiles = std::slice::from_raw_parts((*this).tiles, (*this).num_tiles as usize);
    let edges = std::slice::from_raw_parts((*this).edges, (*this).num_edges as usize);
    for tile in tiles {
        let edge_range = tile.edges.start as usize .. tile.edges.end as usize;
        let offset = tile.output as usize * BYTES_PER_MASK;
        let output = std::slice::from_raw_parts_mut(
            (*this).output.offset(offset as isize),
            BYTES_PER_MASK,
        );

        rasterize_tile(
            &edges[edge_range],
            tile.backdrop,
            tile.fill_rule,
            (*this).tolerance,
            output
        );
    }
}
*/

pub const TILE_SIZE: usize = 16;

pub fn draw_line(from: Point2D<f32>, to: Point2D<f32>, dst: &mut[f32]) {
    // This function rasterizes a line segment in a winding accumulation buffer using the
    // signed area approach for computing antialiasing described in http://nothings.org/gamedev/rasterize/
    //
    // This is heavily inspired by https://github.com/raphlinus/font-rs with the added requirement
    // to handle edges crossing the left boundary of the tile.
    //
    // The values in the accumulation buffer represent the difference between the signed area of
    // the corresponding pixel and its left neighbors affected by the same edge, so that a simple
    // routine can generate the actual coverage mask via a prefix sum per row.
    //
    // It is important to not get confused by notion of coverage *difference*. When doing the math
    // it's easy to think in terms of absolute coverage and forget that:
    //  - The value for each pixel depends the coverage that has already been computed for previous
    //    pixels on the left for the same edge,
    //  - in addition to computing a value for the partially covered pixels, we also write the reminder
    //    of the signed area to the next non-covered pixel immediately after the edge.

    assert!(dst.len() >= TILE_SIZE * TILE_SIZE);

    let (winding, top, bottom) = if to.y > from.y {
        (1.0, from, to)
    } else {
        (-1.0, to, from)
    };

    let y_min = f32::max(top.y, 0.0) as usize;
    let y_max = f32::min(bottom.y.ceil(), TILE_SIZE as f32) as usize;

    if (from.y - to.y).abs() < 0.0001 {
        // horizontal line.
        return;
    }

    let d = bottom - top;
    let dxdy = d.x / d.y;

    //println!("y range {:?} .. {:?}", y_min, y_max);

    let mut x = top.x;

    for y in y_min .. y_max {
        //println!("y = {}", y);
        // y range of the edge overlapping the current row of pixels
        let dy = f32::min((y + 1) as f32, bottom.y) - f32::max(y as f32, top.y);

        // x starting point of the next row.
        let next_x = x + dxdy * dy;

        // Winding sign and y coverage combined in a single quantity since we'd have
        // to multiply them both every time.
        let winding_dy = winding * dy;

        let x_range = if next_x > x { x .. next_x } else { next_x .. x };
        let x_range_snapped = x_range.start.floor() .. x_range.end.ceil();
        //println!("x range {:?}, dy {:?} dxdy {:?}", x_range, dy, dxdy);
        let x_range_int = x_range_snapped.start as i32 .. x_range_snapped.end as i32;

        fn row_idx(idx: i32) -> usize { idx.max(0) as usize }

        //println!(" - x range {:?}", x_range);

        let row = &mut dst[y * TILE_SIZE .. (y + 1) * TILE_SIZE];

        //let start_idx = x_range_int.start.max(0).min(15) as usize;
        let start_idx = x_range_int.start;

        if start_idx >= TILE_SIZE as i32 {
            x = next_x;
            continue;
        }

        if x_range_int.len() <= 1 {
            // The edge overlaps a single pixel on the current row.
            //
            //
            //   +----------+
            //   |          |
            //   |   o......|  |
            //   |     \....|  | dy
            //   |       o..|  |
            //   |          |
            //   +----------+
            //       ^   ^
            //       x   next_x

            // coverage on the left of the edge.
            let left_area = ((x + next_x) * 0.5 - x_range_snapped.start) * winding_dy;

            // (1.0 - x_left) * winding_dy;
            row[row_idx(start_idx)] += winding_dy - left_area;
            if start_idx + 1 < TILE_SIZE as i32 {
                // Non-covered pixel immediately after the edge.
                row[row_idx(start_idx + 1)] += left_area;
            }
        } else {
            // The edge overlaps two or more pixels on the current row.
            //
            //   +-------+-------+-------+-------+-------+
            //   |     o_|___. . |. . . .|. . . .|. . . .|   |
            //   |       |   -------______ . . . | . . . |   | dy
            //   |       |       |       |------____o . .|   |
            //   |       |       |       |       |       |
            //   +-------+-------+-------+-------+-------+
            //         __                         ___
            //         x_left                     x_right
            //    _______ _______ _______________ _______
            //       A       B           C           D
            //
            // In this situation we consider the left-most and right-most
            // pixels separately. For the pixels in-between we can take
            // advantage of the coverage difference between equal.

            let inv_dx = (x_range.end - x_range.start).recip();

            // x overlap of the edge on the left-most and right-most pixels.
            let x_left = 1.0 - (x_range.start - x_range_snapped.start);
            let x_right = x_range.end - x_range_snapped.end + 1.0;

            // Think of the cov_ values as areas but without the winding_dy
            // contributions which are applied at the end.
            let cov_left = 0.5 * inv_dx * x_left * x_left;
            let cov_right = 0.5 * inv_dx * x_right * x_right;

            // (A) Pixel on the left side of the edge partially overlapping on the x axis.
            // The area is a single triangle.
            row[row_idx(start_idx)] += winding_dy * cov_left;

            let x_range_int_end = x_range_int.end;

            if x_range_int.len() == 2 {
                if start_idx + 1 < TILE_SIZE as i32 {
                    // (D) Pixel on the right side of the edge partially overlapping on the x axis.
                    // Special case for when the edge touches only two pixels on this row because
                    // computing the already contributed coverage is simple.
                    row[row_idx(start_idx + 1)] += winding_dy * (1.0 - cov_left - cov_right);
                }
            } else if start_idx + 1 < TILE_SIZE as i32 {

                // (B) Second pixel (fully overlapping the edge)
                let cov_b = inv_dx * (x_left + 0.5);
                row[row_idx(start_idx + 1)] += winding_dy * (cov_b - cov_left);

                // (C) Subsequent pixels fully overlapping with the edge on the x axis.
                // For these the covered area has increased linearly so the coverage difference
                // is constant.
                for i in (start_idx + 2) .. (x_range_int_end - 1).min(15) {
                    row[row_idx(i)] += winding_dy * inv_dx;
                }

                if x_range_int_end - 1 < TILE_SIZE as i32 {
                    // (D) Pixel on the right side of the edge partially overlapping on the x axis.
                    let cov_c = cov_b + (x_range_int.len() - 3) as f32 * inv_dx;
                    row[row_idx(x_range_int_end - 1)] += winding_dy * (1.0 - cov_c - cov_right);
                }
            }

            if x_range_int.end < TILE_SIZE as i32 {
                // Non-covered pixel immediately after the edge.
                row[row_idx(x_range_int_end)] += winding_dy * cov_right;
            }
        }

        x = next_x;
    }
}

pub fn draw_curve(from: Point2D<f32>, ctrl: Point2D<f32>, to: Point2D<f32>, tolerance: f32, dst: &mut[f32]) {
    QuadraticBezierSegment { from, ctrl, to }.for_each_flattened(tolerance, &mut |segment| {
        draw_line(segment.from, segment.to, dst);
    });
}

// Write the winding numbers for auxiliary (backdrop) edges on the left side of the tile that contribute to the tile. 
pub fn add_backdrop(y: f32, winding: f32, dst: &mut[f32]) {
    let start = y as usize;
    if start >= dst.len() {
        return;
    }
    let cov = 1.0 - y.fract();
    dst[start] += cov * winding;
    for e in &mut dst[(start + 1)..] {
        *e += winding;
    }
}

// TODO: this is closer to what font-rs does, would it be better?
//pub fn draw_curve(from: Point2D<f32>, ctrl: Point2D<f32>, to: Point2D<f32>, _tolerance: f32, dst: &mut[f32], backdrops: &mut[f32]) {
//    let ddx = from.x - 2.0 * ctrl.x + to.x;
//    let ddy = from.y - 2.0 * ctrl.y + to.y;
//    let square_dev = ddx * ddx + ddy * ddy;
//    if square_dev < 0.333 {
//        draw_line(from, to, dst);
//        return;
//    }
//
//    let n = 1 + (3.0 * square_dev).sqrt().sqrt().floor() as u32;
//    let inv_n = (n as f32).recip();
//    let mut t = 0.0;
//    let mut prev = from;
//    for _ in 0..(n - 1) {
//        t += inv_n;
//        let next = QuadraticBezierSegment { from, ctrl, to }.sample(t);
//        draw_line(prev, next, dst);
//        prev = next;
//    }
//    draw_line(prev, to, dst);
//}


// Also similar to font-rs with an important caveat: we have a backdrop per row and we
// have to support shapes that aren't closed within the tile (there may be one side of
// the shape on the tile while the other side overlaps another tile), so we can't do
// the fast prefix-sum over the whole tile. Instead we have to a prefix sum per row.
//
// This makes it harder to do an efficient SIMD version the same way font-rs does.
// However, we could interleave rows by groups of four in order to speed up the
// accumulation with SIMD instructions (that means the line routine would become more
// complicated but probably worth it. The mask could be de-interleaved in a shader
// that copies from staging buffer to the atlas texture).

pub fn accumulate_non_zero(src: &[f32], backdrops: &[f32], dst: &mut[u8]) {
    assert!(backdrops.len() >= TILE_SIZE);
    assert!(src.len() >= TILE_SIZE * TILE_SIZE);
    assert!(dst.len() >= TILE_SIZE * TILE_SIZE);
    let mut idx = 0;
    for y in 0..TILE_SIZE {
        let mut acc = backdrops[y];
        for _ in 0..TILE_SIZE {
            acc += src[idx];
            let a = acc.abs();
            let a = if a < 1.0 { a } else { 1.0 };
            dst[idx] = (a * 255.0) as u8;
            idx += 1;
        }
    }
}

pub fn accumulate_even_odd(src: &[f32], backdrops: &[f32], dst: &mut[u8]) {
    assert!(backdrops.len() >= TILE_SIZE);
    assert!(src.len() >= TILE_SIZE * TILE_SIZE);
    assert!(dst.len() >= TILE_SIZE * TILE_SIZE);
    let mut idx = 0;
    for y in 0..TILE_SIZE {
        let mut acc = backdrops[y];
        for _ in 0..TILE_SIZE {
            acc += src[idx];
            let a = acc.abs() % 2.0;
            let a = if a < 1.0 { a } else { 2.0 - a };
            dst[idx] = (a * 255.0) as u8;
            //if (idx+y) % 2 == 0 { dst[idx] = 255; } // debug
            idx += 1;
        }
    }
}

use std::sync::atomic::{Ordering, AtomicU32};
static DUMP_IDX: AtomicU32 = AtomicU32::new(0);
pub fn dump_mask_png(w: u32, h: u32, mask: &[u8]) {
    let idx = DUMP_IDX.fetch_add(1, Ordering::Relaxed);
    let file_name = format!("tmp/mask-{}.png", idx);
    save_mask_png(w, h, mask, &file_name)
}

pub fn save_mask_png(w: u32, h: u32, mask: &[u8], file_name: &str) {
    let mut bytes = Vec::with_capacity(mask.len()*4 * PIXEL_SIZE * PIXEL_SIZE);
    const PIXEL_SIZE: usize = 8;
    for y in 0..TILE_SIZE {
        for _ in 0..PIXEL_SIZE {
            for x in 0..TILE_SIZE {
                let a = mask[y * TILE_SIZE + x];
                let n = 15;
                let a = if a < 255 - n { a + n } else { a };
                for _ in 0..PIXEL_SIZE {
                    bytes.push(a);
                    bytes.push(a);
                    bytes.push(a);
                    bytes.push(255);
                }
            }
        }
    }


    let encoder = image::png::PngEncoder::new(std::fs::File::create(file_name).unwrap());
    encoder.encode(&bytes, w * PIXEL_SIZE as u32, h * PIXEL_SIZE as u32, image::ColorType::Rgba8).unwrap();
}

pub fn save_accum_png(w: u32, h: u32, accum: &[f32], backdrops: &[f32], file_name: &str) {
    let mut bytes = Vec::with_capacity(accum.len()*4);

    for y in 0..h {
        let mut backdrop = backdrops[y as usize];
        for x in 0..w {
            let a = accum[(y * h + x) as usize] + backdrop;
            backdrop = 0.0;

            let v = ((a / 4.0 + 0.5) * 255.0).min(255.0).max(0.0) as u8;
            bytes.push(v);
            bytes.push(v);
            bytes.push(v);
            bytes.push(255);
        }
    }

    let encoder = image::png::PngEncoder::new(std::fs::File::create(file_name).unwrap());
    encoder.encode(&bytes, w, h, image::ColorType::Rgba8).unwrap();
}


#[test]
fn accum_even_odd_01() {
    let src = vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0,-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0,-0.5,-0.5, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.2, 0.6, 0.2, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,

        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];

    let backdrops = [0.0; TILE_SIZE];

    let mut dst = vec![0; 16*16];

    accumulate_even_odd(&src, &backdrops, &mut dst);

    assert_eq!(&dst[0..16], &[255; 16]);

    assert_eq!(dst[16], 0);
    assert_eq!(&dst[17..25], &[255; 8]);
    assert_eq!(&dst[25..32], &[0; 7]);

    assert_eq!(&dst[32..48], &[0, 0, 127, 255, 255, 255, 255, 255,   255, 255, 127, 0, 0, 0, 0, 0]);
    assert_eq!(&dst[48..64], &[0, 0, 51, 204, 255, 255, 255, 255,   255, 255, 255, 255, 255, 255, 255, 255]);
}

#[test]
fn simple_line() {
    let mut accum = [0.0; TILE_SIZE * TILE_SIZE];
    let mut backdrops = [0.0; TILE_SIZE];
    let mut mask = [0; TILE_SIZE * TILE_SIZE];

    let lines = [
        (point2(-3.0, 0.0), point2(5.0, 4.0)),
        (point2(5.0, 4.0), point2(-3.0, 8.0)),
    ];

    for line in &lines {
        if line.0.x < -0.5 && line.0.y != -0.5 {
            add_backdrop(line.0.y, 1.0, &mut backdrops[0..TILE_SIZE]);
        }

        if line.1.x < -0.5 && line.1.y != -0.5 {
            add_backdrop(line.1.y, -1.0, &mut backdrops[0..TILE_SIZE]);
        }

        draw_line(line.0, line.1, &mut accum);
    }

    accumulate_even_odd(&accum, &backdrops, &mut mask);

    save_mask_png(16, 16, &mask, "mask.png");
    save_accum_png(16, 16, &accum, &backdrops, "accum.png");

    println!("backdrops {:?}", backdrops);

    //for y in 0..TILE_SIZE {
    //    println!("{:.2?}", &accum[y* TILE_SIZE .. (y+1) * TILE_SIZE]);
    //}

}

// This a rust port of piet's line rasterization routine, just for the sake of
// understanding and testing the code. There's no reason to run this on the CPU.
pub fn gpu_fill_line(from: Point2D<f32>, to: Point2D<f32>, pos: Point2D<f32>, winding_num: &mut f32) {
    let from = from - pos;
    let to = to - pos;

    // range of the pixel that is in the bounding box of the edge (at most 1.0 since it's a pixel)
    let window = Point2D::new(
        from.y.max(0.0).min(1.0),
        to.y.max(0.0).min(1.0),
    );

    if (window.x - window.y).abs() < 0.001 {
        // pixel no affected.
        return;
    }

    let t = (window - Point2D::new(from.y, from.y)) / (to.y - from.y);
    let xs = Point2D::new(
        from.x * (1.0 - t.x) + to.x * t.x,
        from.x * (1.0 - t.y) + to.x * t.y,
    );
    let xmin: f32 = xs.x.min(xs.y).min(1.0) - 1e-6;
    let xmax: f32 = xs.x.max(xs.y);
    let b: f32 = xmax.min(1.0);
    let c: f32 = b.max(0.0);
    let d: f32 = xmin.max(0.0);
    let area = (b + 0.5 * (d * d - c * c) - xmin) / (xmax - xmin);

    *winding_num += area * (window.x - window.y);
}

#[test]
fn gpu_line_test() {
    let mut mask = [0; TILE_SIZE * TILE_SIZE];

    let lines = [
//        (point2(-0.5, 8.0), point2(8.0, 0.0)),
//        (point2(8.0, 16.0), point2(-0.5, 10.0)),

        (point2(1.0, 1.0), point2(14.0, 2.0)),
        (point2(14.0, 2.0), point2(2.0, 12.0)),
        (point2(2.0, 12.0), point2(1.0, 1.0)),
        //(point2(14.0, 2.0), point2(2.0, 12.0)),
        //(point2(2.0, 12.0), point2(1.0, 1.0)),

        //(point2(-0.5, 12.0), point2(6.0, 8.0)),
        //(point2(6.0, 8.0), point2(-0.5, 4.0)),

        //(point2(8.0, 0.0), point2(15.5, 4.0)),
    ];

    fn even_odd(winding_number: f32) -> f32 {
        1.0 - (winding_number.abs().rem_euclid(2.0) - 1.0).abs()
    }

    for y in 0..TILE_SIZE {
        for x in 0..TILE_SIZE {
            let pos = Point2D::new(x as f32, y as f32);
            let mut winding_number = 0.0;
            for line in &lines {
                gpu_fill_line(line.0, line.1, pos, &mut winding_number);
            }

            assert!(winding_number > -1.1);
            assert!(winding_number < 1.1);

            mask[y * TILE_SIZE + x] = (even_odd(winding_number) * 255.0) as u8
        }
    }

    save_mask_png(16, 16, &mask, "mask.png");

    //for y in 0..TILE_SIZE {
    //    println!("{:.2?}", &accum[y* TILE_SIZE .. (y+1) * TILE_SIZE]);
    //}

}
