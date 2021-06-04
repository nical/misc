use std::f32;
use lyon_path::geom::euclid;
use euclid::default::*;
use euclid::point2;

pub const TILE_SIZE: usize = 16;

pub fn draw_line(from: Point2D<f32>, to: Point2D<f32>, dst: &mut[f32], backdrops: &mut[f32]) {
    // This function rasterizes a line segment in a winding accumulation buffer using the
    // signed area approach for computing antialiasing described in http://nothings.org/gamedev/rasterize/
    //
    // This is heavily inspired by https://github.com/raphlinus/font-rs with the added requirement
    // to handle edges crossing the left boundary of the tile.
    //
    // The values in the accumulation buffer represents the difference between the signed area of
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
    assert!(from.x >= 0.0, "{:?} -> {:?}", from, to);
    assert!(from.y >= 0.0, "{:?} -> {:?}", from, to);
    assert!(from.x <= TILE_SIZE as f32, "{:?} -> {:?}", from, to);
    assert!(from.y <= TILE_SIZE as f32, "{:?} -> {:?}", from, to);
    assert!(to.x >= 0.0, "{:?} -> {:?}", from, to);
    assert!(to.y >= 0.0, "{:?} -> {:?}", from, to);
    assert!(to.x <= TILE_SIZE as f32, "{:?} -> {:?}", from, to);
    assert!(to.y <= TILE_SIZE as f32, "{:?} -> {:?}", from, to);

    let (winding, top, bottom) = if to.y > from.y {
        (1.0, from, to)
    } else {
        (-1.0, to, from)
    };

    let y_min = f32::max(top.y, 0.0) as usize;
    let y_max = f32::min(bottom.y.ceil(), TILE_SIZE as f32) as usize;

    if top.x != bottom.x {
        let mut update_backdrops = false;
        let mut backdrops_y = 0;
        let mut backdrops_winding = 0.0;
        // This is the part where we deal with edges crossing the left tile boundary.
        if bottom.x == 0.0 {
            update_backdrops = true;
            backdrops_y = y_max;
            backdrops_winding = winding;
        } else if top.x == 0.0 {
            update_backdrops = true;
            backdrops_y = y_min;
            backdrops_winding = -winding;
        }

        if update_backdrops {
            for i in backdrops_y..TILE_SIZE {
                backdrops[i] += backdrops_winding;
            }
        }
    }

    if (from.y - to.y).abs() < 0.0001 {
        // horizontal line.
        return;
    }

    let d = bottom - top;
    let dxdy = d.x / d.y;

    //println!("y range {:?} .. {:?}", y_min, y_max);

    let mut x = top.x;

    for y in y_min .. y_max {
        // y range of the edge overlapping the current row of pixels
        let dy = f32::min((y + 1) as f32, bottom.y) - f32::max(y as f32, top.y);

        let next_x = x + dxdy * dy;

        // Winding sign and y coverage combined in a single quantity since we'd have
        // to multiply them both every time.
        let winding_dy = winding * dy;

        let x_range = if next_x > x { x .. next_x } else { next_x .. x };
        let x_range_snapped = x_range.start.floor() .. x_range.end.ceil();
        let x_range_int = x_range_snapped.start as i32 .. x_range_snapped.end as i32;

        //println!(" - x range {:?}", x_range);

        let row = &mut dst[y * TILE_SIZE .. (y + 1) * TILE_SIZE];

        let start_idx = x_range_int.start.max(0) as usize;

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
            row[start_idx] += winding_dy - left_area;
            if start_idx + 1 < TILE_SIZE {
                // Non-covered pixel immediately after the edge.
                row[start_idx + 1] += left_area;
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
            row[start_idx] += winding_dy * cov_left;

            let x_range_int_end = x_range_int.end as usize;

            if x_range_int.len() == 2 {
                if start_idx + 1 < TILE_SIZE {
                    // (D) Pixel on the right side of the edge partially overlapping on the x axis.
                    // Special case for when the edge touches only two pixels on this row because
                    // computing the already contributed coverage is simple.
                    row[start_idx + 1] += winding_dy * (1.0 - cov_left - cov_right);
                }
            } else {

                // (B) Second pixel (fully overlapping the edge)
                let cov_b = inv_dx * (x_left + 0.5);
                row[start_idx + 1] += winding_dy * (cov_b - cov_left);

                // (C) Subsequent pixels fully overlapping with the edge on the x axis.
                // For these the covered area has increased linearly so the coverage difference
                // is constant.
                for i in (start_idx + 2) as usize .. (x_range_int_end - 1) {
                    row[i] += winding_dy * inv_dx;
                }

                if x_range_int_end - 1 < TILE_SIZE {
                    // (D) Pixel on the right side of the edge partially overlapping on the x axis.
                    let cov_c = cov_b + (x_range_int.len() - 3) as f32 * inv_dx;
                    row[x_range_int_end - 1] += winding_dy * (1.0 - cov_c - cov_right);
                }
            }

            if x_range_int.end < TILE_SIZE as i32 {
                // Non-covered pixel immediately after the edge.
                row[x_range_int_end] += winding_dy * cov_right;
            }
        }

        x = next_x;
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
            let a = acc.abs();
            let a = if a < 1.0 { a } else { 1.0 };
            dst[idx] = (a * 255.0) as u8;
            idx += 1;
        }
    }
}


fn save_mask_png(w: u32, h: u32, mask: &[u8], file_name: &str) {
    let mut bytes = Vec::with_capacity(mask.len()*4);

    for a in mask {
        bytes.push(*a);
        bytes.push(*a);
        bytes.push(*a);
        bytes.push(255);
    }

    let encoder = image::png::PNGEncoder::new(std::fs::File::create(file_name).unwrap());
    encoder.encode(&bytes, w, h, image::ColorType::Rgba8).unwrap();
}

fn save_accum_png(w: u32, h: u32, accum: &[f32], backdrops: &[f32], file_name: &str) {
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

    let encoder = image::png::PNGEncoder::new(std::fs::File::create(file_name).unwrap());
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
//        (point2(-0.5, 8.0), point2(8.0, 0.0)),
//        (point2(8.0, 16.0), point2(-0.5, 10.0)),

        (point2(1.0, 1.0), point2(14.0, 2.0)),
        (point2(14.0, 2.0), point2(2.0, 12.0)),
        (point2(2.0, 12.0), point2(1.0, 1.0)),

        //(point2(-0.5, 12.0), point2(6.0, 8.0)),
        //(point2(6.0, 8.0), point2(-0.5, 4.0)),

        //(point2(8.0, 0.0), point2(15.5, 4.0)),
    ];

    for line in &lines {
        draw_line(line.0, line.1, &mut accum, &mut backdrops);
    }

    accumulate_even_odd(&accum, &backdrops, &mut mask);

    save_mask_png(16, 16, &mask, "mask.png");
    save_accum_png(16, 16, &accum, &backdrops, "accum.png");

    //for y in 0..TILE_SIZE {
    //    println!("{:.2?}", &accum[y* TILE_SIZE .. (y+1) * TILE_SIZE]);
    //}


}
