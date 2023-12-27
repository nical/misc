#![allow(unused)]

use core::units::{LocalSpace, SurfaceSpace, SurfaceRect, SurfaceIntRect, Point, vector};
use core::{pattern::BuiltPattern, units::point};

use lyon::geom::{LineSegment, QuadraticBezierSegment, CubicBezierSegment, Box2D};
use lyon::{path::PathEvent, geom::euclid::Transform2D};

use crate::flatten::Flattener;
use crate::{FillOptions, FillRule, Transform, TilePosition};
use crate::occlusion::OcclusionBuffer;

const TILE_SIZE_F32: f32 = 16.0;
const TILE_SIZE: i32 = 16;
const UNITS_PER_TILE: i32 = 256;
const UNITS_PER_TILE_F32: f32 = UNITS_PER_TILE as f32;
const LOCAL_COORD_MASK: i32 = 255;
const LOCAL_COORD_BITS: i32 = 8;
const COORD_SCALE: f32 = UNITS_PER_TILE_F32 / TILE_SIZE_F32;

#[derive(Copy, Clone)]
struct TileInfo {
    tx: i32,
    ty: i32,
    occluded: bool,
    local_x: u8,
    local_y: u8,
}

pub struct Tiler {
    events: Vec<Vec<Event>>,
    tolerance: f32,
    current_tile: (u16, u16),
    current_tile_is_occluded: bool,
    viewport: SurfaceRect,
    scissor: SurfaceRect,
    scissor_tiles: Box2D<i32>,

    // tile_segment2
    prev_tile: Option<TileInfo>,

    // tile_segment
    prev_tx: i32,
    prev_ty: i32,

    //pub dbg: rerun::RecordingStream,
}

const EVENT_BUCKETS: usize = 8;

impl Tiler {
    pub fn new() -> Self {
        let mut events = Vec::with_capacity(EVENT_BUCKETS);
        for _ in 0..EVENT_BUCKETS { events.push(Vec::new())};

        Tiler {
            events,
            tolerance: 0.25,
            current_tile: (0, 0),
            current_tile_is_occluded: false,
            viewport: SurfaceRect {
                min: point(0.0, 0.0),
                max: point(0.0, 0.0),
            },
            scissor: SurfaceRect {
                min: point(0.0, 0.0),
                max: point(0.0, 0.0),
            },
            scissor_tiles: Box2D {
                min: point(0, 0),
                max: point(0, 0),
            },
            prev_tile: None,
            prev_tx: -1,
            prev_ty: -1,
            //dbg: rerun::RecordingStreamBuilder::new("tiler2").spawn().unwrap(),
        }
    }

    #[inline(always)]
    fn events(&mut self, row: usize) -> &mut Vec<Event> {
        &mut self.events[row % EVENT_BUCKETS]
    }

    pub fn begin_target(&mut self, mut viewport: SurfaceIntRect) {
        viewport.min.x = viewport.min.x.max(0);
        viewport.min.y = viewport.min.y.max(0);
        viewport.max.x = viewport.max.x.max(0);
        viewport.max.y = viewport.max.y.max(0);
        self.viewport = viewport.cast();
        self.set_scissor_srect(viewport);
    }

    pub fn set_scissor_srect(&mut self, i32_scissor: SurfaceIntRect) {
        let mut scissor: SurfaceRect = i32_scissor.cast();
        scissor = scissor.intersection_unchecked(&self.viewport);
        self.scissor = scissor;
        self.scissor_tiles = Box2D {
            min: point(
                i32_scissor.min.x / TILE_SIZE,
                i32_scissor.min.y / TILE_SIZE,
            ),
            max: point(
                (scissor.max.x as i32) / TILE_SIZE + (i32_scissor.max.x % TILE_SIZE != 0) as i32,
                (scissor.max.y as i32) / TILE_SIZE + (i32_scissor.max.y % TILE_SIZE != 0) as i32,
            ),
        };
    }

    pub fn get_scissor_rect(&self) -> SurfaceRect {
        self.scissor
    }

    pub fn fill_path(
        &mut self,
        path: impl Iterator<Item = PathEvent>,
        options: &FillOptions,
        pattern: &BuiltPattern,
        output: &mut TilerOutput,
    ) {
        profiling::scope!("Tiler::fill_path");
        //println!("fill path viewport {:?}, scissor {:?}, tiles {:?}", self.viewport, self.scissor, self.scissor_tiles);

        let identity = Transform2D::identity();
        let transform = options.transform.unwrap_or(&identity);

        self.tile_path(path, transform, &mut output.occlusion);

        let mut encoded_fill_rule = match options.fill_rule {
            FillRule::EvenOdd => 0,
            FillRule::NonZero => 1,
        };
        if options.inverted {
            encoded_fill_rule |= 2;
        }

        output.paths.push(PathInfo {
            z_index: options.z_index,
            pattern_data: pattern.data,
            fill_rule: encoded_fill_rule,
            opacity: options.opacity,
            scissor: [
                self.scissor.min.x as u32,
                self.scissor.min.y as u32,
                self.scissor.max.x as u32,
                self.scissor.max.y as u32,
            ],
        }.encode());

        self.generate_tiles(options.fill_rule, options.inverted, pattern, output);
    }

    // Similar to tile_path_nested, except that it flattens curves into a buffer before
    // processing the line segments in bulk, instead of nesting the binning into
    // the flattening loop.
    // In its current states it doesn not appear to improve performance althogh
    // it makes profiles a bit easier to read.
    fn tile_path(
        &mut self,
        path: impl Iterator<Item = PathEvent>,
        transform: &Transform,
        occlusion: &mut OcclusionBuffer,
    ) {
        //println!("\n\n--------");
        profiling::scope!("Tiler::tile_path");

        let transform: &lyon::geom::Transform<f32> = unsafe {
            std::mem::transmute(transform)
        };
        for array in &mut self.events {
            array.clear();
        }

        // Keep track of from manually instead of using the value provided by the
        // iterator because we want to skip tiny edges without intorducing gaps.
        let mut from = point(0.0, 0.0);
        let square_tolerance = self.tolerance * self.tolerance;

        let mut point_buffer = Vec::with_capacity(512);
        let mut flattener = Flattener::new(self.tolerance);
        let mut force_flush = false;
        let mut skipped = None;

        for evt in path {
            match evt {
                PathEvent::Begin { at } => {
                    //println!("# begin {at:?}");
                    point_buffer.push(transform.transform_point(at));
                    from = at;
                }
                PathEvent::End { first, .. } => {
                    //println!("# end first={first:?}");
                    flattener.set_line(transform.transform_point(first));
                    force_flush = true;
                }
                PathEvent::Line { to, .. } => {
                    //println!("# line to={to:?}");
                    let segment = LineSegment { from, to }.transformed(transform);
                    let min_y = segment.from.y.min(segment.to.y);
                    let max_y = segment.from.y.max(segment.to.y);
                    if min_y > self.scissor.max.y + 1.0 || max_y < self.scissor.min.y - 1.0 {
                        from = to;
                        skipped = Some(to);
                        //println!("skip line segment");
                        continue;
                    }

                    if segment.to_vector().square_length() < square_tolerance {
                        continue;
                    }

                    flattener.set_line(transform.transform_point(to));
                    from = to;
                }
                PathEvent::Quadratic { ctrl, to, .. } => {
                    //println!("# quad ctrl= {ctrl:?} to={to:?}");
                    let segment = QuadraticBezierSegment { from, ctrl, to }.transformed(transform);
                    let min_y = segment.from.y.min(segment.ctrl.y).min(segment.to.y);
                    let max_y = segment.from.y.max(segment.ctrl.y).max(segment.to.y);
                    if min_y > self.scissor.max.y + 1.0 || max_y < self.scissor.min.y - 1.0 {
                        from = to;
                        skipped = Some(to);
                        //println!("skip quad segment");
                        continue;
                    }
                    if segment.baseline().to_vector().square_length() < square_tolerance {
                        let center = (segment.from + segment.to.to_vector()) * 0.5;
                        if (segment.ctrl - center).square_length() < square_tolerance {
                            continue;
                        }
                    }

                    flattener.set_quadratic(&segment);
                    from = to;
                }
                PathEvent::Cubic { ctrl1, ctrl2, to, .. } => {
                    //println!("# cubic from={from:?} ctrl1={ctrl1:?} ctrl2={ctrl2:?} to={to:?}");
                    let segment = CubicBezierSegment {
                        from,
                        ctrl1,
                        ctrl2,
                        to,
                    }
                    .transformed(transform);

                    let min_y = segment.from.y.min(segment.ctrl1.y).min(segment.ctrl2.y).min(segment.to.y);
                    let max_y = segment.from.y.max(segment.ctrl1.y).max(segment.ctrl2.y).max(segment.to.y);
                    if min_y > self.scissor.max.y + 1.0 || max_y < self.scissor.min.y - 1.0 {
                        from = to;
                        skipped = Some(to);
                        //println!("skip cubic segment");
                        continue;
                    }

                    if segment.baseline().to_vector().square_length() < square_tolerance {
                        let center = (segment.from + segment.to.to_vector()) * 0.5;
                        if (segment.ctrl1 - center).square_length() < square_tolerance
                            && (segment.ctrl2 - center).square_length() < square_tolerance
                        {
                            continue;
                        }
                    }

                    flattener.set_cubic(&segment);
                    from = to;
                }
            }


            if !flattener.is_done() {
                if let Some(pos) = skipped {
                    //println!("push {pos:?} after skipped segment(s)");
                    point_buffer.push(transform.transform_point(pos));
                    skipped = None;
                }
            }

            while flattener.flatten(&mut point_buffer) || force_flush || point_buffer.capacity() - point_buffer.len() == 0 {
                self.flush(&mut point_buffer, occlusion, force_flush);
                force_flush = false;
            }
        }
    }

    fn flush(&mut self, point_buffer: &mut Vec<Point>, occlusion: &mut OcclusionBuffer, is_last: bool) {
        profiling::scope!("Tiler::flush");

        if point_buffer.len() < 2 {
            return;
        }
        //println!("flush {point_buffer:?}");
        let mut iter = point_buffer.iter();
        let mut from = *iter.next().unwrap();

        let (tx, ty) = self.tile_for_position(from);
        self.prev_tx = tx;
        self.prev_ty = ty;

        for to in iter {
            self.tile_segment(&LineSegment { from, to: *to });
            from = *to;
        }

        point_buffer.clear();
        if !is_last {
            point_buffer.push(from);
        }
    }

    fn tile_segment(&mut self, edge: &LineSegment<f32>) {
        // Leave some margin around this early scissor test so that
        // we keep track of the previous tile near the boundary of the
        // scissor rect.
        let min_y = edge.to.y.min(edge.from.y);
        let max_y = edge.to.y.max(edge.from.y);
        let min_x = edge.to.x.min(edge.from.x);
        if min_y > self.scissor.max.y + 1.0
            || max_y < self.scissor.min.y - 1.0
            || min_x > self.scissor.max.x + 1.0
        {
            return;
        }

        let from_i32_x = (edge.from.x * COORD_SCALE) as i32;
        let from_i32_y = (edge.from.y * COORD_SCALE) as i32;
        let to_i32_x = (edge.to.x * COORD_SCALE) as i32;
        let to_i32_y = (edge.to.y * COORD_SCALE) as i32;
        let src_tx = (from_i32_x >> LOCAL_COORD_BITS);
        let src_ty = (from_i32_y >> LOCAL_COORD_BITS);
        let dst_tx = (to_i32_x >> LOCAL_COORD_BITS);
        let dst_ty = (to_i32_y >> LOCAL_COORD_BITS);
        let scissor_min_tx = self.scissor_tiles.min.x;
        let scissor_min_ty = self.scissor_tiles.min.y;
        let scissor_max_tx = self.scissor_tiles.max.x - 1;
        let scissor_max_ty = self.scissor_tiles.max.y - 1;

        let dx = edge.to.x - edge.from.x;
        let dy = edge.to.y - edge.from.y;
        let dx_dy = dx/dy;
        let dy_dx = 1.0 / dx_dy;

        let x_step = (dst_tx - src_tx).signum();
        let y_step = (dst_ty - src_ty).signum();
        let h_tile_side = x_step.max(0);
        let v_tile_side = y_step.max(0);

        let mut v_x0 = edge.from.x;
        let mut v_y0 = edge.from.y;

        //println!("\ntile_segment {edge:?}, tiles {src_tx} {src_ty} -> {dst_tx} {dst_ty}, step {x_step} {y_step}");

        // x and y index of the current tile.
        let mut tx = src_tx;
        let mut ty = src_ty;
        loop {
            let v_y1;
            let v_x1;
            let h_dst_tx;
            if ty == dst_ty {
                v_y1 = edge.to.y;
                v_x1 = edge.to.x;
                h_dst_tx = dst_tx;
            } else {
                v_y1 = (ty + v_tile_side) as f32 * TILE_SIZE_F32;
                v_x1 = v_x0 + dx_dy * (v_y1 - v_y0);
                h_dst_tx = if x_step == 0 {
                    dst_tx
                } else {
                    (v_x1 * COORD_SCALE) as i32 >> LOCAL_COORD_BITS
                };
            }

            let row_occluded = ty < self.scissor_tiles.min.y || ty >= self.scissor_tiles.max.y;

            // When moving to a different row of tiles, add backdrop events.
            // Note: this relies not only on the input and locals of this function,
            // but also on the assumption that we correctly keep track of self.prev_ty
            // inside and outside of this function.
            if ty != self.prev_ty {
                let positive = ty > self.prev_ty;
                let row = ty + (!positive) as i32;
                if row >= self.scissor_tiles.min.y && row < self.scissor_tiles.max.y {
                    //println!("    - backdrop {} {row}, positive:{positive}", (tx + 1).max(0));
                    self.events(row as usize).push(Event::backdrop((tx + 1).max(0) as u16, row as u16, positive));
                }
                self.prev_ty = ty;
            }

            let mut h_x0 = v_x0;
            let mut h_y0 = v_y0;
            //println!(" - row {ty}, sub ({v_x0} {v_y0}) -> ({v_x1} {v_y1}), tiles x {tx} {h_dst_tx}");

            let mut events = &mut self.events[ty as usize % EVENT_BUCKETS];

            loop {
                let h_x1;
                let h_y1;
                if tx == h_dst_tx {
                    h_x1 = v_x1;
                    h_y1 = v_y1;
                } else {
                    debug_assert!(x_step != 0, "tx {tx}, h_dst_tx {h_dst_tx}, src_tx {src_tx} dst_tx {dst_tx}, segment {edge:?}");
                    h_x1 = (tx + h_tile_side) as f32 * TILE_SIZE_F32;
                    h_y1 = v_y0 + dy_dx * (h_x1 - v_x0);
                }

                let occluded = row_occluded || tx < self.scissor_tiles.min.x || tx >= self.scissor_tiles.max.x;

                if !row_occluded {
                    let offset_y = (ty * UNITS_PER_TILE) as f32;
                    let local_y0 = ((h_y0 * COORD_SCALE) - offset_y).min(255.0) as u8;

                    if tx >= self.scissor_tiles.min.x {
                        let offset_x = (tx * UNITS_PER_TILE) as f32;
                        let local_x0 = ((h_x0 * COORD_SCALE) - offset_x).min(255.0) as u8;
                        let local_x1 = ((h_x1 * COORD_SCALE) - offset_x).min(255.0) as u8;
                        let local_y1 = ((h_y1 * COORD_SCALE) - offset_y).min(255.0) as u8;

                        if !occluded {
                            assert!(tx < self.scissor_tiles.max.x);
                            //println!("   - edge tile {tx} {ty}, step {x_step}, [{h_x0}, {h_y0}, {h_x1}, {h_y1}], offset {offset_x} {offset_y} local {:?}", [local_x0, local_y0, local_x1, local_y1]);
                            events.push(Event::edge(
                                tx as u16, ty as u16,
                                [local_x0, local_y0, local_x1, local_y1]
                            ));
                        }
                    }

                    // Add an auxiliary edge when crossing a vertical boundary (tile x
                    // coordinate changes).
                    if tx != self.prev_tx {
                        // Note, if the edge is going in the ngative x orientation, then
                        // we are actually adding an auxiliary edge to the previous tile
                        // rather than the current one.
                        let aux_tx = tx.max(self.prev_tx);
                        let occluded = aux_tx < self.scissor_tiles.min.x || aux_tx >= self.scissor_tiles.max.x;

                        if !occluded {
                            let (y0, y1) = if tx < self.prev_tx {
                                (local_y0, 255)
                            } else {
                                (255, local_y0)
                            };
                            //println!("   - auxiliary edge tile {aux_tx} {ty}, y-range {y0} {y1}");
                            assert!(aux_tx < self.scissor_tiles.max.x);
                            events.push(Event::edge(
                                aux_tx as u16, ty as u16,
                                [0, y0, 0, y1]
                            ));
                        }
                    }
                }

                self.prev_tx = tx;

                if tx == h_dst_tx {
                    break;
                }
                h_x0 = h_x1;
                h_y0 = h_y1;
                tx += x_step;
            }

            v_x0 = v_x1;
            v_y0 = v_y1;
            if ty == dst_ty {
                break;
            }
            ty += y_step;
        }
    }


    fn tile_path_nested(
        &mut self,
        path: impl Iterator<Item = PathEvent>,
        transform: &Transform,
        occlusion: &mut OcclusionBuffer,
    ) {
        //println!("\n\n-------");
        profiling::scope!("Tiler::tile_path");
        let transform: &lyon::geom::Transform<f32> = unsafe {
            std::mem::transmute(transform)
        };
        for array in &mut self.events {
            array.clear();
        }

        // Keep track of from manually instead of using the value provided by the
        // iterator because we want to skip tiny edges without intorducing gaps.
        let mut from = point(0.0, 0.0);
        let square_tolerance = self.tolerance * self.tolerance;

        for evt in path {
            match evt {
                PathEvent::Begin { at } => {
                    from = at;
                }
                PathEvent::End { first, .. } => {
                    let segment = LineSegment { from, to: first }.transformed(transform);
                    self.tile_segment_dda(&segment, occlusion);
                    from = first;
                }
                PathEvent::Line { to, .. } => {
                    let segment = LineSegment { from, to }.transformed(transform);
                    if segment.to_vector().square_length() < square_tolerance {
                        continue;
                    }
                    self.tile_segment_dda(&segment, occlusion);
                    from = to;
                }
                PathEvent::Quadratic { ctrl, to, .. } => {
                    let segment = QuadraticBezierSegment { from, ctrl, to }.transformed(transform);
                    if segment.baseline().to_vector().square_length() < square_tolerance {
                        let center = (segment.from + segment.to.to_vector()) * 0.5;
                        if (segment.ctrl - center).square_length() < square_tolerance {
                            continue;
                        }
                    }
                    crate::flatten::flatten_quad(&segment, self.tolerance, &mut |segment| {
                        // segment.for_each_flattened(self.draw.tolerance, &mut|segment| {
                        self.tile_segment_dda(segment, occlusion);
                    });
                    from = to;
                }
                PathEvent::Cubic {
                    ctrl1, ctrl2, to, ..
                } => {
                    let segment = CubicBezierSegment {
                        from,
                        ctrl1,
                        ctrl2,
                        to,
                    }
                    .transformed(transform);
                    if segment.baseline().to_vector().square_length() < square_tolerance {
                        let center = (segment.from + segment.to.to_vector()) * 0.5;
                        if (segment.ctrl1 - center).square_length() < square_tolerance
                            && (segment.ctrl2 - center).square_length() < square_tolerance
                        {
                            continue;
                        }
                    }
                    crate::flatten::flatten_cubic(&segment, self.tolerance, &mut |segment| {
                        self.tile_segment_dda(segment, occlusion);
                    });
                    from = to;
                }
            }
        }
    }

    fn tile_segment_dda(&mut self, segment: &LineSegment<f32>, occlusion: &mut OcclusionBuffer) {
        //println!("\nbin {segment:?} ({:?})", segment.to_vector() / TILE_SIZE_F32);
        // Cull above and below the viewport
        let min_y = segment.from.y.min(segment.to.y);
        let max_y = segment.from.y.max(segment.to.y);
        if max_y < self.scissor.min.y || min_y > self.scissor.max.y {
            return;
        }

        // Cull right of the viewport
        let min_x = segment.from.x.min(segment.to.x);
        let max_x = segment.from.x.max(segment.to.x);
        if min_x > self.scissor.max.x {
            return;
        }

        // In number of tiles

        // Cull left of the viewport
        if max_x < self.scissor.min.x {
            let quantized_min_y = (min_y * COORD_SCALE) as i32;
            let quantized_max_y = (max_y * COORD_SCALE) as i32;
            let min_ty = quantized_min_y / UNITS_PER_TILE;
            let max_ty = quantized_max_y / UNITS_PER_TILE;
            let local_min_y_u8 = quantized_min_y.max(0) as u8;
            let local_max_y_u8 = quantized_max_y.max(0) as u8;

            // Backdrops are still affected by content on the left of the viewport.
            let positive_winding = segment.to.y > segment.from.y;
            let mut y_range = min_ty..max_ty;
            if local_min_y_u8 != 0 {
                y_range.start += 1;
            }
            if local_max_y_u8 != 0 {
                y_range.end += 1;
            }

            //println!(" - left backdrops for y range {y_range:?} ({local_min_y_u8})");

            for y in y_range {
                self.events(y as usize).push(Event::backdrop(0, y as u16, positive_winding));
            }

            return;
        }

        // DDA-ish walk over the tiles that the edge touches

        // In number of tiles
        let quantized_x0 = (segment.from.x * COORD_SCALE) as i32;
        let quantized_y0 = (segment.from.y * COORD_SCALE) as i32;
        let quantized_x1 = (segment.to.x * COORD_SCALE) as i32;
        let quantized_y1 = (segment.to.y * COORD_SCALE) as i32;
        let src_tx = quantized_x0 >> LOCAL_COORD_BITS;
        let src_ty = quantized_y0 >> LOCAL_COORD_BITS;
        let dst_tx = quantized_x1 >> LOCAL_COORD_BITS;
        let dst_ty = quantized_y1 >> LOCAL_COORD_BITS;

        let dx_sign = (dst_tx - src_tx).signum();
        let dy_sign = (dst_ty - src_ty).signum();

        // In pixels, local to the current tile.
        let mut local_x0_u8 = (quantized_x0.max(0) & LOCAL_COORD_MASK) as u8;
        let mut local_y0_u8 = (quantized_y0.max(0) & LOCAL_COORD_MASK) as u8;

        // Specialzie the simple and rather common case where there is a single
        // edge in the tile and it does not touch the left ot top sides of the tile.
        // TODO: It's unclear whether this actually helps.
        if src_tx == dst_tx && src_ty == dst_ty && local_x0_u8 != 0 && local_y0_u8 != 0 && src_tx >= 0 {
            let local_x1_u8 = (quantized_x1.max(0) & LOCAL_COORD_MASK) as u8;
            let local_y1_u8 = (quantized_y1.max(0) & LOCAL_COORD_MASK) as u8;
            if local_x1_u8 != 0 && local_y1_u8 != 0 {
                let tile = (src_tx as u16, src_ty as u16);
                if tile != self.current_tile {
                    self.current_tile_is_occluded = occlusion.occluded(tile.0, tile.1);
                    self.current_tile = tile;
                }

                if !self.current_tile_is_occluded {
                    self.events(tile.1 as usize).push(Event::edge(
                        tile.0, tile.1,
                        [local_x0_u8, local_y0_u8, local_x1_u8, local_y1_u8]
                    ));
                }

                return;
            }
        }

        let inv_segment_vx = 1.0 / (segment.to.x - segment.from.x);
        let inv_segment_vy = 1.0 / (segment.to.y - segment.from.y);

        let next_tile_x = (inv_segment_vx > 0.0) as i32;
        let next_tile_y = (inv_segment_vy > 0.0) as i32;
        let mut t_crossing_x = (((src_tx + next_tile_x) as f32 * TILE_SIZE_F32 - segment.from.x) * inv_segment_vx).abs();
        let mut t_crossing_y = (((src_ty + next_tile_y) as f32 * TILE_SIZE_F32 - segment.from.y) * inv_segment_vy).abs();
        let t_delta_x = (TILE_SIZE_F32 * inv_segment_vx).abs();
        let t_delta_y = (TILE_SIZE_F32 * inv_segment_vy).abs();

        let mut tx = src_tx;
        let mut ty = src_ty;

        //println!("tiles {tx} {ty} -> {dst_tx} {dst_ty}, sign {dx_sign} {dy_sign},  t_delta {t_delta_x} {t_delta_y} start {local_x0_u8} {local_y0_u8}");

        loop {
            //assert!(idx < 100, "{segment:?}, {tx} {ty} ({src_tx} {src_ty} -> {dst_tx} {dst_ty})");

            let tcx = t_crossing_x;
            let tcy = t_crossing_y;
            let mut t_split = tcx;
            let mut step_x = 0;
            let mut step_y = 0;
            if tcx <= tcy {
                t_split = t_crossing_x;
                t_crossing_x += t_delta_x;
                step_x = dx_sign;
            }
            if tcy <= tcx {
                t_split = t_crossing_y;
                t_crossing_y += t_delta_y;
                step_y = dy_sign;
            };

            //println!(" * tile {tx} {ty}, crossing {tcx} {tcy} -> {t_split}");
            let t_split = t_split.min(1.0);
            let one_t_split = 1.0 - t_split;
            let x1 = segment.from.x * one_t_split + segment.to.x * t_split;
            let y1 = segment.from.y * one_t_split + segment.to.y * t_split;
            let local_x1_u8 = ((x1 * COORD_SCALE) as i32 - tx * UNITS_PER_TILE).max(0).min(255) as u8;
            let local_y1_u8 = ((y1 * COORD_SCALE) as i32 - ty * UNITS_PER_TILE).max(0).min(255) as u8;

            //println!("            local {x1} {y1}| u8: {local_x0_u8} {local_y0_u8} {local_x1_u8} {local_y1_u8}");
            //println!("            crossings: h0 {h_crossing_0} h1 {h_crossing_1} v0 {v_crossing_0} v1 {v_crossing_1}");

            // discard all tiles that are above, below or right of the view.
            let affects_view = ty >= self.scissor_tiles.min.y
                && ty < self.scissor_tiles.max.y
                && tx < self.scissor_tiles.max.x;

            if affects_view {
                let h_crossing_0 = local_y0_u8 == 0;
                let h_crossing_1 = local_y1_u8 == 0;
                let v_crossing_0 = local_x0_u8 == 0;
                let v_crossing_1 = local_x1_u8 == 0;
                let on_left_side = v_crossing_0 && v_crossing_1;

                // If either but not both endpoints are on the top side of the tile,
                // deal with an horizontal crossings.
                if h_crossing_0 ^ h_crossing_1 {
                    // The tile's x coordinate could be negative if the segment is partially
                    // out of the viewport.
                    // No need to clamp y to positive numbers here because the viewport_tiles
                    // check filters out all tiles with negative y.
                    let x = (tx + (!on_left_side) as i32).max(0);
                    self.events(ty as usize).push(Event::backdrop(x as u16, ty as u16, h_crossing_0));
                    //println!("   - backdrop {x} {ty} | {}", if h_crossing_0 { 1 } else { -1 });
                }

                if !on_left_side && tx >= self.scissor_tiles.min.x {
                    // The viewport can only contain positive coordinates.
                    let tile_x = tx as u16;
                    let tile_y = ty as u16;

                    if (tile_x, tile_y) != self.current_tile {
                        self.current_tile_is_occluded = occlusion.occluded(tile_x, tile_y);
                        self.current_tile = (tile_x, tile_y);
                    }

                    if !self.current_tile_is_occluded {
                        if local_y0_u8 != local_y1_u8 {
                            //println!("   - edge {tx} {ty} | {local_x0_u8} {local_y0_u8}  {local_x1_u8} {local_y1_u8} | {x1} {y1}");
                            self.events(tile_y as usize).push(Event::edge(
                                tile_x, tile_y,
                                [local_x0_u8, local_y0_u8, local_x1_u8, local_y1_u8]
                            ));
                        }

                        // Whether one of the endpoints is exactly on the top-left corner.
                        let on_corner0 = v_crossing_0 && h_crossing_0;
                        let on_corner1 = v_crossing_1 && h_crossing_1;
                        // Add auxiliary edges when an edge crosses the left side.
                        // When either (but not both) endpoints lie at the left boundary
                        // and the segment is not on the top of the tile:
                        if (v_crossing_0 ^ v_crossing_1) && !(on_corner0 || on_corner1) {
                            let auxiliary_edge = if v_crossing_0 {
                                [0, 255, 0, local_y0_u8]
                            } else {
                                [0, local_y1_u8, 0, 255]
                            };
                            //println!("   - auxiliary edge {tile_x} {tile_y} | {auxiliary_edge:?}");
                            self.events(tile_y as usize).push(Event::edge(tile_x, tile_y, auxiliary_edge));
                        }
                    }
                }
            }

            if (tx == dst_tx && ty == dst_ty) || t_split >= 1.0 {
                break;
            }

            tx += step_x;
            ty += step_y;

            local_x0_u8 = if step_x > 0 { 0 } else if step_x < 0 { 255 } else { local_x1_u8 };
            local_y0_u8 = if step_y > 0 { 0 } else if step_y < 0 { 255 } else { local_y1_u8 };
        }
    }

    fn tile_for_position(&mut self, pos: Point) -> (i32, i32) {
        let x_i32 = (pos.x * COORD_SCALE) as i32;
        let y_i32 = (pos.y * COORD_SCALE) as i32;
        let tx = (x_i32 >> LOCAL_COORD_BITS);
        let ty = (y_i32 >> LOCAL_COORD_BITS);

        (tx, ty)
    }

    /*
    fn tile_segment_dda2(&mut self, segment: &LineSegment<f32>) {
        //println!("\nbin {segment:?} ({:?})", segment.to_vector() / TILE_SIZE_F32);
        // Cull above and below the viewport
        let min_y = segment.from.y.min(segment.to.y);
        let max_y = segment.from.y.max(segment.to.y);
        if max_y < self.scissor.min.y || min_y > self.scissor.max.y {
            return;
        }

        // Cull right of the viewport
        let min_x = segment.from.x.min(segment.to.x);
        let max_x = segment.from.x.max(segment.to.x);
        if min_x > self.scissor.max.x {
            return;
        }

        // Cull left of the viewport
        if max_x < self.scissor.min.x {
            let quantized_min_y = (min_y * COORD_SCALE) as i32;
            let quantized_max_y = (max_y * COORD_SCALE) as i32;
            let min_ty = quantized_min_y / UNITS_PER_TILE;
            let max_ty = quantized_max_y / UNITS_PER_TILE;
            let local_min_y_u8 = quantized_min_y.max(0) as u8;
            let local_max_y_u8 = quantized_max_y.max(0) as u8;

            // Backdrops are still affected by content on the left of the viewport.
            let positive_winding = segment.to.y > segment.from.y;
            let mut y_range = min_ty..max_ty;
            if local_min_y_u8 != 0 {
                y_range.start += 1;
            }
            if local_max_y_u8 != 0 {
                y_range.end += 1;
            }

            //println!(" - left backdrops for y range {y_range:?} ({local_min_y_u8})");

            for y in y_range {
                self.events[y as usize % 32].push(Event::backdrop(0, y as u16, positive_winding));
            }

            return;
        }

        // DDA-ish walk over the tiles that the edge touches

        let current = self.prev_tile.get_or_insert_with(|| {
            let quantized_x0 = (segment.from.x * COORD_SCALE) as i32;
            let quantized_y0 = (segment.from.y * COORD_SCALE) as i32;
            let tx = quantized_x0 >> LOCAL_COORD_BITS;
            let ty = quantized_y0 >> LOCAL_COORD_BITS;
            let mut local_x = (quantized_x0.max(0) & LOCAL_COORD_MASK) as u8;
            let mut local_y = (quantized_y0.max(0) & LOCAL_COORD_MASK) as u8;

            let occluded = ty < self.scissor_tiles.min.y
                || ty >= self.scissor_tiles.max.y
                || tx < self.scissor_tiles.min.x
                || tx >= self.scissor_tiles.max.x
                || self.occlusion.occluded(tx as u16, ty as u16);

            TileInfo {
                tx,
                ty,
                occluded,
                local_x,
                local_y,
            }
        });

        let quantized_x1 = (segment.to.x * COORD_SCALE) as i32;
        let quantized_y1 = (segment.to.y * COORD_SCALE) as i32;
        let dst_tx = quantized_x1 >> LOCAL_COORD_BITS;
        let dst_ty = quantized_y1 >> LOCAL_COORD_BITS;

        let dx_sign = (dst_tx - current.tx).signum();
        let dy_sign = (dst_ty - current.ty).signum();

        let inv_segment_vx = 1.0 / (segment.to.x - segment.from.x);
        let inv_segment_vy = 1.0 / (segment.to.y - segment.from.y);

        let next_tile_x = (inv_segment_vx > 0.0) as i32;
        let next_tile_y = (inv_segment_vy > 0.0) as i32;
        let mut t_crossing_x = (((current.tx + next_tile_x) as f32 * TILE_SIZE_F32 - segment.from.x) * inv_segment_vx).abs();
        let mut t_crossing_y = (((current.ty + next_tile_y) as f32 * TILE_SIZE_F32 - segment.from.y) * inv_segment_vy).abs();
        let t_delta_x = (TILE_SIZE_F32 * inv_segment_vx).abs();
        let t_delta_y = (TILE_SIZE_F32 * inv_segment_vy).abs();

        let mut tx = current.tx;
        let mut ty = current.ty;

        //println!("tiles {tx} {ty} -> {dst_tx} {dst_ty}, sign {dx_sign} {dy_sign},  t_delta {t_delta_x} {t_delta_y} start {local_x0_u8} {local_y0_u8}");

        let mut prev_tile_occluded = current.occluded;
        loop {
            //assert!(idx < 100, "{segment:?}, {tx} {ty} ({src_tx} {src_ty} -> {dst_tx} {dst_ty})");

            let tcx = t_crossing_x;
            let tcy = t_crossing_y;
            let mut t_split = tcx;
            let mut step_x = 0;
            let mut step_y = 0;
            if tcx <= tcy {
                t_split = t_crossing_x;
                t_crossing_x += t_delta_x;
                step_x = dx_sign;
            }

            if tcy <= tcx {
                t_split = t_crossing_y;
                t_crossing_y += t_delta_y;
                step_y = dy_sign;
            };

            //println!(" * tile {tx} {ty}, crossing {tcx} {tcy} -> {t_split}");
            let t_split = t_split.min(1.0);
            let one_t_split = 1.0 - t_split;
            let x1 = segment.from.x * one_t_split + segment.to.x * t_split;
            let y1 = segment.from.y * one_t_split + segment.to.y * t_split;
            let local_x1_u8 = ((x1 * COORD_SCALE) as i32 - tx * UNITS_PER_TILE).max(0).min(255) as u8;
            let local_y1_u8 = ((y1 * COORD_SCALE) as i32 - ty * UNITS_PER_TILE).max(0).min(255) as u8;

            //println!("            local {x1} {y1}| u8: {local_x0_u8} {local_y0_u8} {local_x1_u8} {local_y1_u8}");
            //println!("            crossings: h0 {h_crossing_0} h1 {h_crossing_1} v0 {v_crossing_0} v1 {v_crossing_1}");

            // Discard all tiles that are above, below or right of the view.
            let affects_view = ty >= self.scissor_tiles.min.y
                && ty < self.scissor_tiles.max.y
                && tx < self.scissor_tiles.max.x;

            if affects_view {
                if current.ty != ty {
                    // The tile's x coordinate could be negative if the segment is partially
                    // out of the viewport.
                    // No need to clamp y to positive numbers here because the viewport_tiles
                    // check filters out all tiles with negative y.
                    let positive = current.ty < ty;
                    let x = (tx + 1).max(0) as u16;
                    let y = ty as u16 + (!positive) as u16;
                    self.events.push(Event::backdrop(x, y, positive));
                    //println!("   - backdrop {x} {ty} | {}", if h_crossing_0 { 1 } else { -1 });
                }

                if tx >= self.scissor_tiles.min.x {
                    if !current.occluded {
                        // If edge is not horizontal, add it.
                        if current.local_y != local_y1_u8 {
                            //println!("   - edge {tx} {ty} | {local_x0_u8} {local_y0_u8}  {local_x1_u8} {local_y1_u8} | {x1} {y1}");
                            self.events.push(Event::edge(
                                current.tx as u16, current.ty as u16,
                                [current.local_x, current.local_y, local_x1_u8, local_y1_u8]
                            ));
                        }

                        // Add auxiliary edges when an edge crosses the left side.
                        // When either (but not both) endpoints lie at the left boundary
                        // and the segment is not on the top of the tile:
                        if tx != current.tx {
                            let auxiliary_edge = if tx > current.tx {
                                [0, 255, 0, current.local_y]
                            } else {
                                [0, local_y1_u8, 0, 255]
                            };
                            let x = tx as u16 + (tx < current.tx) as u16;
                            self.events.push(Event::edge(x, ty as u16, auxiliary_edge));
                        }
                    }
                }
            }

            current.local_x = if step_x > 0 { 0 } else if step_x < 0 { 255 } else { local_x1_u8 };
            current.local_y = if step_y > 0 { 0 } else if step_y < 0 { 255 } else { local_y1_u8 };

            if step_x != 0 || step_y != 0 {
                current.tx += step_x;
                current.ty += step_y;
                prev_tile_occluded = current.occluded;
                current.occluded = affects_view
                    && tx >= self.scissor_tiles.min.x
                    && self.occlusion.occluded(current.tx as u16, current.ty as u16);
            } else {
                break;
            }

            if t_split >= 1.0 {
                break;
            }
        }
    }
*/
    fn generate_tiles(&mut self, fill_rule: FillRule, inverted: bool, pattern: &BuiltPattern, output: &mut TilerOutput) {
        profiling::scope!("Tiler::generate_tiles");
        //println!("Tiler::generate_tiles");

        let path_index = output.paths.len() as u32 - 1;

        for events in &mut self.events {
            if events.is_empty() {
                continue;
            }

            events.sort_unstable_by_key(|e| e.sort_key);
            // Push a dummy backdrop out of view that will cause the current tile to be flushed
            // at the last iteration without having to replicate the logic out of the loop.
            events.push(Event::backdrop(0, std::u16::MAX, false));

            //for e in events.iter() {
            //    let (tx, ty) = e.tile();
            //    let edge: [u8; 4] = unsafe { std::mem::transmute(e.payload) };
            //    if e.is_edge() {
            //        println!("- {tx} {ty} edge {edge:?}");
            //    } else {
            //        println!("- {tx} {ty} backdrop {}", if e.payload == 0 { -1 } else { 1 });
            //    }
            //}

            let mut current_tile = (0, 0);
            let mut tile_first_edge = output.edges.len();
            let mut backdrop: i16 = 0;

            for evt in events.iter() {
                let tile = evt.tile();
                //if evt.is_edge() {
                //    println!("   * edge {tile:?}");
                //} else {
                //    println!("   * backdrop {tile:?}");
                //}
                if tile != current_tile {
                    let mut x = current_tile.0;
                    let y = current_tile.1;

                    let x_end = if tile.1 == y {
                        tile.0.min(self.scissor_tiles.max.x as u16)
                    } else {
                        self.scissor_tiles.max.x as u16
                    };

                    //if tile.1 != current_tile.1 {
                    //    println!("");
                    //}
                    //println!("* new tile {tile:?} (was {current_tile:?}, backdrop: {backdrop:?}");
                    let tile_last_edge = output.edges.len();
                    if tile_last_edge != tile_first_edge {
                        // This limit isn't necessary but it is useful to catch bugs. In practice there should
                        // never be this many edges in a single tile.
                        const MAX_EDGES_PER_TILE: usize = 512;
                        debug_assert!(tile_last_edge - tile_first_edge < MAX_EDGES_PER_TILE, "bad tile at {current_tile:?}, edges {tile_first_edge} {tile_last_edge}");

                        if !output.occlusion.occluded(x, y) {
                            assert!(x < x_end, "trying to push a mask tile out of bounds at {current_tile:?} scissor: {:?}, event is edge: {:?}", self.scissor_tiles, evt.is_edge());
                            //println!("      * encode tile {} {}, with {} edges, backdrop {backdrop}", current_tile.0, current_tile.1, tile_last_edge - tile_first_edge);
                            output.mask_tiles.push(TileInstance {
                                position: TilePosition::new(x as u32, y as u32),
                                backdrop,
                                first_edge: tile_first_edge as u32,
                                edge_count: usize::min(tile_last_edge - tile_first_edge, MAX_EDGES_PER_TILE) as u16,
                                path_index,
                            }.encode());
                        }

                        tile_first_edge = tile_last_edge;
                        x += 1;
                    }

                    let inside = inverted ^ match fill_rule {
                        FillRule::EvenOdd => backdrop % 2 != 0,
                        FillRule::NonZero => backdrop != 0,
                    };

                    // The dummy tile is the only one expected to be outside the visible
                    // area. No need to fill solid tiles
                    if y >= self.scissor_tiles.max.y as u16 {
                        break;
                    }

                    // Fill solid tiles if any, up to the new tile or the end of the current row.
                    while inside && x < x_end {
                        while x < x_end && !output.occlusion.test(x, y, pattern.is_opaque) {
                            assert!((x as i32) < self.scissor_tiles.max.x, "A");
                            //println!("    skip occluded solid tile {solid_tile_x:?}");
                            x += 1;
                        }

                        if x < x_end {
                            //println!("    begin solid tile {solid_tile_x:?}");
                            let mut position = TilePosition::new(x as u32, y as u32);
                            x += 1;

                            while x < x_end && output.occlusion.test(x, y, pattern.is_opaque) {
                                //println!("    extend solid tile {solid_tile_x:?}");
                                x += 1;
                                position.extend();
                            }

                            let tiles = if pattern.is_opaque {
                                &mut output.opaque_tiles
                            } else {
                                &mut output.mask_tiles
                            };
                            tiles.push(TileInstance {
                                position,
                                backdrop,
                                first_edge: 0,
                                edge_count: 0,
                                path_index,
                            }.encode());
                        }
                    }

                    if tile.1 != y {
                        // We moved to a new row of tiles.
                        //println!("");
                        //println!("   * reset backdrop");
                        backdrop = 0;
                    }
                    current_tile = tile;
                }

                if evt.is_edge() {
                    output.edges.push(unsafe { std::mem::transmute(evt.payload) });
                } else {
                    let winding = if evt.payload == 0 { -1 } else { 1 };
                    backdrop += winding;
                    //println!("   * backdrop {winding:?} -> {backdrop:?}");
                }
            }

        }
    }

    pub fn fill_surface(
        &mut self,
        pattern: &BuiltPattern,
        opacity: f32,
        z_index: u32,
        output: &mut TilerOutput,
    ) {
        //println!("fill_surface");

        output.paths.push(PathInfo {
            z_index,
            pattern_data: pattern.data,
            fill_rule: 0,
            opacity,
            scissor: [
                self.scissor.min.x as u32,
                self.scissor.min.y as u32,
                self.scissor.max.x as u32,
                self.scissor.max.y as u32,
            ],
        }.encode());

        for array in &mut self.events {
            array.clear();
        }

        for y in self.scissor_tiles.min.y..self.scissor_tiles.max.y {
            let x = self.scissor_tiles.max.x as u16;
            let y = y as  u16;
            let events = self.events(y as usize);
            events.push(Event::backdrop(0, y, true));
            events.push(Event::backdrop(x, y, false));
        }

        self.generate_tiles(FillRule::EvenOdd, false, pattern, output);
    }
}

/// A sortable compressed event that encodes either a binned edge or a backdrop update
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct Event {
    sort_key: u32,
    payload: u32,
}

impl Event {
    fn edge(tx: u16, ty: u16, edge: [u8; 4]) -> Self {
        Event {
            sort_key: 1 | ((tx as u32) << 1) | ((ty as u32) << 11),
            payload: unsafe { std::mem::transmute(edge) },
        }
    }

    fn backdrop(tx: u16, ty: u16, positive: bool) -> Self {
        Event {
            sort_key: 0 | ((tx as u32) << 1) | ((ty as u32) << 11),
            payload: if positive { 1 } else { 0 },
        }
    }

    fn is_edge(&self) -> bool {
        self.sort_key & 1 != 0
    }

    fn tile(&self) -> (u16, u16) {
        (
            ((self.sort_key >> 1) & 0x3FF) as u16,
            ((self.sort_key >> 11) & 0x3FF) as u16,
        )
    }
}

#[test]
fn event() {
    let e1 = Event::edge(1, 3, [0, 0, 255, 50]);
    let e2 = Event::edge(2, 3, [0, 0, 255, 50]);
    let e3 = Event::edge(0b1111111111, 0b1101010101, [0, 1, 2, 3]);
    let b1 = Event::backdrop(1, 3, true);
    let b2 = Event::backdrop(2, 3, true);
    assert_eq!(e1.tile(), (1, 3));
    assert_eq!(e2.tile(), (2, 3));
    assert_eq!(e3.tile(), (0b1111111111, 0b1101010101));
    assert_eq!(b1.tile(), (1, 3));
    assert_eq!(b2.tile(), (2, 3));
    let mut v = vec![e3, e2, e1, b2, b1];
    v.sort_unstable_by_key(|e| e.sort_key);
    assert_eq!(v, vec![b1, e1, b2, e2, e3]);
}

pub struct TilerOutput {
    pub paths: Vec<EncodedPathInfo>,
    pub edges: Vec<EncodedEdge>,
    pub mask_tiles: Vec<EncodedTileInstance>,
    pub opaque_tiles: Vec<EncodedTileInstance>,
    pub occlusion: OcclusionBuffer,
}

impl TilerOutput {
    pub fn new() -> Self {
        TilerOutput {
            paths: Vec::new(),
            edges: Vec::new(),
            mask_tiles: Vec::new(),
            opaque_tiles: Vec::new(),
            occlusion: OcclusionBuffer::disabled(),
        }
    }

    pub fn clear(&mut self) {
        self.paths.clear();
        self.edges.clear();
        self.mask_tiles.clear();
        self.opaque_tiles.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
            && self.mask_tiles.is_empty()
            && self.opaque_tiles.is_empty()
    }
}

pub type EncodedEdge = u32;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct TileInstance {
    pub position: TilePosition,
    pub first_edge: u32,
    pub edge_count: u16,
    pub backdrop: i16,
    pub path_index: u32,
}

type EncodedTileInstance = [u32; 4];

impl TileInstance {
    pub fn encode(self) -> EncodedTileInstance {
        [
            self.position.to_u32(),
            self.first_edge,
            (self.edge_count as u32) << 16 | (self.backdrop as i32 + 128) as u32,
            self.path_index,
        ]
    }

    pub fn decode(data: EncodedTileInstance) -> Self {
        TileInstance {
            position: TilePosition(data[0]),
            first_edge: data[1],
            edge_count: (data[2] >> 16) as u16,
            backdrop: ((data[2] & 0xFFFF) as i32 - 128) as i16,
            path_index: data[3],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct PathInfo {
    pub z_index: u32,
    pub pattern_data: u32,
    pub fill_rule: u16,
    pub opacity: f32,

    // TODO: this can be [u16;4]
    pub scissor: [u32; 4],
}

pub(crate) type EncodedPathInfo = [u32; 8];

impl PathInfo {
    pub fn encode(&self) -> EncodedPathInfo {
        [
            self.z_index,
            self.pattern_data,
            (self.fill_rule as u32) << 16 | (self.opacity.max(0.0).min(1.0) * 65535.0) as u32,
            0,
            self.scissor[0],
            self.scissor[1],
            self.scissor[2],
            self.scissor[3],
        ]
    }

    pub fn decode(data: EncodedPathInfo) -> Self {
        PathInfo {
            z_index: data[0],
            pattern_data: data[1],
            fill_rule: (data[2] >> 16) as u16,
            opacity: (data[2] & 0xFFFF) as f32 / 65535.0,

            scissor: [data[4], data[5], data[6], data[7]],
        }
    }
}


#[test]
fn size_of() {
    assert_eq!(std::mem::size_of::<TileInstance>(), 16);
}
#[test]
fn tiler2_svg() {
    use core::path::Builder;
    use core::gpu::shader::ShaderPatternId;
    use core::pattern::BindingsId;

    let mut path = Builder::new();
    path.begin(point(10.0, 0.0));
    //path.line_to(point(50.0, 100.0));
    path.cubic_bezier_to(point(100.0, 0.0), point(100.0, 100.0), point(10.0, 100.0));
    path.end(true);

    let path = path.build();

    let mut tiler = Tiler::new();
    tiler.begin_target(SurfaceIntRect {
        min: point(0, 0),
        max: point(800, 600),
    });

    let options = FillOptions {
        fill_rule: FillRule::EvenOdd,
        inverted: false,
        z_index: 0,
        tolerance: 0.25,
        opacity: 1.0,
        transform: None,
    };

    let pattern = BuiltPattern {
        data: 0,
        shader: ShaderPatternId::from_index(0),
        bindings: BindingsId::from_index(0),
        is_opaque: true,
        can_stretch_horizontally: true,
        favor_prerendering: false,
    };

    let mut output = TilerOutput::new();

    tiler.fill_path(path.iter(), &options, &pattern, &mut output);

    use svg_fmt::*;

    let s = 100.0;
    let tile_style = Style {
        fill: Fill::Color(Color { r: 220, g: 220, b: 220, }),
        stroke: Stroke::Color(Color { r: 150, g: 200, b: 220, }, s * 0.01),
        opacity: 1.0,
        stroke_opacity: 1.0,
    };
    let solid_tile_style = Style {
        fill: Fill::Color(Color { r: 150, g: 150, b: 220, }),
        stroke: Stroke::Color(Color { r: 150, g: 200, b: 220, }, s * 0.01),
        opacity: 0.7,
        stroke_opacity: 1.0,
    };
    println!("xxxx{}", BeginSvg { w: s * 10.0, h: s * 10.0 });
    for tile in output.mask_tiles {
        let tile = TileInstance::decode(tile);
        let x = s * (tile.position.x() as f32);
        let y = s * (tile.position.y() as f32);
        let w = s * (1.0 + tile.position.extension() as f32);
        let h = s * (1.0);
        println!("xxxx{}", Rectangle { x, y, w, h, border_radius: 0.0, style: tile_style });
        let edges_start = tile.first_edge as usize;
        let edges_end = edges_start + tile.edge_count as usize;
        println!("edges: {edges_start}..{edges_end}");
        for edge in &output.edges[edges_start..edges_end] {
            let [x1, y1, x2, y2] = unsafe { std::mem::transmute::<u32, [u8;4]>(*edge) };
            let x1 = x + s * x1 as f32 / 255.0;
            let y1 = y + s * y1 as f32 / 255.0;
            let x2 = x + s * x2 as f32 / 255.0;
            let y2 = y + s * y2 as f32 / 255.0;
            println!("   - {x1} {y1}  {x2} {y2}");
            println!("xxxx    {}", LineSegment { x1, y1, x2, y2, color: Color { r: 250, g: 50, b: 70, }, width: s * 0.05 });
        }
    }
    for tile in output.opaque_tiles {
        let tile = TileInstance::decode(tile);
        let x = s * (tile.position.x() as f32);
        let y = s * (tile.position.y() as f32);
        let w = s * (1.0 + tile.position.extension() as f32);
        let h = s * (1.0);
        println!("xxxx{}", Rectangle { x, y, w, h, border_radius: 0.0, style: solid_tile_style });
    }
    println!("xxxx{}", EndSvg);
    //tiler.dbg.flush_blocking();
}

