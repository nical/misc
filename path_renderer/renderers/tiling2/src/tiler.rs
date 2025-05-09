#![allow(unused)]

use core::gpu::{GpuStoreHandle, GpuStoreWriter, GpuStreamWriter};
use core::units::{LocalSpace, SurfaceSpace, SurfaceRect, SurfaceIntRect, Point, vector};
use core::{pattern::BuiltPattern, units::point};

use lyon::geom::{LineSegment, QuadraticBezierSegment, CubicBezierSegment, Box2D};
use lyon::{path::PathEvent, geom::euclid::Transform2D};

use crate::flatten::{Flattener, FlattenerLevien};
use crate::{FillOptions, FillRule, Transform, TilePosition};
use crate::occlusion::OcclusionBuffer;

// When changing this, also change the corresponding constant in tile.wgsl
const TILE_SIZE: i32 = 16;
const TILE_SIZE_F32: f32 = TILE_SIZE as f32;

const UNITS_PER_TILE: i32 = 256;
const UNITS_PER_TILE_F32: f32 = UNITS_PER_TILE as f32;
const LOCAL_COORD_MASK: i32 = 255;
const LOCAL_COORD_BITS: i32 = 8;
const COORD_SCALE: f32 = UNITS_PER_TILE_F32 / TILE_SIZE_F32;

struct TileUnit;
type TilePoint = core::geom::euclid::Point2D<u16, TileUnit>;

pub struct Tiler {
    events: Vec<Vec<Event>>,
    point_buffer: Vec<Point>,
    tolerance: f32,
    current_tile: (u16, u16),
    current_tile_is_occluded: bool,
    viewport: SurfaceRect,
    scissor: SurfaceRect,
    scissor_tiles: Box2D<i32>,
    edge_buffer: Vec<EncodedEdge>,
}

// TODO: The inverted mode currently only works with 1 event bucket but
// the non-inverted one tends to be a bit faster with 8.
const EVENT_BUCKETS: usize = 1;

impl Tiler {
    pub fn new() -> Self {
        let mut events = Vec::with_capacity(EVENT_BUCKETS);
        for _ in 0..EVENT_BUCKETS { events.push(Vec::new())};

        Tiler {
            events,
            tolerance: 0.25,
            current_tile: (0, 0),
            current_tile_is_occluded: false,
            point_buffer: Vec::with_capacity(512),

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
            edge_buffer: Vec::with_capacity(32),
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

        let path_index = output.paths.push(PathInfo {
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

        self.generate_tiles(
            options.fill_rule,
            options.inverted,
            path_index,
            pattern,
            output,
        );
    }

    // Similar to tile_path_nested, except that it flattens curves into a buffer before
    // processing the line segments in bulk, instead of nesting the binning into
    // the flattening loop.
    // In its current states it doesn not appear to improve performance although
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

        self.point_buffer.clear();
        let mut flattener = FlattenerLevien::new(self.tolerance);
        let mut force_flush = false;
        let mut skipped = None;

        for evt in path {
            match evt {
                PathEvent::Begin { at } => {
                    //println!("# begin {at:?}");
                    self.point_buffer.push(transform.transform_point(at));
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
                    self.point_buffer.push(transform.transform_point(pos));
                    skipped = None;
                }
            }

            while flattener.flatten(&mut self.point_buffer) || force_flush || self.point_buffer.capacity() - self.point_buffer.len() == 0 {
                self.flush(occlusion, force_flush);
                force_flush = false;
            }
        }
    }

    fn flush(&mut self, occlusion: &mut OcclusionBuffer, is_last: bool) {
        profiling::scope!("Tiler::flush");

        if self.point_buffer.len() < 2 {
            return;
        }

        // borrowck dance.
        let mut point_buffer = std::mem::take(&mut self.point_buffer);

        //println!("flush {point_buffer:?}");
        let mut iter = point_buffer.iter();
        let mut from = *iter.next().unwrap();

        for to in iter {
            self.tile_segment(&LineSegment { from, to: *to });
            from = *to;
        }

        point_buffer.clear();
        if !is_last {
            point_buffer.push(from);
        }

        self.point_buffer = point_buffer;
    }

    fn tile_segment(&mut self, edge: &LineSegment<f32>) {

        let _panic = PanicLogger::new("Bug: tile_segment panic with ", edge);

        // Leave some margin around this early scissor test so that
        // we keep track of the previous tile near the boundary of the
        // scissor rect.
        let min_y = edge.to.y.min(edge.from.y);
        let max_y = edge.to.y.max(edge.from.y);
        let min_x = edge.to.x.min(edge.from.x);
        let max_x = edge.to.x.max(edge.from.x);
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
        let mut src_tx = (from_i32_x >> LOCAL_COORD_BITS);
        let mut src_ty = (from_i32_y >> LOCAL_COORD_BITS);
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
                // The min and max operations prevent limited arithmetic precision from
                // allowing the computed vertex to fall on a tile outside of the expected
                // range.
                v_x1 = (edge.from.x + dx_dy * (v_y1 - edge.from.y)).max(min_x).min(max_x);
                h_dst_tx = if x_step == 0 {
                    dst_tx
                } else {
                    (v_x1 * COORD_SCALE) as i32 >> LOCAL_COORD_BITS
                };
            }

            let row_occluded = ty < self.scissor_tiles.min.y || ty >= self.scissor_tiles.max.y;

            // When moving to a different row of tiles, add backdrop events.
            if ty != src_ty {
                let positive = ty > src_ty;
                let row = ty + (!positive) as i32;
                if row >= self.scissor_tiles.min.y && row < self.scissor_tiles.max.y && tx + 1 < self.scissor_tiles.max.x {
                    //println!("    - backdrop {} {row}, positive:{positive}", (tx + 1).max(0));
                    self.events(row as usize).push(Event::backdrop((tx + 1).max(scissor_min_tx) as u16, row as u16, positive));
                }
                src_ty = ty;
            }

            let mut h_x0 = v_x0;
            let mut h_y0 = v_y0;
            //println!(" - row {ty}, sub ({v_x0} {v_y0}) -> ({v_x1} {v_y1}), tiles x {tx} {h_dst_tx}");
            debug_assert!(x_step > 0 || h_dst_tx <= tx);
            debug_assert!(x_step < 0 || h_dst_tx >= tx);

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

                    if !occluded {
                        let offset_x = (tx * UNITS_PER_TILE) as f32; // TODO: multiply with overflow.
                        let local_x0 = ((h_x0 * COORD_SCALE) - offset_x).min(255.0) as u8;
                        let local_x1 = ((h_x1 * COORD_SCALE) - offset_x).min(255.0) as u8;
                        let local_y1 = ((h_y1 * COORD_SCALE) - offset_y).min(255.0) as u8;

                        assert!(tx < self.scissor_tiles.max.x);
                        //println!("   - edge tile {tx} {ty}, step {x_step}, [{h_x0}, {h_y0}, {h_x1}, {h_y1}], offset {offset_x} {offset_y} local {:?}", [local_x0, local_y0, local_x1, local_y1]);
                        events.push(Event::edge(
                            tx as u16, ty as u16,
                            [local_x0, local_y0, local_x1, local_y1]
                        ));
                    }

                    // Add an auxiliary edge when crossing a vertical boundary (tile x
                    // coordinate changes).
                    if tx != src_tx {
                        // Note, if the edge is going in the negative x orientation, then
                        // we are actually adding an auxiliary edge to the previous tile
                        // rather than the current one.
                        let aux_tx = tx.max(src_tx);
                        let aux_occluded = aux_tx < self.scissor_tiles.min.x || aux_tx >= self.scissor_tiles.max.x;

                        if !aux_occluded {
                            let (y0, y1) = if tx < src_tx {
                                (local_y0, 255)
                            } else {
                                (255, local_y0)
                            };
                            //println!("   - auxiliary edge tile {aux_tx} {ty}, y-range {y0} {y1}");
                            debug_assert!(aux_tx < self.scissor_tiles.max.x);
                            events.push(Event::edge(
                                aux_tx as u16, ty as u16,
                                [0, y0, 0, y1]
                            ));
                        }
                    }
                }

                src_tx = tx;
                //println!("    tx: {tx}, h_dst_tx: {h_dst_tx}, dst_tx {dst_tx}");
                debug_assert!(x_step > 0 || tx >= h_dst_tx);
                debug_assert!(x_step < 0 || tx <= h_dst_tx);

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

    fn tile_for_position(&mut self, pos: Point) -> (i32, i32) {
        let x_i32 = (pos.x * COORD_SCALE) as i32;
        let y_i32 = (pos.y * COORD_SCALE) as i32;
        let tx = (x_i32 >> LOCAL_COORD_BITS);
        let ty = (y_i32 >> LOCAL_COORD_BITS);

        (tx, ty)
    }

    fn generate_tiles(&mut self, fill_rule: FillRule, inverted: bool, path_index: GpuStoreHandle, pattern: &BuiltPattern, output: &mut TilerOutput) {
        if inverted {
            self.generate_tiles_inverted(fill_rule, path_index, pattern, output);
            return;
        }

        profiling::scope!("Tiler::generate_tiles");

        for events in &mut self.events {
            if events.is_empty() {
                continue;
            }

            events.sort_unstable_by_key(|e| e.sort_key);
            // Push a dummy backdrop out of view that will cause the current tile to be flushed
            // at the last iteration without having to replicate the logic out of the loop.
            events.push(Event::backdrop(0, std::u16::MAX, false));

            //println!("tiles {:?}", self.scissor_tiles);
            //for e in events.iter() {
            //    let TilePoint { x, y, .. } = e.tile();
            //    let edge: [u8; 4] = unsafe { std::mem::transmute(e.payload) };
            //    if e.is_edge() {
            //        println!("- {x} {y} edge {edge:?}");
            //    } else {
            //        println!("- {x} {y} backdrop {}", if e.payload == 0 { -1 } else { 1 });
            //    }
            //}

            let mut current = TilePoint::new(0, 0);
            let mut backdrop: i16 = 0;

            for evt in events.iter() {
                let tile = evt.tile();

                if current != tile {
                    let edge_count = self.edge_buffer.len();
                    if edge_count != 0 {
                        // This limit isn't necessary but it is useful to catch bugs. In practice there should
                        // never be this many edges in a single tile.
                        const MAX_EDGES_PER_TILE: usize = 4096;
                        debug_assert!(edge_count < MAX_EDGES_PER_TILE, "bad tile at {current:?}, {edge_count} edges");

                        let first_edge = output.edges.push_slice(&self.edge_buffer).to_u32();
                        let edge_count = usize::min(edge_count, MAX_EDGES_PER_TILE) as u16;

                        self.edge_buffer.clear();

                        if !output.occlusion.occluded(current.x, current.y) {
                            output.mask_tiles.push(TileInstance {
                                position: TilePosition::new(current.x as u32, current.y as u32),
                                backdrop,
                                first_edge,
                                edge_count,
                                path_index: path_index.to_u32(),
                            }.encode());
                        }

                        current.x += 1;
                    }

                    let x_end = if current.y == tile.y {
                        tile.x.min(self.scissor_tiles.max.x as u16)
                    } else {
                        self.scissor_tiles.max.x as u16
                    };

                    let inside = match fill_rule {
                        FillRule::EvenOdd => backdrop % 2 != 0,
                        FillRule::NonZero => backdrop != 0,
                    };

                    if current.x < x_end && inside {
                        Self::fill_span(current.x, x_end, current.y, backdrop, pattern.is_opaque, path_index.to_u32(), output);
                    }

                    if current.y != tile.y{
                        // We moved to a new row of tiles.
                        if current.y >= self.scissor_tiles.max.y as u16 {
                            break;
                        }

                        backdrop = 0;
                        current = tile;
                        // println!("\n   * reset backdrop {current_tile:?}");
                    } else {
                        current = tile;
                    }
                }

                if evt.is_edge() {
                    self.edge_buffer.push(unsafe { std::mem::transmute(evt.payload) });
                } else {
                    let winding = if evt.payload == 0 { -1 } else { 1 };
                    backdrop += winding;
                    //println!("   * backdrop {winding:?} -> {backdrop:?}");
                }
            }

        }
    }

    fn generate_tiles_inverted(&mut self, fill_rule: FillRule, path_index: GpuStoreHandle, pattern: &BuiltPattern, output: &mut TilerOutput) {
        let max_x = self.scissor_tiles.max.x as u16;

        let mut events = std::mem::take(&mut self.events);

        for events in &mut events {
            events.sort_unstable_by_key(|e| e.sort_key);
            // Push a dummy backdrop out of view that will cause the current tile to be flushed
            // at the last iteration without having to replicate the logic out of the loop.
            events.push(Event::backdrop(0, std::u16::MAX, false));

            let mut current = TilePoint::new(0, 0);
            let mut backdrop: i16 = 0;

            for evt in events.iter() {
                let tile = evt.tile();
                //println!("    * {evt:?}");

                if current != tile {
                    let edge_count = self.edge_buffer.len();
                    if edge_count != 0 {
                        // This limit isn't necessary but it is useful to catch bugs. In practice there should
                        // never be this many edges in a single tile.
                        const MAX_EDGES_PER_TILE: usize = 4096;
                        debug_assert!(edge_count < MAX_EDGES_PER_TILE, "bad tile at {current:?}, {edge_count} edges");

                        let first_edge = output.edges.push_slice(&self.edge_buffer).to_u32();
                        let edge_count = usize::min(edge_count, MAX_EDGES_PER_TILE) as u16;

                        self.edge_buffer.clear();

                        if !output.occlusion.occluded(current.x, current.y) {
                            output.mask_tiles.push(TileInstance {
                                position: TilePosition::new(current.x as u32, current.y as u32),
                                backdrop,
                                first_edge,
                                edge_count,
                                path_index: path_index.to_u32(),
                            }.encode());
                        }

                        current.x += 1;
                    }
                }

                while current != tile {
                    let next_x = if current.y == tile.y {
                        tile.x.min(max_x)
                    } else {
                        max_x
                    };

                    let inside = !match fill_rule {
                        FillRule::EvenOdd => backdrop % 2 != 0,
                        FillRule::NonZero => backdrop != 0,
                    };

                    // Fill solid tiles if any, up to the new tile or the end of the current row.
                    if inside {
                        Self::fill_span(current.x, next_x, current.y, backdrop, pattern.is_opaque, path_index.to_u32(), output);
                    }

                    if next_x == max_x {
                        // Reached the end of a row, move to the next.
                        backdrop = 0;
                        current.x = 0;
                        current.y += 1;
                        // The dummy tile is the only one expected to be outside the visible
                        // area. No need to fill solid tiles
                        if current.y >= self.scissor_tiles.max.y as u16 {
                            break;
                        }
                    } else {
                        current.x = next_x;
                    }
                }

                if evt.is_edge() {
                    self.edge_buffer.push(unsafe { std::mem::transmute(evt.payload) });
                } else {
                    let winding = if evt.payload == 0 { -1 } else { 1 };
                    backdrop += winding;
                }
            }
        }

        self.events = events;
    }

    fn fill_span(
        mut x: u16,
        x_end: u16,
        y: u16,
        backdrop: i16,
        is_opaque: bool,
        path_index: u32,
        output: &mut TilerOutput,
    ) {
        // Fill solid tiles if any, up to the new tile or the end of the current row.
        while x < x_end {
            // Skip over occluded tiles.
            while x < x_end && !output.occlusion.test(x, y, is_opaque) {
                x += 1;
            }

            if x >= x_end {
                return;
            }

            // Begin a solid tile at `position`
            let mut position = TilePosition::new(x as u32, y as u32);
            x += 1;

            // Extend the solid tile until we reach x_end or an occluded
            // tile.
            while x < x_end && output.occlusion.test(x, y, is_opaque) {
                x += 1;
                position.extend();
            }

            // Add the (strectched) tile and start over if we haven't reached
            // x_end yet.
            let instance = TileInstance {
                position,
                backdrop,
                first_edge: 0,
                edge_count: 0,
                path_index,
            }.encode();
            let tiles = if is_opaque {
                output.opaque_tiles.push(instance);
            } else {
                output.mask_tiles.push(instance);
            };
        }
    }

    fn fill_span_no_occlusion(
        mut x: u16,
        x_end: u16,
        y: u16,
        backdrop: i16,
        is_opaque: bool,
        path_index: u32,
        output: &mut TilerOutput,
    ) {
        if x == x_end {
            return;
        }

        // Begin a solid tile at `position`
        let mut position = TilePosition::new(x as u32, y as u32);
        x += 1;

        // Extend the solid tile until we reach x_end or an occluded
        // tile.
        while x < x_end {
            x += 1;
            position.extend();
        }

        // Add the (strectched) tile and start over if we haven't reached
        // x_end yet.
        let instance = TileInstance {
            position,
            backdrop,
            first_edge: 0,
            edge_count: 0,
            path_index,
        }.encode();
        if is_opaque {
            output.opaque_tiles.push(instance);
        } else {
            output.mask_tiles.push(instance);
        }
    }

    pub fn fill_surface(
        &mut self,
        pattern: &BuiltPattern,
        opacity: f32,
        z_index: u32,
        output: &mut TilerOutput,
    ) {
        let path_index = output.paths.push(PathInfo {
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

        self.generate_tiles(FillRule::EvenOdd, false, path_index, pattern, output);
    }
}

struct Scan {
    is_opaque: bool,
}

impl Scan {
    fn fill_span(&mut self, x0: u16, x1: u16, y: u16, backdrop: i16, output: &mut TilerOutput) {
        todo!();
    }
}

/// A sortable compressed event that encodes either a binned edge or a backdrop update
#[derive(Copy, Clone, PartialEq, Eq)]
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

    fn tile(&self) -> TilePoint {
        TilePoint::new(
            ((self.sort_key >> 1) & 0x3FF) as u16,
            ((self.sort_key >> 11) & 0x3FF) as u16,
        )
    }
}

impl std::fmt::Debug for Event {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        let tile = self.tile();
        if self.is_edge() {
            write!(f, "Edge at {tile:?}")
        } else {
            write!(f, "Backdrop at {tile:?}")
        }
    }
}

#[test]
fn event() {
    let e1 = Event::edge(1, 3, [0, 0, 255, 50]);
    let e2 = Event::edge(2, 3, [0, 0, 255, 50]);
    let e3 = Event::edge(0b1111111111, 0b1101010101, [0, 1, 2, 3]);
    let b1 = Event::backdrop(1, 3, true);
    let b2 = Event::backdrop(2, 3, true);
    assert_eq!(e1.tile().to_tuple(), (1, 3));
    assert_eq!(e2.tile().to_tuple(), (2, 3));
    assert_eq!(e3.tile().to_tuple(), (0b1111111111, 0b1101010101));
    assert_eq!(b1.tile().to_tuple(), (1, 3));
    assert_eq!(b2.tile().to_tuple(), (2, 3));
    let mut v = vec![e3, e2, e1, b2, b1];
    v.sort_unstable_by_key(|e| e.sort_key);
    assert_eq!(v, vec![b1, e1, b2, e2, e3]);
}

pub struct TilerOutput<'l> {
    pub paths: GpuStoreWriter<'l>,
    pub edges: GpuStoreWriter<'l>,
    pub opaque_tiles: GpuStreamWriter<'l> ,
    pub mask_tiles: Vec<EncodedTileInstance>,
    pub occlusion: OcclusionBuffer,
}

impl<'l> TilerOutput<'l> {
    pub fn new(
        paths: GpuStoreWriter<'l>,
        edges: GpuStoreWriter<'l>,
        opaque_tiles: GpuStreamWriter<'l> ,
    ) -> Self {
        TilerOutput {
            paths,
            edges,
            opaque_tiles,
            mask_tiles: Vec::new(),
            occlusion: OcclusionBuffer::disabled(),
        }
    }

    pub fn clear(&mut self) {
    }

    pub fn is_empty(&self) -> bool {
        self.edges.pushed_bytes() == 0
            && self.mask_tiles.is_empty()
            && self.opaque_tiles.pushed_bytes() == 0
    }
}

pub type EncodedEdge = u32;

// Note: The type to use for the memory layout/footprint on GPU is
// EncodedTileInstance.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct TileInstance {
    position: TilePosition,
    first_edge: u32,
    edge_count: u16,
    backdrop: i16,
    path_index: u32,
}

pub type EncodedTileInstance = [u32; 4];

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

struct PanicLogger<'l, T: std::fmt::Debug> {
    _msg: &'l str,
    _payload: T,
}

impl<'l, T: std::fmt::Debug> PanicLogger<'l, T> {
    #[inline]
    pub fn new(_msg: &'l str, _payload: T) -> Self {
        PanicLogger {
            _msg,
            _payload,
        }
    }
}

impl<'l, T: std::fmt::Debug> Drop for PanicLogger<'l, T> {
    fn drop(&mut self) {
        #[cfg(debug_assertions)]
        if std::thread::panicking() {
            println!("{} {:?}", self._msg, self._payload);
        }
    }
}


#[test]
fn size_of() {
    assert_eq!(std::mem::size_of::<TileInstance>(), 16);
}
/*
#[test]
fn tiler2_svg() {
    use core::path::Builder;
    use core::gpu::shader::ShaderPatternId;
    use core::BindingsId;

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
        bindings: BindingsId::NONE,
        is_opaque: true,
        blend_mode: core::gpu::shader::BlendMode::None,
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
}
#[cfg(test)]
fn test_segment(from: Point, to: Point) {
    let seg = LineSegment { from: point(-526043.7, 1466.5547), to: point(-526063.0, 395.3828) };

    use core::path::Builder;
    use core::gpu::shader::ShaderPatternId;
    use core::BindingsId;

    let mut path = Builder::new();
    path.begin(from);
    path.line_to(to);
    path.line_to(from + vector(100.0, 0.0));
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
        bindings: BindingsId::NONE,
        is_opaque: true,
        blend_mode: core::gpu::shader::BlendMode::None,
        can_stretch_horizontally: true,
        favor_prerendering: false,
    };

    let mut output = TilerOutput::new();

    tiler.fill_path(path.iter(), &options, &pattern, &mut output);

}

#[test]
fn segment_01() {
    test_segment(point(-526043.7, 1466.5547), point(-526063.0, 395.3828))
}

#[test]
fn segment_02() {
    test_segment(point(330.918, -0.04), point(331.346, -0.85275))
}
*/
