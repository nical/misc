use core::units::{LocalSpace, SurfaceSpace};
use core::{pattern::BuiltPattern, units::point};

use lyon::geom::{LineSegment, QuadraticBezierSegment, CubicBezierSegment, Box2D};
use lyon::lyon_tessellation::FillRule;
use lyon::{path::PathEvent, geom::euclid::Transform2D};

pub type Transform = lyon::geom::euclid::Transform2D<f32, LocalSpace, SurfaceSpace>;

const TILE_SIZE_F32: f32 = 16.0;
const TILE_SIZE: i32 = 16;

pub struct FillOptions<'l> {
    pub fill_rule: FillRule,
    pub inverted: bool,
    pub z_index: u32,
    pub tolerance: f32,
    pub merge_tiles: bool,
    pub prerender_pattern: bool,
    pub transform: Option<&'l Transform>,
}

impl<'l> FillOptions<'l> {
    pub fn new() -> FillOptions<'static> {
        FillOptions {
            fill_rule: FillRule::EvenOdd,
            inverted: false,
            z_index: 0,
            tolerance: 0.25,
            merge_tiles: true,
            prerender_pattern: false,
            transform: None,
        }
    }

    pub fn transformed<'a>(transform: &'a Transform) -> FillOptions<'a> {
        FillOptions {
            fill_rule: FillRule::EvenOdd,
            inverted: false,
            z_index: 0,
            tolerance: 0.25,
            merge_tiles: true,
            prerender_pattern: false,
            transform: Some(transform),
        }
    }

    pub fn with_transform<'a>(self, transform: Option<&'a Transform>) -> FillOptions<'a>
    where
        'l: 'a,
    {
        FillOptions {
            fill_rule: self.fill_rule,
            inverted: false,
            z_index: self.z_index,
            tolerance: self.tolerance,
            merge_tiles: self.merge_tiles,
            prerender_pattern: self.prerender_pattern,
            transform,
        }
    }

    pub fn with_fill_rule(mut self, fill_rule: FillRule) -> Self {
        self.fill_rule = fill_rule;
        self
    }

    pub fn with_tolerance(mut self, tolerance: f32) -> Self {
        self.tolerance = tolerance;
        self
    }

    pub fn with_inverted(mut self, inverted: bool) -> Self {
        self.inverted = inverted;
        self
    }

    pub fn with_z_index(mut self, z_index: u32) -> Self {
        self.z_index = z_index;
        self
    }
}

pub struct Tiler {
    events: Vec<Event>,
    tolerance: f32,
    current_tile: (u16, u16),
    current_tile_is_occluded: bool,
    viewport: Box2D<f32>,
    viewport_tiles: Box2D<i32>,

    occlusion: OcclusionBuffer,
    //pub dbg: rerun::RecordingStream,
}

impl Tiler {
    pub fn new() -> Self {
        Tiler {
            events: Vec::new(),
            tolerance: 0.25,
            current_tile: (0, 0),
            current_tile_is_occluded: false,
            viewport: Box2D {
                min: point(0.0, 0.0),
                max: point(0.0, 0.0),
            },
            viewport_tiles: Box2D {
                min: point(0, 0),
                max: point(0, 0),
            },
            occlusion: OcclusionBuffer::new(0, 0),
            //dbg: rerun::RecordingStreamBuilder::new("tiler2").spawn().unwrap(),
        }
    }

    pub fn begin_frame(&mut self, mut viewport: Box2D<f32>) {
        viewport.min.x = viewport.min.x.max(0.0);
        viewport.min.y = viewport.min.y.max(0.0);
        viewport.max.x = viewport.max.x.max(0.0);
        viewport.max.y = viewport.max.y.max(0.0);
        self.viewport = viewport;
        let i32_viewport = viewport.to_i32();
        self.viewport_tiles = Box2D {
            min: i32_viewport.min / TILE_SIZE,
            max: point(
                (viewport.max.x as i32) / TILE_SIZE + (i32_viewport.max.x % TILE_SIZE != 0) as i32,
                (viewport.max.y as i32) / TILE_SIZE + (i32_viewport.max.y % TILE_SIZE != 0) as i32,
            ),
        };

        self.occlusion.init(self.viewport_tiles.max.x as u32, self.viewport_tiles.max.y as u32);
    }

    pub fn fill_path(
        &mut self,
        path: impl Iterator<Item = PathEvent>,
        options: &FillOptions,
        pattern: &BuiltPattern,
        output: &mut TilerOutput,
    ) {
        profiling::scope!("Tiler::fill_path");

        let identity = Transform2D::identity();
        let transform = options.transform.unwrap_or(&identity);

        self.tile_path(path, transform);

        let mut encoded_fill_rule = match options.fill_rule {
            FillRule::EvenOdd => 0,
            FillRule::NonZero => 1,
        };
        if options.inverted {
            encoded_fill_rule |= 2;
        }

        output.paths.push(PathInfo {
            //scissor: [
            //    self.viewport.min.x,
            //    self.viewport.min.y,
            //    self.viewport.max.x,
            //    self.viewport.max.y,
            //],
            z_index: options.z_index,
            pattern_data: pattern.data,
            fill_rule: encoded_fill_rule,
            opacity: 255, // TODO
            scissor: 0, // TODO
        }.encode());

        self.generate_tiles(options.fill_rule, options.inverted, pattern, output);
    }

    fn tile_path(
        &mut self,
        path: impl Iterator<Item = PathEvent>,
        transform: &Transform,
    ) {
        //println!("\n\n-------");
        profiling::scope!("Tiler::tile_path");
        let transform: &lyon::geom::Transform<f32> = unsafe {
            std::mem::transmute(transform)
        };
        self.events.clear();

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
                    self.tile_segment(&segment);
                    from = first;
                }
                PathEvent::Line { to, .. } => {
                    let segment = LineSegment { from, to }.transformed(transform);
                    if segment.to_vector().square_length() < square_tolerance {
                        continue;
                    }
                    self.tile_segment(&segment);
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
                    flatten_quad(&segment, self.tolerance, &mut |segment| {
                        // segment.for_each_flattened(self.draw.tolerance, &mut|segment| {
                        self.tile_segment(segment);
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
                    flatten_cubic(&segment, self.tolerance, &mut |segment| {
                        self.tile_segment(segment);
                    });
                    from = to;
                }
            }
        }

    }

    fn tile_segment(&mut self, segment: &LineSegment<f32>) {
        //println!("\n{segment:?} ({:?})", segment.to_vector() / TILE_SIZE_F32);
        // Cull above and below the viewport
        let min_y = segment.from.y.min(segment.to.y);
        let max_y = segment.from.y.max(segment.to.y);
        if max_y < self.viewport.min.y || min_y > self.viewport.max.y {
            return;
        }

        // Cull right of the viewport
        let min_x = segment.from.x.min(segment.to.x);
        let max_x = segment.from.x.max(segment.to.x);
        if min_x > self.viewport.max.x {
            return;
        }

        let inv_tile_size = 1.0 / TILE_SIZE_F32;
        // In number of tiles
        let mut ty = f32::floor(segment.from.y * inv_tile_size).floor() as i32;
        let dst_ty = f32::floor(segment.to.y * inv_tile_size).floor() as i32;

        // Cull left of the viewport
        if max_x < self.viewport.min.x {
            // Backdrops are still affected by content on the left of the viewport.
            let positive_winding = dst_ty > ty;
            let y_range = if positive_winding {ty..dst_ty } else { dst_ty..ty };
            for y in y_range {
                let tile_y = (y + 1) as u16;
                self.events.push(Event::backdrop(0, tile_y, positive_winding));
            }

            return;
        }

        // Skip perfectly horizontal segments since they don't affect the output.
        // This also removes risks of dividing by zero later.
        if segment.from.y == segment.to.y {
            return;
        }

        // In number of tiles
        let mut tx = (segment.from.x * inv_tile_size).floor() as i32;
        let dst_tx = (segment.to.x * inv_tile_size).floor() as i32;
        let src_tx = tx;
        let src_ty = ty;

        // TODO: if dst_dx == tx && dst_ty == ty which is not uncommon, we can fast path and
        // emmit a single segment directly.

        // DDA-ish walk over the tiles that the edge touches

        //let dx_sign = (segment.to.x - segment.from.x).signum() as i32;
        //let dy_sign = (segment.to.y - segment.from.y).signum() as i32;
        let dx_sign = (dst_tx - src_tx).signum();
        let dy_sign = (dst_ty - src_ty).signum();

        // In pixels, local to the current tile.
        let local_x0 = (segment.from.x - tx as f32 * TILE_SIZE_F32).max(0.0);
        let local_y0 = (segment.from.y - ty as f32 * TILE_SIZE_F32).max(0.0);
        let mut local_x0_u8 = (local_x0 * inv_tile_size * 255.0).min(255.0) as u8;
        let mut local_y0_u8 = (local_y0 * inv_tile_size * 255.0).min(255.0) as u8;

        // TODO: division by zero when the segment is vertical.
        let inv_segment_vx = 1.0 / (segment.to.x - segment.from.x);
        let inv_segment_vy = 1.0 / (segment.to.y - segment.from.y);

        let next_tile_x = (inv_segment_vx > 0.0) as i32;
        let next_tile_y = (inv_segment_vy > 0.0) as i32;
        let mut t_crossing_x = (((tx + next_tile_x) as f32 * TILE_SIZE_F32 - segment.from.x) * inv_segment_vx).abs();
        let mut t_crossing_y = (((ty + next_tile_y) as f32 * TILE_SIZE_F32 - segment.from.y) * inv_segment_vy).abs();
        let t_delta_x = (TILE_SIZE_F32 * inv_segment_vx).abs();
        let t_delta_y = (TILE_SIZE_F32 * inv_segment_vy).abs();

        //self.dbg.log(format!("segment-{:?}-{:?}", segment.from.to_tuple(), segment.to.to_tuple()),
        //    &rerun::LineStrips2D::new(
        //        [
        //            [segment.from.to_array(), segment.to.to_array()]
        //        ].to_vec()
        //    )
        //    //.with_labels([format!("sign: {dx_sign} {dy_sign} crossing {t_crossing_x} {t_crossing_y}")])
        //).unwrap();

        //println!("tiles {tx} {ty} -> {dst_tx} {dst_ty}, sign {dx_sign} {dy_sign},  t_delta {t_delta_x} {t_delta_y} start {local_x0_u8} {local_y0_u8}");

        //let mut idx = 0;
        loop {
            //idx += 1;
            //assert!(idx < 100, "{segment:?}, {tx} {ty} ({src_tx} {src_ty} -> {dst_tx} {dst_ty})");
            //let tile_x_px = tx as f32 * TILE_SIZE_F32;
            //let tile_y_px = ty as f32 * TILE_SIZE_F32;
            //self.dbg.log(format!("tile-{tx}-{ty}/rect"), &rerun::Boxes2D::from_mins_and_sizes([(tile_x_px, tile_y_px)], [(TILE_SIZE_F32, TILE_SIZE_F32)])).unwrap();
            //println!("tile {tx} {ty} t_crossing: {t_crossing_x} {t_crossing_y}");

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
            //println!(" - tile {tx} {ty}, crossing {tcx} {tcy} -> {t_split}");
            let t_split = t_split.min(1.0);
            let one_t_split = 1.0 - t_split;
            let x1 = segment.from.x * one_t_split + segment.to.x * t_split;
            let y1 = segment.from.y * one_t_split + segment.to.y * t_split;

            //self.dbg.log(format!("tile-{tx}-{ty}/point"),
            //    &rerun::Points2D::new([(x1, y1)])
            //        .with_radii([1.0])
            //        .with_labels([&format!("t={t_split}")[..]])
            //).unwrap();

            let local_x1 = (x1 - tx as f32 * TILE_SIZE_F32).max(0.0);
            let local_y1 = (y1 - ty as f32 * TILE_SIZE_F32).max(0.0);
            let local_x1_u8 = (local_x1 * (255.0 * inv_tile_size)).min(255.0) as u8;
            let local_y1_u8 = (local_y1 * (255.0 * inv_tile_size)).min(255.0) as u8;
            //println!("            local {x1} {y1}| u8: {local_x0_u8} {local_y0_u8} {local_x1_u8} {local_y1_u8}");
            //if local_x0_u8 == local_x1_u8 && local_y0 == local_y1 {
            //    println!("empty tiled segment");
            //}

            let h_crossing_0 = local_y0_u8 == 0;
            let h_crossing_1 = local_y1_u8 == 0;
            if h_crossing_0 ^ h_crossing_1 {
                // The tile's x coordinate could be negative if the segment is partially
                // out of the viewport.
                let x = (tx + 1).max(0);
                // No need to clamp y to positive numbers here because the viewport_tiles
                // check filters out all tiles with negative y.
                let y = ty;// + step_y.max(0);
                if y >= self.viewport_tiles.min.y
                    && y < self.viewport_tiles.max.y
                    && x < self.viewport_tiles.max.x {
                    //println!(" - backdrop {x} {y} | {}", if h_crossing_0 { 1 } else { -1 });
                    self.events.push(Event::backdrop(x as u16, y as u16, h_crossing_0));
                }
            }

            let in_viewport = self.viewport_tiles.contains(point(tx, ty));
            if in_viewport {
                // The viewport can only contain positive coordinates.
                let tile_x = tx as u16;
                let tile_y = ty as u16;
    
                if (tile_x, tile_y) != self.current_tile {
                    self.current_tile_is_occluded = self.occlusion.occluded(tile_x, tile_y);
                    self.current_tile = (tile_x, tile_y);
                }

                if !self.current_tile_is_occluded {
                    if local_y0_u8 != local_y1_u8 {
                        //println!(" - edge {tx} {ty} | {local_x0_u8} {local_y0_u8}  {local_x1_u8} {local_y1_u8} | {x1} {y1}");
                        self.events.push(Event::edge(
                            tile_x, tile_y,
                            [local_x0_u8, local_y0_u8, local_x1_u8, local_y1_u8]
                        ));    
                    }
    
                    // Add auxiliary edges when an edge crosses the left side.
                    let v_crossing_0 = local_x0_u8 == 0;
                    let v_crossing_1 = local_x1_u8 == 0;
                    // When either (but not both) endpoints lie at the left boundary:
                    if (v_crossing_0 ^ v_crossing_1) && !((h_crossing_0 && v_crossing_0) || (h_crossing_1 && v_crossing_1)) {
                        let auxiliary_edge = if v_crossing_0 {
                            [0, 255, 0, local_y0_u8]
                        } else {
                            [0, local_y1_u8, 0, 255]
                        };
                        //println!(" - auxiliary edge {tile_x} {tile_y} | {auxiliary_edge:?}");
                        self.events.push(Event::edge(tile_x, tile_y, auxiliary_edge));
                    }
                }
            }

            if (tx == dst_tx && ty == dst_ty) || t_split >= 1.0 {
                break;
            }

            tx += step_x;
            ty += step_y;

            local_x0_u8 = local_x1_u8;
            local_y0_u8 = local_y1_u8;
            if step_x > 0 {
                local_x0_u8 = 0;
            } else if step_x < 0 {
                local_x0_u8 = 255;
            } else if step_y > 0 {
                local_y0_u8 = 0;
            } else if step_y < 0 {
                local_y0_u8 = 255;
            }
        }
    }

    fn generate_tiles(&mut self, fill_rule: FillRule, inverted: bool, pattern: &BuiltPattern, output: &mut TilerOutput) {
        profiling::scope!("Tiler::generate_tiles");

        let path_index = output.paths.len() as u32 - 1;

        self.events.sort_unstable_by_key(|e| e.sort_key);
        // Push a dummy backdrop out of view that will cause the current tile to be flushed
        // at the last iteration without having to replicate the logic out of the loop.
        self.events.push(Event::backdrop(0, std::u16::MAX, false));

        //for e in &self.events {
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

        for evt in &self.events {
            let tile = evt.tile();
            //if evt.is_edge() {
            //    println!("   * edge {tile:?}");
            //} else {
            //    println!("   * backdrop {tile:?}");
            //}
            if tile != current_tile {
                //if tile.1 != current_tile.1 {
                //    println!("");
                //}
                //println!("* new tile {tile:?} (was {current_tile:?}, backdrop: {backdrop:?}");
                let tile_last_edge = output.edges.len();
                let mut solid_tile_x = current_tile.0;
                if tile_last_edge != tile_first_edge {
                    debug_assert!(tile_last_edge - tile_first_edge < 512, "bad tile at {current_tile:?}, edges {tile_first_edge} {tile_last_edge}");
                    //println!("      * encode tile {} {}, with {} edges, backdrop {backdrop}", current_tile.0, current_tile.1, tile_last_edge - tile_first_edge);
                    output.mask_tiles.push(TileInstance {
                        position: TilePosition::new(current_tile.0 as u32, current_tile.1 as u32),
                        backdrop,
                        first_edge: tile_first_edge as u32,
                        edge_count: (tile_last_edge - tile_first_edge) as u16,
                        path_index,
                    }.encode());
                    tile_first_edge = tile_last_edge;
                    solid_tile_x += 1;
                }

                let inside = inverted ^ match fill_rule {
                    FillRule::EvenOdd => backdrop % 2 != 0,
                    FillRule::NonZero => backdrop != 0,
                };

                while inside && solid_tile_x < tile.0 && solid_tile_x < self.viewport_tiles.max.x as u16 {
                    while solid_tile_x < tile.0 && self.occlusion.occluded(solid_tile_x, current_tile.1) {
                        assert!((solid_tile_x as i32) < self.viewport_tiles.max.x, "A");
                        //println!("    skip occluded solid tile {solid_tile_x:?}");
                        solid_tile_x += 1;
                    }

                    //println!("    begin solid tile {solid_tile_x:?}");
                    let mut position = TilePosition::new(solid_tile_x as u32, current_tile.1 as u32);
                    solid_tile_x += 1;

                    while solid_tile_x < tile.0  && self.occlusion.test(solid_tile_x, current_tile.1, pattern.is_opaque) {
                        assert!((solid_tile_x as i32) < self.viewport_tiles.max.x, "B");
                        //println!("    extend solid tile {solid_tile_x:?}");
                        solid_tile_x += 1;
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

                if tile.1 != current_tile.1 {
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
}

impl TilerOutput {
    pub fn new() -> Self {
        TilerOutput {
            paths: Vec::new(),
            edges: Vec::new(),
            mask_tiles: Vec::new(),
            opaque_tiles: Vec::new(),
        }
    }

    pub fn clear(&mut self) {
        self.paths.clear();
        self.edges.clear();
        self.mask_tiles.clear();
        self.opaque_tiles.clear();
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
    //pub scissor: [f32; 4], // TODO: move into another texture
    pub z_index: u32,
    pub pattern_data: u32,
    pub fill_rule: u16,
    pub opacity: u16,
    pub scissor: u32,
}

type EncodedPathInfo = [u32; 4];

impl PathInfo {
    pub fn encode(&self) -> EncodedPathInfo {
        [
            self.z_index,
            self.pattern_data,
            (self.fill_rule as u32) << 16 | self.opacity as u32,
            self.scissor,
        ]
    }

    pub fn decode(data: EncodedPathInfo) -> Self {
        PathInfo {
            z_index: data[0],
            pattern_data: data[1],
            fill_rule: (data[2] >> 16) as u16,
            opacity: (data[2] & 0xFFFF) as u16,
            scissor: data[3],
        }
    }
}


#[test]
fn size_of() {
    assert_eq!(std::mem::size_of::<TileInstance>(), 16);
    assert_eq!(std::mem::size_of::<PathInfo>(), 16);
}

fn flatten_cubic<F>(curve: &CubicBezierSegment<f32>, tolerance: f32, callback: &mut F)
where
    F: FnMut(&LineSegment<f32>),
{
    let mut rem = *curve;
    let mut from = rem.from;
    let mut t0 = 0.0;

    let mut split = 0.5;
    loop {
        if rem.is_linear(tolerance) {
            callback(&LineSegment { from, to: rem.to });
            return;
        }

        loop {
            let sub = rem.before_split(split);
            if sub.is_linear(tolerance) {
                let t1 = t0 + (1.0 - t0) * split;
                callback(&LineSegment { from, to: sub.to });
                from = sub.to;
                t0 = t1;
                rem = rem.after_split(split);
                let next_split = split * 2.0;
                if next_split < 1.0 {
                    split = next_split;
                }
                break;
            }
            split *= 0.5;
        }
    }
}

pub fn flatten_quad<F>(curve: &QuadraticBezierSegment<f32>, tolerance: f32, callback: &mut F)
where
    F: FnMut(&LineSegment<f32>),
{
    let mut rem = *curve;
    let mut from = rem.from;
    let mut t0 = 0.0;

    let mut split = 0.5;
    loop {
        if rem.is_linear(tolerance) {
            callback(&LineSegment { from, to: rem.to });
            return;
        }

        loop {
            let sub = rem.before_split(split);
            if sub.is_linear(tolerance) {
                let t1 = t0 + (1.0 - t0) * split;
                callback(&LineSegment { from, to: sub.to });
                from = sub.to;
                t0 = t1;
                rem = rem.after_split(split);
                let next_split = split * 2.0;
                if next_split < 1.0 {
                    split = next_split;
                }
                break;
            }
            split *= 0.5;
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TilePosition(u32);

impl TilePosition {
    const MASK: u32 = 0x3FF;
    pub const ZERO: Self = TilePosition(0);
    pub const INVALID: Self = TilePosition(std::u32::MAX);

    pub fn extended(x: u32, y: u32, extend: u32) -> Self {
        debug_assert!(x <= Self::MASK);
        debug_assert!(y <= Self::MASK);
        debug_assert!(extend <= Self::MASK);

        TilePosition(extend << 20 | x << 10 | y)
    }

    pub fn new(x: u32, y: u32) -> Self {
        debug_assert!(x <= Self::MASK);
        debug_assert!(y <= Self::MASK);

        TilePosition(x << 10 | y)
    }

    pub fn extend(&mut self) {
        self.0 += 1 << 20;
    }

    pub fn with_flag(mut self) -> Self {
        self.add_flag();
        self
    }
    pub fn to_u32(&self) -> u32 {
        self.0
    }
    pub fn x(&self) -> u32 {
        (self.0 >> 10) & Self::MASK
    }
    pub fn y(&self) -> u32 {
        (self.0) & Self::MASK
    }
    pub fn extension(&self) -> u32 {
        (self.0 >> 20) & Self::MASK
    }

    // TODO: we have two unused bits and we use one of them to store
    // whether a tile in an indirection buffer is opaque. That's not
    // great.
    pub fn flag(&self) -> bool {
        self.0 & 1 << 31 != 0
    }
    pub fn add_flag(&mut self) {
        self.0 |= 1 << 31
    }
}

#[test]
fn tile_position() {
    let mut p0 = TilePosition::new(1, 2);
    assert_eq!(p0.x(), 1);
    assert_eq!(p0.y(), 2);
    assert_eq!(p0.extension(), 0);

    p0.extend();

    assert_eq!(p0.x(), 1);
    assert_eq!(p0.y(), 2);
    assert_eq!(p0.extension(), 1);

    p0.extend();

    assert_eq!(p0.x(), 1);
    assert_eq!(p0.y(), 2);
    assert_eq!(p0.extension(), 2);
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
    tiler.begin_frame(Box2D {
        min: point(0.0, 0.0),
        max: point(800.0, 600.0),
    });

    let options = FillOptions {
        fill_rule: FillRule::EvenOdd,
        inverted: false,
        z_index: 0,
        tolerance: 0.25,
        merge_tiles: true,
        prerender_pattern: false,
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

    let mut output = TilerOutput {
        paths: Vec::new(),
        edges: Vec::new(),
        mask_tiles: Vec::new(),
        //solid_tiles: Vec::new(),
        opaque_tiles: Vec::new(),
    };

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

pub struct OcclusionBuffer {
    data: Vec<u8>,
    width: usize,
    height: u32,
}

impl OcclusionBuffer {
    pub fn new(w: u32, h: u32) -> Self {
        let mut m = OcclusionBuffer {
            data: Vec::new(),
            width: 0,
            height: 0,
        };

        m.init(w, h);

        m
    }

    pub fn init(&mut self, w: u32, h: u32) {
        let w = w as usize;
        let h = h as usize;
        if self.data.len() < w * h {
            self.data = vec![0; w * h];
        } else {
            self.data[..(w * h)].fill(0);
        }
        self.width = w;
        self.height = h as u32;
    }

    pub fn resize(&mut self, w: u32, h: u32) {
        if w * h <= self.width as u32 * self.height {
            return;
        }

        let w = w as usize;
        let h = h as usize;

        let mut data = vec![0; w * h];

        for y in 0..self.height {
            let src_start = y as usize * self.width;
            let src_end = src_start + self.width;
            let dst_start = y as usize * w;
            let dst_end = dst_start + self.width;
            data[dst_start..dst_end].copy_from_slice(&self.data[src_start..src_end]);
        }

        self.data = data;

        self.width = w;
        self.height = h as u32;
    }

    pub fn width(&self) -> u32 {
        self.width as u32
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn occluded(&mut self, x: u16, y: u16) -> bool {
        let offset = x as usize + y as usize * self.width;
        assert!(offset < self.data.len(), "occlustion.get({x} {y}) out of bounds {} {}", self.width, self.height);
        self.data[offset] != 0
    }

    pub fn test(&mut self, x: u16, y: u16, write: bool) -> bool {
        let offset = x as usize + y as usize * self.width;
        assert!(offset < self.data.len(), "occlustion.test({x} {y}) out of bounds {} {}", self.width, self.height);
        let payload = &mut self.data[offset as usize];
        let result = *payload == 0;

        if write {
            *payload = 1;
        }

        result
    }

    pub fn clear(&mut self) {
        self.data.fill(0);
    }
}
