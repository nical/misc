use lyon_path::PathEvent;
use lyon_path::geom::euclid::{Box2D, Size2D, Transform2D};
use lyon_path::math::{Point, point};
use lyon_path::geom::{LineSegment, QuadraticBezierSegment};

//use std::ops::Range;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum EdgeKind {
    Linear,
    Quadratic,
}

#[derive(Copy, Clone, Debug, PartialEq)]
struct Edge {
    from: Point,
    to: Point,
    ctrl: Point,
    kind: EdgeKind,
    winding: i16,
}

impl Edge {
    fn linear(mut segment: LineSegment<f32>) -> Self {
        let winding = if segment.from.y > segment.to.y {
            std::mem::swap(&mut segment.from, &mut segment.to);
            -1
        } else {
            1
        };

        Edge {
            from: segment.from,
            to: segment.to,
            ctrl: segment.to,
            kind: EdgeKind::Linear,
            winding,
        }
    }

    fn quadratic(mut segment: QuadraticBezierSegment<f32>) -> Self {
        let winding = if segment.from.y > segment.to.y {
            std::mem::swap(&mut segment.from, &mut segment.to);
            -1
        } else {
            1
        };

        Edge {
            from: segment.from,
            to: segment.to,
            ctrl: segment.ctrl,
            kind: EdgeKind::Linear,
            winding,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
struct RowEdge {
    from: Point,
    to: Point,
    ctrl: Point,
    kind: EdgeKind,
    winding: i16,
    min_x: f32,
    intersects_tile_top: bool,
}


#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ActiveEdge {
    pub from: Point,
    pub to: Point,
    pub ctrl: Point,
    pub kind: EdgeKind,
    pub winding: i16,
    max_x: f32,
}

impl ActiveEdge {
    pub fn clip_horizontally(&self, x_range: std::ops::Range<f32>) -> Self {
        match self.kind {
            EdgeKind::Linear => {
                let mut segment = LineSegment {
                    from: self.from,
                    to: self.to,
                };
                let swap = segment.from.x > segment.to.x;
                if swap {
                    std::mem::swap(&mut segment.from, &mut segment.to);
                }

                let range = clip_line_segment_1d(segment.from.x, segment.to.x, x_range.start, x_range.end);
                let mut segment = segment.split_range(range);

                if swap {
                    std::mem::swap(&mut segment.from, &mut segment.to);
                }

                ActiveEdge {
                    from: segment.from,
                    to: segment.to,
                    ctrl: segment.to,
                    winding: self.winding,
                    kind: EdgeKind::Linear,
                    max_x: 0.0,
                }
            }
            EdgeKind::Quadratic => {
                let mut segment = QuadraticBezierSegment {
                    from: self.from,
                    to: self.to,
                    ctrl: self.ctrl,
                };

                let swap = segment.from.x > segment.to.x;
                if swap {
                    std::mem::swap(&mut segment.from, &mut segment.to);
                }

                let range = clip_quadratic_bezier_1d(segment.from.x, segment.ctrl.x, segment.to.x, x_range.start, x_range.end);

                let mut segment = segment.split_range(range);

                if swap {
                    std::mem::swap(&mut segment.from, &mut segment.to);
                }

                ActiveEdge {
                    from: segment.from,
                    to: segment.to,
                    ctrl: segment.ctrl,
                    winding: self.winding,
                    kind: EdgeKind::Quadratic,
                    max_x: 0.0,
                }
            }
        }
    }
}

pub struct TileInfo {
    pub x: u32,
    pub y: u32,
    pub path_id: u16,
    pub backdrop_winding: i16,
    pub inner_rect: Box2D<f32>,
    pub outer_rect: Box2D<f32>,
}

pub trait TileEncoder {
    fn encode_tile(&mut self, tile: &TileInfo, active_edges: &[ActiveEdge]);
}

impl<T> TileEncoder for T
where
    T: FnMut(&TileInfo, &[ActiveEdge])
{
    fn encode_tile(&mut self, tile: &TileInfo, active_edges: &[ActiveEdge]) {
        (*self)(tile, active_edges)
    }
}

pub struct Tiler {
    tile_size: Size2D<f32>,
    // offset of the first tile
    tile_offset_x: f32,
    tile_offset_y: f32,
    tile_padding: f32,

    num_tiles_x: u32,
    num_tiles_y: f32,

    tolerance: f32,
    flatten: bool,
}

impl Tiler {
    pub fn new(rect: &Box2D<f32>, tile_size: Size2D<f32>, tile_padding: f32) -> Self {
        Tiler {
            tile_size,
            tile_offset_x: rect.min.x,
            tile_offset_y: rect.min.y,
            tile_padding,
            num_tiles_x: f32::ceil(rect.size().width / tile_size.width) as u32,
            num_tiles_y: f32::ceil(rect.size().height / tile_size.height),
            tolerance: 0.01,
            flatten: false,
        }
    }

    pub fn set_tolerance(&mut self, tolerance: f32) {
        self.tolerance = tolerance;
    }

    pub fn set_flattening(&mut self, flatten: bool) {
        self.flatten = flatten;
    }

    pub fn tile_path(
        &self,
        path: impl Iterator<Item = PathEvent>,
        transform: Option<&Transform2D<f32>>,
        path_ctx: &mut PathCtx,
        encoder: &mut dyn TileEncoder,
    ) {
        self.prepare_path_ctx(path_ctx);

        let t0 = time::precise_time_ns();

        match (transform, self.flatten) {
            (None, true) => {
                self.assign_rows_linear(path_ctx, path);
            }
            (None, false) => {
                self.assign_rows_quadratic(path_ctx, path);
            }
            (Some(transform), true) => {
                self.assign_rows_linear_transformed(path_ctx, transform, path);
            }
            (Some(transform), false) => {
                self.assign_rows_quadratic_transformed(path_ctx, transform, path);
            }
        }

        let t1 = time::precise_time_ns();

        self.process_rows(path_ctx, encoder);

        let t2 = time::precise_time_ns();

        path_ctx.row_decomposition_time_ns = t1 - t0;
        path_ctx.tile_decomposition_time_ns = t2 - t1;
    }

    fn prepare_path_ctx(&self, path_ctx: &mut PathCtx) {
        let num_rows = self.num_tiles_y as usize;
        path_ctx.rows.truncate(num_rows);
        for _ in 0..(num_rows - path_ctx.rows.len()) {
            path_ctx.rows.push(Vec::new());
        }

        for row in &mut path_ctx.rows {
            row.clear();
        }
    }

    fn assign_rows_quadratic(&self, path_ctx: &mut PathCtx, path: impl Iterator<Item = PathEvent>) {
        for evt in path {
            match evt {
                PathEvent::MoveTo(..) => {}
                PathEvent::Line(segment) | PathEvent::Close(segment) => {
                    let edge = Edge::linear(segment);
                    self.add_monotonic_edge(path_ctx, &edge);
                }
                PathEvent::Quadratic(segment) => {
                    segment.for_each_monotonic(&mut|monotonic| {
                        let edge = Edge::quadratic(*monotonic.segment());
                        self.add_monotonic_edge(path_ctx, &edge);
                    });
                }
                PathEvent::Cubic(segment) => {
                    segment.for_each_quadratic_bezier(self.tolerance, &mut|segment| {
                        segment.for_each_monotonic(&mut|monotonic| {
                            let edge = Edge::quadratic(*monotonic.segment());
                            self.add_monotonic_edge(path_ctx, &edge);
                        });
                    });
                }
            }
        }
    }

    fn assign_rows_linear(&self, path_ctx: &mut PathCtx, path: impl Iterator<Item = PathEvent>) {
        for evt in path {
            match evt {
                PathEvent::MoveTo(..) => {}
                PathEvent::Line(segment) | PathEvent::Close(segment) => {
                    let edge = Edge::linear(segment);
                    self.add_monotonic_edge(path_ctx, &edge);
                }
                PathEvent::Quadratic(segment) => {
                    let mut from = segment.from;
                    segment.for_each_flattened(self.tolerance, &mut|to| {
                        let edge = Edge::linear(LineSegment { from, to });
                        from = to;
                        self.add_monotonic_edge(path_ctx, &edge);
                    });
                }
                PathEvent::Cubic(segment) => {
                    let mut from = segment.from;
                    segment.for_each_flattened(self.tolerance, &mut|to| {
                        let edge = Edge::linear(LineSegment { from, to });
                        from = to;
                        self.add_monotonic_edge(path_ctx, &edge);
                    });
                }
            }
        }
    }

    fn assign_rows_quadratic_transformed(
        &self,
        path_ctx: &mut PathCtx,
        transform: &Transform2D<f32>,
        path: impl Iterator<Item = PathEvent>,
    ) {
        for evt in path {
            match evt {
                PathEvent::MoveTo(..) => {}
                PathEvent::Line(segment) | PathEvent::Close(segment) => {
                    let segment = segment.transform(transform);
                    let edge = Edge::linear(segment);
                    self.add_monotonic_edge(path_ctx, &edge);
                }
                PathEvent::Quadratic(segment) => {
                    let segment = segment.transform(transform);
                    segment.for_each_monotonic(&mut|monotonic| {
                        let edge = Edge::quadratic(*monotonic.segment());
                        self.add_monotonic_edge(path_ctx, &edge);
                    });
                }
                PathEvent::Cubic(segment) => {
                    let segment = segment.transform(transform);
                    segment.for_each_quadratic_bezier(self.tolerance, &mut|segment| {
                        segment.for_each_monotonic(&mut|monotonic| {
                            let edge = Edge::quadratic(*monotonic.segment());
                            self.add_monotonic_edge(path_ctx, &edge);
                        });
                    });
                }
            }
        }
    }

    fn assign_rows_linear_transformed(
        &self,
        path_ctx: &mut PathCtx,
        transform: &Transform2D<f32>,
        path: impl Iterator<Item = PathEvent>,
    ) {
        for evt in path {
            match evt {
                PathEvent::MoveTo(..) => {}
                PathEvent::Line(segment) | PathEvent::Close(segment) => {
                    let segment = segment.transform(transform);
                    let edge = Edge::linear(segment);
                    self.add_monotonic_edge(path_ctx, &edge);
                }
                PathEvent::Quadratic(segment) => {
                    let segment = segment.transform(transform);
                    let mut from = segment.from;
                    segment.for_each_flattened(self.tolerance, &mut|to| {
                        let edge = Edge::linear(LineSegment { from, to });
                        from = to;
                        self.add_monotonic_edge(path_ctx, &edge);
                    });
                }
                PathEvent::Cubic(segment) => {
                    let segment = segment.transform(transform);
                    let mut from = segment.from;
                    segment.for_each_flattened(self.tolerance, &mut|to| {
                        let edge = Edge::linear(LineSegment { from, to });
                        from = to;
                        self.add_monotonic_edge(path_ctx, &edge);
                    });
                }
            }
        }
    }

    fn add_monotonic_edge(&self, path_ctx: &mut PathCtx, edge: &Edge) {
        debug_assert!(edge.from.y <= edge.to.y);
        //println!("<!-- add edge {} {} -> {} {}  -->", edge.from.x, edge.from.y, edge.to.x, edge.to.y);

        let min = self.tile_offset_y - self.tile_padding;
        let max = self.tile_offset_y + self.num_tiles_y * self.tile_size.height + self.tile_padding;

        if edge.from.y > max || edge.to.y < min {
            return;
        }

        // TODO: probably need to snap edges that are very close to the outer tile boundary so that
        // we don't get an edge that very slightly overlaps the tile but isn't assigned to it (and
        // ends up messing up winding numbers).
        // It would also help with avoiding unnecessary tiny edges.

        let inv_tile_height = 1.0 / self.tile_size.height;
        let y_start_tile = f32::floor((edge.from.y - self.tile_offset_y - self.tile_padding) * inv_tile_height).max(0.0);
        let y_end_tile = f32::ceil((edge.to.y - self.tile_offset_y + self.tile_padding) * inv_tile_height).min(self.num_tiles_y);

        let mut row_f = y_start_tile;
        let mut y_min = self.tile_offset_y + row_f * self.tile_size.height - self.tile_padding;
        let mut y_max = self.tile_offset_y + (row_f + 1.0) * self.tile_size.height + self.tile_padding;
        let start_idx = y_start_tile as usize;
        let end_idx = y_end_tile as usize;
        match edge.kind {
            EdgeKind::Linear => for row in &mut path_ctx.rows[start_idx .. end_idx] {
                let segment = LineSegment { from: edge.from, to: edge.to };
                let range = clip_line_segment_1d(edge.from.y, edge.to.y, y_min, y_max);
                let mut segment = segment.split_range(range);
                let intersects_tile_top = ((segment.from.y - y_min) / self.tolerance) as i32 == 0;
                if intersects_tile_top {
                    segment.from.y = y_min;
                }

                row.push(RowEdge {
                    from: segment.from,
                    to: segment.to,
                    ctrl: segment.to,
                    kind: EdgeKind::Linear,
                    winding: edge.winding,
                    min_x: segment.from.x.min(segment.to.x),
                    intersects_tile_top,
                });

                y_min += self.tile_size.height;
                y_max += self.tile_size.height;
                row_f += 1.0;
            }
            EdgeKind::Quadratic => for row in &mut path_ctx.rows[start_idx .. end_idx] {
                let segment = QuadraticBezierSegment { from: edge.from, ctrl: edge.ctrl, to: edge.to };
                let range = clip_quadratic_bezier_1d(edge.from.y, edge.ctrl.y, edge.to.y, y_min, y_max);
                let mut segment = segment.split_range(range);
                let intersects_tile_top = ((segment.from.y - y_min) / self.tolerance) as i32 == 0;
                if intersects_tile_top {
                    segment.from.y = y_min;
                }

                row.push(RowEdge {
                    from: segment.from,
                    to: segment.to,
                    ctrl: segment.ctrl,
                    kind: EdgeKind::Quadratic,
                    winding: edge.winding,
                    min_x: segment.from.x.min(segment.to.x),
                    intersects_tile_top,
                });

                y_min += self.tile_size.height;
                y_max += self.tile_size.height;
                row_f += 1.0;
            }
        }
    }


    fn process_rows(&self, path_ctx: &mut PathCtx, encoder: &mut dyn TileEncoder) {
        let mut tile_y = 0;
        let mut active_edges = Vec::new();
        for row in &mut path_ctx.rows {
            if !row.is_empty() {
                self.process_row(tile_y, path_ctx.path_id, &mut row[..], &mut active_edges, encoder);
            }
            tile_y += 1;
        }
    }

    fn process_row(
        &self,
        tile_y: u32,
        path_id: u16,
        row: &mut [RowEdge],
        active_edges: &mut Vec<ActiveEdge>,
        encoder: &mut dyn TileEncoder,
    ) {

        row.sort_by(|a, b| a.min_x.partial_cmp(&b.min_x).unwrap());

        active_edges.clear();

        let row_y = tile_y as f32 * self.tile_size.height + self.tile_offset_x;

        //println!("\n\n<!-- row {}   baseline {}-->", tile_y, y_baseline);

        let inner_rect = Box2D {
            min: point(self.tile_offset_x, row_y),
            max: point(
                self.tile_offset_x + self.tile_size.width,
                row_y + self.tile_size.height
            ),
        };

        let mut tile = TileInfo {
            x: 0,
            y: tile_y,
            path_id,
            backdrop_winding: 0,
            inner_rect,
            outer_rect: inner_rect.inflate(self.tile_padding, self.tile_padding),
        };

        let mut current_edge = 0;

        // First iterate on edges until we reach one that starts inside the tiling area.
        // During this phase we only need to keep track of the backdrop winding number
        // and detect edges that end in the tiling area.
        for edge in &row[..] {
            if edge.min_x >= self.tile_offset_x {
                break;
            }

            //println!("<!-- edge: {:?}-->", edge);

            if edge.intersects_tile_top {
                tile.backdrop_winding += edge.winding;
                //println!("<!-- (A) winding {} -> {}     edge: {:?}-->", edge.winding, tile.backdrop_winding, edge);
            }

            let max_x = edge.from.x.max(edge.to.x);

            if max_x >= tile.outer_rect.min.x {
                active_edges.push(ActiveEdge {
                    from: edge.from,
                    to: edge.to,
                    ctrl: edge.ctrl,
                    winding: edge.winding,
                    kind: edge.kind,
                    max_x,
                });
            }

            current_edge += 1;
        }

        // Iterate over edges in the tiling area.
        // Now we produce actual tiles.
        //
        // Each time we get to a new tile, we remove all active edges that end left of the tile.
        // In practice this means all active edges intersect the current tile.
        for edge in &row[current_edge..] {
            //println!("  <!-- edge {:?} -->", edge);
            while edge.min_x > tile.outer_rect.max.x {
                self.finish_tile(&mut tile, active_edges, encoder);
            }

            if tile.x >= self.num_tiles_x {
                break;
            }

            //println!("<!-- edge: {:?}-->", edge);

            if edge.intersects_tile_top {
                tile.backdrop_winding += edge.winding;
                //println!("        <!-- (B) winding {} -> {}     edge: {:?}-->", edge.winding, tile.backdrop_winding, edge);
                //println!("        {}", svg_fmt::Circle { x: edge.from.x, y: edge.from.y, radius: 0.1, style: svg_fmt::Style::default() });
            }

            active_edges.push(ActiveEdge {
                from: edge.from,
                to: edge.to,
                ctrl: edge.ctrl,
                winding: edge.winding,
                kind: edge.kind,
                max_x: edge.from.x.max(edge.to.x),
            });

            current_edge += 1;
        }

        // At this point we visited all edges but not necessarily all tiles.
        while tile.x < self.num_tiles_x {
            self.finish_tile(&mut tile, active_edges, encoder);

            if active_edges.is_empty() {
                break;
            }
        }
    }

    fn finish_tile(&self, tile: &mut TileInfo, active_edges: &mut Vec<ActiveEdge>, encoder: &mut dyn TileEncoder) {
        //println!("  <!-- tile {} {} -->", tile.x, tile.y);
        encoder.encode_tile(tile, active_edges);

        tile.inner_rect.min.x += self.tile_size.width;
        tile.inner_rect.max.x += self.tile_size.width;
        tile.outer_rect.min.x += self.tile_size.width;
        tile.outer_rect.max.x += self.tile_size.width;
        tile.x += 1;

        active_edges.retain(|edge| edge.max_x > tile.outer_rect.min.x);
    }
}

pub struct PathCtx {
    rows: Vec<Vec<RowEdge>>,    
    pub path_id: u16,

    pub row_decomposition_time_ns: u64,
    pub tile_decomposition_time_ns: u64,
}

impl PathCtx {
    pub fn new(path_id: u16) -> Self {
        PathCtx {
            rows: Vec::new(),
            path_id,

            row_decomposition_time_ns: 0,
            tile_decomposition_time_ns: 0,
        }
    }

    pub fn reset_rows(&mut self) {
        for row in &mut self.rows {
            row.clear();
        }
    }
}



pub struct ZBuffer {
    data: Vec<u16>,
    w: usize,
    h: usize,
}

impl ZBuffer {
    pub fn new() -> Self {
        ZBuffer {
            data: Vec::new(),
            w: 0,
            h: 0,
        }
    }

    pub fn init(&mut self, w: usize, h: usize) {
        let size = w * h;

        self.w = w;
        self.h = h;

        if self.data.len() < size {
            self.data = vec![0; size];
        } else {
            for elt in &mut self.data[0..size] {
                *elt = 0;
            }
        }
    }

    pub fn get(&self, x: u32, y: u32) -> u16 {
        self.data[self.index(x, y)]
    }

    pub fn test(&mut self, x: u32, y: u32, z_index: u16, write: bool) -> bool {
        debug_assert!(x < self.w as u32);
        debug_assert!(y < self.h as u32);

        let idx = self.index(x, y);
        let z = &mut self.data[idx];
        let result = *z < z_index;

        if write && result {
            *z = z_index;
        }

        result
    }

    #[inline]
    pub fn index(&self, x: u32, y: u32) -> usize {
        self.w * (y as usize) + (x as usize)
    }
}

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

        for edge in active_edges {
            let edge = edge.clip_horizontally(tile.outer_rect.min.x .. tile.outer_rect.max.x);

            self.edges.push(EdgeInstance {
                from: edge.from.to_array(),
                to: edge.to.to_array(),
                ctrl: edge.ctrl.to_array(),
                tile_index,
            });
        }
    }
}

pub fn load_svg(filename: &str) -> (Box2D<f32>, Vec<lyon_path::Path>) {
    let opt = usvg::Options::default();
    let rtree = usvg::Tree::from_file(filename, &opt).unwrap();
    let mut paths = Vec::new();

    let view_box = rtree.svg_node().view_box;
    for node in rtree.root().descendants() {
        use usvg::NodeExt;
        let t = node.transform();
        let transform = Transform2D::row_major(
            t.a as f32, t.b as f32,
            t.c as f32, t.d as f32,
            t.e as f32, t.f as f32,
        );

        if let usvg::NodeKind::Path(ref usvg_path) = *node.borrow() {
            //if usvg_path.fill.is_none() {
            //    continue;
            //}

            let mut builder = lyon_path::Path::builder();
            for segment in &usvg_path.segments {
                match *segment {
                    usvg::PathSegment::MoveTo { x, y } => {
                        builder.move_to(transform.transform_point(&point(x as f32, y as f32)));
                    }
                    usvg::PathSegment::LineTo { x, y } => {
                        builder.line_to(transform.transform_point(&point(x as f32, y as f32)));
                    }
                    usvg::PathSegment::CurveTo { x1, y1, x2, y2, x, y, } => {
                        builder.cubic_bezier_to(
                            transform.transform_point(&point(x1 as f32, y1 as f32)),
                            transform.transform_point(&point(x2 as f32, y2 as f32)),
                            transform.transform_point(&point(x as f32, y as f32)),
                        );
                    }
                    usvg::PathSegment::ClosePath => {
                        builder.close();
                    }
                }
            }
            let path = builder.build();

            paths.push(path);
        }
    }

    let vb = Box2D {
        min: point(
            view_box.rect.x as f32,
            view_box.rect.y as f32,
        ),
        max: point(
            view_box.rect.x as f32 + view_box.rect.width as f32,
            view_box.rect.y as f32 + view_box.rect.height as f32,
        ),
    };

    (vb, paths)
}

pub fn clip_quadratic_bezier_1d(
    from: f32,
    ctrl: f32,
    to: f32,
    min: f32,
    max: f32,
) -> std::ops::Range<f32> {
    debug_assert!(max >= min);
    debug_assert!(to >= from);

    let a = from + 2.0 * to - 2.0 * ctrl;
    let b = -2.0 * from + 2.0 * ctrl;
    let c1 = from - min;
    let c2 = from - max;

    let delta1 = b * b - 4.0 * a * c1;
    let delta2 = b * b - 4.0 * a * c2;

    let sign = a.signum();
    let two_a = 2.0 * a * sign;

    let mut t1 = 0.0;
    if delta1 >= 0.0 {
        let sqrt_delta = delta1.sqrt();
        let root1 = (-b - sqrt_delta) * sign;
        let root2 = (-b + sqrt_delta) * sign;
        if root1 > 0.0 && root1 < two_a {
            t1 = root1 / two_a;
        } else if root2 > 0.0 && root2 < two_a {
            t1 = root2 / two_a;
        }
    }

    let mut t2 = 1.0;
    if delta2 >= 0.0 {
        let sqrt_delta = delta2.sqrt();
        let root1 = (-b - sqrt_delta) * sign;
        let root2 = (-b + sqrt_delta) * sign;
        if root1 > 0.0 && root1 < two_a {
            t2 = root1 / two_a;
        } else if root2 > 0.0 && root2 < two_a {
            t2 = root2 / two_a;
        }
    }

    t1 .. t2
}

pub fn clip_line_segment_1d(
    from: f32,
    to: f32,
    min: f32,
    max: f32,
) -> std::ops::Range<f32> {
    let d = to - from;
    if d == 0.0 {
        return 1.0 .. 0.0;
    }

    let inv_d = 1.0 / d;

    let t0 = ((min - from) * inv_d).max(0.0);
    let t1 = ((max - from) * inv_d).min(1.0);

    t0 .. t1
}
