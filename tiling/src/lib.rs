pub use lyon_path::geom::euclid::default::{Box2D, Size2D, Transform2D};
pub use lyon_path::math::{Point, point};
use lyon_path::geom::{LineSegment, QuadraticBezierSegment};
use ordered_float::OrderedFloat;
pub use lyon_path::geom::euclid;
pub use lyon_path::PathEvent;
pub use lyon_path::geom;

pub mod pathfinder_encoder;
pub mod load_svg;
pub mod rasterizer;
pub mod raster_encoder;
mod z_buffer;
//mod wr;

pub use z_buffer::ZBuffer;

/// The output of the tiler.
///
/// encode_tile will be called for each tile that isn't fully empty.
pub trait TileEncoder {
    fn encode_tile(
        &mut self,
        tile: &TileInfo,
        active_edges: &[ActiveEdge],
    );
}

impl<T> TileEncoder for T
where
    T: FnMut(&TileInfo, &[ActiveEdge])
{
    fn encode_tile(
        &mut self,
        tile: &TileInfo,
        active_edges: &[ActiveEdge],
    ) {
        (*self)(tile, active_edges)
    }
}

/// A context object that can bin path edges into tile grids.
///
/// The simplest way to use it is through the tile_path method:
///
/// ```ignore
/// let mut tiler = Tiler::new(&config);
/// tiler.tile_path(path.iter(), Some(&transform, &mut encoder));
/// ```
///
/// It is also possible to add edges manually between begin_path and
/// end_path invocations:
///
/// ```ignore
/// let mut tiler = Tiler::new(&config);
///
/// tiler.begin_path();
/// for edge in &edges {
///     tiler.add_line_segment(edge);
/// }
/// tiler.end_path(&mut encoder);
/// ```
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

    rows: Vec<Vec<RowEdge>>,

    pub row_decomposition_time_ns: u64,
    pub tile_decomposition_time_ns: u64,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct TilerConfig {
    pub view_box: Box2D<f32>,
    pub tile_size: Size2D<f32>,
    pub tile_padding: f32,
    pub tolerance: f32,
    pub flatten: bool,
}

impl Tiler {
    /// Constructor.
    pub fn new(config: &TilerConfig) -> Self {
        let rect = config.view_box;
        Tiler {
            tile_size: config.tile_size,
            tile_offset_x: rect.min.x,
            tile_offset_y: rect.min.y,
            tile_padding: config.tile_padding,
            num_tiles_x: f32::ceil(rect.size().width / config.tile_size.width) as u32,
            num_tiles_y: f32::ceil(rect.size().height / config.tile_size.height),
            tolerance: config.tolerance,
            flatten: config.flatten,

            row_decomposition_time_ns: 0,
            tile_decomposition_time_ns: 0,

            rows: Vec::new(),
        }
    }

    /// Using init instead of creating a new tiler allows recycling allocations from
    /// a previous tiling run.
    pub fn init(&mut self, config: &TilerConfig) {
        let rect = config.view_box;
        self.tile_size = config.tile_size;
        self.tile_offset_x = rect.min.x;
        self.tile_offset_y = rect.min.y;
        self.tile_padding = config.tile_padding;
        self.num_tiles_x = f32::ceil(rect.size().width / config.tile_size.width) as u32;
        self.num_tiles_y = f32::ceil(rect.size().height / config.tile_size.height);
        self.tolerance = config.tolerance;
        self.flatten = config.flatten;
    }

    /// Tile an entire path.
    ///
    /// This internally does all of the steps:
    /// - begin_Path
    /// - add_monotonic_edge
    /// - end_path
    pub fn tile_path(
        &mut self,
        path: impl Iterator<Item = PathEvent>,
        transform: Option<&Transform2D<f32>>,
        encoder: &mut dyn TileEncoder,
    ) {
        let t0 = time::precise_time_ns();

        let identity = Transform2D::identity();
        let transform = transform.unwrap_or(&identity);

        self.begin_path();

        if self.flatten {
            self.assign_rows_linear(transform, path);
        } else {
            self.assign_rows_quadratic(transform, path);
        }

        let t1 = time::precise_time_ns();

        self.end_path(encoder);

        let t2 = time::precise_time_ns();

        self.row_decomposition_time_ns = t1 - t0;
        self.tile_decomposition_time_ns = t2 - t1;
    }

    /// Can be use to tile a path manually.
    ///
    /// Should only be called between begin_path and end_path.
    pub fn add_line_segment(&mut self, edge: &LineSegment<f32>) {
        self.add_monotonic_edge(&MonotonicEdge::linear(*edge));
    }

    /// Can be use to tile a path manually.
    ///
    /// Should only be called between begin_path and end_path.
    pub fn add_monotonic_edge(&mut self, edge: &MonotonicEdge) {
        debug_assert!(edge.from.y <= edge.to.y);

        let min = self.tile_offset_y - self.tile_padding;
        let max = self.tile_offset_y + self.num_tiles_y * self.tile_size.height + self.tile_padding;

        if edge.from.y > max || edge.to.y < min {
            return;
        }

        let inv_tile_height = 1.0 / self.tile_size.height;
        let y_start_tile = f32::floor((edge.from.y - self.tile_offset_y - self.tile_padding) * inv_tile_height).max(0.0);
        let y_end_tile = f32::ceil((edge.to.y - self.tile_offset_y + self.tile_padding) * inv_tile_height).min(self.num_tiles_y);

        let mut row_f = y_start_tile;
        let mut y_min = self.tile_offset_y + row_f * self.tile_size.height - self.tile_padding;
        let mut y_max = self.tile_offset_y + (row_f + 1.0) * self.tile_size.height + self.tile_padding;
        let start_idx = y_start_tile as usize;
        let end_idx = y_end_tile as usize;
        match edge.kind {
            EdgeKind::Linear => for row in &mut self.rows[start_idx .. end_idx] {
                let segment = LineSegment { from: edge.from, to: edge.to };
                let range = clip_line_segment_1d(edge.from.y, edge.to.y, y_min, y_max);
                let mut segment = segment.split_range(range);
                let intersects_tile_top = (segment.from.y - y_min).abs() < self.tolerance;
                if intersects_tile_top {
                    segment.from.y = y_min;
                }
                row.push(RowEdge {
                    from: segment.from,
                    to: segment.to,
                    ctrl: segment.to,
                    kind: EdgeKind::Linear,
                    winding: edge.winding,
                    min_x: OrderedFloat(segment.from.x.min(segment.to.x)),
                    intersects_tile_top,
                });

                y_min += self.tile_size.height;
                y_max += self.tile_size.height;
                row_f += 1.0;
            }
            EdgeKind::Quadratic => for row in &mut self.rows[start_idx .. end_idx] {
                let segment = QuadraticBezierSegment { from: edge.from, ctrl: edge.ctrl, to: edge.to };
                let range = clip_quadratic_bezier_1d(edge.from.y, edge.ctrl.y, edge.to.y, y_min, y_max);
                let mut segment = segment.split_range(range);
                let intersects_tile_top = (segment.from.y - y_min).abs() < self.tolerance;
                if intersects_tile_top {
                    segment.from.y = y_min;
                }

                row.push(RowEdge {
                    from: segment.from,
                    to: segment.to,
                    ctrl: segment.ctrl,
                    kind: EdgeKind::Quadratic,
                    winding: edge.winding,
                    min_x: OrderedFloat(segment.from.x.min(segment.to.x)),
                    intersects_tile_top,
                });

                y_min += self.tile_size.height;
                y_max += self.tile_size.height;
                row_f += 1.0;
            }
        }
    }

    /// Initialize the tiler before adding edges manually.
    pub fn begin_path(&mut self) {

        let num_rows = self.num_tiles_y as usize;
        self.rows.truncate(num_rows);
        for _ in 0..(num_rows - self.rows.len()) {
            self.rows.push(Vec::new());
        }

        for row in &mut self.rows {
            row.clear();
        }
    }

    /// Process manually edges and encode them into the output TileEncoder.
    pub fn end_path(&mut self, encoder: &mut dyn TileEncoder) {
        let mut tile_y = 0;
        let mut active_edges = Vec::new();

        // borrow-ck dance.
        let mut rows = Vec::new();
        std::mem::swap(&mut self.rows, &mut rows);

        // This could be done in parallel but it's already quite fast serially.
        for row in &mut rows {
            if !row.is_empty() {
                self.process_row(tile_y, &mut row[..], &mut active_edges, encoder);
            }
            tile_y += 1;
        }

        std::mem::swap(&mut self.rows, &mut rows);

        for row in &mut self.rows {
            row.clear();
        }
    }

    fn assign_rows_quadratic(
        &mut self,
        transform: &Transform2D<f32>,
        path: impl Iterator<Item = PathEvent>,
    ) {
        for evt in path {
            match evt {
                PathEvent::MoveTo(..) => {}
                PathEvent::Line(segment) | PathEvent::Close(segment) => {
                    let segment = segment.transform(transform);
                    let edge = MonotonicEdge::linear(segment);
                    self.add_monotonic_edge(&edge);
                }
                PathEvent::Quadratic(segment) => {
                    let segment = segment.transform(transform);
                    segment.for_each_monotonic(&mut|monotonic| {
                        let edge = MonotonicEdge::quadratic(*monotonic.segment());
                        self.add_monotonic_edge(&edge);
                    });
                }
                PathEvent::Cubic(segment) => {
                    let segment = segment.transform(transform);
                    segment.for_each_quadratic_bezier(self.tolerance, &mut|segment| {
                        segment.for_each_monotonic(&mut|monotonic| {
                            let edge = MonotonicEdge::quadratic(*monotonic.segment());
                            self.add_monotonic_edge(&edge);
                        });
                    });
                }
            }
        }
    }

    fn assign_rows_linear(
        &mut self,
        transform: &Transform2D<f32>,
        path: impl Iterator<Item = PathEvent>,
    ) {
        for evt in path {
            match evt {
                PathEvent::MoveTo(..) => {}
                PathEvent::Line(segment) | PathEvent::Close(segment) => {
                    let segment = segment.transform(transform);
                    let edge = MonotonicEdge::linear(segment);
                    self.add_monotonic_edge(&edge);
                }
                PathEvent::Quadratic(segment) => {
                    let segment = segment.transform(transform);
                    let mut from = segment.from;
                    segment.for_each_flattened(self.tolerance, &mut|to| {
                        let edge = MonotonicEdge::linear(LineSegment { from, to });
                        from = to;
                        self.add_monotonic_edge(&edge);
                    });
                }
                PathEvent::Cubic(segment) => {
                    let segment = segment.transform(transform);
                    let mut from = segment.from;
                    segment.for_each_flattened(self.tolerance, &mut|to| {
                        let edge = MonotonicEdge::linear(LineSegment { from, to });
                        from = to;
                        self.add_monotonic_edge(&edge);
                    });
                }
            }
        }
    }

    fn process_row(
        &self,
        tile_y: u32,
        row: &mut [RowEdge],
        active_edges: &mut Vec<ActiveEdge>,
        encoder: &mut dyn TileEncoder,
    ) {

        row.sort_unstable_by(|a, b| a.min_x.cmp(&b.min_x));

        active_edges.clear();

        let row_y = tile_y as f32 * self.tile_size.height + self.tile_offset_x;

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
            backdrop_winding: 0,
            inner_rect,
            outer_rect: inner_rect.inflate(self.tile_padding, self.tile_padding),
        };

        let mut current_edge = 0;

        // First iterate on edges until we reach one that starts inside the tiling area.
        // During this phase we only need to keep track of the backdrop winding number
        // and detect edges that end in the tiling area.
        for edge in &row[..] {
            if edge.min_x.0 >= self.tile_offset_x {
                break;
            }

            if edge.intersects_tile_top {
                tile.backdrop_winding += edge.winding;
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
            while edge.min_x.0 > tile.outer_rect.max.x {
                self.finish_tile(&mut tile, active_edges, encoder);
            }

            if tile.x >= self.num_tiles_x {
                break;
            }

            if edge.intersects_tile_top {
                tile.backdrop_winding += edge.winding;
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

    fn finish_tile(
        &self,
        tile: &mut TileInfo,
        active_edges: &mut Vec<ActiveEdge>,
        encoder: &mut dyn TileEncoder,
    ) {
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

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum EdgeKind {
    Linear,
    Quadratic,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct MonotonicEdge {
    from: Point,
    to: Point,
    ctrl: Point,
    kind: EdgeKind,
    winding: i16,
}

impl MonotonicEdge {
    pub fn linear(mut segment: LineSegment<f32>) -> Self {
        let winding = if segment.from.y > segment.to.y {
            std::mem::swap(&mut segment.from, &mut segment.to);
            -1
        } else {
            1
        };

        MonotonicEdge {
            from: segment.from,
            to: segment.to,
            ctrl: segment.to,
            kind: EdgeKind::Linear,
            winding,
        }
    }

    pub fn quadratic(mut segment: QuadraticBezierSegment<f32>) -> Self {
        let winding = if segment.from.y > segment.to.y {
            std::mem::swap(&mut segment.from, &mut segment.to);
            -1
        } else {
            1
        };

        MonotonicEdge {
            from: segment.from,
            to: segment.to,
            ctrl: segment.ctrl,
            kind: EdgeKind::Linear,
            winding,
        }
    }
}

/// The edge struct stored once assigned to a particular row.
#[derive(Copy, Clone, Debug, PartialEq)]
struct RowEdge {
    from: Point,
    to: Point,
    ctrl: Point,
    kind: EdgeKind,
    winding: i16,
    min_x: OrderedFloat<f32>,
    intersects_tile_top: bool,
}


/// The edge representation in the list of active edges of a tile.
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
    /// X-offset in number of tiles.
    pub x: u32,
    /// Y-offset in number of tiles.
    pub y: u32,
    /// Winding number of the background of the tile.
    pub backdrop_winding: i16,
    /// Rectangle of the tile aligned with the tile grid.
    pub inner_rect: Box2D<f32>,
    /// Rectangle including the tile padding.
    pub outer_rect: Box2D<f32>,
}

fn clip_quadratic_bezier_1d(
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
