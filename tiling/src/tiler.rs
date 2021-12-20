use ordered_float::OrderedFloat;
pub use lyon::path::math::{Point, point};
pub use lyon::path::{PathEvent, FillRule};
pub use lyon::geom::euclid::default::{Box2D, Size2D, Transform2D};
pub use lyon::geom::euclid;
pub use lyon::geom;
use lyon::geom::{LineSegment, QuadraticBezierSegment, CubicBezierSegment};

pub use crate::z_buffer::{ZBuffer, ZBufferRow};

use parasol::{Context, CachePadded};

/// The output of the tiler.
///
/// encode_tile will be called for each tile that isn't fully empty.
pub trait TileEncoder: Send {
    fn encode_tile(
        &mut self,
        tile: &TileInfo,
        active_edges: &[ActiveEdge],
        side_edges: &SideEdgeTracker,
    );

    fn begin_row(&self) {}
    fn end_row(&self) {}
}

impl<T> TileEncoder for T
where
    T: Send + FnMut(&TileInfo, &[ActiveEdge], &SideEdgeTracker)
{
    fn encode_tile(
        &mut self,
        tile: &TileInfo,
        active_edges: &[ActiveEdge],
        side_edges: &SideEdgeTracker,
    ) {
        (*self)(tile, active_edges, side_edges)
    }
}

struct Row {
    edges: Vec<RowEdge>,
    tile_y: u32,
    z_buffer: ZBufferRow,
}

impl Row {
    fn new() -> Self {
        Row {
            edges: Vec::new(),
            tile_y: 0,
            z_buffer: ZBufferRow::new()
        }
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

    rows: Vec<Row>,

    worker_data: Vec<CachePadded<TilerWorkerData>>,

    first_row: usize,
    last_row: usize,

    pub row_decomposition_time_ns: u64,
    pub tile_decomposition_time_ns: u64,
    pub is_opaque: bool,
    pub fill_rule: FillRule,
    pub z_index: u16,
}

#[derive(Clone)]
struct TilerWorkerData {
    side_edges: SideEdgeTracker,
    active_edges: Vec<ActiveEdge>,
}

impl TilerWorkerData {
    fn new() -> Self {
        TilerWorkerData {
            side_edges: SideEdgeTracker::new(),
            active_edges: Vec::with_capacity(64),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct TilerConfig {
    pub view_box: Box2D<f32>,
    pub tile_size: Size2D<f32>,
    pub tile_padding: f32,
    pub tolerance: f32,
    pub flatten: bool,
}

unsafe impl Sync for Tiler {}

impl Tiler {
    /// Constructor.
    pub fn new(config: &TilerConfig) -> Self {
        let rect = config.view_box;
        let num_tiles_y = f32::ceil(rect.size().height / config.tile_size.height);
        Tiler {
            tile_size: config.tile_size,
            tile_offset_x: rect.min.x,
            tile_offset_y: rect.min.y,
            tile_padding: config.tile_padding,
            num_tiles_x: f32::ceil(rect.size().width / config.tile_size.width) as u32,
            num_tiles_y,
            tolerance: config.tolerance,
            flatten: config.flatten,

            first_row: 0,
            last_row: 0,

            row_decomposition_time_ns: 0,
            tile_decomposition_time_ns: 0,

            rows: Vec::new(),

            is_opaque: true,
            fill_rule: FillRule::EvenOdd,
            z_index: 0,

            worker_data: Vec::new(),
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
        profiling::scope!("tile_path");
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
        self.first_row = self.first_row.min(start_idx);
        self.last_row = self.last_row.max(end_idx);
        match edge.kind {
            EdgeKind::Linear => for row in &mut self.rows[start_idx .. end_idx] {
                let segment = LineSegment { from: edge.from, to: edge.to };
                let range = clip_line_segment_1d(edge.from.y, edge.to.y, y_min, y_max);
                let mut segment = segment.split_range(range);
                let intersects_tile_top = (segment.from.y - y_min).abs() < self.tolerance;
                if intersects_tile_top {
                    segment.from.y = y_min;
                }
                row.edges.push(RowEdge {
                    from: segment.from,
                    to: segment.to,
                    ctrl: point(std::f32::NAN, std::f32::NAN),
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

                row.edges.push(RowEdge {
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
        self.first_row = self.rows.len();
        self.last_row = 0;


        let num_rows = self.num_tiles_y as usize;
        self.rows.truncate(num_rows);
        for i in self.rows.len()..num_rows {
            self.rows.push(Row {
                edges: Vec::new(),
                tile_y: i as u32,
                z_buffer: ZBufferRow::new(),
            });
        }

        for row in &mut self.rows {
            row.edges.clear();
        }
    }

    /// Process manually edges and encode them into the output TileEncoder.
    pub fn end_path(&mut self, encoder: &mut dyn TileEncoder) {
        let mut active_edges = Vec::with_capacity(64);

        // borrow-ck dance.
        let mut rows = std::mem::take(&mut self.rows);

        let mut side_edges = SideEdgeTracker::new();
        // This could be done in parallel but it's already quite fast serially.
        for row in &mut rows[self.first_row..self.last_row] {
            if row.edges.is_empty() {
                continue;
            }

            if row.z_buffer.is_empty() {
                row.z_buffer.init(self.num_tiles_x as usize);
            }
            side_edges.clear();
            self.process_row(
                row.tile_y,
                &mut row.edges[..],
                &mut active_edges,
                &mut side_edges,
                &mut row.z_buffer,
                encoder,
            );
        }

        self.rows = rows;

        for row in &mut self.rows {
            row.edges.clear();
        }
    }

    pub fn tile_path_parallel(
        &mut self,
        ctx: &mut Context,
        path: impl Iterator<Item = PathEvent>,
        transform: Option<&Transform2D<f32>>,
        encoders: &mut [&mut dyn TileEncoder],
    ) where {
        profiling::scope!("tile_path_parallel");
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


        if self.first_row < self.last_row {
            if self.last_row - self.first_row > 16 {
                self.end_path_parallel(ctx, encoders);
            } else {
                self.end_path(encoders[0]);
            }
        }

        let t2 = time::precise_time_ns();

        self.row_decomposition_time_ns = t1 - t0;
        self.tile_decomposition_time_ns = t2 - t1;
    }

    pub fn end_path_parallel(&mut self, ctx: &mut Context, encoders: &mut [&mut dyn TileEncoder]) {
        if self.worker_data.is_empty() {
            self.worker_data = vec![CachePadded::new(TilerWorkerData::new()); ctx.num_contexts() as usize];
        }
        // Basically what we are doing here is passing the encoders rows to the
        // worker threads in the most unsafe of ways, and being careful to never have multiple
        // worker threads accessing the same encoder (there's one per parallel context)
        // we should be using the context_data thing instead but the whole passing &mut dyn TileEncoder
        // around is tedious.
        struct Shared<'a, 'b> {
            encoders: &'b mut [&'a mut dyn TileEncoder],
        }
        unsafe impl<'a, 'b> Send for Shared<'a, 'b> {}
        unsafe impl<'a, 'b> Sync for Shared<'a, 'b> {}

        let mut shared = Shared { encoders, };
        let shared_ptr: UnsafeSendPtr<Shared> = UnsafeSendPtr(&mut shared);

        // borrow-ck dance.
        let mut rows = std::mem::take(&mut self.rows);
        let mut worker_data = std::mem::take(&mut self.worker_data);

        ctx.for_each(&mut rows[self.first_row..self.last_row])
            .with_context_data(&mut worker_data)
            .with_group_size(4)
            .with_priority(parasol::Priority::Low)
            //.filter(|row| !row.edges.is_empty())
            .run(|ctx, args| {
                //println!("ctx {:?} row {:?}", ctx.id(), row.tile_y);

                let worker_data: &mut TilerWorkerData = &mut *args.context_data;

                if args.item.edges.is_empty() {
                    return;
                }

                // Temporarily move the row out of the row array.
                // For once this is not a borrowck dance, we just want to avoid
                // false cache sharing.
                let mut row = std::mem::replace(args.item, Row::new());

                unsafe {
                    worker_data.active_edges.clear();
                    worker_data.side_edges.clear();

                    if row.z_buffer.is_empty() {
                        row.z_buffer.init(self.num_tiles_x as usize);
                    }

                    let idx = if ctx.is_worker_thread()  { ctx.id().index() } else { ctx.num_worker_threads() as usize };
                    self.process_row(
                        row.tile_y,
                        &mut row.edges[..],
                        &mut worker_data.active_edges,
                        &mut worker_data.side_edges,
                        &mut row.z_buffer,
                        (*shared_ptr.get()).encoders[idx],
                    );
                }

                row.edges.clear();

                *args.item = row;
            });

        self.worker_data = worker_data;
        self.rows = rows;
    }

/*
    // A very hacky proof of concept of processing the rows in parallel via rayon.
    // So far it's a spectacular fiasco due to spending all of our time in rayon's glue
    // and less than 10% of the time doing usueful work on worker threads. 
    pub fn end_path_rayon(&mut self, encoders: &mut [&mut dyn TileEncoder]) {
        use rayon::prelude::*;

        // borrow-ck dance.
        let mut rows = std::mem::take(&mut self.rows);
        let mut z_buffer = std::mem::take(&mut self.z_buffer);

        // Basically what we are doing here is passing the encoders and z_buffer rows to the
        // worker threads in the most unsafe of ways, and being careful to never have multiple
        // worker threads accessing the same encoder (there's one per worker), nor the same z buffers
        // (there is one per row and rows are dispatched in parallel)
        struct Shared<'a, 'b> {
            encoders: &'b mut [&'a mut dyn TileEncoder],
            z_buffer: &'b mut [ZBufferRow],
        }
        unsafe impl<'a, 'b> Send for Shared<'a, 'b> {}
        unsafe impl<'a, 'b> Sync for Shared<'a, 'b> {}

        {
        let mut shared = Shared {
            encoders,
            z_buffer: &mut z_buffer[..],
        };

        let shared_ptr: UnsafeSendPtr<Shared> = UnsafeSendPtr(&mut shared);

        rows.par_iter_mut()
            .for_each(|row| {

            if row.edges.is_empty() {
                return;
            }

            unsafe {

            let mut active_edges = Vec::with_capacity(64);
            let mut side_edges = SideEdgeTracker::new();

            let z_buffer: &mut ZBufferRow = &mut (*shared_ptr.get()).z_buffer[row.tile_y as usize];

            if z_buffer.is_empty() {
                z_buffer.init(self.num_tiles_x as usize);
            }
            side_edges.clear();

            let idx = rayon::current_thread_index().unwrap();
            self.process_row(row.tile_y, &mut row.edges[..], &mut active_edges, &mut side_edges, z_buffer, (*shared_ptr.get()).encoders[idx]);

            }
        });

        }

        self.rows = rows;
        self.z_buffer = z_buffer;

        for row in &mut self.rows {
            row.edges.clear();
        }
    }
*/
    pub fn clear_depth(&mut self) {
        for row in &mut self.rows {
            row.z_buffer.clear();
        }
    }

    fn assign_rows_quadratic(
        &mut self,
        transform: &Transform2D<f32>,
        path: impl Iterator<Item = PathEvent>,
    ) {
        profiling::scope!("assign_rows_quadratic");
        for evt in path {
            match evt {
                PathEvent::Begin { .. } => {}
                PathEvent::End { last, first, .. } => {
                    let segment = LineSegment { from: last, to: first }.transformed(transform);
                    let edge = MonotonicEdge::linear(segment);
                    self.add_monotonic_edge(&edge);
                }
                PathEvent::Line { from, to } => {
                    let segment = LineSegment { from, to }.transformed(transform);
                    let edge = MonotonicEdge::linear(segment);
                    self.add_monotonic_edge(&edge);
                }
                PathEvent::Quadratic { from, ctrl, to } => {
                    let segment = QuadraticBezierSegment { from, ctrl, to }.transformed(transform);
                    segment.for_each_monotonic(&mut|monotonic| {
                        let edge = MonotonicEdge::quadratic(*monotonic.segment());
                        self.add_monotonic_edge(&edge);
                    });
                }
                PathEvent::Cubic { from, ctrl1, ctrl2, to } => {
                    let segment = CubicBezierSegment { from, ctrl1, ctrl2, to }.transformed(transform);
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
        profiling::scope!("assign_rows_linear");
        for evt in path {
            match evt {
                PathEvent::Begin { .. } => {}
                PathEvent::End { last, first, .. } => {
                    let segment = LineSegment { from: last, to: first }.transformed(transform);
                    let edge = MonotonicEdge::linear(segment);
                    self.add_monotonic_edge(&edge);
                }
                PathEvent::Line { from, to } => {
                    let segment = LineSegment { from, to }.transformed(transform);
                    let edge = MonotonicEdge::linear(segment);
                    self.add_monotonic_edge(&edge);
                }
                PathEvent::Quadratic { from, ctrl, to } => {
                    let segment = QuadraticBezierSegment { from, ctrl, to }.transformed(transform);
                    let mut from = segment.from;
                    segment.for_each_flattened(self.tolerance, &mut|to| {
                        let edge = MonotonicEdge::linear(LineSegment { from, to });
                        from = to;
                        self.add_monotonic_edge(&edge);
                    });
                }
                PathEvent::Cubic { from, ctrl1, ctrl2, to } => {
                    let segment = CubicBezierSegment { from, ctrl1, ctrl2, to }.transformed(transform);
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
        side_edges: &mut SideEdgeTracker,
        z_buffer: &mut ZBufferRow,
        encoder: &mut dyn TileEncoder,
    ) {
        encoder.begin_row();
        //println!(" -- process row y {:?} edges {:?} first tile {:?} num tiles x {:?}", tile_y, row.len(), self.tile_offset_x, self.num_tiles_x);
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
            x: self.tile_offset_x as u32,
            y: tile_y,
            inner_rect,
            outer_rect: inner_rect.inflate(self.tile_padding, self.tile_padding),
            solid: false,
        };

        let mut current_edge = 0;

        // First iterate on edges until we reach one that starts inside the tiling area.
        // During this phase we only need to keep track of the backdrop winding number
        // and detect edges that end in the tiling area.
        for edge in &row[..] {
            if edge.min_x.0 >= self.tile_offset_x {
                break;
            }

            active_edges.push(ActiveEdge {
                from: edge.from,
                to: edge.to,
                ctrl: edge.ctrl,
                winding: edge.winding,
                kind: edge.kind,
                max_x: edge.from.x.max(edge.to.x),
                intersects_tile_top: edge.intersects_tile_top,
            });

            while edge.min_x.0 > tile.outer_rect.max.x {
                active_edges.retain(|edge| {
                    if edge.max_x < tile.outer_rect.min.x {
                        side_edges.add_edge(edge.from.y, edge.to.y, edge.winding);
                        return false
                    }

                    true
                });
            }

            current_edge += 1;
        }

        // Iterate over edges in the tiling area.
        // Now we produce actual tiles.
        //
        // Each time we get to a new tile, we remove all active edges that end side of the tile.
        // In practice this means all active edges intersect the current tile.
        for edge in &row[current_edge..] {
            while edge.min_x.0 > tile.outer_rect.max.x {
                self.finish_tile(&mut tile, active_edges, side_edges, z_buffer, encoder);
            }

            if tile.x >= self.num_tiles_x {
                break;
            }

            active_edges.push(ActiveEdge {
                from: edge.from,
                to: edge.to,
                ctrl: edge.ctrl,
                winding: edge.winding,
                kind: edge.kind,
                max_x: edge.from.x.max(edge.to.x),
                intersects_tile_top: edge.intersects_tile_top,
            });

            current_edge += 1;
        }

        // At this point we visited all edges but not necessarily all tiles.
        while tile.x < self.num_tiles_x {
            self.finish_tile(&mut tile, active_edges, side_edges, z_buffer, encoder);

            if active_edges.is_empty() {
                break;
            }
        }

        encoder.end_row();
    }

    fn finish_tile(
        &self,
        tile: &mut TileInfo,
        active_edges: &mut Vec<ActiveEdge>,
        side_edges: &mut SideEdgeTracker,
        z_buffer: &mut ZBufferRow,
        encoder: &mut dyn TileEncoder,
    ) {
        //side_edges.print();
        //println!("  <!-- tile {} {} -->", tile.x, tile.y);

        tile.solid = false;
        let mut occluded = false;
        if active_edges.is_empty() {
            if side_edges.is_in(tile.outer_rect.min.y, tile.outer_rect.max.y, self.fill_rule) {
                tile.solid = true;
            } else if side_edges.is_empty() {
                // Empty tile.
                occluded = true;
            }
        }

        if !occluded && z_buffer.test(tile.x, self.z_index, tile.solid && self.is_opaque) {
            encoder.encode_tile(tile, active_edges, side_edges);
        }

        tile.inner_rect.min.x += self.tile_size.width;
        tile.inner_rect.max.x += self.tile_size.width;
        tile.outer_rect.min.x += self.tile_size.width;
        tile.outer_rect.max.x += self.tile_size.width;
        tile.x += 1;

        active_edges.retain(|edge| {
            if edge.max_x < tile.outer_rect.min.x {
                side_edges.add_edge(edge.from.y, edge.to.y, edge.winding);
                return false
            }

            true
        });
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
            ctrl: point(std::f32::NAN, std::f32::NAN),
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
    intersects_tile_top: bool,
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
                    ctrl: point(std::f32::NAN, std::f32::NAN),
                    winding: self.winding,
                    kind: EdgeKind::Linear,
                    max_x: 0.0,
                    intersects_tile_top: self.intersects_tile_top, // TODO
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
                    intersects_tile_top: self.intersects_tile_top, // TODO
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
    /// Rectangle of the tile aligned with the tile grid.
    pub inner_rect: Box2D<f32>,
    /// Rectangle including the tile padding.
    pub outer_rect: Box2D<f32>,
    /// True if the tile is entirely covered by the current path. 
    pub solid: bool,
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

#[derive(Copy, Clone, PartialEq)]
pub struct SideEvent {
    pub y: f32,
    pub winding: i16,
}

impl std::fmt::Debug for SideEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        (self.y, self.winding).fmt(f)
    }
}

#[derive(Clone)]
pub struct SideEdgeTracker {
    events: Vec<SideEvent>,
}

impl SideEdgeTracker {
    pub fn new() -> Self {
        SideEdgeTracker {
            events: Vec::with_capacity(32),
        }
    }

    pub fn print(&self) {
        print!("side events: [");
        for evt in &self.events {
            print!("{}({}), ", evt.y, evt.winding);
        }
        println!("]");
    }

    pub fn clear(&mut self) {
        self.events.clear();
    }

    pub fn events(&self) -> &[SideEvent] { &self.events }

    pub fn is_empty(&self) -> bool { self.events.is_empty() }

    pub fn is_in(&self, from: f32, to: f32, fill_rule: FillRule) -> bool {
        let mut i = 0;
        let len = self.events.len();
        while i < len {
            let evt = &self.events[i];
            i += 1;

            if evt.y > from {
                return false;
            }

            if fill_rule.is_in(evt.winding) {
                break;
            }
        }

        while i < len {
            let evt = &self.events[i];
            i += 1;

            if evt.y >= to {
                return true;
            }

            if fill_rule.is_out(evt.winding) {
                //println!("B {} {:?}", evt.y, fill_rule);
                return false;
            }
        }

        //println!("A");

        false
    }

    pub fn add_edge(&mut self, from: f32, to: f32, edge_winding: i16) {
        if from == to {
            return;
        }

        // TODO: I think they are already top to bottom.
        let y0 = from.min(to);
        let y1 = from.max(to);

        // Keep track of the winding at current y.
        let mut winding = 0;
        let mut i = 0;

        // Iterate over events up to the start of the range.
        loop {
            if i >= self.events.len() {
                self.events.push(SideEvent { y: y0, winding: edge_winding });
                self.events.push(SideEvent { y: y1, winding: 0 });
                return;
            }

            let e = self.events[i];
            if e.y == y0 {
                if e.winding + edge_winding == winding {
                    // Simplify consecutive events as winding is the same.
                    //  +      +
                    //  |      |
                    // -+- -> -|-
                    //  |      |
                    self.events.remove(i);
                } else {
                    //  +      +
                    //  |      |
                    // -+- -> -+-
                    //  |      |
                    winding = e.winding + edge_winding;
                    self.events[i].winding = winding;
                    i += 1;
                }

                break;
            } else if e.y > y0 {
                //  +      +
                //  |      |
                // -|- -> -+-
                //  |      |
                //  +      +
                winding = winding + edge_winding;
                self.events.insert(i, SideEvent { y: y0, winding });
                i += 1;
                break;
            }

            winding = e.winding;
            i += 1;
        }

        // Iterate over events up to the end of the range.
        loop {
            if i == self.events.len() {
                self.events.push(SideEvent { y: y1, winding: winding - edge_winding });
                break;
            }

            let e = self.events[i];
            if e.y > y1 {
                self.events.insert(i, SideEvent { y: y1, winding: winding - edge_winding });
                break;
            } else if e.y == y1 {
                if e.winding == winding {
                    self.events.remove(i);
                }

                break;
            } else {
                self.events[i].winding += edge_winding;
            }

            winding = e.winding + edge_winding;

            i += 1;
        }
    }
}

pub(crate) fn apply_side_edges_to_backdrop(side: &SideEdgeTracker, y_offset: f32, backdrops: &mut [f32; 16]) {
    let mut prev: Option<SideEvent> = None;
    for evt in side.events() {
        if let Some(prev) = prev {
            let y0 = (prev.y - y_offset).max(0.0).min(15.0);
            let y1 = (evt.y - y_offset).max(0.0).min(15.0);
            let winding = prev.winding as f32;
            let y0_px = y0.floor();
            let y1_px = y1.floor();
            let first_px = y0_px as usize;
            let last_px = y1_px as usize;

            backdrops[first_px] += winding * (y0_px + 1.0 - y0);
            for i in first_px + 1 ..= last_px {
                backdrops[i] += winding;
            }
            backdrops[last_px] -= winding * (y1_px + 1.0 - y1);
        }

        if evt.winding != 0 {
            prev = Some(*evt);
        } else {
            prev = None;
        }
    }
}

#[test]
fn side_edges_backdrop() {
    let mut side = SideEdgeTracker::new();
    side.add_edge(0.0, 16.0, 1);

    let mut backdrops = [0.0; 16];
    apply_side_edges_to_backdrop(&side, 0.0, &mut backdrops);

    println!("{:?}", backdrops);

    let mut side = SideEdgeTracker::new();
    side.add_edge(0.25, 3.25, 1);

    let mut backdrops = [0.0; 16];
    apply_side_edges_to_backdrop(&side, 0.0, &mut backdrops);

    println!("{:?}", backdrops);


    let mut side = SideEdgeTracker::new();
    side.add_edge(0.25, 0.75, 1);

    let mut backdrops = [0.0; 16];
    apply_side_edges_to_backdrop(&side, 0.0, &mut backdrops);

    println!("{:?}", backdrops);

    let mut side = SideEdgeTracker::new();
    side.add_edge(-0.5, 5.0, 1);
    side.add_edge(10.0, 16.5, 1);

    let mut backdrops = [0.0; 16];
    apply_side_edges_to_backdrop(&side, 0.0, &mut backdrops);

    println!("{:?}", backdrops);

    //panic!();
}

#[test]
fn side_edges() {
    let mut side = SideEdgeTracker::new();

    side.add_edge(0.2, 0.5, 1);

    assert_eq!(&side.events[..], &[
        SideEvent { y: 0.2, winding: 1 },
        SideEvent { y: 0.5, winding: 0 },
    ]);

    side.add_edge(0.5, 0.7, 1);

    assert_eq!(&side.events[..], &[
        SideEvent { y: 0.2, winding: 1 },
        SideEvent { y: 0.7, winding: 0 },
    ]);

    side.add_edge(0.4, 0.7, -1);

    assert_eq!(&side.events[..], &[
        SideEvent { y: 0.2, winding: 1 },
        SideEvent { y: 0.4, winding: 0 },
    ]);

    side.add_edge(0.8, 0.9, 1);

    assert_eq!(&side.events[..], &[
        SideEvent { y: 0.2, winding: 1 },
        SideEvent { y: 0.4, winding: 0 },
        SideEvent { y: 0.8, winding: 1 },
        SideEvent { y: 0.9, winding: 0 },
    ]);

    side.add_edge(0.4, 0.8, -1);

    assert_eq!(&side.events[..], &[
        SideEvent { y: 0.2, winding: 1 },
        SideEvent { y: 0.4, winding: -1 },
        SideEvent { y: 0.8, winding: 1 },
        SideEvent { y: 0.9, winding: 0 },
    ]);

    side.add_edge(0.4, 0.8, 1);

    assert_eq!(&side.events[..], &[
        SideEvent { y: 0.2, winding: 1 },
        SideEvent { y: 0.4, winding: 0 },
        SideEvent { y: 0.8, winding: 1 },
        SideEvent { y: 0.9, winding: 0 },
    ]);

    side.add_edge(0.4, 0.8, 1);

    assert_eq!(&side.events[..], &[
        SideEvent { y: 0.2, winding: 1 },
        SideEvent { y: 0.9, winding: 0 },
    ]);

    side.add_edge(0.2, 0.9, 1);

    assert_eq!(&side.events[..], &[
        SideEvent { y: 0.2, winding: 2 },
        SideEvent { y: 0.9, winding: 0 },
    ]);

    side.add_edge(0.2, 0.9, -2);

    assert_eq!(side.events.len(), 0);

    side.add_edge(0.3, 0.6, 1);
    side.add_edge(0.2, 0.7, 1);

    assert_eq!(&side.events[..], &[
        SideEvent { y: 0.2, winding: 1 },
        SideEvent { y: 0.3, winding: 2 },
        SideEvent { y: 0.6, winding: 1 },
        SideEvent { y: 0.7, winding: 0 },
    ]);

    side.clear();

    side.add_edge(0.2, 0.7, 1);
    side.add_edge(0.3, 0.6, 1);

    assert_eq!(&side.events[..], &[
        SideEvent { y: 0.2, winding: 1 },
        SideEvent { y: 0.3, winding: 2 },
        SideEvent { y: 0.6, winding: 1 },
        SideEvent { y: 0.7, winding: 0 },
    ]);

    side.clear();

    side.add_edge(0.2, 0.6, 1);
    side.add_edge(0.3, 0.7, 1);

    assert_eq!(&side.events[..], &[
        SideEvent { y: 0.2, winding: 1 },
        SideEvent { y: 0.3, winding: 2 },
        SideEvent { y: 0.6, winding: 1 },
        SideEvent { y: 0.7, winding: 0 },
    ]);

    side.clear();

    side.add_edge(0.3, 0.7, 1);
    side.add_edge(0.2, 0.6, 1);

    assert_eq!(&side.events[..], &[
        SideEvent { y: 0.2, winding: 1 },
        SideEvent { y: 0.3, winding: 2 },
        SideEvent { y: 0.6, winding: 1 },
        SideEvent { y: 0.7, winding: 0 },
    ]);
}

struct UnsafeSendPtr<T>(pub *mut T);
unsafe impl<T> Send for UnsafeSendPtr<T> {}
unsafe impl<T> Sync for UnsafeSendPtr<T> {}
impl<T> Copy for UnsafeSendPtr<T> {}
impl<T> Clone for UnsafeSendPtr<T> { fn clone(&self) -> Self { *self } }
impl<T> UnsafeSendPtr<T> {
    fn get(self) -> *mut T { self.0 }
}
