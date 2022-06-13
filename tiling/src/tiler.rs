use ordered_float::OrderedFloat;
pub use lyon::path::math::{Point, point, Vector, vector};
pub use lyon::path::{PathEvent, FillRule};
pub use lyon::geom::euclid::default::{Box2D, Size2D, Transform2D};
pub use lyon::geom::euclid;
pub use lyon::geom;
use lyon::geom::{LineSegment, QuadraticBezierSegment, CubicBezierSegment};

pub use crate::occlusion::TileMask;
use crate::tile_encoder::TileEncoder;

use parasol::{Context, CachePadded};

use copyless::VecHelper;

struct Row {
    edges: Vec<RowEdge>,
    tile_y: u32,
    z_buffer: TileMask,
}

impl Row {
    fn new() -> Self {
        Row {
            edges: Vec::new(),
            tile_y: 0,
            z_buffer: TileMask::new()
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
    active_edges: Vec<ActiveEdge>,

    worker_data: Vec<CachePadded<TilerWorkerData>>,

    first_row: usize,
    last_row: usize,

    pub row_decomposition_time_ns: u64,
    pub tile_decomposition_time_ns: u64,
    pub is_opaque: bool,
    pub fill_rule: FillRule,
    pub z_index: u16,
    // For debugging.
    pub selected_row: Option<usize>,
}

#[derive(Clone)]
struct TilerWorkerData {
    active_edges: Vec<ActiveEdge>,
    // TODO: put the (parallel) tile encoder here.
}

impl TilerWorkerData {
    fn new() -> Self {
        TilerWorkerData {
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

            active_edges: Vec::with_capacity(64),
            rows: Vec::new(),

            is_opaque: true,
            fill_rule: FillRule::EvenOdd,
            z_index: 0,

            worker_data: Vec::new(),

            selected_row: None,
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
        encoder: &mut TileEncoder,
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

    /// Can be used to tile a segment manually.
    ///
    /// Should only be called between begin_path and end_path.
    pub fn add_line_segment(&mut self, edge: &LineSegment<f32>) {
        self.add_monotonic_edge(&MonotonicEdge::linear(*edge));
    }

    /// Can be used to tile a segment manually.
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

        let start_idx = y_start_tile as usize;
        let end_idx = y_end_tile as usize;
        self.first_row = self.first_row.min(start_idx);
        self.last_row = self.last_row.max(end_idx);

        let offset_min = -self.tile_padding;
        let offset_max = self.tile_size.height + self.tile_padding;
        let mut row_idx = start_idx as u32;
        if edge.is_line() {
            for row in &mut self.rows[start_idx .. end_idx] {
                let y_offset = self.tile_offset_y + row_idx as f32 * self.tile_size.height;

                let mut segment = LineSegment { from: edge.from, to: edge.to };
                segment.from.y -= y_offset;
                segment.to.y -= y_offset;
                let range = clip_line_segment_1d(segment.from.y, segment.to.y, offset_min, offset_max);
                let mut segment = segment.split_range(range.clone());

                // Most of the tiling algorithm isn't affected by float precision hazards except where
                // we split the edges. Ideally we want the split points for edges that cross tile boundaries
                // to be exactly at the tile boundaries, so that the side edge tracker can properly sum up to
                // an empty list of edges by the end of the row (and easily detect full/empty tiles). So we do
                // a bit of snapping here to paper over the imprecision of splitting the edge.
                const SNAP: f32 = 0.05;
                if segment.from.y - offset_min < SNAP { segment.from.y = offset_min };
                if segment.to.y - offset_min < SNAP { segment.to.y = offset_min };
                if offset_max - segment.from.y < SNAP { segment.from.y = offset_max };
                if offset_max - segment.to.y < SNAP { segment.to.y = offset_max };

                if edge.winding < 0 {
                    std::mem::swap(&mut segment.from, &mut segment.to);
                }

                if segment.from.y != segment.to.y {
                    row.edges.alloc().init(RowEdge {
                        from: segment.from,
                        to: segment.to,
                        ctrl: point(segment.from.x, std::f32::NAN),
                        min_x: OrderedFloat(segment.from.x.min(segment.to.x)),
                    });
                }

                row_idx += 1;
            }
        } else {
            for row in &mut self.rows[start_idx .. end_idx] {
                let y_offset = self.tile_offset_y + row_idx as f32 * self.tile_size.height;

                let mut segment = QuadraticBezierSegment { from: edge.from, ctrl: edge.ctrl, to: edge.to };
                segment.from.y -= y_offset;
                segment.ctrl.y -= y_offset;
                segment.to.y -= y_offset;
                let range = clip_quadratic_bezier_1d(segment.from.y, segment.ctrl.y, segment.to.y, offset_min, offset_max);
                let mut segment = segment.split_range(range.clone());

                const SNAP: f32 = 0.05;
                if segment.from.y - offset_min < SNAP { segment.from.y = offset_min };
                if segment.to.y - offset_min < SNAP { segment.to.y = offset_min };
                if offset_max - segment.from.y < SNAP { segment.from.y = offset_max };
                if offset_max - segment.to.y < SNAP { segment.to.y = offset_max };

                if edge.winding < 0 {
                    std::mem::swap(&mut segment.from, &mut segment.to);
                }

                if segment.from.y != segment.to.y {
                    row.edges.alloc().init(RowEdge {
                        from: segment.from,
                        to: segment.to,
                        ctrl: segment.ctrl,
                        min_x: OrderedFloat(segment.from.x.min(segment.to.x)),
                    });
                }

                row_idx += 1;
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
            self.rows.alloc().init(Row {
                edges: Vec::new(),
                tile_y: i as u32,
                z_buffer: TileMask::new(),
            });
        }

        for row in &mut self.rows {
            row.edges.clear();
        }
    }

    /// Process manually edges and encode them into the output encoder.
    pub fn end_path(&mut self, encoder: &mut TileEncoder) {
        let mut active_edges = std::mem::take(&mut self.active_edges);
        active_edges.clear();

        if let Some(row) = self.selected_row {
            self.first_row = row;
            self.last_row = row + 1;
        }

        // borrow-ck dance.
        let mut rows = std::mem::take(&mut self.rows);
        // This could be done in parallel but it's already quite fast serially.
        for row in &mut rows[self.first_row..self.last_row] {
            if row.edges.is_empty() {
                continue;
            }

            if row.z_buffer.is_empty() {
                row.z_buffer.init(self.num_tiles_x as usize);
            }

            self.process_row(
                row.tile_y,
                &mut row.edges[..],
                &mut active_edges,
                &mut row.z_buffer,
                encoder,
            );
        }

        self.active_edges = active_edges;
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
        encoders: &mut [&mut TileEncoder],
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

    pub fn end_path_parallel(&mut self, ctx: &mut Context, encoders: &mut [&mut TileEncoder]) {
        if self.worker_data.is_empty() {
            self.worker_data = vec![CachePadded::new(TilerWorkerData::new()); ctx.num_contexts() as usize];
        }
        // Basically what we are doing here is passing the encoders rows to the
        // worker threads in the most unsafe of ways, and being careful to never have multiple
        // worker threads accessing the same encoder (there's one per parallel context)
        // we should be using the context_data thing instead.
        struct Shared<'a, 'b> {
            encoders: &'b mut [&'a mut TileEncoder],
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

                    if row.z_buffer.is_empty() {
                        row.z_buffer.init(self.num_tiles_x as usize);
                    }

                    let idx = if ctx.is_worker_thread()  { ctx.id().index() } else { ctx.num_worker_threads() as usize };
                    self.process_row(
                        row.tile_y,
                        &mut row.edges[..],
                        &mut worker_data.active_edges,
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
    pub fn end_path_rayon(&mut self, encoders: &mut [&mut TileEncoder]) {
        use rayon::prelude::*;

        // borrow-ck dance.
        let mut rows = std::mem::take(&mut self.rows);
        let mut z_buffer = std::mem::take(&mut self.z_buffer);

        // Basically what we are doing here is passing the encoders and z_buffer rows to the
        // worker threads in the most unsafe of ways, and being careful to never have multiple
        // worker threads accessing the same encoder (there's one per worker), nor the same z buffers
        // (there is one per row and rows are dispatched in parallel)
        struct Shared<'a, 'b> {
            encoders: &'b mut [&'a mut TileEncoder],
            z_buffer: &'b mut [TileMask],
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

            let z_buffer: &mut TileMask = &mut (*shared_ptr.get()).z_buffer[row.tile_y as usize];

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
        z_buffer: &mut TileMask,
        encoder: &mut TileEncoder,
    ) {
        //println!("--------- row {}", tile_y);
        encoder.begin_row();
        row.sort_unstable_by(|a, b| a.min_x.cmp(&b.min_x));

        active_edges.clear();

        let inner_rect = Box2D {
            min: point(self.tile_offset_x, 0.0),
            max: point(
                self.tile_offset_x + self.tile_size.width,
                self.tile_size.height
            ),
        };

        let outer_rect = Box2D {
            min: point(
                inner_rect.min.x - self.tile_padding,
                -self.tile_padding,
            ),
            max: point(
                inner_rect.max.x + self.tile_padding,
                self.tile_size.height + self.tile_padding,
            ),
        };

        let mut tile = TileInfo {
            x: self.tile_offset_x as u32,
            y: tile_y,
            inner_rect,
            outer_rect,
            solid: false,
            backdrop: 0,
        };

        let mut current_edge = 0;

        // First iterate on edges until we reach one that starts inside the tiling area.
        // During this phase we only need to keep track of the backdrop winding number
        // and detect edges that end in the tiling area.
        for edge in &row[..] {
            if edge.min_x.0 >= self.tile_offset_x {
                break;
            }

            active_edges.alloc().init(ActiveEdge {
                from: edge.from,
                to: edge.to,
                ctrl: edge.ctrl,
            });

            while edge.min_x.0 > tile.outer_rect.max.x {
                Self::update_active_edges(active_edges, tile.outer_rect.min.x, tile.outer_rect.min.y, &mut tile.backdrop);
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
                self.finish_tile(&mut tile, active_edges, z_buffer, encoder);
            }

            if tile.x >= self.num_tiles_x {
                break;
            }

            // Note: it is tempting to flatten here to avoid re-flattening edges that
            // touch several tiles, however a naive attempt at doing that was slower.
            // That's partly because other things get slower when the number of active
            // edges grow (for example the side edges bookkeeping).
            // Maybe it would make sense to revisit if the size of the active edge
            // struct can be reduced.
            active_edges.alloc().init(ActiveEdge {
                from: edge.from,
                to: edge.to,
                ctrl: edge.ctrl,
            });

            current_edge += 1;
        }

        // At this point we visited all edges but not necessarily all tiles.
        while tile.x < self.num_tiles_x {
            self.finish_tile(&mut tile, active_edges, z_buffer, encoder);

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
        z_buffer: &mut TileMask,
        encoder: &mut TileEncoder,
    ) {
        tile.solid = false;
        let mut empty = false;
        if active_edges.is_empty() {
            let is_in = self.fill_rule.is_in(tile.backdrop);
            if is_in {
                tile.solid = true;
            } else {
                // Empty tile.
                empty = true;
            }
        }

        if !empty && z_buffer.test(tile.x, tile.solid && self.is_opaque) {
            encoder.encode_tile(tile, active_edges);
        }

        tile.inner_rect.min.x += self.tile_size.width;
        tile.inner_rect.max.x += self.tile_size.width;
        tile.outer_rect.min.x += self.tile_size.width;
        tile.outer_rect.max.x += self.tile_size.width;
        tile.x += 1;

        Self::update_active_edges(
            active_edges,
            tile.outer_rect.min.x,
            tile.outer_rect.min.y,
            &mut tile.backdrop
        );
    }

    #[inline]
    fn update_active_edges(active_edges: &mut Vec<ActiveEdge>, left_x: f32, tile_y: f32, backdrop: &mut i16) {
        // Equivalent to the following snippet but a bit faster and not preserving the
        // edge ordering.
        //
        // active_edges.retain(|edge| {
        //     if edge.max_x() < left_x {
        //         if edge.from.y == tile_y || edge.to.y == tile_y && edge.from.y != edge.to.y {
        //             *backdrop += if edge.from.y < edge.to.y { 1 } else { -1 };
        //         }
        //         return false
        //     }
        //     true
        // });

        if active_edges.is_empty() {
            return;
        }

        let mut i = active_edges.len() - 1;
        loop {
            let edge = &active_edges[i];
            if edge.max_x() < left_x {
                if edge.from.y == tile_y || edge.to.y == tile_y && edge.from.y != edge.to.y {
                    *backdrop += if edge.from.y < edge.to.y { 1 } else { -1 };
                    //println!(" # bacdrop {} (removing edge {:?})", *backdrop, edge);
                } else {
                    //println!(" # remove {:?}", edge);
                }

                active_edges.swap_remove(i);
            }

            if i == 0 {
                break;
            }
            i -= 1;
        }
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
    fn is_line(&self) -> bool { self.ctrl.y.is_nan() }

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
            ctrl: point(segment.from.x, std::f32::NAN),
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
            kind: EdgeKind::Quadratic,
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
    min_x: OrderedFloat<f32>,
}

impl RowEdge {
    #[allow(unused)]
    fn print(&self) {
        if self.ctrl.y.is_nan() {
            println!("Line({:?} {:?}) ", self.from, self.to);
        } else {
            println!("Quad({:?} {:?} {:?}) ", self.from, self.ctrl, self.to);
        }
    }

    #[allow(unused)]
    fn print_svg(&self) {
        self.print_svg_offset(vector(0.0, 0.0));
    }

    #[allow(unused)]
    fn print_svg_offset(&self, offset: Vector) {
        if self.ctrl.y.is_nan() {
            println!("  <path d=\" M {:?} {:?} L {:?} {:?}\"/>", self.from.x - offset.x, self.from.y - offset.x, self.to.x - offset.x, self.to.y - offset.x);
        } else {
            println!("  <path d=\" M {:?} {:?} Q {:?} {:?} {:?} {:?}\"/>", self.from.x - offset.x, self.from.y - offset.x, self.ctrl.x - offset.x, self.ctrl.y - offset.x, self.to.x - offset.x, self.to.y - offset.x);
        }
    }
}

/// The edge representation in the list of active edges of a tile.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ActiveEdge {
    // If we ever decide to flatten early, active edges could be just from/to
    // and fit in 16 bytes instead of 32.
    pub from: Point,
    pub to: Point,
    pub ctrl: Point,
}

impl ActiveEdge {
    pub fn is_line(&self) -> bool { self.ctrl.y.is_nan() }
    pub fn is_curve(&self) -> bool { !self.is_line() }
    fn max_x(&self) -> f32 { self.from.x.max(self.to.x).max(self.ctrl.x) }
}

#[test]
fn active_edge_size() {
    println!("{}", std::mem::size_of::<ActiveEdge>());
}

impl ActiveEdge {
    pub fn clip_horizontally(&self, x_range: std::ops::Range<f32>) -> Self {
        if self.is_line() {
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
                ctrl: point(segment.from.x, std::f32::NAN),
            }
        } else {
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

            let mut segment = split_quad(&segment, range);

            if swap {
                std::mem::swap(&mut segment.from, &mut segment.to);
            }

            ActiveEdge {
                from: segment.from,
                to: segment.to,
                ctrl: segment.ctrl,
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

    pub backdrop: i16,
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

    if from >= min && to <= max {
        return 0.0 .. 1.0;
    }

    // TODO: this is sensible to float errors, should probably
    // be using f64 arithmetic

    // Solve a class quadratic formula "a*x² + b*x + c = 0"
    // using the quadratic bézier's formulation
    // y = (1 - t)² * from + t*(1 - t) * ctrl + t² * to
    // replacing y with min and max we get:
    let a = from + to - 2.0 * ctrl;
    let b = 2.0 * ctrl - 2.0 * from;
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
        return 0.0 .. 1.0;
    }

    let inv_d = 1.0 / d;

    let t0 = ((min - from) * inv_d).max(0.0);
    let t1 = ((max - from) * inv_d).min(1.0);

    t0 .. t1
}

struct UnsafeSendPtr<T>(pub *mut T);
unsafe impl<T> Send for UnsafeSendPtr<T> {}
unsafe impl<T> Sync for UnsafeSendPtr<T> {}
impl<T> Copy for UnsafeSendPtr<T> {}
impl<T> Clone for UnsafeSendPtr<T> { fn clone(&self) -> Self { *self } }
impl<T> UnsafeSendPtr<T> {
    fn get(self) -> *mut T { self.0 }
}

fn split_quad(curve: &QuadraticBezierSegment<f32>, t_range: std::ops::Range<f32>) -> QuadraticBezierSegment<f32> {
    let t0 = t_range.start;
    let t1 = t_range.end;

    let from = if t0 == 0.0 { curve.from } else { curve.sample(t0) };
    let to = if t1 == 1.0 { curve.to } else { curve.sample(t1) };
    let ctrl = from + (curve.ctrl - curve.from).lerp(curve.to - curve.ctrl, t0) * (t1 - t0);

    QuadraticBezierSegment { from, ctrl, to }
}