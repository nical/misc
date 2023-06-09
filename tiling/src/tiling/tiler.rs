use ordered_float::OrderedFloat;
pub use lyon::path::math::{Point, point, Vector, vector};
pub use lyon::path::{PathEvent, FillRule};
pub use lyon::geom::euclid::default::{Box2D, Size2D, Transform2D};
pub use lyon::geom::euclid;
pub use lyon::geom;
use lyon::{geom::{LineSegment, QuadraticBezierSegment, CubicBezierSegment}};

use crate::pattern::BuiltPattern;
use crate::{tiling::*, TILE_SIZE_F32};

use crate::tiling::encoder::*;
use crate::tiling::occlusion::TileMask;

use copyless::VecHelper;

struct Row {
    edges: Vec<RowEdge>,
    tile_y: u32,
}

pub struct DrawParams {
    pub tolerance: f32,
    pub fill_rule: FillRule,
    pub max_edges_per_gpu_tile: usize,
    pub inverted: bool,
    pub encoded_fill_rule: u16,
}

/// A context object that can bin path edges into tile grids.
pub struct Tiler {
    pub draw: DrawParams,

    size: Size2D<f32>,
    scissor: Box2D<f32>,

    num_tiles_x: u32,
    num_tiles_y: f32,

    flatten: bool,

    rows: Vec<Row>,
    active_edges: Vec<ActiveEdge>,

    first_row: usize,
    last_row: usize,

    pub edges: Vec<LineEdge>,

    pub row_decomposition_time_ns: u64,
    pub tile_decomposition_time_ns: u64,
    // For debugging.
    pub selected_row: Option<usize>,

    pub color_tiles_per_row: u32,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct TilerConfig {
    pub view_box: Box2D<f32>,
    pub tolerance: f32,
    pub flatten: bool,
    pub mask_atlas_size: Size2D<u32>,
    pub color_atlas_size: Size2D<u32>,
    pub staging_buffer_size: u32,
}

impl TilerConfig {
    pub fn num_tiles(&self) -> Size2D<u32> {
        let w = self.view_box.size().to_u32().width;
        let h = self.view_box.size().to_u32().height;
        let ts = crate::TILE_SIZE;
        Size2D::new(
            (w + ts - 1) / ts,
            (h + ts - 1) / ts,
        )
    }
}

unsafe impl Sync for Tiler {}

impl Tiler {
    /// Constructor.
    pub fn new(config: &TilerConfig) -> Self {
        let size = config.view_box.size();
        let num_tiles_y = f32::ceil(size.height / crate::TILE_SIZE_F32);
        Tiler {
            draw: DrawParams {
                tolerance: config.tolerance,
                fill_rule: FillRule::NonZero,
                inverted: false,
                max_edges_per_gpu_tile: 4096,
                encoded_fill_rule: 1,
            },
            size,
            scissor: Box2D::from_size(size),
            num_tiles_x: f32::ceil(size.width / crate::TILE_SIZE_F32) as u32,
            num_tiles_y,
            flatten: config.flatten,

            first_row: 0,
            last_row: 0,

            row_decomposition_time_ns: 0,
            tile_decomposition_time_ns: 0,

            active_edges: Vec::with_capacity(64),
            rows: Vec::new(),

            edges: Vec::with_capacity(8192),

            selected_row: None,

            color_tiles_per_row: 0,
        }
    }

    /// Using init instead of creating a new tiler allows recycling allocations from
    /// a previous tiling run.
    pub fn init(&mut self, view_box: &Box2D<f32>) {
        let size = view_box.size();
        self.size = size;
        self.scissor = Box2D::from_size(size);
        self.num_tiles_x = f32::ceil(size.width / TILE_SIZE_F32) as u32;
        self.num_tiles_y = f32::ceil(size.height / TILE_SIZE_F32);
        self.edges.clear();
    }

    fn set_fill_rule(&mut self, fill_rule: FillRule, inverted: bool) {
        self.draw.fill_rule = fill_rule;
        self.draw.encoded_fill_rule = match fill_rule {
            FillRule::EvenOdd => 0,
            FillRule::NonZero => 1,
        };
        if inverted {
            self.draw.encoded_fill_rule |= 2;
        }
        self.draw.inverted = inverted;
    }

    pub fn set_scissor(&mut self, scissor: &Box2D<f32>) {
        self.scissor = scissor.intersection_unchecked(&Box2D::from_size(self.size));
    }

    pub fn scissor(&self) -> &Box2D<f32> {
        &self.scissor
    }

    pub fn fill_path(
        &mut self,
        path: impl Iterator<Item = PathEvent>,
        options: &FillOptions,
        pattern: &BuiltPattern,
        tile_mask: &mut TileMask,
        encoder: &mut TileEncoder,
        device: &wgpu::Device,
    ) {
        profiling::scope!("tile_path");

        assert!(tile_mask.width() >= self.num_tiles_x, "{} >= {}", tile_mask.width(), self.num_tiles_x);
        assert!(tile_mask.height() >= self.num_tiles_y as u32, "{} >= {}", tile_mask.height(), self.num_tiles_y,);

        let t0 = time::precise_time_ns();

        self.set_fill_rule(options.fill_rule, options.inverted);
        self.draw.tolerance = options.tolerance;
        encoder.prerender_pattern = options.prerender_pattern;
        let identity = Transform2D::identity();
        let transform = options.transform.unwrap_or(&identity);

        self.begin_path();

        if self.flatten {
            self.assign_rows_linear(transform, path);
        } else {
            self.assign_rows_quadratic(transform, path);
        }

        let t1 = time::precise_time_ns();

        self.end_path(encoder, tile_mask, pattern, device);

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

    fn affected_rows(&self, y_min: f32, y_max: f32) -> (usize, usize) {
        self.affected_range(
            y_min, y_max,
            self.scissor.min.y,
            self.scissor.max.y,
        )
    }

    fn affected_range(&self, min: f32, max: f32, scissor_min: f32, scissor_max: f32) -> (usize, usize) {
        let inv_tile_size = 1.0 / TILE_SIZE_F32;
        let first_row_y = (scissor_min * inv_tile_size).floor();
        let last_row_y = (scissor_max * inv_tile_size).ceil();

        let y_start_tile = f32::floor(min * inv_tile_size).max(first_row_y);
        let y_end_tile = f32::ceil(max * inv_tile_size).min(last_row_y);

        let start_idx = y_start_tile as usize;
        let end_idx = (y_end_tile as usize).max(start_idx);

        (start_idx, end_idx)
    }

    /// Can be used to tile a segment manually.
    ///
    /// Should only be called between begin_path and end_path.
    pub fn add_monotonic_edge(&mut self, edge: &MonotonicEdge) {
        debug_assert!(edge.from.y <= edge.to.y);

        let max = self.num_tiles_y * TILE_SIZE_F32;

        if edge.from.y > max || edge.to.y < 0.0 {
            return;
        }

        let (start_idx, end_idx) = self.affected_rows(edge.from.y, edge.to.y);

        self.first_row = self.first_row.min(start_idx);
        self.last_row = self.last_row.max(end_idx);

        let offset_min = 0.0;
        let offset_max = TILE_SIZE_F32;
        let mut row_idx = start_idx as u32;
        if edge.is_line() {
            for row in &mut self.rows[start_idx .. end_idx] {
                let y_offset = row_idx as f32 * TILE_SIZE_F32;

                let mut segment = LineSegment { from: edge.from, to: edge.to };
                segment.from.y -= y_offset;
                segment.to.y -= y_offset;
                let range = clip_line_segment_1d(segment.from.y, segment.to.y, offset_min, offset_max);
                let mut segment = segment.split_range(range.clone());

                // Most of the tiling algorithm isn't affected by float precision hazards except where
                // we split the edges. Ideally we want the split points for edges that cross tile boundaries
                // to be exactly at the tile boundaries, so that we can easily detect edges that cross the
                // tile's upper side to count backdrop winding numbers. So we do a bit of snapping here to
                // paper over the imprecision of splitting the edge.
                const SNAP: f32 = 0.05;
                if segment.from.y - offset_min < SNAP { segment.from.y = offset_min };
                if segment.to.y - offset_min < SNAP { segment.to.y = offset_min };
                if offset_max - segment.from.y < SNAP { segment.from.y = offset_max };
                if offset_max - segment.to.y < SNAP { segment.to.y = offset_max };

                if edge.winding < 0 {
                    std::mem::swap(&mut segment.from, &mut segment.to);
                }

                row.edges.alloc().init(RowEdge {
                    from: segment.from,
                    to: segment.to,
                    ctrl: point(segment.from.x, std::f32::NAN),
                    min_x: OrderedFloat(segment.from.x.min(segment.to.x)),
                });

                row_idx += 1;
            }
        } else {
            for row in &mut self.rows[start_idx .. end_idx] {
                let y_offset = row_idx as f32 * TILE_SIZE_F32;

                let mut segment = QuadraticBezierSegment { from: edge.from, ctrl: edge.ctrl, to: edge.to };
                segment.from.y -= y_offset;
                segment.ctrl.y -= y_offset;
                segment.to.y -= y_offset;

                clip_quadratic_bezier_to_row(&mut segment, offset_min, offset_max);

                if edge.winding < 0 {
                    std::mem::swap(&mut segment.from, &mut segment.to);
                }

                row.edges.alloc().init(RowEdge {
                    from: segment.from,
                    to: segment.to,
                    ctrl: segment.ctrl,
                    min_x: OrderedFloat(segment.from.x.min(segment.to.x)),
                });

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
            });
        }

        for row in &mut self.rows {
            row.edges.clear();
        }
    }

    /// Process manually edges and encode them into the output encoder.
    pub fn end_path(&mut self, encoder: &mut TileEncoder, tile_mask: &mut TileMask, pattern: &BuiltPattern, device: &wgpu::Device) {
        if self.draw.inverted {
            self.first_row = 0;
            self.last_row = self.num_tiles_y as usize;
        }
        if self.first_row >= self.last_row {
            return;
        }

        let mut active_edges = std::mem::take(&mut self.active_edges);
        active_edges.clear();

        if let Some(row) = self.selected_row {
            self.first_row = row;
            self.last_row = row + 1;
        }

        encoder.begin_path(pattern);

        let mut edge_buffer = std::mem::take(&mut self.edges);

        // borrow-ck dance.
        let mut rows = std::mem::take(&mut self.rows);
        // This could be done in parallel but it's already quite fast serially.
        for row in &mut rows[self.first_row..self.last_row] {
            if row.edges.is_empty() && !self.draw.inverted {
                continue;
            }

            self.process_row(
                row.tile_y,
                &mut row.edges[..],
                &mut active_edges,
                &mut tile_mask.row(row.tile_y),
                pattern,
                encoder,
                &mut edge_buffer,
                device,
            );
        }

        self.edges = edge_buffer;
        self.active_edges = active_edges;
        self.rows = rows;

        for row in &mut self.rows {
            row.edges.clear();
        }
    }

    fn assign_rows_quadratic(
        &mut self,
        transform: &Transform2D<f32>,
        path: impl Iterator<Item = PathEvent>,
    ) {
        profiling::scope!("assign_rows_quadratic");
        let square_tolerance = self.draw.tolerance * self.draw.tolerance;
        let mut from = point(0.0, 0.0);
        for evt in path {
            match evt {
                PathEvent::Begin { at, .. } => {
                    from = at;
                }
                PathEvent::End { first, .. } => {
                    if first == from {
                        continue;
                    }
                    let segment = LineSegment { from, to: first }.transformed(transform);
                    let edge = MonotonicEdge::linear(segment);
                    self.add_monotonic_edge(&edge);
                    from = first;
                }
                PathEvent::Line { to, .. } => {
                    let segment = LineSegment { from, to }.transformed(transform);
                    if segment.to_vector().square_length() < square_tolerance {
                        continue;
                    }
                    let edge = MonotonicEdge::linear(segment);
                    self.add_monotonic_edge(&edge);
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
                    segment.for_each_monotonic(&mut|monotonic| {
                        let edge = MonotonicEdge::quadratic(*monotonic);
                        self.add_monotonic_edge(&edge);
                    });
                    from = to;
                }
                PathEvent::Cubic { ctrl1, ctrl2, to, .. } => {
                    let segment = CubicBezierSegment { from, ctrl1, ctrl2, to }.transformed(transform);
                    if segment.baseline().to_vector().square_length() < square_tolerance {
                        let center = (segment.from + segment.to.to_vector()) * 0.5;
                        if (segment.ctrl1 - center).square_length() < square_tolerance
                        && (segment.ctrl2 - center).square_length() < square_tolerance {
                            continue;
                        }
                    }
                    segment.for_each_quadratic_bezier(self.draw.tolerance, &mut|segment| {
                        segment.for_each_monotonic(&mut|monotonic| {
                            let edge = MonotonicEdge::quadratic(*monotonic);
                            self.add_monotonic_edge(&edge);
                        });
                    });
                    from = to;
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
        let mut from = point(0.0, 0.0);
        let square_tolerance = self.draw.tolerance * self.draw.tolerance;
        for evt in path {
            match evt {
                PathEvent::Begin { at } => {
                    from = at;
                }
                PathEvent::End { first, .. } => {
                    let segment = LineSegment { from, to: first }.transformed(transform);
                    let edge = MonotonicEdge::linear(segment);
                    self.add_monotonic_edge(&edge);
                    from = first;
                }
                PathEvent::Line { to, .. } => {
                    let segment = LineSegment { from, to }.transformed(transform);
                    let edge = MonotonicEdge::linear(segment);
                    if segment.to_vector().square_length() < square_tolerance {
                        continue;
                    }
                    self.add_monotonic_edge(&edge);
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
                    segment.for_each_flattened(self.draw.tolerance, &mut|segment| {
                        let edge = MonotonicEdge::linear(*segment);
                        self.add_monotonic_edge(&edge);
                    });
                    from = to;
                }
                PathEvent::Cubic { ctrl1, ctrl2, to, .. } => {
                    let segment = CubicBezierSegment { from, ctrl1, ctrl2, to }.transformed(transform);
                    if segment.baseline().to_vector().square_length() < square_tolerance {
                        let center = (segment.from + segment.to.to_vector()) * 0.5;
                        if (segment.ctrl1 - center).square_length() < square_tolerance
                        && (segment.ctrl2 - center).square_length() < square_tolerance {
                            continue;
                        }
                    }
                    segment.for_each_flattened(self.draw.tolerance, &mut|segment| {
                        let edge = MonotonicEdge::linear(*segment);
                        self.add_monotonic_edge(&edge);
                    });
                    from = to;
                }
            }
        }
    }

    fn process_row(
        &self,
        tile_y: u32,
        row: &mut [RowEdge],
        active_edges: &mut Vec<ActiveEdge>,
        coarse_mask: &mut TileMaskRow,
        pattern: &BuiltPattern,
        encoder: &mut TileEncoder,
        edge_buffer: &mut Vec<LineEdge>,
        device: &wgpu::Device,
    )  {
        //println!("--------- row {}", tile_y);
        row.sort_unstable_by(|a, b| a.min_x.cmp(&b.min_x));

        active_edges.clear();

        let inv_tw = 1.0 / TILE_SIZE_F32;
        let tiles_start = (self.scissor.min.x * inv_tw).floor();
        let tiles_end = self.num_tiles_x.min((self.scissor.max.x * inv_tw).ceil() as u32);

        let tx = tiles_start * TILE_SIZE_F32;
        let ty = tile_y as f32 * TILE_SIZE_F32;
        let output_rect = Box2D {
            min: point(tx, ty),
            max: point(tx + TILE_SIZE_F32, ty + TILE_SIZE_F32)
        };

        // The inner rect is equivalent to the output rect with an y offset so that
        // its upper side is at y=0.
        let rect = Box2D {
            min: point(output_rect.min.x, 0.0),
            max: point(output_rect.max.x, TILE_SIZE_F32),
        };

        let x = tiles_start as u32;
        let mut tile = TileInfo {
            x,
            y: tile_y,
            rect,
            backdrop: 0,
        };

        let mut current_edge = 0;

        // First iterate on edges until we reach one that starts inside the tiling area.
        // During this phase we only need to keep track of the backdrop winding number
        // and detect edges that end in the tiling area.
        for edge in &row[..] {
            if edge.min_x.0 >= tile.rect.min.x {
                break;
            }

            active_edges.alloc().init(ActiveEdge {
                from: edge.from,
                to: edge.to,
                ctrl: edge.ctrl,
            });

            while edge.min_x.0 > tile.rect.max.x {
                Self::update_active_edges(active_edges, tile.rect.min.x, tile.rect.min.y, &mut tile.backdrop);
            }

            current_edge += 1;
        }

        // Iterate over edges in the tiling area.
        // Now we produce actual tiles.
        //
        // Each time we get to a new tile, we remove all active edges that end side of the tile.
        // In practice this means all active edges intersect the current tile.
        for edge in &row[current_edge..] {
            while edge.min_x.0 > tile.rect.max.x && tile.x < tiles_end {
                if active_edges.is_empty() {
                    let tx = ((edge.min_x.0 / TILE_SIZE_F32) as u32).min(tiles_end);
                    assert!(tx > tile.x, "next edge {:?} clip {} tile start {:?}", edge, self.scissor.max.x, tile.rect.min.x);
                    if self.draw.fill_rule.is_in(tile.backdrop) ^ self.draw.inverted {
                        encoder.span(tile.x..tx, tile.y, coarse_mask, pattern);
                    }
                    let n = tx-tile.x;
                    self.update_tile_rects(&mut tile, n);
                } else {
                    self.masked_tile(&mut tile, active_edges, coarse_mask, pattern, encoder, edge_buffer, device);
                }
            }

            if tile.x >= tiles_end {
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

        // Continue iterating over tiles until there is no active edge or we are out of the tiling area..
        while tile.x < tiles_end && !active_edges.is_empty() {
            self.masked_tile(&mut tile, active_edges, coarse_mask, pattern, encoder, edge_buffer, device);
        }

        if self.draw.inverted {
            encoder.span(tile.x..tiles_end, tile.y, coarse_mask, pattern);
        }
    }

    pub fn update_tile_rects(&self, tile: &mut TileInfo, num_tiles: u32) {
        // TODO: pass the target tile instead of difference.
        let nt = num_tiles as f32;
        let d = TILE_SIZE_F32 * nt;
        // Note: this is accumulating precision errors, but haven't seen
        // issues in practice.
        tile.rect.min.x += d;
        tile.rect.max.x += d;
        tile.x += num_tiles;
    }

    fn masked_tile(
        &self,
        tile: &mut TileInfo,
        active_edges: &mut Vec<ActiveEdge>,
        coarse_mask: &mut TileMaskRow,
        pattern: &BuiltPattern,
        encoder: &mut TileEncoder,
        edge_buffer: &mut Vec<LineEdge>,
        device: &wgpu::Device,
    ) {
        let visibility = pattern.tile_visibility(tile.x, tile.y);

        let opaque = false;
        if visibility != TileVisibility::Empty && coarse_mask.test(tile.x, opaque) {
            let tile_position = TilePosition::new(tile.x, tile.y);

            let mask_tile = encoder.add_fill_mask(tile, &self.draw, active_edges, edge_buffer, device);
            encoder.add_tile(pattern, opaque, tile_position, mask_tile);
        }

        self.update_tile_rects(tile, 1);

        Self::update_active_edges(
            active_edges,
            tile.rect.min.x,
            tile.rect.min.y,
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
                if (edge.from.y == tile_y || edge.to.y == tile_y) && edge.from.y != edge.to.y {
                    *backdrop += if edge.from.y < edge.to.y { 1 } else { -1 };
                }

                active_edges.swap_remove(i);
            }

            if i == 0 {
                break;
            }
            i -= 1;
        }
    }

    pub fn fill_canvas(
        &mut self,
        pattern: &BuiltPattern,
        tile_mask: &mut TileMask,
        encoder: &mut TileEncoder,
    ) {
        encoder.begin_path(pattern);

        let (column_start, column_end) = self.affected_range(
            self.scissor.min.x,
            self.scissor.max.x,
            self.scissor.min.x,
            self.scissor.max.x,
        );

        for tile_y in 0..self.rows.len() as u32 {
            let mut tile_mask = tile_mask.row(tile_y);

            let range = column_start as u32 .. column_end as u32;
            encoder.span(range, tile_y, &mut tile_mask, pattern);
        }
    }

    pub fn update_stats(&self, stats: &mut Stats) {
        stats.edges += self.edges.len();
        stats.row_time += Duration::from_nanos(self.row_decomposition_time_ns);
        stats.tile_time += Duration::from_nanos(self.tile_decomposition_time_ns);
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

        // TODO: offsetting edges by half a pixel so that the
        // result matches what canvas/SVG expect: lines at integer
        // pixel coordinates show anti-aliasing while lines landing
        // on `.5` corrdinates are crisp.
        // Note also the the offset begin unapplied in mask_fill.wgsl.
        // This is obviously the wrong way to deal with this. The tile
        // bounds probably should be offset instead of of applying/unapplying
        // it on edges.
        let half_px = vector(0.5, 0.5);

        MonotonicEdge {
            from: segment.from - half_px,
            to: segment.to - half_px,
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

        let half_px = vector(0.5, 0.5);
        MonotonicEdge {
            from: segment.from - half_px,
            to: segment.to - half_px,
            ctrl: segment.ctrl - half_px,
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

pub struct TileInfo {
    /// X-offset in number of tiles.
    pub x: u32,
    /// Y-offset in number of tiles.
    pub y: u32,
    /// Rectangle of the tile aligned with the tile grid.
    pub rect: Box2D<f32>,

    pub backdrop: i16,
}

fn clip_quadratic_bezier_to_row(
    segment: &mut QuadraticBezierSegment<f32>,
    min: f32,
    max: f32,
) {
    let from = segment.from.y;
    let ctrl = segment.ctrl.y;
    let to = segment.to.y;

    debug_assert!(max >= min);
    debug_assert!(to >= from);

    const SNAP: f32 = 0.05;

    if from >= min && to <= max {
        if segment.from.y - min < SNAP { segment.from.y = min };
        if segment.to.y - min < SNAP { segment.to.y = min };
        return;
    }

    // TODO: this is sensitive to float errors, should probably
    // be using f64 arithmetic

    // Solve a classic quadratic formula "a*x² + b*x + c = 0"
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

    // Because of precision issues when computing the split range and when
    // splitting, the split point won't be exactly at the tile's upper side.
    // It's not an issue if the split point ends up above, but it breaks counting
    // backdrop winding numbers when the split point is below, so we do two things:
    // If one of the original segment's endpoint is above the tile's upper side the
    // one of the split segment's endpoint *must* be on the upper side (see snap_from
    // and snap_to). We also snap endpoints to the tile sides if they are within
    // a threshold.

    let snap_from = segment.from.y < min;
    let snap_to = segment.to.y < min;

    *segment = segment.split_range(t1..t2);

    if snap_from || segment.from.y - min < SNAP { segment.from.y = min };
    if snap_to || segment.to.y - min < SNAP { segment.to.y = min };
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

pub fn as_scale_offset(m: &Transform2D<f32>) -> Option<(Vector, Vector)> {
    // Same as Skia's SK_ScalarNearlyZero.
    const ESPILON: f32 = 1.0 / 4096.0;

    if m.m12.abs() > ESPILON || m.m21.abs() > ESPILON {
        return None;
    }

    Some((
        vector(m.m11, m.m22),
        vector(m.m31, m.m32),
    ))
}

pub fn affected_range(min: f32, max: f32, scissor_min: f32, scissor_max: f32) -> (usize, usize) {
    let inv_tile_size = 1.0 / crate::TILE_SIZE_F32;
    let first_row_y = (scissor_min * inv_tile_size).floor();
    let last_row_y = (scissor_max * inv_tile_size).ceil();

    let y_start_tile = f32::floor(min * inv_tile_size).max(first_row_y);
    let y_end_tile = f32::ceil(max * inv_tile_size).min(last_row_y);

    let start_idx = y_start_tile as usize;
    let end_idx = (y_end_tile as usize).max(start_idx);

    (start_idx, end_idx)
}

