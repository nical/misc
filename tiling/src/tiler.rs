use ordered_float::OrderedFloat;
pub use lyon::path::math::{Point, point, Vector, vector};
pub use lyon::path::{PathEvent, FillRule};
pub use lyon::geom::euclid::default::{Box2D, Size2D, Transform2D};
pub use lyon::geom::euclid;
pub use lyon::geom;
use lyon::{geom::{LineSegment, QuadraticBezierSegment, CubicBezierSegment}, path::Winding};

use crate::tile_encoder::TileAllocator;

pub use crate::occlusion::{TileMask, TileMaskRow};
use crate::tile_encoder::TileEncoder;
use crate::tile_renderer::TileInstance;

use copyless::VecHelper;

const INVALID_TILE_ID: TilePosition = TilePosition::INVALID;

pub struct FillOptions<'l> {
    pub fill_rule: FillRule,
    pub tolerance: f32,
    pub merge_tiles: bool,
    pub prerender_pattern: bool,
    pub transform: Option<&'l Transform2D<f32>>,
}

impl<'l> FillOptions<'l> {
    pub fn new() -> FillOptions<'static> {
        FillOptions {
            fill_rule: FillRule::EvenOdd,
            tolerance: 0.1,
            merge_tiles: true,
            prerender_pattern: false,
            transform: None,
        }
    }

    pub fn transformed<'a>(transform: &'a Transform2D<f32>) -> FillOptions<'a> {
        FillOptions {
            fill_rule: FillRule::EvenOdd,
            tolerance: 0.1,
            merge_tiles: true,
            prerender_pattern: false,
            transform: Some(transform),
        }
    }

    pub fn with_transform<'a>(self, transform: Option<&'a Transform2D<f32>>) -> FillOptions<'a> 
    where 'l: 'a
    {
        FillOptions {
            fill_rule: self.fill_rule,
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

    pub fn with_merged_tiles(mut self, merge_tiles: bool) -> Self {
        self.merge_tiles = merge_tiles;
        self
    }

    pub fn with_prerendered_pattern(mut self, prerender: bool) -> Self {
        self.prerender_pattern = prerender;
        self
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Stats {
    pub opaque_tiles: usize,
    pub alpha_tiles: usize,
    pub prerendered_tiles: usize,
    pub gpu_mask_tiles: usize,
    pub cpu_mask_tiles: usize,
    pub edges: usize,
    pub batches: usize,
}

impl Stats {
    pub fn new() -> Self {
        Stats {
            opaque_tiles: 0,
            alpha_tiles: 0,
            prerendered_tiles: 0,
            gpu_mask_tiles: 0,
            cpu_mask_tiles: 0,
            edges: 0,
            batches: 0,
        }
    }

    pub fn clear(&mut self) {
        *self = Stats::new();
    }
}

struct Row {
    edges: Vec<RowEdge>,
    tile_y: u32,
}

pub struct DrawParams {
    pub tolerance: f32,
    pub tile_size: Size2D<f32>,
    pub fill_rule: FillRule,
    pub z_index: u16,
    pub is_clip_in: bool,
    pub max_edges_per_gpu_tile: usize,
    pub use_quads: bool,
    pub merge_solid_tiles: bool,
    pub encoded_fill_rule: u16,
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
    pub draw: DrawParams,

    size: Size2D<f32>,
    scissor: Box2D<f32>,
    tile_padding: f32,

    num_tiles_x: u32,
    num_tiles_y: f32,

    flatten: bool,

    rows: Vec<Row>,
    active_edges: Vec<ActiveEdge>,

    first_row: usize,
    last_row: usize,

    output_is_tiled: bool,

    pub row_decomposition_time_ns: u64,
    pub tile_decomposition_time_ns: u64,
    // For debugging.
    pub selected_row: Option<usize>,

    solid_color_tile_id: TilePosition,
    pub color_tiles_per_row: u32,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct TilerConfig {
    pub view_box: Box2D<f32>,
    pub tile_size: Size2D<f32>,
    pub tile_padding: f32,
    pub tolerance: f32,
    pub flatten: bool,
    pub mask_atlas_size: Size2D<u32>,
}

impl TilerConfig {
    pub fn num_tiles(&self) -> Size2D<u32> {
        let w = self.view_box.size().to_u32().width;
        let h = self.view_box.size().to_u32().height;
        let tw = self.tile_size.width as u32;
        let th = self.tile_size.height as u32;
        Size2D::new(
            (w + tw - 1) / tw,
            (h + th - 1) / th,
        )
    }
}

unsafe impl Sync for Tiler {}

impl Tiler {
    /// Constructor.
    pub fn new(config: &TilerConfig) -> Self {
        let size = config.view_box.size();
        let num_tiles_y = f32::ceil(size.height / config.tile_size.height);
        Tiler {
            draw: DrawParams {
                z_index: 0,
                tolerance: config.tolerance,
                tile_size: config.tile_size,
                fill_rule: FillRule::NonZero,
                is_clip_in: false,
                max_edges_per_gpu_tile: 4096,
                use_quads: false,
                merge_solid_tiles: true,
                encoded_fill_rule: 1,
            },
            size,
            scissor: Box2D::from_size(size),
            tile_padding: config.tile_padding,
            num_tiles_x: f32::ceil(size.width / config.tile_size.width) as u32,
            num_tiles_y,
            flatten: config.flatten,

            first_row: 0,
            last_row: 0,

            row_decomposition_time_ns: 0,
            tile_decomposition_time_ns: 0,

            active_edges: Vec::with_capacity(64),
            rows: Vec::new(),

            output_is_tiled: false,

            selected_row: None,

            solid_color_tile_id: TilePosition::ZERO.with_flag(),
            color_tiles_per_row: 0,
        }
    }

    /// Using init instead of creating a new tiler allows recycling allocations from
    /// a previous tiling run.
    pub fn init(&mut self, config: &TilerConfig) {
        let size = config.view_box.size();
        self.size = size;
        self.scissor = Box2D::from_size(size);
        self.draw.tile_size = config.tile_size;
        self.tile_padding = config.tile_padding;
        self.num_tiles_x = f32::ceil(size.width / config.tile_size.width) as u32;
        self.num_tiles_y = f32::ceil(size.height / config.tile_size.height);
        self.draw.tolerance = config.tolerance;
        self.flatten = config.flatten;
    }

    pub fn set_fill_rule(&mut self, fill_rule: FillRule) {
        self.draw.fill_rule = fill_rule;
        self.draw.encoded_fill_rule = match fill_rule {
            FillRule::EvenOdd => 0,
            FillRule::NonZero => 1,
        };
    }

    pub fn set_scissor(&mut self, scissor: &Box2D<f32>) {
        self.scissor = scissor.intersection_unchecked(&Box2D::from_size(self.size));
    }

    /// Tile an entire path.
    ///
    /// This internally does all of the steps:
    /// - begin_Path
    /// - add_monotonic_edge
    /// - end_path
    pub fn fill_path(
        &mut self,
        path: impl Iterator<Item = PathEvent>,
        options: &FillOptions,
        pattern: &mut dyn TilerPattern,
        tile_mask: &mut TileMask,
        tiled_output: Option<&mut TiledOutput>,
        encoder: &mut TileEncoder,
    ) {
        profiling::scope!("tile_path");

        assert!(tile_mask.width() >= self.num_tiles_x);
        assert!(tile_mask.height() >= self.num_tiles_y as u32);

        let t0 = time::precise_time_ns();

        self.draw.tolerance = options.tolerance;
        self.draw.merge_solid_tiles = options.merge_tiles && pattern.is_mergeable();
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

        self.end_path(encoder, tile_mask, tiled_output, pattern);

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
            self.draw.tile_size.height
        )
    }

    fn affected_range(&self, min: f32, max: f32, scissor_min: f32, scissor_max: f32, tile_size: f32) -> (usize, usize) {
        let inv_tile_size = 1.0 / tile_size;
        let first_row_y = (scissor_min * inv_tile_size).floor();
        let last_row_y = (scissor_max * inv_tile_size).ceil();

        let y_start_tile = f32::floor((min - self.tile_padding) * inv_tile_size).max(first_row_y);
        let y_end_tile = f32::ceil((max + self.tile_padding) * inv_tile_size).min(last_row_y);

        let start_idx = y_start_tile as usize;
        let end_idx = (y_end_tile as usize).max(start_idx);

        (start_idx, end_idx)

    }

    /// Can be used to tile a segment manually.
    ///
    /// Should only be called between begin_path and end_path.
    pub fn add_monotonic_edge(&mut self, edge: &MonotonicEdge) {
        debug_assert!(edge.from.y <= edge.to.y);

        let min = -self.tile_padding;
        let max = self.num_tiles_y * self.draw.tile_size.height + self.tile_padding;

        if edge.from.y > max || edge.to.y < min {
            return;
        }

        let (start_idx, end_idx) = self.affected_rows(edge.from.y, edge.to.y);

        self.first_row = self.first_row.min(start_idx);
        self.last_row = self.last_row.max(end_idx);

        let offset_min = -self.tile_padding;
        let offset_max = self.draw.tile_size.height + self.tile_padding;
        let mut row_idx = start_idx as u32;
        if edge.is_line() {
            for row in &mut self.rows[start_idx .. end_idx] {
                let y_offset = row_idx as f32 * self.draw.tile_size.height;

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
                let y_offset = row_idx as f32 * self.draw.tile_size.height;

                let mut segment = QuadraticBezierSegment { from: edge.from, ctrl: edge.ctrl, to: edge.to };
                segment.from.y -= y_offset;
                segment.ctrl.y -= y_offset;
                segment.to.y -= y_offset;

                clip_quadratic_bezier_to_row(&mut segment, offset_min, offset_max);

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
            });
        }

        for row in &mut self.rows {
            row.edges.clear();
        }
    }

    /// Process manually edges and encode them into the output encoder.
    pub fn end_path(&mut self, encoder: &mut TileEncoder, tile_mask: &mut TileMask, mut tiled_output: Option<&mut TiledOutput>, pattern: &mut dyn TilerPattern) {
        if self.first_row >= self.last_row {
            return;
        }

        if let Some(ref mut output) = tiled_output {
            if pattern.pattern_kind() == SOLID_COLOR_PATTERN && pattern.tile_is_opaque() {
                // TODO: if the previous path has the same pattern we don't need to allocate
                // a new solid tile.
                self.solid_color_tile_id = output.tile_allocator.allocate().0.with_flag();
            }
        }

        let mut active_edges = std::mem::take(&mut self.active_edges);
        active_edges.clear();

        if let Some(row) = self.selected_row {
            self.first_row = row;
            self.last_row = row + 1;
        }

        self.output_is_tiled = tiled_output.is_some();

        encoder.begin_path(pattern);

        // borrow-ck dance.
        let mut rows = std::mem::take(&mut self.rows);
        // This could be done in parallel but it's already quite fast serially.
        for row in &mut rows[self.first_row..self.last_row] {
            if row.edges.is_empty() {
                continue;
            }

            let tiled_output = if let Some(ref mut output) = &mut tiled_output {
                Some((output.indirection_buffer.row_mut(row.tile_y), &mut output.tile_allocator))
            } else {
                None
            };

            self.process_row(
                row.tile_y,
                &mut row.edges[..],
                &mut active_edges,
                &mut tile_mask.row(row.tile_y),
                tiled_output,
                pattern,
                encoder,
            );
        }

        self.active_edges = active_edges;
        self.rows = rows;

        for row in &mut self.rows {
            row.edges.clear();
        }
    }

    /*
    pub fn render_indirect_tiles(&mut self, indirection: &IndirectionBuffer) {
        for row_idx in 0..(self.num_tiles_y) as u32 {
            let row = indirection.row(row_idx);
            for (i, &src_tile_id) in row.iter().enumerate() {
                if src_tile_id == INVALID_TILE_ID {
                    continue;
                }

                let src_x = (src_tile_id * self.color_tiles_per_row) as f32 * self.draw.tile_size.width;
                let src_y = (src_tile_id / self.color_tiles_per_row) as f32 * self.draw.tile_size.height;

                let dst_x = i as f32 + self.draw.tile_size.width;
                let dst_y = row_idx as f32 * self.draw.tile_size.height;

                let src_rect = Box2D::from_origin_and_size(point(src_x, src_y), self.draw.tile_size);
                let dst_rect = Box2D::from_origin_and_size(point(dst_x, dst_y), self.draw.tile_size);

                unimplemented!();
            }
        }
    }
    */

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
                        let edge = MonotonicEdge::quadratic(*monotonic);
                        self.add_monotonic_edge(&edge);
                    });
                }
                PathEvent::Cubic { from, ctrl1, ctrl2, to } => {
                    let segment = CubicBezierSegment { from, ctrl1, ctrl2, to }.transformed(transform);
                    segment.for_each_quadratic_bezier(self.draw.tolerance, &mut|segment| {
                        segment.for_each_monotonic(&mut|monotonic| {
                            let edge = MonotonicEdge::quadratic(*monotonic);
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
                    segment.for_each_flattened(self.draw.tolerance, &mut|segment| {
                        let edge = MonotonicEdge::linear(*segment);
                        self.add_monotonic_edge(&edge);
                    });
                }
                PathEvent::Cubic { from, ctrl1, ctrl2, to } => {
                    let segment = CubicBezierSegment { from, ctrl1, ctrl2, to }.transformed(transform);
                    segment.for_each_flattened(self.draw.tolerance, &mut|segment| {
                        let edge = MonotonicEdge::linear(*segment);
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
        coarse_mask: &mut TileMaskRow,
        mut tiled_output: Option<(&mut [TilePosition], &mut TileAllocator)>,
        pattern: &mut dyn TilerPattern,
        encoder: &mut TileEncoder,
    ) {
        //println!("--------- row {}", tile_y);
        encoder.begin_row();
        row.sort_unstable_by(|a, b| a.min_x.cmp(&b.min_x));

        active_edges.clear();

        let inv_tw = 1.0 / self.draw.tile_size.width;
        let tiles_start = (self.scissor.min.x * inv_tw).floor();
        let tiles_end = self.num_tiles_x.min((self.scissor.max.x * inv_tw).ceil() as u32);

        let tx = tiles_start * self.draw.tile_size.width;
        let ty = tile_y as f32 * self.draw.tile_size.height;
        let output_rect = Box2D {
            min: point(tx, ty),
            max: point(tx + self.draw.tile_size.width, ty + self.draw.tile_size.height)
        };

        // The inner rect is equivalent to the output rect with an y offset so that
        // its upper side is at y=0.
        let inner_rect = Box2D {
            min: point(output_rect.min.x, 0.0),
            max: point(output_rect.max.x, self.draw.tile_size.height),
        };

        let outer_rect = Box2D {
            min: point(
                inner_rect.min.x - self.tile_padding,
                -self.tile_padding,
            ),
            max: point(
                inner_rect.max.x + self.tile_padding,
                self.draw.tile_size.height + self.tile_padding,
            ),
        };

        let x = tiles_start as u32;
        let mut tile = TileInfo {
            x,
            y: tile_y,
            index: tile_y * self.num_tiles_x + x,
            inner_rect,
            outer_rect,
            backdrop: 0,
        };

        let mut current_edge = 0;

        // First iterate on edges until we reach one that starts inside the tiling area.
        // During this phase we only need to keep track of the backdrop winding number
        // and detect edges that end in the tiling area.
        for edge in &row[..] {
            if edge.min_x.0 >= tile.outer_rect.min.x {
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
            while edge.min_x.0 > tile.outer_rect.max.x && tile.x < tiles_end {
                self.finish_tile(&mut tile, active_edges, coarse_mask, pattern, &mut tiled_output, encoder);
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
            self.finish_tile(&mut tile, active_edges, coarse_mask, pattern, &mut tiled_output, encoder);
        }

        encoder.end_row();
    }

    fn set_indirect_output_rect(&self, tile_x: u32, opaque: bool, output_indirection_buffer: &mut[TilePosition], tile_alloc: &mut TileAllocator) -> TilePosition {
        let mut color_tile_id = output_indirection_buffer[tile_x as usize];
        if color_tile_id == INVALID_TILE_ID {
            color_tile_id = if opaque {
                // Since we are processing content front-to-back, all solid opaque tiles for
                // a given path are the same, we easily can de-duplicate it.
                // TODO: we are still drawing multiple times over the same opaque tile.
                self.solid_color_tile_id.with_flag()
            } else {
                // allocate a spot in.
                tile_alloc.allocate().0
            };
            output_indirection_buffer[tile_x as usize] = color_tile_id;
        } else if opaque {
            output_indirection_buffer[tile_x as usize].add_flag()
        }

        color_tile_id
    }

    fn finish_tile(
        &self,
        tile: &mut TileInfo,
        active_edges: &mut Vec<ActiveEdge>,
        coarse_mask: &mut TileMaskRow,
        pattern: &mut dyn TilerPattern,
        tiled_output: &mut Option<(&mut[TilePosition], &mut TileAllocator)>,
        encoder: &mut TileEncoder,
    ) {
        let mut full_tile = false;
        let mut empty = false;
        if active_edges.is_empty() {
            let is_in = self.draw.fill_rule.is_in(tile.backdrop);
            full_tile = is_in;
            empty = !is_in;
        }

        pattern.set_tile(tile.x, tile.y);

        let opaque = full_tile && pattern.tile_is_opaque();
        empty = empty || pattern.tile_is_empty();

        // Decide where to draw the tile.
        // Two configurations: Either we render in a plain target in which case the position
        // corresponds to the actual coordinates of the tile's content, or we are rendering
        // to a tiled intermediate target in which case the destination is linearly allocated.
        // The indirection buffer is used to determine whether an aallocation was already made
        // for this position.
        let tile_position = if let Some((indirection_buffer, tile_alloc)) = tiled_output {
            self.set_indirect_output_rect(tile.x, opaque, indirection_buffer, tile_alloc)
        } else {
            TilePosition::new(tile.x, tile.y)
        };

        if !empty && coarse_mask.test(tile.x, opaque) {
            let mask_tile = if full_tile {
                TilePosition::ZERO
            } else {
                encoder.add_fill_mask(tile, &self.draw, active_edges)
            };

            encoder.add_tile(pattern, opaque, tile_position, mask_tile);
        }

        if empty && self.draw.is_clip_in {
            // For clip-in it's the empty tiles that completely mask content out.
            coarse_mask.write_clip(tile.x);
        }

        tile.inner_rect.min.x += self.draw.tile_size.width;
        tile.inner_rect.max.x += self.draw.tile_size.width;
        tile.outer_rect.min.x += self.draw.tile_size.width;
        tile.outer_rect.max.x += self.draw.tile_size.width;
        tile.x += 1;
        tile.index += 1;

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

    pub fn fill_circle(
        &mut self,
        mut center: Point,
        mut radius: f32,
        options: &FillOptions,
        pattern: &mut dyn TilerPattern,
        tile_mask: &mut TileMask,
        tiled_output: Option<&mut TiledOutput>,
        encoder: &mut TileEncoder,
    ) {
        encoder.prerender_pattern = options.prerender_pattern;

        let mut simple = true;
        if let Some(transform) = &options.transform {
            simple = false;
            if let Some((scale, offset)) = as_scale_offset(transform) {
                if (scale.x - scale.y).abs() < 0.01 {
                    center = center * scale.x + offset;
                    radius = radius * scale.x;
                    simple = true;
                }
            }
        }

        if simple {
            self.fill_transformed_circle(
                center,
                radius,
                pattern,
                tile_mask,
                tiled_output,
                encoder
            );
        } else {
            let mut path = lyon::path::Path::builder();
            path.add_circle(center, radius, Winding::Positive);
            let path = path.build();

            self.fill_path(
                path.iter(),
                options,
                pattern,
                tile_mask,
                tiled_output,
                encoder,
            );
        }
    }

    fn fill_transformed_circle(
        &mut self,
        center: Point,
        radius: f32,
        pattern: &mut dyn TilerPattern,
        tile_mask: &mut TileMask,
        mut tiled_output: Option<&mut TiledOutput>,
        encoder: &mut TileEncoder,
    ) {
        self.output_is_tiled = tiled_output.is_some();

        let y_min = center.y - radius;
        let y_max = center.y + radius;
        let x_min = center.x - radius;
        let x_max = center.x + radius;
        let (row_start, row_end) = self.affected_rows(y_min, y_max);
        let (column_start, column_end) = self.affected_range(
            x_min, x_max,
            self.scissor.min.x,
            self.scissor.max.x,
            self.draw.tile_size.width,
        );
        let row_start = row_start as u32;
        let row_end = row_end as u32;
        let column_start = column_start as u32;
        let column_end = column_end as u32;
        let tile_radius = std::f32::consts::SQRT_2 * 0.5 * self.draw.tile_size.width + self.tile_padding;
        encoder.begin_path(pattern);

        for tile_y in row_start..row_end {
            let mut tile_mask = tile_mask.row(tile_y);
            let mut tile_center = point(
                (column_start as f32 + 0.5) * self.draw.tile_size.width,
                (tile_y as f32 + 0.5) * self.draw.tile_size.height,
            );
            encoder.begin_row();

            let mut dummy_tile_alloc = TileAllocator::new(0, 0);
            let dummy_buffer: &mut [TilePosition] = &mut [];
            let (output_indirection_buffer, output_tile_alloc) = if let Some(ref mut output) = &mut tiled_output {
                (output.indirection_buffer.row_mut(tile_y), &mut output.tile_allocator)
            } else {
                (dummy_buffer, &mut dummy_tile_alloc)
            };

            for tile_x in column_start .. column_end {
                let tc = tile_center;
                tile_center.x += self.draw.tile_size.width;

                let d = (tc - center).length();
                if d - tile_radius > radius {
                    continue;
                }

                pattern.set_tile(tile_x, tile_y);

                if pattern.tile_is_empty() {
                    continue;
                }

                let opaque = d + tile_radius < radius && pattern.tile_is_opaque();

                if !tile_mask.test(tile_x as u32, opaque) {
                    continue;
                }

                let mut mask_id = TilePosition::ZERO;
                if !opaque {
                    let tile_offset = vector(tile_x as f32, tile_y as f32) * self.draw.tile_size.width;
                    let center = center - tile_offset;
                    mask_id = encoder.add_cricle_mask(center, radius);
                }

                let tile_position = if self.output_is_tiled {
                    self.set_indirect_output_rect(tile_x, opaque, output_indirection_buffer, output_tile_alloc)
                } else {
                    TilePosition::new(tile_x, tile_y)
                };

                encoder.add_tile(pattern, opaque, tile_position, mask_id);
            }
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

pub struct TileInfo {
    /// X-offset in number of tiles.
    pub x: u32,
    /// Y-offset in number of tiles.
    pub y: u32,
    /// y * width + x
    pub index: u32,
    /// Rectangle of the tile aligned with the tile grid.
    pub inner_rect: Box2D<f32>,
    /// Rectangle including the tile padding.
    pub outer_rect: Box2D<f32>,

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

pub struct TiledOutput {
    pub indirection_buffer: IndirectionBuffer,
    pub tile_allocator: TileAllocator,
}

pub struct IndirectionBuffer {
    data: Vec<TilePosition>,
    size: Size2D<u32>,
}

impl IndirectionBuffer {
    pub fn new(size: Size2D<u32>) -> Self {
        IndirectionBuffer {
            data: vec![INVALID_TILE_ID; size.area() as usize],
            size
        }
    }

    pub fn row(&self, row: u32) -> &[TilePosition] {
        debug_assert!(row < self.size.height);
        let start = (row * self.size.width) as usize;
        let end = start + self.size.width as usize;

        &self.data[start..end]
    }

    pub fn row_mut(&mut self, row: u32) -> &mut[TilePosition] {
        debug_assert!(row < self.size.height);
        let start = (row * self.size.width) as usize;
        let end = start + self.size.width as usize;

        &mut self.data[start..end]
    }

    pub fn reset(&mut self) {
        self.data.fill(INVALID_TILE_ID);
    }

    pub fn get(&self, x: u32, y: u32) -> Option<(TilePosition, bool)> {
        let idx = (y * self.size.width + x) as usize;
        let data = self.data[idx];
        if data == INVALID_TILE_ID {
            return None;
        }

        Some((data, data.flag()))
    }
}

pub type PatternKind = u32;
pub const TILED_IMAGE_PATTERN: PatternKind = 0;
pub const SOLID_COLOR_PATTERN: PatternKind = 1;

pub type PatternData = u32;

// Note: the statefullness with set_tile(x, y) followed by tile-specific getters
// is unfortunate, the reason for it at the moment is the need to query whether
// a tile is empty or fully opaque before culling, while we want to only request
// tile if we really need to, that is after culling.
pub trait TilerPattern {
    /// Simply put, what type of shader do we use to draw the tiles in the main
    /// color pass.
    /// A lot of fancy patterns would pre-render into a tiled color atlas, so their
    /// pattern kind is TILED_IMAGE_PATTERN.
    fn pattern_kind(&self) -> PatternKind;

    fn set_render_pass(&mut self, pass_idx: u32);
    fn set_tile(&mut self, x: u32, y: u32);
    fn tile_data(&mut self) -> PatternData;
    fn opaque_tiles(&mut self) -> &mut Vec<TileInstance>;
    fn prerender_tile(&mut self, atlas: &mut TileAllocator) -> (TilePosition, PatternData);
    fn tile_is_opaque(&self) -> bool { false }
    fn tile_is_empty(&self) -> bool { false }
    fn is_mergeable(&self) -> bool { false }
}

pub struct TiledSourcePattern {
    indirection_buffer: IndirectionBuffer,
    current_tile: u32,
    current_tile_is_opaque: bool,
    current_tile_is_empty: bool,
}

impl TilerPattern for TiledSourcePattern {
    fn set_render_pass(&mut self, _: u32) {}
    fn pattern_kind(&self) -> u32 { TILED_IMAGE_PATTERN }
    fn set_tile(&mut self, x: u32, y: u32) {
        if let Some((tile, opaque)) = self.indirection_buffer.get(x, y) {
            self.current_tile = tile.to_u32();
            self.current_tile_is_opaque = opaque;
            self.current_tile_is_empty = false;
        } else {
            self.current_tile = std::u32::MAX;
            self.current_tile_is_opaque = false;
            self.current_tile_is_empty = true;
        }
    }
    fn opaque_tiles(&mut self) -> &mut Vec<TileInstance> {
        unimplemented!();
    }

    fn prerender_tile(&mut self, _: &mut TileAllocator) -> (TilePosition, PatternData) {
        // TODO: should the first ine be current_tile?
        (TilePosition::INVALID, self.current_tile)
    }

    fn tile_data(&mut self) -> PatternData {
        self.current_tile
    }
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
    pub fn to_u32(&self) -> u32 { self.0 }
    pub fn x(&self) -> u32 { (self.0 >> 10) & Self::MASK }
    pub fn y(&self) -> u32 { (self.0) & Self::MASK }
    pub fn extension(&self) -> u32 { (self.0 >> 20) & Self::MASK }

    // TODO: we have two unused bits and we use one of them to store
    // whether a tile in an indirection buffer is opaque. That's not
    // great.
    pub fn flag(&self) -> bool { self.0 & 1 << 31 != 0 }
    pub fn add_flag(&mut self) { self.0 |= 1 << 31 }
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
