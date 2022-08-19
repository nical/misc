use std::sync::{Arc, atomic::{AtomicU32, Ordering}};
use ordered_float::OrderedFloat;
pub use lyon::path::math::{Point, point, Vector, vector};
pub use lyon::path::{PathEvent, FillRule};
pub use lyon::geom::euclid::default::{Box2D, Size2D, Transform2D};
pub use lyon::geom::euclid;
pub use lyon::geom;
use lyon::geom::{LineSegment, QuadraticBezierSegment, CubicBezierSegment};

pub use crate::occlusion::{TileMask, TileMaskRow};
use crate::tile_encoder::TileEncoder;
use crate::Color;

use copyless::VecHelper;

const INVALID_TILE_ID: u32 = std::u32::MAX;
const TILE_OPACITY_BIT: u32 = 1 << 31;

struct Row {
    edges: Vec<RowEdge>,
    tile_y: u32,
}

pub struct DrawParams {
    pub tolerance: f32,
    pub tile_size: Size2D<f32>,
    pub fill_rule: FillRule,
    pub is_opaque: bool,
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

    pub color_tile_ids: TileIdAllcator,
    solid_color_tile_id: u32,
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
    pub fn new(config: &TilerConfig, color_tile_ids: TileIdAllcator) -> Self {
        let size = config.view_box.size();
        let num_tiles_y = f32::ceil(size.height / config.tile_size.height);
        Tiler {
            draw: DrawParams {
                z_index: 0,
                is_opaque: true,
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

            color_tile_ids,
            solid_color_tile_id: 0,
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
        self.color_tile_ids.reset();
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
    pub fn tile_path(
        &mut self,
        path: impl Iterator<Item = PathEvent>,
        transform: Option<&Transform2D<f32>>,
        tile_mask: &mut TileMask,
        output_indirection: Option<&mut IndirectionBuffer>,
        pattern: &mut dyn TilerPattern,
        encoder: &mut TileEncoder,
    ) {
        profiling::scope!("tile_path");

        assert!(tile_mask.width() >= self.num_tiles_x);
        assert!(tile_mask.height() >= self.num_tiles_y as u32);

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

        self.end_path(encoder, tile_mask, output_indirection, pattern);

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

        let min = -self.tile_padding;
        let max = self.num_tiles_y * self.draw.tile_size.height + self.tile_padding;

        if edge.from.y > max || edge.to.y < min {
            return;
        }

        let inv_tile_height = 1.0 / self.draw.tile_size.height;
        let first_row_y = (self.scissor.min.y * inv_tile_height).floor();
        let last_row_y = (self.scissor.max.y * inv_tile_height).ceil();

        let y_start_tile = f32::floor((edge.from.y - self.tile_padding) * inv_tile_height).max(first_row_y);
        let y_end_tile = f32::ceil((edge.to.y + self.tile_padding) * inv_tile_height).min(last_row_y);

        let start_idx = y_start_tile as usize;
        let end_idx = (y_end_tile as usize).max(start_idx);
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

        if self.output_is_tiled && self.draw.is_opaque {
            // TODO: if the previous path has the same pattern we don't need to allocate
            // a new solid tile.
            self.solid_color_tile_id = self.color_tile_ids.allocate();
        }
    }

    /// Process manually edges and encode them into the output encoder.
    pub fn end_path(&mut self, encoder: &mut TileEncoder, tile_mask: &mut TileMask, mut tiled_output: Option<&mut IndirectionBuffer>, pattern: &mut dyn TilerPattern) {
        if self.first_row >= self.last_row {
            return;
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

            //if self.output_is_tiled && row.output_indirection_buffer.is_empty() {
            //    row.output_indirection_buffer.reserve(self.num_tiles_x as usize);
            //    for _ in 0..self.num_tiles_x {
            //        row.output_indirection_buffer.push(INVALID_TILE_ID);
            //    }
            //}
            let output_indirection = if let Some(ref mut output) = &mut tiled_output {
                output.row_mut(row.tile_y)
            } else {
                &mut []
            };

            self.process_row(
                row.tile_y,
                &mut row.edges[..],
                &mut active_edges,
                &mut tile_mask.row(row.tile_y),
                output_indirection,
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
        output_indirection_buffer: &mut [u32],
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
            output_rect,
            solid: false,
            backdrop: 0,
            pattern_data: 0,
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
                self.finish_tile(&mut tile, active_edges, coarse_mask, pattern, output_indirection_buffer, encoder);
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
            self.finish_tile(&mut tile, active_edges, coarse_mask, pattern, output_indirection_buffer, encoder);
        }

        encoder.end_row();
    }

    fn finish_tile(
        &self,
        tile: &mut TileInfo,
        active_edges: &mut Vec<ActiveEdge>,
        coarse_mask: &mut TileMaskRow,
        pattern: &mut dyn TilerPattern,
        output_indirection_buffer: &mut[u32],
        encoder: &mut TileEncoder,
    ) {
        tile.solid = false;
        let mut empty = false;
        if active_edges.is_empty() {
            let is_in = self.draw.fill_rule.is_in(tile.backdrop);
            if is_in {
                tile.solid = true;
            } else {
                // Empty tile.
                empty = true;
            }
        }

        let (pattern_data, pattern_is_opaque) = match pattern.request_tile(tile.x, tile.y) {
            Some((d, o)) => (d, o),
            None => {
                empty = true;
                (0, false)
            }
        };

        tile.pattern_data = pattern_data;

        // Decide where to draw the tile.
        // Two configurations: Either we render in a plain target in which case the position
        // corresponds to the actual coordinates of the tile's content, or we are rendering
        // to a tiled intermediate target in which case the destination is linearly allocated.
        // The indirection buffer is used to determine whether a location is already allocated
        // for this position.
        if self.output_is_tiled && !empty {
            let mut color_tile_id = output_indirection_buffer[tile.x as usize];
            if color_tile_id == INVALID_TILE_ID {
                if tile.solid && pattern_is_opaque {
                    // Since we are processing content front-to-back, all solid opaque tiles for
                    // a given path are the same, we easily can de-duplicate it.
                    // TODO: we are still drawing multiple times over the same opaque tile.
                    color_tile_id = self.solid_color_tile_id;
                } else {
                    // allocate a spot in.
                    color_tile_id = self.color_tile_ids.allocate();
                }
                output_indirection_buffer[tile.x as usize] = color_tile_id;
            }

            if tile.solid && self.draw.is_opaque {
                output_indirection_buffer[tile.x as usize] |= TILE_OPACITY_BIT
            }

            let tx = (color_tile_id % self.color_tiles_per_row) as f32 * self.draw.tile_size.width;
            let ty = (color_tile_id / self.color_tiles_per_row) as f32 * self.draw.tile_size.height;
            tile.output_rect.min.x = tx;
            tile.output_rect.min.y = ty;
            tile.output_rect.max.x = tx + self.draw.tile_size.width;
            tile.output_rect.max.y = ty + self.draw.tile_size.height;
        } else {
            tile.output_rect.min.x = tile.x as f32 * self.draw.tile_size.width;
            tile.output_rect.max.x = (tile.x as f32 + 1.0) * self.draw.tile_size.width;
        }

        if !empty && coarse_mask.test(tile.x, tile.solid && self.draw.is_opaque) {
            encoder.encode_tile(tile, &self.draw, active_edges);
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
    /// Where to render the tile in pixels.
    pub output_rect: Box2D<f32>,

    pub pattern_data: u32,

    /// True if the tile is entirely covered by the current path. 
    pub solid: bool,

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

    // Because of precision issues when comuting the split range and when
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

pub struct IndirectionBuffer {
    data: Vec<u32>,
    size: Size2D<u32>,
}

impl IndirectionBuffer {
    pub fn new(size: Size2D<u32>) -> Self {
        IndirectionBuffer {
            data: vec![INVALID_TILE_ID; size.area() as usize],
            size
        }
    }

    pub fn row(&self, row: u32) -> &[u32] {
        debug_assert!(row < self.size.height);
        let start = (row * self.size.width) as usize;
        let end = start + self.size.width as usize;

        &self.data[start..end]
    }

    pub fn row_mut(&mut self, row: u32) -> &mut[u32] {
        debug_assert!(row < self.size.height);
        let start = (row * self.size.width) as usize;
        let end = start + self.size.width as usize;

        &mut self.data[start..end]
    }

    pub fn reset(&mut self) {
        self.data.fill(INVALID_TILE_ID);
    }

    pub fn get(&self, x: u32, y: u32) -> Option<(u32, bool)> {
        let idx = (y * self.size.width + x) as usize;
        let data = self.data[idx];
        if data == INVALID_TILE_ID {
            return None;
        }

        Some((data & !TILE_OPACITY_BIT, (data & TILE_OPACITY_BIT) != 0))
    }
}

pub trait TilerPattern {
    fn pattern_kind(&self) -> u32;
    fn request_tile(&mut self, x: u32, y: u32) -> Option<(u32, bool)>;
}

pub struct SolidColorPattern {
    color: u32,
    is_opaque: bool,
}

impl SolidColorPattern {
    pub fn new(color: Color) -> Self {
        SolidColorPattern { color: color.to_u32(), is_opaque: color.is_opaque() }
    }

    pub fn set_color(&mut self, color: Color) {
        self.color = color.to_u32();
        self.is_opaque = color.is_opaque();
    }
}

impl TilerPattern for SolidColorPattern {
    fn pattern_kind(&self) -> u32 { 0 }
    fn request_tile(&mut self, _: u32, _: u32) -> Option<(u32, bool)> {
        Some((self.color, self.is_opaque))
    }
}

pub struct TiledSourcePattern {
    indirection_buffer: IndirectionBuffer
}

impl TilerPattern for TiledSourcePattern {
    fn pattern_kind(&self) -> u32 { 1 }
    fn request_tile(&mut self, x: u32, y: u32) -> Option<(u32, bool)> {
        self.indirection_buffer.get(x, y)
    }
}

impl TilerPattern for () {
    fn pattern_kind(&self) -> u32 { std::u32::MAX }
    fn request_tile(&mut self, _: u32, _: u32) -> Option<(u32, bool)> {
        None
    }
}

#[derive(Clone)]
pub struct TileIdAllcator {
    next_id: Arc<AtomicU32>,
}

impl TileIdAllcator {
    pub fn new() -> Self {
        TileIdAllcator { next_id: Arc::new(AtomicU32::new(0)) }
    }

    pub fn allocate(&self) -> u32 {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    pub fn allocate_n(&self, n: u32) -> u32 {
        self.next_id.fetch_add(n, Ordering::Relaxed)
    }

    pub fn reset(&self) {
        self.next_id.store(0, Ordering::Relaxed);
    }

    pub fn current(&self) -> u32 {
        self.next_id.load(Ordering::Acquire)
    }
}
