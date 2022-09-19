use std::ops::Range;

use ordered_float::OrderedFloat;
pub use lyon::path::math::{Point, point, Vector, vector};
pub use lyon::path::{PathEvent, FillRule};
pub use lyon::geom::euclid::default::{Box2D, Size2D, Transform2D};
pub use lyon::geom::euclid;
pub use lyon::geom;
use lyon::{geom::{LineSegment, QuadraticBezierSegment, CubicBezierSegment}, path::Winding};

use crate::tiling::*;

use crate::tiling::cpu_rasterizer::*;
use crate::tiling::tile_renderer::{TileRenderer, TileInstance, Mask as GpuMask, CircleMask, BumpAllocatedBuffer};
use crate::gpu::mask_uploader::MaskUploader;

use copyless::VecHelper;

struct Row {
    edges: Vec<RowEdge>,
    tile_y: u32,
}

pub struct DrawParams {
    pub tolerance: f32,
    pub tile_size: Size2D<f32>,
    pub fill_rule: FillRule,
    pub max_edges_per_gpu_tile: usize,
    pub use_quads: bool,
    pub merge_solid_tiles: bool,
    pub encoded_fill_rule: u16,
}

/// A context object that can bin path edges into tile grids.
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

    pub edges: EdgeBuffer,

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
                tolerance: config.tolerance,
                tile_size: config.tile_size,
                fill_rule: FillRule::NonZero,
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

            edges: EdgeBuffer {
                line_edges: Vec::with_capacity(8192),
                quad_edges: Vec::with_capacity(0),
            },

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
        self.edges.clear();
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

        let mut edge_buffer = std::mem::take(&mut self.edges);

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
                &mut edge_buffer,
            );
        }

        self.edges = edge_buffer;
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
                if src_tile_id == TilePosition::INVALID {
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
        edge_buffer: &mut EdgeBuffer,
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
                self.finish_tile(&mut tile, active_edges, coarse_mask, pattern, &mut tiled_output, encoder, edge_buffer);
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
            self.finish_tile(&mut tile, active_edges, coarse_mask, pattern, &mut tiled_output, encoder, edge_buffer);
        }

        encoder.end_row();
    }

    fn set_indirect_output_rect(&self, tile_x: u32, opaque: bool, output_indirection_buffer: &mut[TilePosition], tile_alloc: &mut TileAllocator) -> TilePosition {
        let mut color_tile_id = output_indirection_buffer[tile_x as usize];
        if color_tile_id == TilePosition::INVALID {
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
        edge_buffer: &mut EdgeBuffer,
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
                encoder.add_fill_mask(tile, &self.draw, active_edges, edge_buffer)
            };

            encoder.add_tile(pattern, opaque, tile_position, mask_tile);
        }

        //if empty && self.draw.is_clip_in {
        //    // For clip-in it's the empty tiles that completely mask content out.
        //    coarse_mask.write_clip(tile.x);
        //}

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

    pub fn fill_canvas(
        &mut self,
        pattern: &mut dyn TilerPattern,
        tile_mask: &mut TileMask,
        mut tiled_output: Option<&mut TiledOutput>,
        encoder: &mut TileEncoder,
    ) {
        encoder.begin_path(pattern);

        let (column_start, column_end) = self.affected_range(
            self.scissor.min.x,
            self.scissor.max.x,
            self.scissor.min.x,
            self.scissor.max.x,
            self.draw.tile_size.width,
        );

        for tile_y in 0..self.rows.len() as u32 {
            let mut tile_mask = tile_mask.row(tile_y);
            let mut dummy_tile_alloc = TileAllocator::new(0, 0);
            let dummy_buffer: &mut [TilePosition] = &mut [];
            let (output_indirection_buffer, output_tile_alloc) = if let Some(ref mut output) = &mut tiled_output {
                (output.indirection_buffer.row_mut(tile_y), &mut output.tile_allocator)
            } else {
                (dummy_buffer, &mut dummy_tile_alloc)
            };

            for tile_x in column_start .. column_end {
                let tile_x = tile_x as u32;

                pattern.set_tile(tile_x, tile_y);
                let opaque = pattern.tile_is_opaque();

                if pattern.tile_is_empty() {
                    continue;
                }

                if !tile_mask.test(tile_x as u32, opaque) {
                    continue;
                }

                let tile_position = if self.output_is_tiled {
                    self.set_indirect_output_rect(tile_x, opaque, output_indirection_buffer, output_tile_alloc)
                } else {
                    TilePosition::new(tile_x, tile_y)
                };

                encoder.add_tile(pattern, opaque, tile_position, TilePosition::ZERO);
            }
        }
    }

    pub fn update_stats(&mut self, stats: &mut Stats) {
        self.edges.update_stats(stats);
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


#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct QuadEdge(pub Point, pub Point, pub Point, u32, u32);

unsafe impl bytemuck::Pod for QuadEdge {}
unsafe impl bytemuck::Zeroable for QuadEdge {}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct LineEdge(pub Point, pub Point);

unsafe impl bytemuck::Pod for LineEdge {}
unsafe impl bytemuck::Zeroable for LineEdge {}

// If we can't fit all masks into the atlas, we have to break the work into
// multiple passes. Each pass builds the atlas and renders it into the color target.
#[derive(Debug)]
pub struct RenderPass {
    pub opaque_image_tiles: Range<u32>,
    pub batches: Range<usize>,
    pub mask_atlas_index: AtlasIndex,
    pub color_atlas_index: AtlasIndex,
}

pub struct AlphaBatch {
    pub tiles: Range<u32>,
    pub batch_kind: u32,
}


#[derive(Default, Debug)]
pub struct BufferRanges {
    pub opaque_image_tiles: BufferRange,
    pub alpha_tiles: BufferRange,
    pub masks: BufferRange,
    pub circles: BufferRange,
}

impl BufferRanges {
    pub fn reset(&mut self) {
        self.opaque_image_tiles = BufferRange(0, 0);
        self.opaque_image_tiles = BufferRange(0, 0);
        self.alpha_tiles = BufferRange(0, 0);
        self.masks = BufferRange(0, 0);
        self.circles = BufferRange(0, 0);
    }
}

pub struct SourceTiles {
    pub mask_tiles: TileAllocator,
    pub color_tiles: TileAllocator,
}

pub struct EdgeBuffer {
    pub line_edges: Vec<LineEdge>,
    pub quad_edges: Vec<QuadEdge>,
}

impl Default for EdgeBuffer {
    fn default() -> Self {
        EdgeBuffer { line_edges: Vec::new(), quad_edges: Vec::new() }
    }
}

impl EdgeBuffer {
    pub fn upload(&mut self, tile_renderer: &mut TileRenderer, queue: &wgpu::Queue) {
        let edges = if !self.line_edges.is_empty() {
            bytemuck::cast_slice(&self.line_edges)
        } else {
            bytemuck::cast_slice(&self.quad_edges)
        };
        queue.write_buffer(
            &tile_renderer.edges_ssbo.buffer,
            0,
            edges
        );
    }

    pub fn update_stats(&self, stats: &mut Stats) {
        stats.edges += self.line_edges.len() + self.quad_edges.len();
    }

    pub fn allocate_buffer_ranges(&mut self, tile_renderer: &mut TileRenderer) {
        tile_renderer.edges_ssbo.allocator.push(self.line_edges.len());
    }

    pub fn clear(&mut self) {
        self.line_edges.clear();
        self.quad_edges.clear();
    }
}

pub struct TileEncoder {
    // State and output associated with the current group/layer:

    // These should move into dedicated things.
    pub fill_masks: GpuMaskEncoder,
    pub circle_masks: CircleMaskEncoder,

    pub render_passes: Vec<RenderPass>,
    pub batches: Vec<AlphaBatch>,
    pub alpha_tiles: Vec<TileInstance>,
    pub opaque_image_tiles: Vec<TileInstance>,

    current_pattern_kind: Option<u32>,
    // First mask index of the current mask pass.
    batches_start: usize,
    opaque_image_tiles_start: u32,
    //cpu_masks_start: u32,
    // First masked color tile of the current mask pass.
    alpha_tiles_start: u32,
    // Index of the current mask texture (increments every time we run out of space in the
    // mask atlas, which is the primary reason for starting a new mask pass).
    masks_texture_index: AtlasIndex,
    color_texture_index: AtlasIndex,

    reversed: bool,

    /// index of the last opaque solid tile in the current path. Used to detect
    /// consecutive solid tiles that can be merged.
    current_mergeable_tile: u32,

    pub ranges: BufferRanges,

    pub prerender_pattern: bool,

    pub src: SourceTiles,

    pub mask_uploader: MaskUploader,

    pub edge_distributions: [u32; 16],
}

impl TileEncoder {
    pub fn new(config: &TilerConfig, mask_uploader: MaskUploader) -> Self {
        let atlas_tiles_x = config.mask_atlas_size.width as u32 / config.tile_size.width as u32;
        let atlas_tiles_y = config.mask_atlas_size.height as u32 / config.tile_size.height as u32;
        TileEncoder {
            opaque_image_tiles: Vec::with_capacity(2048),
            alpha_tiles: Vec::with_capacity(8192),
            fill_masks: GpuMaskEncoder::new(),
            circle_masks: CircleMaskEncoder::new(),
            render_passes: Vec::with_capacity(16),
            batches: Vec::with_capacity(64),
            current_pattern_kind: None,
            current_mergeable_tile: 0, // Will be set in begin_row
            batches_start: 0,
            opaque_image_tiles_start: 0,
            //cpu_masks_start: 0,
            alpha_tiles_start: 0,
            masks_texture_index: 0,
            color_texture_index: 0,

            mask_uploader,

            reversed: false,

            ranges: BufferRanges::default(),

            edge_distributions: [0; 16],

            src: SourceTiles {
                mask_tiles: TileAllocator::new(atlas_tiles_x, atlas_tiles_y),
                color_tiles: TileAllocator::new(atlas_tiles_x, atlas_tiles_y),
            },
            prerender_pattern: true,
        }
    }

    pub fn create_similar(&self) -> Self {
        TileEncoder {
            opaque_image_tiles: Vec::with_capacity(2000),
            alpha_tiles: Vec::with_capacity(6000),
            fill_masks: GpuMaskEncoder::new(),
            circle_masks: CircleMaskEncoder::new(),
            render_passes: Vec::with_capacity(16),
            batches: Vec::with_capacity(64),
            current_pattern_kind: None,
            current_mergeable_tile: 0,
            batches_start: 0,
            opaque_image_tiles_start: 0,
            //cpu_masks_start: 0,
            alpha_tiles_start: 0,
            masks_texture_index: 0,
            color_texture_index: 0,

            mask_uploader: self.mask_uploader.create_similar(),

            reversed: false,

            ranges: BufferRanges::default(),

            edge_distributions: [0; 16],

            src: SourceTiles {
                mask_tiles: TileAllocator::new(self.src.mask_tiles.width(), self.src.mask_tiles.height()),
                color_tiles: TileAllocator::new(self.src.color_tiles.width(), self.src.color_tiles.height()),
            },
            prerender_pattern: true,
        }
    }

    pub fn reset(&mut self) {
        self.opaque_image_tiles.clear();
        self.alpha_tiles.clear();
        self.fill_masks.reset();
        self.circle_masks.reset();
        self.render_passes.clear();
        self.batches.clear();
        self.mask_uploader.reset();
        self.current_pattern_kind = None;
        self.edge_distributions = [0; 16];
        self.batches_start = 0;
        self.opaque_image_tiles_start = 0;
        //self.cpu_masks_start = 0;
        self.alpha_tiles_start = 0;
        self.masks_texture_index = 0;
        self.color_texture_index = 0;
        self.reversed = false;
        self.ranges.reset();
        self.src.mask_tiles.reset();
        self.src.color_tiles.reset();
    }

    pub fn end_paths(&mut self) {
        self.fill_masks.end_render_pass();
        self.circle_masks.end_render_pass();
        self.flush_render_pass();
    }

    pub fn num_cpu_masks(&self) -> usize {
        self.mask_uploader.copy_instances().len()
    }

    pub fn num_mask_atlases(&self) -> u32 {
        //let id = self.mask_ids.current();
        //id / self.masks_per_atlas + if id % self.masks_per_atlas != 0 { 1 } else { 0 }
        self.src.mask_tiles.current_atlas + 1
    }

    pub fn begin_row(&mut self) {
        self.current_mergeable_tile = std::u32::MAX - 1;
    }

    pub fn begin_path(&mut self, pattern: &mut dyn TilerPattern) {
        let pattern_kind = if self.prerender_pattern {
            TILED_IMAGE_PATTERN
        } else{
            pattern.pattern_kind()
        };

        if self.current_pattern_kind != Some(pattern_kind) {
            self.end_batch();
            self.current_pattern_kind = Some(pattern_kind);
        }
    }

    pub fn end_row(&mut self) {}

    pub fn add_tile(
        &mut self,
        pattern: &mut dyn TilerPattern,
        opaque: bool,
        tile_position: TilePosition,
        mask: TilePosition,
    ) {
        // It is always more efficient to render opaque tiles directly.
        let prerender = self.prerender_pattern && !opaque;
        let mergeable = opaque;

        if !prerender && mergeable && self.current_mergeable_tile + 1 == tile_position.x() {
            let tile = pattern.opaque_tiles().last_mut().unwrap();
            tile.position.extend();
            if !prerender {
                tile.pattern_position.extend();
            }
            self.current_mergeable_tile += 1;
            return;
        }

        if mergeable {
            self.current_mergeable_tile = tile_position.x();
        }

        let (pattern_position, pattern_data) = if prerender {
            let (atlas_position, atlas_index) = self.src.color_tiles.allocate();
            pattern.prerender_tile(atlas_position, atlas_index);
            (atlas_position, 0)
        } else {
            (tile_position, pattern.tile_data())
        };

        {
            // Add the tile that will be rendered into the main pass.
            let tiles = if prerender && opaque {
                &mut self.opaque_image_tiles
            } else if opaque {
                pattern.opaque_tiles()
            } else {
                &mut self.alpha_tiles
            };

            tiles.alloc().init(TileInstance {
                position: tile_position,
                mask,
                pattern_position,
                pattern_data,
            });
        }
    }

    fn maybe_flush_render_pass(&mut self) {
        if self.masks_texture_index != self.src.mask_tiles.current_atlas
        || self.color_texture_index != self.src.color_tiles.current_atlas {
            self.flush_render_pass();
        }
    }

    // TODO: first allocate the mask and color tiles and then flush if either is
    // in a new texture (right now we only flush the mask atlas correctly).
    fn flush_render_pass(&mut self) {
        self.end_batch();
        let batches_end = self.batches.len();
        let opaque_image_tiles_end = self.opaque_image_tiles.len() as u32;
        if batches_end != self.batches_start {
            self.render_passes.alloc().init(RenderPass {
                opaque_image_tiles: self.opaque_image_tiles_start..opaque_image_tiles_end,
                batches: self.batches_start..batches_end,
                mask_atlas_index: self.masks_texture_index,
                color_atlas_index: self.src.color_tiles.current_atlas,
            });
            self.batches_start = batches_end;
            self.opaque_image_tiles_start = opaque_image_tiles_end;
        }

        if self.src.color_tiles.is_nearly_full() {
            self.src.color_tiles.finish_atlas();
        }
        if self.src.mask_tiles.is_nearly_full() {
            self.src.mask_tiles.finish_atlas();
        }

        self.masks_texture_index = self.src.mask_tiles.current_atlas;
        self.color_texture_index = self.src.color_tiles.current_atlas;
    }

    pub fn end_batch(&mut self) {
        let batch_kind = if let Some(kind) = self.current_pattern_kind {
            kind
        } else {
            return;
        };

        let alpha_tiles_end = self.alpha_tiles.len() as u32;
        if self.alpha_tiles_start == alpha_tiles_end {
            return;
        }
        self.batches.alloc().init(AlphaBatch {
            tiles: self.alpha_tiles_start..alpha_tiles_end,
            batch_kind,
        });
        self.alpha_tiles_start = alpha_tiles_end;
    }

    pub fn add_fill_mask(&mut self, tile: &TileInfo, draw: &DrawParams, active_edges: &[ActiveEdge], edges: &mut EdgeBuffer) -> TilePosition {
        let mask = if draw.use_quads {
            self.add_quad_gpu_mask(tile, draw, active_edges, edges)
        } else {
            self.add_line_gpu_mask(tile, draw, active_edges, edges)
        };

        mask.unwrap_or_else(|| self.add_cpu_mask(tile, draw, active_edges))
    }

    pub fn add_cricle_mask(&mut self, center: Point, radius: f32) -> TilePosition {
        let (tile, atlas_index) = self.src.mask_tiles.allocate();
        self.maybe_flush_render_pass();

        self.circle_masks.prerender_mask(atlas_index, CircleMask { tile, radius, center: center.to_array() });

        tile
    }

    pub fn add_line_gpu_mask(&mut self, tile: &TileInfo, draw: &DrawParams, active_edges: &[ActiveEdge], edges: &mut EdgeBuffer) -> Option<TilePosition> {
        //println!(" * tile ({:?}, {:?}), backdrop: {}, {:?}", tile.x, tile.y, tile.backdrop, active_edges);

        let edges_start = edges.line_edges.len();

        let offset = vector(tile.inner_rect.min.x, tile.inner_rect.min.y);

        // If the number of edges is larger than a certain threshold, we'll fall back to
        // rasterizing them on the CPU.
        if (edges.line_edges.len() - edges_start) + active_edges.len() > draw.max_edges_per_gpu_tile {
            edges.line_edges.resize(edges_start, LineEdge(point(0.0, 0.0), point(0.0, 0.0)));
            return None;
        }

        for edge in active_edges {
            // Handle auxiliary edges for edges that cross the tile's left side.
            if edge.from.x < tile.outer_rect.min.x && edge.from.y != tile.outer_rect.min.y {
                edges.line_edges.alloc().init(LineEdge(
                    point(-10.0, tile.outer_rect.max.y),
                    point(-10.0, edge.from.y),
                ));
            }

            if edge.to.x < tile.outer_rect.min.x && edge.to.y != tile.outer_rect.min.y {
                edges.line_edges.alloc().init(LineEdge(
                    point(-10.0, edge.to.y),
                    point(-10.0, tile.outer_rect.max.y),
                ));
            }

            if edge.is_line() {
                edges.line_edges.alloc().init(LineEdge(edge.from - offset, edge.to - offset));
            } else {
                let curve = QuadraticBezierSegment { from: edge.from - offset, ctrl: edge.ctrl - offset, to: edge.to - offset };
                flatten_quad(&curve, draw.tolerance, &mut |segment| {
                    edges.line_edges.alloc().init(LineEdge(segment.from, segment.to));
                });
            }
        }

        let edges_start = edges_start as u32;
        let edges_end = edges.line_edges.len() as u32;
        debug_assert!(edges_end > edges_start, "{} > {} {:?}", edges_end, edges_start, active_edges);
        self.edge_distributions[(edges_end - edges_start).min(15) as usize] += 1;
        let (tile_position, atlas_index) = self.src.mask_tiles.allocate();

        self.maybe_flush_render_pass();

        debug_assert!(tile_position.to_u32() != 0);

        self.fill_masks.prerender_mask(
            atlas_index,
            GpuMask {
                edges: (edges_start, edges_end),
                tile: tile_position,
                backdrop: tile.backdrop + 8192,
                fill_rule: draw.encoded_fill_rule,
            }
        );

        Some(tile_position)
    }

    pub fn add_quad_gpu_mask(&mut self, tile: &TileInfo, draw: &DrawParams, active_edges: &[ActiveEdge], edges: &mut EdgeBuffer) -> Option<TilePosition> {
        let edges_start = edges.quad_edges.len();

        let offset = vector(tile.inner_rect.min.x, tile.inner_rect.min.y);

        // If the number of edges is larger than a certain threshold, we'll fall back to
        // rasterizing them on the CPU.
        if (edges.quad_edges.len() - edges_start) + active_edges.len() > draw.max_edges_per_gpu_tile {
            edges.quad_edges.resize(edges_start, QuadEdge(point(0.0, 0.0), point(123.0, 456.0), point(0.0, 0.0), 0, 0));
            return None;
        }

        for edge in active_edges {
            // Handle auxiliary edges for edges that cross the tile's left side.
            if edge.from.x < tile.outer_rect.min.x && edge.from.y != tile.outer_rect.min.y {
                edges.quad_edges.alloc().init(QuadEdge(
                    point(-10.0, tile.outer_rect.max.y),
                    point(123.0, 456.0),
                    point(-10.0, edge.from.y),
                    0, 0,
                ));
            }

            if edge.to.x < tile.outer_rect.min.x && edge.to.y != tile.outer_rect.min.y {
                edges.line_edges.alloc().init(LineEdge(
                    point(-10.0, edge.to.y),
                    point(-10.0, tile.outer_rect.max.y),
                ));
            }

            if edge.is_line() {
                edges.quad_edges.alloc().init(QuadEdge(edge.from - offset, point(123.0, 456.0), edge.to - offset, 0, 0));
            } else {
                let curve = QuadraticBezierSegment { from: edge.from - offset, ctrl: edge.ctrl - offset, to: edge.to - offset };
                edges.quad_edges.alloc().init(QuadEdge(curve.from, curve.ctrl, curve.to, 1, 0));
            }
        }

        let edges_start = edges_start as u32;
        let edges_end = edges.quad_edges.len() as u32;
        assert!(edges_end > edges_start, "{:?}", active_edges);

        let (tile_position, atlas_index) = self.src.mask_tiles.allocate();
        self.maybe_flush_render_pass();

        unimplemented!();
        // TODO
        //self.gpu_masks.push(GpuMask {
        //    edges: (edges_start, edges_end),
        //    tile: tile_position,
        //    backdrop: tile.backdrop + 8192,
        //    fill_rule: draw.encoded_fill_rule,
        //});

        Some(tile_position)
    }

    pub fn add_cpu_mask(&mut self, tile: &TileInfo, draw: &DrawParams, active_edges: &[ActiveEdge]) -> TilePosition {
        debug_assert!(draw.tile_size.width <= 32.0);
        debug_assert!(draw.tile_size.height <= 32.0);

        let mut accum = [0.0; 32 * 32];
        let mut backdrops = [tile.backdrop as f32; 32];

        let tile_offset = tile.inner_rect.min.to_vector();
        for edge in active_edges {
            // Handle auxiliary edges for edges that cross the tile's left side.
            if edge.from.x < tile.outer_rect.min.x && edge.from.y != tile.outer_rect.min.y {
                add_backdrop(edge.from.y, -1.0, &mut backdrops[0..draw.tile_size.height as usize]);
            }

            if edge.to.x < tile.outer_rect.min.x && edge.to.y != tile.outer_rect.min.y {
                add_backdrop(edge.to.y, 1.0, &mut backdrops[0..draw.tile_size.height as usize]);
            }

            let from = edge.from - tile_offset;
            let to = edge.to - tile_offset;

            if edge.is_line() {
                draw_line(from, to, &mut accum);
            } else {
                let ctrl = edge.ctrl - tile_offset;
                draw_curve(from, ctrl, to, draw.tolerance, &mut accum);
            }
        }

        let (tile_position, _atlas_index) = self.src.mask_tiles.allocate();
        self.maybe_flush_render_pass();

        let mask_buffer_range = self.mask_uploader.new_mask(tile_position);
        unsafe {
            self.mask_uploader.current_mask_buffer.set_len(mask_buffer_range.end as usize);
        }

        let accumulate = match draw.fill_rule {
            FillRule::EvenOdd => accumulate_even_odd,
            FillRule::NonZero => accumulate_non_zero,
        };

        accumulate(
            &accum,
            &backdrops,
            &mut self.mask_uploader.current_mask_buffer[mask_buffer_range.clone()],
        );

        //let mask_name = format!("mask-{}.png", mask_id.to_u32());
        //crate::cpu_rasterizer::save_mask_png(16, 16, &self.mask_uploader.current_mask_buffer[mask_buffer_range.clone()], &mask_name);
        //crate::cpu_rasterizer::save_accum_png(16, 16, &accum, &backdrops, &format!("accum-{}.png", mask_id.to_u32()));

        tile_position
    }

    pub fn reverse_alpha_tiles(&mut self) {
        assert!(!self.reversed);
        self.alpha_tiles.reverse();

        self.batches.reverse();
        let num_alpha_tiles = self.alpha_tiles.len() as u32;
        for batch in &mut self.batches {
            batch.tiles = (num_alpha_tiles - batch.tiles.end) .. (num_alpha_tiles - batch.tiles.start);
        }
        let num_batches = self.batches.len();
        self.render_passes.reverse();
        for mask_pass in &mut self.render_passes {
            mask_pass.batches = (num_batches - mask_pass.batches.end) .. (num_batches - mask_pass.batches.start);
        }
    }

    pub fn allocate_buffer_ranges(&mut self, tile_renderer: &mut TileRenderer) {
        self.fill_masks.allocate_buffer_ranges(&mut tile_renderer.fill_masks);
        self.circle_masks.allocate_buffer_ranges(&mut tile_renderer.circle_masks);
        self.ranges.opaque_image_tiles = tile_renderer.tiles_vbo.allocator.push(self.opaque_image_tiles.len());
        self.ranges.alpha_tiles = tile_renderer.tiles_vbo.allocator.push(self.alpha_tiles.len());
    }

    pub fn upload(&mut self, tile_renderer: &mut TileRenderer, queue: &wgpu::Queue) {
        self.fill_masks.upload(&mut tile_renderer.fill_masks, queue);
        self.circle_masks.upload(&mut tile_renderer.circle_masks, queue);

        queue.write_buffer(
            &tile_renderer.tiles_vbo.buffer,
            self.ranges.opaque_image_tiles.byte_offset::<TileInstance>(),
            bytemuck::cast_slice(&self.opaque_image_tiles),
        );

        queue.write_buffer(
            &tile_renderer.tiles_vbo.buffer,
            self.ranges.alpha_tiles.byte_offset::<TileInstance>(),
            bytemuck::cast_slice(&self.alpha_tiles),
        );
    }

    pub fn update_stats(&self, stats: &mut Stats) {
        stats.opaque_tiles += self.opaque_image_tiles.len();
        stats.alpha_tiles += self.alpha_tiles.len();
        stats.gpu_mask_tiles += self.fill_masks.masks.len() + self.circle_masks.masks.len();
        stats.cpu_mask_tiles += self.num_cpu_masks();
        stats.render_passes += self.render_passes.len();
        stats.batches += self.batches.len();
    }
}

pub fn flatten_quad(curve: &QuadraticBezierSegment<f32>, tolerance: f32, cb: &mut impl FnMut(&LineSegment<f32>)) {
    let sq_error = square_distance_to_point(&curve.baseline().to_line(), curve.ctrl) * 0.25;

    let sq_tolerance = tolerance * tolerance;
    if sq_error <= sq_tolerance {
        cb(&curve.baseline());
    } else if sq_error <= sq_tolerance * 4.0 {
        let ft = curve.from.lerp(curve.to, 0.5);
        let mid = ft.lerp(curve.ctrl, 0.5);
        cb(&LineSegment { from: curve.from, to: mid });
        cb(&LineSegment { from: mid, to: curve.to });
    } else {

        // The baseline cost of this is fairly high and then amortized if the number of segments is high.
        // In practice the number of edges tends to be low in our case due to splitting into small tiles.
        curve.for_each_flattened(tolerance, cb);
        //unsafe { crate::flatten_simd::flatten_quad_sse(curve, tolerance, cb); }
        //crate::flatten_simd::flatten_quad_ref(curve, tolerance, cb);

        // This one is comes from font-rs. It's less work than for_each_flattened but generates
        // more edges, the overall performance is about the same from a quick measurement.
        // Maybe try using a lookup table?
        //let ddx = curve.from.x - 2.0 * curve.ctrl.x + curve.to.x;
        //let ddy = curve.from.y - 2.0 * curve.ctrl.y + curve.to.y;
        //let square_dev = ddx * ddx + ddy * ddy;
        //let n = 1 + (3.0 * square_dev).sqrt().sqrt().floor() as u32;
        //let inv_n = (n as f32).recip();
        //let mut t = 0.0;
        //for _ in 0..(n - 1) {
        //    t += inv_n;
        //    cb(curve.sample(t));
        //}
        //cb(curve.to);
    }
}

use lyon::geom::Line;

#[inline]
fn square_distance_to_point(line: &Line<f32>, p: Point) -> f32 {
    let v = p - line.point;
    let c = line.vector.cross(v);
    (c * c) / line.vector.square_length()
}

pub struct MaskRenderer {
    masks: BumpAllocatedBuffer,
    render_passes: Vec<Range<u32>>,
}

impl MaskRenderer {
    pub fn new<T>(device: &wgpu::Device, label: &'static str, default_size: u32) -> Self {
        MaskRenderer {
            masks: BumpAllocatedBuffer::new::<T>(
                device,
                label,
                default_size,
                wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST
            ),
            render_passes: Vec::with_capacity(16),
        }
    }

    pub fn begin_frame(&mut self) {
        self.masks.begin_frame();
    }

    pub fn ensure_allocated(&mut self, device: &wgpu::Device) {
        self.masks.ensure_allocated(device);
    }

    pub fn has_content(&self, atlas_index: AtlasIndex) -> bool {
        let idx = atlas_index as usize;
        self.render_passes.len() > idx && !self.render_passes[idx].is_empty()
    }

    pub fn render<'a, 'b: 'a>(&'b self, atlas_index: AtlasIndex, pass: &mut wgpu::RenderPass<'a>) {
        let range = self.render_passes[atlas_index as usize].clone();
        pass.set_vertex_buffer(0, self.masks.buffer.slice(..));
        pass.draw_indexed(0..6, 0, range);
    }
}

pub struct MaskEncoder<T> {
    pub masks: Vec<T>,
    masks_start: u32,
    render_passes: Vec<Range<u32>>,
    current_atlas: u32,
    buffer_range: BufferRange,
}

impl<T> MaskEncoder<T> {
    fn new() -> Self {
        MaskEncoder {
            masks: Vec::with_capacity(8192),
            render_passes: Vec::with_capacity(16),
            masks_start: 0,
            current_atlas: 0,
            buffer_range: BufferRange(0, 0),
        }
    }

    fn reset(&mut self) {
        self.masks.clear();
        self.render_passes.clear();
        self.masks_start = 0;
        self.current_atlas = 0;
    }

    fn end_render_pass(&mut self) {
        let masks_end = self.masks.len() as u32;
        if self.masks_start == masks_end {
            return;
        }

        if self.render_passes.len() <= self.current_atlas as usize {
            self.render_passes.resize(self.current_atlas as usize + 1, 0..0);
        }
        self.render_passes[self.current_atlas as usize] = self.masks_start..masks_end;
        self.masks_start = masks_end;
        self.current_atlas += 1;
    }

    fn prerender_mask(&mut self, atlas_index: AtlasIndex, mask: T) {
        if atlas_index != self.current_atlas {
            self.end_render_pass();
            self.current_atlas = atlas_index;
        }

        self.masks.push(mask);
    }

    pub fn allocate_buffer_ranges(&mut self, renderer: &mut MaskRenderer) {
        self.buffer_range = renderer.masks.allocator.push(self.masks.len());
    }

    pub fn upload(&mut self, renderer: &mut MaskRenderer, queue: &wgpu::Queue) where T: bytemuck::Pod {
        queue.write_buffer(
            &renderer.masks.buffer,
            self.buffer_range.byte_offset::<GpuMask>(),
            bytemuck::cast_slice(&self.masks),
        );
        std::mem::swap(&mut self.render_passes, &mut renderer.render_passes);
        self.render_passes.clear();
    }
}

pub type GpuMaskEncoder = MaskEncoder<GpuMask>;
pub type CircleMaskEncoder = MaskEncoder<CircleMask>;
