use lyon::geom::{LineSegment, QuadraticBezierSegment};
use lyon::path::math::{Point, point, vector};
use lyon::path::FillRule;
use std::ops::Range;

use crate::tiler::*;
use crate::cpu_rasterizer::*;
use crate::tile_renderer::{TileInstance, Mask as GpuMask, CircleMask, BumpAllocatedBuffer};
use crate::gpu::mask_uploader::MaskUploader;

use copyless::VecHelper;

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
    pub opaque_image_tiles: Range<u32>, // TODO
    pub batches: Range<usize>,
    //pub circle_masks: Range<u32>,
    pub mask_atlas_index: AtlasIndex,
    pub color_atlas_index: AtlasIndex,
}

pub struct AlphaBatch {
    pub tiles: Range<u32>,
    pub batch_kind: u32,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct BufferRange(pub u32, pub u32);
impl BufferRange {
    pub fn is_empty(&self) -> bool { self.0 >= self.1 }
    pub fn to_u32(&self) -> Range<u32> { self.0 .. self.1 }
    pub fn byte_range<Ty>(&self) -> Range<u64> {
        let s = std::mem::size_of::<Ty>() as u64;
        self.0 as u64 * s .. self.1 as u64 * s
    }
    pub fn byte_offset<Ty>(&self) -> u64 {
        self.0 as u64 * std::mem::size_of::<Ty>() as u64
    }
}

#[derive(Default, Debug)]
pub struct BufferRanges {
    pub opaque_image_tiles: BufferRange,
    pub alpha_tiles: BufferRange,
    pub masks: BufferRange,
    pub circles: BufferRange,
    pub edges: BufferRange,
}

impl BufferRanges {
    pub fn reset(&mut self) {
        self.opaque_image_tiles = BufferRange(0, 0);
        self.opaque_image_tiles = BufferRange(0, 0);
        self.alpha_tiles = BufferRange(0, 0);
        self.masks = BufferRange(0, 0);
        self.circles = BufferRange(0, 0);
        self.edges = BufferRange(0, 0);
    }
}

pub type TileId = u32;
pub type AtlasIndex = u32;

pub struct TileAllocator {
    pub next_id: TileId,
    pub current_atlas: AtlasIndex,
    pub tiles_per_row: u32,
    pub tiles_per_atlas: u32,
}

impl TileAllocator {
    pub fn new(w: u32, h: u32) -> Self {
        TileAllocator {
            next_id: 1,
            current_atlas: 0,
            tiles_per_row: w,
            tiles_per_atlas: w * h,
        }
    }

    pub fn reset(&mut self) {
        self.next_id = 1;
        self.current_atlas = 0;
    }

    pub fn allocate(&mut self) -> (TilePosition, AtlasIndex) {
        let id = self.next_id;
        self.next_id += 1;

        let mut id2 = id % self.tiles_per_atlas;

        if id2 == id {
            // Common path.
            let pos = TilePosition::new(
                id % self.tiles_per_row,
                id / self.tiles_per_row,
            );
            return (pos, self.current_atlas)
        }

        if id2 == 0 {
            // Tile zero is reserved.
            id2 += 1;
        }

        self.next_id = id2 + 1;

        self.current_atlas += 1;
        let pos = TilePosition::new(
            id2 % self.tiles_per_row,
            id2 / self.tiles_per_row,
        );

        (pos, self.current_atlas)
    }

    pub fn finish_atlas(&mut self) {
        self.current_atlas += 1;
        self.next_id = 1;
    }

    pub fn width(&self) -> u32 { self.tiles_per_row }

    pub fn height(&self) -> u32 { self.tiles_per_atlas / self.tiles_per_row }

    pub fn current_atlas(&self) -> u32 { self.current_atlas }

    pub fn is_nearly_full(&self) -> bool {
        (self.next_id * 100) / self.tiles_per_atlas > 70
    }
}

pub struct SourceTiles {
    pub mask_tiles: TileAllocator,
    pub color_tiles: TileAllocator,
}

pub struct TileEncoder {
    // State and output associated with the current group/layer:

    // These should move into dedicated things.
    pub quad_edges: Vec<QuadEdge>,
    pub line_edges: Vec<LineEdge>,
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

    // TODO, pass via parameter instead of owning it.
    pub src: SourceTiles,

    pub mask_uploader: MaskUploader,

    pub edge_distributions: [u32; 16],
}

impl TileEncoder {
    pub fn new(config: &TilerConfig, mask_uploader: MaskUploader) -> Self {
        let atlas_tiles_x = config.mask_atlas_size.width as u32 / config.tile_size.width as u32;
        let atlas_tiles_y = config.mask_atlas_size.height as u32 / config.tile_size.height as u32;
        TileEncoder {
            quad_edges: Vec::with_capacity(0),
            line_edges: Vec::with_capacity(8196),
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
            quad_edges: Vec::with_capacity(8196),
            line_edges: Vec::with_capacity(8196),
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
        self.quad_edges.clear();
        self.line_edges.clear();
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

    pub fn add_fill_mask(&mut self, tile: &TileInfo, draw: &DrawParams, active_edges: &[ActiveEdge]) -> TilePosition {
        let mask = if draw.use_quads {
            self.add_quad_gpu_mask(tile, draw, active_edges)
        } else {
            self.add_line_gpu_mask(tile, draw, active_edges)
        };

        mask.unwrap_or_else(|| self.add_cpu_mask(tile, draw, active_edges))
    }

    pub fn add_cricle_mask(&mut self, center: Point, radius: f32) -> TilePosition {
        let (tile, atlas_index) = self.src.mask_tiles.allocate();
        self.maybe_flush_render_pass();

        self.circle_masks.prerender_mask(atlas_index, CircleMask { tile, radius, center: center.to_array() });

        tile
    }

    pub fn add_line_gpu_mask(&mut self, tile: &TileInfo, draw: &DrawParams, active_edges: &[ActiveEdge]) -> Option<TilePosition> {
        //println!(" * tile ({:?}, {:?}), backdrop: {}, {:?}", tile.x, tile.y, tile.backdrop, active_edges);

        let edges_start = self.line_edges.len();

        let offset = vector(tile.inner_rect.min.x, tile.inner_rect.min.y);

        // If the number of edges is larger than a certain threshold, we'll fall back to
        // rasterizing them on the CPU.
        if (self.line_edges.len() - edges_start) + active_edges.len() > draw.max_edges_per_gpu_tile {
            self.line_edges.resize(edges_start, LineEdge(point(0.0, 0.0), point(0.0, 0.0)));
            return None;
        }

        for edge in active_edges {
            // Handle auxiliary edges for edges that cross the tile's left side.
            if edge.from.x < tile.outer_rect.min.x && edge.from.y != tile.outer_rect.min.y {
                self.line_edges.alloc().init(LineEdge(
                    point(-10.0, tile.outer_rect.max.y),
                    point(-10.0, edge.from.y),
                ));
            }

            if edge.to.x < tile.outer_rect.min.x && edge.to.y != tile.outer_rect.min.y {
                self.line_edges.alloc().init(LineEdge(
                    point(-10.0, edge.to.y),
                    point(-10.0, tile.outer_rect.max.y),
                ));
            }

            if edge.is_line() {
                self.line_edges.alloc().init(LineEdge(edge.from - offset, edge.to - offset));
            } else {
                let curve = QuadraticBezierSegment { from: edge.from - offset, ctrl: edge.ctrl - offset, to: edge.to - offset };
                flatten_quad(&curve, draw.tolerance, &mut |segment| {
                    self.line_edges.alloc().init(LineEdge(segment.from, segment.to));
                });
            }
        }

        let edges_start = edges_start as u32;
        let edges_end = self.line_edges.len() as u32;
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

    pub fn add_quad_gpu_mask(&mut self, tile: &TileInfo, draw: &DrawParams, active_edges: &[ActiveEdge]) -> Option<TilePosition> {
        let edges_start = self.quad_edges.len();

        let offset = vector(tile.inner_rect.min.x, tile.inner_rect.min.y);

        // If the number of edges is larger than a certain threshold, we'll fall back to
        // rasterizing them on the CPU.
        if (self.quad_edges.len() - edges_start) + active_edges.len() > draw.max_edges_per_gpu_tile {
            self.quad_edges.resize(edges_start, QuadEdge(point(0.0, 0.0), point(123.0, 456.0), point(0.0, 0.0), 0, 0));
            return None;
        }

        for edge in active_edges {
            // Handle auxiliary edges for edges that cross the tile's left side.
            if edge.from.x < tile.outer_rect.min.x && edge.from.y != tile.outer_rect.min.y {
                self.line_edges.alloc().init(LineEdge(
                    point(-10.0, tile.outer_rect.max.y),
                    point(-10.0, edge.from.y),
                ));
            }

            if edge.to.x < tile.outer_rect.min.x && edge.to.y != tile.outer_rect.min.y {
                self.line_edges.alloc().init(LineEdge(
                    point(-10.0, edge.to.y),
                    point(-10.0, tile.outer_rect.max.y),
                ));
            }

            if edge.is_line() {
                let e = QuadEdge(edge.from - offset, point(123.0, 456.0), edge.to - offset, 0, 0);
                self.quad_edges.alloc().init(e);
            } else {
                let curve = QuadraticBezierSegment { from: edge.from - offset, ctrl: edge.ctrl - offset, to: edge.to - offset };
                self.quad_edges.alloc().init(QuadEdge(curve.from, curve.ctrl, curve.to, 1, 0));
            }
        }

        let edges_start = edges_start as u32;
        let edges_end = self.quad_edges.len() as u32;
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

    pub fn allocate_buffer_ranges(&mut self, tile_renderer: &mut crate::tile_renderer::TileRenderer) {
        self.fill_masks.allocate_buffer_ranges(&mut tile_renderer.fill_masks);
        self.circle_masks.allocate_buffer_ranges(&mut tile_renderer.circle_masks);
        self.ranges.opaque_image_tiles = tile_renderer.tiles_vbo.allocator.push(self.opaque_image_tiles.len());
        self.ranges.alpha_tiles = tile_renderer.tiles_vbo.allocator.push(self.alpha_tiles.len());
        //self.ranges.masks = tile_renderer.masks_vbo.allocator.push(self.gpu_masks.len());
        //self.ranges.circles = tile_renderer.circles_vbo.allocator.push(self.circle_masks.len());
        self.ranges.edges = tile_renderer.edges_ssbo.allocator.push(self.line_edges.len());
    }

    pub fn upload(&mut self, tile_renderer: &mut crate::tile_renderer::TileRenderer, queue: &wgpu::Queue) {
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

        let (edges, offset) = if !self.line_edges.is_empty() {
            (bytemuck::cast_slice(&self.line_edges), self.ranges.edges.byte_offset::<LineEdge>())
        } else {
            (bytemuck::cast_slice(&self.quad_edges), self.ranges.edges.byte_offset::<QuadEdge>())
        };
        queue.write_buffer(
            &tile_renderer.edges_ssbo.buffer,
            offset,
            edges
        );
    }

    pub fn update_stats(&self, stats: &mut Stats) {
        stats.opaque_tiles += self.opaque_image_tiles.len();
        stats.alpha_tiles += self.alpha_tiles.len();
        stats.gpu_mask_tiles += self.fill_masks.masks.len() + self.circle_masks.masks.len();
        stats.cpu_mask_tiles += self.num_cpu_masks();
        stats.render_passes += self.render_passes.len();
        stats.batches += self.batches.len();
        stats.edges += self.line_edges.len() + self.quad_edges.len();
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
