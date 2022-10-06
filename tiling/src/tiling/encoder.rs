use std::ops::Range;

pub use lyon::path::math::{Point, point, Vector, vector};
pub use lyon::path::{PathEvent, FillRule};
pub use lyon::geom::euclid::default::{Box2D, Size2D, Transform2D};
pub use lyon::geom::euclid;
use lyon::geom::{LineSegment, QuadraticBezierSegment};

use crate::tiling::*;
use crate::tiling::tiler::{CircleMaskEncoder, RectangleMaskEncoder, GpuMaskEncoder};
use crate::tiling::cpu_rasterizer::*;
use crate::tiling::tile_renderer::{TileRenderer, TileInstance, MaskedTileInstance, Mask as GpuMask, CircleMask};
use crate::gpu::mask_uploader::MaskUploader;

use copyless::VecHelper;

fn opaque_span(
    mut x: u32,
    y: u32,
    num_tiles: u32,
    tile_mask: &mut TileMaskRow,
    patterns: &mut Vec<PatternTiles>,
    pattern: &mut dyn TilerPattern,
) {
    let mut start_x = x;
    let last_x = x + num_tiles;

    let mut occluded = !tile_mask.test(x, true);
    let mut is_last = false;

    let tiles = &mut patterns[pattern.index()].opaque;

    // Loop over x indices including 1 index past the range so that we
    // don't need to flush after the loop.
    'outer: loop {
        let flush = (occluded || is_last) && x > start_x;

        if flush {
            let ext = x - start_x - 1;

            let position = TilePosition::extended(start_x, y, ext);
            let pattern_data = pattern.tile_data(start_x, y);
            tiles.alloc().init(TileInstance {
                position,
                mask: TilePosition::ZERO,
                pattern_position: position,
                pattern_data,
            });

            start_x = x;
        }

        if is_last {
            break;
        }

        if occluded {
            // Skip over occluded tiles in a tight loop.
            loop {
                x += 1;
                is_last = x == last_x;
                if is_last {
                    break 'outer;
                }

                if occluded {
                    start_x = x;
                }

                occluded = !tile_mask.test(x, true);
                if !occluded {
                    break;
                }
            }
        }

        // Go over visible span in a tight loop.
        loop {
            x += 1;
            is_last = x == last_x;
            if is_last {
                break;
            }

            occluded = !tile_mask.test(x, true);
            if occluded {
                break;
            }
        }
    }
}

fn alpha_span(
    mut x: u32,
    y: u32,
    num_tiles: u32,
    tile_mask: &mut TileMaskRow,
    pattern: &mut dyn TilerPattern,
    encoder: &mut TileEncoder,
) {
    let mut start_x = x;
    let last_x = x + num_tiles;

    let opaque = false;
    let mut occluded = !tile_mask.test(x, opaque);
    let mut is_last = false;

    // Loop over x indices including 1 index past the range so that we
    // don't need to flush after the loop.
    'outer: loop {
        let flush = (occluded || is_last) && x > start_x;

        if flush {
            let ext = x - start_x - 1;

            let position = TilePosition::extended(start_x, y, ext);
            encoder.alpha_tiles.alloc().init(MaskedTileInstance {
                tile: TileInstance {
                    position: position,
                    mask: TilePosition::ZERO,
                    pattern_position: position,
                    pattern_data: pattern.tile_data(start_x, y),
                },
                mask: [0; 4],
            });

            start_x = x;
        }

        if is_last {
            break;
        }

        if occluded {
            // Skip over occluded tiles in a tight loop.
            loop {
                x += 1;
                is_last = x == last_x;
                if is_last {
                    break 'outer;
                }

                if occluded {
                    start_x = x;
                }

                occluded = !tile_mask.test(x, opaque);
                if !occluded {
                    break;
                }
            }
        }

        // Go over visible span in a tight loop.
        loop {
            x += 1;
            is_last = x == last_x;
            if is_last {
                break;
            }

            occluded = !tile_mask.test(x, opaque);
            if occluded {
                break;
            }
        }
    }
}

fn stretched_prerendered_span(
    mut x: u32,
    y: u32,
    num_tiles: u32,
    tile_mask: &mut TileMaskRow,
    pattern: &mut dyn TilerPattern,
    encoder: &mut TileEncoder,
) {
    let mut start_x = x;
    let last_x = x + num_tiles;

    let mut occluded = !tile_mask.test(x, false);
    let mut is_last = false;
    let mut prerendered_tile = None;

    // Loop over x indices including 1 index past the range so that we
    // don't need to flush after the loop.
    'outer: loop {
        let flush = (occluded || is_last) && x > start_x;

        if flush {
            let ext = x - start_x - 1;

            let position = TilePosition::extended(start_x, y, ext);
            let prerendered = match prerendered_tile {
                Some(tile) => tile,
                None => {
                    let atlas_position = encoder.allocate_color_tile();

                    let tiles = &mut encoder.patterns[pattern.index()].prerendered;

                    tiles.alloc().init(TileInstance {
                        position: atlas_position,
                        mask: TilePosition::ZERO,
                        pattern_position: position,
                        pattern_data: pattern.tile_data(start_x, y),
                    });

                    prerendered_tile = Some(atlas_position);
                    atlas_position
                }
            };

            encoder.alpha_tiles.alloc().init(MaskedTileInstance {
                tile: TileInstance {
                    position: position,
                    mask: TilePosition::ZERO,
                    pattern_position: prerendered,
                    pattern_data: 0,
                },
                mask: [0; 4],
            });

            start_x = x;
        }

        if is_last {
            break;
        }

        if occluded {
            // Skip over occluded tiles in a tight loop.
            loop {
                x += 1;
                is_last = x == last_x;
                if is_last {
                    break 'outer;
                }

                if occluded {
                    start_x = x;
                }

                occluded = !tile_mask.test(x, false);
                if !occluded {
                    break;
                }
            }
        }

        // Go over visible span in a tight loop.
        loop {
            x += 1;
            is_last = x == last_x;
            if is_last {
                break;
            }

            occluded = !tile_mask.test(x, false);
            if occluded {
                break;
            }
        }
    }
}

fn slow_span(
    x: u32,
    y: u32,
    num_tiles: u32,
    tile_mask: &mut TileMaskRow,
    pattern: &mut dyn TilerPattern,
    encoder: &mut TileEncoder,
) {
    for x in x .. x + num_tiles {
        let tile_vis = pattern.tile_visibility(x, y);
        if tile_vis.is_empty() {
            continue;
        }
        let opaque = tile_vis.is_opaque();

        let tile_position = TilePosition::new(x, y);

        if tile_mask.test(x, opaque) {
            let mask_tile = TilePosition::ZERO;
            encoder.add_tile(pattern, opaque, tile_position, mask_tile);
        }
    }
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
    pub alpha_batches: Range<usize>,
    pub mask_atlas_index: AtlasIndex,
    pub color_atlas_index: AtlasIndex,
}

pub struct Batch {
    pub tiles: Range<u32>,
    pub pattern: PatternIndex,
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

        // TODO: non-zero offsets.
        tile_renderer.edges.upload_bytes(0, edges, queue);
    }

    pub fn update_stats(&self, stats: &mut Stats) {
        stats.edges += self.line_edges.len() + self.quad_edges.len();
    }

    pub fn allocate_buffer_ranges(&mut self, tile_renderer: &mut TileRenderer) {
        tile_renderer.edges.bump_allocator().push(self.line_edges.len());
    }

    pub fn clear(&mut self) {
        self.line_edges.clear();
        self.quad_edges.clear();
    }
}

pub struct PatternTiles {
    opaque: Vec<TileInstance>,
    prerendered: Vec<TileInstance>,
    render_pass_start: u32,
    pub prerendered_vbo_range: BufferRange,
    pub opaque_vbo_range: BufferRange,
}

pub struct TileEncoder {
    // State and output associated with the current group/layer:

    // These should move into dedicated things.
    pub fill_masks: GpuMaskEncoder,
    pub circle_masks: CircleMaskEncoder,
    pub rect_masks: RectangleMaskEncoder,

    pub render_passes: Vec<RenderPass>,
    pub alpha_batches: Vec<Batch>,
    pub opaque_batches: Vec<Batch>,
    pub atlas_pattern_batches: Vec<Batch>,
    pub color_atlas_passes: Vec<Range<usize>>,
    pub alpha_tiles: Vec<MaskedTileInstance>,
    pub opaque_image_tiles: Vec<TileInstance>,
    pub patterns: Vec<PatternTiles>,

    current_pattern_index: Option<PatternIndex>,
    // First mask index of the current mask pass.
    alpha_batches_start: usize,
    opaque_image_tiles_start: u32,
    // First masked color tile of the current mask pass.
    alpha_tiles_start: u32,
    // Index of the current mask texture (increments every time we run out of space in the
    // mask atlas, which is the primary reason for starting a new mask pass).
    masks_texture_index: AtlasIndex,
    color_texture_index: AtlasIndex,

    reversed: bool,

    pub ranges: BufferRanges,

    pub prerender_pattern: bool,

    pub src: SourceTiles,

    pub mask_uploader: MaskUploader,

    pub edge_distributions: [u32; 16],
}

impl TileEncoder {
    pub fn new(config: &TilerConfig, mask_uploader: MaskUploader, num_patterns: usize) -> Self {
        let mask_atlas_tiles_x = config.mask_atlas_size.width / config.tile_size;
        let mask_atlas_tiles_y = config.mask_atlas_size.height / config.tile_size;
        let color_atlas_tiles_x = config.color_atlas_size.width / config.tile_size;
        let color_atlas_tiles_y = config.color_atlas_size.height / config.tile_size;
        let mut patterns = Vec::with_capacity(num_patterns);
        for _ in 0..num_patterns {
            patterns.alloc().init(PatternTiles {
                opaque: Vec::with_capacity(2048),
                prerendered: Vec::with_capacity(1024),
                render_pass_start: 0,
                opaque_vbo_range: BufferRange(0, 0),
                prerendered_vbo_range: BufferRange(0, 0),
            });
        }
        TileEncoder {
            opaque_image_tiles: Vec::with_capacity(2048),
            alpha_tiles: Vec::with_capacity(8192),
            fill_masks: GpuMaskEncoder::new(),
            circle_masks: CircleMaskEncoder::new(),
            rect_masks: RectangleMaskEncoder::new(),
            render_passes: Vec::with_capacity(16),
            alpha_batches: Vec::with_capacity(64),
            atlas_pattern_batches: Vec::with_capacity(64),
            color_atlas_passes: Vec::with_capacity(16),
            opaque_batches: Vec::with_capacity(num_patterns),
            patterns,
            current_pattern_index: None,
            alpha_batches_start: 0,
            opaque_image_tiles_start: 0,
            alpha_tiles_start: 0,
            masks_texture_index: 0,
            color_texture_index: 0,

            mask_uploader,

            reversed: false,

            ranges: BufferRanges::default(),

            edge_distributions: [0; 16],

            src: SourceTiles {
                mask_tiles: TileAllocator::new(mask_atlas_tiles_x, mask_atlas_tiles_y),
                color_tiles: TileAllocator::new(color_atlas_tiles_x, color_atlas_tiles_y),
            },
            prerender_pattern: true,
        }
    }

    pub fn get_opaque_batch(&self, pattern_idx: usize) -> Range<u32> {
        self.patterns[pattern_idx].opaque_vbo_range.to_u32()
    }

    pub fn create_similar(&self) -> Self {
        let num_patterns = self.patterns.len();
        let mut patterns = Vec::with_capacity(num_patterns);
        for _ in 0..num_patterns {
            patterns.alloc().init(PatternTiles {
                opaque: Vec::with_capacity(2048),
                prerendered: Vec::with_capacity(1024),
                render_pass_start: 0,
                opaque_vbo_range: BufferRange(0, 0),
                prerendered_vbo_range: BufferRange(0, 0),
            });
        }
        TileEncoder {
            opaque_image_tiles: Vec::with_capacity(2000),
            alpha_tiles: Vec::with_capacity(6000),
            fill_masks: GpuMaskEncoder::new(),
            circle_masks: CircleMaskEncoder::new(),
            rect_masks: RectangleMaskEncoder::new(),
            render_passes: Vec::with_capacity(16),
            alpha_batches: Vec::with_capacity(64),
            atlas_pattern_batches: Vec::with_capacity(64),
            color_atlas_passes: Vec::with_capacity(16),
            opaque_batches: Vec::with_capacity(num_patterns),
            patterns,
            current_pattern_index: None,
            alpha_batches_start: 0,
            opaque_image_tiles_start: 0,
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
        self.rect_masks.reset();
        self.render_passes.clear();
        self.alpha_batches.clear();
        self.opaque_batches.clear();
        self.mask_uploader.reset();
        self.current_pattern_index = None;
        self.edge_distributions = [0; 16];
        self.alpha_batches_start = 0;
        self.opaque_image_tiles_start = 0;
        self.alpha_tiles_start = 0;
        self.masks_texture_index = 0;
        self.color_texture_index = 0;
        self.reversed = false;
        self.ranges.reset();
        self.src.mask_tiles.reset();
        self.src.color_tiles.reset();
        self.atlas_pattern_batches.clear();
        self.color_atlas_passes.clear();
        for pattern in &mut self.patterns {
            pattern.opaque.clear();
            pattern.prerendered.clear();
            pattern.render_pass_start = 0;
        }
    }

    pub fn end_paths(&mut self) {
        self.fill_masks.end_render_pass();
        self.circle_masks.end_render_pass();
        self.rect_masks.end_render_pass();
        self.flush_render_pass(true);
    }

    pub fn num_cpu_masks(&self) -> usize {
        self.mask_uploader.copy_instances().len()
    }

    pub fn num_mask_atlases(&self) -> u32 {
        //let id = self.mask_ids.current();
        //id / self.masks_per_atlas + if id % self.masks_per_atlas != 0 { 1 } else { 0 }
        self.src.mask_tiles.current_atlas + 1
    }

    pub fn begin_path(&mut self, pattern: &mut dyn TilerPattern) {
        let index = if self.prerender_pattern {
            TILED_IMAGE_PATTERN
        } else{
            pattern.index()
        };

        if self.current_pattern_index != Some(index) {
            self.end_batch();
            self.current_pattern_index = Some(index);
        }
    }

    pub fn fill_rows(
        &mut self,
        rows: Range<u32>,
        columns: Range<u32>,
        pattern: &mut dyn TilerPattern,
        tile_mask: &mut TileMask,
    ) {
        for tile_y in rows {
            let mut tile_mask = tile_mask.row(tile_y);
            self.span(columns.clone(), tile_y, &mut tile_mask, pattern);
        }
    }

    pub fn span(
        &mut self,
        x: Range<u32>,
        y: u32,
        tile_mask: &mut TileMaskRow,
        pattern: &mut dyn TilerPattern,
    ) {
        let num_tiles = x.end - x.start;
        if pattern.is_entirely_opaque() {
            opaque_span(x.start, y, num_tiles, tile_mask, &mut self.patterns, pattern);
        } else if !self.prerender_pattern {
            alpha_span(x.start, y, num_tiles, tile_mask, pattern, self);
        } else if self.prerender_pattern && pattern.can_stretch_horizontally() {
            stretched_prerendered_span(x.start, y, num_tiles, tile_mask, pattern, self);
        } else {
            slow_span(x.start, y, num_tiles, tile_mask, pattern, self);
        }
    }

    pub fn add_tile(
        &mut self,
        pattern: &mut dyn TilerPattern,
        opaque: bool,
        tile_position: TilePosition,
        mask: TilePosition,
    ) {
        // It is always more efficient to render opaque tiles directly.
        let prerender = self.prerender_pattern && !opaque;

        let (pattern_position, pattern_data) = if prerender {
            let atlas_position = self.allocate_color_tile();
            let pattern_data = pattern.tile_data(tile_position.x(), tile_position.y());
            let tiles = &mut self.patterns[pattern.index()].prerendered;
            tiles.alloc().init(TileInstance {
                position: atlas_position,
                mask: TilePosition::ZERO,
                pattern_position: tile_position,
                pattern_data
            });

            (atlas_position, 0)
        } else {
            (tile_position, pattern.tile_data(tile_position.x(), tile_position.y()))
        };

        // Add the tile that will be rendered into the main pass.
        if opaque {
            let tiles = if prerender {
                &mut self.opaque_image_tiles
            } else {
                &mut self.patterns[pattern.index()].opaque
            };

            tiles.alloc().init(TileInstance {
                position: tile_position,
                mask,
                pattern_position,
                pattern_data,
            });
        } else {
            self.alpha_tiles.alloc().init(MaskedTileInstance {
                tile: TileInstance {
                    position: tile_position,
                    mask,
                    pattern_position,
                    pattern_data,
                },
                mask: [0; 4],
            });
        }
    }

    pub fn allocate_mask_tile(&mut self) -> TilePosition {
        let (tile, _) = self.src.mask_tiles.allocate();
        self.maybe_flush_render_pass();

        tile
    }

    pub fn allocate_color_tile(&mut self) -> TilePosition {
        let (tile, _) = self.src.color_tiles.allocate();
        self.maybe_flush_render_pass();

        tile
    }

    fn maybe_flush_render_pass(&mut self) {
        if self.masks_texture_index != self.src.mask_tiles.current_atlas
        || self.color_texture_index != self.src.color_tiles.current_atlas {
            self.flush_render_pass(false);
        }
    }

    // TODO: first allocate the mask and color tiles and then flush if either is
    // in a new texture (right now we only flush the mask atlas correctly).
    fn flush_render_pass(&mut self, force_flush: bool) {
        self.end_batch();

        let alpha_batches_end = self.alpha_batches.len();
        let opaque_image_tiles_end = self.opaque_image_tiles.len() as u32;

        if alpha_batches_end != self.alpha_batches_start {
            self.render_passes.alloc().init(RenderPass {
                opaque_image_tiles: self.opaque_image_tiles_start..opaque_image_tiles_end,
                alpha_batches: self.alpha_batches_start..alpha_batches_end,
                mask_atlas_index: self.masks_texture_index,
                color_atlas_index: self.color_texture_index,
            });
            self.alpha_batches_start = alpha_batches_end;
            self.opaque_image_tiles_start = opaque_image_tiles_end;
        }

        if force_flush || self.color_texture_index != self.src.color_tiles.current_atlas {
            let atlas_batches_start = self.atlas_pattern_batches.len();
            for (idx, pattern) in self.patterns.iter_mut().enumerate() {
                let len = pattern.prerendered.len() as u32;
                if len <= pattern.render_pass_start {
                    continue;
                }
                let tiles = pattern.render_pass_start .. len;
                pattern.render_pass_start = len;
                self.atlas_pattern_batches.push(Batch { pattern: idx, tiles });
            }
            let atlas_batches_end = self.atlas_pattern_batches.len();
            while self.color_atlas_passes.len() <= self.color_texture_index as usize {
                self.color_atlas_passes.push(0..0);
            }
            let range = atlas_batches_start .. atlas_batches_end;
            self.color_atlas_passes[self.color_texture_index as usize] = range;
        }

        self.masks_texture_index = self.src.mask_tiles.current_atlas;
        self.color_texture_index = self.src.color_tiles.current_atlas;
    }

    pub fn end_batch(&mut self) {
        let pattern_index = if let Some(index) = self.current_pattern_index {
            index
        } else {
            return;
        };

        let alpha_tiles_end = self.alpha_tiles.len() as u32;
        if self.alpha_tiles_start == alpha_tiles_end {
            return;
        }
        self.alpha_batches.alloc().init(Batch {
            tiles: self.alpha_tiles_start..alpha_tiles_end,
            pattern: pattern_index,
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

    pub fn add_cricle_mask(&mut self, center: Point, radius: f32, inverted: bool) -> TilePosition {
        let (mut tile, atlas_index) = self.src.mask_tiles.allocate();
        self.maybe_flush_render_pass();
        if inverted {
            tile.add_flag();
        }
        self.circle_masks.prerender_mask(atlas_index, CircleMask {
            tile, radius,
            center: center.to_array()
        });

        tile
    }

    pub fn add_rectangle_mask(&mut self, rect: &Box2D<f32>, inverted: bool, tile_size: f32) -> TilePosition {
        let (mut tile, atlas_index) = self.src.mask_tiles.allocate();
        self.maybe_flush_render_pass();

        let zero = point(0.0, 0.0);
        let one = point(1.0, 1.0);
        let min = ((rect.min / tile_size).clamp(zero, one) * std::u16::MAX as f32).to_u32();
        let max = ((rect.max / tile_size).clamp(zero, one) * std::u16::MAX as f32).to_u32();
        self.rect_masks.prerender_mask(atlas_index, RectangleMask {
            tile,
            invert: if inverted { 1 } else { 0 },
            rect: [
                min.x << 16 | min.y,
                max.x << 16 | max.y,
            ]

        });

        tile
    }

    pub fn add_line_gpu_mask(&mut self, tile: &TileInfo, draw: &DrawParams, active_edges: &[ActiveEdge], edges: &mut EdgeBuffer) -> Option<TilePosition> {
        //println!(" * tile ({:?}, {:?}), backdrop: {}, {:?}", tile.x, tile.y, tile.backdrop, active_edges);

        let edges_start = edges.line_edges.len();

        let offset = vector(tile.rect.min.x, tile.rect.min.y);

        // If the number of edges is larger than a certain threshold, we'll fall back to
        // rasterizing them on the CPU.
        if (edges.line_edges.len() - edges_start) + active_edges.len() > draw.max_edges_per_gpu_tile {
            edges.line_edges.resize(edges_start, LineEdge(point(0.0, 0.0), point(0.0, 0.0)));
            return None;
        }

        for edge in active_edges {
            // Handle auxiliary edges for edges that cross the tile's left side.
            if edge.from.x < tile.rect.min.x && edge.from.y != tile.rect.min.y {
                edges.line_edges.alloc().init(LineEdge(
                    point(-10.0, tile.rect.max.y),
                    point(-10.0, edge.from.y),
                ));
            }

            if edge.to.x < tile.rect.min.x && edge.to.y != tile.rect.min.y {
                edges.line_edges.alloc().init(LineEdge(
                    point(-10.0, edge.to.y),
                    point(-10.0, tile.rect.max.y),
                ));
            }

            if edge.is_line() {
                if edge.from.y != edge.to.y {
                    edges.line_edges.alloc().init(LineEdge(edge.from - offset, edge.to - offset));
                }
            } else {
                let curve = QuadraticBezierSegment { from: edge.from - offset, ctrl: edge.ctrl - offset, to: edge.to - offset };
                flatten_quad(&curve, draw.tolerance, &mut |segment| {
                    edges.line_edges.alloc().init(LineEdge(segment.from, segment.to));
                });
            }
        }

        let edges_start = edges_start as u32;
        let edges_end = edges.line_edges.len() as u32;
        //debug_assert!(edges_end > edges_start, "{} > {} {:?}", edges_end, edges_start, active_edges);
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

        let offset = vector(tile.rect.min.x, tile.rect.min.y);

        // If the number of edges is larger than a certain threshold, we'll fall back to
        // rasterizing them on the CPU.
        if (edges.quad_edges.len() - edges_start) + active_edges.len() > draw.max_edges_per_gpu_tile {
            edges.quad_edges.resize(edges_start, QuadEdge(point(0.0, 0.0), point(123.0, 456.0), point(0.0, 0.0), 0, 0));
            return None;
        }

        for edge in active_edges {
            // Handle auxiliary edges for edges that cross the tile's left side.
            if edge.from.x < tile.rect.min.x && edge.from.y != tile.rect.min.y {
                edges.quad_edges.alloc().init(QuadEdge(
                    point(-10.0, tile.rect.max.y),
                    point(123.0, 456.0),
                    point(-10.0, edge.from.y),
                    0, 0,
                ));
            }

            if edge.to.x < tile.rect.min.x && edge.to.y != tile.rect.min.y {
                edges.line_edges.alloc().init(LineEdge(
                    point(-10.0, edge.to.y),
                    point(-10.0, tile.rect.max.y),
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
        debug_assert!(draw.tile_size <= 32.0);
        debug_assert!(draw.tile_size <= 32.0);

        let mut accum = [0.0; 32 * 32];
        let mut backdrops = [tile.backdrop as f32; 32];

        let tile_offset = tile.rect.min.to_vector();
        for edge in active_edges {
            // Handle auxiliary edges for edges that cross the tile's left side.
            if edge.from.x < tile.rect.min.x && edge.from.y != tile.rect.min.y {
                add_backdrop(edge.from.y, -1.0, &mut backdrops[0..draw.tile_size as usize]);
            }

            if edge.to.x < tile.rect.min.x && edge.to.y != tile.rect.min.y {
                add_backdrop(edge.to.y, 1.0, &mut backdrops[0..draw.tile_size as usize]);
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

    pub fn get_color_atlas_batches(&self, color_atlas_index: u32) -> &[Batch] {
        let batch_range = self.color_atlas_passes[color_atlas_index as usize].clone();
        &self.atlas_pattern_batches[batch_range]
    }

    pub fn reverse_alpha_tiles(&mut self) {
        assert!(!self.reversed);
        self.alpha_tiles.reverse();

        self.alpha_batches.reverse();
        let num_alpha_tiles = self.alpha_tiles.len() as u32;
        for batch in &mut self.alpha_batches {
            batch.tiles = (num_alpha_tiles - batch.tiles.end) .. (num_alpha_tiles - batch.tiles.start);
        }
        let num_alpha_batches = self.alpha_batches.len();
        self.render_passes.reverse();
        for mask_pass in &mut self.render_passes {
            mask_pass.alpha_batches = (num_alpha_batches - mask_pass.alpha_batches.end) .. (num_alpha_batches - mask_pass.alpha_batches.start);
        }
    }

    pub fn allocate_buffer_ranges(&mut self, tile_renderer: &mut TileRenderer) {
        self.fill_masks.allocate_buffer_ranges(&mut tile_renderer.fill_masks);
        self.circle_masks.allocate_buffer_ranges(&mut tile_renderer.circle_masks);
        self.rect_masks.allocate_buffer_ranges(&mut tile_renderer.rect_masks);
        self.ranges.opaque_image_tiles = tile_renderer.tiles_vbo.allocator.push(self.opaque_image_tiles.len());
        self.ranges.alpha_tiles = tile_renderer.alpha_tiles_vbo.allocator.push(self.alpha_tiles.len());
        for pattern in &mut self.patterns {
            pattern.opaque_vbo_range = tile_renderer.tiles_vbo.allocator.push(pattern.opaque.len());
            pattern.prerendered_vbo_range = tile_renderer.tiles_vbo.allocator.push(pattern.prerendered.len());
        }
    }

    pub fn upload(&mut self, tile_renderer: &mut TileRenderer, queue: &wgpu::Queue) {
        self.fill_masks.upload(&mut tile_renderer.fill_masks, queue);
        self.circle_masks.upload(&mut tile_renderer.circle_masks, queue);
        self.rect_masks.upload(&mut tile_renderer.rect_masks, queue);

        queue.write_buffer(
            &tile_renderer.tiles_vbo.buffer,
            self.ranges.opaque_image_tiles.byte_offset::<TileInstance>(),
            bytemuck::cast_slice(&self.opaque_image_tiles),
        );

        queue.write_buffer(
            &tile_renderer.alpha_tiles_vbo.buffer,
            self.ranges.alpha_tiles.byte_offset::<TileInstance>(),
            bytemuck::cast_slice(&self.alpha_tiles),
        );

        for pattern in &self.patterns {
            if !pattern.opaque_vbo_range.is_empty() {
                queue.write_buffer(
                    &tile_renderer.tiles_vbo.buffer,
                    pattern.opaque_vbo_range.byte_offset::<TileInstance>(),
                    bytemuck::cast_slice(&pattern.opaque),
                );
            }
            if !pattern.prerendered_vbo_range.is_empty() {
                queue.write_buffer(
                    &tile_renderer.tiles_vbo.buffer,
                    pattern.prerendered_vbo_range.byte_offset::<TileInstance>(),
                    bytemuck::cast_slice(&pattern.prerendered),
                );
            }
        }
    }

    pub fn update_stats(&self, stats: &mut Stats) {
        for pattern in &self.patterns {
            stats.opaque_tiles += pattern.opaque.len();
            stats.prerendered_tiles += pattern.prerendered.len();
        }
        stats.opaque_tiles += self.opaque_image_tiles.len();
        stats.alpha_tiles += self.alpha_tiles.len();
        stats.gpu_mask_tiles += self.fill_masks.masks.len()
            + self.circle_masks.masks.len()
            + self.rect_masks.masks.len();
        stats.cpu_mask_tiles += self.num_cpu_masks();
        stats.render_passes += self.render_passes.len();
        stats.batches += self.alpha_batches.len() + self.opaque_batches.len();
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
