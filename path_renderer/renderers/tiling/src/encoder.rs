use std::ops::Range;

pub use lyon::geom::euclid::default::{Box2D, Size2D, Transform2D};
pub use lyon::path::{FillRule, PathEvent};

use super::mask::MaskEncoder;
use crate::atlas_uploader::TileAtlasUploader;
use crate::cpu_rasterizer::*;
use crate::resources::{Mask as GpuMask, TileInstance};
use crate::TILE_SIZE;
use crate::*;
use core::gpu::shader::ShaderPatternId;
use core::gpu::{DynBufferRange, DynamicStore};
use core::pattern::{BindingsId, BuiltPattern};
pub use core::units::{point, vector, Point, Vector};
use pattern_texture::TextureRenderer;

use copyless::VecHelper;
use core::{bytemuck, SurfaceDrawConfig};
use core::wgpu;

pub const SRC_COLOR_ATLAS_BINDING: BindingsId = BindingsId::from_index(65000);

fn opaque_span(
    mut x: u32,
    y: u32,
    num_tiles: u32,
    tile_mask: &mut TileMaskRow,
    patterns: &mut Vec<PatternTiles>,
    pattern: &BuiltPattern,
) {
    let mut start_x = x;
    let last_x = x + num_tiles;

    let mut occluded = !tile_mask.test(x, true);
    let mut is_last = false;

    let tiles = &mut patterns[pattern.shader.index()].opaque;
    // Loop over x indices including 1 index past the range so that we
    // don't need to flush after the loop.
    'outer: loop {
        let flush = (occluded || is_last) && x > start_x;

        if flush {
            let ext = x - start_x - 1;

            let position = TilePosition::extended(start_x, y, ext);
            let pattern_data = pattern.data;

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
    pattern: &BuiltPattern,
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
            encoder.alpha_tiles.alloc().init(TileInstance {
                position: position,
                mask: TilePosition::ZERO,
                pattern_position: position,
                pattern_data: pattern.data,
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
    pattern: &BuiltPattern,
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

                    let tiles = &mut encoder.patterns[pattern.shader.index()].prerendered;

                    tiles.alloc().init(TileInstance {
                        position: atlas_position,
                        mask: TilePosition::ZERO,
                        pattern_position: position,
                        pattern_data: pattern.data,
                    });

                    prerendered_tile = Some(atlas_position);
                    atlas_position
                }
            };

            encoder.alpha_tiles.alloc().init(TileInstance {
                position: position,
                mask: TilePosition::ZERO,
                pattern_position: prerendered,
                pattern_data: 0,
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
    pattern: &BuiltPattern,
    encoder: &mut TileEncoder,
) {
    for x in x..x + num_tiles {
        let tile_vis = tile_visibility(pattern);
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

pub fn clip_line_segment_1d(from: f32, to: f32, min: f32, max: f32) -> std::ops::Range<f32> {
    let d = to - from;
    if d == 0.0 {
        return 0.0..1.0;
    }

    let inv_d = 1.0 / d;

    let t0 = ((min - from) * inv_d).max(0.0);
    let t1 = ((max - from) * inv_d).min(1.0);

    t0..t1
}

pub fn as_scale_offset(m: &Transform2D<f32>) -> Option<(Vector, Vector)> {
    // Same as Skia's SK_ScalarNearlyZero.
    const EPSILON: f32 = 1.0 / 4096.0;

    if m.m12.abs() > EPSILON || m.m21.abs() > EPSILON {
        return None;
    }

    Some((vector(m.m11, m.m22), vector(m.m31, m.m32)))
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct LineEdge(pub Point, pub Point);

unsafe impl bytemuck::Pod for LineEdge {}
unsafe impl bytemuck::Zeroable for LineEdge {}

// If we can't fit all masks into the atlas, we have to break the work into
// multiple passes. Each pass builds the atlas and renders it into the color target.
#[derive(Debug)]
pub struct RenderPass {
    pub opaque_prerendered_tiles: Range<u32>,
    pub alpha_batches: Range<usize>,
    pub opaque_batches: Range<usize>,
    pub mask_atlas_index: AtlasIndex,
    pub color_atlas_index: AtlasIndex,
    pub z_index: u32,
    pub mask_pre_pass: bool,
    pub color_pre_pass: bool,
    pub surface: SurfaceDrawConfig,
}

#[derive(Debug)]
pub struct Batch {
    pub tiles: Range<u32>,
    pub pattern: ShaderPatternId,
    pub pattern_inputs: BindingsId,
    pub surface: SurfaceDrawConfig,
}

#[derive(Default, Debug)]
pub struct BufferRanges {
    pub opaque_prerendered_tiles: Option<DynBufferRange>,
    pub alpha_tiles: Option<DynBufferRange>,
}

impl BufferRanges {
    pub fn reset(&mut self) {
        self.opaque_prerendered_tiles = None;
        self.alpha_tiles = None;
    }
}

pub struct SourceTiles {
    pub mask_tiles: TileAllocator,
    pub color_tiles: TileAllocator,
}

pub struct PatternTiles {
    opaque: Vec<TileInstance>,
    prerendered: Vec<TileInstance>,
    prerendered_start: u32, // offset of the first prerendered tile for the current render pass
    opaque_start: u32,
    // TODO: probably also need to keep track of the current binding for prerendered tiles?
    // Or use this one.
    current_opaque_binding: BindingsId,
    pub prerendered_vbo_range: Option<DynBufferRange>,
    pub opaque_vbo_range: Option<DynBufferRange>,
}

pub struct TileEncoder {
    // State and output associated with the current group/layer:
    pub edges: Vec<LineEdge>,

    // These should move into dedicated things.
    pub fill_masks: MaskEncoder,

    pub render_passes: Vec<RenderPass>,
    pub alpha_batches: Vec<Batch>,
    pub opaque_batches: Vec<Batch>,
    pub atlas_pattern_batches: Vec<Batch>,
    pub color_atlas_passes: Vec<Range<usize>>,
    pub alpha_tiles: Vec<TileInstance>,
    pub opaque_prerendered_tiles: Vec<TileInstance>,
    pub patterns: Vec<PatternTiles>,

    current_pattern: Option<BuiltPattern>,
    // First mask index of the current mask pass.
    alpha_batches_start: usize,
    opaque_batches_start: usize,
    opaque_prerendered_tiles_start: u32,
    // First masked color tile of the current mask pass.
    alpha_tiles_start: u32,
    // Index of the current mask texture (increments every time we run out of space in the
    // mask atlas, which is the primary reason for starting a new mask pass).
    masks_texture_index: AtlasIndex,
    color_texture_index: AtlasIndex,
    pub current_z_index: u32,
    pub current_surface: SurfaceDrawConfig,
    tile_atlas_pattern: ShaderPatternId,

    reversed: bool,

    pub ranges: BufferRanges,

    pub prerender_pattern: bool,

    pub src: SourceTiles,

    pub mask_uploader: TileAtlasUploader,

    pub edge_distributions: [u32; 16],
}

impl TileEncoder {
    pub fn new(config: &TilerConfig, atlas_shader: &TextureRenderer, num_patterns: usize) -> Self {
        // TODO: should round up?
        let mask_atlas_tiles_x = config.mask_atlas_size.width / TILE_SIZE;
        let mask_atlas_tiles_y = config.mask_atlas_size.height / TILE_SIZE;
        let color_atlas_tiles_x = config.color_atlas_size.width / TILE_SIZE;
        let color_atlas_tiles_y = config.color_atlas_size.height / TILE_SIZE;
        let mut patterns = Vec::with_capacity(num_patterns);
        for _ in 0..num_patterns {
            patterns.alloc().init(PatternTiles {
                opaque: Vec::with_capacity(2048),
                prerendered: Vec::with_capacity(1024),
                prerendered_start: 0,
                opaque_start: 0,
                opaque_vbo_range: None,
                prerendered_vbo_range: None,
                current_opaque_binding: BindingsId::NONE,
            });
        }
        TileEncoder {
            edges: Vec::with_capacity(8182),
            opaque_prerendered_tiles: Vec::with_capacity(2048),
            alpha_tiles: Vec::with_capacity(8192),
            fill_masks: MaskEncoder::new(),
            render_passes: Vec::with_capacity(16),
            alpha_batches: Vec::with_capacity(64),
            atlas_pattern_batches: Vec::with_capacity(64),
            color_atlas_passes: Vec::with_capacity(16),
            opaque_batches: Vec::with_capacity(num_patterns),
            patterns,
            current_pattern: None,
            alpha_batches_start: 0,
            opaque_batches_start: 0,
            opaque_prerendered_tiles_start: 0,
            alpha_tiles_start: 0,
            masks_texture_index: 0,
            color_texture_index: 0,
            current_z_index: 0,
            current_surface: SurfaceDrawConfig::color(),
            tile_atlas_pattern: atlas_shader.load_pattern_id(),

            mask_uploader: TileAtlasUploader::new(config.staging_buffer_size),

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

    pub fn get_opaque_batch_vertices(&self, pattern_idx: usize) -> Option<&DynBufferRange> {
        let pattern = &self.patterns[pattern_idx];
        pattern.opaque_vbo_range.as_ref()
    }

    pub fn reset(&mut self) {
        self.edges.clear();
        self.opaque_prerendered_tiles.clear();
        self.alpha_tiles.clear();
        self.fill_masks.reset();
        self.render_passes.clear();
        self.alpha_batches.clear();
        self.opaque_batches.clear();
        self.mask_uploader.reset();
        self.current_pattern = None;
        self.edge_distributions = [0; 16];
        self.alpha_batches_start = 0;
        self.opaque_batches_start = 0;
        self.opaque_prerendered_tiles_start = 0;
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
            pattern.prerendered_start = 0;
            pattern.opaque_start = 0;
        }
    }

    // Call this to ensure tiles before and after are on separate sub-passes.
    // This is useful, for example to allow interleaving with other types of
    // rendering primitives.
    pub fn split_sub_pass(&mut self) {
        self.flush_render_pass(false);
    }

    // Call this once after tiling and before uploading.
    pub fn finish(&mut self, reversed: bool) {
        self.fill_masks.finish();
        self.flush_render_pass(true);

        assert!(!self.reversed);
        if reversed {
            self.alpha_tiles.reverse();

            self.alpha_batches.reverse();
            let num_alpha_tiles = self.alpha_tiles.len() as u32;
            for batch in &mut self.alpha_batches {
                batch.tiles =
                    (num_alpha_tiles - batch.tiles.end)..(num_alpha_tiles - batch.tiles.start);
            }

            //self.render_passes.reverse(); this is now done per-"batch" in the renderer.
        }

        if reversed {
            let num_alpha_batches = self.alpha_batches.len();
            for pass in &mut self.render_passes {
                pass.alpha_batches = (num_alpha_batches - pass.alpha_batches.end)
                    ..(num_alpha_batches - pass.alpha_batches.start);
            }
        }
    }

    pub fn num_cpu_masks(&self) -> usize {
        self.mask_uploader.num_tiles()
    }

    pub fn num_mask_atlases(&self) -> u32 {
        //let id = self.mask_ids.current();
        //id / self.masks_per_atlas + if id % self.masks_per_atlas != 0 { 1 } else { 0 }
        self.src.mask_tiles.current_atlas + 1
    }

    pub fn begin_path(&mut self, pattern: &BuiltPattern) {
        let pattern = if self.prerender_pattern {
            BuiltPattern::new(self.tile_atlas_pattern, 0).with_bindings(SRC_COLOR_ATLAS_BINDING)
        } else {
            *pattern
        };

        if self.current_pattern != Some(pattern) {
            let flush = self
                .current_pattern
                .map(|p| p.batch_key() != pattern.batch_key())
                .unwrap_or(true);
            if flush {
                self.end_alpha_batch();
                // if the binding changed for the opaque batch, break the batch.
                // this isn't ideal. Instead we could have a separate entry for patterns
                // with different bindings since we know they will never overlap in the
                // opaque pass.
                if self.patterns[pattern.shader.index()].current_opaque_binding != pattern.bindings
                {
                    self.end_opaque_batch(pattern.shader);
                    self.patterns[pattern.shader.index()].current_opaque_binding = pattern.bindings;
                }
            }
            self.current_pattern = Some(pattern);
        }
    }

    pub fn fill_rows(
        &mut self,
        rows: Range<u32>,
        columns: Range<u32>,
        pattern: &BuiltPattern,
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
        pattern: &BuiltPattern,
    ) {
        let num_tiles = x.end - x.start;
        if pattern.is_opaque {
            opaque_span(
                x.start,
                y,
                num_tiles,
                tile_mask,
                &mut self.patterns,
                pattern,
            );
        } else if !self.prerender_pattern {
            alpha_span(x.start, y, num_tiles, tile_mask, pattern, self);
        } else if self.prerender_pattern && pattern.can_stretch_horizontally {
            stretched_prerendered_span(x.start, y, num_tiles, tile_mask, pattern, self);
        } else {
            slow_span(x.start, y, num_tiles, tile_mask, pattern, self);
        }
    }

    pub fn add_tile(
        &mut self,
        pattern: &BuiltPattern,
        opaque: bool,
        tile_position: TilePosition,
        //TODO: Pass the mask data instead.
        mask: TilePosition,
    ) {
        // It is always more efficient to render opaque tiles directly.
        let prerender = self.prerender_pattern && !opaque;

        let (pattern_position, pattern_data) = if prerender {
            let atlas_position = self.allocate_color_tile();
            let pattern_data = pattern.data;
            let tiles = &mut self.patterns[pattern.shader.index()].prerendered;
            tiles.alloc().init(TileInstance {
                position: atlas_position,
                mask: TilePosition::ZERO,
                pattern_position: tile_position,
                pattern_data,
            });

            (atlas_position, 0)
        } else {
            (tile_position, pattern.data)
        };

        // Add the tile that will be rendered into the main pass.
        if opaque {
            let tiles = if prerender {
                &mut self.opaque_prerendered_tiles
            } else {
                &mut self.patterns[pattern.shader.index()].opaque
            };

            tiles.alloc().init(TileInstance {
                position: tile_position,
                mask,
                pattern_position,
                pattern_data,
            });
        } else {
            self.alpha_tiles.alloc().init(TileInstance {
                position: tile_position,
                mask,
                pattern_position,
                pattern_data,
            });
        }
    }

    pub fn allocate_color_tile(&mut self) -> TilePosition {
        let (tile, _) = self.src.color_tiles.allocate();
        self.maybe_flush_render_pass();

        tile
    }

    fn maybe_flush_render_pass(&mut self) {
        if self.masks_texture_index != self.src.mask_tiles.current_atlas
            || self.color_texture_index != self.src.color_tiles.current_atlas
        {
            self.flush_render_pass(false);
        }
    }

    // TODO: first allocate the mask and color tiles and then flush if either is
    // in a new texture (right now we only flush the mask atlas correctly).
    fn flush_render_pass(&mut self, force_flush: bool) {
        self.end_alpha_batch();
        self.end_opaque_batches();

        let alpha_batches_end = self.alpha_batches.len();
        let opaque_batches_end = self.opaque_batches.len();
        let opaque_prerendered_tiles_end = self.opaque_prerendered_tiles.len() as u32;

        if alpha_batches_end != self.alpha_batches_start
            || opaque_batches_end != self.opaque_batches_start
            || opaque_prerendered_tiles_end != self.opaque_prerendered_tiles_start
        {
            self.render_passes.alloc().init(RenderPass {
                opaque_prerendered_tiles: self.opaque_prerendered_tiles_start
                    ..opaque_prerendered_tiles_end,
                alpha_batches: self.alpha_batches_start..alpha_batches_end,
                opaque_batches: self.opaque_batches_start..opaque_batches_end,
                mask_atlas_index: self.masks_texture_index,
                color_atlas_index: self.color_texture_index,
                z_index: self.current_z_index,
                mask_pre_pass: false,
                color_pre_pass: false,
                surface: self.current_surface,
            });
            self.alpha_batches_start = alpha_batches_end;
            self.opaque_batches_start = opaque_batches_end;
            self.opaque_prerendered_tiles_start = opaque_prerendered_tiles_end;
        }

        if force_flush || self.color_texture_index != self.src.color_tiles.current_atlas {
            let atlas_batches_start = self.atlas_pattern_batches.len();
            for (idx, pattern) in self.patterns.iter_mut().enumerate() {
                let len = pattern.prerendered.len() as u32;
                if len <= pattern.prerendered_start {
                    continue;
                }
                let tiles = pattern.prerendered_start..len;
                pattern.prerendered_start = len;
                self.atlas_pattern_batches.push(Batch {
                    pattern: ShaderPatternId::from_index(idx),
                    pattern_inputs: BindingsId::NONE, // TODO
                    tiles,
                    surface: self.current_surface,
                });
            }
            let atlas_batches_end = self.atlas_pattern_batches.len();
            while self.color_atlas_passes.len() <= self.color_texture_index as usize {
                self.color_atlas_passes.push(0..0);
            }
            let range = atlas_batches_start..atlas_batches_end;
            self.color_atlas_passes[self.color_texture_index as usize] = range;
        }

        self.masks_texture_index = self.src.mask_tiles.current_atlas;
        self.color_texture_index = self.src.color_tiles.current_atlas;

        if force_flush {
            self.mask_uploader.flush_batch();
        }
    }

    fn end_alpha_batch(&mut self) {
        let pattern = if let Some(pattern) = self.current_pattern {
            pattern
        } else {
            return;
        };

        let alpha_tiles_end = self.alpha_tiles.len() as u32;
        if self.alpha_tiles_start != alpha_tiles_end {
            self.alpha_batches.alloc().init(Batch {
                tiles: self.alpha_tiles_start..alpha_tiles_end,
                pattern: pattern.shader,
                pattern_inputs: pattern.bindings,
                surface: self.current_surface,
            });
            self.alpha_tiles_start = alpha_tiles_end;
        }
    }

    fn end_opaque_batch(&mut self, pattern_id: ShaderPatternId) {
        let pattern = &mut self.patterns[pattern_id.index()];
        let opaque_end = pattern.opaque.len() as u32;
        if pattern.opaque_start == opaque_end {
            return;
        }

        self.opaque_batches.push(Batch {
            tiles: pattern.opaque_start..opaque_end,
            pattern: pattern_id,
            pattern_inputs: pattern.current_opaque_binding,
            surface: self.current_surface,
        });

        pattern.opaque_start = opaque_end;
    }

    fn end_opaque_batches(&mut self) {
        for (pattern_idx, pattern) in self.patterns.iter_mut().enumerate() {
            let opaque_end = pattern.opaque.len() as u32;
            if pattern.opaque_start == opaque_end {
                continue;
            }

            self.opaque_batches.push(Batch {
                tiles: pattern.opaque_start..opaque_end,
                pattern: ShaderPatternId::from_index(pattern_idx),
                pattern_inputs: pattern.current_opaque_binding,
                surface: self.current_surface,
            });

            pattern.opaque_start = opaque_end;
        }
    }

    pub fn add_fill_mask(
        &mut self,
        tile: &TileInfo,
        draw: &DrawParams,
        active_edges: &[ActiveEdge],
        device: &wgpu::Device,
    ) -> TilePosition {
        self.add_gpu_mask(tile, draw, active_edges)
            .unwrap_or_else(|| self.add_cpu_mask(tile, draw, active_edges, device))
    }

    pub fn allocate_mask_tile(&mut self) -> (TilePosition, AtlasIndex) {
        let (tile, atlas_index) = self.src.mask_tiles.allocate();
        self.maybe_flush_render_pass();

        (tile, atlas_index)
    }

    pub fn add_gpu_mask(
        &mut self,
        tile: &TileInfo,
        draw: &DrawParams,
        active_edges: &[ActiveEdge],
    ) -> Option<TilePosition> {
        //println!(" * tile ({:?}, {:?}), backdrop: {}, {:?}", tile.x, tile.y, tile.backdrop, active_edges);

        if active_edges.len() > draw.max_edges_per_gpu_tile {
            return None;
        }

        let edges_start = self.edges.len();

        let offset = vector(tile.rect.min.x, tile.rect.min.y);

        for edge in active_edges {
            // Handle auxiliary edges for edges that cross the tile's left side.
            if edge.from.x < tile.rect.min.x && edge.from.y != tile.rect.min.y {
                self.edges.alloc().init(LineEdge(
                    point(-10.0, tile.rect.max.y),
                    point(-10.0, edge.from.y),
                ));
            }

            if edge.to.x < tile.rect.min.x && edge.to.y != tile.rect.min.y {
                self.edges.alloc().init(LineEdge(
                    point(-10.0, edge.to.y),
                    point(-10.0, tile.rect.max.y),
                ));
            }

            if edge.from.y != edge.to.y {
                self.edges
                    .alloc()
                    .init(LineEdge(edge.from - offset, edge.to - offset));
            }
        }

        // If the number of edges is larger than a certain threshold, we'll fall back to
        // rasterizing them on the CPU.
        if (self.edges.len() - edges_start) > draw.max_edges_per_gpu_tile {
            self.edges.truncate(edges_start);
            return None;
        }

        let edges_start = edges_start as u32;
        let edges_end = self.edges.len() as u32;
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
            },
        );

        Some(tile_position)
    }

    pub fn add_cpu_mask(
        &mut self,
        tile: &TileInfo,
        draw: &DrawParams,
        active_edges: &[ActiveEdge],
        device: &wgpu::Device,
    ) -> TilePosition {
        let mut accum = [0.0; 32 * 32];
        let mut backdrops = [tile.backdrop as f32; 32];

        let tile_offset = tile.rect.min.to_vector();
        for edge in active_edges {
            // Handle auxiliary edges for edges that cross the tile's left side.
            if edge.from.x < tile.rect.min.x && edge.from.y != tile.rect.min.y {
                add_backdrop(edge.from.y, -1.0, &mut backdrops[0..TILE_SIZE as usize]);
            }

            if edge.to.x < tile.rect.min.x && edge.to.y != tile.rect.min.y {
                add_backdrop(edge.to.y, 1.0, &mut backdrops[0..TILE_SIZE as usize]);
            }

            let from = edge.from - tile_offset;
            let to = edge.to - tile_offset;

            draw_line(from, to, &mut accum);
        }

        let (tile_position, atlas_index) = self.src.mask_tiles.allocate();
        //println!("cpu mask at position {} {} atlas {:?}", tile_position.x(), tile_position.y(), atlas_index);
        self.maybe_flush_render_pass();

        let mask_buffer = self
            .mask_uploader
            .add_tile(device, tile_position, atlas_index);

        let accumulate = match draw.fill_rule {
            FillRule::EvenOdd => accumulate_even_odd,
            FillRule::NonZero => accumulate_non_zero,
        };

        accumulate(&accum, &backdrops, mask_buffer);

        //let mask_name = format!("mask-{}.png", mask_id.to_u32());
        //crate::cpu_rasterizer::save_mask_png(16, 16, &self.mask_uploader.current_mask_buffer[mask_buffer_range.clone()], &mask_name);
        //crate::cpu_rasterizer::save_accum_png(16, 16, &accum, &backdrops, &format!("accum-{}.png", mask_id.to_u32()));

        tile_position
    }

    pub fn get_color_atlas_batches(&self, color_atlas_index: u32) -> &[Batch] {
        let batch_range = self.color_atlas_passes[color_atlas_index as usize].clone();
        &self.atlas_pattern_batches[batch_range]
    }

    pub fn upload(&mut self, vertices: &mut DynamicStore, device: &wgpu::Device) {
        self.fill_masks.upload(vertices, device);

        self.ranges.opaque_prerendered_tiles =
            vertices.upload(device, bytemuck::cast_slice(&self.opaque_prerendered_tiles));
        self.ranges.alpha_tiles = vertices.upload(device, bytemuck::cast_slice(&self.alpha_tiles));

        for pattern in &mut self.patterns {
            pattern.opaque_vbo_range =
                vertices.upload(device, bytemuck::cast_slice(&pattern.opaque));
            pattern.prerendered_vbo_range =
                vertices.upload(device, bytemuck::cast_slice(&pattern.prerendered));
        }
    }

    pub fn update_stats(&self, stats: &mut Stats) {
        stats.edges += self.edges.len();
        for pattern in &self.patterns {
            stats.opaque_tiles += pattern.opaque.len();
            stats.prerendered_tiles += pattern.prerendered.len();
        }
        stats.opaque_tiles += self.opaque_prerendered_tiles.len();
        stats.alpha_tiles += self.alpha_tiles.len();
        stats.gpu_mask_tiles += self.fill_masks.masks.len();
        stats.cpu_mask_tiles += self.num_cpu_masks();
        stats.render_passes += self.render_passes.len();
        stats.batches += self.alpha_batches.len() + self.opaque_batches.len();
    }
}

pub struct TileAllocator {
    pub next_id: u32,
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
            let pos = TilePosition::new(id % self.tiles_per_row, id / self.tiles_per_row);
            return (pos, self.current_atlas);
        }

        if id2 == 0 {
            // Tile zero is reserved.
            id2 += 1;
        }

        self.next_id = id2 + 1;

        self.current_atlas += 1;
        let pos = TilePosition::new(id2 % self.tiles_per_row, id2 / self.tiles_per_row);

        (pos, self.current_atlas)
    }

    pub fn finish_atlas(&mut self) {
        self.current_atlas += 1;
        self.next_id = 1;
    }

    pub fn width(&self) -> u32 {
        self.tiles_per_row
    }

    pub fn height(&self) -> u32 {
        self.tiles_per_atlas / self.tiles_per_row
    }

    pub fn current_atlas(&self) -> u32 {
        self.current_atlas
    }

    pub fn is_nearly_full(&self) -> bool {
        (self.next_id * 100) / self.tiles_per_atlas > 70
    }
}
