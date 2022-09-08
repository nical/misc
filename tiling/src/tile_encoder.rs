use lyon::geom::{LineSegment, QuadraticBezierSegment};
use lyon::path::math::{Point, point, vector};
use lyon::path::FillRule;
use std::ops::Range;

use crate::tiler::*;
use crate::cpu_rasterizer::*;
use crate::tile_renderer::{TileInstance, Mask as GpuMask, CircleMask};
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
pub struct MaskPass {
    pub batches: Range<usize>,
    pub gpu_masks: Range<u32>,
    pub circle_masks: Range<u32>,
    pub mask_atlas_index: u32,
    pub color_atlas_index: u32,
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
pub type TextureIndex = u32;

pub struct TileAllocator {
    pub next_id: TileId,
    pub current_texture: TextureIndex,
    pub tiles_per_row: u32,
    pub tiles_per_atlas: u32,
}

impl TileAllocator {
    pub fn new(w: u32, h: u32) -> Self {
        TileAllocator {
            next_id: 1,
            current_texture: 0,
            tiles_per_row: w,
            tiles_per_atlas: w * h,
        }
    }

    pub fn reset(&mut self) {
        self.next_id = 1;
        self.current_texture = 0;
    }

    pub fn allocate(&mut self) -> (TilePosition, TextureIndex) {
        let id = self.next_id;
        self.next_id += 1;

        let mut id2 = id % self.tiles_per_atlas;

        if id2 == id {
            // Common path.
            let pos = TilePosition::new(
                id % self.tiles_per_row,
                id / self.tiles_per_row,
            );
            return (pos, self.current_texture)
        }

        if id2 == 0 {
            // Tile zero is reserved.
            id2 += 1;
        }

        self.next_id = id2 + 1;

        self.current_texture += 1;
        let pos = TilePosition::new(
            id2 % self.tiles_per_row,
            id2 / self.tiles_per_row,
        );

        (pos, self.current_texture)
    }
}

pub struct SourceTiles {
    pub mask_tiles: TileAllocator,
    pub color_tiles: TileAllocator,
}

pub struct TileEncoder {
    // State and output associated with the current group/layer:

    pub quad_edges: Vec<QuadEdge>,
    pub line_edges: Vec<LineEdge>,
    pub alpha_tiles: Vec<TileInstance>,
    pub opaque_image_tiles: Vec<TileInstance>,
    pub gpu_masks: Vec<GpuMask>,
    pub circle_masks: Vec<CircleMask>,
    pub mask_passes: Vec<MaskPass>,
    pub batches: Vec<AlphaBatch>,
    pub mask_uploader: MaskUploader,

    current_pattern_kind: Option<u32>,
    // First mask index of the current mask pass.
    gpu_masks_start: u32,
    circle_masks_start: u32,
    batches_start: usize,
    //cpu_masks_start: u32,
    // First masked color tile of the current mask pass.
    alpha_tiles_start: u32,
    // Index of the current mask texture (increments every time we run out of space in the
    // mask atlas, which is the primary reason for starting a new mask pass).
    masks_texture_index: u32,

    mask_id_range: Range<u32>,
    // TODO: move to drawing parameters.
    pub masks_per_atlas: u32,
    pub masks_per_row: u32,
    pub reversed: bool,

    // Transient state (only useful within a path):

    /// index of the last opaque solid tile in the current path. Used to detect
    /// consecutive solid tiles that can be merged.
    current_mergeable_tile: u32,

    // State associated with the current pattern and shape.

    // Debugging

    pub edge_distributions: [u32; 16],

    pub ranges: BufferRanges,

    // TODO, pass via parameter instead of owning it.
    pub src: SourceTiles,
    pub prerender_pattern: bool,
}

impl TileEncoder {
    pub fn new(config: &TilerConfig, mask_uploader: MaskUploader) -> Self {
        let atlas_tiles_x = config.mask_atlas_size.width as u32 / config.tile_size.width as u32;
        let atlas_tiles_y = config.mask_atlas_size.height as u32 / config.tile_size.height as u32;
        let tiles_per_atlas = atlas_tiles_x * atlas_tiles_y;
        TileEncoder {
            quad_edges: Vec::with_capacity(0),
            line_edges: Vec::with_capacity(8196),
            opaque_image_tiles: Vec::with_capacity(2048),
            alpha_tiles: Vec::with_capacity(8192),
            gpu_masks: Vec::with_capacity(8192),
            circle_masks: Vec::new(),
            mask_passes: Vec::with_capacity(16),
            batches: Vec::with_capacity(64),
            current_pattern_kind: None,
            current_mergeable_tile: 0, // Will be set in begin_row
            gpu_masks_start: 0,
            circle_masks_start: 0,
            batches_start: 0,
            //cpu_masks_start: 0,
            alpha_tiles_start: 0,
            masks_texture_index: 0,

            mask_uploader,

            mask_id_range: 0..0,
            masks_per_atlas: tiles_per_atlas,
            masks_per_row: atlas_tiles_x,
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
        let atlas_tiles_x = self.masks_per_row;
        let atlas_tiles_y = self.masks_per_atlas / atlas_tiles_x;
        TileEncoder {
            quad_edges: Vec::with_capacity(8196),
            line_edges: Vec::with_capacity(8196),
            opaque_image_tiles: Vec::with_capacity(2000),
            alpha_tiles: Vec::with_capacity(6000),
            gpu_masks: Vec::with_capacity(6000),
            circle_masks: Vec::new(),
            mask_passes: Vec::with_capacity(16),
            batches: Vec::with_capacity(64),
            current_pattern_kind: None,
            current_mergeable_tile: 0,
            gpu_masks_start: 0,
            circle_masks_start: 0,
            batches_start: 0,
            //cpu_masks_start: 0,
            alpha_tiles_start: 0,
            masks_texture_index: 0,

            mask_uploader: self.mask_uploader.create_similar(),

            mask_id_range: 0..0,
            masks_per_atlas: self.masks_per_atlas,
            masks_per_row: self.masks_per_row,
            reversed: false,

            ranges: BufferRanges::default(),

            edge_distributions: [0; 16],

            src: SourceTiles {
                mask_tiles: TileAllocator::new(atlas_tiles_x, atlas_tiles_y),
                color_tiles: TileAllocator::new(atlas_tiles_x, atlas_tiles_y),
            },
            prerender_pattern: self.prerender_pattern,
        }
    }

    pub fn set_tile_texture_size(&mut self, size: u32, tile_size: u32) {
        self.masks_per_atlas = (size * size) / (tile_size * tile_size);
    }

    pub fn reset(&mut self) {
        self.quad_edges.clear();
        self.line_edges.clear();
        self.opaque_image_tiles.clear();
        self.alpha_tiles.clear();
        self.gpu_masks.clear();
        self.circle_masks.clear();
        self.mask_passes.clear();
        self.batches.clear();
        self.mask_uploader.reset();
        self.current_pattern_kind = None;
        self.edge_distributions = [0; 16];
        self.mask_id_range = 0..0;
        self.gpu_masks_start = 0;
        self.circle_masks_start = 0;
        self.batches_start = 0;
        //self.cpu_masks_start = 0;
        self.alpha_tiles_start = 0;
        self.masks_texture_index = 0;
        self.reversed = false;
        self.ranges.reset();
        self.src.mask_tiles.reset();
        self.src.color_tiles.reset();
    }

    pub fn end_paths(&mut self) {
        self.flush_render_pass();
    }

    pub fn num_cpu_masks(&self) -> usize {
        self.mask_uploader.copy_instances().len()
    }

    pub fn num_mask_atlases(&self) -> u32 {
        //let id = self.mask_ids.current();
        //id / self.masks_per_atlas + if id % self.masks_per_atlas != 0 { 1 } else { 0 }
        self.src.mask_tiles.current_texture + 1
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
            pattern.prerender_tile(&mut self.src.color_tiles)
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

            tiles.push(TileInstance {
                position: tile_position,
                mask,
                pattern_position,
                pattern_data,
            });
        }
    }

    fn allocate_mask_tile(&mut self) -> TilePosition {
        let (id, texture) = self.src.mask_tiles.allocate();

        if texture != self.masks_texture_index {
            self.flush_render_pass();
            self.masks_texture_index = texture;
        }

        id
    }

    fn flush_render_pass(&mut self) {
        let gpu_masks_end = self.gpu_masks.len() as u32;
        let circle_masks_end = self.circle_masks.len() as u32;
        //let cpu_end = self.cpu_masks.len() as u32;
        self.end_batch();
        let batches_end = self.batches.len();
        if batches_end != self.batches_start { // TODO: cpu tiles
            self.mask_passes.push(MaskPass {
                gpu_masks: self.gpu_masks_start..gpu_masks_end,
                circle_masks: self.circle_masks_start..circle_masks_end,
                batches: self.batches_start..batches_end,
                mask_atlas_index: self.masks_texture_index,
                color_atlas_index: self.src.color_tiles.current_texture,
            });
            self.gpu_masks_start = gpu_masks_end;
            self.batches_start = batches_end;
            self.circle_masks_start = circle_masks_end;
        }
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
        self.batches.push(AlphaBatch {
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
        let tile = self.allocate_mask_tile();
        self.circle_masks.push(CircleMask { tile, radius, center: center.to_array() });

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
                //println!("   aux edge (enter) {:?}", self.line_edges.last().as_ref().map(|e| (e.0.y, e.1.y)).unwrap());
            }

            if edge.to.x < tile.outer_rect.min.x && edge.to.y != tile.outer_rect.min.y {
                self.line_edges.alloc().init(LineEdge(
                    point(-10.0, edge.to.y),
                    point(-10.0, tile.outer_rect.max.y),
                ));
                //println!("   aux edge (leave) {:?}", self.line_edges.last().as_ref().map(|e| (e.0.y, e.1.y)).unwrap());
            }

            if edge.is_line() {
                self.line_edges.alloc().init(LineEdge(edge.from - offset, edge.to - offset));
            } else {
                let curve = QuadraticBezierSegment { from: edge.from - offset, ctrl: edge.ctrl - offset, to: edge.to - offset };
                flatten_quad(&curve, draw.tolerance, &mut |segment| {
                    self.line_edges.push(LineEdge(segment.from, segment.to));
                });
            }
        }

        let edges_start = edges_start as u32;
        let edges_end = self.line_edges.len() as u32;
        debug_assert!(edges_end > edges_start, "{} > {} {:?}", edges_end, edges_start, active_edges);
        debug_assert!(edges_end - edges_start < 500, "edges {:?}", edges_start..edges_end);
        self.edge_distributions[(edges_end - edges_start).min(15) as usize] += 1;
        let tile_position = self.allocate_mask_tile();
        debug_assert!(tile_position.to_u32() != 0);

        self.gpu_masks.push(GpuMask {
            edges: (edges_start, edges_end),
            tile: tile_position,
            backdrop: tile.backdrop + 8192,
            fill_rule: draw.encoded_fill_rule,
        });

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
                //println!("   aux edge (enter) {:?}", self.line_edges.last().as_ref().map(|e| (e.0.y, e.1.y)).unwrap());
            }

            if edge.to.x < tile.outer_rect.min.x && edge.to.y != tile.outer_rect.min.y {
                self.line_edges.alloc().init(LineEdge(
                    point(-10.0, edge.to.y),
                    point(-10.0, tile.outer_rect.max.y),
                ));
                //println!("   aux edge (leave) {:?}", self.line_edges.last().as_ref().map(|e| (e.0.y, e.1.y)).unwrap());
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
        let tile_position = self.allocate_mask_tile();

        self.gpu_masks.push(GpuMask {
            edges: (edges_start, edges_end),
            tile: tile_position,
            backdrop: tile.backdrop + 8192,
            fill_rule: draw.encoded_fill_rule,
        });

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

        let mask_id = self.allocate_mask_tile();

        let mask_buffer_range = self.mask_uploader.new_mask(mask_id);
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

        mask_id
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
        self.mask_passes.reverse();
        for mask_pass in &mut self.mask_passes {
            mask_pass.batches = (num_batches - mask_pass.batches.end) .. (num_batches - mask_pass.batches.start);
        }
    }

    pub fn allocate_buffer_ranges(&mut self, tile_renderer: &mut crate::tile_renderer::TileRenderer) {
        self.ranges.opaque_image_tiles = tile_renderer.tiles_vbo.allocator.push(self.opaque_image_tiles.len());
        self.ranges.alpha_tiles = tile_renderer.tiles_vbo.allocator.push(self.alpha_tiles.len());
        self.ranges.masks = tile_renderer.masks_vbo.allocator.push(self.gpu_masks.len());
        self.ranges.circles = tile_renderer.circles_vbo.allocator.push(self.circle_masks.len());
        self.ranges.edges = tile_renderer.edges_ssbo.allocator.push(self.line_edges.len());
    }

    pub fn upload(&self, tile_renderer: &mut crate::tile_renderer::TileRenderer, queue: &wgpu::Queue) {
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

        queue.write_buffer(
            &tile_renderer.masks_vbo.buffer,
            self.ranges.masks.byte_offset::<GpuMask>(),
            bytemuck::cast_slice(&self.gpu_masks),
        );

        queue.write_buffer(
            &tile_renderer.circles_vbo.buffer,
            self.ranges.circles.byte_offset::<CircleMask>(),
            bytemuck::cast_slice(&self.circle_masks),
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
        stats.gpu_mask_tiles += self.gpu_masks.len();
        stats.cpu_mask_tiles += self.num_cpu_masks();
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
