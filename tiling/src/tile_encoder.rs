use lyon::geom::{LineSegment, QuadraticBezierSegment};
use lyon::path::math::{Point, point, vector};
use lyon::path::FillRule;
use std::ops::Range;

use crate::tiler::*;
use crate::cpu_rasterizer::*;
use crate::tile_renderer::{TileInstance, Mask as GpuMask};
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
    pub atlas_index: u32,
}

pub struct AlphaBatch {
    pub tiles: Range<u32>,
    pub batch_kind: u32,
}

pub struct TileEncoder {
    // State and output associated with the current group/layer:

    pub quad_edges: Vec<QuadEdge>,
    pub line_edges: Vec<LineEdge>,
    pub alpha_tiles: Vec<TileInstance>,
    pub opaque_solid_tiles: Vec<TileInstance>,
    pub opaque_image_tiles: Vec<TileInstance>,
    pub gpu_masks: Vec<GpuMask>,
    pub mask_passes: Vec<MaskPass>,
    pub batches: Vec<AlphaBatch>,
    pub mask_uploader: MaskUploader,

    current_pattern_kind: Option<u32>,
    // First mask index of the current mask pass.
    gpu_masks_start: u32,
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
    pub reversed: bool,
    mask_ids: TileIdAllcator,

    // Transient state (only useful within a path):

    /// x coordinate of the last opaque solid tile in the current path. Used to detect
    /// consecutive solid tiles that can be merged.
    current_solid_tile: f32,

    // State associated with the current pattern and shape.

    // Debugging

    pub edge_distributions: [u32; 16],
}

impl TileEncoder {
    pub fn new(config: &TilerConfig, mask_uploader: MaskUploader, mask_ids: TileIdAllcator) -> Self {
        TileEncoder {
            quad_edges: Vec::with_capacity(8196),
            line_edges: Vec::with_capacity(8196),
            opaque_solid_tiles: Vec::with_capacity(2000),
            opaque_image_tiles: Vec::with_capacity(2000),
            alpha_tiles: Vec::with_capacity(6000),
            gpu_masks: Vec::with_capacity(6000),
            mask_passes: Vec::with_capacity(16),
            batches: Vec::with_capacity(64),
            current_pattern_kind: None,
            current_solid_tile: std::f32::NAN,
            gpu_masks_start: 0,
            batches_start: 0,
            //cpu_masks_start: 0,
            alpha_tiles_start: 0,
            masks_texture_index: 0,

            mask_uploader,

            mask_ids,
            mask_id_range: 0..0,
            masks_per_atlas: config.mask_atlas_size.area() / config.tile_size.area() as u32,
            reversed: false,

            edge_distributions: [0; 16],
        }
    }

    pub fn create_similar(&self) -> Self {
        TileEncoder {
            quad_edges: Vec::with_capacity(8196),
            line_edges: Vec::with_capacity(8196),
            opaque_solid_tiles: Vec::with_capacity(2000),
            opaque_image_tiles: Vec::with_capacity(2000),
            alpha_tiles: Vec::with_capacity(6000),
            gpu_masks: Vec::with_capacity(6000),
            mask_passes: Vec::with_capacity(16),
            batches: Vec::with_capacity(64),
            current_pattern_kind: None,
            current_solid_tile: std::f32::NAN,
            gpu_masks_start: 0,
            batches_start: 0,
            //cpu_masks_start: 0,
            alpha_tiles_start: 0,
            masks_texture_index: 0,

            mask_uploader: self.mask_uploader.create_similar(),

            mask_ids: TileIdAllcator::new(),
            mask_id_range: 0..0,
            masks_per_atlas: self.masks_per_atlas,
            reversed: false,

            edge_distributions: [0; 16],
        }
    }

    pub fn new_parallel(other: &TileEncoder, config: &TilerConfig, mask_uploader: MaskUploader) -> Self {
        let mut encoder = TileEncoder::new(config, mask_uploader, TileIdAllcator::new());
        encoder.mask_ids = other.mask_ids.clone();
        encoder.masks_per_atlas = other.masks_per_atlas;

        encoder
    }

    pub fn set_tile_texture_size(&mut self, size: u32, tile_size: u32) {
        self.masks_per_atlas = (size * size) / (tile_size * tile_size);
    }

    pub fn reset(&mut self) {
        self.quad_edges.clear();
        self.line_edges.clear();
        self.opaque_solid_tiles.clear();
        self.opaque_image_tiles.clear();
        self.alpha_tiles.clear();
        self.gpu_masks.clear();
        self.mask_passes.clear();
        self.batches.clear();
        self.mask_uploader.reset();
        self.mask_ids.reset();
        self.current_pattern_kind = None;
        self.edge_distributions = [0; 16];
        self.mask_id_range = 0..0;
        self.gpu_masks_start = 0;
        self.batches_start = 0;
        //self.cpu_masks_start = 0;
        self.alpha_tiles_start = 0;
        self.masks_texture_index = 0;
        self.reversed = false;
    }

    pub fn end_paths(&mut self) {
        let gpu_masks_end = self.gpu_masks.len() as u32;
        //let cpu_end = self.cpu_masks.len() as u32;
        let alpha_tiles_end = self.alpha_tiles.len() as u32;
        self.batches.push(AlphaBatch {
            tiles: self.alpha_tiles_start..alpha_tiles_end,
            batch_kind: 0,
        });
        let batches_end = self.batches.len();
        self.mask_passes.push(MaskPass {
            gpu_masks: self.gpu_masks_start..gpu_masks_end,
            batches: self.batches_start..batches_end,
            atlas_index: self.masks_texture_index,
        });
        self.batches_start = batches_end;
        self.gpu_masks_start = gpu_masks_end;
        //self.cpu_masks_start = cpu_end;
        self.alpha_tiles_start = alpha_tiles_end;
    }

    pub fn num_cpu_masks(&self) -> usize {
        self.mask_uploader.copy_instances().len()
    }

    pub fn num_mask_atlases(&self) -> u32 {
        let id = self.mask_ids.current();
        id / self.masks_per_atlas + if id % self.masks_per_atlas != 0 { 1 } else { 0 }
    }

    pub fn begin_row(&mut self) {
        self.current_solid_tile = std::f32::NAN;
    }

    pub fn begin_path(&mut self, pattern: &mut dyn TilerPattern) {
        let pattern_kind = pattern.pattern_kind();

        if self.current_pattern_kind != Some(pattern_kind) {
            self.end_batch();
            self.current_pattern_kind = Some(pattern_kind);
        }
    }

    pub fn end_row(&mut self) {}

    pub fn encode_tile(&mut self, tile: &TileInfo, draw: &DrawParams, active_edges: &[ActiveEdge]) {

        if tile.solid && draw.is_opaque {
            if draw.merge_solid_tiles && (self.current_solid_tile - tile.output_rect.min.x).abs() < 0.01 && self.current_pattern_kind == Some(0) {
                if let Some(solid) = self.opaque_solid_tiles.last_mut() {
                    solid.width += 1;
                }
            } else {
                let tiles = match self.current_pattern_kind {
                    Some(0) => { &mut self.opaque_solid_tiles }
                    Some(1) => { &mut self.opaque_image_tiles }
                    _ => { panic!() }
                };
                tiles.push(TileInstance {
                    tile_id: tile.index,
                    pattern_data: tile.pattern_data,
                    mask: 0,
                    width: 0,
                });
            }
            self.current_solid_tile = tile.output_rect.max.x;

            return;
        }

        // Use masked tiles pipeline for blended full tiles to avoid breaking batches.
        if tile.solid && !draw.is_opaque {
            self.alpha_tiles.push(TileInstance {
                tile_id: tile.index,
                pattern_data: tile.pattern_data,
                mask: 0, // First mask is always fully opaque.
                width: 0,
            });

            return;
        }

        if active_edges.len() > draw.max_edges_per_gpu_tile
            || (draw.use_quads && !self.add_quad_gpu_mask(tile, draw, active_edges))
            || (!draw.use_quads && !self.add_line_gpu_mask(tile, draw, active_edges)) {
            self.add_cpu_mask(tile, draw, active_edges);
        }
    }

    fn allocate_mask_id(&mut self) -> u32 {
        if self.mask_id_range.end > self.mask_id_range.start {
            let id = self.mask_id_range.start;
            self.mask_id_range.start += 1;
            return id;
        }
        let mut id = self.mask_ids.allocate_n(16);
        self.mask_id_range.start = id + 1;
        self.mask_id_range.end = id + 16;

        if id % self.masks_per_atlas == 0 {
            // ID 0 is reserved for full mask.
            self.mask_id_range.start += 1;
            id += 1;
        }


        let texture_index = id / self.masks_per_atlas;
        if texture_index != self.masks_texture_index {
            // Add 1 to account for the special tile 0 which isn't in gpu_mask.
            let gpu_masks_end = self.gpu_masks.len() as u32;
            //let cpu_end = self.cpu_masks.len() as u32;
            let alpha_tiles_end = self.alpha_tiles.len() as u32;
            if gpu_masks_end != self.gpu_masks_start { // TODO: cpu tiles
                self.batches.push(AlphaBatch {
                    tiles: self.alpha_tiles_start..alpha_tiles_end,
                    batch_kind: 0,
                });
                let batches_end = self.batches.len();
                self.mask_passes.push(MaskPass {
                    gpu_masks: self.gpu_masks_start..gpu_masks_end,
                    batches: self.batches_start..batches_end,
                    atlas_index: self.masks_texture_index,
                });
                self.gpu_masks_start = gpu_masks_end;
                self.batches_start = batches_end;
                self.alpha_tiles_start = alpha_tiles_end;
            }
            self.masks_texture_index = texture_index;
        }

        id
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

    fn add_line_gpu_mask(&mut self, tile: &TileInfo, draw: &DrawParams, active_edges: &[ActiveEdge]) -> bool {
        //println!(" * tile ({:?}, {:?}), backdrop: {}, {:?}", tile.x, tile.y, tile.backdrop, active_edges);

        let edges_start = self.line_edges.len();

        let offset = vector(tile.inner_rect.min.x, tile.inner_rect.min.y);

        // If the number of edges is larger than a certain threshold, we'll fall back to
        // rasterizing them on the CPU.
        if (self.line_edges.len() - edges_start) + active_edges.len() > draw.max_edges_per_gpu_tile {
            self.line_edges.resize(edges_start, LineEdge(point(0.0, 0.0), point(0.0, 0.0)));
            return false;
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
        let mask_id = self.allocate_mask_id();
        debug_assert!(mask_id != 0);

        self.gpu_masks.push(GpuMask {
            edges: (edges_start, edges_end),
            mask_id,
            backdrop: tile.backdrop + 8192,
            fill_rule: draw.encoded_fill_rule,
        });

        self.alpha_tiles.push(TileInstance {
            tile_id: tile.index,
            pattern_data: tile.pattern_data,
            mask: mask_id,
            width: 0,
        });

        true
    }

    fn add_quad_gpu_mask(&mut self, tile: &TileInfo, draw: &DrawParams, active_edges: &[ActiveEdge]) -> bool {
        let edges_start = self.quad_edges.len();

        let offset = vector(tile.inner_rect.min.x, tile.inner_rect.min.y);

        // If the number of edges is larger than a certain threshold, we'll fall back to
        // rasterizing them on the CPU.
        if (self.quad_edges.len() - edges_start) + active_edges.len() > draw.max_edges_per_gpu_tile {
            self.quad_edges.resize(edges_start, QuadEdge(point(0.0, 0.0), point(123.0, 456.0), point(0.0, 0.0), 0, 0));
            return false;
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
        assert!(edges_end - edges_start < 500, "edges {:?}", edges_start..edges_end);
        let mask_id = self.allocate_mask_id();

        self.gpu_masks.push(GpuMask {
            edges: (edges_start, edges_end),
            mask_id,
            backdrop: tile.backdrop + 8192,
            fill_rule: draw.encoded_fill_rule,
        });

        self.alpha_tiles.push(TileInstance {
            tile_id: tile.index,
            pattern_data: tile.pattern_data,
            mask: mask_id,
            width: 0,
        });

        true
    }

    fn add_cpu_mask(&mut self, tile: &TileInfo, draw: &DrawParams, active_edges: &[ActiveEdge]) {
        debug_assert!(draw.tile_size.width <= 32.0);
        debug_assert!(draw.tile_size.height <= 32.0);

        let mut accum = [0.0; 32 * 32];
        let mut backdrops = [tile.backdrop as f32; 32];

        //left.print();
        //println!("offset {:?} : {:?}", tile.inner_rect.min.y, backdrops);

        let tile_offset = tile.inner_rect.min.to_vector();
        for edge in active_edges {
            // Handle auxiliary edges for edges that cross the tile's left side.
            if edge.from.x < tile.outer_rect.min.x && edge.from.y != tile.outer_rect.min.y {
                add_backdrop(edge.from.y, 1.0, &mut backdrops[0..draw.tile_size.height as usize]);
            }

            if edge.to.x < tile.outer_rect.min.x && edge.to.y != tile.outer_rect.min.y {
                add_backdrop(edge.to.y, -1.0, &mut backdrops[0..draw.tile_size.height as usize]);
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

        let mask_id = self.allocate_mask_id();

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

        //crate::cpu_rasterizer::dump_mask_png(
        //    draw.tile_size.width as u32, draw.tile_size.height as u32,
        //    &mut self.mask_uploader.current_mask_buffer[mask_buffer_range.clone()]
        //);

        self.alpha_tiles.push(TileInstance {
            tile_id: tile.index,
            pattern_data: tile.pattern_data,
            mask: mask_id,
            width: 0,
        });
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
