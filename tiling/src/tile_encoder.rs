use lyon::geom::QuadraticBezierSegment;
use lyon::path::math::{Point, point, vector};
use lyon::path::FillRule;
use std::sync::{
    Arc,
    atomic::{Ordering, AtomicU32},
};
use std::ops::Range;

use crate::tiler::*;
use crate::cpu_rasterizer::*;
use crate::Color;
use crate::gpu::solid_tiles::TileInstance as SolidTile;
use crate::gpu::masked_tiles::TileInstance as MaskedTile;
use crate::gpu::masked_tiles::Mask as GpuMask;
use crate::gpu::masked_tiles::MaskUploader;
use crate::api::Pattern;


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

struct Shared {
    next_mask_tile_id: AtomicU32,
}

// If we can't fit all masks into the atlas, we have to break the work into
// multiple passes. Each pass builds the atlas and renders it into the color target.
#[derive(Debug)]
pub struct MaskPass {
    pub gpu_masks: Range<u32>,
    //pub cpu_masks: Range<u32>,
    pub masked_tiles: Range<u32>,
    pub atlas_index: u32,
}

pub struct TileEncoder {
    // State and output associated with the current group/layer:

    pub quad_edges: Vec<QuadEdge>,
    pub line_edges: Vec<LineEdge>,
    pub solid_tiles: Vec<SolidTile>,
    pub masked_tiles: Vec<MaskedTile>,
    pub gpu_masks: Vec<GpuMask>,
    pub mask_passes: Vec<MaskPass>,
    pub mask_uploader: MaskUploader,

    // First mask index of the current mask pass.
    gpu_masks_start: u32,
    //cpu_masks_start: u32,
    // First masked color tile of the current mask pass.
    masked_tiles_start: u32,
    // Index of the current mask texture (increments every time we run out of space in the
    // mask atlas, which is the primary reason for starting a new mask pass).
    masks_texture_index: u32,

    mask_id_range: Range<u32>,
    pub masks_per_atlas: u32,
    shared: Arc<Shared>,

    // Transient state (only useful within a path):

    /// x coordinate of the last opaque solid tile in the current path. Used to detect
    /// consecutive solid tiles that can be merged.
    current_solid_tile: f32,

    // State associated with the current pattern and shape.

    // Debugging

    pub edge_distributions: [u32; 16],
}

impl TileEncoder {
    pub fn new(mask_uploader: MaskUploader) -> Self {
        TileEncoder {
            quad_edges: Vec::with_capacity(8196),
            line_edges: Vec::with_capacity(8196),
            solid_tiles: Vec::with_capacity(2000),
            masked_tiles: Vec::with_capacity(6000),
            gpu_masks: Vec::with_capacity(6000),
            mask_passes: Vec::with_capacity(16),
            current_solid_tile: std::f32::NAN,
            gpu_masks_start: 0,
            //cpu_masks_start: 0,
            masked_tiles_start: 0,
            masks_texture_index: 0,

            mask_uploader,

            shared: Arc::new(Shared {
                next_mask_tile_id: AtomicU32::new(0),
            }),
            mask_id_range: 0..0,
            masks_per_atlas: (2048 * 2048) / (16 * 16),

            edge_distributions: [0; 16],
        }
    }

    pub fn create_similar(&self) -> Self {
        TileEncoder {
            quad_edges: Vec::with_capacity(8196),
            line_edges: Vec::with_capacity(8196),
            solid_tiles: Vec::with_capacity(2000),
            masked_tiles: Vec::with_capacity(6000),
            gpu_masks: Vec::with_capacity(6000),
            mask_passes: Vec::with_capacity(16),
            current_solid_tile: std::f32::NAN,
            gpu_masks_start: 0,
            //cpu_masks_start: 0,
            masked_tiles_start: 0,
            masks_texture_index: 0,

            mask_uploader: self.mask_uploader.create_similar(),

            shared: Arc::new(Shared {
                next_mask_tile_id: AtomicU32::new(0),
            }),
            mask_id_range: 0..0,
            masks_per_atlas: (2048 * 2048) / (16 * 16),

            edge_distributions: [0; 16],
        }
    }

    pub fn new_parallel(other: &TileEncoder, mask_uploader: MaskUploader) -> Self {
        let mut encoder = TileEncoder::new(mask_uploader);
        encoder.shared = other.shared.clone();
        encoder.masks_per_atlas = other.masks_per_atlas;

        encoder
    }

    pub fn set_tile_texture_size(&mut self, size: u32, tile_size: u32) {
        self.masks_per_atlas = (size * size) / (tile_size * tile_size);
    }

    pub fn reset(&mut self) {
        self.quad_edges.clear();
        self.line_edges.clear();
        self.solid_tiles.clear();
        self.masked_tiles.clear();
        self.gpu_masks.clear();
        self.mask_passes.clear();
        self.mask_uploader.reset();
        self.shared.next_mask_tile_id.store(0, Ordering::Release);
        self.edge_distributions = [0; 16];
        self.mask_id_range = 0..0;
        self.gpu_masks_start = 0;
        //self.cpu_masks_start = 0;
        self.masked_tiles_start = 0;
        self.masks_texture_index = 0;
    }

    pub fn end_paths(&mut self) {
        let gpu_end = self.gpu_masks.len() as u32;
        //let cpu_end = self.cpu_masks.len() as u32;
        let masked_tiles_end = self.masked_tiles.len() as u32;
        if gpu_end != self.gpu_masks_start {
            self.mask_passes.push(MaskPass {
                gpu_masks: self.gpu_masks_start..gpu_end,
                //cpu_masks: self.cpu_masks_start..cpu_end,
                masked_tiles: self.masked_tiles_start..masked_tiles_end,
                atlas_index: self.masks_texture_index,
            });
            self.gpu_masks_start = gpu_end;
            //self.cpu_masks_start = cpu_end;
            self.masked_tiles_start = masked_tiles_end;
        }
    }

    pub fn num_cpu_masks(&self) -> usize {
        self.mask_uploader.copy_instances().len()
    }

    pub fn num_mask_atlases(&self) -> u32 {
        let id = self.shared.next_mask_tile_id.load(Ordering::Acquire);
        id / self.masks_per_atlas + if id % self.masks_per_atlas != 0 { 1 } else { 0 }
    }

    pub fn begin_row(&mut self) {
        self.current_solid_tile = std::f32::NAN;
    }

    pub fn end_row(&mut self) {}

    pub fn encode_tile(&mut self, tile: &TileInfo, draw: &DrawParams, active_edges: &[ActiveEdge]) {

        if tile.solid {
            if draw.merge_solid_tiles && (self.current_solid_tile - tile.output_rect.min.x).abs() < 0.01 {
                if let Some(solid) = self.solid_tiles.last_mut() {
                    solid.rect.max.x = tile.output_rect.max.x;
                }
            } else {
                self.solid_tiles.push(SolidTile {
                    rect: tile.output_rect,
                    color: match draw.pattern {
                        Pattern::Color(color) => color.to_u32(),
                        Pattern::Image(id) => id,
                    },
                });
            }
            self.current_solid_tile = tile.output_rect.max.x;

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
        let id = self.shared.next_mask_tile_id.fetch_add(16, Ordering::Relaxed);
        self.mask_id_range.start = id + 1;
        self.mask_id_range.end = id + 16;

        let texture_index = id / self.masks_per_atlas;
        if texture_index != self.masks_texture_index {
            let gpu_end = self.gpu_masks.len() as u32;
            //let cpu_end = self.cpu_masks.len() as u32;
            let masked_tiles_end = self.masked_tiles.len() as u32;
            if gpu_end != self.gpu_masks_start { // TODO: cpu tiles
                self.mask_passes.push(MaskPass {
                    gpu_masks: self.gpu_masks_start..gpu_end,
                    //cpu_masks: self.cpu_masks_start..cpu_end,
                    masked_tiles: self.masked_tiles_start..masked_tiles_end,
                    atlas_index: self.masks_texture_index,
                });
                self.gpu_masks_start = gpu_end;
                //self.cpu_masks_start = cpu_end;
                self.masked_tiles_start = masked_tiles_end;
            }
            self.masks_texture_index = texture_index;
        }

        id
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

        // Handle backdrop Auxiliary edges.
        {
            let mut from = tile.outer_rect.min.y;
            let mut to = tile.outer_rect.max.y;
            if tile.backdrop < 0 {
                std::mem::swap(&mut from, &mut to);
            }
            for _ in 0..tile.backdrop.abs() {
                self.line_edges.alloc().init(LineEdge(point(-10.0, from), point(-10.0, to)));
            }
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
                let mut from = curve.from;
                flatten_quad(&curve, draw.tolerance, &mut |to| {
                    self.line_edges.push(LineEdge(from, to));
                    from = to;
                });
            }
        }

        let edges_start = edges_start as u32;
        let edges_end = self.line_edges.len() as u32;
        assert!(edges_end > edges_start, "{:?}", active_edges);
        assert!(edges_end - edges_start < 500, "edges {:?}", edges_start..edges_end);
        self.edge_distributions[(edges_end - edges_start).min(15) as usize] += 1;
        let mask_id = self.allocate_mask_id();

        self.gpu_masks.push(GpuMask {
            edges: (edges_start, edges_end),
            mask_id,
            fill_rule: match draw.fill_rule {
                FillRule::EvenOdd => 0,
                FillRule::NonZero => 1,
            }
        });

        self.masked_tiles.push(MaskedTile {
            rect: tile.output_rect,
            color: match draw.pattern {
                Pattern::Color(color) => color.to_u32(),
                Pattern::Image(id) => id,
            },
            mask: mask_id,
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
            fill_rule: match draw.fill_rule {
                FillRule::EvenOdd => 0,
                FillRule::NonZero => 1,
            }
        });

        self.masked_tiles.push(MaskedTile {
            rect: tile.output_rect,
            color: match draw.pattern {
                Pattern::Color(color) => color.to_u32(),
                Pattern::Image(id) => id,
            },
            mask: mask_id,
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

            //let edge = edge.clip_horizontally(tile.outer_rect.min.x .. tile.outer_rect.max.x);
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

        self.masked_tiles.push(MaskedTile {
            rect: tile.output_rect,
            color: match draw.pattern {
                Pattern::Color(color) => color.to_u32(),
                Pattern::Image(id) => id,
            },
            mask: mask_id,
        });
    }
}

use std::sync::{Mutex, MutexGuard};

pub struct ParallelRasterEncoder<'l> {
    pub workers: Vec<Mutex<&'l mut TileEncoder>>,
}

impl<'l> ParallelRasterEncoder<'l> {
    pub fn lock_encoder(&self) -> MutexGuard<&'l mut TileEncoder> {
        let idx = rayon::current_thread_index().unwrap_or(self.workers.len() - 1);
        self.workers[idx].lock().unwrap()
    }
}

pub fn flatten_quad(curve: &QuadraticBezierSegment<f32>, tolerance: f32, cb: &mut impl FnMut(Point)) {
    let sq_error = square_distance_to_point(&curve.baseline().to_line(), curve.ctrl) * 0.25;

    let sq_tolerance = tolerance * tolerance;
    if sq_error <= sq_tolerance {
        cb(curve.to);
    } else if sq_error <= sq_tolerance * 4.0 {
        let ft = curve.from.lerp(curve.to, 0.5);
        let mid = ft.lerp(curve.ctrl, 0.5);
        cb(mid);
        cb(curve.to);
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
