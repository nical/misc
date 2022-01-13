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
    next_mask_id: AtomicU32,
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

pub struct GpuRasterEncoder {
    pub quad_edges: Vec<QuadEdge>,
    pub line_edges: Vec<LineEdge>,
    pub solid_tiles: Vec<SolidTile>,
    pub mask_tiles: Vec<MaskedTile>,
    pub gpu_masks: Vec<GpuMask>,
    pub mask_passes: Vec<MaskPass>,
    gpu_masks_start: u32,
    //cpu_masks_start: u32,
    masked_tiles_start: u32,
    masks_texture_index: u32,
    pub color: Color,
    pub fill_rule: FillRule,
    pub max_edges_per_gpu_tile: usize,
    pub is_opaque: bool,
    pub use_quads: bool,
    pub tolerance: f32,
    pub mask_uploader: MaskUploader,
    shared: Arc<Shared>,
    mask_id_range: Range<u32>,
    pub masks_per_atlas: u32,
    tile_size: f32,

    pub edge_distributions: [u32; 16],
}

impl GpuRasterEncoder {
    pub fn new(tolerance: f32, mask_uploader: MaskUploader) -> Self {
        GpuRasterEncoder {
            quad_edges: Vec::with_capacity(8196),
            line_edges: Vec::with_capacity(8196),
            solid_tiles: Vec::with_capacity(2000),
            mask_tiles: Vec::with_capacity(6000),
            gpu_masks: Vec::with_capacity(6000),
            gpu_masks_start: 0,
            //cpu_masks_start: 0,
            masked_tiles_start: 0,
            masks_texture_index: 0,
            mask_passes: Vec::with_capacity(16),
            color: Color { r: 0, g: 0, b: 0, a: 0 },
            fill_rule: FillRule::EvenOdd,
            max_edges_per_gpu_tile: 4096,
            is_opaque: true,
            use_quads: false,
            tolerance,
            tile_size: 16.0,

            mask_uploader,

            shared: Arc::new(Shared {
                next_mask_id: AtomicU32::new(0),
            }),
            mask_id_range: 0..0,
            masks_per_atlas: (2048 * 2048) / (16 * 16),

            edge_distributions: [0; 16],
        }
    }

    pub fn new_parallel(other: &GpuRasterEncoder, mask_uploader: MaskUploader) -> Self {
        let mut encoder = GpuRasterEncoder::new(other.tolerance, mask_uploader);
        encoder.max_edges_per_gpu_tile = other.max_edges_per_gpu_tile;
        encoder.use_quads = other.use_quads;
        encoder.shared = other.shared.clone();
        encoder.fill_rule = other.fill_rule;
        encoder.masks_per_atlas = other.masks_per_atlas;
        encoder.tile_size = other.tile_size;

        encoder
    }

    pub fn set_tile_texture_size(&mut self, size: u32, tile_size: u32) {
        self.masks_per_atlas = (size * size) / (tile_size * tile_size);
        self.tile_size = tile_size as f32;
    }

    pub fn reset(&mut self) {
        self.quad_edges.clear();
        self.line_edges.clear();
        self.solid_tiles.clear();
        self.mask_tiles.clear();
        self.gpu_masks.clear();
        self.mask_passes.clear();
        self.mask_uploader.reset();
        self.shared.next_mask_id.store(0, Ordering::Release);
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
        let masked_tiles_end = self.mask_tiles.len() as u32;
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
        let id = self.shared.next_mask_id.load(Ordering::Acquire);
        id / self.masks_per_atlas + if id % self.masks_per_atlas != 0 { 1 } else { 0 }
    }

    fn allocate_mask_id(&mut self) -> u32 {
        if self.mask_id_range.end > self.mask_id_range.start {
            let id = self.mask_id_range.start;
            self.mask_id_range.start += 1;
            return id;
        }
        let id = self.shared.next_mask_id.fetch_add(16, Ordering::Relaxed);
        self.mask_id_range.start = id + 1;
        self.mask_id_range.end = id + 16;

        let texture_index = id / self.masks_per_atlas;
        if texture_index != self.masks_texture_index {
            let gpu_end = self.gpu_masks.len() as u32;
            //let cpu_end = self.cpu_masks.len() as u32;
            let masked_tiles_end = self.mask_tiles.len() as u32;
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


    fn add_line_gpu_mask(&mut self, tile: &TileInfo, active_edges: &[ActiveEdge], left: &SideEdgeTracker) -> bool {
        let tx = tile.x as f32 * self.tile_size;
        let ty = tile.y as f32 * self.tile_size;

        let edges_start = self.line_edges.len();

        // TODO: should it be the outer rect?
        let offset = vector(tile.inner_rect.min.x, tile.inner_rect.min.y);

        let mut prev: Option<SideEvent> = None;
        for evt in left.events() {
            if let Some(prev) = prev {
                let x = -1.0;
                let mut from = point(x, prev.y - offset.y);
                let mut to = point(x, evt.y - offset.y);
                if prev.winding < 0 {
                    std::mem::swap(&mut from, &mut to);
                }

                for _ in 0..prev.winding.abs() {
                    self.line_edges.alloc().init(LineEdge(from, to));
                }
            }

            if evt.winding != 0 {
                prev = Some(*evt);
            } else {
                prev = None;
            }
        }

        // If the number of edges is larger than a certain threshold, we'll fall back to
        // rasterizing them on the CPU.
        if (self.line_edges.len() - edges_start) + active_edges.len() > self.max_edges_per_gpu_tile {
            self.line_edges.resize(edges_start, LineEdge(point(0.0, 0.0), point(0.0, 0.0)));
            return false;
        }

        for edge in active_edges {
            if edge.ctrl.x.is_nan() {
                let mut e = LineEdge(edge.from - offset, edge.to - offset);
                if edge.winding < 0 {
                    std::mem::swap(&mut e.0, &mut e.1);
                }
                self.line_edges.alloc().init(e);
            } else {
                let mut curve = QuadraticBezierSegment { from: edge.from - offset, ctrl: edge.ctrl - offset, to: edge.to - offset };
                if edge.winding < 0 {
                    std::mem::swap(&mut curve.from, &mut curve.to);
                }
                let mut from = curve.from;
                flatten_quad(&curve, self.tolerance, &mut |to| {
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
            fill_rule: match self.fill_rule {
                FillRule::EvenOdd => 0,
                FillRule::NonZero => 1,
            }
        });

        self.mask_tiles.push(MaskedTile {
            rect: Box2D {
                min: point(tx, ty),
                max: point(tx + self.tile_size, ty + self.tile_size),
            },
            color: self.color.to_u32(),
            mask: mask_id,
        });

        true
    }

    fn add_quad_gpu_mask(&mut self, tile: &TileInfo, active_edges: &[ActiveEdge], left: &SideEdgeTracker) -> bool {
        let tx = tile.x as f32 * self.tile_size;
        let ty = tile.y as f32 * self.tile_size;

        let edges_start = self.quad_edges.len();

        // TODO: should it be the outer rect?
        let offset = vector(tile.inner_rect.min.x, tile.inner_rect.min.y);

        let mut prev: Option<SideEvent> = None;
        for evt in left.events() {
            if let Some(prev) = prev {
                let x = -1.0;
                let mut from = point(x, prev.y - offset.y);
                let mut to = point(x, evt.y - offset.y);
                if prev.winding < 0 {
                    std::mem::swap(&mut from, &mut to);
                }

                for _ in 0..prev.winding.abs() {
                    self.quad_edges.alloc().init(QuadEdge(from, point(123.0, 456.0), to, 0, 0));
                }
            }

            if evt.winding != 0 {
                prev = Some(*evt);
            } else {
                prev = None;
            }
        }

        // If the number of edges is larger than a certain threshold, we'll fall back to
        // rasterizing them on the CPU.
        if (self.quad_edges.len() - edges_start) + active_edges.len() > self.max_edges_per_gpu_tile {
            self.quad_edges.resize(edges_start, QuadEdge(point(0.0, 0.0), point(123.0, 456.0), point(0.0, 0.0), 0, 0));
            return false;
        }

        for edge in active_edges {
            if edge.ctrl.x.is_nan() {
                let mut e = QuadEdge(edge.from - offset, point(123.0, 456.0), edge.to - offset, 0, 0);
                if edge.winding < 0 {
                    std::mem::swap(&mut e.0, &mut e.2);
                }
                self.quad_edges.alloc().init(e);
            } else {
                let mut curve = QuadraticBezierSegment { from: edge.from - offset, ctrl: edge.ctrl - offset, to: edge.to - offset };
                if edge.winding < 0 {
                    std::mem::swap(&mut curve.from, &mut curve.to);
                }
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
            fill_rule: match self.fill_rule {
                FillRule::EvenOdd => 0,
                FillRule::NonZero => 1,
            }
        });

        self.mask_tiles.push(MaskedTile {
            rect: Box2D {
                min: point(tx, ty),
                max: point(tx + self.tile_size, ty + self.tile_size),
            },
            color: self.color.to_u32(),
            mask: mask_id,
        });

        true
    }

    fn add_cpu_mask(&mut self, tile: &TileInfo, active_edges: &[ActiveEdge], left: &SideEdgeTracker) {
        debug_assert!(self.tile_size <= 32.0);
        let mut accum = [0.0; 32 * 32];
        let mut backdrops = [0.0 as f32; 32];

        // "Rasterize" the left edges in the backdrop buffer.
        apply_side_edges_to_backdrop(left, tile.inner_rect.min.y, &mut backdrops);

        //left.print();
        //println!("offset {:?} : {:?}", tile.inner_rect.min.y, backdrops);

        let tile_offset = tile.inner_rect.min.to_vector();
        for edge in active_edges {
            //let edge = edge.clip_horizontally(tile.outer_rect.min.x .. tile.outer_rect.max.x);

            let from = edge.from - tile_offset;
            let to = edge.to - tile_offset;

            if edge.ctrl.x.is_nan() {
                draw_line(from, to, &mut accum);
            } else {
                let ctrl = edge.ctrl - tile_offset;
                draw_curve(from, ctrl, to, self.tolerance, &mut accum);
            }
        }

        let mask_id = self.allocate_mask_id();

        let mask_buffer_range = self.mask_uploader.new_mask(mask_id);
        unsafe {
            self.mask_uploader.current_mask_buffer.set_len(mask_buffer_range.end as usize);
        }

        let accumulate = match self.fill_rule {
            FillRule::EvenOdd => accumulate_even_odd,
            FillRule::NonZero => accumulate_non_zero,
        };

        accumulate(
            &accum,
            &backdrops,
            &mut self.mask_uploader.current_mask_buffer[mask_buffer_range.clone()],
        );

        let tx = tile.x as f32 * self.tile_size;
        let ty = tile.y as f32 * self.tile_size;
        self.mask_tiles.push(MaskedTile {
            rect: Box2D {
                min: point(tx, ty),
                max: point(tx + self.tile_size, ty + self.tile_size),
            },
            color: self.color.to_u32(),
            mask: mask_id,
        });
    }
}

impl TileEncoder for GpuRasterEncoder {
    fn encode_tile(&mut self, tile: &TileInfo, active_edges: &[ActiveEdge], left: &SideEdgeTracker) {

        let tx = tile.x as f32 * self.tile_size;
        let ty = tile.y as f32 * self.tile_size;

        if tile.solid {
            self.solid_tiles.push(SolidTile {
                rect: Box2D {
                    min: point(tx, ty),
                    max: point(tx + self.tile_size, ty + self.tile_size),
                },
                color: self.color.to_u32(),
            });
            return;
        }

        if active_edges.len() > self.max_edges_per_gpu_tile
            || (self.use_quads && !self.add_quad_gpu_mask(tile, active_edges, left))
            || (!self.use_quads && !self.add_line_gpu_mask(tile, active_edges, left)) {
            self.add_cpu_mask(tile, active_edges, left);
        }
    }
}

use std::sync::{Mutex, MutexGuard};

pub struct ParallelRasterEncoder<'l> {
    pub workers: Vec<Mutex<&'l mut dyn TileEncoder>>,
}

impl<'l> ParallelRasterEncoder<'l> {
    pub fn lock_encoder(&self) -> MutexGuard<&'l mut dyn TileEncoder> {
        let idx = rayon::current_thread_index().unwrap_or(self.workers.len() - 1);
        self.workers[idx].lock().unwrap()
    }
}

pub fn is_flat(curve: &QuadraticBezierSegment<f32>, tolerance: f32) -> bool {
    let sq_error = (curve.from.lerp(curve.to, 0.5) - curve.ctrl).square_length() * 0.25;
    sq_error <= tolerance * tolerance
}

pub fn flatten_quad(curve: &QuadraticBezierSegment<f32>, tolerance: f32, cb: &mut impl FnMut(Point)) {
    let ft = curve.from.lerp(curve.to, 0.5);
    let sq_error = (ft - curve.ctrl).square_length() * 0.25;
    let sq_tolerance = tolerance * tolerance;

    if sq_error <= sq_tolerance {
        cb(curve.to);
    } else if sq_error * 0.25 <= sq_tolerance {
        let mid = ft.lerp(curve.ctrl, 0.5);
        cb(mid);
        cb(curve.to);
    } else {
        // The baseline cost of this is fairly high and then amortized if the number of segments is high.
        // In practice the number of edges tends to be low in our case due to splitting into small tiles.
        curve.for_each_flattened(tolerance, cb);
        //unsafe { crate::flatten_simd::flatten_quad_sse(curve, tolerance, cb); }
        //crate::flatten_simd::flatten_quad_ref(curve, tolerance, cb);
    }
}
