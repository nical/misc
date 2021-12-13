use lyon::geom::QuadraticBezierSegment;
use lyon::path::math::{Point, point, vector};
use lyon::path::FillRule;
use parasol::ExclusiveCheck;
use std::sync::{
    Arc,
    atomic::{Ordering, AtomicU32},
};

use crate::tiler::*;
use crate::cpu_rasterizer::*;
use crate::Color;
use crate::gpu::solid_tiles::TileInstance as SolidTile;
use crate::gpu::masked_tiles::TileInstance as MaskedTile;
use crate::gpu::masked_tiles::Mask as GpuMask;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Edge(pub Point, pub Point);

#[derive(Copy, Clone, Debug)]
pub struct CpuMask {
    pub mask_id: u32,
    pub byte_offset: u32,
}

struct Shared {
    next_mask_id: AtomicU32,
}

pub struct GpuRasterEncoder {
    pub edges: Vec<Edge>,
    pub solid_tiles: Vec<SolidTile>,
    pub mask_tiles: Vec<MaskedTile>,
    pub gpu_masks: Vec<GpuMask>,
    pub cpu_masks: Vec<CpuMask>,
    pub rasterized_mask_buffer: Vec<u8>,
    pub color: Color,
    pub fill_rule: FillRule,
    pub max_edges_per_gpu_tile: usize,
    pub is_opaque: bool,
    pub tolerance: f32,
    shared: Arc<Shared>,

    lock: ExclusiveCheck<()>,
}

impl GpuRasterEncoder {
    pub fn new(tolerance: f32) -> Self {
        GpuRasterEncoder {
            edges: Vec::with_capacity(8196),
            solid_tiles: Vec::with_capacity(2000),
            mask_tiles: Vec::with_capacity(6000),
            gpu_masks: Vec::with_capacity(6000),
            cpu_masks: Vec::with_capacity(6000),
            rasterized_mask_buffer: Vec::with_capacity(16*16*128),
            color: Color { r: 0, g: 0, b: 0, a: 0 },
            fill_rule: FillRule::EvenOdd,
            max_edges_per_gpu_tile: 4096,
            is_opaque: true,
            tolerance,

            shared: Arc::new(Shared {
                next_mask_id: AtomicU32::new(0),
            }),

            lock: ExclusiveCheck::new(),
        }
    }

    pub fn new_parallel(other: &GpuRasterEncoder) -> Self {
        GpuRasterEncoder {
            edges: Vec::with_capacity(8196),
            solid_tiles: Vec::with_capacity(2000),
            mask_tiles: Vec::with_capacity(1000),
            gpu_masks: Vec::with_capacity(6000),
            cpu_masks: Vec::with_capacity(6000),
            rasterized_mask_buffer: Vec::with_capacity(16*16*128),
            color: Color { r: 0, g: 0, b: 0, a: 0 },
            fill_rule: other.fill_rule,
            max_edges_per_gpu_tile: other.max_edges_per_gpu_tile,
            is_opaque: true,
            tolerance: other.tolerance,

            shared: other.shared.clone(),

            lock: ExclusiveCheck::new(),
        }
    }

    pub fn reset(&mut self) {
        self.edges.clear();
        self.solid_tiles.clear();
        self.mask_tiles.clear();
        self.gpu_masks.clear();
        self.cpu_masks.clear();
        self.rasterized_mask_buffer.clear();
        self.shared.next_mask_id.store(0, Ordering::Release);
    }

    fn allocate_mask_id(&self) -> u32 {
        self.shared.next_mask_id.fetch_add(1, Ordering::SeqCst)
    }


    fn add_gpu_mask(&mut self, tile: &TileInfo, active_edges: &[ActiveEdge], left: &SideEdgeTracker) -> bool {
        const TILE_SIZE: f32 = 16.0;
        let tx = tile.x as f32 * TILE_SIZE;
        let ty = tile.y as f32 * TILE_SIZE;

        let edges_start = self.edges.len();

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
                    self.edges.push(Edge(from, to));
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
        if (self.edges.len() - edges_start) + active_edges.len() > self.max_edges_per_gpu_tile {
            self.edges.resize(edges_start, Edge(point(0.0, 0.0), point(0.0, 0.0)));
            return false;
        }

        for edge in active_edges {
            if edge.ctrl.x.is_nan() {
                let mut e = Edge(edge.from - offset, edge.to - offset);
                if edge.winding < 0 {
                    std::mem::swap(&mut e.0, &mut e.1);
                }
                self.edges.push(e);
            } else {
                let mut curve = QuadraticBezierSegment { from: edge.from, ctrl: edge.ctrl, to: edge.to };
                if edge.winding < 0 {
                    std::mem::swap(&mut curve.from, &mut curve.to);
                }
                let mut from = curve.from;
                curve.for_each_flattened(self.tolerance, &mut |to| {
                    self.edges.push(Edge(from, to));
                    from = to;
                });
            }
        }

        let edges_start = edges_start as u32;
        let edges_end = self.edges.len() as u32;
        assert!(edges_end > edges_start);
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
                max: point(tx + TILE_SIZE, ty + TILE_SIZE),
            },
            color: self.color.to_u32(),
            mask: mask_id,
        });

        true
    }

    fn add_cpu_mask(&mut self, tile: &TileInfo, active_edges: &[ActiveEdge], left: &SideEdgeTracker) {
        const TILE_SIZE: usize = 16;
        let mut accum = [0.0; TILE_SIZE * TILE_SIZE];
        let mut backdrops = [0.0 as f32; TILE_SIZE];

        // "Rasterize" the left edges in the backdrop buffer.
        let mut prev: Option<SideEvent> = None;
        for evt in left.events() {
            if let Some(prev) = prev {
                let y0 = prev.y - tile.inner_rect.min.y;
                let y1 = evt.y - tile.inner_rect.min.y;
                let winding = prev.winding as f32;
                let y0_px = y0.max(0.0).min(15.0).floor();
                let y1_px = y1.max(0.0).min(15.0).floor();
                let first_px = y0_px as usize;
                let last_px = y1_px as usize;
                backdrops[first_px] += winding * (1.0 + y0_px - y0);
                for i in first_px + 1 .. last_px {
                    backdrops[i] += winding;
                }
                if last_px != first_px {
                    backdrops[last_px] += winding * (y1 - y1_px);
                }
            }

            if evt.winding != 0 {
                prev = Some(*evt);
            } else {
                prev = None;
            }
        }

        let tile_offset = lyon::path::math::vector(tile.outer_rect.min.x, tile.outer_rect.min.y);
        for edge in active_edges {
            let edge = edge.clip_horizontally(tile.outer_rect.min.x .. tile.outer_rect.max.x);

            let from = (edge.from - tile_offset).clamp(point(0.0, 0.0), point(16.0, 16.0));
            let to = (edge.to - tile_offset).clamp(point(0.0, 0.0), point(16.0, 16.0));

            draw_line(from, to, &mut accum, &mut backdrops);
        }

        let mask_buffer_offset = self.rasterized_mask_buffer.len();
        self.rasterized_mask_buffer.reserve(TILE_SIZE * TILE_SIZE);
        unsafe {
            // Unfortunately it's measurably faster to leave the bytes uninitialized,
            // we are going to overwrite them anyway.
            self.rasterized_mask_buffer.set_len(mask_buffer_offset + TILE_SIZE * TILE_SIZE);
        }

        accumulate_even_odd(
            &accum,
            &backdrops,
            &mut self.rasterized_mask_buffer[mask_buffer_offset .. mask_buffer_offset + TILE_SIZE * TILE_SIZE],
        );

        let mask_id = self.allocate_mask_id();

        let tx = tile.x as f32 * 16.0;
        let ty = tile.y as f32 * 16.0;
        self.mask_tiles.push(MaskedTile {
            rect: Box2D {
                min: point(tx, ty),
                max: point(tx + 16.0, ty + 16.0),
            },
            color: self.color.to_u32(),
            mask: mask_id,
        });

        self.cpu_masks.push(CpuMask {
            mask_id,
            byte_offset: mask_buffer_offset as u32,
        });
    }
}

impl TileEncoder for GpuRasterEncoder {
    fn encode_tile(&mut self, tile: &TileInfo, active_edges: &[ActiveEdge], left: &SideEdgeTracker) {

        const TILE_SIZE: f32 = 16.0;
        let tx = tile.x as f32 * TILE_SIZE;
        let ty = tile.y as f32 * TILE_SIZE;

        if tile.solid {
            self.solid_tiles.push(SolidTile {
                rect: Box2D {
                    min: point(tx, ty),
                    max: point(tx + TILE_SIZE, ty + TILE_SIZE),
                },
                color: self.color.to_u32(),
            });
            return;
        }

        if active_edges.len() > self.max_edges_per_gpu_tile
            || ! self.add_gpu_mask(tile, active_edges, left) {
            self.add_cpu_mask(tile, active_edges, left);
        }
    }

    fn begin_row(&self) {
        self.lock.begin();
    }
    fn end_row(&self) {
        self.lock.end();
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
