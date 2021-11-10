use lyon::path::math::{Point, point, vector};
use lyon::path::FillRule;
use std::mem::transmute;
use crate::z_buffer::ZBuffer;
use crate::tiler::*;
use crate::cpu_rasterizer::*;

use crate::Color;

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Pattern {
    Color(Color),
    Image(u32),
}

const OP_BACKDROP: i32 = 1;
const OP_FILL: i32 = 3;
//const OP_STROKE: i32 = 4;
const OP_MASK_EVEN_ODD: i32 = 5;
const OP_MASK_NON_ZERO: i32 = 6;
const OP_PATTERN_COLOR: i32 = 7;
//const OP_PATTERN_IMAGE: i32 = 8;
const OP_BLEND_OVER: i32 = 9;
const OP_PUSH_GROUP: i32 = 10;
const OP_POP_GROUP: i32 = 11;


#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Edge(Point, Point);

unsafe impl bytemuck::Pod for Edge {}
unsafe impl bytemuck::Zeroable for Edge {}

pub struct TileCommandEncoder {
    commands: Vec<i32>,
    edges: Vec<Edge>,
}

impl TileCommandEncoder {
    pub fn new() -> Self {
        TileCommandEncoder {
            commands: Vec::new(),
            edges: Vec::new(),
        }
    }

    pub fn set_backdrop(&mut self, winding_number: i32) {
        self.commands.push(OP_BACKDROP);
        self.commands.push(winding_number);
    }

    pub fn mask_even_odd(&mut self) {
        self.commands.push(OP_MASK_EVEN_ODD);
    }

    pub fn mask_non_zero(&mut self) {
        self.commands.push(OP_MASK_NON_ZERO);
    }

    pub fn blend_over(&mut self) {
        self.commands.push(OP_BLEND_OVER);
    }

    pub fn set_color(&mut self, color: Color) {
        let packed = (color.r as u32) << 24
            | (color.g as u32) << 16
            | (color.b as u32) << 8
            | (color.a as u32);
        self.commands.push(OP_PATTERN_COLOR);
        self.commands.push(unsafe { transmute(packed) });
    }

    pub fn fill(&mut self) -> FillBuilder {
        FillBuilder::new(self)
    }

    pub fn push_group(&mut self) -> Group {
        Group::new(self)
    }

    pub fn commands(&self) -> &[i32] {
        &self.commands
    }

    pub fn edges(&self) -> &[Edge] {
        &self.edges
    }
}

pub enum BlendMode {
    Over,
}

pub struct FillBuilder<'l> {
    encoder: &'l mut TileCommandEncoder,
    color: Option<Color>,
    fill_rule: FillRule,
    blend_mode: BlendMode,
    edges_start: i32,
    backdrop: i32,
}

impl<'l> FillBuilder<'l> {
    pub fn new(encoder: &'l mut TileCommandEncoder) -> Self {
        FillBuilder {
            edges_start: encoder.edges.len() as i32,
            encoder,
            color: None,
            fill_rule: FillRule::EvenOdd,
            blend_mode: BlendMode::Over,
            backdrop: 0,
        }
    }

    pub fn with_color(mut self, color: Color) -> Self {
        self.color = Some(color);
        self
    }

    pub fn with_bend_mode(mut self, blend_mode: BlendMode) -> Self {
        self.blend_mode = blend_mode;
        self
    }

    pub fn with_backdrop(mut self, winding_number: i32) -> Self {
        self.backdrop = winding_number;
        self
    }

    pub fn add_edge(&mut self, from: Point, to: Point) {
        self.encoder.edges.push(Edge(from, to));
    }

    pub fn build(self) {
        let edges_end = self.encoder.edges.len() as i32;
        if edges_end == self.edges_start {
            return;
        }

        self.encoder.set_backdrop(self.backdrop);

        self.encoder.commands.push(OP_FILL);
        self.encoder.commands.push(self.edges_start);
        self.encoder.commands.push(edges_end);

        match self.fill_rule {
            FillRule::EvenOdd => { self.encoder.mask_even_odd(); }
            FillRule::NonZero => { self.encoder.mask_non_zero(); }
        }

        if let Some(color) = self.color {
            self.encoder.set_color(color);
        }

        match self.blend_mode {
            BlendMode::Over => { self.encoder.blend_over(); }
        }
    }
}

pub struct Group<'l> {
    encoder: &'l mut TileCommandEncoder,    
}

impl<'l> Group<'l> {
    fn new(encoder: &'l mut TileCommandEncoder) -> Group<'l> {
        encoder.commands.push(OP_PUSH_GROUP);
        Group { encoder }
    }

    pub fn pop(self) {
        self.encoder.commands.push(OP_POP_GROUP);
    }

    pub fn pop_with_mask(self) -> FillBuilder<'l> {
        self.encoder.commands.push(OP_POP_GROUP);
        FillBuilder::new(self.encoder)
    }

    pub fn push_group(&mut self) -> Group {
        Group::new(self.encoder)
    }

    pub fn fill(&mut self) -> FillBuilder {
        FillBuilder::new(self.encoder)
    }
}

use crate::gpu::solid_tiles::TileInstance as SolidTile;
use crate::gpu::masked_tiles::TileInstance as MaskedTile;
use crate::gpu::masked_tiles::Mask as GpuMask;

pub struct CpuMask {
    pub mask_id: u32,
    pub byte_offset: u32,
}

pub struct GpuRasterEncoder<'l> {
    pub edges: Vec<Edge>,
    pub solid_tiles: Vec<SolidTile>,
    pub mask_tiles: Vec<MaskedTile>,
    pub gpu_masks: Vec<GpuMask>,
    pub cpu_masks: Vec<CpuMask>,
    pub rasterized_mask_buffer: Vec<u8>,
    pub z_buffer: &'l mut ZBuffer,
    pub z_index: u16,
    pub path_id: u16,
    pub color: Color,
    pub fill_rule: FillRule,
    pub max_edges_per_gpu_tile: usize,
    pub next_mask_id: u32,
}

impl<'l> GpuRasterEncoder<'l> {
    pub fn new(z_buffer: &'l mut ZBuffer) -> Self {
        GpuRasterEncoder {
            edges: Vec::with_capacity(8196),
            solid_tiles: Vec::with_capacity(2000),
            mask_tiles: Vec::with_capacity(5000),
            gpu_masks: Vec::with_capacity(5000),
            cpu_masks: Vec::with_capacity(5000),
            rasterized_mask_buffer: Vec::with_capacity(16*16*128),
            z_buffer,
            z_index: 0,
            path_id: 0,
            color: Color { r: 0, g: 0, b: 0, a: 0 },
            fill_rule: FillRule::EvenOdd,
            max_edges_per_gpu_tile: 4096,
            next_mask_id: 0,
        }
    }

    pub fn reset(&mut self) {
        self.edges.clear();
        self.solid_tiles.clear();
        self.mask_tiles.clear();
        self.gpu_masks.clear();
        self.cpu_masks.clear();
        self.rasterized_mask_buffer.clear();
        self.z_index = 0;
        self.next_mask_id = 0;
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
            let mut e = Edge(edge.from - offset, edge.to - offset);
            if edge.winding < 0 {
                std::mem::swap(&mut e.0, &mut e.1);
            }
            self.edges.push(e);
        }

        let edges_start = edges_start as u32;
        let edges_end = self.edges.len() as u32;
        assert!(edges_end > edges_start);
        assert!(edges_end - edges_start < 100);
        let mask_id = self.next_mask_id;
        self.next_mask_id += 1;

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

        let mask_id = self.next_mask_id;
        self.next_mask_id += 1;

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

impl<'l> TileEncoder for GpuRasterEncoder<'l> {
    fn encode_tile(&mut self, tile: &TileInfo, active_edges: &[ActiveEdge], left: &SideEdgeTracker) {

        const TILE_SIZE: f32 = 16.0;
        let tx = tile.x as f32 * TILE_SIZE;
        let ty = tile.y as f32 * TILE_SIZE;

        let mut solid = false;
        if active_edges.is_empty() {
            if left.is_in(ty - 0.4, ty + 16.4, self.fill_rule) {
                solid = true;
            } else if left.is_empty() {
                // Empty tile.
                return;
            }
        }

        if !self.z_buffer.test(tile.x, tile.y, self.z_index, solid) {
            // Culled by a solid tile.
            return;
        }

        if solid {
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
}
