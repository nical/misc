use lyon::path::math::{Point, point, vector};
use lyon::path::FillRule;
use std::mem::transmute;
use std::ops::Range;
use crate::z_buffer::ZBuffer;
use crate::tiler::*;

use crate::Color;

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Pattern {
    Color(Color),
    Image(u32),
}

const OP_BACKDROP: i32 = 1;
const OP_FILL: i32 = 3;
const OP_STROKE: i32 = 4;
const OP_MASK_EVEN_ODD: i32 = 5;
const OP_MASK_NON_ZERO: i32 = 6;
const OP_PATTERN_COLOR: i32 = 7;
const OP_PATTERN_IMAGE: i32 = 8;
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
use crate::gpu::masked_tiles::Mask;

/// Encodes tiles of two kinds:
///  - solid tiles.
///  - mask tiles (edges rasterized on the GPU).
pub struct GpuRasterEncoder<'l> {
    pub edges: Vec<Edge>,
    pub solid_tiles: Vec<SolidTile>,
    pub mask_tiles: Vec<MaskedTile>,
    pub masks: Vec<Mask>,
    pub z_buffer: &'l mut ZBuffer,
    pub z_index: u16,
    pub path_id: u16,
    pub color: Color,
}

impl<'l> GpuRasterEncoder<'l> {
    pub fn new(z_buffer: &'l mut ZBuffer) -> Self {
        GpuRasterEncoder {
            edges: Vec::with_capacity(8196),
            solid_tiles: Vec::with_capacity(2000),
            mask_tiles: Vec::with_capacity(5000),
            masks: Vec::with_capacity(5000),
            z_buffer,
            z_index: 0,
            path_id: 0,
            color: Color { r: 0, g: 0, b: 0, a: 0 },
        }
    }

    pub fn reset(&mut self) {
        self.edges.clear();
        self.solid_tiles.clear();
        self.mask_tiles.clear();
        self.z_index = 0;
    }
}

impl<'l> TileEncoder for GpuRasterEncoder<'l> {
    fn encode_tile(&mut self, tile: &TileInfo, active_edges: &[ActiveEdge], left: &SideEdgeTracker) {

        const TILE_SIZE: f32 = 16.0;
        let tx = tile.x as f32 * TILE_SIZE;
        let ty = tile.y as f32 * TILE_SIZE;

        let mut solid = false;
        if active_edges.is_empty() {
            if left.is_in(ty - 0.4, ty + 16.4, FillRule::EvenOdd) {
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

        let edges_start = self.edges.len() as u32;

        let offset = vector(tile.outer_rect.min.x, tile.outer_rect.min.y);
        for edge in active_edges {
            let mut e = Edge(edge.from - offset, edge.to - offset);
            if edge.winding < 0 {
                std::mem::swap(&mut e.0, &mut e.1);
            }
            self.edges.push(e);
        }

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

        const MASKS_PER_ROW: u32 = 2048 / 16; 
        let edges_end = self.edges.len() as u32;
        assert!(edges_end > edges_start);
        assert!(edges_end - edges_start < 100);
        let mask = self.masks.len() as u32;

        //println!(" - mask tile {:?} {:?} {:?}",
        //    active_edges.len(), left.events(),
        //    &self.edges[edges_start as usize .. (edges_end as usize)]
        //);

        self.masks.push(Mask {
            edges: (edges_start, edges_end),
            mask_id: mask,
            //backdrop: tile.backdrop_winding as f32,
            backdrop: 0.0,
        });

        self.mask_tiles.push(MaskedTile {
            rect: Box2D {
                min: point(tx, ty),
                max: point(tx + TILE_SIZE, ty + TILE_SIZE),
            },
            color: self.color.to_u32(),
            mask,
        });
    }
}
