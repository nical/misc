use crate::Color;
use crate::gpu_raster_encoder::Edge;
use lyon::path::math::Point;
use lyon::path::FillRule;
use std::mem::transmute;

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
