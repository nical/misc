use std::sync::Arc;
use lyon::math::{Point, vector};
use lyon::path::{Path, FillRule};
use lyon::geom::euclid::default::{Transform2D, Box2D};
use crate::pattern::checkerboard::*;
use crate::pattern::simple_gradient::*;
use crate::pattern::solid_color::{SolidColorBuilder, SolidColor};
use crate::gpu::GpuStore;
use crate::Color;
use crate::tiling::mask::circle::CircleMaskEncoder;
use crate::tiling::mask::rect::RectangleMaskEncoder;
use crate::tiling::{
    TilerPattern, FillOptions,
    occlusion::TileMask,
    tiler::{Tiler, TilerConfig},
    encoder::TileEncoder,
};


/*

 1--A-----------4--C-----6
    |              |
    +--2--B        +--5
          |
          +--3
.

tile order: 6 C* 5 C 4 A* B* 3 B 2 A 1

renderer order: 3, 2 B, 5, 1 A 4 C 6

for each render pass, first render associated push group(s).
if multiple groups don't fit into an atlas, create a new render pass.

*/

pub trait Shape {
    fn to_command(self) -> RecordedShape;
}

pub trait Pattern {
    fn to_command(self) -> RecordedPattern;
}

pub struct PathShape {
    pub path: Arc<Path>,
    pub fill_rule: FillRule,
    pub inverted: bool,
}

impl PathShape {
    pub fn new(path: Arc<Path>) -> Self {
        PathShape { path, fill_rule: FillRule::EvenOdd, inverted: false }
    }
    pub fn with_fill_rule(mut self, fill_rule: FillRule) -> Self {
        self.fill_rule = fill_rule;
        self
    }
    pub fn inverted(mut self) -> Self {
        self.inverted = !self.inverted;
        self
    }
}

impl Shape for PathShape {
    fn to_command(self) -> RecordedShape {
        RecordedShape::Path(self)
    }
}

impl Shape for Arc<Path> {
    fn to_command(self) -> RecordedShape {
        RecordedShape::Path(PathShape::new(self))
    }
}

pub struct Circle {
    pub center: Point,
    pub radius: f32,
    pub inverted: bool,
}

impl Circle {
    pub fn new(center: Point, radius: f32) -> Self {
        Circle { center, radius, inverted: false }
    }

    pub fn inverted(mut self) -> Self {
        self.inverted = !self.inverted;
        self
    }
}

impl Shape for Circle {
    fn to_command(self) -> RecordedShape {
        RecordedShape::Circle(self)
    }
}

impl Shape for Box2D<f32> {
    fn to_command(self) -> RecordedShape {
        RecordedShape::Rect(self)
    }
}

impl Shape for Box2D<i32> {
    fn to_command(self) -> RecordedShape {
        RecordedShape::Rect(self.to_f32())
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct All;

impl Shape for All {
    fn to_command(self) -> RecordedShape {
        RecordedShape::Canvas
    }
}

impl Pattern for Color {
    fn to_command(self) -> RecordedPattern {
        RecordedPattern::Color(self)
    }
}

impl Pattern for Gradient {
    fn to_command(self) -> RecordedPattern {
        RecordedPattern::Gradient(self)
    }
}

impl Pattern for Checkerboard {
    fn to_command(self) -> RecordedPattern {
        RecordedPattern::Checkerboard(self)
    }
}

pub enum RecordedPattern {
    Color(Color),
    Gradient(Gradient),
    Image(u32),
    Checkerboard(Checkerboard),
}

pub enum RecordedShape {
    Path(PathShape),
    Rect(Box2D<f32>),
    Circle(Circle),
    Canvas,
}

pub struct Fill {
    shape: RecordedShape,
    pattern: RecordedPattern,
    transform: usize,
}

pub struct Group {
    clip: Option<Arc<Path>>,
    // A clip rectangle in screen coordinates.
    clip_rect: Box2D<f32>,
    commands: Vec<Command>,
    transform: usize,
    z_index: u16,
    group_stack_depth: u16,
}

impl Default for Group {
    fn default() -> Self {
        Group {
            clip: None,
            clip_rect: Box2D::zero(),
            commands: Vec::new(),
            z_index: 0,
            transform: 0,
            group_stack_depth: 0,
        }
    }
}

pub struct Transform {
    transform: Transform2D<f32>,
    parent: Option<usize>,
}

pub enum Command {
    Fill(Fill),
    Group(Group),
}

pub struct Commands {
    root: Group,
    transforms: Vec<Transform>,
    max_group_stack_depth: u16,
}

impl Commands {
    pub fn set_transform(&mut self, idx: usize, transform: &Transform2D<f32>) {
        self.transforms[idx].transform = *transform;
    }
}

pub struct Canvas {
    transforms: Vec<Transform>,
    z_index: u16,
    current_transform: usize,
    current_group: Group,
    group_stack: Vec<Group>,
    max_group_stack_depth: u16,
}


impl Canvas {
    pub fn new(size: Box2D<f32>) -> Self {
        Canvas {
            transforms: vec![
                Transform {
                    transform: Transform2D::identity(),
                    parent: None,
                },
            ],
            z_index: 1,
            current_transform: 0,
            current_group: Group {
                commands: Vec::new(),
                clip: None,
                clip_rect: size,
                z_index: 0,
                transform: 0,
                group_stack_depth: 0,
            },
            group_stack: Vec::new(),
            max_group_stack_depth: 1,
        }
    }

    pub fn push_transform(&mut self, transform: &Transform2D<f32>) {
        let id = self.transforms.len();
        if self.current_transform == 0 {
            self.transforms.push(Transform {
                transform: *transform,
                parent: Some(self.current_transform),
            });
        } else {
            let transform = self.transforms[self.current_transform].transform.then(transform);
            self.transforms.push(Transform {
                transform,
                parent: Some(self.current_transform),
            });
        }

        self.current_transform = id;
    }

    pub fn pop_transform(&mut self) {
        assert!(self.current_transform != 0);
        self.current_transform = self.transforms[self.current_transform].parent.unwrap_or(0);
    }

    pub fn fill<S: Shape, P: Pattern>(&mut self, shape: S, pattern: P) {
        self.current_group.commands.push(Command::Fill(Fill {
            shape: shape.to_command(),
            pattern: pattern.to_command(),
            transform: self.current_transform,
        }));
        self.z_index += 1;
    }

    pub fn push_clip(&mut self, path: Arc<Path>) {
        let depth = self.current_group.group_stack_depth + 1;
        let clip_rect = self.current_group.clip_rect;
        self.group_stack.push(std::mem::take(&mut self.current_group));
        self.current_group.clip_rect = clip_rect;
        self.current_group.clip = Some(path);
        self.current_group.z_index = self.z_index;
        self.current_group.transform = self.current_transform;
        self.current_group.group_stack_depth = depth;
        self.max_group_stack_depth = self.max_group_stack_depth.max(depth);
    }

    pub fn pop_clip(&mut self) {
        let mut group = self.group_stack.pop().unwrap();
        std::mem::swap(&mut group, &mut self.current_group);
        assert!(group.clip.is_some());
        self.current_group.commands.push(Command::Group(group));
    }

    pub fn finish(&mut self) -> Commands {
        assert!(self.group_stack.is_empty());

        Commands {
            root: std::mem::take(&mut self.current_group),
            transforms: std::mem::take(&mut self.transforms),
            max_group_stack_depth: self.max_group_stack_depth,
        }
    }
}

pub struct TargetData {
    pub tile_encoder: TileEncoder,
    pub circle_masks: CircleMaskEncoder,
    pub rectangle_masks: RectangleMaskEncoder,
    pub solid_color_pattern: SolidColorBuilder,
    pub checkerboard_pattern: CheckerboardPatternBuilder,
    pub gradient_pattern: SimpleGradientBuilder,
}

pub struct FrameBuilder {
    pub targets: Vec<TargetData>,
    pub tiler: Tiler,
    pub tile_mask: TileMask,
    stats: FrameBuilderStats,
    tolerance: f32,
}

impl FrameBuilder {
    pub fn new(config: &TilerConfig) -> Self {
        let size = config.view_box.size().to_u32();
        let tile_size = config.tile_size as u32;
        let tiles_x = (size.width + tile_size - 1) / tile_size;
        let tiles_y = (size.height + tile_size - 1) / tile_size;
        FrameBuilder {
            targets: vec![TargetData {
                tile_encoder: TileEncoder::new(config, 3),
                circle_masks: CircleMaskEncoder::new(),
                rectangle_masks: RectangleMaskEncoder::new(),
                solid_color_pattern: SolidColorBuilder::new(SolidColor::new(Color::BLACK), 0),
                gradient_pattern: SimpleGradientBuilder::new(SimpleGradient::new(), 1),
                checkerboard_pattern: CheckerboardPatternBuilder::new(CheckerboardPattern::new(), 2),
            }],
            tiler: Tiler::new(config),
            tile_mask: TileMask::new(tiles_x, tiles_y),
            stats: FrameBuilderStats::new(),
            tolerance: config.tolerance,
        }
    }

    pub fn build(&mut self, commands: &Commands, gpu_store: &mut GpuStore, device: &wgpu::Device) {
        let t0 = time::precise_time_ns();

        self.stats = FrameBuilderStats::new();

        let group_stack_depth = commands.max_group_stack_depth as usize;
        while self.targets.len() < group_stack_depth {
            let tile_encoder = self.targets[0].tile_encoder.create_similar();
            self.targets.push(TargetData {
                tile_encoder,
                circle_masks: CircleMaskEncoder::new(),
                rectangle_masks: RectangleMaskEncoder::new(),
                solid_color_pattern: SolidColorBuilder::new(SolidColor::new(Color::BLACK), 1),
                checkerboard_pattern: CheckerboardPatternBuilder::new(CheckerboardPattern::new(), 2),
                gradient_pattern: SimpleGradientBuilder::new(SimpleGradient::new(), 3),
            });
        }

        for target in &mut self.targets[0..group_stack_depth] {
            target.tile_encoder.reset();
            target.circle_masks.reset();
            target.rectangle_masks.reset();
        }
        self.tiler.edges.clear();
        self.tile_mask.clear();

        self.process_group(&commands.root, &commands, gpu_store, device);

        for target in &mut self.targets[0..group_stack_depth] {
            target.tile_encoder.end_paths();
            target.circle_masks.end_render_pass();
            target.rectangle_masks.end_render_pass();
            target.tile_encoder.reverse_alpha_tiles();
        }

        self.stats.total_time = Duration::from_ns(time::precise_time_ns() - t0);
    }

    pub fn process_group(&mut self, group: &Group, commands: &Commands, gpu_store: &mut GpuStore, device: &wgpu::Device) {
        let mut saved_tile_mask = None;
        if let Some(_clip) = &group.clip {
            saved_tile_mask = Some(self.tile_mask.clone());
        }

        for cmd in group.commands.iter().rev() {
            match cmd {
                Command::Fill(fill) => {
                    let target = &mut self.targets[group.group_stack_depth as usize];
                    let encoder = &mut target.tile_encoder;

                    let transform = if fill.transform != 0 {
                        Some(&commands.transforms[fill.transform].transform)
                    } else {
                        None
                    };

                    let mut prerender = false;
                    let pattern: &mut dyn TilerPattern = match fill.pattern {
                        RecordedPattern::Color(color) => {
                            target.solid_color_pattern.set(SolidColor::new(color));
                            &mut target.solid_color_pattern
                        },
                        RecordedPattern::Checkerboard(checkerboard) => {
                            prerender = true;
                            let mut checkerboard = checkerboard;
                            if let Some(transform) = transform {
                                checkerboard.offset = transform.transform_point(checkerboard.offset);
                                checkerboard.scale = transform.transform_vector(vector(0.0, checkerboard.scale)).y
                            }
                            target.checkerboard_pattern.set(add_checkerboard(gpu_store, &checkerboard));
                            &mut target.checkerboard_pattern
                        }
                        RecordedPattern::Gradient(gradient) => {
                            let mut gradient = gradient;
                            if let Some(transform) = transform {
                                gradient.from = transform.transform_point(gradient.from);
                                gradient.to = transform.transform_point(gradient.to);
                            }

                            target.gradient_pattern.set(gradient.write_gpu_data(gpu_store));
                            &mut target.gradient_pattern
                        }
                        _ => { unimplemented!() }
                    };

                    match &fill.shape {
                        RecordedShape::Path(shape) => {
                            let options = FillOptions::new()
                                .with_transform(transform)
                                .with_fill_rule(shape.fill_rule)
                                .with_prerendered_pattern(prerender)
                                .with_tolerance(self.tolerance)
                                .with_inverted(shape.inverted);
                            self.tiler.fill_path(shape.path.iter(), &options, pattern, &mut self.tile_mask, encoder, device);
                        }
                        RecordedShape::Circle(circle) => {
                            let options = FillOptions::new()
                                .with_transform(transform)
                                .with_prerendered_pattern(prerender)
                                .with_tolerance(self.tolerance)
                                .with_inverted(circle.inverted);
                                crate::tiling::mask::circle::fill_circle(
                                    circle.center,
                                    circle.radius,
                                    &options,
                                    pattern,
                                    &mut self.tile_mask,
                                    &mut self.tiler,
                                    encoder,
                                    &mut target.circle_masks,
                                    device,
                                )
                        }
                        RecordedShape::Rect(rect) => {
                            let options = FillOptions::new()
                                .with_transform(transform)
                                .with_prerendered_pattern(prerender)
                                .with_tolerance(self.tolerance);
                            crate::tiling::mask::rect::fill_rect(
                                rect,
                                &options,
                                pattern,
                                &mut self.tile_mask,
                                &mut self.tiler,
                                encoder,
                                &mut target.rectangle_masks,
                                device,
                            )
                        }
                        RecordedShape::Canvas => {
                            self.tiler.fill_canvas(pattern, &mut self.tile_mask, encoder);
                        }
                    }

                    self.stats.row_time += Duration::from_ns(self.tiler.row_decomposition_time_ns);
                    self.stats.tile_time += Duration::from_ns(self.tiler.tile_decomposition_time_ns);
                    self.stats.tiled_paths += 1;
                }
                Command::Group(group) => {
                    self.process_group(group, commands, gpu_store, device);
                }
            }
        }

        if let Some(tile_mask) = saved_tile_mask {
            self.tile_mask = tile_mask;
        }
    }

    pub fn stats(&self) -> &FrameBuilderStats {
        &self.stats
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Duration(u64);

impl Duration {
    pub fn from_ns(ns: u64) -> Self {
        Duration(ns)
    }

    pub fn zero() -> Self {
        Duration(0)
    }

    pub fn ms(self) -> f64 {
        self.ns() as f64 / 1000000.0
    }

    pub fn ns(self) -> u64 {
        self.0
    }
}

use std::fmt;
impl fmt::Debug for Duration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "{:.3}ms", self.ms())
    }
}

impl std::ops::Add<Duration> for Duration {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Duration(self.0 + other.0)
    }
} 

impl std::ops::AddAssign<Duration> for Duration {
    fn add_assign(&mut self, other: Self) {
        self.0 += other.0;
    }
}

#[derive(Copy, Clone, Debug)]
pub struct FrameBuilderStats {
    pub tile_time: Duration,
    pub row_time: Duration,
    pub total_time: Duration,
    pub tiled_paths: u32,
}

impl FrameBuilderStats {
    pub fn new() -> Self {
        FrameBuilderStats {
            tile_time: Duration::zero(),
            row_time: Duration::zero(),
            total_time: Duration::zero(),
            tiled_paths: 0,
        }
    }
}
