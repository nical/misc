use std::sync::Arc;
use lyon::math::{Point, point, vector};
use lyon::path::{Path, FillRule};
use lyon::geom::euclid::default::{Transform2D, Box2D};
use crate::pattern::checkerboard::{CheckerboardPatternBuilder, CheckerboardPattern, add_checkerboard};
use crate::pattern::simple_gradient::{SimpleGradientBuilder, SimpleGradient, add_gradient};
use crate::pattern::solid_color::{SolidColorBuilder, SolidColor};
use crate::gpu::GpuStore;
use crate::{Color};
use crate::tiling::{
    TilerPattern, FillOptions,
    occlusion::TileMask,
    tiler::{Tiler, TilerConfig, TileEncoder}
};
use crate::gpu::mask_uploader::MaskUploader;


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

pub enum Pattern {
    Color(Color),
    Gradient { p0: Point, color0: Color, p1: Point, color1: Color },
    Image(u32),
    Checkerboard { colors: [Color; 2], scale: f32 },
}

pub enum Shape {
    Path(Arc<Path>, FillRule),
    Rect(Box2D<f32>),
    Circle { center: Point, radius: f32 },
    Canvas,
}

pub struct Fill {
    shape: Shape,
    pattern: Pattern,
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

    pub fn fill(&mut self, path: Arc<Path>, fill_rule: FillRule, pattern: Pattern) {
        self.current_group.commands.push(Command::Fill(Fill {
            shape: Shape::Path(path, fill_rule),
            pattern,
            transform: self.current_transform,
        }));
        self.z_index += 1;
    }

    pub fn fill_rect(&mut self, rect: Box2D<f32>, pattern: Pattern) {
        self.current_group.commands.push(Command::Fill(Fill {
            shape: Shape::Rect(rect),
            pattern,
            transform: self.current_transform,
        }));
    }

    pub fn fill_circle(&mut self, center: Point, radius: f32, pattern: Pattern) {
        self.current_group.commands.push(Command::Fill(Fill {
            shape: Shape::Circle { center, radius },
            pattern,
            transform: self.current_transform,
        }));
    }

    pub fn fill_canvas(&mut self, pattern: Pattern) {
        self.current_group.commands.push(Command::Fill(Fill {
            shape: Shape::Canvas,
            pattern,
            transform: self.current_transform,
        }));
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
    pub fn new(config: &TilerConfig, uploader: MaskUploader) -> Self {
        let size = config.view_box.size().to_u32();
        let tile_size = config.tile_size as u32;
        let tiles_x = (size.width + tile_size - 1) / tile_size;
        let tiles_y = (size.height + tile_size - 1) / tile_size;
        FrameBuilder {
            targets: vec![TargetData {
                tile_encoder: TileEncoder::new(config, uploader, 3),
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

    pub fn build(&mut self, commands: &Commands, gpu_store: &mut GpuStore) {
        let t0 = time::precise_time_ns();

        self.stats = FrameBuilderStats::new();

        let group_stack_depth = commands.max_group_stack_depth as usize;
        while self.targets.len() < group_stack_depth {
            let tile_encoder = self.targets[0].tile_encoder.create_similar();
            self.targets.push(TargetData {
                tile_encoder,
                solid_color_pattern: SolidColorBuilder::new(SolidColor::new(Color::BLACK), 1),
                checkerboard_pattern: CheckerboardPatternBuilder::new(CheckerboardPattern::new(), 2),
                gradient_pattern: SimpleGradientBuilder::new(SimpleGradient::new(), 3),
            });
        }

        for target in &mut self.targets[0..group_stack_depth] {
            target.tile_encoder.reset();
        }
        self.tiler.edges.clear();
        self.tile_mask.clear();

        self.process_group(&commands.root, &commands, gpu_store);

        for target in &mut self.targets[0..group_stack_depth] {
            target.tile_encoder.end_paths();
            target.tile_encoder.reverse_alpha_tiles();
        }

        self.stats.total_time = Duration::from_ns(time::precise_time_ns() - t0);
    }

    pub fn process_group(&mut self, group: &Group, commands: &Commands, gpu_store: &mut GpuStore) {
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
                        Pattern::Color(color) => {
                            target.solid_color_pattern.set(SolidColor::new(color));
                            &mut target.solid_color_pattern
                        },
                        Pattern::Checkerboard { colors, scale } => {
                            prerender = true;
                            let mut scale = scale;
                            let mut pos = point(0.0, 0.0);
                            if let Some(transform) = transform {
                                pos = transform.transform_point(pos);
                                scale = transform.transform_vector(vector(0.0, scale)).y
                            }
                            let checkerboard = add_checkerboard(gpu_store, colors[0], colors[1], pos, scale);
                            target.checkerboard_pattern.set(checkerboard);
                            &mut target.checkerboard_pattern
                        }
                        Pattern::Gradient { p0, color0, p1, color1 } => {
                            let mut p0 = p0;
                            let mut p1 = p1;
                            if let Some(transform) = transform {
                                p0 = transform.transform_point(p0);
                                p1 = transform.transform_point(p1);
                            }
                            let gradient = add_gradient(
                                gpu_store,
                                p0, color0,
                                p1, color1,
                            );

                            target.gradient_pattern.set(gradient);
                            &mut target.gradient_pattern
                        }
                        _ => { unimplemented!() }
                    };

                    match &fill.shape {
                        Shape::Path(path, fill_rule) => {
                            let options = FillOptions::new()
                                .with_transform(transform)
                                .with_fill_rule(*fill_rule)
                                .with_prerendered_pattern(prerender)
                                .with_tolerance(self.tolerance);
                            self.tiler.fill_path(path.iter(), &options, pattern, &mut self.tile_mask, None, encoder);
                        }
                        Shape::Circle { center, radius } => {
                            let options = FillOptions::new()
                                .with_transform(transform)
                                .with_prerendered_pattern(prerender)
                                .with_tolerance(self.tolerance);
                            self.tiler.fill_circle(*center, *radius, &options, pattern, &mut self.tile_mask, None, encoder)
                        }
                        Shape::Rect(rect) => {
                            let options = FillOptions::new()
                                .with_transform(transform)
                                .with_prerendered_pattern(prerender)
                                .with_tolerance(self.tolerance);
                            self.tiler.fill_rect(rect, &options, pattern, &mut self.tile_mask, None, encoder)
                        }
                        Shape::Canvas => {
                            self.tiler.fill_canvas(pattern, &mut self.tile_mask, None, encoder);
                        }
                    }

                    self.stats.row_time += Duration::from_ns(self.tiler.row_decomposition_time_ns);
                    self.stats.tile_time += Duration::from_ns(self.tiler.tile_decomposition_time_ns);
                    self.stats.tiled_paths += 1;
                }
                Command::Group(group) => {
                    self.process_group(group, commands, gpu_store);
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
