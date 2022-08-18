use std::sync::Arc;
use lyon::path::Path;
use lyon::geom::euclid::default::{Transform2D, Box2D};
use crate::checkerboard_pattern::CheckerboardPatternBuilder;
use crate::{Color, SolidColorPattern, TileIdAllcator};
use crate::occlusion::TileMask;

pub enum Pattern {
    Color(Color),
    Image(u32),
    Checkerboard { colors: [Color; 2], scale: f32 },
}

pub struct Fill {
    path: Arc<Path>,
    pattern: Pattern,
    transform: usize,
    z_index: u16,
}

pub struct Group {
    clip: Option<Arc<Path>>,
    // A clip rectangle in screen coordinates.
    clip_rect: Box2D<f32>,
    commands: Vec<Command>,
    transform: usize,
    z_index: u16,
    group_stack_depth: u16,
    can_tile_output: bool,
    needs_surface: bool,
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
            can_tile_output: true,
            needs_surface: false,
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
                can_tile_output: false,
                needs_surface: true,
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

    pub fn fill(&mut self, path: Arc<Path>, pattern: Pattern) {
        self.current_group.commands.push(Command::Fill(Fill {
            path,
            pattern,
            transform: self.current_transform,
            z_index: self.z_index,
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
        self.z_index += 1;
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

use crate::tile_encoder::TileEncoder;
use crate::tiler::{Tiler, TilerConfig, TilerPattern};
use crate::gpu::mask_uploader::MaskUploader;

pub struct TargetData {
    pub src_color_tile_ids: TileIdAllcator,
    pub tile_encoder: TileEncoder,
    pub solid_color_pattern: SolidColorPattern,
    pub checkerboard_pattern: CheckerboardPatternBuilder,
}

pub struct FrameBuilder {
    pub targets: Vec<TargetData>,
    pub tiler: Tiler,
    pub tile_mask: TileMask,
    stats: FrameBuilderStats,
}

impl FrameBuilder {
    pub fn new(config: &TilerConfig, uploader: MaskUploader) -> Self {
        let size = config.view_box.size().to_u32();
        let tile_size = config.tile_size.to_u32();
        let tiles_x = (size.width + tile_size.width - 1) / tile_size.width;
        let tiles_y = (size.height + tile_size.height - 1) / tile_size.height;
        let color_tile_ids = TileIdAllcator::new();
        let mask_ids = TileIdAllcator::new();
        FrameBuilder {
            targets: vec![TargetData {
                src_color_tile_ids: color_tile_ids.clone(),
                tile_encoder: TileEncoder::new(config, uploader, mask_ids),
                solid_color_pattern: SolidColorPattern::new(Color::BLACK),
                checkerboard_pattern: CheckerboardPatternBuilder::new(
                    Color::WHITE, Color::BLACK,
                    1.0,
                    color_tile_ids.clone(),
                    config.tile_size.width as f32
                ),
            }],
            tiler: Tiler::new(config, color_tile_ids),
            tile_mask: TileMask::new(tiles_x, tiles_y),
            stats: FrameBuilderStats::new(),
        }
    }

    pub fn build(&mut self, commands: &Commands) {
        let t0 = time::precise_time_ns();

        self.stats = FrameBuilderStats::new();

        let group_stack_depth = commands.max_group_stack_depth as usize;
        while self.targets.len() < group_stack_depth {
            let tile_encoder = self.targets[0].tile_encoder.create_similar();
            let color_tile_allocator = TileIdAllcator::new();
            self.targets.push(TargetData {
                src_color_tile_ids: color_tile_allocator.clone(),
                tile_encoder,
                solid_color_pattern: SolidColorPattern::new(Color::BLACK),
                checkerboard_pattern: CheckerboardPatternBuilder::new(
                    Color::WHITE, Color::BLACK,
                    1.0,
                    color_tile_allocator,
                    16.0,
                ),
            });
        }

        for target in &mut self.targets[0..group_stack_depth] {
            target.tile_encoder.reset();
            target.checkerboard_pattern.reset();
            target.src_color_tile_ids.reset();
        }
        self.tile_mask.clear();

        self.process_group(&commands.root, &commands);

        for target in &mut self.targets[0..group_stack_depth] {
            target.tile_encoder.end_paths();
            target.tile_encoder.reverse_alpha_tiles();
        }

        self.stats.total_time = Duration::from_ns(time::precise_time_ns() - t0);
    }

    pub fn process_group(&mut self, group: &Group, commands: &Commands) {
        if let Some(clip) = &group.clip {
            // TODO: save clip state.

            // TODO: support clipping the root.
            let parent_group_depth = group.group_stack_depth as usize - 1;
            let target = &mut self.targets[parent_group_depth];
            let encoder = &mut target.tile_encoder;

            let transform = if group.transform != 0 {
                Some(&commands.transforms[group.transform].transform)
            } else {
                None
            };

            self.tiler.draw.is_opaque = false;
            self.tiler.draw.is_clip_in = true;

            let mut pattern = (); // TODO

            self.tiler.tile_path(clip.iter(), transform, &mut self.tile_mask, None, &mut pattern, encoder);

            self.stats.row_time += Duration::from_ns(self.tiler.row_decomposition_time_ns);
            self.stats.tile_time += Duration::from_ns(self.tiler.tile_decomposition_time_ns);

            // TODO: tile clip
            // TODO: push clip state.
            self.stats.tiled_paths += 1;
        }

        for cmd in group.commands.iter().rev() {
            match cmd {
                Command::Fill(fill) => {
                    let target = &mut self.targets[group.group_stack_depth as usize];
                    let encoder = &mut target.tile_encoder;

                    self.tiler.draw.z_index = fill.z_index;
                    self.tiler.draw.is_clip_in = false;

                    let transform = if fill.transform != 0 {
                        Some(&commands.transforms[fill.transform].transform)
                    } else {
                        None
                    };

                    let pattern: &mut dyn TilerPattern = match fill.pattern {
                        Pattern::Color(color) => {
                            target.solid_color_pattern.set_color(color);
                            &mut target.solid_color_pattern
                        },
                        Pattern::Checkerboard { colors, scale } => {
                            target.checkerboard_pattern.set(colors[0], colors[1], scale);
                            &mut target.checkerboard_pattern
                        }
                        _ => { unimplemented!() }
                    };

                    self.tiler.tile_path(fill.path.iter(), transform, &mut self.tile_mask, None, pattern, encoder);

                    self.stats.row_time += Duration::from_ns(self.tiler.row_decomposition_time_ns);
                    self.stats.tile_time += Duration::from_ns(self.tiler.tile_decomposition_time_ns);
                    self.stats.tiled_paths += 1;
                }
                Command::Group(group) => {
                    self.process_group(group, commands);
                }
            }
        }

        if group.clip.is_some() {
            // TODO: restore clip state.
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
