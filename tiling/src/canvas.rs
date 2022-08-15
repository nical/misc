use std::sync::Arc;
use lyon::path::Path;
use lyon::geom::euclid::default::{Transform2D, Box2D};
use crate::Color;

pub enum Pattern {
    Color(Color),
    Image(u32),
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
use crate::tiler::{Tiler, TilerConfig, TiledPattern};
use crate::gpu::masked_tiles::MaskUploader;

pub struct FrameBuilder {
    pub tile_encoders: Vec<TileEncoder>,
    pub tiler: Tiler,
    stats: FrameBuilderStats,
}

impl FrameBuilder {
    pub fn new(config: &TilerConfig, uploader: MaskUploader) -> Self {
        FrameBuilder {
            tile_encoders: vec![TileEncoder::new(config, uploader)],
            tiler: Tiler::new(config),
            stats: FrameBuilderStats::new(),
        }
    }

    pub fn build(&mut self, commands: &Commands) {
        let t0 = time::precise_time_ns();

        self.stats = FrameBuilderStats::new();

        let group_stack_depth = commands.max_group_stack_depth as usize;
        while self.tile_encoders.len() < group_stack_depth {
            let encoder = self.tile_encoders[0].create_similar();
            self.tile_encoders.push(encoder);
        }

        for encoder in &mut self.tile_encoders[0..group_stack_depth] {
            encoder.reset();
        }
        self.tiler.clear_depth();

        self.process_group(&commands.root, &commands);

        for encoder in &mut self.tile_encoders[0..group_stack_depth] {
            encoder.end_paths();
            encoder.reverse_alpha_tiles();
        }

        self.stats.total_time = Duration::from_ns(time::precise_time_ns() - t0);
    }

    pub fn process_group(&mut self, group: &Group, commands: &Commands) {
        if let Some(clip) = &group.clip {
            // TODO: save clip state.

            // TODO: support clipping the root.
            let parent_group_depth = group.group_stack_depth as usize - 1;
            let encoder = &mut self.tile_encoders[parent_group_depth];

            let transform = if group.transform != 0 {
                Some(&commands.transforms[group.transform].transform)
            } else {
                None
            };

            self.tiler.draw.is_opaque = false;
            self.tiler.draw.is_clip_in = true;

            self.tiler.tile_path(clip.iter(), transform, encoder);

            self.stats.row_time += Duration::from_ns(self.tiler.row_decomposition_time_ns);
            self.stats.tile_time += Duration::from_ns(self.tiler.tile_decomposition_time_ns);

            // TODO: tile clip
            // TODO: push clip state.
            self.stats.tiled_paths += 1;
        }

        for cmd in group.commands.iter().rev() {
            match cmd {
                Command::Fill(fill) => {
                    let encoder = &mut self.tile_encoders[group.group_stack_depth as usize];
                    self.tiler.draw.z_index = fill.z_index;
                    self.tiler.draw.is_clip_in = false;

                    let pattern = match fill.pattern {
                        Pattern::Color(color) => TiledPattern::Color(color),
                        _ => { unimplemented!() }
                    };

                    self.tiler.set_pattern(pattern);

                    let transform = if fill.transform != 0 {
                        Some(&commands.transforms[fill.transform].transform)
                    } else {
                        None
                    };

                    self.tiler.tile_path(fill.path.iter(), transform, encoder);

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
