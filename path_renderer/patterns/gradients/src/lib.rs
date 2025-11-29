#![allow(exported_private_dependencies)]

use core::geom::traits::Transformation;
use core::geom::Point;

use core::shading::{PatternDescriptor, ShaderPatternId, Shaders, Varying};
use core::gpu::{GpuBufferAddress, GpuBufferWriter};
use core::pattern::BuiltPattern;
use core::{ColorF, Vector};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ExtendMode {
    Clamp,
    Repeat,
}

#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum GradientKind {
    Linear = 0,
    Conic = 1,
    CssRadial = 2,
    SvgRadial = 3,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct GradientStop {
    pub color: ColorF,
    pub offset: f32,
}

pub struct LinearGradientDescriptor<'l> {
    pub stops: &'l [GradientStop],
    pub extend_mode: ExtendMode,
    pub from: Point<f32>,
    pub to: Point<f32>,
}

impl<'l> LinearGradientDescriptor<'l> {
    pub fn transformed<T: Transformation<f32>>(&self, tx: &T) -> Self {
        LinearGradientDescriptor {
            from: tx.transform_point(self.from),
            to: tx.transform_point(self.to),
            extend_mode: self.extend_mode,
            stops: self.stops,
        }
    }
}

pub struct RadialGradientDescriptor<'l> {
    pub stops: &'l [GradientStop],
    pub extend_mode: ExtendMode,
    pub center: Point<f32>,
    pub focal: Option<Point<f32>>,
    pub scale: Vector,
    pub start_radius: f32,
    pub end_radius: f32,
}

impl<'l> RadialGradientDescriptor<'l> {
    pub fn transformed<T: Transformation<f32>>(&self, tx: &T) -> Self {
        RadialGradientDescriptor {
            center: tx.transform_point(self.center),
            focal: self.focal.map(|focal| tx.transform_point(focal)),
            scale: tx.transform_vector(self.scale),
            extend_mode: self.extend_mode,
            stops: self.stops,
            start_radius: self.start_radius, // TODO
            end_radius: self.end_radius, // TODO
        }
    }
}

pub struct ConicGradientDescriptor<'l> {
    pub stops: &'l [GradientStop],
    pub extend_mode: ExtendMode,
    pub center: Point<f32>,
    pub scale: Vector,
    // In radians
    pub start_angle: f32,
    // In radians
    pub end_angle: f32,
}


#[derive(Clone, Debug)]
pub struct GradientRenderer {
    buffer: Vec<f32>,

    /// All css gradient shaders combined (slow and fast paths).
    unified: ShaderPatternId,

    /// Fast path that supports only two gradient stops.
    #[allow(unused)]
    linear_fast: ShaderPatternId,

    /// Variant that supports a focal point.
    svg_radial: ShaderPatternId,
}

impl GradientRenderer {
    pub fn register(shaders: &mut Shaders) -> Self {
        shaders.register_library("pattern::gradient", GRADIENT_LIB_SRC.into());
        shaders.register_library("pattern::linear_gradient", LINEAR_GRADIENT_LIB_SRC.into());
        shaders.register_library("pattern::radial_gradient", RADIAL_GRADIENT_LIB_SRC.into());
        shaders.register_library("pattern::conic_gradient", CONIC_GRADIENT_LIB_SRC.into());

        let linear_fast = shaders.register_pattern(PatternDescriptor {
            name: "gradients::linear_fast".into(),
            source: LINEAR_GRADIENT_FASR_SRC.into(),
            varyings: vec![
                Varying::float32x4("position_stop_offsets").with_interpolation(true),
                Varying::float32x4("dir_offset_extend_mode").flat(),
                Varying::float32x4("color0").flat(),
                Varying::float32x4("color1").flat(),
            ],
            bindings: None,
        });

        let svg_radial = shaders.register_pattern(PatternDescriptor {
            name: "gradients::svg_radial".into(),
            source: SVG_RADIAL_GRADIENT_SRC.into(),
            varyings: vec![
                Varying::float32x4("position_and_center").with_interpolation(true),
                Varying::float32x2("start_end_radius").flat(),
                Varying::uint32x4("gradient_header").flat(),
            ],
            bindings: None,
        });

        let unified = shaders.register_pattern(PatternDescriptor {
            name: "gradients::css_unified".into(),
            source: UNIFIED_GRADIENT_SRC.into(),
            varyings: vec![
                Varying::float32x4("interpolated_data").with_interpolation(true),
                Varying::float32x4("flat_data").flat(),
                Varying::float32x4("stop_offsets").flat(),
                Varying::float32x4("color0").flat(),
                Varying::float32x4("color1").flat(),
                Varying::uint32x4("gradient_header").flat(),
            ],
            bindings: None,
        });

        GradientRenderer {
            buffer: Vec::new(),
            linear_fast,
            svg_radial,
            unified,
        }
    }

    // f32_buffer format:  [Count, repeat_mode, (align 4), offset0..offsetN, (align 4), color0..colorN]
    fn push_gradient_stops(&mut self, kind: GradientKind, extend_mode: ExtendMode, stops: &[GradientStop]) {
        let n = stops.len();

        if n == 0 {
            return self.push_gradient_stops(
                kind,
                ExtendMode::Clamp,
                &[
                    GradientStop { color: ColorF { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }, offset: 0.0, },
                ],
            );
        }

        self.buffer.push(n as f32);
        self.buffer.push(match extend_mode {
            ExtendMode::Clamp => 0.0,
            ExtendMode::Repeat => 1.0
        });
        self.buffer.push((kind as u8) as f32);
        // padding
        self.buffer.push(0.0);

        for stop in stops {
            self.buffer.push(stop.color.r);
            self.buffer.push(stop.color.g);
            self.buffer.push(stop.color.b);
            self.buffer.push(stop.color.a);
        }

        if n > 4 {
            push_stop_offsets(stops, &mut self.buffer);
        } else {
            for stop in stops {
                self.buffer.push(stop.offset);
            }
            while self.buffer.len() % 4 != 0 {
                // padding
                self.buffer.push(0.0);
            }
        }
    }

    fn offsets_storage_size(&self, stops: &[GradientStop]) -> usize {
        let count = stops.len();
        if count <= 4 {
            return count;
        }

        let mut num_blocks_for_level = 1;
        let mut cap: usize = 4;
        while cap < count {
            num_blocks_for_level *= 5;
            cap += num_blocks_for_level * 4;
        }

        // Fix the capacity up to account for the fact that we don't
        // store the entirety of the last level;
        let num_blocks_for_last_level = num_blocks_for_level.min(count / 5 + 1);
        cap += (num_blocks_for_last_level - num_blocks_for_level) * 4;

        return cap;
    }

    fn upload_linear_gradient(&mut self, f32_buffer: &mut GpuBufferWriter, gradient: &LinearGradientDescriptor) -> GpuBufferAddress {
        self.buffer.clear();
        let colors_storage = gradient.stops.len() * 4;
        let offsets_storage = self.offsets_storage_size(&gradient.stops);
        self.buffer.reserve(12 + colors_storage + offsets_storage);

        self.buffer.push(gradient.from.x);
        self.buffer.push(gradient.from.y);
        self.buffer.push(gradient.to.x);
        self.buffer.push(gradient.to.y);
        // Note: this is wasteful but for simplicty we use the same data
        // layout as the other gradients (2x vec4 for the gradient data).
        self.buffer.push(0.0);
        self.buffer.push(0.0);
        self.buffer.push(0.0);
        self.buffer.push(0.0);

        self.push_gradient_stops(GradientKind::Linear, gradient.extend_mode, gradient.stops);

        f32_buffer.push_slice(&self.buffer)
    }

    fn upload_radial_gradient(&mut self, f32_buffer: &mut GpuBufferWriter, gradient: &RadialGradientDescriptor) -> GpuBufferAddress {
        let stops_len = gradient.stops.len() * 5 + 8;

        self.buffer.clear();
        self.buffer.reserve(stops_len + 8);

        self.buffer.push(gradient.center.x);
        self.buffer.push(gradient.center.y);
        if let Some(focal) = gradient.focal {
            self.buffer.push(focal.x);
            self.buffer.push(focal.y);
        } else {
            self.buffer.push(gradient.center.x);
            self.buffer.push(gradient.center.y);
        }

        self.buffer.push(gradient.scale.x);
        self.buffer.push(gradient.scale.y);
        self.buffer.push(gradient.start_radius);
        self.buffer.push(gradient.end_radius);

        let kind = if gradient.focal.is_some() {
            GradientKind::SvgRadial
        } else {
            GradientKind::CssRadial
        };
        self.push_gradient_stops(kind, gradient.extend_mode, gradient.stops);

        f32_buffer.push_slice(&self.buffer)
    }

    fn upload_conic_gradient(&mut self, f32_buffer: &mut GpuBufferWriter, gradient: &ConicGradientDescriptor) -> GpuBufferAddress {
        let stops_len = gradient.stops.len() * 5 + 8;

        self.buffer.clear();
        self.buffer.reserve(stops_len + 8);

        self.buffer.push(gradient.center.x);
        self.buffer.push(gradient.center.y);
        self.buffer.push(gradient.scale.x);
        self.buffer.push(gradient.scale.y);
        self.buffer.push(gradient.start_angle);
        self.buffer.push(gradient.end_angle);
        self.buffer.push(0.0);
        self.buffer.push(0.0);

        self.push_gradient_stops(GradientKind::Conic, gradient.extend_mode, gradient.stops);

        f32_buffer.push_slice(&self.buffer)
    }

    pub fn add_linear(&mut self, f32_buffer: &mut GpuBufferWriter, gradient: &LinearGradientDescriptor) -> BuiltPattern {
        let can_stretch_horizontally = gradient.from.x == gradient.to.y;
        let is_opaque = gradient.stops.iter().all(|stop| stop.color.a >= 1.0);

        let handle = self.upload_linear_gradient(f32_buffer, gradient);

        let shader = self.unified;

        BuiltPattern::new(shader, handle.to_u32())
            .with_opacity(is_opaque)
            .with_horizontal_stretching(can_stretch_horizontally)
    }

    pub fn add_radial(&mut self, f32_buffer: &mut GpuBufferWriter, gradient: &RadialGradientDescriptor) -> BuiltPattern {
        let handle = self.upload_radial_gradient(f32_buffer, gradient);

        let shader = if gradient.focal.is_none() {
            self.unified
        } else {
            self.svg_radial
        };

        let is_opaque = gradient.stops.iter().all(|stop| stop.color.a >= 1.0);

        BuiltPattern::new(shader, handle.to_u32())
            .with_opacity(is_opaque)
    }

    pub fn add_conic(&mut self, f32_buffer: &mut GpuBufferWriter, gradient: &ConicGradientDescriptor) -> BuiltPattern {
        let is_opaque = gradient.stops.iter().all(|stop| stop.color.a >= 1.0);

        let handle = self.upload_conic_gradient(f32_buffer, gradient);

        let shader = self.unified;

        BuiltPattern::new(shader, handle.to_u32())
            .with_opacity(is_opaque)
    }
}

// Push stop offsets in rearranged order so that the search can be carried
// out as an implicit tree traversal.
//
// The structure of the tree is:
//  - Each level is plit into 5 partitions.
//  - The root level has one node (4 offsets -> 5 partitions).
//  - Each level has 5 more nodes than the previous one.
//  - Levels are arranged one by one starting from the root
//
// ```ascii
// level : indices
// ------:---------
//   0   :                                                              24      ...
//   1   :          4         9            14             19             |      ...
//   2   :  0,1,2,3,|,5,6,7,8,|10,11,12,13,| ,15,16,17,18,| ,20,21,22,23,| ,25, ...
// ```
//
// In the example above:
// - The first (root) contains a single block containing the stop offsets from
//   indices [24, 49, 74, 99].
// - The second level contains blocks of offsets from indices [4, 9, 14, 19],
//   [29, 34, 39, 44], etc.
// - The third (leaf) level contains blocks from indices [0,1,2,3], [5,6,7,8],
//   [15, 16, 17, 18], etc.
//
// Placeholder offsets (1.0) are used when a level has more capacity than the
// input number of stops.
//
// Conceptually, blocks [0,1,2,3] and [5,6,7,8] are the first two childrent of
// the node [4,9,14,19], separated by the offset from index 4.
// Links are not explicitly represented via pointers or indices. Instead the
// position in the buffer is sufficient to represent the level and index of the
// stop (at the expense of having to store extra padding to round up each tree
// level to its power-of-5-aligned size).
//
// This scheme is meant to make the traversal efficient loading offsets in
// blocks of 4. The shader can converge to the leaf in very few loads.
fn push_stop_offsets(stops: &[GradientStop], buffer: &mut Vec<f32>) {
    let mut num_levels = 1;
    let mut cap: usize = 4;
    let count = stops.len();
    while cap < count {
        cap = (cap + 1) * 5 - 1;
        num_levels += 1;
    }

    let mut index_stride = 5;
    let mut next_index_stride = 1;
    while index_stride <= count {
        index_stride *= 5;
        next_index_stride *= 5;
    }

    // Number of 4-offsets blocks for the current level.
    // The root has 1, then each level has 5 more than the previous one.
    let mut num_blocks_for_level = 1;

    // Go over each level, starting from the root.
    for level in 0..num_levels {
        // This scheme rounds up the number of offsets to store for each
        // level to the next power of 5, which can represent a lot of wasted
        // space, especially for the last levels. We need each level to start
        // at a specific power-of-5-aligned offset so we can't get around the
        // wasted space for all levels except the last one (which has the most
        // waste).
        let is_last_level = level == num_levels - 1;
        let num_blocks = if is_last_level {
            // A reasonable upper bound for the number of blocks needed in
            // the last level.
            num_blocks_for_level.min(count / 5 + 1)
        } else {
            num_blocks_for_level
        };

        for block_idx in 0..num_blocks {
            for i in 0..4 {
                // index of the stop in the input stop buffer.                    let linear_idx = block_idx * index_stride
                let linear_idx = block_idx * index_stride
                    + i * next_index_stride
                    + next_index_stride - 1;

                let offset = if linear_idx < stops.len() {
                    stops[linear_idx].offset
                } else {
                    // Use a placeholder value for indices that
                    // are out of the range of the input buffer.
                    // these are padding values needed to round
                    // eahc level next to the netx power of 5.
                    1.0
                };
                buffer.push(offset);
            }
        }

        index_stride = next_index_stride;
        next_index_stride /= 5;
        num_blocks_for_level *= 5;
    }
}


const GRADIENT_LIB_SRC: &'static str = include_str!("../shaders/lib/gradient.wgsl");
const LINEAR_GRADIENT_LIB_SRC: &'static str = include_str!("../shaders/lib/linear.wgsl");
const RADIAL_GRADIENT_LIB_SRC: &'static str = include_str!("../shaders/lib/radial.wgsl");
const CONIC_GRADIENT_LIB_SRC: &'static str = include_str!("../shaders/lib/conic.wgsl");

const LINEAR_GRADIENT_FASR_SRC: &'static str = include_str!("../shaders/linear_fast.wgsl");
const SVG_RADIAL_GRADIENT_SRC: &'static str = include_str!("../shaders/svg_radial.wgsl");
const UNIFIED_GRADIENT_SRC: &'static str = include_str!("../shaders/css_unified.wgsl");


#[test]
fn push_offsets_smoke_test() {
    let mut stops = Vec::with_capacity(300);
    let mut buffer = Vec::with_capacity(300);
    for i in 0..300 {
        buffer.clear();
        stops.push(GradientStop {
            offset: i as f32,
            color: ColorF { r: 0.0, g: 0.0, b: 0.0, a: 0.0 },
        });

        push_stop_offsets(&stops, &mut buffer);
        assert!(buffer.len() >= stops.len());
        for stop in &stops {
            assert!(buffer.iter().find(|val| **val == stop.offset).is_some())
        }
    }
}
