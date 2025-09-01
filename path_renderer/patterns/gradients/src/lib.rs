#![allow(exported_private_dependencies)]

use core::geom::traits::Transformation;
use core::geom::Point;

use core::shading::{PatternDescriptor, ShaderPatternId, Shaders, Varying, BlendMode};
use core::gpu::{GpuBufferAddress, GpuBufferWriter};
use core::pattern::BuiltPattern;
use core::{ColorF, Vector};

const PRESORTED_STOPS: bool = true;
const PRESORTED_STOPS_THRESHOLD: usize = 16;

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
    /// Variant that supports a large number of gradient stops.
    linear_slow: ShaderPatternId,
    /// Fast path that supports only two gradient stops.
    linear_fast: ShaderPatternId,
    // Version that supports both the fats and slow paths.
    linear: ShaderPatternId,

    /// Variant that supports a large number of gradient stops.
    css_radial_slow: ShaderPatternId,
    /// Fast path that supports only two gradient stops.
    css_radial_fast: ShaderPatternId,
    /// Variant that supports a large number of gradient stops and
    /// a focal point.
    svg_radial: ShaderPatternId,

    // Conic gradient (slow path only).
    conic: ShaderPatternId,

    // All css gradient shaders combined (slow path only).
    unified_slow: ShaderPatternId,
    // All css gradient shaders combined (slow and fast paths).
    unified: ShaderPatternId,

    pub debug_mode: u32,
}

impl GradientRenderer {
    pub fn register(shaders: &mut Shaders) -> Self {
        shaders.register_library("pattern::gradient", GRADIENT_LIB_SRC.into());
        shaders.register_library("pattern::linear_gradient", LINEAR_GRADIENT_LIB_SRC.into());
        shaders.register_library("pattern::radial_gradient", RADIAL_GRADIENT_LIB_SRC.into());
        shaders.register_library("pattern::conic_gradient", CONIC_GRADIENT_LIB_SRC.into());

        let linear_slow = shaders.register_pattern(PatternDescriptor {
            name: "gradients::linear_slow".into(),
            source: LINEAR_GRADIENT_SLOW_SRC.into(),
            varyings: vec![
                Varying::float32x2("position").with_interpolation(true),
                Varying::float32x3("dir_offset").flat(),
                Varying::uint32x4("gradient_header").flat(),
            ],
            bindings: None,
        });

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

        let linear_unified = shaders.register_pattern(PatternDescriptor {
            name: "gradients::linear".into(),
            source: LINEAR_GRADIENT_SLOW_FAST_SRC.into(),
            varyings: vec![
                Varying::float32x2("position").with_interpolation(true),
                Varying::float32x3("dir_offset").flat(),
                Varying::float32x4("stop_offsets").flat(),
                Varying::float32x4("color0").flat(),
                Varying::float32x4("color1").flat(),
                Varying::uint32x4("header").flat(),
            ],
            bindings: None,
        });

        let css_radial_slow = shaders.register_pattern(PatternDescriptor {
            name: "gradients::css_radial_slow".into(),
            source: CSS_RADIAL_GRADIENT_SLOW_SRC.into(),
            varyings: vec![
                Varying::float32x4("position_and_start").with_interpolation(true),
                Varying::uint32x4("gradient_header").flat(),
            ],
            bindings: None,
        });

        let css_radial_fast = shaders.register_pattern(PatternDescriptor {
            name: "gradients::css_radial_fast".into(),
            source: CSS_RADIAL_GRADIENT_FAST_SRC.into(),
            varyings: vec![
                Varying::float32x4("position_and_start").with_interpolation(true),
                Varying::float32x2("stop_offsets").flat(),
                Varying::float32x4("color0").flat(),
                Varying::float32x4("color1").flat(),
                Varying::uint32("extend_mode").flat(),
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

        let conic = shaders.register_pattern(PatternDescriptor {
            name: "gradients::conic".into(),
            source: CONIC_GRADIENT_SRC.into(),
            varyings: vec![
                Varying::float32x4("dir_start_scale").with_interpolation(true),
                Varying::uint32x4("gradient_header").flat(),
            ],
            bindings: None,
        });

        let unified_slow = shaders.register_pattern(PatternDescriptor {
            name: "gradients::css_unified_slow".into(),
            source: UNIFIED_GRADIENT_SLOW_SRC.into(),
            varyings: vec![
                Varying::float32x4("interpolated_data").with_interpolation(true),
                Varying::float32x4("flat_data").flat(),
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
            linear_slow,
            linear_fast,
            linear: linear_unified,
            css_radial_slow,
            css_radial_fast,
            svg_radial,
            conic,
            unified_slow,
            unified,
            debug_mode: 0,
        }
    }

    // f32_buffer format:  [Count, repeat_mode, (align 4), offset0..offsetN, (align 4), color0..colorN]
    fn push_color_stops(&mut self, kind: GradientKind, extend_mode: ExtendMode, stops: &[GradientStop]) {
        let n = stops.len();

        if n == 0 {
            return self.push_color_stops(
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

        if PRESORTED_STOPS && n > PRESORTED_STOPS_THRESHOLD {
            self.push_pre_sorted_stop_offsets(stops);
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

    fn push_pre_sorted_stop_offsets(&mut self, stops: &[GradientStop]) {
        let mut levels = 0;
        let mut cap: usize = 4;
        let count = stops.len();
        while cap < count {
            cap = (cap + 1) * 5 - 1;
            levels += 1;
        }

        let mut level_indices = [0, 4, 24, 124, 744];
        // The last level contains most of the indices but we know that we'll never
        // need to store more than the initial count of indices in that level (less
        // in fact, but that's a simple enough upper bound).
        // Any stop offset index beyond that can be assumed to be after the last stop.
        let size = (level_indices[levels] + count + 1) & !3;
        assert!(size % 4 == 0);

        let base_addr = self.buffer.len();
        for i in 0..size {
            // Push some value > 1.
            self.buffer.push(1000.0 + i as f32);
        }
        // Padding.
        while self.buffer.len() % 4 != 0 {
            self.buffer.push(2.0);
        }

        let offsets = &mut self.buffer[base_addr..];

        for idx in 0..count {
            let mut l = 0;
            let mut m = 5;
            while (idx + 1) % m == 0 {
                m *= 5;
                l += 1;
            }

            let level = levels - l;
            let out_idx = level_indices[level];

            level_indices[level] += 1;
            offsets[out_idx] = stops[idx].offset;
        }
    }

    fn upload_linear_gradient(&mut self, f32_buffer: &mut GpuBufferWriter, gradient: &LinearGradientDescriptor) -> GpuBufferAddress {
        let stops_len = gradient.stops.len() * 5 + 8;

        self.buffer.clear();
        self.buffer.reserve(stops_len + 8);

        self.buffer.push(gradient.from.x);
        self.buffer.push(gradient.from.y);
        self.buffer.push(gradient.to.x);
        self.buffer.push(gradient.to.y);
        // TODO: this is wasteful but for simplicty we use the same data
        // layout as the other gradients (2x vec4 for the gradient data).
        self.buffer.push(0.0);
        self.buffer.push(0.0);
        self.buffer.push(0.0);
        self.buffer.push(0.0);

        self.push_color_stops(GradientKind::Linear, gradient.extend_mode, gradient.stops);

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
        self.push_color_stops(kind, gradient.extend_mode, gradient.stops);

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

        self.push_color_stops(GradientKind::Conic, gradient.extend_mode, gradient.stops);

        f32_buffer.push_slice(&self.buffer)
    }

    pub fn add_linear(&mut self, f32_buffer: &mut GpuBufferWriter, gradient: &LinearGradientDescriptor) -> BuiltPattern {
        let can_stretch_horizontally = gradient.from.x == gradient.to.y;
        let is_opaque = gradient.stops.iter().all(|stop| stop.color.a >= 1.0);

        let handle = self.upload_linear_gradient(f32_buffer, gradient);

        let shader = if self.debug_mode == 1 {
            println!("Using linear shader (fast+slow)");
            self.linear
        } else if self.debug_mode == 2 {
            println!("Using slow path shader");
            self.linear_slow
        } else if self.debug_mode == 3 {
            println!("Forcing fast path shader");
            self.linear_fast
        } else if self.debug_mode == 4 {
            println!("Forcing unified slow gradient shader");
            self.unified_slow
        } else if self.debug_mode == 5 {
            println!("Forcing unified with fast path gradient shader v2");
            self.unified
        } else if gradient.stops.len() == 2 {
            self.linear_fast
        } else {
            self.linear_slow
        };

        BuiltPattern::new(shader, handle.to_u32())
            .with_opacity(is_opaque)
            .with_horizontal_stretching(can_stretch_horizontally)
            .with_blend_mode(if is_opaque { BlendMode::None } else { BlendMode::PremultipliedAlpha })
    }

    pub fn add_radial(&mut self, f32_buffer: &mut GpuBufferWriter, gradient: &RadialGradientDescriptor) -> BuiltPattern {
        let handle = self.upload_radial_gradient(f32_buffer, gradient);

        // The SVG radial gradient shader is quite a bit more expensive so
        // use the simpler version when we can.
        let shader = if self.debug_mode == 4 && gradient.focal.is_none() {
            println!("Forcing unified slow gradient shader");
            self.unified_slow
        } else if self.debug_mode == 5 && gradient.focal.is_none(){
            println!("Forcing unified with fast path gradient shader");
            self.unified
        } else if gradient.focal.is_none() {
            if gradient.stops.len() == 2 {
                self.css_radial_fast
            } else {
                self.css_radial_slow
            }
        } else {
            self.svg_radial
        };

        let is_opaque = gradient.stops.iter().all(|stop| stop.color.a >= 1.0);

        BuiltPattern::new(shader, handle.to_u32())
            .with_opacity(is_opaque)
            .with_blend_mode(if is_opaque { BlendMode::None } else { BlendMode::PremultipliedAlpha })
    }

    pub fn add_conic(&mut self, f32_buffer: &mut GpuBufferWriter, gradient: &ConicGradientDescriptor) -> BuiltPattern {
        let is_opaque = gradient.stops.iter().all(|stop| stop.color.a >= 1.0);

        let handle = self.upload_conic_gradient(f32_buffer, gradient);

        let shader = if self.debug_mode == 4 {
            println!("Forcing unified slow gradient shader");
            self.unified_slow
        } else if self.debug_mode == 5 {
            println!("Forcing unified with fast path gradient shader");
            self.unified
        } else  {
            self.conic
        };

        BuiltPattern::new(shader, handle.to_u32())
            .with_opacity(is_opaque)
            .with_blend_mode(if is_opaque { BlendMode::None } else { BlendMode::PremultipliedAlpha })
    }
}

const GRADIENT_LIB_SRC: &'static str = include_str!("../shaders/lib/gradient.wgsl");
const LINEAR_GRADIENT_LIB_SRC: &'static str = include_str!("../shaders/lib/linear.wgsl");
const RADIAL_GRADIENT_LIB_SRC: &'static str = include_str!("../shaders/lib/radial.wgsl");
const CONIC_GRADIENT_LIB_SRC: &'static str = include_str!("../shaders/lib/conic.wgsl");

const LINEAR_GRADIENT_SLOW_SRC: &'static str = include_str!("../shaders/linear_slow.wgsl");
const LINEAR_GRADIENT_FASR_SRC: &'static str = include_str!("../shaders/linear_fast.wgsl");
const LINEAR_GRADIENT_SLOW_FAST_SRC: &'static str = include_str!("../shaders/linear.wgsl");
const CSS_RADIAL_GRADIENT_SLOW_SRC: &'static str = include_str!("../shaders/css_radial_slow.wgsl");
const CSS_RADIAL_GRADIENT_FAST_SRC: &'static str = include_str!("../shaders/css_radial_fast.wgsl");
const SVG_RADIAL_GRADIENT_SRC: &'static str = include_str!("../shaders/svg_radial.wgsl");
const CONIC_GRADIENT_SRC: &'static str = include_str!("../shaders/conic.wgsl");
const UNIFIED_GRADIENT_SLOW_SRC: &'static str = include_str!("../shaders/css_unified_slow.wgsl");
const UNIFIED_GRADIENT_SRC: &'static str = include_str!("../shaders/css_unified.wgsl");
