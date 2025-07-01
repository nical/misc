#![allow(exported_private_dependencies)]

use core::geom::traits::Transformation;
use core::geom::Point;

use core::shading::{PatternDescriptor, ShaderPatternId, Shaders, Varying, BlendMode};
use core::gpu::{GpuBufferAddress, GpuBufferWriter};
use core::pattern::BuiltPattern;
use core::{ColorF, Vector};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ExtendMode {
    Clamp,
    Repeat,
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
    pub scale: Vector,
    pub start_radius: f32,
    pub end_radius: f32,
}

impl<'l> RadialGradientDescriptor<'l> {
    pub fn transformed<T: Transformation<f32>>(&self, tx: &T) -> Self {
        RadialGradientDescriptor {
            center: tx.transform_point(self.center),
            scale: tx.transform_vector(self.scale),
            extend_mode: self.extend_mode,
            stops: self.stops,
            start_radius: self.start_radius, // TODO
            end_radius: self.end_radius, // TODO
        }
    }
}

// f32_buffer format:  [Count, repeat_mode, (align 4), offset0..offsetN, (align 4), color0..colorN]
fn push_color_stops(buffer: &mut Vec<f32>, extend_mode: ExtendMode, stops: &[GradientStop]) {
    let n = stops.len();

    if n == 0 {
        return push_color_stops(
            buffer,
            ExtendMode::Clamp,
            &[
                GradientStop { color: ColorF { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }, offset: 0.0, },
                GradientStop { color: ColorF { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }, offset: 1.0, },
            ]
        );
    }

    buffer.push((n + 1) as f32);
    buffer.push(match extend_mode {
        ExtendMode::Clamp => 0.0,
        ExtendMode::Repeat => 1.0
    });
    // padding
    buffer.push(0.0);
    buffer.push(0.0);

    for stop in stops {
        buffer.push(stop.offset);
    }
    buffer.push(f32::MAX);
    while buffer.len() % 4 != 0 {
        // padding
        buffer.push(0.0);
    }

    for stop in stops {
        buffer.push(stop.color.r);
        buffer.push(stop.color.g);
        buffer.push(stop.color.b);
        buffer.push(stop.color.a);
    }

    let last = stops.last().unwrap();
    buffer.push(last.color.r);
    buffer.push(last.color.g);
    buffer.push(last.color.b);
    buffer.push(last.color.a);
}


#[derive(Clone, Debug)]
pub struct GradientRenderer {
    buffer: Vec<f32>,
    /// Variant that supports a large number of gradient stops.
    linear: ShaderPatternId,
    /// Fast path that supports only two gradient stops.
    linear_2: ShaderPatternId,

    /// Variant that supports a large number of gradient stops.
    radial: ShaderPatternId,
    /// Fast path that supports only two gradient stops.
    radial_2: ShaderPatternId,
}

impl GradientRenderer {
    pub fn register(shaders: &mut Shaders) -> Self {
        shaders.register_library("pattern::gradient", GRADIENT_LIB_SRC.into());
        shaders.register_library("pattern::linear_gradient", LINEAR_GRADIENT_LIB_SRC.into());
        shaders.register_library("pattern::radial_gradient", RADIAL_GRADIENT_LIB_SRC.into());

        let linear = shaders.register_pattern(PatternDescriptor {
            name: "gradients::linear".into(),
            source: LINEAR_GRADIENT_SRC.into(),
            varyings: vec![
                Varying::float32x2("position").with_interpolation(true),
                Varying::float32x3("dir_offset").flat(),
                Varying::uint32x4("gradient_header").flat(),
            ],
            bindings: None,
        });

        let linear_2 = shaders.register_pattern(PatternDescriptor {
            name: "gradients::linear2".into(),
            source: LINEAR_GRADIENT_2_SRC.into(),
            varyings: vec![
                Varying::float32x2("position").with_interpolation(true),
                Varying::float32x3("dir_offset").flat(),
                Varying::float32x2("stop_offsets").flat(),
                Varying::float32x4("color0").flat(),
                Varying::float32x4("color1").flat(),
                Varying::uint32("extend_mode").flat(),
            ],
            bindings: None,
        });

        let radial = shaders.register_pattern(PatternDescriptor {
            name: "gradients::radial".into(),
            source: RADIAL_GRADIENT_SRC.into(),
            varyings: vec![
                Varying::float32x3("position_and_start").with_interpolation(true),
                Varying::uint32x4("gradient_header").flat(),
            ],
            bindings: None,
        });

        let radial_2 = shaders.register_pattern(PatternDescriptor {
            name: "gradients::radial2".into(),
            source: RADIAL_GRADIENT_2_SRC.into(),
            varyings: vec![
                Varying::float32x3("position_and_start").with_interpolation(true),
                Varying::float32x2("stop_offsets").flat(),
                Varying::float32x4("color0").flat(),
                Varying::float32x4("color1").flat(),
                Varying::uint32("extend_mode").flat(),
            ],
            bindings: None,
        });

        GradientRenderer {
            buffer: Vec::new(),
            linear,
            linear_2,
            radial,
            radial_2,
        }
    }

    fn upload_linear_gradient(&mut self, f32_buffer: &mut GpuBufferWriter, gradient: &LinearGradientDescriptor) -> GpuBufferAddress {
        let stops_len = gradient.stops.len() * 5 + 8;

        self.buffer.clear();
        self.buffer.reserve(stops_len + 4);

        self.buffer.push(gradient.from.x);
        self.buffer.push(gradient.from.y);
        self.buffer.push(gradient.to.x);
        self.buffer.push(gradient.to.y);

        push_color_stops(&mut self.buffer, gradient.extend_mode, gradient.stops);

        f32_buffer.push_slice(&self.buffer)
    }

    fn upload_radial_gradient(&mut self, f32_buffer: &mut GpuBufferWriter, gradient: &RadialGradientDescriptor) -> GpuBufferAddress {
        let stops_len = gradient.stops.len() * 5 + 8;

        self.buffer.clear();
        self.buffer.reserve(stops_len + 8);

        self.buffer.push(gradient.center.x);
        self.buffer.push(gradient.center.y);
        self.buffer.push(gradient.scale.x);
        self.buffer.push(gradient.scale.y);
        self.buffer.push(gradient.start_radius);
        self.buffer.push(gradient.end_radius);
        self.buffer.push(0.0);
        self.buffer.push(0.0);

        push_color_stops(&mut self.buffer, gradient.extend_mode, gradient.stops);

        f32_buffer.push_slice(&self.buffer)
    }

    pub fn add_linear(&mut self, f32_buffer: &mut GpuBufferWriter, gradient: &LinearGradientDescriptor) -> BuiltPattern {
        let can_stretch_horizontally = gradient.from.x == gradient.to.y;
        let is_opaque = gradient.stops.iter().all(|stop| stop.color.a >= 1.0);

        let handle = self.upload_linear_gradient(f32_buffer, gradient);

        let shader = if gradient.stops.len() == 2 {
            self.linear_2
        } else {
            self.linear
        };

        BuiltPattern::new(shader, handle.to_u32())
            .with_opacity(is_opaque)
            .with_horizontal_stretching(can_stretch_horizontally)
            .with_blend_mode(if is_opaque { BlendMode::None } else { BlendMode::PremultipliedAlpha })
    }

    pub fn add_radial(&mut self, f32_buffer: &mut GpuBufferWriter, gradient: &RadialGradientDescriptor) -> BuiltPattern {
        let is_opaque = gradient.stops.iter().all(|stop| stop.color.a >= 1.0);

        let handle = self.upload_radial_gradient(f32_buffer, gradient);

        let shader = if gradient.stops.len() == 2 {
            self.radial_2
        } else {
            self.radial
        };

        BuiltPattern::new(shader, handle.to_u32())
            .with_opacity(is_opaque)
            .with_blend_mode(if is_opaque { BlendMode::None } else { BlendMode::PremultipliedAlpha })
    }
}

const GRADIENT_LIB_SRC: &'static str = include_str!("../shaders/lib/gradient.wgsl");
const LINEAR_GRADIENT_LIB_SRC: &'static str = include_str!("../shaders/lib/linear.wgsl");
const RADIAL_GRADIENT_LIB_SRC: &'static str = include_str!("../shaders/lib/radial.wgsl");

const LINEAR_GRADIENT_SRC: &'static str = include_str!("../shaders/linear.wgsl");
const LINEAR_GRADIENT_2_SRC: &'static str = include_str!("../shaders/linear2.wgsl");
const RADIAL_GRADIENT_SRC: &'static str = include_str!("../shaders/radial.wgsl");
const RADIAL_GRADIENT_2_SRC: &'static str = include_str!("../shaders/radial2.wgsl");
