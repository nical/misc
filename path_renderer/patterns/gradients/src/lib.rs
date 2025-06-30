#![allow(exported_private_dependencies)]

use core::geom::traits::Transformation;
use core::geom::Point;

use core::shading::{PatternDescriptor, ShaderPatternId, Shaders, Varying, BlendMode};
use core::gpu::{GpuBufferAddress, GpuBufferWriter};
use core::pattern::BuiltPattern;
use core::{ColorF};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ColorStop {
    pub color: ColorF,
    pub offset: f32,
}

pub struct LinearGradientDescriptor<'l> {
    pub stops: &'l [ColorStop],
    pub from: Point<f32>,
    pub to: Point<f32>,
}

impl<'l> LinearGradientDescriptor<'l> {
    pub fn transformed<T: Transformation<f32>>(&self, tx: &T) -> Self {
        LinearGradientDescriptor {
            from: tx.transform_point(self.from),
            to: tx.transform_point(self.to),
            stops: self.stops,
        }
    }
}

// f32_buffer format:  [Count, repeat_mode, (align 4), offset0..offsetN, (align 4), color0..colorN]
fn push_color_stops(buffer: &mut Vec<f32>, stops: &[ColorStop]) {
    let n = stops.len();

    if n == 0 {
        return push_color_stops(
            buffer,
            &[
                ColorStop { color: ColorF { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }, offset: 0.0, },
                ColorStop { color: ColorF { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }, offset: 1.0, },
            ]
        );
    }

    buffer.push((n + 1) as f32);
    buffer.push(0.0); // TODO: repeat mode.
    buffer.push(0.0);
    buffer.push(0.0);

    for stop in stops {
        buffer.push(stop.offset);
    }
    buffer.push(f32::MAX);
    while buffer.len() % 4 != 0 {
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
    linear: ShaderPatternId,
}

impl GradientRenderer {
    pub fn register(shaders: &mut Shaders) -> Self {
        shaders.register_library("pattern::gradient", GRADIENT_LIB_SRC.into());

        let linear = shaders.register_pattern(PatternDescriptor {
            name: "pattern::gradients::linear".into(),
            source: LINEAR_GRADIENT_SRC.into(),
            varyings: vec![
                Varying::float32x2("position").with_interpolation(true),
                Varying::float32x3("dir_offset").with_interpolation(false),
                Varying::uint32("stops_address"),
            ],
            bindings: None,
        });

        GradientRenderer { buffer: Vec::new(), linear }
    }

    fn upload_linear_gradient(&mut self, f32_buffer: &mut GpuBufferWriter, gradient: &LinearGradientDescriptor) -> GpuBufferAddress {
        let stops_len = gradient.stops.len() * 5 + 8;

        self.buffer.clear();
        self.buffer.reserve(stops_len + 4);

        self.buffer.push(gradient.from.x);
        self.buffer.push(gradient.from.y);
        self.buffer.push(gradient.to.x);
        self.buffer.push(gradient.to.y);

        push_color_stops(&mut self.buffer, gradient.stops);

        f32_buffer.push_slice(&self.buffer)
    }

    pub fn add_linear(&mut self, f32_buffer: &mut GpuBufferWriter, gradient: &LinearGradientDescriptor) -> BuiltPattern {
        let can_stretch_horizontally = gradient.from.x == gradient.to.y;
        let is_opaque = gradient.stops.iter().all(|stop| stop.color.a >= 1.0);

        let handle = self.upload_linear_gradient(f32_buffer, gradient);

        BuiltPattern::new(self.linear, handle.to_u32())
            .with_opacity(is_opaque)
            .with_horizontal_stretching(can_stretch_horizontally)
            .with_blend_mode(if is_opaque { BlendMode::None } else { BlendMode::PremultipliedAlpha })
    }
}

const GRADIENT_LIB_SRC: &'static str = "
#import gpu_buffer

fn evaluate_gradient(base_address: u32, offset: f32) -> vec4f {
    var header = f32_gpu_buffer_fetch_1(base_address);
    let count = header.x;
    // TODO: repeat mode

    var addr: u32 = u32(base_address + 1);
    var end_addr: u32 = addr + u32(count);

    // Index of the first gradient stop that is after
    // the current offset.
    var index: u32 = 0;

    var stop_offsets = f32_gpu_buffer_fetch_1(addr);
    var prev_stop_offset = stop_offsets.x;
    var stop_offset = stop_offsets.x;

    while (addr < end_addr) {

        stop_offset = stop_offsets.x;
        if stop_offset > offset { break; }
        index += 1;

        prev_stop_offset = stop_offset;
        stop_offset = stop_offsets.y;
        if stop_offset > offset { break; }
        index += 1;

        prev_stop_offset = stop_offset;
        stop_offset = stop_offsets.z;
        if stop_offset > offset { break; }
        index += 1;

        prev_stop_offset = stop_offset;
        stop_offset = stop_offsets.w;
        if stop_offset > offset { break; }
        index += 1;

        addr += 1;
        stop_offsets = f32_gpu_buffer_fetch_1(addr);
    }

    let colors_address = base_address
        + 1                       // header
        + u32(ceil(count * 0.25)) // offsets
        + max(1, index);          // color index

    let color_pair = f32_gpu_buffer_fetch_2(colors_address - 1);
    let color0 = color_pair.data0;
    let color1 = color_pair.data1;
    var d = stop_offset - prev_stop_offset;
    var factor = 0.0;
    if d > 0.0 {
        factor = clamp((offset - prev_stop_offset) / d, 0.0, 1.0);
    }

    return mix(color0, color1, factor);
}
";

const LINEAR_GRADIENT_SRC: &'static str = "
#import pattern::color
#import pattern::gradient

fn pattern_vertex(pattern_pos: vec2<f32>, pattern_handle: u32) -> Pattern {
    // Fetch the endpoints.
    var endpoints = f32_gpu_buffer_fetch_1(pattern_handle);
    let p0 = endpoints.xy;
    let p1 = endpoints.zw;

    // Gradient stops are stored after the endpoints.
    let stops_address = pattern_handle + 1;

    var dir = p1 - p0;
    dir = dir / dot(dir, dir);
    var offset = dot(p0, dir);

    return Pattern(
        pattern_pos,
        vec3<f32>(dir, offset),
        stops_address,
    );
}

fn pattern_fragment(pattern: Pattern) -> vec4<f32> {
    var offset = clamp(dot(pattern.position, pattern.dir_offset.xy) - pattern.dir_offset.z, 0.0, 1.0);

    return evaluate_gradient(pattern.stops_address, offset);
}
";
