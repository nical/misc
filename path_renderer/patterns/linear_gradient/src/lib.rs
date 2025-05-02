use core::geom::traits::Transformation;
use core::geom::Point;

use core::gpu::shader::{PatternDescriptor, ShaderPatternId, Shaders, Varying, BlendMode};
use core::gpu::GpuStoreWriter;
use core::pattern::BuiltPattern;
use core::Color;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct LinearGradient {
    pub from: Point<f32>,
    pub to: Point<f32>,
    pub color0: Color,
    pub color1: Color,
}

impl LinearGradient {
    pub fn transformed<T: Transformation<f32>>(&self, tx: &T) -> Self {
        LinearGradient {
            from: tx.transform_point(self.from),
            to: tx.transform_point(self.to),
            color0: self.color0,
            color1: self.color1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct LinearGradientRenderer {
    shader: ShaderPatternId,
}

impl LinearGradientRenderer {
    pub fn register(shaders: &mut Shaders) -> Self {
        let shader = shaders.register_pattern(PatternDescriptor {
            name: "pattern::linear_gradient".into(),
            source: SHADER_SRC.into(),
            varyings: vec![
                Varying::float32x2("position").with_interpolation(true),
                Varying::float32x4("color0").with_interpolation(false),
                Varying::float32x4("color1").with_interpolation(false),
                Varying::float32x3("dir_offset").with_interpolation(false),
            ],
            bindings: None,
        });

        LinearGradientRenderer { shader }
    }

    pub fn add(&self, gpu_store: &mut GpuStoreWriter, gradient: LinearGradient) -> BuiltPattern {
        let can_stretch_horizontally =
            gradient.from.x == gradient.to.y || gradient.color0 == gradient.color1;
        let is_opaque = gradient.color0.is_opaque() && gradient.color1.is_opaque();
        let color0 = gradient.color0.to_f32();
        let color1 = gradient.color1.to_f32();

        let handle = gpu_store.push_slice(&[
            gradient.from.x,
            gradient.from.y,
            gradient.to.x,
            gradient.to.y,
            color0[0],
            color0[1],
            color0[2],
            color0[3],
            color1[0],
            color1[1],
            color1[2],
            color1[3],
        ]);

        BuiltPattern::new(self.shader, handle.to_u32())
            .with_opacity(is_opaque)
            .with_horizontal_stretching(can_stretch_horizontally)
            .with_blend_mode(if is_opaque { BlendMode::None } else { BlendMode::PremultipliedAlpha })
    }
}

const SHADER_SRC: &'static str = "
#import pattern::color
#import gpu_store

struct Gradient {
    p0: vec2<f32>,
    p1: vec2<f32>,
    color0: vec4<f32>,
    color1: vec4<f32>,
};

fn fetch_gradient(address: u32) -> Gradient {
    var raw = gpu_store_fetch_3(address);
    var gradient: Gradient;
    gradient.p0 = raw.data0.xy;
    gradient.p1 = raw.data0.zw;
    gradient.color0 = raw.data1;
    gradient.color1 = raw.data2;

    return gradient;
}

fn pattern_vertex(pattern_pos: vec2<f32>, pattern_handle: u32) -> Pattern {
    var gradient = fetch_gradient(pattern_handle);

    var dir = gradient.p1 - gradient.p0;
    dir = dir / dot(dir, dir);
    var offset = dot(gradient.p0, dir);

    return Pattern(
        pattern_pos,
        gradient.color0,
        gradient.color1,
        vec3<f32>(dir, offset),
    );
}

fn pattern_fragment(pattern: Pattern) -> vec4<f32> {
    var d = clamp(dot(pattern.position, pattern.dir_offset.xy) - pattern.dir_offset.z, 0.0, 1.0);
    return mix(pattern.color0, pattern.color1, d);
}
";
