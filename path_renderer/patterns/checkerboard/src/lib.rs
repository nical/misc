use core::geom::euclid::vec2;
use core::geom::traits::Transformation;
use core::Point;

use core::gpu::shader::{PatternDescriptor, ShaderPatternId, Varying};
use core::gpu::{GpuStore, Shaders};
use core::Color;

use core::pattern::{BuiltPattern};


#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Checkerboard {
    pub color0: Color,
    pub color1: Color,
    pub offset: Point,
    pub scale: f32,
}

#[derive(Clone, Debug)]
pub struct CheckerboardRenderer {
    shader: ShaderPatternId,
}

impl Checkerboard {
    pub fn transformed<T: Transformation<f32>>(&self, tx: &T) -> Self {
        Checkerboard {
            offset: tx.transform_point(self.offset),
            scale: tx.transform_vector(vec2(0.0, self.scale)).y,
            color0: self.color0,
            color1: self.color1,
        }
    }
}


impl CheckerboardRenderer {
    pub fn register(shaders: &mut Shaders) -> Self {
        let shader = shaders.register_pattern(PatternDescriptor {
            name: "pattern::checkerboard".into(),
            source: SHADER_SRC.into(),
            varyings: vec![
                Varying::float32x2("uv"),
                Varying::float32x4("color0").flat(),
                Varying::float32x4("color1").flat(),
            ],
            bindings: None,
        });

        CheckerboardRenderer { shader }
    }

    pub fn add(&self, gpu_store: &mut GpuStore, pattern: &Checkerboard) -> BuiltPattern {
        let is_opaque = pattern.color0.is_opaque() && pattern.color1.is_opaque();
        let color0 = pattern.color0.to_f32();
        let color1 = pattern.color1.to_f32();
        let handle = gpu_store.push(&[
            color0[0], color0[1], color0[2], color0[3],
            color1[0], color1[1], color1[2], color1[3],
            pattern.offset.x, pattern.offset.y,
            pattern.scale,
        ]);
    
        BuiltPattern::new(self.shader, handle.to_u32())
            .with_opacity(is_opaque)
            .prerender_by_default()
    }
}

const SHADER_SRC: &'static str = "
#import gpu_store

fn pattern_vertex(pattern_pos: vec2<f32>, pattern_handle: u32) -> Pattern {
    var pattern = gpu_store_fetch_3(pattern_handle);
    var offset = pattern.data2.xy;
    var scale = pattern.data2.z;
    var checker_uv = (pattern_pos - offset) / scale;

    return Pattern(checker_uv, pattern.data0, pattern.data1);
}

fn pattern_fragment(pattern: Pattern) -> vec4<f32> {
    var ab = (i32(pattern.uv.x) + i32(pattern.uv.y)) % 2;
    if (ab < 0) { ab = 1 - ab; }

    return mix(pattern.color0, pattern.color1, f32(ab));
}
";