use core::gpu::shader::{ShaderPatternId, Shaders, Varying, PatternDescriptor};
use core::{Color, pattern::BuiltPattern};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColorPattern {
    data: u32,
    is_opaque: bool,
    shader: ShaderPatternId,
}

#[derive(Clone, Debug)]
pub struct SolidColorRenderer {
    shader: ShaderPatternId,
}

impl SolidColorRenderer {
    pub fn register(shaders: &mut Shaders) -> Self {
        let shader = shaders.register_pattern(PatternDescriptor {
            name: "pattern::solid_color".into(),
            source: SHADER_SRC.into(),
            varyings: vec![Varying::float32x4("color").with_interpolation(false)],
            bindings: None,
        });

        SolidColorRenderer { shader }
    }

    pub fn add(&self, color: Color) -> BuiltPattern {
        BuiltPattern::new(self.shader, color.to_u32())
            .with_opacity(color.is_opaque())
            .with_horizontal_stretching(true)
    }
}

const SHADER_SRC: &'static str = "
#import pattern::color

fn pattern_vertex(pattern_pos: vec2<f32>, pattern_data: u32) -> Pattern {
    return Pattern(decode_color(pattern_data));
}

fn pattern_fragment(pattern: Pattern) -> vec4<f32> {
    return pattern.color;
}
";
