#import pattern::color

fn pattern_vertex(pattern_pos: vec2<f32>, uv: vec2<f32>, pattern_data: u32) -> Pattern {
    return Pattern(decode_color(pattern_data));
}

fn pattern_fragment(pattern: Pattern) -> vec4<f32> {
    return pattern.color;
}
