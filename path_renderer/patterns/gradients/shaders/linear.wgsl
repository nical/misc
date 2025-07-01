#import pattern::color
#import pattern::gradient
#import pattern::linear_gradient

fn pattern_vertex(pattern_pos: vec2<f32>, pattern_handle: u32) -> Pattern {
    let dir_offset = read_linear_gradient(pattern_handle);
    let header = read_gradient_header(pattern_handle + 1);

    return Pattern(pattern_pos, dir_offset, header);
}

fn pattern_fragment(pattern: Pattern) -> vec4<f32> {
    var offset = compute_linear_gradient_offset(pattern.position, pattern.dir_offset);
    return evaluate_gradient(pattern.gradient_header, offset);
}
