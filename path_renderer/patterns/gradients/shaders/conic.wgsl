#import pattern::color
#import pattern::gradient
#import pattern::conic_gradient

fn pattern_vertex(pattern_pos: vec2<f32>, pattern_handle: u32) -> Pattern {
    let conic = read_conic_gradient(pattern_pos, pattern_handle);
    let header = read_gradient_header(pattern_handle + 2);

    return Pattern(conic, header);
}

fn pattern_fragment(pattern: Pattern) -> vec4<f32> {
    var offset = compute_conic_gradient_offset(pattern.dir_start_scale);
    return evaluate_gradient(pattern.gradient_header, offset);
}
