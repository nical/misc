#import pattern::color
#import pattern::gradient
#import pattern::radial_gradient

fn pattern_vertex(pattern_pos: vec2<f32>, pattern_handle: u32) -> Pattern {
    let gradient = read_css_radial_gradient(pattern_pos, pattern_handle);
    let header = read_gradient_header(pattern_handle + 2);

    return Pattern(gradient, header);
}

fn pattern_fragment(pattern: Pattern) -> vec4<f32> {
    var offset = compute_css_radial_gradient_offset(pattern.position_and_start);
    let first_stop_offsets = f32_gpu_buffer_fetch_1(pattern.gradient_header.z);
    return evaluate_gradient(pattern.gradient_header, offset, first_stop_offsets);
}
