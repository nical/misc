#import pattern::color
#import pattern::gradient
#import pattern::radial_gradient

fn pattern_vertex(pattern_pos: vec2<f32>, pattern_handle: u32) -> Pattern {
    let radial_gradient = read_radial_gradient(pattern_pos, pattern_handle);

    // Note: this vertex shader is reading more data than it needs,
    // for example the two color stops could be packed into the header.
    // But this way the shader stays compatible with the format that
    // is used for the general case.

    let gradient = f32_gpu_buffer_fetch_4(pattern_handle + 2);
    let header = gradient.data0;
    let extend_mode = u32(header.y);
    let stops = gradient.data1.xy;
    return Pattern(
        radial_gradient,
        stops,
        gradient.data2,
        gradient.data3,
        extend_mode,
    );
}

fn pattern_fragment(pattern: Pattern) -> vec4<f32> {
    var offset = compute_radial_gradient_offset(pattern.position_and_start);
    return evaluate_simple_gradient_2(pattern.stop_offsets, pattern.color0, pattern.color1, offset, pattern.extend_mode);
}
