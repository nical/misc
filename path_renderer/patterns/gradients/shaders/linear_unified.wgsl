#import pattern::color
#import pattern::gradient
#import pattern::linear_gradient

fn pattern_vertex(pattern_pos: vec2<f32>, pattern_handle: u32) -> Pattern {
    let dir_offset = read_linear_gradient(pattern_handle);

    let gradient = f32_gpu_buffer_fetch_4(pattern_handle + 1);
    let header = decode_gradient_header(pattern_handle + 1, gradient.data0);
    let stops = gradient.data1.xy;
    return Pattern(
        vec4f(pattern_pos, stops),
        dir_offset,
        // If the gradient has two stops, these will contain the two colors.
        // Otherwise they may contain a mix of offsets and colors. We could
        // still use them in the fragment shader to avoid some gpu buffer reads.
        gradient.data2,
        gradient.data3,
        header,
    );
}

fn pattern_fragment(pattern: Pattern) -> vec4<f32> {
    var offset = compute_linear_gradient_offset(
        pattern.position_stop_offsets.xy,
        pattern.dir_offset,
    );

    let count = pattern.header.x;
    if count <= 3 {
        // Count includes the sentinel stops so we have at most two "real"
        // color stops.
        let extend_mode = u32(pattern.header.y);
        return evaluate_simple_gradient_2(
            pattern.position_stop_offsets.zw,
            pattern.color0,
            pattern.color1,
            offset,
            extend_mode,
        );
    }

    return evaluate_gradient(pattern.header, offset);
}
