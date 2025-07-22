#import pattern::color
#import pattern::gradient
#import pattern::linear_gradient

fn pattern_vertex(pattern_pos: vec2<f32>, pattern_handle: u32) -> Pattern {
    let dir_offset = read_linear_gradient(pattern_handle);

    let gradient = f32_gpu_buffer_fetch_4(pattern_handle + 2);
    let header = make_gradient_header(pattern_handle + 2, gradient.data0);
    return Pattern(
        pattern_pos,
        dir_offset,
        gradient.data1,
        gradient.data2,
        gradient.data3,
        header,
    );
}

fn pattern_fragment(pattern: Pattern) -> vec4<f32> {
    var offset = compute_linear_gradient_offset(
        pattern.position,
        pattern.dir_offset,
    );

    let count = gradient_header_stop_count(pattern.header);
    if count <= 2 {
        // Count includes the sentinel stops so we have at most two "real"
        // color stops.
        let extend_mode = gradient_header_extend_mode(pattern.header);
        return evaluate_simple_gradient_2(
            pattern.stop_offsets.xy,
            pattern.color0,
            pattern.color1,
            offset,
            extend_mode,
        );
    }

    return evaluate_gradient(pattern.header, offset, pattern.stop_offsets);
}
