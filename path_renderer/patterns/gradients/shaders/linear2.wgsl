#import pattern::color
#import pattern::gradient
#import pattern::linear_gradient

fn pattern_vertex(pattern_pos: vec2<f32>, pattern_handle: u32) -> Pattern {
    let dir_offset = read_linear_gradient(pattern_handle);

    // Note: this vertex shader is reading more data than it needs,
    // for example the two color stops could be packed into the header.
    // But this way the shader stays compatible with the format that
    // is used for the general case.

    let gradient = f32_gpu_buffer_fetch_4(pattern_handle + 1);
    let header = gradient.data0;
    let stops = gradient.data1.xy;
    return Pattern(
        vec4f(pattern_pos, stops),
        vec4f(dir_offset, header.y),
        gradient.data2,
        gradient.data3,
    );
}

fn pattern_fragment(pattern: Pattern) -> vec4<f32> {
    var offset = compute_linear_gradient_offset(
        pattern.position_stop_offsets.xy,
        pattern.dir_offset_extend_mode.xyz,
    );

    return evaluate_simple_gradient_2(
        pattern.position_stop_offsets.zw,
        pattern.color0,
        pattern.color1,
        offset,
        u32(pattern.dir_offset_extend_mode.w),
    );
}
