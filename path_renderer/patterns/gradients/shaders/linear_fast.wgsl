#import pattern::color
#import pattern::gradient
#import pattern::linear_gradient

fn pattern_vertex(pattern_pos: vec2<f32>, pattern_handle: u32) -> Pattern {
    let endpoints = f32_gpu_buffer_fetch_1(pattern_handle);

    let offset = linear_gradient_vertex(pattern_pos, endpoints);

    // Note: this vertex shader is reading more data than it needs,
    // for example the two color stops could be packed into the header.
    // But this way the shader stays compatible with the format that
    // is used for the general case.

    let gradient = f32_gpu_buffer_fetch_4(pattern_handle + 2);
    let header = gradient.data0;
    let stops = gradient.data1.xy;
    return Pattern(
        vec4f(stops.x, stops.y, offset),
        gradient.data2,
        gradient.data3,
    );
}

fn pattern_fragment(pattern: Pattern) -> vec4<f32> {
    var offset = pattern.stops_and_offset.z;

    return evaluate_simple_gradient_2(
        pattern.stops_and_offset.xy,
        pattern.color0,
        pattern.color1,
        offset,
        u32(pattern.dir_offset_extend_mode.w),
    );
}
