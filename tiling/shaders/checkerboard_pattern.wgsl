#import gpu_store

fn pattern_vertex(pattern_pos: vec2<f32>, uv: vec2<f32>, pattern_handle: u32) -> Pattern {
    var pattern = gpu_store_fetch_3(pattern_handle);
    var offset = pattern[2].xy;
    var scale = pattern[2].z;
    var checker_uv = (pattern_pos - offset) / scale;

    return Pattern(checker_uv, pattern[0], pattern[1]);
}

fn pattern_fragment(pattern: Pattern) -> vec4<f32> {
    var ab = (i32(pattern.uv.x) + i32(pattern.uv.y)) % 2;
    if (ab < 0) { ab = 1 - ab; }

    return mix(pattern.color0, pattern.color1, f32(ab));
}
