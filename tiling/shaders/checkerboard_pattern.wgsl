#import gpu_store

fn pattern_vertex(tile: TileInstance, uv: vec2<f32>) -> Pattern {
    var pattern = gpu_store_fetch_3(tile.pattern_data.y);
    var offset = pattern[2].xy;
    var scale = pattern[2].z;
    var checker_uv = (tiling_decode_position(tile.pattern_data.x, uv) - offset) / scale;

    return Pattern(checker_uv, pattern[0], pattern[1]);
}

fn pattern_fragment(pattern: Pattern) -> vec4<f32> {
    var ab = (i32(pattern.uv.x) + i32(pattern.uv.y)) % 2;
    if (ab < 0) { ab = 1 - ab; }

    return mix(pattern.color0, pattern.color1, f32(ab));
}
