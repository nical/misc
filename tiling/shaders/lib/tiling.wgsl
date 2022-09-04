let TILE_SIZE_U32: u32 = 16u;
let TILE_SIZE_F32: f32 = 16.0;

struct TileInstance {
    tile_index: u32,
    mask_index: u32,
    pattern_data: vec2<u32>,
};

fn tiling_decode_instance(encoded: vec4<u32>) -> TileInstance {
    var instance: TileInstance;
    instance.tile_index = encoded.x;
    instance.mask_index = encoded.y;
    instance.pattern_data = encoded.zw;

    return instance;
}

let TILE_COORD_MASK: u32 = 0x3FFu;

fn tiling_decode_position(encoded: u32, uv: vec2<f32>) -> vec2<f32> {
    var offset = vec2<f32>(
        f32((encoded >> 10u) & TILE_COORD_MASK),
        f32(encoded & TILE_COORD_MASK),
    );
    var extended = vec2<f32>(
        uv.x * f32((encoded >> 20u) & TILE_COORD_MASK),
        0.0
    );

    return (offset + (uv + extended)) * TILE_SIZE_F32;
}
