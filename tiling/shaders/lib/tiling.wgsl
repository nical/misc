const TILE_SIZE_U32: u32 = 16u;
const TILE_SIZE_F32: f32 = 16.0;

struct TileInstance {
    position: vec2<f32>,
    mask_position: vec2<f32>,
    pattern_position: vec2<f32>,
    pattern_data: u32,
};

const TILE_COORD_MASK: u32 = 0x3FFu;

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

fn tiling_decode_instance(encoded: vec4<u32>, uv: vec2<f32>) -> TileInstance {
    var instance: TileInstance;
    instance.position = tiling_decode_position(encoded.x, uv);
    #if TILED_MASK {
        instance.mask_position = tiling_decode_position(encoded.y, uv);
    }
    instance.pattern_position = tiling_decode_position(encoded.z, uv);
    instance.pattern_data = encoded.w;

    return instance;
}

