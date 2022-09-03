let GPU_STORE_16_BITS_MASK: u32 = 0xFFFFu;

@group(0) @binding(1) var gpu_store_texture: texture_2d<f32>;

fn gpu_store_decode_uv(address: u32) -> vec2<i32> {
    return vec2<i32>(
        i32((address >> 16u) & GPU_STORE_16_BITS_MASK),
        i32(address & GPU_STORE_16_BITS_MASK),
    );
}

fn gpu_store_fetch_1(address: u32) -> vec4<f32> {
    var uv = gpu_store_decode_uv(address);
    return textureLoad(gpu_store_texture, uv, 0);
}

fn gpu_store_fetch_2(address: u32) -> array<vec4<f32>, 2> {
    var uv = gpu_store_decode_uv(address);
    return array<vec4<f32>, 2>(
        textureLoad(gpu_store_texture, uv, 0),
        textureLoad(gpu_store_texture, uv + vec2<i32>(1, 0), 0),
    );
}

fn gpu_store_fetch_3(address: u32) -> array<vec4<f32>, 3> {
    var uv = gpu_store_decode_uv(address);
    return array<vec4<f32>, 3>(
        textureLoad(gpu_store_texture, uv, 0),
        textureLoad(gpu_store_texture, uv + vec2<i32>(1, 0), 0),
        textureLoad(gpu_store_texture, uv + vec2<i32>(2, 0), 0),
    );
}

fn gpu_store_fetch_4(address: u32) -> array<vec4<f32>, 4> {
    var uv = gpu_store_decode_uv(address);
    return array<vec4<f32>, 4>(
        textureLoad(gpu_store_texture, uv, 0),
        textureLoad(gpu_store_texture, uv + vec2<i32>(1, 0), 0),
        textureLoad(gpu_store_texture, uv + vec2<i32>(2, 0), 0),
        textureLoad(gpu_store_texture, uv + vec2<i32>(3, 0), 0),
    );
}


