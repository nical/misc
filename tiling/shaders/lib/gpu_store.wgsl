let GPU_STORE_WIDTH: u32 = 2048u;

@group(0) @binding(1) var gpu_store_texture: texture_2d<f32>;

fn gpu_store_decode_uv(address: u32) -> vec2<i32> {
    return vec2<i32>(
        i32(address % GPU_STORE_WIDTH),
        i32(address / GPU_STORE_WIDTH),
    );
}

struct GpuData2 { data0: vec4<f32>, data1: vec4<f32> };
struct GpuData3 { data0: vec4<f32>, data1: vec4<f32>, data2: vec4<f32> };
struct GpuData4 { data0: vec4<f32>, data1: vec4<f32>, data2: vec4<f32>, data3: vec4<f32> };

fn gpu_store_fetch_1(address: u32) -> vec4<f32> {
    var uv = gpu_store_decode_uv(address);
    return textureLoad(gpu_store_texture, uv, 0);
}

fn gpu_store_fetch_2(address: u32) -> GpuData2 {
    var uv = gpu_store_decode_uv(address);
    return GpuData2(
        textureLoad(gpu_store_texture, gpu_store_decode_uv(address), 0),
        textureLoad(gpu_store_texture, gpu_store_decode_uv(address + 1u), 0),
    );
}

fn gpu_store_fetch_3(address: u32) -> GpuData3 {
    var uv = gpu_store_decode_uv(address);
    return GpuData3(
        textureLoad(gpu_store_texture, gpu_store_decode_uv(address), 0),
        textureLoad(gpu_store_texture, gpu_store_decode_uv(address + 1u), 0),
        textureLoad(gpu_store_texture, gpu_store_decode_uv(address + 2u), 0),
    );
}

fn gpu_store_fetch_4(address: u32) -> GpuData4 {
    var uv = gpu_store_decode_uv(address);
    return GpuData4(
        textureLoad(gpu_store_texture, gpu_store_decode_uv(address), 0),
        textureLoad(gpu_store_texture, gpu_store_decode_uv(address + 1u), 0),
        textureLoad(gpu_store_texture, gpu_store_decode_uv(address + 2u), 0),
        textureLoad(gpu_store_texture, gpu_store_decode_uv(address + 3u), 0),
    );
}


