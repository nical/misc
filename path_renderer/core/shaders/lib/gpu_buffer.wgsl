const GPU_BUFFER_WIDTH: u32 = 2048u;
const GPU_BUFFER_ADDRESS_NONE: u32 = 0xFFFFFu;

fn gpu_buffer_decode_uv(address: u32) -> vec2<i32> {
    return vec2<i32>(
        i32(address % GPU_BUFFER_WIDTH),
        i32(address / GPU_BUFFER_WIDTH),
    );
}

struct F32GpuData2 { data0: vec4<f32>, data1: vec4<f32> };
struct F32GpuData3 { data0: vec4<f32>, data1: vec4<f32>, data2: vec4<f32> };
struct F32GpuData4 { data0: vec4<f32>, data1: vec4<f32>, data2: vec4<f32>, data3: vec4<f32> };

struct U32GpuData2 { data0: vec4<u32>, data1: vec4<u32> };
struct U32GpuData3 { data0: vec4<u32>, data1: vec4<u32>, data2: vec4<u32> };
struct U32GpuData4 { data0: vec4<u32>, data1: vec4<u32>, data2: vec4<u32>, data3: vec4<u32> };

// TODO: We can probably rely on the staging buffer chunk alignment to avoid
// decoding consecutive addresses multiple times.

fn f32_gpu_buffer_fetch_1(address: u32) -> vec4<f32> {
    var uv = gpu_buffer_decode_uv(address);
    return textureLoad(f32_gpu_buffer_texture, uv, 0);
}

fn f32_gpu_buffer_fetch_2(address: u32) -> F32GpuData2 {
    return F32GpuData2(
        textureLoad(f32_gpu_buffer_texture, gpu_buffer_decode_uv(address), 0),
        textureLoad(f32_gpu_buffer_texture, gpu_buffer_decode_uv(address + 1u), 0),
    );
}

fn f32_gpu_buffer_fetch_3(address: u32) -> F32GpuData3 {
    return F32GpuData3(
        textureLoad(f32_gpu_buffer_texture, gpu_buffer_decode_uv(address), 0),
        textureLoad(f32_gpu_buffer_texture, gpu_buffer_decode_uv(address + 1u), 0),
        textureLoad(f32_gpu_buffer_texture, gpu_buffer_decode_uv(address + 2u), 0),
    );
}

fn f32_gpu_buffer_fetch_4(address: u32) -> F32GpuData4 {
    return F32GpuData4(
        textureLoad(f32_gpu_buffer_texture, gpu_buffer_decode_uv(address), 0),
        textureLoad(f32_gpu_buffer_texture, gpu_buffer_decode_uv(address + 1u), 0),
        textureLoad(f32_gpu_buffer_texture, gpu_buffer_decode_uv(address + 2u), 0),
        textureLoad(f32_gpu_buffer_texture, gpu_buffer_decode_uv(address + 3u), 0),
    );
}

fn u32_gpu_buffer_fetch_1(address: u32) -> vec4<u32> {
    var uv = gpu_buffer_decode_uv(address);
    return textureLoad(u32_gpu_buffer_texture, uv, 0);
}

fn u32_gpu_buffer_fetch_2(address: u32) -> U32GpuData2 {
    return U32GpuData2(
        textureLoad(u32_gpu_buffer_texture, gpu_buffer_decode_uv(address), 0),
        textureLoad(u32_gpu_buffer_texture, gpu_buffer_decode_uv(address + 1u), 0),
    );
}

fn u32_gpu_buffer_fetch_3(address: u32) -> U32GpuData3 {
    return U32GpuData3(
        textureLoad(u32_gpu_buffer_texture, gpu_buffer_decode_uv(address), 0),
        textureLoad(u32_gpu_buffer_texture, gpu_buffer_decode_uv(address + 1u), 0),
        textureLoad(u32_gpu_buffer_texture, gpu_buffer_decode_uv(address + 2u), 0),
    );
}

fn u32_gpu_buffer_fetch_4(address: u32) -> U32GpuData4 {
    return U32GpuData4(
        textureLoad(u32_gpu_buffer_texture, gpu_buffer_decode_uv(address), 0),
        textureLoad(u32_gpu_buffer_texture, gpu_buffer_decode_uv(address + 1u), 0),
        textureLoad(u32_gpu_buffer_texture, gpu_buffer_decode_uv(address + 2u), 0),
        textureLoad(u32_gpu_buffer_texture, gpu_buffer_decode_uv(address + 3u), 0),
    );
}
