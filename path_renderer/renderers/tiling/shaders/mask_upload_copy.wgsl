#import rect
#import tiling
#import render_target

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) src_offset: u32,
};

@vertex
fn vs_main(
    @location(0) tile: u32,
    @location(1) a_src_offset: u32,
    @builtin(vertex_index) vertex_index: u32,
) -> VertexOutput {
    var uv = rect_get_uv(vertex_index);
    var position = tiling_decode_position(tile, uv);
    var target_pos = canvas_to_target(position);

    return VertexOutput(target_pos, uv * TILE_SIZE_F32, a_src_offset);
}

struct MaskBuffer {
    // Ideally we would want a buffer of 64k u8 but bytes aren't supported by wgpu (yet?)
    // so we interpret the buffer as u32 and do some bit fiddling instead.
    data: array<u32, 16384>,
};

@group(1) @binding(0) var<storage> mask_buffer: MaskBuffer;

@fragment
fn fs_main(
    @location(0) local_uv: vec2<f32>,
    @location(1) @interpolate(flat) src_offset: u32,
) -> @location(0) vec4<f32> {

    var offset = src_offset + u32(floor(local_uv.y)) * TILE_SIZE_U32 + u32(floor(local_uv.x));

    var payload = mask_buffer.data[offset / 4u];
    let shift = (offset % 4u) * 8u;
    var alpha = f32((payload >> shift) & 255u) / 255.0;

    //alpha = max(alpha, 0.3);

    return vec4<f32>(alpha, alpha, alpha, alpha);
}
