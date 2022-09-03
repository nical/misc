#import rect
#import render_target
#import tiling
#import pattern::color
#import gpu_store


@group(0) @binding(0) var<uniform> atlas: TileAtlasDescriptor;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) color0: vec4<f32>,
    @location(2) @interpolate(flat) color1: vec4<f32>,
};

@vertex fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @location(0) instance: vec4<u32>,
) -> VertexOutput {
    var uv = rect_get_uv(vertex_index);
    var tile_pos = tiling_decode_position(instance.x, uv);
    let target_pos = normalized_to_target(tile_pos * atlas.inv_resolution);

    var pattern = gpu_store_fetch_3(instance.w);
    var offset = pattern[2].xy;
    var scale = pattern[2].z;
    var checker_uv = (tiling_decode_position(instance.z, uv) - offset) / scale;

    return VertexOutput(target_pos, checker_uv, pattern[0], pattern[1]);
}

@fragment fn fs_main(
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) color0: vec4<f32>,
    @location(2) @interpolate(flat) color1: vec4<f32>,
 ) -> @location(0) vec4<f32> {

    var ab = (i32(uv.x) + i32(uv.y)) % 2;
    if (ab < 0) { ab = 1 - ab; }

    return mix(color0, color1, f32(ab));
}
