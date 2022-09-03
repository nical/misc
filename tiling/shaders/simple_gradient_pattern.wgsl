// A simple linear gradient pattern shader that only supports two stops and
// no repetitions. Mostly for demo purposes.

#import rect
#import render_target
#import tiling
#import pattern::color
#import gpu_store

@group(0) @binding(0) var<uniform> atlas: TileAtlasDescriptor;

struct Gradient {
    p0: vec2<f32>,
    p1: vec2<f32>,
    color0: vec4<f32>,
    color1: vec4<f32>,
};

fn fetch_gradient(address: u32) -> Gradient {
    var raw = gpu_store_fetch_3(address);
    var gradient: Gradient;
    gradient.p0 = raw[0].xy;
    gradient.p1 = raw[0].zw;
    gradient.color0 = raw[1];
    gradient.color1 = raw[2];

    return gradient;
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) color0: vec4<f32>,
    @location(2) @interpolate(flat) color1: vec4<f32>,
    @location(3) @interpolate(flat) dir: vec2<f32>,
    @location(4) @interpolate(flat) offset: f32,
};

@vertex fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @location(0) instance: vec4<u32>,
) -> VertexOutput {
    var uv = rect_get_uv(vertex_index);
    var tile_pos = tiling_decode_position(instance.x, uv);
    let target_pos = normalized_to_target(tile_pos * atlas.inv_resolution);

    var pattern_pos = tiling_decode_position(instance.z, uv);

    var gradient = fetch_gradient(instance.w);

    var dir = gradient.p1 - gradient.p0;
    dir = dir / dot(dir, dir);
    var offset = dot(gradient.p0, dir);

    return VertexOutput(
        target_pos,
        pattern_pos,
        gradient.color0,
        gradient.color1,
        dir,
        offset,
    );
}

@fragment fn fs_main(
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) color0: vec4<f32>,
    @location(2) @interpolate(flat) color1: vec4<f32>,
    @location(3) @interpolate(flat) dir: vec2<f32>,
    @location(4) @interpolate(flat) offset: f32,
) -> @location(0) vec4<f32> {

    var d = clamp(dot(uv, dir) - offset, 0.0, 1.0);

    return mix(color0, color1, d);
}
