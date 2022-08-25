// A simple linear gradient pattern shader that only supports two stops and
// no repetitions. Mostly for demo purposes.

#import rect
#import render_target
#import tiling
#import pattern::color

@group(0) @binding(0) var<uniform> atlas: TileAtlasDescriptor;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) colors: vec2<u32>,
    @location(2) @interpolate(flat) dir: vec2<f32>,
    @location(3) @interpolate(flat) offset: f32,
};

@vertex fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @location(0) ints: vec4<u32>,
    @location(1) positions: vec4<f32>,
) -> VertexOutput {
    let tile_id = ints.x;
    let colors = ints.zw;

    var uv = rect_get_uv(vertex_index);
    let atlas_uv = tiling_atlas_get_uv(atlas, tile_id, uv);
    let target_pos = normalized_to_target(atlas_uv);

    var pos = uv * TILE_SIZE_F32;

    var dir = positions.zw - positions.xy;
    dir = dir / dot(dir, dir);
    var offset = dot(positions.xy, dir);

    return VertexOutput(target_pos, pos, colors, dir, offset);
}

@fragment fn fs_main(
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) colors: vec2<u32>,
    @location(2) @interpolate(flat) dir: vec2<f32>,
    @location(3) @interpolate(flat) offset: f32,
 ) -> @location(0) vec4<f32> {

    var color0 = decode_color(colors.x);
    var color1 = decode_color(colors.y);

    var d = clamp(dot(uv, dir) - offset, 0.0, 1.0);

    return mix(color0, color1, d);
}
