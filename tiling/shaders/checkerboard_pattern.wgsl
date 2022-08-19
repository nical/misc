#import rect
#import render_target
#import tiling
#import raster::fill
#import pattern::color

@group(0) @binding(0) var<uniform> atlas: TileAtlasDescriptor;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(linear) uv: vec2<f32>,
    @location(1) @interpolate(flat) colors: vec2<u32>,
};

@vertex fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @location(0) mask_id: u32,
    @location(1) scale: f32,
    @location(2) colors: vec2<u32>,
    @location(3) offset: vec2<f32>,
) -> VertexOutput {
    var uv = rect_get_uv(vertex_index);
    let atlas_uv = tiling_atlas_get_uv(atlas, mask_id, uv);
    let target_pos = normalized_to_target(atlas_uv);

    var checker_uv = (uv * TILE_SIZE_F32 + offset) / scale;

    return VertexOutput(target_pos, checker_uv, colors);
}

struct Edges {
    data: array<vec4<f32>>,
};

@group(0) @binding(1) var<storage> edges: Edges;

@fragment fn fs_main(
    @location(0) @interpolate(linear) uv: vec2<f32>,
    @location(1) @interpolate(flat) colors: vec2<u32>,
 ) -> @location(0) vec4<f32> {

    var color0 = decode_color(colors.x);
    var color1 = decode_color(colors.y);

    var ab = (i32(uv.x) + i32(uv.y)) % 2;
    if (ab < 0) { ab = 1 - ab; }

    return mix(color0, color1, f32(ab));
}
