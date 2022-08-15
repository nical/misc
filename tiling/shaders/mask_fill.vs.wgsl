#import quad
#import render_target
#import tiling

@group(0) @binding(0) var<uniform> atlas: TileAtlasDescriptor;

struct VertexOutput {
    @location(0) @interpolate(linear) uv: vec2<f32>,
    @location(1) @interpolate(flat) edges: vec2<u32>,
    @location(2) @interpolate(flat) fill_rule: u32,
    @location(3) @interpolate(flat) backdrop: f32,
    @builtin(position) position: vec4<f32>,
};

@vertex fn main(
    @location(0) in_edges: vec2<u32>,
    @location(1) in_mask_id: u32,
    @location(2) in_fill_rule: u32,
    @builtin(vertex_index) vertex_index: u32,
) -> VertexOutput {

    var fill_rule = in_fill_rule & 0xFFFFu;
    var backdrop = f32(in_fill_rule >> 16u) - 8192.0;

    var uv = quad_get_uv(vertex_index);
    let atlas_uv = tiling_atlas_get_uv(atlas, in_mask_id, uv);
    let target_pos = normalized_to_target(atlas_uv);

    return VertexOutput(
        uv * atlas.tile_size,
        in_edges,
        fill_rule,
        backdrop,
        target_pos,
    );
}
