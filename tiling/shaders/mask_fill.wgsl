// This shader rasterize the mask for a single tile p0 a
// backdrop and and a sequence of edges.
//
// The "backdrop" is the winding number at the top-right corner
// of the tile (following piet and pathfinder's terminology).

#import rect
#import render_target
#import tiling

#define EDGE_STORE_BINDING { @group(0) @binding(1) }
#import mask::fill

struct VertexOutput {
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) edges: vec2<u32>,
    @location(2) @interpolate(flat) fill_rule: u32,
    @location(3) @interpolate(flat) backdrop: f32,
    @builtin(position) position: vec4<f32>,
};

@vertex fn vs_main(
    @location(0) in_edges: vec2<u32>,
    @location(1) in_mask_id: u32,
    @location(2) in_fill_rule: u32,
    @builtin(vertex_index) vertex_index: u32,
) -> VertexOutput {
    var fill_rule = in_fill_rule & 0xFFFFu;
    var backdrop = f32(in_fill_rule >> 16u) - 8192.0;

    var uv = rect_get_uv(vertex_index);
    var pos = tiling_decode_position(in_mask_id, uv);
    let target_pos = canvas_to_target(pos);

    return VertexOutput(
        uv * TILE_SIZE_F32,
        in_edges,
        fill_rule,
        backdrop,
        target_pos,
    );
}

@fragment fn fs_main(
    @location(0) in_uv: vec2<f32>,
    @location(1) @interpolate(flat) in_edges_range: vec2<u32>,
    @location(2) @interpolate(flat) in_fill_rule: u32,
    @location(3) @interpolate(flat) backdrop: f32,
) -> @location(0) vec4<f32> {

    let mask = rasterize_fill_mask(in_uv, in_edges_range, in_fill_rule, backdrop);
    return vec4<f32>(mask, mask, mask, mask);
}
