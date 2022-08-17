// This shader rasterize the mask for a single tile p0 a
// backdrop and and a sequence of edges.
//
// The "backdrop" is the winding number at the top-right corner
// of the tile (following piet and pathfinder's terminology).

#import rect
#import render_target
#import tiling
#import raster::fill

@group(0) @binding(0) var<uniform> atlas: TileAtlasDescriptor;

struct VertexOutput {
    @location(0) @interpolate(linear) uv: vec2<f32>,
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

struct Edges {
    data: array<vec4<f32>>,
};

@group(0) @binding(1) var<storage> edges: Edges;

@fragment fn fs_main(
    @location(0) @interpolate(linear) in_uv: vec2<f32>,
    @location(1) @interpolate(flat) in_edges_range: vec2<u32>,
    @location(2) @interpolate(flat) in_fill_rule: u32,
    @location(3) @interpolate(flat) backdrop: f32,
) -> @location(0) vec4<f32> {

    var winding_number = backdrop;

    var edge_idx = in_edges_range.x;
    loop {
        if (edge_idx >= in_edges_range.y) {
            break;
        }

        var edge = edges.data[edge_idx];
        edge_idx = edge_idx + 1u;

        // Move to coordinates local to the current pixel.
        var p0 = edge.xy - in_uv;
        var p1 = edge.zw - in_uv;

        winding_number = winding_number + rasterize_edge(p0, p1);
    }

    var mask = resolve_mask(winding_number, in_fill_rule);

    var color = vec4<f32>(mask, mask, mask, mask);

    return color;
}
