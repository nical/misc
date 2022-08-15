// This shader rasterize the mask for a single tile p0 a
// backdrop and and a sequence of edges.
//
// The "backdrop" is the winding number at the top-right corner
// of the tile (following piet and pathfinder's terminology).

#import raster::fill

struct Edges {
    data: array<vec4<f32>>,
};

@group(0) @binding(1) var<storage> edges: Edges;

@fragment
fn main(
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
