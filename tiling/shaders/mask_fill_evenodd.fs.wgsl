// This shader rasterize the mask for a single tile from a
// backdrop and and a sequence of edges.
//
// The "backdrop" is the winding number at the top-right corner
// of the tile (following piet and pathfinder's terminology).

struct FragmentOutput {
    [[location(0)]] color: vec4<f32>;
};

[[block]]
struct Edges {
    data: [[stride(16)]] array<vec4<f32>>;
};

[[group(0), binding(2)]] var<storage> edges: Edges;

fn even_odd(winding_number: f32) -> f32 {
    return 1.0 - abs((abs(winding_number) % 2.0) - 1.0);
}

[[stage(fragment)]]
fn main(
    [[location(0), interpolate(linear)]] in_uv: vec4<f32>,
    [[location(1), interpolate(flat)]] in_edges_range: vec2<u32>,
    [[location(2), interpolate(flat)]] in_backdrop: f32,
) -> FragmentOutput {

    var winding_number = in_backdrop;

    var edge_idx = in_edges_range.x;
    loop {
        if (edge_idx >= in_edges_range.y) {
            break;
        }

        var edge = edges.data[edge_idx];
        edge_idx = edge_idx + 1;

        var from = edge.xy - in_uv;
        var to = edge.zw - in_uv;

        var window = vec2<f32>(
            min(max(0.0, from.y), 1.0),
            min(max(0.0, to.y), 1.0)
        );

        if (window.x != window.y) {
            var t = (window - vec2<f32>(from.y, from.y)) / (to.y - from.y);
            var xs = vec2<f32>(
                from.x * (1.0 - t.x) + to.x * t.x,
                from.x * (1.0 - t.y) + to.x * t.y,
            );
            var xmin = min(min(xs.x, xs.y), 1.0) - 1e-6; 
            var xmax = max(xs.x, xs.y);
            var b = min(xmax, 1.0);
            var c = max(b, 0.0);
            var d = max(xmin, 0.0);
            var area = (b + 0.5 * (d * d - c * c) - xmin) / (xmax - xmin);

            winding_number = winding_number + area * (window.x - window.y);
        }
    }

    var  = even_odd(winding_number);

    return FragmentOutput(vec4<f32>(mask));
}
