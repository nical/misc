// This shader rasterize the mask for a single tile from a
// backdrop and and a sequence of edges.
//
// The "backdrop" is the winding number at the top-right corner
// of the tile (following piet and pathfinder's terminology).

struct FragmentOutput {
    [[location(0)]] color: vec4<f32>;
};

struct Edges {
    data: [[stride(16)]] array<vec4<f32>>;
};

[[group(0), binding(1)]] var<storage> edges: Edges;

fn even_odd(winding_number: f32) -> f32 {
    return 1.0 - abs((abs(winding_number) % 2.0) - 1.0);
}

[[stage(fragment)]]
fn main(
    [[location(0), interpolate(linear)]] in_uv: vec2<f32>,
    [[location(1), interpolate(flat)]] in_edges_range: vec2<u32>,
    [[location(2), interpolate(flat)]] in_fill_rule: u32,
) -> FragmentOutput {

    var winding_number = 0.0;

    var edge_idx = in_edges_range.x;
    loop {
        if (edge_idx >= in_edges_range.y) {
            break;
        }

        var edge = edges.data[edge_idx];
        edge_idx = edge_idx + 1u;

        // Move to coordinates local to the current pixel.
        var from = edge.xy - in_uv;
        var to = edge.zw - in_uv;

        // The overlap range on the y axis between the current tow of pixels and the segment.
        // It can be a negative range (negative edge winding).
        var y1 = min(max(0.0, from.y), 1.0);
        var y0 = min(max(0.0, to.y), 1.0);

        if (y0 == y1) {
            continue;
        }

        var inv_dy = 1.0 / (to.y - from.y);
        // The interpolation factors at the start and end of the intersection between the edge
        // and the row of pixels.
        var t0 = (y0 - from.y) * inv_dy;
        var t1 = (y1 - from.y) * inv_dy;
        // X positions at t0 and t1
        var x0 = from.x * (1.0 - t0) + to.x * t0;
        var x1 = from.x * (1.0 - t1) + to.x * t1;

        // Jitter to avoid NaN when dividing by xmin-xmax (for example vertical edges).
        // The original value was 1e-5 but it wasn't sufficient to avoid issues with 32px tiles.
        // TODO: although rare, edges with a certain slope will still cause NaN. Is there a way to
        // make this more robust with a big perf hit?
        var jitter = 1e-5;

        var xmin = min(min(x1, x0), 1.0) - jitter;
        var xmax = max(x1, x0);
        var b = min(xmax, 1.0);
        var c = max(b, 0.0);
        var d = max(xmin, 0.0);
        var area = (b + 0.5 * (d * d - c * c) - xmin) / (xmax - xmin);

        winding_number = winding_number + area * (y1 - y0);
    }

    var mask = even_odd(winding_number);
    var color = vec4<f32>(mask, mask, mask, mask);

    return FragmentOutput(color);
}
