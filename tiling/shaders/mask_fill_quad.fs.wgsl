// This shader rasterize the mask for a single tile from a
// backdrop and and a sequence of edges.
//
// The "backdrop" is the winding number at the top-right corner
// of the tile (following piet and pathfinder's terminology).

struct FragmentOutput {
    [[location(0)]] color: vec4<f32>;
};

struct Edge {
    from: vec2<f32>;
    ctrl: vec2<f32>;
    to: vec2<f32>;
    kind: u32;
    padding: u32;
};

struct Edges {
    data: [[stride(32)]] array<Edge>;
};

[[group(0), binding(1)]] var<storage> edges: Edges;

fn even_odd(winding_number: f32) -> f32 {
    return 1.0 - abs((abs(winding_number) % 2.0) - 1.0);
}

fn approx_parabola_integral(x: f32) -> f32 {
    return x / (0.33 + sqrt(sqrt(0.20151121 + 0.25 * x * x)));
}

fn approx_parabola_inv_integral(x: f32) -> f32 {
    return x * (0.61 + sqrt(0.1521 + 0.25 * x * x));
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

        var from = edge.from - in_uv;
        var ctrl = edge.ctrl - in_uv;
        var to = edge.to - in_uv;

        var tolerance = 0.05;

        var count = 1u;
        var integral_from = 0.0;
        var integral_step = 0.0;
        var div_inv_integral_diff = 0.0;
        if (edge.kind == 1u) {
            var ddx = 2.0 * ctrl.x - from.x - to.x;
            var ddy = 2.0 * ctrl.y - from.y - to.y;
            var cross = (to.x - from.x) * ddy - (to.y - from.y) * ddx;
            var parabola_from = ((ctrl.x - from.x) * ddx + (ctrl.y - from.y) * ddy) / cross;
            var parabola_to = ((to.x - ctrl.x) * ddx + (to.y - ctrl.y) * ddy) / cross;
            var hypot_ddx_ddy = sqrt(ddx * ddx + ddy * ddy);
            var scale = abs(cross) / (hypot_ddx_ddy * abs(parabola_to - parabola_from));
            integral_from = approx_parabola_integral(parabola_from);
            var integral_to = approx_parabola_integral(parabola_to);
            var integral_diff = integral_to - integral_from;

            var inv_integral_from = approx_parabola_inv_integral(integral_from);
            var inv_integral_to = approx_parabola_inv_integral(integral_to);
            div_inv_integral_diff = 1.0 / (inv_integral_to - inv_integral_from);

            // TODO: count could be NaN but we the hardware/driver may not handle NaN so
            // we have to handle this earlier.
            var count_f = ceil(0.5 * abs(integral_diff) * sqrt(scale / tolerance));
            count = min(u32(count_f), 32u);

            integral_step = integral_diff / count_f;
        }

        var i = 0.0;
        var prev = from;
        loop {
            if (count == 0u) {
                break;
            }

            var next = to;
            if (count > 1u) {
                var u = approx_parabola_inv_integral(integral_from + integral_step * i);
                var t = (u - inv_integral_from) * div_inv_integral_diff;
                var one_t = (1.0 - t);

                next = from * one_t * one_t + ctrl * 2.0 * one_t * t + to * t * t;
            }

            var window = vec2<f32>(
                min(max(0.0, prev.y), 1.0),
                min(max(0.0, next.y), 1.0)
            );

            if (window.x != window.y) {
                var t = (window - vec2<f32>(prev.y, prev.y)) / (next.y - prev.y);
                var xs = vec2<f32>(
                    prev.x * (1.0 - t.x) + next.x * t.x,
                    prev.x * (1.0 - t.y) + next.x * t.y,
                );
                var xmin = min(min(xs.x, xs.y), 1.0) - 1e-6;
                var xmax = max(xs.x, xs.y);
                var b = min(xmax, 1.0);
                var c = max(b, 0.0);
                var d = max(xmin, 0.0);
                var area = (b + 0.5 * (d * d - c * c) - xmin) / (xmax - xmin);

                winding_number = winding_number + area * (window.x - window.y);
            }

            prev = next;
            count = count - 1u;
            i = i + 1.0;
        }
    }

    var mask = even_odd(winding_number);
    var color = vec4<f32>(mask, mask, mask, mask);

    return FragmentOutput(color);
}