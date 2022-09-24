fn even_odd(winding_number: f32) -> f32 {
    return 1.0 - abs((abs(winding_number) % 2.0) - 1.0);
}

fn non_zero(winding_number: f32) -> f32 {
    return min(abs(winding_number), 1.0);
}

fn rasterize_edge(p0: vec2<f32>, p1: vec2<f32>) -> f32 {
    // The overlap range on the y axis between the current tow of pixels and the segment.
    // It can be a negative range (negative edge winding).
    var y0 = min(max(0.0, p0.y), 1.0);
    var y1 = min(max(0.0, p1.y), 1.0);

    if (y0 == y1) {
        return 0.0;
    }

    var inv_dy = 1.0 / (p1.y - p0.y);
    // The interpolation factors at the start and end of the intersection between the edge
    // and the row of pixels.
    var t0 = (y0 - p0.y) * inv_dy;
    var t1 = (y1 - p0.y) * inv_dy;
    // X positions at t0 and t1
    var x0 = p0.x * (1.0 - t0) + p1.x * t0;
    var x1 = p0.x * (1.0 - t1) + p1.x * t1;

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

    return area * (y1 - y0);
}

fn resolve_mask(winding_number: f32, fill_rule: u32) -> f32 {
    var mask = 0.0;
    if ((fill_rule & 1u) == 0u) {
        mask = even_odd(winding_number);
    } else {
        mask = non_zero(winding_number);
    }

    // Invert mode.
    if ((fill_rule & 2u) != 0u) {
        mask = 1.0 - mask;
    }

    return mask;
}

