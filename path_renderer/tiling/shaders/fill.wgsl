#if EDGE_STORE_BINDING {
    #if EDGE_TEXTURE {
        #mixin EDGE_STORE_BINDING var edge_texture: texture_2d<f32>;
        let EDGE_TEXTURE_WIDTH: u32 = 1024u;

        fn read_edge(idx: u32) -> vec4<f32> {
            let uv = vec2<i32>(
                i32(idx % EDGE_TEXTURE_WIDTH),
                i32(idx / EDGE_TEXTURE_WIDTH),
            );
            return textureLoad(edge_texture, uv, 0);
        }
    } #else {
        struct Edges { data: array<vec4<f32>> };
        #mixin EDGE_STORE_BINDING var<storage> edge_buffer: Edges;

        fn read_edge(idx: u32) -> vec4<f32> {
            return edge_buffer.data[idx];
        }
    }
}


fn even_odd(winding_number: f32) -> f32 {
    return 1.0 - abs((abs(winding_number) % 2.0) - 1.0);
}

fn non_zero(winding_number: f32) -> f32 {
    return min(abs(winding_number), 1.0);
}

fn rasterize_edge_analytical(p0: vec2<f32>, p1: vec2<f32>) -> f32 {
    // The overlap range on the y axis between the current row of pixels and the segment.
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

fn rasterize_fill_mask_analytical(uv: vec2<f32>, edges: vec2<u32>, fill_rule: u32, backdrop: f32) -> f32 {
    var winding_number = backdrop;

    var edge_idx = edges.x;
    loop {
        if (edge_idx >= edges.y) {
            break;
        }

        var edge = read_edge(edge_idx);
        edge_idx = edge_idx + 1u;

        // Position of this pixel's top-left corner (in_uv points to the pixel's center).
        // See comment in tiler.rs about the half-pixel offset.
        let pixel_offset = uv - vec2<f32>(0.5);

        // Move to coordinates local to the current pixel.
        var p0 = edge.xy - pixel_offset;
        var p1 = edge.zw - pixel_offset;

        winding_number = winding_number + rasterize_edge_analytical(p0, p1);
    }

    var mask = resolve_mask(winding_number, fill_rule);

    return mask;
}

fn rasterize_edge_ssaa(upper: vec2<f32>, lower: vec2<f32>) -> f32 {
    let y_range = step(upper.y, 0.0) * step(0.0, lower.y);

    let v1 = lower - upper;
    let v2 = upper;
    let is_right_side = step(0.0, v1.x * v2.y - v2.x * v1.y);

    return y_range * is_right_side;
}

fn rasterize_fill_mask_ssaa4(uv: vec2<f32>, edges: vec2<u32>, fill_rule: u32, backdrop: f32) -> f32 {
    var wn = vec4<f32>(backdrop);

    var edge_idx = edges.x;
    loop {
        if (edge_idx >= edges.y) {
            break;
        }

        var edge = read_edge(edge_idx);
        edge_idx = edge_idx + 1u;

        var p0 = edge.xy;
        var p1 = edge.zw;

        let s = sign(p1.y - p0.y);

        let select = step(0.0, s);
        let upper = mix(p0, p1, 1.0 - select) - uv;
        let lower = mix(p0, p1, select) - uv;
        // Sample positions:
        // +---+---+---+---+
        // |   |   | b |   |
        // +---+---+---+---+
        // | a |   |   |   |
        // +---+---o---+---+
        // |   |   |   | c |
        // +---+---+---+---+
        // |   | d |   |   |
        // +---+---+---+---+
        let a = vec2<f32>(-3.0/8.0, -1.0/8.0);
        let b = vec2<f32>( 1.0/8.0, -3.0/8.0);
        let c = vec2<f32>( 3.0/8.0,  1.0/8.0);
        let d = vec2<f32>(-1.0/8.0,  3.0/8.0);
        wn.x = wn.x + s * rasterize_edge_ssaa(upper + a, lower + a);
        wn.y = wn.y + s * rasterize_edge_ssaa(upper + b, lower + b);
        wn.z = wn.z + s * rasterize_edge_ssaa(upper + c, lower + c);
        wn.w = wn.w + s * rasterize_edge_ssaa(upper + d, lower + d);
    }

    var mask = 0.0;
    if ((fill_rule & 1u) == 0u) {
        mask = even_odd(wn.x) + even_odd(wn.y) + even_odd(wn.z) + even_odd(wn.w);
    } else {
        mask = non_zero(wn.x) + non_zero(wn.y) + non_zero(wn.z) + non_zero(wn.w);
    }

    mask *= 0.25;

    // Invert mode.
    if ((fill_rule & 2u) != 0u) {
        mask = 1.0 - mask;
    }

    return mask;
}

fn rasterize_fill_mask(uv: vec2<f32>, edges: vec2<u32>, fill_rule: u32, backdrop: f32) -> f32 {
    #if FILL_SSAA4 {
        return rasterize_fill_mask_ssaa4(uv, edges, fill_rule, backdrop);
    } #else {
        return rasterize_fill_mask_analytical(uv, edges, fill_rule, backdrop);
    }
}