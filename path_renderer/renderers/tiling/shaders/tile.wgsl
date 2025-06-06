#import rect
#import z_index
#import render_task

override TILE_SIZE: f32;
const TILE_COORD_MASK: u32 = 0x3FFu;

struct TileInstance {
    rect: vec4f,
    edges: vec2u, // start..end
    backdrop: i32,
    path: u32,
};

struct PathInfo {
    z: f32,
    opacity: f32,
    pattern_data: u32,
    fill_rule: u32,
    render_task: u32,
    scissor: vec4f,
};

fn tiling_decode_rect(encoded: u32) -> vec4<f32> {
    var offset = vec2<f32>(
        f32((encoded >> 10u) & TILE_COORD_MASK),
        f32(encoded & TILE_COORD_MASK),
    );
    var extend_x = f32((encoded >> 20u) & TILE_COORD_MASK);

    return vec4f(
        offset.x,
        offset.y,
        offset.x + 1.0 + extend_x,
        offset.y + 1.0,
    ) * TILE_SIZE;
}

fn decode_instance(encoded: vec4<u32>) -> TileInstance {
    var instance: TileInstance;

    instance.rect = tiling_decode_rect(encoded.x);
    var first_edge = encoded.y;
    var num_edges = encoded.z >> 16u;
    instance.edges = vec2u(first_edge, first_edge + num_edges);
    instance.backdrop = i32(encoded.z & 0xFFFFu) - 128i;
    instance.path = encoded.w;

    return instance;
}

fn fetch_path(path_idx: u32) -> PathInfo {
    // Must match the gpu_buffer texture width.
    const PATH_TEXTURE_WIDTH: u32 = 2048;
    // The path data occupies two u32x4 pixels in the path texture.
    let idx = path_idx * 2u;
    let path_uv = vec2u(idx % PATH_TEXTURE_WIDTH, idx / PATH_TEXTURE_WIDTH);
    var encoded0 = textureLoad(path_texture, path_uv, 0);
    var encoded1 = textureLoad(path_texture, path_uv + vec2u(1u, 0u), 0);

    var path: PathInfo;

    path.z = z_index_to_f32(encoded0.x);
    path.pattern_data = encoded0.y;
    path.fill_rule = encoded0.z >> 16u;
    path.opacity = f32(encoded0.z & 0xFFFFu) / 65535.0;

    path.scissor = vec4f(encoded1);

    return path;
}

fn geometry_vertex(vertex_index: u32, instance_data: vec4<u32>) -> GeometryVertex {
    var uv = rect_get_uv(vertex_index);
    var tile = decode_instance(instance_data);

    var path = fetch_path(tile.path);

    var task = render_task_fetch(path.render_task);

    // Clip to the scissor rect.
    var position = mix(tile.rect.xy, tile.rect.zw, uv);
    position = max(position, path.scissor.xy);
    position = min(position, path.scissor.zw);
    uv = (position - tile.rect.xy) / (tile.rect.zw - tile.rect.xy);

    var target_position = render_task_target_position(task, position);

    return GeometryVertex(
        vec4f(target_position.x, target_position.y, path.z, 1.0),
        position,
        path.pattern_data,

        uv * TILE_SIZE,
        tile.edges,
        tile.backdrop,
        path.fill_rule,
        path.opacity,
    );
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

fn rasterize_edge_ssaa(upper: vec2<f32>, lower: vec2<f32>) -> f32 {
    // 1 if the sample is in the edge's y range, 0 otherwise.
    let y_range_test = step(upper.y, 0.0) * step(0.0, lower.y);

    let v1 = lower - upper;
    let v2 = upper;
    let is_right_side = step(0.0, v1.x * v2.y - v2.x * v1.y);

    return y_range_test * is_right_side;
}

fn even_odd(winding_number: f32) -> f32 {
    return 1.0 - abs((abs(winding_number) % 2.0) - 1.0);
}

fn non_zero(winding_number: f32) -> f32 {
    return min(abs(winding_number), 1.0);
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

fn geometry_fragment(uv: vec2f, edges: vec2u, backdrop: i32, fill_rule: u32, opacity: f32) -> f32 {
    #if TILING_SSAA4 {
        // Note: This could be packed into a single u32.
        var wn = vec4<f32>(backdrop);
    } #else if TILING_SSAA8 {
        // Note: This could be packed into a single u32.
        var wn0 = vec4<f32>(backdrop);
        var wn1 = vec4<f32>(backdrop);
    } #else {
        var winding_number: f32 = f32(backdrop);
    }

    var edge_idx = edges.x;
    // This isn't necessary but to be on the safe side and make sure we don't accidentally
    // hang from of some corrupted data, restrict the loop to a large-ish number of segments.
    var end = min(edges.y, edges.x + 512u);
    loop {
        if (edge_idx >= end) {
            break;
        }

        // Must match the gpu_buffer texture width.
        const EDGE_TEXTURE_WIDTH: u32 = 2048;
        let edge_uv = vec2i(
            i32(edge_idx % EDGE_TEXTURE_WIDTH),
            i32(edge_idx / EDGE_TEXTURE_WIDTH),
        );
        var edge = textureLoad(edge_texture, edge_uv, 0) * TILE_SIZE;

        edge_idx = edge_idx + 1u;

        #if TILING_SSAA4 {
            var p0 = edge.xy;
            var p1 = edge.zw;

            let s = sign(p1.y - p0.y);

            let select = step(0.0, s);
            var upper = mix(p0, p1, 1.0 - select) - uv;
            var lower = mix(p0, p1, select) - uv;
            // Sample positions:
            // +---+---+---+---+
            // |   |   | a |   |
            // +---+---+---+---+
            // | b |   |   |   |
            // +---+---o---+---+
            // |   |   |   | c |
            // +---+---+---+---+
            // |   | d |   |   |
            // +---+---+---+---+
            // Instead of expressing a, b, c, and d relative to the pixel center, we
            // store them as offsets from the previous sample position. This way we don't
            // need to hold both upper/lower and the result of adding the offset.
            // This saves 4 VGPR and was just enough to get to 32 VGPR which hit maximum
            // occupancy with the color pattern on my RDNA3 iGPU, reducing by 10% the GPU
            // time of the draw call accoring to RGP.
            // This is still slower than the area coverage code path.
            const a = vec2<f32>( 1.0/8.0, -3.0/8.0);
            const b = vec2<f32>(-3.0/8.0, -1.0/8.0);
            const c = vec2<f32>( 3.0/8.0,  1.0/8.0);
            const d = vec2<f32>(-1.0/8.0,  3.0/8.0);
            upper += a;
            lower += a;
            wn.x += s * rasterize_edge_ssaa(upper, lower);
            upper += (b - a);
            lower += (b - a);
            wn.y += s * rasterize_edge_ssaa(upper, lower);
            upper += (c - b);
            lower += (c - b);
            wn.z += s * rasterize_edge_ssaa(upper, lower);
            upper += (d - c);
            lower += (d - c);
            wn.w += s * rasterize_edge_ssaa(upper, lower);
        } #else if TILING_SSAA8 {
            var p0 = edge.xy;
            var p1 = edge.zw;
            let s = sign(p1.y - p0.y);
            let select = step(0.0, s);
            var upper = mix(p0, p1, 1.0 - select) - uv;
            var lower = mix(p0, p1, select) - uv;
            // Sample positions:
            // +---+---+---+---+---+---+---+---+
            // |   |   |   |   |   |   |   | a |
            // +---+---+---+---+---+---+---+---+
            // |   |   | b |   |   |   |   |   |
            // +---+---+---+---+---+---+---+---+
            // |   |   |   |   | c |   |   |   |
            // +---+---+---+---+---+---+---+---+
            // | d |   |   |   |   |   |   |   |
            // +---+---+---+---o---+---+---+---+
            // |   |   |   |   |   |   | e |   |
            // +---+---+---+---+---+---+---+---+
            // |   |   |   | f |   |   |   |   |
            // +---+---+---+---+---+---+---+---+
            // |   | g |   |   |   |   |   | c |
            // +---+---+---+---+---+---+---+---+
            // |   |   |   |   |   | h |   |   |
            // +---+---+---+---+---+---+---+---+
            const a = vec2<f32>(7.0/16.0, -7.0/16.0);
            const b = -10.0/16.0;
            const c =  4.0/16.0;
            const d = -8.0/16.0;
            const e = 12.0/16.0;
            const f = -6.0/16.0;
            const g = -4.0/16.0;
            const h =  8.0/16.0;
            const dy = 2.0/16.0;

            upper += a;
            lower += a;
            wn0.x += s * rasterize_edge_ssaa(upper, lower);
            upper.x += b;
            upper.y += dy;
            lower.x += b;
            lower.y += dy;
            wn0.y += s * rasterize_edge_ssaa(upper, lower);
            upper.x += c;
            upper.y += dy;
            lower.x += c;
            lower.y += dy;
            wn0.z += s * rasterize_edge_ssaa(upper, lower);
            upper.x += d;
            upper.y += dy;
            lower.x += d;
            lower.y += dy;
            wn0.w += s * rasterize_edge_ssaa(upper, lower);
            upper.x += e;
            upper.y += dy;
            lower.x += e;
            lower.y += dy;
            wn1.x += s * rasterize_edge_ssaa(upper, lower);
            upper.x += f;
            upper.y += dy;
            lower.x += f;
            lower.y += dy;
            wn1.y += s * rasterize_edge_ssaa(upper, lower);
            upper.x += g;
            upper.y += dy;
            lower.x += g;
            lower.y += dy;
            wn1.z += s * rasterize_edge_ssaa(upper, lower);
            upper.x += h;
            upper.y += dy;
            lower.x += h;
            lower.y += dy;
            wn1.w += s * rasterize_edge_ssaa(upper, lower);
        } #else {
            // Position of this pixel's top-left corner (uv points to the pixel's center).
            // See comment in tiler.rs about the half-pixel offset.
            let pixel_offset = uv - vec2<f32>(0.5);

            // Move to coordinates local to the current pixel.
            var p0 = edge.xy - pixel_offset;
            var p1 = edge.zw - pixel_offset;

            winding_number += rasterize_edge_analytical(p0, p1);
        }
    }

    #if TILING_SSAA4 {
        var mask = 0.0;
        if ((fill_rule & 1u) == 0u) {
            mask = even_odd(wn.x) + even_odd(wn.y) + even_odd(wn.z) + even_odd(wn.w);
        } else {
            mask = non_zero(wn.x) + non_zero(wn.y) + non_zero(wn.z) + non_zero(wn.w);
        }

        mask *= 0.25;
    } #else if TILING_SSAA8 {
        var mask = 0.0;
        if ((fill_rule & 1u) == 0u) {
            mask = even_odd(wn0.x) + even_odd(wn0.y) + even_odd(wn0.z) + even_odd(wn0.w)
                + even_odd(wn1.x) + even_odd(wn1.y) + even_odd(wn1.z) + even_odd(wn1.w);
        } else {
            mask = non_zero(wn0.x) + non_zero(wn0.y) + non_zero(wn0.z) + non_zero(wn0.w)
                + non_zero(wn1.x) + non_zero(wn1.y) + non_zero(wn1.z) + non_zero(wn1.w);
        }

        mask *= 0.125;
    } #else {
        var mask = resolve_mask(winding_number, fill_rule);
    }

    // Debug: uncomment to see the grid in alpha tiles.
    //if (edges.x != edges.y && (min(uv.x, uv.y) < 1.0 || max(uv.x, uv.y) > 15.0)) {
    //    mask = 0.4;
    //}

    return mask * opacity;
}
