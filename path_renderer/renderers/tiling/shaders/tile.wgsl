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
    path.render_task = encoded0.w;

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

// The winding number contribution of one edge to one sample.
//
// `offset` is the sample's position, `edge` the vector from the upper to the lower
// endpoint of the edge, `winding` the edge's winding direction, and `cross_at_origin`
// the cross product of `edge` with the vector from the sample pattern's origin to the
// upper endpoint. `y_min` and `y_max` are the edge's y range in the sample pattern's
// coordinate system.
//
// The two quantities the test needs are affine in `offset`, so everything that does
// not depend on the sample is computed once per edge by the caller and only two fused
// multiply-adds and three comparisons are left here.
fn ssaa_sample(
    offset: vec2<f32>,
    edge: vec2<f32>,
    cross_at_origin: f32,
    y_min: f32,
    y_max: f32,
    winding: f32,
) -> f32 {
    // The sample is between the two endpoints on the y axis.
    let in_y_range = offset.y >= y_min && offset.y <= y_max;
    // The sample is on the right hand side of the edge.
    let cross = cross_at_origin + edge.x * offset.y - edge.y * offset.x;

    return select(0.0, winding, in_y_range && cross >= 0.0);
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

            // A horizontal edge contributes zero to the winding number of every
            // sample. Skipping it here also means the orientation below is decided
            // by a strict comparison.
            if (p0.y == p1.y) {
                continue;
            }

            let downwards = p1.y > p0.y;
            let s = select(-1.0, 1.0, downwards);
            let upper = select(p1, p0, downwards) - uv;
            let lower = select(p0, p1, downwards) - uv;

            // The offsets below are applied to the edge rather than to the sample, so
            // the edge's y range in the sample pattern's coordinate system is the
            // negated and flipped y range of the edge.
            let y_min = -lower.y;
            let y_max = -upper.y;

            // The four samples sit on rows 3/8 and 1/8 above and below the pixel
            // center. An edge whose y range does not reach them covers none of them
            // and cannot change any of the winding numbers. For a given pixel most
            // edges of the tile are several rows away, so this is the common case.
            if (y_min > 3.0/8.0 || y_max < -3.0/8.0) {
                continue;
            }

            // Sample offsets:
            // +---+---+---+---+
            // |   |   | a |   |
            // +---+---+---+---+
            // | b |   |   |   |
            // +---+---o---+---+
            // |   |   |   | c |
            // +---+---+---+---+
            // |   | d |   |   |
            // +---+---+---+---+
            // These are offsets applied to the edge, so the sample pattern is this
            // grid mirrored through the pixel center. Both are rotated grids with one
            // sample per row and per column, which is what matters here.
            const a = vec2<f32>( 1.0/8.0, -3.0/8.0);
            const b = vec2<f32>(-3.0/8.0, -1.0/8.0);
            const c = vec2<f32>( 3.0/8.0,  1.0/8.0);
            const d = vec2<f32>(-1.0/8.0,  3.0/8.0);

            let v = lower - upper;
            let cross_at_origin = v.x * upper.y - upper.x * v.y;

            wn.x += ssaa_sample(a, v, cross_at_origin, y_min, y_max, s);
            wn.y += ssaa_sample(b, v, cross_at_origin, y_min, y_max, s);
            wn.z += ssaa_sample(c, v, cross_at_origin, y_min, y_max, s);
            wn.w += ssaa_sample(d, v, cross_at_origin, y_min, y_max, s);
        } #else if TILING_SSAA8 {
            var p0 = edge.xy;
            var p1 = edge.zw;

            // See the comment in the ssaa4 branch.
            if (p0.y == p1.y) {
                continue;
            }

            let downwards = p1.y > p0.y;
            let s = select(-1.0, 1.0, downwards);
            let upper = select(p1, p0, downwards) - uv;
            let lower = select(p0, p1, downwards) - uv;

            // See the comment in the ssaa4 branch.
            let y_min = -lower.y;
            let y_max = -upper.y;

            // The eight samples sit on rows 7/16, 5/16, 3/16 and 1/16 above and below
            // the pixel center. See the comment in the ssaa4 branch.
            if (y_min > 7.0/16.0 || y_max < -7.0/16.0) {
                continue;
            }

            // Sample offsets:
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
            // |   | g |   |   |   |   |   |   |
            // +---+---+---+---+---+---+---+---+
            // |   |   |   |   |   | h |   |   |
            // +---+---+---+---+---+---+---+---+
            // As in the ssaa4 branch, these are offsets applied to the edge, so the
            // sample pattern is this grid mirrored through the pixel center.
            const a = vec2<f32>( 7.0/16.0, -7.0/16.0);
            const b = vec2<f32>(-3.0/16.0, -5.0/16.0);
            const c = vec2<f32>( 1.0/16.0, -3.0/16.0);
            const d = vec2<f32>(-7.0/16.0, -1.0/16.0);
            const e = vec2<f32>( 5.0/16.0,  1.0/16.0);
            const f = vec2<f32>(-1.0/16.0,  3.0/16.0);
            const g = vec2<f32>(-5.0/16.0,  5.0/16.0);
            const h = vec2<f32>( 3.0/16.0,  7.0/16.0);

            let v = lower - upper;
            let cross_at_origin = v.x * upper.y - upper.x * v.y;

            wn0.x += ssaa_sample(a, v, cross_at_origin, y_min, y_max, s);
            wn0.y += ssaa_sample(b, v, cross_at_origin, y_min, y_max, s);
            wn0.z += ssaa_sample(c, v, cross_at_origin, y_min, y_max, s);
            wn0.w += ssaa_sample(d, v, cross_at_origin, y_min, y_max, s);
            wn1.x += ssaa_sample(e, v, cross_at_origin, y_min, y_max, s);
            wn1.y += ssaa_sample(f, v, cross_at_origin, y_min, y_max, s);
            wn1.z += ssaa_sample(g, v, cross_at_origin, y_min, y_max, s);
            wn1.w += ssaa_sample(h, v, cross_at_origin, y_min, y_max, s);
        } #else {
            // uv is a position in pixels relative to the tile's origin, sampled at the
            // center of the pixel. rasterize_edge_analytical works in coordinates local
            // to the pixel, with the pixel covering the unit square, so shift by half a
            // pixel to get the top-left corner of the pixel.
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
