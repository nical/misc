#import render_task
#import rect
#import z_index

fn geometry_vertex(
    vertex_index: u32,
    rect: vec4<f32>,
    z_index: u32,
    pattern: u32,
    render_task: u32,
    cv_start: u32,
    cv_count: u32,
    fill_rule: u32,
    path_start: u32,
    _padding: u32,
) -> GeometryVertex {
    let uv = rect_get_uv(vertex_index);
    let device_position = mix(
        rect.xy,
        rect.zw, // -vec2(0.0, 1.0),
        uv,
    );

    let task = render_task_fetch(render_task);
    let target_position = render_task_target_position(task, device_position);

    let position = vec4<f32>(
        target_position.x,
        target_position.y,
        z_index_to_f32(z_index),
        1.0,
    );

    // TODO: render task clip.

    return GeometryVertex(
        position,
        device_position,
        pattern,
        device_position,
        vec4u(
            cv_start,
            cv_count,
            fill_rule,
            path_start,
        ),
    );
}

// Distance from point to line segment A-B.
fn sd_segment(p: vec2<f32>, A: vec2<f32>, B: vec2<f32>) -> f32 {
    let pa = p - A;
    let ba = B - A;
    let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}

// Distance to a quadratic Bezier curve, with fallback for degenerate (line) cases.
// Based on https://www.iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm
fn sd_bezier(pos: vec2<f32>, A: vec2<f32>, B: vec2<f32>, C: vec2<f32>) -> f32 {
    let a = B - A;
    let b = A - 2.0 * B + C;

    // Degenerate case: control points are collinear (curve is a line).
    let bb = dot(b, b);
    if bb < 1e-6 {
        return sd_segment(pos, A, C);
    }

    let c = a * 2.0;
    let d = A - pos;
    let kk = 1.0 / bb;
    let kx = kk * dot(a, b);
    let ky = kk * (2.0 * dot(a, a) + dot(d, b)) / 3.0;
    let kz = kk * dot(d, a);
    var res = 0.0;
    let p = ky - kx * kx;
    let p3 = p * p * p;
    let q = kx * (2.0 * kx * kx + -3.0 * ky) + kz;
    var h = q * q + 4.0 * p3;
    if h >= 0.0 {
        h = sqrt(h);
        let x = (vec2<f32>(h, -h) - q) / 2.0;
        let uv = sign(x) * pow(abs(x), vec2<f32>(1.0 / 3.0));
        let t = clamp(uv.x + uv.y - kx, 0.0, 1.0);
        res = dot(d + (c + b * t) * t, d + (c + b * t) * t);
    } else {
        let z = sqrt(-p);
        let v = acos(q / (p * z * 2.0)) / 3.0;
        let m = cos(v);
        let n = sin(v) * 1.732050808;
        let t = clamp(vec3<f32>(m + m, -n - m, n - m) * z - kx, vec3<f32>(0.0), vec3<f32>(1.0));
        res = min(
            dot(d + (c + b * t.x) * t.x, d + (c + b * t.x) * t.x),
            dot(d + (c + b * t.y) * t.y, d + (c + b * t.y) * t.y),
        );
    }
    return sqrt(res);
}

// Ray-casting test: does the horizontal ray from p to -inf cross segment A-B?
fn line_test(p: vec2<f32>, A: vec2<f32>, B: vec2<f32>) -> f32 {
    let cs = i32(A.y < p.y) * 2 + i32(B.y < p.y);
    if cs == 0 || cs == 3 {
        return 0.0;
    }
    let v = B - A;
    let t = (p.y - A.y) / v.y;

    if (A.x + t * v.x) < p.x {
        return 1.0;
    }

    return 0.0;
}

// Is the point within the area between the curve and line segment A-C?
// Returns false for degenerate (collinear) cases.
fn bezier_test(p: vec2<f32>, A: vec2<f32>, B: vec2<f32>, C: vec2<f32>) -> bool {
    let v0 = B - A;
    let v1 = C - A;
    let det = v0.x * v1.y - v1.x * v0.y;
    // Collinear points: no area between curve and chord.
    if abs(det) < 1e-6 {
        return false;
    }

    let v2 = p - A;
    let s = (v2.x * v1.y - v1.x * v2.y) / det;
    let t = (v0.x * v2.y - v2.x * v0.y) / det;
    if s < 0.0 || t < 0.0 || (1.0 - s - t) < 0.0 {
        return false;
    }
    let u = s * 0.5 + t;
    let v = t;
    return u * u < v;
}

fn rasterize_edge_ssaa(a: vec2f, b: vec2f, c: vec2f) -> vec4f {
    let sign = sign(c.y - a.y);
    let select = step(0.0, sign);
    var upper = mix(a, c, 1.0 - select);
    var lower = mix(a, c, select);
    var ctrl = b;

    let v0 = ctrl - upper;
    let v1 = lower - upper;
    let det = v0.x * v1.y - v1.x * v0.y;
    var inv_det = 0.0;
    if abs(det) > 1e-6 {
        inv_det = 1.0 / det;
    }

    // Sample offsets:
    // +---+---+---+---+
    // |   |   | 0 |   |
    // +---+---+---+---+
    // | 1 |   |   |   |
    // +---+---p---+---+
    // |   |   |   | 2 |
    // +---+---+---+---+
    // |   | 3 |   |   |
    // +---+---+---+---+
    const s0 = vec2<f32>( 1.0/8.0, -3.0/8.0);
    const s1 = vec2<f32>(-3.0/8.0, -1.0/8.0);
    const s2 = vec2<f32>( 3.0/8.0,  1.0/8.0);
    const s3 = vec2<f32>(-1.0/8.0,  3.0/8.0);

    upper += s0;
    ctrl += s0;
    lower += s0;
    let wn0 = sign * rasterize_edge_ssaa_impl(upper, ctrl, lower, inv_det);
    upper += (s1 - s0);
    ctrl += (s1 - s0);
    lower += (s1 - s0);
    let wn1 = sign * rasterize_edge_ssaa_impl(upper, ctrl, lower, inv_det);
    upper += (s2 - s1);
    ctrl += (s2 - s1);
    lower += (s2 - s1);
    let wn2 = sign * rasterize_edge_ssaa_impl(upper, ctrl, lower, inv_det);
    upper += (s3 - s2);
    ctrl += (s3 - s2);
    lower += (s3 - s2);
    let wn3 = sign * rasterize_edge_ssaa_impl(upper, ctrl, lower, inv_det);

    return vec4f(wn0, wn1, wn2, wn3);
}

fn rasterize_edge_ssaa_impl(upper: vec2f, b: vec2f, lower: vec2f, inv_det: f32) -> f32 {

    // line test

    // 1 if the sample is in the edge's y range, 0 otherwise.
    let y_range_test = step(upper.y, 0.0) * step(0.0, lower.y);

    let v0 = b - upper;
    let v1 = lower - upper;
    let v2 = -upper;

    let v2_cross_v1 = v2.x * v1.y - v1.x * v2.y;

    let is_right_side = step(0.0, v2_cross_v1);
    let line_test = y_range_test * is_right_side;

    // bezier test

    let s = v2_cross_v1 * inv_det;
    let t = (v0.x * v2.y - v2.x * v0.y) * inv_det;

    // 1 if we are inside of the trangle a b c, 0 otherwise.
    let triangle_test = step(0.0, s) * step(0.0, t) * step(0.0, 1.0 - s - t);

    var bezier_test = 0.0;
    let u = s * 0.5 + t;
    // This is similar to step(u*u, t) but behaves differently when u * u == t.
    if u * u < t {
        bezier_test = 1.0;
    }

    return abs(line_test - triangle_test * bezier_test);
}


fn geometry_fragment(
    // Positions are transformed to device spac on the CPU.
    pos: vec2<f32>,
    quad_data: vec4u,
) -> f32 {
    let fw = length(fwidth(pos));

    let seg_start = quad_data.x;
    let seg_count = quad_data.y;
    let fill_rule = quad_data.z; // TODO: implement the non-zero fill rule.
    let path_start = quad_data.w;

    #if BANDS_SSAA4 {
        // Note: This could be packed into a single u32.
        var wn = vec4<f32>(0.0);
    } #else {
        var wn = 0.0;
        var dist = 1.0;
    }

    for (var i: u32 = 0u; i < seg_count; i = i + 1u) {
        let j = seg_start + i;
        // Must match the same constant in bands/resource.rs
        const DATA_TEXTURE_WIDTH: u32 = 4096;

        let edge_index_uv = vec2i(
            i32(j % DATA_TEXTURE_WIDTH),
            i32(j / DATA_TEXTURE_WIDTH),
        );

        let edge_index = path_start + textureLoad(curve_indices, edge_index_uv, 0).x;
        let edge_uv = vec2i(
            i32(edge_index % DATA_TEXTURE_WIDTH),
            i32(edge_index / DATA_TEXTURE_WIDTH),
        );
        let edge_index2 = edge_index + 1;
        let edge_uv2 = vec2i(
            i32(edge_index2 % DATA_TEXTURE_WIDTH),
            i32(edge_index2 / DATA_TEXTURE_WIDTH),
        );

        let from_ctrl = textureLoad(curves, edge_uv, 0);
        let a = from_ctrl.xy;
        let b = from_ctrl.zw;
        let c = textureLoad(curves, edge_uv2, 0).xy;

        #if BANDS_SSAA4 {
            wn += rasterize_edge_ssaa(a - pos, b - pos, c - pos);
        } #else {
            let sign = sign(c.y - a.y);
            let lt = line_test(pos, a, c);

            let xmax = pos.x + fw;
            let xmin = pos.x - fw;

            var skip = false;
            if a.x > xmax && b.x > xmax && c.x > xmax {
                skip = true;
            } else if a.x < xmin && b.x < xmin && c.x < xmin {
                skip = true;
            }

            var d = 1.0;
            var bt = 0.0;
            if !skip {
                dist = min(dist, sd_bezier(pos, a, b, c));
                if bezier_test(pos, a, b, c) {
                    bt = 1.0;
                }
            }

            let cov = sign * abs(lt - bt);
            wn += cov;
        }
    }

    // Resolve coverage.

    var coverage = 0.0;
    #if BANDS_SSAA4 {
        if ((fill_rule & 1u) == 0u) {
            coverage = even_odd(wn.x) + even_odd(wn.y) + even_odd(wn.z) + even_odd(wn.w);
        } else {
            coverage = non_zero(wn.x) + non_zero(wn.y) + non_zero(wn.z) + non_zero(wn.w);
        }
        coverage *= 0.25;
    } #else {
        if ((fill_rule & 1u) == 0u) {
            coverage = even_odd(wn);
        } else {
            coverage = non_zero(wn);
        }

        // TODO: that's not quite right. I think this only modulates
        // coverage inside of the shape.
        coverage = min(coverage, dist);
    }

    // Uncomment this line to see the pixels covered by the quad for debugging.
    //coverage = max(coverage, 0.2);
    return coverage;
}

fn even_odd(winding_number: f32) -> f32 {
    return 1.0 - abs((abs(winding_number) % 2.0) - 1.0);
}

fn non_zero(winding_number: f32) -> f32 {
    return min(abs(winding_number), 1.0);
}
