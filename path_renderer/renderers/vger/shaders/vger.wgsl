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
    padding: vec2<u32>,
) -> GeometryVertex {
    let uv = rect_get_uv(vertex_index);
    let device_position = mix(rect.xy, rect.zw, uv);

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
        cv_start,
        cv_count,
        fill_rule,
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

// Ray-casting test: does the horizontal ray from p to +inf cross segment A-C?
fn line_test(p: vec2<f32>, A: vec2<f32>, B: vec2<f32>) -> bool {
    let cs = i32(A.y < p.y) * 2 + i32(B.y < p.y);
    if cs == 0 || cs == 3 {
        return false;
    }
    let v = B - A;
    let t = (p.y - A.y) / v.y;
    return (A.x + t * v.x) > p.x;
}

// Is the point within the area between the curve and line segment A-C?
// Returns false for degenerate (collinear) cases.
fn bezier_test(p: vec2<f32>, A: vec2<f32>, B: vec2<f32>, C: vec2<f32>) -> bool {
    let v0 = B - A;
    let v1 = C - A;
    let v2 = p - A;
    let det = v0.x * v1.y - v1.x * v0.y;
    // Collinear points: no area between curve and chord.
    if abs(det) < 1e-6 {
        return false;
    }
    let s = (v2.x * v1.y - v1.x * v2.y) / det;
    let t = (v0.x * v2.y - v2.x * v0.y) / det;
    if s < 0.0 || t < 0.0 || (1.0 - s - t) < 0.0 {
        return false;
    }
    let u = s * 0.5 + t;
    let v = t;
    return u * u < v;
}

fn geometry_fragment(
    // Positions are transformed to device spac on the CPU.
    device_pos: vec2<f32>,
    seg_start: u32,
    seg_count: u32,
    fill_rule: u32, // TODO: implement the non-zero fill rule.
) -> f32 {
    let fw = length(fwidth(device_pos));

    var d = 1e10;
    var s: f32 = 1.0;

    for (var i: u32 = 0u; i < seg_count; i = i + 1u) {
        let j = seg_start + 3u * i;
        let a = curves[j];
        let b = curves[j + 1u];
        let c = curves[j + 2u];

        var skip = false;
        let xmax = device_pos.x + fw;
        let xmin = device_pos.x - fw;

        if a.x > xmax && b.x > xmax && c.x > xmax {
            skip = true;
        } else if a.x < xmin && b.x < xmin && c.x < xmin {
            skip = true;
        }

        if line_test(device_pos, a, c) {
            s = -s;
        }

        if !skip {
            d = min(d, sd_bezier(device_pos, a, b, c));
            if bezier_test(device_pos, a, b, c) {
                s = -s;
            }
        }
    }

    d = d * s;

    return 1.0 - smoothstep(-fw / 2.0, fw / 2.0, d);
}
