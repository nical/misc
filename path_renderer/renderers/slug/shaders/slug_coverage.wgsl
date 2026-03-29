// Slug winding-number coverage computation.
//
// Ported from https://github.com/EricLengyel/Slug

// Band texture width is fixed at 4096 texels.
const K_LOG_BAND_TEXTURE_WIDTH: u32 = 12u;

// Three-bit classification: determine which roots of the quadratic contribute
// to the winding number, based on the signs of y1, y2, y3.
fn calc_root_code(y1: f32, y2: f32, y3: f32) -> u32 {
    let i1 = bitcast<u32>(y1) >> 31u;
    let i2 = bitcast<u32>(y2) >> 30u;
    let i3 = bitcast<u32>(y3) >> 29u;
    var shift = (i2 & 2u) | (i1 & ~2u);
    shift = (i3 & 4u) | (shift & ~4u);
    return (0x2E74u >> shift) & 0x0101u;
}

// Solve for x-intercepts of a quadratic Bézier with a horizontal ray.
fn solve_horiz_poly(from_ctrl: vec4<f32>, to: vec2<f32>) -> vec2<f32> {
    let a = from_ctrl.xy - from_ctrl.zw * 2.0 + to;
    let b = from_ctrl.xy - from_ctrl.zw;
    let ra = 1.0 / a.y;
    let rb = 0.5 / b.y;
    let d = sqrt(max(b.y * b.y - a.y * from_ctrl.y, 0.0));
    var t1 = (b.y - d) * ra;
    var t2 = (b.y + d) * ra;
    if abs(a.y) < 1.0 / 65536.0 {
        t1 = from_ctrl.y * rb;
        t2 = t1;
    }
    return vec2<f32>(
        (a.x * t1 - b.x * 2.0) * t1 + from_ctrl.x,
        (a.x * t2 - b.x * 2.0) * t2 + from_ctrl.x,
    );
}

// Solve for y-intercepts of a quadratic Bézier with a vertical ray.
fn solve_vert_poly(from_ctrl: vec4<f32>, to: vec2<f32>) -> vec2<f32> {
    let a = from_ctrl.xy - from_ctrl.zw * 2.0 + to;
    let b = from_ctrl.xy - from_ctrl.zw;
    let ra = 1.0 / a.x;
    let rb = 0.5 / b.x;
    let d = sqrt(max(b.x * b.x - a.x * from_ctrl.x, 0.0));
    var t1 = (b.x - d) * ra;
    var t2 = (b.x + d) * ra;
    if abs(a.x) < 1.0 / 65536.0 {
        t1 = from_ctrl.x * rb;
        t2 = t1;
    }
    return vec2<f32>(
        (a.y * t1 - b.y * 2.0) * t1 + from_ctrl.y,
        (a.y * t2 - b.y * 2.0) * t2 + from_ctrl.y,
    );
}

fn band_address(shape_loc: vec2<i32>, offset: u32) -> vec2<i32> {
    var loc = vec2<i32>(shape_loc.x + i32(offset), shape_loc.y);
    loc.y += loc.x >> K_LOG_BAND_TEXTURE_WIDTH;
    loc.x &= (1 << K_LOG_BAND_TEXTURE_WIDTH) - 1;
    return loc;
}

fn calc_coverage(x_cov: f32, y_cov: f32, x_weight: f32, y_weight: f32) -> f32 {
    let coverage = max(
        abs(x_cov * x_weight + y_cov * y_weight) / max(x_weight + y_weight, 1.0 / 65536.0),
        min(abs(x_cov), abs(y_cov)),
    );

    // TODO: this is the non-zero fill rule, add support for even-odd.
    return clamp(coverage, 0.0, 1.0);
}

fn slug_render(
    render_coord: vec2<f32>,
    band_transform: vec4<f32>,
    shape_loc: vec2<i32>,
    band_max: vec2<i32>,
) -> f32 {
    let ems_per_pixel = fwidth(render_coord);
    let pixels_per_em = 1.0 / ems_per_pixel;

    let band_index = clamp(
        vec2i(render_coord * band_transform.xy + band_transform.zw),
        vec2i(0),
        band_max,
    );

    // Horizontal bands (fire ray in +x)
    var x_cov: f32 = 0.0;
    var x_weight: f32 = 0.0;

    let hband_uv = vec2i(shape_loc.x + band_index.y, shape_loc.y);
    let hband_data = textureLoad(band_texture, hband_uv, 0).xy;

    var curve_indirect_address = band_address(shape_loc, hband_data.y);

    for (var ci: i32 = 0; ci < i32(hband_data.x); ci++) {
        var curve_address = vec2i(textureLoad(band_texture, curve_indirect_address, 0).xy);
        curve_indirect_address.x += 1;

        // Fetch the quadratic curve.

        let from_ctrl = textureLoad(curve_texture, curve_address, 0) - vec4f(render_coord, render_coord);
        curve_address.x += 1;
        let to = textureLoad(curve_texture, curve_address, 0).xy - render_coord;

        if max(max(from_ctrl.x, from_ctrl.z), to.x) * pixels_per_em.x < -0.5 {
            break;
        }

        let code = calc_root_code(from_ctrl.y, from_ctrl.w, to.y);
        if code != 0u {
            let r = solve_horiz_poly(from_ctrl, to) * pixels_per_em.x;
            if (code & 1u) != 0u {
                x_cov += clamp(r.x + 0.5, 0.0, 1.0);
                x_weight = max(x_weight, clamp(1.0 - abs(r.x) * 2.0, 0.0, 1.0));
            }
            if code > 1u {
                x_cov -= clamp(r.y + 0.5, 0.0, 1.0);
                x_weight = max(x_weight, clamp(1.0 - abs(r.y) * 2.0, 0.0, 1.0));
            }
        }
    }

    // Vertical bands (fire ray in +y)
    var y_cov: f32 = 0.0;
    var y_weight: f32 = 0.0;

    let vband_uv = vec2<i32>(shape_loc.x + band_max.y + 1 + band_index.x, shape_loc.y);
    let vband_data = textureLoad(band_texture, vband_uv, 0).xy;

    curve_indirect_address = band_address(shape_loc, vband_data.y);

    for (var ci: i32 = 0; ci < i32(vband_data.x); ci++) {
        var curve_address = vec2i(textureLoad(band_texture, curve_indirect_address, 0).xy);
        curve_indirect_address.x += 1;

        let from_ctrl = textureLoad(curve_texture, curve_address, 0) - vec4<f32>(render_coord, render_coord);
        curve_address.x += 1;
        let to = textureLoad(curve_texture, curve_address, 0).xy - render_coord;

        if max(max(from_ctrl.y, from_ctrl.w), to.y) * pixels_per_em.y < -0.5 {
            break;
        }

        let code = calc_root_code(from_ctrl.x, from_ctrl.z, to.x);
        if code != 0u {
            let r = solve_vert_poly(from_ctrl, to) * pixels_per_em.y;
            if (code & 1u) != 0u {
                y_cov -= clamp(r.x + 0.5, 0.0, 1.0);
                y_weight = max(y_weight, clamp(1.0 - abs(r.x) * 2.0, 0.0, 1.0));
            }
            if code > 1u {
                y_cov += clamp(r.y + 0.5, 0.0, 1.0);
                y_weight = max(y_weight, clamp(1.0 - abs(r.y) * 2.0, 0.0, 1.0));
            }
        }
    }

    return calc_coverage(x_cov, y_cov, x_weight, y_weight);
}
