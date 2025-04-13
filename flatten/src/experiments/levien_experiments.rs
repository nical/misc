//! A so far unsuccessful attempt at speeding up the quadratic bézier flattening code using SSE
//! instructions.
//! I was hoping that batching approx_parabola_integral and approx_parabola_inv_integral in pairs
//! would halve the cost of the expensive square roots but it doesn't make a significant difference
//! in the profiles so far.

use arrayvec::ArrayVec;
use lyon_path::geom::{CubicBezierSegment, LineSegment};
use lyon_path::math::{point, Point, Vector};

#[cfg(target_arch = "x86")]
use std::arch::x86 as arch;
#[cfg(target_arch = "x86_64")]
use std::f32;

use crate::simd4::*;

use crate::levien::FlatteningParams;
use crate::{polynomial_form_cubic, QuadraticBezierPolynomial};
use crate::fast_recip;

#[inline(always)]
unsafe fn approx_parabola_integral(x: f32x4) -> f32x4 {
    let d_pow4 = splat(0.67 * 0.67 * 0.67 * 0.67);
    let sqr = sqrt(sqrt(add(d_pow4, mul(splat(0.25), mul(x, x)))));
    div(x, add(splat(1.0 - 0.67), sqr))
}

#[inline(always)]
unsafe fn approx_parabola_inv_integral(x: f32x4) -> f32x4 {
    mul(x, add(splat(1.0 - 0.39), sqrt(add(splat(0.39 * 0.39), mul(splat(0.25), mul(x, x))))))
}

#[inline(never)]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx"))]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
unsafe fn flattening_params_simd4(
    curve: &CubicBezierSegment<f32>,
    s_num_quads: f32,
    s_first_quad: f32,
    s_quad_step: f32,
    sqrt_tolerance: f32,
    output: &mut ArrayVec<(FlatteningParams, QuadraticBezierPolynomial), 16>
) -> (Point, f32) {
    let quad_step = splat(s_quad_step);
    let t0 = mul(add(splat(s_first_quad), vec4(0.0, 1.0, 2.0, 3.0)), quad_step);
    let t1 = add(t0, quad_step);

    // polynomial_form:
    let s_poly = polynomial_form_cubic(curve);
    let a0_x = splat(s_poly.a0.x);
    let a0_y = splat(s_poly.a0.y);
    let a1_x = splat(s_poly.a1.x);
    let a1_y = splat(s_poly.a1.y);
    let a2_x = splat(s_poly.a2.x);
    let a2_y = splat(s_poly.a2.y);
    let a3_x = splat(s_poly.a3.x);
    let a3_y = splat(s_poly.a3.y);

    let (from_x, from_y) = sample_cubic_horner_simd4(a0_x, a0_y, a1_x, a1_y, a2_x, a2_y, a3_x, a3_y, t0);
    // TODO: 3 out of 4 values are already in from_
    let (to_x, to_y) = sample_cubic_horner_simd4(a0_x, a0_y, a1_x, a1_y, a2_x, a2_y, a3_x, a3_y, t1);

    // Compute the control points of the sub-curves.

    // CubicBezierSegment::split_range
    let qx0 = splat(curve.ctrl1.x - curve.from.x);
    let qy0 = splat(curve.ctrl1.y - curve.from.y);
    let qx1 = splat(curve.ctrl2.x - curve.ctrl1.x);
    let qy1 = splat(curve.ctrl2.y - curve.ctrl1.y);
    let qx2 = splat(curve.to.x - curve.ctrl2.x);
    let qy2 = splat(curve.to.y - curve.ctrl2.y);

    let one = splat(1.0);
    let two = splat(2.0);

    // QuadraticBezierSegment::sample(t0)
    let t0_2 = mul(t0, t0);
    let one_t0 = sub(one, t0);
    let one_t0_2 = mul(one_t0, one_t0);
    let two_t0_one_t0 = mul(two, mul(one_t0, t0));
    let qx = add(mul_add(qx0, one_t0_2, mul(qx1, two_t0_one_t0)), mul(qx2, t0_2));
    let qy = add(mul_add(qy0, one_t0_2, mul(qy1, two_t0_one_t0)), mul(qy2, t0_2));

    let ctrl1_x = mul_add(qx, quad_step, from_x);
    let ctrl1_y = mul_add(qy, quad_step, from_y);

    // QuadraticBezierSegment::sample(t1)
    let t1_2 = mul(t1, t1);
    let one_t1 = sub(one, t1);
    let one_t1_2 = mul(one_t1, one_t1);
    let two_t1_one_t1 = mul(two, mul(one_t1, t1));
    let qx = add(mul_add(qx0, one_t1_2, mul(qx1, two_t1_one_t1)), mul(qx2, t1_2));
    let qy = add(mul_add(qy0, one_t1_2, mul(qy1, two_t1_one_t1)), mul(qy2, t1_2));

    let ctrl2_x = sub(to_x, mul(qx, quad_step));
    let ctrl2_y = sub(to_y, mul(qy, quad_step));

    // Approximate the sub-curves with quadratics
    let three = splat(3.0);
    let half = splat(0.5);
    let c1x = mul(mul_sub(ctrl1_x, three, from_x), half);
    let c1y = mul(mul_sub(ctrl1_y, three, from_y), half);
    let c2x = mul(mul_sub(ctrl2_x, three, to_x), half);
    let c2y = mul(mul_sub(ctrl2_y, three, to_y), half);

    let ctrl_x = mul(add(c1x, c2x), half);
    let ctrl_y = mul(add(c1y, c2y), half);

    // Now that we have our four quadratics, compute the flattening parameters
    let ddx = sub(mul_sub(two, ctrl_x, from_x), to_x);
    let ddy = sub(mul_sub(two, ctrl_y, from_y), to_y);
    //println!("ddx {ddx:?} ddy {ddy:?}");
    let cross = mul_sub(sub(to_x, from_x), ddy, mul(sub(to_y, from_y), ddx));
    //println!("cross {cross:?}");
    let rcp_cross = recip(cross);
    let parabola_from = mul(mul_add(sub(ctrl_x, from_x), ddx, mul(sub(ctrl_y, from_y), ddy)), rcp_cross);
    let parabola_to = mul(mul_add(sub(to_x, ctrl_x), ddx, mul(sub(to_y, ctrl_y), ddy)), rcp_cross);
    let parabola_diff = sub(parabola_to, parabola_from);
    let scale = abs(div(cross, mul(sqrt(mul_add(ddx, ddx, mul(ddy, ddy))), parabola_diff)));

    let integral_from = approx_parabola_integral(parabola_from);
    let integral_to = approx_parabola_integral(parabola_to);
    //println!("parabola_to {parabola_to:?} parabola_from {parabola_from:?}");
    //println!("intergal_to {integral_to:?} integral_from {integral_from:?}");

    let inv_integral_from = approx_parabola_inv_integral(integral_from);
    let inv_integral_to = approx_parabola_inv_integral(integral_to);

    let div_inv_integral_diff = recip(sub(inv_integral_to, inv_integral_from));

    let sqrt_tolerance = splat(sqrt_tolerance);
    let integral_diff = abs(sub(integral_to, integral_from));
    let sqrt_scale = sqrt(scale);

    let mut scaled_count = mul(integral_diff, sqrt_scale);

    let is_cusp = neq(signum(parabola_from), signum(parabola_to));
    // Handle a cusp case (segment contains curvature maximum)
    // Assuming the cusp case is fairly rare and because it is expensive
    // we add a branch to test wether any of the lanes need to take it
    // and skip it otherwise.
    if any(is_cusp) {
        let xmin = div(sqrt_tolerance, sqrt_scale);
        let scaled_count2 = mul(sqrt_tolerance, div(integral_diff, approx_parabola_integral(xmin)));
        scaled_count = select(is_cusp, scaled_count2, scaled_count);
    }

    // Handle another kind of cusp.
    scaled_count = select_or_zero(is_finite(scale), scaled_count);

    // Convert the quadratic curves into polynomial form.
    let a1_x = mul(sub(ctrl_x, from_x), two);
    let a1_y = mul(sub(ctrl_y, from_y), two);
    let a2_x = sub(add(from_x, to_x), mul(ctrl_x, two));
    let a2_y = sub(add(from_y, to_y), mul(ctrl_y, two));

    // Go back to scalar land to produce the output.

    let scaled_count = unpack(scaled_count);
    let integral_from = unpack(integral_from);
    let integral_to = unpack(integral_to);
    let inv_integral_from = unpack(inv_integral_from);
    let div_inv_integral_diff = unpack(div_inv_integral_diff);

    let a0_x = unpack(from_x);
    let a0_y = unpack(from_y);
    let a1_x = unpack(a1_x);
    let a1_y = unpack(a1_y);
    let a2_x = unpack(a2_x);
    let a2_y = unpack(a2_y);
    let to_x = unpack(to_x);
    let to_y = unpack(to_y);

    let mut sum = scaled_count.0;

    assert!(output.capacity() >= output.len() + 4);

    let count = s_num_quads as u32;
    output.push((
        FlatteningParams {
            scaled_count: scaled_count.0,
            integral_from: integral_from.0,
            integral_to: integral_to.0,
            inv_integral_from: inv_integral_from.0,
            div_inv_integral_diff: div_inv_integral_diff.0,
        },
        QuadraticBezierPolynomial {
            a0: Vector::new(a0_x.0, a0_y.0),
            a1: Vector::new(a1_x.0, a1_y.0),
            a2: Vector::new(a2_x.0, a2_y.0),
        },
    ));

    if count < 2 {
        return (point(to_x.0, to_y.0), sum);
    }
    sum += scaled_count.1;
    output.push((
        FlatteningParams {
            scaled_count: scaled_count.1,
            integral_from: integral_from.1,
            integral_to: integral_to.1,
            inv_integral_from: inv_integral_from.1,
            div_inv_integral_diff: div_inv_integral_diff.1,
        },
        QuadraticBezierPolynomial {
            a0: Vector::new(a0_x.1, a0_y.1),
            a1: Vector::new(a1_x.1, a1_y.1),
            a2: Vector::new(a2_x.1, a2_y.1),
        },
    ));

    if count < 3 {
        return (point(to_x.1, to_y.1), sum);
    }

    sum += scaled_count.2;
    output.push((
        FlatteningParams {
            scaled_count: scaled_count.2,
            integral_from: integral_from.2,
            integral_to: integral_to.2,
            inv_integral_from: inv_integral_from.2,
            div_inv_integral_diff: div_inv_integral_diff.2,
        },
        QuadraticBezierPolynomial {
            a0: Vector::new(a0_x.2, a0_y.2),
            a1: Vector::new(a1_x.2, a1_y.2),
            a2: Vector::new(a2_x.2, a2_y.2),
        },
    ));

    if count < 4 {
        return (point(to_x.2, to_y.2), sum);
    }

    sum += scaled_count.3;
    output.push((
        FlatteningParams {
            scaled_count: scaled_count.3,
            integral_from: integral_from.3,
            integral_to: integral_to.3,
            inv_integral_from: inv_integral_from.3,
            div_inv_integral_diff: div_inv_integral_diff.3,
        },
        QuadraticBezierPolynomial {
            a0: Vector::new(a0_x.3, a0_y.3),
            a1: Vector::new(a1_x.3, a1_y.3),
            a2: Vector::new(a2_x.3, a2_y.3),
        },
    ));

    (point(to_x.3, to_y.3), sum)
}


/// Was a quick attempt at packing simd values accross quads of the
/// flattening loop.
/// I did not try very hard because doing the packing is expensive
/// enough to offset potential gains from being able to process more
/// line segments (and more approx_parabola_inv_integral) per iteration
#[inline(never)]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx"))]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
pub unsafe fn flatten_cubic_simd4_merged_quads(curve: &CubicBezierSegment<f32>, tolerance: f32, cb: &mut dyn FnMut(&LineSegment<f32>)) {

    let quads_tolerance = tolerance * 0.1;
    let flatten_tolerance = tolerance * 0.9;
    let sqrt_flatten_tolerance = flatten_tolerance.sqrt();
    let inv_sqrt_flatten_tolerance = fast_recip(sqrt_flatten_tolerance);

    let num_quadratics = crate::levien::num_quadratics_impl(curve, quads_tolerance);
    //println!("{num_quadratics:?} quads");

    //counters_inc(num_quadratics.max(0.0) as usize);

    let mut quads: ArrayVec<(FlatteningParams, QuadraticBezierPolynomial), 16> = ArrayVec::new();

    let mut tmp_recip_scaled_count = AlignedBuf::new();
    let mut tmp_quad_a0x = AlignedBuf::new();
    let mut tmp_quad_a1x = AlignedBuf::new();
    let mut tmp_quad_a2x = AlignedBuf::new();
    let mut tmp_quad_a0y = AlignedBuf::new();
    let mut tmp_quad_a1y = AlignedBuf::new();
    let mut tmp_quad_a2y = AlignedBuf::new();
    let mut tmp_integral_from = AlignedBuf::new();
    let mut tmp_integral_diff = AlignedBuf::new();
    let mut tmp_inv_integral_from = AlignedBuf::new();
    let mut tmp_div_inv_integral_diff = AlignedBuf::new();
    let mut tmp_scaled_count_sum = AlignedBuf::new();

    let quad_step = fast_recip(num_quadratics);
    let num_quadratics = num_quadratics as u32;
    let mut quad_idx = 0;
    let mut from = curve.from;

    loop {
        let mut sum = 0.0;
        let mut quads_last_to = point(0.0, 0.0);
        while quad_idx < num_quadratics && quads.capacity() > quads.len() {
            let (to, s) = flattening_params_simd4(
                curve,
                (num_quadratics - quad_idx).min(4) as f32,
                quad_idx as f32,
                quad_step,
                sqrt_flatten_tolerance,
                &mut quads
            );
            sum += s;
            quads_last_to = to;

            //println!(" + {quad:?} {t0} .. {t1} scaled count: {:?}", params.scaled_count);
            quad_idx += 4;
        }

        let num_edges = ((0.5 * sum * inv_sqrt_flatten_tolerance).ceil() as u32).max(1);

        // Iterate through the quadratics, outputting the points of
        // subdivisions that fall within that quadratic.
        let step = sum / (num_edges as f32);
        let mut i = 1;
        let mut scaled_count_sum = 0.0;
        let v_step = splat(step);

        let mut v_i = vec4(1.0, 2.0, 3.0, 4.0);

        let mut lane = 0;
        //println!("------ {num_edges:?} edges step {step:?}");
        for (quad_idx, (params, quad)) in quads.iter().enumerate() {
            let n = u32::min(num_edges, ((scaled_count_sum + params.scaled_count) / step).ceil() as u32);
            let recip_scaled_count = fast_recip(params.scaled_count);

            while i < n {
                while i < n && lane < 4 {
                    *tmp_recip_scaled_count.ptr(lane) = recip_scaled_count;
                    *tmp_quad_a0x.ptr(lane) = quad.a0.x;
                    *tmp_quad_a0y.ptr(lane) = quad.a0.y;
                    *tmp_quad_a1x.ptr(lane) = quad.a1.x;
                    *tmp_quad_a1y.ptr(lane) = quad.a1.y;
                    *tmp_quad_a2x.ptr(lane) = quad.a2.x;
                    *tmp_quad_a2y.ptr(lane) = quad.a2.y;
                    *tmp_integral_from.ptr(lane) = params.integral_from;
                    *tmp_integral_diff.ptr(lane) = params.integral_to - params.integral_from;
                    *tmp_inv_integral_from.ptr(lane) = params.inv_integral_from;
                    *tmp_div_inv_integral_diff.ptr(lane) = params.div_inv_integral_diff;
                    *tmp_scaled_count_sum.ptr(lane) = scaled_count_sum;

                    i += 1;
                    lane += 1;
                }

                if lane == 4 || quad_idx == quads.len() - 1 {
                    let recip_scaled_count = aligned_load(tmp_recip_scaled_count.ptr(0));
                    let quad_a0x = aligned_load(tmp_quad_a0x.ptr(0));
                    let quad_a0y = aligned_load(tmp_quad_a0y.ptr(0));
                    let quad_a1x = aligned_load(tmp_quad_a1x.ptr(0));
                    let quad_a1y = aligned_load(tmp_quad_a1y.ptr(0));
                    let quad_a2x = aligned_load(tmp_quad_a2x.ptr(0));
                    let quad_a2y = aligned_load(tmp_quad_a2y.ptr(0));
                    let integral_from = aligned_load(tmp_integral_from.ptr(0));
                    let integral_diff = aligned_load(tmp_integral_diff.ptr(0));
                    let inv_integral_from = aligned_load(tmp_inv_integral_from.ptr(0));
                    let div_inv_integral_diff = aligned_load(tmp_div_inv_integral_diff.ptr(0));
                    let v_scaled_count_sum = aligned_load(tmp_scaled_count_sum.ptr(0));

                    let targets = mul(v_i, v_step);
                    let u = mul(sub(targets, v_scaled_count_sum), recip_scaled_count);

                    let u = approx_parabola_inv_integral(mul_add(integral_diff, u, integral_from));
                    let t = mul(sub(u, inv_integral_from), div_inv_integral_diff);

                    let px = sample_quadratic_horner_simd4(quad_a0x, quad_a1x, quad_a2x, t);
                    let py = sample_quadratic_horner_simd4(quad_a0y, quad_a1y, quad_a2y, t);

                    let px: [f32; 4] = std::mem::transmute(px);
                    let py: [f32; 4] = std::mem::transmute(py);

                    for (x, y) in px.iter().zip(py.iter()).take(lane) {
                        let to = point(*x, *y);
                        cb(&LineSegment { from, to });
                        from = to;
                    }

                    v_i = add(v_i, splat(4.0));
                    lane = 0;
                }
            }

            scaled_count_sum += params.scaled_count;
        }

        cb(&LineSegment { from, to: quads_last_to });
        if quad_idx >= num_quadratics {
            break;
        }

        quads.clear();
    }
}

const NUM_QUADS: usize = 16;

pub struct FlatteningParamsSimd {
    //quads: ArrayVec<(FlatteningParams, QuadraticBezierPolynomial), 16>,
    count: usize,
    a0_x: AlignedBuf,
    a0_y: AlignedBuf,
    a1_x: AlignedBuf,
    a1_y: AlignedBuf,
    a2_x: AlignedBuf,
    a2_y: AlignedBuf,
    scaled_count: AlignedBuf,
    integral_from: AlignedBuf,
    integral_to: AlignedBuf,
    inv_integral_from: AlignedBuf,
    div_inv_integral_diff: AlignedBuf,
}


/// This version uses a point buffer to avoid invoking the callback during
/// the hot flattening loop (and use supposedly more efficient alined_load
/// instead of unpacking the vectors), but that was a regression.
#[inline(never)]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx"))]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
pub unsafe fn flatten_cubic_simd4_with_point_buffer(curve: &CubicBezierSegment<f32>, tolerance: f32, cb: &mut dyn FnMut(&LineSegment<f32>)) {

    let quads_tolerance = tolerance * 0.1;
    let flatten_tolerance = tolerance * 0.9;
    let sqrt_flatten_tolerance = flatten_tolerance.sqrt();
    let inv_sqrt_flatten_tolerance = fast_recip(sqrt_flatten_tolerance);

    let num_quadratics = crate::levien::num_quadratics_impl(curve, quads_tolerance);
    //println!("{num_quadratics:?} quads");

    let mut quads: ArrayVec<(FlatteningParams, QuadraticBezierPolynomial), 16> = ArrayVec::new();
    let mut point_buffer: Aligned<[f32; 64]> = Aligned([0.0; 64]);

    let quad_step = fast_recip(num_quadratics);
    let num_quadratics = num_quadratics as u32;
    let mut quad_idx = 0;
    let mut from = curve.from;

    loop {
        let mut sum = 0.0;
        let mut quads_last_to = point(0.0, 0.0);
        while quad_idx < num_quadratics && quads.capacity() > quads.len() {
            let (to, s) = flattening_params_simd4(
                curve,
                (num_quadratics - quad_idx).min(4) as f32,
                quad_idx as f32,
                quad_step,
                sqrt_flatten_tolerance,
                &mut quads
            );
            sum += s;
            quads_last_to = to;

            //println!(" + {quad:?} {t0} .. {t1} scaled count: {:?}", params.scaled_count);
            quad_idx += 4;
        }

        let num_edges = ((0.5 * sum * inv_sqrt_flatten_tolerance).ceil() as u32).max(1);

        // Iterate through the quadratics, outputting the points of
        // subdivisions that fall within that quadratic.
        let step = sum / (num_edges as f32);
        let mut i = 1;
        let mut scaled_count_sum = 0.0;
        let v_step = splat(step);
        //println!("------ {num_edges:?} edges step {step:?}");
        for (params, quad) in &quads {
            let recip_scaled_count = splat(fast_recip(params.scaled_count));
            let n = u32::min(num_edges, ((scaled_count_sum + params.scaled_count) / step).ceil() as u32);

            if i < n {
                let quad_a0 = interleave_splat(quad.a0.x, quad.a0.y);
                let quad_a1 = interleave_splat(quad.a1.x, quad.a1.y);
                let quad_a2 = interleave_splat(quad.a2.x, quad.a2.y);

                while i < n {
                    // Set up the point buffer
                    let mut points_cap = 64;
                    let mut points_out_ptr = point_buffer.0.as_mut_ptr();
                    let copy_count = (points_cap / 2).min((n - i) as i32) as usize;

                    while i < n && points_cap >= 4 {
                        let i1 = i as f32;
                        let i2 = i1 + 1.0;
                        let targets = mul(vec4(i1, i1, i2, i2), v_step);
                        let u = mul(sub(targets, splat(scaled_count_sum)), recip_scaled_count);

                        // Note: if we were processing 4 points at a time instead of
                        // interleaving x and y, approx_parabola_inv_integral would
                        // process 4 points instead of 2 as many points at the same
                        // cost.
                        let u = approx_parabola_inv_integral(mul_add(splat(params.integral_to - params.integral_from), u, splat(params.integral_from)));
                        let t = mul(sub(u, splat(params.inv_integral_from)), splat(params.div_inv_integral_diff));

                        let p = sample_quadratic_horner_simd4(quad_a0, quad_a1, quad_a2, t);
                        aligned_store(points_out_ptr, p);

                        points_cap -= 4;
                        points_out_ptr = points_out_ptr.add(4);
                        i = (i + 2).min(n);
                    }

                    // Go over the point buffer a invoke the callback.
                    let mut ptr = point_buffer.0.as_mut_ptr();
                    for _ in 0..copy_count {
                        let x = ptr.read();
                        let y = ptr.add(1).read();
                        ptr = ptr.add(2);

                        let to = point(x, y);

                        cb(&LineSegment { from, to});
                        from = to;
                    }
                }
            }
            scaled_count_sum += params.scaled_count;
        }

        cb(&LineSegment { from, to: quads_last_to });
        if quad_idx >= num_quadratics {
            break;
        }

        quads.clear();
    }
}

/// This version only differs in the way flattening_params_simd4_v2 outputs
/// the flattening params, using SOA instead of AOS layout and using aligned_store
/// into the arrays instead of unpacking the lanes and doing an awkward unrolled
/// loop to store each flattening param.
/// I'm a bit surprised and disappointed that this version is slower than the
/// dumb and inelegant equivalent.
#[inline(never)]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx"))]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
pub unsafe fn flatten_cubic_simd4_v2(curve: &CubicBezierSegment<f32>, tolerance: f32, cb: &mut dyn FnMut(&LineSegment<f32>)) {

    let quads_tolerance = tolerance * 0.1;
    let flatten_tolerance = tolerance * 0.9;
    let sqrt_flatten_tolerance = flatten_tolerance.sqrt();
    let inv_sqrt_flatten_tolerance = fast_recip(sqrt_flatten_tolerance);

    let num_quadratics = crate::levien::num_quadratics_impl(curve, quads_tolerance);
    //println!("{num_quadratics:?} quads");

    let mut params = FlatteningParamsSimd {
        count: 0,
        a0_x: AlignedBuf::new(),
        a0_y: AlignedBuf::new(),
        a1_x: AlignedBuf::new(),
        a1_y: AlignedBuf::new(),
        a2_x: AlignedBuf::new(),
        a2_y: AlignedBuf::new(),
        scaled_count: AlignedBuf::new(),
        integral_from: AlignedBuf::new(),
        integral_to: AlignedBuf::new(),
        inv_integral_from: AlignedBuf::new(),
        div_inv_integral_diff: AlignedBuf::new(),
    };

    let quad_step = fast_recip(num_quadratics);
    let num_quadratics = num_quadratics as u32;
    let mut quad_idx = 0;
    let mut from = curve.from;

    loop {
        let (quads_last_to, sum) = flattening_params_simd4_v2(
            curve,
            num_quadratics,
            quad_idx,
            quad_step,
            sqrt_flatten_tolerance,
            &mut params
        );

        let num_edges = ((0.5 * sum * inv_sqrt_flatten_tolerance).ceil() as u32).max(1);

        // Iterate through the quadratics, outputting the points of
        // subdivisions that fall within that quadratic.
        let step = sum / (num_edges as f32);
        let mut i = 1;
        let mut scaled_count_sum = 0.0;
        let v_step = splat(step);
        //println!("------ {num_edges:?} edges step {step:?}");
        //assert!(params.count <= NUM_QUADS);
        for idx in 0..params.count {
            let n = u32::min(num_edges, ((scaled_count_sum + params.scaled_count.get(idx)) / step).ceil() as u32);

            if i < n {
                let recip_scaled_count = splat(fast_recip(params.scaled_count.get(idx)));
                let quad_a0 = interleave_splat(params.a0_x.get(idx), params.a0_y.get(idx));
                let quad_a1 = interleave_splat(params.a1_x.get(idx), params.a1_y.get(idx));
                let quad_a2 = interleave_splat(params.a2_x.get(idx), params.a2_y.get(idx));
                let integral_from = splat(params.integral_from.get(idx));
                let integral_diff = splat(params.integral_to.get(idx) - params.integral_from.get(idx));
                let div_inv_integral_diff = splat(params.div_inv_integral_diff.get(idx));
                let inv_integral_from = splat(params.inv_integral_from.get(idx));
                let v_scaled_count_sum = splat(scaled_count_sum);

                while i < n {
                    let i1 = i as f32;
                    let i2 = i1 + 1.0;
                    let targets = mul(vec4(i1, i1, i2, i2), v_step);
                    let u = mul(sub(targets, v_scaled_count_sum), recip_scaled_count);

                    // Note: if we were processing 4 points at a time instead of
                    // interleaving x and y, approx_parabola_inv_integral would
                    // process 4 points instead of 2 at the same cost.
                    let u = approx_parabola_inv_integral(mul_add(integral_diff, u, integral_from));
                    let t = mul(sub(u, inv_integral_from), div_inv_integral_diff);

                    let p = sample_quadratic_horner_simd4(quad_a0, quad_a1, quad_a2, t);

                    let (x0, y0, x1, y1) = unpack(p);
                    let to = point(x0, y0);
                    cb(&LineSegment { from, to });
                    from = to;

                    if i + 1 < n {
                        let to = point(x1, y1);
                        cb(&LineSegment { from, to });
                        from = to;
                        i += 1;
                    }

                    i += 1;
                }
            }
            scaled_count_sum += params.scaled_count.get(idx);
        }

        cb(&LineSegment { from, to: quads_last_to });
        quad_idx += params.count as u32;
        params.count = 0;

        if quad_idx >= num_quadratics {
            break;
        }
    }
}

#[inline(never)]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx"))]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
unsafe fn flattening_params_simd4_v2(
    curve: &CubicBezierSegment<f32>,
    s_num_quads: u32,
    s_first_quad: u32,
    s_quad_step: f32,
    sqrt_tolerance: f32,
    output: &mut FlatteningParamsSimd,
) -> (Point, f32) {
    let mut quad_idx = s_first_quad;
    let mut last_x = 0.0;
    let mut last_y = 0.0;
    let quad_step = splat(s_quad_step);
    while quad_idx < s_num_quads && output.count < NUM_QUADS {
        let t0 = mul(add(splat(quad_idx as f32), vec4(0.0, 1.0, 2.0, 3.0)), quad_step);
        let t1 = add(t0, quad_step);

        // polynomial_form:
        let s_poly = polynomial_form_cubic(curve);
        let a0_x = splat(s_poly.a0.x);
        let a0_y = splat(s_poly.a0.y);
        let a1_x = splat(s_poly.a1.x);
        let a1_y = splat(s_poly.a1.y);
        let a2_x = splat(s_poly.a2.x);
        let a2_y = splat(s_poly.a2.y);
        let a3_x = splat(s_poly.a3.x);
        let a3_y = splat(s_poly.a3.y);

        let (from_x, from_y) = sample_cubic_horner_simd4(a0_x, a0_y, a1_x, a1_y, a2_x, a2_y, a3_x, a3_y, t0);
        // TODO: 3 out of 4 values are already in from_
        let (to_x, to_y) = sample_cubic_horner_simd4(a0_x, a0_y, a1_x, a1_y, a2_x, a2_y, a3_x, a3_y, t1);

        // Compute the control points of the sub-curves.

        // CubicBezierSegment::split_range
        let qx0 = splat(curve.ctrl1.x - curve.from.x);
        let qy0 = splat(curve.ctrl1.y - curve.from.y);
        let qx1 = splat(curve.ctrl2.x - curve.ctrl1.x);
        let qy1 = splat(curve.ctrl2.y - curve.ctrl1.y);
        let qx2 = splat(curve.to.x - curve.ctrl2.x);
        let qy2 = splat(curve.to.y - curve.ctrl2.y);

        let one = splat(1.0);
        let two = splat(2.0);

        // QuadraticBezierSegment::sample(t0)
        let t0_2 = mul(t0, t0);
        let one_t0 = sub(one, t0);
        let one_t0_2 = mul(one_t0, one_t0);
        let two_t0_one_t0 = mul(two, mul(one_t0, t0));
        let qx = add(mul_add(qx0, one_t0_2, mul(qx1, two_t0_one_t0)), mul(qx2, t0_2));
        let qy = add(mul_add(qy0, one_t0_2, mul(qy1, two_t0_one_t0)), mul(qy2, t0_2));

        let ctrl1_x = mul_add(qx, quad_step, from_x);
        let ctrl1_y = mul_add(qy, quad_step, from_y);

        // QuadraticBezierSegment::sample(t1)
        let t1_2 = mul(t1, t1);
        let one_t1 = sub(one, t1);
        let one_t1_2 = mul(one_t1, one_t1);
        let two_t1_one_t1 = mul(two, mul(one_t1, t1));
        let qx = add(mul_add(qx0, one_t1_2, mul(qx1, two_t1_one_t1)), mul(qx2, t1_2));
        let qy = add(mul_add(qy0, one_t1_2, mul(qy1, two_t1_one_t1)), mul(qy2, t1_2));

        let ctrl2_x = sub(to_x, mul(qx, quad_step));
        let ctrl2_y = sub(to_y, mul(qy, quad_step));

        // Approximate the sub-curves with quadratics
        let three = splat(3.0);
        let half = splat(0.5);
        let c1x = mul(mul_sub(ctrl1_x, three, from_x), half);
        let c1y = mul(mul_sub(ctrl1_y, three, from_y), half);
        let c2x = mul(mul_sub(ctrl2_x, three, to_x), half);
        let c2y = mul(mul_sub(ctrl2_y, three, to_y), half);

        let ctrl_x = mul(add(c1x, c2x), half);
        let ctrl_y = mul(add(c1y, c2y), half);

        // Now that we have our four quadratics, compute the flattening parameters
        let ddx = sub(mul_sub(two, ctrl_x, from_x), to_x);
        let ddy = sub(mul_sub(two, ctrl_y, from_y), to_y);
        //println!("ddx {ddx:?} ddy {ddy:?}");
        let cross = mul_sub(sub(to_x, from_x), ddy, mul(sub(to_y, from_y), ddx));
        //println!("cross {cross:?}");
        let rcp_cross = recip(cross);
        let parabola_from = mul(mul_add(sub(ctrl_x, from_x), ddx, mul(sub(ctrl_y, from_y), ddy)), rcp_cross);
        let parabola_to = mul(mul_add(sub(to_x, ctrl_x), ddx, mul(sub(to_y, ctrl_y), ddy)), rcp_cross);
        let parabola_diff = sub(parabola_to, parabola_from);
        let scale = abs(div(cross, mul(sqrt(mul_add(ddx, ddx, mul(ddy, ddy))), parabola_diff)));

        let integral_from = approx_parabola_integral(parabola_from);
        let integral_to = approx_parabola_integral(parabola_to);
        //println!("parabola_to {parabola_to:?} parabola_from {parabola_from:?}");
        //println!("intergal_to {integral_to:?} integral_from {integral_from:?}");

        let inv_integral_from = approx_parabola_inv_integral(integral_from);
        let inv_integral_to = approx_parabola_inv_integral(integral_to);

        let div_inv_integral_diff = recip(sub(inv_integral_to, inv_integral_from));

        let sqrt_tolerance = splat(sqrt_tolerance);
        let integral_diff = abs(sub(integral_to, integral_from));
        let sqrt_scale = sqrt(scale);

        let mut scaled_count = mul(integral_diff, sqrt_scale);

        let is_cusp = neq(signum(parabola_from), signum(parabola_to));
        // Handle a cusp case (segment contains curvature maximum)
        // Assuming the cusp case is fairly rare and because it is expensive
        // we add a branch to test wether any of the lanes need to take it
        // and skip it otherwise.
        if any(is_cusp) {
            let xmin = div(sqrt_tolerance, sqrt_scale);
            let scaled_count2 = mul(sqrt_tolerance, div(integral_diff, approx_parabola_integral(xmin)));
            scaled_count = select(is_cusp, scaled_count2, scaled_count);
        }

        // Handle another kind of cusp.
        scaled_count = select_or_zero(is_finite(scale), scaled_count);

        // Convert the quadratic curves into polynomial form.
        let a1_x = mul(sub(ctrl_x, from_x), two);
        let a1_y = mul(sub(ctrl_y, from_y), two);
        let a2_x = sub(add(from_x, to_x), mul(ctrl_x, two));
        let a2_y = sub(add(from_y, to_y), mul(ctrl_y, two));

        // Go back to scalar land to produce the output.

        let offset = output.count;
        aligned_store(output.scaled_count.ptr(offset), scaled_count);
        aligned_store(output.integral_from.ptr(offset), integral_from);
        aligned_store(output.integral_to.ptr(offset), integral_to);
        aligned_store(output.inv_integral_from.ptr(offset), inv_integral_from);
        aligned_store(output.div_inv_integral_diff.ptr(offset), div_inv_integral_diff);

        aligned_store(output.a0_x.ptr(offset), from_x);
        aligned_store(output.a0_y.ptr(offset), from_y);
        aligned_store(output.a1_x.ptr(offset), a1_x);
        aligned_store(output.a1_y.ptr(offset), a1_y);
        aligned_store(output.a2_x.ptr(offset), a2_x);
        aligned_store(output.a2_y.ptr(offset), a2_y);

        let mut tmp_to_x = Aligned([0.0; 4]);
        let mut tmp_to_y = Aligned([0.0; 4]);
        aligned_store(tmp_to_x.0.as_mut_ptr(), to_x);
        aligned_store(tmp_to_y.0.as_mut_ptr(), to_y);

        let count = u32::min(s_num_quads - quad_idx, 4) as usize;

        last_x = tmp_to_x.0[count - 1];
        last_y = tmp_to_y.0[count - 1];

        output.count += count;
        quad_idx += 4;
    }

    let count = usize::min((s_num_quads - s_first_quad) as usize, NUM_QUADS);
    let sum = output.scaled_count.0.assume_init_mut()[0..count].iter().sum();

    (point(last_x, last_y), sum)
}

/// This was the base version for a while.
/// x and y are interleaved in the flattening loop (so the loop processes
/// two points at a time instead of 4).
/// The non-interleaved version turned out to be faster in some cases
/// or similar performance otherwise.
#[inline(never)]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx"))]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
pub unsafe fn flatten_cubic_simd4_interleaved(curve: &CubicBezierSegment<f32>, tolerance: f32, cb: &mut dyn FnMut(&LineSegment<f32>)) {

    let quads_tolerance = tolerance * 0.1;
    let flatten_tolerance = tolerance * 0.9;
    let sqrt_flatten_tolerance = flatten_tolerance.sqrt();
    let inv_sqrt_flatten_tolerance = fast_recip(sqrt_flatten_tolerance);

    let num_quadratics = crate::levien::num_quadratics_impl(curve, quads_tolerance);
    //println!("{num_quadratics:?} quads");

    //counters_inc(num_quadratics.max(0.0) as usize);

    let mut quads: ArrayVec<(FlatteningParams, QuadraticBezierPolynomial), 16> = ArrayVec::new();

    let quad_step = fast_recip(num_quadratics);
    let num_quadratics = num_quadratics as u32;
    let mut quad_idx = 0;
    let mut from = curve.from;

    loop {
        let mut sum = 0.0;
        let mut quads_last_to = point(0.0, 0.0);
        while quad_idx < num_quadratics && quads.capacity() > quads.len() {
            let (to, s) = flattening_params_simd4(
                curve,
                (num_quadratics - quad_idx).min(4) as f32,
                quad_idx as f32,
                quad_step,
                sqrt_flatten_tolerance,
                &mut quads
            );
            sum += s;
            quads_last_to = to;

            //println!(" + {quad:?} {t0} .. {t1} scaled count: {:?}", params.scaled_count);
            quad_idx += 4;
        }

        let num_edges = ((0.5 * sum * inv_sqrt_flatten_tolerance).ceil() as u32).max(1);

        // Iterate through the quadratics, outputting the points of
        // subdivisions that fall within that quadratic.
        let step = sum / (num_edges as f32);
        let mut i = 1;
        let mut scaled_count_sum = 0.0;
        let v_step = splat(step);
        //println!("------ {num_edges:?} edges step {step:?}");
        for (params, quad) in &quads {
            let n = u32::min(num_edges, ((scaled_count_sum + params.scaled_count) / step).ceil() as u32);

            if i < n {
                let recip_scaled_count = splat(fast_recip(params.scaled_count));
                let quad_a0 = interleave_splat(quad.a0.x, quad.a0.y);
                let quad_a1 = interleave_splat(quad.a1.x, quad.a1.y);
                let quad_a2 = interleave_splat(quad.a2.x, quad.a2.y);
                let integral_from = splat(params.integral_from);
                let integral_diff = splat(params.integral_to - params.integral_from);
                let inv_integral_from = splat(params.inv_integral_from);
                let div_inv_integral_diff = splat(params.div_inv_integral_diff);
                let v_scaled_count_sum = splat(scaled_count_sum);

                while i < n {
                    let i1 = i as f32;
                    let i2 = i1 + 1.0;
                    let targets = mul(vec4(i1, i1, i2, i2), v_step);
                    let u = mul(sub(targets, v_scaled_count_sum), recip_scaled_count);

                    // Note: if we were processing 4 points at a time instead of
                    // interleaving x and y, approx_parabola_inv_integral would
                    // process 4 points instead of 2 as many points at the same
                    // cost.
                    let u = approx_parabola_inv_integral(mul_add(integral_diff, u, integral_from));
                    let t = mul(sub(u, inv_integral_from), div_inv_integral_diff);

                    let p = sample_quadratic_horner_simd4(quad_a0, quad_a1, quad_a2, t);

                    // TODO: instead of unpacking, and using a callback, write directly into
                    // a point buffer. It would probably avoid a lot of bookkeeping from
                    // reorganizing register and calling a (potentially non-inlined function)
                    // in the hot loop.

                    let (x0, y0, x1, y1) = unpack(p);
                    let to = point(x0, y0);
                    cb(&LineSegment { from, to });
                    from = to;

                    if i + 1 < n {
                        let to = point(x1, y1);
                        cb(&LineSegment { from, to });
                        from = to;
                        i += 1;
                    }

                    i += 1;
                }
            }
            scaled_count_sum += params.scaled_count;
        }

        cb(&LineSegment { from, to: quads_last_to });
        if quad_idx >= num_quadratics {
            break;
        }

        quads.clear();
    }
}
