//! A so far unsuccessful attempt at speeding up the quadratic bézier flattening code using SSE
//! instructions.
//! I was hoping that batching approx_parabola_integral and approx_parabola_inv_integral in pairs
//! would halve the cost of the expensive square roots but it doesn't make a significant difference
//! in the profiles so far.

use arrayvec::ArrayVec;
use lyon_path::geom::{CubicBezierSegment, LineSegment, QuadraticBezierSegment};
use lyon_path::math::point;

#[cfg(target_arch = "x86")]
use std::arch::x86 as arch;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64 as arch;
use std::f32;
use std::ops::Range;

use arch::*;

use arch::_mm_set1_ps as splat;
use arch::_mm_setr_ps as vec4;
use arch::_mm_add_ps as add;
use arch::_mm_sub_ps as sub;
use arch::_mm_mul_ps as mul;
use arch::_mm_div_ps as div;
use arch::_mm_sqrt_ps as sqrt;
use arch::_mm_ceil_ps as ceil;
use arch::_mm_fmadd_ps as mul_add;
use arch::_mm_fmsub_ps as mul_sub;
use arch::_mm_rcp_ps as recip;
use arch::_mm_cmpeq_ps as eq;
use arch::_mm_cmpneq_ps as neq;
use arch::_mm_cmplt_ps as lt;
use arch::_mm_cmpgt_ps as gt;
use arch::_mm_and_ps as and;
use arch::_mm_andnot_ps as and_not;
use arch::_mm_or_ps as or;

#[inline(always)]
unsafe fn abs(val: __m128) -> __m128 {
    let minus_1 = arch::_mm_set1_epi32(-1);
    let mask = arch::_mm_castsi128_ps(arch::_mm_srli_epi32(minus_1, 1));
    arch::_mm_and_ps(mask, val)
}

#[inline(always)]
unsafe fn unpack(lanes: __m128) -> (f32, f32, f32, f32) {
    std::mem::transmute(lanes)
}

#[inline(always)]
unsafe fn select(cond: __m128, a: __m128, b: __m128) -> __m128 {
    or(
        arch::_mm_andnot_ps(cond, b),
        and(cond, a),
    )
}

#[inline(always)]
unsafe fn is_finite(a: __m128) -> __m128 {
    lt(abs(a), splat(f32::INFINITY))
}

#[inline(always)]
unsafe fn sign_bit(a: __m128) -> __m128 {
    and(a, splat(std::mem::transmute(1u32 << 31)))
}

#[inline(always)]
unsafe fn signum(a: __m128) -> __m128 {
    or(and(a, splat(std::mem::transmute(1u32 << 31))), splat(1.0))
}

#[inline(always)]
unsafe fn not(a: __m128) -> __m128 {
    and_not(a, a)
}

#[inline(always)]
unsafe fn any(a: __m128) -> bool {
    const MASK_13: i32 = 1 | (0 << 2) | (3 << 4) | (2 << 6);
    let a13 = arch::_mm_shuffle_ps::<MASK_13>(a, a); // a1, a0, a3, a2
    let or13 =  or(a13, a); // a0|a1, a0|a1, a2|a3, a2|a3

    const MASK_02: i32 = 2 | 0 | 0 | 0;
    let a02 = arch::_mm_shuffle_ps::<MASK_02>(or13, or13); // a2|a3, a0|a1, a0|a1, a0|a1
    let or02 =  or(or13, a02);

    arch::_mm_extract_ps::<0>(or02) != 0
}




#[test]
fn sse_cmp() {
    unsafe {
        println!("eq {:?}", unpack(eq(vec4(1.0, 2.0, 3.0, 4.0), vec4(2.0, 2.0, 2.0, 2.0))));
        let lt_cond = lt(vec4(1.0, 2.0, 3.0, 4.0), vec4(2.0, 2.0, 2.0, 2.0));
        println!("lt {:?}", unpack(lt_cond));
        println!("select {:?}", unpack(select(lt_cond, splat(1.0), splat(2.0))));
        println!("is_finite {:?}", unpack(is_finite(vec4(f32::NAN, f32::INFINITY, -f32::INFINITY, 42.0))));
        println!("sign {:?}", unpack(sign_bit(vec4(0.0, -0.0, 1.0, -2.0))));
        println!("signum {:?}", unpack(signum(vec4(0.0, -0.0, 1.0, -2.0))));
        println!("any 1000 {:?}", any(vec4(1.0, 0.0, 0.0, 0.0)));
        println!("any 0100 {:?}", any(vec4(0.0, 1.0, 0.0, 0.0)));
        println!("any 0010 {:?}", any(vec4(0.0, 0.0, 1.0, 0.0)));
        println!("any 0001 {:?}", any(vec4(0.0, 0.0, 0.0, 1.0)));
        println!("any 0000 {:?}", any(vec4(0.0, 0.0, 0.0, 0.0)));
        panic!();
    }
}

use crate::levien::FlatteningParams;
use crate::polynomial_form_cubic;

#[inline(always)]
unsafe fn approx_parabola_integral(x: __m128) -> __m128 {
    let d_pow4 = splat(0.67 * 0.67 * 0.67 * 0.67);
    let sqr = sqrt(sqrt(add(d_pow4, mul(splat(0.25), mul(x, x)))));
    div(x, add(splat(1.0 - 0.67), sqr))
}

#[inline(always)]
unsafe fn approx_parabola_inv_integral(x: __m128) -> __m128 {
    mul(x, add(splat(1.0 - 0.39), sqrt(add(splat(0.39 * 0.39), mul(splat(0.25), mul(x, x))))))
}

pub fn flatten_quadratic(curve: &QuadraticBezierSegment<f32>, tolerance: f32, cb: &mut impl FnMut(&LineSegment<f32>)) {
    unsafe {

    let ddx = 2.0 * curve.ctrl.x - curve.from.x - curve.to.x;
    let ddy = 2.0 * curve.ctrl.y - curve.from.y - curve.to.y;
    let cross = (curve.to.x - curve.from.x) * ddy - (curve.to.y - curve.from.y) * ddx;
    let parabola_from = ((curve.ctrl.x - curve.from.x) * ddx + (curve.ctrl.y - curve.from.y) * ddy) / cross;
    let parabola_to = ((curve.to.x - curve.ctrl.x) * ddx + (curve.to.y - curve.ctrl.y) * ddy) / cross;
    // Note, scale can be NaN, for example with straight lines. When it happens the NaN will
    // propagate to other parameters. We catch it all by setting the iteration count to zero
    // and leave the rest as garbage.
    let scale = cross.abs() / ((ddx * ddx + ddy * ddy).sqrt() * (parabola_to - parabola_from).abs());

    let integral = approx_parabola_integral(vec4(parabola_from, parabola_to, 0.0, 0.0));
    let (integral_to, integral_from, _, _) = unpack(integral);
    let inv_integral = approx_parabola_inv_integral(integral);
    let (inv_integral_to, inv_integral_from, _, _) = unpack(inv_integral);

    let integral_diff = integral_to - integral_from;
    let div_inv_integral_diff = 1.0 / (inv_integral_to - inv_integral_from);

    let mut count = (0.5 * integral_diff.abs() * (scale / tolerance).sqrt()).ceil();
    // If count is NaN the curve can be approximated by a single straight line or a point.
    if !count.is_finite() {
        count = 0.0;
    }

    let integral_from = splat(integral_from);
    let integral_step = splat(integral_diff / count);
    let mut iteration = vec4(4.0, 3.0, 2.0, 1.0);
    let mut i = 1;
    let count = count as usize;
    let mut from = curve.from;
    while i < count {

        let integ = add(integral_from, mul(integral_step, iteration));
        let u = approx_parabola_inv_integral(integ);
        let t = mul(sub(u, splat(inv_integral_from)), splat(div_inv_integral_diff));
        let t2 = mul(t, t);
        let one_t = sub(splat(1.0), t);
        let one_t2 = mul(one_t, one_t);

        let x = add(
            mul(splat(curve.from.x), one_t2),
            add(
                mul(mul(mul(splat(curve.ctrl.x), splat(2.0)), one_t), t),
                mul(splat(curve.to.x), t2)
            )
        );

        let y = add(
            mul(splat(curve.from.y), one_t2),
            add(
                mul(mul(mul(splat(curve.ctrl.y), splat(2.0)), one_t), t),
                mul(splat(curve.to.y), t2)
            )
        );

        let x: [f32; 4] = std::mem::transmute(x);
        let y: [f32; 4] = std::mem::transmute(y);

        for i in 0..(count - i).min(4) {
            let p = point(x[i], y[i]);
            cb(&LineSegment {from, to: p });
            from = p;
        }

        iteration = add(iteration, splat(4.0));
        i += 4;
    }

    cb(&LineSegment {from, to: curve.to });

    }
}

#[inline]
unsafe fn horner_sample_sse2(
    a0_x: __m128,
    a0_y: __m128,
    a1_x: __m128,
    a1_y: __m128,
    a2_x: __m128,
    a2_y: __m128,
    a3_x: __m128,
    a3_y: __m128,
    t: __m128,
) -> (__m128, __m128) {
    let mut vx = a0_x;
    let mut vy = a0_y;

    vx = mul_add(a1_x, t, vx);
    vy = mul_add(a1_y, t, vy);

    let mut t2 = mul(t, t);

    vx = mul_add(a2_x, t2, vx);
    vy = mul_add(a2_y, t2, vy);

    t2 = mul(t2, t);

    vx = mul_add(a3_x, t2, vx);
    vy = mul_add(a3_y, t2, vy);

    (vx, vy)
}

unsafe fn flattening_params_sse2(
    curve: &CubicBezierSegment<f32>,
    s_num_quads: f32,
    s_first_quad: f32,
    s_quad_step: f32,
    sqrt_tolerance: f32,
    output: &mut ArrayVec<(QuadraticBezierSegment<f32>, FlatteningParams), 16>
) -> f32 {
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

    let (from_x, from_y) = horner_sample_sse2(a0_x, a0_y, a1_x, a1_y, a2_x, a2_y, a3_x, a3_y, t0);
    // TODO: 3 out of 4 values are already in from_
    let (to_x, to_y) = horner_sample_sse2(a0_x, a0_y, a1_x, a1_y, a2_x, a2_y, a3_x, a3_y, t1);

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
    let qx = add(add(mul(qx0, one_t0_2), mul(qx1, two_t0_one_t0)), mul(qx2, t0_2));
    let qy = add(add(mul(qy0, one_t0_2), mul(qy1, two_t0_one_t0)), mul(qy2, t0_2));

    let ctrl1_x = mul_add(qx, quad_step, from_x);
    let ctrl1_y = mul_add(qy, quad_step, from_y);

    // QuadraticBezierSegment::sample(t1)
    let t1_2 = mul(t1, t1);
    let one_t1 = sub(one, t1);
    let one_t1_2 = mul(one_t1, one_t1);
    let two_t1_one_t1 = mul(two, mul(one_t1, t1));
    let qx = add(add(mul(qx0, one_t1_2), mul(qx1, two_t1_one_t1)), mul(qx2, t1_2));
    let qy = add(add(mul(qy0, one_t1_2), mul(qy1, two_t1_one_t1)), mul(qy2, t1_2));

    let ctrl2_x = sub(to_x, mul(qx, quad_step));
    let ctrl2_y = sub(to_y, mul(qy, quad_step));

    // Approximate the sub-curves with quadratics
    let three = splat(3.0);
    let half = splat(0.5);
    let c1x = mul(sub(mul(ctrl1_x, three), from_x), half);
    let c1y = mul(sub(mul(ctrl1_y, three), from_y), half);
    let c2x = mul(sub(mul(ctrl2_x, three), to_x), half);
    let c2y = mul(sub(mul(ctrl2_y, three), to_y), half);

    let ctrl_x = mul(add(c1x, c2x), half);
    let ctrl_y = mul(add(c1y, c2y), half);

    // Now that we have our four quadratics, compute the flattening parameters
    let ddx = sub(mul_sub(two, ctrl_x, from_x), to_x);
    let ddy = sub(mul_sub(two, ctrl_y, from_y), to_y);
    //println!("ddx {ddx:?} ddy {ddy:?}");
    let cross = sub(mul(sub(to_x, from_x), ddy), mul(sub(to_y, from_y), ddx));
    //println!("cross {cross:?}");
    let rcp_cross = recip(cross);
    let parabola_from = mul(add(mul(sub(ctrl_x, from_x), ddx), mul(sub(ctrl_y, from_y), ddy)), rcp_cross);
    let parabola_to = mul(add(mul(sub(to_x, ctrl_x), ddx), mul(sub(to_y, ctrl_y), ddy)), rcp_cross);
    let parabola_diff = sub(parabola_to, parabola_from);
    let scale = abs(div(cross, mul(sqrt(add(mul(ddx, ddx), mul(ddy, ddy))), parabola_diff)));

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

    //println!("is_cusp {:?} integral_diff {:?} sqrt_scale {:?}", unpack(is_cusp), unpack(integral_diff), unpack(sqrt_scale));

    //let xmin = div(sqrt_tolerance, sqrt_scale);
    //let scaled_count2 = mul(sqrt_tolerance, div(integral_diff, approx_parabola_integral(xmin)));
    //scaled_count = select(is_cusp, scaled_count2, scaled_count);

    // Handle another kind of cusp.
    scaled_count = and(is_finite(scale), scaled_count);

    // Go back to scalar land to produce the output.

    let scaled_count = unpack(scaled_count);
    let integral_from = unpack(integral_from);
    let integral_to = unpack(integral_to);
    let inv_integral_from = unpack(inv_integral_from);
    let div_inv_integral_diff = unpack(div_inv_integral_diff);

    let from_x = unpack(from_x);
    let from_y = unpack(from_y);
    let ctrl_x = unpack(ctrl_x);
    let ctrl_y = unpack(ctrl_y);
    let to_x = unpack(to_x);
    let to_y = unpack(to_y);

    let mut sum = scaled_count.0;

    let count = s_num_quads as u32;
    output.push((
        QuadraticBezierSegment {
            from: point(from_x.0, from_y.0),
            ctrl: point(ctrl_x.0, ctrl_y.0),
            to: point(to_x.0, to_y.0),
        },
        FlatteningParams {
            scaled_count: scaled_count.0,
            integral_from: integral_from.0,
            integral_to: integral_to.0,
            inv_integral_from: inv_integral_from.0,
            div_inv_integral_diff: div_inv_integral_diff.0,
        }
    ));

    if count < 2 {
        return sum;
    }
    sum += scaled_count.1;
    output.push((
        QuadraticBezierSegment {
            from: point(from_x.1, from_y.1),
            ctrl: point(ctrl_x.1, ctrl_y.1),
            to: point(to_x.1, to_y.1),
        },
        FlatteningParams {
            scaled_count: scaled_count.1,
            integral_from: integral_from.1,
            integral_to: integral_to.1,
            inv_integral_from: inv_integral_from.1,
            div_inv_integral_diff: div_inv_integral_diff.1,
        }
    ));

    if count < 3 {
        return sum;
    }

    sum += scaled_count.2;
    output.push((
        QuadraticBezierSegment {
            from: point(from_x.2, from_y.2),
            ctrl: point(ctrl_x.2, ctrl_y.2),
            to: point(to_x.2, to_y.2),
        },
        FlatteningParams {
            scaled_count: scaled_count.2,
            integral_from: integral_from.2,
            integral_to: integral_to.2,
            inv_integral_from: inv_integral_from.2,
            div_inv_integral_diff: div_inv_integral_diff.2,
        }
    ));

    if count < 4 {
        return sum;
    }

    sum += scaled_count.3;
    output.push((
        QuadraticBezierSegment {
            from: point(from_x.3, from_y.3),
            ctrl: point(ctrl_x.3, ctrl_y.3),
            to: point(to_x.3, to_y.3),
        },
        FlatteningParams {
            scaled_count: scaled_count.3,
            integral_from: integral_from.3,
            integral_to: integral_to.3,
            inv_integral_from: inv_integral_from.3,
            div_inv_integral_diff: div_inv_integral_diff.2,
        }
    ));

    sum
}

pub fn flatten_cubic_sse2(curve: &CubicBezierSegment<f32>, tolerance: f32, cb: &mut impl FnMut(&LineSegment<f32>)) {
    unsafe {

    let quads_tolerance = tolerance * 0.1;
    let flatten_tolerance = tolerance * 0.9;
    let sqrt_flatten_tolerance = flatten_tolerance.sqrt();

    let num_quadratics = crate::levien::num_quadratics_impl(curve, quads_tolerance);
    //println!("{num_quadratics:?} quads");

    let mut quads: ArrayVec<(QuadraticBezierSegment<f32>, FlatteningParams), 16> = ArrayVec::new();

    let quad_step = 1.0 / num_quadratics;
    let num_quadratics = num_quadratics as u32;
    let mut quad_idx = 0;
    let mut from = curve.from;
    let sqrt_tolerance = tolerance.sqrt();

    loop {
        let mut sum = 0.0;
        while quad_idx < num_quadratics && quads.capacity() > quads.len() {
            sum += flattening_params_sse2(
                curve,
                (num_quadratics - quad_idx).min(4) as f32,
                quad_idx as f32,
                quad_step,
                sqrt_tolerance,
                &mut quads
            );

            //println!(" + {quad:?} {t0} .. {t1} scaled count: {:?}", params.scaled_count);
            quad_idx += 4;
        }

        let num_edges = ((0.5 * sum / sqrt_flatten_tolerance).ceil() as u32).max(1);

        // Iterate through the quadratics, outputting the points of
        // subdivisions that fall within that quadratic.
        let step = sum / (num_edges as f32);
        let mut i = 1;
        let mut scaled_count_sum = 0.0;
        //println!("------ {num_edges:?} edges step {step:?}");
        for (quad, params) in &quads {
            let mut target = (i as f32) * step;
            //println!("  target {target} ({i})");
            let recip_scaled_count = params.scaled_count.recip();
            while target < scaled_count_sum + params.scaled_count {
                let u = (target - scaled_count_sum) * recip_scaled_count;
                let t = params.get_t(u);
                let to = quad.sample(t);
                //println!("     u={u}, t={t}");
                cb(&LineSegment { from, to });
                from = to;
                if i == num_edges {
                    break;
                }
                i += 1;
                target = (i as f32) * step;
            }
            scaled_count_sum += params.scaled_count;
        }

        cb(&LineSegment { from, to: quads.last().unwrap().0.to });
        if quad_idx >= num_quadratics {
            break;
        }

        quads.clear();
    }

    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn flatten_simd() {
    unsafe {
        let (_, _, _, a) = unpack(approx_parabola_inv_integral(splat(0.5)));
        let aref = crate::levien::approx_parabola_inv_integral(0.5);

        println!("{} {}", a, aref);

        let curve = QuadraticBezierSegment {
            from: point(0.0, 0.0),
            ctrl: point(100.0, 0.0),
            to: point(100.0, 100.0),
        };

        let mut n0: u32 = 0;
        flatten_quadratic(&curve, 0.01, &mut |_p| {
            //println!("{:?}", _p);
            n0 += 1;
        });

        //println!("-----");
        let mut n1: u32 = 0;
        crate::levien::flatten_quad_scalar(&curve, 0.01, &mut |_p| {
            //println!("{:?}", _p);
            n1 += 1;
        });

        assert_eq!(n0, n1);
    }
}

#[test]
fn flatten_params() {
    let curve = CubicBezierSegment {
        from: point(0.0, 0.0),
        ctrl1: point(100.0, 0.0),
        ctrl2: point(0.0, 100.0),
        to: point(0.0, 100.0),
    };

    //let curve = CubicBezierSegment {
    //    from: point(0.0, 0.0),
    //    ctrl1: point(100.0, 0.0),
    //    ctrl2: point(100.0, 0.0),
    //    to: point(100.0, 100.0),
    //};


    let tolerance: f32 = 0.25;
    let sqrt_tolerance = tolerance.sqrt();

    let num_quadratics = crate::levien::num_quadratics_impl(&curve, tolerance);
    println!("{num_quadratics:?} quads");

    let mut quads_sse: ArrayVec<(QuadraticBezierSegment<f32>, FlatteningParams), 16> = ArrayVec::new();
    let mut quads_scalar: ArrayVec<(QuadraticBezierSegment<f32>, FlatteningParams), 16> = ArrayVec::new();
    let quad_step = 1.0 / num_quadratics;

    let sum_sse = unsafe {
        flattening_params_sse2(
            &curve,
            num_quadratics,
            0.0, quad_step,
            sqrt_tolerance,
            &mut quads_sse,
        )
    };

    println!("-----");

    let mut t0 = 0.0;
    let mut sum_scalar = 0.0;
    let mut quad_idx = 0;
    while quad_idx < num_quadratics as u32 {
        if quad_idx >= 4 {
            break;
        }
        let t1 = t0 + quad_step;
        println!("{:?}", t0 .. t1);
        let quad = split_range(&curve, t0..t1).to_quadratic();
        let params = FlatteningParams::new(&quad, tolerance);
        sum_scalar += params.scaled_count;
        //println!(" + {quad:?} {t0} .. {t1} scaled count: {:?}", params.scaled_count);
        quads_scalar.push((quad, params));
        t0 = t1;
        quad_idx += 1;
    }

    println!("-----");

    println!("sse: {quads_sse:?}");
    println!("scalar: {quads_scalar:?}");
    assert!((sum_sse - sum_scalar).abs() < 0.05, "sum sse {sum_sse} scalar {sum_scalar}");
}

pub fn split_range(curve: &CubicBezierSegment<f32>, t_range: Range<f32>) -> CubicBezierSegment<f32> {
    let (t0, t1) = (t_range.start, t_range.end);
    let from = curve.sample(t0);
    let to = curve.sample(t1);

    let d = QuadraticBezierSegment {
        from: (curve.ctrl1 - curve.from).to_point(),
        ctrl: (curve.ctrl2 - curve.ctrl1).to_point(),
        to: (curve.to - curve.ctrl2).to_point(),
    };

    let dt = t1 - t0;
    let ctrl1 = from + d.sample(t0).to_vector() * dt;
    let ctrl2 = to - d.sample(t1).to_vector() * dt;

    CubicBezierSegment {
        from,
        ctrl1,
        ctrl2,
        to,
    }
}
