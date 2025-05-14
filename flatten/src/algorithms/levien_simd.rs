use arrayvec::ArrayVec;
use crate::{CubicBezierSegment, LineSegment, QuadraticBezierSegment};
use crate::{point, Point, Vector};

#[cfg(target_arch = "x86")]
use std::arch::x86 as arch;
#[cfg(target_arch = "x86_64")]
use std::f32;
use std::ops::Range;

use crate::{fast_recip, fast_recip_sqrt, polynomial_form_quadratic, simd4::*};

use crate::levien::FlatteningParams;
//use crate::testing::counters_inc;
use crate::{polynomial_form_cubic, QuadraticBezierPolynomial};


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

#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx"))]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
pub unsafe fn flatten_quadratic(curve: &QuadraticBezierSegment, tolerance: f32, cb: &mut impl FnMut(&LineSegment)) {

    let ddx = 2.0 * curve.ctrl.x - curve.from.x - curve.to.x;
    let ddy = 2.0 * curve.ctrl.y - curve.from.y - curve.to.y;
    let cross = (curve.to.x - curve.from.x) * ddy - (curve.to.y - curve.from.y) * ddx;
    let inv_cross = fast_recip(cross);

    // Attempting to vectorize these two lines did not improve performance.
    let parabola_from = ((curve.ctrl.x - curve.from.x) * ddx + (curve.ctrl.y - curve.from.y) * ddy) * inv_cross;
    let parabola_to   = ((curve.to.x   - curve.ctrl.x) * ddx + (curve.to.y   - curve.ctrl.y) * ddy) * inv_cross;

    // Note, scale can be NaN, for example with straight lines. When it happens the NaN will
    // propagate to other parameters. We catch it all by setting the iteration count to zero
    // and leave the rest as garbage.
    //let scale = cross.abs() / ((ddx * ddx + ddy * ddy).sqrt() * (parabola_to - parabola_from).abs());
    let scale = cross.abs() * fast_recip_sqrt(ddx * ddx + ddy * ddy) * fast_recip((parabola_to - parabola_from).abs());

    let integral = approx_parabola_integral(vec4(parabola_from, parabola_to, 0.0, 0.0));
    let inv_integral = approx_parabola_inv_integral(integral);
    let (integral_to, integral_from, _, _) = unpack(integral);
    let (inv_integral_to, inv_integral_from, _, _) = unpack(inv_integral);

    let integral_diff = integral_to - integral_from;
    let div_inv_integral_diff = fast_recip(inv_integral_to - inv_integral_from);

    let mut count = (0.5 * integral_diff.abs() * (scale * fast_recip(tolerance)).sqrt()).ceil();
    // If count is NaN the curve can be approximated by a single straight line or a point.
    if !count.is_finite() {
        count = 0.0;
    }

    let poly = polynomial_form_quadratic(curve);
    let quad_a0x = splat(poly.a0.x);
    let quad_a0y = splat(poly.a0.y);
    let quad_a1x = splat(poly.a1.x);
    let quad_a1y = splat(poly.a1.y);
    let quad_a2x = splat(poly.a2.x);
    let quad_a2y = splat(poly.a2.y);

    let integral_from = splat(integral_from);
    let integral_step = splat(integral_diff * fast_recip(count));
    let mut iteration = vec4(1.0, 2.0, 3.0, 4.0);
    let mut i = 1;
    let count = count as usize;
    let mut from = curve.from;
    while i < count {
        let integ = add(integral_from, mul(integral_step, iteration));
        let u = approx_parabola_inv_integral(integ);

        let t = mul(sub(u, splat(inv_integral_from)), splat(div_inv_integral_diff));

        let x = sample_quadratic_horner_simd4(quad_a0x, quad_a1x, quad_a2x, t);
        let y = sample_quadratic_horner_simd4(quad_a0y, quad_a1y, quad_a2y, t);

        let x: [f32; 4] = std::mem::transmute(x);
        let y: [f32; 4] = std::mem::transmute(y);

        for (x, y) in x.iter().zip(y.iter()).take((count - i).min(4)) {
            let p = point(*x, *y);
            cb(&LineSegment {from, to: p });
            from = p;
        }

        iteration = add(iteration, splat(4.0));
        i += 4;
    }

    cb(&LineSegment {from, to: curve.to });
}

#[inline(never)]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx"))]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
unsafe fn flattening_params_simd4(
    curve: &CubicBezierSegment,
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

#[inline(never)]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx"))]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
pub unsafe fn flatten_cubic_simd4(curve: &CubicBezierSegment, tolerance: f32, cb: &mut dyn FnMut(&LineSegment)) {

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
                let quad_a0x = splat(quad.a0.x);
                let quad_a0y = splat(quad.a0.y);
                let quad_a1x = splat(quad.a1.x);
                let quad_a1y = splat(quad.a1.y);
                let quad_a2x = splat(quad.a2.x);
                let quad_a2y = splat(quad.a2.y);
                let integral_from = splat(params.integral_from);
                let integral_diff = splat(params.integral_to - params.integral_from);
                let inv_integral_from = splat(params.inv_integral_from);
                let div_inv_integral_diff = splat(params.div_inv_integral_diff);
                let v_scaled_count_sum = splat(scaled_count_sum);

                let mut v_i = add(splat(i as f32), vec4(0.0, 1.0, 2.0, 3.0));

                while i < n {
                    let targets = mul(v_i, v_step);
                    let u = mul(sub(targets, v_scaled_count_sum), recip_scaled_count);

                    let u = approx_parabola_inv_integral(mul_add(integral_diff, u, integral_from));
                    let t = mul(sub(u, inv_integral_from), div_inv_integral_diff);

                    let px = sample_quadratic_horner_simd4(quad_a0x, quad_a1x, quad_a2x, t);
                    let py = sample_quadratic_horner_simd4(quad_a0y, quad_a1y, quad_a2y, t);

                    let px: [f32; 4] = std::mem::transmute(px);
                    let py: [f32; 4] = std::mem::transmute(py);

                    for (x, y) in px.iter().zip(py.iter()) {
                        let to = point(*x, *y);
                        cb(&LineSegment { from, to });
                        from = to;
                        i += 1;
                        if i >= n {
                            break;
                        }
                    }
                    v_i = add(v_i, splat(4.0));
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

    let mut quads_sse: ArrayVec<(FlatteningParams, QuadraticBezierPolynomial), 16> = ArrayVec::new();
    let mut quads_scalar: ArrayVec<(FlatteningParams, QuadraticBezierPolynomial), 16> = ArrayVec::new();
    let quad_step = 1.0 / num_quadratics;

    let sum_sse = unsafe {
        flattening_params_simd4(
            &curve,
            num_quadratics,
            0.0, quad_step,
            sqrt_tolerance,
            &mut quads_sse,
        ).1
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
        quads_scalar.push((params, crate::polynomial_form_quadratic(&quad)));
        t0 = t1;
        quad_idx += 1;
    }

    println!("-----");

    println!("sse: {quads_sse:?}");
    println!("scalar: {quads_scalar:?}");
    assert!((sum_sse - sum_scalar).abs() < 0.05, "sum sse {sum_sse} scalar {sum_scalar}");
}

pub fn split_range(curve: &CubicBezierSegment, t_range: Range<f32>) -> CubicBezierSegment {
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
