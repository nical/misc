//! A so far unsuccessful attempt at speeding up the quadratic bézier flattening code using SSE
//! instructions.
//! I was hoping that batching approx_parabola_integral and approx_parabola_inv_integral in pairs
//! would halve the cost of the expensive square roots but it doesn't make a significant difference
//! in the profiles so far.

use lyon::geom::QuadraticBezierSegment;
use lyon::path::math::{Point, point};

#[cfg(target_arch = "x86")]
use std::arch::x86 as arch;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64 as arch;

use arch::*;

use arch::_mm_add_ps as add;
use arch::_mm_sub_ps as sub;
use arch::_mm_mul_ps as mul;
use arch::_mm_div_ps as div;
use arch::_mm_set1_ps as splat;
use arch::_mm_set_ps as vec4;
use arch::_mm_sqrt_ps as sqrt;

#[inline]
unsafe fn approx_parabola_integral(x: __m128) -> __m128 {
    let d_pow4 = splat(0.67 * 0.67 * 0.67 * 0.67);
    let sqr = sqrt(sqrt(add(d_pow4, mul(splat(0.25), mul(x, x)))));
    div(x, add(splat(1.0 - 0.67), sqr))
}

#[inline]
unsafe fn approx_parabola_inv_integral(x: __m128) -> __m128 {
    mul(x, add(splat(1.0 - 0.39), sqrt(add(splat(0.39 * 0.39), mul(splat(0.25), mul(x, x))))))
}

#[inline(always)]
unsafe fn unpack(lanes: __m128) -> (f32, f32, f32, f32) {
    std::mem::transmute(lanes)
}

pub unsafe fn flatten_quad_sse(curve: &QuadraticBezierSegment<f32>, tolerance: f32, cb: &mut impl FnMut(Point)) {
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
    let (_, _, integral_to, integral_from) = unpack(integral);
    let inv_integral = approx_parabola_inv_integral(integral);
    let (_, _, inv_integral_to, inv_integral_from) = unpack(inv_integral);

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
            cb(p);
        }

        iteration = add(iteration, splat(4.0));
        i += 4;
    }

    cb(curve.to);
}

/// Compute an approximation to integral (1 + 4x^2) ^ -0.25 dx used in the flattening code.
fn approx_parabola_integral_ref(x: f32) -> f32 {
    let d = 0.67;
    let quarter = 0.25;
    x / (1.0 - d + (d*d*d*d + quarter * x * x).sqrt().sqrt())
}

/// Approximate the inverse of the function above.
fn approx_parabola_inv_integral_ref(x: f32) -> f32 {
    let b = 0.39;
    let quarter = 0.25;
    x * (1.0 - b + (b * b + quarter * x * x).sqrt())
}

pub fn flatten_quad_ref(curve: &QuadraticBezierSegment<f32>, tolerance: f32, cb: &mut impl FnMut(Point)) {
    // Map the quadratic bézier segment to y = x^2 parabola.
    let ddx = 2.0 * curve.ctrl.x - curve.from.x - curve.to.x;
    let ddy = 2.0 * curve.ctrl.y - curve.from.y - curve.to.y;
    let cross = (curve.to.x - curve.from.x) * ddy - (curve.to.y - curve.from.y) * ddx;
    let parabola_from =
        ((curve.ctrl.x - curve.from.x) * ddx + (curve.ctrl.y - curve.from.y) * ddy) / cross;
    let parabola_to =
        ((curve.to.x - curve.ctrl.x) * ddx + (curve.to.y - curve.ctrl.y) * ddy) / cross;
    // Note, scale can be NaN, for example with straight lines. When it happens the NaN will
    // propagate to other parameters. We catch it all by setting the iteration count to zero
    // and leave the rest as garbage.
    //let scale = cross.abs() / (ddx.hypot(ddy) * (parabola_to - parabola_from).abs());
    let scale = cross.abs() / ((ddx * ddx + ddy * ddy).sqrt() * (parabola_to - parabola_from).abs());

    let integral_from = approx_parabola_integral_ref(parabola_from);
    let integral_to = approx_parabola_integral_ref(parabola_to);
    let integral_diff = integral_to - integral_from;

    let inv_integral_from = approx_parabola_inv_integral_ref(integral_from);
    let inv_integral_to = approx_parabola_inv_integral_ref(integral_to);
    let div_inv_integral_diff = 1.0 / (inv_integral_to - inv_integral_from);

    // We could store this as an integer but the generic code makes that awkward and we'll
    // use it as a scalar again while iterating, so it's kept as a scalar.
    let mut count = (0.5 * integral_diff.abs() * (scale / tolerance).sqrt()).ceil();
    // If count is NaN the curve can be approximated by a single straight line or a point.
    if !count.is_finite() {
        count = 0.0;
    }

    let integral_step = integral_diff / count;

    let mut i = 1.0;
    for _ in 1..(count as u32) {
        let u = approx_parabola_inv_integral_ref(integral_from + integral_step * i);
        let t = (u - inv_integral_from) * div_inv_integral_diff;
        i += 1.0;
        cb(curve.sample(t));
    }

    cb(curve.to);
}

#[test]
fn flatten_simd() {
    unsafe {
        let (_, _, _, a) = unpack(approx_parabola_inv_integral(splat(0.5)));
        let aref = approx_parabola_inv_integral_ref(0.5);

        println!("{} {}", a, aref);

        let curve = QuadraticBezierSegment {
            from: point(0.0, 0.0),
            ctrl: point(100.0, 0.0),
            to: point(100.0, 100.0),
        };

        let mut n0: u32 = 0;
        flatten_quad_simd(&curve, 0.01, &mut |p| {
            println!("{:?}", p);
            n0 += 1;
        });

        println!("-----");
        let mut n1: u32 = 0;
        flatten_quad_ref(&curve, 0.01, &mut |p| {
            println!("{:?}", p);
            n1 += 1;
        });

        assert_eq!(n0, n1);
    }
}