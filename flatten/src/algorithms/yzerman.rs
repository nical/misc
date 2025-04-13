use crate::{LineSegment, CubicBezierSegment, point};
use crate::{fast_ceil, fast_cubic_root, fast_recip, split_range};

fn num_quadratics(curve: &CubicBezierSegment, tolerance: f32) -> f32 {
    let q = curve.from - curve.to + (curve.ctrl2 - curve.ctrl1) * 3.0;
    const K: f32 = 20.784609691; // (12.0 * 3.0f32.sqrt());
    fast_ceil(fast_cubic_root(tolerance * K * q.length())).max(1.0) // TODO: looks like ceil is not inlined.
}

pub fn flatten_cubic<F>(curve: &CubicBezierSegment, tolerance: f32, callback: &mut F)
where
    F:  FnMut(&LineSegment)
{
    let simplify_tolerance = tolerance * 0.2;
    let flatten_tolerance = tolerance - simplify_tolerance;

    let num_quads = num_quadratics(curve, simplify_tolerance);
    let step = fast_recip(num_quads);
    let mut t0 = 0.0;
    while t0 < 1.0 {
        let mut t1 = t0 + step;
        if t1 > 0.999 {
            t1 = 1.0;
        }

        let quad = split_range(&curve, t0..t1).to_quadratic();

        // Check Whether the quadratic curve is flat enough that we can
        // skip the exepensive evaluation of the number of line segments
        // per quad.
        // This shortcut is a win for small curves or when the tolerance
        // is large enough that a fair amount of quads will be flat.
        // Overall it is a 5~12% win in the benchmarks at tolerance 0.25.
        // The SIMD version does not have this optimization because it would
        // require 4 consecutive flat quads to take advantage of it without
        // expensive bookkeeping.
        if crate::flatness::quadratic_is_flat(&quad, flatten_tolerance) {
            callback(&quad.baseline());
        } else {
            crate::wang::flatten_quadratic(&quad, flatten_tolerance, callback);
        }

        t0 = t1;
    }
}

const NUM_QUADS: usize = 16;

#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx"))]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
pub unsafe fn flatten_cubic_simd4<F>(curve: &CubicBezierSegment, tolerance: f32, callback: &mut F)
where
    F:  FnMut(&LineSegment)
{
    use crate::simd4::*;

    let simplify_tolerance = tolerance * 0.2;
    let flatten_tolerance = tolerance - simplify_tolerance;

    let mut num_segments = AlignedBuf::new();
    let mut a0x = AlignedBuf::new();
    let mut a0y = AlignedBuf::new();
    let mut a1x = AlignedBuf::new();
    let mut a1y = AlignedBuf::new();
    let mut a2x = AlignedBuf::new();
    let mut a2y = AlignedBuf::new();

    let num_quads = num_quadratics(curve, simplify_tolerance);
    let quad_step = fast_recip(num_quads);

    let mut from = curve.from;
    let mut quad_offset = 0;
    while quad_offset < num_quads as usize {
        let first_quad = quad_offset as f32;
        flattening_params_simd4(
            curve,
            num_quads,
            first_quad,
            quad_step,
            flatten_tolerance,
            &FlatteningParams {
                num_segments: num_segments.ptr(quad_offset),
                a0x: a0x.ptr(quad_offset),
                a0y: a0y.ptr(quad_offset),
                a1x: a1x.ptr(quad_offset),
                a1y: a1y.ptr(quad_offset),
                a2x: a2x.ptr(quad_offset),
                a2y: a2y.ptr(quad_offset),
            }
        );

        let is_last = quad_offset + NUM_QUADS >= num_quads as usize;
        let quad_count = (num_quads as usize - quad_offset).min(NUM_QUADS);
        for i in 0..quad_count {
            let a0x = splat(a0x.get(i));
            let a0y = splat(a0y.get(i));
            let a1x = splat(a1x.get(i));
            let a1y = splat(a1y.get(i));
            let a2x = splat(a2x.get(i));
            let a2y = splat(a2y.get(i));
            let num_segments = num_segments.get(i);
            let flatten_step = fast_recip(num_segments);
            let flatten_step = splat(flatten_step);
            let mut t = mul(flatten_step, vec4(1.0, 2.0, 3.0, 4.0));
            let flatten_step4 = mul(flatten_step, splat(4.0));

            let mut n = num_segments as i32;
            if is_last && i + 1 == n as usize {
                // We'll add the last one manually so that it is exactly
                // at the endpoint.
                n -= 1;
            }
            while n > 0 {
                let x = crate::simd4::sample_quadratic_horner_simd4(a0x, a1x, a2x, t);
                let y = crate::simd4::sample_quadratic_horner_simd4(a0y, a1y, a2y, t);

                let x: [f32; 4] = std::mem::transmute(x);
                let y: [f32; 4] = std::mem::transmute(y);

                for (x, y) in x.iter().zip(y.iter()).take(n.min(4) as usize) {
                    let p = point(*x, *y);
                    callback(&LineSegment { from, to: p });
                    from = p;
                }

                t = add(t, flatten_step4);
                n -= 4;
            }
        }

        quad_offset += NUM_QUADS;
    }

    callback(&LineSegment { from, to: curve.to });
}

struct FlatteningParams {
    num_segments: *mut f32,
    a0x: *mut f32,
    a0y: *mut f32,
    a1x: *mut f32,
    a1y: *mut f32,
    a2x: *mut f32,
    a2y: *mut f32,
}

#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx"))]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
unsafe fn flattening_params_simd4(
    curve: &CubicBezierSegment,
    s_num_quads: f32,
    s_first_quad: f32,
    s_quad_step: f32,
    tolerance: f32,
    output: &FlatteningParams,
) {
    use crate::simd4::*;

    let quad_step = splat(s_quad_step);
    let mut t0 = mul(add(splat(s_first_quad), vec4(0.0, 1.0, 2.0, 3.0)), quad_step);
    let mut t1 = add(t0, quad_step);
    let quad_step_4 = mul(quad_step, splat(4.0));
    let mut quad_idx = 0;
    let count = (s_num_quads as usize - s_first_quad as usize).min(NUM_QUADS);

    while quad_idx < count {
        // polynomial_form:
        let s_poly = crate::polynomial_form_cubic(curve);
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

        let two = splat(2.0);
        let lx = mul(sub(add(to_x, from_x), mul(ctrl_x, two)), two);
        let ly = mul(sub(add(to_y, from_y), mul(ctrl_y, two)), two);
        let recip_8tol = recip(splat(tolerance * 8.0));
        let num_segments = sqrt(mul(sqrt(add(mul(lx, lx), mul(ly, ly))), recip_8tol));
        let num_segments = max(splat(1.0), ceil(num_segments));

        // Convert the quadratic curves into polynomial form.
        let a1_x = mul(sub(ctrl_x, from_x), two);
        let a1_y = mul(sub(ctrl_y, from_y), two);
        let a2_x = sub(add(from_x, to_x), mul(ctrl_x, two));
        let a2_y = sub(add(from_y, to_y), mul(ctrl_y, two));

        let offset = quad_idx;

        aligned_store(output.num_segments.add(offset), num_segments);
        aligned_store(output.a0x.add(offset), from_x);
        aligned_store(output.a0y.add(offset), from_y);
        aligned_store(output.a1x.add(offset), a1_x);
        aligned_store(output.a1y.add(offset), a1_y);
        aligned_store(output.a2x.add(offset), a2_x);
        aligned_store(output.a2y.add(offset), a2_y);

        t0 = add(t0, quad_step_4);
        t1 = add(t1, quad_step_4);
        quad_idx += 4;
    }
}
