use arrayvec::ArrayVec;
use crate::{CubicBezierSegment, LineSegment, QuadraticBezierSegment, point};

use crate::{polynomial_form_quadratic, QuadraticBezierPolynomial};

pub fn flatten_cubic_19<F>(curve: &CubicBezierSegment, tolerance: f32, callback: &mut F)
where
    F:  FnMut(&LineSegment)
{
    let quads_tolerance = tolerance * 0.1;
    let flatten_tolerance = tolerance * 0.9;
    curve.for_each_quadratic_bezier(quads_tolerance, &mut |quad| {
        flatten_quad_scalar(quad, flatten_tolerance, callback);
    });
}

pub fn flatten_cubic_37<F>(curve: &CubicBezierSegment, tolerance: f32, callback: &mut F)
where
    F:  FnMut(&LineSegment)
{
    let quads_tolerance = tolerance * 0.3;
    let flatten_tolerance = tolerance * 0.7;
    curve.for_each_quadratic_bezier(quads_tolerance, &mut |quad| {
        flatten_quad_scalar(quad, flatten_tolerance, callback);
    });
}

pub fn flatten_cubic_55<F>(curve: &CubicBezierSegment, tolerance: f32, callback: &mut F)
where
    F:  FnMut(&LineSegment)
{
    let quads_tolerance = tolerance * 0.5;
    let flatten_tolerance = tolerance * 0.5;
    curve.for_each_quadratic_bezier(quads_tolerance, &mut |quad| {
        flatten_quad_scalar(quad, flatten_tolerance, callback);
    });
}

pub fn flatten_quadratic<F>(curve: &QuadraticBezierSegment, tolerance: f32, callback: &mut F)
where
    F:  FnMut(&LineSegment)
{
    flatten_quad_scalar(curve, tolerance, callback);
}


/// Compute an approximation to integral (1 + 4x^2) ^ -0.25 dx used in the flattening code.
pub(crate) fn approx_parabola_integral(x: f32) -> f32 {
    let d = 0.67;
    let quarter = 0.25;
    x / (1.0 - d + (d*d*d*d + quarter * x * x).sqrt().sqrt())
}

/// Approximate the inverse of the function above.
pub(crate) fn approx_parabola_inv_integral(x: f32) -> f32 {
    let b = 0.39;
    let quarter = 0.25;
    x * (1.0 - b + (b * b + quarter * x * x).sqrt())
}

#[derive(Debug)]
pub struct FlatteningParams {
    // Edge count * 2 * sqrt(tolerance).
    pub scaled_count: f32,
    pub integral_from: f32,
    pub integral_to: f32,
    pub inv_integral_from: f32,
    pub div_inv_integral_diff: f32,
}

impl FlatteningParams {
    pub fn new(curve: &QuadraticBezierSegment, sqrt_tolerance: f32) -> Self {
        //println!("quad: {curve:?}");
        // Map the quadratic bézier segment to y = x^2 parabola.
        let ddx = 2.0 * curve.ctrl.x - curve.from.x - curve.to.x;
        let ddy = 2.0 * curve.ctrl.y - curve.from.y - curve.to.y;
        //println!("ddx {ddx:?} ddy {ddy:?}");
        let cross = (curve.to.x - curve.from.x) * ddy - (curve.to.y - curve.from.y) * ddx;
        //println!("cross {cross:?}");
        let parabola_from =
            ((curve.ctrl.x - curve.from.x) * ddx + (curve.ctrl.y - curve.from.y) * ddy) / cross;
        let parabola_to =
            ((curve.to.x - curve.ctrl.x) * ddx + (curve.to.y - curve.ctrl.y) * ddy) / cross;

        let scale = (cross / ((ddx * ddx + ddy * ddy).sqrt() * (parabola_to - parabola_from))).abs();

        let integral_from = approx_parabola_integral(parabola_from);
        let integral_to = approx_parabola_integral(parabola_to);
        //println!("parabola_to {parabola_to:?} parabola_from {parabola_from:?}");
        //println!("intergal_to {integral_to:?} integral_from {integral_from:?}");

        let inv_integral_from = approx_parabola_inv_integral(integral_from);
        let inv_integral_to = approx_parabola_inv_integral(integral_to);
        let div_inv_integral_diff = 1.0 / (inv_integral_to - inv_integral_from);

        // Note: lyon's version doesn't have the cusp handling path.
        // TODO: Kurbo does not not check for zero scale here. However
        // that would cause scaled_count to be NaN when dividing by sqrt_scale.
        let scaled_count = if scale != 0.0 && scale.is_finite() {
            let integral_diff = (integral_to - integral_from).abs();
            let sqrt_scale = scale.sqrt();
            if parabola_from.signum() == parabola_to.signum() {
                //println!("case A integral diff {integral_diff:?} sqrt_scale {sqrt_scale:?}");
                integral_diff * sqrt_scale
            } else {
                //println!("case B (cusp) integral diff {integral_diff:?} sqrt_scale {sqrt_scale:?}");
                // Handle cusp case (segment contains curvature maximum)
                let xmin = sqrt_tolerance / sqrt_scale;
                sqrt_tolerance * integral_diff / approx_parabola_integral(xmin)
            }
        } else {
            0.0
        };

        debug_assert!(scaled_count.is_finite());

        FlatteningParams {
            scaled_count,
            integral_from,
            integral_to,
            inv_integral_from,
            div_inv_integral_diff,
        }
    }

    pub fn get_t(&self, norm_step: f32) -> f32 {
        let u = approx_parabola_inv_integral(self.integral_from + (self.integral_to - self.integral_from) * norm_step);
        let t = (u - self.inv_integral_from) * self.div_inv_integral_diff;

        t
    }
}

pub fn flatten_quad_scalar(curve: &QuadraticBezierSegment, tolerance: f32, cb: &mut impl FnMut(&LineSegment)) {
    let params = FlatteningParams::new(curve, tolerance.sqrt());

    let sqrt_tol = tolerance.sqrt();
    let n = ((0.5 * params.scaled_count / sqrt_tol).ceil() as u32).max(1);
    let step = 1.0 / (n as f32);
    let mut from = curve.from;
    for i in 1..n {
        let u = (i as f32) * step;
        let t = params.get_t(u);
        let to = curve.sample(t);
        cb(&LineSegment { from, to });
        from = to;
    }

    cb(&LineSegment { from, to: curve.to });
}

pub fn flatten_cubic_scalar(curve: &CubicBezierSegment, tolerance: f32, cb: &mut impl FnMut(&LineSegment)) {
    let quads_tolerance = tolerance * 0.1;
    let flatten_tolerance = tolerance * 0.9;
    let sqrt_flatten_tolerance = flatten_tolerance.sqrt();

    let num_quadratics = num_quadratics_impl(curve, quads_tolerance);
    //println!("{num_quadratics:?} quads");

    let mut quads: ArrayVec<(FlatteningParams, QuadraticBezierPolynomial), 16> = ArrayVec::new();

    let quad_step = 1.0 / num_quadratics;
    let num_quadratics = num_quadratics as u32;
    let mut quad_idx = 0;
    let mut t0 = 0.0;

    let mut from = curve.from;

    loop {
        let mut last_quad_to = point(0.0, 0.0);
        let mut sum = 0.0;
        while quad_idx < num_quadratics && quads.capacity() > quads.len() {
            let t1 = t0 + quad_step;
            let quad = curve.split_range(t0..t1).to_quadratic();
            let params = FlatteningParams::new(&quad, sqrt_flatten_tolerance);
            sum += params.scaled_count;
            //println!(" + {quad:?} {t0} .. {t1} scaled count: {:?}", params.scaled_count);
            last_quad_to = quad.to;
            quads.push((params, polynomial_form_quadratic(&quad)));
            t0 = t1;
            quad_idx += 1;
        }

        let num_edges = ((0.5 * sum / sqrt_flatten_tolerance).ceil() as u32).max(1);

        // Iterate through the quadratics, outputting the points of
        // subdivisions that fall within that quadratic.
        let step = sum / (num_edges as f32);
        let mut i = 1;
        let mut scaled_count_sum = 0.0;
        //println!("------ {num_edges:?} edges step {step:?}");
        for (params, quad) in &quads {
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

        cb(&LineSegment { from, to: last_quad_to });
        if quad_idx == num_quadratics {
            break;
        }

        quads.clear();
    }
}

pub fn flatten_cubic_scalar2(curve: &CubicBezierSegment, tolerance: f32, cb: &mut impl FnMut(&LineSegment)) {
    let quads_tolerance = tolerance * 0.1;
    let flatten_tolerance = tolerance * 0.9;
    let sqrt_flatten_tolerance = flatten_tolerance.sqrt();

    let num_quadratics = num_quadratics_impl(curve, quads_tolerance);
    //println!("{num_quadratics:?} quads");

    let mut quads: ArrayVec<(FlatteningParams, QuadraticBezierPolynomial), 16> = ArrayVec::new();

    let quad_step = 1.0 / num_quadratics;
    let num_quadratics = num_quadratics as u32;
    let mut quad_idx = 0;
    let mut t0 = 0.0;

    let mut from = curve.from;

    loop {
        let mut last_quad_to = point(0.0, 0.0);
        let mut sum = 0.0;
        while quad_idx < num_quadratics && quads.capacity() > quads.len() {
            let t1 = t0 + quad_step;
            let quad = curve.split_range(t0..t1).to_quadratic();
            let params = FlatteningParams::new(&quad, sqrt_flatten_tolerance);
            sum += params.scaled_count;
            //println!(" + {quad:?} {t0} .. {t1} scaled count: {:?}", params.scaled_count);
            last_quad_to = quad.to;
            let poly = polynomial_form_quadratic(&quad);
            quads.push((params, poly));
            t0 = t1;
            quad_idx += 1;
        }

        let num_edges = ((0.5 * sum / sqrt_flatten_tolerance).ceil() as u32).max(1);

        // Iterate through the quadratics, outputting the points of
        // subdivisions that fall within that quadratic.
        let step = sum / (num_edges as f32);
        let mut i = 1;
        let mut scaled_count_sum = 0.0;
        //println!("------ {num_edges:?} edges step {step:?}");
        for (params, quad) in &quads {
            //println!("  target {target} ({i})");
            let recip_scaled_count = params.scaled_count.recip();
            let n = u32::min(num_edges, ((scaled_count_sum + params.scaled_count) / step).ceil() as u32);
            while i < n {
                let target = (i as f32) * step;
                let u = (target - scaled_count_sum) * recip_scaled_count;
                let t = params.get_t(u);
                let to = quad.sample(t);
                //println!("     u={u}, t={t}");
                cb(&LineSegment { from, to });
                from = to;
                i += 1;
            }
            scaled_count_sum += params.scaled_count;
        }

        cb(&LineSegment { from, to: last_quad_to });
        if quad_idx == num_quadratics {
            break;
        }

        quads.clear();
    }
}

pub fn num_quadratics_impl(curve: &CubicBezierSegment, tolerance: f32) -> f32 {
    debug_assert!(tolerance > 0.0);

    let x = curve.from.x - 3.0 * curve.ctrl1.x + 3.0 * curve.ctrl2.x - curve.to.x;
    let y = curve.from.y - 3.0 * curve.ctrl1.y + 3.0 * curve.ctrl2.y - curve.to.y;

    let err = x * x + y * y;

    (err / (432.0 * tolerance * tolerance))
        .powf(1.0 / 6.0)
        .ceil()
        .max(1.0)
}

#[test]
fn flat_cusp() {
    use lyon_path::math::point;
    let curve = CubicBezierSegment {
        from: point(0.0, 10.0),
        ctrl1: point(-10.0, 10.0),
        ctrl2: point(180.0, 10.0),
        to: point(60.0, 10.0),
    };

    flatten_cubic_scalar(&curve, 0.1, &mut |seg| {
        println!(" - {seg:?}");
    });
}

#[test]
fn flatten_v2() {
    use lyon_path::math::point;
    let curve = CubicBezierSegment {
        from: point(0.0, 10.0),
        ctrl1: point(-10.0, 10.0),
        ctrl2: point(180.0, 10.0),
        to: point(60.0, 10.0),
    };

    let mut p1 = Vec::new();
    let mut p2 = Vec::new();
    flatten_cubic_scalar(&curve, 0.01, &mut|segment| {
        p1.push(segment.to);
    });
    flatten_cubic_scalar2(&curve, 0.01, &mut|segment| {
        p2.push(segment.to);
    });

    println!("p1: {p1:?}\np2: {p2:?}");
    for (v1, v2) in p1.iter().zip(p2.iter()) {
        assert!(v1.distance_to(*v2) < 0.001);
    }
}
