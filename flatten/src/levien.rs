use lyon_path::geom::{QuadraticBezierSegment, CubicBezierSegment, LineSegment};

pub fn flatten_cubic_19<F>(curve: &CubicBezierSegment<f32>, tolerance: f32, callback: &mut F)
where
    F:  FnMut(&LineSegment<f32>)
{
    let quads_tolerance = tolerance * 0.1;
    let flatten_tolerance = tolerance * 0.9;
    curve.for_each_quadratic_bezier(quads_tolerance, &mut |quad| {
        flatten_quad_scalar(quad, flatten_tolerance, callback);
    });
}

pub fn flatten_cubic_28<F>(curve: &CubicBezierSegment<f32>, tolerance: f32, callback: &mut F)
where
    F:  FnMut(&LineSegment<f32>)
{
    let quads_tolerance = tolerance * 0.2;
    let flatten_tolerance = tolerance * 0.8;
    curve.for_each_quadratic_bezier(quads_tolerance, &mut |quad| {
        flatten_quad_scalar(quad, flatten_tolerance, callback);
    });
}

pub fn flatten_cubic_37<F>(curve: &CubicBezierSegment<f32>, tolerance: f32, callback: &mut F)
where
    F:  FnMut(&LineSegment<f32>)
{
    let quads_tolerance = tolerance * 0.3;
    let flatten_tolerance = tolerance * 0.7;
    curve.for_each_quadratic_bezier(quads_tolerance, &mut |quad| {
        flatten_quad_scalar(quad, flatten_tolerance, callback);
    });
}

pub fn flatten_cubic_55<F>(curve: &CubicBezierSegment<f32>, tolerance: f32, callback: &mut F)
where
    F:  FnMut(&LineSegment<f32>)
{
    let quads_tolerance = tolerance * 0.5;
    let flatten_tolerance = tolerance * 0.5;
    curve.for_each_quadratic_bezier(quads_tolerance, &mut |quad| {
        flatten_quad_scalar(quad, flatten_tolerance, callback);
    });
}

pub fn flatten_quadratic<F>(curve: &QuadraticBezierSegment<f32>, tolerance: f32, callback: &mut F)
where
    F:  FnMut(&LineSegment<f32>)
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

pub fn flatten_quad_scalar(curve: &QuadraticBezierSegment<f32>, tolerance: f32, cb: &mut impl FnMut(&LineSegment<f32>)) {
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
    let scale = (cross / ((ddx * ddx + ddy * ddy).sqrt() * (parabola_to - parabola_from))).abs();

    let integral_from = approx_parabola_integral(parabola_from);
    let integral_to = approx_parabola_integral(parabola_to);
    let integral_diff = integral_to - integral_from;

    let inv_integral_from = approx_parabola_inv_integral(integral_from);
    let inv_integral_to = approx_parabola_inv_integral(integral_to);
    let div_inv_integral_diff = 1.0 / (inv_integral_to - inv_integral_from);

    // We could store this as an integer but the generic code makes that awkward and we'll
    // use it as a scalar again while iterating, so it's kept as a scalar.
    let mut count = (0.5 * integral_diff.abs() * (scale / tolerance).sqrt()).ceil();
    // If count is NaN the curve can be approximated by a single straight line or a point.
    if !count.is_finite() {
        count = 0.0;
    }

    let integral_step = integral_diff / count;

    let mut from = curve.from;
    let mut i = 1.0;
    for _ in 1..(count as u32) {
        let u = approx_parabola_inv_integral(integral_from + integral_step * i);
        let t = (u - inv_integral_from) * div_inv_integral_diff;
        i += 1.0;
        let to = curve.sample(t);
        cb(&LineSegment { from, to });
        from = to;
    }

    cb(&LineSegment { from, to: curve.to });
}
