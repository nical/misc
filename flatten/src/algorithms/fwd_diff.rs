use crate::{QuadraticBezierSegment, CubicBezierSegment, LineSegment};

use crate::{polynomial_form_cubic, polynomial_form_quadratic};

/// Flatten using forward difference
///
/// This is the simple (non-adaptative) version of the forward difference
/// algorithm, pre-calculating the number of edges using the formula given
/// in section 10.6 of CAGD.
pub fn flatten_cubic<F>(curve: &CubicBezierSegment, tolerance: f32, callback: &mut F)
where
    F:  FnMut(&LineSegment)
{
    let poly = polynomial_form_cubic(&curve);
    let n = crate::wang::num_segments_cubic(curve, tolerance);
    let dt = 1.0 / n;

    let mut from = curve.from;
    let mut to = from;

    let mut fd1 = poly.a1 * dt;
    let mut fd2 = poly.a2 * 2.0 * dt * dt;
    let fd3 = poly.a3 * 6.0 * dt * dt * dt;

    for _ in 1..(n as u32) {
        to += fd1 + fd2 * 0.5 + fd3 / 6.0;
        fd1 += fd2 + fd3 * 0.5;
        fd2 += fd3;

        callback(&LineSegment { from, to });

        from = to;
    }

    callback(&LineSegment { from, to: curve.to });
}

/// Flatten using forward difference
///
/// This is the simple (non-adaptative) version of the forward difference
/// algorithm, pre-calculating the number of edges using the formula given
/// in section 10.6 of CAGD.
pub fn flatten_quadratic<F>(curve: &QuadraticBezierSegment, tolerance: f32, callback: &mut F)
where
    F:  FnMut(&LineSegment)
{
    let poly = polynomial_form_quadratic(&curve);
    let n = crate::wang::num_segments_quadratic(curve, tolerance);
    let dt = 1.0 / n;

    let mut from = curve.from;
    let mut to = from;

    let mut fd1 = poly.a1 * dt;
    let fd2 = poly.a2 * 2.0 * dt * dt;

    for _ in 1..(n as u32) {
        to += fd1 + fd2 * 0.5;
        fd1 += fd2;

        callback(&LineSegment { from, to });

        from = to;
    }

    callback(&LineSegment { from, to: curve.to });
}
