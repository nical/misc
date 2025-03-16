use lyon_path::geom::{QuadraticBezierSegment, CubicBezierSegment, LineSegment, Point};

use crate::cubic_is_linear;

/// Flatten using a simple recursive algorithm
pub fn flatten_cubic<F>(curve: &CubicBezierSegment<f32>, tolerance: f32, callback: &mut F)
where
    F:  FnMut(&LineSegment<f32>)
{
    let mut prev = curve.from;
    flatten_recursive_cubic_impl(curve, tolerance, callback, &mut prev, 0.0, 1.0);
}

fn flatten_recursive_cubic_impl<F>(curve: &CubicBezierSegment<f32>, tolerance: f32, callback: &mut F, prev: &mut Point<f32>, t0: f32, t1: f32)
where
    F:  FnMut(&LineSegment<f32>)
{
    if cubic_is_linear(curve, tolerance) {
        callback(&LineSegment { from: *prev, to: curve.to });
        *prev = curve.to;
        return;
    }

    let t = (t0 + t1) * 0.5;
    let (c0, c1) = curve.split(0.5);

    flatten_recursive_cubic_impl(&c0, tolerance, callback, prev, t0, t);
    flatten_recursive_cubic_impl(&c1, tolerance, callback, prev, t, t1);
}


/// Flatten using a simple recursive algorithm
pub fn flatten_quadratic<F>(curve: &QuadraticBezierSegment<f32>, tolerance: f32, callback: &mut F)
where
    F:  FnMut(&LineSegment<f32>)
{
    let mut prev = curve.from;
    flatten_recursive_quadratic_impl(curve, tolerance, callback, &mut prev, 0.0, 1.0);
}

fn flatten_recursive_quadratic_impl<F>(curve: &QuadraticBezierSegment<f32>, tolerance: f32, callback: &mut F, prev: &mut Point<f32>, t0: f32, t1: f32)
where
    F:  FnMut(&LineSegment<f32>)
{
    if curve.is_linear(tolerance) {
        callback(&LineSegment { from: *prev, to: curve.to });
        *prev = curve.to;
        return;
    }

    let t = (t0 + t1) * 0.5;
    let (c0, c1) = curve.split(0.5);

    flatten_recursive_quadratic_impl(&c0, tolerance, callback, prev, t0, t);
    flatten_recursive_quadratic_impl(&c1, tolerance, callback, prev, t, t1);
}
