use crate::{QuadraticBezierSegment, CubicBezierSegment, LineSegment, Point};

use crate::flatness::CubicFlatness;

/// Flatten using a simple recursive algorithm
pub fn flatten_cubic<Flatness, F>(curve: &CubicBezierSegment, tolerance: f32, callback: &mut F)
where
    Flatness: CubicFlatness,
    F:  FnMut(&LineSegment)
{
    if crate::cubic_is_a_point(&curve, tolerance) {
        return;
    }
    let mut prev = curve.from;
    flatten_recursive_cubic_impl::<Flatness, _>(curve, tolerance, callback, &mut prev, 0.0, 1.0);
}

fn flatten_recursive_cubic_impl<Flatness, F>(curve: &CubicBezierSegment, tolerance: f32, callback: &mut F, prev: &mut Point, t0: f32, t1: f32)
where
    Flatness: CubicFlatness,
    F:  FnMut(&LineSegment)
{
    if t1 < t0 + 0.001 || Flatness::is_flat(curve, tolerance) {
        callback(&LineSegment { from: *prev, to: curve.to });
        *prev = curve.to;
        return;
    }

    let t = (t0 + t1) * 0.5;
    let (c0, c1) = curve.split(0.5);

    flatten_recursive_cubic_impl::<Flatness, _>(&c0, tolerance, callback, prev, t0, t);
    flatten_recursive_cubic_impl::<Flatness, _>(&c1, tolerance, callback, prev, t, t1);
}


/// Flatten using a simple recursive algorithm
pub fn flatten_quadratic<F>(curve: &QuadraticBezierSegment, tolerance: f32, callback: &mut F)
where
    F:  FnMut(&LineSegment)
{
    let mut prev = curve.from;
    flatten_recursive_quadratic_impl(curve, tolerance, callback, &mut prev, 0.0, 1.0);
}

fn flatten_recursive_quadratic_impl<F>(curve: &QuadraticBezierSegment, tolerance: f32, callback: &mut F, prev: &mut Point, t0: f32, t1: f32)
where
    F:  FnMut(&LineSegment)
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
