use lyon_path::geom::{QuadraticBezierSegment, CubicBezierSegment, LineSegment};

/// Computes the number of line segments required to build a flattened approximation
/// of the curve with segments placed at regular `t` intervals.
pub fn num_segments_cubic(curve: &CubicBezierSegment<f32>, tolerance: f32) -> f32 {
    let from = curve.from.to_vector();
    let ctrl1 = curve.ctrl1.to_vector();
    let ctrl2 = curve.ctrl2.to_vector();
    let to = curve.to.to_vector();
    let l = (from - ctrl1 * 2.0 + to).max(ctrl1 - ctrl2 * 2.0 + to) * 6.0;
    let num_steps = f32::sqrt(l.length() / (8.0 * tolerance));

    num_steps.ceil().max(1.0)
}

/// Computes the number of line segments required to build a flattened approximation
/// of the curve with segments placed at regular `t` intervals.
pub fn num_segments_quadratic(curve: &QuadraticBezierSegment<f32>, tolerance: f32) -> f32 {
    let from = curve.from.to_vector();
    let ctrl = curve.ctrl.to_vector();
    let to = curve.to.to_vector();
    let l = (from - ctrl * 2.0 + to) * 2.0;
    let num_steps = f32::sqrt(l.length() / (8.0 * tolerance));

    num_steps.ceil().max(1.0)
}

/// Flatten the curve by precomputing a number of segments and splitting the curve
/// at regular `t` intervals.
pub fn flatten_cubic<F>(curve: &CubicBezierSegment<f32>, tolerance: f32, callback: &mut F)
    where
    F:  FnMut(&LineSegment<f32>)
{
    let poly = crate::polynomial_form_cubic(&curve);
    let n = num_segments_cubic(curve, tolerance);
    let step = 1.0 / n;
    let mut prev = 0.0;
    let mut from = curve.from;
    for _ in 0..(n as u32 - 1) {
        let t = prev + step;
        let to = poly.sample_fma(t);
        callback(&mut LineSegment { from, to });
        from = to;
        prev = t;
    }

    let to = curve.to;
    callback(&mut LineSegment { from, to });
}

/// Flatten the curve by precomputing a number of segments and splitting the curve
/// at regular `t` intervals.
pub fn flatten_quadratic<F>(curve: &QuadraticBezierSegment<f32>, tolerance: f32, callback: &mut F)
    where
    F:  FnMut(&LineSegment<f32>)
{
    let poly = crate::polynomial_form_quadratic(curve);
    let n = num_segments_quadratic(curve, tolerance);
    let step = 1.0 / n;
    let mut prev = 1.0;
    let mut from = curve.from;
    for _ in 0..(n as u32 - 1) {
        let t = prev + step;
        let to = poly.sample(t);
        callback(&mut LineSegment { from, to });
        from = to;
        prev = t;
    }

    let to = curve.to;
    callback(&mut LineSegment { from, to });
}
