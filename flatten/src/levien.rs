use lyon_path::geom::{QuadraticBezierSegment, CubicBezierSegment, LineSegment};

pub fn flatten_cubic<F>(curve: &CubicBezierSegment<f32>, tolerance: f32, callback: &mut F)
where
    F:  FnMut(&LineSegment<f32>)
{
    curve.for_each_flattened(tolerance, callback);
}

pub fn flatten_quadratic<F>(curve: &QuadraticBezierSegment<f32>, tolerance: f32, callback: &mut F)
where
    F:  FnMut(&LineSegment<f32>)
{
    curve.for_each_flattened(tolerance, callback);
}
