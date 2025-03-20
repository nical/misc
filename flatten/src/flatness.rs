use lyon_path::geom::{CubicBezierSegment, Vector};


pub trait CubicFlatness {
    fn is_flat(curve: &CubicBezierSegment<f32>, tolerance: f32) -> bool;
}

pub trait QuadraticFlatness {
    fn is_flat(curve: &CubicBezierSegment<f32>, tolerance: f32) -> bool;
}

pub struct DefaultFlatness;
impl CubicFlatness for DefaultFlatness {
    fn is_flat(curve: &CubicBezierSegment<f32>, tolerance: f32) -> bool {
        cubic_is_flat(curve, tolerance)
    }
}

pub struct HfdFlatness;
impl CubicFlatness for HfdFlatness {
    fn is_flat(curve: &CubicBezierSegment<f32>, tolerance: f32) -> bool {
        fn approx_norm(v: &Vector<f32>) -> f32 {
            v.x.abs().max(v.y.abs())
        }

        let v212 = curve.ctrl1 - curve.from;
        let v214 = curve.ctrl2 - curve.ctrl1;
        let v216 = curve.to - curve.ctrl2;
        let v218 = v214 - v212;
        let v220 = v216 - v214;
        approx_norm(&v218).max(approx_norm(&v220)) <= tolerance
    }
}

/// The flatness criterion described in https://agg.sourceforge.net/antigrain.com/research/adaptive_bezier/index.html
pub struct AggFlatness;
impl CubicFlatness for AggFlatness {
    fn is_flat(curve: &CubicBezierSegment<f32>, tolerance: f32) -> bool {
        let baseline = curve.to - curve.from;
        let c1 = baseline.cross(curve.ctrl1 - curve.to);
        let c2 = baseline.cross(curve.ctrl2 - curve.to);

        let flat = (c1 + c2) * (c1 + c2) <= tolerance * tolerance * baseline.square_length();

        flat
    }
}

/// Returns true if the curve can be approximated with a single line segment, given
/// a tolerance threshold.
pub fn cubic_is_flat(curve: &CubicBezierSegment<f32>, tolerance: f32) -> bool {
    // Similar to Line::square_distance_to_point, except we keep
    // the sign of c1 and c2 to compute tighter upper bounds as we
    // do in fat_line_min_max.
    let baseline = curve.to - curve.from;
    let v1 = curve.ctrl1 - curve.from;
    let v2 = curve.ctrl2 - curve.from;
    let v3 = curve.ctrl2 - curve.to;

    let c1 = baseline.cross(v1);
    let c2 = baseline.cross(v2);
    // TODO: it would be faster to multiply the threshold with baseline_len2
    // instead of dividing d1 and d2, but it changes the behavior when the
    // baseline length is zero in ways that breaks some of the cubic intersection
    // tests.
    let inv_baseline_len2 = 1.0 / baseline.square_length();
    let d1 = (c1 * c1) * inv_baseline_len2;
    let d2 = (c2 * c2) * inv_baseline_len2;

    let factor = if (c1 * c2) > 0.0 {
        3.0 / 4.0
    } else {
        4.0 / 9.0
    };

    let f2 = factor * factor;
    let threshold = tolerance * tolerance;

    // Check whether the curve is a point.
    // TODO: It is not great to have to do this at each step
    // it would probably be better to check this early in the
    // flattening algorithms and then only check that the
    // subdivision is not an empty range.
    // if baseline.square_length() < tolerance
    //     && v1.square_length() < tolerance
    //     && v2.square_length() < tolerance {
    //     return true;
    // }

    d1 * f2 <= threshold
        && d2 * f2 <= threshold
        // TODO: This check is missing from CubicBezierSegment::is_linear, which
        // misses flat-ish curves with control points that aren't between the
        // endpoints.
        && baseline.dot(v1) > -tolerance
        && baseline.dot(v3) < tolerance
}
