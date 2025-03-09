use lyon_path::geom::{CubicBezierSegment, LineSegment};
use arrayvec::ArrayVec;

const EPSILON: f32 = 1e-4;

pub fn flatten_cubic<F: FnMut(&LineSegment<f32>)>(
    bezier: &CubicBezierSegment<f32>,
    tolerance: f32,
    call_back: &mut F,
) {
    let mut bezier = *bezier;
    let mut inflections: ArrayVec<f32, 2> = ArrayVec::new();
    find_cubic_bezier_inflection_points(&bezier, &mut |t| {
        inflections.push(t);
    });

    for t_inflection in inflections {
        let (before, mut after) = bezier.split(t_inflection);

        // Flatten up to the inflection point.
        flatten_cubic_no_inflection(before, tolerance, call_back);

        // Approximate the inflection with a segment if need be.
        if let Some(tf) = inflection_approximation_range(&after, tolerance) {
            let from = after.from;
            after = after.after_split(tf);
            call_back(&LineSegment { from, to: after.from});
        }

        bezier = after;
    }

    // Do the rest of the curve.
    flatten_cubic_no_inflection(bezier, tolerance, call_back);
}

// The algorithm implemented here is based on: "Fast, precise flattening of cubic Bézier path and offset curves"
// http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.106.5344&rep=rep1&type=pdf
//
// The basic premise is that for a small t the third order term in the
// equation of a cubic bezier curve is insignificantly small. This can
// then be approximated by a quadratic equation for which the maximum
// difference from a linear approximation can be much more easily determined.
fn flatten_cubic_no_inflection<F: FnMut(&LineSegment<f32>)>(
    mut curve: CubicBezierSegment<f32>,
    tolerance: f32,
    call_back: &mut F,
) {
    let end = curve.to;
    let mut from = curve.from;

    loop {
        let step = no_inflection_flattening_step(&curve, tolerance);

        if step >= 1.0 {
            if !crate::cubic_is_a_point(&curve, 0.0) {
                call_back(&LineSegment { from, to: end });
            }

            break;
        }
        curve = curve.after_split(step);
        call_back(&LineSegment { from, to: curve.from });
        from = curve.from;
    }
}

fn no_inflection_flattening_step(bezier: &CubicBezierSegment<f32>, tolerance: f32) -> f32 {
    let v1 = bezier.ctrl1 - bezier.from;
    let v2 = bezier.ctrl2 - bezier.from;

    // This function assumes that the bézier segment is not starting at an inflection point,
    // otherwise the following cross product may result in very small numbers which will hit
    // floating point precision issues.

    // To remove divisions and check for divide-by-zero, this is optimized from:
    // s2 = (v2.x * v1.y - v2.y * v1.x) / hypot(v1.x, v1.y);
    // t = sqrt(tolerance / (3. * abs(s2)));
    let v2_cross_v1 = v2.cross(v1);
    if v2_cross_v1 == 0.0 {
        return 1.0;
    }
    let s2inv = v1.x.hypot(v1.y) / (3.0 * v2_cross_v1);

    let t = f32::sqrt(tolerance * f32::abs(s2inv));

    // TODO: We start having floating point precision issues if this constant
    // is closer to 1.0 with a small enough tolerance threshold.
    if t >= 0.995 || t == 0.0 {
        return 1.0;
    }

    return t;
}

// Find the inflection points of a cubic bezier curve.
pub(crate) fn find_cubic_bezier_inflection_points<F>(bezier: &CubicBezierSegment<f32>, cb: &mut F)
where
    F: FnMut(f32),
{
    // Find inflection points.
    // See www.faculty.idc.ac.il/arik/quality/appendixa.html for an explanation
    // of this approach.
    let pa = bezier.ctrl1 - bezier.from;
    let pb =
        bezier.ctrl2.to_vector() - (bezier.ctrl1.to_vector() * 2.0) + bezier.from.to_vector();
    let pc = bezier.to.to_vector() - (bezier.ctrl2.to_vector() * 3.0)
        + (bezier.ctrl1.to_vector() * 3.0)
        - bezier.from.to_vector();

    let a = pb.cross(pc);
    let b = pa.cross(pc);
    let c = pa.cross(pb);

    if f32::abs(a) < EPSILON {
        // Not a quadratic equation.
        if f32::abs(b) < EPSILON {
            // Instead of a linear acceleration change we have a constant
            // acceleration change. This means the equation has no solution
            // and there are no inflection points, unless the constant is 0.
            // In that case the curve is a straight line, essentially that means
            // the easiest way to deal with is is by saying there's an inflection
            // point at t == 0. The inflection point approximation range found will
            // automatically extend into infinity.
            if f32::abs(c) < EPSILON {
                cb(0.0);
            }
        } else {
            let t = -c / b;
            if in_range(t) {
                cb(t);
            }
        }

        return;
    }

    fn in_range(t: f32) -> bool {
        t >= 0.0 && t < 1.0
    }

    let discriminant = b * b - 4.0 * a * c;

    if discriminant < 0.0 {
        return;
    }

    if discriminant < EPSILON {
        let t = -b / (2.0 * a);

        if in_range(t) {
            cb(t);
        }

        return;
    }

    // This code is derived from https://www2.units.it/ipl/students_area/imm2/files/Numerical_Recipes.pdf page 184.
    // Computing the roots this way avoids precision issues when a, c or both are small.
    let discriminant_sqrt = f32::sqrt(discriminant);
    let sign_b = if b >= 0.0 { 1.0 } else { -1.0 };
    let q = -0.5 * (b + sign_b * discriminant_sqrt);
    let mut first_inflection = q / a;
    let mut second_inflection = c / q;

    if first_inflection > second_inflection {
        std::mem::swap(&mut first_inflection, &mut second_inflection);
    }

    if in_range(first_inflection) {
        cb(first_inflection);
    }

    if in_range(second_inflection) {
        cb(second_inflection);
    }
}

// Find the range around the start of the curve where the curve can locally be approximated
// with a line segment, given a tolerance threshold.
fn inflection_approximation_range(
    bezier: &CubicBezierSegment<f32>,
    tolerance: f32,
) -> Option<f32> {
    // Transform the curve such that it starts at the origin.
    let p1 = bezier.ctrl1 - bezier.from;
    let p2 = bezier.ctrl2 - bezier.from;
    let p3 = bezier.to - bezier.from;

    // Thus, curve(t) = t^3 * (3*p1 - 3*p2 + p3) + t^2 * (-6*p1 + 3*p2) + t * (3*p1).
    // Since curve(0) is an inflection point, cross(p1, p2) = 0, i.e. p1 and p2 are parallel.

    // Let s(t) = s3 * t^3 be the (signed) perpendicular distance of curve(t) from a line that will be determined below.
    let s3;
    if f32::abs(p1.x) < EPSILON && f32::abs(p1.y) < EPSILON {
        // Assume p1 = 0.
        if f32::abs(p2.x) < EPSILON && f32::abs(p2.y) < EPSILON {
            // Assume p2 = 0.
            // The curve itself is a line or a point.
            return None;
        } else {
            // In this case p2 is away from zero.
            // Choose the line in direction p2.
            s3 = p2.cross(p3) / p2.length();
        }
    } else {
        // In this case p1 is away from zero.
        // Choose the line in direction p1 and use that p1 and p2 are parallel.
        s3 = p1.cross(p3) / p1.length();
    }

    // Calculate the maximal t value such that the (absolute) distance is within the tolerance.
    let tf = f32::abs(tolerance / s3).powf(1.0 / 3.0);

    return if tf < 1.0 { Some(tf) } else { None };
}
