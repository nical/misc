/// The algorithm implemented here is based on:
///     "Fast, precise flattening of cubic Bézier path and offset curves"
///     http://sibgrapi.sid.inpe.br/col/sid.inpe.br/banon/2004/08.13.18.12/doc/BezierOffsetRendering.pdf
///
/// The basic premise is that for a small t the third order term in the
/// equation of a cubic bezier curve is insignificantly small. This can
/// then be approximated by a quadratic equation for which the maximum
/// difference from a linear approximation can be much more easily determined.

use crate::{CubicBezierSegment, LineSegment};

const EPSILON: f32 = 1e-4;

#[inline(always)]
fn in_range(t: f32) -> bool {
    t >= 0.0 && t < 1.0
}


pub fn flatten_cubic<F: FnMut(&LineSegment)>(
    bezier: &CubicBezierSegment,
    tolerance: f32,
    call_back: &mut F,
) {
    let mut inflection_1 = -1.0;
    let mut inflection_2 = 2.0;
    let inflection_count = find_cubic_bezier_inflection_points(&bezier, &mut inflection_1, &mut inflection_2);

    if inflection_count == 0 {
        flatten_cubic_no_inflection(*bezier, tolerance, call_back);
        return;
    }

    // From now on we know that we have one or two inflection points.

    let mut t1_min = inflection_1;
    let mut t1_max = inflection_1;
    inflection_approximation_range(bezier, inflection_1, tolerance, &mut t1_min, &mut t1_max);

    let mut t2_min = inflection_2;
    let mut t2_max = inflection_2;
    if inflection_count == 2 {
        inflection_approximation_range(bezier, inflection_2, tolerance, &mut t2_min, &mut t2_max);
    }

    if inflection_count == 1 && t1_min < 0.0 && t1_max > 1.0 {
        // The first inflection range covers the entire curve.
        call_back(&bezier.baseline());
        return;
    }

    let mut from = bezier.from;

    if t1_min > 0.0 {
        let before = bezier.before_split(t1_min);
        flatten_cubic_no_inflection(before, tolerance, call_back);
        from = before.to;
    }

    if in_range(t1_max) && (inflection_count == 1 || t2_min > t1_max) {
        // There is no second inflection point or the second inflection point's range is in the first's.
        // Add a line to the end of the first approximation range;
        let after = bezier.after_split(t1_max);
        call_back(&LineSegment { from, to: after.from });
        from = after.from;

        if inflection_count == 1 || t2_min > 1.0 {
            flatten_cubic_no_inflection(after, tolerance, call_back);
            from = after.to;
        }
    } else if inflection_count == 2 && t2_min > 1.0 {
        call_back(&LineSegment { from, to: bezier.to });
        return;
    }

    if inflection_count == 2 && t2_min < 1.0 && t2_max > 0.0 {
        if t2_min > 0.0 && t2_min < t1_max {
            // Skip t2_min since it is in the first approximation range.
            let to = bezier.sample(t1_max);
            call_back(&LineSegment { from, to });
            from = to;
        } else if t2_min > 0.0 && t1_max > 0.0 {
            let curve = bezier.split_range(t1_max..t2_min);
            flatten_cubic_no_inflection(curve, tolerance, call_back);
            from = curve.to;
        } else if t2_min > 0.0 {
            let before = bezier.before_split(t2_min);
            flatten_cubic_no_inflection(before, tolerance, call_back);
            from = before.to;
        }

        if t2_max < 1.0 {
            let after = bezier.after_split(t1_max);
            call_back(&LineSegment { from, to: after.from });
            flatten_cubic_no_inflection(after, tolerance, call_back);
        } else {
            call_back(&LineSegment { from, to: bezier.from });
        }
    }
}

fn flatten_cubic_no_inflection<F: FnMut(&LineSegment)>(
    mut curve: CubicBezierSegment,
    tolerance: f32,
    call_back: &mut F,
) {
    let end = curve.to;
    let mut from = curve.from;

    loop {
        let step = no_inflection_flattening_step(&curve, tolerance);

        if step >= 1.0 {
            call_back(&LineSegment { from, to: end });
            break;
        }

        curve = curve.after_split(step);
        call_back(&LineSegment { from, to: curve.from });
        from = curve.from;
    }
}

fn no_inflection_flattening_step(bezier: &CubicBezierSegment, tolerance: f32) -> f32 {
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
    let s2inv = (v1.x * v1.x + v1.y * v1.y).sqrt() / (3.0 * v2_cross_v1);

    // The paper says that we should be able to split at 2.0 * t,
    // But that produces incorrect results at most of the tolerance threasholds
    // used here.
    // Note: The assumption of this algorithm is that there is a cubic
    // term that is negligible when t is "small enough". The paper used
    // a flatness criterion of 0.0005 for testing which is way smaller
    // than what the other algorithms are targetting (for example 0.25).
    // So it could be that this algorithm is only adequate for very
    // small tolerance thresholds.
    //
    // It looks like the flattening quality indeed holds up at very small
    // tolerance thresholds, so we tweak the factor to get close to the
    // constant two for small tolerances and closer to 1.0 otherwise.
    let factor = 2.0 - (tolerance * 4.0).min(1.0);

    // Note: We are using f' = tolerance
    // whereas the paper has f' = tolerance / (1 - d * s2 / (3 r1²))
    // with d being the line width. We don't have a line width here
    // so d=0.
    let t = factor * f32::sqrt(tolerance * f32::abs(s2inv));

    // TODO: We start having floating point precision issues if this constant
    // is closer to 1.0 with a small enough tolerance threshold.
    if t >= 0.995 || t == 0.0 {
        return 1.0;
    }

    return t;
}

// Find the inflection points of a cubic bezier curve.
pub(crate) fn find_cubic_bezier_inflection_points(
    bezier: &CubicBezierSegment,
    t1: &mut f32,
    t2: &mut f32,
) -> u32 {
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
                *t1 = 0.0;
                return 1;
            }
        } else {
            let t = -c / b;
            if in_range(t) {
                *t1 = t;
                return 1;
            }
        }

        return 0;
    }

    let discriminant = b * b - 4.0 * a * c;

    if discriminant < 0.0 {
        return 0;
    }

    if discriminant < EPSILON {
        let t = -b / (2.0 * a);

        *t1 = t;
        return 1;
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

    let mut next = t1;
    let mut count = 0;

    if in_range(first_inflection) {
        *next = first_inflection;
        next = t2;
        count += 1;
    }

    if in_range(second_inflection) {
        *next = second_inflection;
        count += 1;
    }

    return count
}

// Find the range around the start of the curve where the curve can locally be approximated
// with a line segment, given a tolerance threshold.
fn inflection_approximation_range(
    bezier: &CubicBezierSegment,
    t: f32,
    tolerance: f32,
    t_min: &mut f32,
    t_max: &mut f32,
) {
    let next_curve = bezier.after_split(t);

    // Transform the curve such that it starts at the origin.
    let mut p1 = next_curve.ctrl1 - next_curve.from;
    let p2 = next_curve.ctrl2 - next_curve.from;
    let p3 = next_curve.to - next_curve.from;

    if p1.x == 0.0 && p1.y == 0.0 {
        p1 = p2
    }

    if p1.x == 0.0 && p1.y == 0.0 {
        let s3 = p3.x - p3.y;

        if s3 == 0.0 {
            *t_min = -1.0;
            *t_max = 2.0;
        } else {
            let r = (tolerance / s3).abs().powf(1.0/3.0);
            *t_min = t - r;
            *t_max = t + r;
        }
        return;
    }

    // Thus, curve(t) = t^3 * (3*p1 - 3*p2 + p3) + t^2 * (-6*p1 + 3*p2) + t * (3*p1).
    // Since curve(0) is an inflection point, cross(p1, p2) = 0, i.e. p1 and p2 are parallel.

    let s3 = p2.cross(p3) / p2.length();
    if s3 == 0.0 {
        *t_min = -1.0;
        *t_max = 2.0;
        return;
    }

    let r_next = (tolerance / s3).abs().powf(1.0/3.0);
    let r = r_next * (1.0 - t);
    *t_min = t - r;
    *t_max = t + r;
}


// A few curves that produce poor approximations with this algorithm (tolerance 0.01):
// CubicBezierSegment { from: (523.588, 182.392), ctrl1: (523.588, 182.481), ctrl2: (523.554, 182.549), to: (523.554, 182.631) }, error = 157.79745
// CubicBezierSegment { from: (523.588, 182.392), ctrl1: (523.588, 182.481), ctrl2: (523.554, 182.549), to: (523.554, 182.631) }, error = 224.91916
// CubicBezierSegment { from: (525.16, 186.036), ctrl1: (525.21, 186.081), ctrl2: (525.209, 186.165), to: (525.227, 186.23) }, error = 160.81334
// CubicBezierSegment { from: (525.16, 186.036), ctrl1: (525.21, 186.081), ctrl2: (525.209, 186.165), to: (525.227, 186.23) }, error = 265.02643
// CubicBezierSegment { from: (525.16, 186.036), ctrl1: (525.21, 186.081), ctrl2: (525.209, 186.165), to: (525.227, 186.23) }, error = 341.30396
// CubicBezierSegment { from: (517.937, 185.96), ctrl1: (517.965, 185.859), ctrl2: (518.033, 185.777), to: (518.055, 185.676) }, error = 276.2007
// CubicBezierSegment { from: (517.937, 185.96), ctrl1: (517.965, 185.859), ctrl2: (518.033, 185.777), to: (518.055, 185.676) }, error = 370.13004
// CubicBezierSegment { from: (517.937, 185.96), ctrl1: (517.965, 185.859), ctrl2: (518.033, 185.777), to: (518.055, 185.676) }, error = 490.8672
// CubicBezierSegment { from: (517.937, 185.96), ctrl1: (517.965, 185.859), ctrl2: (518.033, 185.777), to: (518.055, 185.676) }, error = 644.92163
// CubicBezierSegment { from: (517.937, 185.96), ctrl1: (517.965, 185.859), ctrl2: (518.033, 185.777), to: (518.055, 185.676) }, error = 840.21936
// CubicBezierSegment { from: (517.937, 185.96), ctrl1: (517.965, 185.859), ctrl2: (518.033, 185.777), to: (518.055, 185.676) }, error = 1086.4247
// CubicBezierSegment { from: (517.937, 185.96), ctrl1: (517.965, 185.859), ctrl2: (518.033, 185.777), to: (518.055, 185.676) }, error = 1395.3298
// CubicBezierSegment { from: (517.937, 185.96), ctrl1: (517.965, 185.859), ctrl2: (518.033, 185.777), to: (518.055, 185.676) }, error = 1781.3925
// CubicBezierSegment { from: (517.937, 185.96), ctrl1: (517.965, 185.859), ctrl2: (518.033, 185.777), to: (518.055, 185.676) }, error = 2262.2986
// CubicBezierSegment { from: (517.937, 185.96), ctrl1: (517.965, 185.859), ctrl2: (518.033, 185.777), to: (518.055, 185.676) }, error = 2859.9673
