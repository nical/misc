use crate::{CubicBezierSegment, LineSegment, QuadraticBezierSegment};
use crate::flatness::{quadratic_is_flat, CubicFlatness};

pub fn flatten_cubic<Flat, F>(curve: &CubicBezierSegment, tolerance: f32, callback: &mut F)
where
    Flat: CubicFlatness,
    F:  FnMut(&LineSegment)
{
    if crate::cubic_is_a_point(&curve, tolerance) {
        return;
    }

    let mut rem = *curve;
    let mut from = rem.from;

    let mut split = 0.5;
    loop {
        // Only check flatness of the entire remaining chunk if
        // we are not in the process of doing very fine subdivision.
        // This gives a 9% improvement.
        if split >= 0.25 && Flat::is_flat(&rem, tolerance) {
            callback(&LineSegment { from, to: rem.to });
            return;
        }

        loop {
            let sub = rem.before_split(split);
            if Flat::is_flat(&sub, tolerance) {
                callback(&LineSegment { from, to: sub.to });
                from = sub.to;
                rem = rem.after_split(split);
                let next_split = split * 2.0;
                if next_split < 1.0 {
                    split = next_split;
                }
                break;
            }
            split *= 0.5;
        }
    }
}

pub fn flatten_quadratic<F>(curve: &QuadraticBezierSegment, tolerance: f32, callback: &mut F)
where
    F:  FnMut(&LineSegment)
{
    let mut rem = *curve;
    let mut from = rem.from;

    let mut split = 0.5;
    loop {
        if split >= 0.25 && quadratic_is_flat(&rem, tolerance) {
            callback(&LineSegment { from, to: rem.to });
            return;
        }

        loop {
            let sub = rem.before_split(split);
            if quadratic_is_flat(&sub, tolerance) {
                callback(&LineSegment { from, to: sub.to });
                from = sub.to;
                rem = rem.after_split(split);
                let next_split = split * 2.0;
                if next_split < 1.0 {
                    split = next_split;
                }
                break;
            }
            split *= 0.5;
        }
    }
}
