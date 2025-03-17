use lyon_path::{geom::{CubicBezierSegment, LineSegment, QuadraticBezierSegment}, math::point};

use crate::flatness::{CubicFlatness};

/// Same as `flatten_linear` with an optimization to more quickly find the split point.
pub fn flatten_cubic<Flat, F>(curve: &CubicBezierSegment<f32>, tolerance: f32, callback: &mut F)
where
    Flat: CubicFlatness,
    F:  FnMut(&LineSegment<f32>)
{
    let mut rem = *curve;
    let mut from = rem.from;

    let mut split = 0.5;
    loop {
        if Flat::is_flat(&rem, tolerance) {
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

pub fn flatten_quadratic<F>(curve: &QuadraticBezierSegment<f32>, tolerance: f32, callback: &mut F)
where
    F:  FnMut(&LineSegment<f32>)
{
    let mut rem = *curve;
    let mut from = rem.from;

    let mut split = 0.5;
    loop {
        if rem.is_linear(tolerance) {
            callback(&LineSegment { from, to: rem.to });
            return;
        }

        loop {
            let sub = rem.before_split(split);
            if sub.is_linear(tolerance) {
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
