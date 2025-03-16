use lyon_path::geom::{QuadraticBezierSegment, CubicBezierSegment, LineSegment};

use crate::cubic_is_linear;

/// Same as `flatten_linear` with an optimization to more quickly find the split point.
pub fn flatten_cubic<F>(curve: &CubicBezierSegment<f32>, tolerance: f32, callback: &mut F)
where
    F:  FnMut(&LineSegment<f32>)
{
    let mut rem = *curve;
    let mut from = rem.from;

    let mut split = 0.5;
    loop {
        if cubic_is_linear(&rem, tolerance) {
            callback(&LineSegment { from, to: rem.to });
            return;
        }

        loop {
            let sub = rem.before_split(split);
            if cubic_is_linear(&sub, tolerance) {
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
