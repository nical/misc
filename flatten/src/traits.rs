use crate::{CubicBezierSegment, LineSegment, QuadraticBezierSegment, point};
use crate::{flatness::{AggFlatness, DefaultFlatness, HfdFlatness}, polynomial_form_cubic, polynomial_form_quadratic};

pub trait Flatten {
    fn cubic<Cb: FnMut(&LineSegment)>(_curve: &CubicBezierSegment, _tolerance: f32, _cb: &mut Cb) {
        unimplemented!()
    }
    fn quadratic<Cb: FnMut(&LineSegment)>(_curve: &QuadraticBezierSegment, _tolerance: f32, _cb: &mut Cb) {
        unimplemented!()
    }
}

pub struct Fixed1;
impl Flatten for Fixed1 {
    fn cubic<Cb: FnMut(&LineSegment)>(curve: &CubicBezierSegment, _: f32, cb: &mut Cb) {
        cb(&curve.baseline());
    }
    fn quadratic<Cb: FnMut(&LineSegment)>(curve: &QuadraticBezierSegment, _: f32, cb: &mut Cb) {
        cb(&curve.baseline());
    }
}

pub struct Fixed16;
impl Flatten for Fixed16 {
    fn cubic<Cb: FnMut(&LineSegment)>(curve: &CubicBezierSegment, _: f32, cb: &mut Cb) {
        let poly = polynomial_form_cubic(curve);
        let step = 1.0 / 16.0;

        let mut t = step;
        let mut from = curve.from;
        for _ in 0..15 {
            let to = poly.sample_fma(t);
            cb(&LineSegment { from, to });
            from = to;
            t += step;
        }

        cb(&LineSegment { from, to: curve.to });
    }

    fn quadratic<Cb: FnMut(&LineSegment)>(curve: &QuadraticBezierSegment, _: f32, cb: &mut Cb) {
        let poly = polynomial_form_quadratic(curve);
        let step = 1.0 / 16.0;

        let mut t = step;
        let mut from = curve.from;
        for _ in 0..15 {
            let to = poly.sample(t);
            cb(&LineSegment { from, to });
            from = to;
            t += step;
        }

        cb(&LineSegment { from, to: curve.to });
    }
}

pub struct FwdDiff;
impl Flatten for FwdDiff {
    fn cubic<Cb: FnMut(&LineSegment)>(curve: &CubicBezierSegment, tolerance: f32, cb: &mut Cb) {
        crate::fwd_diff::flatten_cubic(curve, tolerance, cb);
    }
    fn quadratic<Cb: FnMut(&LineSegment)>(curve: &QuadraticBezierSegment, tolerance: f32, cb: &mut Cb) {
        crate::fwd_diff::flatten_quadratic(curve, tolerance, cb);
    }
}

pub struct HybridFwdDiff;
impl Flatten for HybridFwdDiff {
    fn cubic<Cb: FnMut(&LineSegment)>(curve: &CubicBezierSegment, tolerance: f32, cb: &mut Cb) {
        crate::hybrid_fwd_diff::flatten_cubic(curve, tolerance, cb);
    }
}

pub struct Wang;
impl Flatten for Wang {
    fn cubic<Cb: FnMut(&LineSegment)>(curve: &CubicBezierSegment, tolerance: f32, cb: &mut Cb) {
        crate::wang::flatten_cubic(curve, tolerance, cb);
    }
    fn quadratic<Cb: FnMut(&LineSegment)>(curve: &QuadraticBezierSegment, tolerance: f32, cb: &mut Cb) {
        crate::wang::flatten_quadratic(curve, tolerance, cb);
    }
}

pub struct WangSimd4;
impl Flatten for WangSimd4 {
    fn cubic<Cb: FnMut(&LineSegment)>(curve: &CubicBezierSegment, tolerance: f32, cb: &mut Cb) {
        unsafe {
            crate::wang::flatten_cubic_simd4(curve, tolerance, cb);
        }
    }
}

pub struct LevienQuads;
impl Flatten for LevienQuads {
    fn cubic<Cb: FnMut(&LineSegment)>(curve: &CubicBezierSegment, tolerance: f32, cb: &mut Cb) {
        crate::levien::flatten_cubic_19(curve, tolerance, cb);
    }
    fn quadratic<Cb: FnMut(&LineSegment)>(curve: &QuadraticBezierSegment, tolerance: f32, cb: &mut Cb) {
        crate::levien::flatten_quadratic(curve, tolerance, cb);
    }
}

pub struct Levien;
impl Flatten for Levien {
    fn cubic<Cb: FnMut(&LineSegment)>(curve: &CubicBezierSegment, tolerance: f32, cb: &mut Cb) {
        crate::levien::flatten_cubic_scalar2(curve, tolerance, cb);
    }
    fn quadratic<Cb: FnMut(&LineSegment)>(curve: &QuadraticBezierSegment, tolerance: f32, cb: &mut Cb) {
        crate::levien::flatten_quadratic(curve, tolerance, cb);
    }
}

pub struct LevienSimd;
impl Flatten for LevienSimd {
    fn cubic<Cb: FnMut(&LineSegment)>(curve: &CubicBezierSegment, tolerance: f32, cb: &mut Cb) {
        unsafe {
            crate::levien_simd::flatten_cubic_simd4(curve, tolerance, cb);
        }
    }
    fn quadratic<Cb: FnMut(&LineSegment)>(curve: &QuadraticBezierSegment, tolerance: f32, cb: &mut Cb) {
        unsafe {
            crate::levien_simd::flatten_quadratic(curve, tolerance, cb);
        }
    }
}

pub struct LevienSimd2;
impl Flatten for LevienSimd2 {
    fn cubic<Cb: FnMut(&LineSegment)>(curve: &CubicBezierSegment, tolerance: f32, cb: &mut Cb) {
        unsafe {
            crate::experiments::levien_experiments::flatten_cubic_simd4_v2(curve, tolerance, cb);
        }
    }
    fn quadratic<Cb: FnMut(&LineSegment)>(curve: &QuadraticBezierSegment, tolerance: f32, cb: &mut Cb) {
        unsafe {
            crate::levien_simd::flatten_quadratic(curve, tolerance, cb);
        }
    }
}

pub struct LevienSimd3;
impl Flatten for LevienSimd3 {
    fn cubic<Cb: FnMut(&LineSegment)>(curve: &CubicBezierSegment, tolerance: f32, cb: &mut Cb) {
        unsafe {
            crate::experiments::levien_experiments::flatten_cubic_simd4_interleaved(curve, tolerance, cb);
        }
    }
}

pub struct LevienSimdBuf;
impl Flatten for LevienSimdBuf {
    fn cubic<Cb: FnMut(&LineSegment)>(curve: &CubicBezierSegment, tolerance: f32, cb: &mut Cb) {
        unsafe {
            crate::experiments::levien_experiments::flatten_cubic_simd4_with_point_buffer(curve, tolerance, cb);
        }
    }
}

pub struct LevienLinear;
impl Flatten for LevienLinear {
    fn cubic<Cb: FnMut(&LineSegment)>(curve: &CubicBezierSegment, tolerance: f32, cb: &mut Cb) {
        unsafe {
            let dx = (curve.to.x - curve.from.x).abs();
            let dy = (curve.to.y - curve.from.y).abs();
            if dx + dy > 64.0 * tolerance {
                crate::levien_simd::flatten_cubic_simd4(curve, tolerance, cb);
            } else {
                crate::linear::flatten_cubic::<DefaultFlatness, _>(curve, tolerance, cb);
            }
        }
    }
    fn quadratic<Cb: FnMut(&LineSegment)>(curve: &QuadraticBezierSegment, tolerance: f32, cb: &mut Cb) {
        unsafe {
            let dx = (curve.to.x - curve.from.x).abs();
            let dy = (curve.to.y - curve.from.y).abs();
            if dx + dy > 64.0 * tolerance {
                crate::levien_simd::flatten_quadratic(curve, tolerance, cb);
            } else {
                crate::linear::flatten_quadratic(curve, tolerance, cb);
            }
        }
    }
}


pub struct Recursive;
impl Flatten for Recursive {
    fn cubic<Cb: FnMut(&LineSegment)>(curve: &CubicBezierSegment, tolerance: f32, cb: &mut Cb) {
        crate::recursive::flatten_cubic::<DefaultFlatness, _>(curve, tolerance, cb);
    }
    fn quadratic<Cb: FnMut(&LineSegment)>(curve: &QuadraticBezierSegment, tolerance: f32, cb: &mut Cb) {
        crate::recursive::flatten_quadratic(curve, tolerance, cb);
    }
}

pub struct RecursiveHfd;
impl Flatten for RecursiveHfd {
    fn cubic<Cb: FnMut(&LineSegment)>(curve: &CubicBezierSegment, tolerance: f32, cb: &mut Cb) {
        crate::recursive::flatten_cubic::<HfdFlatness, _>(curve, tolerance, cb);
    }
}

pub struct RecursiveAgg;
impl Flatten for RecursiveAgg {
    fn cubic<Cb: FnMut(&LineSegment)>(curve: &CubicBezierSegment, tolerance: f32, cb: &mut Cb) {
        crate::recursive::flatten_cubic::<AggFlatness, _>(curve, tolerance, cb);
    }
}

pub struct Linear;
impl Flatten for Linear {
    fn cubic<Cb: FnMut(&LineSegment)>(curve: &CubicBezierSegment, tolerance: f32, cb: &mut Cb) {
        crate::linear::flatten_cubic::<DefaultFlatness, _>(curve, tolerance, cb);
    }
    fn quadratic<Cb: FnMut(&LineSegment)>(curve: &QuadraticBezierSegment, tolerance: f32, cb: &mut Cb) {
        crate::linear::flatten_quadratic(curve, tolerance, cb);
    }
}

pub struct LinearHfd;
impl Flatten for LinearHfd {
    fn cubic<Cb: FnMut(&LineSegment)>(curve: &CubicBezierSegment, tolerance: f32, cb: &mut Cb) {
        crate::linear::flatten_cubic::<HfdFlatness, _>(curve, tolerance, cb);
    }
}

pub struct LinearAgg;
impl Flatten for LinearAgg {
    fn cubic<Cb: FnMut(&LineSegment)>(curve: &CubicBezierSegment, tolerance: f32, cb: &mut Cb) {
        crate::linear::flatten_cubic::<AggFlatness, _>(curve, tolerance, cb);
    }
}


pub struct Hain;
impl Flatten for Hain {
    fn cubic<Cb: FnMut(&LineSegment)>(curve: &CubicBezierSegment, tolerance: f32, cb: &mut Cb) {
        crate::hain::flatten_cubic(curve, tolerance, cb);
    }
}

pub struct Yzerman;
impl Flatten for Yzerman {
    fn cubic<Cb: FnMut(&LineSegment)>(curve: &CubicBezierSegment, tolerance: f32, cb: &mut Cb) {
        crate::yzerman::flatten_cubic(curve, tolerance, cb);
    }
}

pub struct YzermanSimd4;
impl Flatten for YzermanSimd4 {
    fn cubic<Cb: FnMut(&LineSegment)>(curve: &CubicBezierSegment, tolerance: f32, cb: &mut Cb) {
        unsafe {
            crate::yzerman::flatten_cubic_simd4(curve, tolerance, cb);
        }
    }
}

pub struct Kurbo;
impl Flatten for Kurbo {
    fn cubic<Cb: FnMut(&LineSegment)>(curve: &CubicBezierSegment, tolerance: f32, cb: &mut Cb) {
        let mut from = curve.from;
        kurbo::flatten(
            [
                kurbo::PathEl::MoveTo(kurbo::Point { x: curve.from.x as f64, y: curve.from.y as f64 }),
                kurbo::PathEl::CurveTo(
                    kurbo::Point { x: curve.ctrl1.x as f64, y: curve.ctrl1.y as f64 },
                    kurbo::Point { x: curve.ctrl2.x as f64, y: curve.ctrl2.y as f64 },
                    kurbo::Point { x: curve.to.x as f64, y: curve.to.y as f64 },
                ),
            ],
            tolerance as f64,
            &mut |elt| {
                match elt {
                    kurbo::PathEl::LineTo(p) => {
                        let to = point(p.x as f32, p.y as f32);
                        cb(&LineSegment { from, to });
                        from = to;
                    }
                    kurbo::PathEl::MoveTo(_) => {}
                    kurbo::PathEl::ClosePath => {}
                    _ => { unreachable!() }
                }
        });
    }
    fn quadratic<Cb: FnMut(&LineSegment)>(curve: &QuadraticBezierSegment, tolerance: f32, cb: &mut Cb) {
        let mut from = curve.from;
        kurbo::flatten(
            [
                kurbo::PathEl::MoveTo(kurbo::Point { x: curve.from.x as f64, y: curve.from.y as f64 }),
                kurbo::PathEl::QuadTo(
                    kurbo::Point { x: curve.ctrl.x as f64, y: curve.ctrl.y as f64 },
                    kurbo::Point { x: curve.to.x as f64, y: curve.to.y as f64 },
                ),
            ],
            tolerance as f64,
            &mut |elt| {
                match elt {
                    kurbo::PathEl::LineTo(p) => {
                        let to = point(p.x as f32, p.y as f32);
                        cb(&LineSegment { from, to });
                        from = to;
                    }
                    kurbo::PathEl::MoveTo(_) => {}
                    kurbo::PathEl::ClosePath => {}
                    _ => { unreachable!() }
                }
        });
    }
}
