use flatness::{AggFlatness, DefaultFlatness, HfdFlatness};
use lyon_path::{geom::{CubicBezierSegment, LineSegment, Point, QuadraticBezierSegment, Vector}, math::point};
use std::ops::Range;

pub mod linear;
pub mod levien;
//#[cfg(target_arch = "x86_64")]
pub mod levien_simd;
pub mod recursive;
pub mod wang;
pub mod fwd_diff;
pub mod hybrid_fwd_diff;
pub mod hain;
pub mod simd4;
pub mod testing;
pub mod flatness;
#[cfg(test)]
pub mod show;
#[cfg(test)]
pub mod edge_count;

// Using mul_add causes a large perf regression on x86_64.
// By default the regression is huge for wang and even
// with `-C target-feature=+fma` passed (for example using
// the `RUSTFLAGS` environment variable), the regression is
// quite large.
// TODO: Am I doing something wrong?

#[cfg(not(target_feature="fma"))]
#[inline(always)]
pub fn fma(val: f32, mul: f32, add: f32) -> f32 {
    val * mul + add
}

#[cfg(target_feature="fma")]
#[inline(always)]
pub fn fma(val: f32, mul: f32, add: f32) -> f32 {
    val.mul_add(mul, add)
}

pub struct CubicBezierPolynomial {
    pub a0: Vector<f32>,
    pub a1: Vector<f32>,
    pub a2: Vector<f32>,
    pub a3: Vector<f32>,
}

#[inline(always)]
pub fn polynomial_form_cubic(curve: &CubicBezierSegment<f32>) -> CubicBezierPolynomial {
    CubicBezierPolynomial {
        a0: curve.from.to_vector(),
        a1: (curve.ctrl1 - curve.from) * 3.0,
        a2: curve.from * 3.0 - curve.ctrl1 * 6.0 + curve.ctrl2.to_vector() * 3.0,
        a3: curve.to - curve.from + (curve.ctrl1 - curve.ctrl2) * 3.0
    }
}

impl CubicBezierPolynomial {
    #[inline(always)]
    pub fn sample(&self, t: f32) -> Point<f32> {
        // Horner's method.
        let mut v = self.a0;
        let mut t2 = t;
        v += self.a1 * t2;
        t2 *= t;
        v += self.a2 * t2;
        t2 *= t;
        v += self.a3 * t2;

        v.to_point()
    }

    #[inline(always)]
    pub fn sample_fma(&self, t: f32) -> Point<f32> {
        let mut vx = self.a0.x;
        let mut vy = self.a0.y;
        let mut t2 = t;
        vx = fma(self.a1.x, t2, vx);
        vy = fma(self.a1.y, t2, vy);
        t2 *= t;
        vx = fma(self.a2.x, t2, vx);
        vy = fma(self.a2.y, t2, vy);
        t2 *= t;
        vx = fma(self.a3.x, t2, vx);
        vy = fma(self.a3.y, t2, vy);

        Point::new(vx, vy)
    }
}

#[inline]
pub fn polynomial_form_quadratic(curve: &QuadraticBezierSegment<f32>) -> QuadraticBezierPolynomial {
    let from = curve.from.to_vector();
    let ctrl = curve.ctrl.to_vector();
    let to = curve.to.to_vector();
    QuadraticBezierPolynomial {
        a0: from,
        a1: (ctrl - from) * 2.0,
        a2: from + to - ctrl * 2.0
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct QuadraticBezierPolynomial {
    pub a0: Vector<f32>,
    pub a1: Vector<f32>,
    pub a2: Vector<f32>,
}

impl QuadraticBezierPolynomial {
    #[inline(always)]
    pub fn sample(&self, t: f32) -> Point<f32> {
        // Horner's method.
        let mut v = self.a0;
        let mut t2 = t;
        v += self.a1 * t2;
        t2 *= t;
        v += self.a2 * t2;

        v.to_point()
    }
}


/// Just approximate the curve with a single line segment.
pub fn flatten_noop<F>(curve: &CubicBezierSegment<f32>, _tolerance: f32, callback: &mut F)
where
    F:  FnMut(&LineSegment<f32>, Range<f32>)
{
    callback(&LineSegment { from: curve.from, to: curve.to }, 0.0..1.0);
}

/// Returns whether the curve can be approximated with a single point, given
/// a tolerance threshold.
#[inline]
pub(crate) fn cubic_is_a_point(&curve: &CubicBezierSegment<f32>, tolerance: f32) -> bool {
    let tolerance_squared = tolerance * tolerance;
    // Use <= so that tolerance can be zero.
    (curve.from - curve.to).square_length() <= tolerance_squared
        && (curve.from - curve.ctrl1).square_length() <= tolerance_squared
        && (curve.to - curve.ctrl2).square_length() <= tolerance_squared
}

pub trait Flatten {
    fn cubic<Cb: FnMut(&LineSegment<f32>)>(_curve: &CubicBezierSegment<f32>, _tolerance: f32, _cb: &mut Cb) {
        unimplemented!()
    }
    fn quadratic<Cb: FnMut(&LineSegment<f32>)>(_curve: &QuadraticBezierSegment<f32>, _tolerance: f32, _cb: &mut Cb) {
        unimplemented!()
    }
}

pub struct Fixed1;
impl Flatten for Fixed1 {
    fn cubic<Cb: FnMut(&LineSegment<f32>)>(curve: &CubicBezierSegment<f32>, _: f32, cb: &mut Cb) {
        cb(&curve.baseline());
    }
    fn quadratic<Cb: FnMut(&LineSegment<f32>)>(curve: &QuadraticBezierSegment<f32>, _: f32, cb: &mut Cb) {
        cb(&curve.baseline());
    }
}

pub struct Fixed16;
impl Flatten for Fixed16 {
    fn cubic<Cb: FnMut(&LineSegment<f32>)>(curve: &CubicBezierSegment<f32>, _: f32, cb: &mut Cb) {
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

    fn quadratic<Cb: FnMut(&LineSegment<f32>)>(curve: &QuadraticBezierSegment<f32>, _: f32, cb: &mut Cb) {
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
    fn cubic<Cb: FnMut(&LineSegment<f32>)>(curve: &CubicBezierSegment<f32>, tolerance: f32, cb: &mut Cb) {
        crate::fwd_diff::flatten_cubic(curve, tolerance, cb);
    }
    fn quadratic<Cb: FnMut(&LineSegment<f32>)>(curve: &QuadraticBezierSegment<f32>, tolerance: f32, cb: &mut Cb) {
        crate::fwd_diff::flatten_quadratic(curve, tolerance, cb);
    }
}

pub struct HybridFwdDiff;
impl Flatten for HybridFwdDiff {
    fn cubic<Cb: FnMut(&LineSegment<f32>)>(curve: &CubicBezierSegment<f32>, tolerance: f32, cb: &mut Cb) {
        crate::hybrid_fwd_diff::flatten_cubic(curve, tolerance, cb);
    }
}

pub struct Wang;
impl Flatten for Wang {
    fn cubic<Cb: FnMut(&LineSegment<f32>)>(curve: &CubicBezierSegment<f32>, tolerance: f32, cb: &mut Cb) {
        crate::wang::flatten_cubic(curve, tolerance, cb);
    }
    fn quadratic<Cb: FnMut(&LineSegment<f32>)>(curve: &QuadraticBezierSegment<f32>, tolerance: f32, cb: &mut Cb) {
        crate::wang::flatten_quadratic(curve, tolerance, cb);
    }
}

pub struct WangSimd4;
impl Flatten for WangSimd4 {
    fn cubic<Cb: FnMut(&LineSegment<f32>)>(curve: &CubicBezierSegment<f32>, tolerance: f32, cb: &mut Cb) {
        unsafe {
            crate::wang::flatten_cubic_simd4(curve, tolerance, cb);
        }
    }
}

pub struct WangSimd42;
impl Flatten for WangSimd42 {
    fn cubic<Cb: FnMut(&LineSegment<f32>)>(curve: &CubicBezierSegment<f32>, tolerance: f32, cb: &mut Cb) {
        unsafe {
            crate::wang::flatten_cubic_simd4_with_point_buffer(curve, tolerance, cb);
        }
    }
}

pub struct LevienQuads;
impl Flatten for LevienQuads {
    fn cubic<Cb: FnMut(&LineSegment<f32>)>(curve: &CubicBezierSegment<f32>, tolerance: f32, cb: &mut Cb) {
        crate::levien::flatten_cubic_19(curve, tolerance, cb);
    }
    fn quadratic<Cb: FnMut(&LineSegment<f32>)>(curve: &QuadraticBezierSegment<f32>, tolerance: f32, cb: &mut Cb) {
        crate::levien::flatten_quadratic(curve, tolerance, cb);
    }
}

pub struct Levien;
impl Flatten for Levien {
    fn cubic<Cb: FnMut(&LineSegment<f32>)>(curve: &CubicBezierSegment<f32>, tolerance: f32, cb: &mut Cb) {
        crate::levien::flatten_cubic_scalar2(curve, tolerance, cb);
    }
    fn quadratic<Cb: FnMut(&LineSegment<f32>)>(curve: &QuadraticBezierSegment<f32>, tolerance: f32, cb: &mut Cb) {
        crate::levien::flatten_quadratic(curve, tolerance, cb);
    }
}

pub struct LevienSimd;
impl Flatten for LevienSimd {
    fn cubic<Cb: FnMut(&LineSegment<f32>)>(curve: &CubicBezierSegment<f32>, tolerance: f32, cb: &mut Cb) {
        unsafe {
            crate::levien_simd::flatten_cubic_simd4(curve, tolerance, cb);
        }
    }
    fn quadratic<Cb: FnMut(&LineSegment<f32>)>(curve: &QuadraticBezierSegment<f32>, tolerance: f32, cb: &mut Cb) {
        unsafe {
            crate::levien_simd::flatten_quadratic(curve, tolerance, cb);
        }
    }
}

pub struct LevienSimd2;
impl Flatten for LevienSimd2 {
    fn cubic<Cb: FnMut(&LineSegment<f32>)>(curve: &CubicBezierSegment<f32>, tolerance: f32, cb: &mut Cb) {
        unsafe {
            crate::levien_simd::flatten_cubic_simd4_with_point_buffer(curve, tolerance, cb);
        }
    }
    fn quadratic<Cb: FnMut(&LineSegment<f32>)>(curve: &QuadraticBezierSegment<f32>, tolerance: f32, cb: &mut Cb) {
        unsafe {
            crate::levien_simd::flatten_quadratic(curve, tolerance, cb);
        }
    }
}

pub struct Levien37;
impl Flatten for Levien37 {
    fn cubic<Cb: FnMut(&LineSegment<f32>)>(curve: &CubicBezierSegment<f32>, tolerance: f32, cb: &mut Cb) {
        crate::levien::flatten_cubic_37(curve, tolerance, cb);
    }
    fn quadratic<Cb: FnMut(&LineSegment<f32>)>(curve: &QuadraticBezierSegment<f32>, tolerance: f32, cb: &mut Cb) {
        crate::levien::flatten_quadratic(curve, tolerance, cb);
    }
}

pub struct Levien55;
impl Flatten for Levien55 {
    fn cubic<Cb: FnMut(&LineSegment<f32>)>(curve: &CubicBezierSegment<f32>, tolerance: f32, cb: &mut Cb) {
        crate::levien::flatten_cubic_55(curve, tolerance, cb);
    }
    fn quadratic<Cb: FnMut(&LineSegment<f32>)>(curve: &QuadraticBezierSegment<f32>, tolerance: f32, cb: &mut Cb) {
        crate::levien::flatten_quadratic(curve, tolerance, cb);
    }
}

pub struct Recursive;
impl Flatten for Recursive {
    fn cubic<Cb: FnMut(&LineSegment<f32>)>(curve: &CubicBezierSegment<f32>, tolerance: f32, cb: &mut Cb) {
        crate::recursive::flatten_cubic::<DefaultFlatness, _>(curve, tolerance, cb);
    }
    fn quadratic<Cb: FnMut(&LineSegment<f32>)>(curve: &QuadraticBezierSegment<f32>, tolerance: f32, cb: &mut Cb) {
        crate::recursive::flatten_quadratic(curve, tolerance, cb);
    }
}

pub struct RecursiveHfd;
impl Flatten for RecursiveHfd {
    fn cubic<Cb: FnMut(&LineSegment<f32>)>(curve: &CubicBezierSegment<f32>, tolerance: f32, cb: &mut Cb) {
        crate::recursive::flatten_cubic::<HfdFlatness, _>(curve, tolerance, cb);
    }
}

pub struct RecursiveAgg;
impl Flatten for RecursiveAgg {
    fn cubic<Cb: FnMut(&LineSegment<f32>)>(curve: &CubicBezierSegment<f32>, tolerance: f32, cb: &mut Cb) {
        crate::recursive::flatten_cubic::<AggFlatness, _>(curve, tolerance, cb);
    }
}

pub struct Linear;
impl Flatten for Linear {
    fn cubic<Cb: FnMut(&LineSegment<f32>)>(curve: &CubicBezierSegment<f32>, tolerance: f32, cb: &mut Cb) {
        crate::linear::flatten_cubic::<DefaultFlatness, _>(curve, tolerance, cb);
    }
    fn quadratic<Cb: FnMut(&LineSegment<f32>)>(curve: &QuadraticBezierSegment<f32>, tolerance: f32, cb: &mut Cb) {
        crate::linear::flatten_quadratic(curve, tolerance, cb);
    }
}

pub struct LinearHfd;
impl Flatten for LinearHfd {
    fn cubic<Cb: FnMut(&LineSegment<f32>)>(curve: &CubicBezierSegment<f32>, tolerance: f32, cb: &mut Cb) {
        crate::linear::flatten_cubic::<HfdFlatness, _>(curve, tolerance, cb);
    }
}

pub struct LinearAgg;
impl Flatten for LinearAgg {
    fn cubic<Cb: FnMut(&LineSegment<f32>)>(curve: &CubicBezierSegment<f32>, tolerance: f32, cb: &mut Cb) {
        crate::linear::flatten_cubic::<AggFlatness, _>(curve, tolerance, cb);
    }
}


pub struct Hain;
impl Flatten for Hain {
    fn cubic<Cb: FnMut(&LineSegment<f32>)>(curve: &CubicBezierSegment<f32>, tolerance: f32, cb: &mut Cb) {
        crate::hain::flatten_cubic(curve, tolerance, cb);
    }
}

pub struct Kurbo;
impl Flatten for Kurbo {
    fn cubic<Cb: FnMut(&LineSegment<f32>)>(curve: &CubicBezierSegment<f32>, tolerance: f32, cb: &mut Cb) {
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
    fn quadratic<Cb: FnMut(&LineSegment<f32>)>(curve: &QuadraticBezierSegment<f32>, tolerance: f32, cb: &mut Cb) {
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
