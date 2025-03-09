use lyon_path::geom::{QuadraticBezierSegment, CubicBezierSegment, LineSegment, Vector, Point};
use std::ops::Range;

pub mod linear;
pub mod levien;
pub mod levien_simd;
pub mod recursive;
pub mod sedeberg;
pub mod fwd_diff;
pub mod hybrid_fwd_diff;
pub mod parabola_approx;
#[cfg(feature = "testing")]
pub mod testing;
#[cfg(test)]
pub mod edge_count;

// Using mul_add causes a large perf regression on x86_64.
// By default the regression is huge for sedeberg and even
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

#[inline]
pub fn polynomial_form_cubic(curve: &CubicBezierSegment<f32>) -> CubicBezierPolynomial {
    CubicBezierPolynomial {
        a0: curve.from.to_vector(),
        a1: (curve.ctrl1 - curve.from) * 3.0,
        a2: curve.from * 3.0 - curve.ctrl1 * 6.0 + curve.ctrl2.to_vector() * 3.0,
        a3: curve.to - curve.from + (curve.ctrl1 - curve.ctrl2) * 3.0
    }
}

impl CubicBezierPolynomial {
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

    pub fn sample_fma(&self, t: f32) -> Point<f32> {
        let mut vx = self.a1.x;
        let mut vy = self.a1.y;
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

/// Just approcimate the curve with a single line segment.
pub fn flatten_noop<F>(curve: &CubicBezierSegment<f32>, _tolerance: f32, callback: &mut F)
where
    F:  FnMut(&LineSegment<f32>, Range<f32>)
{
    callback(&LineSegment { from: curve.from, to: curve.to }, 0.0..1.0);
}

/// Returns whether the curve can be approximated with a single point, given
/// a tolerance threshold.
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

pub struct Sedeberg;
impl Flatten for Sedeberg {
    fn cubic<Cb: FnMut(&LineSegment<f32>)>(curve: &CubicBezierSegment<f32>, tolerance: f32, cb: &mut Cb) {
        crate::sedeberg::flatten_cubic(curve, tolerance, cb);
    }
    fn quadratic<Cb: FnMut(&LineSegment<f32>)>(curve: &QuadraticBezierSegment<f32>, tolerance: f32, cb: &mut Cb) {
        crate::sedeberg::flatten_quadratic(curve, tolerance, cb);
    }
}

pub struct Levien;
impl Flatten for Levien {
    fn cubic<Cb: FnMut(&LineSegment<f32>)>(curve: &CubicBezierSegment<f32>, tolerance: f32, cb: &mut Cb) {
        crate::levien::flatten_cubic(curve, tolerance, cb);
    }
    fn quadratic<Cb: FnMut(&LineSegment<f32>)>(curve: &QuadraticBezierSegment<f32>, tolerance: f32, cb: &mut Cb) {
        crate::levien::flatten_quadratic(curve, tolerance, cb);
    }
}

pub struct LevienSimd;
impl Flatten for LevienSimd {
    fn quadratic<Cb: FnMut(&LineSegment<f32>)>(curve: &QuadraticBezierSegment<f32>, tolerance: f32, cb: &mut Cb) {
        crate::levien_simd::flatten_quadratic(curve, tolerance, cb);
    }
}

pub struct Recursive;
impl Flatten for Recursive {
    fn cubic<Cb: FnMut(&LineSegment<f32>)>(curve: &CubicBezierSegment<f32>, tolerance: f32, cb: &mut Cb) {
        crate::recursive::flatten_cubic(curve, tolerance, cb);
    }
    fn quadratic<Cb: FnMut(&LineSegment<f32>)>(curve: &QuadraticBezierSegment<f32>, tolerance: f32, cb: &mut Cb) {
        crate::recursive::flatten_quadratic(curve, tolerance, cb);
    }
}

pub struct Linear;
impl Flatten for Linear {
    fn cubic<Cb: FnMut(&LineSegment<f32>)>(curve: &CubicBezierSegment<f32>, tolerance: f32, cb: &mut Cb) {
        crate::linear::flatten_cubic(curve, tolerance, cb);
    }
    fn quadratic<Cb: FnMut(&LineSegment<f32>)>(curve: &QuadraticBezierSegment<f32>, tolerance: f32, cb: &mut Cb) {
        crate::linear::flatten_quadratic(curve, tolerance, cb);
    }
}

pub struct ParabolaApprox;
impl Flatten for ParabolaApprox {
    fn cubic<Cb: FnMut(&LineSegment<f32>)>(curve: &CubicBezierSegment<f32>, tolerance: f32, cb: &mut Cb) {
        crate::parabola_approx::flatten_cubic(curve, tolerance, cb);
    }
}
