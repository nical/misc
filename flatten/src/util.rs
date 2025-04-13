use std::ops::Range;

pub type CubicBezierSegment = lyon_path::geom::CubicBezierSegment<f32>;
pub type QuadraticBezierSegment = lyon_path::geom::QuadraticBezierSegment<f32>;
pub type LineSegment = lyon_path::geom::LineSegment<f32>;
pub type Point = lyon_path::geom::Point<f32>;
pub type Vector = lyon_path::geom::Vector<f32>;
pub use lyon_path::math::point;

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

#[inline(always)]
#[cfg(target_arch = "x86_64")]
pub fn fast_recip(a: f32) -> f32 {
    use std::arch::x86_64::*;
    unsafe { _mm_cvtss_f32(_mm_rcp_ss(_mm_set_ss(a))) }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn fast_recip(a: f32) -> f32 {
    1.0 / a
}

#[inline(always)]
#[cfg(target_arch = "x86_64")]
pub fn fast_ceil(x: f32) -> f32 {
    // For some reason this is not inlined.
    x.ceil()

    // This is slower.
    // It actually produces a fair amount of instructions.
    //(x + 0.5) as i32 as f32

    // This is slower and so is using _mm_ceil_ps.
    //use std::arch::x86_64::*;
    //unsafe {
    //    _mm_cvtss_f32(_mm_ceil_ss(simd4::splat(0.0), _mm_set_ss(x)))
    //}
}

#[inline(always)]
#[cfg(not(target_arch = "x86_64"))]
pub fn fast_ceil(x: f32) -> f32 {
    x.ceil()
}

#[inline(always)]
#[cfg(target_arch = "x86_64")]
pub fn fast_recip_sqrt(x: f32) -> f32 {
    use std::arch::x86_64::*;
    unsafe { _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(x))) }
}

#[inline(always)]
#[cfg(not(target_arch = "x86_64"))]
pub fn fast_recip_sqrt(x: f32) -> f32 {
    1.0 / x.sqrt()
}

// This is pretty expensive but I haven't found a faster alternative
// yet.
#[inline(always)]
#[cfg(not(feature = "approx"))]
pub fn fast_cubic_root(x: f32) -> f32 {
    x.powf(1.0/3.0)
}

// From https://github.com/666rayen999/x-math/blob/main/src/lib.rs
// So far this version has not shown any performance improvement
// over `x.powf(1.0/3.0)` and I have not evaluated how it affects
// precision.
#[inline]
#[cfg(feature = "approx")]
pub fn fast_cubic_root(x: f32) -> f32 {
    const A: f32 = f32::from_bits(0x3fe04c03);
    const B: f32 = f32::from_bits(0x3f0266d9);
    const C: f32 = f32::from_bits(0xbfa01f36);
    let xi = x.to_bits();
    let sx = (xi & 0x80000000) | 0x3f800000;
    let ax = xi & 0x7fffffff;
    let i = 0x548c2b4b - (ax / 3);
    let y = f32::from_bits(i);
    let c = x * y * y * y;
    let y = y * (A + c * (B * c + C));
    let d = x * y * y;
    let c = d - d * d * y;
    let c = c * f32::from_bits(0x3eaaaaab) + d;
    f32::from_bits(sx) * c
}


pub struct CubicBezierPolynomial {
    pub a0: Vector,
    pub a1: Vector,
    pub a2: Vector,
    pub a3: Vector,
}

#[inline(always)]
pub fn polynomial_form_cubic(curve: &CubicBezierSegment) -> CubicBezierPolynomial {
    CubicBezierPolynomial {
        a0: curve.from.to_vector(),
        a1: (curve.ctrl1 - curve.from) * 3.0,
        a2: curve.from * 3.0 - curve.ctrl1 * 6.0 + curve.ctrl2.to_vector() * 3.0,
        a3: curve.to - curve.from + (curve.ctrl1 - curve.ctrl2) * 3.0
    }
}

impl CubicBezierPolynomial {
    #[inline(always)]
    pub fn sample(&self, t: f32) -> Point {
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
    pub fn sample_fma(&self, t: f32) -> Point {
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
pub fn polynomial_form_quadratic(curve: &QuadraticBezierSegment) -> QuadraticBezierPolynomial {
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
    pub a0: Vector,
    pub a1: Vector,
    pub a2: Vector,
}

impl QuadraticBezierPolynomial {
    #[inline(always)]
    pub fn sample(&self, t: f32) -> Point {
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
pub fn flatten_noop<F>(curve: &CubicBezierSegment, _tolerance: f32, callback: &mut F)
where
    F:  FnMut(&LineSegment, Range<f32>)
{
    callback(&LineSegment { from: curve.from, to: curve.to }, 0.0..1.0);
}

/// Returns whether the curve can be approximated with a single point, given
/// a tolerance threshold.
#[inline]
pub(crate) fn cubic_is_a_point(&curve: &CubicBezierSegment, tolerance: f32) -> bool {
    let tolerance_squared = tolerance * tolerance;
    // Use <= so that tolerance can be zero.
    (curve.from - curve.to).square_length() <= tolerance_squared
        && (curve.from - curve.ctrl1).square_length() <= tolerance_squared
        && (curve.to - curve.ctrl2).square_length() <= tolerance_squared
}


#[repr(align(16))]
pub struct AlignedBuf<const N: usize>(std::mem::MaybeUninit<[f32; N]>);

impl<const N: usize> AlignedBuf<N> {
    #[inline(always)]
    pub fn new() -> Self {
        AlignedBuf(std::mem::MaybeUninit::uninit())
    }

    #[inline(always)]
    pub unsafe fn get(&self, offset: usize) -> f32 {
        *self.0.assume_init_ref().as_ptr().add(offset)
    }

    #[inline(always)]
    pub unsafe fn ptr(&mut self, offset: usize) -> *mut f32 {
        self.0.assume_init_mut().as_mut_ptr().add(offset)
    }
}

/// Aame as CubicBezierSegment::split_range but #[inline]
/// which makes a big difference.
#[inline(always)]
pub fn split_range(curve: &CubicBezierSegment, t_range: std::ops::Range<f32>) -> CubicBezierSegment {
    let (t0, t1) = (t_range.start, t_range.end);
    let from = curve.sample(t0);
    let to = curve.sample(t1);

    let d = QuadraticBezierSegment {
        from: (curve.ctrl1 - curve.from).to_point(),
        ctrl: (curve.ctrl2 - curve.ctrl1).to_point(),
        to: (curve.to - curve.ctrl2).to_point(),
    };

    let dt = t1 - t0;
    let ctrl1 = from + d.sample(t0).to_vector() * dt;
    let ctrl2 = to - d.sample(t1).to_vector() * dt;

    CubicBezierSegment {
        from,
        ctrl1,
        ctrl2,
        to,
    }
}
