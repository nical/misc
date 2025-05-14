#![allow(exported_private_dependencies)]
#![allow(unused)]

use core::units::Vector;
use core::{point, Point};
use lyon::geom::{arrayvec::ArrayVec, CubicBezierSegment, LineSegment, QuadraticBezierSegment};

#[cfg(target_arch = "x86")]
use std::arch::x86 as arch;
#[cfg(target_arch = "x86_64")]
use std::f32;
use std::ops::Range;

use crate::{simd4::*};

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
    pub a0: Vector,
    pub a1: Vector,
    pub a2: Vector,
    pub a3: Vector,
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

#[derive(Debug)]
pub struct FlatteningParams {
    // Edge count * 2 * sqrt(tolerance).
    pub scaled_count: f32,
    pub integral_from: f32,
    pub integral_to: f32,
    pub inv_integral_from: f32,
    pub div_inv_integral_diff: f32,
}

#[inline(always)]
unsafe fn approx_parabola_integral(x: f32x4) -> f32x4 {
    let d_pow4 = splat(0.67 * 0.67 * 0.67 * 0.67);
    let sqr = sqrt(sqrt(add(d_pow4, mul(splat(0.25), mul(x, x)))));
    div(x, add(splat(1.0 - 0.67), sqr))
}

#[inline(always)]
unsafe fn approx_parabola_inv_integral(x: f32x4) -> f32x4 {
    mul(x, add(splat(1.0 - 0.39), sqrt(add(splat(0.39 * 0.39), mul(splat(0.25), mul(x, x))))))
}



#[derive(PartialEq, Debug)]
enum CurveType {
    None,
    Line,
    Quadratic,
    Cubic,
}

pub struct Flattener {
    rem: CubicBezierSegment<f32>,
    sq_tolerance: f32,
    t0: f32,
    split: f32,
    ty: CurveType,
}

impl Flattener {
    pub fn new(tolerance: f32) -> Self {
        Flattener {
            rem: CubicBezierSegment {
                from: point(0.0, 0.0),
                ctrl1: point(0.0, 0.0),
                ctrl2: point(0.0, 0.0),
                to: point(0.0, 0.0),
            },
            sq_tolerance: tolerance * tolerance,
            t0: 0.0,
            split: 0.5,
            ty: CurveType::None,
        }
    }

    #[inline]
    pub fn is_done(&self) -> bool {
        self.ty == CurveType::None
    }

    pub fn set_cubic(&mut self, curve: &CubicBezierSegment<f32>) {
        self.rem = *curve;
        self.split = 0.5;
        self.ty = CurveType::Cubic;
    }

    pub fn set_quadratic(&mut self, curve: &QuadraticBezierSegment<f32>) {
        // TODO
        self.rem = curve.to_cubic();
        self.ty = CurveType::Cubic;
        self.split = 0.5;
    }

    pub fn set_line(&mut self, to: Point) {
        self.rem.to = to;
        self.ty = CurveType::Line;
    }

    pub fn flatten(&mut self, output: &mut Vec<Point>) -> bool {
        match self.ty {
            CurveType::Cubic => self.flatten_cubic(output),
            CurveType::Quadratic => { unimplemented!() },
            CurveType::Line => self.flatten_line(output),
            CurveType::None => false,
        }
    }

    pub fn flatten_line(&mut self, output: &mut Vec<Point>) -> bool {
        if self.ty == CurveType::Line {
            //println!("flatten line");
            self.ty = CurveType::None;
            output.push(self.rem.to);
        }
        return false;
    }

    pub fn flatten_cubic(&mut self, output: &mut Vec<Point>) -> bool {
        if self.ty != CurveType::Cubic {
            return false;
        }
        //println!("flatten cubic");

        // TODO: is_linear assumes from and to aren't at the same position,
        // (it returns true in this case regardless of the control points).
        // If it's the case we should force a split.

        let mut cap = output.capacity() - output.len();
        loop {
            if cap == 0 {
                return true;
            }

            if is_linear(&self.rem, self.sq_tolerance) {
                output.push(self.rem.to);
                self.ty = CurveType::None;
                return false;
            }

            loop {
                let sub = self.rem.before_split(self.split);
                if is_linear(&sub, self.sq_tolerance) {
                    let t1 = self.t0 + (1.0 - self.t0) * self.split;
                    output.push(sub.to);
                    cap -= 1;
                    self.t0 = t1;
                    self.rem = self.rem.after_split(self.split);
                    let next_split = self.split * 2.0;
                    if next_split < 1.0 {
                        self.split = next_split;
                    }
                    break;
                }
                self.split *= 0.5;
            }
        }
    }
}

pub fn is_linear(curve: &CubicBezierSegment<f32>, sq_tolerance: f32) -> bool {
    let baseline = curve.to - curve.from;
    let v1 = curve.ctrl1 - curve.from;
    let v2 = curve.ctrl2 - curve.from;
    let c1 = baseline.cross(v1);
    let c2 = baseline.cross(v2);
    let sqlen = baseline.square_length();
    let d1 = c1 * c1;
    let d2 = c2 * c2;

    let f2 = 0.5625; // (4/3)^2
    let threshold = sq_tolerance * sqlen;

    d1 * f2 <= threshold && d2 * f2 <= threshold
}

#[derive(Debug)]
pub struct FlattenerLevien {
    curve: CubicBezierSegment<f32>,
    quads: ArrayVec<(QuadraticBezierSegment<f32>, FlatteningParams), 16>,
    tolerance: f32,
    sqrt_tolerance: f32,
    ty: CurveType,
    init_quads: bool,
    num_quads: f32,
    num_edges: u32,
    quad_idx: u32,
    flattened_quad_idx: u32,
    t0: f32,
    i: u32,
    sum: f32,
    scaled_count_sum:f32,
}

impl FlattenerLevien {
    pub fn new(tolerance: f32) -> Self {
        FlattenerLevien {
            curve: CubicBezierSegment {
                from: point(0.0, 0.0),
                ctrl1: point(0.0, 0.0),
                ctrl2: point(0.0, 0.0),
                to: point(0.0, 0.0),
            },
            quads: ArrayVec::new(),
            tolerance,
            sqrt_tolerance: (tolerance * 0.9).sqrt(),
            ty: CurveType::None,
            init_quads: true,
            num_quads: 1.0,
            num_edges: 0,
            quad_idx: 0,
            flattened_quad_idx: 0,
            t0: 0.0,
            i: 0,
            sum: 0.0,
            scaled_count_sum: 0.0,
        }
    }

    #[inline]
    pub fn is_done(&self) -> bool {
        self.ty == CurveType::None
    }

    pub fn set_cubic(&mut self, curve: &CubicBezierSegment<f32>) {
        //println!("set {curve:?}");
        self.curve = *curve;
        self.ty = CurveType::Cubic;
    }

    pub fn set_quadratic(&mut self, curve: &QuadraticBezierSegment<f32>) {
        self.curve.from = curve.from;
        self.curve.ctrl1 = curve.ctrl;
        self.curve.to = curve.to;
        self.ty = CurveType::Quadratic;
        self.init_quads = true;
    }

    pub fn set_line(&mut self, to: Point) {
        self.curve.to = to;
        self.ty = CurveType::Line;
    }

    pub fn flatten(&mut self, output: &mut Vec<Point>) -> bool {
        match self.ty {
            CurveType::Cubic => self.flatten_cubic(output),
            CurveType::Quadratic => self.flatten_quadratic(output),
            CurveType::Line => self.flatten_line(output),
            CurveType::None => false,
        }
    }

    pub fn flatten_line(&mut self, output: &mut Vec<Point>) -> bool {
        debug_assert_eq!(self.ty, CurveType::Line);
        self.ty = CurveType::None;
        output.push(self.curve.to);
        return false;
    }

    pub fn flatten_cubic(&mut self, output: &mut Vec<Point>) -> bool {
        debug_assert_eq!(self.ty, CurveType::Cubic);
        let dx = (self.curve.to.x - self.curve.from.x).abs();
        let dy = (self.curve.to.y - self.curve.from.y).abs();
        if dx + dy < 64.0 * self.tolerance {
            return self.flatten_cubic_linear(output);
        }

        unsafe {
            flatten_cubic_simd4(&self.curve, self.tolerance, &mut|s| {
                output.push(s.to);
            });
        }

        return false;
    }

    pub fn flatten_quadratic(&mut self, output: &mut Vec<Point>) -> bool {
        debug_assert_eq!(self.ty, CurveType::Quadratic);

        let quad = QuadraticBezierSegment {
            from: self.curve.from,
            ctrl: self.curve.ctrl1,
            to: self.curve.to,
        };

        let dx = (quad.to.x - quad.from.x).abs();
        let dy = (quad.to.y - quad.from.y).abs();
        if dx + dy < 64.0 * self.tolerance {
            flatten_quadratic_linear(&quad, self.tolerance, output);
            return false;
        }

        unsafe {
            flatten_quadratic_simd4(&quad, self.tolerance, &mut|s| {
                output.push(s.to);
            });
        }

        return false;
    }

    pub fn flatten_cubic_linear(&mut self, output: &mut Vec<Point>) -> bool {
        let mut rem = self.curve;
        let mut split = 0.5;
        let sq_tolerance = self.tolerance * self.tolerance;

        let mut cap = output.capacity() - output.len();
        loop {
            if cap == 0 {
                return true;
            }

            if is_linear(&rem, sq_tolerance) {
                output.push(rem.to);
                self.ty = CurveType::None;
                return false;
            }

            loop {
                let sub = rem.before_split(split);
                if is_linear(&sub, sq_tolerance) {
                    let t1 = self.t0 + (1.0 - self.t0) * split;
                    output.push(sub.to);
                    cap -= 1;
                    self.t0 = t1;
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
}

pub fn flatten_quadratic_linear(curve: &QuadraticBezierSegment<f32>, tolerance: f32, output: &mut Vec<Point>) {
    let mut rem = *curve;
    let mut from = rem.from;

    let mut split = 0.5;
    loop {
        if split >= 0.25 && quadratic_is_flat(&rem, tolerance) {
            output.push(rem.to);
            return;
        }

        loop {
            let sub = rem.before_split(split);
            if quadratic_is_flat(&sub, tolerance) {
                output.push(sub.to);
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

#[inline]
pub fn quadratic_is_flat(curve: &QuadraticBezierSegment<f32>, tolerance: f32) -> bool {
    let baseline = curve.to - curve.from;
    let v = curve.ctrl - curve.from;
    let c = baseline.cross(v);

    let baseline_len2 = baseline.square_length();
    let threshold = baseline_len2 * tolerance * tolerance;

    c * c * 0.25 <= threshold
}

#[inline(never)]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx"))]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
unsafe fn flattening_params_simd4(
    curve: &CubicBezierSegment<f32>,
    s_num_quads: f32,
    s_first_quad: f32,
    s_quad_step: f32,
    sqrt_tolerance: f32,
    output: &mut ArrayVec<(FlatteningParams, QuadraticBezierPolynomial), 16>
) -> (Point, f32) {
    let quad_step = splat(s_quad_step);
    let t0 = mul(add(splat(s_first_quad), vec4(0.0, 1.0, 2.0, 3.0)), quad_step);
    let t1 = add(t0, quad_step);

    // polynomial_form:
    let s_poly = polynomial_form_cubic(curve);
    let a0_x = splat(s_poly.a0.x);
    let a0_y = splat(s_poly.a0.y);
    let a1_x = splat(s_poly.a1.x);
    let a1_y = splat(s_poly.a1.y);
    let a2_x = splat(s_poly.a2.x);
    let a2_y = splat(s_poly.a2.y);
    let a3_x = splat(s_poly.a3.x);
    let a3_y = splat(s_poly.a3.y);

    let (from_x, from_y) = sample_cubic_horner_simd4(a0_x, a0_y, a1_x, a1_y, a2_x, a2_y, a3_x, a3_y, t0);
    // TODO: 3 out of 4 values are already in from_
    let (to_x, to_y) = sample_cubic_horner_simd4(a0_x, a0_y, a1_x, a1_y, a2_x, a2_y, a3_x, a3_y, t1);

    // Compute the control points of the sub-curves.

    // CubicBezierSegment::split_range
    let qx0 = splat(curve.ctrl1.x - curve.from.x);
    let qy0 = splat(curve.ctrl1.y - curve.from.y);
    let qx1 = splat(curve.ctrl2.x - curve.ctrl1.x);
    let qy1 = splat(curve.ctrl2.y - curve.ctrl1.y);
    let qx2 = splat(curve.to.x - curve.ctrl2.x);
    let qy2 = splat(curve.to.y - curve.ctrl2.y);

    let one = splat(1.0);
    let two = splat(2.0);

    // QuadraticBezierSegment::sample(t0)
    let t0_2 = mul(t0, t0);
    let one_t0 = sub(one, t0);
    let one_t0_2 = mul(one_t0, one_t0);
    let two_t0_one_t0 = mul(two, mul(one_t0, t0));
    let qx = add(mul_add(qx0, one_t0_2, mul(qx1, two_t0_one_t0)), mul(qx2, t0_2));
    let qy = add(mul_add(qy0, one_t0_2, mul(qy1, two_t0_one_t0)), mul(qy2, t0_2));

    let ctrl1_x = mul_add(qx, quad_step, from_x);
    let ctrl1_y = mul_add(qy, quad_step, from_y);

    // QuadraticBezierSegment::sample(t1)
    let t1_2 = mul(t1, t1);
    let one_t1 = sub(one, t1);
    let one_t1_2 = mul(one_t1, one_t1);
    let two_t1_one_t1 = mul(two, mul(one_t1, t1));
    let qx = add(mul_add(qx0, one_t1_2, mul(qx1, two_t1_one_t1)), mul(qx2, t1_2));
    let qy = add(mul_add(qy0, one_t1_2, mul(qy1, two_t1_one_t1)), mul(qy2, t1_2));

    let ctrl2_x = sub(to_x, mul(qx, quad_step));
    let ctrl2_y = sub(to_y, mul(qy, quad_step));

    // Approximate the sub-curves with quadratics
    let three = splat(3.0);
    let half = splat(0.5);
    let c1x = mul(mul_sub(ctrl1_x, three, from_x), half);
    let c1y = mul(mul_sub(ctrl1_y, three, from_y), half);
    let c2x = mul(mul_sub(ctrl2_x, three, to_x), half);
    let c2y = mul(mul_sub(ctrl2_y, three, to_y), half);

    let ctrl_x = mul(add(c1x, c2x), half);
    let ctrl_y = mul(add(c1y, c2y), half);

    // Now that we have our four quadratics, compute the flattening parameters
    let ddx = sub(mul_sub(two, ctrl_x, from_x), to_x);
    let ddy = sub(mul_sub(two, ctrl_y, from_y), to_y);
    //println!("ddx {ddx:?} ddy {ddy:?}");
    let cross = mul_sub(sub(to_x, from_x), ddy, mul(sub(to_y, from_y), ddx));
    //println!("cross {cross:?}");
    let rcp_cross = recip(cross);
    let parabola_from = mul(mul_add(sub(ctrl_x, from_x), ddx, mul(sub(ctrl_y, from_y), ddy)), rcp_cross);
    let parabola_to = mul(mul_add(sub(to_x, ctrl_x), ddx, mul(sub(to_y, ctrl_y), ddy)), rcp_cross);
    let parabola_diff = sub(parabola_to, parabola_from);
    let scale = abs(div(cross, mul(sqrt(mul_add(ddx, ddx, mul(ddy, ddy))), parabola_diff)));

    let integral_from = approx_parabola_integral(parabola_from);
    let integral_to = approx_parabola_integral(parabola_to);
    //println!("parabola_to {parabola_to:?} parabola_from {parabola_from:?}");
    //println!("intergal_to {integral_to:?} integral_from {integral_from:?}");

    let inv_integral_from = approx_parabola_inv_integral(integral_from);
    let inv_integral_to = approx_parabola_inv_integral(integral_to);

    let div_inv_integral_diff = recip(sub(inv_integral_to, inv_integral_from));

    let sqrt_tolerance = splat(sqrt_tolerance);
    let integral_diff = abs(sub(integral_to, integral_from));
    let sqrt_scale = sqrt(scale);

    let mut scaled_count = mul(integral_diff, sqrt_scale);

    let is_cusp = neq(signum(parabola_from), signum(parabola_to));
    // Handle a cusp case (segment contains curvature maximum)
    // Assuming the cusp case is fairly rare and because it is expensive
    // we add a branch to test wether any of the lanes need to take it
    // and skip it otherwise.
    if any(is_cusp) {
        let xmin = div(sqrt_tolerance, sqrt_scale);
        let scaled_count2 = mul(sqrt_tolerance, div(integral_diff, approx_parabola_integral(xmin)));
        scaled_count = select(is_cusp, scaled_count2, scaled_count);
    }

    // Handle another kind of cusp.
    scaled_count = select_or_zero(is_finite(scale), scaled_count);

    // Convert the quadratic curves into polynomial form.
    let a1_x = mul(sub(ctrl_x, from_x), two);
    let a1_y = mul(sub(ctrl_y, from_y), two);
    let a2_x = sub(add(from_x, to_x), mul(ctrl_x, two));
    let a2_y = sub(add(from_y, to_y), mul(ctrl_y, two));

    // Go back to scalar land to produce the output.

    let scaled_count = unpack(scaled_count);
    let integral_from = unpack(integral_from);
    let integral_to = unpack(integral_to);
    let inv_integral_from = unpack(inv_integral_from);
    let div_inv_integral_diff = unpack(div_inv_integral_diff);

    let a0_x = unpack(from_x);
    let a0_y = unpack(from_y);
    let a1_x = unpack(a1_x);
    let a1_y = unpack(a1_y);
    let a2_x = unpack(a2_x);
    let a2_y = unpack(a2_y);
    let to_x = unpack(to_x);
    let to_y = unpack(to_y);

    let mut sum = scaled_count.0;

    assert!(output.capacity() >= output.len() + 4);

    let count = s_num_quads as u32;
    output.push((
        FlatteningParams {
            scaled_count: scaled_count.0,
            integral_from: integral_from.0,
            integral_to: integral_to.0,
            inv_integral_from: inv_integral_from.0,
            div_inv_integral_diff: div_inv_integral_diff.0,
        },
        QuadraticBezierPolynomial {
            a0: Vector::new(a0_x.0, a0_y.0),
            a1: Vector::new(a1_x.0, a1_y.0),
            a2: Vector::new(a2_x.0, a2_y.0),
        },
    ));

    if count < 2 {
        return (point(to_x.0, to_y.0), sum);
    }
    sum += scaled_count.1;
    output.push((
        FlatteningParams {
            scaled_count: scaled_count.1,
            integral_from: integral_from.1,
            integral_to: integral_to.1,
            inv_integral_from: inv_integral_from.1,
            div_inv_integral_diff: div_inv_integral_diff.1,
        },
        QuadraticBezierPolynomial {
            a0: Vector::new(a0_x.1, a0_y.1),
            a1: Vector::new(a1_x.1, a1_y.1),
            a2: Vector::new(a2_x.1, a2_y.1),
        },
    ));

    if count < 3 {
        return (point(to_x.1, to_y.1), sum);
    }

    sum += scaled_count.2;
    output.push((
        FlatteningParams {
            scaled_count: scaled_count.2,
            integral_from: integral_from.2,
            integral_to: integral_to.2,
            inv_integral_from: inv_integral_from.2,
            div_inv_integral_diff: div_inv_integral_diff.2,
        },
        QuadraticBezierPolynomial {
            a0: Vector::new(a0_x.2, a0_y.2),
            a1: Vector::new(a1_x.2, a1_y.2),
            a2: Vector::new(a2_x.2, a2_y.2),
        },
    ));

    if count < 4 {
        return (point(to_x.2, to_y.2), sum);
    }

    sum += scaled_count.3;
    output.push((
        FlatteningParams {
            scaled_count: scaled_count.3,
            integral_from: integral_from.3,
            integral_to: integral_to.3,
            inv_integral_from: inv_integral_from.3,
            div_inv_integral_diff: div_inv_integral_diff.3,
        },
        QuadraticBezierPolynomial {
            a0: Vector::new(a0_x.3, a0_y.3),
            a1: Vector::new(a1_x.3, a1_y.3),
            a2: Vector::new(a2_x.3, a2_y.3),
        },
    ));

    (point(to_x.3, to_y.3), sum)
}

#[inline(never)]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx"))]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
pub unsafe fn flatten_cubic_simd4(curve: &CubicBezierSegment<f32>, tolerance: f32, cb: &mut dyn FnMut(&LineSegment<f32>)) {
    let quads_tolerance = tolerance * 0.1;
    let flatten_tolerance = tolerance * 0.9;
    let sqrt_flatten_tolerance = flatten_tolerance.sqrt();
    let inv_sqrt_flatten_tolerance = fast_recip(sqrt_flatten_tolerance);

    let num_quadratics = num_quadratics_impl(curve, quads_tolerance);
    //println!("{num_quadratics:?} quads");

    //counters_inc(num_quadratics.max(0.0) as usize);

    let mut quads: ArrayVec<(FlatteningParams, QuadraticBezierPolynomial), 16> = ArrayVec::new();

    let quad_step = fast_recip(num_quadratics);
    let num_quadratics = num_quadratics as u32;
    let mut quad_idx = 0;
    let mut from = curve.from;

    loop {
        let mut sum = 0.0;
        let mut quads_last_to = point(0.0, 0.0);
        while quad_idx < num_quadratics && quads.capacity() > quads.len() {
            let (to, s) = flattening_params_simd4(
                curve,
                (num_quadratics - quad_idx).min(4) as f32,
                quad_idx as f32,
                quad_step,
                sqrt_flatten_tolerance,
                &mut quads
            );
            sum += s;
            quads_last_to = to;

            //println!(" + {quad:?} {t0} .. {t1} scaled count: {:?}", params.scaled_count);
            quad_idx += 4;
        }

        let num_edges = ((0.5 * sum * inv_sqrt_flatten_tolerance).ceil() as u32).max(1);

        // Iterate through the quadratics, outputting the points of
        // subdivisions that fall within that quadratic.
        let step = sum / (num_edges as f32);
        let mut i = 1;
        let mut scaled_count_sum = 0.0;
        let v_step = splat(step);
        //println!("------ {num_edges:?} edges step {step:?}");
        for (params, quad) in &quads {
            let n = u32::min(num_edges, ((scaled_count_sum + params.scaled_count) / step).ceil() as u32);

            if i < n {
                let recip_scaled_count = splat(fast_recip(params.scaled_count));
                let quad_a0x = splat(quad.a0.x);
                let quad_a0y = splat(quad.a0.y);
                let quad_a1x = splat(quad.a1.x);
                let quad_a1y = splat(quad.a1.y);
                let quad_a2x = splat(quad.a2.x);
                let quad_a2y = splat(quad.a2.y);
                let integral_from = splat(params.integral_from);
                let integral_diff = splat(params.integral_to - params.integral_from);
                let inv_integral_from = splat(params.inv_integral_from);
                let div_inv_integral_diff = splat(params.div_inv_integral_diff);
                let v_scaled_count_sum = splat(scaled_count_sum);

                let mut v_i = add(splat(i as f32), vec4(0.0, 1.0, 2.0, 3.0));

                while i < n {
                    let targets = mul(v_i, v_step);
                    let u = mul(sub(targets, v_scaled_count_sum), recip_scaled_count);

                    let u = approx_parabola_inv_integral(mul_add(integral_diff, u, integral_from));
                    let t = mul(sub(u, inv_integral_from), div_inv_integral_diff);

                    let px = sample_quadratic_horner_simd4(quad_a0x, quad_a1x, quad_a2x, t);
                    let py = sample_quadratic_horner_simd4(quad_a0y, quad_a1y, quad_a2y, t);

                    let px: [f32; 4] = std::mem::transmute(px);
                    let py: [f32; 4] = std::mem::transmute(py);

                    for (x, y) in px.iter().zip(py.iter()) {
                        let to = point(*x, *y);
                        cb(&LineSegment { from, to });
                        from = to;
                        i += 1;
                        if i >= n {
                            break;
                        }
                    }
                    v_i = add(v_i, splat(4.0));
                }
            }
            scaled_count_sum += params.scaled_count;
        }

        cb(&LineSegment { from, to: quads_last_to });
        if quad_idx >= num_quadratics {
            break;
        }

        quads.clear();
    }
}

#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx"))]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
pub unsafe fn flatten_quadratic_simd4(curve: &QuadraticBezierSegment<f32>, tolerance: f32, cb: &mut impl FnMut(&LineSegment<f32>)) {

    let ddx = 2.0 * curve.ctrl.x - curve.from.x - curve.to.x;
    let ddy = 2.0 * curve.ctrl.y - curve.from.y - curve.to.y;
    let cross = (curve.to.x - curve.from.x) * ddy - (curve.to.y - curve.from.y) * ddx;
    let inv_cross = fast_recip(cross);

    // Attempting to vectorize these two lines did not improve performance.
    let parabola_from = ((curve.ctrl.x - curve.from.x) * ddx + (curve.ctrl.y - curve.from.y) * ddy) * inv_cross;
    let parabola_to   = ((curve.to.x   - curve.ctrl.x) * ddx + (curve.to.y   - curve.ctrl.y) * ddy) * inv_cross;

    // Note, scale can be NaN, for example with straight lines. When it happens the NaN will
    // propagate to other parameters. We catch it all by setting the iteration count to zero
    // and leave the rest as garbage.
    //let scale = cross.abs() / ((ddx * ddx + ddy * ddy).sqrt() * (parabola_to - parabola_from).abs());
    let scale = cross.abs() * fast_recip_sqrt(ddx * ddx + ddy * ddy) * fast_recip((parabola_to - parabola_from).abs());

    let integral = approx_parabola_integral(vec4(parabola_from, parabola_to, 0.0, 0.0));
    let inv_integral = approx_parabola_inv_integral(integral);
    let (integral_to, integral_from, _, _) = unpack(integral);
    let (inv_integral_to, inv_integral_from, _, _) = unpack(inv_integral);

    let integral_diff = integral_to - integral_from;
    let div_inv_integral_diff = fast_recip(inv_integral_to - inv_integral_from);

    let mut count = (0.5 * integral_diff.abs() * (scale * fast_recip(tolerance)).sqrt()).ceil();
    // If count is NaN the curve can be approximated by a single straight line or a point.
    if !count.is_finite() {
        count = 0.0;
    }

    let poly = polynomial_form_quadratic(curve);
    let quad_a0x = splat(poly.a0.x);
    let quad_a0y = splat(poly.a0.y);
    let quad_a1x = splat(poly.a1.x);
    let quad_a1y = splat(poly.a1.y);
    let quad_a2x = splat(poly.a2.x);
    let quad_a2y = splat(poly.a2.y);

    let integral_from = splat(integral_from);
    let integral_step = splat(integral_diff * fast_recip(count));
    let mut iteration = vec4(1.0, 2.0, 3.0, 4.0);
    let mut i = 1;
    let count = count as usize;
    let mut from = curve.from;
    while i < count {
        let integ = add(integral_from, mul(integral_step, iteration));
        let u = approx_parabola_inv_integral(integ);

        let t = mul(sub(u, splat(inv_integral_from)), splat(div_inv_integral_diff));

        let x = sample_quadratic_horner_simd4(quad_a0x, quad_a1x, quad_a2x, t);
        let y = sample_quadratic_horner_simd4(quad_a0y, quad_a1y, quad_a2y, t);

        let x: [f32; 4] = std::mem::transmute(x);
        let y: [f32; 4] = std::mem::transmute(y);

        for (x, y) in x.iter().zip(y.iter()).take((count - i).min(4)) {
            let p = point(*x, *y);
            cb(&LineSegment {from, to: p });
            from = p;
        }

        iteration = add(iteration, splat(4.0));
        i += 4;
    }

    cb(&LineSegment {from, to: curve.to });
}


fn num_quadratics_impl(curve: &CubicBezierSegment<f32>, tolerance: f32) -> f32 {
    debug_assert!(tolerance > 0.0);

    let x = curve.from.x - 3.0 * curve.ctrl1.x + 3.0 * curve.ctrl2.x - curve.to.x;
    let y = curve.from.y - 3.0 * curve.ctrl1.y + 3.0 * curve.ctrl2.y - curve.to.y;

    let err = x * x + y * y;

    (err / (432.0 * tolerance * tolerance))
        .powf(1.0 / 6.0)
        .ceil()
        .max(1.0)
}

#[test]
fn quads_oob_01() {
    //set CubicBezierSegment { from: (597.0059, 321.65604), ctrl1: (597.23145, 318.45694), ctrl2: (595.8107, 313.43985), to: (592.74384, 306.60468) }
    //  flatten int: true num_quads: 1.0 i: 0, remaining cap: 511
    //    Done.

    let curve = CubicBezierSegment {
        from: point(597.00592, 321.656036),
        ctrl1: point(597.231445, 318.45694,),
        ctrl2: point(595.81073, 313.43985),
        to: point(592.743835, 306.604675),
    };

    let mut flattener = FlattenerLevien::new(0.25);

    flattener.set_cubic(&curve);
    let mut points = Vec::with_capacity(512);
    for _ in 0..(512 - 2) {
        points.push(point(0.0, 0.0));
    }
    while flattener.flatten(&mut points) {
        points.clear();
    }
}
