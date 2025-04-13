use crate::{CubicBezierSegment, LineSegment, Vector};


// The algorithm is described in detail in the 1995 patent # 5367617 "System and
// method of hybrid forward differencing to render Bezier splines" to be found
// on the Microsoft legal dept. web site (LCAWEB).  Additional references are:
//     Lien, Shantz and Vaughan Pratt, "Adaptive Forward Differencing for
//     Rendering Curves and Surfaces", Computer Graphics, July 1987
//     Chang and Shantz, "Rendering Trimmed NURBS with Adaptive Forward
//         Differencing", Computer Graphics, August 1988
//     Foley and Van Dam, "Fundamentals of Interactive Computer Graphics"
//
// The basic idea is to replace the Bernstein basis (underlying Bezier curves)
// with the Hybrid Forward Differencing (HFD) basis which is more efficient
// for flattening.  Each one of the 3 actions - Step, Halve and Double (step
// size) this basis affords very efficient formulas for computing coefficients
// for the new interval.
//
// The coefficients of the HFD basis are defined in terms of the Bezier
// coefficients as follows:
//
//          e0 = p0, e1 = p3 - p0, e2 = 6(p1 - 2p2 + p3), e3 = 6(p0 - 2p1 + p2),
//
// but formulas may be easier to understand by going through the power basis
// representation:  f(t) = a*t + b*t + c * t^2 + d * t^3.
//
//  The conversion is then:
//                              e0 = a
//                              e1 = f(1) - f(0) = b + c + d
//                              e2 = f"(1) = 2c + 6d
//                              e3 = f"(0) = 2c
//
// This is inverted to:
//                              a = e0
//                              c = e3 / 2
//                              d = (e2 - 2c) / 6 = (e2 - e3) / 6
//                              b = e1 - c - d = e1 - e2 / 6 - e3 / 3
//
// a, b, c, d for the new (halved, doubled or forwarded) interval are derived
// and then converted to e0, e1, e2, e3 using these relationships.
//
// This code is an adaptation of a Rust port by Jeff Muizelaar of WPF's C++
// implementation.
struct HfdFlattener {
    current: Vector,
    v1: Vector,
    v2: Vector,
    v3: Vector,
    steps: u32,
    step_size: f32,
    t: f32,
    tol: f32,
    quarter_tol: f32,
}

impl HfdFlattener {
    #[inline]
    fn new(curve: &CubicBezierSegment, tolerance: f32) -> Self {
        let mut flattener = HfdFlattener {
            current: curve.from.to_vector(),
            v1: curve.to - curve.from,
            v2: (curve.ctrl1 - curve.ctrl2 * 2.0 + curve.to.to_vector()) * 6.0,
            v3: (curve.from - curve.ctrl1 * 2.0 + curve.ctrl2.to_vector()) * 6.0,
            tol: 6.0 * tolerance,
            quarter_tol:  6.0 / 4.0 * tolerance,
            steps: 1,
            step_size: 1.0,
            t: 0.0,
        };

        while approx_norm(&flattener.v2) > flattener.tol || approx_norm(&flattener.v3) > flattener.tol {
            flattener.halve_step();
        }

        flattener
    }

    fn step(&mut self) {
        self.current += self.v1;
        let pt = self.v2;
        self.v1 += self.v2;
        self.v2 = self.v2 + self.v2 - self.v3;
        self.v3 = pt;

        self.t += self.step_size;

        self.steps -= 1;
    }

    fn halve_step(&mut self) {
        self.v2 = (self.v2 + self.v3) * 0.125;
        self.v1 = (self.v1 - self.v2) * 0.5;
        self.v3 *= 0.25;
        self.steps *= 2;
        self.step_size *= 0.5;
    }

    fn maybe_double_step(&mut self) -> bool {
        let tmp = self.v2 * 2.0 - self.v3;
        let doubled = approx_norm(&self.v3) <= self.quarter_tol && approx_norm(&tmp) <= self.quarter_tol;
        if doubled {
            self.v1 = self.v1 * 2.0 + self.v2;
            self.v3 *= 4.0;
            self.v2 = tmp * 4.0;
            self.steps /= 2;
            self.step_size *= 2.0;
        }

        doubled
    }

    fn adjust_step(&mut self) {
        if approx_norm(&self.v2) > self.tol && self.step_size > 1e-3 {
            // Halving the step once is provably sufficient (see Notes above)...
            self.halve_step();
        } else if (self.steps & 1) == 0 {
            // ...but the step can possibly be more than doubled, hence the while loop.
            while self.maybe_double_step() {}
        }
    }
}

fn approx_norm(v: &Vector) -> f32 {
    v.x.abs().max(v.y.abs())
}


/// Flatten using the hybrid forward difference algortihm.
pub fn flatten_cubic<F>(curve: &CubicBezierSegment, tolerance: f32, callback: &mut F)
where
    F:  FnMut(&LineSegment)
{
    let mut flattener = HfdFlattener::new(curve, tolerance);

    let mut prev = curve.from;

    while flattener.steps > 1 {
        flattener.step();

        let to = flattener.current.to_point();
        callback(&LineSegment { from: prev, to, });
        prev = to;

        flattener.adjust_step();
    }

    callback(&LineSegment { from: prev, to: curve.to, });
}
