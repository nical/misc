#![allow(exported_private_dependencies)]
#![allow(unused)]

use core::{point, Point};

use lyon::geom::{arrayvec::ArrayVec, CubicBezierSegment, LineSegment, QuadraticBezierSegment};


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
        self.init_quads = true;
        self.num_quads = num_quadratics_impl(curve, self.tolerance * 0.1);
        self.num_edges = 0;
        self.quad_idx = 0;
        self.flattened_quad_idx = 0;
        self.t0 = 0.0;
        self.i = 0;
        self.sum = 0.0;
        self.scaled_count_sum = 0.0;
    }

    pub fn set_quadratic(&mut self, curve: &QuadraticBezierSegment<f32>) {
        // TODO
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
        if self.ty == CurveType::Line {
            //println!("flatten line");
            self.ty = CurveType::None;
            output.push(self.curve.to);
        }
        return false;
    }

    pub fn flatten_cubic(&mut self, output: &mut Vec<Point>) -> bool {
        if self.ty != CurveType::Cubic {
            return false;
        }

        //println!("  flatten int: {:?} num_quads: {:?} i: {:?}, remaining cap: {:?}", self.init_quads, self.num_quads, self.i, output.capacity() - output.len());

        let quad_step = 1.0 / self.num_quads;

        let num_quadratics = self.num_quads as u32;

        loop {
            if self.init_quads {
                self.init_quads = false;
                self.quads.clear();
                self.sum = 0.0;
                while self.quad_idx < num_quadratics && self.quads.capacity() > self.quads.len() {
                    let t1 = self.t0 + quad_step;
                    let quad = self.curve.split_range(self.t0..t1).to_quadratic();
                    let params = FlatteningParams::new(&quad, self.sqrt_tolerance);
                    self.sum += params.scaled_count;
                    //println!(" + {quad:?} {t0} .. {t1} scaled count: {:?}", params.scaled_count);
                    self.quads.push((quad, params));
                    self.t0 = t1;
                    self.quad_idx += 1;
                }

                self.num_edges = ((0.5 * self.sum / self.sqrt_tolerance).ceil() as u32).max(1);
                self.i = 1;
            }

            let mut cap = output.capacity() - output.len();

            // Iterate through the quadratics, outputting the points of
            // subdivisions that fall within that quadratic.
            let step = self.sum / (self.num_edges as f32);
            self.scaled_count_sum = 0.0;
            //println!("------ {num_edges:?} edges step {step:?}");
            //if self.flattened_quad_idx as usize >= self.quads.len() {
            //    panic!("{:?}", self);
            //}
            for (quad, params) in &self.quads[self.flattened_quad_idx as usize ..] {
                let mut target = (self.i as f32) * step;
                //println!("  target {target} ({i})");
                let recip_scaled_count = params.scaled_count.recip();
                while target < self.scaled_count_sum + params.scaled_count {
                    let u = (target - self.scaled_count_sum) * recip_scaled_count;
                    let t = params.get_t(u);
                    let to = quad.sample(t);
                    output.push(to);
                    if self.i == self.num_edges {
                        // TODO: I think that this should break out
                        // of the outer for loop instead of the while loop.
                        break;
                    }
                    cap -= 1;
                    self.i += 1;
                    target = (self.i as f32) * step;

                    if cap == 0 {
                        //println!("    point buffer full quad idx {} / {}", self.flattened_quad_idx, self.quads.len());
                        return true;
                    }
                }
                self.flattened_quad_idx += 1;
                self.scaled_count_sum += params.scaled_count;
            }

            //if self.quads.is_empty() {
            //    println!("{:?}", self);
            //}
            output.push(self.quads.last().unwrap().0.to);

            if self.quad_idx == num_quadratics {
                //println!("    Done.");
                self.ty = CurveType::None;
                return false;
            }

            self.init_quads = true;
        }
    }

    pub fn flatten_quadratic(&mut self, output: &mut Vec<Point>) -> bool {
        if self.ty != CurveType::Quadratic {
            return false;
        }

        if self.init_quads {
            self.init_quads = false;
            self.quads.clear();
            let quad = QuadraticBezierSegment {
                from: self.curve.from,
                ctrl: self.curve.ctrl1,
                to: self.curve.to,
            };
            // TODO: 1.234567901 compensates for the squared 0.9 factor in self.sqrt_tolerance.
            let sqrt_tolerance = self.sqrt_tolerance * 1.234567901;
            let params = FlatteningParams::new(&quad, sqrt_tolerance);
            self.num_edges = ((0.5 * params.scaled_count / sqrt_tolerance).ceil() as u32).max(1);
            self.quads.push((quad, params));
            self.i = 1;
        }

        let (curve, params) = self.quads.first().unwrap();

        let step = 1.0 / (self.num_edges as f32);
        while self.i < self.num_edges {
            let u = (self.i as f32) * step;
            let t = params.get_t(u);
            let p = curve.sample(t);
            output.push(p);
            self.i += 1;
            if output.capacity() == output.len() {
                return true;
            }
        }

        output.push(curve.to);
        self.ty = CurveType::None;
        return false;
    }
}

#[derive(Debug)]
pub struct FlatteningParams {
    // Edge count * 2 * sqrt(tolerance).
    scaled_count: f32,
    integral_from: f32,
    integral_to: f32,
    inv_integral_from: f32,
    div_inv_integral_diff: f32,
}

impl FlatteningParams {
    pub fn new(curve: &QuadraticBezierSegment<f32>, sqrt_tolerance: f32) -> Self {
        // Map the quadratic bézier segment to y = x^2 parabola.
        let ddx = 2.0 * curve.ctrl.x - curve.from.x - curve.to.x;
        let ddy = 2.0 * curve.ctrl.y - curve.from.y - curve.to.y;
        let cross = (curve.to.x - curve.from.x) * ddy - (curve.to.y - curve.from.y) * ddx;
        let parabola_from =
            ((curve.ctrl.x - curve.from.x) * ddx + (curve.ctrl.y - curve.from.y) * ddy) / cross;
        let parabola_to =
            ((curve.to.x - curve.ctrl.x) * ddx + (curve.to.y - curve.ctrl.y) * ddy) / cross;

        let scale = (cross / ((ddx * ddx + ddy * ddy).sqrt() * (parabola_to - parabola_from))).abs();

        let integral_from = approx_parabola_integral(parabola_from);
        let integral_to = approx_parabola_integral(parabola_to);

        let inv_integral_from = approx_parabola_inv_integral(integral_from);
        let inv_integral_to = approx_parabola_inv_integral(integral_to);
        let div_inv_integral_diff = 1.0 / (inv_integral_to - inv_integral_from);

        // TODO: Kurbo does not not check for zero scale here. However
        // that would cause scaled_count to be NaN when dividing by sqrt_scale.
        let scaled_count = if scale != 0.0 && scale.is_finite() {
            let integral_diff = (integral_to - integral_from).abs();
            let sqrt_scale = scale.sqrt();
            if parabola_from.signum() == parabola_to.signum() {
                integral_diff * sqrt_scale
            } else {
                // Handle cusp case (segment contains curvature maximum)
                let xmin = sqrt_tolerance / sqrt_scale;
                sqrt_tolerance * integral_diff / approx_parabola_integral(xmin)
            }
        } else {
            0.0
        };

        debug_assert!(scaled_count.is_finite());

        FlatteningParams {
            scaled_count,
            integral_from,
            integral_to,
            inv_integral_from,
            div_inv_integral_diff,
        }
    }

    fn get_t(&self, norm_step: f32) -> f32 {
        let u = approx_parabola_inv_integral(self.integral_from + (self.integral_to - self.integral_from) * norm_step);
        let t = (u - self.inv_integral_from) * self.div_inv_integral_diff;

        t
    }
}


pub fn flatten_cubic(curve: &CubicBezierSegment<f32>, tolerance: f32, cb: &mut impl FnMut(&LineSegment<f32>)) {
    let quads_tolerance = tolerance * 0.1;
    let flatten_tolerance = tolerance * 0.9;
    let sqrt_flatten_tolerance = flatten_tolerance.sqrt();

    let num_quadratics = num_quadratics_impl(curve, quads_tolerance);
    //println!("{num_quadratics:?} quads");

    let mut quads: ArrayVec<(QuadraticBezierSegment<f32>, FlatteningParams), 16> = ArrayVec::new();

    let quad_step = 1.0 / num_quadratics;
    let num_quadratics = num_quadratics as u32;
    let mut quad_idx = 0;
    let mut t0 = 0.0;

    let mut from = curve.from;

    loop {
        let mut sum = 0.0;
        while quad_idx < num_quadratics && quads.capacity() > quads.len() {
            let t1 = t0 + quad_step;
            let quad = curve.split_range(t0..t1).to_quadratic();
            let params = FlatteningParams::new(&quad, sqrt_flatten_tolerance);
            sum += params.scaled_count;
            //println!(" + {quad:?} {t0} .. {t1} scaled count: {:?}", params.scaled_count);
            quads.push((quad, params));
            t0 = t1;
            quad_idx += 1;
        }

        let num_edges = ((0.5 * sum / sqrt_flatten_tolerance).ceil() as u32).max(1);

        // Iterate through the quadratics, outputting the points of
        // subdivisions that fall within that quadratic.
        let step = sum / (num_edges as f32);
        let mut i = 1;
        let mut scaled_count_sum = 0.0;
        //println!("------ {num_edges:?} edges step {step:?}");
        for (quad, params) in &quads {
            let mut target = (i as f32) * step;
            //println!("  target {target} ({i})");
            let recip_scaled_count = params.scaled_count.recip();
            while target < scaled_count_sum + params.scaled_count {
                let u = (target - scaled_count_sum) * recip_scaled_count;
                let t = params.get_t(u);
                let to = quad.sample(t);
                //println!("     u={u}, t={t}");
                cb(&LineSegment { from, to });
                from = to;
                if i == num_edges {
                    break;
                }
                i += 1;
                target = (i as f32) * step;
            }
            scaled_count_sum += params.scaled_count;
        }

        cb(&LineSegment { from, to: quads.last().unwrap().0.to });
        if quad_idx == num_quadratics {
            break;
        }

        quads.clear();
    }
}


/// Compute an approximation to integral (1 + 4x^2) ^ -0.25 dx used in the flattening code.
pub(crate) fn approx_parabola_integral(x: f32) -> f32 {
    let d = 0.67;
    let quarter = 0.25;
    x / (1.0 - d + (d*d*d*d + quarter * x * x).sqrt().sqrt())
}

/// Approximate the inverse of the function above.
pub(crate) fn approx_parabola_inv_integral(x: f32) -> f32 {
    let b = 0.39;
    let quarter = 0.25;
    x * (1.0 - b + (b * b + quarter * x * x).sqrt())
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
