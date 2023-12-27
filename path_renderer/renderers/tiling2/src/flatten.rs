use core::{units::point, Point};

use lyon::geom::{CubicBezierSegment, LineSegment, QuadraticBezierSegment};


#[derive(PartialEq)]
enum CurveType {
    None,
    Line,
    //Quadratic,
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


pub fn flatten_cubic<F>(curve: &CubicBezierSegment<f32>, tolerance: f32, callback: &mut F)
where
    F: FnMut(&LineSegment<f32>),
{
    let mut rem = *curve;
    let mut from = rem.from;
    let mut t0 = 0.0;

    let mut split = 0.5;
    loop {
        if rem.is_linear(tolerance) {
            callback(&LineSegment { from, to: rem.to });
            return;
        }

        loop {
            let sub = rem.before_split(split);
            if sub.is_linear(tolerance) {
                let t1 = t0 + (1.0 - t0) * split;
                callback(&LineSegment { from, to: sub.to });
                from = sub.to;
                t0 = t1;
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

pub fn flatten_quad<F>(curve: &QuadraticBezierSegment<f32>, tolerance: f32, callback: &mut F)
where
    F: FnMut(&LineSegment<f32>),
{
    let mut rem = *curve;
    let mut from = rem.from;
    let mut t0 = 0.0;

    let mut split = 0.5;
    loop {
        if rem.is_linear(tolerance) {
            callback(&LineSegment { from, to: rem.to });
            return;
        }

        loop {
            let sub = rem.before_split(split);
            if sub.is_linear(tolerance) {
                let t1 = t0 + (1.0 - t0) * split;
                callback(&LineSegment { from, to: sub.to });
                from = sub.to;
                t0 = t1;
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

