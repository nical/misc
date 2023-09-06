// TODO: this doesn't belong in the core crate.

use lyon::{
    path::{
        builder::PathBuilder,
        NO_ATTRIBUTES, PathEvent, EndpointId, Attributes,
    },
    geom::{QuadraticBezierSegment, CubicBezierSegment, Line, utils::{tangent, cubic_polynomial_roots}, LineSegment, arrayvec::ArrayVec}
};

use crate::{units::{Point, Vector}, path::Path};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum LineCap {
    Butt,
    Square,
    Round,
    RoundInverted,
    Triangle,
    TriangleInverted,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum LineJoin {
    /// A sharp corner is to be used to join path segments.
    Miter,
    /// Same as a miter join, but if the miter limit is exceeded,
    /// the miter is clipped at a miter length equal to the miter limit value
    /// multiplied by the stroke width.
    MiterClip,
    /// A round corner is to be used to join path segments.
    Round,
    /// A round-ish corner using a single quadratic bézier segment.
    Quad,
    /// A beveled corner is to be used to join path segments.
    /// The bevel shape is a triangle that fills the area between the two stroked
    /// segments.
    Bevel,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct StrokeOptions {
    pub offsets: (f32, f32),
    pub miter_limit: f32,
    pub tolerance: f32,
    pub line_join: LineJoin,
    pub start_cap: LineCap,
    pub end_cap: LineCap,
    pub add_empty_caps: bool,
}

impl StrokeOptions {
    #[inline]
    pub fn line_with(w: f32) -> Self {
        Self::default().with_line_width(w)
    }

    #[inline]
    pub fn tolerance(tolerance: f32) -> Self {
        StrokeOptions {
            tolerance,
            offsets: (0.5, -0.5),
            miter_limit: lyon::tessellation::StrokeOptions::DEFAULT_MITER_LIMIT,
            line_join: LineJoin::Miter,
            start_cap: LineCap::Square,
            end_cap: LineCap::Square,
            add_empty_caps: false,
        }
    }

    #[inline]
    pub fn with_line_width(mut self, width: f32) -> Self {
        self.offsets = (-width * 0.5, width * 0.5);
        self
    }

    #[inline]
    pub fn with_line_join(mut self, join: LineJoin) -> Self {
        self.line_join = join;
        self
    }

    #[inline]
    pub fn with_line_cap(mut self, cap: LineCap) -> Self {
        self.start_cap = cap;
        self.end_cap = cap;
        self
    }

    #[inline]
    pub fn with_start_cap(mut self, cap: LineCap) -> Self {
        self.start_cap = cap;
        self
    }

    #[inline]
    pub fn with_end_cap(mut self, cap: LineCap) -> Self {
        self.end_cap = cap;
        self
    }

    #[inline]
    pub fn with_miter_limit(mut self, limit: f32) -> Self {
        self.miter_limit = limit;
        self
    }

    #[inline]
    pub fn with_tolerance(mut self, tolerance: f32) -> Self {
        self.tolerance = tolerance;
        self
    }
}

impl Default for StrokeOptions {
    fn default() -> Self {
        StrokeOptions::tolerance(0.25)
    }
}

pub struct Stroker {
    // TODO: recycle inner builder
}

impl Stroker {
    pub fn new() -> Stroker {
        Stroker {}
    }

    pub fn stroke_to_fill(&mut self, path: impl Iterator<Item = PathEvent>, options: &StrokeOptions) -> Path {
        let mut builder = Path::builder();

        {
            let mut stroker = StrokeToFillBuilder::new(&mut builder, &options);
            for event in path {
                match event {
                    PathEvent::Begin { at } => { stroker.begin(at); }
                    PathEvent::End { close, .. } => { stroker.end(close); }
                    PathEvent::Line { to, .. } => { stroker.line_to(to); }
                    PathEvent::Quadratic { ctrl, to, .. } => { stroker.quadratic_bezier_to(ctrl, to); }
                    PathEvent::Cubic { ctrl1, ctrl2, to, .. } => { stroker.cubic_bezier_to(ctrl1, ctrl2, to); }
                }
            }
        }

        builder.build()
    }
}

impl<'a> PathBuilder for OffsetBuilder<'a> {
    fn num_attributes(&self) -> usize { 0 }

    fn begin(&mut self, at: Point, _: Attributes) -> EndpointId {
        self.begin(at);
        EndpointId::INVALID
    }

    fn end(&mut self, close: bool) {
        self.end(close);
    }

    fn line_to(&mut self, to: Point, _: Attributes) -> EndpointId {
        self.line_to(to);
        EndpointId::INVALID
    }

    fn quadratic_bezier_to(&mut self, ctrl: Point, to: Point, _: Attributes) -> EndpointId {
        self.quadratic_bezier_to(ctrl, to);
        EndpointId::INVALID
    }

    fn cubic_bezier_to(&mut self, ctrl1: Point, ctrl2: Point, to: Point, _: Attributes) -> EndpointId {
        self.cubic_bezier_to(ctrl1, ctrl2, to);
        EndpointId::INVALID
    }
}

impl<'a> PathBuilder for StrokeToFillBuilder<'a> {
    fn num_attributes(&self) -> usize { 0 }

    fn begin(&mut self, at: Point, _: Attributes) -> EndpointId {
        self.begin(at);
        EndpointId::INVALID
    }

    fn end(&mut self, close: bool) {
        self.end(close);
    }

    fn line_to(&mut self, to: Point, _: Attributes) -> EndpointId {
        self.line_to(to);
        EndpointId::INVALID
    }

    fn quadratic_bezier_to(&mut self, ctrl: Point, to: Point, _: Attributes) -> EndpointId {
        self.quadratic_bezier_to(ctrl, to);
        EndpointId::INVALID
    }

    fn cubic_bezier_to(&mut self, ctrl1: Point, ctrl2: Point, to: Point, _: Attributes) -> EndpointId {
        self.cubic_bezier_to(ctrl1, ctrl2, to);
        EndpointId::INVALID
    }
}

fn normal(v: Vector) -> Vector {
    Vector::new(-v.y, v.x)
}

pub mod offset {
    use lyon::geom::{Scalar, QuadraticBezierSegment, Vector, point, vector};
    use lyon::geom::euclid;

    pub fn for_each_offset<S: Scalar, F>(curve: &QuadraticBezierSegment<S>, dist: S, tolerance: S, cb: &mut F)
    where F: FnMut(&QuadraticBezierSegment<S>) {
        let tolerance = tolerance.max(S::value(0.001));

        let (t1, t2) = critical_points(curve, dist);

        let mut  t0 = S::ZERO;
        if let Some(t1) = t1 {
            let sub_curve = curve.split_range(t0..t1);
            for_each_offset_inner(&sub_curve, dist, tolerance, cb);
            t0 = t1;
        }

        if let Some(t2) = t2 {
            let sub_curve = curve.split_range(t0..t2);
            for_each_offset_inner(&sub_curve, dist, tolerance, cb);
            t0 = t2;
        }

        let sub_curve = curve.split_range(t0..S::ONE);
        for_each_offset_inner(&sub_curve, dist, tolerance, cb);
/*
        if let Some(t) = find_sharp_turn(curve) {
            let t0 = t * S::value(0.8);
            let t1 = t + (S::ONE - t) * S::value(0.8);
            for range in [S::ZERO..t0, t0..t, t..t1, t1..S::ONE] {
                let sub_curve = curve.split_range(range);
                for_each_offset_inner(&sub_curve, dist, tolerance, cb);
            }
        } else {
            for_each_offset_inner(curve, dist, tolerance, cb);
        }
*/
    }

    fn for_each_offset_inner<S: Scalar, F>(curve: &QuadraticBezierSegment<S>, dist: S, tolerance: S, cb: &mut F)
    where F: FnMut(&QuadraticBezierSegment<S>) {
        let mut range = S::ZERO..S::ONE;
        let end = range.end;

        fn normal<S: Scalar>(v: Vector<S>) -> Vector<S> {
            Vector::new(-v.y, v.x)
        }

        let mut v0 = (curve.ctrl - curve.from).normalize();
        let mut from = curve.from + normal(v0) * dist;

        while range.start < range.end {
            let sub_curve = curve.split_range(range.clone());

            // Offset the sub-curve
            let v1 = (sub_curve.to - sub_curve.ctrl).normalize();
            let n1 = normal(v1);
            let n = normal((v0 + v1).normalize());
            let inv_len = n.dot(n1);
            let ctrl_offset = if inv_len != S::ZERO { dist / inv_len } else { dist };
            let ctrl = sub_curve.ctrl + n * ctrl_offset;

            let to = sub_curve.to + n1 * dist;
            let offset_curve = QuadraticBezierSegment { from, ctrl, to, };

            // Evaluate the error.
            let err = if range.end - range.start < S::value(0.05) {
                S::ZERO
            } else {
                let d = (sub_curve.sample(S::HALF) - offset_curve.sample(S::HALF)).length();
                (d - dist.abs()).abs()
            };

            if err <= tolerance {
                cb(&offset_curve);
                range = range.end .. end;
                from = to;
                v0 = v1;
            } else {
                range.end = (range.start + range.end) * S::HALF;
            }
        }
    }

    fn _find_sharp_turn<S: Scalar>(curve: &QuadraticBezierSegment<S>) -> Option<S> {
        // TODO: There's a better approach described in https://blend2d.com/research/precise_offset_curves.pdf
        // TODO: The various thresholds here should take the line width into account.
        let baseline = curve.to - curve.from;
        let v = curve.ctrl - curve.from;
        let n = vector(-baseline.y, baseline.x);
        let v_dot_b = v.dot(baseline);
        let v_dot_n = v.dot(n);

        // If the projection of the control point on the baseline is between the endpoints, we
        // can only get a sharp turn with a control point that is very far away.
        let long_axis = if (v_dot_b >= S::ZERO && v_dot_b <= baseline.dot(baseline)) || v_dot_n.abs() * S::TWO >= v_dot_b.abs() {
            // The control point is far enough from the endpoints It can cause a sharp turn.
            if baseline.square_length() * S::value(30.0) > v.square_length() {
                return None;
            }

            v
        } else {
            baseline
        };

        // Rotate the curve to find its extremum along the long axis, where we should split to
        // avoid the sharp turn.
        let rot = euclid::Rotation2D::new(-long_axis.angle_from_x_axis());
        let rotated = QuadraticBezierSegment {
            from: point(S::ZERO, S::ZERO),
            ctrl: rot.transform_vector(v).to_point(),
            to: rot.transform_vector(baseline).to_point(),
        };

        rotated.local_x_extremum_t()
    }

    fn critical_points<S: Scalar>(curve: &QuadraticBezierSegment<S>, offset: S) -> (Option<S>, Option<S>) {
        // See https://blend2d.com/research/precise_offset_curves.pdf
        let from = curve.from;
        let ctrl = curve.ctrl;
        let to = curve.to;
        let v1 = from - ctrl;
        let a = (ctrl - to) * S::TWO - v1;
        let b = v1 * S::TWO;

        let axbx_ayby = a.x * b.x + a.y * b.y;
        let axbx_ayby_2 = axbx_ayby * axbx_ayby;
        let ax2_ay2 = a.x * a.x + a.y * a.y;
        let bx2_by2 = b.x * b.x + b.y * b.y;
        let dot_ab = b.x * a.y - a.x * b.y;

        let r3 = (offset * offset * dot_ab * dot_ab).cbrt();
        let r2 = (axbx_ayby_2 - ax2_ay2 * (bx2_by2 - r3)).sqrt();

        let mut t1 = (-axbx_ayby + r2) / ax2_ay2;
        let mut t2 = (-axbx_ayby - r2) / ax2_ay2;

        if t2 < t1 {
            std::mem::swap(&mut t1, &mut t2);
        }

        // We don't want to split very close to the endpoints.
        let eps = S::value(0.05);

        let t1 = if t1 > S::ZERO + eps && t1 < S::ONE - eps { Some(t1) } else { None };
        let t2 = if t2 > S::ZERO + eps && t2 < S::ONE - eps { Some(t2) } else { None };

        (t1, t2)
    }
}

fn _quads_per_arc(angle: f32, radius: f32, tolerance: f32) -> f32 {
    let e = tolerance / radius;
    let n = 4.0 *  ((2.0 + e - (e + (2.0 + e)).sqrt()).sqrt() / std::f32::consts::SQRT_2).acos();

    (n * 0.5 * angle/std::f32::consts::PI).ceil()
}

pub type CapBuilder = fn(
    output: &mut dyn PathBuilder,
    position: Point,
    tangent: Vector,
    end: Point,
    width: f32,
    sign: f32,
    tolerance: f32,
) -> ();

pub type JoinBuilder = fn(
    output: &mut dyn PathBuilder,
    position: Point,
    start_tangent: Vector,
    end_tangent: Vector,
    width: f32,
    miter_limit:f32,
    tolerance: f32,
    prev_is_line: bool,
    next_is_line: bool,
) -> ();

fn bevel_join(
    output: &mut dyn PathBuilder,
    position: Point,
    start_tangent: Vector,
    end_tangent: Vector,
    width: f32,
    _miter_limit: f32,
    _tolerance: f32,
    prev_is_line: bool,
    _next_is_line: bool,
) {
    if prev_is_line {
        let start = position + normal(start_tangent) * width;
        output.line_to(start, NO_ATTRIBUTES);
    }
    let end = position + normal(end_tangent) * width;
    output.line_to(end, NO_ATTRIBUTES);
}

fn try_miter_join(
    output: &mut dyn PathBuilder,
    normal: Vector,
    position: Point,
    end_offset: Vector,
    width: f32,
    miter_limit: f32,
    next_is_line: bool,
) -> bool {
    if miter_limit_is_exceeded(normal, miter_limit) {
        return false;
    }

    output.line_to(position + normal * width, NO_ATTRIBUTES);
    if !next_is_line {
        output.line_to(position + end_offset, NO_ATTRIBUTES);
    }

    true
}

fn miter_join(
    output: &mut dyn PathBuilder,
    position: Point,
    start_tangent: Vector,
    end_tangent: Vector,
    width: f32,
    miter_limit: f32,
    _tolerance: f32,
    prev_is_line: bool,
    next_is_line: bool,
) {
    if prev_is_line {
        let start = position + normal(start_tangent) * width;
        output.line_to(start, NO_ATTRIBUTES);
    }

    let end_offset = normal(end_tangent) * width;
    let normal = compute_normal(start_tangent, end_tangent);

    if try_miter_join(output, normal, position, end_offset, width, miter_limit, next_is_line) {
        return;
    }

    output.line_to(position + end_offset, NO_ATTRIBUTES);
}

fn miter_clip_join(
    output: &mut dyn PathBuilder,
    position: Point,
    start_tangent: Vector,
    end_tangent: Vector,
    width: f32,
    miter_limit: f32,
    _tolerance: f32,
    prev_is_line: bool,
    next_is_line: bool,
) {
    let start_offset = normal(start_tangent) * width;
    let end_offset = normal(end_tangent) * width;
    let normal = compute_normal(start_tangent, end_tangent);

    if prev_is_line {
        output.line_to(position + start_offset, NO_ATTRIBUTES);
    }

    if try_miter_join(output, normal, position, end_offset, width, miter_limit, next_is_line) {
        return;
    }

    let (v0, v1) = get_clip_intersections(
        start_offset,
        end_offset,
        normal * width.signum(),
        miter_limit * width.abs() * 0.5,
    );

    output.line_to(position + v0, NO_ATTRIBUTES);
    output.line_to(position + v1, NO_ATTRIBUTES);
    if !next_is_line {
        output.line_to(position + end_offset, NO_ATTRIBUTES);
    }
}

fn quad_join(
    output: &mut dyn PathBuilder,
    position: Point,
    start_tangent: Vector,
    end_tangent: Vector,
    width: f32,
    _miter_limit: f32,
    _tolerance: f32,
    _prev_is_line: bool,
    _next_is_line: bool,
) {
    let start = position + normal(start_tangent) * width;
    output.line_to(start, NO_ATTRIBUTES);

    let end = position + normal(end_tangent) * width;
    let normal = compute_normal(start_tangent, end_tangent);

    let ctrl = position + normal * width;
    output.quadratic_bezier_to(ctrl, end, NO_ATTRIBUTES);
}

fn cubic_join(
    output: &mut dyn PathBuilder,
    position: Point,
    start_tangent: Vector,
    end_tangent: Vector,
    width: f32,
    _miter_limit: f32,
    _tolerance: f32,
    prev_is_line: bool,
    _next_is_line: bool,
) {
    let sign = start_tangent.cross(end_tangent).signum();
    let start_normal = normal(start_tangent);
    let end_normal = normal(end_tangent);
    let start = position + start_normal * width;
    let end = position + end_normal * width;
    if prev_is_line {
        output.line_to(start, NO_ATTRIBUTES);
    }

    let angle = (start_normal * sign).angle_to(end_normal * sign).radians.abs();

    // From https://pomax.github.io/bezierinfo/#circles
    let k = width.abs() * 4.0/3.0 * (angle * 0.25).tan();

    let ctrl1 = start + start_tangent * k;
    let ctrl2 = end - end_tangent * k;

    output.cubic_bezier_to(ctrl1, ctrl2, end, NO_ATTRIBUTES);
}

fn round_join(
    output: &mut dyn PathBuilder,
    position: Point,
    start_tangent: Vector,
    end_tangent: Vector,
    width: f32,
    miter_limit: f32,
    tolerance: f32,
    prev_is_line: bool,
    next_is_line: bool,
) {
    cubic_join(output, position, start_tangent, end_tangent, width, miter_limit, tolerance, prev_is_line, next_is_line)
}

fn butt_cap(
    output: &mut dyn PathBuilder,
    _position: Point,
    _tangent: Vector,
    end: Point,
    _width: f32,
    _sign: f32,
    _tolerance: f32) {
    output.line_to(end, NO_ATTRIBUTES);
}

fn square_cap(
    output: &mut dyn PathBuilder,
    pivot: Point,
    tangent: Vector,
    end: Point,
    width: f32,
    sign: f32,
    _tolerance: f32,
) {
    let tw = tangent * width;
    let n = normal(tw) * sign;
    let mid = pivot + tw;
    let a = mid + n;
    let b = mid - n;

    output.line_to(a, NO_ATTRIBUTES);
    output.line_to(b, NO_ATTRIBUTES);
    output.line_to(end, NO_ATTRIBUTES);
}

fn _round_cap_quad(
    output: &mut dyn PathBuilder,
    pivot: Point,
    tangent: Vector,
    end: Point,
    width: f32,
    sign: f32,
    _tolerance: f32,
) {
    let tangent = tangent.normalize();
    let tw = tangent * width;
    let n = normal(tangent) * sign;
    let mid = pivot + tw;

    const SIN_FRACT_PI_4: f32 = 0.70710678118;
    const SIN_FRACT_PI_8: f32 = SIN_FRACT_PI_4 * 0.5;
    let d1 = SIN_FRACT_PI_4 * width;
    let d2 = SIN_FRACT_PI_8 * width;

    let start = pivot + n * width;

    let p0 = start + tangent * d2;
    let p1 = pivot + (tangent + n) * d1;
    let p2 = mid + n * d2;

    let p3 = mid - n * d2;
    let p4 = pivot + (tangent - n) * d1;
    let p5 = end + tangent * d2;

    output.quadratic_bezier_to(p0, p1, NO_ATTRIBUTES);
    output.quadratic_bezier_to(p2, mid, NO_ATTRIBUTES);
    output.quadratic_bezier_to(p3, p4, NO_ATTRIBUTES);
    output.quadratic_bezier_to(p5, end, NO_ATTRIBUTES);
}

fn round_cap_cubic(
    output: &mut dyn PathBuilder,
    pivot: Point,
    tangent: Vector,
    end: Point,
    width: f32,
    sign: f32,
    _tolerance: f32,
) {
    let normal = normal(tangent) * sign;
    let start = pivot + normal * width;

    // From https://pomax.github.io/bezierinfo/#circles
    let angle = std::f32::consts::PI * 0.5;
    let k = width.abs() * 4.0/3.0 * (angle * 0.25).tan();

    let mid = pivot + tangent * width;
    let ctrl1 = start + tangent * k;
    let ctrl2 = mid + normal * k;
    output.cubic_bezier_to(ctrl1, ctrl2, mid, NO_ATTRIBUTES);
    let ctrl1 = mid - normal * k;
    let ctrl2 = end + tangent * k;
    output.cubic_bezier_to(ctrl1, ctrl2, end, NO_ATTRIBUTES);
}

fn round_inverted_cap(
    output: &mut dyn PathBuilder,
    pivot: Point,
    tangent: Vector,
    end: Point,
    width: f32,
    sign: f32,
    _tolerance: f32,
) {
    round_cap_cubic(output, pivot, -tangent, end, width, -sign, _tolerance);
}

fn triangle_cap(
    output: &mut dyn PathBuilder,
    pivot: Point,
    tangent: Vector,
    end: Point,
    width: f32,
    _sign: f32,
    _tolerance: f32,
) {
    let tw = tangent * width;
    let mid = pivot + tw;

    output.line_to(mid, NO_ATTRIBUTES);
    output.line_to(end, NO_ATTRIBUTES);
}

fn triangle_inverted_cap(
    output: &mut dyn PathBuilder,
    pivot: Point,
    tangent: Vector,
    end: Point,
    width: f32,
    sign: f32,
    tolerance: f32,
) {
    triangle_cap(output, pivot, -tangent, end, width, -sign, tolerance)
}

// Derived from:
// miter_limit = miter_length / stroke_width
// miter_limit = (normal.length() * half_width) / (2.0 * half_width)
fn miter_limit_is_exceeded(normal: Vector, miter_limit: f32) -> bool {
    normal.square_length() > miter_limit * miter_limit * 4.0
}

/// Compute a normal vector at a point P such that ```x ---e1----> P ---e2---> x```
///
/// The resulting vector is not normalized. The length is such that extruding the shape
/// would yield parallel segments exactly 1 unit away from their original. (useful
/// for generating strokes and vertex-aa).
/// The normal points towards the positive side of e1.
///
/// v1 and v2 are expected to be normalized.
pub fn compute_normal(v1: Vector, v2: Vector) -> Vector {
    //debug_assert!((v1.length() - 1.0).abs() < 0.001, "v1 should be normalized ({})", v1.length());
    //debug_assert!((v2.length() - 1.0).abs() < 0.001, "v2 should be normalized ({})", v2.length());

    let epsilon = 1e-4;

    let n1 = Vector::new(-v1.y, v1.x);

    let v12 = v1 + v2;

    // !(a < b) instead of a <= b to catch NaN situations.
    if !(v12.square_length() > epsilon) {
        return Vector::new(0.0, 0.0);
    }

    let tangent = v12.normalize();
    let n = Vector::new(-tangent.y, tangent.x);

    let inv_len = n.dot(n1);

    if inv_len.abs() < epsilon {
        return n1;
    }

    n / inv_len
}

fn get_clip_intersections(
    start_offset: Vector,
    end_offset: Vector,
    normal: Vector,
    clip_distance: f32,
) -> (Vector, Vector) {
    let clip_line = Line {
        point: normal.normalize().to_point() * clip_distance,
        vector: tangent(normal),
    };

    let prev_line = Line {
        point: start_offset.to_point(),
        vector: tangent(start_offset),
    };

    let next_line = Line {
        point: end_offset.to_point(),
        vector: tangent(end_offset),
    };

    let i1 = clip_line
        .intersection(&prev_line)
        .map(|p| p.to_f32())
        .unwrap_or_else(|| normal.to_point())
        .to_vector();
    let i2 = clip_line
        .intersection(&next_line)
        .map(|p| p.to_f32())
        .unwrap_or_else(|| normal.to_point())
        .to_vector();

    (i1, i2)
}

pub type Output<'l> = &'l mut dyn PathBuilder;

#[derive(Copy, Clone, Debug)]
enum Seg {
    Line(LineSegment<f32>),
    Quadratic(QuadraticBezierSegment<f32>),
}

impl Seg {
    pub fn from(&self) -> Point {
        match self {
            Seg::Line(line) => line.from,
            Seg::Quadratic(quad) => quad.from,
        }
    }

    pub fn to(&self) -> Point {
        match self {
            Seg::Line(line) => line.to,
            Seg::Quadratic(quad) => quad.to,
        }
    }

    pub fn is_line(&self) -> bool {
        match self {
            Seg::Line { .. } => true,
            Seg::Quadratic { .. } => false,
        }
    }
}

pub struct OffsetOptions {
    pub offset: f32,
    pub join: LineJoin,
    pub miter_limit: f32,
    pub tolerance: f32,
    pub simplify_inner_joins: bool,
}

pub struct OffsetBuilder<'l> {
    output: Output<'l>,
    prev: Seg,
    prev_endpoint: Point,
    first_endpoint: Point,
    prev_tangent: Vector,
    first_tangent: Vector,
    offset: f32,
    miter_limit: f32,
    tolerance: f32,
    simplify_inner_joins: bool,
    skip_begin: bool,
    skip_end: bool,
    is_first_segment: bool,
    join: JoinBuilder,
}

impl<'l> OffsetBuilder<'l> {
    pub fn new(output: &'l mut dyn PathBuilder, options: &OffsetOptions) -> Self {
        OffsetBuilder {
            output,
            miter_limit: options.miter_limit.max(1.0),
            tolerance: options.tolerance.max(0.0001),
            offset: options.offset,
            prev: Seg::Line(LineSegment { from: Point::zero(), to: Point::zero() }),
            prev_endpoint: Point::zero(),
            prev_tangent: Vector::zero(),
            first_endpoint: Point::zero(),
            first_tangent: Vector::zero(),
            is_first_segment: true,
            skip_begin: false,
            skip_end: false,
            join: match options.join {
                LineJoin::Bevel => bevel_join,
                LineJoin::Miter => miter_join,
                LineJoin::MiterClip => miter_clip_join,
                LineJoin::Round => round_join,
                LineJoin::Quad => quad_join,
            },
            simplify_inner_joins: options.simplify_inner_joins,
        }
    }

    pub fn begin(&mut self, at: Point) {
        self.prev_endpoint = at;
        self.first_endpoint = at;
        self.is_first_segment = true;
    }

    pub fn end(&mut self, close: bool) {
        if self.is_first_segment {
            return;
        }

        let end_at_start = (self.prev_endpoint - self.first_endpoint).square_length() <= self.tolerance * self.tolerance;
        let close = close || end_at_start;

        if close {
            // Add the join between last and first segment.
            self.line_to(self.first_endpoint);

            let simplify = self.simplify_inner_joins;
            self.simplify_inner_joins = false;
            self.line_to(self.first_endpoint + self.first_tangent);
            self.simplify_inner_joins = simplify;

            self.output.end(true);
        } else {
            self.add_prev_segment();
            if !self.skip_end {
                self.output.end(false);
            }
        }
    }

    pub fn line_to(&mut self, to: Point) {
        if (to - self.prev_endpoint).square_length() < self.tolerance * self.tolerance {
            return;
        }

        let unit_tangent = (to - self.prev_endpoint).normalize();
        let d = normal(unit_tangent) * self.offset;

        let mut segment = Seg::Line(LineSegment {
            from: self.prev_endpoint + d,
            to: to + d,
        });

        self.step(&mut segment, unit_tangent);
        self.prev_tangent = unit_tangent;
        self.prev_endpoint = to;
    }

    pub fn quadratic_bezier_to(&mut self, ctrl: Point, to: Point) {
        let from = self.prev_endpoint;
        let curve = QuadraticBezierSegment { from, ctrl, to };

        match classify_quadratic_bezier(&curve, self.tolerance) {
            QuadType::Linear => {
                self.line_to(to);
                return;
            }
            QuadType::Degenerate => {
                self.line_to(curve.sample(0.5));
                self.line_to(curve.to);
                return;
            }
            QuadType::Curve => {}
        }

        let mut first = true;
        offset::for_each_offset(&curve, self.offset, self.tolerance, &mut|sub_curve| {
            let mut segment = Seg::Quadratic(*sub_curve);
            if first {
                first = false;
                let start_tangent = (curve.ctrl - curve.from).normalize();
                self.step(&mut segment, start_tangent);
            } else {
                self.step_without_join(&segment)
            }
        });
        self.prev_tangent = (to - ctrl).normalize();
        self.prev_endpoint = to;
    }

    pub fn cubic_bezier_to(&mut self, ctrl1: Point, ctrl2: Point, to: Point) {
        let curve = CubicBezierSegment { from: self.prev_endpoint, ctrl1, ctrl2, to };

        if curve.is_linear(self.tolerance) {
            self.line_to(to);
            return;
        }

        let cusps = find_cubic_bezier_cusps(&curve);
        //println!("-----");
        let mut t0 = 0.0;
        let eps = 0.0001;
        for cusp in cusps {
            //println!("cusp at {:?}", cusp);
            let sub_curve = curve.split_range(t0..(cusp - eps));
            sub_curve.for_each_quadratic_bezier(self.tolerance, &mut |quad| {
                self.quadratic_bezier_to(quad.ctrl, quad.to);
            });
            t0 = cusp+eps;
        }

        let sub_curve = curve.split_range(t0..1.0);
        sub_curve.for_each_quadratic_bezier(self.tolerance, &mut |quad| {
            self.quadratic_bezier_to(quad.ctrl, quad.to)
        });
    }

    fn step_without_join(&mut self, segment: &Seg) {
        debug_assert!(!self.is_first_segment);
        self.add_prev_segment();
        self.prev = *segment;
    }

    fn step(&mut self, segment: &mut Seg, unit_tangent: Vector) {
        let no_join = (self.prev.to() - segment.from()).square_length() < self.tolerance * self.tolerance;

        if self.is_first_segment {
            self.first_tangent = unit_tangent;
            self.is_first_segment = false;
            if !self.skip_begin {
                self.output.begin(segment.from(), NO_ATTRIBUTES);
            }
        } else if no_join {
            self.add_prev_segment();
        } else {
            let t01 = unit_tangent - self.prev_tangent;
            let n01 = (normal(unit_tangent) + normal(self.prev_tangent)) * self.offset.signum();
            // We are a bit conserative about inner joins when the dot product is close to zero.
            // That's when the next segment goes back along the previous one  and are close to
            // parallel at which point it becomes difficult
            let inner_join = t01.dot(n01) > -0.01;

            //println!("> w:{:?} inner:{inner_join:?} t01:{t01:?} n01:{n01:?} t01.n01{:?} cross:{cross:?} dot:{dot:?} pt: {positive_turn:?} pos: {:?}", self.offset, t01.dot(n01), self.prev_endpoint);

            if inner_join {
                self.inner_join(segment);
            } else {
                self.outer_join(segment, unit_tangent);
            }
        }

        self.prev = *segment;
    }

    // Inner joins are expected to be inside the shape so we simplify them into a
    // single line segment and even try to prevent the folding that happens where
    // the two offset segments intersect in the simple cases.
    fn inner_join(&mut self, next: &mut Seg) {
        // If the two segments are lines unfold the join.
        let mut simplified = false;
        if self.simplify_inner_joins {
            if let (Seg::Line(l0), Seg::Line(l1)) = (&mut self.prev, &mut *next) {
                if let Some(intersection) = l0.intersection(l1) {
                    l0.to = intersection;
                    l1.from = intersection;
                    simplified = true;
                }
            }
        }

        self.add_prev_segment();

        if !simplified {
            self.output.line_to(next.from(), NO_ATTRIBUTES);
        }
    }

    fn add_prev_segment(&mut self) {
        match &self.prev {
            Seg::Line(line) => {
                self.output.line_to(line.to, NO_ATTRIBUTES);
            }
            Seg::Quadratic(curve) => {
                self.output.quadratic_bezier_to(curve.ctrl, curve.to, NO_ATTRIBUTES);
            }
        }
    }

    fn outer_join(&mut self, next: &Seg, unit_tangent: Vector) {
        if let Seg::Quadratic(curve) = self.prev {
            self.output.quadratic_bezier_to(curve.ctrl, curve.to, NO_ATTRIBUTES);
        }

        (self.join)(
            self.output,
            self.prev_endpoint,
            self.prev_tangent,
            unit_tangent,
            self.offset,
            self.miter_limit,
            self.tolerance,
            self.prev.is_line(),
            next.is_line(),
        );
    }
}

fn get_cap_fn(cap : LineCap) -> CapBuilder {
    match cap {
        LineCap::Butt => butt_cap,
        LineCap::Square => square_cap,
        LineCap::Round => round_cap_cubic,
        LineCap::RoundInverted => round_inverted_cap,
        LineCap::Triangle => triangle_cap,
        LineCap::TriangleInverted => triangle_inverted_cap,
    }
}

pub struct StrokeToFillBuilder<'l> {
    offsetter: OffsetBuilder<'l>,
    offsets: (f32, f32),
    opposite: lyon::path::path::Builder,
    start_cap: CapBuilder,
    end_cap: CapBuilder,
    add_empty_caps: bool,
}

impl<'l> StrokeToFillBuilder<'l> {
    pub fn new(output: &'l mut dyn PathBuilder, options: &StrokeOptions) -> Self {
        let offset0 = options.offsets.0;
        let offset1 = options.offsets.1;

        let offset_options = OffsetOptions {
            offset: offset0,
            tolerance: options.tolerance,
            miter_limit: options.miter_limit,
            join: options.line_join,
            simplify_inner_joins: true,
        };

        let mut offset = OffsetBuilder::new(output, &offset_options);
        offset.skip_end = true;

        StrokeToFillBuilder {
            offsetter: offset,
            offsets: (offset0, offset1),
            start_cap: get_cap_fn(options.start_cap),
            end_cap: get_cap_fn(options.end_cap),
            opposite: lyon::path::Path::builder(),
            add_empty_caps: options.add_empty_caps,
        }
    }

    pub fn begin(&mut self, at: Point) {
        self.offsetter.begin(at);
        self.opposite.begin(at);
    }

    pub fn line_to(&mut self, to: Point) {
        self.offsetter.line_to(to);
        self.opposite.line_to(to);
    }

    pub fn quadratic_bezier_to(&mut self, ctrl: Point, to: Point) {
        self.offsetter.quadratic_bezier_to(ctrl, to);
        self.opposite.quadratic_bezier_to(ctrl, to);
    }

    pub fn cubic_bezier_to(&mut self, ctrl1: Point, ctrl2: Point, to: Point) {
        self.offsetter.cubic_bezier_to(ctrl1, ctrl2, to);
        self.opposite.cubic_bezier_to(ctrl1, ctrl2, to);
    }

    pub fn end(&mut self, close: bool) {
        if self.offsetter.is_first_segment {
            if self.add_empty_caps {
                self.add_empty_cap();
            }
            self.opposite = Default::default();
            return;
        }

        let start_endpoint = self.offsetter.first_endpoint;
        let start_tangent = self.offsetter.first_tangent;

        self.offsetter.end(close);

        let need_caps = !close;

        if close {
            self.opposite.end(true);
        }

        if need_caps {
            self.add_end_cap();
            self.opposite.end(false);
        }

        // Swap the offset while replaying the opposite contour.
        // Invert the sign since we are going to replay in reverse order.
        self.offsetter.offset = -self.offsets.1;
        self.offsetter.skip_begin = !close;

        let rev = std::mem::take(&mut self.opposite).build();

        for evt in rev.reversed() {
            match evt {
                PathEvent::Line { to, .. } => {
                    self.offsetter.line_to(to);
                }
                PathEvent::Quadratic { ctrl, to, .. } => {
                    self.offsetter.quadratic_bezier_to(ctrl, to);
                }
                PathEvent::Cubic {ctrl1, ctrl2, to, .. } => {
                    self.offsetter.cubic_bezier_to(ctrl1, ctrl2, to);
                }
                PathEvent::Begin { at } => {
                    self.offsetter.begin(at);
                }
                PathEvent::End { close, .. } => {
                    self.offsetter.end(close);
                }
            }
        }

        if need_caps {
            self.add_start_cap(start_endpoint, -start_tangent);
        }

        self.offsetter.skip_begin = false;
        self.offsetter.offset = self.offsets.0;
    }

    fn add_start_cap(&mut self, endpoint: Point, tangent: Vector) {
        let n = normal(tangent);
        let width = (self.offsets.0 - self.offsets.1).abs() * 0.5;
        let pivot_offset = (self.offsets.0 + self.offsets.1) * 0.5;
        let pivot = endpoint - n * pivot_offset;
        let end = endpoint - n * self.offsets.0;
        let sign = (self.offsets.0 - self.offsets.1).signum();
        (self.start_cap)(self.offsetter.output, pivot, tangent, end, width, sign, self.offsetter.tolerance);
        self.offsetter.output.end(true);
    }

    fn add_end_cap(&mut self) {
        let tangent = self.offsetter.prev_tangent;
        let n = normal(tangent);
        let width = (self.offsets.0 - self.offsets.1).abs() * 0.5;
        let pivot_offset = (self.offsets.0 + self.offsets.1) * 0.5;
        let pivot = self.offsetter.prev_endpoint + n * pivot_offset;
        let end = self.offsetter.prev_endpoint + n * self.offsets.1;
        let sign = (self.offsets.0 - self.offsets.1).signum();
        (self.end_cap)(self.offsetter.output, pivot, tangent, end, width, sign, self.offsetter.tolerance);
    }

    fn add_empty_cap(&mut self) {
        let tangent = Vector::new(1.0, 0.0);
        let normal = normal(tangent);

        // We don't have a real tangent so we don't offset the pivot.
        let pivot = self.offsetter.first_endpoint;

        let width = (self.offsets.0 - self.offsets.1).abs() * 0.5;
        let p0 = pivot + normal * width;
        let p1 = pivot - normal * width;

        let tolerance = self.offsetter.tolerance;
        let output = &mut *self.offsetter.output;
        output.begin(p0, NO_ATTRIBUTES);
        (self.end_cap)(output, pivot, tangent, p1, width, 1.0, tolerance);
        (self.start_cap)(output, pivot, -tangent, p0, width, 1.0, tolerance);
        output.end(true);
    }
}

enum QuadType {
    Curve,
    Linear,
    Degenerate,
}

fn classify_quadratic_bezier(curve: &QuadraticBezierSegment<f32>, tolerance: f32) -> QuadType {
    if !curve.is_linear(tolerance) {
        return QuadType::Curve
    }

    let baseline = curve.to - curve.from;
    if baseline.dot(curve.ctrl - curve.from) < 0.0
    || (-baseline).dot(curve.ctrl - curve.to) < 0.0 {
        return QuadType::Degenerate
    }

    QuadType::Linear
}

fn find_cubic_bezier_cusps(curve: &CubicBezierSegment<f32>) -> ArrayVec<f32, 4> {
    let mut cusps = ArrayVec::new();

    let x_deriv = deriv_dot_deriv2_1d(curve.from.x, curve.ctrl1.x, curve.ctrl2.x, curve.to.x);
    let y_deriv = deriv_dot_deriv2_1d(curve.from.x, curve.ctrl1.x, curve.ctrl2.x, curve.to.x);

    let a = x_deriv[0] + y_deriv[0];
    let b = x_deriv[1] + y_deriv[1];
    let c = x_deriv[2] + y_deriv[2];
    let d = x_deriv[3] + y_deriv[3];

    for root in cubic_polynomial_roots(a, b, c, d) {
        if root > 0.0 && root < 1.0 {
            cusps.push(root);
        }
    }

    cusps.sort_by(|a, b| a.partial_cmp(b).unwrap() );

    cusps
}

// Compute F' dot F''
//
// F' = 3ct^2 + 6bt + 3a
// F'' = 6ct + 6b
//
// F' dot F'' = cct^3 + 3bct^2 + (2bb + ca)t + ab
fn deriv_dot_deriv2_1d(from: f32, ctrl1:f32, ctrl2: f32, to: f32) -> [f32; 4] {
    let a = ctrl1 - from;
    let b = ctrl2 - 2.0 * ctrl1 + from;
    let c = to + 3.0 * (ctrl1 - ctrl2) - from;

    [c * c, 3.0 * b * c, 2.0 * b * b + c * a, a * b]
}
