// TODO: this doesn't belong in the core crate.

use lyon::{
    path::{
        builder::PathBuilder,
        NO_ATTRIBUTES, PathEvent, EndpointId, Attributes,
    },
    geom::{QuadraticBezierSegment, CubicBezierSegment, Line, utils::tangent, LineSegment}
};

use crate::{units::{Point, Vector}, path::Path};

pub use lyon::path::LineJoin;

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
    use lyon::geom::{Scalar, QuadraticBezierSegment, Point, Vector, point, vector};
    use lyon::geom::euclid;

    struct QuadOffsetter<S> {
        prev_tangent: Vector<S>,
        potential_prev_tangent: Vector<S>,
        from: Point<S>,
        d: S,
    }

    impl<S: Scalar> QuadOffsetter<S> {
        fn new(curve: &QuadraticBezierSegment<S>, d: S) -> Self {
            let v0 = (curve.ctrl - curve.from).normalize();
            let n0 = vector(-v0.y, v0.x);
            let from = curve.from + n0 * d;

            QuadOffsetter { prev_tangent: v0, potential_prev_tangent: v0, from, d }
        }

        fn offset(&mut self, curve: &QuadraticBezierSegment<S>) -> QuadraticBezierSegment<S> {
            let v0 = self.prev_tangent;
            let from = curve.from + vector(-v0.y, v0.x) * self.d;

            let v1 = (curve.to - curve.ctrl).normalize();
            let n1 = vector(-v1.y, v1.x);
            let to = curve.to + n1 * self.d;
            self.potential_prev_tangent = v1;

            let v01 = v0 + v1;

            let tangent = v01.normalize();
            let n = vector(-tangent.y, tangent.x);

            let inv_len = n.dot(n1);

            let d2 = if inv_len != S::ZERO {
                self.d / inv_len
            } else {
                self.d
            };

            let ctrl = curve.ctrl + n * d2;

            QuadraticBezierSegment {
                from, ctrl, to,
            }
        }

        fn next(&mut self) {
            self.prev_tangent = self.potential_prev_tangent;
        }
    }

    pub fn for_each_offset<S: Scalar, F>(curve: &QuadraticBezierSegment<S>, dist: S, tolerance: S, cb: &mut F)
    where F: FnMut(&QuadraticBezierSegment<S>) {
        let tolerance = tolerance.max(S::value(0.001));
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
    }

    fn for_each_offset_inner<S: Scalar, F>(curve: &QuadraticBezierSegment<S>, dist: S, tolerance: S, cb: &mut F)
    where F: FnMut(&QuadraticBezierSegment<S>) {
        let mut range = S::ZERO..S::ONE;
        let end = range.end;
        let mut offsetter = QuadOffsetter::new(curve, dist);

        while range.start < range.end {
            let sub_curve = curve.split_range(range.clone());
            let offset = offsetter.offset(&sub_curve);

            let err = if range.end - range.start < S::value(0.1) {
                S::ZERO
            } else {
                let d = (sub_curve.sample(S::HALF) - offset.sample(S::HALF)).length();
                (d - dist.abs()).abs()
            };

            if err <= tolerance {
                cb(&offset);
                range = range.end .. end;
                offsetter.next();
            } else {
                range.end = (range.start + range.end) * S::HALF;
            }
        }
    }

    fn find_sharp_turn<S: Scalar>(curve: &QuadraticBezierSegment<S>) -> Option<S> {
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

fn round_cap(
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
    let ctrl1 = mid + n;
    let ctrl2 = mid - n;

    // TODO: this is a pretty poor approximation.
    output.quadratic_bezier_to(ctrl1, mid, NO_ATTRIBUTES);
    output.quadratic_bezier_to(ctrl2, end, NO_ATTRIBUTES);
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
    let tw = tangent * width;
    let n = normal(tw) * sign;
    let mid = pivot - tw;
    let ctrl1 = mid + n;
    let ctrl2 = mid - n;

    output.quadratic_bezier_to(ctrl1, mid, NO_ATTRIBUTES);
    output.quadratic_bezier_to(ctrl2, end, NO_ATTRIBUTES);
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
    _sign: f32,
    _tolerance: f32,
) {
    let tw = tangent * width;
    let mid = pivot - tw;

    output.line_to(mid, NO_ATTRIBUTES);
    output.line_to(end, NO_ATTRIBUTES);
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
                LineJoin::Round => quad_join,
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
        offset::for_each_offset(&curve, self.offset, self.tolerance, &mut|curve| {
            let start_tangent = (curve.ctrl - curve.from).normalize();
            self.step(&mut Seg::Quadratic(*curve), start_tangent);
            self.prev_tangent = (to - ctrl).normalize();
            self.prev_endpoint = to;
        });
    }

    pub fn cubic_bezier_to(&mut self, ctrl1: Point, ctrl2: Point, to: Point) {
        let curve = CubicBezierSegment { from: self.prev_endpoint, ctrl1, ctrl2, to };
        curve.for_each_quadratic_bezier(self.tolerance, &mut |quad| {
            self.quadratic_bezier_to(quad.ctrl, quad.to)
        });
    }

    fn step(&mut self, segment: &mut Seg, unit_tangent: Vector) {
        //println!("segment {:?}", segment);

        let no_join = self.prev.to() == segment.from();

        if self.is_first_segment {
            self.first_tangent = unit_tangent;
            self.is_first_segment = false;
            if !self.skip_begin {
                self.output.begin(segment.from(), NO_ATTRIBUTES);
            }
        } else if no_join {
            self.add_prev_segment();
        } else {
            let positive_turn = self.prev_tangent.cross(unit_tangent) >= 0.0;
            let inner_join = positive_turn ^ (self.offset < 0.0);

            if inner_join {
                self.inner_join(segment);
            } else {
                self.outer_join(segment, unit_tangent);
            }
        }

        self.prev = *segment;
    }

    fn inner_join(&mut self, next: &mut Seg) {
        // TODO: also simplify in the presence of curves?
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
        LineCap::Round => round_cap,
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
