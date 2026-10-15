// TODO: this doesn't belong in the core crate.

use core::geom::{
    arrayvec::ArrayVec,
    utils::{cubic_polynomial_roots, tangent},
    CubicBezierSegment, Line, LineSegment, QuadraticBezierSegment,
};
use core::units::{Point, Vector};
use core::path::{builder::PathBuilder, Path, Attributes, EndpointId, PathEvent, NO_ATTRIBUTES};


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
            miter_limit: 4.0,
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

    pub fn stroke_to_fill(
        &mut self,
        path: impl Iterator<Item = PathEvent>,
        options: &StrokeOptions,
    ) -> Path {
        let mut builder = Path::builder();

        {
            let mut stroker = StrokeToFillBuilder::new(&mut builder, &options);
            for event in path {
                match event {
                    PathEvent::Begin { at } => {
                        stroker.begin(at);
                    }
                    PathEvent::End { close, .. } => {
                        stroker.end(close);
                    }
                    PathEvent::Line { to, .. } => {
                        stroker.line_to(to);
                    }
                    PathEvent::Quadratic { ctrl, to, .. } => {
                        stroker.quadratic_bezier_to(ctrl, to);
                    }
                    PathEvent::Cubic {
                        ctrl1, ctrl2, to, ..
                    } => {
                        stroker.cubic_bezier_to(ctrl1, ctrl2, to);
                    }
                }
            }
        }

        builder.build()
    }
}

impl<'a> PathBuilder for OffsetBuilder<'a> {
    fn num_attributes(&self) -> usize {
        0
    }

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

    fn cubic_bezier_to(
        &mut self,
        ctrl1: Point,
        ctrl2: Point,
        to: Point,
        _: Attributes,
    ) -> EndpointId {
        self.cubic_bezier_to(ctrl1, ctrl2, to);
        EndpointId::INVALID
    }
}

impl<'a> PathBuilder for StrokeToFillBuilder<'a> {
    fn num_attributes(&self) -> usize {
        0
    }

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

    fn cubic_bezier_to(
        &mut self,
        ctrl1: Point,
        ctrl2: Point,
        to: Point,
        _: Attributes,
    ) -> EndpointId {
        self.cubic_bezier_to(ctrl1, ctrl2, to);
        EndpointId::INVALID
    }
}

fn normal(v: Vector) -> Vector {
    Vector::new(-v.y, v.x)
}

pub mod offset {
    use core::geom::arrayvec::ArrayVec;
    use core::geom::euclid;
    use core::geom::{point, vector, Point, QuadraticBezierSegment, Scalar, Vector};

    pub fn for_each_offset<S: Scalar, F>(
        curve: &QuadraticBezierSegment<S>,
        dist: S,
        tolerance: S,
        cb: &mut F,
    ) where
        F: FnMut(&QuadraticBezierSegment<S>),
    {
        let tolerance = tolerance.max(S::value(0.001));

        let mut splits: ArrayVec<S, 5> = ArrayVec::new();
        let (t1, t2) = critical_points(curve, dist);
        splits.extend(t1);
        splits.extend(t2);
        splits.extend(turn_splits(curve));
        splits.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut t0 = S::ZERO;
        for t1 in splits.into_iter().chain(Some(S::ONE)) {
            if t1 > t0 {
                for_each_offset_inner(curve, t0..t1, dist, tolerance, cb);
                t0 = t1;
            }
        }
    }

    fn normal<S: Scalar>(v: Vector<S>) -> Vector<S> {
        Vector::new(-v.y, v.x)
    }

    /// Maximum angle that the tangent of the curve turns by over each of the
    /// ranges passed to for_each_offset_inner.
    ///
    /// for_each_offset_inner places the control point of each offset sub-curve
    /// at dist / cos(turn / 2) from the curve, which grows without bound as the
    /// turn gets close to 180 degrees. Bounding the turn also bounds the error of
    /// the sub-curves that are accepted without meeting the tolerance (see
    /// min_step): a quadratic bézier curve approximates a 45 degree arc within
    /// 0.3% of its radius.
    const MAX_TURN: f32 = std::f32::consts::FRAC_PI_4;

    /// Parameters that split the curve into parts that turn by at most MAX_TURN.
    fn turn_splits<S: Scalar>(curve: &QuadraticBezierSegment<S>) -> ArrayVec<S, 3> {
        let mut splits = ArrayVec::new();

        // The derivative is 2 (p + t q).
        let p = curve.ctrl - curve.from;
        let q = curve.from.to_vector() - curve.ctrl.to_vector() * S::TWO + curve.to.to_vector();
        let p_end = p + q;
        // Less than 180 degrees.
        let turn = S::atan2(p.cross(p_end), p.dot(p_end));
        let max_turn = S::value(MAX_TURN);
        let count = (turn.abs() / max_turn).ceil();
        if !(count > S::ONE) {
            return splits;
        }

        let p_unit = p.normalize();
        let mut i = S::ONE;
        while i < count {
            // Solve cross(p + t q, direction) = 0.
            let angle = turn * i / count;
            let (sin, cos) = euclid::Angle::radians(angle).sin_cos();
            let direction = p_unit * cos + normal(p_unit) * sin;
            let t = -p.cross(direction) / q.cross(direction);
            if t > S::ZERO && t < S::ONE {
                splits.push(t);
            }
            i = i + S::ONE;
        }

        splits
    }

    fn for_each_offset_inner<S: Scalar, F>(
        curve: &QuadraticBezierSegment<S>,
        range: std::ops::Range<S>,
        dist: S,
        tolerance: S,
        cb: &mut F,
    ) where
        F: FnMut(&QuadraticBezierSegment<S>),
    {
        debug_assert!(curve.from.is_finite());
        debug_assert!(curve.ctrl.is_finite());
        debug_assert!(curve.to.is_finite());

        // Sub-curves are accepted regardless of the error below this parameter
        // range.
        let min_step = (range.end - range.start) * S::value(0.05);

        let end = range.end;
        let mut range = range;

        // Tangents are computed from the derivative of the curve rather than the
        // control points of the sub-curves, which lose precision as the
        // sub-curves get small.
        let tangent = |t: S| curve.derivative(t).normalize();
        let offset_point = |t: S| curve.sample(t) + normal(tangent(t)) * dist;

        // Distance between a point and the offset curve, estimated from the
        // closest point of the curve. It is found with Newton iterations on the
        // derivative of the squared distance, starting at parameter t and
        // staying within t_min..t_max.
        let second_derivative = (curve.from.to_vector() - curve.ctrl.to_vector() * S::TWO
            + curve.to.to_vector())
            * S::TWO;
        let offset_error = |p: Point<S>, mut t: S, t_min: S, t_max: S| {
            for _ in 0..2 {
                let v = curve.sample(t) - p;
                let d1 = curve.derivative(t);
                let g = v.dot(d1);
                let dg = d1.square_length() + v.dot(second_derivative);
                // Not a minimum of the distance, which is the case on the
                // part of the offset curve that goes backwards between its
                // cusps.
                if !(dg > S::ZERO) {
                    break;
                }
                t = (t - g / dg).max(t_min).min(t_max);
            }
            ((p - curve.sample(t)).dot(normal(tangent(t))) - dist).abs()
        };

        let mut v0 = tangent(range.start);
        let mut from = curve.sample(range.start) + normal(v0) * dist;

        while range.start < end {
            let sub_curve = curve.split_range(range.clone());

            // Offset the sub-curve
            let v1 = tangent(range.end);
            let n1 = normal(v1);
            let n = normal((v0 + v1).normalize());
            // At least cos(MAX_TURN / 2).
            let inv_len = n.dot(n1);
            if !(inv_len > S::ZERO) || !from.is_finite() {
                // The derivative of the curve vanishes, which only happens for
                // degenerate curves. Join the offset of the end of the curve
                // with a straight line.
                let to = offset_point(end);
                if from.is_finite() && to.is_finite() {
                    let ctrl = from.lerp(to, S::HALF);
                    cb(&QuadraticBezierSegment { from, ctrl, to });
                }
                return;
            }
            let ctrl = sub_curve.ctrl + n * (dist / inv_len);
            let to = sub_curve.to + n1 * dist;
            let offset_curve = QuadraticBezierSegment { from, ctrl, to };

            // Evaluate the error.
            let step = range.end - range.start;
            let mut err = S::ZERO;
            if step > min_step {
                for s in [S::value(0.25), S::HALF, S::value(0.75)] {
                    let t = range.start + step * s;
                    let e = offset_error(offset_curve.sample(s), t, range.start, range.end);
                    err = err.max(e);
                }
            }

            if err <= tolerance {
                cb(&offset_curve);
                range = range.end..end;
                from = to;
                v0 = v1;
            } else {
                range.end = range.start + step * S::HALF;
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
        let long_axis = if (v_dot_b >= S::ZERO && v_dot_b <= baseline.dot(baseline))
            || v_dot_n.abs() * S::TWO >= v_dot_b.abs()
        {
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

    fn critical_points<S: Scalar>(
        curve: &QuadraticBezierSegment<S>,
        offset: S,
    ) -> (Option<S>, Option<S>) {
        // See https://blend2d.com/research/precise_offset_curves.pdf
        //
        // The offset curve has a cusp where the radius of curvature of the curve
        // is equal to the offset distance d. With B'(t) = 2At + B and B'' = 2A,
        // the radius of curvature is |B'|³ / |B' x B''| = |B'|³ / (2 |A x B|),
        // so these points are the solutions of |B'(t)|² = cbrt(4 d² (A x B)²):
        //
        // 4 |A|² t² + 4 (A.B) t + |B|² - cbrt(4 d² (A x B)²) = 0
        //
        // Only the solutions on the concave side are cusps of the offset curve.
        // Splitting at the others is harmless.
        let from = curve.from.to_vector();
        let ctrl = curve.ctrl.to_vector();
        let to = curve.to.to_vector();
        let a = from - ctrl * S::TWO + to;
        let b = (ctrl - from) * S::TWO;

        let a_dot_b = a.dot(b);
        let a_len2 = a.square_length();
        let a_cross_b = a.cross(b);

        let r = (S::FOUR * offset * offset * a_cross_b * a_cross_b).cbrt();
        // NaN when there is no solution, in which case both comparisons below fail.
        let sqrt_delta = (a_dot_b * a_dot_b - a_len2 * (b.square_length() - r)).sqrt();

        let t1 = (-a_dot_b - sqrt_delta) / (S::TWO * a_len2);
        let t2 = (-a_dot_b + sqrt_delta) / (S::TWO * a_len2);

        // We don't want to split very close to the endpoints.
        let eps = S::value(0.05);

        let t1 = if t1 > S::ZERO + eps && t1 < S::ONE - eps {
            Some(t1)
        } else {
            None
        };
        let t2 = if t2 > S::ZERO + eps && t2 < S::ONE - eps {
            Some(t2)
        } else {
            None
        };

        (t1, t2)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn quad(from: (f32, f32), ctrl: (f32, f32), to: (f32, f32)) -> QuadraticBezierSegment<f32> {
            QuadraticBezierSegment {
                from: point(from.0, from.1),
                ctrl: point(ctrl.0, ctrl.1),
                to: point(to.0, to.1),
            }
        }

        const CURVES: [((f32, f32), (f32, f32), (f32, f32)); 4] = [
            ((0.0, 0.0), (50.0, 100.0), (100.0, 0.0)),
            ((0.0, 0.0), (100.0, 100.0), (20.0, 0.0)),
            ((10.0, 30.0), (200.0, 0.0), (0.0, 90.0)),
            ((0.0, 0.0), (1000.0, 1.0), (0.0, 2.0)),
        ];

        #[test]
        fn critical_points_match_radius_of_curvature() {
            for (from, ctrl, to) in CURVES {
                let curve = quad(from, ctrl, to);
                let b2 = (curve.from.to_vector() - curve.ctrl.to_vector() * 2.0 + curve.to.to_vector()) * 2.0;
                let radius = |t: f32| {
                    let b1 = curve.derivative(t);
                    b1.length().powi(3) / b1.cross(b2).abs()
                };

                for offset in [1.0f32, 5.0, 20.0, 40.0, 80.0, -40.0] {
                    // Find where the radius of curvature crosses the offset distance
                    // by sampling, away from the endpoints where critical_points
                    // discards solutions.
                    let mut expected = Vec::new();
                    let n = 100_000;
                    let mut prev = radius(0.0) - offset.abs();
                    for i in 1..=n {
                        let t = i as f32 / n as f32;
                        let r = radius(t) - offset.abs();
                        if r.signum() != prev.signum() && t > 0.051 && t < 0.949 {
                            expected.push(t);
                        }
                        prev = r;
                    }

                    let (t1, t2) = critical_points(&curve, offset);
                    let found: Vec<f32> = [t1, t2].into_iter().flatten().collect();
                    assert_eq!(found.len(), expected.len(), "{curve:?} {offset}: {found:?} != {expected:?}");
                    for (a, b) in found.iter().zip(expected.iter()) {
                        assert!((a - b).abs() < 1e-3, "{curve:?} {offset}: {found:?} != {expected:?}");
                    }
                }
            }
        }

        #[test]
        fn offset_is_within_tolerance() {
            let tolerance = 0.25;
            for (from, ctrl, to) in CURVES {
                let curve = quad(from, ctrl, to);
                for dist in [1.0f32, -1.0, 5.0, -5.0, 20.0, -20.0, 50.0, -50.0] {
                    let exact = |t: f32| {
                        let tangent = curve.derivative(t).normalize();
                        curve.sample(t) + vector(-tangent.y, tangent.x) * dist
                    };
                    // Sample the exact offset curve uniformly in t and in tangent
                    // angle, so that sharp turns are sampled densely as well.
                    let n = 2000;
                    let p0 = curve.ctrl - curve.from;
                    let q = curve.from.to_vector() - curve.ctrl.to_vector() * 2.0 + curve.to.to_vector();
                    let turn = p0.angle_to(p0 + q).radians;
                    let mut ts: Vec<f32> = (0..=n).map(|i| i as f32 / n as f32).collect();
                    for i in 1..n {
                        // The tangent is parallel to p0 + t q.
                        let angle = turn * i as f32 / n as f32;
                        let dir = p0.normalize() * angle.cos() + vector(-p0.y, p0.x).normalize() * angle.sin();
                        let t = -p0.cross(dir) / q.cross(dir);
                        if t > 0.0 && t < 1.0 {
                            ts.push(t);
                        }
                    }
                    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let samples: Vec<_> = ts.iter().map(|t| exact(*t)).collect();

                    // Distance to the exact offset curve: closest sample, refined
                    // with a ternary search between its neighbors.
                    let distance = |p: Point<f32>| {
                        let i = (0..samples.len())
                            .min_by(|a, b| {
                                let da = (samples[*a] - p).square_length();
                                let db = (samples[*b] - p).square_length();
                                da.partial_cmp(&db).unwrap()
                            })
                            .unwrap();
                        let mut lo = ts[i.saturating_sub(1)];
                        let mut hi = ts[(i + 1).min(ts.len() - 1)];
                        for _ in 0..40 {
                            let t1 = lo + (hi - lo) / 3.0;
                            let t2 = hi - (hi - lo) / 3.0;
                            if (exact(t1) - p).length() < (exact(t2) - p).length() {
                                hi = t2;
                            } else {
                                lo = t1;
                            }
                        }
                        (exact((lo + hi) * 0.5) - p).length()
                    };

                    let mut max_err: f32 = 0.0;
                    for_each_offset(&curve, dist, tolerance, &mut |sub_curve| {
                        for i in 0..=16 {
                            max_err = max_err.max(distance(sub_curve.sample(i as f32 / 16.0)));
                        }
                    });
                    assert!(max_err < tolerance * 1.2, "{curve:?} {dist}: {max_err}");
                }
            }
        }
    }
}

fn _quads_per_arc(angle: f32, radius: f32, tolerance: f32) -> f32 {
    let e = tolerance / radius;
    let n = 4.0 * ((2.0 + e - (e + (2.0 + e)).sqrt()).sqrt() / std::f32::consts::SQRT_2).acos();

    (n * 0.5 * angle / std::f32::consts::PI).ceil()
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
    miter_limit: f32,
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

    if try_miter_join(
        output,
        normal,
        position,
        end_offset,
        width,
        miter_limit,
        next_is_line,
    ) {
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

    if try_miter_join(
        output,
        normal,
        position,
        end_offset,
        width,
        miter_limit,
        next_is_line,
    ) {
        return;
    }

    // A zero normal means that the segments go in (nearly) opposite directions,
    // in which case the miter points forward, along the bisector.
    let miter_direction = if normal == Vector::zero() {
        start_tangent - end_tangent
    } else {
        normal * width.signum()
    };

    let (v0, v1) = get_clip_intersections(
        start_offset,
        end_offset,
        miter_direction,
        // Half of the miter length limit (miter_limit * stroke_width).
        miter_limit * width.abs(),
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

    let angle = (start_normal * sign)
        .angle_to(end_normal * sign)
        .radians
        .abs();

    // From https://pomax.github.io/bezierinfo/#circles
    let k = width.abs() * 4.0 / 3.0 * (angle * 0.25).tan();

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
    cubic_join(
        output,
        position,
        start_tangent,
        end_tangent,
        width,
        miter_limit,
        tolerance,
        prev_is_line,
        next_is_line,
    )
}

fn butt_cap(
    output: &mut dyn PathBuilder,
    _position: Point,
    _tangent: Vector,
    end: Point,
    _width: f32,
    _sign: f32,
    _tolerance: f32,
) {
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
    let k = width.abs() * 4.0 / 3.0 * (angle * 0.25).tan();

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

// The miter length is the distance between the inner corner and the tip of the
// miter, which are both normal.length() * half_width away from the join point.
// miter_limit = miter_length / stroke_width
// miter_limit = (2.0 * normal.length() * half_width) / (2.0 * half_width)
//
// compute_normal returns a zero vector when the miter length is infinite.
fn miter_limit_is_exceeded(normal: Vector, miter_limit: f32) -> bool {
    normal == Vector::zero() || normal.square_length() > miter_limit * miter_limit
}

/// Compute a normal vector at a point P such that ```x ---e1----> P ---e2---> x```
///
/// The resulting vector is not normalized. The length is such that extruding the shape
/// would yield parallel segments exactly 1 unit away from their original. (useful
/// for generating strokes and vertex-aa).
/// The normal points towards the positive side of e1.
/// Returns a zero vector if v1 and v2 point in (nearly) opposite directions.
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
    /// Treat open sub-paths that end at their start point as closed.
    close_if_end_at_start: bool,
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
            prev: Seg::Line(LineSegment {
                from: Point::zero(),
                to: Point::zero(),
            }),
            prev_endpoint: Point::zero(),
            prev_tangent: Vector::zero(),
            first_endpoint: Point::zero(),
            first_tangent: Vector::zero(),
            is_first_segment: true,
            skip_begin: false,
            skip_end: false,
            close_if_end_at_start: true,
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

        let end_at_start = (self.prev_endpoint - self.first_endpoint).square_length()
            <= self.tolerance * self.tolerance;
        let close = close || (self.close_if_end_at_start && end_at_start);

        if close {
            self.line_to(self.first_endpoint);

            // Add the join between the last and first segments by stepping into
            // a stand-in for the first segment. It bypasses line_to so that it is
            // never discarded for being shorter than the tolerance.
            let d = normal(self.first_tangent) * self.offset;
            let mut first = Seg::Line(LineSegment {
                from: self.first_endpoint + d,
                to: self.first_endpoint + self.first_tangent + d,
            });

            let simplify = self.simplify_inner_joins;
            self.simplify_inner_joins = false;
            self.step(&mut first, self.first_tangent);
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
        offset::for_each_offset(&curve, self.offset, self.tolerance, &mut |sub_curve| {
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
        let curve = CubicBezierSegment {
            from: self.prev_endpoint,
            ctrl1,
            ctrl2,
            to,
        };

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
            t0 = cusp + eps;
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
        let no_join =
            (self.prev.to() - segment.from()).square_length() < self.tolerance * self.tolerance;

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
            // When the next segment goes back along the previous one, the offset
            // segments end on opposite sides of the stroke on both sides, so both
            // sides get an outer join to wrap around the tip.
            let reversal = self.prev_tangent.dot(unit_tangent) < 0.0
                && self.prev_tangent.cross(unit_tangent).abs() < 0.005;
            // We are a bit conserative about inner joins when the dot product is close to zero.
            let inner_join = !reversal && t01.dot(n01) > -0.01;

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
                self.output
                    .quadratic_bezier_to(curve.ctrl, curve.to, NO_ATTRIBUTES);
            }
        }
    }

    fn outer_join(&mut self, next: &Seg, unit_tangent: Vector) {
        if let Seg::Quadratic(curve) = self.prev {
            self.output
                .quadratic_bezier_to(curve.ctrl, curve.to, NO_ATTRIBUTES);
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

fn get_cap_fn(cap: LineCap) -> CapBuilder {
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
    opposite: core::path::BuilderInner,
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
        // Open sub-paths get caps even if they end at their start point, and
        // end() relies on the offsetter closing exactly when the caller asks.
        offset.close_if_end_at_start = false;

        StrokeToFillBuilder {
            offsetter: offset,
            offsets: (offset0, offset1),
            start_cap: get_cap_fn(options.start_cap),
            end_cap: get_cap_fn(options.end_cap),
            opposite: core::path::PathInner::builder(),
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
                PathEvent::Cubic {
                    ctrl1, ctrl2, to, ..
                } => {
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
        (self.start_cap)(
            self.offsetter.output,
            pivot,
            tangent,
            end,
            width,
            sign,
            self.offsetter.tolerance,
        );
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
        (self.end_cap)(
            self.offsetter.output,
            pivot,
            tangent,
            end,
            width,
            sign,
            self.offsetter.tolerance,
        );
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
        return QuadType::Curve;
    }

    let baseline = curve.to - curve.from;
    if baseline.dot(curve.ctrl - curve.from) < 0.0 || (-baseline).dot(curve.ctrl - curve.to) < 0.0 {
        return QuadType::Degenerate;
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

    cusps.sort_by(|a, b| a.partial_cmp(b).unwrap());

    cusps
}

// Compute F' dot F''
//
// F' = 3ct^2 + 6bt + 3a
// F'' = 6ct + 6b
//
// F' dot F'' = cct^3 + 3bct^2 + (2bb + ca)t + ab
fn deriv_dot_deriv2_1d(from: f32, ctrl1: f32, ctrl2: f32, to: f32) -> [f32; 4] {
    let a = ctrl1 - from;
    let b = ctrl2 - 2.0 * ctrl1 + from;
    let c = to + 3.0 * (ctrl1 - ctrl2) - from;

    [c * c, 3.0 * b * c, 2.0 * b * b + c * a, a * b]
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::units::point;

    fn stroke_polyline(points: &[Point], close: bool, options: &StrokeOptions) -> Path {
        let mut builder = Path::builder();
        {
            let mut stroker = StrokeToFillBuilder::new(&mut builder, options);
            stroker.begin(points[0]);
            for p in &points[1..] {
                stroker.line_to(*p);
            }
            stroker.end(close);
        }
        builder.build()
    }

    /// Absolute area of each contour. Only supports line segments.
    fn contour_areas(path: &Path) -> Vec<f32> {
        let mut areas = Vec::new();
        let mut area = 0.0;
        for event in path.iter() {
            match event {
                PathEvent::Begin { .. } => area = 0.0,
                PathEvent::Line { from, to } => area += from.to_vector().cross(to.to_vector()),
                PathEvent::End { last, first, .. } => {
                    area += last.to_vector().cross(first.to_vector());
                    areas.push(area.abs() * 0.5);
                }
                _ => panic!("unexpected curve"),
            }
        }
        areas
    }

    #[test]
    fn open_sub_path_ending_at_start() {
        let options = StrokeOptions::line_with(10.0).with_line_cap(LineCap::Square);
        let triangle = [
            point(0.0, 0.0),
            point(100.0, 0.0),
            point(100.0, 100.0),
            point(0.0, 0.0),
        ];
        let hairpin = [point(0.0, 0.0), point(100.0, 0.0), point(0.0, 0.0)];

        for points in [&triangle[..], &hairpin[..]] {
            let path = stroke_polyline(points, false, &options);

            // A single contour that goes out and back, with a cap at each end.
            let mut begins = 0;
            let mut ends = 0;
            for event in path.iter() {
                match event {
                    PathEvent::Begin { .. } => begins += 1,
                    PathEvent::End { close, .. } => {
                        assert!(close);
                        ends += 1;
                    }
                    _ => {}
                }
            }
            assert_eq!((begins, ends), (1, 1));

            // The square cap at the start point extends behind it.
            let min_x = path
                .iter()
                .filter_map(|event| match event {
                    PathEvent::Line { to, .. } => Some(to.x),
                    _ => None,
                })
                .fold(f32::MAX, f32::min);
            assert!(min_x < -4.9, "min_x = {min_x}");
        }
    }

    #[test]
    fn closed_sub_path_with_coarse_tolerance() {
        let square = [
            point(0.0, 0.0),
            point(100.0, 0.0),
            point(100.0, 100.0),
            point(0.0, 100.0),
        ];

        for tolerance in [0.25, 1.5] {
            let options = StrokeOptions::line_with(10.0)
                .with_line_join(LineJoin::Miter)
                .with_tolerance(tolerance);
            let path = stroke_polyline(&square, true, &options);

            // The outer contour has miter corners, the inner one a small fold at
            // the closing corner.
            let areas = contour_areas(&path);
            assert_eq!(areas.len(), 2);
            assert!((areas[0] - 110.0 * 110.0).abs() < 1.0, "tolerance {tolerance}: {areas:?}");
            assert!((areas[1] - 90.0 * 90.0).abs() < 50.0, "tolerance {tolerance}: {areas:?}");
        }
    }

    /// Distance that the join of a wedge reaches past the corner, along the outer
    /// bisector. The two segments meet at the given angle.
    fn wedge_join_extent(angle: f32, join: LineJoin, miter_limit: f32, half_width: f32) -> f32 {
        let corner = point(100.0, 0.0);
        let d_out = Vector::new(-angle.cos(), angle.sin());
        let wedge = [point(0.0, 0.0), corner, corner + d_out * 100.0];
        let options = StrokeOptions::line_with(half_width * 2.0)
            .with_line_join(join)
            .with_miter_limit(miter_limit)
            .with_line_cap(LineCap::Butt);
        let path = stroke_polyline(&wedge, false, &options);

        let bisector = (Vector::new(1.0, 0.0) - d_out).normalize();
        path.iter()
            .filter_map(|event| match event {
                PathEvent::Line { to, .. } => Some((to - corner).dot(bisector)),
                _ => None,
            })
            .fold(f32::MIN, f32::max)
    }

    #[test]
    fn miter_limit() {
        let miter_limit = 4.0;
        let hw = 5.0;
        for degrees in (10..=60).step_by(2) {
            let angle = (degrees as f32).to_radians();
            // Ratio between the miter length and the stroke width.
            let ratio = 1.0 / (angle * 0.5).sin();

            let miter = wedge_join_extent(angle, LineJoin::Miter, miter_limit, hw);
            if ratio <= miter_limit {
                assert!((miter - ratio * hw).abs() < 1e-3, "{degrees} degrees: {miter}");
            } else {
                // Bevel.
                assert!(miter < hw, "{degrees} degrees: {miter}");
            }

            let clipped = wedge_join_extent(angle, LineJoin::MiterClip, miter_limit, hw);
            let expected = (ratio * hw).min(miter_limit * hw);
            assert!((clipped - expected).abs() < 1e-3, "{degrees} degrees: {clipped} != {expected}");
        }
    }

    /// Furthest extent of the outline along a direction, sampling curves.
    fn extent(path: &Path, direction: Vector) -> f32 {
        let mut extent = f32::MIN;
        for event in path.iter() {
            for i in 0..=32 {
                let t = i as f32 / 32.0;
                let p = match event {
                    PathEvent::Begin { at } => at,
                    PathEvent::Line { from, to } => from.lerp(to, t),
                    PathEvent::Quadratic { from, ctrl, to } => {
                        QuadraticBezierSegment { from, ctrl, to }.sample(t)
                    }
                    PathEvent::Cubic { from, ctrl1, ctrl2, to } => {
                        CubicBezierSegment { from, ctrl1, ctrl2, to }.sample(t)
                    }
                    PathEvent::End { .. } => continue,
                };
                extent = extent.max(p.to_vector().dot(direction));
            }
        }
        extent
    }

    #[test]
    fn hairpin_turns() {
        let hw = 5.0;
        let miter_limit = 4.0;
        // Turning angles in degrees, up to a complete reversal.
        for degrees in [179.0f32, 179.5, 179.6, 179.8, 179.95, 180.0] {
            let a = (180.0 - degrees).to_radians();
            let corner = point(100.0, 0.0);
            let d_out = Vector::new(-a.cos(), a.sin());
            let hairpin = [point(0.0, 0.0), corner, corner + d_out * 100.0];
            // Distance past the corner along the outer bisector.
            let bisector = (Vector::new(1.0, 0.0) - d_out).normalize();
            let past_corner = |path: &Path| extent(path, bisector) - corner.to_vector().dot(bisector);
            let stroke = |join| {
                let options = StrokeOptions::line_with(hw * 2.0)
                    .with_line_join(join)
                    .with_miter_limit(miter_limit)
                    .with_line_cap(LineCap::Butt);
                stroke_polyline(&hairpin, false, &options)
            };

            // The round join wraps around the tip.
            let round = past_corner(&stroke(LineJoin::Round));
            assert!((round - hw).abs() < 0.01, "{degrees}: {round}");

            // The miter limit is exceeded so miter joins are bevels.
            let miter: Vec<PathEvent> = stroke(LineJoin::Miter).iter().collect();
            let bevel: Vec<PathEvent> = stroke(LineJoin::Bevel).iter().collect();
            assert_eq!(miter, bevel, "{degrees}");

            // The miter is clipped half of the miter length limit past the corner.
            let clipped = past_corner(&stroke(LineJoin::MiterClip));
            assert!((clipped - miter_limit * hw).abs() < 0.01, "{degrees}: {clipped}");
        }
    }

    #[test]
    fn closed_two_point_sub_path() {
        let options = StrokeOptions::line_with(10.0).with_line_join(LineJoin::Round);
        let path = stroke_polyline(&[point(0.0, 0.0), point(100.0, 0.0)], true, &options);
        let max = extent(&path, Vector::new(1.0, 0.0));
        let min = -extent(&path, Vector::new(-1.0, 0.0));
        assert!((max - 105.0).abs() < 0.01, "{max}");
        assert!((min + 5.0).abs() < 0.01, "{min}");
    }
}
