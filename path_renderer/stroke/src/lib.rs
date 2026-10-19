// TODO: this doesn't belong in the core crate.

use core::geom::{
    arrayvec::ArrayVec,
    utils::{cubic_polynomial_roots, tangent},
    CubicBezierSegment, Line, LineSegment, QuadraticBezierSegment,
};
use core::units::{Point, Vector};
use core::path::{builder::PathBuilder, Path, Attributes, EndpointId, PathEvent, NO_ATTRIBUTES};

pub mod skia;


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
    /// Storage for StrokeToFillBuilder::opposite, kept between calls to avoid
    /// reallocating it.
    opposite: SegmentBuffer,
}

impl Stroker {
    pub fn new() -> Stroker {
        Stroker {
            opposite: SegmentBuffer::default(),
        }
    }

    pub fn stroke_to_fill(
        &mut self,
        path: impl Iterator<Item = PathEvent>,
        options: &StrokeOptions,
    ) -> Path {
        let mut builder = Path::builder();

        {
            let mut stroker = StrokeToFillBuilder::new(&mut builder, &options);
            stroker.opposite = std::mem::take(&mut self.opposite);
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
            self.opposite = std::mem::take(&mut stroker.opposite);
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
        for_each_offsets(curve, [dist], tolerance, &mut |sub_curves| cb(&sub_curves[0]));
    }

    /// Offsets a curve by up to two distances at once.
    ///
    /// The offset curves are approximated over the same parameter ranges, so
    /// that the analysis of the curve is shared: the curve is split at the
    /// critical points of every distance, and a range is split further until
    /// the offset sub-curves of every distance are within the tolerance. The
    /// callback receives the offset sub-curves of each range, in the order of
    /// `dists`.
    pub fn for_each_offsets<S: Scalar, F, const N: usize>(
        curve: &QuadraticBezierSegment<S>,
        dists: [S; N],
        tolerance: S,
        cb: &mut F,
    ) where
        F: FnMut(&[QuadraticBezierSegment<S>; N]),
    {
        const { assert!(N >= 1 && N <= 2) };

        let tolerance = tolerance.max(S::value(0.001));

        let mut splits: ArrayVec<S, 7> = ArrayVec::new();
        for (i, dist) in dists.iter().enumerate() {
            // The critical points only depend on the magnitude of the distance.
            if dists[..i].iter().any(|d| d.abs() == dist.abs()) {
                continue;
            }
            let (t1, t2) = critical_points(curve, *dist);
            splits.extend(t1);
            splits.extend(t2);
        }
        splits.extend(turn_splits(curve));
        splits.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut t0 = S::ZERO;
        for t1 in splits.into_iter().chain(Some(S::ONE)) {
            if t1 > t0 {
                for_each_offset_inner(curve, t0..t1, dists, tolerance, cb);
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

    fn for_each_offset_inner<S: Scalar, F, const N: usize>(
        curve: &QuadraticBezierSegment<S>,
        range: std::ops::Range<S>,
        dists: [S; N],
        tolerance: S,
        cb: &mut F,
    ) where
        F: FnMut(&[QuadraticBezierSegment<S>; N]),
    {
        debug_assert!(curve.from.is_finite());
        debug_assert!(curve.ctrl.is_finite());
        debug_assert!(curve.to.is_finite());

        // Sub-curves are accepted regardless of the error below this parameter
        // range, to guarantee termination. This is reached near the cusps of the
        // offset curve, or when the tolerance is too small for the precision of
        // the coordinates.
        let min_step = (range.end - range.start) * S::value(0.05);

        let end = range.end;
        let mut range = range;

        // Tangents are computed from the derivative of the curve rather than the
        // control points of the sub-curves, which lose precision as the
        // sub-curves get small.
        let tangent = |t: S| curve.derivative(t).normalize();
        let offset_point = |t: S, dist: S| curve.sample(t) + normal(tangent(t)) * dist;

        // Distance between a point and the offset curve, estimated from the
        // closest point of the curve. It is found with Newton iterations on the
        // derivative of the squared distance, starting at parameter t and
        // staying within t_min..t_max.
        let second_derivative = (curve.from.to_vector() - curve.ctrl.to_vector() * S::TWO
            + curve.to.to_vector())
            * S::TWO;
        let offset_error = |p: Point<S>, dist: S, mut t: S, t_min: S, t_max: S| {
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
        let mut from = dists.map(|dist| offset_point(range.start, dist));

        while range.start < end {
            let sub_curve = curve.split_range(range.clone());

            // Offset the sub-curve
            let v1 = tangent(range.end);
            let n1 = normal(v1);
            let n = normal((v0 + v1).normalize());
            // At least cos(MAX_TURN / 2).
            let inv_len = n.dot(n1);
            if !(inv_len > S::ZERO) || !from.iter().all(|p| p.is_finite()) {
                // The derivative of the curve vanishes, which only happens for
                // degenerate curves. Join the offset of the end of the curve
                // with a straight line.
                let to = dists.map(|dist| offset_point(end, dist));
                if from.iter().chain(&to).all(|p| p.is_finite()) {
                    cb(&std::array::from_fn(|i| QuadraticBezierSegment {
                        from: from[i],
                        ctrl: from[i].lerp(to[i], S::HALF),
                        to: to[i],
                    }));
                }
                return;
            }
            let offset_curves: [QuadraticBezierSegment<S>; N] =
                std::array::from_fn(|i| QuadraticBezierSegment {
                    from: from[i],
                    ctrl: sub_curve.ctrl + n * (dists[i] / inv_len),
                    to: sub_curve.to + n1 * dists[i],
                });

            // Evaluate the error. The sub-curve matches the offset curve's
            // endpoints and tangents, so the error vanishes at both ends and
            // peaks in between: near the middle, or up to about a quarter of the
            // way from one end when the curvature changes a lot over the range.
            // The peak is then up to about 1.5 times the error at the middle.
            // The error is almost always far below the tolerance, so the other
            // samples are only measured when the middle one is not.
            let step = range.end - range.start;
            let mut err = S::ZERO;
            if step > min_step {
                for (offset_curve, dist) in offset_curves.iter().zip(dists) {
                    let error_at = |s: S| {
                        let t = range.start + step * s;
                        offset_error(offset_curve.sample(s), dist, t, range.start, range.end)
                    };
                    let mut e = error_at(S::HALF);
                    if e > tolerance * S::HALF {
                        e = e.max(error_at(S::value(0.25))).max(error_at(S::value(0.75)));
                    }
                    err = err.max(e);
                }
            }

            if err <= tolerance {
                cb(&offset_curves);
                range = range.end..end;
                from = offset_curves.map(|c| c.to);
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

/// One side of the offset path.
#[derive(Copy, Clone)]
struct Side {
    offset: f32,
    /// The last offset segment. It is added to the output once the join that
    /// follows it is known, since an inner join can shorten it.
    prev: Seg,
}

/// Offsets a path by one or two distances, each one being a side of the
/// offset path.
///
/// The source path is processed once for all sides: short segments are
/// discarded, curves are split and approximated, and joins are classified as
/// inner or outer. The sides can't disagree about any of these decisions.
///
/// Each side is written to its own output. The methods take the outputs in the
/// order of the sides.
struct Offsetter<const N: usize> {
    sides: [Side; N],
    /// End of the last segment added to the outputs.
    prev_endpoint: Point,
    /// End of the last segment of the source path, including the segments that
    /// were discarded for being too short. The next segment starts here, and is
    /// displaced to start at `prev_endpoint` instead (see
    /// `displace_quadratic_bezier` and `displace_cubic_bezier`).
    position: Point,
    first_endpoint: Point,
    prev_tangent: Vector,
    first_tangent: Vector,
    miter_limit: f32,
    tolerance: f32,
    simplify_inner_joins: bool,
    /// Treat open sub-paths that end at their start point as closed.
    close_if_end_at_start: bool,
    is_first_segment: bool,
    /// Line segments shorter than this are discarded. See
    /// `default_min_segment_length`.
    min_segment_length: f32,
    /// Endpoints of the line segments discarded since the beginning of the
    /// sub-path, while no segment has been added yet.
    discarded: Vec<Point>,
    /// Number of segments added to the outputs so far.
    num_steps: u32,
    join: JoinBuilder,
}

impl<const N: usize> Offsetter<N> {
    fn new(
        offsets: [f32; N],
        join: LineJoin,
        miter_limit: f32,
        tolerance: f32,
        simplify_inner_joins: bool,
    ) -> Self {
        let no_segment = Seg::Line(LineSegment {
            from: Point::zero(),
            to: Point::zero(),
        });
        Offsetter {
            sides: offsets.map(|offset| Side {
                offset,
                prev: no_segment,
            }),
            miter_limit: miter_limit.max(1.0),
            tolerance: tolerance.max(0.0001),
            prev_endpoint: Point::zero(),
            position: Point::zero(),
            prev_tangent: Vector::zero(),
            first_endpoint: Point::zero(),
            first_tangent: Vector::zero(),
            is_first_segment: true,
            min_segment_length: tolerance.max(0.0001) * 0.5,
            discarded: Vec::new(),
            num_steps: 0,
            close_if_end_at_start: true,
            join: match join {
                LineJoin::Bevel => bevel_join,
                LineJoin::Miter => miter_join,
                LineJoin::MiterClip => miter_clip_join,
                LineJoin::Round => round_join,
                LineJoin::Quad => quad_join,
            },
            simplify_inner_joins,
        }
    }

    fn begin(&mut self, at: Point) {
        self.prev_endpoint = at;
        self.position = at;
        self.first_endpoint = at;
        self.is_first_segment = true;
        self.discarded.clear();
    }

    /// Length below which line segments are discarded, outside of the special
    /// cases that reduce it.
    ///
    /// The following curve is displaced by up to this distance to start where
    /// the output stops (see `displace_cubic_bezier`), on top of the error of the
    /// offset curves, which can reach the tolerance. Half of the tolerance keeps
    /// the sum of both reasonably close to the tolerance.
    fn default_min_segment_length(&self) -> f32 {
        self.tolerance * 0.5
    }

    /// If all segments of the current sub-path were discarded for being too
    /// short, adds them back with a much smaller length threshold.
    ///
    /// Without this, sub-paths that fit within the tolerance would not produce
    /// any output, although their stroke can be much larger than the tolerance.
    /// The threshold stays reduced until the end of the sub-path.
    fn recover_discarded_segments(&mut self, outputs: &mut [&mut dyn PathBuilder; N]) {
        if !self.is_first_segment || self.discarded.is_empty() {
            return;
        }
        self.min_segment_length = self.tolerance * 0.001;
        let discarded = std::mem::take(&mut self.discarded);
        for to in &discarded {
            self.line_to(*to, outputs);
        }
        self.discarded = discarded;
    }

    /// Ends the current sub-path.
    ///
    /// Returns `None` if nothing was added to the outputs, otherwise whether
    /// the sub-path is closed. The sub-paths of the outputs are not ended. If
    /// the sub-path is closed, they stop after the closing join.
    fn end(&mut self, close: bool, outputs: &mut [&mut dyn PathBuilder; N]) -> Option<bool> {
        self.recover_discarded_segments(outputs);
        if self.is_first_segment {
            self.min_segment_length = self.default_min_segment_length();
            return None;
        }

        let end_at_start = (self.prev_endpoint - self.first_endpoint).square_length()
            <= self.tolerance * self.tolerance;
        let close = close || (self.close_if_end_at_start && end_at_start);

        if close {
            // The closing segment is kept even if it is too short.
            // Discarding it would merge the joins at both of its ends into a single
            // one, which turns the wrong way when their angles add up to more than
            // 180 degrees.
            let min_segment_length = self.min_segment_length;
            self.min_segment_length = min_segment_length.min(self.tolerance * 0.001);
            self.line_to(self.first_endpoint, outputs);
            self.min_segment_length = min_segment_length;

            // Add the join between the last and first segments by stepping into
            // a stand-in for the first segment. It bypasses line_to so that it is
            // never discarded for being too short.
            let n = normal(self.first_tangent);
            let mut first = self.sides.map(|side| {
                let d = n * side.offset;
                Seg::Line(LineSegment {
                    from: self.first_endpoint + d,
                    to: self.first_endpoint + self.first_tangent + d,
                })
            });
            self.step(&mut first, self.first_tangent, false, outputs);
        } else {
            for (side, output) in self.sides.iter().zip(outputs.iter_mut()) {
                add_segment(&side.prev, *output);
            }
        }

        self.min_segment_length = self.default_min_segment_length();

        Some(close)
    }

    fn line_to(&mut self, to: Point, outputs: &mut [&mut dyn PathBuilder; N]) {
        self.position = to;
        if (to - self.prev_endpoint).square_length()
            < self.min_segment_length * self.min_segment_length
        {
            if self.is_first_segment {
                self.discarded.push(to);
            }
            return;
        }

        let unit_tangent = (to - self.prev_endpoint).normalize();
        let n = normal(unit_tangent);
        let mut segments = self.sides.map(|side| {
            let d = n * side.offset;
            Seg::Line(LineSegment {
                from: self.prev_endpoint + d,
                to: to + d,
            })
        });

        self.step(&mut segments, unit_tangent, self.simplify_inner_joins, outputs);
        self.prev_tangent = unit_tangent;
        self.prev_endpoint = to;
    }

    /// Control point and endpoint of a quadratic bézier curve that starts at
    /// the current position, displaced so that the curve starts at
    /// `prev_endpoint`.
    ///
    /// The control point moves with the start of the curve, weighted by how
    /// close it is to the start along the largest axis of the curve's extent
    /// between its endpoints. This keeps the tangent at the end the control
    /// point is closest to. If the curve ends where it starts, the whole curve
    /// moves, and the next segment absorbs the displacement.
    fn displace_quadratic_bezier(&self, ctrl: Point, to: Point) -> (Point, Point) {
        let from = self.position;
        let delta = self.prev_endpoint - from;
        if delta == Vector::zero() {
            return (ctrl, to);
        }
        if to == from {
            return (ctrl + delta, to + delta);
        }

        let v = to - from;
        let weight = if v.x.abs() >= v.y.abs() {
            (to.x - ctrl.x) / v.x
        } else {
            (to.y - ctrl.y) / v.y
        };

        (ctrl + delta * weight.max(0.0).min(1.0), to)
    }

    /// Control points and endpoint of a cubic bézier curve that starts at the
    /// current position, displaced so that the curve starts at `prev_endpoint`.
    ///
    /// The first control point moves with the start of the curve, so the
    /// tangents at both ends are unchanged and the curve moves by at most the
    /// displacement.
    fn displace_cubic_bezier(&self, ctrl1: Point, ctrl2: Point, to: Point) -> (Point, Point, Point) {
        let delta = self.prev_endpoint - self.position;
        (ctrl1 + delta, ctrl2, to)
    }

    fn quadratic_bezier_to(&mut self, ctrl: Point, to: Point, outputs: &mut [&mut dyn PathBuilder; N]) {
        let (displaced_ctrl, displaced_to) = self.displace_quadratic_bezier(ctrl, to);
        let curve = QuadraticBezierSegment {
            from: self.prev_endpoint,
            ctrl: displaced_ctrl,
            to: displaced_to,
        };
        let num_steps = self.num_steps;
        self.add_quadratic_bezier(&curve, outputs);
        if self.num_steps != num_steps {
            // See cubic_bezier_to.
            self.prev_endpoint = curve.to;
        }
        self.position = to;
    }

    fn add_quadratic_bezier(&mut self, curve: &QuadraticBezierSegment<f32>, outputs: &mut [&mut dyn PathBuilder; N]) {
        match classify_quadratic_bezier(curve, self.tolerance) {
            QuadType::Linear => {
                self.line_to(curve.to, outputs);
                return;
            }
            QuadType::Degenerate(t) => {
                self.line_to(curve.sample(t), outputs);
                self.line_to(curve.to, outputs);
                return;
            }
            QuadType::Curve => {}
        }

        let offsets = self.sides.map(|side| side.offset);
        let mut first = true;
        offset::for_each_offsets(curve, offsets, self.tolerance, &mut |sub_curves| {
            let mut segments = sub_curves.map(Seg::Quadratic);
            if first {
                first = false;
                let start_tangent = (curve.ctrl - curve.from).normalize();
                self.step(&mut segments, start_tangent, self.simplify_inner_joins, outputs);
            } else {
                self.step_without_join(&segments, outputs)
            }
        });
        self.prev_tangent = (curve.to - curve.ctrl).normalize();
        self.prev_endpoint = curve.to;
    }

    fn cubic_bezier_to(&mut self, ctrl1: Point, ctrl2: Point, to: Point, outputs: &mut [&mut dyn PathBuilder; N]) {
        let (ctrl1, ctrl2, to) = self.displace_cubic_bezier(ctrl1, ctrl2, to);
        let curve = CubicBezierSegment {
            from: self.prev_endpoint,
            ctrl1,
            ctrl2,
            to,
        };
        // The quadratic bézier curves that approximate the cubic bézier curve
        // start on the displaced curve.
        self.position = curve.from;
        let num_steps = self.num_steps;
        self.add_cubic_bezier(&curve, outputs);

        // If the curve added segments to the outputs, it ends exactly at `to`,
        // even where its last part does not: split_range has rounding errors, the
        // last part can stop before a cusp, or be discarded for being too short.
        // This avoids tiny closing segments with arbitrary directions.
        if self.num_steps != num_steps {
            self.prev_endpoint = to;
        }
        self.position = to;
    }

    fn add_cubic_bezier(&mut self, curve: &CubicBezierSegment<f32>, outputs: &mut [&mut dyn PathBuilder; N]) {
        if cubic_bezier_is_linear(curve, self.tolerance) {
            for t in cubic_bezier_turnarounds(curve) {
                self.line_to(curve.sample(t), outputs);
            }
            self.line_to(curve.to, outputs);
            return;
        }

        // Leave a small gap around the cusps, where the tangent is undefined. The
        // sub-curves on each side are connected with a join.
        let cusps = find_cubic_bezier_cusps(curve, self.tolerance);
        let mut t0 = 0.0;
        let eps = 0.0001;
        for cusp in cusps {
            if cusp - eps > t0 {
                let sub_curve = curve.split_range(t0..(cusp - eps));
                sub_curve.for_each_quadratic_bezier(self.tolerance, &mut |quad| {
                    self.quadratic_bezier_to(quad.ctrl, quad.to, outputs);
                });
            }
            t0 = cusp + eps;
        }

        if t0 < 1.0 {
            let sub_curve = curve.split_range(t0..1.0);
            sub_curve.for_each_quadratic_bezier(self.tolerance, &mut |quad| {
                self.quadratic_bezier_to(quad.ctrl, quad.to, outputs)
            });
        }
    }

    fn step_without_join(&mut self, segments: &[Seg; N], outputs: &mut [&mut dyn PathBuilder; N]) {
        debug_assert!(!self.is_first_segment);
        self.num_steps += 1;
        for ((side, segment), output) in self.sides.iter_mut().zip(segments).zip(outputs.iter_mut()) {
            add_segment(&side.prev, *output);
            side.prev = *segment;
        }
    }

    fn step(
        &mut self,
        segments: &mut [Seg; N],
        unit_tangent: Vector,
        simplify_inner_joins: bool,
        outputs: &mut [&mut dyn PathBuilder; N],
    ) {
        self.num_steps += 1;

        if self.is_first_segment {
            self.first_tangent = unit_tangent;
            self.is_first_segment = false;
            for ((side, segment), output) in self.sides.iter_mut().zip(segments).zip(outputs.iter_mut()) {
                output.begin(segment.from(), NO_ATTRIBUTES);
                side.prev = *segment;
            }
            return;
        }

        // Whether a join is inner or outer only depends on the sign of the
        // offset of each side.
        let t01 = unit_tangent - self.prev_tangent;
        let n01 = normal(unit_tangent) + normal(self.prev_tangent);
        let turn = t01.dot(n01);
        // When the next segment goes back along the previous one, the offset
        // segments end on opposite sides of the stroke on both sides, so both
        // sides get an outer join to wrap around the tip.
        let reversal = self.prev_tangent.dot(unit_tangent) < 0.0
            && self.prev_tangent.cross(unit_tangent).abs() < 0.005;

        for i in 0..N {
            let output = &mut *outputs[i];
            let segment = &mut segments[i];
            let side = &mut self.sides[i];
            let no_join =
                (side.prev.to() - segment.from()).square_length() < self.tolerance * self.tolerance;
            // We are a bit conserative about inner joins when the dot product is close to zero.
            let inner_join = !reversal && turn * side.offset.signum() > -0.01;

            if no_join {
                add_segment(&side.prev, output);
            } else if inner_join {
                Self::inner_join(side, segment, simplify_inner_joins, output);
            } else {
                // If the previous segment is a line, the join adds its end.
                if let Seg::Quadratic(curve) = side.prev {
                    output.quadratic_bezier_to(curve.ctrl, curve.to, NO_ATTRIBUTES);
                }
                (self.join)(
                    output,
                    self.prev_endpoint,
                    self.prev_tangent,
                    unit_tangent,
                    side.offset,
                    self.miter_limit,
                    self.tolerance,
                    side.prev.is_line(),
                    segment.is_line(),
                );
            }

            side.prev = *segment;
        }
    }

    // Inner joins are expected to be inside the shape so we simplify them into a
    // single line segment and even try to prevent the folding that happens where
    // the two offset segments intersect in the simple cases.
    fn inner_join(side: &mut Side, next: &mut Seg, simplify: bool, output: &mut dyn PathBuilder) {
        // If the two segments are lines unfold the join.
        let mut simplified = false;
        if simplify {
            if let (Seg::Line(l0), Seg::Line(l1)) = (&mut side.prev, &mut *next) {
                if let Some(intersection) = l0.intersection(l1) {
                    l0.to = intersection;
                    l1.from = intersection;
                    simplified = true;
                }
            }
        }

        add_segment(&side.prev, output);

        if !simplified {
            output.line_to(next.from(), NO_ATTRIBUTES);
        }
    }
}

fn add_segment(segment: &Seg, output: &mut dyn PathBuilder) {
    match segment {
        Seg::Line(line) => {
            output.line_to(line.to, NO_ATTRIBUTES);
        }
        Seg::Quadratic(curve) => {
            output.quadratic_bezier_to(curve.ctrl, curve.to, NO_ATTRIBUTES);
        }
    }
}

pub struct OffsetBuilder<'l> {
    output: &'l mut dyn PathBuilder,
    offsetter: Offsetter<1>,
}

impl<'l> OffsetBuilder<'l> {
    pub fn new(output: &'l mut dyn PathBuilder, options: &OffsetOptions) -> Self {
        OffsetBuilder {
            output,
            offsetter: Offsetter::new(
                [options.offset],
                options.join,
                options.miter_limit,
                options.tolerance,
                options.simplify_inner_joins,
            ),
        }
    }

    pub fn begin(&mut self, at: Point) {
        self.offsetter.begin(at);
    }

    pub fn end(&mut self, close: bool) {
        if let Some(closed) = self.offsetter.end(close, &mut [&mut *self.output]) {
            self.output.end(closed);
        }
    }

    pub fn line_to(&mut self, to: Point) {
        self.offsetter.line_to(to, &mut [&mut *self.output]);
    }

    pub fn quadratic_bezier_to(&mut self, ctrl: Point, to: Point) {
        self.offsetter.quadratic_bezier_to(ctrl, to, &mut [&mut *self.output]);
    }

    pub fn cubic_bezier_to(&mut self, ctrl1: Point, ctrl2: Point, to: Point) {
        self.offsetter.cubic_bezier_to(ctrl1, ctrl2, to, &mut [&mut *self.output]);
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

#[derive(Copy, Clone)]
enum RecordedSegment {
    Line { to: Point },
    Quadratic { ctrl: Point, to: Point },
    Cubic { ctrl1: Point, ctrl2: Point, to: Point },
}

impl RecordedSegment {
    fn to(&self) -> Point {
        match *self {
            RecordedSegment::Line { to }
            | RecordedSegment::Quadratic { to, .. }
            | RecordedSegment::Cubic { to, .. } => to,
        }
    }
}

/// Records a sub-path, to add it to another path in reverse order.
#[derive(Default)]
struct SegmentBuffer {
    start: Point,
    segments: Vec<RecordedSegment>,
}

impl SegmentBuffer {
    fn last_point(&self) -> Point {
        self.segments.last().map_or(self.start, RecordedSegment::to)
    }

    /// Adds the segments to `output` in reverse order, starting from the
    /// current position of `output`.
    fn add_reversed_to(&self, output: &mut dyn PathBuilder) {
        // Each segment, reversed, ends where the previous one starts.
        for (i, segment) in self.segments.iter().enumerate().rev() {
            let to = match i {
                0 => self.start,
                _ => self.segments[i - 1].to(),
            };
            match *segment {
                RecordedSegment::Line { .. } => {
                    output.line_to(to, NO_ATTRIBUTES);
                }
                RecordedSegment::Quadratic { ctrl, .. } => {
                    output.quadratic_bezier_to(ctrl, to, NO_ATTRIBUTES);
                }
                RecordedSegment::Cubic { ctrl1, ctrl2, .. } => {
                    output.cubic_bezier_to(ctrl2, ctrl1, to, NO_ATTRIBUTES);
                }
            }
        }
    }
}

impl PathBuilder for SegmentBuffer {
    fn num_attributes(&self) -> usize {
        0
    }

    fn begin(&mut self, at: Point, _: Attributes) -> EndpointId {
        self.start = at;
        self.segments.clear();
        EndpointId::INVALID
    }

    fn end(&mut self, _close: bool) {}

    fn line_to(&mut self, to: Point, _: Attributes) -> EndpointId {
        self.segments.push(RecordedSegment::Line { to });
        EndpointId::INVALID
    }

    fn quadratic_bezier_to(&mut self, ctrl: Point, to: Point, _: Attributes) -> EndpointId {
        self.segments.push(RecordedSegment::Quadratic { ctrl, to });
        EndpointId::INVALID
    }

    fn cubic_bezier_to(&mut self, ctrl1: Point, ctrl2: Point, to: Point, _: Attributes) -> EndpointId {
        self.segments.push(RecordedSegment::Cubic { ctrl1, ctrl2, to });
        EndpointId::INVALID
    }
}

pub struct StrokeToFillBuilder<'l> {
    output: &'l mut dyn PathBuilder,
    /// The second side of the stroke, in the direction of the source path. It
    /// is added to the output in reverse order at the end of each sub-path.
    opposite: SegmentBuffer,
    offsetter: Offsetter<2>,
    offsets: (f32, f32),
    start_cap: CapBuilder,
    end_cap: CapBuilder,
    add_empty_caps: bool,
}

impl<'l> StrokeToFillBuilder<'l> {
    pub fn new(output: &'l mut dyn PathBuilder, options: &StrokeOptions) -> Self {
        let mut offsetter = Offsetter::new(
            [options.offsets.0, options.offsets.1],
            options.line_join,
            options.miter_limit,
            options.tolerance,
            true,
        );
        // Open sub-paths get caps even if they end at their start point.
        offsetter.close_if_end_at_start = false;

        StrokeToFillBuilder {
            output,
            opposite: SegmentBuffer::default(),
            offsetter,
            offsets: options.offsets,
            start_cap: get_cap_fn(options.start_cap),
            end_cap: get_cap_fn(options.end_cap),
            add_empty_caps: options.add_empty_caps,
        }
    }

    pub fn begin(&mut self, at: Point) {
        self.offsetter.begin(at);
    }

    pub fn line_to(&mut self, to: Point) {
        let mut outputs: [&mut dyn PathBuilder; 2] = [&mut *self.output, &mut self.opposite];
        self.offsetter.line_to(to, &mut outputs);
    }

    pub fn quadratic_bezier_to(&mut self, ctrl: Point, to: Point) {
        let mut outputs: [&mut dyn PathBuilder; 2] = [&mut *self.output, &mut self.opposite];
        self.offsetter.quadratic_bezier_to(ctrl, to, &mut outputs);
    }

    pub fn cubic_bezier_to(&mut self, ctrl1: Point, ctrl2: Point, to: Point) {
        let mut outputs: [&mut dyn PathBuilder; 2] = [&mut *self.output, &mut self.opposite];
        self.offsetter.cubic_bezier_to(ctrl1, ctrl2, to, &mut outputs);
    }

    pub fn end(&mut self, close: bool) {
        let mut outputs: [&mut dyn PathBuilder; 2] = [&mut *self.output, &mut self.opposite];
        let Some(closed) = self.offsetter.end(close, &mut outputs) else {
            if self.add_empty_caps {
                self.add_empty_cap();
            }
            return;
        };

        if closed {
            // Each side is a separate contour.
            self.output.end(true);
            self.output.begin(self.opposite.last_point(), NO_ATTRIBUTES);
            self.opposite.add_reversed_to(&mut *self.output);
            self.output.end(true);
        } else {
            // A single contour that goes along one side, around the end cap, back
            // along the other side and around the start cap.
            self.add_end_cap();
            self.opposite.add_reversed_to(&mut *self.output);
            self.add_start_cap();
            self.output.end(true);
        }
    }

    fn add_start_cap(&mut self) {
        let endpoint = self.offsetter.first_endpoint;
        let tangent = -self.offsetter.first_tangent;
        let n = normal(tangent);
        let width = (self.offsets.0 - self.offsets.1).abs() * 0.5;
        let pivot_offset = (self.offsets.0 + self.offsets.1) * 0.5;
        let pivot = endpoint - n * pivot_offset;
        let end = endpoint - n * self.offsets.0;
        let sign = (self.offsets.0 - self.offsets.1).signum();
        (self.start_cap)(
            &mut *self.output,
            pivot,
            tangent,
            end,
            width,
            sign,
            self.offsetter.tolerance,
        );
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
            &mut *self.output,
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
        let output = &mut *self.output;
        output.begin(p0, NO_ATTRIBUTES);
        (self.end_cap)(output, pivot, tangent, p1, width, 1.0, tolerance);
        (self.start_cap)(output, pivot, -tangent, p0, width, 1.0, tolerance);
        output.end(true);
    }
}

enum QuadType {
    Curve,
    Linear,
    /// The curve lies on a line and changes direction at the given parameter.
    Degenerate(f32),
}

fn classify_quadratic_bezier(curve: &QuadraticBezierSegment<f32>, tolerance: f32) -> QuadType {
    if !curve.is_linear(tolerance) {
        return QuadType::Curve;
    }

    // The derivative is 2 (p + t q). On a line, it vanishes where the curve
    // changes direction. This is where it is the shortest, which also gives the
    // turning point of curves that are only close to a line.
    let p = curve.ctrl - curve.from;
    let q = curve.from.to_vector() - curve.ctrl.to_vector() * 2.0 + curve.to.to_vector();
    // NaN if q is zero, which means that the curve goes straight from start to end.
    let t = -p.dot(q) / q.square_length();
    if t > 0.0 && t < 1.0 {
        return QuadType::Degenerate(t);
    }

    QuadType::Linear
}

/// Direction of the line that a cubic bézier curve lies on, if it does. The
/// longest vector from the start to another point still gives it when the
/// endpoints are equal.
fn cubic_bezier_direction(curve: &CubicBezierSegment<f32>) -> Vector {
    [
        curve.ctrl1 - curve.from,
        curve.ctrl2 - curve.from,
        curve.to - curve.from,
    ]
    .into_iter()
    .max_by(|a, b| a.square_length().total_cmp(&b.square_length()))
    .unwrap()
}

fn cubic_bezier_is_linear(curve: &CubicBezierSegment<f32>, tolerance: f32) -> bool {
    if curve.from != curve.to {
        return curve.is_linear(tolerance);
    }

    // CubicBezierSegment::is_linear measures the distance to the line that goes
    // through the endpoints, which is not defined when they are equal.
    let line = Line {
        point: curve.from,
        vector: cubic_bezier_direction(curve),
    };
    if line.vector == Vector::zero() {
        return true;
    }

    let tolerance2 = tolerance * tolerance;
    line.square_distance_to_point(curve.ctrl1) <= tolerance2
        && line.square_distance_to_point(curve.ctrl2) <= tolerance2
}

/// Parameters, in increasing order, at which a cubic bézier curve that lies on
/// a line changes direction.
fn cubic_bezier_turnarounds(curve: &CubicBezierSegment<f32>) -> ArrayVec<f32, 3> {
    let mut turnarounds = ArrayVec::new();

    let direction = cubic_bezier_direction(curve);

    // The derivative is 3 (a + 2 b t + c t²). Find where its projection on the
    // line vanishes.
    let a = curve.ctrl1 - curve.from;
    let b = curve.ctrl2.to_vector() - curve.ctrl1.to_vector() * 2.0 + curve.from.to_vector();
    let c = curve.to.to_vector() + (curve.ctrl1 - curve.ctrl2) * 3.0 - curve.from.to_vector();
    let roots = cubic_polynomial_roots(0.0, c.dot(direction), 2.0 * b.dot(direction), a.dot(direction));
    for t in roots {
        if t > 0.0 && t < 1.0 {
            turnarounds.push(t);
        }
    }
    turnarounds.sort_by(|a, b| a.total_cmp(b));

    turnarounds
}

/// Parameters, in increasing order, at which a cubic bézier curve has a cusp,
/// or turns so tightly that it is indistinguishable from one at this tolerance.
fn find_cubic_bezier_cusps(curve: &CubicBezierSegment<f32>, tolerance: f32) -> ArrayVec<f32, 3> {
    let mut cusps = ArrayVec::new();

    // F' = 3 (a + 2 b t + c t²) and F'' = 6 (b + c t).
    let a = curve.ctrl1 - curve.from;
    let b = curve.ctrl2.to_vector() - curve.ctrl1.to_vector() * 2.0 + curve.from.to_vector();
    let c = curve.to.to_vector() + (curve.ctrl1 - curve.ctrl2) * 3.0 - curve.from.to_vector();

    // Rule out cusps without solving the cubic polynomial below, which is
    // comparatively expensive. F' is a quadratic bézier curve with control
    // points h0, h1 and h2, and F'' = 2 ((1 - t) (h1 - h0) + t (h2 - h1)). If
    // the control points of F' are all on the positive side of a direction u,
    // so is F', and |F'| >= min(h.u) / |u|. |F''| is at most twice the longest
    // of h1 - h0 and h2 - h1. Together, they bound the radius of curvature at
    // the extrema of the speed from below.
    let h0 = curve.ctrl1 - curve.from;
    let h1 = curve.ctrl2 - curve.ctrl1;
    let h2 = curve.to - curve.ctrl2;
    let u = curve.to - curve.from;
    let min_speed = h0.dot(u).min(h1.dot(u)).min(h2.dot(u));
    if min_speed > 0.0 {
        let max_d2 = (h1 - h0).square_length().max((h2 - h1).square_length()).sqrt() * 6.0;
        // Both sides are multiplied by |u|² and divided by 9.
        if min_speed * min_speed > tolerance * max_d2 * u.square_length() / 9.0 {
            return cusps;
        }
    }

    // The speed |F'| has its extrema where F'.F'' = 0, and cusps are among them.
    // F'.F'' / 18 = c.c t³ + 3 b.c t² + (2 b.b + a.c) t + a.b
    let p3 = c.dot(c);
    let p2 = 3.0 * b.dot(c);
    let p1 = 2.0 * b.dot(b) + a.dot(c);
    let p0 = a.dot(b);
    // When a control point is on an endpoint, the speed vanishes there and the
    // polynomial has a root at that end. Factoring it out leaves a quadratic
    // polynomial, which is cheaper to solve. It is a common case in SVG files.
    let roots = if h0 == Vector::zero() {
        cubic_polynomial_roots(0.0, p3, p2, p1)
    } else if h2 == Vector::zero() {
        cubic_polynomial_roots(0.0, p3, p3 + p2, p3 + p2 + p1)
    } else {
        cubic_polynomial_roots(p3, p2, p1, p0)
    };
    for t in roots {
        if !(t > 0.0 && t < 1.0) {
            continue;
        }

        // F' and F'' are orthogonal at the extrema of the speed, so the radius
        // of curvature there is |F'|² / |F''|.
        let d1 = (a + b * (2.0 * t) + c * (t * t)) * 3.0;
        let d2 = (b + c * t) * 6.0;
        if d1.square_length() <= tolerance * d2.length() {
            cusps.push(t);
        }
    }

    cusps.sort_by(|a, b| a.total_cmp(b));

    cusps
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

    #[test]
    fn sub_paths_smaller_than_the_tolerance() {
        // A closed polygon about 0.18 units wide (from the paris-30k.svg test file).
        let polygon = [
            point(0.396, 0.975),
            point(0.396, 0.916),
            point(0.426, 0.878),
            point(0.465, 0.857),
            point(0.504, 0.857),
            point(0.548, 0.881),
            point(0.572, 0.935),
            point(0.568, 0.985),
            point(0.542, 1.014),
            point(0.504, 1.034),
            point(0.46, 1.034),
            point(0.416, 1.009),
        ];
        let directions = [
            Vector::new(1.0, 0.0),
            Vector::new(-1.0, 0.0),
            Vector::new(0.0, 1.0),
            Vector::new(0.0, -1.0),
        ];
        let polygon_extents = directions.map(|d| {
            polygon.iter().map(|p| p.to_vector().dot(d)).fold(f32::MIN, f32::max)
        });

        for join in [LineJoin::Round, LineJoin::Bevel, LineJoin::Miter] {
            for close in [true, false] {
                let options = StrokeOptions::line_with(1.0)
                    .with_line_join(join)
                    .with_line_cap(LineCap::Round)
                    .with_tolerance(0.25);
                let path = stroke_polyline(&polygon, close, &options);
                for (d, polygon_extent) in directions.iter().zip(polygon_extents) {
                    // The stroke extends at least the half width beyond the polygon,
                    // and not much more (the miter limit is 4).
                    let e = extent(&path, *d) - polygon_extent;
                    assert!(e > 0.5 - 0.25 && e < 2.0, "{join:?} close:{close} {d:?}: {e}");
                }
            }
        }

        // Sub-paths made of a single point are still empty without add_empty_caps.
        let options = StrokeOptions::line_with(1.0).with_line_join(LineJoin::Round);
        let path = stroke_polyline(&[point(1.0, 1.0), point(1.0, 1.0), point(1.0, 1.0)], true, &options);
        assert_eq!(path.iter().count(), 0);
    }

    #[test]
    fn short_first_segment_of_closed_sub_path() {
        // A thin closed shape (from the paris-30k.svg test file) whose first segment
        // is short enough to be discarded. Both sides of the stroke must see the
        // same simplified sub-path, otherwise neither side gets an outer join at
        // the top.
        let mut builder = Path::builder();
        builder.begin(point(0.087, 10.391));
        builder.line_to(point(0.3, 10.391));
        builder.cubic_bezier_to(point(0.234, 10.257), point(0.196, 10.106), point(0.196, 9.947));
        builder.line_to(point(0.196, 7.799));
        builder.cubic_bezier_to(point(0.196, 7.52), point(0.31, 7.268), point(0.495, 7.086));
        builder.cubic_bezier_to(point(0.31, 6.904), point(0.196, 6.652), point(0.196, 6.373));
        builder.line_to(point(0.196, 4.635));
        builder.cubic_bezier_to(point(0.196, 4.476), point(0.234, 4.324), point(0.3, 4.191));
        builder.line_to(point(0.087, 4.191));
        builder.end(true);
        let source = builder.build();

        let tolerance = 0.5;
        for width in [1.0, 25.0] {
            let options = StrokeOptions::line_with(width)
                .with_line_join(LineJoin::Round)
                .with_tolerance(tolerance);
            let path = Stroker::new().stroke_to_fill(source.iter(), &options);
            let top = extent(&path, Vector::new(0.0, 1.0));
            let bottom = -extent(&path, Vector::new(0.0, -1.0));
            let expected_top = 10.391 + width * 0.5;
            let expected_bottom = 4.191 - width * 0.5;
            assert!((top - expected_top).abs() < tolerance, "width {width}: top {top}");
            assert!((bottom - expected_bottom).abs() < tolerance, "width {width}: bottom {bottom}");
        }
    }

    #[test]
    fn short_closing_segment() {
        // A closed quadrilateral about 0.35 units wide (from the paris-30k.svg test
        // file). Some of its segments, including the closing one, are short enough
        // to be discarded, and the stroke is much wider than the shape.
        let polygon = [
            point(0.779, 0.176),
            point(1.091, 0.084),
            point(1.155, 0.289),
            point(0.878, 0.381),
        ];
        let tolerance = 0.5;
        let half_width = 12.5;
        let options = StrokeOptions::line_with(half_width * 2.0)
            .with_line_join(LineJoin::Round)
            .with_tolerance(tolerance);
        let path = stroke_polyline(&polygon, true, &options);

        // The outer side of the stroke is a disk around the shape.
        for i in 0..16 {
            let a = i as f32 * std::f32::consts::TAU / 16.0;
            let d = Vector::new(a.cos(), a.sin());
            let ideal = polygon.iter().map(|p| p.to_vector().dot(d)).fold(f32::MIN, f32::max)
                + half_width;
            let e = extent(&path, d);
            assert!((e - ideal).abs() < tolerance, "{d:?}: {e} (ideal {ideal})");
        }
    }

    #[test]
    fn collinear_curves_that_change_direction() {
        let hw = 5.0;
        let options = StrokeOptions::line_with(hw * 2.0)
            .with_line_join(LineJoin::Round)
            .with_line_cap(LineCap::Butt);
        let quad = |ctrl: (f32, f32), to: (f32, f32)| {
            QuadraticBezierSegment { from: point(0.0, 0.0), ctrl: point(ctrl.0, ctrl.1), to: point(to.0, to.1) }
        };
        let cubic = |ctrl1: (f32, f32), ctrl2: (f32, f32), to: (f32, f32)| CubicBezierSegment {
            from: point(0.0, 0.0),
            ctrl1: point(ctrl1.0, ctrl1.1),
            ctrl2: point(ctrl2.0, ctrl2.1),
            to: point(to.0, to.1),
        };
        let quads = [
            // Goes out to x = 50 and back to the start.
            quad((100.0, 0.0), (0.0, 0.0)),
            // Goes past its end to x = 112.5.
            quad((150.0, 0.0), (100.0, 0.0)),
            // Goes back past its start.
            quad((-100.0, 0.0), (10.0, 0.0)),
        ];
        let cubics = [
            // Goes past both ends.
            cubic((100.0, 0.0), (-100.0, 0.0), (10.0, 0.0)),
            cubic((0.0, 0.0), (100.0, 0.0), (0.0, 0.0)),
        ];

        let check = |path: Path, sample: &dyn Fn(f32) -> Point| {
            // The round joins at the turning points extend the stroke by the
            // half width.
            let n = 100_000;
            let (min, max) = (0..=n)
                .map(|i| sample(i as f32 / n as f32).x)
                .fold((f32::MAX, f32::MIN), |(min, max), x| (min.min(x), max.max(x)));
            let start = sample(0.0).x;
            let end = sample(1.0).x;
            let expected_max = if max > start.max(end) { max + hw } else { max };
            let expected_min = if min < start.min(end) { min - hw } else { min };

            let max_x = extent(&path, Vector::new(1.0, 0.0));
            let min_x = -extent(&path, Vector::new(-1.0, 0.0));
            assert!((max_x - expected_max).abs() < 0.01, "{max_x} != {expected_max}");
            assert!((min_x - expected_min).abs() < 0.01, "{min_x} != {expected_min}");
        };

        for curve in quads {
            let mut builder = Path::builder();
            {
                let mut stroker = StrokeToFillBuilder::new(&mut builder, &options);
                stroker.begin(curve.from);
                stroker.quadratic_bezier_to(curve.ctrl, curve.to);
                stroker.end(false);
            }
            check(builder.build(), &|t| curve.sample(t));
        }

        for curve in cubics {
            let mut builder = Path::builder();
            {
                let mut stroker = StrokeToFillBuilder::new(&mut builder, &options);
                stroker.begin(curve.from);
                stroker.cubic_bezier_to(curve.ctrl1, curve.ctrl2, curve.to);
                stroker.end(false);
            }
            check(builder.build(), &|t| curve.sample(t));
        }
    }

    /// Winding number of the outline around a point, flattening curves.
    fn winding_number(path: &Path, p: Point, tolerance: f32) -> i32 {
        let mut winding = 0;
        let mut add_edge = |from: Point, to: Point| {
            let side = (to - from).cross(p - from);
            if from.y <= p.y && to.y > p.y && side > 0.0 {
                winding += 1;
            } else if from.y > p.y && to.y <= p.y && side < 0.0 {
                winding -= 1;
            }
        };
        for event in path.iter() {
            match event {
                PathEvent::Begin { .. } => {}
                PathEvent::Line { from, to } => add_edge(from, to),
                PathEvent::Quadratic { from, ctrl, to } => QuadraticBezierSegment { from, ctrl, to }
                    .for_each_flattened(tolerance, &mut |s| add_edge(s.from, s.to)),
                PathEvent::Cubic { from, ctrl1, ctrl2, to } => CubicBezierSegment { from, ctrl1, ctrl2, to }
                    .for_each_flattened(tolerance, &mut |s| add_edge(s.from, s.to)),
                PathEvent::End { last, first, .. } => add_edge(last, first),
            }
        }
        winding
    }

    #[test]
    fn closing_join_after_cubic() {
        // A closed shape made of two cubic bézier curves (from the tiger.svg test
        // file). On the reversed side of the stroke, the quadratic bézier curves
        // that approximate the last cubic bézier curve do not end exactly at the
        // start of the sub-path. Both sides must still agree about the closing
        // join.
        let start = point(246.51247, 807.6567);
        let mut builder = Path::builder();
        builder.begin(start);
        builder.cubic_bezier_to(start, point(266.99396, 768.10626), point(228.50287, 813.6599));
        builder.cubic_bezier_to(point(228.50287, 813.6599), point(247.92499, 796.35657), start);
        builder.end(true);
        let source = builder.build();

        let half_width = 12.5;
        for tolerance in [0.25, 0.05, 0.01] {
            let options = StrokeOptions::line_with(half_width * 2.0)
                .with_line_join(LineJoin::Round)
                .with_tolerance(tolerance);
            let path = Stroker::new().stroke_to_fill(source.iter(), &options);

            // The start point is on the source path, so the stroke covers a disk
            // around it.
            for i in 0..32 {
                let a = i as f32 * std::f32::consts::TAU / 32.0;
                let p = start + Vector::new(a.cos(), a.sin()) * (half_width - 2.0 * tolerance);
                let winding = winding_number(&path, p, tolerance * 0.1);
                assert!(winding != 0, "tolerance {tolerance}: {p:?} is not covered");
            }
        }
    }

    /// Extents of the outline in 32 directions.
    fn extents(path: &Path) -> Vec<f32> {
        (0..32)
            .map(|i| {
                let a = i as f32 * std::f32::consts::TAU / 32.0;
                extent(path, Vector::new(a.cos(), a.sin()))
            })
            .collect()
    }

    #[test]
    fn curve_after_discarded_segments() {
        // Segments shorter than the tolerance that go back and sideways from the
        // end of a line, followed by a curve that continues in the direction of
        // the line. The curve is displaced to start at the end of the line, and
        // the stroke is the same as without the short segments.
        let tiny = [point(99.95, 0.05), point(99.92, 0.08)];
        let from = tiny[1];
        let curves: [&dyn Fn(&mut core::path::Builder, Point); 2] = [
            &|builder, from| {
                // The tangent at the start is given by the second control point.
                builder.cubic_bezier_to(from, point(150.0, from.y), point(150.0, 50.0));
            },
            &|builder, from| {
                builder.quadratic_bezier_to(from + Vector::new(0.05, 0.0), point(130.0, 10.0));
            },
        ];

        for curve in curves {
            for join in [LineJoin::Miter, LineJoin::Bevel, LineJoin::Round] {
                let options = StrokeOptions::line_with(10.0)
                    .with_line_join(join)
                    .with_line_cap(LineCap::Butt)
                    .with_tolerance(0.25);
                let stroke = |tiny: &[Point], from: Point| {
                    let mut builder = Path::builder();
                    builder.begin(point(0.0, 0.0));
                    builder.line_to(point(100.0, 0.0));
                    for p in tiny {
                        builder.line_to(*p);
                    }
                    curve(&mut builder, from);
                    builder.end(false);
                    Stroker::new().stroke_to_fill(builder.build().iter(), &options)
                };

                let expected = extents(&stroke(&[], point(100.0, 0.0)));
                let actual = extents(&stroke(&tiny, from));
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert!((a - e).abs() < 0.5, "{join:?}: {actual:?} != {expected:?}");
                }
            }
        }
    }

    #[test]
    fn curve_ending_at_its_start_after_discarded_segments() {
        // A quadratic bézier curve that ends where it starts (a line that goes to
        // its turning point and back) cannot be stretched to start at the end of
        // the previous segment, so it is moved as a whole. As the last segment of
        // a closed sub-path, it does not end at the first point, and a line closes
        // the sub-path.
        let mut builder = Path::builder();
        builder.begin(point(0.0, 0.0));
        builder.line_to(point(100.0, 0.0));
        builder.line_to(point(100.0, 100.0));
        builder.line_to(point(0.1, 0.1));
        builder.line_to(point(0.05, 0.12));
        builder.quadratic_bezier_to(point(-30.0, 30.0), point(0.05, 0.12));
        builder.end(true);
        let source = builder.build();

        let half_width = 5.0;
        for join in [LineJoin::Miter, LineJoin::Bevel, LineJoin::Round] {
            let options = StrokeOptions::line_with(half_width * 2.0)
                .with_line_join(join)
                .with_tolerance(0.25);
            let path = Stroker::new().stroke_to_fill(source.iter(), &options);

            // The curve up to near its turning point, the corners and the middle
            // of the sides are covered. The turning point itself is on the edge of
            // the stroke with miter and bevel joins.
            let tip = QuadraticBezierSegment {
                from: point(0.05, 0.12),
                ctrl: point(-30.0, 30.0),
                to: point(0.05, 0.12),
            }
            .sample(0.45);
            for p in [tip, point(0.0, 0.0), point(100.0, 0.0), point(100.0, 100.0), point(50.0, 0.0), point(50.0, 50.0)] {
                for offset in [Vector::new(0.0, 0.0), Vector::new(3.0, 0.0), Vector::new(0.0, -3.0)] {
                    let winding = winding_number(&path, p + offset, 0.01);
                    assert!(winding != 0, "{join:?}: {:?} is not covered", p + offset);
                }
            }
        }
    }

    fn cubic(points: [(f32, f32); 4]) -> CubicBezierSegment<f32> {
        let [from, ctrl1, ctrl2, to] = points.map(|(x, y)| point(x, y));
        CubicBezierSegment { from, ctrl1, ctrl2, to }
    }

    #[test]
    fn cubic_bezier_cusps() {
        let tolerance = 0.25;
        let cusp = cubic([(0.0, 0.0), (100.0, 100.0), (0.0, 100.0), (100.0, 0.0)]);
        let cusps = find_cubic_bezier_cusps(&cusp, tolerance);
        assert_eq!(cusps.len(), 1);
        assert!((cusps[0] - 0.5).abs() < 1e-5, "{cusps:?}");

        // The speed of these curves has extrema, but no cusps.
        let arch = cubic([(0.0, 0.0), (0.0, 100.0), (100.0, 100.0), (100.0, 0.0)]);
        let s_curve = cubic([(0.0, 0.0), (100.0, 0.0), (0.0, 100.0), (100.0, 100.0)]);
        let with_loop = cubic([(0.0, 0.0), (150.0, 100.0), (-50.0, 100.0), (100.0, 0.0)]);
        // The speed of these curves vanishes at one end, where a control point is
        // on the endpoint. It is not a cusp within the curve.
        let flat_start = cubic([(0.0, 0.0), (0.0, 0.0), (100.0, 100.0), (100.0, 0.0)]);
        let flat_end = cubic([(0.0, 0.0), (0.0, 100.0), (30.0, 10.0), (30.0, 10.0)]);
        for curve in [arch, s_curve, with_loop, flat_start, flat_end] {
            assert!(find_cubic_bezier_cusps(&curve, tolerance).is_empty(), "{curve:?}");
        }

        // Moving a control point of the cusp rounds its tip, with a radius of
        // curvature of 0.093 and 0.37 respectively.
        let tight = cubic([(0.0, 0.0), (100.0, 100.0), (10.0, 100.0), (100.0, 0.0)]);
        let round = cubic([(0.0, 0.0), (100.0, 100.0), (20.0, 100.0), (100.0, 0.0)]);
        assert_eq!(find_cubic_bezier_cusps(&tight, tolerance).len(), 1);
        assert!(find_cubic_bezier_cusps(&tight, 0.05).is_empty());
        assert!(find_cubic_bezier_cusps(&round, tolerance).is_empty());

        // The stroke gets a round join at the cusp, whose tip is at y = 75.
        let options = StrokeOptions::line_with(20.0)
            .with_line_join(LineJoin::Round)
            .with_line_cap(LineCap::Butt);
        let mut builder = Path::builder();
        {
            let mut stroker = StrokeToFillBuilder::new(&mut builder, &options);
            stroker.begin(cusp.from);
            stroker.cubic_bezier_to(cusp.ctrl1, cusp.ctrl2, cusp.to);
            stroker.end(false);
        }
        let max_y = extent(&builder.build(), Vector::new(0.0, 1.0));
        assert!((max_y - 85.0).abs() < 0.01, "{max_y}");
    }
}
