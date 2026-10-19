//! A port of Skia's stroker.
//!
//! Sources (skia main branch):
//! - `src/core/SkStroke.cpp` (`SkPathStroker`, `SkStroke::strokePath`),
//! - `src/core/SkStrokerPriv.cpp` (caps and joins),
//! - the parts of `src/core/SkGeometry.cpp` that the stroker depends on.
//!
//! The structure and the numerical heuristics follow Skia's code closely so that
//! the output can serve as a reference. The main points of interest:
//! - Each side of the stroke is built in its own contour (`outer` and `inner`).
//!   The inner contour is reversed and appended to the outer one at the end of
//!   the sub-path.
//! - Curves are offset by approximating each side with quadratic bézier curves:
//!   the control point is the intersection of the offset tangents at the ends of
//!   a t-range. The approximation is checked by casting a ray from the middle of
//!   the source curve along its normal, and the t-range is split in two if the
//!   intersection is not within tolerance.
//! - Cubic bézier curves are split at their inflections before offsetting. A
//!   circle is added at cusps.
//! - Inner joins go through the pivot point and the output is meant to be
//!   filled with the non-zero fill rule.
//!
//! Differences with Skia:
//! - Skia emits conics for round caps and joins. Here they are converted to
//!   quadratic bézier curves with `SkConic::computeQuadPOW2`, which is what Skia
//!   does when filling the stroked path. Skia's scan converters use a tolerance
//!   of 0.25 in device space; here the stroke tolerance is used.
//! - Skia strokes closed rectangles with a dedicated fast path (`strokeRect`).
//!   It is not ported: rectangles go through the generic code.
//! - Skia's tolerance is expressed with a resolution scale. `res_scale = 1.0`
//!   corresponds to a tolerance of 0.25, so the resolution scale is derived from
//!   the tolerance as `0.25 / tolerance`.
//! - Options that Skia does not support are mapped to the closest supported
//!   option: `MiterClip` joins are treated as `Miter`, `Quad` joins as `Round`,
//!   and the triangle and inverted caps as `Butt`. Skia uses the same cap at
//!   both ends of a sub-path: `start_cap` is used and `end_cap` is ignored.
//!   Skia only adds caps to empty sub-paths if they are closed. With
//!   `add_empty_caps`, empty open sub-paths get caps too. Asymmetric offsets are not
//!   supported: the stroke is centered and its width is the distance between
//!   the two offsets.

use crate::{LineCap, LineJoin, StrokeOptions};
use core::geom::arrayvec::ArrayVec;
use core::path::{builder::PathBuilder, Path, PathEvent, NO_ATTRIBUTES};
use core::units::{Point, Vector};

/// `SK_ScalarNearlyZero`.
const NEARLY_ZERO: f32 = 1.0 / 4096.0;

const ROOT_2_OVER_2: f32 = std::f32::consts::FRAC_1_SQRT_2;

// Recursion limits, indexed by curve type.
//
// Quads with extreme widths (e.g. (0,1) (1,6) (0,3) width=5e7) recurse to point of failure.
// Largest seen for normal cubics: 5, 26. Largest seen for normal quads: 11.
// The limits are chosen to be the largest depth that won't timeout Skia's fuzzer.
const TANGENT_RECURSIVE_LIMIT: i32 = 5 * 3;
const CUBIC_RECURSIVE_LIMIT: i32 = 24;
const QUAD_RECURSIVE_LIMIT: i32 = 16;

/// Converts paths into the outline of their stroke, using Skia's algorithm.
///
/// Has the same interface as `crate::Stroker`.
pub struct Stroker {
    segments: Vec<Segment>,
    stroker: Option<PathStroker>,
}

impl Stroker {
    pub fn new() -> Self {
        Stroker {
            segments: Vec::new(),
            stroker: None,
        }
    }

    pub fn stroke_to_fill(
        &mut self,
        path: impl Iterator<Item = PathEvent>,
        options: &StrokeOptions,
    ) -> Path {
        let mut builder = Path::builder();
        self.stroke_to_fill_into(path, options, &mut builder);
        builder.build()
    }

    pub fn stroke_to_fill_into(
        &mut self,
        path: impl Iterator<Item = PathEvent>,
        options: &StrokeOptions,
        output: &mut dyn PathBuilder,
    ) {
        let radius = (options.offsets.0 - options.offsets.1).abs() * 0.5;
        if !(radius > 0.0) {
            return;
        }

        let cap = match options.start_cap {
            LineCap::Round => Cap::Round,
            LineCap::Square => Cap::Square,
            _ => Cap::Butt,
        };
        let join = match options.line_join {
            LineJoin::Miter | LineJoin::MiterClip => Join::Miter,
            LineJoin::Round | LineJoin::Quad => Join::Round,
            LineJoin::Bevel => Join::Bevel,
        };
        let res_scale = 0.25 / options.tolerance;

        let mut stroker = self.stroker.take().unwrap_or_else(PathStroker::default);
        stroker.init(radius, options.miter_limit, cap, join, res_scale);

        // Skia's stroker looks ahead in the sub-path to decide whether a degenerate
        // line should get caps, so each sub-path is buffered before being stroked.
        let mut segments = std::mem::take(&mut self.segments);
        let mut first = Point::zero();
        // Like in SkStroke::strokePath, this is not reset between sub-paths.
        let mut last_is_line = false;
        for event in path {
            match event {
                PathEvent::Begin { at } => {
                    first = at;
                    segments.clear();
                }
                PathEvent::Line { to, .. } => segments.push(Segment::Line(to)),
                PathEvent::Quadratic { ctrl, to, .. } => segments.push(Segment::Quad(ctrl, to)),
                PathEvent::Cubic {
                    ctrl1, ctrl2, to, ..
                } => segments.push(Segment::Cubic(ctrl1, ctrl2, to)),
                PathEvent::End { last, close, .. } => {
                    stroke_sub_path(
                        &mut stroker,
                        first,
                        last,
                        &mut segments,
                        close,
                        options.add_empty_caps,
                        &mut last_is_line,
                    );
                }
            }
        }
        // SkPathStroker::done.
        stroker.finish_contour(false, last_is_line);
        stroker.outer.emit(options.tolerance, output);
        stroker.outer.clear();

        self.segments = segments;
        self.stroker = Some(stroker);
    }
}

#[derive(Copy, Clone, Debug)]
enum Segment {
    Line(Point),
    Quad(Point, Point),
    Cubic(Point, Point, Point),
}

impl Segment {
    fn is_degenerate(&self, from: Point) -> bool {
        match *self {
            Segment::Line(to) => from == to,
            Segment::Quad(ctrl, to) => from == ctrl && from == to,
            Segment::Cubic(ctrl1, ctrl2, to) => from == ctrl1 && from == ctrl2 && from == to,
        }
    }

    fn to(&self) -> Point {
        match *self {
            Segment::Line(to) | Segment::Quad(_, to) | Segment::Cubic(_, _, to) => to,
        }
    }
}

/// The body of the loop in `SkStroke::strokePath`, for a single sub-path.
///
/// The contour is finished when the next sub-path starts or when the whole
/// path is done, like in Skia.
fn stroke_sub_path(
    stroker: &mut PathStroker,
    first: Point,
    last: Point,
    segments: &mut Vec<Segment>,
    close: bool,
    add_empty_caps: bool,
    last_is_line: &mut bool,
) {
    // SkPath::Iter emits a line back to the start when closing a sub-path.
    if close && last != first {
        segments.push(Segment::Line(first));
    }

    stroker.move_to(first);

    for (idx, segment) in segments.iter().enumerate() {
        match *segment {
            Segment::Line(to) => {
                let has_valid_tangent = || {
                    let mut prev = to;
                    segments[idx + 1..].iter().any(|s| {
                        let degenerate = s.is_degenerate(prev);
                        prev = s.to();
                        !degenerate
                    })
                };
                stroker.line_to(to, Some(&has_valid_tangent));
                *last_is_line = true;
            }
            Segment::Quad(ctrl, to) => {
                stroker.quad_to(ctrl, to);
                *last_is_line = false;
            }
            Segment::Cubic(ctrl1, ctrl2, to) => {
                stroker.cubic_to(ctrl1, ctrl2, to);
                *last_is_line = false;
            }
        }
    }

    if close {
        if stroker.cap != Cap::Butt {
            // If the stroke consists of a moveTo followed by a close, treat it
            // as if it were followed by a zero-length line. Lines without length
            // can have square and round end caps.
            if stroker.has_only_move_to() {
                stroker.line_to(stroker.move_to_pt(), None);
                *last_is_line = true;
                return;
            }
            // If the stroke consists of a moveTo followed by one or more zero-length
            // verbs, then followed by a close, treat is as if it were followed by a
            // zero-length line. Lines without length can have square & round end caps.
            if stroker.is_current_contour_empty() {
                *last_is_line = true;
                return;
            }
        }
        stroker.close(*last_is_line);
    } else if add_empty_caps && stroker.cap != Cap::Butt && stroker.has_only_move_to() {
        // Skia only adds caps to empty sub-paths when they are closed.
        stroker.line_to(stroker.move_to_pt(), None);
        *last_is_line = true;
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Cap {
    Butt,
    Round,
    Square,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Join {
    Miter,
    Round,
    Bevel,
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum Verb {
    Move,
    Line,
    Quad,
    Conic(f32),
    Close,
}

/// A minimal equivalent of `SkPathBuilder`.
#[derive(Default)]
struct PathBuf {
    verbs: Vec<Verb>,
    points: Vec<Point>,
    // Index of the last move_to point, used to implicitly start a new contour after
    // a close, like SkPathBuilder does.
    last_move: usize,
    needs_move: bool,
}

impl PathBuf {
    fn clear(&mut self) {
        self.verbs.clear();
        self.points.clear();
        self.last_move = 0;
        self.needs_move = false;
    }

    fn is_empty(&self) -> bool {
        self.verbs.is_empty()
    }

    fn move_to(&mut self, p: Point) {
        self.last_move = self.points.len();
        self.needs_move = false;
        self.verbs.push(Verb::Move);
        self.points.push(p);
    }

    fn inject_move_to_if_needed(&mut self) {
        if self.needs_move {
            let p = self.points.get(self.last_move).copied().unwrap_or(Point::zero());
            self.move_to(p);
        } else if self.verbs.is_empty() {
            self.move_to(Point::zero());
        }
    }

    fn line_to(&mut self, p: Point) {
        self.inject_move_to_if_needed();
        self.verbs.push(Verb::Line);
        self.points.push(p);
    }

    fn quad_to(&mut self, ctrl: Point, p: Point) {
        self.inject_move_to_if_needed();
        self.verbs.push(Verb::Quad);
        self.points.push(ctrl);
        self.points.push(p);
    }

    fn conic_to(&mut self, ctrl: Point, p: Point, w: f32) {
        self.inject_move_to_if_needed();
        self.verbs.push(Verb::Conic(w));
        self.points.push(ctrl);
        self.points.push(p);
    }

    fn close(&mut self) {
        if let Some(&last) = self.verbs.last() {
            if last != Verb::Close {
                self.verbs.push(Verb::Close);
            }
        }
        self.needs_move = true;
    }

    fn last_point(&self) -> Option<Point> {
        self.points.last().copied()
    }

    fn set_last_point(&mut self, p: Point) {
        match self.points.last_mut() {
            Some(last) => *last = p,
            None => self.move_to(p),
        }
    }

    /// `SkPathBuilder::privateReversePathTo`: appends the segments of `src` (a
    /// single contour) in reverse order, ignoring its move_to.
    fn reverse_path_to(&mut self, src: &PathBuf) {
        if src.verbs.is_empty() {
            return;
        }
        let mut pt_idx = src.points.len() - 1;
        for verb in src.verbs[1..].iter().rev() {
            match *verb {
                Verb::Line => {
                    pt_idx -= 1;
                    self.line_to(src.points[pt_idx]);
                }
                Verb::Quad => {
                    pt_idx -= 2;
                    self.quad_to(src.points[pt_idx + 1], src.points[pt_idx]);
                }
                Verb::Conic(w) => {
                    pt_idx -= 2;
                    self.conic_to(src.points[pt_idx + 1], src.points[pt_idx], w);
                }
                Verb::Move | Verb::Close => {
                    debug_assert!(false, "expected a single contour");
                }
            }
        }
    }

    fn add_path(&mut self, src: &PathBuf) {
        let mut pts = src.points.iter().copied();
        for verb in &src.verbs {
            match *verb {
                Verb::Move => self.move_to(pts.next().unwrap()),
                Verb::Line => self.line_to(pts.next().unwrap()),
                Verb::Quad => {
                    let ctrl = pts.next().unwrap();
                    self.quad_to(ctrl, pts.next().unwrap());
                }
                Verb::Conic(w) => {
                    let ctrl = pts.next().unwrap();
                    self.conic_to(ctrl, pts.next().unwrap(), w);
                }
                Verb::Close => self.close(),
            }
        }
    }

    fn add_circle(&mut self, center: Point, radius: f32) {
        let r = radius;
        let c = center;
        let p = |x: f32, y: f32| Point::new(c.x + x * r, c.y + y * r);
        self.move_to(p(1.0, 0.0));
        self.conic_to(p(1.0, 1.0), p(0.0, 1.0), ROOT_2_OVER_2);
        self.conic_to(p(-1.0, 1.0), p(-1.0, 0.0), ROOT_2_OVER_2);
        self.conic_to(p(-1.0, -1.0), p(0.0, -1.0), ROOT_2_OVER_2);
        self.conic_to(p(1.0, -1.0), p(1.0, 0.0), ROOT_2_OVER_2);
        self.close();
    }

    /// Conics are converted to quadratic bézier curves within `conic_tolerance`.
    fn emit(&self, conic_tolerance: f32, output: &mut dyn PathBuilder) {
        let mut pts = self.points.iter().copied();
        let mut open = false;
        let mut current = Point::zero();
        for verb in &self.verbs {
            match *verb {
                Verb::Move => {
                    if open {
                        output.end(false);
                    }
                    current = pts.next().unwrap();
                    output.begin(current, NO_ATTRIBUTES);
                    open = true;
                }
                Verb::Line => {
                    current = pts.next().unwrap();
                    output.line_to(current, NO_ATTRIBUTES);
                }
                Verb::Quad => {
                    let ctrl = pts.next().unwrap();
                    current = pts.next().unwrap();
                    output.quadratic_bezier_to(ctrl, current, NO_ATTRIBUTES);
                }
                Verb::Conic(w) => {
                    let ctrl = pts.next().unwrap();
                    let to = pts.next().unwrap();
                    let conic = Conic { pts: [current, ctrl, to], w };
                    let pow2 = conic.compute_quad_pow2(conic_tolerance);
                    conic.emit_quads(pow2, output);
                    current = to;
                }
                Verb::Close => {
                    if open {
                        output.end(true);
                        open = false;
                    }
                }
            }
        }
        if open {
            output.end(false);
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct Conic {
    pts: [Point; 3],
    w: f32,
}

impl Conic {
    /// `SkConic::computeQuadPOW2`.
    ///
    /// "High order approximation of conic sections by quadratic splines"
    /// by Michael Floater, 1993.
    fn compute_quad_pow2(&self, tol: f32) -> u32 {
        let finite = self.pts.iter().all(|p| p.x.is_finite() && p.y.is_finite());
        if tol < 0.0 || !tol.is_finite() || !finite || self.w < 0.0 || !self.w.is_finite() {
            return 0;
        }
        let a = self.w - 1.0;
        let k = a / (4.0 * (2.0 + a));
        let x = k * (self.pts[0].x - 2.0 * self.pts[1].x + self.pts[2].x);
        let y = k * (self.pts[0].y - 2.0 * self.pts[1].y + self.pts[2].y);

        let mut error = (x * x + y * y).sqrt();
        let mut pow2 = 0;
        while pow2 < 5 {
            if error <= tol {
                break;
            }
            error *= 0.25;
            pow2 += 1;
        }
        pow2
    }

    /// `SkConic::chop`.
    fn chop(&self) -> [Conic; 2] {
        let scale = 1.0 / (1.0 + self.w);
        let t0 = self.pts[0].to_vector() * scale;
        let t1 = self.pts[1].to_vector() * (self.w * scale);
        let t2 = self.pts[2].to_vector() * scale;
        let p1 = (t0 + t1).to_point();
        let p3 = (t1 + t2).to_point();
        let p2 = (t0 * 0.5 + t1 + t2 * 0.5).to_point();
        let w = (0.5 + self.w * 0.5).sqrt();
        [
            Conic { pts: [self.pts[0], p1, p2], w },
            Conic { pts: [p2, p3, self.pts[2]], w },
        ]
    }

    /// `SkConic::chopIntoQuadsPOW2`, without the adjustments for Skia's scan
    /// converters.
    fn emit_quads(&self, level: u32, output: &mut dyn PathBuilder) {
        if level == 0 {
            output.quadratic_bezier_to(self.pts[1], self.pts[2], NO_ATTRIBUTES);
            return;
        }
        let [a, b] = self.chop();
        a.emit_quads(level - 1, output);
        b.emit_quads(level - 1, output);
    }
}

/// `SkQuadConstruct`: the state of the quad stroke under construction.
#[derive(Copy, Clone, Default)]
struct QuadConstruct {
    /// The stroked quad parallel to the original curve.
    quad: [Point; 3],
    /// Tangent vector at quad[0].
    tangent_start: Vector,
    /// Tangent vector at quad[2].
    tangent_end: Vector,
    /// A segment of the original curve.
    start_t: f32,
    mid_t: f32,
    end_t: f32,
    /// State to share common points across structs.
    start_set: bool,
    end_set: bool,
    /// Set if coincident tangents have opposite directions.
    opposite_tangents: bool,
}

impl QuadConstruct {
    /// Returns false if start and end are too close to have a unique middle.
    fn init(&mut self, start: f32, end: f32) -> bool {
        self.start_t = start;
        self.mid_t = (start + end) * 0.5;
        self.end_t = end;
        self.start_set = false;
        self.end_set = false;
        self.start_t < self.mid_t && self.mid_t < self.end_t
    }

    fn init_with_start(&mut self, parent: &QuadConstruct) -> bool {
        if !self.init(parent.start_t, parent.mid_t) {
            return false;
        }
        self.quad[0] = parent.quad[0];
        self.tangent_start = parent.tangent_start;
        self.start_set = true;
        true
    }

    fn init_with_end(&mut self, parent: &QuadConstruct) -> bool {
        if !self.init(parent.mid_t, parent.end_t) {
            return false;
        }
        self.quad[2] = parent.quad[2];
        self.tangent_end = parent.tangent_end;
        self.end_set = true;
        true
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ResultType {
    /// The caller should split the quad stroke in two.
    Split,
    /// The caller should add a line.
    Degenerate,
    /// The caller should (continue to try to) add a quad stroke.
    Quad,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ReductionType {
    /// All curve points are practically identical.
    Point,
    /// The control point is on the line between the ends.
    Line,
    /// The control point is outside the line between the ends.
    Quad,
    /// The control points are on the line but outside the ends. The number of
    /// reduction points (1 to 3) is stored separately.
    Degenerate,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum IntersectRayType {
    CtrlPt,
    ResultType,
}

/// `SkPathStroker`.
#[derive(Default)]
struct PathStroker {
    radius: f32,
    inv_miter_limit: f32,
    res_scale: f32,
    inv_res_scale: f32,
    inv_res_scale_squared: f32,

    first_normal: Vector,
    prev_normal: Vector,
    first_unit_normal: Vector,
    prev_unit_normal: Vector,
    first_pt: Point,
    prev_pt: Point,
    first_outer_pt: Point,
    first_outer_pt_index_in_contour: usize,
    segment_count: i32,
    prev_is_line: bool,

    cap: Cap,
    join: Join,

    // The outer contour is the working answer, inner is temporary.
    inner: PathBuf,
    outer: PathBuf,
    cusper: PathBuf,

    /// 1.0 for the outer stroke, -1.0 for the inner one.
    stroke_type: f32,

    /// Tracks the stack depth to abort if numerics run amok.
    recursion_depth: i32,
    /// Do less work until tangents meet (cubic).
    found_tangents: bool,
    /// Previous join was not degenerate.
    join_completed: bool,
}

impl Default for Cap {
    fn default() -> Self {
        Cap::Butt
    }
}

impl Default for Join {
    fn default() -> Self {
        Join::Miter
    }
}

const OUTER: f32 = 1.0;
const INNER: f32 = -1.0;

impl PathStroker {
    fn init(&mut self, radius: f32, miter_limit: f32, cap: Cap, mut join: Join, res_scale: f32) {
        self.radius = radius;
        self.inv_miter_limit = 0.0;
        if join == Join::Miter {
            if miter_limit <= 1.0 {
                join = Join::Bevel;
            } else {
                self.inv_miter_limit = 1.0 / miter_limit;
            }
        }
        self.cap = cap;
        self.join = join;
        self.segment_count = -1;
        self.first_outer_pt_index_in_contour = 0;
        self.prev_is_line = false;
        self.inner.clear();
        self.outer.clear();
        self.cusper.clear();
        self.res_scale = res_scale;
        // The '4' below matches the fill scan converter's error term.
        self.inv_res_scale = 1.0 / (res_scale * 4.0);
        self.inv_res_scale_squared = self.inv_res_scale * self.inv_res_scale;
        self.recursion_depth = 0;
    }

    fn has_only_move_to(&self) -> bool {
        self.segment_count == 0
    }

    fn move_to_pt(&self) -> Point {
        self.first_pt
    }

    fn is_current_contour_empty(&self) -> bool {
        is_zero_length_since_point(&self.inner.points, 0)
            && is_zero_length_since_point(&self.outer.points, self.first_outer_pt_index_in_contour)
    }

    fn close(&mut self, is_line: bool) {
        self.finish_contour(true, is_line);
    }

    fn pre_join_to(
        &mut self,
        curr_pt: Point,
        curr_is_line: bool,
    ) -> Option<(Vector, Vector)> {
        debug_assert!(self.segment_count >= 0);

        let (normal, unit_normal) =
            match set_normal_unit_normal(self.prev_pt, curr_pt, self.res_scale, self.radius) {
                Some(n) => n,
                None => {
                    if self.cap == Cap::Butt {
                        return None;
                    }
                    // Square caps and round caps draw even if the segment length is zero.
                    // Since the zero length segment has no direction, set the orientation
                    // to upright as the default orientation.
                    (Vector::new(self.radius, 0.0), Vector::new(1.0, 0.0))
                }
            };

        if self.segment_count == 0 {
            self.first_normal = normal;
            self.first_unit_normal = unit_normal;
            self.first_outer_pt = self.prev_pt + normal;

            self.outer.move_to(self.first_outer_pt);
            self.inner.move_to(self.prev_pt - normal);
        } else {
            // We have a previous segment.
            join(
                self.join,
                &mut self.outer,
                &mut self.inner,
                self.prev_unit_normal,
                self.prev_pt,
                unit_normal,
                self.radius,
                self.inv_miter_limit,
                self.prev_is_line,
                curr_is_line,
            );
        }
        self.prev_is_line = curr_is_line;
        Some((normal, unit_normal))
    }

    fn post_join_to(&mut self, curr_pt: Point, normal: Vector, unit_normal: Vector) {
        self.join_completed = true;
        self.prev_pt = curr_pt;
        self.prev_unit_normal = unit_normal;
        self.prev_normal = normal;
        self.segment_count += 1;
    }

    fn finish_contour(&mut self, close: bool, curr_is_line: bool) {
        if self.segment_count > 0 {
            if close {
                join(
                    self.join,
                    &mut self.outer,
                    &mut self.inner,
                    self.prev_unit_normal,
                    self.prev_pt,
                    self.first_unit_normal,
                    self.radius,
                    self.inv_miter_limit,
                    self.prev_is_line,
                    curr_is_line,
                );
                self.outer.close();

                // Now add the inner contour as its own contour.
                if let Some(pt) = self.inner.last_point() {
                    self.outer.move_to(pt);
                    self.outer.reverse_path_to(&self.inner);
                    self.outer.close();
                }
            } else {
                // Add caps to start and end.
                if let Some(pt) = self.inner.last_point() {
                    // Cap the end.
                    cap(self.cap, &mut self.outer, self.prev_pt, self.prev_normal, pt, curr_is_line);
                    self.outer.reverse_path_to(&self.inner);
                    // Cap the start.
                    // Note: Skia passes prev_is_line (whether the last segment is a line)
                    // to the start cap.
                    cap(
                        self.cap,
                        &mut self.outer,
                        self.first_pt,
                        -self.first_normal,
                        self.first_outer_pt,
                        self.prev_is_line,
                    );
                    self.outer.close();
                }
            }
            if !self.cusper.is_empty() {
                self.outer.add_path(&self.cusper);
                self.cusper.clear();
            }
        }
        self.inner.clear();
        self.segment_count = -1;
        self.first_outer_pt_index_in_contour = self.outer.points.len();
    }

    fn move_to(&mut self, pt: Point) {
        if self.segment_count > 0 {
            self.finish_contour(false, false);
        }
        self.segment_count = 0;
        self.first_pt = pt;
        self.prev_pt = pt;
        self.join_completed = false;
    }

    fn add_line(&mut self, curr_pt: Point, normal: Vector) {
        self.outer.line_to(curr_pt + normal);
        self.inner.line_to(curr_pt - normal);
    }

    /// `has_valid_tangent` returns whether a non-degenerate segment follows in
    /// the sub-path.
    fn line_to(&mut self, curr_pt: Point, has_valid_tangent: Option<&dyn Fn() -> bool>) {
        let teeny_line =
            equals_within_tolerance(self.prev_pt, curr_pt, NEARLY_ZERO * self.inv_res_scale);
        if self.cap == Cap::Butt && teeny_line {
            return;
        }
        if teeny_line && (self.join_completed || has_valid_tangent.map_or(false, |f| f())) {
            return;
        }

        let Some((normal, unit_normal)) = self.pre_join_to(curr_pt, true) else {
            return;
        };
        self.add_line(curr_pt, normal);
        self.post_join_to(curr_pt, normal, unit_normal);
    }

    fn set_quad_end_normal(
        &self,
        quad: &[Point; 3],
        normal_ab: Vector,
        unit_normal_ab: Vector,
    ) -> (Vector, Vector) {
        set_normal_unit_normal(quad[1], quad[2], self.res_scale, self.radius)
            .unwrap_or((normal_ab, unit_normal_ab))
    }

    fn set_cubic_end_normal(
        &self,
        cubic: &[Point; 4],
        normal_ab: Vector,
        unit_normal_ab: Vector,
    ) -> (Vector, Vector) {
        let mut ab = cubic[1] - cubic[0];
        let mut cd = cubic[3] - cubic[2];

        let mut degenerate_ab = degenerate_vector(ab);
        let mut degenerate_cd = degenerate_vector(cd);

        if degenerate_ab && degenerate_cd {
            return (normal_ab, unit_normal_ab);
        }

        if degenerate_ab {
            ab = cubic[2] - cubic[0];
            degenerate_ab = degenerate_vector(ab);
        }
        if degenerate_cd {
            cd = cubic[3] - cubic[1];
            degenerate_cd = degenerate_vector(cd);
        }
        if degenerate_ab || degenerate_cd {
            return (normal_ab, unit_normal_ab);
        }

        set_normal_unit_normal_vec(cd, self.radius).unwrap()
    }

    fn init_quad(&mut self, stroke_type: f32, quad_pts: &mut QuadConstruct, t_start: f32, t_end: f32) {
        self.stroke_type = stroke_type;
        self.found_tangents = false;
        self.recursion_depth = 0;
        quad_pts.init(t_start, t_end);
    }

    fn quad_to(&mut self, pt1: Point, pt2: Point) {
        let quad = [self.prev_pt, pt1, pt2];
        let (reduction_type, reduction) = check_quad_linear(&quad);
        match reduction_type {
            ReductionType::Point | ReductionType::Line => {
                // If the stroke consists of a moveTo followed by a degenerate curve, treat it
                // as if it were followed by a zero-length line. Lines without length
                // can have square and round end caps.
                self.line_to(pt2, None);
                return;
            }
            ReductionType::Degenerate => {
                self.line_to(reduction, None);
                let saved_join = self.join;
                self.join = Join::Round;
                self.line_to(pt2, None);
                self.join = saved_join;
                return;
            }
            ReductionType::Quad => {}
        }

        let Some((normal_ab, unit_ab)) = self.pre_join_to(pt1, false) else {
            self.line_to(pt2, None);
            return;
        };
        let mut quad_pts = QuadConstruct::default();
        self.init_quad(OUTER, &mut quad_pts, 0.0, 1.0);
        self.quad_stroke(&quad, &mut quad_pts);
        self.init_quad(INNER, &mut quad_pts, 0.0, 1.0);
        self.quad_stroke(&quad, &mut quad_pts);
        let (normal_bc, unit_bc) = self.set_quad_end_normal(&quad, normal_ab, unit_ab);

        self.post_join_to(pt2, normal_bc, unit_bc);
    }

    fn cubic_to(&mut self, pt1: Point, pt2: Point, pt3: Point) {
        let cubic = [self.prev_pt, pt1, pt2, pt3];
        let (reduction_type, reduction, tangent_pt) = check_cubic_linear(&cubic);
        match reduction_type {
            ReductionType::Point | ReductionType::Line => {
                // If the stroke consists of a moveTo followed by a degenerate curve, treat it
                // as if it were followed by a zero-length line. Lines without length
                // can have square and round end caps.
                self.line_to(pt3, None);
                return;
            }
            ReductionType::Degenerate => {
                let saved_join = self.join;
                for (i, p) in reduction.iter().enumerate() {
                    self.line_to(*p, None);
                    if i == 0 {
                        self.join = Join::Round;
                    }
                }
                self.line_to(pt3, None);
                self.join = saved_join;
                return;
            }
            ReductionType::Quad => {}
        }

        let Some((normal_ab, unit_ab)) = self.pre_join_to(tangent_pt, false) else {
            self.line_to(pt3, None);
            return;
        };
        let t_values = find_cubic_inflections(&cubic);
        let mut last_t = 0.0;
        for index in 0..=t_values.len() {
            let next_t = t_values.get(index).copied().unwrap_or(1.0);
            let mut quad_pts = QuadConstruct::default();
            self.init_quad(OUTER, &mut quad_pts, last_t, next_t);
            self.cubic_stroke(&cubic, &mut quad_pts);
            self.init_quad(INNER, &mut quad_pts, last_t, next_t);
            self.cubic_stroke(&cubic, &mut quad_pts);
            last_t = next_t;
        }
        if let Some(cusp) = find_cubic_cusp(&cubic) {
            let cusp_loc = eval_cubic(&cubic, cusp);
            self.cusper.add_circle(cusp_loc, self.radius);
        }
        // Emit the join even if one stroke succeeded but the last one failed.
        // This avoids reversing an inner stroke with a partial path followed by another moveto.
        let (normal_cd, unit_cd) = self.set_cubic_end_normal(&cubic, normal_ab, unit_ab);

        self.post_join_to(pt3, normal_cd, unit_cd);
    }

    /// Given a point on the curve and its derivative, scale the derivative by the radius, and
    /// compute the perpendicular point and its tangent.
    fn set_ray_pts(&self, t_pt: Point, dxy: Vector) -> (Point, Vector) {
        let dxy = set_length(dxy, self.radius).unwrap_or(Vector::new(self.radius, 0.0));
        // Go opposite ways for outer, inner.
        let axis_flip = self.stroke_type;
        let on_pt = Point::new(t_pt.x + axis_flip * dxy.y, t_pt.y - axis_flip * dxy.x);
        (on_pt, dxy)
    }

    /// Given a cubic and t, returns the point on curve, its perpendicular, and the
    /// perpendicular tangent.
    fn cubic_perp_ray(&self, cubic: &[Point; 4], t: f32) -> (Point, Point, Vector) {
        let t_pt = eval_cubic(cubic, t);
        let mut dxy = eval_cubic_tangent(cubic, t);
        if dxy == Vector::zero() {
            let mut c_pts: [Point; 4] = *cubic;
            if t.abs() <= NEARLY_ZERO {
                dxy = cubic[2] - cubic[0];
            } else if (1.0 - t).abs() <= NEARLY_ZERO {
                dxy = cubic[3] - cubic[1];
            } else {
                // If the cubic inflection falls on the cusp, subdivide the cubic
                // to find the tangent at that point.
                let chopped = chop_cubic_at(cubic, t);
                dxy = chopped[3] - chopped[2];
                if dxy == Vector::zero() {
                    dxy = chopped[3] - chopped[1];
                    c_pts = [chopped[0], chopped[1], chopped[2], chopped[3]];
                }
            }
            if dxy == Vector::zero() {
                dxy = c_pts[3] - c_pts[0];
            }
        }
        let (on_pt, tangent) = self.set_ray_pts(t_pt, dxy);
        (t_pt, on_pt, tangent)
    }

    /// Given a cubic and a t range, finds the start and end if they haven't been found already.
    fn cubic_quad_ends(&self, cubic: &[Point; 4], quad_pts: &mut QuadConstruct) {
        if !quad_pts.start_set {
            let (_, on_pt, tangent) = self.cubic_perp_ray(cubic, quad_pts.start_t);
            quad_pts.quad[0] = on_pt;
            quad_pts.tangent_start = tangent;
            quad_pts.start_set = true;
        }
        if !quad_pts.end_set {
            let (_, on_pt, tangent) = self.cubic_perp_ray(cubic, quad_pts.end_t);
            quad_pts.quad[2] = on_pt;
            quad_pts.tangent_end = tangent;
            quad_pts.end_set = true;
        }
    }

    fn cubic_quad_mid(&self, cubic: &[Point; 4], quad_pts: &QuadConstruct) -> Point {
        self.cubic_perp_ray(cubic, quad_pts.mid_t).1
    }

    /// Given a quad and t, returns the point on curve, its perpendicular, and the
    /// perpendicular tangent.
    fn quad_perp_ray(&self, quad: &[Point; 3], t: f32) -> (Point, Point, Vector) {
        let t_pt = eval_quad(quad, t);
        let mut dxy = eval_quad_tangent(quad, t);
        if dxy == Vector::zero() {
            dxy = quad[2] - quad[0];
        }
        let (on_pt, tangent) = self.set_ray_pts(t_pt, dxy);
        (t_pt, on_pt, tangent)
    }

    /// Finds the intersection of the stroke tangents to construct a stroke quad.
    /// Returns whether the stroke is a degenerate (a line), a quad, or must be split.
    /// Optionally computes the quad's control point.
    fn intersect_ray(&self, quad_pts: &mut QuadConstruct, intersect_ray_type: IntersectRayType) -> ResultType {
        let start = quad_pts.quad[0];
        let end = quad_pts.quad[2];
        let a_len = quad_pts.tangent_start;
        let b_len = quad_pts.tangent_end;
        // Slopes match when denom goes to zero:
        //                   axLen / ayLen ==                   bxLen / byLen
        // (ayLen * byLen) * axLen / ayLen == (ayLen * byLen) * bxLen / byLen
        //          byLen  * axLen         ==  ayLen          * bxLen
        //          byLen  * axLen         -   ayLen          * bxLen         ( == denom )
        let denom = a_len.cross(b_len);
        if denom == 0.0 || !denom.is_finite() {
            quad_pts.opposite_tangents = a_len.dot(b_len) < 0.0;
            return ResultType::Degenerate;
        }
        quad_pts.opposite_tangents = false;
        let ab0 = start - end;
        let mut numer_a = b_len.cross(ab0);
        let numer_b = a_len.cross(ab0);
        if (numer_a >= 0.0) == (numer_b >= 0.0) {
            // If the control point is outside the quad ends.

            // If the perpendicular distances from the quad points to the opposite tangent line
            // are small, a straight line is good enough.
            let dist1 = pt_to_tangent_line(start, end, quad_pts.tangent_end);
            let dist2 = pt_to_tangent_line(end, start, quad_pts.tangent_start);
            if dist1.max(dist2) <= self.inv_res_scale_squared {
                return ResultType::Degenerate;
            }
            return ResultType::Split;
        }
        // Check to see if the denominator is teeny relative to the numerator.
        // If the offset by one will be lost, the ratio is too large.
        numer_a /= denom;
        let valid_divide = numer_a > numer_a - 1.0;
        if valid_divide {
            if intersect_ray_type == IntersectRayType::CtrlPt {
                // The intersection of the tangents need not be on the tangent segment
                // so 0 <= numer_a <= 1 is not necessarily true.
                quad_pts.quad[1] = start + quad_pts.tangent_start * numer_a;
            }
            return ResultType::Quad;
        }
        quad_pts.opposite_tangents = a_len.dot(b_len) < 0.0;
        // If the lines are parallel, straight line is good enough.
        ResultType::Degenerate
    }

    /// Given a cubic and a t-range, determines if the stroke can be described by a quadratic.
    fn tangents_meet(&self, cubic: &[Point; 4], quad_pts: &mut QuadConstruct) -> ResultType {
        self.cubic_quad_ends(cubic, quad_pts);
        self.intersect_ray(quad_pts, IntersectRayType::ResultType)
    }

    /// Returns true if the point is close to the bounds of the quad. This is used as a quick reject.
    fn pt_in_quad_bounds(&self, quad: &[Point; 3], pt: Point) -> bool {
        let x_min = quad[0].x.min(quad[1].x).min(quad[2].x);
        if pt.x + self.inv_res_scale < x_min {
            return false;
        }
        let x_max = quad[0].x.max(quad[1].x).max(quad[2].x);
        if pt.x - self.inv_res_scale > x_max {
            return false;
        }
        let y_min = quad[0].y.min(quad[1].y).min(quad[2].y);
        if pt.y + self.inv_res_scale < y_min {
            return false;
        }
        let y_max = quad[0].y.max(quad[1].y).max(quad[2].y);
        if pt.y - self.inv_res_scale > y_max {
            return false;
        }
        true
    }

    fn stroke_close_enough(&self, stroke: &[Point; 3], ray: &[Point; 2], quad_pts: &QuadConstruct) -> ResultType {
        let stroke_mid = eval_quad(stroke, 0.5);
        // Measure the distance from the curve to the quad-stroke midpoint, compare to radius.
        if points_within_dist(ray[0], stroke_mid, self.inv_res_scale) {
            // If the difference is small.
            if sharp_angle(&quad_pts.quad) {
                return ResultType::Split;
            }
            return ResultType::Quad;
        }
        // Measure the distance to quad's bounds (quick reject).
        // An alternative: look for point in triangle.
        if !self.pt_in_quad_bounds(stroke, ray[0]) {
            // If far, subdivide.
            return ResultType::Split;
        }
        // Measure the curve ray distance to the quad-stroke.
        let roots = intersect_quad_ray(ray, stroke);
        if roots.len() != 1 {
            return ResultType::Split;
        }
        let quad_pt = eval_quad(stroke, roots[0]);
        let error = self.inv_res_scale * (1.0 - (roots[0] - 0.5).abs() * 2.0);
        if points_within_dist(ray[0], quad_pt, error) {
            // If the difference is small, we're done.
            if sharp_angle(&quad_pts.quad) {
                return ResultType::Split;
            }
            return ResultType::Quad;
        }
        // Otherwise, subdivide.
        ResultType::Split
    }

    fn compare_quad_cubic(&self, cubic: &[Point; 4], quad_pts: &mut QuadConstruct) -> ResultType {
        // Get the quadratic approximation of the stroke.
        self.cubic_quad_ends(cubic, quad_pts);
        let result_type = self.intersect_ray(quad_pts, IntersectRayType::CtrlPt);
        if result_type != ResultType::Quad {
            return result_type;
        }
        // Project a ray from the curve to the stroke.
        // Points near midpoint on quad, midpoint on cubic.
        let (t_pt, on_pt, _) = self.cubic_perp_ray(cubic, quad_pts.mid_t);
        let ray = [on_pt, t_pt];
        self.stroke_close_enough(&quad_pts.quad, &ray, quad_pts)
    }

    fn compare_quad_quad(&self, quad: &[Point; 3], quad_pts: &mut QuadConstruct) -> ResultType {
        // Get the quadratic approximation of the stroke.
        if !quad_pts.start_set {
            let (_, on_pt, tangent) = self.quad_perp_ray(quad, quad_pts.start_t);
            quad_pts.quad[0] = on_pt;
            quad_pts.tangent_start = tangent;
            quad_pts.start_set = true;
        }
        if !quad_pts.end_set {
            let (_, on_pt, tangent) = self.quad_perp_ray(quad, quad_pts.end_t);
            quad_pts.quad[2] = on_pt;
            quad_pts.tangent_end = tangent;
            quad_pts.end_set = true;
        }
        let result_type = self.intersect_ray(quad_pts, IntersectRayType::CtrlPt);
        if result_type != ResultType::Quad {
            return result_type;
        }
        // Project a ray from the curve to the stroke.
        let (t_pt, on_pt, _) = self.quad_perp_ray(quad, quad_pts.mid_t);
        let ray = [on_pt, t_pt];
        self.stroke_close_enough(&quad_pts.quad, &ray, quad_pts)
    }

    fn sink(&mut self) -> &mut PathBuf {
        if self.stroke_type == OUTER {
            &mut self.outer
        } else {
            &mut self.inner
        }
    }

    fn add_degenerate_line(&mut self, quad_pts: &QuadConstruct) {
        let to = quad_pts.quad[2];
        self.sink().line_to(to);
    }

    fn add_quad(&mut self, quad_pts: &QuadConstruct) {
        let ctrl = quad_pts.quad[1];
        let to = quad_pts.quad[2];
        self.sink().quad_to(ctrl, to);
    }

    fn cubic_mid_on_line(&self, cubic: &[Point; 4], quad_pts: &QuadConstruct) -> bool {
        let stroke_mid = self.cubic_quad_mid(cubic, quad_pts);
        let dist = pt_to_line(stroke_mid, quad_pts.quad[0], quad_pts.quad[2]);
        dist < self.inv_res_scale_squared
    }

    fn cubic_stroke(&mut self, cubic: &[Point; 4], quad_pts: &mut QuadConstruct) -> bool {
        if !self.found_tangents {
            let result_type = self.tangents_meet(cubic, quad_pts);
            if result_type != ResultType::Quad {
                if (result_type == ResultType::Degenerate
                    || points_within_dist(quad_pts.quad[0], quad_pts.quad[2], self.inv_res_scale))
                    && self.cubic_mid_on_line(cubic, quad_pts)
                {
                    self.add_degenerate_line(quad_pts);
                    return true;
                }
            } else {
                self.found_tangents = true;
            }
        }
        if self.found_tangents {
            let result_type = self.compare_quad_cubic(cubic, quad_pts);
            if result_type == ResultType::Quad {
                self.add_quad(quad_pts);
                return true;
            }
            if result_type == ResultType::Degenerate && !quad_pts.opposite_tangents {
                self.add_degenerate_line(quad_pts);
                return true;
            }
        }
        if !(quad_pts.quad[2].x.is_finite() && quad_pts.quad[2].y.is_finite()) {
            // Just abort if projected quad isn't representable.
            return false;
        }
        self.recursion_depth += 1;
        let limit = if self.found_tangents {
            CUBIC_RECURSIVE_LIMIT
        } else {
            TANGENT_RECURSIVE_LIMIT
        };
        if self.recursion_depth > limit {
            // If we stop making progress, just emit a line and move on.
            self.add_degenerate_line(quad_pts);
            self.recursion_depth -= 1;
            return true;
        }
        let mut half = QuadConstruct::default();
        if !half.init_with_start(quad_pts) {
            self.add_degenerate_line(quad_pts);
            self.recursion_depth -= 1;
            return true;
        }
        if !self.cubic_stroke(cubic, &mut half) {
            return false;
        }
        if !half.init_with_end(quad_pts) {
            self.add_degenerate_line(quad_pts);
            self.recursion_depth -= 1;
            return true;
        }
        if !self.cubic_stroke(cubic, &mut half) {
            return false;
        }
        self.recursion_depth -= 1;
        true
    }

    fn quad_stroke(&mut self, quad: &[Point; 3], quad_pts: &mut QuadConstruct) -> bool {
        let result_type = self.compare_quad_quad(quad, quad_pts);
        if result_type == ResultType::Quad {
            self.add_quad(quad_pts);
            return true;
        }
        if result_type == ResultType::Degenerate {
            self.add_degenerate_line(quad_pts);
            return true;
        }
        if !(quad_pts.quad[2].x.is_finite() && quad_pts.quad[2].y.is_finite()) {
            // Just abort if projected quad isn't representable.
            return false;
        }
        self.recursion_depth += 1;
        if self.recursion_depth > QUAD_RECURSIVE_LIMIT {
            // If we stop making progress, just emit a line and move on.
            self.add_degenerate_line(quad_pts);
            self.recursion_depth -= 1;
            return true;
        }
        let mut half = QuadConstruct::default();
        if !half.init_with_start(quad_pts) {
            self.add_degenerate_line(quad_pts);
            self.recursion_depth -= 1;
            return true;
        }
        if !self.quad_stroke(quad, &mut half) {
            return false;
        }
        if !half.init_with_end(quad_pts) {
            self.add_degenerate_line(quad_pts);
            self.recursion_depth -= 1;
            return true;
        }
        if !self.quad_stroke(quad, &mut half) {
            return false;
        }
        self.recursion_depth -= 1;
        true
    }
}

fn is_zero_length_since_point(points: &[Point], start_pt_index: usize) -> bool {
    if points.len() < start_pt_index + 2 {
        return true;
    }
    let pts = &points[start_pt_index..];
    pts[1..].iter().all(|p| *p == pts[0])
}

//
// Caps and joins (SkStrokerPriv.cpp).
//

fn rotate_cw(v: Vector) -> Vector {
    Vector::new(-v.y, v.x)
}

fn rotate_ccw(v: Vector) -> Vector {
    Vector::new(v.y, -v.x)
}

fn cap(cap: Cap, sink: &mut PathBuf, pivot: Point, normal: Vector, stop: Point, extend_last_pt: bool) {
    match cap {
        Cap::Butt => {
            sink.line_to(stop);
        }
        Cap::Round => {
            let parallel = rotate_cw(normal);
            let projected_center = pivot + parallel;
            sink.conic_to(projected_center + normal, projected_center, ROOT_2_OVER_2);
            sink.conic_to(projected_center - normal, stop, ROOT_2_OVER_2);
        }
        Cap::Square => {
            let parallel = rotate_cw(normal);
            if extend_last_pt {
                sink.set_last_point(pivot + normal + parallel);
                sink.line_to(pivot - normal + parallel);
            } else {
                sink.line_to(pivot + normal + parallel);
                sink.line_to(pivot - normal + parallel);
                sink.line_to(stop);
            }
        }
    }
}

fn is_clockwise(before: Vector, after: Vector) -> bool {
    before.x * after.y > before.y * after.x
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum AngleType {
    Nearly180,
    Sharp,
    Shallow,
    NearlyLine,
}

fn dot_to_angle_type(dot: f32) -> AngleType {
    if dot >= 0.0 {
        // Shallow or line.
        if (1.0 - dot).abs() <= NEARLY_ZERO {
            AngleType::NearlyLine
        } else {
            AngleType::Shallow
        }
    } else {
        // Sharp or 180.
        if (1.0 + dot).abs() <= NEARLY_ZERO {
            AngleType::Nearly180
        } else {
            AngleType::Sharp
        }
    }
}

fn handle_inner_join(inner: &mut PathBuf, pivot: Point, after: Vector) {
    // In the degenerate case that the stroke radius is larger than our segments
    // just connecting the two inner segments may "show through" as a funny
    // diagonal. To pseudo-fix this, we go through the pivot point. This adds
    // an extra point/edge, but I can't see a cheap way to know when this is
    // not needed :(
    inner.line_to(pivot);
    inner.line_to(pivot - after);
}

fn join(
    join: Join,
    outer: &mut PathBuf,
    inner: &mut PathBuf,
    before_unit_normal: Vector,
    pivot: Point,
    after_unit_normal: Vector,
    radius: f32,
    inv_miter_limit: f32,
    prev_is_line: bool,
    curr_is_line: bool,
) {
    match join {
        Join::Bevel => bevel_join(outer, inner, before_unit_normal, pivot, after_unit_normal, radius),
        Join::Round => round_join(outer, inner, before_unit_normal, pivot, after_unit_normal, radius),
        Join::Miter => miter_join(
            outer,
            inner,
            before_unit_normal,
            pivot,
            after_unit_normal,
            radius,
            inv_miter_limit,
            prev_is_line,
            curr_is_line,
        ),
    }
}

fn bevel_join<'a>(
    mut outer: &'a mut PathBuf,
    mut inner: &'a mut PathBuf,
    before_unit_normal: Vector,
    pivot: Point,
    after_unit_normal: Vector,
    radius: f32,
) {
    let mut after = after_unit_normal * radius;

    if !is_clockwise(before_unit_normal, after_unit_normal) {
        std::mem::swap(&mut outer, &mut inner);
        after = -after;
    }

    outer.line_to(pivot + after);
    handle_inner_join(inner, pivot, after);
}

fn round_join<'a>(
    mut outer: &'a mut PathBuf,
    mut inner: &'a mut PathBuf,
    before_unit_normal: Vector,
    pivot: Point,
    after_unit_normal: Vector,
    radius: f32,
) {
    let dot_prod = before_unit_normal.dot(after_unit_normal);
    let angle_type = dot_to_angle_type(dot_prod);

    if angle_type == AngleType::NearlyLine {
        return;
    }

    let mut before = before_unit_normal;
    let mut after = after_unit_normal;
    let mut cw = true;

    if !is_clockwise(before, after) {
        std::mem::swap(&mut outer, &mut inner);
        before = -before;
        after = -after;
        cw = false;
    }

    let conics = build_unit_arc(before, after, cw, radius, pivot);
    if !conics.is_empty() {
        for conic in &conics {
            outer.conic_to(conic.pts[1], conic.pts[2], conic.w);
        }
        handle_inner_join(inner, pivot, after * radius);
    }
}

fn miter_join<'a>(
    mut outer: &'a mut PathBuf,
    mut inner: &'a mut PathBuf,
    before_unit_normal: Vector,
    pivot: Point,
    after_unit_normal: Vector,
    radius: f32,
    inv_miter_limit: f32,
    prev_is_line: bool,
    mut curr_is_line: bool,
) {
    // Negate the dot since we're using normals instead of tangents.
    let dot_prod = before_unit_normal.dot(after_unit_normal);
    let angle_type = dot_to_angle_type(dot_prod);
    let mut before = before_unit_normal;
    let mut after = after_unit_normal;

    if angle_type == AngleType::NearlyLine {
        return;
    }

    // Some(mid) if a miter should be added, None for a blunt join.
    let miter = 'miter: {
        if angle_type == AngleType::Nearly180 {
            curr_is_line = false;
            break 'miter None;
        }

        let ccw = !is_clockwise(before, after);
        if ccw {
            std::mem::swap(&mut outer, &mut inner);
            before = -before;
            after = -after;
        }

        // Before we enter the world of square-roots and divides,
        // check if we're trying to join an upright right angle
        // (common case for stroking rectangles). If so, special case
        // that (for speed an accuracy).
        // Note: we only need to check one normal if dot==0
        if dot_prod == 0.0 && inv_miter_limit <= ROOT_2_OVER_2 {
            break 'miter Some((before + after) * radius);
        }

        // midLength = radius / sinHalfAngle
        // if (midLength > miterLimit * radius) abort
        // if (radius / sinHalf > miterLimit * radius) abort
        // if (1 / sinHalf > miterLimit) abort
        // if (1 / miterLimit > sinHalf) abort
        // My dotProd is opposite sign, since it is built from normals and not tangents
        // hence 1 + dot instead of 1 - dot in the formula
        let sin_half_angle = ((1.0 + dot_prod) / 2.0).sqrt();
        if sin_half_angle < inv_miter_limit {
            curr_is_line = false;
            break 'miter None;
        }

        // Choose the most accurate way to form the initial mid-vector.
        let mut mid = if angle_type == AngleType::Sharp {
            let mid = Vector::new(after.y - before.y, before.x - after.x);
            if ccw { -mid } else { mid }
        } else {
            before + after
        };

        mid = set_length(mid, radius / sin_half_angle).unwrap_or(Vector::zero());
        Some(mid)
    };

    if let Some(mid) = miter {
        if prev_is_line {
            outer.set_last_point(pivot + mid);
        } else {
            outer.line_to(pivot + mid);
        }
    }

    let after = after * radius;
    if !curr_is_line {
        outer.line_to(pivot + after);
    }
    handle_inner_join(inner, pivot, after);
}

/// `SkConic::BuildUnitArc`, followed by the scale + translate transform used by
/// the round join.
fn build_unit_arc(
    u_start: Vector,
    u_stop: Vector,
    cw: bool,
    radius: f32,
    pivot: Point,
) -> ArrayVec<Conic, 5> {
    let mut dst = ArrayVec::new();

    // Rotate by x,y so that u_start is (1.0).
    let x = u_start.dot(u_stop);
    let mut y = u_start.cross(u_stop);

    let abs_y = y.abs();

    // Check for (effectively) coincident vectors.
    // This can happen if our angle is nearly 0 or nearly 180 (y == 0).
    // We use the dot-prod to distinguish between 0 and 180 (x > 0).
    if abs_y <= NEARLY_ZERO && x > 0.0 && ((y >= 0.0 && cw) || (y <= 0.0 && !cw)) {
        return dst;
    }

    if !cw {
        y = -y;
    }

    // We decide to use 1-conic per quadrant of a circle. What quadrant does [xy] lie in?
    //      0 == [0  .. 90)
    //      1 == [90 ..180)
    //      2 == [180..270)
    //      3 == [270..360)
    let mut quadrant = 0;
    if y == 0.0 {
        // 180
        quadrant = 2;
    } else if x == 0.0 {
        // 90 : 270
        quadrant = if y > 0.0 { 1 } else { 3 };
    } else {
        if y < 0.0 {
            quadrant += 2;
        }
        if (x < 0.0) != (y < 0.0) {
            quadrant += 1;
        }
    }

    let quadrant_pts = [
        Point::new(1.0, 0.0),
        Point::new(1.0, 1.0),
        Point::new(0.0, 1.0),
        Point::new(-1.0, 1.0),
        Point::new(-1.0, 0.0),
        Point::new(-1.0, -1.0),
        Point::new(0.0, -1.0),
        Point::new(1.0, -1.0),
    ];

    for i in 0..quadrant {
        dst.push(Conic {
            pts: [quadrant_pts[i * 2], quadrant_pts[i * 2 + 1], quadrant_pts[(i * 2 + 2) % 8]],
            w: ROOT_2_OVER_2,
        });
    }

    // Now compute any remaing (sub-90-degree) arc for the last conic.
    let final_p = Point::new(x, y);
    let last_q = quadrant_pts[quadrant * 2]; // Will already be a unit-vector.
    let dot = last_q.to_vector().dot(final_p.to_vector());
    if dot.is_nan() {
        return ArrayVec::new();
    }

    if dot < 1.0 {
        let off_curve = Vector::new(last_q.x + x, last_q.y + y);
        // Compute the bisector vector, and then rescale to be the off-curve point.
        // We compute its length from cos(theta/2) = length / 1, using half-angle identity we get
        // length = sqrt(2 / (1 + cos(theta)). We already have cos() when to computed the dot.
        // This is nice, since our computed weight is cos(theta/2) as well!
        let cos_theta_over_2 = ((1.0 + dot) / 2.0).sqrt();
        let off_curve = set_length(off_curve, 1.0 / cos_theta_over_2).unwrap_or(Vector::zero());
        if !equals_within_tolerance(last_q, off_curve.to_point(), NEARLY_ZERO) {
            dst.push(Conic {
                pts: [last_q, off_curve.to_point(), final_p],
                w: cos_theta_over_2,
            });
        }
    }

    // Now handle counter-clockwise and the initial u_start rotation.
    let (sin, cos) = (u_start.y, u_start.x);
    let y_scale = if cw { 1.0 } else { -1.0 };
    for conic in &mut dst {
        for p in &mut conic.pts {
            let px = p.x;
            let py = p.y * y_scale;
            *p = Point::new(
                (px * cos - py * sin) * radius + pivot.x,
                (px * sin + py * cos) * radius + pivot.y,
            );
        }
    }

    dst
}

//
// Geometry helpers (SkStroke.cpp, SkGeometry.cpp, SkPointPriv.h).
//

/// `!SkPointPriv::CanNormalize`.
fn degenerate_vector(v: Vector) -> bool {
    !(v.x.is_finite() && v.y.is_finite() && (v.x != 0.0 || v.y != 0.0))
}

/// `SkPointPriv::EqualsWithinTolerance`.
fn equals_within_tolerance(a: Point, b: Point, tol: f32) -> bool {
    (a.x - b.x).abs() <= tol && (a.y - b.y).abs() <= tol
}

/// `SkPoint::setLength`. Returns `None` if the vector can't be normalized.
fn set_length(v: Vector, length: f32) -> Option<Vector> {
    let x = v.x as f64;
    let y = v.y as f64;
    let mag = (x * x + y * y).sqrt();
    if !mag.is_finite() || mag == 0.0 {
        return None;
    }
    let scale = length as f64 / mag;
    let r = Vector::new((x * scale) as f32, (y * scale) as f32);
    if !(r.x.is_finite() && r.y.is_finite()) || r == Vector::zero() {
        return None;
    }
    Some(r)
}

/// Returns `(normal, unit_normal)`.
fn set_normal_unit_normal(before: Point, after: Point, scale: f32, radius: f32) -> Option<(Vector, Vector)> {
    set_normal_unit_normal_vec((after - before) * scale, radius)
}

fn set_normal_unit_normal_vec(v: Vector, radius: f32) -> Option<(Vector, Vector)> {
    let unit_normal = rotate_ccw(set_length(v, 1.0)?);
    Some((unit_normal * radius, unit_normal))
}

/// Returns the distance squared from the point to the line segment.
fn pt_to_line(pt: Point, line_start: Point, line_end: Point) -> f32 {
    let dxy = line_end - line_start;
    let ab0 = pt - line_start;
    let numer = dxy.dot(ab0);
    let denom = dxy.dot(dxy);
    let t = numer / denom;
    if t >= 0.0 && t <= 1.0 {
        let hit = Point::new(
            line_start.x * (1.0 - t) + line_end.x * t,
            line_start.y * (1.0 - t) + line_end.y * t,
        );
        (hit - pt).square_length()
    } else {
        (pt - line_start).square_length()
    }
}

/// Returns the distance squared from the point to the line.
fn pt_to_tangent_line(pt: Point, line_start: Point, tangent: Vector) -> f32 {
    let ab0 = pt - line_start;
    let numer = tangent.dot(ab0);
    let denom = tangent.dot(tangent);
    let t = numer / denom;
    if t >= 0.0 && t <= 1.0 {
        let hit = line_start + tangent * t;
        (hit - pt).square_length()
    } else {
        (pt - line_start).square_length()
    }
}

fn points_within_dist(near_pt: Point, far_pt: Point, limit: f32) -> bool {
    (near_pt - far_pt).square_length() <= limit * limit
}

fn sharp_angle(quad: &[Point; 3]) -> bool {
    let mut smaller = quad[1] - quad[0];
    let mut larger = quad[1] - quad[2];
    let smaller_len = smaller.square_length();
    let mut larger_len = larger.square_length();
    if smaller_len > larger_len {
        std::mem::swap(&mut smaller, &mut larger);
        larger_len = smaller_len;
    }
    let Some(smaller) = set_length(smaller, larger_len) else {
        return false;
    };
    smaller.dot(larger) > 0.0
}

/// Intersects the line with the quad and returns the t values on the quad where
/// the line crosses.
fn intersect_quad_ray(line: &[Point; 2], quad: &[Point; 3]) -> ArrayVec<f32, 2> {
    let vec = line[1] - line[0];
    let r = [
        vec.cross(quad[0] - line[0]),
        vec.cross(quad[1] - line[0]),
        vec.cross(quad[2] - line[0]),
    ];
    let mut a = r[2];
    let mut b = r[1];
    let c = r[0];
    a += c - 2.0 * b; // A = a - 2*b + c
    b -= c; // B = -(b - c)
    find_unit_quad_roots(a, 2.0 * b, c)
}

/// Given a cubic, determines if all four points are in a line.
/// Returns true if the inner points are close to a line connecting the outermost points.
fn cubic_in_line(cubic: &[Point; 4]) -> bool {
    let mut pt_max = -1.0;
    let mut outer1 = 0;
    let mut outer2 = 0;
    for index in 0..3 {
        for inner in index + 1..4 {
            let test_diff = cubic[inner] - cubic[index];
            let test_max = test_diff.x.abs().max(test_diff.y.abs());
            if pt_max < test_max {
                outer1 = index;
                outer2 = inner;
                pt_max = test_max;
            }
        }
    }
    let mid1 = (1 + (2 >> outer2)) >> outer1;
    let mid2 = outer1 ^ outer2 ^ mid1;
    // This multiplier is pulled out of the air.
    let line_slop = pt_max * pt_max * 0.00001;
    pt_to_line(cubic[mid1], cubic[outer1], cubic[outer2]) <= line_slop
        && pt_to_line(cubic[mid2], cubic[outer1], cubic[outer2]) <= line_slop
}

/// Given a quad, sees if all three points are in a line.
/// Returns true if the inside point is close to a line connecting the outermost points.
fn quad_in_line(quad: &[Point; 3]) -> bool {
    let mut pt_max = -1.0;
    let mut outer1 = 0;
    let mut outer2 = 0;
    for index in 0..2 {
        for inner in index + 1..3 {
            let test_diff = quad[inner] - quad[index];
            let test_max = test_diff.x.abs().max(test_diff.y.abs());
            if pt_max < test_max {
                outer1 = index;
                outer2 = inner;
                pt_max = test_max;
            }
        }
    }
    let mid = outer1 ^ outer2 ^ 3;
    // This multiplier is pulled out of the air.
    const CURVATURE_SLOP: f32 = 0.000005;
    let line_slop = pt_max * pt_max * CURVATURE_SLOP;
    pt_to_line(quad[mid], quad[outer1], quad[outer2]) <= line_slop
}

/// Returns the reduction type, the reduction points and the point used to compute
/// the start tangent.
fn check_cubic_linear(cubic: &[Point; 4]) -> (ReductionType, ArrayVec<Point, 3>, Point) {
    let mut reduction = ArrayVec::new();
    let degenerate_ab = degenerate_vector(cubic[1] - cubic[0]);
    let degenerate_bc = degenerate_vector(cubic[2] - cubic[1]);
    let degenerate_cd = degenerate_vector(cubic[3] - cubic[2]);
    if degenerate_ab & degenerate_bc & degenerate_cd {
        return (ReductionType::Point, reduction, cubic[1]);
    }
    if degenerate_ab as u32 + degenerate_bc as u32 + degenerate_cd as u32 == 2 {
        return (ReductionType::Line, reduction, cubic[1]);
    }
    if !cubic_in_line(cubic) {
        let tangent_pt = if degenerate_ab { cubic[2] } else { cubic[1] };
        return (ReductionType::Quad, reduction, tangent_pt);
    }
    let t_values = find_cubic_max_curvature(cubic);
    // Now loop over the t-values, and reject any that evaluate to either end-point.
    for t in t_values {
        if 0.0 >= t || t >= 1.0 {
            continue;
        }
        let p = eval_cubic(cubic, t);
        if p != cubic[0] && p != cubic[3] {
            reduction.push(p);
        }
    }
    if reduction.is_empty() {
        return (ReductionType::Line, reduction, cubic[1]);
    }
    (ReductionType::Degenerate, reduction, cubic[1])
}

fn check_quad_linear(quad: &[Point; 3]) -> (ReductionType, Point) {
    let degenerate_ab = degenerate_vector(quad[1] - quad[0]);
    let degenerate_bc = degenerate_vector(quad[2] - quad[1]);
    if degenerate_ab & degenerate_bc {
        return (ReductionType::Point, quad[0]);
    }
    if degenerate_ab | degenerate_bc {
        return (ReductionType::Line, quad[0]);
    }
    if !quad_in_line(quad) {
        return (ReductionType::Quad, quad[0]);
    }
    let t = find_quad_max_curvature(quad);
    if t == 0.0 || t == 1.0 {
        return (ReductionType::Line, quad[0]);
    }
    (ReductionType::Degenerate, eval_quad(quad, t))
}

fn eval_quad(q: &[Point; 3], t: f32) -> Point {
    let c = q[0].to_vector();
    let b = (q[1].to_vector() - c) * 2.0;
    let a = q[2].to_vector() - q[1].to_vector() * 2.0 + c;
    ((a * t + b) * t + c).to_point()
}

fn eval_quad_tangent(q: &[Point; 3], t: f32) -> Vector {
    // The derivative equation is 2(b - a +(a - 2b +c)t). This returns a
    // zero tangent vector when t is 0 or 1, and the control point is equal
    // to the end point. In this case, use the quad end points to compute the tangent.
    if (t == 0.0 && q[0] == q[1]) || (t == 1.0 && q[1] == q[2]) {
        return q[2] - q[0];
    }
    let b = q[1] - q[0];
    let a = q[2] - q[1] - b;
    let t = a * t + b;
    t + t
}

fn eval_cubic(c: &[Point; 4], t: f32) -> Point {
    let p0 = c[0].to_vector();
    let p1 = c[1].to_vector();
    let p2 = c[2].to_vector();
    let p3 = c[3].to_vector();
    let a = p3 + (p1 - p2) * 3.0 - p0;
    let b = (p2 - p1 * 2.0 + p0) * 3.0;
    let cc = (p1 - p0) * 3.0;
    (((a * t + b) * t + cc) * t + p0).to_point()
}

/// One third of the derivative.
fn eval_cubic_derivative(c: &[Point; 4], t: f32) -> Vector {
    let p0 = c[0].to_vector();
    let p1 = c[1].to_vector();
    let p2 = c[2].to_vector();
    let p3 = c[3].to_vector();
    let a = p3 + (p1 - p2) * 3.0 - p0;
    let b = (p2 - p1 * 2.0 + p0) * 2.0;
    let cc = p1 - p0;
    (a * t + b) * t + cc
}

fn eval_cubic_tangent(c: &[Point; 4], t: f32) -> Vector {
    // The derivative equation returns a zero tangent vector when t is 0 or 1, and the
    // adjacent control point is equal to the end point. In this case, use the
    // next control point or the end points to compute the tangent.
    if (t == 0.0 && c[0] == c[1]) || (t == 1.0 && c[2] == c[3]) {
        let tangent = if t == 0.0 { c[2] - c[0] } else { c[3] - c[1] };
        if tangent == Vector::zero() {
            return c[3] - c[0];
        }
        return tangent;
    }
    eval_cubic_derivative(c, t)
}

fn chop_cubic_at(c: &[Point; 4], t: f32) -> [Point; 7] {
    if t == 1.0 {
        return [c[0], c[1], c[2], c[3], c[3], c[3], c[3]];
    }
    // (b - a) * t + a: this does not return b when t == 1, but it otherwise seems to
    // get better precision than a * (1 - t) + b * t.
    let mix = |a: Point, b: Point| a + (b - a) * t;
    let ab = mix(c[0], c[1]);
    let bc = mix(c[1], c[2]);
    let cd = mix(c[2], c[3]);
    let abc = mix(ab, bc);
    let bcd = mix(bc, cd);
    let abcd = mix(abc, bcd);
    [c[0], ab, abc, abcd, bcd, cd, c[3]]
}

fn valid_unit_divide(mut numer: f32, mut denom: f32) -> Option<f32> {
    if numer < 0.0 {
        numer = -numer;
        denom = -denom;
    }
    if denom == 0.0 || numer == 0.0 || numer >= denom {
        return None;
    }
    let r = numer / denom;
    if r.is_nan() || r == 0.0 {
        return None;
    }
    Some(r)
}

/// `SkFindUnitQuadRoots`. From Numerical Recipes in C.
///
/// Q = -1/2 (B + sign(B) sqrt[B*B - 4*A*C])
/// x1 = Q / A
/// x2 = C / Q
fn find_unit_quad_roots(a: f32, b: f32, c: f32) -> ArrayVec<f32, 2> {
    let mut roots = ArrayVec::new();
    if a == 0.0 {
        roots.extend(valid_unit_divide(-c, b));
        return roots;
    }

    // Use doubles so we don't overflow temporarily trying to compute R.
    let dr = b as f64 * b as f64 - 4.0 * a as f64 * c as f64;
    if dr < 0.0 {
        return roots;
    }
    let r = dr.sqrt() as f32;
    if !r.is_finite() {
        return roots;
    }

    let q = if b < 0.0 { -(b - r) / 2.0 } else { -(b + r) / 2.0 };
    roots.extend(valid_unit_divide(q, a));
    roots.extend(valid_unit_divide(c, q));
    if roots.len() == 2 {
        if roots[0] > roots[1] {
            roots.swap(0, 1);
        } else if roots[0] == roots[1] {
            // Skip the double root.
            roots.pop();
        }
    }
    roots
}

/// F'(t) dot F''(t) == 0
///
/// t = - (Ax Bx + Ay By) / (Bx ^ 2 + By ^ 2)
fn find_quad_max_curvature(src: &[Point; 3]) -> f32 {
    let ax = src[1].x - src[0].x;
    let ay = src[1].y - src[0].y;
    let bx = src[0].x - src[1].x - src[1].x + src[2].x;
    let by = src[0].y - src[1].y - src[1].y + src[2].y;

    let mut numer = -(ax * bx + ay * by);
    let mut denom = bx * bx + by * by;
    if denom < 0.0 {
        numer = -numer;
        denom = -denom;
    }
    if numer <= 0.0 {
        return 0.0;
    }
    // Also catches denom == 0.
    if numer >= denom {
        return 1.0;
    }
    numer / denom
}

/// Solves coeff(t) == 0, returning the roots that lie within 0 <= t <= 1.
/// coeff[0]t^3 + coeff[1]t^2 + coeff[2]t + coeff[3]
///
/// Eliminates repeated roots (so that all t values are distinct, and are always
/// in increasing order).
fn solve_cubic_poly(coeff: &[f32; 4]) -> ArrayVec<f32, 3> {
    let mut t_values = ArrayVec::new();
    if coeff[0].abs() <= NEARLY_ZERO {
        // We're just a quadratic.
        t_values.extend(find_unit_quad_roots(coeff[1], coeff[2], coeff[3]));
        return t_values;
    }

    let inva = 1.0 / coeff[0];
    let a = coeff[1] * inva;
    let b = coeff[2] * inva;
    let c = coeff[3] * inva;

    let q = (a * a - b * 3.0) / 9.0;
    let r = (2.0 * a * a * a - 9.0 * a * b + 27.0 * c) / 54.0;

    let q3 = q * q * q;
    let r2_minus_q3 = r * r - q3;
    let adiv3 = a / 3.0;

    if r2_minus_q3 < 0.0 {
        // We have 3 real roots.
        // The divide/root can, due to finite precisions, be slightly outside of -1...1.
        let theta = (r / q3.sqrt()).clamp(-1.0, 1.0).acos();
        let neg2_root_q = -2.0 * q.sqrt();
        let pi = std::f32::consts::PI;

        let mut roots = [
            (neg2_root_q * (theta / 3.0).cos() - adiv3).clamp(0.0, 1.0),
            (neg2_root_q * ((theta + 2.0 * pi) / 3.0).cos() - adiv3).clamp(0.0, 1.0),
            (neg2_root_q * ((theta - 2.0 * pi) / 3.0).cos() - adiv3).clamp(0.0, 1.0),
        ];
        // Now sort the roots and remove duplicates.
        roots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        for root in roots {
            if t_values.last() != Some(&root) {
                t_values.push(root);
            }
        }
    } else {
        // We have 1 real root.
        let mut a = r.abs() + r2_minus_q3.sqrt();
        a = a.powf(0.3333333);
        if r > 0.0 {
            a = -a;
        }
        if a != 0.0 {
            a += q / a;
        }
        t_values.push((a - adiv3).clamp(0.0, 1.0));
    }
    t_values
}

/// Looking for F' dot F'' == 0
///
/// A = b - a
/// B = c - 2b + a
/// C = d - 3c + 3b - a
///
/// F' = 3Ct^2 + 6Bt + 3A
/// F'' = 6Ct + 6B
///
/// F' dot F'' -> CCt^3 + 3BCt^2 + (2BB + CA)t + AB
fn find_cubic_max_curvature(src: &[Point; 4]) -> ArrayVec<f32, 3> {
    let coeffs = |p0: f32, p1: f32, p2: f32, p3: f32| {
        let a = p1 - p0;
        let b = p2 - 2.0 * p1 + p0;
        let c = p3 + 3.0 * (p1 - p2) - p0;
        [c * c, 3.0 * b * c, 2.0 * b * b + c * a, a * b]
    };
    let cx = coeffs(src[0].x, src[1].x, src[2].x, src[3].x);
    let cy = coeffs(src[0].y, src[1].y, src[2].y, src[3].y);
    let coeff = [cx[0] + cy[0], cx[1] + cy[1], cx[2] + cy[2], cx[3] + cy[3]];
    solve_cubic_poly(&coeff)
}

/// A = b - a
/// B = c - 2b + a
/// C = d - 3c + 3b - a
/// (BxCy - ByCx)t^2 + (AxCy - AyCx)t + AxBy - AyBx == 0
fn find_cubic_inflections(src: &[Point; 4]) -> ArrayVec<f32, 2> {
    let ax = src[1].x - src[0].x;
    let ay = src[1].y - src[0].y;
    let bx = src[2].x - 2.0 * src[1].x + src[0].x;
    let by = src[2].y - 2.0 * src[1].y + src[0].y;
    let cx = src[3].x + 3.0 * (src[1].x - src[2].x) - src[0].x;
    let cy = src[3].y + 3.0 * (src[1].y - src[2].y) - src[0].y;

    find_unit_quad_roots(bx * cy - by * cx, ax * cy - ay * cx, ax * by - ay * bx)
}

/// Returns true if both points src[test_index], src[test_index+1] are in the same half plane
/// defined by the line segment src[line_index], src[line_index+1].
fn on_same_side(src: &[Point; 4], test_index: usize, line_index: usize) -> bool {
    let origin = src[line_index];
    let line = src[line_index + 1] - origin;
    let c0 = line.cross(src[test_index] - origin);
    let c1 = line.cross(src[test_index + 1] - origin);
    c0 * c1 >= 0.0
}

/// Returns the location (in t) of the cubic cusp, if there is one.
fn find_cubic_cusp(src: &[Point; 4]) -> Option<f32> {
    // When the adjacent control point matches the end point, it behaves as if
    // the cubic has a cusp: there's a point of max curvature where the derivative
    // goes to zero. Ideally, this would be where t is zero or one, but math
    // error makes not so. It is not uncommon to create cubics this way; skip them.
    if src[0] == src[1] || src[2] == src[3] {
        return None;
    }
    // Cubics only have a cusp if the line segments formed by the control and end points cross.
    // Detect crossing if line ends are on opposite sides of plane formed by the other line.
    if on_same_side(src, 0, 2) || on_same_side(src, 2, 0) {
        return None;
    }
    // Cubics may have multiple points of maximum curvature, although at most only
    // one is a cusp.
    for test_t in find_cubic_max_curvature(src) {
        // No need to consider max curvature on the end.
        if 0.0 >= test_t || test_t >= 1.0 {
            continue;
        }
        // A cusp is at the max curvature, and also has a derivative close to zero.
        // Choose the 'close to zero' meaning by comparing the derivative length
        // with the overall cubic size.
        let d_pt = eval_cubic_derivative(src, test_t);
        let precision = ((src[1] - src[0]).square_length()
            + (src[2] - src[1]).square_length()
            + (src[3] - src[2]).square_length())
            * 1e-8;
        if d_pt.square_length() < precision {
            // All three max curvature t values may be close to the cusp;
            // return the first one.
            return Some(test_t);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::geom::{CubicBezierSegment, QuadraticBezierSegment};
    use core::units::point;

    fn path(build: impl FnOnce(&mut core::path::Builder)) -> Path {
        let mut builder = Path::builder();
        build(&mut builder);
        builder.build()
    }

    /// Flattens the path and calls the callback for each sub-path.
    fn for_each_polygon(path: &Path, tolerance: f32, mut cb: impl FnMut(&[Point])) {
        let mut polygon = Vec::new();
        for event in path.iter() {
            match event {
                PathEvent::Begin { at } => {
                    polygon.clear();
                    polygon.push(at);
                }
                PathEvent::Line { to, .. } => polygon.push(to),
                PathEvent::Quadratic { from, ctrl, to } => {
                    QuadraticBezierSegment { from, ctrl, to }
                        .for_each_flattened(tolerance, &mut |s| polygon.push(s.to));
                }
                PathEvent::Cubic {
                    from,
                    ctrl1,
                    ctrl2,
                    to,
                } => {
                    CubicBezierSegment {
                        from,
                        ctrl1,
                        ctrl2,
                        to,
                    }
                    .for_each_flattened(tolerance, &mut |s| polygon.push(s.to));
                }
                PathEvent::End { .. } => cb(&polygon),
            }
        }
    }

    /// Signed area of each sub-path.
    fn signed_areas(path: &Path) -> Vec<f32> {
        let mut areas = Vec::new();
        for_each_polygon(path, 0.001, |polygon| {
            let mut area = 0.0;
            for i in 0..polygon.len() {
                let a = polygon[i].to_vector();
                let b = polygon[(i + 1) % polygon.len()].to_vector();
                area += a.cross(b);
            }
            areas.push(area * 0.5);
        });
        areas
    }

    fn total_area(path: &Path) -> f32 {
        signed_areas(path).iter().sum::<f32>().abs()
    }

    fn stroke(path: &Path, options: &StrokeOptions) -> Path {
        Stroker::new().stroke_to_fill(path.iter(), options)
    }

    fn assert_approx(actual: f32, expected: f32, eps: f32) {
        assert!(
            (actual - expected).abs() <= eps,
            "expected {expected}, got {actual} (eps {eps})"
        );
    }

    #[test]
    fn line_caps() {
        let line = path(|b| {
            b.begin(point(0.0, 0.0));
            b.line_to(point(100.0, 0.0));
            b.end(false);
        });

        let options = StrokeOptions::line_with(10.0).with_line_cap(LineCap::Butt);
        assert_approx(total_area(&stroke(&line, &options)), 1000.0, 0.01);

        let options = options.with_line_cap(LineCap::Square);
        assert_approx(total_area(&stroke(&line, &options)), 1100.0, 0.01);

        let options = options.with_line_cap(LineCap::Round);
        let expected = 1000.0 + std::f32::consts::PI * 25.0;
        assert_approx(total_area(&stroke(&line, &options)), expected, 0.5);
    }

    #[test]
    fn closed_square() {
        let square = path(|b| {
            b.begin(point(0.0, 0.0));
            b.line_to(point(100.0, 0.0));
            b.line_to(point(100.0, 100.0));
            b.line_to(point(0.0, 100.0));
            b.end(true);
        });

        let options = StrokeOptions::line_with(10.0).with_line_join(LineJoin::Miter);
        let areas = signed_areas(&stroke(&square, &options));
        assert_eq!(areas.len(), 2);
        // The outer and inner contours have opposite orientations.
        assert!(areas[0] * areas[1] < 0.0);
        assert_approx(areas[0].abs(), 110.0 * 110.0, 0.01);
        // Inner joins go through the pivot, which adds a 5x5 loop with the opposite
        // orientation at each corner of the inner contour.
        assert_approx(areas[1].abs(), 90.0 * 90.0 - 4.0 * 25.0, 0.01);

        let options = options.with_line_join(LineJoin::Bevel);
        let areas = signed_areas(&stroke(&square, &options));
        assert_approx(areas[0].abs(), 110.0 * 110.0 - 4.0 * 12.5, 0.01);
    }

    #[test]
    fn empty_closed_sub_path() {
        let dot = path(|b| {
            b.begin(point(10.0, 10.0));
            b.end(true);
        });

        let options = StrokeOptions::line_with(10.0).with_line_cap(LineCap::Square);
        assert_approx(total_area(&stroke(&dot, &options)), 100.0, 0.01);

        let options = options.with_line_cap(LineCap::Butt);
        assert_eq!(stroke(&dot, &options).iter().count(), 0);
    }

    /// Checks that the boundary of the stroke of a single curve with butt caps is
    /// within tolerance of the offset curves, and that its area is the arc length
    /// times the width.
    fn check_curve_stroke(curve: &CubicBezierSegment<f32>, width: f32, tolerance: f32) {
        let p = path(|b| {
            b.begin(curve.from);
            b.cubic_bezier_to(curve.ctrl1, curve.ctrl2, curve.to);
            b.end(false);
        });
        let options = StrokeOptions::line_with(width)
            .with_line_cap(LineCap::Butt)
            .with_tolerance(tolerance);
        let stroked = stroke(&p, &options);

        let mut samples = Vec::new();
        let n = 2000;
        for i in 0..=n {
            samples.push(curve.sample(i as f32 / n as f32));
        }
        let distance_to_curve = |p: Point| {
            samples
                .iter()
                .map(|s| (*s - p).length())
                .fold(f32::MAX, f32::min)
        };

        let radius = width * 0.5;
        let sampling_error = curve.approximate_length(0.01) / n as f32;
        for_each_polygon(&stroked, 0.01, |polygon| {
            for i in 0..polygon.len() {
                let a = polygon[i];
                let b = polygon[(i + 1) % polygon.len()];
                for j in 0..10 {
                    let p = a.lerp(b, j as f32 / 10.0);
                    let d = distance_to_curve(p);
                    assert!(
                        d <= radius + tolerance + sampling_error,
                        "point {p:?} is at distance {d} of the curve"
                    );
                }
            }
        });

        let expected = curve.approximate_length(0.001) * width;
        assert_approx(total_area(&stroked), expected, expected * 0.002);
    }

    #[test]
    fn curves() {
        let curves = [
            CubicBezierSegment {
                from: point(0.0, 0.0),
                ctrl1: point(100.0, 0.0),
                ctrl2: point(200.0, 100.0),
                to: point(200.0, 200.0),
            },
            // S-shape (one inflection).
            CubicBezierSegment {
                from: point(0.0, 0.0),
                ctrl1: point(200.0, 0.0),
                ctrl2: point(0.0, 200.0),
                to: point(200.0, 200.0),
            },
            // A quadratic bézier curve.
            QuadraticBezierSegment {
                from: point(0.0, 0.0),
                ctrl: point(150.0, 20.0),
                to: point(300.0, 200.0),
            }
            .to_cubic(),
        ];

        for curve in &curves {
            for tolerance in [0.25, 0.05] {
                check_curve_stroke(curve, 10.0, tolerance);
            }
        }
    }

    #[test]
    fn cusp_adds_a_circle() {
        let p = path(|b| {
            b.begin(point(0.0, 0.0));
            b.cubic_bezier_to(point(100.0, 100.0), point(0.0, 100.0), point(100.0, 0.0));
            b.end(false);
        });
        let options = StrokeOptions::line_with(10.0).with_line_cap(LineCap::Butt);
        let areas = signed_areas(&stroke(&p, &options));
        assert_eq!(areas.len(), 2);
        assert_approx(areas[1].abs(), std::f32::consts::PI * 25.0, 0.5);
    }

    #[test]
    fn same_area_as_default_stroker() {
        let p = path(|b| {
            b.begin(point(0.0, 0.0));
            b.line_to(point(100.0, 0.0));
            b.quadratic_bezier_to(point(200.0, 0.0), point(200.0, 100.0));
            b.cubic_bezier_to(point(200.0, 200.0), point(100.0, 300.0), point(0.0, 200.0));
            b.line_to(point(-50.0, 100.0));
            b.end(false);
        });

        for join in [LineJoin::Miter, LineJoin::Bevel, LineJoin::Round] {
            for cap in [LineCap::Butt, LineCap::Square, LineCap::Round] {
                let options = StrokeOptions::line_with(20.0)
                    .with_line_join(join)
                    .with_line_cap(cap)
                    .with_tolerance(0.05);
                let skia = total_area(&stroke(&p, &options));
                let reference = total_area(&crate::Stroker::new().stroke_to_fill(p.iter(), &options));
                assert_approx(skia, reference, reference * 0.002);
            }
        }
    }
}
