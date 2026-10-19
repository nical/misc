//! Measures how far the output of a stroker is from the ideal stroke.
//!
//! With round joins and caps, the ideal stroke is the set of points within
//! `radius` of the source path. Two one-sided errors are measured:
//! - Overshoot: how far the output extends beyond the ideal stroke. The output
//!   edges are sampled and each sample contributes its distance to the source
//!   path minus the radius.
//! - Undershoot: how far the ideal stroke extends beyond the output. The
//!   boundary of the ideal stroke (offset segments and arcs around vertices of
//!   the flattened source) is sampled. Samples that are not covered by the output
//!   (with the non-zero fill rule) contribute their distance to the output edges.
//!
//! The measurement is only meaningful for round joins and caps.

use core::geom::{CubicBezierSegment, QuadraticBezierSegment};
use core::path::{Path, PathEvent};
use core::units::{Point, Vector};

#[derive(Copy, Clone, Debug, Default)]
pub struct Accuracy {
    pub max_overshoot: f32,
    pub max_undershoot: f32,
    pub samples: u64,
    pub samples_over_tolerance: u64,
    /// Where the maximum errors were measured: (path index, position).
    pub max_overshoot_at: (usize, Point),
    pub max_undershoot_at: (usize, Point),
}

impl Accuracy {
    pub fn merge(&mut self, other: &Accuracy) {
        if other.max_overshoot > self.max_overshoot {
            self.max_overshoot = other.max_overshoot;
            self.max_overshoot_at = other.max_overshoot_at;
        }
        if other.max_undershoot > self.max_undershoot {
            self.max_undershoot = other.max_undershoot;
            self.max_undershoot_at = other.max_undershoot_at;
        }
        self.samples += other.samples;
        self.samples_over_tolerance += other.samples_over_tolerance;
    }
}

/// `path_index` is only used to report where the maximum errors are.
pub fn measure(
    path_index: usize,
    source: &Path,
    stroked: &Path,
    radius: f32,
    tolerance: f32,
) -> Accuracy {
    let mut result = Accuracy::default();

    let flattening_tolerance = tolerance * 0.02;
    // Spacing between samples. The distance functions are 1-lipschitz, so the
    // maximum error between samples is at most half of the spacing above the
    // measured one.
    let spacing = tolerance * 0.25;
    // Searches stop at this distance, errors are clamped to it.
    let max_error = 100.0 * tolerance;

    let source_polylines = flatten(source, flattening_tolerance);
    let output_polylines = flatten(stroked, flattening_tolerance);
    if source_polylines.is_empty() {
        return result;
    }

    let mut source_segments = Vec::new();
    for polyline in &source_polylines {
        polyline.for_each_segment(&mut |a, b| source_segments.push([a, b]));
    }
    let mut output_segments = Vec::new();
    for polyline in &output_polylines {
        // Filled sub-paths are implicitly closed.
        polyline.for_each_edge(&mut |a, b| output_segments.push([a, b]));
    }

    let mut min = Point::new(f32::MAX, f32::MAX);
    let mut max = Point::new(f32::MIN, f32::MIN);
    for seg in source_segments.iter().chain(output_segments.iter()) {
        for p in seg {
            min = min.min(*p);
            max = max.max(*p);
        }
    }
    let margin = Vector::new(radius + 1.0, radius + 1.0);
    let (min, max) = (min - margin, max + margin);

    let source_grid = Grid::new(source_segments, min, max, radius.max(spacing));
    let output_grid = Grid::new(output_segments, min, max, radius.max(spacing));

    let mut add_sample = |error: f32, overshoot: bool, at: Point| {
        result.samples += 1;
        if error > tolerance {
            result.samples_over_tolerance += 1;
        }
        if overshoot && error > result.max_overshoot {
            result.max_overshoot = error;
            result.max_overshoot_at = (path_index, at);
        } else if !overshoot && error > result.max_undershoot {
            result.max_undershoot = error;
            result.max_undershoot_at = (path_index, at);
        }
    };

    // Overshoot.
    for [a, b] in &output_grid.segments {
        let n = ((*b - *a).length() / spacing).ceil().max(1.0) as u32;
        for i in 0..n {
            let p = a.lerp(*b, i as f32 / n as f32);
            let d = source_grid.nearest(p, radius + max_error);
            add_sample((d - radius).max(0.0), true, p);
        }
    }

    // Undershoot.
    let interior_threshold = radius - 2.0 * flattening_tolerance;
    let mut check_ideal_point = |p: Point| {
        if source_grid.nearest(p, radius) < interior_threshold {
            // Inside the ideal stroke, not on its boundary.
            return;
        }
        let d = output_grid.nearest(p, max_error);
        let error = if d <= tolerance * 0.01 || output_grid.winding(p) != 0 {
            // On the boundary (within a negligible distance), or covered.
            d.min(tolerance * 0.01)
        } else {
            d
        };
        add_sample(error, false, p);
    };

    for polyline in &source_polylines {
        let pts = &polyline.points;
        if pts.len() < 2 {
            continue;
        }
        // Offset segments, on both sides.
        polyline.for_each_segment(&mut |a, b| {
            let v = b - a;
            let n = Vector::new(-v.y, v.x) / v.length() * radius;
            let count = (v.length() / spacing).ceil().max(1.0) as u32;
            for i in 0..=count {
                let p = a.lerp(b, i as f32 / count as f32);
                check_ideal_point(p + n);
                check_ideal_point(p - n);
            }
        });

        // Arcs around vertices.
        let len = pts.len();
        for i in 0..len {
            let is_end = !polyline.closed && (i == 0 || i == len - 1);
            if is_end {
                // Round cap: sample the whole circle, the samples on the stroke
                // side are discarded as interior points.
                let count = (std::f32::consts::TAU * radius / spacing).ceil().max(4.0) as u32;
                for j in 0..count {
                    let angle = j as f32 * std::f32::consts::TAU / count as f32;
                    check_ideal_point(pts[i] + Vector::new(angle.cos(), angle.sin()) * radius);
                }
                continue;
            }
            let prev = pts[(i + len - 1) % len];
            let next = pts[(i + 1) % len];
            let t0 = (pts[i] - prev).normalize();
            let t1 = (next - pts[i]).normalize();
            let cross = t0.cross(t1);
            // The join is on the side opposite to the turn.
            let side = if cross > 0.0 { -1.0 } else { 1.0 };
            let n0 = Vector::new(-t0.y, t0.x) * side;
            let n1 = Vector::new(-t1.y, t1.x) * side;
            let start = n0.y.atan2(n0.x);
            let mut sweep = n1.y.atan2(n1.x) - start;
            if sweep > std::f32::consts::PI {
                sweep -= std::f32::consts::TAU;
            }
            if sweep < -std::f32::consts::PI {
                sweep += std::f32::consts::TAU;
            }
            let count = (sweep.abs() * radius / spacing).ceil() as u32;
            for j in 1..count {
                let angle = start + sweep * j as f32 / count as f32;
                check_ideal_point(pts[i] + Vector::new(angle.cos(), angle.sin()) * radius);
            }
        }
    }

    result
}

struct Polyline {
    points: Vec<Point>,
    closed: bool,
}

impl Polyline {
    /// Calls the callback for each segment, including the closing segment of
    /// closed polylines. Single points are reported as a degenerate segment.
    fn for_each_segment(&self, cb: &mut dyn FnMut(Point, Point)) {
        if self.points.len() == 1 {
            cb(self.points[0], self.points[0]);
            return;
        }
        for w in self.points.windows(2) {
            cb(w[0], w[1]);
        }
        if self.closed {
            cb(*self.points.last().unwrap(), self.points[0]);
        }
    }

    /// Calls the callback for each edge of the filled polygon.
    fn for_each_edge(&self, cb: &mut dyn FnMut(Point, Point)) {
        let n = self.points.len();
        for i in 0..n {
            cb(self.points[i], self.points[(i + 1) % n]);
        }
    }
}

/// Flattens the path, removing consecutive duplicate points.
fn flatten(path: &Path, tolerance: f32) -> Vec<Polyline> {
    let mut polylines = Vec::new();
    let mut points = Vec::new();
    let push = |points: &mut Vec<Point>, p: Point| {
        if points.last() != Some(&p) {
            points.push(p);
        }
    };
    for event in path.iter() {
        match event {
            PathEvent::Begin { at } => {
                points.clear();
                points.push(at);
            }
            PathEvent::Line { to, .. } => push(&mut points, to),
            PathEvent::Quadratic { from, ctrl, to } => {
                QuadraticBezierSegment { from, ctrl, to }
                    .for_each_flattened(tolerance, &mut |s| push(&mut points, s.to));
            }
            PathEvent::Cubic { from, ctrl1, ctrl2, to } => {
                CubicBezierSegment { from, ctrl1, ctrl2, to }
                    .for_each_flattened(tolerance, &mut |s| push(&mut points, s.to));
            }
            PathEvent::End { close, .. } => {
                if close && points.len() > 1 && points.last() == points.first() {
                    points.pop();
                }
                polylines.push(Polyline {
                    points: std::mem::take(&mut points),
                    closed: close,
                });
            }
        }
    }
    polylines
}

/// A uniform grid of segments, for distance queries and winding numbers.
struct Grid {
    min: Point,
    cell_size: f32,
    cols: i32,
    rows: i32,
    /// Index of the first item of each cell in `items`, plus one at the end.
    cell_starts: Vec<u32>,
    items: Vec<u32>,
    segments: Vec<[Point; 2]>,
}

const MAX_GRID_SIZE: f32 = 1024.0;

impl Grid {
    fn new(segments: Vec<[Point; 2]>, min: Point, max: Point, cell_size_hint: f32) -> Self {
        let size = max - min;
        let cell_size = cell_size_hint
            .max(size.x / MAX_GRID_SIZE)
            .max(size.y / MAX_GRID_SIZE)
            .max(1e-3);
        let cols = ((size.x / cell_size).ceil() as i32).max(1);
        let rows = ((size.y / cell_size).ceil() as i32).max(1);

        let mut grid = Grid {
            min,
            cell_size,
            cols,
            rows,
            cell_starts: vec![0; (cols * rows) as usize + 1],
            items: Vec::new(),
            segments: Vec::new(),
        };

        // Count, then fill.
        let mut counts = vec![0u32; (cols * rows) as usize];
        for seg in &segments {
            grid.for_each_cell(seg, &mut |cell| counts[cell] += 1);
        }
        let mut offset = 0;
        for (i, count) in counts.iter().enumerate() {
            grid.cell_starts[i] = offset;
            offset += count;
        }
        grid.cell_starts[counts.len()] = offset;
        grid.items = vec![0; offset as usize];
        let mut cursors = grid.cell_starts.clone();
        for (idx, seg) in segments.iter().enumerate() {
            let items = &mut grid.items;
            let cursors = &mut cursors;
            grid_for_each_cell(min, cell_size, cols, rows, seg, &mut |cell| {
                items[cursors[cell] as usize] = idx as u32;
                cursors[cell] += 1;
            });
        }
        grid.segments = segments;

        grid
    }

    fn for_each_cell(&self, seg: &[Point; 2], cb: &mut dyn FnMut(usize)) {
        grid_for_each_cell(self.min, self.cell_size, self.cols, self.rows, seg, cb);
    }

    fn cell_coord(&self, v: f32, min: f32, count: i32) -> i32 {
        grid_coord(v, min, self.cell_size, count)
    }

    fn cell_items(&self, x: i32, y: i32) -> &[u32] {
        let cell = (y * self.cols + x) as usize;
        &self.items[self.cell_starts[cell] as usize..self.cell_starts[cell + 1] as usize]
    }

    /// Returns the distance to the nearest segment, or `max_dist` if there is
    /// no segment closer than that.
    fn nearest(&self, p: Point, max_dist: f32) -> f32 {
        let cx = self.cell_coord(p.x, self.min.x, self.cols);
        let cy = self.cell_coord(p.y, self.min.y, self.rows);
        let mut best = max_dist * max_dist;
        let max_ring = self.cols.max(self.rows);
        for k in 0..=max_ring {
            // Cells in ring k are at least (k - 1) cells away from p.
            let ring_dist = (k - 1).max(0) as f32 * self.cell_size;
            if ring_dist * ring_dist >= best {
                break;
            }
            let mut visit = |x: i32, y: i32| {
                if x < 0 || y < 0 || x >= self.cols || y >= self.rows {
                    return;
                }
                for &idx in self.cell_items(x, y) {
                    let [a, b] = self.segments[idx as usize];
                    best = best.min(square_distance_to_segment(p, a, b));
                }
            };
            if k == 0 {
                visit(cx, cy);
                continue;
            }
            for x in cx - k..=cx + k {
                visit(x, cy - k);
                visit(x, cy + k);
            }
            for y in cy - k + 1..cy + k {
                visit(cx - k, y);
                visit(cx + k, y);
            }
        }
        best.sqrt()
    }

    /// Winding number of the polygons at p.
    fn winding(&self, p: Point) -> i32 {
        let row = ((p.y - self.min.y) / self.cell_size).floor() as i32;
        if row < 0 || row >= self.rows {
            return 0;
        }
        let first_col = self.cell_coord(p.x, self.min.x, self.cols);
        let mut winding = 0;
        for col in first_col..self.cols {
            for &idx in self.cell_items(col, row) {
                let [a, b] = self.segments[idx as usize];
                if (a.y <= p.y) == (b.y <= p.y) {
                    continue;
                }
                let x = a.x + (p.y - a.y) * (b.x - a.x) / (b.y - a.y);
                if x <= p.x {
                    continue;
                }
                // Segments can be in several cells of the row. Only count them
                // in the first visited one.
                let (c0, _) = row_col_range(self.min, self.cell_size, self.cols, &[a, b], row);
                if c0.max(first_col) != col {
                    continue;
                }
                winding += if b.y > a.y { 1 } else { -1 };
            }
        }
        winding
    }
}

/// Calls the callback for each cell that the segment touches.
fn grid_for_each_cell(
    min: Point,
    cell_size: f32,
    cols: i32,
    rows: i32,
    seg: &[Point; 2],
    cb: &mut dyn FnMut(usize),
) {
    let [a, b] = *seg;
    let r0 = grid_coord(a.y.min(b.y), min.y, cell_size, rows);
    let r1 = grid_coord(a.y.max(b.y), min.y, cell_size, rows);
    for row in r0..=r1 {
        let (c0, c1) = row_col_range(min, cell_size, cols, seg, row);
        for col in c0..=c1 {
            cb((row * cols + col) as usize);
        }
    }
}

fn grid_coord(v: f32, min: f32, cell_size: f32, count: i32) -> i32 {
    (((v - min) / cell_size).floor() as i32).clamp(0, count - 1)
}

/// Range of columns of the cells that the segment touches in a row.
fn row_col_range(min: Point, cell_size: f32, cols: i32, seg: &[Point; 2], row: i32) -> (i32, i32) {
    let [a, b] = *seg;
    // Portion of the segment within the row.
    let y0 = min.y + row as f32 * cell_size;
    let y1 = y0 + cell_size;
    let (x0, x1) = if a.y == b.y {
        (a.x.min(b.x), a.x.max(b.x))
    } else {
        let x_at = |y: f32| {
            let t = ((y - a.y) / (b.y - a.y)).clamp(0.0, 1.0);
            a.x + (b.x - a.x) * t
        };
        let (xa, xb) = (x_at(y0), x_at(y1));
        (xa.min(xb), xa.max(xb))
    };
    (
        grid_coord(x0, min.x, cell_size, cols),
        grid_coord(x1, min.x, cell_size, cols),
    )
}

fn square_distance_to_segment(p: Point, a: Point, b: Point) -> f32 {
    let ab = b - a;
    let len2 = ab.square_length();
    let t = if len2 > 0.0 {
        ((p - a).dot(ab) / len2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (a + ab * t - p).square_length()
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::units::point;

    /// A polygon approximating the ideal round-capped stroke of the segment
    /// (0, 0) -> (100, 0), with its radius offset by `delta`.
    fn capsule(radius: f32, delta: f32) -> Path {
        let r = radius + delta;
        let mut builder = Path::builder();
        let n = 2000;
        builder.begin(point(0.0, -r));
        builder.line_to(point(100.0, -r));
        for i in 1..n {
            let a = -std::f32::consts::FRAC_PI_2 + std::f32::consts::PI * i as f32 / n as f32;
            builder.line_to(point(100.0 + a.cos() * r, a.sin() * r));
        }
        builder.line_to(point(100.0, r));
        builder.line_to(point(0.0, r));
        for i in 1..n {
            let a = std::f32::consts::FRAC_PI_2 + std::f32::consts::PI * i as f32 / n as f32;
            builder.line_to(point(a.cos() * r, a.sin() * r));
        }
        builder.end(true);
        builder.build()
    }

    fn segment() -> Path {
        let mut builder = Path::builder();
        builder.begin(point(0.0, 0.0));
        builder.line_to(point(100.0, 0.0));
        builder.end(false);
        builder.build()
    }

    #[test]
    fn exact_output() {
        let result = measure(0, &segment(), &capsule(10.0, 0.0), 10.0, 0.1);
        assert!(result.max_overshoot < 0.01, "{result:?}");
        assert!(result.max_undershoot < 0.01, "{result:?}");
        assert_eq!(result.samples_over_tolerance, 0);
    }

    #[test]
    fn overshoot() {
        let result = measure(0, &segment(), &capsule(10.0, 1.0), 10.0, 0.1);
        assert!((result.max_overshoot - 1.0).abs() < 0.02, "{result:?}");
        assert!(result.max_undershoot < 0.01, "{result:?}");
    }

    #[test]
    fn undershoot() {
        let result = measure(0, &segment(), &capsule(10.0, -0.5), 10.0, 0.1);
        assert!(result.max_overshoot < 0.01, "{result:?}");
        assert!((result.max_undershoot - 0.5).abs() < 0.02, "{result:?}");
    }

    #[test]
    fn covered_inner_side_is_not_an_error() {
        // The inner offset of a tight turn is covered by the other side of the stroke.
        let mut builder = Path::builder();
        builder.begin(point(0.0, 0.0));
        builder.line_to(point(100.0, 0.0));
        builder.line_to(point(0.0, 5.0));
        builder.end(false);
        let source = builder.build();
        let options = stroke::StrokeOptions::line_with(20.0)
            .with_line_join(stroke::LineJoin::Round)
            .with_line_cap(stroke::LineCap::Round)
            .with_tolerance(0.1);
        let stroked = stroke::skia::Stroker::new().stroke_to_fill(source.iter(), &options);
        let result = measure(0, &source, &stroked, 10.0, 0.1);
        assert!(result.max_overshoot < 0.1, "{result:?}");
        assert!(result.max_undershoot < 0.1, "{result:?}");
    }
}
