//! Compares the output size and the speed of the stroke-to-fill implementations.
//!
//! Usage: stroke_bench [--no-accuracy] [--accuracy-stride N] [file.svg...]
//!
//! Every path of each SVG file is stroked (fills included, to get more
//! geometry) with a range of widths, joins and tolerances.
//!
//! The accuracy of the output is then measured with round joins and caps (see
//! the `accuracy` module), on every Nth path if `--accuracy-stride` is set.

mod accuracy;

use core::geom::{CubicBezierSegment, QuadraticBezierSegment};
use core::path::{Path, PathEvent};
use core::units::point;
use std::time::{Duration, Instant};
use stroke::{LineCap, LineJoin, StrokeOptions};

#[derive(Copy, Clone, Default)]
struct Counts {
    sub_paths: usize,
    lines: usize,
    quads: usize,
    cubics: usize,
    /// Number of edges after flattening the output with the stroke tolerance.
    edges: usize,
}

impl Counts {
    fn add(&mut self, path: &Path, tolerance: f32) {
        for event in path.iter() {
            match event {
                PathEvent::Begin { .. } => self.sub_paths += 1,
                PathEvent::Line { .. } => {
                    self.lines += 1;
                    self.edges += 1;
                }
                PathEvent::Quadratic { from, ctrl, to } => {
                    self.quads += 1;
                    QuadraticBezierSegment { from, ctrl, to }
                        .for_each_flattened(tolerance, &mut |_| self.edges += 1);
                }
                PathEvent::Cubic { from, ctrl1, ctrl2, to } => {
                    self.cubics += 1;
                    CubicBezierSegment { from, ctrl1, ctrl2, to }
                        .for_each_flattened(tolerance, &mut |_| self.edges += 1);
                }
                PathEvent::End { close, .. } => {
                    // The closing edge.
                    if close {
                        self.edges += 1;
                    }
                }
            }
        }
    }

    fn segments(&self) -> usize {
        self.lines + self.quads + self.cubics
    }
}

trait Implementation {
    fn name(&self) -> &'static str;
    fn stroke(&mut self, path: &Path, options: &StrokeOptions) -> Path;
}

struct Default(stroke::Stroker);
impl Implementation for Default {
    fn name(&self) -> &'static str {
        "default"
    }
    fn stroke(&mut self, path: &Path, options: &StrokeOptions) -> Path {
        self.0.stroke_to_fill(path.iter(), options)
    }
}

struct Skia(stroke::skia::Stroker);
impl Implementation for Skia {
    fn name(&self) -> &'static str {
        "skia"
    }
    fn stroke(&mut self, path: &Path, options: &StrokeOptions) -> Path {
        self.0.stroke_to_fill(path.iter(), options)
    }
}

fn main() {
    let mut files: Vec<String> = Vec::new();
    let mut measure_accuracy = true;
    let mut accuracy_stride = 1;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--no-accuracy" => measure_accuracy = false,
            "--accuracy-stride" => {
                accuracy_stride = args
                    .next()
                    .and_then(|n| n.parse().ok())
                    .expect("--accuracy-stride expects a number");
            }
            _ => files.push(arg),
        }
    }
    if files.is_empty() {
        let demo = concat!(env!("CARGO_MANIFEST_DIR"), "/../../demo");
        files = ["tiger.svg", "gearflowers.svg", "paris-30k.svg"]
            .iter()
            .map(|f| format!("{demo}/{f}"))
            .collect();
    }

    let mut implementations: Vec<Box<dyn Implementation>> = vec![
        Box::new(Default(stroke::Stroker::new())),
        Box::new(Skia(stroke::skia::Stroker::new())),
    ];

    let styles = [
        ("miter/butt", LineJoin::Miter, LineCap::Butt),
        ("round/round", LineJoin::Round, LineCap::Round),
        ("bevel/square", LineJoin::Bevel, LineCap::Square),
    ];

    for file in &files {
        let paths = load_svg(file);
        let mut input = Counts::default();
        for path in &paths {
            input.add(path, 0.25);
        }
        println!(
            "\n{} ({} paths, {} segments: {} lines, {} quads, {} cubics)",
            file.rsplit('/').next().unwrap(),
            paths.len(),
            input.segments(),
            input.lines,
            input.quads,
            input.cubics,
        );
        println!(
            "  {:<12} {:>5} {:>5} | {:<7} {:>9} {:>9} {:>9} {:>9} {:>10} {:>10}",
            "style", "width", "tol", "impl", "lines", "quads", "cubics", "segments", "edges", "time (ms)"
        );

        for tolerance in [0.25, 0.05] {
            for width in [1.0, 5.0, 25.0] {
                for (style_name, join, cap) in styles {
                    let options = StrokeOptions::line_with(width)
                        .with_line_join(join)
                        .with_line_cap(cap)
                        .with_tolerance(tolerance);

                    for imp in &mut implementations {
                        let mut counts = Counts::default();
                        for path in &paths {
                            counts.add(&imp.stroke(path, &options), tolerance);
                        }
                        let time = measure(|| {
                            for path in &paths {
                                std::hint::black_box(imp.stroke(path, &options));
                            }
                        });
                        println!(
                            "  {:<12} {:>5} {:>5} | {:<7} {:>9} {:>9} {:>9} {:>9} {:>10} {:>10.3}",
                            style_name,
                            width,
                            tolerance,
                            imp.name(),
                            counts.lines,
                            counts.quads,
                            counts.cubics,
                            counts.segments(),
                            counts.edges,
                            time.as_secs_f64() * 1000.0,
                        );
                    }
                }
            }
        }

        if measure_accuracy {
            print_accuracy(&paths, accuracy_stride, &mut implementations);
        }
    }
}

fn print_accuracy(paths: &[Path], stride: usize, implementations: &mut [Box<dyn Implementation>]) {
    println!(
        "\n  accuracy with round joins and caps{}:",
        if stride > 1 { format!(" (every {stride}th path)") } else { String::new() }
    );
    println!(
        "  {:>5} {:>5} | {:<7} {:>13} {:>14} {:>12} {:>15}",
        "width", "tol", "impl", "max overshoot", "max undershoot", "samples", "over tolerance"
    );
    for tolerance in [0.25, 0.05] {
        for width in [1.0, 5.0, 25.0] {
            let options = StrokeOptions::line_with(width)
                .with_line_join(LineJoin::Round)
                .with_line_cap(LineCap::Round)
                .with_tolerance(tolerance);
            for imp in implementations.iter_mut() {
                let selected: Vec<&Path> = paths.iter().step_by(stride.max(1)).collect();
                let stroked: Vec<Path> = selected.iter().map(|p| imp.stroke(p, &options)).collect();
                let result = measure_in_parallel(&selected, &stroked, width * 0.5, tolerance);
                println!(
                    "  {:>5} {:>5} | {:<7} {:>13.4} {:>14.4} {:>12} {:>14.4}%",
                    width,
                    tolerance,
                    imp.name(),
                    result.max_overshoot,
                    result.max_undershoot,
                    result.samples,
                    100.0 * result.samples_over_tolerance as f64 / result.samples.max(1) as f64,
                );
                if std::env::var("STROKE_BENCH_VERBOSE").is_ok() {
                    let (oi, op) = result.max_overshoot_at;
                    let (ui, up) = result.max_undershoot_at;
                    println!(
                        "      worst overshoot: path {} at {:?}, worst undershoot: path {} at {:?}",
                        oi * stride.max(1), op, ui * stride.max(1), up
                    );
                }
            }
        }
    }
}

fn measure_in_parallel(
    paths: &[&Path],
    stroked: &[Path],
    radius: f32,
    tolerance: f32,
) -> accuracy::Accuracy {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut result = accuracy::Accuracy::default();
    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| {
                    let mut result = accuracy::Accuracy::default();
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if i >= paths.len() {
                            break;
                        }
                        result.merge(&accuracy::measure(i, paths[i], &stroked[i], radius, tolerance));
                    }
                    result
                })
            })
            .collect();
        for worker in workers {
            result.merge(&worker.join().unwrap());
        }
    });
    result
}

/// Returns the fastest run, repeating for at least 300ms (and at least 5 times).
fn measure(mut f: impl FnMut()) -> Duration {
    let start = Instant::now();
    let mut best = Duration::MAX;
    let mut runs = 0;
    while runs < 5 || start.elapsed() < Duration::from_millis(300) {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed());
        runs += 1;
    }
    best
}

fn load_svg(filename: &str) -> Vec<Path> {
    use usvg::{NodeExt, TreeParsing};

    let opt = usvg::Options::default();
    let svg_src = std::fs::read_to_string(filename).unwrap();
    let rtree = usvg::Tree::from_str(&svg_src, &opt).unwrap();

    let mut paths = Vec::new();
    for node in rtree.root.descendants() {
        let t = node.abs_transform();
        if let usvg::NodeKind::Path(ref usvg_path) = *node.borrow() {
            let mut builder = Path::builder().with_svg();
            for segment in usvg_path.data.segments() {
                use usvg::tiny_skia_path::PathSegment;
                let p = |pt: usvg::tiny_skia_path::Point| {
                    point(
                        pt.x * t.sx + pt.y * t.kx + t.tx,
                        pt.x * t.ky + pt.y * t.sy + t.ty,
                    )
                };
                match segment {
                    PathSegment::MoveTo(to) => {
                        builder.move_to(p(to));
                    }
                    PathSegment::LineTo(to) => {
                        builder.line_to(p(to));
                    }
                    PathSegment::QuadTo(ctrl, to) => {
                        builder.quadratic_bezier_to(p(ctrl), p(to));
                    }
                    PathSegment::CubicTo(ctrl1, ctrl2, to) => {
                        builder.cubic_bezier_to(p(ctrl1), p(ctrl2), p(to));
                    }
                    PathSegment::Close => {
                        builder.close();
                    }
                }
            }
            paths.push(builder.build());
        }
    }

    paths
}
