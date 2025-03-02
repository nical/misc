extern crate lyon_path;
#[macro_use]
extern crate criterion;

use criterion::Criterion;
use lyon_path::geom::euclid::point2 as point;
use lyon_path::geom::{CubicBezierSegment, QuadraticBezierSegment};
use lyon_path::geom::euclid::default::{Box2D, Transform2D};
use lyon_path::math::Point;
use lyon_path::{Path, PathEvent};

use flatten::{Flatten};

const N: usize = 1;

fn bench_flatten<A: Flatten>(curves: &[CubicBezierSegment<f32>], tolerance: f32) {
    for _ in 0..N {
        for curve in curves {
            A::cubic(&curve, tolerance, &mut |seg| {
                std::hint::black_box(seg);
            });
        }
    }
}

fn bench_flatten_quad<A: Flatten>(curves: &[QuadraticBezierSegment<f32>], tolerance: f32) {
    for _ in 0..N {
        for curve in curves {
            A::quadratic(&curve, tolerance, &mut |seg| {
                std::hint::black_box(seg);
            });
        }
    }
}

use criterion::BenchmarkId;

static TOLERANCES: [f32; 10] = [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.5, 1.0];

fn cubic_flatten(c: &mut Criterion) {
    let curves = generate_bezier_curves();
    let mut g = c.benchmark_group("cubic");
    for tol in &TOLERANCES {
        g.bench_with_input(BenchmarkId::new("recursive", tol), tol, |b, tol| { b.iter(|| bench_flatten::<flatten::Recursive>(&curves, *tol)) });
        g.bench_with_input(BenchmarkId::new("fwd-iff", tol), tol, |b, tol| { b.iter(|| bench_flatten::<flatten::FwdDiff>(&curves, *tol)) });
        g.bench_with_input(BenchmarkId::new("hfd", tol), tol, |b, tol| { b.iter(|| bench_flatten::<flatten::HybridFwdDiff>(&curves, *tol)) });
        g.bench_with_input(BenchmarkId::new("pa", tol), tol, |b, tol| { b.iter(|| bench_flatten::<flatten::ParabolaApprox>(&curves, *tol)) });
        g.bench_with_input(BenchmarkId::new("levien", tol), tol, |b, tol| { b.iter(|| bench_flatten::<flatten::Levien>(&curves, *tol)) });
        g.bench_with_input(BenchmarkId::new("linear", tol), tol, |b, tol| { b.iter(|| bench_flatten::<flatten::Linear>(&curves, *tol)) });
        g.bench_with_input(BenchmarkId::new("sedebrg", tol), tol, |b, tol| { b.iter(|| bench_flatten::<flatten::Sedeberg>(&curves, *tol)) });
    }
}

fn quad_flatten(c: &mut Criterion) {
    let curves = generate_quadratic_curves();
    let mut g = c.benchmark_group("quadratic");
    for tol in &TOLERANCES {
        g.bench_with_input(BenchmarkId::new("recursive", tol), tol, |b, tol| { b.iter(|| bench_flatten_quad::<flatten::Recursive>(&curves, *tol)) });
        g.bench_with_input(BenchmarkId::new("fwd-diff", tol), tol, |b, tol| { b.iter(|| bench_flatten_quad::<flatten::FwdDiff>(&curves, *tol)) });
        g.bench_with_input(BenchmarkId::new("levien", tol), tol, |b, tol| { b.iter(|| bench_flatten_quad::<flatten::Levien>(&curves, *tol)) });
        g.bench_with_input(BenchmarkId::new("levien-simd", tol), tol, |b, tol| { b.iter(|| bench_flatten_quad::<flatten::LevienSimd>(&curves, *tol)) });
        g.bench_with_input(BenchmarkId::new("linear2", tol), tol, |b, tol| { b.iter(|| bench_flatten_quad::<flatten::Linear>(&curves, *tol)) });
        g.bench_with_input(BenchmarkId::new("cagd", tol), tol, |b, tol| { b.iter(|| bench_flatten_quad::<flatten::Sedeberg>(&curves, *tol)) });
    }
}


pub fn generate_bezier_curves() -> Vec<CubicBezierSegment<f32>> {
    let (_, paths) = load_svg("tiger.svg", 1.0);

    let mut curves = Vec::new();
    for (path, _) in paths {
        for evt in path.iter() {
            match evt {
                PathEvent::Cubic { from, ctrl1, ctrl2, to } => {
                    curves.push(CubicBezierSegment { from, ctrl1, ctrl2, to });
                }
                PathEvent::Quadratic { from, ctrl, to } => {
                    curves.push(QuadraticBezierSegment { from, ctrl, to }.to_cubic());
                }
                _ => {}
            }
        }
    }

    curves
}

pub fn generate_quadratic_curves() -> Vec<QuadraticBezierSegment<f32>> {
    let cubics = generate_bezier_curves();
    let mut quads = Vec::new();
    for cubic in &cubics {
        cubic.for_each_quadratic_bezier(0.25, &mut |quad| {
            quads.push(*quad);
        })
    }

    quads
}


pub type Color = (u8, u8, u8, u8);

pub const FALLBACK_COLOR: Color = (0, 255, 0, 255 );

#[derive(Clone, Debug)]
pub enum SvgPattern {
    Color(Color),
    Gradient { color0: Color, color1: Color, from: Point, to: Point },
}

pub fn load_svg(filename: &str, scale_factor: f32) -> (Box2D<f32>, Vec<(Path, SvgPattern)>) {
    let opt = usvg::Options::default();
    let file_data = std::fs::read(filename).unwrap();
    let rtree = usvg::Tree::from_data(&file_data, &opt).unwrap();
    let mut paths = Vec::new();

    let s = scale_factor;

    let mut gradients = std::collections::HashMap::new();

    let view_box = rtree.svg_node().view_box;
    for node in rtree.root().descendants() {
        use usvg::NodeExt;
        let t = node.transform();
        let transform = Transform2D::new(
            t.a as f32, t.b as f32,
            t.c as f32, t.d as f32,
            t.e as f32, t.f as f32,
        );

        match *node.borrow() {
            usvg::NodeKind::LinearGradient(ref gradient) => {
                let color0 = gradient.base.stops.first().map(|stop| {
                    (
                        stop.color.red,
                        stop.color.green,
                        stop.color.blue,
                        (stop.opacity.value() * 255.0) as u8,
                    )
                }).unwrap_or(FALLBACK_COLOR);
                let color1 = gradient.base.stops.last().map(|stop| {
                    (
                        stop.color.red,
                        stop.color.green,
                        stop.color.blue,
                        (stop.opacity.value() * 255.0) as u8,
                    )
                }).unwrap_or(FALLBACK_COLOR);
                gradients.insert(gradient.id.clone(), SvgPattern::Gradient {
                    color0,
                    color1,
                    from: point(gradient.x1 as f32, gradient.y1 as f32),
                    to: point(gradient.x2 as f32, gradient.y2 as f32),
                });
            }
            usvg::NodeKind::Path(ref usvg_path) => {
                let pattern = match usvg_path.fill {
                    Some(ref fill) => {
                        match fill.paint {
                            usvg::Paint::Color(c) => SvgPattern::Color((c.red, c.green, c.blue, 255)),
                            usvg::Paint::Link(ref id) => {
                                gradients.get(id).cloned().unwrap_or_else(|| {
                                    println!("Could not find pattern {:?}", id);
                                    SvgPattern::Color(FALLBACK_COLOR)
                                })
                            }
                        }
                    }
                    None => {
                        continue;
                    }
                };

                let mut builder = Path::builder().with_svg();
                for segment in usvg_path.data.iter() {
                    match *segment {
                        usvg::PathSegment::MoveTo { x, y } => {
                            builder.move_to(transform.transform_point(point(x as f32, y as f32)) * s);
                        }
                        usvg::PathSegment::LineTo { x, y } => {
                            builder.line_to(transform.transform_point(point(x as f32, y as f32)) * s);
                        }
                        usvg::PathSegment::CurveTo { x1, y1, x2, y2, x, y, } => {
                            builder.cubic_bezier_to(
                                transform.transform_point(point(x1 as f32, y1 as f32)) * s,
                                transform.transform_point(point(x2 as f32, y2 as f32)) * s,
                                transform.transform_point(point(x as f32, y as f32)) * s,
                            );
                        }
                        usvg::PathSegment::ClosePath => {
                            builder.close();
                        }
                    }
                }
                let path = builder.build();

                paths.push((path, pattern));
            }
            _ => {}
        }
    }

    let vb = Box2D {
        min: point(
            view_box.rect.x() as f32 * s,
            view_box.rect.y() as f32 * s,
        ),
        max: point(
            view_box.rect.x() as f32 + view_box.rect.width() as f32 * s,
            view_box.rect.y() as f32 + view_box.rect.height() as f32 * s,
        ),
    };

    (vb, paths)
}




criterion_group!(flatten, cubic_flatten, quad_flatten );
criterion_main!(flatten);
