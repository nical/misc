extern crate lyon_path;
#[macro_use]
extern crate criterion;

use criterion::Criterion;
use lyon_path::geom::{CubicBezierSegment, QuadraticBezierSegment};
use flatten::testing::*;

use flatten::Flatten;

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
        g.bench_with_input(BenchmarkId::new("fwd-diff", tol), tol, |b, tol| { b.iter(|| bench_flatten::<flatten::FwdDiff>(&curves, *tol)) });
        g.bench_with_input(BenchmarkId::new("hfd", tol), tol, |b, tol| { b.iter(|| bench_flatten::<flatten::HybridFwdDiff>(&curves, *tol)) });
        g.bench_with_input(BenchmarkId::new("hain", tol), tol, |b, tol| { b.iter(|| bench_flatten::<flatten::Hain>(&curves, *tol)) });
        g.bench_with_input(BenchmarkId::new("levien", tol), tol, |b, tol| { b.iter(|| bench_flatten::<flatten::Levien>(&curves, *tol)) });
        g.bench_with_input(BenchmarkId::new("levien-simd", tol), tol, |b, tol| { b.iter(|| bench_flatten::<flatten::LevienSimd>(&curves, *tol)) });
        g.bench_with_input(BenchmarkId::new("levien-quads", tol), tol, |b, tol| { b.iter(|| bench_flatten::<flatten::LevienQuads>(&curves, *tol)) });
        g.bench_with_input(BenchmarkId::new("linear", tol), tol, |b, tol| { b.iter(|| bench_flatten::<flatten::Linear>(&curves, *tol)) });
        g.bench_with_input(BenchmarkId::new("linear-agg", tol), tol, |b, tol| { b.iter(|| bench_flatten::<flatten::LinearAgg>(&curves, *tol)) });
        g.bench_with_input(BenchmarkId::new("linear-hfd", tol), tol, |b, tol| { b.iter(|| bench_flatten::<flatten::LinearHfd>(&curves, *tol)) });
        g.bench_with_input(BenchmarkId::new("wang", tol), tol, |b, tol| { b.iter(|| bench_flatten::<flatten::Wang>(&curves, *tol)) });
        g.bench_with_input(BenchmarkId::new("fixed-16", tol), tol, |b, tol| { b.iter(|| bench_flatten::<flatten::Fixed16>(&curves, *tol)) });
    }
}

fn quad_flatten(c: &mut Criterion) {
    let curves = generate_quadratic_curves();
    let mut g = c.benchmark_group("quadratic");
    for tol in &TOLERANCES {
        g.bench_with_input(BenchmarkId::new("recursive", tol), tol, |b, tol| { b.iter(|| bench_flatten_quad::<flatten::Recursive>(&curves, *tol)) });
        g.bench_with_input(BenchmarkId::new("fwd-diff", tol), tol, |b, tol| { b.iter(|| bench_flatten_quad::<flatten::FwdDiff>(&curves, *tol)) });
        g.bench_with_input(BenchmarkId::new("levien", tol), tol, |b, tol| { b.iter(|| bench_flatten_quad::<flatten::LevienQuads>(&curves, *tol)) });
        g.bench_with_input(BenchmarkId::new("levien-simd", tol), tol, |b, tol| { b.iter(|| bench_flatten_quad::<flatten::LevienSimd>(&curves, *tol)) });
        g.bench_with_input(BenchmarkId::new("linear", tol), tol, |b, tol| { b.iter(|| bench_flatten_quad::<flatten::Linear>(&curves, *tol)) });
        g.bench_with_input(BenchmarkId::new("wang", tol), tol, |b, tol| { b.iter(|| bench_flatten_quad::<flatten::Wang>(&curves, *tol)) });
    }
}




criterion_group!(flatten, cubic_flatten, quad_flatten );
criterion_main!(flatten);
