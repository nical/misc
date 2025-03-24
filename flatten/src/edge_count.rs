use lyon_path::geom::{QuadraticBezierSegment, CubicBezierSegment};
use crate::{Fixed16, Flatten, FwdDiff, Hain, HybridFwdDiff, Kurbo, Levien, Levien37, Levien55, LevienQuads, LevienSimd, Linear, LinearAgg, LinearHfd, Recursive, RecursiveAgg, RecursiveHfd, Wang};
use crate::testing::*;

static TOLERANCES: [f32; 10] = [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.5, 1.0];

fn count_edges_cubic<F: Flatten>(curves: &[CubicBezierSegment<f32>], tolerance: f32) -> u32 {
    let mut count = 0;
    for curve in curves {
        let start = count;
        F::cubic(curve, tolerance, &mut |_,| {
            count += 1;
            assert!(count - start < 5000, "too many edges on {curve:?} tolerance {tolerance:?}");
        });
    }

    count
}

fn count_edges_quad<F: Flatten>(curves: &[QuadraticBezierSegment<f32>], tolerance: f32) -> u32 {
    let mut count = 0;
    for curve in curves {
        F::quadratic(curve, tolerance, &mut |_| { count += 1; });
    }

    count
}

#[test]
fn flatten_edge_count() {
    let curves = generate_bezier_curves();
    let mut hain = Vec::new();
    let mut rec = Vec::new();
    let mut rec_agg = Vec::new();
    let mut rec_hfd = Vec::new();
    let mut linear = Vec::new();
    let mut linear_hfd = Vec::new();
    let mut linear_agg = Vec::new();
    let mut levien19 = Vec::new();
    let mut kurbo = Vec::new();
    let mut levien_partial = Vec::new();
    let mut levien_simd = Vec::new();
    let mut levien37 = Vec::new();
    let mut levien55 = Vec::new();
    let mut fd = Vec::new();
    let mut hfd = Vec::new();
    let mut wang = Vec::new();
    let mut fixed_16 = Vec::new();
    for tolerance in TOLERANCES {
        hain.push(count_edges_cubic::<Hain>(&curves, tolerance));
        rec.push(count_edges_cubic::<Recursive>(&curves, tolerance));
        rec_hfd.push(count_edges_cubic::<RecursiveHfd>(&curves, tolerance));
        rec_agg.push(count_edges_cubic::<RecursiveAgg>(&curves, tolerance));
        linear.push(count_edges_cubic::<Linear>(&curves, tolerance));
        linear_hfd.push(count_edges_cubic::<LinearHfd>(&curves, tolerance));
        linear_agg.push(count_edges_cubic::<LinearAgg>(&curves, tolerance));
        levien19.push(count_edges_cubic::<LevienQuads>(&curves, tolerance));
        levien_simd.push(count_edges_cubic::<LevienSimd>(&curves, tolerance));
        kurbo.push(count_edges_cubic::<Kurbo>(&curves, tolerance));
        levien_partial.push(count_edges_cubic::<Levien>(&curves, tolerance));
        levien37.push(count_edges_cubic::<Levien37>(&curves, tolerance));
        levien55.push(count_edges_cubic::<Levien55>(&curves, tolerance));
        fd.push(count_edges_cubic::<FwdDiff>(&curves, tolerance));
        hfd.push(count_edges_cubic::<HybridFwdDiff>(&curves, tolerance));
        wang.push(count_edges_cubic::<Wang>(&curves, tolerance));
        fixed_16.push(count_edges_cubic::<Fixed16>(&curves, tolerance));
    }

    fn print_first_row_md() {
        print!("| tolerance  ");
        for tolerance in &TOLERANCES {
            print!("|  {:.2} ", tolerance);
        }
        println!("|");
        print!("|-----------");
        for _ in 0..TOLERANCES.len() {
            print!("| -----:");
        }
        println!("|");
    }

    fn print_edges_md(name: &str, vals: &[u32]) {
        print!("|{}", name);
        for val in vals {
            print!("| {:.2} ", val);
        }
        println!("|");
    }

    fn _print_first_row_csv() {
        print!("tolerance, ");
        for tolerance in &TOLERANCES {
            print!("{}, ", tolerance);
        }
        println!("");
    }

    fn _print_edges_csv(name: &str, vals: &[u32]) {
        print!("{}, ", name);
        for val in vals {
            print!("{:.2}, ", val);
        }
        println!(",");
    }

    println!("Cubic bézier curves:\n");
    print_first_row_md();
    print_edges_md(" recursive    ", &rec);
    print_edges_md(" recursive-agg", &rec_agg);
    print_edges_md(" recursive-hfd", &rec_hfd);
    print_edges_md(" linear       ", &linear);
    print_edges_md(" linear-agg   ", &linear_agg);
    print_edges_md(" linear-hfd   ", &linear_hfd);
    print_edges_md(" levien       ", &levien_partial);
    print_edges_md(" levien-simd  ", &levien_simd);
    print_edges_md(" kurbo        ", &kurbo);
    print_edges_md(" levien-quads ", &levien19);
    //print_edges_md(" levien-37  ", &levien37);
    //print_edges_md(" levien-55  ", &levien55);
    print_edges_md(" hain         ", &hain);
    print_edges_md(" wang         ", &wang);
    print_edges_md(" fwd-diff     ", &fd);
    print_edges_md(" hfd          ", &hfd);
    print_edges_md(" fixed-16     ", &fixed_16);

    println!();

    let curves = generate_quadratic_curves();
    let mut rec = Vec::new();
    let mut levien = Vec::new();
    let mut fd = Vec::new();
    let mut cagd = Vec::new();
    let mut lin = Vec::new();
    for tolerance in TOLERANCES {
        rec.push(count_edges_quad::<Recursive>(&curves, tolerance));
        levien.push(count_edges_quad::<LevienQuads>(&curves, tolerance));
        fd.push(count_edges_quad::<FwdDiff>(&curves, tolerance));
        cagd.push(count_edges_quad::<Wang>(&curves, tolerance));
        lin.push(count_edges_quad::<Linear>(&curves, tolerance));
    }

    println!("");
    println!("Quadratic bézier curves:\n");
    print_first_row_md();
    print_edges_md(" recursive ", &rec);
    print_edges_md(" linear    ", &lin);
    print_edges_md(" wang      ", &cagd);
    print_edges_md(" levien    ", &levien);
    print_edges_md(" fwd-diff  ", &fd);
}
