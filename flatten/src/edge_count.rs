use lyon_path::geom::{QuadraticBezierSegment, CubicBezierSegment};
use crate::{Flatten, FwdDiff, HybridFwdDiff, Hain, Sedeberg, Recursive, Linear, Levien, Levien37, Levien55};
use crate::testing::*;

static TOLERANCES: [f32; 10] = [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.5, 1.0];

fn count_edges_cubic<F: Flatten>(curves: &[CubicBezierSegment<f32>], tolerance: f32) -> u32 {
    let mut count = 0;
    for curve in curves {
        F::cubic(curve, tolerance, &mut |_,| { count += 1; });
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
    let mut linear = Vec::new();
    let mut levien19 = Vec::new();
    let mut levien37 = Vec::new();
    let mut levien55 = Vec::new();
    let mut fd = Vec::new();
    let mut hfd = Vec::new();
    let mut sedeberg = Vec::new();
    for tolerance in TOLERANCES {
        hain.push(count_edges_cubic::<Hain>(&curves, tolerance));
        rec.push(count_edges_cubic::<Recursive>(&curves, tolerance));
        linear.push(count_edges_cubic::<Linear>(&curves, tolerance));
        levien19.push(count_edges_cubic::<Levien>(&curves, tolerance));
        levien37.push(count_edges_cubic::<Levien37>(&curves, tolerance));
        levien55.push(count_edges_cubic::<Levien55>(&curves, tolerance));
        fd.push(count_edges_cubic::<FwdDiff>(&curves, tolerance));
        hfd.push(count_edges_cubic::<HybridFwdDiff>(&curves, tolerance));
        sedeberg.push(count_edges_cubic::<Sedeberg>(&curves, tolerance));
    }

    fn print_first_row_md() {
        print!("| tolerance ");
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
    print_edges_md(" recursive ", &rec);
    print_edges_md(" linear    ", &linear);
    print_edges_md(" levien-19 ", &levien19);
    print_edges_md(" levien-37 ", &levien37);
    print_edges_md(" levien-55 ", &levien55);
    print_edges_md(" hain      ", &hain);
    print_edges_md(" sedeberg  ", &sedeberg);
    print_edges_md(" fwd-diff  ", &fd);
    print_edges_md(" hfd       ", &hfd);

    println!();

    let curves = generate_quadratic_curves();
    let mut rec = Vec::new();
    let mut levien = Vec::new();
    let mut fd = Vec::new();
    let mut cagd = Vec::new();
    let mut lin = Vec::new();
    for tolerance in TOLERANCES {
        rec.push(count_edges_quad::<Recursive>(&curves, tolerance));
        levien.push(count_edges_quad::<Levien>(&curves, tolerance));
        fd.push(count_edges_quad::<FwdDiff>(&curves, tolerance));
        cagd.push(count_edges_quad::<Sedeberg>(&curves, tolerance));
        lin.push(count_edges_quad::<Linear>(&curves, tolerance));
    }

    println!("");
    println!("Quadratic bézier curves:\n");
    print_first_row_md();
    print_edges_md(" recursive ", &rec);
    print_edges_md(" linear    ",    &lin);
    print_edges_md(" sedeberg  ", &cagd);
    print_edges_md(" levien    ", &levien);
    print_edges_md(" fwd-diff  ", &fd);
}
