use std::io::Write;
use std::path::PathBuf;

use lyon_path::geom::{QuadraticBezierSegment, CubicBezierSegment};
use crate::{Flatten, FwdDiff, Hain, HybridFwdDiff, Kurbo, Levien, LevienLinear, LevienQuads, LevienSimd, Linear, LinearAgg, LinearHfd, Recursive, RecursiveAgg, RecursiveHfd, Wang, Yzerman, TOLERANCES};
use crate::testing::*;

fn count_edges_cubic<F: Flatten>(curves: &[CubicBezierSegment<f32>], tolerance: f32) -> u32 {
    counters_reset();
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
    counters_reset();
    let mut count = 0;
    for curve in curves {
        F::quadratic(curve, tolerance, &mut |_| { count += 1; });
    }

    count
}

fn print_first_row_md(output: &mut dyn Write) {
    let _ = write!(output, "| tolerance  ");
    for tolerance in &TOLERANCES {
        let _ = write!(output, "|  {} ", tolerance);
    }
    let _ = writeln!(output, "|");
    let _ = write!(output, "|-----------");
    for _ in 0..TOLERANCES.len() {
        let _ = write!(output, "| -----:");
    }
    let _ = writeln!(output, "|");
}

fn print_edges_md(output: &mut dyn Write, name: &str, vals: &[u32]) {
    if vals.is_empty() {
        return;
    }
    let _ = write!(output, "|{}", name);
    for val in vals {
        let _ = write!(output, "| {:.2} ", val);
    }
    let _ = writeln!(output, "|");
}

fn get_flatten_output() -> Option<String> {
    let mut out_name = None;
    for (var, val) in std::env::vars() {
        if var == "FLATTEN_OUTPUT" {
            out_name = Some(val);
        }
    }

    out_name
}

#[test]
fn edge_count_cubic() {
    let curves = generate_bezier_curves();
    let mut hain = Vec::new();
    let mut rec = Vec::new();
    let mut rec_agg = Vec::new();
    let mut rec_hfd = Vec::new();
    let mut linear = Vec::new();
    let mut linear_hfd = Vec::new();
    let mut linear_agg = Vec::new();
    let mut levien_quads = Vec::new();
    let mut kurbo = Vec::new();
    let mut levien = Vec::new();
    let mut levien_simd = Vec::new();
    let mut levien_simd2 = Vec::new();
    let mut levien_linear = Vec::new();
    let mut fd = Vec::new();
    let mut hfd = Vec::new();
    let mut wang = Vec::new();
    let mut yzerman = Vec::new();
    //let mut fixed_16 = Vec::new();
    for tolerance in TOLERANCES {
        hain.push(count_edges_cubic::<Hain>(&curves, tolerance));
        rec.push(count_edges_cubic::<Recursive>(&curves, tolerance));
        //rec_hfd.push(count_edges_cubic::<RecursiveHfd>(&curves, tolerance));
        //rec_agg.push(count_edges_cubic::<RecursiveAgg>(&curves, tolerance));
        linear.push(count_edges_cubic::<Linear>(&curves, tolerance));
        //linear_hfd.push(count_edges_cubic::<LinearHfd>(&curves, tolerance));
        //linear_agg.push(count_edges_cubic::<LinearAgg>(&curves, tolerance));
        levien_quads.push(count_edges_cubic::<LevienQuads>(&curves, tolerance));
        //levien_simd.push(count_edges_cubic::<LevienSimd>(&curves, tolerance));
        //levien_simd2.push(count_edges_cubic::<crate::LevienSimd3>(&curves, tolerance));
        //#[cfg(feature = "stats")] {
        //    print!("| {tolerance:?} | ");
        //    for i in 0..NUM_COUNTERS {
        //        print!(" {} |", counters_get(i));
        //    }
        //    println!("");
        //}
        levien_linear.push(count_edges_cubic::<LevienLinear>(&curves, tolerance));

        //kurbo.push(count_edges_cubic::<Kurbo>(&curves, tolerance));
        levien.push(count_edges_cubic::<Levien>(&curves, tolerance));
        fd.push(count_edges_cubic::<FwdDiff>(&curves, tolerance));
        hfd.push(count_edges_cubic::<HybridFwdDiff>(&curves, tolerance));
        wang.push(count_edges_cubic::<Wang>(&curves, tolerance));
        yzerman.push(count_edges_cubic::<Yzerman>(&curves, tolerance));
        //fixed_16.push(count_edges_cubic::<Fixed16>(&curves, tolerance));
    }

    let out_name = get_flatten_output();

    let mut _std_out = None;
    let mut _out_file = None;
    let output: &mut dyn std::io::Write = match out_name {
        None => {
            _std_out = Some(std::io::stdout().lock());
            _std_out.as_mut().unwrap()
        }
        Some(file_name) => {
            let path: PathBuf = file_name.into();
            _out_file = Some(std::fs::File::create(path).unwrap());
            _out_file.as_mut().unwrap()
        }
    };

    println!("Cubic bézier curves:\n");
    print_first_row_md(output);
    print_edges_md(output, " recursive    ", &rec);
    print_edges_md(output, " recursive-agg", &rec_agg);
    print_edges_md(output, " recursive-hfd", &rec_hfd);
    print_edges_md(output, " linear       ", &linear);
    print_edges_md(output, " linear-agg   ", &linear_agg);
    print_edges_md(output, " linear-hfd   ", &linear_hfd);
    print_edges_md(output, " levien       ", &levien);
    print_edges_md(output, " levien-simd  ", &levien_simd);
    print_edges_md(output, " levien-simd-v3", &levien_simd2);
    print_edges_md(output, " kurbo        ", &kurbo);
    print_edges_md(output, " levien-quads ", &levien_quads);
    print_edges_md(output, " levien-linear", &levien_linear);
    print_edges_md(output, " hain         ", &hain);
    print_edges_md(output, " wang         ", &wang);
    print_edges_md(output, " yzerman      ", &yzerman);
    print_edges_md(output, " fwd-diff     ", &fd);
    print_edges_md(output, " hfd          ", &hfd);
    //print_edges_md(" fixed-16     ", &fixed_16);

    println!();

    let curves = generate_quadratic_curves();
    let mut rec = Vec::new();
    let mut levien = Vec::new();
    let mut levien_simd = Vec::new();
    let mut levien_linear = Vec::new();
    let mut fd = Vec::new();
    let mut cagd = Vec::new();
    let mut lin = Vec::new();
    for tolerance in TOLERANCES {
        rec.push(count_edges_quad::<Recursive>(&curves, tolerance));
        levien.push(count_edges_quad::<LevienQuads>(&curves, tolerance));
        levien_simd.push(count_edges_quad::<LevienSimd>(&curves, tolerance));
        levien_linear.push(count_edges_quad::<LevienLinear>(&curves, tolerance));
        fd.push(count_edges_quad::<FwdDiff>(&curves, tolerance));
        cagd.push(count_edges_quad::<Wang>(&curves, tolerance));
        lin.push(count_edges_quad::<Linear>(&curves, tolerance));
    }
}

#[test]
fn edge_count_quadratic() {
    let out_name = get_flatten_output();

    let mut _std_out = None;
    let mut _out_file = None;
    let output: &mut dyn std::io::Write = match out_name {
        None => {
            _std_out = Some(std::io::stdout().lock());
            _std_out.as_mut().unwrap()
        }
        Some(file_name) => {
            let path: PathBuf = file_name.into();
            _out_file = Some(std::fs::File::create(path).unwrap());
            _out_file.as_mut().unwrap()
        }
    };

    let curves = generate_quadratic_curves();
    let mut rec = Vec::new();
    let mut levien = Vec::new();
    let mut levien_simd = Vec::new();
    let mut levien_simd2 = Vec::new();
    let mut levien_linear = Vec::new();
    let mut fd = Vec::new();
    let mut cagd = Vec::new();
    let mut lin = Vec::new();
    for tolerance in TOLERANCES {
        rec.push(count_edges_quad::<Recursive>(&curves, tolerance));
        levien.push(count_edges_quad::<LevienQuads>(&curves, tolerance));
        levien_simd.push(count_edges_quad::<LevienSimd>(&curves, tolerance));
        levien_simd2.push(count_edges_quad::<crate::LevienSimd2>(&curves, tolerance));
        levien_linear.push(count_edges_quad::<LevienLinear>(&curves, tolerance));
        fd.push(count_edges_quad::<FwdDiff>(&curves, tolerance));
        cagd.push(count_edges_quad::<Wang>(&curves, tolerance));
        lin.push(count_edges_quad::<Linear>(&curves, tolerance));
    }

    print_first_row_md(output);
    print_edges_md(output, " recursive    ", &rec);
    print_edges_md(output, " levien       ", &levien);
    //print_edges_md(output, " levien-simd  ", &levien_simd);
    //print_edges_md(output, " levien-simd2 ", &levien_simd);
    print_edges_md(output, " linear       ", &lin);
    print_edges_md(output, " levien-linear", &levien_linear);
    print_edges_md(output, " wang         ", &cagd);
    print_edges_md(output, " fwd-diff     ", &fd);

}
