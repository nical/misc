pub mod levien_experiments;

use lyon_path::{geom::LineSegment, math::{point, Point}};

use crate::simd4::*;

#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx"))]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
pub unsafe fn output_points_1(x: f32x4, y: f32x4, count: usize, from: &mut Point, cb: &mut dyn FnMut(&LineSegment<f32>)) {
    let x: [f32; 4] = std::mem::transmute(x);
    let y: [f32; 4] = std::mem::transmute(y);

    for (x, y) in x.iter().zip(y.iter()).take(count.min(4)) {
        let p = point(*x, *y);
        cb(&LineSegment { from: *from, to: p });
        *from = p;
    }
}

/// This generates fewer instructions but is a lot slower than
/// output_points_1. There are a handful of very expensive vmovss
/// instructions that write at rsp + offset.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx"))]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
pub unsafe fn output_points_2(x: f32x4, y: f32x4, count: usize, from: &mut Point, cb: &mut dyn FnMut(&LineSegment<f32>)) {
    use crate::AlignedBuf;
    let mut bx: AlignedBuf<4> = AlignedBuf::new();
    let mut by: AlignedBuf<4> = AlignedBuf::new();

    unaligned_store(bx.ptr(0), x);
    unaligned_store(by.ptr(0), y);

    for i in 0..count {
        let x = bx.get(i);
        let y = by.get(i);
        let p = point(x, y);
        cb(&LineSegment { from: *from, to: p });
        *from = p;
    }
}
