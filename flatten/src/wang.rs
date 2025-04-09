use lyon_path::{geom::{CubicBezierSegment, LineSegment, QuadraticBezierSegment}, math::point};

use crate::simd4::unpack;

/// Computes the number of line segments required to build a flattened approximation
/// of the curve with segments placed at regular `t` intervals.
pub fn num_segments_cubic(curve: &CubicBezierSegment<f32>, tolerance: f32) -> f32 {
    let from = curve.from.to_vector();
    let ctrl1 = curve.ctrl1.to_vector();
    let ctrl2 = curve.ctrl2.to_vector();
    let to = curve.to.to_vector();
    let l = (from - ctrl1 * 2.0 + to).max(ctrl1 - ctrl2 * 2.0 + to) * 6.0;
    let num_steps = f32::sqrt(l.length() / (8.0 * tolerance));

    num_steps.ceil().max(1.0)
}

/// Computes the number of line segments required to build a flattened approximation
/// of the curve with segments placed at regular `t` intervals.
pub fn num_segments_quadratic(curve: &QuadraticBezierSegment<f32>, tolerance: f32) -> f32 {
    let from = curve.from.to_vector();
    let ctrl = curve.ctrl.to_vector();
    let to = curve.to.to_vector();
    let l = (from - ctrl * 2.0 + to) * 2.0;
    let num_steps = f32::sqrt(l.length() / (8.0 * tolerance));

    num_steps.ceil().max(1.0)
}

/// Flatten the curve by precomputing a number of segments and splitting the curve
/// at regular `t` intervals.
pub fn flatten_cubic<F>(curve: &CubicBezierSegment<f32>, tolerance: f32, callback: &mut F)
    where
    F:  FnMut(&LineSegment<f32>)
{
    let poly = crate::polynomial_form_cubic(&curve);
    let n = num_segments_cubic(curve, tolerance);
    let step = 1.0 / n;
    let mut prev = 0.0;
    let mut from = curve.from;
    for _ in 0..(n as u32 - 1) {
        let t = prev + step;
        let to = poly.sample_fma(t);
        callback(&mut LineSegment { from, to });
        from = to;
        prev = t;
    }

    let to = curve.to;
    callback(&mut LineSegment { from, to });
}

/// Flatten the curve by precomputing a number of segments and splitting the curve
/// at regular `t` intervals.
pub fn flatten_quadratic<F>(curve: &QuadraticBezierSegment<f32>, tolerance: f32, callback: &mut F)
    where
    F:  FnMut(&LineSegment<f32>)
{
    let poly = crate::polynomial_form_quadratic(curve);
    let n = num_segments_quadratic(curve, tolerance);
    let step = 1.0 / n;
    let mut prev = 0.0;
    let mut from = curve.from;
    for _ in 0..(n as u32 - 1) {
        let t = prev + step;
        let to = poly.sample(t);
        callback(&mut LineSegment { from, to });
        from = to;
        prev = t;
    }

    let to = curve.to;
    callback(&mut LineSegment { from, to });
}

/// A 128-bit wide SIMD version of cubic flattening using
/// Wang's formula.
///
/// Wang's formula itself remains scalar, but the loop over
/// the curve to split at regular t inverval is manually
/// vectorized.
/// The strategy is to evaluate the curve for two points
/// at a time and write the result into a scratch buffer.
/// A second loop traverses the scratch buffer and emmits
/// the callback. The motivations behind the scratch buffer
/// are:
///  - Avoiding the cost of extracting each lane from the
///    SIMD result.
///  - Avoiding the callback in the tight simd loop.
/// In practice this point buffer approach is only a win at
/// very low tolerance threshods and is a regression overall.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx"))]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
pub unsafe fn flatten_cubic_simd4_with_point_buffer<F>(curve: &CubicBezierSegment<f32>, tolerance: f32, callback: &mut F)
    where
    F:  FnMut(&LineSegment<f32>)
{
    use crate::simd4::{vec4, splat, interleave_splat, add, mul, aligned_store};

    let poly = crate::polynomial_form_cubic(&curve);
    let n = num_segments_cubic(curve, tolerance);
    let mut from = curve.from;

    let mut point_buffer: [f32; 64] = [0.0; 64];

    let a0 = interleave_splat(poly.a0.x, poly.a0.y);
    let a1 = interleave_splat(poly.a1.x, poly.a1.y);
    let a2 = interleave_splat(poly.a2.x, poly.a2.y);
    let a3 = interleave_splat(poly.a3.x, poly.a3.y);
    let step = 1.0 / n;
    let step = splat(step);
    let mut t = mul(step, vec4(1.0, 1.0, 2.0, 2.0));
    let step = add(step, step);

    // minus one because we'll add the last point explicitly
    let mut n = n as i32 - 1;
    while n > 0 {
        let mut cap: i32 = 32;
        let mut out_ptr: *mut f32 = &mut point_buffer[0];
        let offset = out_ptr.align_offset(16);
        out_ptr = out_ptr.add(offset);
        cap -= offset as i32;
        let base_ptr = out_ptr;
        let copy_count = (cap / 2).min(n) as usize;

        while n > 0 && cap >= 4 {
            debug_assert_eq!(out_ptr.align_offset(16), 0);

            let p = crate::simd4::sample_cubic_horner_interlaved_simd4(a0, a1, a2, a3, t);
            aligned_store(out_ptr, p);
            out_ptr = out_ptr.add(4);

            t = add(t, step);
            n -= 2;
            cap -= 4;
        }

        let mut ptr = base_ptr;
        for _ in 0..copy_count {
            let x = ptr.read();
            let y = ptr.add(1).read();
            ptr = ptr.add(2);

            let to = point(x, y);

            callback(&LineSegment { from, to});
            from = to;
        }
    }

    let to = curve.to;
    callback(&mut LineSegment { from, to });
}

#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx"))]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
pub unsafe fn flatten_cubic_simd4<F>(curve: &CubicBezierSegment<f32>, tolerance: f32, callback: &mut F)
    where
    F:  FnMut(&LineSegment<f32>)
{
    use crate::simd4::{vec4, splat, interleave_splat, add, mul};

    let poly = crate::polynomial_form_cubic(&curve);
    let n = num_segments_cubic(curve, tolerance);
    let mut from = curve.from;

    let a0 = interleave_splat(poly.a0.x, poly.a0.y);
    let a1 = interleave_splat(poly.a1.x, poly.a1.y);
    let a2 = interleave_splat(poly.a2.x, poly.a2.y);
    let a3 = interleave_splat(poly.a3.x, poly.a3.y);
    let step = 1.0 / n;
    let step = splat(step);
    let mut t = mul(step, vec4(1.0, 1.0, 2.0, 2.0));
    let step = add(step, step);

    // minus one because we'll add the last point explicitly
    let mut n = n as i32 - 1;
    while n > 0 {
        let p = crate::simd4::sample_cubic_horner_interlaved_simd4(a0, a1, a2, a3, t);
        let (x0, y0, x1, y1) = unpack(p);

        let to = point(x0, y0);
        callback(&LineSegment { from, to});
        from = to;

        if n > 1 {
            let to = point(x1, y1);
            callback(&LineSegment { from, to});
            from = to;
        }

        t = add(t, step);
        n -= 2;
    }

    let to = curve.to;
    callback(&mut LineSegment { from, to });
}

#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx"))]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
pub unsafe fn flatten_quadratic_simd4<F>(curve: &QuadraticBezierSegment<f32>, tolerance: f32, callback: &mut F)
    where
    F:  FnMut(&LineSegment<f32>)
{
    use crate::simd4::{vec4, splat, add, mul};

    let poly = crate::polynomial_form_quadratic(&curve);
    let n = num_segments_quadratic(curve, tolerance);
    let mut from = curve.from;

    let a0x = splat(poly.a0.x);
    let a0y = splat(poly.a0.y);
    let a1x = splat(poly.a1.x);
    let a1y = splat(poly.a1.y);
    let a2x = splat(poly.a2.x);
    let a2y = splat(poly.a2.y);
    let step = 1.0 / n;
    let step = splat(step);
    let mut t = mul(step, vec4(1.0, 2.0, 3.0, 4.0));
    let step4 = mul(step, splat(4.0));

    // minus one because we'll add the last point explicitly
    let mut n = n as i32 - 1;
    while n > 0 {
        let x = crate::simd4::sample_quadratic_horner_simd4(a0x, a1x, a2x, t);
        let y = crate::simd4::sample_quadratic_horner_simd4(a0y, a1y, a2y, t);

        let x: [f32; 4] = std::mem::transmute(x);
        let y: [f32; 4] = std::mem::transmute(y);

        for (x, y) in x.iter().zip(y.iter()).take(n.min(4) as usize) {
            let p = point(*x, *y);
            callback(&LineSegment { from, to: p });
            from = p;
        }

        t = add(t, step4);
        n -= 4;
    }

    let to = curve.to;
    callback(&mut LineSegment { from, to });
}

#[test]
fn wang_simd_scalar() {
    let curve = CubicBezierSegment {
        from: point(100.0, 100.0),
        ctrl1: point(0.0, 100.0),
        ctrl2: point(100.0, 0.0),
        to: point(10.0, 0.0),
    };

    let mut scalar = Vec::new();
    flatten_cubic(&curve, 0.001, &mut |seg| {
        scalar.push(seg.to)
    });

    let mut simd = Vec::new();
    unsafe {
        flatten_cubic_simd4(&curve, 0.001, &mut |seg| {
            simd.push(seg.to)
        });
    }

    assert_eq!(scalar.len(), simd.len());
    for (i, (scalar, simd)) in scalar.iter().zip(simd.iter()).enumerate() {
        assert!(scalar.distance_to(*simd) < 0.01, "{i:?}: scalar {scalar:?} simd {simd:?}");
    }
}
