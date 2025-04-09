use lyon_path::geom::{LineSegment, CubicBezierSegment, QuadraticBezierSegment};


fn num_quadratics(curve: &CubicBezierSegment<f32>, tolerance: f32) -> f32 {
    let q = curve.from - curve.to + (curve.ctrl2 - curve.ctrl1) * 3.0;
    const K: f32 = 0.048112522; // 1.0 / (12.0 * 3.0f32.sqrt());
    (tolerance / K * q.length()).powf(1.0 / 3.0).ceil().max(1.0)
}

pub fn flatten_cubic<F>(curve: &CubicBezierSegment<f32>, tolerance: f32, callback: &mut F)
where
    F:  FnMut(&LineSegment<f32>)
{
    let simplify_tolerance = tolerance * 0.2;
    let flatten_tolerance = tolerance - simplify_tolerance;

    let num_quads = num_quadratics(curve, simplify_tolerance);
    let step = 1.0 / num_quads;

    let mut t0 = 0.0;
    while t0 < 1.0 {
        let mut t1 = t0 + step;
        if t1 > 0.999 {
            t1 = 1.0;
        }

        let quad = split_range(&curve, t0..t1).to_quadratic();

        crate::wang::flatten_quadratic(&quad, flatten_tolerance, callback);

        t0 = t1;
    }
}

pub unsafe fn flatten_cubic_simd4<F>(curve: &CubicBezierSegment<f32>, tolerance: f32, callback: &mut F)
where
    F:  FnMut(&LineSegment<f32>)
{
    let simplify_tolerance = tolerance * 0.2;
    let flatten_tolerance = tolerance - simplify_tolerance;

    let num_quads = num_quadratics(curve, simplify_tolerance);
    let step = 1.0 / num_quads;

    let mut t0 = 0.0;
    while t0 < 1.0 {
        let mut t1 = t0 + step;
        if t1 > 0.999 {
            t1 = 1.0;
        }

        let quad = split_range(curve, t0..t1).to_quadratic();

        crate::wang::flatten_quadratic_simd4(&quad, flatten_tolerance, callback);

        t0 = t1;
    }
}

#[inline(always)]
pub fn split_range(curve: &CubicBezierSegment<f32>, t_range: std::ops::Range<f32>) -> CubicBezierSegment<f32> {
    let (t0, t1) = (t_range.start, t_range.end);
    let from = curve.sample(t0);
    let to = curve.sample(t1);

    let d = QuadraticBezierSegment {
        from: (curve.ctrl1 - curve.from).to_point(),
        ctrl: (curve.ctrl2 - curve.ctrl1).to_point(),
        to: (curve.to - curve.ctrl2).to_point(),
    };

    let dt = t1 - t0;
    let ctrl1 = from + d.sample(t0).to_vector() * dt;
    let ctrl2 = to - d.sample(t1).to_vector() * dt;

    CubicBezierSegment {
        from,
        ctrl1,
        ctrl2,
        to,
    }
}
