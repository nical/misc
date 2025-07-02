const PI: f32 = 3.141592653589793;
const INV_PI: f32 = 1.0 / PI;
const INV_2_PI: f32 = INV_PI * 0.5;

// From https://math.stackexchange.com/questions/1098487/atan2-faster-approximation
fn approx_atan2(v: vec2f) -> f32 {
    let abs_v = abs(v);
    let slope = min(abs_v.x, abs_v.y) / max(abs_v.x, abs_v.y);
    let s2 = slope * slope;
    var r = ((-0.0464964749 * s2 + 0.15931422) * s2 - 0.327622764) * s2 * slope + slope;

    if abs_v.y > abs_v.x {
        r = 1.57079637 - r;
    }
    if v.x < 0.0 {
        r = 3.14159274 - r;
    }
    if v.y < 0.0 {
        r = -r;
    }

    return r;
}
