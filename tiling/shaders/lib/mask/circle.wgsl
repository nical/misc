
fn circle(pos: vec2<f32>, center: vec2<f32>, radius: f32) -> f32 {
    var d = length(pos - center) - radius;
    return 1.0 - clamp(d, 0.0, 1.0);
}
