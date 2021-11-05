
struct FragmentOutput {
    [[location(0)]] color: vec4<f32>;
};

// TODO actually sample a mask.

[[stage(fragment)]]
fn main(
    [[location(0), interpolate(linear)]] in_uv: vec2<f32>,
    [[location(1), interpolate(flat)]] in_color: vec4<f32>,
) -> FragmentOutput {

    var color = in_color;
    // TODO
    color.a = 0.5;

    // Premultiply.
    color.r = color.r * color.a;
    color.g = color.g * color.a;
    color.b = color.b * color.a;

    return FragmentOutput(color);
}