
[[group(1), binding(0)]] var mask_texture: texture_2d<f32>;

struct FragmentOutput {
    [[location(0)]] color: vec4<f32>;
};

[[stage(fragment)]]
fn main(
    [[location(0), interpolate(linear)]] in_uv: vec2<f32>,
    [[location(1), interpolate(flat)]] in_color: vec4<f32>,
) -> FragmentOutput {

    var color = in_color;

    var uv = vec2<i32>(i32(in_uv.x), i32(in_uv.y));
    color.a = textureLoad(mask_texture, uv, 0).r;

    // Premultiply.
    color.r = color.r * color.a;
    color.g = color.g * color.a;
    color.b = color.b * color.a;

    return FragmentOutput(color);
}