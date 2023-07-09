
fn decode_color(color: u32) -> vec4<f32> {
    var r = f32((color >> 24u) & 255u) / 255.0;
    var g = f32((color >> 16u) & 255u) / 255.0;
    var b = f32((color >> 8u)  & 255u) / 255.0;
    var a = f32((color >> 0u)  & 255u) / 255.0;
    return vec4<f32>(r, g, b, a);
}

fn premultiply_color(input: vec4<f32>) -> vec4<f32> {
    var color = input;
    color.r *= color.a;
    color.g *= color.a;
    color.b *= color.a;
    return color;
}

fn unpremultiply_color(color: vec4<f32>) -> vec4<f32> {
    let inv_a = 1.0 / color.a;
    return vec4<f32>(
        color.r * inv_a,
        color.g * inv_a,
        color.b * inv_a,
        color.a,
    );
}
