
fn decode_color(color: u32) -> vec4<f32> {
    var r = f32((color >> 24u) & 255u) / 255.0;
    var g = f32((color >> 16u) & 255u) / 255.0;
    var b = f32((color >> 8u)  & 255u) / 255.0;
    var a = f32((color >> 0u)  & 255u) / 255.0;
    return vec4<f32>(r, g, b, a);
}
