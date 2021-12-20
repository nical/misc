struct Globals {
    resolution: vec2<f32>;
};

[[group(0), binding(0)]] var<uniform> globals: Globals;

struct VertexOutput {
    [[location(0), interpolate(flat)]] color: vec4<f32>;
    [[builtin(position)]] position: vec4<f32>;
};

fn decode_color(color: u32) -> vec4<f32> {
    var r = f32((color >> 24u) & 255u) / 255.0;
    var g = f32((color >> 16u) & 255u) / 255.0;
    var b = f32((color >> 8u)  & 255u) / 255.0;
    var a = f32((color >> 0u)  & 255u) / 255.0;
    return vec4<f32>(r, g, b, a);
}

[[stage(vertex)]]
fn main(
    [[location(0)]] a_rect: vec4<f32>,
    [[location(1)]] a_color: u32,
    [[builtin(vertex_index)]] vertex_index: u32,
) -> VertexOutput {

    var vertices = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 1.0)
    );

    var uv = vertices[vertex_index];

    var screen_pos = mix(a_rect.xy, a_rect.zw, uv) / globals.resolution - vec2<f32>(0.5);
    screen_pos.y = -screen_pos.y;

    return VertexOutput(
        decode_color(a_color),
        vec4<f32>(screen_pos.x, screen_pos.y, 0.0, 1.0),
    );
}
