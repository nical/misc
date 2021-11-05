[[block]]
struct Globals {
    resolution: vec2<f32>;
};

[[group(0), binding(0)]] var<uniform> globals: Globals;

struct VertexOutput {
    [[location(0), interpolate(linear)]] uv_res: vec4<f32>;
    [[location(1), interpolate(flat)]] cmd_range: vec2<u32>;
    [[builtin(position)]] position: vec4<f32>;
};

[[stage(vertex)]]
fn main(
    [[location(0)]] a_rect: vec4<f32>,
    [[location(1)]] a_cmd_range: vec2<u32>,
    [[builtin(vertex_index)]] vertex_index: u32,
) -> VertexOutput {

    var vertices = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 1.0)
    );

    var uv = vertices[vertex_index];

    var world_position = mix(a_rect.xy, a_rect.zw, uv);

    return VertexOutput(
        vec4<f32>(world_position.x, world_position.y, globals.resolution.x, globals.resolution.y),
        a_cmd_range,
        vec4<f32>(uv * 2.0 - vec2<f32>(1.0, 1.0), 0.0, 1.0)
    );
}
