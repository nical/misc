struct VertexOutput {
    [[location(0), interpolate(linear)]] uv: vec2<f32>;
    [[location(1), interpolate(flat)]] edges: vec2<u32>;
    [[location(2), interpolate(flat)]] backdrop: f32;
    [[builtin(position)]] position: vec4<f32>;
};

[[stage(vertex)]]
fn main(
    [[location(0)]] in_edges: vec2<u32>,
    [[location(1)]] in_mask_id: u32,
    [[location(2)]] in_backdrop: f32,
    [[builtin(vertex_index)]] vertex_index: u32,
) -> VertexOutput {

    var vertices = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 1.0)
    );

    var uv = vertices[vertex_index % 4u];

    var TILE_SIZE: f32 = 16.0;
    var MASKS_PER_ROW: u32 = 2048u / 16u;
    var tile_x = f32(in_mask_id % MASKS_PER_ROW);
    var tile_y = f32(in_mask_id / MASKS_PER_ROW);
    var normalized_mask_uv = ((vec2<f32>(tile_x, tile_y) + uv) * TILE_SIZE) / 2048.0;
    var screen_pos = normalized_mask_uv * 2.0 - vec2<f32>(1.0);
    screen_pos.y = - screen_pos.y;

    return VertexOutput(
        uv * 16.0,
        in_edges,
        in_backdrop,
        vec4<f32>(screen_pos.x, screen_pos.y, 0.0, 1.0),
    );
}
