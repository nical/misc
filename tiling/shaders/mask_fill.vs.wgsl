struct MaskParams {
    tile_size: f32;
    inv_atlas_width: f32;
    masks_per_row: u32;
};

[[group(0), binding(0)]] var<uniform> globals: MaskParams;

struct VertexOutput {
    [[location(0), interpolate(linear)]] uv: vec2<f32>;
    [[location(1), interpolate(flat)]] edges: vec2<u32>;
    [[location(2), interpolate(flat)]] fill_rule: u32;
    [[builtin(position)]] position: vec4<f32>;
};

[[stage(vertex)]]
fn main(
    [[location(0)]] in_edges: vec2<u32>,
    [[location(1)]] in_mask_id: u32,
    [[location(2)]] in_fill_rule: u32,
    [[builtin(vertex_index)]] vertex_index: u32,
) -> VertexOutput {

    var vertices = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 1.0)
    );

    var uv = vertices[vertex_index % 4u];

    var tile_size = globals.tile_size;
    var masks_per_row = globals.masks_per_row;

    var tile_x = f32(in_mask_id % masks_per_row);
    var tile_y = f32(in_mask_id / masks_per_row);
    var normalized_mask_uv = ((vec2<f32>(tile_x, tile_y) + uv) * tile_size) * globals.inv_atlas_width;

    var screen_pos = normalized_mask_uv * 2.0 - vec2<f32>(1.0);
    screen_pos.y = -screen_pos.y;

    return VertexOutput(
        uv * tile_size,
        in_edges,
        in_fill_rule,
        vec4<f32>(screen_pos.x, screen_pos.y, 0.0, 1.0),
    );
}
