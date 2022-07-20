struct Globals {
    resolution: vec2<f32>,
    tile_size: u32,
    tile_atlas_size: u32,
};

@group(0) @binding(0) var<uniform> globals: Globals;

struct VertexOutput {
    @location(0) @interpolate(linear) uv: vec2<f32>,
    @location(1) @interpolate(flat) src_offset: u32,
    @builtin(position) position: vec4<f32>,
};

@vertex
fn main(
    @location(0) a_mask_id: u32,
    @location(1) a_src_offset: u32,
    @builtin(vertex_index) vertex_index: u32,
) -> VertexOutput {
    var vertices = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 1.0)
    );

    var tile_size = f32(globals.tile_size);
    var uv = vertices[vertex_index] * tile_size;
    var masks_per_row = globals.tile_atlas_size / globals.tile_size;
    var mask_index = a_mask_id % (masks_per_row * masks_per_row);

    var dst = vec2<f32>(
        f32(mask_index % masks_per_row),
        f32(mask_index / masks_per_row)
    );

    var pos = (dst * tile_size + uv) / f32(globals.tile_atlas_size);
    var screen_pos = pos * 2.0 - vec2<f32>(1.0);
    screen_pos.y = -screen_pos.y;

    return VertexOutput(
        uv,
        a_src_offset,
        vec4<f32>(screen_pos.x, screen_pos.y, 0.0, 1.0),
    );
}
