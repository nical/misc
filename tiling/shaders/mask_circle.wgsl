// This shader rasterize the mask for a single tile p0 a
// backdrop and and a sequence of edges.
//
// The "backdrop" is the winding number at the top-right corner
// of the tile (following piet and pathfinder's terminology).

#import rect
#import render_target
#import tiling
#import mask::circle

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) radius: f32,
    @location(2) @interpolate(flat) center: vec2<f32>,
    @location(3) @interpolate(flat) invert: u32,
};

@vertex fn vs_main(
    @location(0) tile_position: u32,
    @location(1) radius: f32,
    @location(2) center: vec2<f32>,
    @builtin(vertex_index) vertex_index: u32,
) -> VertexOutput {
    var uv = rect_get_uv(vertex_index);
    var pos = tiling_decode_position(tile_position, uv);
    let target_pos = canvas_to_target(pos);
    let invert = tile_position >> 31u;
    return VertexOutput(
        target_pos,
        uv * TILE_SIZE_F32,
        radius,
        center,
        invert,
    );
}

@fragment fn fs_main(
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) radius: f32,
    @location(2) @interpolate(flat) center: vec2<f32>,
    @location(3) @interpolate(flat) invert: u32,
) -> @location(0) vec4<f32> {
    var mask = circle(uv, center, radius);
    if (invert != 0u) {
        mask = 1.0 - mask;
    }
    return vec4<f32>(mask, mask, mask, mask);
}
