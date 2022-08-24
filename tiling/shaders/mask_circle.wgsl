// This shader rasterize the mask for a single tile p0 a
// backdrop and and a sequence of edges.
//
// The "backdrop" is the winding number at the top-right corner
// of the tile (following piet and pathfinder's terminology).

#import rect
#import render_target
#import tiling
#import mask::circle

@group(0) @binding(0) var<uniform> atlas: TileAtlasDescriptor;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) center: vec2<f32>,
    @location(2) @interpolate(flat) radius: f32,
};

@vertex fn vs_main(
    @location(0) mask_id: u32,
    @location(1) radius: f32,
    @location(2) center: vec2<f32>,
    @builtin(vertex_index) vertex_index: u32,
) -> VertexOutput {
    var uv = rect_get_uv(vertex_index);
    let atlas_uv = tiling_atlas_get_uv(atlas, mask_id, uv);
    let target_pos = normalized_to_target(atlas_uv);

    return VertexOutput(
        target_pos,
        uv * TILE_SIZE_F32,
        center,
        radius,
    );
}

@fragment fn fs_main(
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) center: vec2<f32>,
    @location(2) @interpolate(flat) radius: f32,
) -> @location(0) vec4<f32> {
    var mask = circle(uv, center, radius);
    return vec4<f32>(mask, mask, mask, mask);
}
