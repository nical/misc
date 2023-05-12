#import rect
#import render_target

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
) -> VertexOutput {
    var uv = rect_get_uv(vertex_index);
    return VertexOutput(normalized_to_target(uv), uv);
}

@group(0) @binding(0) var src_texture: texture_2d<f32>;

@fragment fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    var texel_pos: vec2<u32> = vec2<u32>(uv * vec2<f32>(textureDimensions(src_texture).xy));
    var color: vec4<f32> = textureLoad(src_texture, texel_pos, 0);
    return color;
}
