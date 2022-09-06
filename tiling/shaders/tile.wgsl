#import tiling
#import render_target
#import rect
#import pattern::color

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    #if TILED_MASK { @location(0) mask_uv: vec2<f32>, }
    #if SOLID_PATTERN { @location(1) @interpolate(flat) color: vec4<f32>, }
    #if TILED_IMAGE_PATTERN { @location(1) src_uv: vec2<f32>, }
};

@vertex fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @location(0) instance_data: vec4<u32>,
) -> VertexOutput {
    var uv = rect_get_uv(vertex_index);
    var tile = tiling_decode_instance(instance_data, uv);

    var target_pos = canvas_to_target(tile.position);

    #if SOLID_PATTERN {
        var color = decode_color(tile.pattern_data);
    }

    return VertexOutput(
        target_pos,
        #if TILED_MASK { tile.mask_position, }
        #if SOLID_PATTERN { color, }
        #if TILED_IMAGE_PATTERN { tile.pattern_position, }
    );
}

#if TILED_MASK {
    @group(1) @binding(0) var mask_texture: texture_2d<f32>;
}

#if TILED_IMAGE_PATTERN {
    @group(2) @binding(0) var src_texture: texture_2d<f32>;
}

@fragment fn fs_main(
    #if TILED_MASK { @location(0) mask_uv: vec2<f32>, }
    #if SOLID_PATTERN { @location(1) @interpolate(flat) in_color: vec4<f32>, }
    #if TILED_IMAGE_PATTERN { @location(1) tiled_image_uv: vec2<f32>, }
) -> @location(0) vec4<f32> {

    var color = vec4(1.0);

    #if SOLID_PATTERN {{
        color *= in_color;
    }}

    #if TILED_IMAGE_PATTERN {{
        var uv = vec2<i32>(i32(tiled_image_uv.x), i32(tiled_image_uv.y));
        color *= textureLoad(src_texture, uv, 0);
    }}

    #if TILED_MASK {{
        var uv = vec2<i32>(i32(mask_uv.x), i32(mask_uv.y));
        color.a *= textureLoad(mask_texture, uv, 0).r;
    }}

    // Premultiply.
    color.r = color.r * color.a;
    color.g = color.g * color.a;
    color.b = color.b * color.a;

    return color;
}
