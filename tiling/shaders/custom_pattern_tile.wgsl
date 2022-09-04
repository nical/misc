#import tiling
#import render_target
#import rect

struct Pattern {
#mixin custom_pattern_struct_src
}

#mixin custom_pattern_src

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    #if TILED_MASK { @location(0) mask_uv: vec2<f32>, }
    #mixin custom_pattern_varyings_src
};

@vertex fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @location(0) instance_data: vec4<u32>,
) -> VertexOutput {
    var tile = tiling_decode_instance(instance_data);

    var uv = rect_get_uv(vertex_index);
    var pos = tiling_decode_position(tile.tile_index, uv);
    var target_pos = canvas_to_target(pos);

    var pattern = pattern_vertex(tile, uv);

    #if TILED_MASK {
        var mask_uv = tiling_decode_position(tile.mask_index, uv);
    }

    return VertexOutput(
        target_pos,
        #if TILED_MASK { mask_uv, }
        #mixin custom_pattern_pass_varyings_src
    );
}

#if TILED_MASK {
    @group(1) @binding(0) var mask_texture: texture_2d<f32>;
}

@fragment fn fs_main(
    #if TILED_MASK { @location(0) mask_uv: vec2<f32>, }
    #mixin custom_pattern_varyings_src
) -> @location(0) vec4<f32> {
    var pattern = Pattern(
        #mixin custom_pattern_fragment_arguments_src
    );

    var color = pattern_fragment(pattern);

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
