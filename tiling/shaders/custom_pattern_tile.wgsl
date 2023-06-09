#import tiling
#import render_target
#import rect

//#define EDGE_STORE_BINDING { @group(1) @binding(1) }
//#import mask::fill

const MASK_KIND_NONE: u32 = 0u;
const MASK_KIND_TILED: u32 = 1u;
const MASK_KIND_FILL: u32 = 2u;

struct Pattern {
#mixin custom_pattern_struct_src
}

#mixin custom_pattern_src

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    #if TILED_MASK {
        @location(0) mask_uv: vec2<f32>,
        //@location(1) maks_data: vec4<u32>,
    }
    #mixin custom_pattern_varyings_src
};

@vertex fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @location(0) instance_data: vec4<u32>,
    //#if TILED_MASK { @location(1) mask_data: vec4<u32>, }
) -> VertexOutput {
    var uv = rect_get_uv(vertex_index);
    var tile = tiling_decode_instance(instance_data, uv);
    var target_pos = canvas_to_target(tile.position);

    var pattern = pattern_vertex(tile.pattern_position, tile.pattern_data);
    #if TILED_MASK {
        var mask_uv = tile.mask_position;
        #if FILL_MASK {
            if mask_data.x == MASK_KIND_FILL {
                mask_uv = tile.pattern_position;
            }
        }
    }

    return VertexOutput(
        target_pos,
        #if TILED_MASK {
            mask_uv,
            //mask_data,
        }
        #mixin custom_pattern_pass_varyings_src
    );
}

#if TILED_MASK {
    @group(1) @binding(0) var mask_texture: texture_2d<f32>;

    fn mask_fragment(mask_uv: vec2<f32>, /*mask_data: vec4<u32>*/) -> f32 {
        var alpha = 1.0;
        //let mask_kind = mask_data.x;

        #if FILL_MASK {
            if (mask_kind == MASK_KIND_FILL) {
                let fill_rule = mask_data.y & 0xFFFFu;
                let backdrop = f32((mask_data.y > 16u)) - 8192.0;
                let edges = mask_data.zw;
                alpha *= rasterize_fill_mask(mask_uv, edges, fill_rule, backdrop);
            }
        }

        //if (mask_kind == MASK_KIND_TILED) {
            var uv = vec2<i32>(i32(mask_uv.x), i32(mask_uv.y));
            alpha *= textureLoad(mask_texture, uv, 0).r;
        //}

        return alpha;
    }
}

@fragment fn fs_main(
    #if TILED_MASK {
        @location(0) mask_uv: vec2<f32>,
        //@location(1) mask_data: vec4<u32>,
    }
    #mixin custom_pattern_varyings_src
) -> @location(0) vec4<f32> {
    var pattern = Pattern(
        #mixin custom_pattern_fragment_arguments_src
    );

    var color = pattern_fragment(pattern);

    #if TILED_MASK {
        color.a *= mask_fragment(mask_uv, /*mask_data*/);
    }

    // Premultiply.
    color.r = color.r * color.a;
    color.g = color.g * color.a;
    color.b = color.b * color.a;

    return color;
}
