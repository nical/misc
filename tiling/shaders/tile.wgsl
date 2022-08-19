#import tiling
#import render_target
#import rect
#import pattern::color

struct Globals {
    target_tiles: TileAtlasDescriptor,
    src_masks: TileAtlasDescriptor,
    src_color: TileAtlasDescriptor,
};

@group(0) @binding(0) var<uniform> globals: Globals;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    #if TILED_MASK { @location(0) @interpolate(linear) mask_uv: vec2<f32>, }
    #if SOLID_PATTERN { @location(1) @interpolate(flat) color: vec4<f32>, }
    #if TILED_IMAGE_PATTERN { @location(1) @interpolate(linear) src_uv: vec2<f32>, }
};

@vertex fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @location(0) tile_index: u32,
    @location(1) a_mask: u32,
    @location(2) a_pattern: u32,
    @location(3) a_width: u32, // TODO: encode in the tile index.
    #if OPACITY { @location(4) a_opacity: f32, }
) -> VertexOutput {

    var uv = rect_get_uv(vertex_index);
    var pos = tiling_atlas_get_position(globals.target_tiles, tile_index, uv);
    pos.x += uv.x * f32(a_width) * TILE_SIZE_F32;
    var normalized_pos = pos * globals.target_tiles.inv_resolution;
    var target_pos = normalized_to_target(normalized_pos);

    #if TILED_MASK {
        var mask_uv = tiling_atlas_get_position(globals.src_masks, a_mask, uv);
    }

    #if TILED_IMAGE_PATTERN {
        var tiled_image_uv = tiling_atlas_get_position(globals.src_color, a_pattern, uv);
    }

    #if SOLID_PATTERN {
        var color = decode_color(a_pattern);
    }

    return VertexOutput(
        target_pos,
        #if TILED_MASK { mask_uv, }
        #if SOLID_PATTERN { color, }
        #if TILED_IMAGE_PATTERN { tiled_image_uv, }
        #if OPACITY { a_opacity, }
    );
}


#if TILED_MASK {
    @group(1) @binding(0) var mask_texture: texture_2d<f32>;
}

#if TILED_IMAGE_PATTERN {
    @group(2) @binding(0) var src_texture: texture_2d<f32>;
}

@fragment fn fs_main(
    #if TILED_MASK { @location(0) @interpolate(linear) mask_uv: vec2<f32>, }
    #if SOLID_PATTERN { @location(1) @interpolate(flat) in_color: vec4<f32>, }
    #if TILED_IMAGE_PATTERN { @location(1) @interpolate(linear) tiled_image_uv: vec2<f32>, }
    #if OPACITY { @location(2) @interpolate(flat) opacity: f32, }
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

    #if OPACITY {{
        color.a *= opacity;
    }}

    // Premultiply.
    color.r = color.r * color.a;
    color.g = color.g * color.a;
    color.b = color.b * color.a;

    return color;
}
