#import render_target
#import rect
#import pattern::color

struct Globals {
    resolution: vec2<f32>,
    tile_size: u32,
    tile_atlas_size: u32,
};

@group(0) @binding(0) var<uniform> globals: Globals;

struct VertexOutput {
    #if TILED_MASK { @location(0) @interpolate(linear) mask_uv: vec2<f32>, }
    #if SOLID_PATTERN { @location(1) @interpolate(flat) color: vec4<f32>, }
    #if TILED_IMAGE_PATTERN { @location(1) @interpolate(linear) src_uv: vec2<f32>, }
    @builtin(position) position: vec4<f32>,
};

fn tile_uv(tile_idx: u32, tiles_per_row: u32, uv: vec2<f32>) -> vec2<f32> {
    var tile_size = f32(globals.tile_size);
    var tile_idx = tile_idx % (tiles_per_row * tiles_per_row);
    var tile_x = f32(tile_idx % tiles_per_row) * tile_size;
    var tile_y = f32(tile_idx / tiles_per_row) * tile_size;
    return vec2<f32>(tile_x, tile_y) + uv * tile_size;
}

@vertex fn vs_main(
    @location(0) a_rect: vec4<f32>,
    @location(1) a_mask: u32,
    @location(2) a_pattern: u32,
    #if OPACITY { @location(3) a_opacity: f32, }
    @builtin(vertex_index) vertex_index: u32,
) -> VertexOutput {

    var uv = rect_get_uv(vertex_index);
    var pos = mix(a_rect.xy, a_rect.zw, uv);
    var target_pos = render_target_normalized_position(pos, globals.resolution);

    var tiles_per_row: u32 = globals.tile_atlas_size / globals.tile_size;

    #if TILED_MASK {
        var mask_uv = tile_uv(a_mask, tiles_per_row, uv);
    }

    #if SOLID_PATTERN {
        var color = decode_color(a_pattern);
    }

    #if TILED_IMAGE_PATTERN {
        var tiled_image_uv = tile_uv(a_pattern, tiles_per_row, uv);
    }

    // Uncomment to get a checkerboard-ish pattern on the tiles (to help with visualizing
    // tile boundaries).
    //if (mask_index % 2u == 0u) {
    //    color.b = color.b + 0.25;
    //    color.r = color.r - 0.25;
    //}

    return VertexOutput(
        #if TILED_MASK { mask_uv, }
        #if SOLID_PATTERN { color, }
        #if TILED_IMAGE_PATTERN { tiled_image_uv, }
        #if OPACITY { a_opacity, }
        target_pos
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
