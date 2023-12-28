#import render_target
#import tiling
#import rect

fn base_vertex(vertex_index: u32, instance_data: vec4<u32>) -> BaseVertex {
    var uv = rect_get_uv(vertex_index);
    var tile = tiling_decode_instance(instance_data, uv);
    var target_position = canvas_to_target(tile.position);

    // TODO: z_index
    return BaseVertex(
        target_position,
        tile.pattern_position,
        tile.pattern_data,
        tile.mask_position,
        0u,
    );
}

fn base_fragment() -> f32 { return 1.0; }
