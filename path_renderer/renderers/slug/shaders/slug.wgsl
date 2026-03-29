#import render_task
#import rect
#import gpu_buffer
#import z_index
#import slug_coverage

fn fetch_transform(address: u32) -> mat3x2<f32> {
    if address == GPU_BUFFER_ADDRESS_NONE {
        return mat3x2<f32>(
            vec2<f32>(1.0, 0.0),
            vec2<f32>(0.0, 1.0),
            vec2<f32>(0.0, 0.0),
        );
    }
    let d = f32_gpu_buffer_fetch_2(address);
    return mat3x2<f32>(d.data0.xy, d.data0.zw, d.data1.xy);
}

fn geometry_vertex(
    vertex_index: u32,
    local_rect: vec4<f32>,
    band_params: vec4<f32>,
    band_loc: u32,
    band_max: u32,
    z_index: u32,
    pattern: u32,
    render_task: u32,
    flags_transform: u32,
) -> GeometryVertex {
    let transform_id = flags_transform & 0xFFFFFu;
    let uv = rect_get_uv(vertex_index);

    let transform = fetch_transform(transform_id);

    let local_position = mix(local_rect.xy, local_rect.zw, uv);

    let task = render_task_fetch(render_task);
    var clamped = local_position;
    if transform_id == GPU_BUFFER_ADDRESS_NONE {
        clamped = clamp(clamped, task.clip.xy, task.clip.zw);
    }

    let canvas_position = (transform * vec3(clamped, 1.0)).xy;
    let target_position = render_task_target_position(task, canvas_position);

    let position = vec4<f32>(
        target_position.x,
        target_position.y,
        z_index_to_f32(z_index),
        1.0,
    );

    let glyph_loc = vec2<i32>(
        i32(band_loc & 0xFFFFu),
        i32(band_loc >> 16u),
    );
    let band_max_val = vec2<i32>(
        i32(band_max & 0xFFFFu),
        i32(band_max >> 16u),
    );

    return GeometryVertex(
        position,
        local_position,
        pattern,
        // varyings:
        local_position,  // texcoord
        band_params,
        glyph_loc,
        band_max_val,
    );
}

fn geometry_fragment(
    texcoord: vec2<f32>,
    band_params: vec4<f32>,
    glyph_loc: vec2<i32>,
    band_max_val: vec2<i32>,
) -> f32 {
    return slug_render(texcoord, band_params, glyph_loc, band_max_val);
}
