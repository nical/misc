#import render_target

fn geometry(vertex_index: u32, canvas_position: vec2<f32>, z_index: u32, pattern_data: u32) -> Geometry {
    var target_position = canvas_to_target(canvas_position);

    var position = vec4<f32>(
        target_position.x,
        target_position.y,
        f32(z_index) / 8192.0,
        1.0,
    );

    return Geometry(
        position,
        canvas_position,
        pattern_data,
        // No suport for masks.
        vec2<f32>(0.0),
        0u,
    );
}
