// Note: the RenderTarget struct is produced by the generator.

fn canvas_to_target(pos: vec2<f32>) -> vec4<f32> {
    var p = pos * render_target.inv_resolution * 2.0 - vec2<f32>(1.0);
    p.y = -p.y;

    return vec4<f32>(p, 0.0, 1.0);
}

fn normalized_to_target(pos: vec2<f32>) -> vec4<f32> {
    var p = pos * 2.0 - vec2<f32>(1.0);
    p.y = -p.y;

    return vec4<f32>(p, 0.0, 1.0);
}

fn render_target_normalized_position(screen_pos: vec2<f32>, resolution: vec2<f32>) -> vec4<f32> {
    return normalized_to_target(screen_pos / resolution);
}
