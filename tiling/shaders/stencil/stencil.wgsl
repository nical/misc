#import render_target

@vertex fn vs_main(@location(0) position: vec2<f32>) -> @builtin(position) vec4<f32> {
    var pos = canvas_to_target(position);
    return vec4<f32>(pos.x, pos.y, 0.0, 1.0);
}

@fragment fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0);
}
