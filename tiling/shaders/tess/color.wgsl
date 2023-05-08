#import render_target
#import pattern::color

struct VertexOutput {
    @location(0) color: vec4<f32>,
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(
    @location(0) position: vec2<f32>,
    @location(1) z_index: u32,
    @location(2) pattern: u32,
) -> VertexOutput {
    var color = decode_color(pattern);
    var target_position = canvas_to_target(position);

    var position = vec4<f32>(
        target_position.x,
        target_position.y,
        f32(z_index) / 8192.0,
        1.0,
    );
    return VertexOutput(color, position);
}

@fragment
fn fs_main(@location(0) color: vec4<f32>) -> @location(0) vec4<f32> {
    return color;
}
