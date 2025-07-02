#import gpu_buffer
#import trigonometry

fn read_conic_gradient(position: vec2f, address: u32) -> vec4f {
    var gradient = f32_gpu_buffer_fetch_2(address);
    let center = gradient.data0.xy;
    let scale = gradient.data0.zw;
    var start = gradient.data1.x * INV_2_PI;
    let end = gradient.data1.y * INV_2_PI;

    let da = (end - start);
    var offset_scale = 0.0;
    if da != 0.0 {
        offset_scale = 1.0 / da;
    }
    start = start * offset_scale;

    var dir = (position - center) * scale;

    return vec4f(dir, start, offset_scale);
}

fn compute_conic_gradient_offset(dir_start_scale: vec4f) -> f32 {
    let dir = dir_start_scale.xy;
    let start = dir_start_scale.z;
    let offset_scale = dir_start_scale.w;

    let current_angle = approx_atan2(dir);
    return fract(current_angle * INV_2_PI) * offset_scale - start;
}
