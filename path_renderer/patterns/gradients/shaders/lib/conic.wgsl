#import gpu_buffer
#import trigonometry

fn read_conic_gradient(position: vec2f, address: u32) -> vec4f {
    var gradient = f32_gpu_buffer_fetch_2(address);
    return decode_conic_gradient(position, gradient.data0, gradient.data1);
}

fn decode_conic_gradient(position: vec2f, data0: vec4f, data1: vec4f) -> vec4f {
    let center = data0.xy;
    let scale = data0.zw;
    var start = data1.x * INV_2_PI;
    let end = data1.y * INV_2_PI;

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
