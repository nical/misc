#import gpu_buffer

fn read_linear_gradient(address: u32) -> vec3f {
    var endpoints = f32_gpu_buffer_fetch_1(address);
    return decode_linear_gradient(endpoints);
}

fn decode_linear_gradient(endpoints: vec4f) -> vec3f {
    let p0 = endpoints.xy;
    let p1 = endpoints.zw;

    var dir = p1 - p0;
    dir = dir / dot(dir, dir);
    var offset = dot(p0, dir);

    return vec3f(dir, offset);
}

fn compute_linear_gradient_offset(position: vec2f, gradient_dir_offset: vec3f) -> f32 {
    return dot(position, gradient_dir_offset.xy) - gradient_dir_offset.z;
}
