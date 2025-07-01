#import gpu_buffer

fn read_radial_gradient(position: vec2f, address: u32) -> vec3f {
    var gradient = f32_gpu_buffer_fetch_2(address);
    let center = gradient.data0.xy;
    let scale = gradient.data0.zw;
    var start_radius = gradient.data1.x;
    let end_radius = gradient.data1.y;

    let dr = end_radius - start_radius;
    var radius_scale = 0.0;
    if (dr != 0.0) {
        radius_scale = 1.0 / dr;
    }

    // Normalize start_radius and the position
    start_radius = start_radius * radius_scale;
    var normalized_pos = (position - center) * scale * radius_scale;
    // TODO: WR has an extra parameter to tweak the y scale here.
    return vec3f(normalized_pos, start_radius);
}

fn compute_radial_gradient_offset(position_and_start: vec3f) -> f32 {
    return length(position_and_start.xy) - position_and_start.z;
}
