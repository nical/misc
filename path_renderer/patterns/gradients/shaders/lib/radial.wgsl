#import gpu_buffer

fn read_css_radial_gradient(position: vec2f, address: u32) -> vec3f {
    var gradient = f32_gpu_buffer_fetch_2(address);
    let center = gradient.data0.xy;
    // focal parameter data0.zw is ignored in the css version.

    let scale = gradient.data1.xy;
    var start_radius = gradient.data1.z;
    let end_radius = gradient.data1.w;

    let dr = end_radius - start_radius;
    var radius_scale = 0.0;
    if (dr != 0.0) {
        radius_scale = 1.0 / dr;
    }

    // Normalize start_radius and the position
    start_radius = start_radius * radius_scale;
    var normalized_pos = (position - center) * scale * radius_scale;
    return vec3f(normalized_pos, start_radius);
}

fn compute_css_radial_gradient_offset(position_and_start: vec3f) -> f32 {
    return length(position_and_start.xy) - position_and_start.z;
}

struct SvgRadialGradient {
    position: vec2f,
    center: vec2f,
    start_radius: f32,
    end_radius: f32,
};

fn read_svg_radial_gradient(position: vec2f, address: u32) -> SvgRadialGradient {
    var gradient = f32_gpu_buffer_fetch_2(address);
    let center = gradient.data0.xy;
    let focal = gradient.data0.zw;

    let scale = gradient.data1.xy;
    var start_radius = gradient.data1.z;
    let end_radius = gradient.data1.w;

    let dr = end_radius - start_radius;
    var radius_scale = 0.0;
    if (dr != 0.0) {
        radius_scale = 1.0 / dr;
    }

    // Normalize start_radius and the position
    //start_radius = start_radius * radius_scale;
    var normalized_pos = (position - focal) * scale;
    var normalized_center = (center - focal) * scale;
    // TODO: WR has an extra parameter to tweak the y scale here.
    return SvgRadialGradient(
        normalized_pos,
        normalized_center,
        start_radius,
        end_radius
    );
}

fn compute_svg_radial_gradient_offset(position_and_center: vec4f, start_end: vec2f) -> f32 {
    let pos = position_and_center.xy;
    let center = position_and_center.zw;
    let start_radius = start_end.x;
    let end_radius = start_end.y;

    // TODO: handle x == 0
    //if pos.x != 0.0 {
        // Line equation from focal to point `y = mx + u` where u is zero because
        // the focal point is at the origin.
        let m = pos.y / pos.x;
        // Intersection between the line and the circle around center with end_radius
        // is a quadratic equation y = ax² + bx + c;
        let a = m * m + 1.0;
        let b = -2.0 * (m * center.y + center.x);
        let c = center.y * center.y
            + center.x * center.x
            - end_radius * end_radius;

        let delta = b * b - 4.0 * a * c;
        if delta < 0.0 {
            // We are outside of the cone. SVG spec defaults to 0.
            return 0.0;
        }
        // `a = m² + 1` is always positive and non-null.
        let inv_2a = 0.5 / a;
        let sqrt_delta = sqrt(delta);
        // TODO: explain why we pick the root based on the sign of pox.x.
        let x = (-b + sqrt_delta * sign(pos.x)) * inv_2a;
        // derive y from the line equation.
        let y = m * x;
    //}

    let k = start_radius / end_radius;
    let pc = vec2f(x, y);
    if dot(pc, pos) < 0.0 {
        return 0.0;
    }

    let t = length(pos - pc * k) / length(pc * (1.0 - k));
    return t;
}
