#import gpu_buffer

fn linear_gradient_vertex(position: vec2f, endpoints: vec4f) -> f32 {
    let p0 = endpoints.xy;
    let p1 = endpoints.zw;

    var dir = p1 - p0;
    dir = dir / dot(dir, dir);
    var start_offset = dot(p0, dir);
    var offset = dot(position, dir) - start_offset;

    // Adding an epsilon value to the offset works around interpolation precision
    // issues which cause hard stops to look wrong for in cases. The same workaround
    // exists in Skia and WebRender.
    offset += 0.000001;

    return offset;
}
