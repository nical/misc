#import pattern::color
#import gpu_store

struct Gradient {
    p0: vec2<f32>,
    p1: vec2<f32>,
    color0: vec4<f32>,
    color1: vec4<f32>,
};

fn fetch_gradient(address: u32) -> Gradient {
    var raw = gpu_store_fetch_3(address);
    var gradient: Gradient;
    gradient.p0 = raw.data0.xy;
    gradient.p1 = raw.data0.zw;
    gradient.color0 = raw.data1;
    gradient.color1 = raw.data2;

    return gradient;
}

fn pattern_vertex(pattern_pos: vec2<f32>, pattern_handle: u32) -> Pattern {
    var gradient = fetch_gradient(pattern_handle);

    var dir = gradient.p1 - gradient.p0;
    dir = dir / dot(dir, dir);
    var offset = dot(gradient.p0, dir);

    return Pattern(
        pattern_pos,
        gradient.color0,
        gradient.color1,
        vec3<f32>(dir, offset),
    );
}

fn pattern_fragment(pattern: Pattern) -> vec4<f32> {
    var d = clamp(dot(pattern.position, pattern.dir_offset.xy) - pattern.dir_offset.z, 0.0, 1.0);
    return mix(pattern.color0, pattern.color1, d);
}
