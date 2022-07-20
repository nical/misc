
struct MaskBuffer {
    // Ideally we would want a buffer of 64k u8 but bytes aren't supported by wgpu (yet?)
    // so we interpret the buffer as u32 and do some bit fiddling instead.
    data: array<u32, 16384>,
};

@group(1) @binding(0) var<storage> mask_buffer: MaskBuffer;

struct FragmentOutput {
    @location(0) color: vec4<f32>,
};


@fragment
fn main(
    @location(0) @interpolate(linear) local_uv: vec2<f32>,
    @location(1) @interpolate(flat) src_offset: u32,
) -> FragmentOutput {

    var offset = src_offset + u32(floor(local_uv.y)) * 16u + u32(floor(local_uv.x));

    var payload = mask_buffer.data[offset / 4u];
    let shift = (offset % 4u) * 8u;
    var alpha = f32((payload >> shift) & 255u) / 255.0;

    return FragmentOutput(vec4<f32>(alpha, alpha, alpha, alpha));
}
