#import rect
#import render_target
#import tiling

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) pos: vec2<f32>,
    @location(1) @interpolate(flat) rect: vec4<f32>,
    @location(2) @interpolate(flat) invert: u32,
};

@vertex fn vs_main(
    @location(0) instance: vec4<u32>,
    @builtin(vertex_index) vertex_index: u32,
) -> VertexOutput {
    let uv = rect_get_uv(vertex_index);
    let pos = tiling_decode_position(instance.x, uv);
    let target_pos = canvas_to_target(pos);
    let invert = instance.y;
    let rect = instance.zw;
    let mask = 0xFFFFu;
    let k = TILE_SIZE_F32 / f32(mask);
    let x0 = f32(rect.x >> 16u) * k;
    let x1 = f32(rect.x & mask) * k;
    let y0 = f32(rect.y >> 16u) * k;
    let y1 = f32(rect.y & mask) * k;

    return VertexOutput(
        target_pos,
        uv * TILE_SIZE_F32,
        vec4<f32>(x0, x1, y0, y1),
        invert,
    );
}

@fragment fn fs_main(
    @location(0) pos: vec2<f32>,
    @location(1) @interpolate(flat) rect: vec4<f32>,
    @location(2) @interpolate(flat) invert: u32,
) -> @location(0) vec4<f32> {
    let m0 = rect.xy - pos;
    let m1 = pos - rect.zw;
    let zero = vec2<f32>(0.0);
    let one = vec2<f32>(1.0);
    let m0_clamped = max(zero, min(one, m0));
    let m1_clamped = max(zero, min(one, m1));
    let d2 = max(m0_clamped, m1_clamped);
    var mask = 1.0 - max(d2.x, d2.y);

    if (invert != 0u) {
        mask = 1.0 - mask;
    }
    return vec4<f32>(mask, mask, mask, mask);
}
