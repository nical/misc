#import gpu_buffer

const GRADIENT_EXTEND_MODE_CLAMP: u32 = 0;
const GRADIENT_EXTEND_MODE_REPEAT: u32 = 1;

// Fetch information about the gradient stops that is independent from
// the pixel position (preferrably used in the vertex shader).
fn read_gradient_header(base_address: u32) -> vec4u {
    let header = f32_gpu_buffer_fetch_1(base_address);
    let count = header.x;
    let extend_mode = header.y;
    let offsets_address = base_address + 1;
    let colors_address = offsets_address + u32(ceil(count * 0.25));

    return vec4u(
        u32(count),
        u32(extend_mode),
        offsets_address,
        colors_address,
    );
}

fn apply_extend_mode(offset: f32, extend_mode: u32) -> f32 {
    if extend_mode == GRADIENT_EXTEND_MODE_REPEAT {
        return fract(offset);
    }

    return clamp(offset, 0.0, 1.0);
}

fn evaluate_gradient(gradient_header: vec4u, original_offset: f32) -> vec4f {
    let count = gradient_header.x;
    let extend_mode = gradient_header.y;
    var addr: u32 = gradient_header.z;
    let colors_base_address = gradient_header.w;

    var offset = apply_extend_mode(original_offset, extend_mode);

    var end_addr: u32 = addr + count;

    // Index of the first gradient stop that is after
    // the current offset.
    var index: u32 = 0;

    var stop_offsets = f32_gpu_buffer_fetch_1(addr);
    var prev_stop_offset = stop_offsets.x;
    var stop_offset = stop_offsets.x;

    while (addr < end_addr) {

        stop_offset = stop_offsets.x;
        if stop_offset > offset { break; }
        index += 1;

        prev_stop_offset = stop_offset;
        stop_offset = stop_offsets.y;
        if stop_offset > offset { break; }
        index += 1;

        prev_stop_offset = stop_offset;
        stop_offset = stop_offsets.z;
        if stop_offset > offset { break; }
        index += 1;

        prev_stop_offset = stop_offset;
        stop_offset = stop_offsets.w;
        if stop_offset > offset { break; }
        index += 1;

        addr += 1;
        stop_offsets = f32_gpu_buffer_fetch_1(addr);
    }

    let color_pair_address = colors_base_address + max(1, index) - 1;
    let color_pair = f32_gpu_buffer_fetch_2(color_pair_address);

    var d = stop_offset - prev_stop_offset;
    var factor = 0.0;
    if d > 0.0 {
        factor = clamp((offset - prev_stop_offset) / d, 0.0, 1.0);
    }

    return mix(color_pair.data0, color_pair.data1, factor);
}

fn evaluate_simple_gradient_2(
    stop_offsets: vec2f,
    color0: vec4f,
    color1: vec4f,
    original_offset: f32,
    extend_mode: u32,
) -> vec4f {
    var offset = apply_extend_mode(original_offset, extend_mode);
    var d = stop_offsets.y - stop_offsets.x;
    var factor = 0.0;

    if offset < stop_offsets.x {
        factor = 0.0;
        d = 1.0;
    } else if offset > stop_offsets.y {
        factor = 1.0;
        d = 1.0;
    } else if d > 0.0 {
        factor = clamp((offset - stop_offsets.x) / d, 0.0, 1.0);
    }

    return mix(color0, color1, factor);
}
