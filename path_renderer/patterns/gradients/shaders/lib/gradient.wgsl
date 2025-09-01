#import gpu_buffer

const GRADIENT_EXTEND_MODE_CLAMP: u32 = 0;
const GRADIENT_EXTEND_MODE_REPEAT: u32 = 1;

const GRADIENT_KIND_LINEAR: u32 = 0;
const GRADIENT_KIND_CONIC: u32 = 1;
const GRADIENT_KIND_CSS_RADIAL: u32 = 2;
const GRADIENT_KIND_SVG_RADIAL: u32 = 3;

// Fetch information about the gradient stops that is independent from
// the pixel position (preferrably used in the vertex shader).
fn read_gradient_header(base_address: u32) -> vec4u {
    let header = f32_gpu_buffer_fetch_1(base_address);
    return make_gradient_header(base_address, header);
}

fn make_gradient_header(base_address: u32, payload: vec4f) -> vec4u {
    let gradient_kind = u32(payload.z);
    let count = u32(payload.x);
    let extend_mode = payload.y;
    let colors_address = base_address + 1;
    let offsets_address = colors_address + count;

    return vec4u(
        gradient_kind,
        count | (u32(extend_mode) << 24),
        offsets_address,
        colors_address,
    );
}

fn gradient_header_kind(header: vec4u) -> u32 {
    return header.x;
}

fn gradient_header_extend_mode(header: vec4u) -> u32 {
    return header.y >> 24;
}

fn gradient_header_stop_count(header: vec4u) -> u32 {
    return header.y & 0x0FFFFFF;
}

fn gradient_header_offsets_address(header: vec4u) -> u32 {
    return header.z;
}

fn gradient_header_colors_address(header: vec4u) -> u32 {
    return header.w;
}

fn apply_extend_mode(offset: f32, extend_mode: u32) -> f32 {
    if (extend_mode) == GRADIENT_EXTEND_MODE_REPEAT {
        return fract(offset);
    }

    return clamp(offset, 0.0, 1.0);
}

fn evaluate_gradient(gradient_header: vec4u, original_offset: f32, first_offsets: vec4f) -> vec4f {
    let count = gradient_header_stop_count(gradient_header);
    let extend_mode = gradient_header_extend_mode(gradient_header);
    var addr: u32 = gradient_header.z;
    let colors_base_address = gradient_header.w;

    var offset = apply_extend_mode(original_offset, extend_mode);

    // Index of the first gradient stop that is after
    // the current offset.
    var index: u32 = 0;

    // We take advantage of the fact that the first 4 stop offsets were passed via
    // varyings to avoid fetching them from the buffer.
    var stop_offsets = first_offsets;
    var prev_stop_offset = stop_offsets.x;
    var stop_offset = stop_offsets.x;

    loop {
        prev_stop_offset = stop_offset;
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
        if index >= count {
            // If we exit the loop through here, it means that there isn't a
            // gradient stop after the current offset. In this case we must
            // use the color of the last gradient stop. We do so by noticing
            // that the index is greater or equal to the stop count and set
            // the interpolation factor to 1.0.
            break;
        }

        stop_offsets = f32_gpu_buffer_fetch_1(addr);
    }

    // Clamp the address to the valid range of gradient stops.
    let color_pair_address = colors_base_address + min(max(1, index), count - 1) - 1;
    let color_pair = f32_gpu_buffer_fetch_2(color_pair_address);

    // If we are before the first gradient stop, stop_offset and prev_stop_offset
    // will be equal, in which case the interpolaiton factor will remain zero.
    var d = stop_offset - prev_stop_offset;
    var factor = 0.0;
    if index >= count {
        // The current offset is after the last gradient stop.
        factor = 1.0;
    } else if d > 0.0 {
        // We are between two gradient stops, compute the interpolation factor.
        factor = clamp((offset - prev_stop_offset) / d, 0.0, 1.0);
    }

    //return vec4f(factor, factor, factor, 1.0);
    //return vec4f(offset, offset, offset, 1.0);
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

// From https://www.shadertoy.com/view/sd2yDd
fn dithering_noise(coords: vec2f) -> f32 {
    return fract(52.9829189 * fract(dot(coords, vec2f(0.06711056, 0.00583715)))) - 0.5;
}

fn dither_gradient_offset(offset: f32, coords: vec2f) -> f32 {
    let dither = dithering_noise(coords) * 0.05;
    return smoothstep(0.0, 1.0, clamp(offset + dither, 0.0, 1.0));
}
