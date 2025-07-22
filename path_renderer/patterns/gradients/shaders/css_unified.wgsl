// Same as css_unified.wgsl but leverages the fact that the first 4 stop offsets
// are passed via varyings for both the fast and slow paths.

#import pattern::color
#import pattern::gradient
#import pattern::linear_gradient
#import pattern::conic_gradient
#import pattern::radial_gradient

fn pattern_vertex(pattern_pos: vec2<f32>, pattern_handle: u32) -> Pattern {
    let data = f32_gpu_buffer_fetch_4(pattern_handle);
    let header = make_gradient_header(pattern_handle + 2, data.data2);

    var interpolated_data = vec4f(0.0);
    var flat_data = vec4f(0.0);
    switch gradient_header_kind(header) {
        case GRADIENT_KIND_LINEAR: {
            let dir_offset = decode_linear_gradient(data.data0);
            interpolated_data = vec4f(pattern_pos, 0.0, 0.0);
            flat_data = vec4f(dir_offset, 0.0);
        }
        case GRADIENT_KIND_CONIC: {
            interpolated_data = decode_conic_gradient(pattern_pos, data.data0, data.data1);
        }
        case GRADIENT_KIND_CSS_RADIAL: {
            interpolated_data = decode_css_radial_gradient(pattern_pos, data.data0, data.data1);
        }
        default: {}
    }

    let offsets = data.data3;
    var color0 = vec4f(0.0);
    var color1 = vec4f(0.0);
    if gradient_header_stop_count(header) <= 2 {
        let fast_path_colors = f32_gpu_buffer_fetch_2(pattern_handle + 4);
        color0 = fast_path_colors.data0;
        color1 = fast_path_colors.data1;
    }

    return Pattern(
        interpolated_data,
        flat_data,
        offsets,
        color0,
        color1,
        header,
    );
}

fn pattern_fragment(pattern: Pattern) -> vec4<f32> {
    var offset = 0.0;
    switch gradient_header_kind(pattern.gradient_header) {
        case GRADIENT_KIND_LINEAR: {
            offset = compute_linear_gradient_offset(
                pattern.interpolated_data.xy,
                pattern.flat_data.xyz,
            );
        }
        case GRADIENT_KIND_CONIC: {
            offset = compute_conic_gradient_offset(
                pattern.interpolated_data,
            );
        }
        case GRADIENT_KIND_CSS_RADIAL: {
            offset = compute_css_radial_gradient_offset(
                pattern.interpolated_data,
            );
        }
        default: {}
    }

    if gradient_header_stop_count(pattern.gradient_header) <= 2 {
        // Count includes the sentinel stop so we have at most two "real"
        // color stops.
        let extend_mode = gradient_header_extend_mode(pattern.gradient_header);
        return evaluate_simple_gradient_2(
            pattern.stop_offsets.xy,
            pattern.color0,
            pattern.color1,
            offset,
            extend_mode,
        );
    }

    return evaluate_gradient(pattern.gradient_header, offset, pattern.stop_offsets);
}
