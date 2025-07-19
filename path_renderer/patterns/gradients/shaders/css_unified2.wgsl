#import pattern::color
#import pattern::gradient
#import pattern::linear_gradient
#import pattern::conic_gradient
#import pattern::radial_gradient

fn pattern_vertex(pattern_pos: vec2<f32>, pattern_handle: u32) -> Pattern {
    let header = read_gradient_header(pattern_handle + 2);
    var interpolated_data = vec4f(0.0);
    var flat_data = vec4f(0.0);
    switch gradient_header_kind(header) {
        case GRADIENT_KIND_LINEAR: {
            let dir_offset = read_linear_gradient(pattern_handle);
            interpolated_data = vec4f(pattern_pos, 0.0, 0.0);
            flat_data = vec4f(dir_offset, 0.0);
        }
        case GRADIENT_KIND_CONIC: {
            interpolated_data = read_conic_gradient(pattern_pos, pattern_handle);
        }
        case GRADIENT_KIND_CSS_RADIAL: {
            interpolated_data = read_css_radial_gradient(pattern_pos, pattern_handle);
        }
        default: {}
    }

    // TODO: Would it be better to conditionally read these?
    // We could also leverage the fact that we've read the first
    // few vec4s to skip some reads in the slow path of the fragment
    // shader.
    let fast_path_data = f32_gpu_buffer_fetch_3(pattern_handle + 3);
    let offsets = fast_path_data.data0.xy;
    let color0 = fast_path_data.data1;
    let color1 = fast_path_data.data2;

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

    if gradient_header_stop_count(pattern.gradient_header) <= 3 {
        // Count includes the sentinel stop so we have at most two "real"
        // color stops.
        let extend_mode = gradient_header_extend_mode(pattern.gradient_header);
        return evaluate_simple_gradient_2(
            pattern.stop_offsets,
            pattern.color0,
            pattern.color1,
            offset,
            extend_mode,
        );
    }

    return evaluate_gradient(pattern.gradient_header, offset);
}
