// This is heavily inspired from piet-gpu kernal4 (or k4) shader.
// It acts as a sort of small interpreter to execute 2d graphics
// related commands on a region of pixels.
//
// It's likely that performance won't be very good without a way
// to vectorize the loops. Unfortunately wgpu doesn't have subgroup
// operations yet which would be needed to do that in the fragment shader.
// A better is be to do this in a compute shader instead (as piet-gpu
// does) so that we at least have control over the scheduling of pixels
// within a wave. However I'm keeping this version at least until I have
// been able to measure how far this approached can be pushed in a simple
// fragment shader.

struct FragmentOutput {
    @location(0) color: vec4<f32>;
};

struct Commands {
    data: array<i32>;
};

struct Edges {
    data: array<vec4<f32>>;
};

@group(0) @binding(1) var<storage> commands: Commands;
@group(0) @binding(2) var<storage> edges: Edges;

fn even_odd(mask: f32) -> f32 {
    return 1.0 - abs((abs(mask) % 2.0) - 1.0);
}

fn non_zero(mask: f32) -> f32 {
    return even_odd(mask); // TODO!
}

fn blend_over(a: vec4<f32>, b: vec4<f32>, mask: f32) -> vec4<f32> {
    var alpha = a.a * mask;
    let out_alpha = a.a + b.a * (1.0 - a.a);
    return vec4<f32>(
        (a.r * alpha + b.r * b.a * (1.0 - alpha)) / out_alpha,
        (a.g * alpha + b.g * b.a * (1.0 - alpha)) / out_alpha,
        (a.b * alpha + b.b * b.a * (1.0 - alpha)) / out_alpha,
        out_alpha
    );
}

@fragment
fn main(
    @location(0) @interpolate(linear) in_uv_res: vec4<f32>,
    @location(1) @interpolate(flat) in_cmd_range: vec2<u32>,
) -> FragmentOutput {

    // TODO: piet-gpu was originally doing this for their blend stack and is moving away
    // because lots of drivers don't implement it properly.
    // see https://github.com/linebender/piet-gpu/issues/83#issuecomment-989001504
    var blend_stack: array<vec4<f32>, 128>;
    var blend_stack_offset: u32 = 0u;

    var OP_NOP: i32 = 0;
    var OP_BACKDROP: i32 = 1;
    var OP_FILL: i32 = 3;
    var OP_STROKE: i32 = 4;
    var OP_MASK_EVEN_ODD: i32 = 5;
    var OP_MASK_NON_ZERO: i32 = 6;
    var OP_PATTERN_COLOR: i32 = 7;
    var OP_PATTERN_IMAGE: i32 = 8;
    var OP_BLEND_OVER: i32 = 9;
    var OP_PUSH_GROUP: i32 = 10;
    var OP_POP_GROUP: i32 = 11;

    // The destination color of the pixel.
    var out_color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    // The Source color is set by pattern ops and read by blend ops
    var src_color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    // The signification of mask depends on the operation.
    // - the backdrop op initializes it as a winding number.
    // - while fill and stroke ops read and write it as winding numbers 
    // - even-odd and non-zero ops read the winding number and write it as alpha values
    // - the blend ops read it as alpha values.
    var mask: f32 = 0.0;

    var cmd_idx = in_cmd_range.x;

    loop {
        if (cmd_idx > in_cmd_range.y) {
            break;
        }

        var op = commands.data[cmd_idx];
        cmd_idx = cmd_idx + 1u;

        if (op == OP_FILL) {
            var edges_start = commands.data[cmd_idx];
            cmd_idx = cmd_idx + 1u;
            var edges_end = commands.data[cmd_idx];
            cmd_idx = cmd_idx + 1u;

            var edge_idx = edges_start;
            loop {
                if (edge_idx >= edges_end) {
                    break;
                }

                var edge = edges.data[edge_idx];
                edge_idx = edge_idx + 1;

                var from = edge.xy - in_uv_res.xy;
                var to = edge.zw - in_uv_res.xy;

                var window = vec2<f32>(
                    min(max(0.0, from.y), 1.0),
                    min(max(0.0, to.y), 1.0)
                );

                if (window.x != window.y) {
                    var t = (window - vec2<f32>(from.y, from.y)) / (to.y - from.y);
                    var xs = vec2<f32>(
                        from.x * (1.0 - t.x) + to.x * t.x,
                        from.x * (1.0 - t.y) + to.x * t.y,
                    );
                    var xmin = min(min(xs.x, xs.y), 1.0) - 1e-6; 
                    var xmax = max(xs.x, xs.y);
                    var b = min(xmax, 1.0);
                    var c = max(b, 0.0);
                    var d = max(xmin, 0.0);
                    var area = (b + 0.5 * (d * d - c * c) - xmin) / (xmax - xmin);

                    mask = mask + area * (window.x - window.y);
                }
            }
        } elseif (op == OP_BACKDROP) {
            var backdrop = f32(commands.data[cmd_idx]);
            cmd_idx = cmd_idx + 1u;
            mask = backdrop;
        } elseif (op == OP_MASK_EVEN_ODD) {
            mask = even_odd(mask);
        } elseif (op == OP_MASK_NON_ZERO) {
            mask = non_zero(mask);
        } elseif (op == OP_PATTERN_COLOR) {
            var packed = bitcast<u32>(commands.data[cmd_idx]);
            cmd_idx = cmd_idx + 1u;
            src_color = vec4<f32>(
                f32((packed >> 24u) & 255u) / 255.0,
                f32((packed >> 16u) & 255u) / 255.0,
                f32((packed >> 8u) & 255u) / 255.0,
                f32(packed & 255u) / 255.0
            );
        } elseif (op == OP_BLEND_OVER) {
            out_color = blend_over(src_color, out_color, mask);
        } elseif (op == OP_PUSH_GROUP) {
            blend_stack[blend_stack_offset] = out_color;
            blend_stack_offset = blend_stack_offset + 1u;
            out_color = vec4<f32>(0.0);
        } elseif (op == OP_POP_GROUP) {
            src_color = out_color;
            blend_stack_offset = blend_stack_offset - 1u;
            out_color = blend_stack[blend_stack_offset];
        }
    }

    return FragmentOutput(out_color);
}
