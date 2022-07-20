
struct FragmentOutput {
    @location(0) color: vec4<f32>,
};


@fragment
fn main(
    @location(0) @interpolate(flat) a_color: vec4<f32>,
) -> FragmentOutput {
    return FragmentOutput(a_color);
}
