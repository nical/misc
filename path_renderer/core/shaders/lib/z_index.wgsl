
fn z_index_to_f32(z_index: u32) -> f32 {
    // TODO: the constant is a bit arbitrary, all we need is for the transformation
    // to preserve ordering and put the range we are interested in between 0 and 1. 
    return f32(z_index) / 65536.0;
}
