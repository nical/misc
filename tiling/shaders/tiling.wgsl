struct TileAtlasDescriptor {
    tile_size: f32,
    inv_atlas_size: f32,
    masks_per_row: u32,
};

fn tiling_atlas_get_uv(atlas: TileAtlasDescriptor, tile_index: u32, uv: vec2<f32>) -> vec2<f32> {
    let index = tile_index % (atlas.masks_per_row * atlas.masks_per_row);
    var tile_x = f32(index % atlas.masks_per_row);
    var tile_y = f32(index / atlas.masks_per_row);
    var target_pos = (vec2<f32>(tile_x, tile_y) + uv) * atlas.tile_size;
    var atlas_uv = target_pos * atlas.inv_atlas_size;

    return atlas_uv;
}
