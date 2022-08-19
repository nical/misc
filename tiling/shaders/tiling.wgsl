let TILE_SIZE_U32: u32 = 16u;
let TILE_SIZE_F32: f32 = 16.0;

struct TileAtlasDescriptor {
    inv_resolution: vec2<f32>,
    tiles_per_row: u32,
    tiles_per_atlas: u32,
};

fn tiling_atlas_get_position(atlas: TileAtlasDescriptor, tile_idx: u32, uv: vec2<f32>) -> vec2<f32> {
    var tile_idx = tile_idx % atlas.tiles_per_atlas;
    var tile_x = f32(tile_idx % atlas.tiles_per_row);
    var tile_y = f32(tile_idx / atlas.tiles_per_row);
    return (vec2<f32>(tile_x, tile_y) + uv) * TILE_SIZE_F32;
}

fn tiling_atlas_get_uv(atlas: TileAtlasDescriptor, tile_idx: u32, uv: vec2<f32>) -> vec2<f32> {
    return tiling_atlas_get_position(atlas, tile_idx, uv) * atlas.inv_resolution;
}
