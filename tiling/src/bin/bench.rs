use tiling::*;
use lyon_path::geom::euclid::{Box2D, Size2D, vec2, size2, Transform2D};
use lyon_path::math::{Point, point};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (view_box, paths) = load_svg(&args[1]);

    let ts = 16;

    let mut tiler = Tiler::new(
        &view_box,
        size2(ts as f32, ts as f32),
        0.0,
    );
    tiler.set_flattening(true);

    let mut path_ctx = PathCtx::new(0);
    let mut z_buffer = ZBuffer::new();
    z_buffer.init(view_box.max.x as usize / ts, view_box.max.y as usize / ts);

    let mut encoder = PathfinderLikeEncoder {
        edges: Vec::new(),
        solid_tiles: Vec::new(),
        alpha_tiles: Vec::new(),
        next_tile_index: 0,
        z_buffer: &mut z_buffer,
    };

    let mut row_time: u64 = 0;
    let mut tile_time: u64 = 0;

    //let N = 1;
    let n = 1000;
    let t0 = time::precise_time_ns();
    let transform = Transform2D::create_translation(1.0, 1.0);
    for _ in 0..n {
        encoder.z_buffer.init(view_box.max.x as usize / ts + 1, view_box.max.y as usize / ts + 1);
        encoder.edges.clear();
        encoder.solid_tiles.clear();
        encoder.alpha_tiles.clear();
        encoder.next_tile_index = 0;

        let mut path_id = paths.len() as u16;
        for path in paths.iter().rev() {
            path_ctx.path_id = path_id;

            tiler.tile_path(path.iter(), Some(&transform), &mut path_ctx, &mut encoder);

            path_ctx.reset_rows();

            path_id -= 1;

            row_time += path_ctx.row_decomposition_time_ns;
            tile_time += path_ctx.tile_decomposition_time_ns;
        }
    }
    let t1 = time::precise_time_ns();

    let t = (t1 - t0) / n;

    row_time = row_time / n as u64;
    tile_time = tile_time / n as u64;

    println!("view box: {:?}", view_box);
    println!("{} edges", encoder.edges.len());
    println!("{} solid_tiles", encoder.solid_tiles.len());
    println!("{} alpha_tiles", encoder.alpha_tiles.len());
    println!("");
    println!("-> {}ns", t);
    println!("-> {}ms", t as f64 / 1000000.0);
    println!("-> row decomposition: {}ms", row_time as f64 / 1000000.0);
    println!("-> tile decomposition: {}ms", tile_time as f64 / 1000000.0);
}
