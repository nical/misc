use tiling::*;
use tiling::pathfinder_encoder::*;
use tiling::load_svg::*;
use lyon_path::geom::euclid::{size2, Transform2D};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (view_box, paths) = load_svg(&args[1]);

    let ts = 16;

    let mut tiler = Tiler::new(
        &TilerConfig {
            view_box,
            tile_size: size2(ts as f32, ts as f32),
            tile_padding: 0.0,
            tolerance: 0.1,
            flatten: true,
        }
    );

    let mut z_buffer = ZBuffer::new();
    z_buffer.init(view_box.max.x as usize / ts, view_box.max.y as usize / ts);

    let mut encoder = crate::wr_encoder::WrEncoder::new(&mut z_buffer);

    //let mut encoder = PathfinderLikeEncoder {
    //    edges: Vec::with_capacity(20000),
    //    solid_tiles: Vec::with_capacity(2000),
    //    alpha_tiles: Vec::with_capacity(5000),
    //    next_tile_index: 0,
    //    z_buffer: &mut z_buffer,
    //    z_index: 0,
    //};

    let mut row_time: u64 = 0;
    let mut tile_time: u64 = 0;

    let n = 100;
    let t0 = time::precise_time_ns();
    let transform = Transform2D::create_translation(1.0, 1.0);
    for _ in 0..n {
        encoder.reset();
        encoder.z_buffer.init(view_box.max.x as usize / ts + 1, view_box.max.y as usize / ts + 1);
        encoder.z_index = paths.len() as u16;

        // Loop over the paths in front-to-back order to take advantage of
        // occlusion culling.
        for path in paths.iter().rev() {

            tiler.tile_path(path.iter(), Some(&transform), &mut encoder);

            encoder.z_index -= 1;

            row_time += tiler.row_decomposition_time_ns;
            tile_time += tiler.tile_decomposition_time_ns;
        }

        // Since the paths were processed front-to-back we have to reverse
        // the alpha tiles to render then back-to-front.
        // This surprisingly doesn't show up in profiles.
        encoder.mask_tiles.reverse();
    }
    let t1 = time::precise_time_ns();

    let t = (t1 - t0) / n;

    row_time = row_time / n as u64;
    tile_time = tile_time / n as u64;

    println!("view box: {:?}", view_box);
    //println!("{} edges", encoder.edges.len());
    println!("{} solid_tiles", encoder.solid_tiles.len());
    println!("{} alpha_tiles", encoder.mask_tiles.len());
    println!("{} mask bytes", encoder.mask_buffer.len());
    println!("");
    println!("-> {}ns", t);
    println!("-> {}ms", t as f64 / 1000000.0);
    println!("-> row decomposition: {}ms", row_time as f64 / 1000000.0);
    println!("-> tile decomposition: {}ms", tile_time as f64 / 1000000.0);
}
