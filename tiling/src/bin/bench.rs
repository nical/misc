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

    let mut encoder = crate::raster_encoder::RasterEncoder::new(&mut z_buffer);

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
/*

    let mut dt = raqote::DrawTarget::new(900, 900);
    let mut raqote_paths = Vec::new();
    for path in &paths {
        for evt in path {
            let mut builder = raqote::PathBuilder::new();
            match evt {
                PathEvent::MoveTo(at) => {
                    builder.move_to(at.x, at.y);
                }
                PathEvent::Line(segment) => {
                    builder.line_to(segment.to.x, segment.to.y);
                }
                PathEvent::Quadratic(s) => {
                    builder.quad_to(
                        s.ctrl.x, s.ctrl.y,
                        s.to.x, s.to.y,
                    );
                }
                PathEvent::Cubic(s) => {
                    builder.cubic_to(
                        s.ctrl1.x, s.ctrl1.y,
                        s.ctrl2.x, s.ctrl2.y,
                        s.to.x, s.to.y,
                    );
                }
                PathEvent::Close(..) => {
                    builder.close();
                }
            }
            raqote_paths.push(builder.finish());
        }
    }

    let raqote_start = time::precise_time_ns();
    for _ in 0..n {
        for path in &raqote_paths {
            let source = raqote::Source::Solid(raqote::SolidSource { r: 128, g: 128, b: 128, a: 255 });
            dt.fill(path, &source, &raqote::DrawOptions::new());
        }
    }
    let raqote_time = (time::precise_time_ns() - raqote_start) / n;

    println!("-> raqote: {}ns = {}ms", raqote_time, raqote_time as f64 / 1000000.0);    
*/


    let mut tinyskia_paths = Vec::new();
    for path in &paths {
        for evt in path {
            let mut builder = tiny_skia::PathBuilder::new();
            match evt {
                PathEvent::MoveTo(at) => {
                    builder.move_to(at.x, at.y);
                }
                PathEvent::Line(segment) => {
                    builder.line_to(segment.to.x, segment.to.y);
                }
                PathEvent::Quadratic(s) => {
                    builder.quad_to(
                        s.ctrl.x, s.ctrl.y,
                        s.to.x, s.to.y,
                    );
                }
                PathEvent::Cubic(s) => {
                    builder.cubic_to(
                        s.ctrl1.x, s.ctrl1.y,
                        s.ctrl2.x, s.ctrl2.y,
                        s.to.x, s.to.y,
                    );
                }
                PathEvent::Close(..) => {
                    builder.close();
                }
            }
            if let Some(path) = builder.finish() {
                tinyskia_paths.push(path);
            } else {
                println!("skipping path!");
                println!("{:?}", path);
            }
        }
    }

    let tinyskia_start = time::precise_time_ns();
    let mut dt = tiny_skia::Pixmap::new(900, 900).unwrap();
    for _ in 0..n {
        for path in &tinyskia_paths {
            let mut paint = tiny_skia::Paint::default();
            paint.set_color_rgba8(50, 127, 150, 200);
            paint.anti_alias = true;

            dt.fill_path(&path, &paint, tiny_skia::FillRule::Winding, tiny_skia::Transform::identity(), None);
        }
    }
    let tinyskia_time = (time::precise_time_ns() - tinyskia_start) / n;

    println!("-> tiny-skia: {}ns = {}ms", tinyskia_time, tinyskia_time as f64 / 1000000.0);    
}
