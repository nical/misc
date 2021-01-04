use tiling::*;
use tiling::load_svg::*;
use lyon_path::geom::euclid::{size2};

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
    z_buffer.init(view_box.max.x as usize / ts + 1, view_box.max.y as usize / ts + 1);

    let mut encoder = Encoder {
        z_buffer: &mut z_buffer,
        z_index: paths.len() as u16,
        culled_opaque_tiles: 0,
        culled_solid_tiles: 0,
        culled_alpha_tiles: 0,
        visible_opaque_tiles: 0,
        visible_solid_tiles: 0,
        visible_alpha_tiles: 0,
        empty_tiles: 0,
    };

    println!("{}", svg_fmt::BeginSvg { w: view_box.max.x, h: view_box.max.y });

    println!("<!-- {} paths -->", paths.len());
    for path in paths.iter().rev() {

        tiler.tile_path(path.iter(), None, &mut encoder);

        encoder.z_index -= 1;
    }

    println!("\n<!-- size: {:?} ({}px, {:?} tiles) -->", view_box.size(), view_box.area(), (view_box / ts as f32).round_out().size().to_i32() );

    let stats = &|name, val| {
        let px = val * (ts * ts) as usize;
        let percent = px as f32 / view_box.area() as f32 * 100.0;
        println!("<!-- {} {} ({}px, {:.2}%) -->", val, name, px, percent);
    };

    stats("visible solid tiles", encoder.visible_solid_tiles);
    stats("visible alpha tiles", encoder.visible_alpha_tiles);
    stats("culled opaque solid tiles", encoder.culled_opaque_tiles);
    stats("culled transparent solid tiles", encoder.culled_solid_tiles);
    stats("culled alpha tiles", encoder.culled_alpha_tiles);
    stats("empty tiles", encoder.empty_tiles);

    println!("{}", svg_fmt::EndSvg);
}

struct Encoder<'l> {
    z_buffer: &'l mut ZBuffer,
    z_index: u16,
    culled_opaque_tiles: usize,
    culled_solid_tiles: usize,
    culled_alpha_tiles: usize,
    visible_opaque_tiles: usize,
    visible_solid_tiles: usize,
    visible_alpha_tiles: usize,
    empty_tiles: usize,
}

impl<'l> TileEncoder for Encoder<'l> {
    fn encode_tile(&mut self, tile: &TileInfo, edges: &[ActiveEdge]) {
        let opaque = true;
        let mut solid = false;
        if edges.is_empty() {
            if tile.backdrop_winding % 2 != 0 {
                solid = true;
            } else {
                self.empty_tiles += 1;
                return;
            }
        }

        if !self.z_buffer.test(tile.x, tile.y, self.z_index, opaque && solid) {
            println!("<!-- culled tile {} {} path {}-->", tile.x, tile.y, self.z_index);
            if solid && opaque {
                self.culled_opaque_tiles += 1;
            } else if solid {
                self.culled_solid_tiles += 1;
            } else {
                self.culled_alpha_tiles += 1;
            }
            return;
        }

        if solid {
            println!("<!-- solid tile {} {} path {}-->", tile.x, tile.y, self.z_index);
            if opaque {
                self.visible_opaque_tiles += 1;
            } else {
                self.visible_solid_tiles += 1;
            }

            use std::ops::Rem;
            let solid_tile_color = svg_fmt::Color { r: 0, g: 0, b: (self.z_index * 17).rem(150) as u8 + 100 };
            println!("  {}",
                svg_fmt::rectangle(
                    tile.inner_rect.min.x,
                    tile.inner_rect.min.y,
                    tile.inner_rect.size().width,
                    tile.inner_rect.size().height,
                )
                .fill(solid_tile_color)
                .opacity(0.4)
            );
        } else {
            println!("<!-- regular tile {} {} path {}-->", tile.x, tile.y, self.z_index);
            self.visible_alpha_tiles += 1;
        }

        for edge in edges {
            let edge = edge.clip_horizontally(tile.outer_rect.min.x .. tile.outer_rect.max.x);
            let color = if edge.winding > 0 { svg_fmt::white() } else { svg_fmt::red() };
            println!("  {}", svg_fmt::line_segment(edge.from.x, edge.from.y, edge.to.x, edge.to.y).color(color));
        }
        println!("  {}",
            svg_fmt::rectangle(
                tile.inner_rect.min.x,
                tile.inner_rect.min.y,
                tile.inner_rect.size().width,
                tile.inner_rect.size().height,
            )
            .fill(svg_fmt::Fill::None)
            .stroke(svg_fmt::Stroke::Color(svg_fmt::black(), 0.1))
        );
        println!("  {}",
            svg_fmt::rectangle(
                tile.outer_rect.min.x,
                tile.outer_rect.min.y,
                tile.outer_rect.size().width,
                tile.outer_rect.size().height,
            )
            .fill(svg_fmt::Fill::None)
            .stroke(svg_fmt::Stroke::Color(svg_fmt::green(), 0.1))
        );
    }
}
