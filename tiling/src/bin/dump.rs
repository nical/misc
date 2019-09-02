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
    };

    println!("{}", svg_fmt::BeginSvg { w: view_box.max.x, h: view_box.max.y });

    println!("<!-- {} paths -->", paths.len());
    let mut z_index = paths.len() as u16;
    for path in paths.iter().rev() {

        tiler.tile_path(path.iter(), None, z_index, &mut encoder);

        z_index -= 1;
    }

    println!("{}", svg_fmt::EndSvg);
}

struct Encoder<'l> {
    z_buffer: &'l mut ZBuffer,
}

impl<'l> TileEncoder for Encoder<'l> {
    fn encode_tile(&mut self, tile: &TileInfo, edges: &[ActiveEdge]) {
        let mut solid = false;
        if edges.is_empty() {
            if tile.backdrop_winding % 2 != 0 {
                solid = true;
            } else {
                return;
            }
        }

        if !self.z_buffer.test(tile.x, tile.y, tile.z_index, solid) {
            println!("<!-- culled tile {} {} path {}-->", tile.x, tile.y, tile.z_index);
            return;
        }

        if solid {
            println!("<!-- solid tile {} {} path {}-->", tile.x, tile.y, tile.z_index);

            use std::ops::Rem;
            let solid_tile_color = svg_fmt::Color { r: 0, g: 0, b: (tile.z_index * 17).rem(150) as u8 + 100 };
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
            println!("<!-- regular tile {} {} path {}-->", tile.x, tile.y, tile.z_index);
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
