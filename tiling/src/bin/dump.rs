use tiling::*;
use lyon_path::geom::euclid::{Box2D, Size2D, vec2, size2, Transform2D};
use lyon_path::math::{Point, point};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (view_box, paths) = load_svg(&args[1]);

    let ts = 10;

    let mut tiler = Tiler::new(
        &view_box,
        size2(ts as f32, ts as f32),
        0.0,
    );

    let mut z_buffer = ZBuffer::new();
    z_buffer.init(view_box.max.x as usize / ts + 1, view_box.max.y as usize / ts + 1);

    let mut encoder = Encoder {
        z_buffer: &mut z_buffer,
    };

    let mut path_ctx = PathCtx::new(0);

    println!("{}", svg_fmt::BeginSvg { w: view_box.max.x, h: view_box.max.y });

    println!("<!-- {} paths -->", paths.len());
    let mut path_id = paths.len() as u16;
    for path in paths.iter().rev() {
        path_ctx.path_id = path_id;

        tiler.tile_path(path.iter(), None, &mut path_ctx, &mut encoder);

        path_ctx.reset_rows();

        path_id -= 1;
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

        if !self.z_buffer.test(tile.x, tile.y, tile.path_id, solid) {
            println!("<!-- culled tile {} {} path {}-->", tile.x, tile.y, tile.path_id);
            return;
        }

        if solid {
            println!("<!-- solid tile {} {} path {}-->", tile.x, tile.y, tile.path_id);            
        }

        for edge in edges {
            let edge = edge.clip_horizontally(tile.outer_rect.min.x .. tile.outer_rect.max.x);
            let color = if edge.winding > 0 {
                svg_fmt::Color { r: 0, g: 0, b: 255 }                
            } else{
                svg_fmt::Color { r: 255, g: 0, b: 0 }                
            };
            println!("  {}", svg_fmt::line_segment(edge.from.x, edge.from.y, edge.to.x, edge.to.y).color(color));
        }
        if solid {
            println!("  {}",
                svg_fmt::rectangle(
                    tile.inner_rect.min.x,
                    tile.inner_rect.min.y,
                    tile.inner_rect.size().width,
                    tile.inner_rect.size().height,
                )
                .fill(svg_fmt::blue())
                .opacity(0.3)
            );
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
