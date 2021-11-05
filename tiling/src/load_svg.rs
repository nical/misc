use lyon::path::geom::euclid::default::{Box2D, Transform2D};
use lyon::path::math::point;
use lyon::path::Path;
use crate::Color;

pub const FALLBACK_COLOR: Color = Color {
    r: 0,
    g: 255,
    b: 0,
    a: 255,
};

pub fn load_svg(filename: &str) -> (Box2D<f32>, Vec<(Path, Color)>) {
    let opt = usvg::Options::default();
    let rtree = usvg::Tree::from_file(filename, &opt).unwrap();
    let mut paths = Vec::new();

    let view_box = rtree.svg_node().view_box;
    for node in rtree.root().descendants() {
        use usvg::NodeExt;
        let t = node.transform();
        let transform = Transform2D::new(
            t.a as f32, t.b as f32,
            t.c as f32, t.d as f32,
            t.e as f32, t.f as f32,
        );

        if let usvg::NodeKind::Path(ref usvg_path) = *node.borrow() {
            let color = match usvg_path.fill {
                Some(ref fill) => {
                    match fill.paint {
                        usvg::Paint::Color(c) => Color {
                            r: c.red,
                            g: c.green,
                            b: c.blue,
                            a: 255,
                        },
                        _ => FALLBACK_COLOR,
                    }
                }
                None => {
                    continue;
                }
            };

            let mut builder = Path::builder().with_svg();
            for segment in &usvg_path.segments {
                match *segment {
                    usvg::PathSegment::MoveTo { x, y } => {
                        builder.move_to(transform.transform_point(point(x as f32, y as f32)));
                    }
                    usvg::PathSegment::LineTo { x, y } => {
                        builder.line_to(transform.transform_point(point(x as f32, y as f32)));
                    }
                    usvg::PathSegment::CurveTo { x1, y1, x2, y2, x, y, } => {
                        builder.cubic_bezier_to(
                            transform.transform_point(point(x1 as f32, y1 as f32)),
                            transform.transform_point(point(x2 as f32, y2 as f32)),
                            transform.transform_point(point(x as f32, y as f32)),
                        );
                    }
                    usvg::PathSegment::ClosePath => {
                        builder.close();
                    }
                }
            }
            let path = builder.build();

            paths.push((path, color));
        }
    }

    let vb = Box2D {
        min: point(
            view_box.rect.x as f32,
            view_box.rect.y as f32,
        ),
        max: point(
            view_box.rect.x as f32 + view_box.rect.width as f32,
            view_box.rect.y as f32 + view_box.rect.height as f32,
        ),
    };

    (vb, paths)
}

