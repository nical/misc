use lyon_path::geom::euclid::{Box2D, Transform2D};
use lyon_path::math::point;

pub fn load_svg(filename: &str) -> (Box2D<f32>, Vec<lyon_path::Path>) {
    let opt = usvg::Options::default();
    let rtree = usvg::Tree::from_file(filename, &opt).unwrap();
    let mut paths = Vec::new();

    let view_box = rtree.svg_node().view_box;
    for node in rtree.root().descendants() {
        use usvg::NodeExt;
        let t = node.transform();
        let transform = Transform2D::row_major(
            t.a as f32, t.b as f32,
            t.c as f32, t.d as f32,
            t.e as f32, t.f as f32,
        );

        if let usvg::NodeKind::Path(ref usvg_path) = *node.borrow() {
            //if usvg_path.fill.is_none() {
            //    continue;
            //}

            let mut builder = lyon_path::Path::builder();
            for segment in &usvg_path.segments {
                match *segment {
                    usvg::PathSegment::MoveTo { x, y } => {
                        builder.move_to(transform.transform_point(&point(x as f32, y as f32)));
                    }
                    usvg::PathSegment::LineTo { x, y } => {
                        builder.line_to(transform.transform_point(&point(x as f32, y as f32)));
                    }
                    usvg::PathSegment::CurveTo { x1, y1, x2, y2, x, y, } => {
                        builder.cubic_bezier_to(
                            transform.transform_point(&point(x1 as f32, y1 as f32)),
                            transform.transform_point(&point(x2 as f32, y2 as f32)),
                            transform.transform_point(&point(x as f32, y as f32)),
                        );
                    }
                    usvg::PathSegment::ClosePath => {
                        builder.close();
                    }
                }
            }
            let path = builder.build();

            paths.push(path);
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

