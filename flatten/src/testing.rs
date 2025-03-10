use lyon_path::geom::euclid::point2 as point;
use lyon_path::geom::{CubicBezierSegment, QuadraticBezierSegment};
use lyon_path::geom::euclid::default::{Box2D, Transform2D};
use lyon_path::math::Point;
use lyon_path::{Path, PathEvent};
use usvg::TreeParsing;

pub fn generate_bezier_curves() -> Vec<CubicBezierSegment<f32>> {

    let mut curves = Vec::new();

    for asset in &[
        "tiger.svg",
        "nehab_blender.svg",
        "nehab_lorenz.svg",
        "nehab_roads.svg",
        "nehab_spiral.svg",
        "nehab_spirograph.svg",
        "nehab_waves.svg",
    ] {
        let file = format!("assets/{asset}");
        let (_, paths) = load_svg(&file, 1.0);
        println!("{file}: {:?} paths", paths.len());
        for path in &paths {
            for evt in path.iter() {
                match evt {
                    PathEvent::Cubic { from, ctrl1, ctrl2, to } => {
                        curves.push(CubicBezierSegment { from, ctrl1, ctrl2, to });
                    }
                    PathEvent::Quadratic { from, ctrl, to } => {
                        curves.push(QuadraticBezierSegment { from, ctrl, to }.to_cubic());
                    }
                    _ => {}
                }
            }
        }
    }

    println!("generated {:?} cubic bézier curves", curves.len());

    curves
}

pub fn generate_quadratic_curves() -> Vec<QuadraticBezierSegment<f32>> {
    let cubics = generate_bezier_curves();
    let mut quads = Vec::new();
    for cubic in &cubics {
        cubic.for_each_quadratic_bezier(0.25, &mut |quad| {
            quads.push(*quad);
        })
    }

    quads
}


pub type Color = (u8, u8, u8, u8);

pub const FALLBACK_COLOR: Color = (0, 255, 0, 255 );

#[derive(Clone, Debug)]
pub enum SvgPattern {
    Color(Color),
    Gradient { color0: Color, color1: Color, from: Point, to: Point },
}

pub fn load_svg(
    filename: &str,
    scale_factor: f32,
) -> (
    Box2D<f32>,
    Vec<Path>,
) {
    let opt = usvg::Options::default();

    let svg_src = std::fs::read_to_string(filename).unwrap();
    let rtree = usvg::Tree::from_str(&svg_src, &opt).unwrap();
    let mut paths = Vec::new();

    let s = scale_factor;

    let view_box = rtree.view_box;
    for node in rtree.root.descendants() {
        use usvg::NodeExt;
        let t = node.transform();
        let transform = Transform2D::new(
            t.sx as f32, t.kx as f32, t.ky as f32, t.sy as f32, t.tx as f32, t.ty as f32,
        );

        match *node.borrow() {
            usvg::NodeKind::Path(ref usvg_path) => {
                let mut builder = Path::builder().with_svg();
                for segment in usvg_path.data.segments() {
                    use usvg::tiny_skia_path::PathSegment;
                    match segment {
                        PathSegment::MoveTo(p) => {
                            builder
                                .move_to(transform.transform_point(point(p.x, p.y)) * s);
                        }
                        PathSegment::LineTo(p) => {
                            builder
                                .line_to(transform.transform_point(point(p.x, p.y)) * s);
                        }
                        PathSegment::QuadTo (ctrl, to) => {
                            builder.quadratic_bezier_to(
                                transform.transform_point(point(ctrl.x, ctrl.y)) * s,
                                transform.transform_point(point(to.x, to.y)) * s,
                            );
                        }
                        PathSegment::CubicTo (ctrl1, ctrl2, to) => {
                            builder.cubic_bezier_to(
                                transform.transform_point(point(ctrl1.x, ctrl1.y)) * s,
                                transform.transform_point(point(ctrl2.x, ctrl2.y)) * s,
                                transform.transform_point(point(to.x, to.y)) * s,
                            );
                        }
                        PathSegment::Close => {
                            builder.close();
                        }
                    }
                }
                let path = builder.build();

                paths.push(path);
            }
            _ => {}
        }
    }

    let vb = Box2D {
        min: point(view_box.rect.x() * s, view_box.rect.y() * s),
        max: point(
            view_box.rect.x() + view_box.rect.width() * s,
            view_box.rect.y() + view_box.rect.height() * s,
        ),
    };

    (vb, paths)
}
