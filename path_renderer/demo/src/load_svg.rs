use core::stroke::{LineCap, LineJoin};
use std::sync::Arc;

use core::path::Path;
use core::Color;
use lyon::path::geom::euclid::default::{Box2D, Transform2D};
use lyon::path::math::{point, Point};
use usvg::TreeParsing;

pub const FALLBACK_COLOR: Color = Color {
    r: 0,
    g: 255,
    b: 0,
    a: 255,
};

#[derive(Clone, Debug)]
pub enum SvgPattern {
    Color(Color),
    Gradient {
        color0: Color,
        color1: Color,
        from: Point,
        to: Point,
    },
}

pub struct Stroke {
    pub pattern: SvgPattern,
    pub line_width: f32,
    pub line_cap: LineCap,
    pub line_join: LineJoin,
}

pub fn load_svg(
    filename: &str,
    scale_factor: f32,
) -> (
    Box2D<f32>,
    Vec<(Arc<Path>, Option<SvgPattern>, Option<Stroke>)>,
) {
    let opt = usvg::Options::default();

    let svg_src = std::fs::read_to_string(filename).unwrap();
    let rtree = usvg::Tree::from_str(&svg_src, &opt).unwrap();
    let mut paths = Vec::new();

    let s = scale_factor;

    let view_box = rtree.view_box;
    for node in rtree.root.descendants() {
        use usvg::NodeExt;
        let t = node.abs_transform();
        let transform = Transform2D::new(
            t.sx as f32, t.kx as f32, t.ky as f32, t.sy as f32, t.tx as f32, t.ty as f32,
        );

        match *node.borrow() {
            usvg::NodeKind::Path(ref usvg_path) => {
                let fill_pattern = usvg_path.fill.as_ref().map(|fill| match &fill.paint {
                    usvg::Paint::Color(c) => {
                        SvgPattern::Color(Color::new(c.red, c.green, c.blue, 255))
                    }
                    usvg::Paint::LinearGradient(gradient) => {
                        let color0 = gradient
                            .base
                            .stops
                            .first()
                            .map(|stop| Color {
                                r: stop.color.red,
                                g: stop.color.green,
                                b: stop.color.blue,
                                a: (stop.opacity.get() * 255.0) as u8,
                            })
                            .unwrap_or(FALLBACK_COLOR);
                        let color1 = gradient
                            .base
                            .stops
                            .last()
                            .map(|stop| Color {
                                r: stop.color.red,
                                g: stop.color.green,
                                b: stop.color.blue,
                                a: (stop.opacity.get() * 255.0) as u8,
                            })
                            .unwrap_or(FALLBACK_COLOR);
                        SvgPattern::Gradient {
                            color0,
                            color1,
                            from: point(gradient.x1 as f32, gradient.y1 as f32),
                            to: point(gradient.x2 as f32, gradient.y2 as f32),
                        }
                    }
                    _ => SvgPattern::Color(FALLBACK_COLOR),
                });

                let stroke_pattern = usvg_path.stroke.as_ref().map(|stroke| Stroke {
                    pattern: match &stroke.paint {
                        usvg::Paint::Color(c) => {
                            SvgPattern::Color(Color::new(c.red, c.green, c.blue, 255))
                        }
                        usvg::Paint::LinearGradient(gradient) => {
                            let color0 = gradient
                                .base
                                .stops
                                .first()
                                .map(|stop| Color {
                                    r: stop.color.red,
                                    g: stop.color.green,
                                    b: stop.color.blue,
                                    a: (stop.opacity.get() * 255.0) as u8,
                                })
                                .unwrap_or(FALLBACK_COLOR);
                            let color1 = gradient
                                .base
                                .stops
                                .last()
                                .map(|stop| Color {
                                    r: stop.color.red,
                                    g: stop.color.green,
                                    b: stop.color.blue,
                                    a: (stop.opacity.get() * 255.0) as u8,
                                })
                                .unwrap_or(FALLBACK_COLOR);
                            SvgPattern::Gradient {
                                color0,
                                color1,
                                from: point(gradient.x1 as f32, gradient.y1 as f32),
                                to: point(gradient.x2 as f32, gradient.y2 as f32),
                            }
                        }
                        _ => SvgPattern::Color(FALLBACK_COLOR),
                    },
                    line_width: stroke.width.get() as f32,
                    line_cap: match stroke.linecap {
                        usvg::LineCap::Butt => LineCap::Butt,
                        usvg::LineCap::Square => LineCap::Square,
                        usvg::LineCap::Round => LineCap::Round,
                    },
                    line_join: match stroke.linejoin {
                        usvg::LineJoin::Miter => LineJoin::Miter,
                        usvg::LineJoin::MiterClip => LineJoin::MiterClip,
                        usvg::LineJoin::Round => LineJoin::Round,
                        usvg::LineJoin::Bevel => LineJoin::Bevel,
                    },
                });

                if fill_pattern.is_none() && stroke_pattern.is_none() {
                    continue;
                }

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

                paths.push((Arc::new(path), fill_pattern, stroke_pattern));
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
