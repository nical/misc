use std::io::{self, Write};

use crate::{Vector, CubicBezierSegment, Point};
use svg_fmt::{Circle, Color, Fill, Stroke, PathOp, Style};

use crate::Flatten;



pub fn show_cubic<F: Flatten>(curve: &CubicBezierSegment, tolerance: f32, offset: (f32, f32), output: &mut dyn Write) -> io::Result<u32> {
    fn back_point(p: Point, radius: f32, output: &mut dyn Write) -> io::Result<()> {
        let style: Style = Fill::Color(Color { r: 200, g: 200, b: 200 }).into();
        writeln!(output, "{}", Circle { x: p.x, y: p.y, radius: radius * 2.0, style, comment: None })
    }

    fn point(p: Point, radius: f32, output: &mut dyn Write) -> io::Result<()> {
        let style: Style = Fill::Color(Color { r: 0, g: 0, b: 0 }).into();
        writeln!(output, "{}", Circle { x: p.x, y: p.y, radius, style, comment: None })?;
        let style: Style = Fill::Color(Color { r: 255, g: 255, b: 255 }).into();
        writeln!(output, "{}", Circle { x: p.x, y: p.y, radius: radius * 0.8, style, comment: None })
    }

    let v = Vector::new(offset.0, offset.1);
    let curve = &CubicBezierSegment {
        from: curve.from + v,
        ctrl1: curve.ctrl1 + v,
        ctrl2: curve.ctrl2 + v,
        to: curve.to + v,
    };

    let pt_size = tolerance * 0.8;

    let mut count = 0;
    let mut poly_line = Vec::new();
    poly_line.push(PathOp::MoveTo { x: curve.from.x, y: curve.from.y });
    F::cubic(curve, tolerance, &mut|segment| {
        let _ = back_point(segment.to, tolerance, output);
        poly_line.push(PathOp::LineTo { x: segment.to.x, y: segment.to.y });
        count += 1;
    });
    back_point(curve.from, tolerance, output)?;

    writeln!(output, "{}", svg_fmt::Path {
        ops: vec![
            PathOp::MoveTo { x: curve.from.x, y: curve.from.y },
            PathOp::CubicTo { ctrl1_x: curve.ctrl1.x, ctrl1_y: curve.ctrl1.y, ctrl2_x: curve.ctrl2.x, ctrl2_y: curve.ctrl2.y, x: curve.to.x, y: curve.to.y }
        ],
        style: Stroke::Color(Color { r: 180, g: 200, b: 250 }, tolerance*2.0).into(),
        comment: None,
    })?;

    writeln!(output, "{}", svg_fmt::Path {
        ops: poly_line,
        style: Stroke::Color(Color { r: 0, g: 0, b: 0 }, tolerance*0.2).into(),
        comment: None,
    })?;

    point(curve.from, pt_size, output)?;

    F::cubic(curve, tolerance, &mut|segment| {
        let _ = point(segment.to, pt_size, output);
        count += 1;
    });

    Ok(count)
}

fn print_card(output: &mut dyn Write, rect: &(i32, i32, i32, i32), title: &str) {
    let _ = write!(output, "{}", svg_fmt::rectangle(rect.0 as f32, rect.1 as f32, rect.2 as f32, rect.3 as f32)
        .border_radius(2.0)
        .fill(Fill::Color(Color { r: 255, g: 255, b: 255}))
    );

    if title != "" {
        let x = (rect.0 + rect.2 / 2) as f32;
        let y = (rect.1 + rect.3 - 2) as f32;

        let _ = write!(output, "{}", svg_fmt::text(x, y, title).align(svg_fmt::Align::Center));
    }
}

fn compare_cubic(output: &mut dyn Write, curve: &CubicBezierSegment, tolerance: f32, offset: (i32, i32)) {
    let mut x = offset.0;
    let y = offset.1;

    let _ = write!(output, "{}", svg_fmt::text(
        x as f32 + 2.0, y as f32 - 2.0,
        &format!("M {} {} C {} {}  {} {}  {} {}", curve.from.x, curve.from.y, curve.ctrl1.x, curve.ctrl1.y, curve.ctrl2.x, curve.ctrl2.y, curve.to.x, curve.to.y)
    ));

    print_card(output, &(x, y, 110, 110), "Levien-simd");
    let _ = show_cubic::<crate::LevienSimd>(
        curve,
        tolerance,
        (x as f32 + 5.0, y as f32 + 5.0),
        output
    );

    if true {
        x += 115;
        print_card(output, &(x, y, 110, 110), "Yzerman");
        let _ = show_cubic::<crate::YzermanSimd4>(
            curve,
            tolerance,
            (x as f32 + 5.0, y as f32 + 5.0),
            output
        );
    }

    x += 115;
    print_card(output, &(x, y, 110, 110), "Levien-quads");
    let _ = show_cubic::<crate::LevienQuads>(
        curve,
        tolerance,
        (x as f32 + 5.0, y as f32 + 5.0),
        output
    );

    x += 115;
    print_card(output, &(x, y, 110, 110), "Yzerman");
    let _ = show_cubic::<crate::Yzerman>(
        curve,
        tolerance,
        (x as f32 + 5.0, y as f32 + 5.0),
        output
    );

    x += 115;
    print_card(output, &(x, y, 110, 110), "Wang");
    let _ = show_cubic::<crate::Wang>(
        curve,
        tolerance,
        (x as f32 + 5.0, y as f32 + 5.0),
        output
    );

    x += 115;
    print_card(output, &(x, y, 110, 110), "Hain");
    let _ = show_cubic::<crate::Hain>(
        curve,
        tolerance,
        (x as f32 + 5.0, y as f32 + 5.0),
        output
    );

    x += 115;
    print_card(output, &(x, y, 110, 110), "Linear");
    let _ = show_cubic::<crate::Linear>(
        curve,
        tolerance,
        (x as f32 + 5.0, y as f32 + 5.0),
        output
    );

    x += 115;
    print_card(output, &(x, y, 110, 110), "Recursive");
    let _ = show_cubic::<crate::Recursive>(
        curve,
        tolerance,
        (x as f32 + 5.0, y as f32 + 5.0),
        output
    );
}

#[test]
fn print_cubics() {
    let mut stdout = std::io::stdout().lock();
    let mut output: &mut dyn Write = &mut stdout;
    let mut _file = None;
    for (key, val) in std::env::vars() {
        if key == "FLATTEN_OUTPUT" {
            _file = Some(std::fs::File::create(val.as_str()).unwrap());
            output = _file.as_mut().unwrap();
        }
    }

    let _ = write!(output, "{}", svg_fmt::BeginSvg { w: 700.0, h: 800.0 });

    let tol = 0.5;
    let mut y = 15;
    let row_h = 125;

    let _ = write!(output, "{}", svg_fmt::rectangle(0.0, 0.0, 700.0, 800.0)
        .border_radius(4.0)
        .fill(Fill::Color(Color { r: 230, g: 230, b: 230 }))
    );

    compare_cubic(output,
        &CubicBezierSegment {
            from: Point::new(0.0, 0.0),
            ctrl1: Point::new(70.0, 0.0),
            ctrl2: Point::new(100.0, 30.0),
            to: Point::new(100.0, 100.0),
        },
        tol,
        (5, y)
    );

    y += row_h;
    compare_cubic(output,
        &CubicBezierSegment {
            from: Point::new(0.0, 0.0),
            ctrl1: Point::new(160.0, 0.0),
            ctrl2: Point::new(0.0, 100.0),
            to: Point::new(100.0, 100.0),
        },
        tol,
        (5, y)
    );

    y += row_h;
    compare_cubic(output,
        &CubicBezierSegment {
            from: Point::new(40.0, 40.0),
            ctrl1: Point::new(-100.0, 40.0),
            ctrl2: Point::new(250.0, 55.0),
            to: Point::new(20.0, 55.0),
        },
        tol,
        (5, y)
    );

    y += row_h;
    compare_cubic(output,
        &CubicBezierSegment {
            from: Point::new(0.0, 10.0),
            ctrl1: Point::new(-10.0, 10.0),
            ctrl2: Point::new(180.0, 10.0),
            to: Point::new(60.0, 10.0),
        },
        tol,
        (5, y)
    );

    y += row_h;
    compare_cubic(output,
        &CubicBezierSegment {
            from: Point::new(0.0, 100.0),
            ctrl1: Point::new(100.0, 0.0),
            ctrl2: Point::new(0.0, 0.0),
            to: Point::new(100.0, 100.0),
        },
        tol,
        (5, y)
    );

    let _ = write!(output, "{}", svg_fmt::EndSvg);
}
