use std::{collections::HashMap, io::Write};

use svg_fmt as svg;

use crate::{criterion::ToleranceKey, BenchResults, TOLERANCES};

pub enum ColorScheme {
    Light,
    Dark,
}

impl ColorScheme {
    pub fn background(&self) -> svg::Color {
        match self {
            ColorScheme::Light => svg::white(),
            ColorScheme::Dark => svg::rgb(30, 30, 30),
        }
    }

    pub fn text(&self) -> svg::Color {
        match self {
            ColorScheme::Light => svg::black(),
            ColorScheme::Dark => svg::white(),
        }
    }

    pub fn muted_text(&self) -> svg::Color {
        match self {
            ColorScheme::Light => svg::rgb(30, 30, 30),
            ColorScheme::Dark => svg::rgb(200, 200, 200),
        }
    }

    pub fn palette(&self) -> Vec<svg::Color> {
        vec![
            svg::rgb(253, 127, 111),
            svg::rgb(126, 176, 213),
            svg::rgb(178, 224, 97),
            svg::rgb(189, 126, 190),
            svg::rgb(255, 181, 90),
            svg::rgb(255, 238, 101),
            svg::rgb(190, 185, 219),
            svg::rgb(253, 204, 229),
            svg::rgb(139, 211, 199),
        ]
    }
}

pub fn plot_graph(
    title: &str,
    subtitle: &str,
    results: &BenchResults,
    color_scheme: ColorScheme,
    colors: &[(&str, (u8, u8, u8))],
    output: &mut dyn Write,
) -> std::io::Result<()> {
    const W: f32 = 800.0;
    const H: f32 = 300.0;

    let mut color_map = HashMap::new();
    for (algo, color) in colors {
        if !results.contains_key(*algo) {
            continue;
        }
        color_map.insert(algo.to_string(), svg::rgb(color.0, color.1, color.2));
    }

    let mut fallback_colors = color_scheme.palette();

    for algo in results.keys() {
        if color_map.contains_key(algo) {
            continue;
        }

        color_map.insert(algo.clone(), fallback_colors.pop().unwrap_or(svg::black()));
    }

    let write = &mut|val: &dyn std::fmt::Display| {
        write!(output, "{}", val).unwrap();
    };

    write(&svg::BeginSvg { w: W, h: H });


    write(&svg::rectangle(0.0, 0.0, W, H)
        .border_radius(6.0)
        .style(svg::Fill::Color(color_scheme.background()).into()),
    );

    let min_x = 30.0;
    let max_x = W - 20.0;
    let min_y = 20.0;
    let max_y = H - 30.0;

    let axis_color = color_scheme.muted_text();
    write(&svg::line_segment(min_x, min_y, min_x, max_y).color(axis_color));
    write(&svg::line_segment(min_x, max_y, max_x, max_y).color(axis_color));

    let x_axis = TOLERANCES;
    //let x_axis = &[0.01, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0];
    for tol in x_axis {
        let x = x_for_tolerance(*tol, min_x..max_x);
        write(&svg::line_segment(x, max_y, x, max_y + 2.0).color(axis_color));
        write(&svg::text(x, max_y + 12.0, &format!("{tol}")).align(svg::Align::Center).size(10.0).color(axis_color));
    }
    let y_axis = &[0.0, 0.25, 0.5, 0.75, 1.0];
    for val in y_axis {
        let y = y_for_val(*val * 1000.0, min_y..max_y);
        write(&svg::line_segment(min_x - 2.0, y, min_x, y).color(axis_color));
        write(&svg::text(min_x - 4.0, y + 4.0, &format!("{val}")).align(svg::Align::Right).size(10.0).color(axis_color));
    }
    write(&svg::text(W/2.0, max_y + 24.0, "Tolerance").align(svg::Align::Center).size(10.0).color(color_scheme.muted_text()));


    for (algo, values) in results {
        let mut prev: Option<(f32, f32)> = None;
        for tol in TOLERANCES {
            let Some(val) = values.get(&ToleranceKey::new(*tol)) else {
                continue;
            };
            let color = *color_map.get(algo).unwrap();
            let x = x_for_tolerance(*tol, min_x..max_x);
            let y = y_for_val(*val, min_y..max_y);
            if let Some(prev) = prev {
                write(&svg::line_segment(prev.0, prev.1, x, y).color(color));
            }
            write(&svg::Circle {
                x,
                y,
                radius: 2.0,
                style: svg::Fill::Color(color).into(),
                comment: None,
            });
            prev = Some((x, y));
        }
    }

    let x = W - 120.0;
    let mut y = 20.0;
    let bs = 20.0;
    write(
        &svg::rectangle(x - 4.0, y - 4.0, 80.0, color_map.len() as f32 * (bs + 1.0) + 7.0 )
            .fill(svg::Fill::Color(color_scheme.background()))
            .opacity(0.8)
            .border_radius(4.0)
    );

    let mut sorted_color_map: Vec<_> = color_map.iter().collect();
    sorted_color_map.sort_by_key(|item| item.0.as_str());

    for (algo, color) in &sorted_color_map {
        write(&svg::rectangle(x, y, bs, bs)
            .fill(svg::Fill::Color(**color))
            .border_radius(2.0)
        );
        write(&svg::text(x + bs + 5.0, y + bs - 7.0, algo.as_str())
            .size(bs / 2.0)
            .color(color_scheme.text())
        );
        y += bs + 1.0;
    }

    write(
        &svg::text(W/2.0, 30.0, title)
            .size(20.0)
            .align(svg::Align::Center)
            .color(color_scheme.text())
    );

    write(
        &svg::text(W/2.0, 42.0, subtitle)
            .align(svg::Align::Center)
            .size(8.0)
            .color(color_scheme.muted_text())
    );

    write(&svg::EndSvg);

    Ok(())
}

fn x_for_tolerance(tol: f32, range: std::ops::Range<f32>) -> f32 {
    let dx = range.end - range.start;
    let min_tol = 0.01;
    let max_tol = 1.0;

    let log_min = f32::log2(min_tol);
    let log_max = f32::log2(max_tol);

    range.start + (f32::log2(tol) / (log_max - log_min)).abs() * dx

    //range.start + tol * dx
}

fn y_for_val(score: f32, range: std::ops::Range<f32>) -> f32 {
    let dy = range.end - range.start;
    range.end - score / 1000.0 * dy
}
