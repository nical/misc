//! Headless benchmark for the CPU side of the tiling renderer.
//!
//! This exercises the same per-frame work as `TileRenderer::prepare_single_thread`
//! (flattening, binning, sorting, tile generation, occlusion culling and staging
//! buffer writes) without requiring a GPU device: the staging buffer pool runs
//! in CPU-only mode.
//!
//! Example:
//! ```text
//! cargo run --release --bin tiling_bench -- demo/tiger.svg --frames 200 --scale 2
//! ```

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use core::gpu::{
    GpuBufferDescriptor, GpuBufferResources, GpuStreamsDescritptor, GpuStreamsResources,
    StagingBufferPool,
};
use core::pattern::BuiltPattern;
use core::shading::ShaderPatternId;
use core::units::SurfaceIntRect;
use core::wgpu;
use lyon::geom::euclid;
use lyon::path::Path;
use tiling::occlusion::OcclusionBuffer;
use tiling::tiler::{EncodedTileInstance, Tiler, TilerOutput};
use tiling::{FillOptions, FillRule, Transform};

struct SceneItem {
    path: Path,
    fill_rule: FillRule,
    pattern: BuiltPattern,
}

struct Options {
    svgs: Vec<String>,
    frames: u32,
    warmup: u32,
    tolerance: f32,
    scale: f32,
    occlusion: bool,
    top_paths: usize,
}

fn main() {
    let options = parse_args();
    if options.svgs.is_empty() {
        eprintln!("No input SVG. Usage: tiling_bench <file.svg>... [--frames N] [--warmup N] [--tolerance T] [--scale S] [--no-occlusion] [--top-paths N]");
        std::process::exit(1);
    }

    for svg in &options.svgs {
        run_svg(svg, &options);
    }
}

fn parse_args() -> Options {
    let mut options = Options {
        svgs: Vec::new(),
        frames: 100,
        warmup: 10,
        tolerance: 0.25,
        scale: 1.0,
        occlusion: true,
        top_paths: 0,
    };

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let num = |args: &mut dyn Iterator<Item = String>| -> f64 {
            args.next()
                .and_then(|v| v.parse().ok())
                .unwrap_or_else(|| {
                    eprintln!("Expected a number after {arg:?}");
                    std::process::exit(1);
                })
        };
        match arg.as_str() {
            "--frames" => options.frames = num(&mut args) as u32,
            "--warmup" => options.warmup = num(&mut args) as u32,
            "--tolerance" => options.tolerance = num(&mut args) as f32,
            "--scale" => options.scale = num(&mut args) as f32,
            "--no-occlusion" => options.occlusion = false,
            "--top-paths" => options.top_paths = num(&mut args) as usize,
            _ if arg.starts_with("--") => {
                eprintln!("Unknown option {arg:?}");
                std::process::exit(1);
            }
            _ => options.svgs.push(arg),
        }
    }

    options
}

fn run_svg(filename: &str, options: &Options) {
    let (view_box, scene) = load_svg(filename);

    let transform = Transform::scale(options.scale, options.scale);
    let width = (view_box.max.x * options.scale).ceil() as i32;
    let height = (view_box.max.y * options.scale).ceil() as i32;
    let viewport = SurfaceIntRect {
        min: euclid::point2(0, 0),
        max: euclid::point2(width.max(1), height.max(1)),
    };

    println!("{filename}: {} paths, viewport {width}x{height}, tolerance {}, occlusion {}", scene.len(), options.tolerance, options.occlusion);

    let paths_store = GpuBufferResources::new(&GpuBufferDescriptor::Texture {
        format: wgpu::TextureFormat::Rgba32Uint,
        width: 2048,
        label: Some("tiling::paths"),
        alignment: 32,
    });
    let edges_store = GpuBufferResources::new(&GpuBufferDescriptor::rgba8_unorm_texture("tiling::edges"));
    let instances_store = GpuStreamsResources::new(&GpuStreamsDescritptor {
        usages: wgpu::BufferUsages::VERTEX,
        buffer_size: 1024 * 96,
        chunk_size: 1024 * 6,
        label: Some("instances"),
    });

    let mut tiler = Tiler::new();
    let mut frame_times = Vec::with_capacity(options.frames as usize);
    let mut stats = FrameStats::default();

    for frame in 0..(options.warmup + options.frames) {
        let start = Instant::now();

        // A fresh pool every frame keeps CPU-side staging chunks from
        // accumulating (they are never recycled in CPU-only mode).
        let staging = Arc::new(Mutex::new(StagingBufferPool::new_cpu_only(1024 * 128)));
        let paths_buffer = paths_store.begin_frame(staging.clone());
        let edges_buffer = edges_store.begin_frame(staging.clone());
        let instances = instances_store.begin_frame(staging.clone());

        let opaque_stream = instances.next_stream_id();
        let mut output = TilerOutput {
            opaque_tiles: instances.write(opaque_stream, 0),
            mask_tiles: Vec::new(),
            paths: paths_buffer.write(),
            edges: edges_buffer.write(),
            occlusion: OcclusionBuffer::disabled(),
        };

        tiler.begin_target(viewport);
        if options.occlusion {
            // The single-threaded prepare path initializes the occlusion
            // buffer with the surface size in pixels.
            output.occlusion.init(viewport.max.x as u32, viewport.max.y as u32);
        }

        // Paint back-to-front like prepare_single_thread does when cpu
        // occlusion culling is enabled.
        for (i, item) in scene.iter().enumerate().rev() {
            let fill_options = FillOptions::new()
                .with_transform(Some(&transform))
                .with_fill_rule(item.fill_rule)
                .with_tolerance(options.tolerance)
                .with_z_index(i as u32 + 1);
            tiler.fill_path(item.path.iter(), &fill_options, &item.pattern, &mut output);
        }

        // Like the renderer, copy the accumulated mask tiles into a stream.
        if !output.mask_tiles.is_empty() {
            let masked_stream = instances.next_stream_id();
            let mut writer = instances.write(masked_stream, 0);
            writer.push_slice(&output.mask_tiles);
        }

        stats = FrameStats {
            opaque_tiles: output.opaque_tiles.pushed_items::<EncodedTileInstance>(),
            mask_tiles: output.mask_tiles.len() as u32,
            edge_bytes: output.edges.pushed_bytes(),
            path_bytes: output.paths.pushed_bytes(),
        };
        drop(output);

        if frame >= options.warmup {
            frame_times.push(start.elapsed());
        }
    }

    report(&frame_times, &stats);

    if options.top_paths > 0 {
        report_top_paths(&scene, &transform, viewport, options, &paths_store, &edges_store, &instances_store, &mut tiler);
    }
}

#[derive(Default)]
struct FrameStats {
    opaque_tiles: u32,
    mask_tiles: u32,
    edge_bytes: u32,
    path_bytes: u32,
}

fn report(frame_times: &[Duration], stats: &FrameStats) {
    let mut sorted: Vec<Duration> = frame_times.to_vec();
    sorted.sort();
    let ms = |d: Duration| d.as_secs_f64() * 1000.0;
    let total: Duration = sorted.iter().sum();
    let avg = total / sorted.len() as u32;
    let median = sorted[sorted.len() / 2];
    let p95 = sorted[p95_idx(sorted.len())];

    println!("  tiles: {} opaque, {} masked; edges: {} bytes; paths: {} bytes",
        stats.opaque_tiles, stats.mask_tiles, stats.edge_bytes, stats.path_bytes);
    println!(
        "  frame cpu time (ms): min {:.3}  avg {:.3}  median {:.3}  p95 {:.3}  max {:.3}",
        ms(sorted[0]), ms(avg), ms(median), ms(p95), ms(*sorted.last().unwrap()),
    );
}

fn p95_idx(len: usize) -> usize {
    ((len as f64 * 0.95) as usize).min(len - 1)
}

// Times each path individually over a frame and prints the most expensive ones.
fn report_top_paths(
    scene: &[SceneItem],
    transform: &Transform,
    viewport: SurfaceIntRect,
    options: &Options,
    paths_store: &GpuBufferResources,
    edges_store: &GpuBufferResources,
    instances_store: &GpuStreamsResources,
    tiler: &mut Tiler,
) {
    let staging = Arc::new(Mutex::new(StagingBufferPool::new_cpu_only(1024 * 128)));
    let paths_buffer = paths_store.begin_frame(staging.clone());
    let edges_buffer = edges_store.begin_frame(staging.clone());
    let instances = instances_store.begin_frame(staging.clone());

    let opaque_stream = instances.next_stream_id();
    let mut output = TilerOutput {
        opaque_tiles: instances.write(opaque_stream, 0),
        mask_tiles: Vec::new(),
        paths: paths_buffer.write(),
        edges: edges_buffer.write(),
        occlusion: OcclusionBuffer::disabled(),
    };

    tiler.begin_target(viewport);
    if options.occlusion {
        output.occlusion.init(viewport.max.x as u32, viewport.max.y as u32);
    }

    let mut timings: Vec<(usize, Duration)> = Vec::with_capacity(scene.len());
    for (i, item) in scene.iter().enumerate().rev() {
        let fill_options = FillOptions::new()
            .with_transform(Some(transform))
            .with_fill_rule(item.fill_rule)
            .with_tolerance(options.tolerance)
            .with_z_index(i as u32 + 1);
        let start = Instant::now();
        tiler.fill_path(item.path.iter(), &fill_options, &item.pattern, &mut output);
        timings.push((i, start.elapsed()));
    }

    timings.sort_by(|a, b| b.1.cmp(&a.1));
    println!("  most expensive paths:");
    for (i, time) in timings.iter().take(options.top_paths) {
        println!("    path #{i}: {:.3} ms", time.as_secs_f64() * 1000.0);
    }
}

fn load_svg(filename: &str) -> (lyon::geom::Box2D<f32>, Vec<SceneItem>) {
    use usvg::{NodeExt, TreeParsing};

    let opt = usvg::Options::default();
    let svg_src = std::fs::read_to_string(filename).unwrap();
    let rtree = usvg::Tree::from_str(&svg_src, &opt).unwrap();

    let mut scene = Vec::new();
    for node in rtree.root.descendants() {
        let t = node.abs_transform();
        let transform = euclid::default::Transform2D::new(
            t.sx, t.kx, t.ky, t.sy, t.tx, t.ty,
        );

        if let usvg::NodeKind::Path(ref usvg_path) = *node.borrow() {
            let Some(fill) = usvg_path.fill.as_ref() else {
                continue;
            };

            let fill_rule = match fill.rule {
                usvg::FillRule::NonZero => FillRule::NonZero,
                usvg::FillRule::EvenOdd => FillRule::EvenOdd,
            };

            let is_opaque = fill.opacity.get() >= 1.0
                && matches!(fill.paint, usvg::Paint::Color(_));

            let mut builder = Path::builder().with_svg();
            for segment in usvg_path.data.segments() {
                use usvg::tiny_skia_path::PathSegment;
                let p = |pt: usvg::tiny_skia_path::Point| {
                    transform.transform_point(euclid::point2(pt.x, pt.y))
                };
                match segment {
                    PathSegment::MoveTo(to) => {
                        builder.move_to(p(to));
                    }
                    PathSegment::LineTo(to) => {
                        builder.line_to(p(to));
                    }
                    PathSegment::QuadTo(ctrl, to) => {
                        builder.quadratic_bezier_to(p(ctrl), p(to));
                    }
                    PathSegment::CubicTo(ctrl1, ctrl2, to) => {
                        builder.cubic_bezier_to(p(ctrl1), p(ctrl2), p(to));
                    }
                    PathSegment::Close => {
                        builder.close();
                    }
                }
            }

            scene.push(SceneItem {
                path: builder.build(),
                fill_rule,
                pattern: BuiltPattern::new(ShaderPatternId::from_index(0), 0)
                    .with_opacity(is_opaque),
            });
        }
    }

    let vb = rtree.view_box.rect;
    let view_box = lyon::geom::Box2D {
        min: euclid::point2(vb.x(), vb.y()),
        max: euclid::point2(vb.x() + vb.width(), vb.y() + vb.height()),
    };

    (view_box, scene)
}
