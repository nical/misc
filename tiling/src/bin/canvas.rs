use std::sync::Arc;
use std::time::Duration;
use lyon::path::Path;
//use lyon::path::Path;
//use tiling::canvas::*;
use tiling::custom_pattern::CustomPatterns;
use tiling::pattern::{
    checkerboard::{Checkerboard, CheckerboardPattern},
    simple_gradient::{Gradient, SimpleGradient},
    solid_color::SolidColor,
};
use tiling::canvas::*;
use tiling::load_svg::*;
use tiling::gpu::{ShaderSources, GpuStore};
use tiling::tiling::*;
use lyon::path::geom::euclid::size2;
use tiling::{Color, Size2D, Transform2D};
//use lyon::extra::rust_logo::build_logo_path;

use winit::dpi::PhysicalSize;
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::Window;
use futures::executor::block_on;

fn main() {
    profiling::register_thread!("Main");

    let args: Vec<String> = std::env::args().collect();

    let mut tolerance = 0.25;

    let mut trace = None;
    let mut force_gl = false;
    let mut asap = false;
    let mut read_tolerance = false;
    for arg in &args {
        if read_tolerance {
            read_tolerance = false;
            tolerance = arg.parse::<f32>().unwrap();
            println!("tolerance: {}", tolerance);
        }
        if arg == "--gl" { force_gl = true; }
        if arg == "--x11" {
            // This used to get this demo to work in renderdoc (with the gl backend) but now
            // it runs into new issues.
            std::env::set_var("WINIT_UNIX_BACKEND", "x11");
        }
        if arg == "--trace" {
            trace = Some(std::path::Path::new("./trace"));
        }
        if arg == "--asap" {
            asap = true;
        }
        if arg == "--tolerance" {
            read_tolerance = true;
        }
    }

    let tile_size = 16;
    let scale_factor = 2.0;
    let max_edges_per_gpu_tile = 16;
    let mask_atlas_size: u32 = 2048;
    let color_atlas_size: u32 = 2048;
    let inital_window_size = size2(1200u32, 1000);

    let mut tiler_config = TilerConfig {
        view_box: Box2D::from_size(inital_window_size.to_f32()),
        tile_size,
        tolerance,
        flatten: false,
        mask_atlas_size: size2(mask_atlas_size, mask_atlas_size),
        color_atlas_size: size2(color_atlas_size, color_atlas_size),
        staging_buffer_size: tiling::BYTES_PER_MASK as u32 * 2048,
    };

    let event_loop = EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .with_inner_size(winit::dpi::Size::Physical(PhysicalSize::new(inital_window_size.width, inital_window_size.height)))
        .build(&event_loop).unwrap();
    let window_size = window.inner_size();

    let backends = if force_gl {
        wgpu::Backends::GL
    } else {
        wgpu::Backends::all()
    };
    // create an instance
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends,
        .. wgpu::InstanceDescriptor::default()
    });

    // create an surface
    let surface = unsafe { instance.create_surface(&window).unwrap() };

    // create an adapter
    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: Some(&surface),
        force_fallback_adapter: false,
    })).unwrap();
    // create a device and a queue
    let (device, queue) = block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: None,
            features: wgpu::Features::default(),
            limits: wgpu::Limits::default(),
        },
        trace,
    )).unwrap();

    let (view_box, paths) = if args.len() > 1 && !args[1].starts_with('-') {
        load_svg(&args[1], scale_factor)
    } else {
        let mut builder = lyon::path::Path::builder();
        builder.begin(point(0.0, 0.0));
        builder.line_to(point(50.0, 400.0));
        builder.line_to(point(450.0, 450.0));
        builder.line_to(point(400.0, 50.0));
        builder.end(true);

        (Box2D { min: point(0.0, 0.0), max: point(500.0, 500.0) }, vec![(Arc::new(builder.build()), SvgPattern::Color(Color { r: 50, g: 200, b: 100, a: 255 }))])
    };

    tiler_config.view_box = view_box;

    let tiling_handle = ResourcesHandle::new(0);
    let mut canvas = Canvas::new();
    let mut tiling = TileRenderer::new(0, tiling_handle, &tiler_config);
    let mut dummy = DummyRenderer::new(1);

    let mut shaders = ShaderSources::new();

    let mut gpu_store = GpuStore::new(2048, &device);

    let mut tiling_resources = TilingGpuResources::new(
        &device,
        &mut shaders,
        Size2D::new(window_size.width, window_size.height),
        mask_atlas_size,
        color_atlas_size,
        &gpu_store,
    );

    let mut pattern_builder = CustomPatterns::new(
        &device,
        &mut shaders,
        &tiling_resources.target_and_gpu_store_layout,
        &tiling_resources.mask_texture_bind_group_layout,
    );

    tiling_resources.register_pattern(SolidColor::create_pipelines(&device, &mut pattern_builder));
    tiling_resources.register_pattern(SimpleGradient::create_pipelines(&device, &mut pattern_builder));
    tiling_resources.register_pattern(CheckerboardPattern::create_pipelines(&device, &mut pattern_builder));

    let mut gpu_resources = GpuResources::new(vec![Box::new(tiling_resources)]);

    tiling.tiler.draw.max_edges_per_gpu_tile = max_edges_per_gpu_tile;

    paint_scene(&paths, &mut canvas, &mut tiling, &Transform2D::translation(10.0, 1.0));

    tiling.prepare(&canvas, &mut gpu_store, &device);
    canvas.build_render_passes(&mut[&mut tiling]);


    let mut surface_desc = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: wgpu::TextureFormat::Bgra8UnormSrgb,
        width: window_size.width,
        height: window_size.height,
        present_mode: wgpu::PresentMode::AutoVsync,
        alpha_mode: wgpu::CompositeAlphaMode::Opaque,
        view_formats: vec![wgpu::TextureFormat::Bgra8UnormSrgb],
    };

    surface.configure(&device, &surface_desc);

    window.request_redraw();

    let mut scene = SceneGlobals {
        zoom: 1.0,
        target_zoom: 1.0,
        pan: [0.0, 0.0],
        target_pan: [0.0, 0.0],
        window_size,
        wireframe: false,
        size_changed: true,
        render: true,
    };

    let mut frame_build_time = Duration::ZERO;
    let mut render_time = Duration::ZERO;
    let mut row_time = Duration::ZERO;
    let mut tile_time = Duration::ZERO;
    let mut frame_idx = 0;
    event_loop.run(move |event, _, control_flow| {
        device.poll(wgpu::Maintain::Poll);

        if !update_inputs(event, &window, control_flow, &mut scene) {
            return;
        }

        if scene.size_changed {
            scene.size_changed = false;
            let physical = scene.window_size;
            surface_desc.width = physical.width;
            surface_desc.height = physical.height;
            surface.configure(&device, &surface_desc);
            tiler_config.view_box = Box2D::from_size(size2(physical.width, physical.height).to_f32());
            tiling.tiler.init(&tiler_config);
            let tiles = tiler_config.num_tiles();
            tiling.occlusion_mask.init(tiles.width, tiles.height);

            gpu_resources[tiling_handle].resize(Size2D::new(scene.window_size.width as u32, scene.window_size.height as u32), &queue)
        }

        if !scene.render {
            return;
        }
        scene.render = false;

        let frame = match surface.get_current_texture() {
            Ok(texture) => texture,
            Err(e) => {
                println!("Swap-chain error: {:?}", e);
                return;
            }
        };

        let frame_view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

        gpu_store.clear();
        tiling.begin_frame();
        canvas.begin_frame();
        gpu_resources.begin_frame();

        let tx = scene.pan[0];
        let ty = scene.pan[1];
        let ww = (scene.window_size.width as f32) * 0.5;
        let wh = (scene.window_size.height as f32) * 0.5;
        let transform = Transform2D::translation(tx, ty)
            .then_translate(-vector(ww, wh))
            .then_scale(scene.zoom, scene.zoom)
            .then_translate(vector(ww, wh));

        dummy.command(&mut canvas);
        paint_scene(&paths, &mut canvas, &mut tiling, &transform);
        dummy.command(&mut canvas);

        let frame_build_start = time::precise_time_ns();

        tiling.prepare(&canvas, &mut gpu_store, &device);
        dummy.prepare(&canvas);
        canvas.build_render_passes(&mut[&mut tiling, &mut dummy]);

        frame_build_time += Duration::from_nanos(time::precise_time_ns() - frame_build_start);

        let render_start = time::precise_time_ns();

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Canvas"),
        });

        tiling.upload(&mut gpu_resources, &device, &queue, &mut encoder);
        gpu_store.upload(&device, &queue);

        canvas.render(&[&tiling, &dummy], &gpu_resources, &frame_view, &mut encoder);

        queue.submit(Some(encoder.finish()));

        frame.present();

        render_time += Duration::from_nanos(time::precise_time_ns() - render_start);

        fn ms(duration: Duration) -> f64 {
            duration.as_micros() as f64 / 1000.0
        }

        let n = 300;
        frame_idx += 1;
        if frame_idx == n {
            let fbt = ms(frame_build_time) / (n as f64);
            let rt = ms(render_time) / (n as f64);
            let row = ms(row_time) / (n as f64);
            let tile = ms(tile_time) / (n as f64);
            frame_build_time = Duration::ZERO;
            render_time = Duration::ZERO;
            row_time = Duration::ZERO;
            tile_time = Duration::ZERO;
            frame_idx = 0;
            println!("frame build {:.2} (row: {:.2}, tile {:.2}) render {:.2}", fbt, row, tile, rt);
            print_stats(&tiling, scene.window_size);
        }

        gpu_resources.end_frame();

        if asap {
            window.request_redraw();
        }
    });
}

fn paint_scene(paths: &[(Arc<Path>, SvgPattern)], canvas: &mut Canvas, tiling: &mut TileRenderer, transform: &Transform2D<f32>) {
    let mut builder = lyon::path::Path::builder();
    builder.begin(point(0.0, 0.0));
    builder.line_to(point(50.0, 400.0));
    builder.line_to(point(450.0, 450.0));
    builder.line_to(point(400.0, 50.0));
    builder.end(true);

    tiling.fill(
        canvas,
        All, Gradient {
            from: point(100.0, 100.0), color0: Color { r: 10, g: 50, b: 250, a: 255},
            to: point(100.0, 1500.0), color1: Color { r: 50, g: 0, b: 50, a: 255},
        }
    );

    canvas.transforms.push(transform);

    tiling.fill(
        canvas,
        Circle::new(point(500.0, 500.0), 800.0),
        Gradient {
            from: point(100.0, 100.0), color0: Color { r: 200, g: 150, b: 0, a: 255},
            to: point(100.0, 1000.0), color1: Color { r: 250, g: 50, b: 10, a: 255},
        }
    );

    for (path, pattern) in paths {
        match pattern {
            &SvgPattern::Color(color) => {
                tiling.fill(canvas, path.clone(), color);
            }
            &SvgPattern::Gradient { color0, color1, from, to } => {
                tiling.fill(canvas, path.clone(), Gradient { color0, color1, from, to });
            }
        }
    }

    tiling.fill(
        canvas,
        Circle::new(point(500.0, 300.0), 200.0),
        Gradient {
            from: point(300.0, 100.0), color0: Color { r: 10, g: 200, b: 100, a: 255},
            to: point(700.0, 100.0), color1: Color { r: 200, g: 100, b: 250, a: 255},
        }
    );

    tiling.fill(
        canvas,
        Arc::new(builder.build()),
        Checkerboard { color0: Color { r: 10, g: 100, b: 250, a: 255 }, color1: Color::WHITE, scale: 25.0, offset: point(0.0, 0.0) }
    );

    canvas.transforms.pop();

    //tiling.fill(canvas, Circle::new(point(600.0, 400.0,), 100.0), Color { r: 200, g: 100, b: 120, a: 180});
    tiling.fill(canvas, Box2D { min: point(10.0, 10.0), max: point(50.0, 50.0) }, Color::BLACK);
    tiling.fill(canvas, Box2D { min: point(60.5, 10.5), max: point(100.5, 50.5) }, Color::BLACK);

    canvas.transforms.push(&Transform2D::translation(10.0, 1.0));
    canvas.transforms.pop();
}


fn print_stats(tiling: &TileRenderer, window_size: PhysicalSize<u32>) {
    let mut stats = Stats::new();
    tiling.update_stats(&mut stats);
    println!("{:#?}", stats);
    println!("Data:");
    println!("      tiles: {:2} kb", stats.tiles_bytes() as f32 / 1000.0);
    println!("      edges: {:2} kb", stats.edges_bytes() as f32 / 1000.0);
    println!("  cpu masks: {:2} kb", stats.cpu_masks_bytes() as f32 / 1000.0);
    println!("   uploaded: {:2} kb", stats.uploaded_bytes() as f32 / 1000.0);
    let win_bytes = (window_size.width * window_size.height * 4) as f32;
    println!(
        " resolution: {}x{} ({:2} kb)  overhead {:2}%",
        window_size.width, window_size.height, win_bytes / 1000.0,
        stats.uploaded_bytes() as f32 * 100.0 / win_bytes as f32,
    );
    println!("#edge distributions: {:?}", tiling.encoder.edge_distributions);
    println!("\n");

}

// Default scene has all values set to zero
#[derive(Copy, Clone, Debug)]
pub struct SceneGlobals {
    pub zoom: f32,
    pub target_zoom: f32,
    pub pan: [f32; 2],
    pub target_pan: [f32; 2],
    pub window_size: PhysicalSize<u32>,
    pub wireframe: bool,
    pub size_changed: bool,
    pub render: bool,
}

fn update_inputs(
    event: Event<()>,
    window: &Window,
    control_flow: &mut ControlFlow,
    scene: &mut SceneGlobals,
) -> bool {
    let p = scene.pan;
    let z = scene.zoom;
    match event {
        Event::RedrawRequested(_) => {
            scene.render = true;
        }
        Event::WindowEvent {
            event: WindowEvent::Destroyed,
            ..
        }
        | Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            *control_flow = ControlFlow::Exit;
            return false;
        }
        Event::WindowEvent {
            event: WindowEvent::Resized(size),
            ..
        } => {
            scene.window_size = size;
            scene.size_changed = true
        }
        Event::WindowEvent { event: WindowEvent::MouseWheel { delta , .. }, ..} => {
            use winit::event::MouseScrollDelta::*;
            let (dx, dy) = match delta {
                LineDelta(x, y) => (x * 20.0, -y * 20.0),
                PixelDelta(v) => (-v.x as f32, -v.y as f32),
            };
            let dx = dx / scene.target_zoom;
            let dy = dy / scene.target_zoom;
            if dx != 0.0 || dy != 0.0 {
                scene.target_pan[0] -= dx;
                scene.target_pan[1] -= dy;
                scene.pan[0] -= dx;
                scene.pan[1] -= dy;
            }
        }
        Event::WindowEvent {
            event:
                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            state: ElementState::Pressed,
                            virtual_keycode: Some(key),
                            ..
                        },
                    ..
                },
            ..
        } => match key {
            VirtualKeyCode::Escape => {
                *control_flow = ControlFlow::Exit;
                return false;
            }
            VirtualKeyCode::PageDown => {
                scene.target_zoom *= 0.8;
            }
            VirtualKeyCode::PageUp => {
                scene.target_zoom *= 1.25;
            }
            VirtualKeyCode::Left => {
                scene.target_pan[0] += 100.0 / scene.target_zoom;
            }
            VirtualKeyCode::Right => {
                scene.target_pan[0] -= 100.0 / scene.target_zoom;
            }
            VirtualKeyCode::Up => {
                scene.target_pan[1] += 100.0 / scene.target_zoom;
            }
            VirtualKeyCode::Down => {
                scene.target_pan[1] -= 100.0 / scene.target_zoom;
            }
            VirtualKeyCode::W => {
                scene.wireframe = !scene.wireframe;
            }
            _key => {}
        },
        _evt => {
            //println!("{:?}", _evt);
        }
    }

    scene.zoom += (scene.target_zoom - scene.zoom) * 0.15;
    scene.pan[0] += (scene.target_pan[0] - scene.pan[0]) * 0.15;
    scene.pan[1] += (scene.target_pan[1] - scene.pan[1]) * 0.15;
    if p != scene.pan || z != scene.zoom {
        window.request_redraw();
    }

    *control_flow = ControlFlow::Poll;

    true
}
