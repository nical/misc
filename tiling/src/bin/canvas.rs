use std::sync::Arc;
//use lyon::path::Path;
use tiling::canvas::*;
use tiling::custom_pattern::CustomPatterns;
use tiling::pattern::{
    checkerboard::{Checkerboard, CheckerboardPattern},
    simple_gradient::{Gradient, SimpleGradient},
    solid_color::SolidColor,
};
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

        (Box2D { min: point(0.0, 0.0), max: point(500.0, 500.0) }, vec![(builder.build(), SvgPattern::Color(Color { r: 50, g: 200, b: 100, a: 255 }))])
    };

    tiler_config.view_box = view_box;

    let mut canvas = Canvas::new(tiler_config.view_box);

    let mut shaders = ShaderSources::new();

    let mut gpu_store = GpuStore::new(2048, &device);

    let mut tile_renderer = TileRenderer::new(
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
        &tile_renderer.target_and_gpu_store_layout,
        &tile_renderer.mask_texture_bind_group_layout,
    );

    tile_renderer.register_pattern(SolidColor::create_pipelines(&device, &mut pattern_builder));
    tile_renderer.register_pattern(SimpleGradient::create_pipelines(&device, &mut pattern_builder));
    tile_renderer.register_pattern(CheckerboardPattern::create_pipelines(&device, &mut pattern_builder));

    let mut frame_builder = FrameBuilder::new(&tiler_config);

    //frame_builder.tiler.set_scissor(&Box2D { min: point(500.0, 600.0), max: point(1000.0, 800.0) });
    frame_builder.tiler.draw.max_edges_per_gpu_tile = max_edges_per_gpu_tile;

    let mut builder = lyon::path::Path::builder();
    builder.begin(point(0.0, 0.0));
    builder.line_to(point(50.0, 400.0));
    builder.line_to(point(450.0, 450.0));
    builder.line_to(point(400.0, 50.0));
    builder.end(true);

    canvas.fill(
        All,
        Gradient {
            from: point(100.0, 100.0), color0: Color { r: 10, g: 50, b: 250, a: 255},
            to: point(100.0, 1500.0), color1: Color { r: 50, g: 0, b: 50, a: 255},
        }
    );

    canvas.push_transform(&Transform2D::translation(10.0, 1.0));

    canvas.fill(
        Circle::new(point(500.0, 500.0), 800.0),
        Gradient {
            from: point(100.0, 100.0), color0: Color { r: 200, g: 150, b: 0, a: 255},
            to: point(100.0, 1000.0), color1: Color { r: 250, g: 50, b: 10, a: 255},
        }
    );

    for (path, pattern) in paths {
        match pattern {
            SvgPattern::Color(color) => {
                canvas.fill(Arc::new(path), color);
            }
            SvgPattern::Gradient { color0, color1, from, to } => {
                canvas.fill(Arc::new(path), Gradient { color0, color1, from, to });
            }
        }
    }

    canvas.fill(
        Circle::new(point(500.0, 300.0), 200.0),
        Gradient {
            from: point(300.0, 100.0), color0: Color { r: 10, g: 200, b: 100, a: 255},
            to: point(700.0, 100.0), color1: Color { r: 200, g: 100, b: 250, a: 255},
        }
    );

    canvas.fill(
        PathShape::new(Arc::new(builder.build())),
        Checkerboard { color0: Color { r: 10, g: 100, b: 250, a: 255 }, color1: Color::WHITE, scale: 25.0, offset: point(0.0, 0.0) }
    );

    canvas.pop_transform();

    //canvas.fill(Circle::new(point(600.0, 400.0,), 100.0), Color { r: 200, g: 100, b: 120, a: 180});
    canvas.fill(Box2D { min: point(10.0, 10.0), max: point(50.0, 50.0) }, Color::BLACK);
    canvas.fill(Box2D { min: point(60.5, 10.5), max: point(100.5, 50.5) }, Color::BLACK);

    canvas.push_transform(&Transform2D::translation(10.0, 1.0));
    canvas.pop_transform();

    let mut commands = canvas.finish();

    frame_builder.build(&commands, &mut gpu_store, &device);

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

    let mut frame_build_time = Duration::zero();
    let mut render_time = Duration::zero();
    let mut row_time = Duration::zero();
    let mut tile_time = Duration::zero();
    let mut frame_idx = 0;
    event_loop.run(move |event, _, control_flow| {
        device.poll(wgpu::Maintain::Poll);

        if !update_inputs(event, &window, control_flow, &mut scene) {
            return;
        }

        let tx = scene.pan[0];
        let ty = scene.pan[1];
        let ww = (scene.window_size.width as f32) * 0.5;
        let wh = (scene.window_size.height as f32) * 0.5;
        let transform = Transform2D::translation(tx, ty)
            .then_translate(-vector(ww, wh))
            .then_scale(scene.zoom, scene.zoom)
            .then_translate(vector(ww, wh));
        commands.set_transform(1, &transform);

        if scene.size_changed {
            scene.size_changed = false;
            let physical = scene.window_size;
            surface_desc.width = physical.width;
            surface_desc.height = physical.height;
            surface.configure(&device, &surface_desc);
            tiler_config.view_box = Box2D::from_size(size2(physical.width, physical.height).to_f32());
            frame_builder.tiler.init(&tiler_config);
            let tiles = tiler_config.num_tiles();
            frame_builder.tile_mask.init(tiles.width, tiles.height);

            tile_renderer.resize(Size2D::new(scene.window_size.width as u32, scene.window_size.height as u32), &queue)
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
        gpu_store.push(&[0.0]);
        gpu_store.push(&[0.0]);

        let frame_build_start = time::precise_time_ns();
        frame_builder.build(&commands, &mut gpu_store, &device);
        frame_build_time += Duration::from_ns(time::precise_time_ns() - frame_build_start);
        row_time += frame_builder.stats().row_time;
        tile_time += frame_builder.stats().tile_time;

        let render_start = time::precise_time_ns();

        tile_renderer.begin_frame();

        tile_renderer.edges.bump_allocator().push(frame_builder.tiler.edges.len());

        tile_renderer.allocate(&device);

        tile_renderer.edges.upload_bytes(0, bytemuck::cast_slice(&frame_builder.tiler.edges), &queue);
        for target in &mut frame_builder.targets {
            target.tile_encoder.upload(&mut tile_renderer, &device);
        }
        gpu_store.upload(&device, &queue);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tile"),
        });

        let target = &mut frame_builder.targets[0];
        tile_renderer.render(
            &mut target.tile_encoder,
            &device,
            &frame_view,
            &mut encoder,
        );

        queue.submit(Some(encoder.finish()));

        render_time += Duration::from_ns(time::precise_time_ns() - render_start);

        let n = 300;
        frame_idx += 1;
        if frame_idx == n {
            let fbt = frame_build_time.ms() / (n as f64);
            let rt = render_time.ms() / (n as f64);
            let row = row_time.ms() / (n as f64);
            let tile = tile_time.ms() / (n as f64);
            frame_build_time = Duration::zero();
            render_time = Duration::zero();
            row_time = Duration::zero();
            tile_time = Duration::zero();
            frame_idx = 0;
            println!("frame build {:.2} (row: {:.2}, tile {:.2}) render {:.2}", fbt, row, tile, rt);
            print_stats(&frame_builder, scene.window_size);
        }

        frame.present();

        if asap {
            window.request_redraw();
        }
    });
}

fn print_stats(frame_builder: &FrameBuilder, window_size: PhysicalSize<u32>) {
    let mut stats = Stats::new();
    for target in &frame_builder.targets {
        target.tile_encoder.update_stats(&mut stats);
    }
    frame_builder.tiler.update_stats(&mut stats);
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
    println!("#edge distributions: {:?}", frame_builder.targets[0].tile_encoder.edge_distributions);
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
