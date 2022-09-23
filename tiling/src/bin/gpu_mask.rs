use tiling::{Color, point, Box2D};
use tiling::tiling::*;
use tiling::gpu::GpuStore;
use tiling::load_svg::*;
use lyon::path::geom::euclid::{size2, Transform2D};

use winit::dpi::PhysicalSize;
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::Window;
use futures::executor::block_on;

use tiling::tiling::tiler::{Tiler, TileEncoder, TilerConfig};
use tiling::tiling::tile_renderer::TileRenderer;
use tiling::gpu::{ShaderSources};
use tiling::pattern::solid_color::*;
use tiling::custom_pattern::*;

fn main() {
    profiling::register_thread!("Main");

    let args: Vec<String> = std::env::args().collect();

    let mut select_path = None;
    let mut select_path_range = None;
    let mut select_row = None;
    let mut profile = false;
    let mut use_quads = false;
    let mut sp = false;
    let mut spr = false;
    let mut sr = false;
    for arg in &args {
        if spr {
            select_path_range = Some(arg.parse::<u16>().unwrap());
            sp = false;
        }
        if sp {
            select_path = Some(arg.parse::<u16>().unwrap());
            sp = false;
        }
        if sr {
            select_row = Some(arg.parse::<usize>().unwrap());
            sr = false;
        }
        if arg == "--path-range" { spr = true; }
        if arg == "--path" { sp = true; }
        if arg == "--row" { sr = true; }
        if arg == "--profile" { profile = true; }
        if arg == "--quads" { use_quads = true; }
    }

    let tile_size = 16;
    let tolerance = 0.1;
    let scale_factor = 2.0;
    let max_edges_per_gpu_tile = 64;
    let n = if profile { 1000 } else { 1 };
    let tile_atlas_size: u32 = 2048;

    let mut tiler_config = TilerConfig {
        view_box: Box2D::from_size(size2(1200.0, 1000.0)),
        tile_size,
        tile_padding: 0.5,
        tolerance,
        flatten: false,
        mask_atlas_size: size2(tile_atlas_size, tile_atlas_size),
        color_atlas_size: size2(tile_atlas_size, tile_atlas_size),
    };

    let event_loop = EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .with_inner_size(winit::dpi::Size::Physical(PhysicalSize::new(1200, 1000)))
        .build(&event_loop).unwrap();
    let window_size = window.inner_size();

    // create an instance
    let instance = wgpu::Instance::new(wgpu::Backends::all());

    // create an surface
    let surface = unsafe { instance.create_surface(&window) };

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
        None,
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

        (Box2D { min: point(0.0, 0.0), max: point(500.0, 500.0) }, vec![(builder.build(), Color { r: 50, g: 200, b: 100, a: 255 })])
    };

    let mut tiler = Tiler::new(&tiler_config);
    tiler.selected_row = select_row;

    let size = tiler_config.view_box.size().to_u32();
    let ts = tiler_config.tile_size;
    let tiles_x = (size.width + ts - 1) / ts;
    let tiles_y = (size.height + ts - 1) / ts;
    let mut tile_mask = TileMask::new(tiles_x, tiles_y);

    use tiling::gpu::mask_uploader::MaskUploader;
    let mut shaders = ShaderSources::new();

    let mut gpu_store = GpuStore::new(1024, 1024, &device);
    let mut tile_renderer = TileRenderer::new(
        &device,
        &mut shaders,
        size,
        tile_atlas_size as u32,
        tile_atlas_size as u32,
        &mut gpu_store,
    );

    let mask_upload_copies = tiling::gpu::mask_uploader::MaskUploadCopies::new(&device, &mut shaders, &tile_renderer.target_and_gpu_store_layout);
    let mask_uploader = MaskUploader::new(&device, &mask_upload_copies.bind_group_layout, tile_atlas_size);

    // Main builder.
    let mut builder = TileEncoder::new(&tiler_config, mask_uploader, 1);

    tiler.draw.max_edges_per_gpu_tile = max_edges_per_gpu_tile;
    tiler.draw.use_quads = use_quads;

    let mut custom_patterns = CustomPatterns::new(
        &device,
        &mut shaders,
        &tile_renderer.target_and_gpu_store_layout,
        &tile_renderer.mask_texture_bind_group_layout,
    );

    let mut color_pattern = SolidColorBuilder::new(SolidColor::new(Color::BLACK), 0);
    let color_pipelines = SolidColor::create_pipelines(&device, &mut custom_patterns);
    tile_renderer.register_pattern(color_pipelines);

    let mut row_time: u64 = 0;
    let mut tile_time: u64 = 0;

    let t0 = time::precise_time_ns();
    for _run in 0..n {
        let transform = Transform2D::translation(1.0, 1.0);

        builder.reset();
        tile_mask.clear();
        let mut z_index = paths.len() as u16;
        // Loop over the paths in front-to-back order to take advantage of
        // occlusion culling.
        for (path, color) in paths.iter().rev() {
            if let Some(idx) = select_path_range {
                if idx < z_index {
                    z_index -= 1;
                    continue;
                }
            }
            if let Some(idx) = select_path {
                if idx != z_index {
                    z_index -= 1;
                    continue;
                }
            }

            color_pattern.set(SolidColor::new(*color));

            let options = FillOptions::new()
                .with_transform(Some(&transform))
                .with_tolerance(tiler_config.tolerance);

            tiler.fill_path(
                path.iter(),
                &options,
                &mut color_pattern,
                &mut tile_mask,
                None,
                &mut builder,
            );

            z_index -= 1;

            row_time += tiler.row_decomposition_time_ns;
            tile_time += tiler.tile_decomposition_time_ns;
        }

        builder.end_paths();

        // Since the paths were processed front-to-back we have to reverse
        // the alpha tiles to render then back-to-front.
        // This doesn't show up in profiles.
        builder.reverse_alpha_tiles();
    }

    let t1 = time::precise_time_ns();

    let t = (t1 - t0) / n;

    let mut stats = Stats::new();
    builder.update_stats(&mut stats);

    println!("view box: {:?}", view_box);
    println!("{:#?}", stats);
    println!("#edge distributions: {:?}", builder.edge_distributions);
    println!("");
    println!("-> {}ns", t);
    println!("-> {:.3}ms", t as f64 / 1000000.0);
    println!("-> row decomposition: {:.3}ms", (row_time / n) as f64 / 1000000.0);
    println!("-> tile decomposition: {:.3}ms", (tile_time / n) as f64 / 1000000.0);

    if profile {
        // Ensures the MaskUploader's buffers are properly cleaned up since we won't
        // consume them.
        builder.reset();
        return;
    }

    let mut surface_desc = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: wgpu::TextureFormat::Bgra8UnormSrgb,
        width: window_size.width,
        height: window_size.height,
        present_mode: wgpu::PresentMode::AutoVsync,
    };

    surface.configure(&device, &surface_desc);

    window.request_redraw();

    let mut scene = SceneGlobals {
        zoom: 1.0,
        pan: [0.0, 0.0],
        window_size,
        wireframe: false,
        size_changed: true,
        render: true,
    };

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
            tiler.init(&tiler_config);
            let tiles = tiler_config.num_tiles();
            tile_mask.init(tiles.width, tiles.height);
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

        tile_renderer.begin_frame();

        builder.allocate_buffer_ranges(&mut tile_renderer);
        tile_renderer.allocate(&device);
        builder.upload(&mut tile_renderer, &queue);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tile"),
        });

        tile_renderer.render(&mut builder, &device, &frame_view, &mut encoder);

        queue.submit(Some(encoder.finish()));
        frame.present();
    });
}

// Default scene has all values set to zero
#[derive(Copy, Clone, Debug)]
pub struct SceneGlobals {
    pub zoom: f32,
    pub pan: [f32; 2],
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
    match event {
        Event::RedrawRequested(_) => {
            scene.render = true;
        }
        Event::RedrawEventsCleared => { 
            window.request_redraw();
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
                scene.zoom *= 0.8;
            }
            VirtualKeyCode::PageUp => {
                scene.zoom *= 1.25;
            }
            VirtualKeyCode::Left => {
                scene.pan[0] -= 50.0 / scene.pan[0];
            }
            VirtualKeyCode::Right => {
                scene.pan[0] += 50.0 / scene.pan[0];
            }
            VirtualKeyCode::Up => {
                scene.pan[1] += 50.0 / scene.pan[1];
            }
            VirtualKeyCode::Down => {
                scene.pan[1] -= 50.0 / scene.pan[1];
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

    *control_flow = ControlFlow::Poll;

    true
}
