use tiling::*;
use tiling::load_svg::*;
use lyon::path::geom::euclid::{size2, Transform2D};
use lyon::path::math::vector;

use winit::dpi::PhysicalSize;
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::Window;
use wgpu::util::DeviceExt;
use futures::executor::block_on;

use parasol::CachePadded;

use tiling::tile_encoder::{TileEncoder, MaskPass};

fn main() {
    profiling::register_thread!("Main");

    let args: Vec<String> = std::env::args().collect();

    // This is an incomplete prototype so there are some configurations that don't work:
    //  - in parallel mode or with quadratic curves, make sure the tile atlas size is large enough to

    let mut select_path = None;
    let mut select_row = None;
    let mut profile = false;
    let mut use_quads = false;
    let mut sp = false;
    let mut sr = false;
    let mut parallel = false;
    for arg in &args {
        if sp {
            select_path = Some(arg.parse::<u16>().unwrap());
            sp = false;
        }
        if sr {
            select_row = Some(arg.parse::<usize>().unwrap());
            sr = false;
        }
        if arg == "--path" { sp = true; }
        if arg == "--row" { sr = true; }
        if arg == "--parallel" { parallel = true; }
        if arg == "--profile" { profile = true; }
        if arg == "--quads" { use_quads = true; }
    }

    let tile_size = 16.0;
    let tolerance = 0.1;
    let scale_factor = 2.0;
    let max_edges_per_gpu_tile = 64;
    let n = if profile { 1000 } else { 1 };
    let tile_atlas_size: u32 = 2024;

    let mut tiler_config = TilerConfig {
        view_box: Box2D::zero(),
        tile_size: size2(tile_size, tile_size),
        tile_padding: 0.5,
        tolerance,
        flatten: false,
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

    let globals_ubo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Globals"),
        contents: bytemuck::cast_slice(&[
            tiling::gpu::GpuGlobals {
                resolution: vector(window_size.width as f32, window_size.height as f32),
                tile_size: tile_size as u32,
                tile_atlas_size,
            }
        ]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let globals_bind_group_layout = tiling::gpu::GpuGlobals::create_bind_group_layout(&device);
    let globals_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Globals"),
        layout: &globals_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(globals_ubo.as_entire_buffer_binding())
            },
        ],
    });

    let mask_upload_copies = tiling::gpu::masked_tiles::MaskUploadCopies::new(&device, &globals_bind_group_layout);

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

    tiler_config.view_box = view_box;

    let thread_pool = parasol::ThreadPool::builder()
        .with_worker_threads(3)
        .with_contexts(10)
        .build();

    let mut ctx = thread_pool.pop_context().unwrap();

    let mut tiler = Tiler::new(&tiler_config);
    tiler.selected_row = select_row;

    use tiling::gpu::masked_tiles::MaskUploader;
    let mask_uploader = MaskUploader::new(&device, &mask_upload_copies.bind_group_layout, tile_atlas_size);
    let mask_uploader_0 = MaskUploader::new(&device, &mask_upload_copies.bind_group_layout, tile_atlas_size);
    let mask_uploader_1 = MaskUploader::new(&device, &mask_upload_copies.bind_group_layout, tile_atlas_size);
    let mask_uploader_2 = MaskUploader::new(&device, &mask_upload_copies.bind_group_layout, tile_atlas_size);

    // Main builder.
    let mut builder = CachePadded::new(TileEncoder::new(&tiler_config, mask_uploader));
    builder.set_tile_texture_size(tile_atlas_size, tile_size as u32);
    // Extra builders for worker threads.
    let mut b0 = CachePadded::new(TileEncoder::new_parallel(&builder, &tiler_config, mask_uploader_0));
    let mut b1 = CachePadded::new(TileEncoder::new_parallel(&builder, &tiler_config, mask_uploader_1));
    let mut b2 = CachePadded::new(TileEncoder::new_parallel(&builder, &tiler_config, mask_uploader_2));

    tiler.draw.max_edges_per_gpu_tile = max_edges_per_gpu_tile;
    tiler.draw.use_quads = use_quads;

    let mut row_time: u64 = 0;
    let mut tile_time: u64 = 0;

    let t0 = time::precise_time_ns();
    for _run in 0..n {
        let transform = Transform2D::translation(1.0, 1.0);

        b0.reset();
        b1.reset();
        b2.reset();

        builder.reset();
        tiler.clear_depth();
        tiler.draw.z_index = paths.len() as u16;
        // Loop over the paths in front-to-back order to take advantage of
        // occlusion culling.
        for (path, color) in paths.iter().rev() {
            if let Some(idx) = select_path {
                if idx != tiler.draw.z_index {
                    tiler.draw.z_index -= 1;
                    continue;
                }
            }

            tiler.set_pattern(TiledPattern::Color(*color));

            if parallel {
                tiler.tile_path_parallel(&mut ctx, path.iter(), Some(&transform), &mut [
                    &mut *b0, &mut *b1, &mut *b2, &mut *builder
                ]);

                // The order of the mask tiles doesn't matter within a path but it does between paths,
                // so extend the main builder's mask tiles buffer between each path.
                builder.masked_tiles.reserve(b0.masked_tiles.len() + b1.masked_tiles.len() + b2.masked_tiles.len());
                builder.masked_tiles.extend_from_slice(&b0.masked_tiles);
                builder.masked_tiles.extend_from_slice(&b1.masked_tiles);
                builder.masked_tiles.extend_from_slice(&b2.masked_tiles);
                b0.masked_tiles.clear();
                b1.masked_tiles.clear();
                b2.masked_tiles.clear();
            } else {
                tiler.tile_path(path.iter(), Some(&transform), &mut *builder);
            }

            tiler.draw.z_index -= 1;

            row_time += tiler.row_decomposition_time_ns;
            tile_time += tiler.tile_decomposition_time_ns;
        }

        builder.end_paths();
        b0.end_paths();
        b1.end_paths();
        b2.end_paths();

        // Since the paths were processed front-to-back we have to reverse
        // the alpha tiles to render then back-to-front.
        // This doesn't show up in profiles.
        builder.masked_tiles.reverse();
    }

    let t1 = time::precise_time_ns();

    let t = (t1 - t0) / n;

    println!("view box: {:?}", view_box);
    println!("{} solid_tiles", builder.solid_tiles.len());
    println!("{} alpha_tiles", builder.masked_tiles.len());
    println!("{} gpu_masks", builder.gpu_masks.len());
    println!("{} cpu_masks", builder.num_cpu_masks());
    println!("{} line edges", builder.line_edges.len());
    println!("{} quad edges", builder.quad_edges.len());
    println!("#edge distributions: {:?}", builder.edge_distributions);
    println!("");
    println!("-> {}ns", t);
    println!("-> {:.3}ms", t as f64 / 1000000.0);
    println!("-> row decomposition: {:.3}ms", (row_time / n) as f64 / 1000000.0);
    println!("-> tile decomposition: {:.3}ms", (tile_time / n) as f64 / 1000000.0);
    println!("{:?}", ctx.stats());

    if profile {
        // Ensures the MaskUploader's buffers are properly cleaned up since we won't
        // consume them.
        builder.reset();
        b0.reset();
        b1.reset();
        b2.reset();
        return;
    }

    let mut tile_renderer = tiling::tile_renderer::TileRenderer::new(
        &device,
        tile_size as u32,
        tile_atlas_size as u32,
        &globals_bind_group_layout,
    );

    tile_renderer.update2(&mut builder, &mut b0, &mut b1, &mut b2, parallel);

    let mut surface_desc = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: wgpu::TextureFormat::Bgra8Unorm,
        width: window_size.width,
        height: window_size.height,
        present_mode: wgpu::PresentMode::Mailbox,
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

        tile_renderer.begin_frame(
            &device,
            &queue,
            &*builder,
            scene.window_size.width as f32,
            scene.window_size.height as f32
        );

        queue.write_buffer(
            &globals_ubo,
            0,
            bytemuck::cast_slice(&[tiling::gpu::GpuGlobals {
                resolution: vector(
                    scene.window_size.width as f32,
                    scene.window_size.height as f32,
                ),
                tile_size: tile_size as u32,
                tile_atlas_size,
            }]),
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tile"),
        });

        let mask_uploaders = &mut[&mut builder.mask_uploader, &mut b0.mask_uploader, &mut b1.mask_uploader, &mut b2.mask_uploader];
        tile_renderer.render(&device, &frame_view, &mut encoder, &globals_bind_group, mask_uploaders);

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
