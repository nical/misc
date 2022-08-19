use std::sync::Arc;
use tiling::*;
use tiling::canvas::*;
use tiling::gpu::GpuTileAtlasDescriptor;
use tiling::load_svg::*;
use tiling::gpu::mask_uploader::MaskUploader;
use lyon::path::geom::euclid::{size2, Transform2D};

use winit::dpi::PhysicalSize;
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::Window;
use wgpu::util::DeviceExt;
use futures::executor::block_on;

fn main() {
    profiling::register_thread!("Main");

    let args: Vec<String> = std::env::args().collect();

    let mut use_quads = false;
    for arg in &args {
        if arg == "--quads" { use_quads = true; }
    }

    let tile_size = 16.0;
    let tolerance = 0.1;
    let scale_factor = 2.0;
    let max_edges_per_gpu_tile = 128;
    let tile_atlas_size: u32 = 2048;
    let inital_window_size = size2(1200u32, 1000);

    let mut tiler_config = TilerConfig {
        view_box: Box2D::from_size(inital_window_size.to_f32()),
        tile_size: size2(tile_size, tile_size),
        tile_padding: 0.5,
        tolerance,
        flatten: false,
        mask_atlas_size: size2(tile_atlas_size, tile_atlas_size),
    };

    let event_loop = EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .with_inner_size(winit::dpi::Size::Physical(PhysicalSize::new(inital_window_size.width, inital_window_size.height)))
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
                target_tiles: GpuTileAtlasDescriptor::new(window_size.width as u32, window_size.height, tile_size as u32),
                src_color: GpuTileAtlasDescriptor::new(tile_atlas_size, tile_atlas_size, tile_size as u32),
                src_masks: GpuTileAtlasDescriptor::new(tile_atlas_size, tile_atlas_size, tile_size as u32),
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

    let mask_upload_copies = tiling::gpu::mask_uploader::MaskUploadCopies::new(&device, &globals_bind_group_layout);

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

    let mut canvas = Canvas::new(tiler_config.view_box);

    let mask_uploader = MaskUploader::new(&device, &mask_upload_copies.bind_group_layout, tile_atlas_size);

    let mut frame_builder = FrameBuilder::new(&tiler_config, mask_uploader);

    //frame_builder.tiler.set_scissor(&Box2D { min: point(500.0, 600.0), max: point(1000.0, 800.0) });
    frame_builder.tiler.draw.max_edges_per_gpu_tile = max_edges_per_gpu_tile;
    frame_builder.tiler.draw.use_quads = use_quads;

    //frame_builder.tiler.output_is_tiled = true;
    //frame_builder.tiler.color_tiles_per_row = 128;

    canvas.push_transform(&Transform2D::translation(1.0, 1.0));
    for (path, color) in paths {
        canvas.fill(Arc::new(path), Pattern::Color(color));
    }
    canvas.pop_transform();

    let mut builder = lyon::path::Path::builder();
    builder.begin(point(0.0, 0.0));
    builder.line_to(point(50.0, 400.0));
    builder.line_to(point(450.0, 450.0));
    builder.line_to(point(400.0, 50.0));
    builder.end(true);
    canvas.fill(Arc::new(builder.build()), Pattern::Checkerboard { colors: [Color { r: 10, g: 100, b: 250, a: 255 }, Color::WHITE], scale: 30.0 });


    let commands = canvas.finish();

    let mut tile_renderer = tiling::tile_renderer::TileRenderer::new(
        &device,
        tile_size as u32,
        tile_atlas_size as u32,
        &globals_bind_group_layout,
    );

    frame_builder.build(&commands);

    println!("view box: {:?}", view_box);
    println!("{} solid tiles", frame_builder.targets[0].tile_encoder.opaque_solid_tiles.len() + frame_builder.targets[0].tile_encoder.opaque_image_tiles.len());
    println!("{} alpha tiles", frame_builder.targets[0].tile_encoder.alpha_tiles.len());
    println!("{} gpu masks", frame_builder.targets[0].tile_encoder.gpu_masks.len());
    println!("{} cpu masks", frame_builder.targets[0].tile_encoder.num_cpu_masks());
    println!("{} line edges", frame_builder.targets[0].tile_encoder.line_edges.len());
    println!("{} quad edges", frame_builder.targets[0].tile_encoder.quad_edges.len());
    println!("{} batches", frame_builder.targets[0].tile_encoder.batches.len());
    println!("#edge distributions: {:?}", frame_builder.targets[0].tile_encoder.edge_distributions);
    println!("");
    println!("{:?}", frame_builder.stats());

    tile_renderer.update(&mut frame_builder.targets[0].tile_encoder);

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
            frame_builder.tiler.init(&tiler_config);
            let tiles = tiler_config.num_tiles();
            frame_builder.tile_mask.init(tiles.width, tiles.height);

            queue.write_buffer(
                &globals_ubo,
                0,
                bytemuck::cast_slice(&[tiling::gpu::GpuGlobals {
                    target_tiles: GpuTileAtlasDescriptor::new(scene.window_size.width as u32, scene.window_size.height, tile_size as u32),
                    src_color: GpuTileAtlasDescriptor::new(tile_atlas_size, tile_atlas_size, tile_size as u32),
                    src_masks: GpuTileAtlasDescriptor::new(tile_atlas_size, tile_atlas_size, tile_size as u32),
                }]),
            );    
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

        frame_builder.build(&commands);
        tile_renderer.update(&mut frame_builder.targets[0].tile_encoder);

        tile_renderer.begin_frame(
            &device,
            &queue,
            &frame_builder.targets[0].tile_encoder,
            &frame_builder.targets[0].checkerboard_pattern,
            scene.window_size.width as f32,
            scene.window_size.height as f32
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tile"),
        });

        tile_renderer.render(
            &device,
            &frame_view,
            &mut encoder,
            &globals_bind_group,
            &mut frame_builder.targets[0].tile_encoder.mask_uploader,
        );

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
