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

fn main() {
    profiling::register_thread!("Main");

    let args: Vec<String> = std::env::args().collect();
    let (view_box, paths) = if args.len() > 1 {
        load_svg(&args[1])
    } else {
        let mut builder = lyon::path::Path::builder();
        builder.begin(point(0.0, 0.0));
        builder.line_to(point(50.0, 400.0));
        builder.line_to(point(450.0, 450.0));
        builder.line_to(point(400.0, 50.0));
        builder.end(true);

        (Box2D { min: point(0.0, 0.0), max: point(500.0, 500.0) }, vec![(builder.build(), Color { r: 50, g: 200, b: 100, a: 255 })])
    };

    // The tile size.
    let ts = 16;

    let mut tiler = Tiler::new(
        &TilerConfig {
            view_box,
            tile_size: size2(ts as f32, ts as f32),
            tile_padding: 0.5,
            tolerance: 0.05,
            flatten: true,
        }
    );


    let mut b0 = tiling::gpu_raster_encoder::GpuRasterEncoder::new();
    let mut b1 = tiling::gpu_raster_encoder::GpuRasterEncoder::new();
    let mut b2 = tiling::gpu_raster_encoder::GpuRasterEncoder::new();
    let mut b3 = tiling::gpu_raster_encoder::GpuRasterEncoder::new();


    let mut builder = tiling::gpu_raster_encoder::GpuRasterEncoder::new();

    let mut row_time: u64 = 0;
    let mut tile_time: u64 = 0;

    let n = 500;

    let t0 = time::precise_time_ns();
    for _run in 0..n {
        //println!("-- {:?}", _run);
        let transform = Transform2D::translation(1.0, 1.0);

        b0.reset();
        b1.reset();
        b2.reset();
        b3.reset();

        builder.reset();
        tiler.clear_depth();
        tiler.z_index = paths.len() as u16;
        //builder.max_edges_per_gpu_tile  = 32;

        // Loop over the paths in front-to-back order to take advantage of
        // occlusion culling.
        for (path, color) in paths.iter().rev() {
            builder.color = *color;

            //tiler.tile_path(path.iter(), Some(&transform), &mut builder);
            tiler.tile_path_parallel(path.iter(), Some(&transform), &mut [&mut b0, &mut b1, &mut b2, &mut b3]);

            tiler.z_index -= 1;

            row_time += tiler.row_decomposition_time_ns;
            tile_time += tiler.tile_decomposition_time_ns;
        }

        // Since the paths were processed front-to-back we have to reverse
        // the alpha tiles to render then back-to-front.
        // This doesn't show up in profiles.
        builder.mask_tiles.reverse();
    }

    let num_solid_tiles = builder.solid_tiles.len() as u32;
    let num_masked_tiles = builder.mask_tiles.len() as u32;
    let num_gpu_masks = builder.gpu_masks.len() as u32;

    let t1 = time::precise_time_ns();

    let t = (t1 - t0) / n;

    println!("view box: {:?}", view_box);
    //println!("{} edges", builder.edges.len());
    println!("{} solid_tiles", builder.solid_tiles.len());
    println!("{} alpha_tiles", builder.mask_tiles.len());
    println!("{} gpu_masks", builder.gpu_masks.len());
    println!("{} cpu_masks", builder.cpu_masks.len());
    println!("{} edges", builder.edges.len());
    println!("");
    println!("-> {}ns", t);
    println!("-> {:.3}ms", t as f64 / 1000000.0);
    println!("-> row decomposition: {:.3}ms", (row_time / n) as f64 / 1000000.0);
    println!("-> tile decomposition: {:.3}ms", (tile_time / n) as f64 / 1000000.0);


    return;

    let event_loop = EventLoop::new();
    let window = Window::new(&event_loop).unwrap();
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

    let quad_indices = [0u16, 1, 2, 0, 2, 3];
    let quad_ibo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Quad indices"),
        contents: bytemuck::cast_slice(&quad_indices),
        usage: wgpu::BufferUsages::INDEX,
    });

    let solid_tiles_vbo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Solid tile instances"),
        contents: bytemuck::cast_slice(&builder.solid_tiles),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let masked_tiles_vbo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Masked tile instances"),
        contents: bytemuck::cast_slice(&builder.mask_tiles),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let masks_vbo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Mask instances"),
        contents: bytemuck::cast_slice(&builder.gpu_masks),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let edges_ssbo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Edges"),
        contents: bytemuck::cast_slice(&builder.edges),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let solid_tiles = tiling::gpu::solid_tiles::SolidTiles::new(&device);

    let masks = tiling::gpu::masked_tiles::Masks::new(&device);

    let masked_tiles = tiling::gpu::masked_tiles::MaskedTiles::new(&device);

    let globals_ubo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Globals"),
        contents: bytemuck::cast_slice(&[
            tiling::gpu::GpuGlobals {
                resolution: vector(window_size.width as f32, window_size.height as f32),
            }
        ]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let solid_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Solid tiles"),
        layout: &solid_tiles.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(globals_ubo.as_entire_buffer_binding())
            },
        ],
    });

    let mask_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Mask atlas"),
        dimension: wgpu::TextureDimension::D2,
        size: wgpu::Extent3d {
            width: 2048,
            height: 2048,
            depth_or_array_layers: 1,
        },
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        mip_level_count: 1,
        sample_count: 1,
    });

    let mask_texture_view = mask_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let masked_tiles_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Masked tiles"),
        layout: &masked_tiles.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(globals_ubo.as_entire_buffer_binding())
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&mask_texture_view),
            },
        ],
    });

    let masks_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Masks"),
        layout: &masks.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(globals_ubo.as_entire_buffer_binding())
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Buffer(edges_ssbo.as_entire_buffer_binding()),
            },
        ],
    });

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

    println!("{:?}", &builder.mask_tiles[0..8]);

    event_loop.run(move |event, _, control_flow| {
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

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tile"),
        });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view: &mask_texture_view,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: true,
                    },
                    resolve_target: None,
                }],
                depth_stencil_attachment: None,
            });

            pass.set_pipeline(&masks.evenodd_pipeline);
            pass.set_bind_group(0, &masks_bind_group, &[]);
            pass.set_index_buffer(quad_ibo.slice(..), wgpu::IndexFormat::Uint16);
            pass.set_vertex_buffer(0, masks_vbo.slice(..));
            pass.draw_indexed(0..6, 0, 0..num_gpu_masks);
        }

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view: &frame_view,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: true,
                    },
                    resolve_target: None,
                }],
                depth_stencil_attachment: None,
            });

            pass.set_index_buffer(quad_ibo.slice(..), wgpu::IndexFormat::Uint16);

            pass.set_pipeline(&solid_tiles.pipeline);
            pass.set_bind_group(0, &solid_bind_group, &[]);
            pass.set_vertex_buffer(0, solid_tiles_vbo.slice(..));
            pass.draw_indexed(0..6, 0, 0..num_solid_tiles);

            pass.set_pipeline(&masked_tiles.pipeline);
            pass.set_bind_group(0, &masked_tiles_bind_group, &[]);
            pass.set_vertex_buffer(0, masked_tiles_vbo.slice(..));
            pass.draw_indexed(0..6, 0, 0..num_masked_tiles);
        }


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