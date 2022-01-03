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

fn main() {
    profiling::register_thread!("Main");

    let args: Vec<String> = std::env::args().collect();

    let mut select_path = None;
    let mut profile = false;
    let mut use_quads = false;
    let mut sp = false;
    let mut parallel = false;
    for arg in &args {
        if sp {
            println!("{:?}", arg);
            select_path = Some(arg.parse::<u16>().unwrap());
            sp = false;
        }
        if arg == "--path" { sp = true; }
        if arg == "--parallel" { parallel = true; }
        if arg == "--profile" { profile = true; }
        if arg == "--quads" { use_quads = true; }
    }

    // The tile size.
    let tolerance = 0.1;
    let scale_factor = 2.0;
    let max_edges_per_gpu_tile = 64;
    let n = if profile { 1000 } else { 1 };

    let mut tiler_config = TilerConfig {
        view_box: Box2D::zero(),
        tile_size: size2(16.0 as f32, 16.0 as f32),
        tile_padding: 0.5,
        tolerance,
        flatten: false,
    };

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

    let mask_upload_copies = tiling::gpu::masked_tiles::MaskUploadCopies::new(&device);


    let (view_box, paths) = if args.len() > 1 {
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

    use tiling::gpu::masked_tiles::MaskUploader;
    let mask_uploader = MaskUploader::new(&device, &mask_upload_copies.bind_group_layout);
    let mask_uploader_0 = MaskUploader::new(&device, &mask_upload_copies.bind_group_layout);
    let mask_uploader_1 = MaskUploader::new(&device, &mask_upload_copies.bind_group_layout);
    let mask_uploader_2 = MaskUploader::new(&device, &mask_upload_copies.bind_group_layout);

    // Main builder.
    let mut builder = CachePadded::new(tiling::gpu_raster_encoder::GpuRasterEncoder::new(tolerance, mask_uploader));
    // Extra builders for worker threads.
    let mut b0 = CachePadded::new(tiling::gpu_raster_encoder::GpuRasterEncoder::new_parallel(&builder, mask_uploader_0));
    let mut b1 = CachePadded::new(tiling::gpu_raster_encoder::GpuRasterEncoder::new_parallel(&builder, mask_uploader_1));
    let mut b2 = CachePadded::new(tiling::gpu_raster_encoder::GpuRasterEncoder::new_parallel(&builder, mask_uploader_2));

    builder.max_edges_per_gpu_tile = max_edges_per_gpu_tile;
    builder.use_quads = use_quads;
    b0.max_edges_per_gpu_tile = max_edges_per_gpu_tile;
    b0.use_quads = use_quads;
    b1.max_edges_per_gpu_tile = max_edges_per_gpu_tile;
    b1.use_quads = use_quads;
    b2.max_edges_per_gpu_tile = max_edges_per_gpu_tile;
    b2.use_quads = use_quads;

    let mut row_time: u64 = 0;
    let mut tile_time: u64 = 0;

    let t0 = time::precise_time_ns();
    for _run in 0..n {
        //println!("-- {:?}", _run);
        let transform = Transform2D::translation(1.0, 1.0);

        b0.reset();
        b1.reset();
        b2.reset();

        builder.reset();
        tiler.clear_depth();
        tiler.z_index = paths.len() as u16;
        // Loop over the paths in front-to-back order to take advantage of
        // occlusion culling.
        for (path, color) in paths.iter().rev() {
            if let Some(idx) = select_path {
                println!("z-index: {:?}", tiler.z_index);
                if idx != tiler.z_index {
                    tiler.z_index -= 1;
                    continue;
                }
            }

            builder.color = *color;

            if parallel {
                b0.color = *color;
                b1.color = *color;
                b2.color = *color;

                tiler.tile_path_parallel(&mut ctx, path.iter(), Some(&transform), &mut [
                    &mut *b0, &mut *b1, &mut *b2, &mut *builder
                ]);

                // The order of the mask tiles doesn't matter within a path but it does between paths,
                // so extend the main builder's mask tiles buffer between each path.
                builder.mask_tiles.reserve(b0.mask_tiles.len() + b1.mask_tiles.len() + b2.mask_tiles.len());
                builder.mask_tiles.extend_from_slice(&b0.mask_tiles);
                builder.mask_tiles.extend_from_slice(&b1.mask_tiles);
                builder.mask_tiles.extend_from_slice(&b2.mask_tiles);
                b0.mask_tiles.clear();
                b1.mask_tiles.clear();
                b2.mask_tiles.clear();
            } else {
                tiler.tile_path(path.iter(), Some(&transform), &mut *builder);
            }

            tiler.z_index -= 1;

            row_time += tiler.row_decomposition_time_ns;
            tile_time += tiler.tile_decomposition_time_ns;
        }

        //let idx_str: &str = &args.last().as_ref().unwrap();
        //let idx: usize = idx_str.parse().unwrap();
        //builder.color = paths[idx].1;
        //tiler.tile_path(paths[idx].0.iter(), Some(&transform), &mut *builder);

        // Since the paths were processed front-to-back we have to reverse
        // the alpha tiles to render then back-to-front.
        // This doesn't show up in profiles.
        builder.mask_tiles.reverse();
    }

    if parallel {
        // Merge worker data into the main builder. TODO: It's not necessary, we should upload directly
        // off of each worker's data.
        builder.solid_tiles.reserve(b0.solid_tiles.len() + b1.solid_tiles.len() + b2.solid_tiles.len());
        builder.solid_tiles.extend_from_slice(&b0.solid_tiles);
        builder.solid_tiles.extend_from_slice(&b1.solid_tiles);
        builder.solid_tiles.extend_from_slice(&b2.solid_tiles);

        builder.gpu_masks.reserve(b0.gpu_masks.len() + b1.gpu_masks.len() + b2.gpu_masks.len());
        use tiling::gpu::masked_tiles::Mask;
        let mut offset = builder.quad_edges.len() as u32;
        for mask in &b0.gpu_masks {
            builder.gpu_masks.push(Mask {
                edges: (mask.edges.0 + offset, mask.edges.1 + offset),
                ..*mask
            });
        }
        offset += b0.quad_edges.len() as u32;
        for mask in &b1.gpu_masks {
            builder.gpu_masks.push(Mask {
                edges: (mask.edges.0 + offset, mask.edges.1 + offset),
                ..*mask
            });
        }
        offset += b1.quad_edges.len() as u32;
        for mask in &b2.gpu_masks {
            builder.gpu_masks.push(Mask {
                edges: (mask.edges.0 + offset, mask.edges.1 + offset),
                ..*mask
            });
        }


        builder.quad_edges.reserve(b0.quad_edges.len() + b1.quad_edges.len() + b2.quad_edges.len());
        builder.quad_edges.extend_from_slice(&b0.quad_edges);
        builder.quad_edges.extend_from_slice(&b1.quad_edges);
        builder.quad_edges.extend_from_slice(&b2.quad_edges);

        for i in 0..16 {
            builder.edge_distributions[i] += b0.edge_distributions[i]
                + b1.edge_distributions[i]
                + b2.edge_distributions[i];
        }
    }

    let num_solid_tiles = builder.solid_tiles.len() as u32;
    let num_masked_tiles = builder.mask_tiles.len() as u32;
    let num_gpu_masks = builder.gpu_masks.len() as u32;

    let t1 = time::precise_time_ns();

    let t = (t1 - t0) / n;

    println!("view box: {:?}", view_box);
    println!("{} solid_tiles", builder.solid_tiles.len());
    println!("{} alpha_tiles", builder.mask_tiles.len());
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
        return;
    }

    let solid_tiles = tiling::gpu::solid_tiles::SolidTiles::new(&device);

    let masks = tiling::gpu::masked_tiles::Masks::new(&device);

    let masked_tiles = tiling::gpu::masked_tiles::MaskedTiles::new(&device);

    let quad_indices = [0u16, 1, 2, 0, 2, 3];
    let quad_ibo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Quad indices"),
        contents: bytemuck::cast_slice(&quad_indices),
        usage: wgpu::BufferUsages::INDEX,
    });

    let globals_ubo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Globals"),
        contents: bytemuck::cast_slice(&[
            tiling::gpu::GpuGlobals {
                resolution: vector(window_size.width as f32, window_size.height as f32),
            }
        ]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let mut mask_textures = Vec::new();
    let mut mask_texture_views = Vec::new();
    let num_atlases = builder.num_mask_atlases();
    for i in 0..num_atlases {
        let label = format!("mask atlas #{}", i);
        let mask_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&label),
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

        mask_textures.push((mask_texture, masked_tiles_bind_group));
        mask_texture_views.push(mask_texture_view)
    }

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

    let line_edges_ssbo = if builder.line_edges.is_empty() {
        None
    } else {
        Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Line edges"),
            contents: bytemuck::cast_slice(&builder.line_edges),
            usage: wgpu::BufferUsages::STORAGE,
        }))
    };

    let quad_edges_ssbo = if builder.quad_edges.is_empty() {
        None
    } else {
        Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Quad edges"),
            contents: bytemuck::cast_slice(&builder.quad_edges),
            usage: wgpu::BufferUsages::STORAGE,
        }))
    };

    let mask_params_ubo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Mask params"),
        contents: bytemuck::cast_slice(&[
            tiling::gpu::masked_tiles::MaskParams {
                tile_size: 16.0,
                inv_atlas_width: 1.0 / 2048.0,
                masks_per_row: 2048 / 16,
            }
        ]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
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

    let line_masks_bind_group = line_edges_ssbo.map(|buffer| device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Masks"),
        layout: &masks.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(mask_params_ubo.as_entire_buffer_binding())
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Buffer(buffer.as_entire_buffer_binding()),
            },
        ],
    }));

    let quad_masks_bind_group = quad_edges_ssbo.map(|buffer| device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Masks"),
        layout: &masks.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(mask_params_ubo.as_entire_buffer_binding())
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Buffer(buffer.as_entire_buffer_binding()),
            },
        ],
    }));

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


        queue.write_buffer(
            &globals_ubo,
            0,
            bytemuck::cast_slice(&[tiling::gpu::GpuGlobals {
                resolution: vector(
                    scene.window_size.width as f32,
                    scene.window_size.height as f32,
                ),
            }]),
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tile"),
        });

        if let Some(line_masks_bind_group) = &line_masks_bind_group {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view: &mask_texture_views[0],
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: true,
                    },
                    resolve_target: None,
                }],
                depth_stencil_attachment: None,
            });

            pass.set_pipeline(&masks.line_evenodd_pipeline);
            pass.set_bind_group(0, &line_masks_bind_group, &[]);
            pass.set_index_buffer(quad_ibo.slice(..), wgpu::IndexFormat::Uint16);
            pass.set_vertex_buffer(0, masks_vbo.slice(..));
            pass.draw_indexed(0..6, 0, 0..num_gpu_masks);
        }

        if let Some(quad_masks_bind_group) = &quad_masks_bind_group {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view: &mask_texture_views[0],
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: true,
                    },
                    resolve_target: None,
                }],
                depth_stencil_attachment: None,
            });

            pass.set_pipeline(&masks.quad_evenodd_pipeline);
            pass.set_bind_group(0, &quad_masks_bind_group, &[]);
            pass.set_index_buffer(quad_ibo.slice(..), wgpu::IndexFormat::Uint16);
            pass.set_vertex_buffer(0, masks_vbo.slice(..));
            pass.draw_indexed(0..6, 0, 0..num_gpu_masks);
        }

        builder.mask_uploader.upload(&device, &mut encoder, &mask_upload_copies.pipeline, &quad_ibo, &mask_texture_views);
        b0.mask_uploader.upload(&device, &mut encoder, &mask_upload_copies.pipeline, &quad_ibo, &mask_texture_views);
        b1.mask_uploader.upload(&device, &mut encoder, &mask_upload_copies.pipeline, &quad_ibo, &mask_texture_views);
        b2.mask_uploader.upload(&device, &mut encoder, &mask_upload_copies.pipeline, &quad_ibo, &mask_texture_views);

        {
            let bg_color = wgpu::Color { r: 0.8, g: 0.8, b: 0.8, a: 1.0 };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view: &frame_view,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(bg_color),
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
            pass.set_bind_group(0, &mask_textures[0].1, &[]);
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
