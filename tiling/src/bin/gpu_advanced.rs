//use lyon::tessellation;
//use lyon::tessellation::geometry_builder::*;
//use lyon::tessellation::{FillOptions, FillTessellator};
//use lyon::tessellation::{StrokeOptions, StrokeTessellator};
use lyon::math::point;
use lyon::geom::{Vector, Box2D};

use winit::dpi::PhysicalSize;
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::Window;
use wgpu::util::DeviceExt;
use futures::executor::block_on;

use tiling::advanced_raster_encoder::*;
use tiling::Color;

use tiling::gpu::GpuGlobals;
use tiling::gpu::advanced_tiles::TileInstance;

fn main() {
    let event_loop = EventLoop::new();
    let window = Window::new(&event_loop).unwrap();

    let size = window.inner_size();

    let mut encoder = TileCommandEncoder::new();

    let mut fill = encoder.fill()
        .with_color(Color { r: 255, g: 128, b: 10, a: 255 });
    fill.add_edge(point(50.0, 50.0), point(400.0, 250.0));
    fill.add_edge(point(400.0, 250.0), point(220.0, 420.0));
    fill.add_edge(point(220.0, 420.0), point(50.0, 50.0));
    fill.build();

    let mut fill = encoder.fill()
        .with_color(Color { r: 55, g: 128, b: 210, a: 100 });
    fill.add_edge(point(150.0, 50.0), point(500.0, 250.0));
    fill.add_edge(point(500.0, 250.0), point(320.0, 420.0));
    fill.add_edge(point(320.0, 420.0), point(150.0, 50.0));
    fill.build();

    let mut g = encoder.push_group();

    let mut fill = g.fill()
        .with_color(Color { r: 10, g: 130, b: 100, a: 150 });
    fill.add_edge(point(300.0, 30.0), point(100.0, 400.0));
    fill.add_edge(point(100.0, 400.0), point(500.0, 200.0));
    fill.add_edge(point(500.0, 200.0), point(300.0, 30.0));
    fill.build();

    let mut mask = g.pop_with_mask();
    //.with_color(Color { r: 0, g: 200, b: 200, a: 100 });
    mask.add_edge(point(200.0, 50.0), point(400.0, 50.0));
    mask.add_edge(point(400.0, 50.0), point(500.0, 350.0));
    mask.add_edge(point(500.0, 350.0), point(300.0, 350.0));
    mask.add_edge(point(300.0, 350.0), point(200.0, 50.0));
    mask.build();

    let tiles = [
        TileInstance {
            rect: Box2D {
                min: point(0.0, 0.0),
                max: point(size.width as f32, size.height as f32),
            },
            path_data_start: 0,
            path_data_end: encoder.commands().len() as u32
        },
    ];

    let quad_indices = [0u16, 1, 2, 0, 2, 3];


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

    let advanced_tiles = tiling::gpu::advanced_tiles::create_pipeline(&device);

    let commands_ssbo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Commands"),
        contents: bytemuck::cast_slice(encoder.commands()),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let points_ssbo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Points"),
        contents: bytemuck::cast_slice(encoder.edges()),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let globals_ubo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Globals"),
        contents: bytemuck::cast_slice(&[
            GpuGlobals { resolution: Vector::new(size.width as f32, size.height as f32) }
        ]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let quad_ibo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Quad indices"),
        contents: bytemuck::cast_slice(&quad_indices),
        usage: wgpu::BufferUsages::INDEX,
    });

    let tiles_vbo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Tile instances"),
        contents: bytemuck::cast_slice(&tiles),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let tile_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Tiles"),
        layout: &advanced_tiles.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(globals_ubo.as_entire_buffer_binding())
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Buffer(commands_ssbo.as_entire_buffer_binding())
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Buffer(points_ssbo.as_entire_buffer_binding())
            },
        ],
    });

    let mut surface_desc = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: wgpu::TextureFormat::Bgra8Unorm,
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Mailbox,
    };

    surface.configure(&device, &surface_desc);

    window.request_redraw();

    let mut scene = SceneGlobals {
        zoom: 1.0,
        pan: [0.0, 0.0],
        window_size: size,
        wireframe: false,
        size_changed: true,
        render: true,
    };

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
                    view: &frame_view,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: true,
                    },
                    resolve_target: None,
                }],
                depth_stencil_attachment: None,
            });

            pass.set_pipeline(&advanced_tiles.pipeline);
            pass.set_bind_group(0, &tile_bind_group, &[]);
            pass.set_index_buffer(quad_ibo.slice(..), wgpu::IndexFormat::Uint16);
            pass.set_vertex_buffer(0, tiles_vbo.slice(..));
            pass.draw_indexed(0..6, 0, 0..1);
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
