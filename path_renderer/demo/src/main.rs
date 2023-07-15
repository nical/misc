use core::canvas::*;
use core::shape::*;
use core::geom::euclid::default::Size2D;
use core::gpu::{GpuStore, PipelineDefaults, Shaders};
use core::path::Path;
use core::pattern::BindingsId;
use core::resources::{CommonGpuResources, GpuResources};
use core::stroke::*;
use core::units::{
    point, vector, LocalRect, LocalToSurfaceTransform, LocalTransform, SurfaceIntSize,
};
use core::wgpu::util::DeviceExt;
use core::{BindingResolver, Color};
use lyon::geom::Angle;
use lyon::path::geom::euclid::size2;
use rectangles::{Aa, RectangleGpuResources, RectangleRenderer};
use std::sync::Arc;
use std::time::Duration;
use stencil::{StencilAndCoverRenderer, StencilAndCoverResources};
use tess::{MeshGpuResources, MeshRenderer};
use tiling::*;

use pattern_checkerboard::{Checkerboard, CheckerboardRenderer};
use pattern_color::SolidColorRenderer;
use pattern_linear_gradient::{LinearGradient, LinearGradientRenderer};
use pattern_texture::TextureRenderer;

use futures::executor::block_on;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::Window;

use core::wgpu;

mod load_svg;
use load_svg::*;

const TILING: usize = 0;
const TESS: usize = 1;
const STENCIL: usize = 2;

fn main() {
    color_backtrace::install();
    profiling::register_thread!("Main");

    let args: Vec<String> = std::env::args().collect();

    let mut tolerance = 0.25;

    let mut trace = None;
    let mut force_gl = false;
    let mut force_vk = false;
    let mut asap = false;
    let mut read_tolerance = false;
    let mut use_ssaa4 = false;
    for arg in &args {
        if read_tolerance {
            read_tolerance = false;
            tolerance = arg.parse::<f32>().unwrap();
            println!("tolerance: {}", tolerance);
        }
        if arg == "--ssaa" {
            use_ssaa4 = true;
        }
        if arg == "--gl" {
            force_gl = true;
        }
        if arg == "--vulkan" {
            force_vk = true;
        }
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

    let scale_factor = 2.0;
    let max_edges_per_gpu_tile = 16;
    let mask_atlas_size: u32 = 2048;
    let color_atlas_size: u32 = 2048;
    let inital_window_size = size2(1200u32, 1000);

    let mut tiler_config = TilerConfig {
        view_box: Box2D::from_size(inital_window_size.to_f32()),
        tolerance,
        mask_atlas_size: size2(mask_atlas_size, mask_atlas_size),
        color_atlas_size: size2(color_atlas_size, color_atlas_size),
        staging_buffer_size: tiling::BYTES_PER_MASK as u32 * 2048,
    };

    let event_loop = EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .with_inner_size(winit::dpi::Size::Physical(PhysicalSize::new(
            inital_window_size.width,
            inital_window_size.height,
        )))
        .build(&event_loop)
        .unwrap();
    let window_size = window.inner_size();

    let backends = if force_gl {
        wgpu::Backends::GL
    } else if force_vk {
        wgpu::Backends::VULKAN
    } else {
        wgpu::Backends::all()
    };
    // create an instance
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends,
        ..wgpu::InstanceDescriptor::default()
    });

    // create an surface
    let surface = unsafe { instance.create_surface(&window).unwrap() };

    // create an adapter
    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: Some(&surface),
        force_fallback_adapter: false,
    }))
    .unwrap();
    println!("{:#?}", adapter.get_info());
    // create a device and a queue
    let (device, queue) = block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: None,
            features: wgpu::Features::default(),
            limits: wgpu::Limits::default(),
        },
        trace,
    ))
    .unwrap();

    let (view_box, paths) = if args.len() > 1 && !args[1].starts_with('-') {
        load_svg(&args[1], scale_factor)
    } else {
        let mut builder = core::path::Path::builder();
        builder.begin(point(0.0, 0.0));
        builder.line_to(point(50.0, 400.0));
        builder.line_to(point(450.0, 450.0));
        builder.line_to(point(400.0, 50.0));
        builder.end(true);

        (
            Box2D {
                min: point(0.0, 0.0),
                max: point(500.0, 500.0),
            },
            vec![(
                Arc::new(builder.build()),
                SvgPattern::Color(Color {
                    r: 50,
                    g: 200,
                    b: 100,
                    a: 255,
                }),
            )],
        )
    };

    tiler_config.view_box = view_box;

    let mut shaders = Shaders::new();

    let patterns = Patterns {
        colors: SolidColorRenderer::register(&mut shaders),
        gradients: LinearGradientRenderer::register(&mut shaders),
        checkerboards: CheckerboardRenderer::register(&mut shaders),
        textures: TextureRenderer::register(&device, &mut shaders),
    };

    let mut canvas = Canvas::new();

    let mut gpu_store = GpuStore::new(2048, &device);

    let mut common_resources = CommonGpuResources::new(
        &device,
        SurfaceIntSize::new(window_size.width as i32, window_size.height as i32),
        &gpu_store,
        &mut shaders,
    );

    let tiling_resources = TilingGpuResources::new(
        &mut common_resources,
        &device,
        &mut shaders,
        &patterns.textures,
        mask_atlas_size,
        color_atlas_size,
        use_ssaa4,
    );

    let mesh_resources = MeshGpuResources::new(&device, &mut shaders);

    let stencil_resources =
        StencilAndCoverResources::new(&mut common_resources, &device, &mut shaders);

    let rectangle_resources = RectangleGpuResources::new(&device, &mut shaders);

    let mut gpu_resources = GpuResources::new();
    let common_handle = gpu_resources.register(common_resources);
    let tiling_handle = gpu_resources.register(tiling_resources);
    let mesh_handle = gpu_resources.register(mesh_resources);
    let stencil_handle = gpu_resources.register(stencil_resources);
    let rectangle_handle = gpu_resources.register(rectangle_resources);

    let mut tiling = TileRenderer::new(
        0,
        common_handle,
        tiling_handle,
        &tiler_config,
        &patterns.textures,
    );
    let mut meshes = MeshRenderer::new(1, common_handle, mesh_handle);
    let mut stencil = StencilAndCoverRenderer::new(2, common_handle, stencil_handle);
    let mut rectangles = RectangleRenderer::new(3, common_handle, rectangle_handle);

    tiling.tiler.draw.max_edges_per_gpu_tile = max_edges_per_gpu_tile;

    let mut source_textures = SourceTextures::new();

    let img_bgl = shaders.get_bind_group_layout(patterns.textures.bind_group_layout());
    let image_binding =
        source_textures.add_texture(create_image(&device, &queue, &img_bgl.handle, 800, 600));

    let mut surface_desc = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: wgpu::TextureFormat::Bgra8Unorm,
        width: window_size.width,
        height: window_size.height,
        present_mode: wgpu::PresentMode::AutoVsync,
        alpha_mode: wgpu::CompositeAlphaMode::Opaque,
        view_formats: vec![wgpu::TextureFormat::Bgra8Unorm],
    };

    surface.configure(&device, &surface_desc);

    window.request_redraw();

    let mut scene = SceneGlobals {
        zoom: 1.0,
        target_zoom: 1.0,
        pan: [0.0, 0.0],
        target_pan: [0.0, 0.0],
        window_size: SurfaceIntSize::new(window_size.width as i32, window_size.height as i32),
        wireframe: false,
        size_changed: true,
        render: true,
        selected_renderer: 0,
    };

    update_title(&window, scene.selected_renderer);

    let mut depth_texture = None;
    let mut msaa_texture = None;
    let mut msaa_depth_texture = None;
    let mut temporary_texture = None;

    let mut frame_build_time = Duration::ZERO;
    let mut render_time = Duration::ZERO;
    let mut present_time = Duration::ZERO;
    let mut frame_idx = 0;
    event_loop.run(move |event, _, control_flow| {
        device.poll(wgpu::Maintain::Poll);

        if !update_inputs(event, &window, control_flow, &mut scene) {
            return;
        }

        if scene.size_changed {
            scene.size_changed = false;
            let physical = scene.window_size;
            surface_desc.width = physical.width as u32;
            surface_desc.height = physical.height as u32;
            surface.configure(&device, &surface_desc);
            gpu_resources[common_handle].resize_target(
                SurfaceIntSize::new(scene.window_size.width, scene.window_size.height),
                &queue,
            );

            depth_texture = None;
            msaa_texture = None;
            msaa_depth_texture = None;
            temporary_texture = None;
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

        //println!("\n\n\n ----- \n\n");

        let frame_view = frame.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(shaders.defaults.color_format()),
            ..wgpu::TextureViewDescriptor::default()
        });
        let size = Size2D::new(
            scene.window_size.width as u32,
            scene.window_size.height as u32,
        );

        gpu_store.clear();
        canvas.begin_frame(SurfaceParameters::new(
            size,
            SurfaceState {
                depth: false,
                msaa: false,
                stencil: false,
            },
        ));
        tiling.begin_frame(&canvas);
        meshes.begin_frame(&canvas);
        stencil.begin_frame(&canvas);
        rectangles.begin_frame(&canvas);

        gpu_resources.begin_frame();

        let tx = scene.pan[0].round();
        let ty = scene.pan[1].round();
        let hw = (size.width as f32) * 0.5;
        let hh = (size.height as f32) * 0.5;
        let transform = LocalTransform::translation(tx, ty)
            .then_translate(-vector(hw, hh))
            .then_scale(scene.zoom, scene.zoom)
            .then_translate(vector(hw, hh));

        paint_scene(
            &paths,
            scene.selected_renderer,
            &mut canvas,
            &mut tiling,
            &mut meshes,
            &mut stencil,
            &mut rectangles,
            &patterns,
            &mut gpu_store,
            &transform,
        );
        //canvas.reconfigure_surface(SurfaceState { depth: false, msaa: false, stencil: false });
        tiling.fill_circle(
            &mut canvas,
            Circle {
                center: point(10.0, 600.0),
                radius: 100.0,
                inverted: false,
            },
            patterns.textures.sample_rect(
                &mut gpu_store,
                image_binding,
                &Box2D {
                    min: point(0.0, 0.0),
                    max: point(800.0, 600.0),
                },
                &Box2D {
                    min: point(-100.0, 500.0),
                    max: point(700.0, 1100.0),
                },
                true,
            ),
        );

        let frame_build_start = time::precise_time_ns();

        canvas.prepare();
        tiling.prepare(&canvas, &device);
        meshes.prepare(&canvas);
        stencil.prepare(&canvas);
        rectangles.prepare(&canvas);

        let requirements = canvas.build_render_passes(&mut [
            &mut tiling,
            &mut meshes,
            &mut stencil,
            &mut rectangles,
        ]);

        frame_build_time += Duration::from_nanos(time::precise_time_ns() - frame_build_start);

        create_render_targets(
            &device,
            &requirements,
            size,
            &shaders.defaults,
            &mut depth_texture,
            &mut msaa_texture,
            &mut msaa_depth_texture,
            &mut temporary_texture,
        );
        let temporary_src_bind_group = temporary_texture.as_ref().map(|tex| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &gpu_resources[common_handle].msaa_blit_src_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(tex),
                }],
            })
        });
        let target = SurfaceResources {
            main: &frame_view,
            depth: depth_texture.as_ref(),
            msaa_color: msaa_texture.as_ref(),
            msaa_depth: msaa_depth_texture.as_ref(),
            temporary_color: temporary_texture.as_ref(),
            temporary_src_bind_group: temporary_src_bind_group.as_ref(),
        };

        let render_start = time::precise_time_ns();

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        tiling.upload(&mut gpu_resources, &device, &queue);
        meshes.upload(&mut gpu_resources, &mut shaders, &device, &queue);
        stencil.upload(&mut gpu_resources, &mut shaders, &device);
        rectangles.upload(&mut gpu_resources, &mut shaders, &device, &queue);
        gpu_store.upload(&device, &queue);

        gpu_resources.begin_rendering(&mut encoder);

        canvas.render(
            &[&tiling, &meshes, &stencil, &rectangles],
            &gpu_resources,
            &source_textures,
            &mut shaders,
            &device,
            common_handle,
            &target,
            &mut encoder,
        );

        queue.submit(Some(encoder.finish()));

        let present_start = time::precise_time_ns();
        render_time += Duration::from_nanos(present_start - render_start);

        frame.present();

        present_time += Duration::from_nanos(time::precise_time_ns() - present_start);

        fn ms(duration: Duration) -> f64 {
            duration.as_micros() as f64 / 1000.0
        }

        let n = 300;
        frame_idx += 1;
        if frame_idx == n {
            let fbt = ms(frame_build_time) / (n as f64);
            let rt = ms(render_time) / (n as f64);
            let pt = ms(present_time) / (n as f64);
            frame_build_time = Duration::ZERO;
            render_time = Duration::ZERO;
            present_time = Duration::ZERO;
            frame_idx = 0;
            println!(
                "frame {:.2}ms (prepare {:.2}ms, render {:.2}ms, present {:.2}ms)",
                fbt + rt,
                fbt,
                rt,
                pt
            );
            print_stats(&tiling, &stencil, scene.window_size);
        }

        gpu_resources.end_frame();

        if asap {
            window.request_redraw();
        }
    });
}

fn paint_scene(
    paths: &[(Arc<Path>, SvgPattern)],
    selected_renderer: usize,
    canvas: &mut Canvas,
    tiling: &mut TileRenderer,
    meshes: &mut MeshRenderer,
    stencil: &mut StencilAndCoverRenderer,
    rectangles: &mut RectangleRenderer,
    patterns: &Patterns,
    gpu_store: &mut GpuStore,
    transform: &LocalTransform,
) {
    tiling.fill_canvas(
        canvas,
        patterns.gradients.add(
            gpu_store,
            LinearGradient {
                from: point(100.0, 100.0),
                color0: Color {
                    r: 10,
                    g: 50,
                    b: 250,
                    a: 255,
                },
                to: point(100.0, 1500.0),
                color1: Color {
                    r: 50,
                    g: 0,
                    b: 50,
                    a: 255,
                },
            }
            .transformed(&canvas.transforms.get_current().matrix().to_untyped()),
        ),
    );

    canvas.transforms.push(transform);

    tiling.fill_circle(
        canvas,
        Circle::new(point(500.0, 500.0), 800.0),
        patterns.gradients.add(
            gpu_store,
            LinearGradient {
                from: point(100.0, 100.0),
                color0: Color {
                    r: 200,
                    g: 150,
                    b: 0,
                    a: 255,
                },
                to: point(100.0, 1000.0),
                color1: Color {
                    r: 250,
                    g: 50,
                    b: 10,
                    a: 255,
                },
            }
            .transformed(&canvas.transforms.get_current().matrix().to_untyped()),
        ),
    );

    if selected_renderer == TESS {
        canvas.reconfigure_surface(SurfaceState {
            depth: true,
            msaa: true,
            stencil: false,
        });
    } else if selected_renderer == STENCIL {
        canvas.reconfigure_surface(SurfaceState {
            depth: true,
            msaa: true,
            stencil: true,
        });
    }

    for (path, pattern) in paths {
        let pattern = match pattern {
            &SvgPattern::Color(color) => patterns.colors.add(color),
            &SvgPattern::Gradient {
                color0,
                color1,
                from,
                to,
            } => patterns.gradients.add(
                gpu_store,
                LinearGradient {
                    color0,
                    color1,
                    from,
                    to,
                }
                .transformed(&canvas.transforms.get_current().matrix().to_untyped()),
            ),
        };

        let path = PathShape::new(path.clone());
        match selected_renderer {
            TILING => {
                tiling.fill_path(canvas, path, pattern);
            }
            TESS => {
                meshes.fill_path(canvas, path, pattern);
            }
            STENCIL => {
                stencil.fill_path(canvas, path, pattern);
            }
            _ => unimplemented!(),
        }
    }

    canvas.reconfigure_surface(SurfaceState {
        depth: true,
        msaa: false,
        stencil: true,
    });

    canvas
        .transforms
        .set(&LocalToSurfaceTransform::rotation(Angle::radians(0.2)));
    let transform_handle = canvas.transforms.get_current_gpu_handle(gpu_store);

    rectangles.fill_rect(
        canvas,
        &LocalRect {
            min: point(500.0, 200.0),
            max: point(600.0, 400.0),
        },
        Aa::ALL,
        patterns.gradients.add(
            gpu_store,
            LinearGradient {
                from: point(0.0, 200.0),
                to: point(0.0, 400.0),
                color0: Color {
                    r: 0,
                    g: 30,
                    b: 100,
                    a: 255,
                },
                color1: Color {
                    r: 0,
                    g: 60,
                    b: 250,
                    a: 255,
                },
            },
        ),
        transform_handle,
    );
    rectangles.fill_rect(
        canvas,
        &LocalRect {
            min: point(610.5, 200.5),
            max: point(710.5, 400.5),
        },
        Aa::LEFT | Aa::RIGHT | Aa::ALL,
        patterns.gradients.add(
            gpu_store,
            LinearGradient {
                from: point(0.0, 200.0),
                to: point(0.0, 400.0),
                color0: Color {
                    r: 0,
                    g: 30,
                    b: 100,
                    a: 255,
                },
                color1: Color {
                    r: 0,
                    g: 60,
                    b: 250,
                    a: 255,
                },
            },
        ),
        transform_handle,
    );
    canvas.transforms.pop();

    //if selected_renderer != TILING {
    canvas.reconfigure_surface(SurfaceState {
        depth: false,
        msaa: false,
        stencil: false,
    });
    //}

    tiling.fill_circle(
        canvas,
        Circle::new(point(500.0, 300.0), 200.0),
        patterns.gradients.add(
            gpu_store,
            LinearGradient {
                from: point(300.0, 100.0),
                color0: Color {
                    r: 10,
                    g: 200,
                    b: 100,
                    a: 100,
                },
                to: point(700.0, 100.0),
                color1: Color {
                    r: 200,
                    g: 100,
                    b: 250,
                    a: 255,
                },
            }
            .transformed(&canvas.transforms.get_current().matrix().to_untyped()),
        ),
    );

    let mut builder = core::path::Path::builder();
    builder.begin(point(0.0, 0.0));
    builder.line_to(point(50.0, 400.0));
    builder.line_to(point(450.0, 450.0));
    builder.line_to(point(400.0, 50.0));
    builder.end(true);

    tiling.fill_path(
        canvas,
        builder.build(),
        patterns.checkerboards.add(
            gpu_store,
            &Checkerboard {
                color0: Color {
                    r: 10,
                    g: 100,
                    b: 250,
                    a: 255,
                },
                color1: Color::WHITE,
                scale: 25.0,
                offset: point(0.0, 0.0),
            }
            .transformed(&canvas.transforms.get_current().matrix().to_untyped()),
        ),
    );

    canvas.transforms.pop();

    let black = patterns.colors.add(Color::BLACK);
    tiling.fill_rect(
        canvas,
        LocalRect {
            min: point(10.0, 10.0),
            max: point(50.0, 50.0),
        },
        black,
    );
    tiling.fill_rect(
        canvas,
        LocalRect {
            min: point(60.5, 10.5),
            max: point(100.5, 50.5),
        },
        black,
    );

    let mut builder2 = core::path::Path::builder();
    {
        let o = transform.m31 * 0.1;
        let mut stroker = StrokeToFillBuilder::new(
            &mut builder2,
            &StrokeOptions {
                tolerance: 0.25,
                offsets: (-50.0, -60.0-o),
                miter_limit: 0.5,
                line_join: LineJoin::Miter,
                start_cap: LineCap::TriangleInverted,
                end_cap: LineCap::Round,
                add_empty_caps: true,
                .. Default::default()
            },
        );

        stroker.begin(point(110.0, 110.0));
        stroker.quadratic_bezier_to(point(200.0, 110.0), point(200.0, 200.0));
        stroker.quadratic_bezier_to(point(200.0, 300.0), point(110.0, 200.0));
        stroker.end(false);

        stroker.begin(point(300.0, 100.0));
        stroker.line_to(point(400.0, 100.0));
        stroker.line_to(point(400.0, 200.0));
        stroker.line_to(point(390.0, 250.0));
        stroker.end(true);

        stroker.begin(point(600.0, 400.0));
        stroker.end(true);
        stroker.begin(point(700.0, 400.0));
        stroker.end(true);
    }
    {
        let mut offsetter = OffsetBuilder::new(&mut builder2, &OffsetOptions {
            offset: transform.m31 * 0.1,
            join: LineJoin::Round,
            miter_limit: 0.5,
            tolerance: 0.25,
            simplify_inner_joins: true,
        });

        offsetter.begin(point(500.0, 500.0));
        offsetter.line_to(point(600.0, 500.0));
        offsetter.line_to(point(650.0, 400.0));
        offsetter.line_to(point(700.0, 500.0));
        offsetter.line_to(point(800.0, 500.0));
        offsetter.line_to(point(650.0, 600.0));
        offsetter.end(true);


        //offsetter.begin(point(800.0, 500.0));
        //offsetter.line_to(point(900.0, 500.0));
        //offsetter.line_to(point(900.0, 600.0));
        //offsetter.line_to(point(800.0, 600.0));
        //offsetter.end(true);

        offsetter.begin(point(800.0, 700.0));
        offsetter.line_to(point(800.0, 800.0));
        offsetter.line_to(point(900.0, 800.0));
        offsetter.line_to(point(900.0, 700.0));
        offsetter.end(false);
    }

    let offset_path = builder2.build();
    tiling.fill_path(canvas, offset_path.clone(), patterns.colors.add(Color::RED));

    meshes.stroke_path(canvas, offset_path.clone(), 1.0, patterns.colors.add(Color::BLACK));

    let mut b = Path::builder();
    b.begin(point(500.0, 500.0));
    b.line_to(point(600.0, 500.0));
    b.line_to(point(650.0, 400.0));
    b.line_to(point(700.0, 500.0));
    b.line_to(point(800.0, 500.0));
    b.line_to(point(650.0, 600.0));
    b.end(true);

    b.begin(point(110.0, 110.0));
    b.quadratic_bezier_to(point(200.0, 110.0), point(200.0, 200.0));
    b.quadratic_bezier_to(point(200.0, 300.0), point(110.0, 200.0));
    b.end(false);
    meshes.stroke_path(canvas, b.build(), 1.0, patterns.colors.add(Color::BLACK));


    let green = patterns.colors.add(Color::GREEN);
    let blue = patterns.colors.add(Color::BLUE);
    let white = patterns.colors.add(Color::WHITE);
    for evt in offset_path.as_slice() {
        match evt {
            PathEvent::Begin { at } => {
                meshes.fill_circle(canvas, Circle { center: at.cast_unit(), radius: 3.0, inverted: false }, green);
            }
            PathEvent::Line { to, .. } => {
                meshes.fill_circle(canvas, Circle { center: to.cast_unit(), radius: 3.0, inverted: false }, green);
            }
            PathEvent::Quadratic { ctrl, to, .. } => {
                meshes.fill_circle(canvas, Circle { center: ctrl.cast_unit(), radius: 4.0, inverted: false }, blue);
                meshes.fill_circle(canvas, Circle { center: to.cast_unit(), radius: 3.0, inverted: false }, white);
            }
            PathEvent::Cubic { ctrl1, ctrl2, to, .. } => {
                meshes.fill_circle(canvas, Circle { center: ctrl1.cast_unit(), radius: 4.0, inverted: false }, blue);
                meshes.fill_circle(canvas, Circle { center: ctrl2.cast_unit(), radius: 4.0, inverted: false }, blue);
                meshes.fill_circle(canvas, Circle { center: to.cast_unit(), radius: 2.0, inverted: false }, white);
            }
            _ => {}
        }
    }

    canvas
        .transforms
        .push(&LocalTransform::translation(10.0, 1.0));
    canvas.transforms.pop();
}

fn print_stats(
    tiling: &TileRenderer,
    stencil: &StencilAndCoverRenderer,
    window_size: SurfaceIntSize,
) {
    let mut stats = Stats::new();
    tiling.update_stats(&mut stats);
    println!("Tiling: {:#?}", stats);
    println!("Stencil-and-cover: {:?}", stencil.stats);
    println!("Data:");
    println!("      tiles: {:2} kb", stats.tiles_bytes() as f32 / 1000.0);
    println!("      edges: {:2} kb", stats.edges_bytes() as f32 / 1000.0);
    println!(
        "  cpu masks: {:2} kb",
        stats.cpu_masks_bytes() as f32 / 1000.0
    );
    println!(
        "   uploaded: {:2} kb",
        stats.uploaded_bytes() as f32 / 1000.0
    );
    let win_bytes = (window_size.width * window_size.height * 4) as f32;
    println!(
        " resolution: {}x{} ({:2} kb)  overhead {:2}%",
        window_size.width,
        window_size.height,
        win_bytes / 1000.0,
        stats.uploaded_bytes() as f32 * 100.0 / win_bytes as f32,
    );
    println!(
        "#edge distributions: {:?}",
        tiling.encoder.edge_distributions
    );
    println!("\n");
}

fn create_render_targets(
    device: &wgpu::Device,
    requirements: &RenderPassesRequirements,
    size: Size2D<u32>,
    defaults: &PipelineDefaults,
    depth_texture: &mut Option<wgpu::TextureView>,
    msaa_texture: &mut Option<wgpu::TextureView>,
    msaa_depth_texture: &mut Option<wgpu::TextureView>,
    temporary_texture: &mut Option<wgpu::TextureView>,
) {
    let size = wgpu::Extent3d {
        width: size.width,
        height: size.height,
        depth_or_array_layers: 1,
    };

    if requirements.depth_stencil && depth_texture.is_none() {
        println!("create depth texture");
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: defaults.depth_stencil_format().unwrap(),
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            label: Some("depth"),
            view_formats: &[],
        });

        *depth_texture = Some(depth.create_view(&wgpu::TextureViewDescriptor::default()));
    }

    if requirements.msaa && msaa_texture.is_none() {
        println!("create msaa texture");
        let msaa = device.create_texture(&wgpu::TextureDescriptor {
            size,
            mip_level_count: 1,
            sample_count: defaults.msaa_sample_count(),
            dimension: wgpu::TextureDimension::D2,
            format: defaults.msaa_format(),
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            label: Some("msaa"),
            view_formats: &[],
        });

        *msaa_texture = Some(msaa.create_view(&wgpu::TextureViewDescriptor::default()));
    }

    if requirements.msaa_depth_stencil && msaa_depth_texture.is_none() {
        println!("create msaa depth texture");
        let msaa_depth = device.create_texture(&wgpu::TextureDescriptor {
            size,
            mip_level_count: 1,
            sample_count: defaults.msaa_sample_count(),
            dimension: wgpu::TextureDimension::D2,
            format: defaults.depth_stencil_format().unwrap(),
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            label: Some("depth+msaa"),
            view_formats: &[],
        });

        *msaa_depth_texture = Some(msaa_depth.create_view(&wgpu::TextureViewDescriptor::default()));
    }

    if requirements.temporary && temporary_texture.is_none() {
        println!("create temp texture");
        let temporary = device.create_texture(&wgpu::TextureDescriptor {
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: defaults.color_format(),
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            label: Some("Temporary color"),
            view_formats: &[],
        });

        *temporary_texture = Some(temporary.create_view(&wgpu::TextureViewDescriptor::default()))
    }
}

// Default scene has all values set to zero
#[derive(Copy, Clone, Debug)]
pub struct SceneGlobals {
    pub zoom: f32,
    pub target_zoom: f32,
    pub pan: [f32; 2],
    pub target_pan: [f32; 2],
    pub window_size: SurfaceIntSize,
    pub wireframe: bool,
    pub size_changed: bool,
    pub render: bool,
    pub selected_renderer: usize,
}

fn update_inputs(
    event: Event<()>,
    window: &Window,
    control_flow: &mut ControlFlow,
    scene: &mut SceneGlobals,
) -> bool {
    let p = scene.pan;
    let z = scene.zoom;
    let r = scene.selected_renderer;
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
            let size = SurfaceIntSize::new(size.width as i32, size.height as i32);
            if scene.window_size != size {
                scene.window_size = size;
                scene.size_changed = true
            }
        }
        Event::WindowEvent {
            event: WindowEvent::MouseWheel { delta, .. },
            ..
        } => {
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
            VirtualKeyCode::J => {
                if scene.selected_renderer == 0 {
                    scene.selected_renderer = 2;
                } else {
                    scene.selected_renderer -= 1;
                }
                update_title(window, scene.selected_renderer);
            }
            VirtualKeyCode::K => {
                scene.selected_renderer = (scene.selected_renderer + 1) % 3;
                update_title(window, scene.selected_renderer);
            }

            _key => {}
        },
        _evt => {
            //println!("{:?}", _evt);
        }
    }

    if r != scene.selected_renderer {
        println!(
            "{}",
            &["tiling", "tessellation", "stencil and cover"][scene.selected_renderer]
        )
    }

    scene.zoom += (scene.target_zoom - scene.zoom) * 0.15;
    scene.pan[0] += (scene.target_pan[0] - scene.pan[0]) * 0.15;
    scene.pan[1] += (scene.target_pan[1] - scene.pan[1]) * 0.15;
    if p != scene.pan || z != scene.zoom || r != scene.selected_renderer {
        window.request_redraw();
    }

    *control_flow = ControlFlow::Poll;

    true
}

fn update_title(window: &Window, selected_renderer: usize) {
    window.set_title(match selected_renderer {
        TILING => "demo - tiling",
        TESS => "demo - tessellation",
        STENCIL => "demo - stencil_and_colver",
        _ => "demo",
    });
}

struct Patterns {
    colors: SolidColorRenderer,
    gradients: LinearGradientRenderer,
    checkerboards: CheckerboardRenderer,
    textures: TextureRenderer,
}

struct SourceTexture {
    //width: u32,
    //height: u32,
    //format: wgpu::TextureFormat,
    //handle: wgpu::Texture,
    //view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
}

struct SourceTextures {
    textures: Vec<SourceTexture>,
}

impl SourceTextures {
    fn new() -> Self {
        SourceTextures {
            textures: Vec::new(),
        }
    }

    fn add_texture(&mut self, texture: SourceTexture) -> BindingsId {
        let id = BindingsId::from_index(self.textures.len());
        self.textures.push(texture);
        id
    }
}

impl BindingResolver for SourceTextures {
    fn resolve(&self, id: core::pattern::BindingsId) -> Option<&wgpu::BindGroup> {
        if id.is_none() {
            return None;
        }

        Some(&self.textures[id.index()].bind_group)
    }
}

impl SourceTexture {
    fn from_data(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        desc: &wgpu::TextureDescriptor,
        data: &[u8],
    ) -> Self {
        let handle = device.create_texture_with_data(queue, desc, data);
        let view = handle.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("source texture"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            }],
        });
        SourceTexture {
            //width: desc.size.width,
            //height: desc.size.height,
            //format: desc.format,
            //handle,
            //view: view,
            bind_group,
        }
    }
}

fn create_image(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    w: u32,
    h: u32,
) -> SourceTexture {
    let mut img = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let r = (x * 255 / w) as u8;
            let g = (y * 255 / h) as u8;
            let b = if x % 50 == 0 || y % 50 == 0 { 255 } else { 0 };

            img.push(r);
            img.push(g);
            img.push(b);
            img.push(255);
        }
    }

    SourceTexture::from_data(
        device,
        queue,
        layout,
        &wgpu::TextureDescriptor {
            label: Some("image"),
            mip_level_count: 1,
            sample_count: 1,
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
        &img,
    )
}
