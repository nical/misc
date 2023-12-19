use core::canvas::*;
use core::gpu::shader::{RenderPipelineBuilder, PrepareRenderPipelines};
use core::gpu::{GpuStore, PipelineDefaults, Shaders};
use core::path::Path;
use core::pattern::BindingsId;
use core::resources::{CommonGpuResources, GpuResources};
use core::shape::*;
use core::stroke::*;
use core::units::{
    point, vector, LocalRect, LocalToSurfaceTransform, LocalTransform, SurfaceIntSize,
};
use core::wgpu::util::DeviceExt;
use core::{BindingResolver, Color};
use lyon::geom::Angle;
use lyon::path::geom::euclid::size2;
use lyon::path::traits::PathBuilder;
//use lyon::path::traits::PathBuilder;
use rectangles::{Aa, RectangleGpuResources, RectangleRenderer};
use wpf::{WpfGpuResources, WpfMeshRenderer};
use std::sync::Arc;
use std::time::Duration;
use stencil::{StencilAndCoverRenderer, StencilAndCoverResources};
use tess::{MeshGpuResources, MeshRenderer};
use msaa_stroke::{MsaaStrokeGpuResources, MsaaStrokeRenderer};
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
//const TESS: usize = 1;
const STENCIL: usize = 2;
const WPF: usize = 3;
const TILING2: usize = 4;
const FILL_RENDERER_STRINGS: &[&str] = &["tiling", "tessellation", "stencil and cover", "wpf", "tiling2"];

const STROKE_TO_FILL: usize = 0;
const INSTANCED: usize = 1;
const STROKE_RENDERER_STRINGS: &[&str] = &["stroke-to-fill", "instanced"];

const NUM_SCENES: u32 = 2;

struct Renderers {
    tiling: TileRenderer,
    tiling2: tiling2::TileRenderer,
    meshes: MeshRenderer,
    stencil: StencilAndCoverRenderer,
    wpf: WpfMeshRenderer,
    rectangles: RectangleRenderer,
    msaa_strokes: MsaaStrokeRenderer,
}

impl Renderers {
    fn begin_frame(&mut self, ctx: &Context) {
        self.tiling.begin_frame(ctx);
        self.tiling2.begin_frame(ctx);
        self.meshes.begin_frame(ctx);
        self.stencil.begin_frame(ctx);
        self.wpf.begin_frame(ctx);
        self.rectangles.begin_frame(ctx);
        self.msaa_strokes.begin_frame(ctx);
    }

    fn prepare(&mut self, ctx: &Context, prep: &mut PrepareRenderPipelines, device: &wgpu::Device) {
        self.tiling.prepare(ctx, prep, &device);
        self.tiling2.prepare(ctx, prep);
        self.meshes.prepare(ctx, prep);
        self.stencil.prepare(ctx, prep);
        self.rectangles.prepare(ctx, prep);
        self.wpf.prepare(ctx, prep);
        self.msaa_strokes.prepare(ctx, prep);
    }

    fn upload(&mut self, gpu_resources: &mut GpuResources, shaders: &Shaders, device: &wgpu::Device, queue: &wgpu::Queue) {
        self.tiling.upload(gpu_resources, &device, &queue);
        self.tiling2.upload(gpu_resources, &device, &queue);
        self.meshes.upload(gpu_resources, &device, &queue);
        self.stencil.upload(gpu_resources, &device);
        self.rectangles.upload(gpu_resources, &device, &queue);
        self.wpf.upload(gpu_resources, &device, &queue);
        self.msaa_strokes.upload(gpu_resources, &shaders, &device, &queue);
    }

    fn fill(&mut self, idx: usize) -> &mut dyn FillPath {
        [
            &mut self.tiling as &mut dyn FillPath,
            &mut self.meshes as &mut dyn FillPath,
            &mut self.stencil as &mut dyn FillPath,
            &mut self.wpf as &mut dyn FillPath,
            &mut self.tiling2 as &mut dyn FillPath,
        ][idx]
    }
}

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
    let max_edges_per_gpu_tile = 512;
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
                Some(SvgPattern::Color(Color {
                    r: 50,
                    g: 200,
                    b: 100,
                    a: 255,
                })),
                None,
            )],
        )
    };

    tiler_config.view_box = view_box;

    let mut shaders = Shaders::new();
    let mut render_pipelines = core::gpu::shader::RenderPipelines::new();

    let patterns = Patterns {
        colors: SolidColorRenderer::register(&mut shaders),
        gradients: LinearGradientRenderer::register(&mut shaders),
        checkerboards: CheckerboardRenderer::register(&mut shaders),
        textures: TextureRenderer::register(&device, &mut shaders),
    };

    let mut ctx = Context::new(CanvasParams { tolerance });

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

    let wpf_resources = WpfGpuResources::new(&device, &mut shaders);

    let stroke_resources = MsaaStrokeGpuResources::new(&device, &mut shaders);

    let tiling2_resources = tiling2::TileGpuResources::new(&device, &mut shaders);

    let mut gpu_resources = GpuResources::new();
    let common_handle = gpu_resources.register(common_resources);
    let tiling_handle = gpu_resources.register(tiling_resources);
    let mesh_handle = gpu_resources.register(mesh_resources);
    let stencil_handle = gpu_resources.register(stencil_resources);
    let rectangle_handle = gpu_resources.register(rectangle_resources);
    let wpf_handle = gpu_resources.register(wpf_resources);
    let stroke_handle = gpu_resources.register(stroke_resources);
    let tiling2_handle = gpu_resources.register(tiling2_resources);

    let mut renderers = Renderers {
        tiling: TileRenderer::new(
            0,
            common_handle,
            tiling_handle,
            &gpu_resources[tiling_handle],
            &patterns.textures,
            &tiler_config,
            &patterns.textures,
        ),
        meshes: MeshRenderer::new(
            1,
            common_handle,
            mesh_handle,
            &gpu_resources[mesh_handle],
        ),
        stencil: StencilAndCoverRenderer::new(
            2,
            common_handle,
            stencil_handle,
            &gpu_resources[stencil_handle],
        ),
        rectangles: RectangleRenderer::new(
            3,
            common_handle,
            rectangle_handle,
            &gpu_resources[rectangle_handle],
        ),
        wpf: WpfMeshRenderer::new(
            4,
            common_handle,
            wpf_handle,
            &gpu_resources[wpf_handle],
        ),
        msaa_strokes: MsaaStrokeRenderer::new(
            5,
            common_handle,
            stroke_handle,
            &gpu_resources[stroke_handle],
        ),
        tiling2: tiling2::TileRenderer::new(
            6,
            common_handle,
            tiling2_handle,
            &gpu_resources[tiling2_handle],
        )
    };

    renderers.tiling.tiler.draw.max_edges_per_gpu_tile = max_edges_per_gpu_tile;

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
        view_formats: vec![],
    };

    surface.configure(&device, &surface_desc);

    window.request_redraw();

    let mut demo = Demo {
        zoom: 1.0,
        target_zoom: 1.0,
        pan: [0.0, 0.0],
        target_pan: [0.0, 0.0],
        window_size: SurfaceIntSize::new(window_size.width as i32, window_size.height as i32),
        wireframe: false,
        size_changed: true,
        render: true,
        fill_renderer: 0,
        stroke_renderer: 0,
        msaa: MsaaMode::Auto,
        scene_idx: 0,
    };

    update_title(&window, demo.fill_renderer, demo.stroke_renderer, demo.msaa);

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

        if !update_inputs(event, &window, control_flow, &mut demo) {
            return;
        }

        if demo.size_changed {
            demo.size_changed = false;
            let physical = demo.window_size;
            surface_desc.width = physical.width as u32;
            surface_desc.height = physical.height as u32;
            surface.configure(&device, &surface_desc);
            gpu_resources[common_handle].resize_target(
                SurfaceIntSize::new(demo.window_size.width, demo.window_size.height),
                &queue,
            );

            depth_texture = None;
            msaa_texture = None;
            msaa_depth_texture = None;
            temporary_texture = None;
        }

        if !demo.render {
            return;
        }
        demo.render = false;

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
        let size = SurfaceIntSize::new(demo.window_size.width, demo.window_size.height);

        gpu_store.clear();
        ctx.begin_frame(
            size,
            SurfacePassConfig {
                depth: false,
                msaa: if demo.msaa == MsaaMode::Enabled { true } else { false },
                stencil: false,
                kind: SurfaceKind::Color,
            },
        );

        renderers.begin_frame(&ctx);

        gpu_resources.begin_frame();

        let tx = demo.pan[0].round();
        let ty = demo.pan[1].round();
        let hw = (size.width as f32) * 0.5;
        let hh = (size.height as f32) * 0.5;
        let transform = LocalTransform::translation(tx, ty)
            .then_translate(-vector(hw, hh))
            .then_scale(demo.zoom, demo.zoom)
            .then_translate(vector(hw, hh));

        let test_stuff = demo.scene_idx == 0;
        paint_scene(
            &paths,
            demo.fill_renderer,
            demo.stroke_renderer,
            test_stuff,
            demo.msaa,
            &mut ctx,
            &mut renderers,
            &patterns,
            &mut gpu_store,
            &transform,
        );

        if false {
            renderers.tiling.fill_circle(
                &mut ctx,
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
        }
        let frame_build_start = time::precise_time_ns();

        let mut prep_pipelines = render_pipelines.prepare();

        ctx.prepare();
        renderers.prepare(&ctx, &mut prep_pipelines, &device);

        let changes = prep_pipelines.finish();
        render_pipelines.build(
            &[&changes],
            &mut RenderPipelineBuilder(&device, &mut shaders),
        );

        let requirements = ctx.build_render_passes(&mut [
            &mut renderers.tiling,
            &mut renderers.meshes,
            &mut renderers.stencil,
            &mut renderers.rectangles,
            &mut renderers.wpf,
            &mut renderers.msaa_strokes,
            &mut renderers.tiling2,
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

        renderers.upload(&mut gpu_resources, &shaders, &device, &queue);
        gpu_store.upload(&device, &queue);

        gpu_resources.begin_rendering(&mut encoder);

        ctx.render(
            &[
                &renderers.tiling,
                &renderers.meshes,
                &renderers.stencil,
                &renderers.rectangles,
                &renderers.wpf,
                &renderers.msaa_strokes,
                &renderers.tiling2,
            ],
            &gpu_resources,
            &source_textures,
            &mut render_pipelines,
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
            print_stats(&renderers, demo.window_size);
        }

        gpu_resources.end_frame();

        if asap {
            window.request_redraw();
        }
    });
}

fn paint_scene(
    paths: &[(Arc<Path>, Option<SvgPattern>, Option<load_svg::Stroke>)],
    fill_renderer: usize,
    stroke_renderer: usize,
    testing: bool,
    msaa: MsaaMode,
    ctx: &mut Context,
    renderers: &mut Renderers,
    patterns: &Patterns,
    gpu_store: &mut GpuStore,
    transform: &LocalTransform,
) {
    if testing {
        renderers.tiling.fill_canvas(
            ctx,
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
                .transformed(&ctx.transforms.get_current().matrix().to_untyped()),
            ),
        );
    }

    ctx.transforms.push(transform);

    if testing {
        renderers.tiling.fill_circle(
            ctx,
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
                .transformed(&ctx.transforms.get_current().matrix().to_untyped()),
            ),
        );
    }

    let msaa_default = match msaa {
        MsaaMode::Disabled => false,
        _ => true,
    };
    let msaa_tiling = match msaa {
        MsaaMode::Enabled => true,
        _ => false,
    };

    ctx.reconfigure_surface(SurfacePassConfig {
        depth: fill_renderer != TILING,
        msaa: if fill_renderer == TILING || fill_renderer == TILING2 || fill_renderer == WPF { msaa_tiling } else { msaa_default },
        stencil: fill_renderer == STENCIL,
        kind: SurfaceKind::Color,
    });

    for (path, fill, stroke) in paths {
        if let Some(fill) = fill {
            let pattern = match fill {
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
                    .transformed(&ctx.transforms.get_current().matrix().to_untyped()),
                ),
            };

            let path = FilledPath::new(path.clone());
            renderers.fill(fill_renderer).fill_path(ctx, path, pattern);
        }

        if let Some(stroke) = stroke {
            let scale = transform.m11;
            let width = f32::max(stroke.line_width, 1.0 / scale);

            let pattern = match stroke.pattern {
                SvgPattern::Color(color) => patterns.colors.add(color),
                SvgPattern::Gradient {
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
                    .transformed(&ctx.transforms.get_current().matrix().to_untyped()),
                ),
            };

            match stroke_renderer {
                crate::INSTANCED => {
                    renderers.msaa_strokes.stroke_path(ctx, path.clone(), pattern, width);
                }
                crate::STROKE_TO_FILL => {
                    let w = width * 0.5;
                    let mut stroked_path = Path::builder();
                    {
                        let mut stroker = StrokeToFillBuilder::new(
                            &mut stroked_path,
                            &StrokeOptions {
                                tolerance: 0.25 / scale,
                                offsets: (-w, w),
                                miter_limit: 0.5,
                                line_join: stroke.line_join,
                                start_cap: stroke.line_cap,
                                end_cap: stroke.line_cap,
                                add_empty_caps: true,
                                ..Default::default()
                            },
                        );
                        for evt in path.iter() {
                            stroker.path_event(evt, &[]);
                        }
                    }
                    let stroked_path = stroked_path.build();
                    let path = FilledPath::new(Arc::new(stroked_path));
                    renderers.fill(fill_renderer).fill_path(ctx, path, pattern);
                }
                _ => {
                    unimplemented!();
                }
            }
        }
    }

    if testing {
        ctx.reconfigure_surface(SurfacePassConfig {
            depth: true,
            msaa: msaa_default,
            stencil: true,
            kind: SurfaceKind::Color,
        });

        ctx.transforms
            .set(&LocalToSurfaceTransform::rotation(Angle::radians(0.2)));
        let transform_handle = ctx.transforms.get_current_gpu_handle(gpu_store);
        let gradient = patterns.gradients.add(
            gpu_store,
            LinearGradient {
                from: point(0.0, 700.0),
                to: point(0.0, 900.0),
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
        );

        renderers.rectangles.fill_rect(
            ctx,
            &LocalRect {
                min: point(200.0, 700.0),
                max: point(300.0, 900.0),
            },
            Aa::ALL,
            gradient,
            transform_handle,
        );
        renderers.rectangles.fill_rect(
            ctx,
            &LocalRect {
                min: point(310.5, 700.5),
                max: point(410.5, 900.5),
            },
            Aa::LEFT | Aa::RIGHT | Aa::ALL,
            gradient,
            transform_handle,
        );
        ctx.transforms.pop();

        ctx.reconfigure_surface(SurfacePassConfig {
            depth: false,
            msaa: msaa_tiling,
            stencil: false,
            kind: SurfaceKind::Color,
        });

        renderers.tiling.fill_circle(
            ctx,
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
                .transformed(&ctx.transforms.get_current().matrix().to_untyped()),
            ),
        );

        let mut builder = core::path::Path::builder();
        builder.begin(point(600.0, 0.0));
        builder.line_to(point(650.0, 400.0));
        builder.line_to(point(1050.0, 450.0));
        builder.line_to(point(1000.0, 50.0));
        builder.end(true);

        renderers.tiling.fill_path(
            ctx,
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
                .transformed(&ctx.transforms.get_current().matrix().to_untyped()),
            ),
        );

        ctx.transforms.pop();

        let black = patterns.colors.add(Color::BLACK);
        renderers.tiling.fill_rect(
            ctx,
            LocalRect {
                min: point(10.0, 10.0),
                max: point(50.0, 50.0),
            },
            black,
        );
        renderers.tiling.fill_rect(
            ctx,
            LocalRect {
                min: point(60.5, 10.5),
                max: point(100.5, 50.5),
            },
            black,
        );

        let mut p = Path::builder();

        //    p.begin(point(110.0, 110.0));
        //    p.quadratic_bezier_to(point(200.0, 110.0), point(200.0, 200.0));
        //    p.quadratic_bezier_to(point(200.0, 300.0), point(110.0, 200.0));
        //    p.end(false);
        //
        //    p.begin(point(300.0, 100.0));
        //    p.line_to(point(400.0, 100.0));
        //    p.line_to(point(400.0, 200.0));
        //    p.line_to(point(390.0, 250.0));
        //    p.end(true);

        p.begin(point(700.0, 100.0));
        p.quadratic_bezier_to(point(700.0, 330.0), point(900.0, 300.0));
        p.quadratic_bezier_to(point(700.0, 300.0), point(700.0, 500.0));
        //p.cubic_bezier_to(point(500.0, 100.0), point(500.0, 500.0), point(700.0, 100.0));
        p.end(false);

        //p.begin(point(600.0, 400.0));
        //p.end(true);
        //p.begin(point(700.0, 400.0));
        //p.end(true);
        let path_to_offset = p.build();

        let mut builder2 = core::path::Path::builder();
        {
            let o = transform.m31 * 0.1;
            let mut stroker = StrokeToFillBuilder::new(
                &mut builder2,
                &StrokeOptions {
                    tolerance: 0.25,
                    offsets: (10.0, -10.0 - o),
                    miter_limit: 0.5,
                    line_join: LineJoin::Round,
                    start_cap: LineCap::TriangleInverted,
                    end_cap: LineCap::TriangleInverted,
                    add_empty_caps: true,
                    ..Default::default()
                },
            );

            for evt in path_to_offset.iter() {
                stroker.path_event(evt, &[]);
            }
        }

        if false {
            let mut offsetter = OffsetBuilder::new(
                &mut builder2,
                &OffsetOptions {
                    offset: transform.m31 * 0.1,
                    join: LineJoin::Round,
                    miter_limit: 0.5,
                    tolerance: 0.25,
                    simplify_inner_joins: true,
                },
            );

            offsetter.begin(point(500.0, 500.0));
            offsetter.line_to(point(600.0, 500.0));
            offsetter.line_to(point(650.0, 400.0));
            offsetter.line_to(point(700.0, 500.0));
            offsetter.line_to(point(800.0, 500.0));
            offsetter.line_to(point(650.0, 600.0));
            offsetter.end(true);

            offsetter.begin(point(800.0, 500.0));
            offsetter.line_to(point(900.0, 500.0));
            offsetter.line_to(point(900.0, 600.0));
            offsetter.line_to(point(800.0, 600.0));
            offsetter.end(true);

            offsetter.begin(point(800.0, 700.0));
            offsetter.line_to(point(800.0, 800.0));
            offsetter.line_to(point(900.0, 800.0));
            offsetter.line_to(point(900.0, 700.0));
            offsetter.end(false);
        }

        //{
        //    msaa_stroker.stroke_path(
        //        ctx,
        //        path_to_offset.clone(),
        //        patterns.colors.add(Color { r: 100, g: 0, b: 0, a: 255 }),
        //        10.0
        //    );
        //}

        if false {
            let offset_path = builder2.build();
            renderers.tiling.fill_path(ctx, offset_path.clone(), patterns.colors.add(Color::RED));

            renderers.meshes.stroke_path(
                ctx,
                offset_path.clone(),
                1.0,
                patterns.colors.add(Color {
                    r: 100,
                    g: 0,
                    b: 0,
                    a: 255,
                }),
            );
            renderers.meshes.stroke_path(
                ctx,
                path_to_offset.clone(),
                1.0,
                patterns.colors.add(Color::BLACK),
            );

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
            renderers.meshes.stroke_path(ctx, b.build(), 1.0, patterns.colors.add(Color::BLACK));

            let green = patterns.colors.add(Color::GREEN);
            let blue = patterns.colors.add(Color::BLUE);
            let white = patterns.colors.add(Color::WHITE);
            for evt in offset_path.as_slice() {
                match evt {
                    PathEvent::Begin { at } => {
                        renderers.meshes.fill_circle(
                            ctx,
                            Circle {
                                center: at.cast_unit(),
                                radius: 3.0,
                                inverted: false,
                            },
                            green,
                        );
                    }
                    PathEvent::Line { to, .. } => {
                        renderers.meshes.fill_circle(
                            ctx,
                            Circle {
                                center: to.cast_unit(),
                                radius: 3.0,
                                inverted: false,
                            },
                            green,
                        );
                    }
                    PathEvent::Quadratic { ctrl, to, .. } => {
                        renderers.meshes.fill_circle(
                            ctx,
                            Circle {
                                center: ctrl.cast_unit(),
                                radius: 4.0,
                                inverted: false,
                            },
                            blue,
                        );
                        renderers.meshes.fill_circle(
                            ctx,
                            Circle {
                                center: to.cast_unit(),
                                radius: 3.0,
                                inverted: false,
                            },
                            white,
                        );
                    }
                    PathEvent::Cubic {
                        ctrl1, ctrl2, to, ..
                    } => {
                        renderers.meshes.fill_circle(
                            ctx,
                            Circle {
                                center: ctrl1.cast_unit(),
                                radius: 4.0,
                                inverted: false,
                            },
                            blue,
                        );
                        renderers.meshes.fill_circle(
                            ctx,
                            Circle {
                                center: ctrl2.cast_unit(),
                                radius: 4.0,
                                inverted: false,
                            },
                            blue,
                        );
                        renderers.meshes.fill_circle(
                            ctx,
                            Circle {
                                center: to.cast_unit(),
                                radius: 2.0,
                                inverted: false,
                            },
                            white,
                        );
                    }
                    _ => {}
                }
            }
        }
    }

    ctx.transforms.push(&LocalTransform::translation(10.0, 1.0));
    ctx.transforms.pop();
}

fn print_stats(
    renderers:&Renderers,
    window_size: SurfaceIntSize,
) {
    let mut stats = Stats::new();
    renderers.tiling.update_stats(&mut stats);
    println!("Tiling: {:#?}", stats);
    println!("Stencil-and-cover: {:?}", renderers.stencil.stats);
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
        renderers.tiling.encoder.edge_distributions
    );
    println!("\n");
}

fn create_render_targets(
    device: &wgpu::Device,
    requirements: &RenderPassesRequirements,
    size: SurfaceIntSize,
    defaults: &PipelineDefaults,
    depth_texture: &mut Option<wgpu::TextureView>,
    msaa_texture: &mut Option<wgpu::TextureView>,
    msaa_depth_texture: &mut Option<wgpu::TextureView>,
    temporary_texture: &mut Option<wgpu::TextureView>,
) {
    let size = wgpu::Extent3d {
        width: size.width as u32,
        height: size.height as u32,
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

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MsaaMode {
    Auto,
    Enabled,
    Disabled,
}

// Default scene has all values set to zero
#[derive(Copy, Clone, Debug)]
pub struct Demo {
    pub zoom: f32,
    pub target_zoom: f32,
    pub pan: [f32; 2],
    pub target_pan: [f32; 2],
    pub window_size: SurfaceIntSize,
    pub wireframe: bool,
    pub size_changed: bool,
    pub render: bool,
    pub fill_renderer: usize,
    pub stroke_renderer: usize,
    pub msaa: MsaaMode,
    pub scene_idx: u32,
}

fn update_inputs(
    event: Event<()>,
    window: &Window,
    control_flow: &mut ControlFlow,
    demo: &mut Demo,
) -> bool {
    let p = demo.pan;
    let z = demo.zoom;
    let fr = demo.fill_renderer;
    let sr = demo.stroke_renderer;
    let mut redraw = false;
    match event {
        Event::RedrawRequested(_) => {
            demo.render = true;
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
            if demo.window_size != size {
                demo.window_size = size;
                demo.size_changed = true
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
            let dx = dx / demo.target_zoom;
            let dy = dy / demo.target_zoom;
            if dx != 0.0 || dy != 0.0 {
                demo.target_pan[0] -= dx;
                demo.target_pan[1] -= dy;
                demo.pan[0] -= dx;
                demo.pan[1] -= dy;
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
                demo.target_zoom *= 0.8;
            }
            VirtualKeyCode::PageUp => {
                demo.target_zoom *= 1.25;
            }
            VirtualKeyCode::Left => {
                demo.target_pan[0] += 100.0 / demo.target_zoom;
            }
            VirtualKeyCode::Right => {
                demo.target_pan[0] -= 100.0 / demo.target_zoom;
            }
            VirtualKeyCode::Up => {
                demo.target_pan[1] += 100.0 / demo.target_zoom;
            }
            VirtualKeyCode::Down => {
                demo.target_pan[1] -= 100.0 / demo.target_zoom;
            }
            VirtualKeyCode::W => {
                demo.wireframe = !demo.wireframe;
            }
            VirtualKeyCode::K => {
                demo.scene_idx = (demo.scene_idx + 1) % NUM_SCENES;
                update_title(window, demo.fill_renderer, demo.stroke_renderer, demo.msaa);
                redraw = true;
            }
            VirtualKeyCode::J => {
                if demo.scene_idx == 0 {
                    demo.scene_idx = NUM_SCENES - 1;
                } else {
                    demo.scene_idx -= 1;
                }
                update_title(window, demo.fill_renderer, demo.stroke_renderer, demo.msaa);
                redraw = true;
            }
            VirtualKeyCode::F => {
                demo.fill_renderer = (demo.fill_renderer + 1) % FILL_RENDERER_STRINGS.len();
                update_title(window, demo.fill_renderer, demo.stroke_renderer, demo.msaa);
                redraw = true;
            }
            VirtualKeyCode::S => {
                demo.stroke_renderer = (demo.stroke_renderer + 1) % STROKE_RENDERER_STRINGS.len();
                update_title(window, demo.fill_renderer, demo.stroke_renderer, demo.msaa);
                redraw = true;
            }
            VirtualKeyCode::M => {
                demo.msaa = match demo.msaa {
                    MsaaMode::Auto => MsaaMode::Disabled,
                    MsaaMode::Disabled => MsaaMode::Enabled,
                    MsaaMode::Enabled => MsaaMode::Auto,
                };
                update_title(window, demo.fill_renderer, demo.stroke_renderer, demo.msaa);
                redraw = true;
            }

            _key => {}
        },
        _evt => {
            //println!("{:?}", _evt);
        }
    }

    if fr != demo.fill_renderer {
        println!("Fill: {}", FILL_RENDERER_STRINGS[demo.fill_renderer]);
    }
    if sr != demo.stroke_renderer {
        println!("Stroke: {}", STROKE_RENDERER_STRINGS[demo.stroke_renderer]);
    }

    demo.zoom += (demo.target_zoom - demo.zoom) * 0.15;
    demo.pan[0] += (demo.target_pan[0] - demo.pan[0]) * 0.15;
    demo.pan[1] += (demo.target_pan[1] - demo.pan[1]) * 0.15;
    redraw |= p != demo.pan || z != demo.zoom;

    if redraw {
        window.request_redraw();
    }

    *control_flow = ControlFlow::Wait;

    true
}

fn update_title(window: &Window, fill_renderer: usize, stroke_renderer: usize, msaa: MsaaMode) {
    let title = format!("Demo - fill: {}, stroke: {}, msaa: {msaa:?}",
        FILL_RENDERER_STRINGS[fill_renderer],
        STROKE_RENDERER_STRINGS[stroke_renderer],
    );
    window.set_title(&title);
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
