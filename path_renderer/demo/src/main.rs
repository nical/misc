use core::{context::*, FillPath};
use core::gpu::shader::{RenderPipelineBuilder, PrepareRenderPipelines, BlendMode};
use core::gpu::{GpuStore, PipelineDefaults, Shaders};
use core::path::Path;
use core::pattern::BindingsId;
use core::resources::{CommonGpuResources, GpuResources};
use core::shape::*;
use core::stroke::*;
use core::transform::Transforms;
use core::units::{
    point, vector, LocalRect, LocalToSurfaceTransform, LocalTransform, SurfaceIntSize,
};
use core::wgpu::util::DeviceExt;
use core::{BindingResolver, Color};
use debug_overlay::{CounterId, Counters, Overlay, CounterDescriptor, Orientation, Graphs, Column, Table};
use debug_overlay::wgpu::{Renderer as OverlayRenderer, RendererOptions as OverlayOptions};
use lyon::geom::{Angle, Box2D};
use lyon::path::PathEvent;
use lyon::path::traits::PathBuilder;
//use lyon::path::traits::PathBuilder;
use rectangles::{Aa, RectangleGpuResources, RectangleRenderer};
//use stats::{StatsRenderer, StatsRendererOptions, Overlay};
//use stats::views::{Column, Counter, Layout, Style};
use tiling2::Occlusion;
use winit::keyboard::{Key, NamedKey};
use wpf::{WpfGpuResources, WpfMeshRenderer};
use std::sync::Arc;
use std::time::{Instant, Duration};
use stencil::{StencilAndCoverRenderer, StencilAndCoverResources};
use tess::{MeshGpuResources, MeshRenderer};
use msaa_stroke::{MsaaStrokeGpuResources, MsaaStrokeRenderer};
//use tiling::*;

use pattern_checkerboard::{Checkerboard, CheckerboardRenderer};
use pattern_color::SolidColorRenderer;
use pattern_linear_gradient::{LinearGradient, LinearGradientRenderer};
use pattern_texture::TextureRenderer;

use futures::executor::block_on;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, Event, KeyEvent, WindowEvent};
use winit::event_loop::{EventLoop, EventLoopWindowTarget};
use winit::window::Window;

use core::wgpu;

mod load_svg;
use load_svg::*;

const TILING: usize = 0;
//const TESS: usize = 1;
const STENCIL: usize = 2;
const WPF: usize = 3;
const FILL_RENDERER_STRINGS: &[&str] = &["tiling", "tessellation", "stencil and cover", "wpf"];

const STROKE_TO_FILL: usize = 0;
const INSTANCED: usize = 1;
const STROKE_RENDERER_STRINGS: &[&str] = &["stroke-to-fill", "instanced"];

const NUM_SCENES: u32 = 2;

struct Renderers {
    //tiling: TileRenderer,
    tiling2: tiling2::TileRenderer,
    meshes: MeshRenderer,
    stencil: StencilAndCoverRenderer,
    wpf: WpfMeshRenderer,
    rectangles: RectangleRenderer,
    msaa_strokes: MsaaStrokeRenderer,
}

impl Renderers {
    fn begin_frame(&mut self) {
        //self.tiling.begin_frame(ctx);
        self.tiling2.begin_frame();
        self.meshes.begin_frame();
        self.stencil.begin_frame();
        self.wpf.begin_frame();
        self.rectangles.begin_frame();
        self.msaa_strokes.begin_frame();
    }

    fn prepare(&mut self, pass: &BuiltRenderPass, transforms: &Transforms, prep: &mut PrepareRenderPipelines, _device: &wgpu::Device) {
        //self.tiling.prepare(pass, transforms, prep, &device);
        self.tiling2.prepare(pass, transforms, prep);
        self.meshes.prepare(pass, transforms, prep);
        self.stencil.prepare(pass, transforms, prep);
        self.rectangles.prepare(pass, prep);
        self.wpf.prepare(pass, transforms, prep);
        self.msaa_strokes.prepare(pass, transforms, prep);
    }

    fn upload(&mut self, gpu_resources: &mut GpuResources, shaders: &Shaders, device: &wgpu::Device, queue: &wgpu::Queue) {
        //self.tiling.upload(gpu_resources, &device, &queue);
        self.tiling2.upload(gpu_resources, &device, &queue);
        self.meshes.upload(gpu_resources, &device, &queue);
        self.stencil.upload(gpu_resources, &device);
        self.rectangles.upload(gpu_resources, &device, &queue);
        self.wpf.upload(gpu_resources, &device, &queue);
        self.msaa_strokes.upload(gpu_resources, &shaders, &device, &queue);
    }

    fn fill(&mut self, idx: usize) -> &mut dyn FillPath {
        [
            //&mut self.tiling as &mut dyn FillPath,
            &mut self.tiling2 as &mut dyn FillPath,
            &mut self.meshes as &mut dyn FillPath,
            &mut self.stencil as &mut dyn FillPath,
            &mut self.wpf as &mut dyn FillPath,
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
    let mut read_fill = false;
    let mut _use_ssaa4 = false;
    let mut read_occlusion = false;
    let mut z_buffer = None;
    let mut cpu_occlusion = None;
    let mut fill_renderer = 0;
    for arg in &args {
        if read_tolerance {
            tolerance = arg.parse::<f32>().unwrap();
            println!("tolerance: {}", tolerance);
        }
        if read_fill {
            fill_renderer = arg.parse::<usize>().unwrap() % FILL_RENDERER_STRINGS.len();
        }
        if read_occlusion {
            cpu_occlusion = Some(arg.contains("cpu") || arg.contains("all"));
            z_buffer = Some(arg.contains("gpu") || arg.contains("all") || arg.contains("z-buffer"));
        }
        if arg == "--x11" {
            // This used to get this demo to work in renderdoc (with the gl backend) but now
            // it runs into new issues.
            std::env::set_var("WINIT_UNIX_BACKEND", "x11");
        }
        if arg == "--trace" {
            trace = Some(std::path::Path::new("./trace"));
        }
        _use_ssaa4 |= arg == "--ssaa";
        force_gl |= arg == "--gl";
        force_vk |= arg == "--vulkan";
        asap |= arg == "--asap";
        read_tolerance = arg == "--tolerance";
        read_fill = arg == "--fill";
        read_occlusion = arg == "--occlusion";
    }

    let scale_factor = 2.0;

    let event_loop = EventLoop::new().unwrap();
    let window = winit::window::WindowBuilder::new()
        .with_inner_size(winit::dpi::Size::Physical(PhysicalSize::new(1200, 1000)))
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
    let surface = instance.create_surface(&window).unwrap();

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
            required_features: wgpu::Features::default(),
            required_limits: wgpu::Limits::default(),
        },
        trace,
    ))
    .unwrap();

    let (_, paths) = if args.len() > 1 && !args[1].starts_with('-') {
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

    //tiler_config.view_box = view_box;

    let mut shaders = Shaders::new();
    let mut render_pipelines = core::gpu::shader::RenderPipelines::new();

    let patterns = Patterns {
        colors: SolidColorRenderer::register(&mut shaders),
        gradients: LinearGradientRenderer::register(&mut shaders),
        checkerboards: CheckerboardRenderer::register(&mut shaders),
        textures: TextureRenderer::register(&device, &mut shaders),
    };

    let mut main_pass = RenderPassBuilder::new();
    let mut transforms = Transforms::new();

    let mut gpu_store = GpuStore::new(2048, &device);

    let mut common_resources = CommonGpuResources::new(
        &device,
        SurfaceIntSize::new(window_size.width as i32, window_size.height as i32),
        &gpu_store,
        &mut shaders,
    );

    //let tiling_resources = TilingGpuResources::new(
    //    &mut common_resources,
    //    &device,
    //    &mut shaders,
    //    &patterns.textures,
    //    mask_atlas_size,
    //    color_atlas_size,
    //    _use_ssaa4,
    //);

    let tiling2_resources = tiling2::TileGpuResources::new(&device, &mut shaders);

    let mesh_resources = MeshGpuResources::new(&device, &mut shaders);

    let stencil_resources =
        StencilAndCoverResources::new(&mut common_resources, &device, &mut shaders);

    let rectangle_resources = RectangleGpuResources::new(&device, &mut shaders);

    let wpf_resources = WpfGpuResources::new(&device, &mut shaders);

    let stroke_resources = MsaaStrokeGpuResources::new(&device, &mut shaders);

    let mut gpu_resources = GpuResources::new();
    let common_handle = gpu_resources.register(common_resources);
    let tiling2_handle = gpu_resources.register(tiling2_resources);
    //let tiling_handle = gpu_resources.register(tiling_resources);
    let mesh_handle = gpu_resources.register(mesh_resources);
    let stencil_handle = gpu_resources.register(stencil_resources);
    let rectangle_handle = gpu_resources.register(rectangle_resources);
    let wpf_handle = gpu_resources.register(wpf_resources);
    let stroke_handle = gpu_resources.register(stroke_resources);

    let mut stats_renderer = OverlayRenderer::new(&device, &queue, &OverlayOptions {
        target_format: wgpu::TextureFormat::Bgra8Unorm,
        .. OverlayOptions::default()
    });

    let mut overlay = Overlay::new();

    let mut counters = counters();
    counters.enable_history(BATCHING);
    counters.enable_history(PREPARE);
    counters.enable_history(RENDER);

    let tiling_occlusion = Occlusion {
        cpu: cpu_occlusion.unwrap_or(true),
        gpu: z_buffer.unwrap_or(false),
    };

    let mut renderers = Renderers {
        tiling2: tiling2::TileRenderer::new(
            0,
            common_handle,
            tiling2_handle,
            &gpu_resources[tiling2_handle],
            &tiling2::RendererOptions {
                tolerance,
                occlusion: tiling_occlusion,
                no_opaque_batches: !tiling_occlusion.cpu && !tiling_occlusion.gpu,
            }
        ),
        //tiling: TileRenderer::new(
        //    0,
        //    common_handle,
        //    tiling_handle,
        //    &gpu_resources[tiling_handle],
        //    &patterns.textures,
        //    &tiler_config,
        //    &patterns.textures,
        //),
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
    };

    renderers.tiling2.tolerance = tolerance;
    renderers.stencil.tolerance = tolerance;
    renderers.meshes.tolerance = tolerance;
    renderers.msaa_strokes.tolerance = tolerance;

    //renderers.tiling.tiler.draw.max_edges_per_gpu_tile = max_edges_per_gpu_tile;

    let mut source_textures = SourceTextures::new();

    let img_bgl = shaders.get_bind_group_layout(patterns.textures.bind_group_layout());
    let _image_binding =
        source_textures.add_texture(create_image(&device, &queue, &img_bgl.handle, 800, 600));

    let mut surface_desc = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: wgpu::TextureFormat::Bgra8Unorm,
        width: window_size.width,
        height: window_size.height,
        present_mode: wgpu::PresentMode::AutoVsync,
        alpha_mode: wgpu::CompositeAlphaMode::Opaque,
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
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
        fill_renderer,
        stroke_renderer: 0,
        msaa: MsaaMode::Auto,
        scene_idx: 0,
        debug_overlay: false,
    };

    update_title(&window, demo.fill_renderer, demo.stroke_renderer, demo.msaa);

    let window = &window;

    let mut depth_texture = None;
    let mut msaa_texture = None;
    let mut msaa_depth_texture = None;
    let mut temporary_texture = None;

    let mut sum_frame_build_time = Duration::ZERO;
    let mut sum_render_time = Duration::ZERO;
    let mut sum_present_time = Duration::ZERO;
    event_loop.run(move |event, window_target| {
        device.poll(wgpu::Maintain::Poll);

        if !update_inputs(event, window, window_target, &mut demo) {
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
            ..Default::default()
        });
        let size = SurfaceIntSize::new(demo.window_size.width, demo.window_size.height);

        let msaa_default = match demo.msaa {
            MsaaMode::Disabled => false,
            _ => true,
        };
        let msaa_tiling = match demo.msaa {
            MsaaMode::Enabled => true,
            _ => false,
        };

        let surface_cfg = SurfacePassConfig {
            depth: z_buffer.unwrap_or(demo.fill_renderer != TILING),
            msaa: if demo.fill_renderer == TILING || demo.fill_renderer == WPF { msaa_tiling } else { msaa_default },
            stencil: demo.fill_renderer == STENCIL,
            kind: SurfaceKind::Color,
        };

        gpu_store.clear();
        transforms.clear();

        renderers.begin_frame();

        overlay.begin_frame();

        if demo.debug_overlay {
            overlay.draw_item(&"Hello world\nNew line");
            overlay.end_group();

            let mut selection = Vec::new();
            counters.select_counters([BATCHING, PREPARE, RENDER].iter().cloned(), &mut selection);

            overlay.draw_item(&Table {
                    columns: &[
                        Column::color(),
                        Column::name().with_unit().label("CPU timings"),
                        Column::avg().label("Avg"),
                        Column::max().label("Max"),
                        Column::history_graph(),
                    ],
                    rows: &selection,
                    labels: true,
            });

            overlay.draw_item(&Graphs {
                counters: &selection,
                width: Some(120),
                height: None,
                reference_value: 8.0,
                orientation: Orientation::Vertical,
            });

            overlay.end_group();

            overlay.push_column();

            overlay.draw_item(&"More text");
            overlay.end_group();

            overlay.finish();
        }

        main_pass.begin(size, surface_cfg);


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

        let record_start = Instant::now();

        paint_scene(
            &paths,
            demo.fill_renderer,
            demo.stroke_renderer,
            test_stuff,
            &mut main_pass,
            &mut transforms,
            &mut renderers,
            &patterns,
            &mut gpu_store,
            &transform,
        );

        let frame_build_start = Instant::now();
        let record_time = frame_build_start - record_start;

        let mut prep_pipelines = render_pipelines.prepare();

        let built_pass = main_pass.end();
        renderers.prepare(&built_pass, &transforms, &mut prep_pipelines, &device);

        let changes = prep_pipelines.finish();
        render_pipelines.build(
            &[&changes],
            &mut RenderPipelineBuilder(&device, &mut shaders),
        );

        let frame_build_time = Instant::now() - frame_build_start;
        sum_frame_build_time += frame_build_time;

        let requirements = RenderPassesRequirements {
            depth_stencil: !surface_cfg.msaa && (surface_cfg.depth || surface_cfg.stencil),
            msaa: surface_cfg.msaa,
            msaa_depth_stencil: surface_cfg.msaa && (surface_cfg.depth || surface_cfg.stencil),
            temporary: false,
        };

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

        //let temporary_src_bind_group = temporary_texture.as_ref().map(|tex| {
        //    device.create_bind_group(&wgpu::BindGroupDescriptor {
        //        label: None,
        //        layout: &gpu_resources[common_handle].msaa_blit_src_bind_group_layout,
        //        entries: &[wgpu::BindGroupEntry {
        //            binding: 0,
        //            resource: wgpu::BindingResource::TextureView(tex),
        //        }],
        //    })
        //});

        let render_start = Instant::now();

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        renderers.upload(&mut gpu_resources, &shaders, &device, &queue);
        gpu_store.upload(&device, &queue);

        gpu_resources.begin_rendering(&mut encoder);

        let msaa_resolve = surface_cfg.msaa;
        let surface_cfg = main_pass.surface.config();
        let ops = wgpu::Operations {
            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
            store: if msaa_resolve {
                wgpu::StoreOp::Discard
            } else {
                wgpu::StoreOp::Store
            },
        };
        let pass_descriptor = &wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: if surface_cfg.msaa { msaa_texture.as_ref().unwrap() } else { &frame_view },
                resolve_target: if msaa_resolve {
                    Some(&frame_view)
                } else {
                    None
                },
                ops,
            })],
            depth_stencil_attachment: if surface_cfg.depth_or_stencil() {
                Some(wgpu::RenderPassDepthStencilAttachment {
                    view: if surface_cfg.msaa {
                        msaa_depth_texture.as_ref()
                    } else {
                        depth_texture.as_ref()
                    }
                    .unwrap(),
                    depth_ops: if surface_cfg.depth {
                        Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(0.0),
                            store: wgpu::StoreOp::Discard,
                        })
                    } else {
                        None
                    },
                    stencil_ops: if surface_cfg.stencil {
                        Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(128),
                            store: wgpu::StoreOp::Discard,
                        })
                    } else {
                        None
                    },
                })
            } else {
                None
            },
            timestamp_writes: None,
            occlusion_query_set: None,
        };

        stats_renderer.update(
            &overlay.geometry,
            (size.width as u32, size.height as u32),
            1.0,
            &device,
            &queue,
        );

        {
            let mut render_pass = encoder.begin_render_pass(&pass_descriptor);

            built_pass.encode(
                &[
                    &renderers.tiling2,
                    &renderers.meshes,
                    &renderers.stencil,
                    &renderers.rectangles,
                    &renderers.wpf,
                    &renderers.msaa_strokes,
                ],
                &gpu_resources,
                &source_textures,
                &render_pipelines,
                &mut render_pass,
            );
        }

        if demo.debug_overlay {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Debug overlay"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &frame_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            stats_renderer.render(&mut render_pass);
        }

        queue.submit(Some(encoder.finish()));

        let present_start = Instant::now();
        let render_time = present_start - render_start;
        sum_render_time += render_time;

        frame.present();

        let present_time = Instant::now() - present_start;
        sum_present_time += present_time;

        fn ms(duration: Duration) -> f32 {
            (duration.as_micros() as f64 / 1000.0) as f32
        }

        let rec_t = ms(record_time);
        let fbt = ms(frame_build_time);
        let rt = ms(render_time);
        let pt = ms(present_time);
        counters.set(BATCHING, rec_t);
        counters.set(PREPARE, fbt);
        counters.set(RENDER, rt);
        counters.set(PRESENT, pt);
        counters.set(CPU_TOTAL, rec_t + fbt + rt);

        counters.update();

        gpu_resources.end_frame();

        if asap {
            window.request_redraw();
        }
    }).unwrap();
}

fn paint_scene(
    paths: &[(Arc<Path>, Option<SvgPattern>, Option<load_svg::Stroke>)],
    fill_renderer: usize,
    stroke_renderer: usize,
    testing: bool,
    pass: &mut RenderPassBuilder,
    transforms: &mut Transforms,
    renderers: &mut Renderers,
    patterns: &Patterns,
    gpu_store: &mut GpuStore,
    transform: &LocalTransform,
) {
    if testing {
        let gradient = patterns.gradients.add(
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
            .transformed(&transforms.get_current().matrix().to_untyped()),
        );

        renderers.tiling2.fill_surface(&mut pass.ctx(), &transforms, gradient);
    }

    transforms.push(transform);

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
                    .transformed(&transforms.get_current().matrix().to_untyped()),
                ),
            };

            let path = FilledPath::new(path.clone());
            renderers.fill(fill_renderer).fill_path(&mut pass.ctx(), transforms, path, pattern);
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
                    .transformed(&transforms.get_current().matrix().to_untyped()),
                ),
            };

            match stroke_renderer {
                crate::INSTANCED => {
                    renderers.msaa_strokes.stroke_path(pass, transforms, path.clone(), pattern, width);
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
                    renderers.fill(fill_renderer).fill_path(&mut pass.ctx(), transforms, path, pattern);
                }
                _ => {
                    unimplemented!();
                }
            }
        }
    }

    if testing {

        transforms.set(&LocalToSurfaceTransform::rotation(Angle::radians(0.2)));
        let transform_handle = transforms.get_current_gpu_handle(gpu_store);
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
            pass,
            transforms,
            &LocalRect {
                min: point(200.0, 700.0),
                max: point(300.0, 900.0),
            },
            Aa::ALL,
            gradient,
            transform_handle,
        );
        renderers.rectangles.fill_rect(
            pass,
            transforms,
            &LocalRect {
                min: point(310.5, 700.5),
                max: point(410.5, 900.5),
            },
            Aa::LEFT | Aa::RIGHT | Aa::ALL,
            gradient,
            transform_handle,
        );
        transforms.pop();

        //renderers.tiling.fill_circle(
        //    ctx,
        //    transforms,
        //    Circle::new(point(500.0, 300.0), 200.0),
        //    patterns.gradients.add(
        //        gpu_store,
        //        LinearGradient {
        //            from: point(300.0, 100.0),
        //            color0: Color {
        //                r: 10,
        //                g: 200,
        //                b: 100,
        //                a: 100,
        //            },
        //            to: point(700.0, 100.0),
        //            color1: Color {
        //                r: 200,
        //                g: 100,
        //                b: 250,
        //                a: 255,
        //            },
        //        }
        //        .transformed(&transforms.get_current().matrix().to_untyped()),
        //    ),
        //);

        let mut builder = core::path::Path::builder();
        builder.begin(point(600.0, 0.0));
        builder.line_to(point(650.0, 400.0));
        builder.line_to(point(1050.0, 450.0));
        builder.line_to(point(1000.0, 50.0));
        builder.end(true);

        let fill: FilledPath = builder.build().into();
        renderers.fill(fill_renderer).fill_path(
            &mut pass.ctx(),
            transforms,
            fill,//.inverted(),
            patterns.checkerboards.add(
                gpu_store,
                &Checkerboard {
                    color0: Color {
                        r: 10,
                        g: 100,
                        b: 250,
                        a: 50,
                    },
                    color1: Color { r: 255, g: 255, b: 255, a: 100 },
                    scale: 25.0,
                    offset: point(0.0, 0.0),
                }
                .transformed(&transforms.get_current().matrix().to_untyped()),
            ).with_blend_mode(BlendMode::Screen),
        );

        transforms.pop();

        //let black = patterns.colors.add(Color::BLACK);
        //renderers.tiling.fill_rect(
        //    ctx,
        //    transforms,
        //    LocalRect {
        //        min: point(10.0, 10.0),
        //        max: point(50.0, 50.0),
        //    },
        //    black,
        //);
        //renderers.tiling.fill_rect(
        //    ctx,
        //    transforms,
        //    LocalRect {
        //        min: point(60.5, 10.5),
        //        max: point(100.5, 50.5),
        //    },
        //    black,
        //);

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
            renderers.tiling2.fill_path(&mut pass.ctx(), transforms, offset_path.clone(), patterns.colors.add(Color::RED));

            renderers.meshes.stroke_path(
                &mut pass.ctx(),
                transforms,
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
                &mut pass.ctx(),
                transforms,
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
            renderers.meshes.stroke_path(&mut pass.ctx(), transforms, b.build(), 1.0, patterns.colors.add(Color::BLACK));

            let green = patterns.colors.add(Color::GREEN);
            let blue = patterns.colors.add(Color::BLUE);
            let white = patterns.colors.add(Color::WHITE);
            for evt in offset_path.as_slice() {
                match evt {
                    PathEvent::Begin { at } => {
                        renderers.meshes.fill_circle(
                            &mut pass.ctx(),
                            transforms,
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
                            &mut pass.ctx(),
                            transforms,
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
                            &mut pass.ctx(),
                            transforms,
                            Circle {
                                center: ctrl.cast_unit(),
                                radius: 4.0,
                                inverted: false,
                            },
                            blue,
                        );
                        renderers.meshes.fill_circle(
                            &mut pass.ctx(),
                            transforms,
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
                            &mut pass.ctx(),
                            transforms,
                            Circle {
                                center: ctrl1.cast_unit(),
                                radius: 4.0,
                                inverted: false,
                            },
                            blue,
                        );
                        renderers.meshes.fill_circle(
                            &mut pass.ctx(),
                            transforms,
                            Circle {
                                center: ctrl2.cast_unit(),
                                radius: 4.0,
                                inverted: false,
                            },
                            blue,
                        );
                        renderers.meshes.fill_circle(
                            &mut pass.ctx(),
                            transforms,
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

    transforms.push(&LocalTransform::translation(10.0, 1.0));
    transforms.pop();
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

        *depth_texture = Some(depth.create_view(&Default::default()));
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

        *msaa_texture = Some(msaa.create_view(&Default::default()));
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

        *msaa_depth_texture = Some(msaa_depth.create_view(&Default::default()));
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

        *temporary_texture = Some(temporary.create_view(&Default::default()))
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
    pub debug_overlay: bool,
}

fn update_inputs(
    event: Event<()>,
    window: &Window,
    window_target: &EventLoopWindowTarget<()>,
    demo: &mut Demo,
) -> bool {
    let p = demo.pan;
    let z = demo.zoom;
    let fr = demo.fill_renderer;
    let sr = demo.stroke_renderer;
    let mut redraw = false;
    match event {
        Event::WindowEvent {
            event: WindowEvent::RedrawRequested,
            ..
        } => {
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
            window_target.exit();
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
                    event: KeyEvent {
                        state: ElementState::Pressed,
                        logical_key,
                        ..
                    },
                    ..
                },
            ..
        } => match logical_key {
            Key::Named(NamedKey::Escape) => {
                window_target.exit();
                return false;
            }
            Key::Named(NamedKey::PageDown) => {
                demo.target_zoom *= 0.8;
            }
            Key::Named(NamedKey::PageUp) => {
                demo.target_zoom *= 1.25;
            }
            Key::Named(NamedKey::ArrowLeft) => {
                demo.target_pan[0] += 100.0 / demo.target_zoom;
            }
            Key::Named(NamedKey::ArrowRight) => {
                demo.target_pan[0] -= 100.0 / demo.target_zoom;
            }
            Key::Named(NamedKey::ArrowUp) => {
                demo.target_pan[1] += 100.0 / demo.target_zoom;
            }
            Key::Named(NamedKey::ArrowDown) => {
                demo.target_pan[1] -= 100.0 / demo.target_zoom;
            }
            Key::Character(c) => {
                if c == "w" {
                    demo.wireframe = !demo.wireframe;
                } else if c == "k" {
                    demo.scene_idx = (demo.scene_idx + 1) % NUM_SCENES;
                    update_title(window, demo.fill_renderer, demo.stroke_renderer, demo.msaa);
                    redraw = true;
                } else if c == "j" {
                    if demo.scene_idx == 0 {
                        demo.scene_idx = NUM_SCENES - 1;
                    } else {
                        demo.scene_idx -= 1;
                    }
                    update_title(window, demo.fill_renderer, demo.stroke_renderer, demo.msaa);
                    redraw = true;
                } else if c == "f" {
                    demo.fill_renderer = (demo.fill_renderer + 1) % FILL_RENDERER_STRINGS.len();
                    update_title(window, demo.fill_renderer, demo.stroke_renderer, demo.msaa);
                    redraw = true;
                } else if c == "s" {
                    demo.stroke_renderer = (demo.stroke_renderer + 1) % STROKE_RENDERER_STRINGS.len();
                    update_title(window, demo.fill_renderer, demo.stroke_renderer, demo.msaa);
                    redraw = true;
                } else if c == "m" {
                    demo.msaa = match demo.msaa {
                        MsaaMode::Auto => MsaaMode::Disabled,
                        MsaaMode::Disabled => MsaaMode::Enabled,
                        MsaaMode::Enabled => MsaaMode::Auto,
                    };
                    update_title(window, demo.fill_renderer, demo.stroke_renderer, demo.msaa);
                    redraw = true;
                } else if c == "o" {
                    demo.debug_overlay = !demo.debug_overlay;
                    redraw = true;
                }
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
        let handle = device.create_texture_with_data(queue, desc, wgpu::util::TextureDataOrder::default(), data);
        let view = handle.create_view(&Default::default());
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

pub struct RenderPassesRequirements {
    pub msaa: bool,
    pub depth_stencil: bool,
    pub msaa_depth_stencil: bool,
    // Temporary color target used in place of the main target if we need
    // to read from it but can't.
    pub temporary: bool,
}

const BATCHING: CounterId = 0;
const PREPARE: CounterId = 1;
const RENDER: CounterId = 2;
const PRESENT: CounterId = 3;
const CPU_TOTAL: CounterId = 4;
const BATCHES: CounterId = 5;

fn counters() -> Counters {
    let float = &CounterDescriptor::float;
    let int = &CounterDescriptor::int;
    Counters::new(
        &[
            float("Batching", "ms", BATCHING).safe_range(0.0..4.0).color((100, 150, 200, 255)),
            float("Prepare",  "ms", PREPARE).safe_range(0.0..6.0).color((200, 150, 100, 255)),
            float("Render",   "ms", RENDER).safe_range(0.0..4.0).color((50, 50, 200, 255)),
            float("Present",  "ms", PRESENT).safe_range(0.0..12.0).color((200, 200, 30, 255)),
            float("Total",    "ms", CPU_TOTAL).safe_range(0.0..16.0).color((50, 250, 250, 255)),
            int( "Batches",   "",   BATCHES),
        ],
        60,
    )
}
