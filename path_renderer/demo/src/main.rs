use core::frame::{RenderNodeDescriptor, RenderPass};
use core::instance::Instance;
use core::render_graph::{Allocation, Resource, BuiltGraph, ColorAttachment};
use core::{FillPath, Renderer};
use core::gpu::shader::BlendMode;
use core::path::Path;
use core::frame::Frame;
use core::resources::GpuResource;
use core::shape::*;
use core::stroke::*;
use core::units::{
    point, vector, LocalRect, LocalToSurfaceTransform, LocalTransform, SurfaceIntSize
};
use core::wgpu::util::DeviceExt;
use core::{BindingResolver, Color};
use core::{BindingsId, BindingsNamespace};
use std::u32;
use debug_overlay::{Column, Counters, Graphs, Orientation, Overlay, Table};
use debug_overlay::wgpu::{Renderer as OverlayRenderer, RendererOptions as OverlayOptions};
use lyon::geom::{Angle, Box2D};
use lyon::path::PathEvent;
use lyon::path::traits::PathBuilder;
//use lyon::path::traits::PathBuilder;
use rectangles::{Aa, RectangleRenderer, Rectangles};
//use stats::{StatsRenderer, StatsRendererOptions, Overlay};
//use stats::views::{Column, Counter, Layout, Style};
use tiling2::{Occlusion, Tiling};
use wpf::{Wpf, WpfMeshRenderer};
use std::sync::Arc;
use std::time::{Instant, Duration};
use stencil::{StencilAndCoverRenderer, StencilAndCover};
use tess::{MeshRenderer, Tessellation};
use msaa_stroke::{MsaaStroke, MsaaStrokeRenderer};
//use tiling::*;

use pattern_checkerboard::{Checkerboard, CheckerboardRenderer};
use pattern_color::SolidColorRenderer;
use pattern_linear_gradient::{LinearGradient, LinearGradientRenderer};
use pattern_texture::TextureRenderer;

use futures::executor::block_on;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

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
const ATLAS_SIZE: SurfaceIntSize = SurfaceIntSize::new(2048, 2048);

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

enum AppState {
    Initializing,
    Running(App),
    Closing,
}

struct App {
    window: Arc<Window>,
    view: Demo,

    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_desc: wgpu::SurfaceConfiguration,

    instance: Instance,

    renderers: Renderers,
    patterns: Patterns,

    //shaders: Shaders,
    //gpu_resources: GpuResources,
    //render_pipelines: core::gpu::shader::RenderPipelines,
    //gpu_store: GpuStore,

    overlay: Overlay,
    stats_renderer: OverlayRenderer,
    counters: Counters,
    wgpu_counters: counters::wgpu::Ids,
    renderer_counters: counters::render::Ids,
    paths: Vec<(Arc<Path>, Option<SvgPattern>, Option<Stroke>)>,

    asap: bool,
    z_buffer: Option<bool>,
}

impl ApplicationHandler for AppState {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let win_attrs = Window::default_attributes();
        let window = Arc::new(event_loop.create_window(win_attrs).unwrap());

        match self {
            AppState::Initializing => {
                if let Some(app) = App::init(window) {
                    *self = AppState::Running(app);
                    return;
                }
            }
            _ => {
                // TODO
            }
        }

        event_loop.exit();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        let this = match self {
            AppState::Running(this) => this,
            _ => {
                event_loop.exit();
                return;
            }
        };

        this.update_inputs(event_loop, id, event);

        if this.view.render {
            this.render();
        }

        if event_loop.exiting() {
            *self = AppState::Closing;
        }
    }
}

impl App {
    fn init(window: Arc<Window>) -> Option<Self> {
        let args: Vec<String> = std::env::args().collect();

        let mut tolerance = 0.25;

        let mut trace = None;
        let mut force_gl = false;
        let mut force_vk = false;
        let mut asap = false;
        let mut read_tolerance = false;
        let mut read_fill = false;
        let mut use_ssaa4 = false;
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
            use_ssaa4 |= arg == "--ssaa";
            force_gl |= arg == "--gl";
            force_vk |= arg == "--vulkan";
            asap |= arg == "--asap";
            read_tolerance = arg == "--tolerance";
            read_fill = arg == "--fill";
            read_occlusion = arg == "--occlusion";
        }

        let scale_factor = 2.0;

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
        let surface = instance.create_surface(window.clone()).unwrap();

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
                memory_hints: wgpu::MemoryHints::MemoryUsage,
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

        let mut instance = Instance::new(&device);

        let patterns = Patterns {
            colors: SolidColorRenderer::register(&mut instance.shaders),
            gradients: LinearGradientRenderer::register(&mut instance.shaders),
            checkerboards: CheckerboardRenderer::register(&mut instance.shaders),
            textures: TextureRenderer::register(&device, &mut instance.shaders),
        };

        let stats_renderer = OverlayRenderer::new(&device, &queue, &OverlayOptions {
            target_format: wgpu::TextureFormat::Bgra8Unorm,
            .. OverlayOptions::default()
        });

        let overlay = Overlay::new();

        let mut counters = Counters::new(60);

        let wgpu_counters = counters::wgpu::register("wgpu", &mut counters);
        let renderer_counters = counters::render::register("renderer", &mut counters);

        counters.enable_history(renderer_counters.batching());
        counters.enable_history(renderer_counters.prepare());
        counters.enable_history(renderer_counters.render());
        counters.enable_history(wgpu_counters.texture_memory());
        counters.enable_history(wgpu_counters.buffer_memory());
        counters.enable_history(wgpu_counters.memory_allocations());

        let tiling_occlusion = Occlusion {
            cpu: cpu_occlusion.unwrap_or(true),
            gpu: z_buffer.unwrap_or(false),
        };

        let rectangles = Rectangles::new(&device, &mut instance.shaders);
        let tessellation = Tessellation::new(&device, &mut instance.shaders);
        let tiling = Tiling::new(&device, &mut instance.shaders, use_ssaa4);
        let stencil_and_cover = StencilAndCover::new(&mut instance.resources.common, &device, &mut instance.shaders);
        let wpf = Wpf::new(&device, &mut instance.shaders);
        let msaa_stroke = MsaaStroke::new(&device, &mut instance.shaders);

        let mut renderers = Renderers {
            tiling2: tiling.new_renderer(
                &device,
                &mut instance.shaders,
                0,
                &tiling2::RendererOptions {
                    tolerance,
                    occlusion: tiling_occlusion,
                    no_opaque_batches: !tiling_occlusion.cpu && !tiling_occlusion.gpu,
                }
            ),
            meshes: tessellation.new_renderer(1),
            stencil: stencil_and_cover.new_renderer(2),
            rectangles: rectangles.new_renderer(3),
            wpf: wpf.new_renderer(4),
            msaa_strokes: msaa_stroke.new_renderer(&device, 5),
        };

        renderers.tiling2.tolerance = tolerance;
        renderers.stencil.tolerance = tolerance;
        renderers.meshes.tolerance = tolerance;
        renderers.msaa_strokes.tolerance = tolerance;

        //renderers.tiling.tiler.draw.max_edges_per_gpu_tile = max_edges_per_gpu_tile;

        let surface_desc = wgpu::SurfaceConfiguration {
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

        let view = Demo {
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

        Some(App {
            window,
            view,
            device,
            queue,
            surface,
            surface_desc,
            instance,
            renderers,
            patterns,
            overlay,
            stats_renderer,
            counters,
            wgpu_counters,
            renderer_counters,
            paths,
            asap,
            z_buffer,
        })
    }

    fn update_inputs(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let view = &mut self.view;
        let initial_scroll = view.pan;
        let initial_zoom = view.zoom;
        let mut redraw = false;
        match event {
            WindowEvent::RedrawRequested => {
                view.render = true;
            }
            WindowEvent::Destroyed | WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                view.window_size.width = size.width as i32;
                view.window_size.height = size.height as i32;
                view.size_changed = true;
                view.render = true;
            }
            WindowEvent::MouseWheel { delta, .. } => {
                use winit::event::MouseScrollDelta::*;
                let (dx, dy) = match delta {
                    LineDelta(x, y) => (x * 20.0, -y * 20.0),
                    PixelDelta(v) => (-v.x as f32, -v.y as f32),
                };
                let dx = dx / view.target_zoom;
                let dy = dy / view.target_zoom;
                if dx != 0.0 || dy != 0.0 {
                    view.target_pan[0] -= dx;
                    view.target_pan[1] -= dy;
                    view.pan[0] -= dx;
                    view.pan[1] -= dy;
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(key_code),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => match key_code {
                KeyCode::Escape => event_loop.exit(),
                KeyCode::PageDown => view.target_zoom *= 0.8,
                KeyCode::PageUp => view.target_zoom *= 1.25,
                KeyCode::ArrowLeft => view.target_pan[0] += 50.0 / view.target_zoom,
                KeyCode::ArrowRight => view.target_pan[0] -= 50.0 / view.target_zoom,
                KeyCode::ArrowUp => view.target_pan[1] -= 50.0 / view.target_zoom,
                KeyCode::ArrowDown => view.target_pan[1] += 50.0 / view.target_zoom,
                KeyCode::KeyW => {
                    view.wireframe = !view.wireframe;
                }
                KeyCode::KeyK => {
                    view.scene_idx = (view.scene_idx + 1) % NUM_SCENES;
                    update_title(&self.window, view.fill_renderer, view.stroke_renderer, view.msaa);
                    redraw = true;
                }
                KeyCode::KeyJ => {
                    if view.scene_idx == 0 {
                        view.scene_idx = NUM_SCENES - 1;
                    } else {
                        view.scene_idx -= 1;
                    }
                    update_title(&self.window, view.fill_renderer, view.stroke_renderer, view.msaa);
                    redraw = true;
                }
                KeyCode::KeyF => {
                    view.fill_renderer = (view.fill_renderer + 1) % FILL_RENDERER_STRINGS.len();
                    update_title(&self.window, view.fill_renderer, view.stroke_renderer, view.msaa);
                    redraw = true;
                }
                KeyCode::KeyS => {
                    view.stroke_renderer = (view.stroke_renderer + 1) % STROKE_RENDERER_STRINGS.len();
                    update_title(&self.window, view.fill_renderer, view.stroke_renderer, view.msaa);
                    redraw = true;
                }
                KeyCode::KeyM => {
                    view.msaa = match view.msaa {
                        MsaaMode::Auto => MsaaMode::Disabled,
                        MsaaMode::Disabled => MsaaMode::Enabled,
                        MsaaMode::Enabled => MsaaMode::Auto,
                    };
                    update_title(&self.window, view.fill_renderer, view.stroke_renderer, view.msaa);
                    redraw = true;
                }
                KeyCode::KeyO => {
                    view.debug_overlay = !view.debug_overlay;
                    redraw = true;
                }
                _key => {}
            },
            _evt => {}
        };

        if event_loop.exiting() {
            view.render = false;
            return;
        }

        view.zoom += (view.target_zoom - view.zoom) / 3.0;
        view.pan[0] = view.pan[0] + (view.target_pan[0] - view.pan[0]) / 3.0;
        view.pan[1] = view.pan[1] + (view.target_pan[1] - view.pan[1]) / 3.0;

        redraw |= view.pan != initial_scroll || view.zoom != initial_zoom;

        if redraw {
            self.window.request_redraw();
        }
    }

    fn render(&mut self) {
        if self.view.size_changed {
            self.view.size_changed = false;
            let physical = self.view.window_size;
            self.surface_desc.width = physical.width as u32;
            self.surface_desc.height = physical.height as u32;
            self.surface.configure(&self.device, &self.surface_desc);
        }

        if !self.view.render {
            return;
        }
        self.view.render = false;

        let wgpu_frame = match self.surface.get_current_texture() {
            Ok(texture) => texture,
            Err(e) => {
                println!("Swap-chain error: {:?}", e);
                return;
            }
        };

        //println!("\n\n\n ----- \n\n");

        let frame_view = wgpu_frame.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(self.instance.shaders.defaults.color_format()),
            ..Default::default()
        });

        let size = SurfaceIntSize::new(self.view.window_size.width, self.view.window_size.height);

        let msaa_default = match self.view.msaa {
            MsaaMode::Disabled => false,
            _ => true,
        };
        let msaa_tiling = match self.view.msaa {
            MsaaMode::Enabled => true,
            _ => false,
        };

        let mut frame = self.instance.begin_frame();

        self.renderers.begin_frame();

        self.overlay.begin_frame();

        if self.view.debug_overlay {
            self.overlay.draw_item(&"Hello world\nNew line");
            self.overlay.end_group();

            let mut selection = Vec::new();
            self.counters.select_counters([
                self.renderer_counters.batching(),
                self.renderer_counters.prepare(),
                self.renderer_counters.render()
            ].iter().cloned(), &mut selection);

            self.overlay.draw_item(&Table {
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

            self.overlay.draw_item(&Graphs {
                counters: &selection,
                width: Some(120),
                height: None,
                reference_value: 8.0,
                orientation: Orientation::Vertical,
            });

            self.overlay.end_group();

            self.overlay.push_column();

            self.overlay.draw_item(&"More text");
            self.overlay.end_group();

            selection.clear();
            self.counters.select_counters(
                self.wgpu_counters.all(),
                &mut selection
            );

            self.overlay.draw_item(&Table {
                    columns: &[
                        Column::name().with_unit().label("wgpu internals"),
                        Column::value(),
                        Column::history_graph(),
                    ],
                    rows: &selection,
                    labels: true,
            });

            self.overlay.finish();
        }

        let record_start = Instant::now();

        let depth = self.z_buffer.unwrap_or(self.view.fill_renderer != TILING);
        let stencil = self.view.fill_renderer == STENCIL;
        let msaa = if self.view.fill_renderer == TILING || self.view.fill_renderer == WPF { msaa_tiling } else { msaa_default };

        let attachments = [ColorAttachment::color().with_external(0, false)];
        let mut descriptor = RenderNodeDescriptor::new(size)
            .msaa(msaa)
            .attachments(&attachments);
        if depth || stencil {
            descriptor = descriptor.depth_stencil(Resource::Auto, depth, stencil);
        }

        let mut main_surface = frame.begin_render_pass(descriptor);
        frame.add_root(main_surface.node_id().color(0));

        let tx = self.view.pan[0].round();
        let ty = self.view.pan[1].round();
        let hw = (size.width as f32) * 0.5;
        let hh = (size.height as f32) * 0.5;
        let transform = LocalTransform::translation(tx, ty)
            .then_translate(-vector(hw, hh))
            .then_scale(self.view.zoom, self.view.zoom)
            .then_translate(vector(hw, hh));

        let test_stuff = self.view.scene_idx == 0;

        paint_scene(
            &self.paths,
            self.view.fill_renderer,
            self.view.stroke_renderer,
            test_stuff,
            &mut main_surface,
            &mut frame,
            &mut self.instance,
            &mut self.renderers,
            &self.patterns,
            &transform,
        );

        frame.end_render_pass(main_surface);

        let frame_build_start = Instant::now();
        let record_time = frame_build_start - record_start;

        // TODO
        self.stats_renderer.update(
            &self.overlay.geometry,
            (size.width as u32, size.height as u32),
            1.0,
            &self.device,
            &self.queue,
        );

        let mut encoder =
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let render_stats = self.instance.render(
            frame,
            &mut [
                &mut self.renderers.tiling2 as &mut dyn Renderer,
                &mut self.renderers.meshes,
                &mut self.renderers.stencil,
                &mut self.renderers.rectangles,
                &mut self.renderers.wpf,
                &mut self.renderers.msaa_strokes,
            ],
            &[],
            &[Some(&frame_view)],
            &self.device,
            &self.queue,
            &mut encoder,
        );

        if self.view.debug_overlay {
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

            self.stats_renderer.render(&mut render_pass);
        }

        let present_start = Instant::now();

        self.queue.submit(Some(encoder.finish()));

        wgpu_frame.present();

        let present_time = Instant::now() - present_start;

        fn ms(duration: Duration) -> f32 {
            (duration.as_micros() as f64 / 1000.0) as f32
        }

        let rec_t = ms(record_time);
        let fbt = render_stats.prepare_time_ms;
        let rt = render_stats.upload_time_ms + render_stats.render_time_ms;
        let pt = ms(present_time);
        self.counters.set(self.renderer_counters.batching(), rec_t);
        self.counters.set(self.renderer_counters.prepare(), fbt);
        self.counters.set(self.renderer_counters.render(), rt);
        self.counters.set(self.renderer_counters.present(), pt);
        self.counters.set(self.renderer_counters.cpu_total(), rec_t + fbt + rt);

        let wgpu_counters = self.device.get_internal_counters();
        debug_overlay::update_wgpu_internal_counters(&mut self.counters, self.wgpu_counters, &wgpu_counters);

        self.counters.update();

        if self.asap {
            self.window.request_redraw();
        }

        self.device.poll(wgpu::Maintain::Poll);
    }
}

fn main() {
    color_backtrace::install();
    profiling::register_thread!("Main");


    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Wait);

    let mut app = AppState::Initializing;
    event_loop.run_app(&mut app).unwrap();
}

fn paint_scene(
    paths: &[(Arc<Path>, Option<SvgPattern>, Option<load_svg::Stroke>)],
    fill_renderer: usize,
    stroke_renderer: usize,
    testing: bool,
    surface: &mut RenderPass,
    frame: &mut Frame,
    _instance: &mut Instance,
    renderers: &mut Renderers,
    patterns: &Patterns,
    transform: &LocalTransform,
) {
    if testing {
        let gradient = patterns.gradients.add(
            &mut frame.gpu_store,
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
            .transformed(&frame.transforms.get_current().matrix().to_untyped()),
        );

        renderers.tiling2.fill_surface(&mut surface.ctx(), &frame.transforms, gradient);
    }

    frame.transforms.push(transform);

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
                    &mut frame.gpu_store,
                    LinearGradient {
                        color0,
                        color1,
                        from,
                        to,
                    }
                    .transformed(&frame.transforms.get_current().matrix().to_untyped()),
                ),
            };

            let path = FilledPath::new(path.clone());
            renderers.fill(fill_renderer).fill_path(&mut surface.ctx(), &frame.transforms, path, pattern);
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
                    &mut frame.gpu_store,
                    LinearGradient {
                        color0,
                        color1,
                        from,
                        to,
                    }
                    .transformed(&frame.transforms.get_current().matrix().to_untyped()),
                ),
            };

            match stroke_renderer {
                crate::INSTANCED => {
                    renderers.msaa_strokes.stroke_path(&mut surface.ctx(), &frame.transforms, path.clone(), pattern, width);
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
                    renderers.fill(fill_renderer).fill_path(&mut surface.ctx(), &frame.transforms, path, pattern);
                }
                _ => {
                    unimplemented!();
                }
            }
        }
    }

    if testing {

        frame.transforms.set(&LocalToSurfaceTransform::rotation(Angle::radians(0.2)));
        let transform_handle = frame.transforms.get_current_gpu_handle(&mut frame.gpu_store);
        let gradient = patterns.gradients.add(
            &mut frame.gpu_store,
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
            &mut surface.ctx(),
            &frame.transforms,
            &LocalRect {
                min: point(200.0, 700.0),
                max: point(300.0, 900.0),
            },
            Aa::ALL,
            gradient,
            transform_handle,
        );
        renderers.rectangles.fill_rect(
            &mut surface.ctx(),
            &frame.transforms,
            &LocalRect {
                min: point(310.5, 700.5),
                max: point(410.5, 900.5),
            },
            Aa::LEFT | Aa::RIGHT | Aa::ALL,
            gradient,
            transform_handle,
        );
        frame.transforms.pop();

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
            &mut surface.ctx(),
            &frame.transforms,
            fill,//.inverted(),
            patterns.checkerboards.add(
                &mut frame.gpu_store,
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
                .transformed(&frame.transforms.get_current().matrix().to_untyped()),
            ).with_blend_mode(BlendMode::Screen),
        );

        frame.transforms.pop();

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
            renderers.tiling2.fill_path(&mut surface.ctx(), &frame.transforms, offset_path.clone(), patterns.colors.add(Color::RED));

            renderers.meshes.stroke_path(
                &mut surface.ctx(),
                &frame.transforms,
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
                &mut surface.ctx(),
                &frame.transforms,
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
            renderers.meshes.stroke_path(&mut surface.ctx(), &frame.transforms, b.build(), 1.0, patterns.colors.add(Color::BLACK));

            let green = patterns.colors.add(Color::GREEN);
            let blue = patterns.colors.add(Color::BLUE);
            let white = patterns.colors.add(Color::WHITE);
            for evt in offset_path.as_slice() {
                match evt {
                    PathEvent::Begin { at } => {
                        renderers.meshes.fill_circle(
                            &mut surface.ctx(),
                            &frame.transforms,
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
                            &mut surface.ctx(),
                            &frame.transforms,
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
                            &mut surface.ctx(),
                            &frame.transforms,
                            Circle {
                                center: ctrl.cast_unit(),
                                radius: 4.0,
                                inverted: false,
                            },
                            blue,
                        );
                        renderers.meshes.fill_circle(
                            &mut surface.ctx(),
                            &frame.transforms,
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
                            &mut surface.ctx(),
                            &frame.transforms,
                            Circle {
                                center: ctrl1.cast_unit(),
                                radius: 4.0,
                                inverted: false,
                            },
                            blue,
                        );
                        renderers.meshes.fill_circle(
                            &mut surface.ctx(),
                            &frame.transforms,
                            Circle {
                                center: ctrl2.cast_unit(),
                                radius: 4.0,
                                inverted: false,
                            },
                            blue,
                        );
                        renderers.meshes.fill_circle(
                            &mut surface.ctx(),
                            &frame.transforms,
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

    frame.transforms.push(&LocalTransform::translation(10.0, 1.0));
    frame.transforms.pop();
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

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct AttachmentId(u32);

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

pub mod counters {
    debug_overlay::declare_counters!(render = {
        batching: float = "batching" with { unit: "ms", safe_range: Some(0.0..4.0), color: (100, 150, 200, 255) },
        prepare: float = "prepare"   with { unit: "ms", safe_range: Some(0.0..6.0), color: (200, 150, 100, 255) },
        render: float = "render"     with { unit: "ms", safe_range: Some(0.0..4.0), color: (50, 50, 200, 255) },
        present: float = "present"   with { unit: "ms", safe_range: Some(0.0..12.0), color: (200, 200, 30, 255) },
        cpu_total: float = "total"   with { unit: "ms", safe_range: Some(0.0..16.0), color: (50, 250, 250, 255) },
        batches: int = "batches"
    });

    pub use debug_overlay::wgpu_counters as wgpu;
}


pub struct Bindings<'l> {
    pub graph: &'l BuiltGraph,
    pub external_inputs: &'l[Option<&'l wgpu::BindGroup>],
    pub external_attachments: &'l[Option<&'l wgpu::TextureView>],
    pub resources: &'l[GpuResource],
}

impl<'l> BindingResolver for Bindings<'l> {
    fn resolve_input(&self, binding: BindingsId) -> Option<&wgpu::BindGroup> {
        match binding.namespace() {
            BindingsNamespace::RenderGraph => {
                if let Some(id) = self.graph.resolve_binding(binding) {
                    let index = id.index as usize;
                    match id.allocation {
                        Allocation::Temporary => self.resources[index].as_input.as_ref(),
                        Allocation::External => self.external_inputs[index],
                    }
                } else {
                    None
                }
            }
            BindingsNamespace::External => {
                self.external_inputs[binding.index()]
            }
            _ => None
        }
    }

    fn resolve_attachment(&self, binding: BindingsId) -> Option<&wgpu::TextureView> {
        match binding.namespace() {
            BindingsNamespace::RenderGraph => {
                if let Some(id) = self.graph.resolve_binding(binding) {
                    let index = id.index as usize;
                    match id.allocation {
                        Allocation::Temporary => self.resources[index].as_attachment.as_ref(),
                        Allocation::External => self.external_attachments[index]
                    }
                } else {
                    return None;
                }
            }
            BindingsNamespace::External => {
                self.external_attachments[binding.index()]
            }
            _ => None
        }
    }
}
