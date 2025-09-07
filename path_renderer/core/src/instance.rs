use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::gpu::{GpuBuffer, GpuStreams, StagingBufferPool, UploadStats};
use crate::graph::render_nodes::RenderNodes;
use crate::transform::Transforms;
use crate::worker::Workers;
use crate::{BindingResolver, BindingsId, BindingsNamespace, PrepareContext, PrepareWorkerData, Renderer, RendererStats, UploadContext, WgpuContext};
use crate::graph::{Allocation, GraphBindings, PassId, PassRenderContext};
use crate::resources::{GpuResource, GpuResources};
use crate::shading::{RenderPipelineBuilder, Shaders, RenderPipelines};
use crate::graph::GraphSystem;

pub fn ms(duration: Duration) -> f32 {
    (duration.as_micros() as f64 / 1000.0) as f32
}

pub struct Instance {
    pub shaders: Shaders,
    render_pipelines: RenderPipelines,
    pub resources: GpuResources,
    pub workers: Workers,
    pub staging_buffers: Arc<Mutex<StagingBufferPool>>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub gpu_profiler: wgpu_profiler::GpuProfiler,
    pub gpu_profiling_enabled: bool,
    next_frame_index: u32,
}

impl Instance {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, num_workers: usize, gpu_profiling_enabled: bool) -> Self {
        let staging_buffers = unsafe {
            Arc::new(Mutex::new(StagingBufferPool::new(1024 * 64, device.clone())))
        };

        let shaders = Shaders::new(&device);
        let render_pipelines = RenderPipelines::new();
        let resources = GpuResources::new(
            &device,
            staging_buffers.clone(),
        );

        let workers = Workers::new(num_workers);

        let gpu_profiler = wgpu_profiler::GpuProfiler::new(device, wgpu_profiler::GpuProfilerSettings {
            enable_timer_queries: gpu_profiling_enabled,
            enable_debug_groups: gpu_profiling_enabled,
            max_num_pending_frames: 4,
        }).unwrap();

        Instance {
            shaders,
            render_pipelines,
            resources,
            workers,
            staging_buffers,
            device: device.clone(),
            queue: queue.clone(),
            next_frame_index: 0,
            gpu_profiler,
            gpu_profiling_enabled,
        }
    }

    pub fn begin_frame(&mut self) -> Frame {
        let idx = self.next_frame_index;
        self.next_frame_index += 1;
        Frame::new(
            idx,
            self.resources.common.f32_buffer.begin_frame(
                self.staging_buffers.clone()
            ),
            self.resources.common.u32_buffer.begin_frame(
                self.staging_buffers.clone()
            ),
            self.resources.common.vertices.begin_frame(
                self.staging_buffers.clone()
            ),
            self.resources.common.indices.begin_frame(
                self.staging_buffers.clone()
            ),
            self.resources.common.instances.begin_frame(
                self.staging_buffers.clone()
            ),
        )
    }

    // TODO: reduce the number of arguments
    pub fn render(
        &mut self,
        frame: Frame,
        commands: &[PassId],
        // TODO: ideally we could remove this one, but we need some way
        // for the prepare phase to iterate over the batches per render
        // pass (to be able to do per-pass things like occlusion culling)
        render_nodes: &RenderNodes,
        graph_bindings: &GraphBindings,
        renderers: &mut[&mut dyn Renderer],
        external_inputs: &[Option<&wgpu::BindGroup>],
        external_attachments: &[Option<&wgpu::TextureView>],
        encoder: &mut wgpu::CommandEncoder,
    ) -> RenderStats {

        let mut prep = self.prepare_phase(frame, graph_bindings, encoder);

        let (mut ctx, stats) = prep.ctx();
        render_nodes.prepare(&mut ctx, renderers, &mut stats.renderers);

        let mut upload = prep.next();
        let (mut ctx, stats) = upload.ctx();

        for (idx, renderer) in renderers.iter_mut().enumerate() {
            let renderer_upload_start = Instant::now();
            stats.uploads += renderer.upload(&mut ctx);
            let stats = &mut stats.renderers[idx];
            stats.upload_time = ms(Instant::now() - renderer_upload_start);
        }

        let render = upload.next();

        // TODO

        let mut stats = render.next();

        let render_start = Instant::now();

        self.resources.begin_rendering(encoder);

        let bindings = Bindings {
            graph: &graph_bindings,
            external_inputs,
            external_attachments,
            resources: &self.resources.graph.resources(),
        };

        // Cast `&mut [&mut dyn Renderer]` into `&[&dyn Renderer]`
        let const_renderers: &[&dyn Renderer] = unsafe {
            std::mem::transmute(renderers)
        };

        let mut ctx = PassRenderContext {
            encoder,
            renderers: const_renderers,
            resources: &self.resources,
            bindings: &bindings,
            render_pipelines: &self.render_pipelines,
            gpu_profiler: &mut self.gpu_profiler,
            stats: &mut stats,
        };

        for pass in commands {
            render_nodes.render(&mut ctx, *pass)
        }

        if self.gpu_profiling_enabled {
            self.gpu_profiler.resolve_queries(encoder);
        }

        stats.render_time_ms = ms(Instant::now() - render_start);

        stats
    }

    fn prepare_phase<'l>(
        &'l mut self,
        frame: Frame,
        graph_bindings: &'l GraphBindings,
        encoder: &'l mut wgpu::CommandEncoder
    ) -> PreparePhase<'l> {
        let mut stats = RenderStats::default();

        let num_renderers = 8; // TODO
        stats.renderers = vec![RendererStats::default(); num_renderers];

        // TODO: does this need to be a separate step from upload?
        self.resources.begin_frame();

        let prepare_start_time = Instant::now();

        let num_workers = self.workers.num_workers();
        let mut worker_data = Vec::with_capacity(num_workers);
        for _ in 0..(num_workers - 1) {
            worker_data.push(PrepareWorkerData {
                pipelines: self.render_pipelines.prepare(),
                f32_buffer: frame.f32_buffer.clone(),
                u32_buffer: frame.u32_buffer.clone(),
                vertices: frame.vertices.clone(),
                indices: frame.indices.clone(),
                instances: frame.instances.clone(),
            });
        }
        worker_data.push(PrepareWorkerData {
            pipelines: self.render_pipelines.prepare(),
            f32_buffer: frame.f32_buffer,
            u32_buffer: frame.u32_buffer,
            vertices: frame.vertices,
            indices: frame.indices,
            instances: frame.instances,
        });

        PreparePhase {
            instance: self,
            transforms: frame.transforms,
            worker_data,
            encoder,
            graph_bindings,
            prepare_start_time,
            stats,
        }
    }

    /// Should be called after submitting the frame.
    pub fn end_frame(&mut self) {
        self.resources.end_frame();
        {
            let mut staging_buffers = self.staging_buffers.lock().unwrap();
            staging_buffers.triage_available_buffers();
            staging_buffers.recycle_active_buffers();
        }
        if self.gpu_profiling_enabled {
            match self.gpu_profiler.end_frame() {
                Ok(_) => {
                    let timestamp_period = self.queue.get_timestamp_period();
                    let results = self.gpu_profiler.process_finished_frame(timestamp_period);

                    if let Some(queries) = &results {
                        println!("-------- GPU timestamp queries --------");
                        for query in queries {
                            if let Some(time) = &query.time {
                                println!("{}: {:.4}ms", query.label, (time.end - time.start) * 1000.0);
                            } else {
                                println!("{}: --", query.label);
                            }
                        }
                    }
                }
                Err(e) => {
                    println!("Gpu profiler error: {e:?}");
                }
            }
        }
    }
}

pub struct Frame {
    pub f32_buffer: GpuBuffer,
    pub u32_buffer: GpuBuffer,
    pub vertices: GpuBuffer,
    pub indices: GpuStreams,
    pub instances: GpuStreams,
    pub transforms: Transforms,
    // pub allocator: FrameAllocator, // TODO
    index: u32,
}

impl Frame {
    pub(crate) fn new(
        index: u32,
        f32_buffer: GpuBuffer,
        u32_buffer: GpuBuffer,
        vertices: GpuBuffer,
        indices: GpuStreams,
        instances: GpuStreams,
    ) -> Self {
        Frame {
            f32_buffer,
            u32_buffer,
            vertices,
            indices,
            instances,
            transforms: Transforms::new(),
            index,
        }
    }

    pub fn index(&self) -> u32 {
        self.index
    }
}

#[derive(Clone, Default, Debug)]
pub struct RenderStats {
    pub renderers: Vec<RendererStats>,
    pub render_passes: u32,
    pub draw_calls: u32,
    pub prepare_time_ms: f32,
    pub upload_time_ms: f32,
    pub render_time_ms: f32,
    pub uploads: UploadStats,
    pub staging_buffers: u32,
    pub stagin_buffer_chunks: u32,
}

pub struct Bindings<'l> {
    graph: &'l GraphBindings,
    external_inputs: &'l[Option<&'l wgpu::BindGroup>],
    external_attachments: &'l[Option<&'l wgpu::TextureView>],
    resources: &'l[GpuResource],
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

pub struct PreparePhase<'l> {
    instance: &'l mut Instance,
    transforms: Transforms,
    worker_data: Vec<PrepareWorkerData>,
    encoder: &'l mut wgpu::CommandEncoder,
    graph_bindings: &'l GraphBindings,
    prepare_start_time: Instant,
    stats: RenderStats,
}

impl<'l> PreparePhase<'l> {
    pub fn ctx(&mut self) -> (PrepareContext, &mut RenderStats) {
        (
            PrepareContext {
                transforms: &self.transforms,
                workers: self.instance.workers.ctx_with(&mut self.worker_data[..]),
                staging_buffers: self.instance.staging_buffers.clone(),
            },
            &mut self.stats,
        )
    }

    pub fn next(mut self) -> UploadPhase<'l> {
        let mut f32_buffer_ops = Vec::with_capacity(self.worker_data.len());
        let mut u32_buffer_ops = Vec::with_capacity(self.worker_data.len());
        let mut vtx_ops = Vec::with_capacity(self.worker_data.len());
        let mut idx_ops = Vec::with_capacity(self.worker_data.len());
        let mut inst_ops = Vec::with_capacity(self.worker_data.len());
        let mut pipeline_changes = Vec::with_capacity(self.worker_data.len());
        for mut wd in self.worker_data {
            f32_buffer_ops.push(wd.f32_buffer.finish());
            u32_buffer_ops.push(wd.u32_buffer.finish());
            vtx_ops.push(wd.vertices.finish());
            idx_ops.push(wd.indices.finish());
            inst_ops.push(wd.instances.finish());
            pipeline_changes.push(wd.pipelines.finish());
        }

        let device = &self.instance.device;
        let queue = &self.instance.queue;

        self.instance.render_pipelines.build(
            &pipeline_changes[..],
            &mut RenderPipelineBuilder(device, &mut self.instance.shaders),
        );

        let upload_start = Instant::now();

        unsafe {
            let mut staging_buffers = self.instance.staging_buffers.lock().unwrap();
            staging_buffers.unmap_active_buffers();
        }

        self.instance.resources.upload(
            device,
            queue,
            &self.instance.shaders,
            &self.graph_bindings.temporary_resources,
        );

        let mut upload_stats = UploadStats::default();
        {
            let staging_buffers = self.instance.staging_buffers.lock().unwrap();
            upload_stats += self.instance.resources.common.f32_buffer.upload(
                &f32_buffer_ops,
                &staging_buffers,
                &self.instance.device,
                self.encoder,
            );
            upload_stats += self.instance.resources.common.u32_buffer.upload(
                &u32_buffer_ops,
                &staging_buffers,
                &self.instance.device,
                self.encoder,
            );
            upload_stats += self.instance.resources.common.vertices.upload(
                &vtx_ops,
                &staging_buffers,
                &self.instance.device,
                self.encoder,
            );
            upload_stats += self.instance.resources.common.indices.upload(
                &idx_ops,
                &staging_buffers,
                &self.instance.device,
                self.encoder,
            );
            upload_stats += self.instance.resources.common.instances.upload(
                &inst_ops,
                &staging_buffers,
                &self.instance.device,
                self.encoder,
            );
            self.stats.staging_buffers = staging_buffers.active_staging_buffer_count();
        }

        self.stats.prepare_time_ms = ms(upload_start - self.prepare_start_time);

        UploadPhase {
            instance: self.instance,
            encoder: self.encoder,
            graph_bindings: self.graph_bindings,
            stats: self.stats,
            upload_start_time: upload_start,
        }
    }
}

pub struct UploadPhase<'l> {
    instance: &'l mut Instance,
    encoder: &'l mut wgpu::CommandEncoder,
    graph_bindings: &'l GraphBindings,
    stats: RenderStats,
    upload_start_time: Instant,
}

impl<'l> UploadPhase<'l> {
    pub fn ctx(&mut self) -> (UploadContext, &mut RenderStats) {
        (
            UploadContext {
                resources: &mut self.instance.resources,
                shaders: &self.instance.shaders,
                wgpu: WgpuContext {
                    device: &self.instance.device,
                    queue: &self.instance.queue,
                    encoder: self.encoder,
                },
            },
            &mut self.stats,
        )
    }

    pub fn next(mut self) -> RenderPhase<'l> {

        self.stats.draw_calls = self.stats.renderers.iter().map(|s| s.draw_calls).sum();

        let render_start_time = Instant::now();
        self.stats.upload_time_ms = ms(render_start_time - self.upload_start_time);

        RenderPhase {
            instance: self.instance,
            encoder: self.encoder,
            stats: self.stats,
            render_start_time,
        }
    }
}

// TODO
pub struct RenderPhase<'l> {
    instance: &'l mut Instance,
    encoder: &'l mut wgpu::CommandEncoder,
    stats: RenderStats,
    render_start_time: Instant,
}

impl<'l> RenderPhase<'l> {
    pub fn next(self) -> RenderStats {
        self.stats
    }
}
