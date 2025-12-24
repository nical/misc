use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::gpu::{GpuBuffer, GpuStreams, StagingBufferPool, UploadStats};
use crate::render_pass::{BuiltRenderPass, PassRenderContext};
use crate::transform::Transforms;
use crate::worker::Workers;
use crate::{BindingResolver, BindingsId, BindingsNamespace, PrepareContext, PrepareWorkerData, Renderer, RendererStats, UploadContext, WgpuContext};
use crate::resources::{GpuResource, GpuResources, ResourceKey, Allocation, ResourceIndex};
use crate::shading::{RenderPipelineBuilder, Shaders, RenderPipelines};

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

    pub fn render_frame(
        &mut self,
        frame: Frame,
        external_inputs: &[Option<&wgpu::BindGroup>],
        external_attachments: &[Option<&wgpu::TextureView>],
        encoder: &mut wgpu::CommandEncoder,
        renderers: &mut [&mut dyn Renderer],
    ) -> RenderStats {
        let mut stats = RenderStats::new(renderers.len());

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

        let mut ctx = PrepareContext {
            transforms: &frame.transforms,
            workers: self.workers.ctx_with(&mut worker_data[..]),
            staging_buffers: self.staging_buffers.clone(),
        };

        let mut start = Instant::now();
        for (idx, renderer) in renderers.iter_mut().enumerate() {
            renderer.prepare(&mut ctx, &frame.passes.render_passes);
            let end = Instant::now();
            stats.renderers[idx].prepare_time += ms(end - start);
            start = end;
        }

        let mut f32_buffer_ops = Vec::with_capacity(worker_data.len());
        let mut u32_buffer_ops = Vec::with_capacity(worker_data.len());
        let mut vtx_ops = Vec::with_capacity(worker_data.len());
        let mut idx_ops = Vec::with_capacity(worker_data.len());
        let mut inst_ops = Vec::with_capacity(worker_data.len());
        let mut pipeline_changes = Vec::with_capacity(worker_data.len());
        for mut wd in worker_data {
            f32_buffer_ops.push(wd.f32_buffer.finish());
            u32_buffer_ops.push(wd.u32_buffer.finish());
            vtx_ops.push(wd.vertices.finish());
            idx_ops.push(wd.indices.finish());
            inst_ops.push(wd.instances.finish());
            pipeline_changes.push(wd.pipelines.finish());
        }

        let device = &self.device;
        let queue = &self.queue;

        self.render_pipelines.build(
            &pipeline_changes[..],
            &mut RenderPipelineBuilder(device, &mut self.shaders),
        );

        let upload_start = Instant::now();
        stats.prepare_time_ms = ms(upload_start - prepare_start_time);

        unsafe {
            let mut staging_buffers = self.staging_buffers.lock().unwrap();
            staging_buffers.unmap_active_buffers();
        }

        self.resources.upload(
            device,
            queue,
            &self.shaders,
            frame.passes.resources.descriptors(),
        );

        let mut upload_stats = UploadStats::default();
        {
            let staging_buffers = self.staging_buffers.lock().unwrap();
            upload_stats += self.resources.common.f32_buffer.upload(
                &f32_buffer_ops,
                &staging_buffers,
                &self.device,
                encoder,
            );
            upload_stats += self.resources.common.u32_buffer.upload(
                &u32_buffer_ops,
                &staging_buffers,
                &self.device,
                encoder,
            );
            upload_stats += self.resources.common.vertices.upload(
                &vtx_ops,
                &staging_buffers,
                &self.device,
                encoder,
            );
            upload_stats += self.resources.common.indices.upload(
                &idx_ops,
                &staging_buffers,
                &self.device,
                encoder,
            );
            upload_stats += self.resources.common.instances.upload(
                &inst_ops,
                &staging_buffers,
                &self.device,
                encoder,
            );
            stats.staging_buffers = staging_buffers.active_staging_buffer_count();
        }

        let mut ctx = UploadContext {
            resources: &mut self.resources,
            shaders: &self.shaders,
            wgpu: WgpuContext {
                device: &self.device,
                queue: &self.queue,
                encoder,
            },
        };

        for (idx, renderer) in renderers.iter_mut().enumerate() {
            let renderer_upload_start = Instant::now();
            stats.uploads += renderer.upload(&mut ctx);
            let stats = &mut stats.renderers[idx];
            stats.upload_time = ms(Instant::now() - renderer_upload_start);
        }

        let render_start_time = Instant::now();
        stats.upload_time_ms = ms(render_start_time - upload_start);

        self.resources.begin_rendering(encoder);

        let bindings = Bindings {
            graph: &frame.passes.bindings,
            external_inputs,
            external_attachments,
            resources: self.resources.temp.resources()
        };

        let mut ctx = PassRenderContext {
            encoder,
            resources: &self.resources,
            bindings: &bindings,
            render_pipelines: &self.render_pipelines,
            gpu_profiler: &mut self.gpu_profiler,
            stats: &mut stats.renderers,
        };

        for pass_id in &frame.passes.ordered_passes {
            match *pass_id {
                PassId::Render(index) => {
                    let pass = &frame.passes.render_passes[index as usize];
                    pass.render(&mut ctx, renderers);
                }
            }
        }

        if self.gpu_profiling_enabled {
            self.gpu_profiler.resolve_queries(encoder);
        }

        stats.render_passes = frame.passes.render_passes.len() as u32;
        stats.draw_calls = stats.renderers.iter().map(|s| s.draw_calls).sum();

        stats.render_time_ms = ms(Instant::now() - render_start_time);

        stats
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
    pub passes: Passes,
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
            passes: Passes::new(),
            index,
        }
    }

    pub fn index(&self) -> u32 {
        self.index
    }
}

#[derive(Clone, Debug)]
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

impl RenderStats {
    pub fn new(num_renderers: usize) -> Self {
        RenderStats {
            renderers: vec![RendererStats::default(); num_renderers],
            render_passes: 0,
            draw_calls: 0,
            prepare_time_ms: 0.0,
            upload_time_ms: 0.0,
            render_time_ms: 0.0,
            uploads: UploadStats { bytes: 0, copy_ops: 0 },
            staging_buffers: 0,
            stagin_buffer_chunks: 0
        }
    }
}

pub struct Bindings<'l> {
    graph: &'l [Option<ResourceIndex>],
    external_inputs: &'l[Option<&'l wgpu::BindGroup>],
    external_attachments: &'l[Option<&'l wgpu::TextureView>],
    resources: &'l[GpuResource],
}

impl<'l> BindingResolver for Bindings<'l> {
    // TODO: return a Result with a helpful error message.
    fn resolve_input(&self, binding: BindingsId) -> Option<&wgpu::BindGroup> {
        match binding.namespace() {
            BindingsNamespace::RenderGraph => {
                if let Some(id) = self.graph[binding.index()] {
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
                if let Some(id) = self.graph[binding.index()] {
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


#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
enum PassId {
    Render(u16),
}

pub struct Passes {
    render_passes: Vec<BuiltRenderPass>,
    // TODO: compute passes, transfer passes, etc.
    ordered_passes: Vec<PassId>,

    // TODO: SHould the two below be in a separate struct?
    pub resources: TemporaryResources,
    bindings: Vec<Option<ResourceIndex>>,
}

impl Passes {
    pub fn new() -> Self {
        Passes {
            render_passes: Vec::with_capacity(32),
            resources: TemporaryResources::new(),
            bindings: Vec::with_capacity(32),
            ordered_passes: Vec::with_capacity(32),
        }
    }

    pub fn push_render_pass(&mut self, pass: BuiltRenderPass) {
        let index = self.render_passes.len() as u16;
        self.render_passes.push(pass);
        self.ordered_passes.push(PassId::Render(index));
    }

    pub fn set_binding(&mut self, idx: usize, binding: Option<ResourceIndex>) {
        while self.bindings.len() <= idx {
            self.bindings.push(None)
        }

        debug_assert!(self.bindings[idx].is_none());
        self.bindings[idx] = binding;
    }
}

pub struct TemporaryResources {
    available: Vec<(ResourceKey, Vec<u16>)>,
    resources: Vec<ResourceKey>,
}

impl TemporaryResources {
    fn new() -> Self {
        TemporaryResources {
            available: Vec::with_capacity(32),
            resources: Vec::with_capacity(32),
        }
    }

    pub fn get(&mut self, descriptor: ResourceKey) -> u16 {
        let mut resource = None;
        for (desc, resources) in &mut self.available {
            if *desc == descriptor {
                resource = resources.pop();
                break;
            }
        }

        resource.unwrap_or_else(|| {
            let res = self.resources.len() as u16;
            self.resources.push(descriptor);
            res
        })
    }

    pub fn recycle(&mut self, descriptor: ResourceKey, index: u16) {
        let mut pool_idx = self.available.len();
        for (idx, (desc, _)) in self.available.iter().enumerate() {
            if *desc == descriptor {
                pool_idx = idx;
                break;
            }
        }
        if pool_idx == self.available.len() {
            self.available.push((descriptor, Vec::with_capacity(8)));
        }

        self.available[pool_idx].1.push(index);
    }

    pub fn descriptors(&self) -> &[ResourceKey] {
        &self.resources
    }
}
