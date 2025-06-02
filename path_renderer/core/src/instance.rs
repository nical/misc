use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::gpu::{StagingBufferPool, UploadStats};
use crate::worker::Workers;
use crate::{BindingResolver, BindingsId, BindingsNamespace, PrepareContext, PrepareWorkerData, Renderer, RendererStats, UploadContext, WgpuContext};
use crate::frame::Frame;
use crate::graph::{Allocation, BuiltGraph, CommandContext};
use crate::resources::{GpuResource, GpuResources};
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
    next_frame_index: u32,
}

impl Instance {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, num_workers: usize) -> Self {
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

        Instance {
            shaders,
            render_pipelines,
            resources,
            workers,
            staging_buffers,
            device: device.clone(),
            queue: queue.clone(),
            next_frame_index: 0,
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

    pub fn render(
        &mut self,
        frame: Frame,
        renderers: &mut[&mut dyn Renderer],
        external_inputs: &[Option<&wgpu::BindGroup>],
        external_attachments: &[Option<&wgpu::TextureView>],
        encoder: &mut wgpu::CommandEncoder,
    ) -> RenderStats {
        let mut stats = RenderStats::default();
        stats.renderers = vec![RendererStats::default(); renderers.len()];

        let mut guard = frame.graph.inner.lock().unwrap();
        let graph = &mut *guard;
        let render_cmds = &mut graph.render_passes;
        let graph = graph.graph.schedule(render_cmds).unwrap(); // TODO

        // TODO: does this need to be a separate step from upload?
        self.resources.begin_frame();

        let prepare_start = Instant::now();

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

        for pass in render_cmds.passes() {
            let Some(pass) = pass else { continue; };
            for (idx, renderer) in renderers.iter_mut().enumerate() {
                let renderer_prepare_start = Instant::now();
                let stats = &mut stats.renderers[idx];
                renderer.prepare(&mut PrepareContext {
                    pass,
                    transforms: &frame.transforms,
                    workers: self.workers.ctx_with(&mut worker_data[..]),
                    staging_buffers: self.staging_buffers.clone(),
                });
                stats.prepare_time += ms(Instant::now() - renderer_prepare_start);
            }
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

        unsafe {
            let mut staging_buffers = self.staging_buffers.lock().unwrap();
            staging_buffers.unmap_active_buffers();
        }

        self.resources.upload(
            device,
            queue,
            &self.shaders,
            &graph.temporary_resources,
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

        for (idx, renderer) in renderers.iter_mut().enumerate() {
            let renderer_upload_start = Instant::now();
            let stats = &mut stats.renderers[idx];
            upload_stats += renderer.upload(&mut UploadContext {
                resources: &mut self.resources,
                shaders: &self.shaders,
                wgpu: WgpuContext { device, queue, encoder },
            });
            stats.upload_time = ms(Instant::now() - renderer_upload_start);
        }

        let render_start = Instant::now();

        self.resources.begin_rendering(encoder);

        let bindings = Bindings {
            graph: &graph,
            external_inputs,
            external_attachments,
            resources: &self.resources.graph.resources(),
        };

        // Cast `&mut [&mut dyn Renderer]` into `&[&dyn Renderer]`
        let const_renderers: &[&dyn Renderer] = unsafe {
            std::mem::transmute(renderers)
        };

        graph.commands.execute(&mut CommandContext {
            encoder,
            renderers: const_renderers,
            resources: &self.resources,
            bindings: &bindings,
            render_pipelines: &self.render_pipelines,
            stats: &mut stats,
        });

        self.resources.end_frame();

        stats.prepare_time_ms = ms(upload_start - prepare_start);
        stats.upload_time_ms = ms(render_start - upload_start);
        stats.render_time_ms = ms(Instant::now() - render_start);
        stats.draw_calls = stats.renderers.iter().map(|s| s.draw_calls).sum();
        stats.uploads_kb = upload_stats.bytes as f32 / 1000.0;
        stats.copy_ops = upload_stats.copy_ops;
        stats
    }

    /// Should be called after submitting the frame.
    pub fn end_frame(&mut self) {
        {
            let mut staging_buffers = self.staging_buffers.lock().unwrap();
            staging_buffers.triage_available_buffers();
            staging_buffers.recycle_active_buffers();
        }
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
    pub uploads_kb: f32,
    pub copy_ops: u32,
    pub staging_buffers: u32,
    pub stagin_buffer_chunks: u32,
}

pub struct Bindings<'l> {
    pub graph: &'l BuiltGraph<'l>,
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
