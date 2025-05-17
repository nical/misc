use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::gpu::{StagingBufferPool, UploadStats};
use crate::worker::Workers;
use crate::{BindingResolver, BindingsId, BindingsNamespace, PrepareContext, PrepareWorkerData, Renderer, RendererStats, UploadContext, WgpuContext};
use crate::frame::Frame;
use crate::graph::{Allocation, BuiltGraph};
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

        let mut shaders = Shaders::new(&device);
        let render_pipelines = RenderPipelines::new();
        let resources = GpuResources::new(
            &device,
            &mut shaders,
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
            self.resources.common.gpu_store.begin_frame(
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

        let graph = frame.graph.schedule().unwrap(); // TODO

        // TODO: does this need to be a separate step from upload?
        self.resources.begin_frame();

        let prepare_start = Instant::now();

        let num_workers = self.workers.num_workers();
        let mut worker_data = Vec::with_capacity(num_workers);
        for _ in 0..(num_workers - 1) {
            worker_data.push(PrepareWorkerData {
                pipelines: self.render_pipelines.prepare(),
                gpu_store: frame.gpu_store.clone(),
                vertices: frame.vertices.clone(),
                indices: frame.indices.clone(),
                instances: frame.instances.clone(),
            });
        }
        worker_data.push(PrepareWorkerData {
            pipelines: self.render_pipelines.prepare(),
            gpu_store: frame.gpu_store,
            vertices: frame.vertices,
            indices: frame.indices,
            instances: frame.instances,
        });

        for cmd in &graph {
            let pass = &frame.built_render_passes[cmd.task_id().0 as usize];
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

        let mut gpu_store_ops = Vec::with_capacity(worker_data.len());
        let mut vtx_ops = Vec::with_capacity(worker_data.len());
        let mut idx_ops = Vec::with_capacity(worker_data.len());
        let mut inst_ops = Vec::with_capacity(worker_data.len());
        let mut pipeline_changes = Vec::with_capacity(worker_data.len());
        for mut wd in worker_data {
            gpu_store_ops.push(wd.gpu_store.finish());
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
            &graph.pass_data,
        );

        let mut upload_stats = UploadStats::default();
        {
            let staging_buffers = self.staging_buffers.lock().unwrap();
            upload_stats += self.resources.common.gpu_store.upload(
                &gpu_store_ops,
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

        let mut attachments = Vec::new();
        for command in &graph {
            if let Some(command) = command.as_render_command() {
                let Some(built_pass) = &frame.built_render_passes[command.task_id().0 as usize] else {
                    continue;
                };
                if built_pass.is_empty() {
                    continue;
                }

                stats.render_passes += 1;

                attachments.clear();
                for item in command.color_attachments() {
                    let Some(color_attachment) = item.attachment else {
                        attachments.push(None);
                        continue;
                    };

                    let view = bindings.resolve_attachment(color_attachment.binding()).unwrap();

                    let resolve_target = item.resolve_target.and_then(|attachment| {
                        bindings.resolve_attachment(attachment.binding())
                    });

                    attachments.push(Some(wgpu::RenderPassColorAttachment {
                        view,
                        resolve_target,
                        ops: wgpu::Operations {
                            load: if color_attachment.load() {
                                wgpu::LoadOp::Load
                            } else {
                                wgpu::LoadOp::Clear(wgpu::Color::BLACK)
                            },
                            store: if color_attachment.store() {
                                wgpu::StoreOp::Store
                            } else {
                                wgpu::StoreOp::Discard
                            }
                        }
                    }));
                }

                let depth_stencil_attachment = command.depth_stencil_attachment().map(|attachment| {
                    let view = bindings.resolve_attachment(attachment.binding()).unwrap();

                    wgpu::RenderPassDepthStencilAttachment {
                        view,
                        depth_ops: if attachment.depth() {
                            Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(0.0),
                                store: wgpu::StoreOp::Discard,
                            })
                        } else {
                            None
                        },
                        stencil_ops: if attachment.stencil() {
                            Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(128),
                                store: wgpu::StoreOp::Discard,
                            })
                        } else {
                            None
                        },
                    }
                });

                let pass_descriptor = &wgpu::RenderPassDescriptor {
                    label: command.label(),
                    color_attachments: &attachments,
                    depth_stencil_attachment,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                };

                let mut wgpu_pass = encoder.begin_render_pass(&pass_descriptor);

                // Cast `&mut [&mut dyn Renderer]` into `&[&dyn Renderer]`
                let const_renderers: &&[&dyn Renderer] = unsafe {
                    std::mem::transmute(&renderers)
                };

                built_pass.encode(
                    command.pass_data_index(),
                    *const_renderers,
                    &self.resources,
                    &bindings,
                    &self.render_pipelines,
                    &mut wgpu_pass,
                    &mut stats,
                );
            }
        }

        self.resources.end_frame();

        stats.prepare_time_ms = ms(upload_start - prepare_start);
        stats.upload_time_ms = ms(render_start - upload_start);
        stats.render_time_ms = ms(Instant::now() - render_start);
        stats.draw_calls = stats.renderers.iter().map(|s| s.draw_calls).sum();
        stats.uploads_kb = upload_stats.bytes as f32 / 1000.0;
        stats.copy_ops = upload_stats.copy_ops;
        stats
    }

    /// should be called after submitting the frame.
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
