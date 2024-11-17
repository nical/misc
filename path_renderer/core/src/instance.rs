use std::time::{Duration, Instant};

use crate::worker::Workers;
use crate::{BindingResolver, BindingsId, BindingsNamespace, PrepareContext, Renderer, UploadContext, WgpuContext};
use crate::frame::Frame;
use crate::render_graph::{Allocation, BuiltGraph};
use crate::resources::{GpuResource, GpuResources};
use crate::gpu::{
    shader::{RenderPipelineBuilder, RenderPipelines},
    Shaders
};


pub struct Instance {
    pub shaders: Shaders,
    render_pipelines: RenderPipelines,
    pub resources: GpuResources,
    pub workers: Workers,
    next_frame_index: u32,
}

impl Instance {
    pub fn new(device: &wgpu::Device, num_workers: usize) -> Self {
        let mut shaders = Shaders::new(&device);
        let render_pipelines = RenderPipelines::new();
        let resources = GpuResources::new(
            &device,
            &mut shaders,
        );

        let workers = Workers::new(num_workers);

        Instance {
            shaders,
            render_pipelines,
            resources,
            workers,
            next_frame_index: 0,
        }
    }

    pub fn begin_frame(&mut self) -> Frame {
        let idx = self.next_frame_index;
        self.next_frame_index += 1;
        Frame::new(idx)
    }

    pub fn render(
        &mut self,
        frame: Frame,
        renderers: &mut[&mut dyn Renderer],
        external_inputs: &[Option<&wgpu::BindGroup>],
        external_attachments: &[Option<&wgpu::TextureView>],
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
    ) -> RenderStats {
        let mut stats = RenderStats {
            render_passes: 0,
            prepare_time_ms: 0.0,
            upload_time_ms: 0.0,
            render_time_ms: 0.0,
        };

        let graph = frame.graph.schedule().unwrap(); // TODO

        // TODO: does this need to be a separate step from upload?
        self.resources.begin_frame();

        let prepare_start = Instant::now();

        let mut prep_pipelines = self.render_pipelines.prepare();

        for cmd in &graph {
            let pass = &frame.built_render_passes[cmd.task_id().0 as usize];
            let Some(pass) = pass else { continue; };
            for renderer in renderers.iter_mut() {
                renderer.prepare(&mut PrepareContext {
                    pass,
                    transforms: &frame.transforms,
                    pipelines: &mut prep_pipelines,
                    workers: &self.workers,
                });
            }
        }

        let changes = prep_pipelines.finish();
        self.render_pipelines.build(
            &[&changes],
            &mut RenderPipelineBuilder(device, &mut self.shaders),
        );

        let upload_start = Instant::now();

        self.resources.upload(
            device,
            queue,
            &self.shaders,
            &graph.temporary_resources,
            &graph.pass_data,
        );

        for renderer in renderers.iter_mut() {
            renderer.upload(&mut UploadContext {
                resources: &mut self.resources,
                shaders: &self.shaders,
                wgpu: WgpuContext { device, queue },
            });
        }

        frame.gpu_store.upload(device, queue, &mut self.resources.common.gpu_store);

        self.resources.begin_rendering(encoder);

        let bindings = Bindings {
            graph: &graph,
            external_inputs,
            external_attachments,
            resources: &self.resources.graph.resources(),
        };

        let render_start = Instant::now();

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
                );
            }
        }

        self.resources.end_frame();

        fn ms(duration: Duration) -> f32 {
            (duration.as_micros() as f64 / 1000.0) as f32
        }

        stats.prepare_time_ms = ms(upload_start - prepare_start);
        stats.upload_time_ms = ms(render_start - upload_start);
        stats.render_time_ms = ms(Instant::now() - render_start);

        stats
    }
}

pub struct RenderStats {
    pub render_passes: u32,
    pub prepare_time_ms: f32,
    pub upload_time_ms: f32,
    pub render_time_ms: f32,
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
