use crate::{canvas::{RendererResources, CommonGpuResources}, gpu::{PipelineDefaults, VertexBuilder, Shaders, shader::{OutputType, PipelineDescriptor, BlendMode, GeneratedPipelineId, ShaderMaskId}}};


pub struct StencilAndCoverResources {
    pub stencil_pipeline: wgpu::RenderPipeline,
    pub msaa_stencil_pipeline: wgpu::RenderPipeline,
    pub opaque_cover_pipeline: GeneratedPipelineId,
    pub alpha_cover_pipeline: GeneratedPipelineId,
}

impl StencilAndCoverResources {
    pub fn new(
        common: &mut CommonGpuResources,
        device: &wgpu::Device,
        shaders: &mut Shaders,
    ) -> Self {
        let stencil_src = include_str!("./../../shaders/stencil/stencil.wgsl");

        let stencil_module = shaders.create_shader_module(device, "stencil", stencil_src, &[]);

        let vertex_attributes = VertexBuilder::from_slice(
            wgpu::VertexStepMode::Vertex,
            &[wgpu::VertexFormat::Float32x2],
        );

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("stencil"),
            bind_group_layouts: &[
                &shaders.get_bind_group_layout(common.target_and_gpu_store_layout).handle,
            ],
            push_constant_ranges: &[],
        });

        let targets = &[Some(wgpu::ColorTargetState {
            format: PipelineDefaults::color_format(),
            blend: None,
            write_mask: wgpu::ColorWrites::empty(),
        })];

        let mut descriptor = wgpu::RenderPipelineDescriptor {
            label: Some(&"stencil"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &stencil_module,
                entry_point: "vs_main",
                buffers: &[vertex_attributes.buffer_layout()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &stencil_module,
                entry_point: "fs_main",
                targets,
            }),
            primitive: PipelineDefaults::primitive_state(),
            depth_stencil: Some(wgpu::DepthStencilState {
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::GreaterEqual,
                stencil: wgpu::StencilState {
                    front: wgpu::StencilFaceState {
                        compare: wgpu::CompareFunction::Always,
                        fail_op: wgpu::StencilOperation::Keep,
                        depth_fail_op: wgpu::StencilOperation::Keep,
                        pass_op: wgpu::StencilOperation::IncrementWrap,
                    },
                    back: wgpu::StencilFaceState {
                        compare: wgpu::CompareFunction::Always,
                        fail_op: wgpu::StencilOperation::Keep,
                        depth_fail_op: wgpu::StencilOperation::Keep,
                        pass_op: wgpu::StencilOperation::DecrementWrap,
                    },
                    read_mask: 0xFFFFFFFF,
                    write_mask: 0xFFFFFFFF,
                },
                bias: wgpu::DepthBiasState::default(),
                format: PipelineDefaults::depth_format(),
            }),
            multiview: None,
            multisample: wgpu::MultisampleState::default(),
        };
        let stencil_pipeline = device.create_render_pipeline(&descriptor);

        descriptor.multisample = wgpu::MultisampleState {
            count: PipelineDefaults::msaa_sample_count(),
            .. wgpu::MultisampleState::default()
        };
        let msaa_stencil_pipeline = device.create_render_pipeline(&descriptor);

        // TODO: this crates an implicit dependency to the mesh renderer.
        let cover_geom = shaders.find_geometry("geometry::simple_mesh").unwrap();

        // TODO: these pipelines happen to be identical to the mesh renderer's.
        let opaque_cover_pipeline = shaders.register_pipeline(PipelineDescriptor {
            label: "cover(opaque)",
            geometry: cover_geom,
            mask: ShaderMaskId::NONE,
            user_flags: 0,
            output: OutputType::Color,
            blend: BlendMode::None,
        });
        let alpha_cover_pipeline = shaders.register_pipeline(PipelineDescriptor {
            label: "cover(alpha)",
            geometry: cover_geom,
            mask: ShaderMaskId::NONE,
            user_flags: 0,
            output: OutputType::Color,
            blend: BlendMode::PremultipliedAlpha,
        });

        StencilAndCoverResources {
            stencil_pipeline,
            msaa_stencil_pipeline,
            opaque_cover_pipeline,
            alpha_cover_pipeline,
        }
    }
}

impl RendererResources for StencilAndCoverResources {
    fn name(&self) -> &'static str { "StencilAndCoverResources" }

    fn begin_frame(&mut self) {

    }

    fn end_frame(&mut self) {

    }
}
