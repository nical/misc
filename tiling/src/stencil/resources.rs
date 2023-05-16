use crate::{canvas::{RendererResources, CommonGpuResources}, gpu::{PipelineDefaults, VertexBuilder, ShaderSources}};


pub struct StencilAndCoverResources {
    pub stencil_pipeline: wgpu::RenderPipeline,
    pub evenodd_color_pipeline: wgpu::RenderPipeline,
    pub msaa_stencil_pipeline: wgpu::RenderPipeline,
    pub msaa_evenodd_color_pipeline: wgpu::RenderPipeline,
}

impl StencilAndCoverResources {
    pub fn new(
        common: &CommonGpuResources,
        device: &wgpu::Device,
        shaders: &mut ShaderSources,
    ) -> Self {
        let stencil_src = include_str!("./../../shaders/stencil/stencil.wgsl");
        let cover_color_src = include_str!("./../../shaders/stencil/cover_color.wgsl");

        let stencil_module = shaders.create_shader_module(device, "stencil", stencil_src, &[]);
        let cover_module = shaders.create_shader_module(device, "stencil", cover_color_src, &[]);

        let defaults = PipelineDefaults::new();
        let vertex_attributes = VertexBuilder::from_slice(
            wgpu::VertexStepMode::Vertex,
            &[wgpu::VertexFormat::Float32x2],
        );

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("stencil"),
            bind_group_layouts: &[
                &common.target_and_gpu_store_layout,
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


        let vertex_attributes = VertexBuilder::from_slice(
            wgpu::VertexStepMode::Vertex,
            &[
                wgpu::VertexFormat::Float32x2,
                wgpu::VertexFormat::Uint32,
                wgpu::VertexFormat::Uint32,
            ]
        );

        let face_state = wgpu::StencilFaceState {
            compare: wgpu::CompareFunction::NotEqual,
            // reset the stencil buffer.
            fail_op: wgpu::StencilOperation::Replace,
            depth_fail_op: wgpu::StencilOperation::Replace,
            pass_op: wgpu::StencilOperation::Replace,
        };

        let mut descriptor = wgpu::RenderPipelineDescriptor {
            label: Some(&"stencil: color even-odd"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &cover_module,
                entry_point: "vs_main",
                buffers: &[vertex_attributes.buffer_layout()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &cover_module,
                entry_point: "fs_main",
                targets: defaults.color_target_state(),
            }),
            primitive: PipelineDefaults::primitive_state(),
            depth_stencil: Some(wgpu::DepthStencilState {
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::GreaterEqual,
                stencil: wgpu::StencilState {
                    front: face_state,
                    back: face_state,
                    read_mask: 1,
                    write_mask: 0xFFFFFFFF,
                },
                bias: wgpu::DepthBiasState::default(),
                format: PipelineDefaults::depth_format(),
            }),
            multiview: None,
            multisample: wgpu::MultisampleState::default(),
        };

        let evenodd_color_pipeline = device.create_render_pipeline(&descriptor);

        descriptor.multisample = wgpu::MultisampleState {
            count: PipelineDefaults::msaa_sample_count(),
            .. wgpu::MultisampleState::default()
        };
        let msaa_evenodd_color_pipeline = device.create_render_pipeline(&descriptor);

        StencilAndCoverResources {
            stencil_pipeline,
            evenodd_color_pipeline,
            msaa_stencil_pipeline,
            msaa_evenodd_color_pipeline,
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
