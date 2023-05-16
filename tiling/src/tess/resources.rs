use crate::{gpu::{DynamicStore, ShaderSources, PipelineDefaults, VertexBuilder}, canvas::{CommonGpuResources, RendererResources}};

pub struct Pipelines {
    pub opaque_color: wgpu::RenderPipeline,
    pub alpha_color: wgpu::RenderPipeline,
    //pub opaque_image: wgpu::RenderPipeline,
    //pub alpha_image: wgpu::RenderPipeline,
}

pub struct MeshGpuResources {
    pub depth_pipelines: Option<Pipelines>,
    pub depth_msaa_pipelines: Option<Pipelines>,
    pub nodepth_pipelines: Option<Pipelines>,
    pub nodepth_msaa_pipelines: Option<Pipelines>,
}

impl MeshGpuResources {
    pub fn new(
        common: &CommonGpuResources,
        device: &wgpu::Device,
        shaders: &mut ShaderSources,
    ) -> Self {
        let src = include_str!("./../../shaders/tess/color.wgsl");
        let color_module = shaders.create_shader_module(device, "tess:color", src, &[]);

        let defaults = PipelineDefaults::new();
        let vertex_attributes = VertexBuilder::from_slice(
            wgpu::VertexStepMode::Vertex,
            &[
                wgpu::VertexFormat::Float32x2,
                wgpu::VertexFormat::Uint32,
                wgpu::VertexFormat::Uint32,
            ]
        );

        let color_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tess: color"),
            bind_group_layouts: &[
                &common.target_and_gpu_store_layout,
            ],
            push_constant_ranges: &[],
        });

        let primitive = PipelineDefaults::primitive_state();
        let no_multisample = wgpu::MultisampleState::default();

        let descriptor = wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&color_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &color_module,
                entry_point: "vs_main",
                buffers: &[vertex_attributes.buffer_layout()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &color_module,
                entry_point: "fs_main",
                targets: defaults.color_target_state_no_blend()
            }),
            primitive,
            depth_stencil: None,
            multiview: None,
            multisample: no_multisample,
        };

        let depth_read = wgpu::DepthStencilState {
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::GreaterEqual,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
            format: PipelineDefaults::depth_format(),
        };

        let depth_read_write = wgpu::DepthStencilState {
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::GreaterEqual,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
            format: PipelineDefaults::depth_format(),
        };

        let fragment_no_blend = Some(wgpu::FragmentState {
            module: &color_module,
            entry_point: "fs_main",
            targets: defaults.color_target_state_no_blend()
        });

        let fragment_blend = Some(wgpu::FragmentState {
            module: &color_module,
            entry_point: "fs_main",
            targets: defaults.color_target_state_no_blend()
        });

        let msaa_state = wgpu::MultisampleState {
            count: PipelineDefaults::msaa_sample_count(),
            .. wgpu::MultisampleState::default()
        };

        let opaque_color = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tess: opaque color"),
            multisample: no_multisample.clone(),
            depth_stencil: None,
            fragment: fragment_no_blend.clone(),
            .. descriptor.clone()
        });
        let alpha_color = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tess: alpha color"),
            multisample: no_multisample.clone(),
            depth_stencil: None,
            fragment: fragment_blend.clone(),
            .. descriptor.clone()
        });

        let msaa_opaque_color = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tess: msaa opaque color"),
            multisample: msaa_state.clone(),
            depth_stencil: None,
            fragment: fragment_no_blend.clone(),
            .. descriptor.clone()
        });
        let msaa_alpha_color = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tess: msaa alpha color"),
            multisample: msaa_state.clone(),
            depth_stencil: None,
            fragment: fragment_blend.clone(),
            .. descriptor.clone()
        });

        let depth_opaque_color = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tess: depth opaque color"),
            multisample: no_multisample.clone(),
            depth_stencil: Some(depth_read_write.clone()),
            fragment: fragment_no_blend.clone(),
            .. descriptor.clone()
        });
        let depth_alpha_color = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tess: depth alpha color"),
            multisample: no_multisample,
            depth_stencil: Some(depth_read.clone()),
            fragment: fragment_blend.clone(),
            .. descriptor.clone()
        });

        let depth_msaa_opaque_color = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tess: depth msaa opaque color"),
            multisample: msaa_state.clone(),
            depth_stencil: Some(depth_read_write.clone()),
            fragment: fragment_no_blend.clone(),
            .. descriptor.clone()
        });
        let depth_msaa_alpha_color = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tess: depth msaa alpha color"),
            multisample: msaa_state.clone(),
            depth_stencil: Some(depth_read.clone()),
            fragment: fragment_blend.clone(),
            .. descriptor.clone()
        });

        MeshGpuResources {
            depth_pipelines: Some(Pipelines {
                opaque_color: depth_opaque_color,
                alpha_color: depth_alpha_color,
            }),
            depth_msaa_pipelines: Some(Pipelines {
                opaque_color: depth_msaa_opaque_color,
                alpha_color: depth_msaa_alpha_color,
            }),
            nodepth_pipelines: Some(Pipelines {
                opaque_color: opaque_color,
                alpha_color: alpha_color,
            }),
            nodepth_msaa_pipelines: Some(Pipelines {
                opaque_color: msaa_opaque_color,
                alpha_color: msaa_alpha_color,
            }),
        }
    }

    pub fn pipelines(&self, with_depth_test: bool, with_msaa: bool) -> &Pipelines {
        match (with_depth_test, with_msaa) {
            (false, false) => &self.nodepth_pipelines,
            (false, true) => &self.nodepth_msaa_pipelines,
            (true, false) => &self.depth_pipelines,
            (true, true) => &self.depth_msaa_pipelines,
        }.as_ref().unwrap()
    }
}

impl RendererResources for MeshGpuResources {
    fn name(&self) -> &'static str { "MeshGpuResources" }

    fn begin_frame(&mut self) {
    }

    fn begin_rendering(&mut self, encoder: &mut wgpu::CommandEncoder) {
    }

    fn end_frame(&mut self) {
    }
}
