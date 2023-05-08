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
    pub indices: DynamicStore,
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

        let primitive = defaults.primitive_state();
        let multisample = wgpu::MultisampleState::default();

        let opaque_color_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
            label: Some("tess: opque color"),
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
            depth_stencil: Some(wgpu::DepthStencilState {
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::GreaterEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
                format: defaults.depth_format(),
            }),
            multiview: None,
            multisample,
        };

        let alpha_color_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
            label: Some("tess: alpha color with depth"),
            layout: Some(&color_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &color_module,
                entry_point: "vs_main",
                buffers: &[vertex_attributes.buffer_layout()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &color_module,
                entry_point: "fs_main",
                targets: defaults.color_target_state()
            }),
            primitive,
            depth_stencil: Some(wgpu::DepthStencilState {
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::GreaterEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
                format: defaults.depth_format(),
            }),
            multiview: None,
            multisample,
        };

        let depth_pipelines = Some(Pipelines {
            opaque_color: device.create_render_pipeline(&opaque_color_pipeline_descriptor),
            alpha_color: device.create_render_pipeline(&alpha_color_pipeline_descriptor),
        });

        let opaque_color_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
            label: Some("tess: opaque color no depth"),
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
            multisample,
        };

        let alpha_color_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
            label: Some("tess: alpha color no depth"),
            layout: Some(&color_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &color_module,
                entry_point: "vs_main",
                buffers: &[vertex_attributes.buffer_layout()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &color_module,
                entry_point: "fs_main",
                targets: defaults.color_target_state()
            }),
            primitive,
            depth_stencil: None,
            multiview: None,
            multisample,
        };

        let nodepth_pipelines = Some(Pipelines {
            opaque_color: device.create_render_pipeline(&opaque_color_pipeline_descriptor),
            alpha_color: device.create_render_pipeline(&alpha_color_pipeline_descriptor),
        });

        MeshGpuResources {
            depth_pipelines,
            depth_msaa_pipelines: None,
            nodepth_pipelines,
            nodepth_msaa_pipelines: None,
            indices: DynamicStore::new(8192, wgpu::BufferUsages::INDEX, "Mesh:Index"),
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
        self.indices.unmap(encoder);
    }

    fn end_frame(&mut self) {
        self.indices.end_frame();
    }
}
