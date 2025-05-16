use core::gpu::PipelineDefaults;
use core::wgpu;
use core::{
    gpu::{
        shader::GeometryId,
        Shaders, VertexBuilder,
    },
    resources::CommonGpuResources,
    context::RendererId,
};
use std::sync::Arc;

use crate::StencilAndCoverRenderer;

pub struct StencilAndCover {
    resources: Arc<StencilAndCoverResources>,
}

pub(crate) struct StencilAndCoverResources {
    pub stencil_pipeline: wgpu::RenderPipeline,
    pub msaa_stencil_pipeline: wgpu::RenderPipeline,
    pub cover_geometry: GeometryId,
}

const STENCIL_SHADER_SRC: &'static str = "
@group(0) @binding(0) var<uniform> render_target: RenderTarget;

#import render_target

@vertex fn vs_main(@location(0) position: vec2<f32>) -> @builtin(position) vec4<f32> {
    var pos = canvas_to_target(position);
    return vec4<f32>(pos.x, pos.y, 0.0, 1.0);
}

@fragment fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0);
}
";

impl StencilAndCover {
    pub fn new(
        _common: &mut CommonGpuResources,
        device: &wgpu::Device,
        shaders: &mut Shaders,
    ) -> Self {
        let stencil_module =
            shaders.create_shader_module(device, "stencil", STENCIL_SHADER_SRC, &[]);

        let vertex_attributes = VertexBuilder::from_slice(
            wgpu::VertexStepMode::Vertex,
            &[wgpu::VertexFormat::Float32x2],
        );

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("stencil"),
            bind_group_layouts: &[&shaders
                .get_bind_group_layout(shaders.common_bind_group_layouts.target_and_gpu_store)
                .handle],
            push_constant_ranges: &[],
        });

        let targets = &[Some(wgpu::ColorTargetState {
            format: shaders.defaults.color_format(),
            blend: None,
            write_mask: wgpu::ColorWrites::empty(),
        })];

        let mut descriptor = wgpu::RenderPipelineDescriptor {
            label: Some(&"stencil"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &stencil_module,
                entry_point: Some("vs_main"),
                buffers: &[vertex_attributes.buffer_layout()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &stencil_module,
                entry_point: Some("fs_main"),
                targets,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: PipelineDefaults::primitive_state(),
            depth_stencil: Some(wgpu::DepthStencilState {
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
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
                format: shaders.defaults.depth_stencil_format().unwrap(),
            }),
            multiview: None,
            multisample: wgpu::MultisampleState::default(),
            cache: None,
        };
        let stencil_pipeline = device.create_render_pipeline(&descriptor);

        descriptor.multisample = wgpu::MultisampleState {
            count: shaders.defaults.msaa_sample_count(),
            ..wgpu::MultisampleState::default()
        };
        let msaa_stencil_pipeline = device.create_render_pipeline(&descriptor);

        // TODO: this creates an implicit dependency to the mesh renderer.
        let cover_geometry = shaders.find_geometry("geometry::simple_mesh").unwrap();

        StencilAndCover {
            resources: Arc::new(StencilAndCoverResources {
                stencil_pipeline,
                msaa_stencil_pipeline,
                cover_geometry,
            })
        }
    }

    pub fn new_renderer(
        &self,
        renderer_id: RendererId,
    ) -> StencilAndCoverRenderer {
        StencilAndCoverRenderer::new(self.resources.clone(), renderer_id)
    }
}
