use lyon::math::Point;

use crate::gpu_store::{GpuStoreHandle, GpuStore};
use crate::tile_renderer::TileInstance;
use crate::{Color, TilePosition};
use crate::gpu::{ShaderSources, VertexBuilder, PipelineDefaults};

use crate::custom_pattern::*;

pub type CheckerboardPatternBuilder = CustomPatternBuilder<CheckerboardPattern>;
pub type CheckerboardPatternRenderer = CustomPatternRenderer<CheckerboardPattern>;

pub fn add_checkerboard(gpu_store: &mut GpuStore, color0: Color, color1: Color, offset: Point, scale: f32) -> CheckerboardPattern {
    let is_opaque = color0.is_opaque() && color1.is_opaque();
    let color0 = color0.to_f32();
    let color1 = color1.to_f32();
    let handle = gpu_store.push(&[
        color0[0], color0[1], color0[2], color0[3],
        color1[0], color1[1], color1[2], color1[3],
        offset.x, offset.y,
        scale,
    ]);

    CheckerboardPattern { handle, is_opaque }
}

pub struct CheckerboardPattern {
    handle: GpuStoreHandle,
    is_opaque: bool,
}

impl CheckerboardPattern {
    pub fn new() -> Self {
        CheckerboardPattern {
            handle: GpuStoreHandle::INVALID,
            is_opaque: false,
        }
    }
}

impl CustomPattern for CheckerboardPattern {
    type Instance = TileInstance;
    type RenderData = ();

    fn is_opaque(&self) -> bool { self.is_opaque }

    fn new_tile(&mut self, position: TilePosition, x: u32, y: u32) -> Self::Instance {
        TileInstance {
            position,
            mask: TilePosition::ZERO,
            pattern_data: [
                TilePosition::new(x, y).to_u32(),
                self.handle.to_u32(),
            ]
        }
    }

    fn new_renderer(
        device: &wgpu::Device,
        shaders: &mut ShaderSources,
        tile_atlas_desc_layout: &wgpu::BindGroupLayout,
    ) -> CustomPatternRenderer<Self> {
        let label = &"Checkerboard";

        let src = include_str!("../shaders/checkerboard_pattern.wgsl");
        let module = shaders.create_shader_module(device, label, src, &[]);

        let defaults = PipelineDefaults::new();
        let attributes = VertexBuilder::from_slice(&[
            wgpu::VertexFormat::Uint32x4,
        ]);

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[&tile_atlas_desc_layout],
            push_constant_ranges: &[],
        });
        let pipeline_descriptor = wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: "vs_main",
                buffers: &[attributes.buffer_layout()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: "fs_main",
                targets: defaults.color_target_state_no_blend(),
            }),
            primitive: defaults.primitive_state(),
            depth_stencil: None,
            multiview: None,
            multisample: wgpu::MultisampleState::default(),
        };

        let pipeline = device.create_render_pipeline(&pipeline_descriptor);

        CustomPatternRenderer::new(label, device, pipeline, ())
    }

    fn set_render_pass_state<'a, 'b: 'a>(_: &'a Self::RenderData, _: &mut wgpu::RenderPass<'b>) {}
}
