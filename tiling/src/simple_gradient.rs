use crate::gpu_store::{GpuStore, GpuStoreHandle};
use crate::tile_renderer::TileInstance;
use crate::{Color, Point, TilePosition};
use crate::gpu::{ShaderSources, VertexBuilder, PipelineDefaults};

use crate::custom_pattern::*;

pub type SimpleGradientBuilder = CustomPatternBuilder<SimpleGradient>;
pub type SimpleGradientRenderer = CustomPatternRenderer<SimpleGradient>;

pub fn add_gradient(store: &mut GpuStore, p0: Point, color0: Color, p1: Point, color1: Color) -> SimpleGradient {
    let is_opaque = color0.is_opaque() && color1.is_opaque();
    let color0 = color0.to_f32();
    let color1 = color1.to_f32();

    let handle = store.push(&[
        p0.x, p0.y, p1.x, p1.y,
        color0[0], color0[1], color0[2], color0[3],
        color1[0], color1[1], color1[2], color1[3],
    ]);

    SimpleGradient {
        handle,
        is_opaque,
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Stop {
    pub position: Point,
    pub color: Color,
}
pub struct SimpleGradient {
    handle: GpuStoreHandle,
    is_opaque: bool,
}

impl SimpleGradient {
    pub fn new() -> Self {
        SimpleGradient {
            handle: GpuStoreHandle::INVALID,
            is_opaque: false,
        }
    }
}

impl CustomPattern for SimpleGradient {
    type Instance = TileInstance;
    type RenderData = ();

    fn is_opaque(&self) -> bool { self.is_opaque }

    fn new_tile(&mut self, atlas_tile_id: TilePosition, x: u32, y: u32) -> Self::Instance {
        let pattern_position = TilePosition::new(x, y);

        TileInstance {
            position: atlas_tile_id,
            mask: TilePosition::ZERO,
            pattern_data: [
                pattern_position.to_u32(),
                self.handle.to_u32(),
            ],
        }
    }

    fn new_renderer(
        device: &wgpu::Device,
        shaders: &mut ShaderSources,
        tile_atlas_desc_layout: &wgpu::BindGroupLayout,
    ) -> CustomPatternRenderer<Self> {
        let label = &"simple_gradient";

        let src = include_str!("../shaders/simple_gradient_pattern.wgsl");
        let module = shaders.create_shader_module(device, label, src, &[]);

        let defaults = PipelineDefaults::new();
        let attributes = VertexBuilder::from_slice(&[wgpu::VertexFormat::Uint32x4]);

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
