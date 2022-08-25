use crate::{Color, Point, vector};
use crate::gpu::{ShaderSources, VertexBuilder, PipelineDefaults};

use crate::custom_pattern::*;

pub type SimpleGradientBuilder = CustomPatternBuilder<SimpleGradient>;
pub type SimpleGradientRenderer = CustomPatternRenderer<SimpleGradient>;


#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct SimpleGradientTile {
    pub tile_id: u32,
    pub padding: u32,
    pub from_color: u32,
    pub to_color: u32,
    pub from_pos: [f32; 2],
    pub to_pos: [f32; 2],
}

unsafe impl bytemuck::Pod for SimpleGradientTile {}
unsafe impl bytemuck::Zeroable for SimpleGradientTile {}

#[derive(Copy, Clone, Debug)]
pub struct Stop {
    pub position: Point,
    pub color: Color,
}
pub struct SimpleGradient {
    stops: [Stop; 2],
    is_opaque: bool,
}

impl SimpleGradient {
    pub fn new(stops: [Stop; 2]) -> Self {
        SimpleGradient {
            is_opaque: stops[0].color.is_opaque() && stops[1].color.is_opaque(),
            stops,
        }
    }
}

impl CustomPattern for SimpleGradient {
    type Instance = SimpleGradientTile;
    type RenderData = ();

    fn is_opaque(&self) -> bool { self.is_opaque }

    fn new_tile(&self, tile_id: u32, x: f32, y: f32) -> Self::Instance {
        let offset = vector(x, y);

        SimpleGradientTile {
            tile_id,
            padding: 0,
            from_color: self.stops[0].color.to_u32(),
            to_color: self.stops[1].color.to_u32(),
            from_pos: (self.stops[0].position - offset).to_array(),
            to_pos: (self.stops[1].position - offset).to_array(),
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
        let attributes = VertexBuilder::from_slice(&[
            wgpu::VertexFormat::Uint32x4,
            wgpu::VertexFormat::Float32x4,
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
