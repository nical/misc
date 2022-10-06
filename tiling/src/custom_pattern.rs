use crate::tiling::{
    TilePosition, PatternData, PatternIndex, TileVisibility,
};
use crate::gpu::{ShaderSources, PipelineDefaults, VertexBuilder};

pub struct TilePipelines {
    pub opaque: wgpu::RenderPipeline,
    pub masked: wgpu::RenderPipeline,
}

pub struct Varying<'l> {
    pub name: &'l str,
    pub kind: &'l str,
    pub interpolated: bool,
}

pub struct CustomPatternDescriptor<'l> {
    pub name: &'l str,
    pub source: &'l str,
    pub varyings: &'l[Varying<'l>],
    pub extra_bind_groups: &'l [wgpu::BindGroupLayout],
}

pub struct CustomPatterns<'l> {
    shaders: &'l mut ShaderSources,
    defaults: PipelineDefaults,
    masked_attributes: VertexBuilder,
    opaque_attributes: VertexBuilder,
    opaque_layout: wgpu::PipelineLayout,
    masked_layout: wgpu::PipelineLayout,
    next_index: PatternIndex,
}

impl<'l> CustomPatterns<'l> {
    pub fn new(
        device: &wgpu::Device,
        shaders: &'l mut ShaderSources,
        tile_atlas_desc_layout: &wgpu::BindGroupLayout,
        mask_src_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        CustomPatterns {
            shaders,
            defaults: PipelineDefaults::new(),
            masked_attributes: VertexBuilder::from_slice(&[wgpu::VertexFormat::Uint32x4, wgpu::VertexFormat::Uint32x4]),
            opaque_attributes: VertexBuilder::from_slice(&[wgpu::VertexFormat::Uint32x4]),
            opaque_layout: device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("opaque pattern"),
                bind_group_layouts: &[tile_atlas_desc_layout],
                push_constant_ranges: &[],
            }),
            masked_layout: device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("masked pattern"),
                bind_group_layouts: &[tile_atlas_desc_layout, mask_src_layout],
                push_constant_ranges: &[],
            }),
            next_index: 0,
        }
    }

    pub fn new_index(&mut self) -> PatternIndex {
        let index = self.next_index;
        self.next_index += 1;

        index
    }

    pub fn create_tile_render_pipelines(
        &mut self,
        device: &wgpu::Device,
        descriptor: &CustomPatternDescriptor,
    ) -> TilePipelines {

        let mut location = 2;
        let mut varyings_src = String::new();
        let mut pass_varyings_src = String::new();
        let mut frag_arguments_src = String::new();
        let mut pattern_struct_src = String::new();
        for varying in descriptor.varyings {
            let interpolate = if varying.interpolated { "perspective" } else { "flat" };
            varyings_src.push_str(&format!(
                "    @location({}) @interpolate({}) {}: {},\n",
                location, interpolate, varying.name, varying.kind
            ));
            pass_varyings_src.push_str(&format!("    pattern.{},\n", varying.name));
            frag_arguments_src.push_str(&format!("    {},\n", varying.name));
            pattern_struct_src.push_str(&format!("    {}: {},\n", varying.name, varying.kind));
            location += 1;
        }

        self.shaders.define("custom_pattern_src", &descriptor.source);
        self.shaders.define("custom_pattern_varyings_src", &varyings_src);
        self.shaders.define("custom_pattern_pass_varyings_src", &pass_varyings_src);
        self.shaders.define("custom_pattern_fragment_arguments_src", &frag_arguments_src);
        self.shaders.define("custom_pattern_struct_src", &pattern_struct_src);

        let base_src = include_str!("./../shaders/custom_pattern_tile.wgsl");

        let opaque_module = self.shaders.create_shader_module(device, descriptor.name, base_src, &[]);
        let masked_module = self.shaders.create_shader_module(device, descriptor.name, base_src, &["TILED_MASK"]);

        let opaque_layout = if descriptor.extra_bind_groups.is_empty() {
            &self.opaque_layout
        } else {
            unimplemented!()
        };

        let masked_layout = if descriptor.extra_bind_groups.is_empty() {
            &self.masked_layout
        } else {
            unimplemented!()
        };

        let opaque_label = format!("opaque {}", descriptor.name);
        let opaque_descriptor = wgpu::RenderPipelineDescriptor {
            label: Some(&opaque_label),
            layout: Some(opaque_layout),
            vertex: wgpu::VertexState {
                module: &opaque_module,
                entry_point: "vs_main",
                buffers: &[self.opaque_attributes.buffer_layout()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &opaque_module,
                entry_point: "fs_main",
                targets: self.defaults.color_target_state_no_blend(),
            }),
            primitive: self.defaults.primitive_state(),
            depth_stencil: None,
            multiview: None,
            multisample: wgpu::MultisampleState::default(),
        };

        let masked_label = format!("masked {}", descriptor.name);
        let masked_descriptor = wgpu::RenderPipelineDescriptor {
            label: Some(&masked_label),
            layout: Some(masked_layout),
            vertex: wgpu::VertexState {
                module: &masked_module,
                entry_point: "vs_main",
                buffers: &[self.masked_attributes.buffer_layout()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &masked_module,
                entry_point: "fs_main",
                targets: self.defaults.color_target_state(),
            }),
            primitive: self.defaults.primitive_state(),
            depth_stencil: None,
            multiview: None,
            multisample: wgpu::MultisampleState::default(),
        };

        TilePipelines {
            opaque: device.create_render_pipeline(&opaque_descriptor),
            masked: device.create_render_pipeline(&masked_descriptor),
        }
    }
}

pub trait CustomPattern: Sized {
    type RenderData;

    fn is_opaque(&self) -> bool;

    fn can_stretch_horizontally(&self) -> bool { false }

    fn new_tile(&mut self, pattern_position: TilePosition) -> PatternData;
}

pub struct CustomPatternBuilder<Parameters: CustomPattern> {
    parameters: Parameters,
    index: PatternIndex,
}

impl<Parameters: CustomPattern> CustomPatternBuilder<Parameters> {
    pub fn new(parameters: Parameters, index: PatternIndex) -> Self {
        CustomPatternBuilder {
            parameters,
            index
        }
    }

    pub fn set(&mut self, parameters: Parameters) {
        self.parameters = parameters;
    }
}

impl<Parameters: CustomPattern> crate::tiling::TilerPattern for CustomPatternBuilder<Parameters> {
    fn index(&self) -> PatternIndex { self.index }
    fn is_entirely_opaque(&self) -> bool { self.parameters.is_opaque() }
    fn tile_visibility(&self, _: u32, _: u32) -> TileVisibility {
        if self.parameters.is_opaque() {
            TileVisibility::Opaque
        } else {
            TileVisibility::Alpha
        }
        // TODO: support empty tiles
    }

    fn can_stretch_horizontally(&self) -> bool {
        self.parameters.can_stretch_horizontally()
    }

    fn tile_data(&mut self, x: u32, y: u32) -> PatternData {
        self.parameters.new_tile(TilePosition::new(x, y))
    }
}

