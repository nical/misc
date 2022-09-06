use crate::{TilePosition, PatternData, PatternKind, Stats};
use crate::gpu::{ShaderSources, PipelineDefaults, VertexBuilder};
use crate::tile_encoder::{BufferRange, TileAllocator};
use crate::tile_renderer::{PatternRenderer, BufferBumpAllocator, TileInstance};

use std::ops::Range;

pub struct TilePipelines {
    pub opaque: wgpu::RenderPipeline,
    pub masked: wgpu::RenderPipeline,
}

pub struct Varying<'l> {
    pub name: &'l str,
    pub kind: &'l str,
    pub interpolate: &'l str,
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
    attributes: VertexBuilder,
    opaque_layout: wgpu::PipelineLayout,
    masked_layout: wgpu::PipelineLayout,
    next_pattern_kind: PatternKind,
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
            attributes: VertexBuilder::from_slice(&[wgpu::VertexFormat::Uint32x4]),
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
            next_pattern_kind: 0,
        }
    }

    pub fn new_pattern_kind(&mut self) -> PatternKind {
        let kind = self.next_pattern_kind;
        self.next_pattern_kind += 1;

        kind
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
            varyings_src.push_str(&format!(
                "    @location({}) @interpolate({}) {}: {},\n",
                location, varying.interpolate, varying.name, varying.kind
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
                buffers: &[self.attributes.buffer_layout()],
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
                buffers: &[self.attributes.buffer_layout()],
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

    fn is_mergeable(&self) -> bool { false }

    fn new_tile(&mut self, pattern_position: TilePosition) -> PatternData;

    fn set_render_pass_state<'a, 'b: 'a>(_data: &'a Self::RenderData, _pass: &mut wgpu::RenderPass<'b>) {}

    fn allocate_buffer_ranges(&mut self, _render_data: &mut Self::RenderData) {}

    fn upload(&self, _render_data: &mut Self::RenderData, _queue: &wgpu::Queue) {}
}

pub struct CustomPatternBuilder<Parameters: CustomPattern> {
    parameters: Parameters,
    prerendered_tiles: Vec<TileInstance>,
    opaque_tiles: Vec<TileInstance>,
    prerender_vbo_range: BufferRange,
    opaque_vbo_range: BufferRange,
    render_passes: Vec<Range<u32>>,
    current_texture: u32,
    render_pass_start: u32,
    x: u32,
    y: u32,
    kind: PatternKind,
}

impl<Parameters: CustomPattern> CustomPatternBuilder<Parameters> {
    pub fn new(parameters: Parameters, kind: PatternKind) -> Self {
        CustomPatternBuilder {
            parameters,
            prerendered_tiles: Vec::new(),
            opaque_tiles: Vec::new(),
            prerender_vbo_range: BufferRange(0, 0),
            opaque_vbo_range: BufferRange(0, 0),
            render_passes: Vec::new(),
            current_texture: 0,
            render_pass_start: 0,
            x: 0,
            y: 0,
            kind
        }
    }

    pub fn set(&mut self, parameters: Parameters) {
        self.parameters = parameters;
    }

    pub fn reset(&mut self) {
        self.prerendered_tiles.clear();
        self.opaque_tiles.clear();
        self.render_pass_start = 0;
        self.current_texture = 0;
        self.render_passes.clear();
    }

    pub fn allocate_buffer_ranges(&mut self, renderer: &mut CustomPatternRenderer<Parameters>) {
        self.prerender_vbo_range = renderer.vbo_alloc.push(self.prerendered_tiles.len());
        self.opaque_vbo_range = renderer.vbo_alloc.push(self.opaque_tiles.len());
        self.parameters.allocate_buffer_ranges(&mut renderer.data);
    }

    pub fn upload(&self, renderer: &mut CustomPatternRenderer<Parameters>, queue: &wgpu::Queue) {
        if !self.prerendered_tiles.is_empty() {
            queue.write_buffer(
                &renderer.vbo,
                self.prerender_vbo_range.byte_offset::<TileInstance>(),
                bytemuck::cast_slice(&self.prerendered_tiles),
            );
            renderer.render_passes.clear();
            renderer.render_passes.extend_from_slice(&self.render_passes);
        }

        if !self.opaque_tiles.is_empty() {
            queue.write_buffer(
                &renderer.vbo,
                self.opaque_vbo_range.byte_offset::<TileInstance>(),
                bytemuck::cast_slice(&self.opaque_tiles),
            );
            renderer.opaque_pass = self.opaque_vbo_range.to_u32();
        }

        self.parameters.upload(&mut renderer.data, queue);
    }

    pub fn end_render_pass(&mut self) {
        while self.render_passes.len() as u32 <= self.current_texture {
            self.render_passes.push(0..0);
        }
        let render_pass_end = self.prerendered_tiles.len() as u32;
        self.render_passes[self.current_texture as usize] = self.render_pass_start..render_pass_end;
        self.render_pass_start = render_pass_end;
    }

    pub fn opaque_tiles_mut(&mut self) -> &mut[TileInstance] {
        &mut self.opaque_tiles
    }

    pub fn update_stats(&self, stats: &mut Stats) {
        stats.opaque_tiles += self.opaque_tiles.len();
        stats.prerendered_tiles += self.prerendered_tiles.len();
    }
}

impl<Parameters: CustomPattern> crate::tiler::TilerPattern for CustomPatternBuilder<Parameters> {
    fn pattern_kind(&self) -> u32 { self.kind }
    fn tile_is_opaque(&self) -> bool { self.parameters.is_opaque() }
    fn set_tile(&mut self, x: u32, y: u32) {
        self.x = x;
        self.y = y;
    }

    fn is_mergeable(&self) -> bool {
        self.parameters.is_mergeable()
    }

    fn opaque_tiles(&mut self) -> &mut Vec<TileInstance> {
        &mut self.opaque_tiles
    }

    fn tile_data(&mut self) -> PatternData {
        self.parameters.new_tile(TilePosition::new(self.x, self.y))
    }

    fn prerender_tile(&mut self, atlas: &mut TileAllocator) -> (TilePosition, PatternData) {
        let (atlas_tile_position, texture) = atlas.allocate();

        if texture != self.current_texture {
            self.end_render_pass();
            self.current_texture = texture;
        }

        let pattern_position = TilePosition::new(self.x, self.y);
        let pattern_data = self.parameters.new_tile(pattern_position);

        self.prerendered_tiles.push(TileInstance {
            position: atlas_tile_position,
            mask: TilePosition::ZERO,
            pattern_position,
            pattern_data,
        });

        (atlas_tile_position, 0)
    }

    fn set_render_pass(&mut self, pass_index: u32) {
        if pass_index == self.current_texture {
            return;
        }

        self.end_render_pass();

        self.current_texture = pass_index;
    }
}

pub struct CustomPatternRenderer<Pattern: CustomPattern> {
    pipelines: TilePipelines,
    // TODO: should be in a different struct.
    vbo: wgpu::Buffer,
    vbo_alloc: BufferBumpAllocator,
    render_passes: Vec<Range<u32>>,
    opaque_pass: Range<u32>,
    pub data: Pattern::RenderData,
}

impl<Pattern: CustomPattern> CustomPatternRenderer<Pattern> {
    pub fn new(
        device: &wgpu::Device,
        helper: &mut CustomPatterns,
        descriptor: &CustomPatternDescriptor,
        data: Pattern::RenderData,
    ) -> Self {

        let pipelines = helper.create_tile_render_pipelines(device, descriptor);

        let vbo = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(descriptor.name),
            size: 4096 * 64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        CustomPatternRenderer {
            pipelines,
            vbo,
            vbo_alloc: BufferBumpAllocator::new(),
            render_passes: Vec::new(),
            opaque_pass: 0..0,
            data,
        }
    }

    fn get_tiles(&self, pass_idx: u32) -> Option<Range<u32>> {
        let pass_idx = pass_idx as usize;
        if pass_idx >= self.render_passes.len() {
            return None;
        }

        let range = self.render_passes[pass_idx].clone();

        if range.is_empty() {
            None
        } else {
            Some(range)
        }
    }
}

impl<Pattern: CustomPattern> PatternRenderer for CustomPatternRenderer<Pattern> {
    fn begin_frame(&mut self) {
        self.vbo_alloc.clear();
    }

    fn has_content(&self, pass_idx: u32) -> bool {
        self.get_tiles(pass_idx).is_some()
    }

    fn render<'a, 'b: 'a>(&'b self, pass_idx: u32, pass: &mut wgpu::RenderPass<'a>) {
        if let Some(tile_range) = self.get_tiles(pass_idx) {
            pass.set_pipeline(&self.pipelines.opaque);
            pass.set_vertex_buffer(0, self.vbo.slice(..));
            Pattern::set_render_pass_state(&self.data, pass);
            pass.draw_indexed(0..6, 0, tile_range);
        }
    }

    fn render_opaque_pass<'a, 'b: 'a>(&'b self, pass: &mut wgpu::RenderPass<'a>) {
        if self.opaque_pass.is_empty() {
            return;
        }
        pass.set_pipeline(&self.pipelines.opaque);
        pass.set_vertex_buffer(0, self.vbo.slice(..));
        Pattern::set_render_pass_state(&self.data, pass);
        pass.draw_indexed(0..6, 0, self.opaque_pass.clone());
    }

    fn prepare_alpha_pass<'a, 'b: 'a>(&'b self, pass: &mut wgpu::RenderPass<'a>) {
        pass.set_pipeline(&self.pipelines.masked);
        Pattern::set_render_pass_state(&self.data, pass);
    }
}
