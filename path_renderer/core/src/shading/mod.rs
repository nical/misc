mod generator;
mod shaders;
pub mod preprocessor;

use lyon::path::FillRule;
use preprocessor::Source;
pub use shaders::Shaders;
use std::borrow::Cow;

// If this grows to use up more than 4 bits, BatchKey must be adjusted.
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BlendMode {
    /// Overwrite the destination.
    None,
    /// The default blend mode for alpha blending.
    PremultipliedAlpha,
    Add,
    Subtract,
    Multiply,
    Screen,
    Lighter,
    Exclusion,
    /// Discards pixels of the destination where the source alpha is zero.
    ClipIn,
    /// Discards pixels of the destination where the source alpha is one.
    ClipOut,
}

impl BlendMode {
    pub fn with_alpha(self, alpha: bool) -> Self {
        match (self, alpha) {
            (BlendMode::None, true) => BlendMode::PremultipliedAlpha,
            (BlendMode::PremultipliedAlpha, false) => BlendMode::None,
            (mode, _) => mode,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum WgslType {
    Float32,
    Uint32,
    Sint32,
    Bool,
    Float32x2,
    Float32x3,
    Float32x4,
    Uint32x2,
    Uint32x3,
    Uint32x4,
    Sint32x2,
    Sint32x3,
    Sint32x4,
}

#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ShaderPatternId(u16);
impl ShaderPatternId {
    pub fn from_index(idx: usize) -> Self {
        debug_assert!(idx < (u16::MAX - 1) as usize);
        ShaderPatternId(idx as u16)
    }
    pub fn index(self) -> usize {
        self.0 as usize
    }
    pub fn get(self) -> u16 {
        self.0
    }
    pub fn is_none(self) -> bool {
        self.0 == u16::MAX
    }
    pub const NONE: Self = ShaderPatternId(u16::MAX);
    fn map<Out>(self, cb: impl Fn(Self) -> Out) -> Option<Out> {
        if self.is_none() {
            return None;
        }

        Some(cb(self))
    }
}

#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct GeometryId(u16);
impl GeometryId {
    pub fn from_index(idx: usize) -> Self {
        debug_assert!(idx < (u16::MAX - 1) as usize);
        GeometryId(idx as u16)
    }
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

// Note: The pipeline layout key hash relies on BindGroupLayoutId using
// 16 bits.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct BindGroupLayoutId(pub(crate) u16);

impl BindGroupLayoutId {
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }

    #[inline]
    fn from_index(idx: usize) -> Self {
        BindGroupLayoutId(idx as u16)
    }
}

/// Describes data produced by the vertex shader and consumed by the
/// fragment shader.
#[derive(Clone)]
pub struct Varying {
    pub name: Cow<'static, str>,
    pub kind: WgslType,
    pub interpolated: bool,
}

impl Varying {
    #[inline]
    pub fn float32(name: &'static str) -> Self {
        Varying {
            name: name.into(),
            kind: WgslType::Float32,
            interpolated: true,
        }
    }
    #[inline]
    pub fn float32x2(name: &'static str) -> Self {
        Varying {
            name: name.into(),
            kind: WgslType::Float32x2,
            interpolated: true,
        }
    }
    #[inline]
    pub fn float32x3(name: &'static str) -> Self {
        Varying {
            name: name.into(),
            kind: WgslType::Float32x3,
            interpolated: true,
        }
    }
    #[inline]
    pub fn float32x4(name: &'static str) -> Self {
        Varying {
            name: name.into(),
            kind: WgslType::Float32x4,
            interpolated: true,
        }
    }
    #[inline]
    pub fn uint32(name: &'static str) -> Self {
        Varying {
            name: name.into(),
            kind: WgslType::Uint32,
            interpolated: false,
        }
    }
    #[inline]
    pub fn uint32x2(name: &'static str) -> Self {
        Varying {
            name: name.into(),
            kind: WgslType::Uint32x2,
            interpolated: false,
        }
    }
    #[inline]
    pub fn uint32x3(name: &'static str) -> Self {
        Varying {
            name: name.into(),
            kind: WgslType::Uint32x3,
            interpolated: false,
        }
    }
    #[inline]
    pub fn uint32x4(name: &'static str) -> Self {
        Varying {
            name: name.into(),
            kind: WgslType::Uint32x4,
            interpolated: false,
        }
    }
    #[inline]
    pub fn sint32(name: &'static str) -> Self {
        Varying {
            name: name.into(),
            kind: WgslType::Sint32,
            interpolated: false,
        }
    }
    #[inline]
    pub fn sint32x2(name: &'static str) -> Self {
        Varying {
            name: name.into(),
            kind: WgslType::Sint32x2,
            interpolated: false,
        }
    }
    #[inline]
    pub fn sint32x3(name: &'static str) -> Self {
        Varying {
            name: name.into(),
            kind: WgslType::Sint32x3,
            interpolated: false,
        }
    }
    #[inline]
    pub fn sint32x4(name: &'static str) -> Self {
        Varying {
            name: name.into(),
            kind: WgslType::Sint32x4,
            interpolated: false,
        }
    }
    #[inline]
    pub fn with_interpolation(mut self, interpolated: bool) -> Self {
        self.interpolated = interpolated;
        self
    }
    #[inline]
    pub fn interpolated(self) -> Self {
        self.with_interpolation(true)
    }
    #[inline]
    pub fn flat(self) -> Self {
        self.with_interpolation(false)
    }
}

/// Describes per-vertex data consumed by the vertex shader.
#[derive(Clone)]
pub struct VertexAtribute {
    pub name: Cow<'static, str>,
    pub kind: WgslType,
}

impl VertexAtribute {
    #[inline]
    pub fn float32(name: &'static str) -> Self {
        VertexAtribute {
            name: name.into(),
            kind: WgslType::Float32,
        }
    }
    #[inline]
    pub fn float32x2(name: &'static str) -> Self {
        VertexAtribute {
            name: name.into(),
            kind: WgslType::Float32x2,
        }
    }
    #[inline]
    pub fn float32x3(name: &'static str) -> Self {
        VertexAtribute {
            name: name.into(),
            kind: WgslType::Float32x3,
        }
    }
    #[inline]
    pub fn float32x4(name: &'static str) -> Self {
        VertexAtribute {
            name: name.into(),
            kind: WgslType::Float32x4,
        }
    }
    #[inline]
    pub fn uint32(name: &'static str) -> Self {
        VertexAtribute {
            name: name.into(),
            kind: WgslType::Uint32,
        }
    }
    #[inline]
    pub fn uint32x2(name: &'static str) -> Self {
        VertexAtribute {
            name: name.into(),
            kind: WgslType::Uint32x2,
        }
    }
    #[inline]
    pub fn uint32x3(name: &'static str) -> Self {
        VertexAtribute {
            name: name.into(),
            kind: WgslType::Uint32x3,
        }
    }
    #[inline]
    pub fn uint32x4(name: &'static str) -> Self {
        VertexAtribute {
            name: name.into(),
            kind: WgslType::Uint32x4,
        }
    }
    #[inline]
    pub fn int32(name: &'static str) -> Self {
        VertexAtribute {
            name: name.into(),
            kind: WgslType::Sint32,
        }
    }
    #[inline]
    pub fn int32x2(name: &'static str) -> Self {
        VertexAtribute {
            name: name.into(),
            kind: WgslType::Sint32x2,
        }
    }
    #[inline]
    pub fn int32x3(name: &'static str) -> Self {
        VertexAtribute {
            name: name.into(),
            kind: WgslType::Sint32x3,
        }
    }
    #[inline]
    pub fn int32x4(name: &'static str) -> Self {
        VertexAtribute {
            name: name.into(),
            kind: WgslType::Sint32x4,
        }
    }

    fn to_wgpu(&self) -> wgpu::VertexFormat {
        match self.kind {
            WgslType::Float32 => wgpu::VertexFormat::Float32,
            WgslType::Sint32 => wgpu::VertexFormat::Sint32,
            WgslType::Uint32 => wgpu::VertexFormat::Uint32,
            WgslType::Float32x2 => wgpu::VertexFormat::Float32x2,
            WgslType::Sint32x2 => wgpu::VertexFormat::Sint32x2,
            WgslType::Uint32x2 => wgpu::VertexFormat::Uint32x2,
            WgslType::Float32x3 => wgpu::VertexFormat::Float32x3,
            WgslType::Sint32x3 => wgpu::VertexFormat::Sint32x3,
            WgslType::Uint32x3 => wgpu::VertexFormat::Uint32x3,
            WgslType::Float32x4 => wgpu::VertexFormat::Float32x4,
            WgslType::Sint32x4 => wgpu::VertexFormat::Sint32x4,
            WgslType::Uint32x4 => wgpu::VertexFormat::Uint32x4,
            _ => unimplemented!(),
        }
    }
}

/// The description of a bind group layout usable by generated render pipelines.
pub struct BindGroupLayout {
    pub name: String,
    pub handle: wgpu::BindGroupLayout,
    pub entries: Vec<Binding>,
}

impl BindGroupLayout {
    pub fn new(device: &wgpu::Device, name: String, entries: Vec<Binding>) -> Self {
        let mut bg = Vec::with_capacity(entries.len());
        for (idx, entry) in entries.iter().enumerate() {
            bg.push(wgpu::BindGroupLayoutEntry {
                binding: idx as u32,
                ty: entry.ty,
                visibility: entry.visibility,
                count: None,
            });
        }

        let handle = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &bg,
            label: Some(name.as_str()),
        });

        BindGroupLayout {
            name,
            handle,
            entries,
        }
    }
}

/// The description of a shader binding usable by generated render pipelines.
pub struct Binding {
    pub name: String,
    pub struct_type: String,
    pub ty: wgpu::BindingType,
    pub visibility: wgpu::ShaderStages,
}

impl Binding {
    pub fn uniform_buffer(name: &str, struct_type: &str) -> Self {
        Binding {
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            // TODO: min binding size.
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            name: name.into(),
            struct_type: struct_type.into(),
        }
    }

    pub fn storage_buffer(name: &str, struct_type: &str) -> Self {
        Binding {
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            // TODO: min binding size.
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            name: name.into(),
            struct_type: struct_type.into(),
        }
    }
}

/// Describes a pattern for generated render pipelines.
pub struct PatternDescriptor {
    pub name: Cow<'static, str>,
    pub source: Source,
    pub varyings: Vec<Varying>,
    pub bindings: Option<BindGroupLayoutId>,
}

/// Desribes a geometry for generated render pipelines.
#[derive(Clone)]
pub struct GeometryDescriptor {
    pub name: Cow<'static, str>,
    pub primitive: wgpu::PrimitiveState,
    pub source: Source,
    pub vertex_attributes: Vec<VertexAtribute>,
    pub instance_attributes: Vec<VertexAtribute>,
    pub varyings: Vec<Varying>,
    pub bindings: Option<BindGroupLayoutId>,
    pub shader_defines: Vec<&'static str>,
    pub constants: Vec<(&'static str, f64)>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum StencilMode {
    EvenOdd,
    NonZero,
    Ignore,
    None,
}

impl From<FillRule> for StencilMode {
    fn from(fill_rule: FillRule) -> Self {
        match fill_rule {
            FillRule::EvenOdd => StencilMode::EvenOdd,
            FillRule::NonZero => StencilMode::NonZero,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DepthMode {
    Enabled,
    Ignore,
    None,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum SurfaceKind {
    None,
    Color,
    Alpha,
    HdrColor,
    HdrAlpha,
}

impl SurfaceKind {
    pub fn to_bits(self) -> u16 {
        match self {
            SurfaceKind::None => 0,
            SurfaceKind::Color => 1,
            SurfaceKind::Alpha => 2,
            SurfaceKind::HdrColor => 3,
            SurfaceKind::HdrAlpha => 4,
        }
    }

    pub fn from_bits(bits: u16) -> Self {
        match bits {
            1 => SurfaceKind::Color,
            2 => SurfaceKind::Alpha,
            3 => SurfaceKind::HdrColor,
            4 => SurfaceKind::HdrAlpha,
            _ => SurfaceKind::None,
        }
    }
}

// The surface parameters for draw calls.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SurfaceDrawConfig {
    pub msaa: bool,
    pub depth: DepthMode,
    pub stencil: StencilMode,
    pub attachments: [SurfaceKind; 3],
}

impl SurfaceDrawConfig {
    pub(crate) fn hash(&self) -> u16 {
        return if self.msaa { 1 } else { 0 }
            + (match self.depth {
                DepthMode::Enabled => 1,
                DepthMode::Ignore => 2,
                DepthMode::None => 0,
            } << 1)
            + (match self.stencil {
                StencilMode::EvenOdd => 1,
                StencilMode::NonZero => 2,
                StencilMode::Ignore => 3,
                StencilMode::None => 0,
            } << 3)
            + (self.attachments[0].to_bits() << 6)
            + (self.attachments[1].to_bits() << 9)
            + (self.attachments[2].to_bits() << 12);
    }

    pub(crate) fn from_hash(val: u16) -> Self {
        SurfaceDrawConfig {
            msaa: val & 1 != 0,
            depth: match (val >> 1) & 3 {
                1 => DepthMode::Enabled,
                2 => DepthMode::Ignore,
                _ => DepthMode::None,
            },
            stencil: match (val >> 3) & 3 {
                1 => StencilMode::EvenOdd,
                2 => StencilMode::NonZero,
                3 => StencilMode::Ignore,
                _ => StencilMode::None,
            },
            attachments: [
                SurfaceKind::from_bits((val >> 6) & 0b111),
                SurfaceKind::from_bits((val >> 9) & 0b111),
                SurfaceKind::from_bits((val >> 12) & 0b111),
            ],
        }
    }

    pub fn color() -> Self {
        SurfaceDrawConfig {
            msaa: false,
            depth: DepthMode::None,
            stencil: StencilMode::None,
            attachments: [SurfaceKind::Color, SurfaceKind::None, SurfaceKind::None],
        }
    }

    pub fn alpha() -> Self {
        SurfaceDrawConfig {
            msaa: false,
            depth: DepthMode::None,
            stencil: StencilMode::None,
            attachments: [SurfaceKind::Alpha, SurfaceKind::None, SurfaceKind::None],
        }
    }

    pub fn with_msaa(mut self, msaa: bool) -> Self {
        self.msaa = msaa;
        self
    }

    pub fn with_stencil(mut self, stencil: StencilMode) -> Self {
        self.stencil = stencil;
        self
    }

    pub fn with_depth(mut self, depth: DepthMode) -> Self {
        self.depth = depth;
        self
    }

    pub fn num_color_attachments(&self) -> usize {
        use SurfaceKind::None;
        match self.attachments {
            [None, None, None] => 0,
            [_, None, None] => 1,
            [_, _, None] => 2,
            [_, _, _] => 3,
        }
    }

    pub fn color_attachments(&self) -> &[SurfaceKind] {
        &self.attachments[0..self.num_color_attachments()]
    }
}

impl Default for SurfaceDrawConfig {
    fn default() -> Self {
        SurfaceDrawConfig::color()
    }
}

/// Allows fast lookup of a generated render pipeline in a `RenderPipelines` registry.
pub type RenderPipelineIndex = crate::cache::Index<RenderPipelineKey>;

/// A registry of generated render pipelines.
pub type RenderPipelines = crate::cache::Registry<RenderPipelineKey, wgpu::RenderPipeline>;

/// Turns `RenderPipelineKey`s into `RenderPipelineIndex`es and keeps track of what
/// generated render pipelines must be compiled this frame (if any).
pub type PrepareRenderPipelines = crate::cache::Prepare<RenderPipelineKey>;

pub(crate) struct RenderPipelineBuilder<'l>(pub &'l wgpu::Device, pub &'l mut Shaders);

/// Identifies a generated render pipeline.
///
/// In order to to access the render pipeline, a `RenderPipelineIndex` must be created
/// from this key using `PrepareRenderPipelines::prepare`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct RenderPipelineKey(u64);

impl RenderPipelineKey {
    pub fn new(
        base: GeometryId,
        pattern: ShaderPatternId,
        blend_mode: BlendMode,
        surf: SurfaceDrawConfig,
    ) -> Self {
        Self(
            surf.hash() as u64
                | ((base.0 as u64) << 16)
                | ((pattern.get() as u64) << 32)
                | ((blend_mode as u64) << 48),
        )
    }

    pub fn unpack(&self) -> (GeometryId, ShaderPatternId, BlendMode, SurfaceDrawConfig) {
        let base = GeometryId((self.0 >> 16) as u16);
        let pattern = ShaderPatternId((self.0 >> 32) as u16);
        let surf = SurfaceDrawConfig::from_hash(self.0 as u16);
        let blend: BlendMode = unsafe { std::mem::transmute((self.0 >> 48) as u8) };
        (base, pattern, blend, surf)
    }
}

impl<'l> crate::cache::Build<RenderPipelineKey, wgpu::RenderPipeline>
    for RenderPipelineBuilder<'l>
{
    fn build(&mut self, key: RenderPipelineKey) -> wgpu::RenderPipeline {
        let (pipeline, pattern, blend, surface) = key.unpack();
        self.1
            .generate_pipeline_variant(self.0, pipeline, pattern, blend, &surface)
    }

    fn finish(&mut self) {}
}

/// A Utility for building vertex attributes.
#[derive(Clone)]
pub struct VertexBuilder {
    location: u32,
    offset: u64,
    attributes: Vec<wgpu::VertexAttribute>,
    step_mode: wgpu::VertexStepMode,
}

impl VertexBuilder {
    pub fn new(step_mode: wgpu::VertexStepMode) -> Self {
        VertexBuilder {
            location: 0,
            offset: 0,
            attributes: Vec::with_capacity(16),
            step_mode,
        }
    }

    pub fn from_slice(step_mode: wgpu::VertexStepMode, formats: &[wgpu::VertexFormat]) -> Self {
        let mut attributes = VertexBuilder::new(step_mode);
        for format in formats {
            attributes.push(*format);
        }

        attributes
    }

    pub fn push(&mut self, format: wgpu::VertexFormat) {
        self.attributes.push(wgpu::VertexAttribute {
            format,
            offset: self.offset,
            shader_location: self.location,
        });
        self.offset += format.size();
        self.location += 1;
    }

    pub fn get(&self) -> &[wgpu::VertexAttribute] {
        &self.attributes
    }

    pub fn clear(&mut self) {
        self.location = 0;
        self.offset = 0;
        self.attributes.clear();
    }

    pub fn buffer_layout(&self) -> wgpu::VertexBufferLayout {
        wgpu::VertexBufferLayout {
            array_stride: self.offset,
            step_mode: self.step_mode,
            attributes: &self.attributes,
        }
    }
}

// TODO: manage the number of shader configurations so that it remains reasonable
// while not constraining renderers too much.
// for example we don't want depth and stencil in the tile atlas passes
// but we should allow the tiling renderer to work in a render pass that has depth
// and/or stencil enabled (ignoring them).
pub struct PipelineDefaults {
    color_format: wgpu::TextureFormat,
    mask_format: wgpu::TextureFormat,
    hdr_color_format: wgpu::TextureFormat,
    hdr_alpha_format: wgpu::TextureFormat,
    depth_buffer: bool,
    stencil_buffer: bool,
    msaa_samples: u32,
}

impl PipelineDefaults {
    pub fn new() -> Self {
        PipelineDefaults {
            color_format: wgpu::TextureFormat::Bgra8Unorm,
            mask_format: wgpu::TextureFormat::R8Unorm,
            hdr_color_format: wgpu::TextureFormat::Rgba16Float,
            hdr_alpha_format: wgpu::TextureFormat::R16Float,
            depth_buffer: true,
            stencil_buffer: true,
            msaa_samples: 4,
        }
    }

    pub fn primitive_state() -> wgpu::PrimitiveState {
        wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            polygon_mode: wgpu::PolygonMode::Fill,
            front_face: wgpu::FrontFace::Ccw,
            strip_index_format: None,
            cull_mode: None,
            unclipped_depth: false,
            conservative: false,
        }
    }

    pub fn color_target_state(&self) -> Option<wgpu::ColorTargetState> {
        Some(wgpu::ColorTargetState {
            format: self.color_format,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        })
    }

    pub fn color_target_state_no_blend(&self) -> Option<wgpu::ColorTargetState> {
        Some(wgpu::ColorTargetState {
            format: self.color_format,
            blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
            write_mask: wgpu::ColorWrites::ALL,
        })
    }

    pub fn alpha_target_state(&self) -> Option<wgpu::ColorTargetState> {
        Some(wgpu::ColorTargetState {
            format: self.mask_format,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        })
    }

    pub fn depth_stencil_format(&self) -> Option<wgpu::TextureFormat> {
        match (self.depth_buffer, self.stencil_buffer) {
            (false, false) => None,
            (true, false) => Some(wgpu::TextureFormat::Depth32Float),
            (false, true) => Some(wgpu::TextureFormat::Stencil8),
            (true, true) => Some(wgpu::TextureFormat::Depth24PlusStencil8),
        }
    }

    pub fn color_format(&self) -> wgpu::TextureFormat {
        self.color_format
    }

    pub fn mask_format(&self) -> wgpu::TextureFormat {
        self.mask_format
    }

    pub fn hdr_color_format(&self) -> wgpu::TextureFormat {
        self.hdr_color_format
    }

    pub fn hdr_alpha_format(&self) -> wgpu::TextureFormat {
        self.hdr_alpha_format
    }

    pub fn msaa_format(&self) -> wgpu::TextureFormat {
        self.color_format
    }

    pub fn msaa_sample_count(&self) -> u32 {
        self.msaa_samples
    }
}
