use crate::{gpu::shader::{ShaderPatternId}};

// TODO: Pattern coordinate system:
// currently patterns take layout space coordinates which works well for things like gradients which are specified in that space.
// However it is not convenient for, say, sampling an image via a rectangle where we would instead prefer normalized image coordinates per vertex.
// - layout space is convenient for arbitray mesh geometry to provide in general
// - normalized and layout spaces are very easy for rects
// - The pattern rect could be specified as a layout-space axis-aligned rect that the vertex shader can use to build the normalized coordinates
//   - this is similar to how WR has the local rect which defines the bounds of the image and the local clip rect that allows selecting a sub-region.
//   - rotating the pattern would need to be done via a transform in the shader.
//   - the pattern would be responsible for owning that rect if it needs it (gradients don't).
//     - that allows passing the rect only if the pattern needs it, but it forces the rect to be bundled with other
//       data if any (for example atlas sample bounds). Glyphs would benefit from having atlas data
//       in one reusable place. Maybe use only 16 bits for the gpu store handles?


#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct BindingsId(u16);
impl BindingsId {
    pub const NONE: Self = BindingsId(std::u16::MAX);
    #[inline] pub fn is_none(&self) -> bool { *self == Self::NONE }
    #[inline] pub fn is_some(&self) -> bool { *self != Self::NONE }
    pub const fn from_index(idx: usize) -> Self {
        debug_assert!(idx < std::u16::MAX as usize - 1);
        BindingsId(idx as u16)
    }
    pub fn index(&self) -> usize {
        self.0 as usize
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct BuiltPattern {
    /// payload passed to the shaders.
    ///
    /// The interpretation of the value is shader-dependent. Usually an offset into a buffer
    /// to fetch more information, or the shader's parameter itself if it fits in 32 bits.
    pub data: u32,
    /// The type of pattern (color, gradient, etc.).
    pub shader: ShaderPatternId,
    /// Optional Id of a bindgroup to use when emitting the draw call.
    pub bindings: BindingsId,

    pub is_opaque: bool,
    pub can_stretch_horizontally: bool,
    pub favor_prerendering: bool,
}

impl BuiltPattern {
    #[inline]
    pub fn new(shader: ShaderPatternId, data: u32) -> Self {
        BuiltPattern { data, shader, bindings: BindingsId::NONE, is_opaque: false, can_stretch_horizontally: false, favor_prerendering: false }
    }

    #[inline]
    pub fn with_bindings(mut self, id: BindingsId) -> Self {
        self.bindings = id;
        self
    }

    #[inline]
    pub fn with_opacity(mut self, is_opaque: bool) -> Self {
        self.is_opaque = is_opaque;
        self
    }

    #[inline]
    pub fn with_horizontal_stretching(mut self, allow: bool) -> Self {
        self.can_stretch_horizontally = allow;
        self
    }

    #[inline]
    pub fn prerender_by_default(mut self) -> Self {
        self.favor_prerendering = true;
        self
    }

    #[inline]
    pub fn batch_key(&self) -> (ShaderPatternId, BindingsId) {
        (self.shader, self.bindings)
    }
}
