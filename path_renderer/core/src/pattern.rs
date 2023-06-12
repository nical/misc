use crate::{gpu::shader::ShaderPatternId};

pub trait PatternRenderer {

}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct BuiltPattern {
    pub data: u32,
    pub shader: ShaderPatternId,
    pub is_opaque: bool,
    pub can_stretch_horizontally: bool,
    pub favor_prerendering: bool,
}

impl BuiltPattern {
    pub fn new(shader: ShaderPatternId, data: u32) -> Self {
        BuiltPattern { data, shader, is_opaque: false, can_stretch_horizontally: false, favor_prerendering: false }
    }

    pub fn opaque(mut self) -> Self {
        self.is_opaque = true;
        self
    }

    pub fn allow_stretching_horizontally(mut self) -> Self {
        self.can_stretch_horizontally = true;
        self
    }

    pub fn prerender_by_default(mut self) -> Self {
        self.favor_prerendering = true;
        self
    }
}
