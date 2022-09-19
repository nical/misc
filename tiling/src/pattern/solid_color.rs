use crate::{Color};
use crate::tiling::{TilePosition, PatternData};
use crate::custom_pattern::*;

pub type SolidColorBuilder = CustomPatternBuilder<SolidColor>;
pub type SolidColorRenderer = CustomPatternRenderer<SolidColor>;

pub struct SolidColor {
    pattern_data: u32,
    is_opaque: bool,
}

impl SolidColor {
    pub fn new(color: Color) -> Self {
        SolidColor {
            pattern_data: color.to_u32(),
            is_opaque: color.is_opaque(),
        }
    }

    pub fn new_renderer(
        device: &wgpu::Device,
        helper: &mut CustomPatterns,
    ) -> CustomPatternRenderer<Self> {
        let descriptor = CustomPatternDescriptor {
            name: "solid color",
            source: include_str!("../../shaders/pattern/solid_color.wgsl"),
            varyings: &[
                Varying { name: "color", kind: "vec4<f32>", interpolate: "flat" },
            ],
            extra_bind_groups: &[],
        };

        CustomPatternRenderer::new(device, helper, &descriptor, ())
    }
}

impl CustomPattern for SolidColor {
    type RenderData = ();

    fn is_opaque(&self) -> bool { self.is_opaque }

    fn is_mergeable(&self) -> bool { true }

    fn new_tile(&mut self, _pattern_position: TilePosition) -> PatternData {
        self.pattern_data
    }
}
