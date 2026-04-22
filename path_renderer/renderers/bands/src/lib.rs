pub extern crate core;

mod renderer;
mod resources;

pub use renderer::*;
pub use resources::*;

#[derive(Copy, Clone, Debug)]
pub enum AaMode {
    /// High quality and fast, has conflation artifacts.
    AreaCoverage,
    /// Low quality, prevents conflation artifacts.
    Ssaa4,
}

#[derive(Clone, Debug)]
pub struct BandsOptions {
    pub antialiasing: AaMode,
}

impl Default for BandsOptions {
    fn default() -> Self {
        BandsOptions {
            antialiasing: AaMode::AreaCoverage,
        }
    }
}
