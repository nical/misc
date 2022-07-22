use std::ops::Range;
use std::collections::HashMap;

pub mod shader_list;
pub mod loader;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ShaderId(u32);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ShaderName(pub u32);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct PrimaryFeatures(pub u32);

impl std::ops::BitOr<PrimaryFeatures> for PrimaryFeatures {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        PrimaryFeatures(self.0 | rhs.0)
    }
}

impl std::ops::BitAnd<PrimaryFeatures> for PrimaryFeatures {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self {
        PrimaryFeatures(self.0 & rhs.0)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SecondaryFeatures(pub u32);


impl std::ops::BitOr<SecondaryFeatures> for SecondaryFeatures {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        SecondaryFeatures(self.0 | rhs.0)
    }
}

impl std::ops::BitAnd<SecondaryFeatures> for SecondaryFeatures {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self {
        SecondaryFeatures(self.0 & rhs.0)
    }
}

pub struct InitFlags(pub u32);

impl InitFlags {
    pub const CACHED: Self = InitFlags(1 << 0);
    pub const LAZY: Self = InitFlags(1 << 1);
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ShaderKey {
    pub shader: ShaderName,
    pub primary_features: PrimaryFeatures,
    pub secondary_features: SecondaryFeatures,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ShaderState {
    Disabled = 0,
    NotBuilt = 1,
    Building = 2,
    Built = 3,
}


pub struct Shader {
    secondary_features: SecondaryFeatures,
    state: ShaderState,
}

pub struct ShaderGroup {
    range: Range<usize>
}

pub struct Shaders {
    shaders: Vec<Shader>,
    groups: HashMap<(ShaderName, PrimaryFeatures), ShaderGroup>,
    low_priority_build_tasks: Vec<ShaderId>,
    high_priority_build_tasks: Vec<ShaderId>,
}

impl Shaders {
    pub fn get(&self, id: ShaderId) -> &Shader {
        &self.shaders[id.0 as usize]
    }

    pub fn get_id_for_rendering(&mut self, key: &ShaderKey) -> Option<ShaderId> {
        let range = self.groups.get(&(key.shader, key.primary_features))?.range.clone();

        let mut first_compatible = None;
        let mut first_built = None;
        for idx in range {
            let shader = &self.shaders[idx];

            if shader.secondary_features.0 & key.secondary_features.0 != key.secondary_features.0 {
                continue;
            }

            if shader.state != ShaderState::Built {
                if shader.state != ShaderState::Disabled && first_compatible.is_none() {
                    first_compatible = Some(idx);
                }
                continue;
            }

            first_built = Some(ShaderId(idx as u32));
            break;
        }

        if let Some(idx) = first_compatible {
            let shader = &mut self.shaders[idx];
            if shader.state == ShaderState::NotBuilt {
                shader.state = ShaderState::Building;
                let tasks = if first_built.is_none() {
                    &mut self.high_priority_build_tasks
                } else {
                    &mut self.low_priority_build_tasks
                };

                tasks.push(ShaderId(idx as u32));
            }

            return Some(ShaderId(idx as u32));
        }

        if first_built.is_some() {
            first_built
        } else {
            first_compatible.map(|idx| ShaderId(idx as u32))
        }
    }
}

pub struct ShaderDescriptor {
    pub name: ShaderName,
    pub file_name: &'static str,
    pub permutations: Vec<(PrimaryFeatures, SecondaryFeatures, InitFlags)>,
}
