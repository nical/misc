use std::ops::Range;
use std::collections::HashMap;
use std::sync::Arc;

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

pub enum ShaderState {
    Disabled,
    NotBuilt,
    Building,
    Built(wgpu::RenderPipeline),
    Error(String),
}

impl ShaderState {
    pub fn built(&self) -> Option<&wgpu::RenderPipeline> {
        match self {
            ShaderState::Built(pipeline) => Some(pipeline),
            _ => None,
        }
    }

    pub fn error(&self) -> Option<&str> {
        match self {
            ShaderState::Error(string) => Some(&string[..]),
            _ => None,
        }
    }
}

impl ShaderState {
    pub fn is_built(&self) -> bool {
        match self {
            ShaderState::Built(..) => true,
            _ => false,
        }
    }

    pub fn can_build(&self) -> bool {
        match self {
            ShaderState::NotBuilt => true,
            _ => false,
        }
    }

    pub fn is_error(&self) -> bool {
        match self {
            ShaderState::Error(..) => true,
            _ => false,
        }
    }

    pub fn is_disabled(&self) -> bool {
        match self {
            ShaderState::Disabled => true,
            _ => false,
        }
    }

}

pub type CreatePipelineFn = dyn Fn(&wgpu::Device, ShaderKey) -> Result<wgpu::RenderPipeline, String> + Send + Sync;

struct Shader {
    state: ShaderState,
    key: ShaderKey,
}

struct ShaderGroup {
    range: Range<usize>,
    create_fn: Arc<CreatePipelineFn>,
}

pub struct Shaders {
    shaders: Vec<Shader>,
    groups: HashMap<(ShaderName, PrimaryFeatures), ShaderGroup>,
    build_tasks: Vec<(ShaderId, parasol::handle::OwnedHandle<Result<wgpu::RenderPipeline, String>>)>,
}

impl Shaders {
    pub fn new() -> Shaders {
        Shaders { shaders: Vec::new(), groups: HashMap::default(), build_tasks: Vec::new() }
    }

    pub fn get(&self, id: ShaderId) -> &ShaderState {
        &self.shaders[id.0 as usize].state
    }

    pub fn get_best_id(&self, key: ShaderKey) -> Option<ShaderId> {
        let range = self.groups.get(&(key.shader, key.primary_features))?.range.clone();

        for idx in range {
            let shader = &self.shaders[idx];

            if shader.key.secondary_features.0 & key.secondary_features.0 != key.secondary_features.0 {
                continue;
            }

            return Some(ShaderId(idx as u32));
        }

        None
    }

    pub fn get_built_id(&self, key: ShaderKey) -> Option<ShaderId> {
        let range = self.groups.get(&(key.shader, key.primary_features))?.range.clone();

        for idx in range {
            let shader = &self.shaders[idx];

            if shader.key.secondary_features.0 & key.secondary_features.0 != key.secondary_features.0 {
                continue;
            }

            if shader.state.is_built() {
                return Some(ShaderId(idx as u32));
            }
        }

        None
    }

    pub fn register_group(
        &mut self,
        name: ShaderName,
        features: PrimaryFeatures,
        variants: &[SecondaryFeatures],
        create_fn: Arc<CreatePipelineFn>,
    ) {
        let start = self.shaders.len();
        for variant in variants {
            self.shaders.push(Shader {
                key: ShaderKey { primary_features: features, secondary_features: *variant, shader: name },
                state: ShaderState::NotBuilt,
            });
        }
        let end = self.shaders.len();
        self.groups.insert((name, features), ShaderGroup {
            range: start..end,
            create_fn,
        });
    }

    pub fn build_shader(&mut self, id: ShaderId, device: &wgpu::Device) -> &ShaderState {
        let mut shader = &mut self.shaders[id.0 as usize];
        let create_fn = &(*self.groups.get(&(shader.key.shader, shader.key.primary_features))
            .unwrap()
            .create_fn);

        shader.state = match (create_fn)(device, shader.key) {
            Ok(pipeline) => ShaderState::Built(pipeline),
            Err(string) => ShaderState::Error(string)
        };

        &shader.state
    }

    pub unsafe fn build_shader_async(
        &mut self,
        id: ShaderId,
        device: &wgpu::Device,
        par: &mut parasol::Context,
    ) {
        let device: &'static wgpu::Device = std::mem::transmute(device);

        let key = self.shaders[id.0 as usize].key;
        let create_fn = self.groups.get(&(key.shader, key.primary_features))
            .unwrap()
            .create_fn.clone();

        let handle = par.task().run(move|_, _| {
            create_fn(device, key)
        });

        self.build_tasks.push((id, handle));
    }

    pub fn poll_build_tasks(&mut self) {
        for idx in (0..self.build_tasks.len()).rev() {
            if self.build_tasks[idx].1.poll() {
                let (id, handle) = self.build_tasks.swap_remove(idx);
                self.shaders[id.0 as usize].state = match handle.resolve_assuming_ready() {
                    Ok(pipeline) => {
                        ShaderState::Built(pipeline)
                    }
                    Err(string) => {
                        println!("{:?}", string);
                        ShaderState::Error(string)
                    }
                }
            }
        }
    }
}

pub struct ShaderDescriptor {
    pub name: ShaderName,
    pub file_name: &'static str,
    pub permutations: Vec<(PrimaryFeatures, SecondaryFeatures, InitFlags)>,
}
