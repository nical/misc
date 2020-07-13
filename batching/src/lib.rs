mod ordered;
mod order_independent;

pub type Rect = euclid::default::Rect<f32>;
pub type Point = euclid::default::Point2D<f32>;

pub use ordered::*;
pub use order_independent::*;

/// Some options to tune the behavior of the batching phase.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BatchingConfig {
    /// Don't aggressively merge batches until the number of batches is larger
    /// than this threshold.
    pub ideal_batch_count: usize,

    /// Don't aggressively merge batches if the sum of their cost is larger
    /// than this threshold.
    ///
    /// This allows reducing the likely hood that we'll use a very expensive
    /// shader for a very large amount of pixels.
    pub max_merge_cost: f32,

    /// Maximum amount of batches to go through when looking for a batch to
    /// assign a primitive to.
    pub max_lookback: usize,
}

pub trait Batch {
    type Key;
    type Instance;

    fn new(key: &Self::Key, instances: &[Self::Instance], rect: &Rect) -> Self;

    fn add_instances(&mut self, key: &Self::Key, instances: &[Self::Instance], rect: &Rect) -> bool;

    fn can_merge(&self, _other: &Self) -> bool { false }

    fn merge(&mut self, _other: &mut Self) -> bool { false }

    fn num_instances(&self) -> usize;
}

#[derive(Copy, Clone, Debug)]
pub struct Stats {
    pub num_batches: u32,
    pub num_instances: u32,
    pub hit_lookback_limit: u32,
}

impl Stats {
    pub fn combine(&self, other: &Self) -> Self {
        Stats {
            num_batches: self.num_batches + other.num_batches,
            num_instances: self.num_instances + other.num_instances,
            hit_lookback_limit: self.hit_lookback_limit + other.hit_lookback_limit,
        }
    }
}

#[test]
fn simple() {
    use euclid::rect;

    /// Per-instance data to send to the GPU.
    #[derive(Copy, Clone)]
    struct Instance;

    /// Bit sets describing features that will be required of the shader.
    ///
    /// The general idea is to have a few "über-shaders" that we can fall back
    /// to and some specialized shaders that the render can select, if they match
    /// the requested config.
    #[derive(Copy, Clone, Debug, PartialEq, Hash)]
    pub struct ShaderFeatures {
        /// Features of the shader that are expensive to combine.
        ///
        /// In other words don't mix primitives with different primary features in the same
        /// batch unless we really need to reduce the number of draw calls.
        pub primary: u32,

        /// Features of the shader that are cheaper to combine.
        ///
        /// In other words we prefer to mix primitives with different secondary features
        /// over generating more draw calls.
        pub secondary: u32,
    }

    impl ShaderFeatures {
        pub fn combined_with(&self, other: Self) -> Self {
            ShaderFeatures {
                primary: self.primary | other.primary,
                secondary: self.secondary | other.secondary,
            }
        }
    }

    #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
    pub enum BlendMode {
        None,
        Alpha,
        // etc.
    }

    #[derive(Copy, Clone, Debug, PartialEq, Hash)]
    pub struct Key {
        shader: ShaderFeatures,
        blend_mode: BlendMode,
        textures: [u32; 4],
    }

    impl Key {
        fn combine(&mut self, other: &Self) {
            debug_assert_eq!(self.blend_mode, other.blend_mode);
            debug_assert_eq!(self.textures, other.textures);
            self.shader = self.shader.combined_with(other.shader);
        }

        fn should_add_to_batch(&self, batch: &Self) -> bool {
            self.textures == batch.textures
                && self.blend_mode == batch.blend_mode
                && self.shader.primary == batch.shader.primary
        }
    }

    struct AlphaBatch {
        key: Key,
        instances: Vec<Instance>,
    }


    impl Batch for AlphaBatch {
        type Key = Key;
        type Instance = Instance;

        fn new(key: &Key, instances: &[Instance], _rect: &Rect) -> Self {
            AlphaBatch {
                key: *key,
                instances: instances.to_vec(),
            }
        }

        fn add_instances(&mut self, key: &Key, instances: &[Instance], _rect: &Rect) -> bool {
            if !key.should_add_to_batch(&self.key) {
                return false;
            }

            self.instances.extend_from_slice(instances);
            self.key.combine(&key);

            true
        }

        fn can_merge(&self, other: &Self) -> bool {
            self.key.textures == other.key.textures
                && self.key.blend_mode == other.key.blend_mode
                && shader_configuration_exists(self.key.shader.combined_with(other.key.shader))
        }

        fn num_instances(&self) -> usize { self.instances.len() }

        fn merge(&mut self, other: &mut Self) -> bool {
            if self.can_merge(other) {
                self.instances.extend(other.instances.drain(..));
                self.key.combine(&other.key);

                return true
            }

            false
        }
    }

    fn shader_configuration_exists(_shader: ShaderFeatures) -> bool {
        // In practice we'd need some logic here to avoid having to support every possible shader configuration.
        true
    }

    impl Key {
        fn default() -> Self {
            Key {
                shader: ShaderFeatures { primary: 0, secondary: 0 },
                textures: [0; 4],
                blend_mode: BlendMode::None
            }            
        }
        fn solid() -> Self { Key { shader: ShaderFeatures { primary: 1, secondary: 0 }, ..Self::default() } }
        fn image() -> Self { Key { shader: ShaderFeatures { primary: 2, secondary: 0 }, ..Self::default() } }
        fn text() -> Self { Key { shader: ShaderFeatures { primary: 4, secondary: 0 }, ..Self::default() } }
    }

    let cfg = BatchingConfig {
        ideal_batch_count: 10,
        max_lookback: 10,
        max_merge_cost: 1000.0,
    };

    let mut batches = OrderedBatchList::<AlphaBatch>::new(&cfg);

    batches.add_instance(&Key::solid(), Instance, &rect(0.0, 0.0, 100.0, 100.0));
    batches.add_instance(&Key::image(), Instance, &rect(100.0, 0.0, 100.0, 100.0));
    batches.add_instance(&Key::text(), Instance, &rect(200.0, 0.0, 100.0, 100.0));
    batches.add_instance(&Key::solid(), Instance, &rect(10.0, 10.0, 10.0, 10.0));
    batches.add_instance(&Key::solid(), Instance, &rect(300.0, 0.0, 10.0, 10.0));
    batches.add_instance(&Key::text(), Instance, &rect(320.0, 0.0, 10.0, 10.0));

    batches.optimize(&cfg);
}
