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

/// Describes the requirements of a batch.
///
/// This is the means by which we decide how 
pub trait BatchKey: Clone {
    /// Create a dummy key.
    fn invalid() -> Self;

    /// Combine this key with another compatible key.
    fn combine(&mut self, other: &Self);

    /// Whether it is possible and relatively cheap to add to a certain batch.
    fn should_add_to_batch(&self, batch_key: &Self) -> bool;

    /// Whether it is possible (even if somewhat costly) to merge the two batches
    /// when attempting to reduce the number of draw calls.
    fn can_merge_batches(a: &Self, b: &Self) -> bool;
}


struct Batch<Key, Instance> {
    key: Key,
    instances: Vec<Instance>,
    cost: f32,
}

impl<Key: BatchKey, Instance> Batch<Key, Instance> {
    fn with_key(key: Key) -> Self {
        Batch {
            key,
            instances: Vec::new(),
            cost: 0.0,
        }
    }

    fn merge(&mut self, other: Self) {
        self.instances.extend(other.instances);
        self.key.combine(&other.key);
        self.cost += other.cost;
    }
}


#[test]
fn simple() {
    use euclid::rect;

    /// Per-instance data to send to the GPU.
    #[derive(Clone)]
    struct Instance;

    /// Bit sets describing features that will be required of the shader.
    ///
    /// The general idea is to have a few "über-shaders" that we can fall back
    /// to and some specialized shaders that the render can select, if they match
    /// the requested config
    #[derive(Copy, Clone, Debug, PartialEq, Hash)]
    pub struct ShaderFeatures {
        /// Features of the shader that are expensive to combine.
        ///
        /// In other words don't mix primitves with different primary features in the same
        /// batch unless we really need to reduce the number of draw calls.
        pub primary: u32,

        /// Features of the shader that are cheaper to combine.
        ///
        /// In other words we prefer to mix primitives with different secondary features
        /// over generating more draw calls.
        pub secondary: u32,
    }

    #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
    pub enum BlendMode {
        None,
        Alpha,
        // etc.
    }

    #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
    pub struct Key {
        shader: ShaderFeatures,
        blend_mode: BlendMode,
        textures: [u32; 4],
    }

    impl BatchKey for Key {
        fn invalid() -> Self {
            Key {
                shader: ShaderFeatures { primary: 0, secondary: 0 },
                blend_mode: BlendMode::None,
                textures: [0; 4],
            }
        }

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

        fn can_merge_batches(a: &Self, b: &Self) -> bool {
            a.textures == b.textures
                && a.blend_mode == b.blend_mode
                && shader_configuration_exists(a.shader.combined_with(b.shader))
        }
    }

    fn shader_configuration_exists(_shader: ShaderFeatures) -> bool {
        // In practive we'd need some logic here to avoid having to support every possible shader configuration.
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

    let mut batches = OrderedBatchList::new(&cfg);

    batches.add_instance(&Key::solid(), Instance, &rect(0.0, 0.0, 100.0, 100.0));
    batches.add_instance(&Key::image(), Instance, &rect(100.0, 0.0, 100.0, 100.0));
    batches.add_instance(&Key::text(), Instance, &rect(200.0, 0.0, 100.0, 100.0));
    batches.add_instance(&Key::solid(), Instance, &rect(10.0, 10.0, 10.0, 10.0));
    batches.add_instance(&Key::solid(), Instance, &rect(300.0, 0.0, 10.0, 10.0));
    batches.add_instance(&Key::text(), Instance, &rect(320.0, 0.0, 10.0, 10.0));

    batches.optimize(&cfg);
}
