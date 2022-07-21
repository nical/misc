mod ordered;
mod order_independent;
mod store;

pub type Rect = euclid::default::Rect<f32>;
pub type Point = euclid::default::Point2D<f32>;

pub use ordered::*;
pub use order_independent::*;
pub use store::*;

// TODO: it would be nice if Batcher had the entire logic of adding instances, so that it
// could internally dispatch to an alpha or opaque batcher.
pub trait Batcher {
    fn add_to_existing_batch(
        &mut self,
        system_id: SystemId,
        rect: &Rect,
        callback: &mut impl FnMut(BatchIndex) -> bool,
    ) -> bool;

    fn add_batch(
        &mut self,
        batch: BatchId,
        rect: &Rect,
    );
}

pub trait BatchType {
    type Key: Clone;
    type Instance: Copy;

    fn is_compatible(&self, a: &Self::Key, b: &Self::Key) -> bool;
    fn combine_keys(&self, key: &Self::Key, other: &Self::Key) -> Self::Key;
    /// A rough approximation of the per-area-unit cost of a batch.
    fn cost(&self, _key: &Self::Key) -> f32 { 1.0 }
}

pub type SystemId = u32;
pub type BatchIndex = u32;

#[derive(Copy, Clone, Debug)]
pub struct BatchId {
    pub system: SystemId,
    pub index: BatchIndex,
}

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

#[derive(Copy, Clone, Debug)]
pub struct Stats {
    pub num_batches: u32,
    pub num_instances: u32,
    pub hit_lookback_limit: u32,
}

#[test]
fn simple_batch_store() {
    use euclid::rect;

    struct SolidColor;
    struct Image;
    struct Text;

    impl SolidColor {
        fn key() -> Key<()> {
            Key {
                shader: ShaderFeatures { primary: 1, secondary: 0 },
                blend_mode: BlendMode::Alpha,
                handles: (),
            }
        }
    }

    impl Image {
        fn key(src_texture: u32, mask_texture: u32) -> Key<[u32; 2]> {
            Key {
                shader: ShaderFeatures { primary: 1, secondary: 0 },
                blend_mode: BlendMode::Alpha,
                handles: [src_texture, mask_texture],
            }
        }
    }

    impl Text {
        fn key(atlas_texture: u32) -> Key<u32> {
            Key {
                shader: ShaderFeatures { primary: 1, secondary: 0 },
                blend_mode: BlendMode::Alpha,
                handles: atlas_texture,
            }
        }
    }

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
        Alpha,
        // etc.
    }

    #[derive(Copy, Clone, Debug, PartialEq, Hash)]
    pub struct Key<Handles> {
        shader: ShaderFeatures,
        blend_mode: BlendMode,
        handles: Handles,
    }

    impl<Handles: Copy> Key<Handles> {
        fn combine(&self, other: &Self) -> Self {
            Key { shader: self.shader.combined_with(other.shader), blend_mode: self.blend_mode, handles: self.handles }
        }
    }

    impl BatchType for SolidColor {
        type Key = Key<()>;
        type Instance = Instance;

        fn is_compatible(&self, a: &Self::Key, b: &Self::Key) -> bool {
            a.handles == b.handles
                && a.blend_mode == b.blend_mode
                && shader_configuration_exists(a.shader.combined_with(b.shader))
        }

        fn combine_keys(&self, key: &Self::Key, other: &Self::Key) -> Self::Key {
            key.combine(other)
        }
    }

    impl BatchType for Image {
        type Key = Key<[u32; 2]>;
        type Instance = Instance;

        fn is_compatible(&self, a: &Self::Key, b: &Self::Key) -> bool {
            a.handles == b.handles
                && a.blend_mode == b.blend_mode
                && shader_configuration_exists(a.shader.combined_with(b.shader))
        }

        fn combine_keys(&self, key: &Self::Key, other: &Self::Key) -> Self::Key {
            key.combine(other)
        }
    }

    impl BatchType for Text {
        type Key = Key<u32>;
        type Instance = Instance;

        fn is_compatible(&self, a: &Self::Key, b: &Self::Key) -> bool {
            a.handles == b.handles
                && a.blend_mode == b.blend_mode
                && shader_configuration_exists(a.shader.combined_with(b.shader))
        }

        fn combine_keys(&self, key: &Self::Key, other: &Self::Key) -> Self::Key {
            key.combine(other)
        }
    }

    fn shader_configuration_exists(_shader: ShaderFeatures) -> bool {
        // In practice we'd need some logic here to avoid having to support every possible shader configuration.
        true
    }

    let cfg = BatchingConfig {
        ideal_batch_count: 10,
        max_lookback: 10,
        max_merge_cost: 1000.0,
    };

    let mut batcher = OrderedBatcher::new(&cfg);

    let mut solid_color_batches = BatchStore::new(SolidColor, &cfg, 0);
    let mut image_batches = BatchStore::new(Image, &cfg, 1);
    let mut text_batches = BatchStore::new(Text, &cfg, 2);

    solid_color_batches.add_instance(&mut batcher, &SolidColor::key(), Instance, &rect(0.0, 0.0, 100.0, 100.0));
    image_batches.add_instance(&mut batcher, &Image::key(0, 0), Instance, &rect(100.0, 0.0, 100.0, 100.0));
    text_batches.add_instance(&mut batcher, &Text::key(0), Instance, &rect(200.0, 0.0, 100.0, 100.0));
    solid_color_batches.add_instance(&mut batcher, &SolidColor::key(), Instance, &rect(10.0, 10.0, 10.0, 10.0));
    solid_color_batches.add_instance(&mut batcher, &SolidColor::key(), Instance, &rect(300.0, 0.0, 10.0, 10.0));
    text_batches.add_instance(&mut batcher, &Text::key(0), Instance, &rect(320.0, 0.0, 10.0, 10.0));
}

