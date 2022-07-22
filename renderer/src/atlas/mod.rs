use crate::types::units::*;

use euclid::{point2, default::Box2D};

pub use guillotiere::AtlasAllocator as GuillotineAllocator;
pub use guillotiere::AllocatorOptions as GuillotineAllocatorOptions;
pub use etagere::AllocatorOptions as ShelfAllocatorOptions;
pub use etagere::BucketedAtlasAllocator as BucketedShelfAllocator;
pub use etagere::AtlasAllocator as ShelfAllocator;

#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TextureId(pub u32);

/// ID of an allocation within a given allocator texture.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
pub struct AllocId(pub u32);

/// Common interface for a single atlas allocator surface.
pub trait AtlasAllocator {
    type Parameters;

    fn new(size: DeviceIntSize, parameters: &Self::Parameters) -> Self;
    fn allocate(&mut self, size: DeviceIntSize) -> Option<(AllocId, DeviceIntRect)>;
    fn deallocate(&mut self, id: AllocId);
    fn is_empty(&self) -> bool;
    fn allocated_space(&self) -> i32;
    fn dump_into_svg(&self, rect: &Box2D<f32>, output: &mut dyn std::io::Write) -> std::io::Result<()>;

    const PREFERS_SORTED_ALLOCATIONS: bool;
}

/// Common interface for an atlas allocator surface that can allocate into multiple surfaces
/// and grow the number of surfaces as needed.
pub trait AtlasAllocatorSet<TextureParameters> {
    /// Allocate a rectangle in one of the atlases.
    ///
    /// If needed, add a new allocator and call the provided callback.
    fn allocate(
        &mut self,
        size: DeviceIntSize,
        texture_alloc_cb: &mut dyn FnMut(DeviceIntSize, &TextureParameters) -> TextureId,
    ) -> (TextureId, AllocId, DeviceIntRect);

    fn try_allocate(
        &mut self,
        size: DeviceIntSize
    ) -> Option<(TextureId, AllocId, DeviceIntRect)>;

    fn deallocate(&mut self, texture_id: TextureId, alloc_id: AllocId);

    fn allocated_space(&self) -> i32;

    fn texture_parameters(&self) -> &TextureParameters;

    fn prefers_sorted_allocations(&self) -> bool { false }

    fn release_empty_textures<'l>(&mut self, texture_dealloc_cb: &'l mut dyn FnMut(TextureId));
}

#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
struct TextureUnit<Allocator> {
    allocator: Allocator,
    texture_id: TextureId,
}

#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
pub struct CacheTextures<Allocator: AtlasAllocator, TextureParameters> {
    units: Vec<TextureUnit<Allocator>>,
    size: DeviceIntSize,
    atlas_parameters: Allocator::Parameters,
    texture_parameters: TextureParameters,
}

impl<Allocator: AtlasAllocator, TextureParameters> CacheTextures<Allocator, TextureParameters> {
    pub fn new(
        size: DeviceIntSize,
        atlas_parameters: Allocator::Parameters,
        texture_parameters: TextureParameters,
    ) -> Self {
        CacheTextures {
            units: Vec::new(),
            size,
            atlas_parameters,
            texture_parameters,
        }
    }

    pub fn texture_size(&self) -> DeviceIntSize {
        self.size
    }

    pub fn try_allocate(
        &mut self,
        requested_size: DeviceIntSize,
    ) -> Option<(TextureId, AllocId, DeviceIntRect)> {
        // Try to allocate from one of the existing textures.
        for unit in &mut self.units {
            if let Some((alloc_id, rect)) = unit.allocator.allocate(requested_size) {
                return Some((unit.texture_id, alloc_id, rect));
            }
        }

        None
    }

    pub fn allocate(
        &mut self,
        requested_size: DeviceIntSize,
        texture_alloc_cb: &mut dyn FnMut(DeviceIntSize, &TextureParameters) -> TextureId,
    ) -> (TextureId, AllocId, DeviceIntRect) {
        if let Some(result) = self.try_allocate(requested_size) {
            return result;
        }

        // Need to create a new texture to hold the allocation.
        let texture_id = texture_alloc_cb(self.size, &self.texture_parameters);
        let unit_index = self.units.len();

        self.units.push(TextureUnit {
            allocator: Allocator::new(self.size, &self.atlas_parameters),
            texture_id,
        });

        let (alloc_id, rect) = self.units[unit_index]
            .allocator
            .allocate(requested_size)
            .unwrap();

        (texture_id, alloc_id, rect)
    }

    pub fn deallocate(&mut self, texture_id: TextureId, alloc_id: AllocId) {
        let unit = self.units
            .iter_mut()
            .find(|unit| unit.texture_id == texture_id)
            .expect("Unable to find the associated texture array unit");

        unit.allocator.deallocate(alloc_id);
    }

    pub fn release_empty_textures<'l>(&mut self, texture_dealloc_cb: &'l mut dyn FnMut(TextureId)) {
        self.units.retain(|unit| {
            if unit.allocator.is_empty() {
                texture_dealloc_cb(unit.texture_id);

                false
            } else{
                true
            }
        });
    }

    pub fn clear(&mut self, texture_dealloc_cb: &mut dyn FnMut(TextureId)) {
        for unit in self.units.drain(..) {
            texture_dealloc_cb(unit.texture_id);
        }
    }

    #[allow(dead_code)]
    pub fn dump_as_svg(&self, output: &mut dyn std::io::Write) -> std::io::Result<()> {
        use svg_fmt::*;

        let num_arrays = self.units.len() as f32;

        let text_spacing = 15.0;
        let unit_spacing = 30.0;
        // TODO: non-square textures.
        let texture_size = self.size.width as f32 / 2.0;

        let svg_w = unit_spacing * 2.0 + texture_size;
        let svg_h = unit_spacing + num_arrays * (texture_size + text_spacing + unit_spacing);

        writeln!(output, "{}", BeginSvg { w: svg_w, h: svg_h })?;

        // Background.
        writeln!(output,
            "    {}",
            rectangle(0.0, 0.0, svg_w, svg_h)
                .inflate(1.0, 1.0)
                .fill(rgb(50, 50, 50))
        )?;

        let mut y = unit_spacing;
        for unit in &self.units {
            writeln!(output, "    {}", text(unit_spacing, y, format!("{:?}", unit.texture_id)).color(rgb(230, 230, 230)))?;

            let rect = Box2D {
                min: point2(unit_spacing, y),
                max: point2(unit_spacing + texture_size, y + texture_size),
            };

            unit.allocator.dump_into_svg(&rect, output)?;

            y += unit_spacing + texture_size + text_spacing;
        }

        writeln!(output, "{}", EndSvg)
    }

    pub fn allocated_space(&self) -> i32 {
        let mut accum = 0;
        for unit in &self.units {
            accum += unit.allocator.allocated_space();
        }

        accum
    }

    pub fn allocated_textures(&self) -> usize {
        self.units.len()
    }
}

impl<Allocator: AtlasAllocator, TextureParameters> AtlasAllocatorSet<TextureParameters> 
for CacheTextures<Allocator, TextureParameters> {
    fn allocate(
        &mut self,
        requested_size: DeviceIntSize,
        texture_alloc_cb: &mut dyn FnMut(DeviceIntSize, &TextureParameters) -> TextureId,
    ) -> (TextureId, AllocId, DeviceIntRect) {
        self.allocate(requested_size, texture_alloc_cb)
    }

    fn try_allocate(
        &mut self,
        requested_size: DeviceIntSize,
    ) -> Option<(TextureId, AllocId, DeviceIntRect)> {
        self.try_allocate(requested_size)
    }

    fn deallocate(&mut self, texture_id: TextureId, alloc_id: AllocId) {
        self.deallocate(texture_id, alloc_id);
    }

    fn allocated_space(&self) -> i32 {
        self.allocated_space()
    }

    fn texture_parameters(&self) -> &TextureParameters {
        &self.texture_parameters
    }

    fn prefers_sorted_allocations(&self) -> bool {
        Allocator::PREFERS_SORTED_ALLOCATIONS
    }

    fn release_empty_textures<'l>(&mut self, texture_dealloc_cb: &'l mut dyn FnMut(TextureId)) {
        self.release_empty_textures(texture_dealloc_cb);
    }
}

impl AtlasAllocator for ShelfAllocator {
    type Parameters = ShelfAllocatorOptions;

    fn new(size: DeviceIntSize, options: &Self::Parameters) -> Self {
        ShelfAllocator::with_options(size.cast_unit(), options)
    }

    fn allocate(&mut self, size: DeviceIntSize) -> Option<(AllocId, DeviceIntRect)> {
        self.allocate(size.to_untyped()).map(|alloc| {
            (AllocId(alloc.id.serialize()), alloc.rectangle.cast_unit())
        })
    }

    fn deallocate(&mut self, id: AllocId) {
        self.deallocate(etagere::AllocId::deserialize(id.0));
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }

    fn allocated_space(&self) -> i32 {
        self.allocated_space()
    }

    fn dump_into_svg(&self, rect: &Box2D<f32>, output: &mut dyn std::io::Write) -> std::io::Result<()> {
        self.dump_into_svg(Some(&rect.to_i32().cast_unit()), output)
    }

    const PREFERS_SORTED_ALLOCATIONS: bool = true;
}

impl AtlasAllocator for BucketedShelfAllocator {
    type Parameters = ShelfAllocatorOptions;

    fn new(size: DeviceIntSize, options: &Self::Parameters) -> Self {
        BucketedShelfAllocator::with_options(size.cast_unit(), options)
    }

    fn allocate(&mut self, size: DeviceIntSize) -> Option<(AllocId, DeviceIntRect)> {
        self.allocate(size.to_untyped()).map(|alloc| {
            (AllocId(alloc.id.serialize()), alloc.rectangle.cast_unit())
        })
    }

    fn deallocate(&mut self, id: AllocId) {
        self.deallocate(etagere::AllocId::deserialize(id.0));
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }

    fn allocated_space(&self) -> i32 {
        self.allocated_space()
    }

    fn dump_into_svg(&self, rect: &Box2D<f32>, output: &mut dyn std::io::Write) -> std::io::Result<()> {
        self.dump_into_svg(Some(&rect.to_i32().cast_unit()), output)
    }

    const PREFERS_SORTED_ALLOCATIONS: bool = false;
}

impl AtlasAllocator for GuillotineAllocator {
    type Parameters = GuillotineAllocatorOptions;

    fn new(size: DeviceIntSize, options: &Self::Parameters) -> Self {
        GuillotineAllocator::with_options(size.cast_unit(), options)
    }

    fn allocate(&mut self, size: DeviceIntSize) -> Option<(AllocId, DeviceIntRect)> {
        self.allocate(size.to_untyped()).map(|alloc| {
            (AllocId(alloc.id.serialize()), alloc.rectangle.cast_unit())
        })
    }

    fn deallocate(&mut self, id: AllocId) {
        self.deallocate(guillotiere::AllocId::deserialize(id.0));
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }

    fn allocated_space(&self) -> i32 {
        unimplemented!();
    }

    fn dump_into_svg(&self, rect: &Box2D<f32>, output: &mut dyn std::io::Write) -> std::io::Result<()> {
        guillotiere::dump_into_svg(self, Some(&rect.cast_unit().to_i32()), output)
    }

    const PREFERS_SORTED_ALLOCATIONS: bool = true;
}

