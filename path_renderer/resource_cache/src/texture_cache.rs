use std::{collections::HashMap, num::NonZeroUsize, sync::Arc};

use etagere::{AllocId, AllocatorOptions, AtlasAllocator};
use lru::LruCache;

use core::units::{SurfaceIntPoint, SurfaceIntRect, SurfaceIntSize};
use crate::{Eviction, ImageData, ImageDescriptor};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TextureId(u32);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TextureCacheItemId(u32);

pub type Epoch = u64;
pub type Generation = u32;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Pressure {
    Low,
    High,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum TextureKind {
    Color,
    Alpha,
    HdrColor,
    HdrAlpha,
}

impl TextureKind {
    pub fn from_format(format: ImageFormat) -> Self {
        match format {
            ImageFormat::Rgba8 => TextureKind::Color,
            ImageFormat::Alpha8 => TextureKind::Alpha,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ImageFormat {
    Rgba8,
    Alpha8,
    // TODO
}

pub struct TextureCacheItem {
    texture_id: TextureId,
    texture_index: usize,
    alloc: AllocId,
    rectangle: SurfaceIntRect,
    generation: u32,
    last_used: Epoch,
    eviction: Eviction,
}

struct TextureAtlas {
    allocator: AtlasAllocator,
    id: TextureId,
}

pub struct TextureCache {
    // TODO: would it make more sense to place manually evicted
    // items in another container?
    lru: LruCache<TextureCacheItemId, TextureCacheItem>,
    current_frame: Epoch,
    target_area: f32,

    textures: Vec<TextureAtlas>,
    allocated_space: u32,
    next_item_id: u32,

    texture_size: SurfaceIntSize,
    texture_kind: TextureKind,
    allocator_options: AllocatorOptions,

    // TODO: per-frame gpu data addresses
}

impl TextureCache {
    pub fn new(
        max_items: usize,
        texture_size: SurfaceIntSize,
        texture_kind: TextureKind,
        allocator_options: AllocatorOptions,
        target_area: f32,
    ) -> Self {
        let lru = match NonZeroUsize::new(max_items) {
            Some(cap) => LruCache::new(cap),
            None => LruCache::unbounded(),
        };
        TextureCache {
            lru,
            current_frame: 0,
            target_area,
            textures: Vec::with_capacity(8),
            allocated_space: 0,
            next_item_id: 1,
            texture_size,
            texture_kind,
            allocator_options,
        }
    }

    pub fn touch(&mut self, id: TextureCacheItemId, size: Option<SurfaceIntSize>) -> Option<Generation> {
        match self.lru.get_mut(&id) {
            Some(item) => {
                item.last_used = self.current_frame;

                if let Some(size) = size {
                    // If the size changed, throw away the item, we'll realloate it.
                    if item.rectangle.size() != size {
                        self.textures[item.texture_index]
                            .allocator
                            .deallocate(item.alloc);

                        return None;
                    }
                }

                Some(item.generation)
            }
            None => {
                None
            }
        }
    }

    pub fn request(
        &mut self,
        id: &mut Option<TextureCacheItemId>,
        image: &ImageDescriptor,
        generation: Generation,
        eviction: Eviction,
        updates: &mut TextureCacheUpdates,
    ) {
        let mut is_update = false;
        if let Some(item_id) = id {
            // It is important to touch requested items in the LRU cache now before we update
            // texture texture cache. It enables us to do some evictions during the update.
            match self.touch(*item_id, Some(image.size)) {
                Some(current_gen) if current_gen == generation => {
                    // The item is already present and up to date, no need to add or update it.
                    return;
                }
                Some(_) => {
                    // The item exists in the cache but with a different generation,
                    // we need to upload the new version.
                    is_update = true;
                }
                None => {
                    // The item has been evicted from the cache.
                }
            }
        }

        let request_id = id.unwrap_or_else(|| self.generate_item_id());
        *id = Some(request_id);

        let dst_texture;
        let dst_position;
        if is_update {
            let item = self.lru.get_mut(&request_id).unwrap();
            item.generation = generation;
            dst_texture = item.texture_id;
            dst_position = item.rectangle.min;
        } else {
            // Try to allocate without adding textures. If it fails, evict up to 50 items and try
            // again, this time adding a new texture if need be.
            let (texture, texture_index, alloc_id, rectangle) = match self.try_allocate(image.size) {
                Some(result) => result,
                None => {
                    self.evict(50, Pressure::High);

                    let texture_size = self.texture_size;
                    let texture_kind = self.texture_kind;
                    let texture_alloc_cb = &mut || {
                        let texture_id = updates.id_generator.new_texture_id();
                        updates.updates.insert(texture_id, TextureUpdate {
                            allocate: Some((texture_size, texture_kind)),
                            delete: false,
                            uploads: Vec::new(),
                        });

                        texture_id
                    };

                    self.allocate(
                        image.size,
                        texture_alloc_cb,
                    )
                }
            };

            // The allocator may have returned a larger rect, but we want
            // to track exactly the requested bounds.
            let mut rectangle = rectangle.cast_unit();
            rectangle.set_size(image.size);

            self.lru.put(request_id, TextureCacheItem {
                texture_id: texture,
                texture_index,
                alloc: alloc_id,
                rectangle,
                generation,
                last_used: self.current_frame,
                eviction,
            });

            dst_texture = texture;
            dst_position = rectangle.min;
        }

        if let ImageData::Buffer(data) = &image.data {
            let texture_update = updates.updates
                .entry(dst_texture)
                .or_insert_with(TextureUpdate::new);

            texture_update.uploads.push(TextureUpload {
                data: data.clone(),
                size: image.size,
                dest: dst_position,
                stride: image.stride,
            });
        }
    }

    /// Call this method after having done the requests for the current frame.
    pub fn resolve_item(&self, id: TextureCacheItemId) -> Option<(TextureId, SurfaceIntRect)> {
        self.lru.peek(&id).map(|item| (item.texture_id, item.rectangle))
    }

    /// Evict up to `n` items.
    pub fn evict(&mut self, n: u32, pressure: Pressure) -> u32 {
        let (factor, frame_threshold) = match pressure {
            Pressure::Low => (0.7, 60),
            Pressure::High => (0.5, 1),
        };
        let target = self.target_area * factor;

        let mut i = 0;
        while i < n {
            let allocated = self.allocated_space as f32;
            if allocated < target {
                return i;
            }

            if let Some((id, item)) = self.lru.pop_lru() {
                let auto_eviction = item.eviction == Eviction::Manual;
                // TODO: handle eager eviction.

                if auto_eviction && item.last_used + frame_threshold <= self.current_frame {
                    self.textures[item.texture_index].allocator.deallocate(item.alloc);
                    i += 1;
                }

                // Can't evict, put it back.
                self.lru.put(id, item);

                if auto_eviction {
                    // No point continuing, all the remaining items are too young.
                    return i;
                }
            }
        }

        n
    }

    // Should be called once per frame at the beginning or end of the frame.
    pub fn maintain(&mut self, updates: &mut TextureCacheUpdates) {
        self.evict(10, Pressure::Low);

        self.textures.retain(|tex| {
            if tex.allocator.is_empty() {
                updates.updates.insert(tex.id, TextureUpdate::delete());

                false
            } else {
                true
            }
        });

        self.current_frame += 1;
    }

    pub fn evict_item(&mut self, id: TextureCacheItemId) {
        if let Some(item) = self.lru.pop(&id) {
            self.textures[item.texture_index].allocator.deallocate(item.alloc);
        }
    }

    fn try_allocate(
        &mut self,
        size: SurfaceIntSize,
    ) -> Option<(TextureId, usize, AllocId, SurfaceIntRect)> {
        // Try to allocate from one of the existing textures.
        for (index, texture) in self.textures.iter_mut().enumerate() {
            if let Some(alloc) = texture.allocator.allocate(size.cast_unit()) {
                return Some((texture.id, index, alloc.id, alloc.rectangle.cast_unit()));
            }
        }

        None
    }

    fn allocate(
        &mut self,
        size: SurfaceIntSize,
        texture_alloc_cb: &mut dyn FnMut() -> TextureId,
    ) -> (TextureId, usize, AllocId, SurfaceIntRect) {
        if let Some(result) = self.try_allocate(size) {
            return result;
        }

        let texture_id = texture_alloc_cb();
        let texture_index = self.textures.len();

        self.textures.push(TextureAtlas {
            allocator: AtlasAllocator::with_options(
                self.texture_size.cast_unit(),
                &self.allocator_options
            ),
            id: texture_id,
        });

        let alloc = self.textures[texture_index]
            .allocator
            .allocate(size.cast_unit())
            .unwrap();

        (texture_id, texture_index, alloc.id, alloc.rectangle.cast_unit())
    }

    fn generate_item_id(&mut self) -> TextureCacheItemId {
        let id = TextureCacheItemId(self.next_item_id);
        self.next_item_id += 1;

        id
    }
}

pub struct TextureCaches {
    pub color: TextureCache,
    pub alpha: TextureCache,
    pub hdr_color: TextureCache,
    pub hdr_alpha: TextureCache,
    // TODO: standalone textures
}

impl TextureCaches {
    pub fn index(_size: SurfaceIntSize, kind: TextureKind) -> usize {
        match kind {
            TextureKind::Color => { 0 }
            TextureKind::Alpha => { 1 }
            TextureKind::HdrColor => { 2 }
            TextureKind::HdrAlpha => { 3 }
            // TODO: Standalone textures
        }
    }

    pub fn maintain(&mut self, updates: &mut TextureCacheUpdates) {
        self.color.maintain(updates);
        self.alpha.maintain(updates);
        self.hdr_color.maintain(updates);
        self.hdr_alpha.maintain(updates);
    }
}

impl std::ops::Index<usize> for TextureCaches {
    type Output = TextureCache;
    fn index(&self, idx: usize) -> &TextureCache {
        [
            &self.color,
            &self.alpha,
            &self.hdr_color,
            &self.hdr_alpha,
        ][idx]
    }
}

impl std::ops::IndexMut<usize> for TextureCaches {
    fn index_mut(&mut self, idx: usize) -> &mut TextureCache {
        [
            &mut self.color,
            &mut self.alpha,
            &mut self.hdr_color,
            &mut self.hdr_alpha,
        ][idx]
    }
}

pub struct TextureCacheUpdates {
    pub updates: HashMap<TextureId, TextureUpdate>,
    pub id_generator: TextureIdGenerator,
}

impl TextureCacheUpdates {
    pub fn reset(&mut self) {
        self.updates.clear();
    }
}

pub struct TextureUpload {
    pub data: Arc<Vec<u8>>,
    pub size: SurfaceIntSize,
    pub dest: SurfaceIntPoint,
    pub stride: Option<u32>,
}

pub struct TextureUpdate {
    pub allocate: Option<(SurfaceIntSize, TextureKind)>,
    pub delete: bool,
    pub uploads: Vec<TextureUpload>,
}

impl TextureUpdate {
    pub fn new() -> Self {
        TextureUpdate {
            allocate: None,
            delete: false,
            uploads: Vec::new(),
        }
    }

    pub fn allocate(size: SurfaceIntSize, format: TextureKind) -> Self {
        TextureUpdate {
            allocate: Some((size, format)),
            delete: false,
            uploads: Vec::new(),
        }
    }

    pub fn delete() -> Self {
        TextureUpdate {
            allocate: None,
            delete: true,
            uploads: Vec::new(),
        }
    }
}

pub struct TextureIdGenerator {
    next: TextureId,
}

impl TextureIdGenerator {
    pub fn new() -> Self {
        TextureIdGenerator {
            next: TextureId(1),
        }
    }

    pub fn new_texture_id(&mut self) -> TextureId {
        let id = self.next;
        self.next.0 += 1;

        id
    }
}
