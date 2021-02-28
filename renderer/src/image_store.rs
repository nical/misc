use crate::units::*;
use crate::texture_update::*;
use crate::texture_atlas::{TextureId, AtlasAllocatorSet, AllocId, CacheTextures, ShelfAllocator};

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use lru::LruCache;

#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ImageFormat {
    Rgba8,
    Alpha8,
}

#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ImageKey(pub u32);

#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
pub struct ImageItem {
    pub size: DeviceIntSize,
    pub data: Arc<Vec<u8>>,
    pub format: ImageFormat,
    pub texture_cache_id: Option<TextureCacheItemId>,
    pub generation: u32,
    pub cache_index: usize
}

pub struct CacheItemIdGenerator {
    next: TextureCacheItemId,
}

impl CacheItemIdGenerator {
    pub fn new() -> Self {
        CacheItemIdGenerator {
            next: TextureCacheItemId(1),
        }
    }

    pub fn new_texture_cache_item_id(&mut self) -> TextureCacheItemId {
        let id = self.next;
        self.next.0 += 1;

        id
    }
}

#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
pub struct ImageStore {
    images: HashMap<ImageKey, ImageItem>,
    requests: HashSet<ImageKey>,
    removed_images: Vec<(usize, TextureCacheItemId)>,
}

fn cache_index(size: DeviceIntSize, format: ImageFormat) -> usize {
    if size.width > 512 || size.height > 512 {
        return 0;
    }
    match format {
        ImageFormat::Rgba8 => { 1 }
        ImageFormat::Alpha8 => { 2 }
    }
}

impl ImageStore {
    pub fn new() -> Self {
        ImageStore {
            images: HashMap::new(),
            requests: HashSet::new(),
            removed_images: Vec::new(),
        }
    }

    pub fn add_image(
        &mut self,
        key: ImageKey,
        data: Arc<Vec<u8>>,
        size: DeviceIntSize,
        format: ImageFormat,
    ) {
        self.images.insert(key, ImageItem {
            size,
            data,
            format,
            texture_cache_id: None,
            generation: 0,
            cache_index: cache_index(size, format),
        });
    }

    pub fn update_image(
        &mut self,
        key: ImageKey,
        data: Arc<Vec<u8>>,
    ) {
        let image = self.images.get_mut(&key).unwrap();
        image.data = data;
        image.generation += 1;
    }

    pub fn remove_image(&mut self, key: ImageKey) {
        let image = self.images.remove(&key).unwrap();
        if let Some(id) = image.texture_cache_id {
            self.removed_images.push((image.cache_index, id));
        }
    }

    pub fn request(&mut self, key: ImageKey) {
        debug_assert!(self.images.contains_key(&key));
        self.requests.insert(key);
    }

    // TODO: should return the rect handle as well.
    pub fn resolve(&self, key: ImageKey, cache: &mut Cache) -> Option<TextureId> {
        let image = self.images.get(&key)?;
        let cache_item_id = image.texture_cache_id?;
        cache.texture_caches()[image.cache_index].resolve(cache_item_id)
    }

    fn flush_requests(
        &mut self,
        caches: &mut [&mut dyn TextureCache],
        ids: &mut CacheItemIdGenerator,
    ) {
        for (index, id) in self.removed_images.drain(..) {
            caches[index].mark_unused(id);
        }

        for key in self.requests.drain() {
            let item = self.images.get(&key).unwrap();
            let tex_cache = &mut caches[item.cache_index];

            let mut is_update = false;
            if let Some(id) = item.texture_cache_id {
                // It is important to touch requested items in the LRU cache now before we update
                // texture texture cache. It enables us to do some evictions during the update.
                match tex_cache.touch(id) {
                    Some(generation) if generation == item.generation => {
                        // The item is already present and up to date, no need to add or update it.
                        continue;
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

            let id = item.texture_cache_id.unwrap_or_else(|| ids.new_texture_cache_item_id());

            let request = TextureCacheRequest {
                id,
                generation: item.generation,
                size: item.size,
                data: Some(item.data.clone()),
                format: item.format,
            };

            if is_update {
                tex_cache.update_item(request);
            } else {
                tex_cache.insert_item(request);
            }
        }
    }
}

#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Pressure {
    Low,
    High,
}

#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TextureCacheItemId(u32);

#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
pub struct TextureCacheItem {
    pub texture: TextureId,
    pub alloc_id: AllocId,
    pub rectangle: DeviceIntRect,
    pub generation: u32,
    pub last_used: u64,
}

pub struct TextureCacheRequest {
    id: TextureCacheItemId,
    generation: u32,
    size: DeviceIntSize,
    data: Option<Arc<Vec<u8>>>,
    format: ImageFormat,
}

pub trait TextureCache {
    fn touch(&mut self, id: TextureCacheItemId) -> Option<u32>;
    fn update_item(&mut self, request: TextureCacheRequest);
    fn insert_item(&mut self, request: TextureCacheRequest);
    fn mark_unused(&mut self, id: TextureCacheItemId);
    // TODO: rect handle
    fn resolve(&self, id: TextureCacheItemId) -> Option<TextureId>;
}

#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
pub struct SharedTextureCache {
    lru: LruCache<TextureCacheItemId, TextureCacheItem>,
    requested_updates: Vec<TextureCacheRequest>,
    requested_insertions: Vec<TextureCacheRequest>,
    deallocations: Vec<(TextureId, AllocId)>,
    current_frame: u64,
    target_area: f32,
}

impl SharedTextureCache {
    pub fn new(target_area: f32) -> Self {
        SharedTextureCache {
            lru: LruCache::unbounded(),
            requested_updates: Vec::new(),
            requested_insertions: Vec::new(),
            deallocations: Vec::new(),
            current_frame: 0,
            target_area,
        }
    }

    pub fn touch(&mut self, id: TextureCacheItemId) -> Option<u32> {
        match self.lru.get_mut(&id) {
            Some(item) => {
                item.last_used = self.current_frame;
                Some(item.generation)
            }
            None => {
                None
            }
        }
    }

    pub fn resolve_texture_id(&self, id: TextureCacheItemId) -> Option<TextureId> {
        self.lru.peek(&id).map(|item| item.texture)
    }

    /// Call once per frame after all ImageStore::flush_requests invocations.
    pub fn update(
        &mut self,
        textures: &mut dyn AtlasAllocatorSet<TextureParameters>,
        updates: &mut TextureUpdateState,
    ) {
        for (texture_id, alloc_id) in self.deallocations.drain(..) {
            textures.deallocate(texture_id, alloc_id);
        }

        self.evict(10, Pressure::Low, textures);

        if textures.prefers_sorted_allocations() {
            self.requested_insertions.sort_by_key(|item| (-item.size.height, -item.size.width));
        }

        let mut requested_insertions = std::mem::take(&mut self.requested_insertions);
        for request in requested_insertions.drain(..) {
            let texture_alloc_cb = &mut |size: DeviceIntSize, params: &TextureParameters| {
                let texture_id = updates.id_generator.new_texture_id();
                updates.textures.insert(texture_id, TextureUpdate {
                    allocate: Some((size, *params)),
                    delete: false,
                    uploads: Vec::new(),
                });

                texture_id
            };

            // Try to allocate without adding textures. If it fails, evict up to 50 items and try
            // again, this time adding a new texture if need be.
            let (texture, alloc_id, rectangle) = match textures.try_allocate(request.size.cast_unit()) {
                Some(result) => result,
                None => {
                    self.evict(50, Pressure::High, textures);

                    textures.allocate(
                        request.size.cast_unit(),
                        texture_alloc_cb,
                    )                    
                }
            };

            self.lru.put(request.id, TextureCacheItem {
                texture,
                alloc_id,
                rectangle: rectangle.cast_unit(),
                generation: request.generation,
                last_used: self.current_frame,
            });

            if let Some(data) = request.data {
                let texture_update = updates.textures
                    .entry(texture)
                    .or_insert_with(TextureUpdate::new);

                texture_update.uploads.push(TextureUpload {
                    data,
                    size: request.size,
                    dest: rectangle.min.cast_unit(),
                });
            }
        }
 
        let mut requested_updates = std::mem::take(&mut self.requested_updates);
        for request in requested_updates.drain(..) {
            let item = self.lru.get_mut(&request.id).unwrap();
            item.generation = request.generation;

            if let Some(data) = request.data {
                let texture_update = updates.textures
                    .entry(item.texture)
                    .or_insert_with(TextureUpdate::new);

                texture_update.uploads.push(TextureUpload {
                    data,
                    size: request.size,
                    dest: item.rectangle.min,
                });        
            }
        }

        textures.release_empty_textures(&mut |id| {
            updates.textures.insert(id, TextureUpdate::delete());
        });

        self.current_frame += 1;
    }

    pub fn evict(
        &mut self,
        n: u32,
        pressure: Pressure,
        textures: &mut dyn AtlasAllocatorSet<TextureParameters>,
    ) -> u32 {
        let (factor, frame_threshold) = match pressure {
            Pressure::Low => (0.7, 60),
            Pressure::High => (0.5, 1),
        };
        let target = self.target_area * factor;
        for i in 0..n {
            let allocated = textures.allocated_space() as f32;
            if allocated < target {
                return i;
            }

            if let Some((id, item)) = self.lru.pop_lru() {
                if item.last_used + frame_threshold > self.current_frame {
                    // Can't evict something that's used this frame, put it back.
                    self.lru.put(id, item);
                    // No point continuing, all the remaining items are too young.
                    return i;
                }

                textures.deallocate(item.texture, item.alloc_id);
            }
        }

        n
    }

    pub fn mark_unused(&mut self, id: TextureCacheItemId) {
        if let Some(item) = self.lru.pop(&id) {
            self.deallocations.push((item.texture, item.alloc_id));
        }
    }

    pub fn clear_all(
        &mut self,
        textures: &mut dyn AtlasAllocatorSet<TextureParameters>,
        updates: &mut TextureUpdateState,
    ) {
        while let Some((_, item)) = self.lru.pop_lru() {
            textures.deallocate(item.texture, item.alloc_id);
        }

        textures.release_empty_textures(&mut |id| {
            updates.textures.insert(id, TextureUpdate::delete());
        });
    }
}

impl TextureCache for SharedTextureCache {
    fn touch(&mut self, id: TextureCacheItemId) -> Option<u32> {
        self.touch(id)
    }

    fn update_item(&mut self, request: TextureCacheRequest) {
        self.requested_updates.push(request);
    }

    fn insert_item(&mut self, request: TextureCacheRequest) {
        self.requested_insertions.push(request);
    }

    fn mark_unused(&mut self, id: TextureCacheItemId) {
        self.mark_unused(id);
    }

    fn resolve(&self, id: TextureCacheItemId) -> Option<TextureId> {
        self.lru.peek(&id).map(|item| { item.texture })
    }
}

#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
pub struct StandaloneTextureCacheItem {
    pub texture: TextureId,
    pub generation: u32,
    pub last_used: u64,
}

pub struct StandaloneTextureCache {
    lru: LruCache<TextureCacheItemId, StandaloneTextureCacheItem>,
    requested_updates: Vec<TextureCacheRequest>,
    requested_insertions: Vec<TextureCacheRequest>,
    deallocations: Vec<TextureId>,
    current_frame: u64,
}

impl StandaloneTextureCache {
    fn update(&mut self, updates: &mut TextureUpdateState) {
        for texture_id in self.deallocations.drain(..) {
            updates.textures.insert(texture_id, TextureUpdate::delete());
        }

        for request in self.requested_insertions.drain(..) {
            let texture = updates.id_generator.new_texture_id();

            self.lru.put(request.id, StandaloneTextureCacheItem {
                texture,
                generation: request.generation,
                last_used: self.current_frame,
            });

            updates.textures.insert(texture, TextureUpdate {
                allocate: Some((
                    request.size,
                    TextureParameters { format: request.format }
                )),
                delete: false,
                uploads: Vec::new(),
            });
        }

        for request in self.requested_updates.drain(..) {
            let item = self.lru.get_mut(&request.id).unwrap();
            item.generation = request.generation;

            if let Some(data) = request.data {
                let texture_update = updates.textures
                    .entry(item.texture)
                    .or_insert_with(TextureUpdate::new);

                texture_update.uploads.push(TextureUpload {
                    data,
                    size: request.size,
                    dest: point2(0, 0),
                });        
            }
        }
    
        self.current_frame += 1;
    }
}

impl TextureCache for StandaloneTextureCache {
    fn touch(&mut self, id: TextureCacheItemId) -> Option<u32> {
        match self.lru.get_mut(&id) {
            Some(item) => {
                item.last_used = self.current_frame;
                Some(item.generation)
            }
            None => {
                None
            }
        }
    }

    fn update_item(&mut self, request: TextureCacheRequest) {
        self.requested_updates.push(request);
    }

    fn insert_item(&mut self, request: TextureCacheRequest) {
        self.requested_insertions.push(request);
    }

    fn mark_unused(&mut self, id: TextureCacheItemId) {
        if let Some(item) = self.lru.pop(&id) {
            self.deallocations.push(item.texture);
        }
    }

    fn resolve(&self, id: TextureCacheItemId) -> Option<TextureId> {
        self.lru.peek(&id).map(|item| { item.texture })
    }
}

pub struct Resources {
    pub images: ImageStore,
    // glyphs: GlyphStore,
}

impl Resources {
    pub fn update(&mut self, cache: &mut Cache) {
        self.images.flush_requests(
            &mut [
                &mut cache.standalone_cache,
                &mut cache.rgba8_shared_cache,
                &mut cache.alpha8_shared_cache,
            ],
            &mut cache.ids,
        );
    }
}

pub struct Cache {
    pub standalone_cache: StandaloneTextureCache,
    pub rgba8_shared_cache: SharedTextureCache,
    pub alpha8_shared_cache: SharedTextureCache,

    pub rgba8_shared_textures: CacheTextures<ShelfAllocator, TextureParameters>,
    pub alpha8_shared_textures: CacheTextures<ShelfAllocator, TextureParameters>,

    ids: CacheItemIdGenerator,
}

impl Cache {
    pub fn update(&mut self, texture_updates: &mut TextureUpdateState) {
        self.rgba8_shared_cache.update(
            &mut self.rgba8_shared_textures,
            texture_updates,
        );

        self.alpha8_shared_cache.update(
            &mut self.alpha8_shared_textures,
            texture_updates,
        );

        self.standalone_cache.update(texture_updates);
    }

    fn texture_caches(&mut self) -> [&mut dyn TextureCache; 3] {
        [
            &mut self.standalone_cache,
            &mut self.rgba8_shared_cache,
            &mut self.alpha8_shared_cache,
        ]
    }
}
