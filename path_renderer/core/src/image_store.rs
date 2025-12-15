use std::collections::{HashMap, HashSet};

use crate::units::{SurfaceIntRect, SurfaceIntSize};
use crate::texture_cache::*;

// TODO:
//  - texture uploads
//  - Tiled images
//  - Placeholder images
//  - per frame gpu data
//    - should it be managed and queried to each texture cache?
//    - the image store?
//    - some unified abstraction for image sources like WR's render
//      task graph?

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ImageId(u32);

pub struct ImageDescriptor {
    pub data: ImageData,
    pub format: ImageFormat,
    pub size: SurfaceIntSize,
    // In bytes.
    pub stride: Option<u32>,
    // In bytes.
    pub offset: u32,
}

pub struct ImageItem {
    pub descriptor: ImageDescriptor,
    pub texture_cache_id: Option<TextureCacheItemId>,
    pub generation: u32,
    pub eviction: Eviction,
    pub cache_index: usize
}

pub struct ImageStore {
    images: HashMap<ImageId, ImageItem>,
    removed_images: Vec<(usize, TextureCacheItemId)>,

    // Per-frame data.
    requests: HashSet<ImageId>,
}

impl ImageStore {
    pub fn new() -> Self {
        ImageStore {
            images: HashMap::with_capacity(256),
            removed_images: Vec::with_capacity(16),
            requests: HashSet::with_capacity(64),
        }
    }

    pub fn add(
        &mut self,
        id: ImageId,
        descriptor: ImageDescriptor,
        eviction: Eviction,
    ) {
        let size = descriptor.size;
        let tex_kind = TextureKind::from_format(descriptor.format);
        self.images.insert(id, ImageItem {
            descriptor,
            texture_cache_id: None,
            generation: 0,
            cache_index: TextureCaches::index(size, tex_kind),
            eviction,
        });
    }

    pub fn update(
        &mut self,
        id: ImageId,
        descriptor: ImageDescriptor,
    ) {
        let image = self.images.get_mut(&id).unwrap();
        image.descriptor = descriptor;
        image.generation += 1;
    }

    pub fn remove(&mut self, id: ImageId) {
        let image = self.images.remove(&id).unwrap();
        if let Some(id) = image.texture_cache_id {
            self.removed_images.push((image.cache_index, id));
        }
    }

    pub fn request(&mut self, key: ImageId) {
        debug_assert!(self.images.contains_key(&key));
        self.requests.insert(key);
    }

    pub fn flush_requests(
        &mut self,
        caches: &mut TextureCaches,
        updates: &mut TextureCacheUpdates,
    ) {
        for (index, id) in self.removed_images.drain(..) {
            caches[index].evict_item(id);
        }

        for key in self.requests.drain() {
            let item = self.images.get_mut(&key).unwrap();
            let tex_cache = &mut caches[item.cache_index];
            tex_cache.request(
                &mut item.texture_cache_id,
                &item.descriptor,
                item.generation,
                item.eviction,
                updates,
            );
        }
    }

    pub fn resolve(&self, id: ImageId, caches: &TextureCaches) -> Option<(TextureId, SurfaceIntRect)> {
        let image = self.images.get(&id)?;
        let tex_cache_id = image.texture_cache_id?;
        caches[image.cache_index].resolve_item(tex_cache_id)
    }
}

#[test]
fn simple_cache() {
    use etagere::AllocatorOptions;

    let tex_size = SurfaceIntSize::new(2048, 2048);
    let alloc_options = AllocatorOptions::default();
    let target_area = tex_size.area() as f32 * 0.9;

    let mut caches = TextureCaches {
        color: TextureCache::new(0, tex_size, TextureKind::Color, alloc_options, target_area),
        alpha: TextureCache::new(0, tex_size, TextureKind::Alpha, alloc_options, target_area),
        hdr_color: TextureCache::new(0, tex_size, TextureKind::HdrColor, alloc_options, target_area),
        hdr_alpha: TextureCache::new(0, tex_size, TextureKind::HdrAlpha, alloc_options, target_area),
        glyphs: TextureCache::new(0, tex_size, TextureKind::Glyphs, alloc_options, target_area),
    };

    let mut images = ImageStore::new();

    let image_data = ImageData::new_buffer(vec![
        // First image (offset 0)
        255, 255, 255, 255,  255, 255, 255, 255,
        255, 255, 255, 255,  255, 255, 255, 255,
        // Second image (offset 16)
        0, 0, 0, 255,  0, 0, 0, 255,
        0, 0, 0, 255,  0, 0, 0, 255,
    ]);

    let img1 = ImageId(1);
    images.add(
        img1,
        ImageDescriptor {
            data: image_data.clone(),
            format: ImageFormat::Rgba8,
            size: SurfaceIntSize::new(2, 2),
            offset: 0,
            stride: None,
        },
        Eviction::Auto,
    );

    let img2 = ImageId(2);
    images.add(
        img2,
        ImageDescriptor {
            data: image_data.clone(),
            format: ImageFormat::Rgba8,
            size: SurfaceIntSize::new(2, 2),
            offset: 16,
            stride: None,
        },
        Eviction::Auto,
    );


    let mut updates = TextureCacheUpdates {
        id_generator: TextureIdGenerator::new(),
        updates: HashMap::new(),
    };


    images.request(img1);

    images.flush_requests(&mut caches, &mut updates);
    images.resolve(img1, &caches).unwrap();
    caches.maintain(&mut updates);

    assert_eq!(updates.updates.len(), 1);


    updates.reset();

    images.flush_requests(&mut caches, &mut updates);
    images.resolve(img1, &caches).unwrap();
    caches.maintain(&mut updates);

    assert_eq!(updates.updates.len(), 0);



    updates.reset();

    images.request(img1);
    images.request(img2);

    images.flush_requests(&mut caches, &mut updates);
    images.resolve(img1, &caches).unwrap();
    images.resolve(img2, &caches).unwrap();
    caches.maintain(&mut updates);

    assert_eq!(updates.updates.len(), 1);



    updates.reset();

    images.remove(img1);
    images.remove(img2);

    images.flush_requests(&mut caches, &mut updates);
    caches.maintain(&mut updates);

    assert_eq!(updates.updates.len(), 1);
}
