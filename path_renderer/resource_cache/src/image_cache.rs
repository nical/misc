use std::fmt;

use core::units::SurfaceIntRect;

use crate::{Eviction, ImageDescriptor};
use crate::texture_cache::{TextureCacheItemId, TextureCacheUpdates, TextureCaches, TextureId, TextureKind};

// TODO:
//  - texture uploads
//  - Tiled images
//  - Placeholder images
//  - per frame gpu data
//    - should it be managed and queried to each texture cache?
//    - the image store?
//    - some unified abstraction for image sources like WR's render
//      task graph?

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct ImageId {
    index: u16,
    generation: u16,
}

impl ImageId {
    #[inline]
    fn index(&self) -> usize { self.index as usize }

    fn next_generation(self) -> Self {
        ImageId {
            index: self.index,
            generation: self.generation.wrapping_add(1),
        }
    }
}

impl fmt::Debug for ImageId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{}:{}", self.index, self.generation)
    }
}

pub(crate) struct ImageItem {
    pub descriptor: ImageDescriptor,
    pub texture_cache_id: Option<TextureCacheItemId>,
    pub version: u32,
    pub eviction: Eviction,
    pub cache_index: usize,
    pub id_generation: u16,
}


pub struct ImageCache {
    items: Vec<Option<ImageItem>>,
    free_slots: Vec<ImageId>,

    removed_images: Vec<(usize, TextureCacheItemId)>,
}

impl ImageCache {
    pub fn new() -> Self {
        ImageCache {
            items: Vec::new(),
            free_slots: Vec::new(),
            removed_images: Vec::new(),
        }
    }

    pub fn add(&mut self, descriptor: ImageDescriptor, eviction: Eviction) -> ImageId {
        let size = descriptor.size;
        let tex_kind = TextureKind::from_format(descriptor.format);
        let mut item = ImageItem {
            descriptor,
            texture_cache_id: None,
            version: 0,
            cache_index: TextureCaches::index(size, tex_kind),
            eviction,
            id_generation: 1,
        };

        match self.free_slots.pop() {
            Some(id) => {
                item.id_generation = id.generation;
                self.items[id.index()] = Some(item);
                id
            }
            None => {
                let gen = 1;
                let id = ImageId {
                    index: self.items.len() as u16,
                    generation: gen,
                };
                self.items.push(Some(item));

                id
            }
        }
    }

    pub fn remove(&mut self, id: ImageId) {
        if let Some(image) = self.items[id.index()].take() {
            debug_assert_eq!(image.id_generation, id.generation);
            if let Some(cache_id) = image.texture_cache_id {
                self.removed_images.push((image.cache_index, cache_id));
            }
            self.free_slots.push(id.next_generation());
        }
    }

    pub fn update(
        &mut self,
        id: ImageId,
        descriptor: ImageDescriptor,
    ) {
        let image = self.get_mut(id);
        image.descriptor = descriptor;
        image.version += 1;
    }

    pub fn create_requests(&self) -> ImageRequests {
        ImageRequests {
            items: vec![false; self.items.len()],
            count: 0,
        }
    }

    pub fn flush_requests(
        &mut self,
        requests: &[ImageRequests],
        caches: &mut TextureCaches,
        updates: &mut TextureCacheUpdates,
    ) {
        for (index, id) in self.removed_images.drain(..) {
            caches[index].evict_item(id);
        }

        for i in 0..self.items.len() {
            for req in requests {
                if req.items[i] {
                    let item = self.items[i].as_mut().unwrap();
                    let tex_cache = &mut caches[item.cache_index];
                    tex_cache.request(
                        &mut item.texture_cache_id,
                        &item.descriptor,
                        item.version,
                        item.eviction,
                        updates,
                    );

                    continue;
                }
            }
        }
    }

    /// Use after having called `flush_request` this frame.
    pub fn resolve(&self, id: ImageId, caches: &TextureCaches) -> Option<(TextureId, SurfaceIntRect)> {
        let image = self.get(id);
        let tex_cache_id = image.texture_cache_id?;
        caches[image.cache_index].resolve_item(tex_cache_id)
    }

    fn get(&self, id: ImageId) -> &ImageItem {
        let image = self.items[id.index()].as_ref().unwrap();
        debug_assert_eq!(image.id_generation, id.generation);

        image
    }

    fn get_mut(&mut self, id: ImageId) -> &mut ImageItem {
        let image = self.items[id.index()].as_mut().unwrap();
        debug_assert_eq!(image.id_generation, id.generation);

        image
    }
}

/// An object that contains image requests for a specific frame.
/// The object must be re-created each frame.
pub struct ImageRequests {
    items: Vec<bool>,
    count: usize,
}

impl ImageRequests {
    #[inline]
    pub fn request(&mut self, id: ImageId) {
        let slot = &mut self.items[id.index()];
        if !(*slot) {
            self.count += 1;
        }
        *slot = true;
    }

    pub fn num_requests(&self) -> usize {
        self.count
    }
}

#[test]
fn simple_cache() {
    use crate::ImageData;
    use crate::texture_cache::{ImageFormat, TextureCache, TextureIdGenerator};
    use core::units::SurfaceIntSize;
    use std::collections::HashMap;

    use etagere::AllocatorOptions;

    let tex_size = SurfaceIntSize::new(2048, 2048);
    let alloc_options = AllocatorOptions::default();
    let target_area = tex_size.area() as f32 * 0.9;

    let mut caches = TextureCaches {
        color: TextureCache::new(0, tex_size, TextureKind::Color, alloc_options, target_area),
        alpha: TextureCache::new(0, tex_size, TextureKind::Alpha, alloc_options, target_area),
        hdr_color: TextureCache::new(0, tex_size, TextureKind::HdrColor, alloc_options, target_area),
        hdr_alpha: TextureCache::new(0, tex_size, TextureKind::HdrAlpha, alloc_options, target_area),
    };

    let mut images = ImageCache::new();

    let image_data = ImageData::new_buffer(vec![
        // First image (offset 0)
        255, 255, 255, 255,  255, 255, 255, 255,
        255, 255, 255, 255,  255, 255, 255, 255,
        // Second image (offset 16)
        0, 0, 0, 255,  0, 0, 0, 255,
        0, 0, 0, 255,  0, 0, 0, 255,
    ]);

    let img1 = images.add(
        ImageDescriptor {
            data: image_data.clone(),
            format: ImageFormat::Rgba8,
            size: SurfaceIntSize::new(2, 2),
            offset: 0,
            stride: None,
        },
        Eviction::Auto,
    );

    let img2 = images.add(
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


    let mut requests = images.create_requests();
    requests.request(img1);

    images.flush_requests(&[requests], &mut caches, &mut updates);
    images.resolve(img1, &caches).unwrap();
    caches.maintain(&mut updates);

    assert_eq!(updates.updates.len(), 1);



    updates.reset();

    let requests = images.create_requests();
    images.flush_requests(&[requests], &mut caches, &mut updates);
    images.resolve(img1, &caches).unwrap();
    caches.maintain(&mut updates);

    assert_eq!(updates.updates.len(), 0);



    updates.reset();

    let mut req1 = images.create_requests();
    let mut req2 = images.create_requests();
    req1.request(img1);
    req2.request(img2);

    images.flush_requests(&[req1, req2], &mut caches, &mut updates);
    images.resolve(img1, &caches).unwrap();
    images.resolve(img2, &caches).unwrap();
    caches.maintain(&mut updates);

    assert_eq!(updates.updates.len(), 1);



    updates.reset();

    images.remove(img1);
    images.remove(img2);

    images.flush_requests(&[], &mut caches, &mut updates);
    caches.maintain(&mut updates);

    assert_eq!(updates.updates.len(), 1);


    let img3 = images.add(
        ImageDescriptor {
            data: image_data.clone(),
            format: ImageFormat::Rgba8,
            size: SurfaceIntSize::new(2, 2),
            offset: 0,
            stride: None,
        },
        Eviction::Auto,
    );

    // The new image id should be different from the removed ones...
    assert!(img3 != img1);
    assert!(img3 != img2);
    // ...but it should reuse one of their slots.
    assert!(img3.index == img1.index || img3.index == img2.index);
}
