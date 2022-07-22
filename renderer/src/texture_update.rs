use crate::types::units::*;
use crate::atlas::TextureId;
use crate::image_store::ImageFormat;

use std::collections::HashMap;
use std::sync::Arc;

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

#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct TextureParameters {
    pub format: ImageFormat,
}

pub struct TextureUpload {
    pub data: Arc<Vec<u8>>,
    pub size: DeviceIntSize,
    pub dest: DeviceIntPoint,
}

pub struct TextureUpdate {
    pub allocate: Option<(DeviceIntSize, TextureParameters)>,
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

    pub fn allocate(size: DeviceIntSize, format: ImageFormat) -> Self {
        TextureUpdate {
            allocate: Some((size, TextureParameters { format })),
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

pub struct TextureUpdateState {
    pub textures: HashMap<TextureId, TextureUpdate>,
    pub id_generator: TextureIdGenerator,
}

impl TextureUpdateState {
    pub fn new() -> Self {
        TextureUpdateState {
            textures: HashMap::new(),
            id_generator: TextureIdGenerator::new(),
        }
    }

    pub fn add_texture(&mut self, size: DeviceIntSize, format: ImageFormat) -> TextureId {
        let id = self.id_generator.new_texture_id();

        self.textures.insert(id, TextureUpdate::allocate(size, format));

        id
    }

    pub fn delete_texture(&mut self, id: TextureId) {
        self.textures.insert(id, TextureUpdate::delete());
    }
}
