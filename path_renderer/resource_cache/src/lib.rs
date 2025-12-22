use core::units::SurfaceIntSize;
use std::sync::Arc;

use crate::texture_cache::ImageFormat;

pub mod texture_cache;
pub mod image_cache;

#[derive(Clone)]
pub enum ImageData {
    Buffer(Arc<Vec<u8>>),
    External(u64),
    Render,
}

impl ImageData {
    pub fn new_buffer(data: Vec<u8>) -> Self {
        ImageData::Buffer(Arc::new(data))
    }
}

pub struct ImageDescriptor {
    pub data: ImageData,
    pub format: ImageFormat,
    pub size: SurfaceIntSize,
    // In bytes.
    pub stride: Option<u32>,
    // In bytes.
    pub offset: u32,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Eviction {
    Auto,
    Eager,
    Manual,
}
