use crate::gpu::{DynBufferRange, DynamicStore};
use std::ops::Range;

use super::{AtlasIndex, Stats};

pub mod circle;
pub mod rect;

pub struct MaskEncoder {
    pub masks: Vec<u8>,
    masks_start: u32,
    masks_end: u32,
    render_passes: Vec<Range<u32>>,
    current_atlas: u32,
    buffer_range: Option<DynBufferRange>,
}

impl MaskEncoder {
    pub fn new() -> Self {
        MaskEncoder::with_size(2048)
    }

    pub fn with_size(bytes: usize) -> Self {
        MaskEncoder {
            masks: Vec::with_capacity(bytes),
            render_passes: Vec::with_capacity(16),
            masks_start: 0,
            masks_end: 0,
            current_atlas: 0,
            buffer_range: None,
        }
    }

    pub fn reset(&mut self) {
        self.masks.clear();
        self.render_passes.clear();
        self.masks_start = 0;
        self.masks_end = 0;
        self.current_atlas = 0;
        self.buffer_range = None;
    }

    pub fn end_render_pass(&mut self) {
        if self.masks_start == self.masks_end {
            return;
        }

        if self.render_passes.len() <= self.current_atlas as usize {
            self.render_passes.resize(self.current_atlas as usize + 1, 0..0);
        }
        self.render_passes[self.current_atlas as usize] = self.masks_start..self.masks_end;
        self.masks_start = self.masks_end;
        self.current_atlas += 1;
    }

    pub fn prerender_mask<T>(&mut self, atlas_index: AtlasIndex, mask: T) where T: bytemuck::Pod {
        if atlas_index != self.current_atlas {
            self.end_render_pass();
            self.current_atlas = atlas_index;
        }

        self.masks_end += 1;
        self.masks.extend_from_slice(bytemuck::bytes_of(&mask));
    }

    pub fn upload(&mut self, vertices: &mut DynamicStore, device: &wgpu::Device) {
        self.buffer_range = vertices.upload(device, bytemuck::cast_slice(&self.masks));
    }

    pub fn has_content(&self, atlas_index: AtlasIndex) -> bool {
        let idx = atlas_index as usize;
        self.render_passes.len() > idx && !self.render_passes[idx].is_empty()
    }

    pub fn buffer_and_instance_ranges(&self, atlas_index: AtlasIndex) -> Option<(&DynBufferRange, Range<u32>)> {
        if !self.has_content(atlas_index) {
            return None;
        }
        let buffer_range = self.buffer_range.as_ref().unwrap();
        let range = self.render_passes[atlas_index as usize].clone();

        Some((buffer_range, range))
    }

    pub fn update_stats(&self, stats: &mut Stats) {
        stats.gpu_mask_tiles += self.masks_end as usize;
    }
}

pub type FillMaskEncoder = MaskEncoder;
