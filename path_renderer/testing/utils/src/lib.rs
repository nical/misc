use futures::executor::block_on;

pub use core;
use core::wgpu;

pub mod args;

use core::{Instance};

#[derive(Copy, Clone, Debug)]
pub struct Reftest {
    pub max_difference: u32,
    pub max_differing_pixels: u32,
}

pub struct Image {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
    // pub format: Format,
}

pub fn read_back(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
) -> Image {
    let mut cmd_buf = device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    let size = texture.size();
    let bpp = 4; // TODO support other image formats.

    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback buffer"),
        size: size.width as u64 * size.height as u64 * bpp,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    cmd_buf.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size.width * bpp as u32),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width: size.width,
            height: size.height,
            depth_or_array_layers: 1,
        },
    );

    queue.submit(Some(cmd_buf.finish()));

    let readback_buffer_slice = readback_buffer.slice(..);
    readback_buffer_slice.map_async(wgpu::MapMode::Read, |_| ());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

    let bytes = readback_buffer_slice.get_mapped_range().to_vec();

    Image {
        width: size.width as u32,
        height: size.height as u32,
        data: bytes,
    }
}

pub struct ImageComparison {
    // Number of pixels in the with a a difference greater or equal to
    // the index.
    pub differences: [u32; 255],
}

pub fn compare_images(
    img1: &Image,
    img2: &Image,
    check_alpha: bool,
) -> ImageComparison {
    let mut result = ImageComparison {
        differences: [0; 255],
    };

    let iter1 = img1.data.chunks(4);
    let iter2 = img2.data.chunks(4);
    for (p1, p2) in iter1.zip(iter2) {
        let mut diff = 0;
        diff = diff.max((p1[0] as i16 - p2[0] as i16).abs());
        diff = diff.max((p1[1] as i16 - p2[1] as i16).abs());
        diff = diff.max((p1[2] as i16 - p2[2] as i16).abs());
        if check_alpha {
            diff = diff.max((p1[4] as i16 - p2[4] as i16).abs());
        }
        diff = diff.min(255);

        result.differences[diff as usize] += 1;
    }

    let mut sum = 0;
    for diff in result.differences.iter_mut().rev() {
        *diff += sum;
        sum = *diff;
    }

    result
}

pub fn check_comparison(
    comparison: &ImageComparison,
    requirements: &[Reftest],
    extra_fuzz: u32,
) -> Result<(), Reftest> {
    if requirements.is_empty() {
        return check_comparison(
            comparison,
            &[Reftest {
                max_difference: 1 + extra_fuzz,
                max_differing_pixels: 0,
            }],
            0,
        )
    }

    for req in requirements {
        let idx = (req.max_difference + extra_fuzz).min(255).max(1) as usize;
        if comparison.differences[idx] > req.max_differing_pixels {
            return Err(*req);
        }
    }

    Ok(())
}

pub fn init(backends: wgpu::Backends) -> core::Instance {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends,
        ..wgpu::InstanceDescriptor::default()
    });

    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: false,
    })).unwrap();


    let (device, queue) = block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: None,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            required_features: wgpu::Features::default(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
        },
    )).unwrap();

    Instance::new(&device, &queue, 0, false)
}
