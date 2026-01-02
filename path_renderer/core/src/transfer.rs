use crate::BindingsId;
use crate::render_pass::PassRenderContext;
use crate::units::{SurfaceIntPoint, SurfaceIntSize};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ReadbackError {
    Unknown
}

pub struct ReadbackImage<'l> {
    pub data: &'l [u8],
    pub size: SurfaceIntSize,
    // pub format: ImageFormat,
}

pub type ImageReadbackCallback = Box<dyn FnOnce(Result<ReadbackImage, ReadbackError>) + Send>;

pub enum Transfer {
    TextureToTexture {
        src: BindingsId,
        dst: BindingsId,
        size: SurfaceIntSize,
        src_offset: SurfaceIntPoint,
        dst_offset: SurfaceIntPoint,
    },
    TextureToBuffer {
        src: BindingsId,
        dst: BindingsId,
        size: SurfaceIntSize,
        src_offset: SurfaceIntPoint,
        dst_offset: u64,
    },
    BufferToTexture {
        src: BindingsId,
        dst: BindingsId,
        size: SurfaceIntSize,
        src_offset: u64,
        dst_offset: SurfaceIntPoint,
    },
    BufferToBuffer {
        src: BindingsId,
        dst: BindingsId,
        size: u64,
        src_offset: u64,
        dst_offset: u64,
    },
    ReadbackTexture {
        src: BindingsId,
        size: SurfaceIntSize,
        src_offset: SurfaceIntPoint,
        callback: Option<ImageReadbackCallback>,
    }
}

impl Transfer {
    pub fn run(
        &mut self,
        transfers: &mut Transfers,
        ctx: &mut PassRenderContext,
        device: &wgpu::Device,
    ) {
        match self {
            Self::ReadbackTexture { src, size, src_offset, callback } => {
                let texture = ctx.bindings.resolve_texture(*src).unwrap();
                let format = texture.format();
                let bpp = format.block_copy_size(None).unwrap();

                let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("readback buffer"),
                    size: size.width as u64 * size.height as u64 * bpp as u64,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });

                ctx.encoder.copy_texture_to_buffer(
                    wgpu::TexelCopyTextureInfo {
                        texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d {
                            x: src_offset.x as u32,
                            y: src_offset.y as u32,
                            z: 0,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyBufferInfo {
                        buffer: &readback_buffer,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(size.width as u32 * bpp as u32),
                            rows_per_image: None,
                        },
                    },
                    wgpu::Extent3d {
                        width: size.width as u32,
                        height: size.height as u32,
                        depth_or_array_layers: 1,
                    },
                );

                transfers.pending_readbacks.push(PendingReadback {
                    buffer: readback_buffer,
                    size: *size,
                    format,
                    callback: callback.take(),
                });
            }
            _ => {
                todo!()
            }
        }
    }
}

struct PendingReadback {
    buffer: wgpu::Buffer,
    size: SurfaceIntSize,
    format: wgpu::TextureFormat,
    callback: Option<ImageReadbackCallback>,
}

pub struct Transfers {
    // TODO: move staging buffer pool here.
    // TODO: readback buffer pool.
    pending_readbacks: Vec<PendingReadback>,
}

impl Transfers {
    pub fn new() -> Self {
        Transfers {
            pending_readbacks: Vec::new(),
        }
    }

    pub fn end_frame(&mut self) {
        for readback in &mut self.pending_readbacks {
            if let Some(callback) = readback.callback.take() {
                let size = readback.size;
                let readback_buffer = readback.buffer.clone();
                readback.buffer.map_async(
                    wgpu::MapMode::Read,
                    ..,
                    move |result| {
                        match result {
                            Ok(()) => {
                                let view = readback_buffer.get_mapped_range(..);
                                let image = ReadbackImage {
                                    size,
                                    data: &view[..],
                                };
                                callback(Ok(image));
                            }
                            Err(_error) => {
                                callback(Err(ReadbackError::Unknown));
                            }
                        }
                    }
                );
            }
        }
    }
}
