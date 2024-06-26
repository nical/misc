use std::ops::Range;
use crate::buffer::{Buffer, UniformBufferPool};
use crate::gpu::ShaderSources;
use crate::tiling::TilePosition;

const STAGING_BUFFER_SIZE: u32 = 65536;

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct CpuMask {
    pub position: TilePosition,
    pub byte_offset: u32,
}

unsafe impl bytemuck::Pod for CpuMask {}
unsafe impl bytemuck::Zeroable for CpuMask {}


pub struct MaskUploadCopies {
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl MaskUploadCopies {
    pub fn new(device: &wgpu::Device, shaders: &mut ShaderSources, globals_bg_layout: &wgpu::BindGroupLayout) -> Self {
        create_mask_upload_pipeline(device, shaders, globals_bg_layout)
    }
}

fn create_mask_upload_pipeline(device: &wgpu::Device, shaders: &mut ShaderSources, globals_bg_layout: &wgpu::BindGroupLayout) -> MaskUploadCopies {
    let src = include_str!("../shaders/mask_upload_copy.wgsl");
    let module = shaders.create_shader_module(device, "mask upload copy", src, &[]);

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Mask upload copy"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(65536),
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Mask upload copy"),
        bind_group_layouts: &[&globals_bg_layout, &bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline_descriptor = wgpu::RenderPipelineDescriptor {
        label: Some("Mask upload copy"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: "vs_main",
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<CpuMask>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[
                    wgpu::VertexAttribute {
                        offset: 0,
                        format: wgpu::VertexFormat::Uint32,
                        shader_location: 0,
                    },
                    wgpu::VertexAttribute {
                        offset: 4,
                        format: wgpu::VertexFormat::Uint32,
                        shader_location: 1,
                    },
                ],
            }],
        },
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: "fs_main",
            targets: &[
                Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
            ],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            polygon_mode: wgpu::PolygonMode::Fill,
            front_face: wgpu::FrontFace::Ccw,
            strip_index_format: None,
            cull_mode: None,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: None,
        multiview: None,
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
    };

    let pipeline = device.create_render_pipeline(&pipeline_descriptor);

    MaskUploadCopies {
        pipeline,
        bind_group_layout,
    }
}

struct MaskUploadBatch {
    instances: Range<u32>,
    src_index: u32,
    dst_atlas: u32,
}


pub struct MaskUploader {
    pool: UniformBufferPool<u8>,

    batches: Vec<MaskUploadBatch>,
    copy_instances: Vec<CpuMask>,
    copy_instance_buffer: Option<wgpu::Buffer>,

    current_atlas: u32,
    current_instance_start: u32,

    pub current_mask_buffer: Buffer<u8>,

    masks_per_atlas: u32,
}

unsafe impl Send for MaskUploader {}

impl MaskUploader {
    pub fn new(device: *const wgpu::Device, bind_group_layout: *const wgpu::BindGroupLayout, atlas_size: u32) -> Self {
        MaskUploader {
            pool: UniformBufferPool::new(
                STAGING_BUFFER_SIZE,
                device,
                bind_group_layout,
            ),
            batches: Vec::new(),
            copy_instances: Vec::new(),
            copy_instance_buffer: None,
            current_atlas: 0,
            current_instance_start: 0,
            current_mask_buffer: Buffer::empty(),
            masks_per_atlas: (atlas_size * atlas_size) / (16 * 16),
        }
    }

    pub fn create_similar(&self) -> Self {
        MaskUploader {
            // TODO: we should use the same pool instead.
            pool: self.pool.create_similar(),
            batches: Vec::new(),
            copy_instances: Vec::new(),
            copy_instance_buffer: None,
            current_atlas: 0,
            current_instance_start: 0,
            current_mask_buffer: Buffer::empty(),
            masks_per_atlas: self.masks_per_atlas,
        }
    }

    pub fn reset(&mut self) {
        if self.current_mask_buffer.capacity() > 0 {
            self.pool.return_buffer(std::mem::replace(&mut self.current_mask_buffer, Buffer::empty()));
        }
        self.pool.reset();
        self.batches.clear();
        self.copy_instances.clear();
        self.current_atlas = 0;
        self.current_instance_start = 0;
        self.copy_instance_buffer = None;
    }

    pub fn new_mask(&mut self, position: TilePosition) -> Range<usize> {

        const TILE_SIZE: usize = 16;

        //let atlas_index = id / self.masks_per_atlas;
        let atlas_index = self.current_atlas;

        if atlas_index != self.current_atlas || self.current_mask_buffer.remaining_capacity() < TILE_SIZE * TILE_SIZE {
            let instance_end = self.copy_instances.len() as u32;
            if instance_end > self.current_instance_start {
                self.batches.push(MaskUploadBatch {
                    instances: self.current_instance_start .. instance_end,
                    src_index: self.current_mask_buffer.index(),
                    dst_atlas: self.current_atlas,
                });

                self.current_instance_start = instance_end;
            }

            if (self.current_mask_buffer.remaining_capacity() as u32) < STAGING_BUFFER_SIZE / 2 {
                let buf = std::mem::replace(&mut self.current_mask_buffer, self.pool.get_buffer());

                if buf.capacity() > 0 {
                    self.pool.return_buffer(buf);
                }
            }

            self.current_atlas = atlas_index;
        }

        let start = self.current_mask_buffer.len();
        let end = start + (TILE_SIZE * TILE_SIZE);

        self.copy_instances.push(CpuMask {
            position,
            byte_offset: start as u32,
        });

        start..end
    }

    pub fn needs_upload(&self) -> bool {
        !self.copy_instances.is_empty()
    }

    pub fn upload<'a, 'c, 'b: 'a>(
        &'b mut self,
        device: &'b wgpu::Device,
        pass: &'c mut wgpu::RenderPass<'a>,
        globals_bind_group: &'b wgpu::BindGroup,
        pipeline: &'b wgpu::RenderPipeline,
        quad_ibo: &'b wgpu::Buffer,
        mask_pass_index: u32,
    ) -> bool {
        use wgpu::util::DeviceExt;

        let instance_end = self.copy_instances.len() as u32;
        if instance_end > self.current_instance_start {
            self.batches.push(MaskUploadBatch {
                instances: self.current_instance_start .. instance_end,
                src_index: self.current_mask_buffer.index(),
                dst_atlas: self.current_atlas,
            });

            self.current_instance_start = instance_end;
        }

        if self.current_mask_buffer.capacity() > 0 {
            self.pool.return_buffer(std::mem::replace(&mut self.current_mask_buffer, Buffer::empty()));
        }

        if self.batches.is_empty() {
            return false;
        }

        if self.copy_instance_buffer.is_none() {
            self.copy_instance_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Mask copy instances"),
                contents: bytemuck::cast_slice(&self.copy_instances),
                usage: wgpu::BufferUsages::VERTEX,
            }));
        }

        let instances = self.copy_instance_buffer.as_ref().unwrap();

        let mut batches_start = 0;
        while batches_start < self.batches.len() {
            let current_atlas = self.batches[batches_start].dst_atlas;
            if current_atlas != mask_pass_index {
                batches_start += 1;
                continue;
            }

            pass.set_pipeline(pipeline);
            pass.set_index_buffer(quad_ibo.slice(..), wgpu::IndexFormat::Uint16);
            pass.set_vertex_buffer(0, instances.slice(..));
            pass.set_bind_group(0, globals_bind_group, &[]);

            let mut idx = batches_start;
            while idx < self.batches.len() && self.batches[idx].dst_atlas == current_atlas {
                let batch = &self.batches[idx];
                let bind_group = &self.pool.get_bind_group(batch.src_index);

                pass.set_bind_group(1, bind_group, &[]);
                pass.draw_indexed(0..6, 0, batch.instances.clone());

                idx += 1;
            }

            batches_start = idx;
        }

        //self.batches.clear();

        true
    }

    pub fn copy_instances(&self) -> &[CpuMask] {
        &self.copy_instances
    }
}

pub struct MaskDst {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
    pub texture: u32, // TODO
}

struct CopyInstance {
    src_offset: u32,
    src_stride: u32,
    dst_x: u16,
    dst_y: u16,
    width: u16,
    height: u16,
}

pub struct MappedBuffer {
    ptr: *mut u8,
    rem: usize,
    buffer: Option<wgpu::Buffer>,
}

pub struct SimpleUploader {
    current: MappedBuffer,
    used: Vec<wgpu::Buffer>,
    staging_buffer_size: usize,
    copy_instances: Vec<CopyInstance>
}

impl SimpleUploader {
    pub fn new (staging_buffer_size: usize) -> Self {
        SimpleUploader {
            current: MappedBuffer {
                ptr: std::ptr::null_mut(),
                rem: 0,
                buffer: None,
            },
            used: Vec::new(),
            staging_buffer_size,
            copy_instances: Vec::new(),
        }
    }

    pub fn write(
        &mut self, 
        device: &wgpu::Device,
        data: &[u8],
        dst: MaskDst,
    ) {
        let len = data.len();
        if self.current.rem < len {
            if data.len() > self.staging_buffer_size as usize {
                unimplemented!();
            }

            if let Some(buffer) = self.current.buffer.take() {
                buffer.unmap();
                self.used.push(buffer);
            }

            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("staging"),
                size: self.staging_buffer_size as u64,
                mapped_at_creation: true,
                usage: wgpu::BufferUsages::COPY_SRC
            });

            {
                let slice = buf.slice(..);
                let mut range = slice.get_mapped_range_mut();

                self.current.ptr = range.as_mut_ptr();
                self.current.rem = self.staging_buffer_size as usize;
            }

            self.current.buffer = Some(buf);
        }

        let offset = self.staging_buffer_size - self.current.rem;
        self.current.rem -= len;

        unsafe {
            let dst_ptr = self.current.ptr.offset(offset as isize);
            let dst = std::slice::from_raw_parts_mut(dst_ptr, len);
            dst.copy_from_slice(data);
        }

        self.copy_instances.push(CopyInstance {
            src_offset: offset as u32,
            src_stride: dst.w,
            dst_x: dst.x as u16,
            dst_y: dst.y as u16,
            width: dst.w as u16,
            height: dst.h as u16,
        });
    }
}