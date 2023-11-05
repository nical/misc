use crate::TilePosition;
use std::ops::Range;

use core::gpu::{
    gpu_store::{DynBufferRange, DynamicStore},
    Shaders,
};

use core::bytemuck;
use core::wgpu;

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct CpuMask {
    pub position: TilePosition,
    pub byte_offset: u32,
}

unsafe impl bytemuck::Pod for CpuMask {}
unsafe impl bytemuck::Zeroable for CpuMask {}

use crate::BYTES_PER_MASK;

/*

The tiled atlas uploader is meant to efficiently upload a large number of small
mask tiles each frame. It does so by first linearly pushing them into a staging buffer,
which is copied into a gpu-visible intermediate buffer and the mask atlas is
then populated by via a draw call that reads from the intermediate buffer.

The use of an intermediate buffer causes an extra copy but makes it possible to copy many
tiles with a single batched draw call, as opposed to emmitting a buffer-to-texture copy
command per mask tile (which is typically slow passed a certain threshold).

Uploads don't need to happen at the beginning of the frame's gpu commands submission.
It is possible to alternate between transfering to and then sampling from a tile atlas,
as long as each of the alternating iterations have a separate dst_index.
*/

struct StagingBuffer {
    cpu_visible: wgpu::Buffer,
    mapped: *mut u8,
}

struct MaskUploadBatch {
    staging_range: Range<u32>, // In bytes.
    instances: Range<u32>,     // In instances.
    src_index: u32,
    dst_index: u32,
}

pub struct TileAtlasUploader {
    staging_buffers: Vec<StagingBuffer>,
    unused_staging_buffers: Vec<StagingBuffer>,
    staging_buffer_size: usize,
    current_staging_buffer_offset: usize,

    batches: Vec<MaskUploadBatch>,
    copy_instances: Vec<CpuMask>,
    current_dst: u32,
    current_instance_start: u32,
    staging_range_start: u32,

    // Where in the dynamic vertex buffers our data will be this frame.
    uploaded_copy_instances_range: Option<DynBufferRange>,
}

impl TileAtlasUploader {
    pub fn new(staging_buffer_size: u32) -> Self {
        TileAtlasUploader {
            staging_buffers: Vec::new(),
            unused_staging_buffers: Vec::new(),
            staging_buffer_size: staging_buffer_size as usize,
            current_staging_buffer_offset: staging_buffer_size as usize,

            batches: Vec::new(),
            copy_instances: Vec::new(),
            current_dst: 0,
            current_instance_start: 0,
            staging_range_start: 0,

            uploaded_copy_instances_range: None,
        }
    }

    pub fn num_tiles(&self) -> usize {
        self.copy_instances.len()
    }

    pub fn staging_buffer_size(&self) -> u32 {
        self.staging_buffer_size as u32
    }

    pub fn add_tile(
        &mut self,
        device: &wgpu::Device,
        dst_position: TilePosition,
        dst_index: u32,
    ) -> &mut [u8] {
        if dst_index != self.current_dst {
            self.flush_batch();
            self.current_dst = dst_index;
        }
        if dst_index != self.current_dst
            || self.staging_buffer_size < self.current_staging_buffer_offset + BYTES_PER_MASK
        {
            self.flush_batch();
            self.allocate_staging_buffer(device);
        }

        self.copy_instances.push(CpuMask {
            position: dst_position,
            byte_offset: self.current_staging_buffer_offset as u32,
        });

        let slice = unsafe {
            let ptr = self
                .staging_buffers
                .last()
                .unwrap()
                .mapped
                .add(self.current_staging_buffer_offset as usize);
            std::slice::from_raw_parts_mut(ptr, BYTES_PER_MASK)
        };

        self.current_staging_buffer_offset += BYTES_PER_MASK;

        slice
    }

    pub fn write_tile(
        &mut self,
        device: &wgpu::Device,
        data: &[u8],
        dst_position: TilePosition,
        dst_index: u32,
    ) {
        self.add_tile(device, dst_position, dst_index)
            .copy_from_slice(data);
    }

    pub fn flush_batch(&mut self) {
        let start = self.current_instance_start;
        let end = self.copy_instances.len() as u32;

        if start == end {
            return;
        }

        let staging_start = self.staging_range_start;
        let staging_end = self.current_staging_buffer_offset as u32;

        self.staging_range_start = self.current_staging_buffer_offset as u32;
        self.current_instance_start = end;

        let src_index = self.staging_buffers.len() as u32 - 1;

        self.batches.push(MaskUploadBatch {
            staging_range: staging_start..staging_end,
            instances: start..end,
            src_index,
            dst_index: self.current_dst,
        });
    }

    fn allocate_staging_buffer(&mut self, device: &wgpu::Device) {
        if let Some(buf) = self.unused_staging_buffers.pop() {
            self.staging_buffers.push(buf)
        }

        let size = self.staging_buffer_size as wgpu::BufferAddress;
        let cpu_visible = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging buffer (CPU visible)"),
            size,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::MAP_WRITE,
            mapped_at_creation: true,
        });
        let ptr = cpu_visible.slice(..).get_mapped_range_mut().as_mut_ptr();

        self.staging_buffers.push(StagingBuffer {
            cpu_visible,
            mapped: ptr,
        });

        self.staging_range_start = 0;
        self.current_staging_buffer_offset = 0;
    }

    pub fn unmap(&mut self) {
        for buffer in &mut self.staging_buffers {
            buffer.cpu_visible.unmap();
            buffer.mapped = std::ptr::null_mut();
        }
    }

    pub fn upload_vertices(&mut self, device: &wgpu::Device, vertices: &mut DynamicStore) {
        self.uploaded_copy_instances_range =
            vertices.upload(device, bytemuck::cast_slice(&self.copy_instances));
    }

    pub fn reset(&mut self) {
        self.copy_instances.clear();
        self.batches.clear();
        self.staging_buffers.clear(); // TODO recycle staging buffers.
        self.current_staging_buffer_offset = self.staging_buffer_size;
        self.current_dst = 0;
        self.current_instance_start = 0;
        self.staging_range_start = 0;
    }
}

pub struct MaskUploadCopies {
    pub intermediate_buffer: wgpu::Buffer,
    pub intermediate_buffer_bind_group: wgpu::BindGroup,
    pub pipeline: wgpu::RenderPipeline,
}

impl MaskUploadCopies {
    pub fn new(
        device: &wgpu::Device,
        shaders: &mut Shaders,
        staging_buffer_size: u32,
    ) -> MaskUploadCopies {
        create_mask_upload_pipeline(device, shaders, staging_buffer_size)
    }

    pub fn update_target(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        quad_ibo: &wgpu::Buffer,
        vertices: &DynamicStore,
        uploader: &TileAtlasUploader,
        dst_index: u32,
        dst: &wgpu::TextureView,
        dst_info: &wgpu::BindGroup,
    ) {
        let vbo_range = match uploader.uploaded_copy_instances_range.as_ref() {
            Some(range) => range,
            None => {
                return;
            }
        };

        let verts = vertices.get_buffer_slice(vbo_range);

        for batch in &uploader.batches {
            if batch.dst_index != dst_index {
                continue;
            }

            let src = &uploader.staging_buffers[batch.src_index as usize].cpu_visible;
            let offset = batch.staging_range.start as wgpu::BufferAddress;
            let copy_size = batch.staging_range.end as wgpu::BufferAddress - offset;
            //println!("copy range {offset} .. {}, ({copy_size})", offset + copy_size);
            encoder.copy_buffer_to_buffer(
                src,
                offset,
                &self.intermediate_buffer,
                offset,
                copy_size,
            );

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Tile atlas upload copy"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: dst,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            //println!("upload cpu masks {:?} instances ({} bytes)", batch.instances, (batch.instances.end - batch.instances.start) * BYTES_PER_MASK as u32);

            pass.set_pipeline(&self.pipeline);
            pass.set_index_buffer(quad_ibo.slice(..), wgpu::IndexFormat::Uint16);
            pass.set_vertex_buffer(0, verts);
            pass.set_bind_group(0, dst_info, &[]);
            pass.set_bind_group(1, &self.intermediate_buffer_bind_group, &[]);
            pass.draw_indexed(0..6, 0, batch.instances.clone());
        }
    }
}

fn create_mask_upload_pipeline(
    device: &wgpu::Device,
    shaders: &mut Shaders,
    staging_buffer_size: u32,
) -> MaskUploadCopies {
    let src = include_str!("../shaders/mask_upload_copy.wgsl");
    let module = shaders.create_shader_module(device, "mask upload copy", src, &[]);

    let target_and_gpu_store_layout = &shaders.get_base_bind_group_layout().handle;

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Mask upload copy"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(65536),
            },
            count: None,
        }],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Mask upload copy"),
        bind_group_layouts: &[&target_and_gpu_store_layout, &bind_group_layout],
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
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::R8Unorm,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
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

    let intermediate_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging buffer (GPU visible)"),
        size: staging_buffer_size as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let intermediate_buffer_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: intermediate_buffer.as_entire_binding(),
        }],
    });

    MaskUploadCopies {
        intermediate_buffer,
        intermediate_buffer_bind_group,
        pipeline,
    }
}
