use crate::gpu::{GpuBufferDescriptor, GpuBufferResources, GpuStreamsDescritptor, GpuStreamsResources, StagingBufferPoolRef};
use crate::graph::TempResourceKey;
use crate::shading::Shaders;
use std::u32;
use wgpu::util::DeviceExt;

pub struct GpuResources {
    pub common: CommonGpuResources,
    pub graph: RenderGraphResources,
}

impl GpuResources {
    pub fn new(
        device: &wgpu::Device,
        staging_buffers: StagingBufferPoolRef,
    ) -> Self {
        let common = CommonGpuResources::new(device, staging_buffers);
        let graph = RenderGraphResources::new();

        GpuResources {
            common,
            graph,
        }
    }

    pub fn upload(&mut self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        shaders: &Shaders,
        allocations: &[TempResourceKey],
    ) {
        self.common.upload(device, shaders);
        self.graph.upload(device, shaders, allocations);
    }

    pub fn begin_rendering(&mut self, encoder: &mut wgpu::CommandEncoder) {
        self.common.begin_rendering(encoder);
    }

    pub fn end_frame(&mut self) {
        self.common.end_frame();
        self.graph.end_frame();
    }
}

pub struct CommonGpuResources {
    pub quad_ibo: wgpu::Buffer,
    pub vertices: GpuBufferResources,
    pub indices: GpuStreamsResources,
    pub instances: GpuStreamsResources,

    pub default_sampler: wgpu::Sampler,
    pub f32_buffer: GpuBufferResources,
    pub u32_buffer: GpuBufferResources,
    pub base_bind_group: Option<wgpu::BindGroup>,

    pub staging_buffers: StagingBufferPoolRef,

    f32_buffer_epoch: u32,
    u32_buffer_epoch: u32,
}

impl CommonGpuResources {
    pub fn new(
        device: &wgpu::Device,
        staging_buffers: StagingBufferPoolRef,
    ) -> Self {
        let quad_indices = [0u16, 1, 2, 0, 2, 3];
        let quad_ibo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Quad indices"),
            contents: bytemuck::cast_slice(&quad_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let mut f32_buffer = GpuBufferResources::new(&GpuBufferDescriptor::rgba32_float_texture("f32 buffer"));
        f32_buffer.allocate(4096 * 1024, device);

        let mut u32_buffer = GpuBufferResources::new(&GpuBufferDescriptor::rgba32_uint_texture("u32 buffer"));
        u32_buffer.allocate(4096 * 1024, device);

        let vertices = GpuBufferResources::new(&GpuBufferDescriptor::Buffers {
            usages: wgpu::BufferUsages::VERTEX,
            min_size: 1024 * 128,
            default_alignment: 16,
            label: Some("vertices"),
        });

        let indices = GpuStreamsResources::new(&GpuStreamsDescritptor {
            usages: wgpu::BufferUsages::INDEX,
            buffer_size: 1024 * 16,
            chunk_size: 1024 * 2,
            label: Some("indices"),
        });

        let instances = GpuStreamsResources::new(&GpuStreamsDescritptor {
            usages: wgpu::BufferUsages::VERTEX,
            buffer_size: 1024 * 128,
            chunk_size: 1024 * 2,
            label: Some("instances"),
        });

        let default_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("default sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: 0.0,
            compare: None,
            anisotropy_clamp: 1,
            border_color: None,
        });

        CommonGpuResources {
            quad_ibo,
            vertices,
            indices,
            instances,
            f32_buffer,
            u32_buffer,
            default_sampler,
            base_bind_group: None,
            staging_buffers,
            f32_buffer_epoch: u32::MAX,
            u32_buffer_epoch: u32::MAX,
        }
    }

    fn upload(&mut self, device: &wgpu::Device, shaders: &Shaders) {
        if self.f32_buffer_epoch != self.f32_buffer.epoch() || self.u32_buffer_epoch != self.u32_buffer.epoch() {
            self.f32_buffer_epoch = self.f32_buffer.epoch();
            self.u32_buffer_epoch = self.u32_buffer.epoch();
            // If the gpu store texture ch_anges, we have to re-create the bind groups.
            self.base_bind_group = None;
        }

        if self.base_bind_group.is_none() {
            let target_and_gpu_buffers_layout = shaders.get_bind_group_layout(shaders.common_bind_group_layouts.target_and_gpu_buffer);

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Pass descriptor & gpu buffers"),
                layout: &target_and_gpu_buffers_layout.handle,
                entries: &[
                    self.f32_buffer.as_bind_group_entry(0).unwrap(),
                    self.u32_buffer.as_bind_group_entry(1).unwrap(),
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.default_sampler),
                    },
                ],
            });

            self.base_bind_group = Some(bind_group);
        }
    }

    // Should be called only between upload and end_frame.
    pub fn get_base_bindgroup(&self) -> &wgpu::BindGroup {
        &self.base_bind_group.as_ref().unwrap()
    }

    fn begin_rendering(&mut self, _encoder: &mut wgpu::CommandEncoder) {
    }

    fn end_frame(&mut self) {
    }
}

pub struct GpuResource {
    pub key: TempResourceKey,
    pub as_input: Option<wgpu::BindGroup>,
    pub as_attachment: Option<wgpu::TextureView>,
}

// TODO: the name isn't great.
pub struct RenderGraphResources {
    pool: Vec<GpuResource>,
    resources: Vec<GpuResource>,
}

impl RenderGraphResources {
    pub fn new() -> Self {
        RenderGraphResources {
            pool: Vec::new(),
            resources: Vec::new(),
        }
    }

    pub fn upload(
        &mut self,
        device: &wgpu::Device,
        shaders: &Shaders,
        allocations: &[TempResourceKey],
    ) {
        for key in allocations {
            let mut pool_idx = self.pool.len();
            for (idx, res) in self.pool.iter().enumerate() {
                if res.key == *key {
                    pool_idx = idx;
                    break;
                }
            }
            let resource = if pool_idx < self.pool.len() {
                self.pool.swap_remove(pool_idx)
            } else {
                if key.kind.is_texture() {
                    self.allocate_texture(device, shaders, *key)
                } else {
                    unimplemented!()
                }
            };

            self.resources.push(resource);
        }
    }

    fn allocate_texture(&self,
        device: &wgpu::Device,
        shaders: &Shaders,
        key: TempResourceKey,
    ) -> GpuResource {
        let kind = key.kind.as_texture().unwrap();

        println!("allocate texture {key:?}");

        let mut usage = wgpu::TextureUsages::empty();
        usage.set(wgpu::TextureUsages::COPY_SRC, kind.is_copy_src());
        usage.set(wgpu::TextureUsages::COPY_DST, kind.is_copy_dst());
        usage.set(wgpu::TextureUsages::RENDER_ATTACHMENT, kind.is_attachment() | kind.is_depth_stencil());
        usage.set(wgpu::TextureUsages::TEXTURE_BINDING, kind.is_binding());

        let format = if kind.is_color() {
            shaders.defaults.color_format()
        } else if kind.is_alpha() {
            wgpu::TextureFormat::R8Unorm
        } else if kind.is_depth_stencil() {
            shaders.defaults.depth_stencil_format().unwrap()
        } else {
            unimplemented!()
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("color atlas"),
            dimension: wgpu::TextureDimension::D2,
            sample_count: if kind.is_msaa() { 4 } else { 1 },
            mip_level_count: 1,
            format,
            size: wgpu::Extent3d {
                width: key.size.0,
                height: key.size.1,
                depth_or_array_layers: 1,
            },
            usage,
            view_formats: &[],
        });

        let view = if kind.is_binding() || kind.is_attachment() || kind.is_depth_stencil() {
            Some(texture.create_view(&Default::default()))
        } else {
            None
        };

        let bind_group = if kind.is_binding() {
            Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &shaders.get_bind_group_layout(shaders.common_bind_group_layouts.color_texture).handle,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view.as_ref().unwrap())
                    },
                ],
                label: None,
            }))
        } else {
            None
        };

        GpuResource {
            key,
            as_input: bind_group,
            as_attachment: view,
        }
    }

    // Should be called only between upload and end_frame.
    pub fn get_resource(&self, index: u16) -> &GpuResource {
        &self.resources[index as usize]
    }

    pub fn resources(&self) -> &[GpuResource] {
        &self.resources
    }

    pub fn end_frame(&mut self) {
        // Discard resources that we did not use this frame.
        self.pool.clear();

        // Keep the one sthat were used.
        self.pool.reserve(self.resources.len());
        while let Some(res) = self.resources.pop() {
            self.pool.push(res);
        }
    }
}
