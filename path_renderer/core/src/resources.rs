use crate::gpu::{GpuBufferDescriptor, GpuBufferResources, GpuStreamsDescritptor, GpuStreamsResources, StagingBufferPoolRef};
use crate::graph::{RenderPassData, TempResourceKey};
use crate::shading::Shaders;
use std::u32;
use std::marker::PhantomData;
use wgpu::util::DeviceExt;
use wgpu::BufferUsages;

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
        let graph = RenderGraphResources::new(device);

        GpuResources {
            common,
            graph,
        }
    }

    pub fn begin_frame(&mut self) {
        self.common.begin_frame();
        self.graph.begin_frame();
    }

    pub fn upload(&mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        shaders: &Shaders,
        allocations: &[TempResourceKey],
        pass_data: &[RenderPassData],
    ) {
        self.graph.upload(device, queue, shaders, &self.common, allocations, pass_data);
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

    pub f32_buffer: GpuBufferResources,
    pub u32_buffer: GpuBufferResources,

    pub default_sampler: wgpu::Sampler,

    pub staging_buffers: StagingBufferPoolRef,
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

        CommonGpuResources {
            quad_ibo,
            vertices,
            indices,
            instances,
            f32_buffer,
            u32_buffer,
            default_sampler,
            staging_buffers,
        }
    }

    fn begin_frame(&mut self) {}

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
    pass_bind_groups: Vec<(wgpu::Buffer, wgpu::BindGroup)>,
    default_sampler: wgpu::Sampler,
    f32_buffer_epoch: u32,
    u32_buffer_epoch: u32,
}

impl RenderGraphResources {
    pub fn new(device: &wgpu::Device) -> Self {

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

        RenderGraphResources {
            pool: Vec::new(),
            resources: Vec::new(),
            pass_bind_groups: Vec::new(),
            default_sampler,
            f32_buffer_epoch: u32::MAX,
            u32_buffer_epoch: u32::MAX,
        }
    }

    pub fn begin_frame(&mut self) {

    }

    pub fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        shaders: &Shaders,
        resources: &CommonGpuResources,
        allocations: &[TempResourceKey],
        pass_data: &[RenderPassData],
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

        if self.f32_buffer_epoch != resources.f32_buffer.epoch() || self.u32_buffer_epoch != resources.u32_buffer.epoch() {
            self.f32_buffer_epoch = resources.f32_buffer.epoch();
            self.u32_buffer_epoch = resources.u32_buffer.epoch();
            // If the gpu store texture ch_anges, we have to re-create the bind groups.
            self.pass_bind_groups.clear();
        }

        // Unfortunately there is a default minimum alignment of 256 bytes for UBOs, so
        // storing the pass data in a single uniform buffer and creating bindings at
        // different offsets is quite a bit wastful and tedious. On the other hand creating
        // an UBO per pass data is probably even more wasteful but that's what this does for
        // now.
        // Ideally, the pass data would be provided via push constants.
        if self.pass_bind_groups.len() < pass_data.len() {
            let target_and_gpu_buffers_layout = shaders.get_bind_group_layout(shaders.common_bind_group_layouts.target_and_gpu_buffer);
            let size = std::mem::size_of::<RenderPassGpuData>() as u64;

            while self.pass_bind_groups.len() < pass_data.len() {
                let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    size,
                    usage: BufferUsages::COPY_DST | BufferUsages::UNIFORM,
                    label: Some("Render pass descriptor"),
                    mapped_at_creation: false,
                });
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Pass descriptor & gpu buffers"),
                    layout: &target_and_gpu_buffers_layout.handle,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &buffer,
                                offset: 0,
                                size: wgpu::BufferSize::new(size),
                            }),
                        },
                        resources.f32_buffer.as_bind_group_entry(1).unwrap(),
                        resources.u32_buffer.as_bind_group_entry(2).unwrap(),
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::Sampler(&self.default_sampler),
                        },
                    ],
                });

                self.pass_bind_groups.push((buffer, bind_group));
            }
        }

        for (data, (buffer, _)) in pass_data.iter().zip(self.pass_bind_groups.iter()) {
            let w = data.target_size.0;
            let h = data.target_size.1;
            queue.write_buffer(
                buffer,
                0,
                bytemuck::cast_slice(&[RenderPassGpuData::new(w, h)]),
            );
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

    // Should be called only between upload and end_frame.
    pub fn get_base_bindgroup(&self, index: u16) -> &wgpu::BindGroup {
        &self.pass_bind_groups[index as usize].1
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

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct RenderPassGpuData {
    pub width: f32,
    pub height: f32,
    pub inv_width: f32,
    pub inv_height: f32,
}

impl RenderPassGpuData {
    pub fn new(w: u32, h: u32) -> Self {
        let width = w as f32;
        let height = h as f32;
        let inv_width = 1.0 / width;
        let inv_height = 1.0 / height;
        RenderPassGpuData {
            width,
            height,
            inv_width,
            inv_height,
        }
    }
}

unsafe impl bytemuck::Pod for RenderPassGpuData {}
unsafe impl bytemuck::Zeroable for RenderPassGpuData {}

pub struct ResourcesHandle<T> {
    index: u8,
    _marker: PhantomData<T>,
}

impl<T> ResourcesHandle<T> {
    pub fn new(index: u8) -> Self {
        ResourcesHandle {
            index,
            _marker: PhantomData,
        }
    }

    pub fn index(&self) -> usize {
        self.index as usize
    }
}

impl<T> Copy for ResourcesHandle<T> {}
impl<T> Clone for ResourcesHandle<T> {
    fn clone(&self) -> Self {
        *self
    }
}
