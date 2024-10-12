use crate::gpu::{DynamicStore, GpuStore, RenderPassDescriptor, PipelineDefaults, Shaders};
use crate::render_graph::{RenderPassData, ResourceKind, TempResourceKey};
use std::u32;
use std::{any::Any, marker::PhantomData};
use wgpu::util::DeviceExt;
use wgpu::BufferUsages;

pub struct GpuResources {
    systems: Vec<Box<dyn RendererResources>>,
    next_handle: u8,

    pub common: CommonGpuResources,
    pub graph: RenderGraphResources,
}

impl GpuResources {
    pub fn new(
        device: &wgpu::Device,
        gpu_store: &GpuStore,
        shaders: &mut crate::gpu::Shaders,
    ) -> Self {
        let common = CommonGpuResources::new(device, gpu_store, shaders);
        let mut graph = RenderGraphResources::new(device);
        let copy = wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC;
        let render = wgpu::TextureUsages::RENDER_ATTACHMENT;
        let sample = wgpu::TextureUsages::TEXTURE_BINDING;
        let color_format = shaders.defaults.color_format();
        let depth_stencil_format = shaders.defaults.depth_stencil_format().unwrap();
        let color_texture = register_texture_kind(&mut graph, color_format, false, render | sample | copy);
        let alpha_texture = register_texture_kind(&mut graph, wgpu::TextureFormat::R8Unorm, false, render | sample | copy);
        let ds_texture = register_texture_kind(&mut graph, depth_stencil_format, false, render);
        let msaa_color_texture = register_texture_kind(&mut graph, color_format, true, render);
        let msaa_alpha_texture = register_texture_kind(&mut graph, wgpu::TextureFormat::R8Unorm, true, render);
        let msaa_ds_texture = register_texture_kind(&mut graph, depth_stencil_format, true, render);
        // TODO: register storage buffer resources.

        assert_eq!(color_texture, ResourceKind::COLOR_TEXTURE);
        assert_eq!(alpha_texture, ResourceKind::ALPHA_TEXTURE);
        assert_eq!(ds_texture, ResourceKind::DEPTH_STENCIL_TEXTURE);
        assert_eq!(msaa_color_texture, ResourceKind::MSAA_COLOR_TEXTURE);
        assert_eq!(msaa_alpha_texture, ResourceKind::MSAA_ALPHA_TEXTURE);
        assert_eq!(msaa_ds_texture, ResourceKind::MSAA_DEPTH_STENCIL_TEXTURE);

        GpuResources {
            systems: Vec::with_capacity(32),
            next_handle: 0,
            common,
            graph,
        }
    }

    pub fn register<T: RendererResources + 'static>(&mut self, system: T) -> ResourcesHandle<T> {
        let handle = ResourcesHandle::new(self.next_handle);
        self.next_handle += 1;

        self.systems.push(Box::new(system));

        handle
    }

    pub fn get<T: 'static>(&self, handle: ResourcesHandle<T>) -> &T {
        let result: Option<&T> = (*self.systems[handle.index()]).as_any().downcast_ref();
        #[cfg(debug_assertions)]
        if result.is_none() {
            panic!(
                "Invalid type, got {:?}",
                self.systems[handle.index()].name()
            );
        }
        result.unwrap()
    }

    pub fn get_mut<T: 'static>(&mut self, handle: ResourcesHandle<T>) -> &mut T {
        #[cfg(debug_assertions)]
        let name = self.systems[handle.index()].name();
        let result: Option<&mut T> = (*self.systems[handle.index()]).as_any_mut().downcast_mut();
        #[cfg(debug_assertions)]
        if result.is_none() {
            panic!("Invalid type, got {:?}", name);
        }

        result.unwrap()
    }

    pub fn begin_frame(&mut self) {
        self.common.begin_frame();
        self.graph.begin_frame();
        for sys in &mut self.systems {
            sys.begin_frame();
        }
    }

    pub fn upload(&mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        shaders: &Shaders,
        gpu_store: &GpuStore,
        allocations: &[TempResourceKey],
        pass_data: &[RenderPassData],
    ) {
        self.graph.upload(device, queue, shaders, gpu_store, allocations, pass_data);
    }

    pub fn begin_rendering(&mut self, encoder: &mut wgpu::CommandEncoder) {
        self.common.begin_rendering(encoder);
        //self.common.begin_rendering(..); // TODO
        for sys in &mut self.systems {
            sys.begin_rendering(encoder);
        }
    }

    pub fn end_frame(&mut self) {
        self.common.end_frame();
        self.graph.end_frame();
        for sys in &mut self.systems {
            sys.end_frame();
        }
    }
}

impl<T: 'static> std::ops::Index<ResourcesHandle<T>> for GpuResources {
    type Output = T;
    fn index(&self, handle: ResourcesHandle<T>) -> &T {
        self.get(handle)
    }
}

impl<T: 'static> std::ops::IndexMut<ResourcesHandle<T>> for GpuResources {
    fn index_mut(&mut self, handle: ResourcesHandle<T>) -> &mut T {
        self.get_mut(handle)
    }
}

pub struct CommonGpuResources {
    pub quad_ibo: wgpu::Buffer,
    pub vertices: DynamicStore,
    pub indices: DynamicStore,

    pub gpu_store_view: wgpu::TextureView,
    pub default_sampler: wgpu::Sampler,

    pub msaa_blit_layout: wgpu::PipelineLayout,
    pub msaa_blit_pipeline: wgpu::RenderPipeline,
    pub msaa_blit_with_depth_stencil_pipeline: wgpu::RenderPipeline,
    pub msaa_blit_src_bind_group_layout: wgpu::BindGroupLayout,
}

impl CommonGpuResources {
    pub fn new(
        device: &wgpu::Device,
        gpu_store: &GpuStore,
        shaders: &mut crate::gpu::Shaders,
    ) -> Self {
        let quad_indices = [0u16, 1, 2, 0, 2, 3];
        let quad_ibo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Quad indices"),
            contents: bytemuck::cast_slice(&quad_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let gpu_store_view = gpu_store.create_texture_view();

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

        let vertices = DynamicStore::new_vertices(4096 * 32);
        let indices = DynamicStore::new(8192, wgpu::BufferUsages::INDEX, "Common:Index");

        let msaa_blit_src_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Msaa blit source"),
                entries: &[
                    // target descriptor
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });

        let src = include_str!("./../shaders/msaa_blit.wgsl");
        let color_module = shaders.create_shader_module(device, "msaa-blit", src, &[]);

        let msaa_blit_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("color-to-msaa"),
            bind_group_layouts: &[&msaa_blit_src_bind_group_layout],
            push_constant_ranges: &[],
        });

        let targets = &[shaders.defaults.color_target_state_no_blend()];
        let mut descriptor = wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&msaa_blit_layout),
            vertex: wgpu::VertexState {
                module: &color_module,
                entry_point: "vs_main",
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &color_module,
                entry_point: "fs_main",
                targets,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: PipelineDefaults::primitive_state(),
            depth_stencil: None,
            multiview: None,
            multisample: wgpu::MultisampleState {
                count: shaders.defaults.msaa_sample_count(),
                ..wgpu::MultisampleState::default()
            },
            cache: None,
        };

        let msaa_blit = device.create_render_pipeline(&descriptor);

        descriptor.depth_stencil = Some(wgpu::DepthStencilState {
            format: shaders.defaults.depth_stencil_format().unwrap(),
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::Always,
            bias: wgpu::DepthBiasState::default(),
            stencil: wgpu::StencilState::default(),
        });

        let msaa_blit_depth_stencil = device.create_render_pipeline(&descriptor);

        CommonGpuResources {
            quad_ibo,
            vertices,
            indices,
            gpu_store_view,
            default_sampler,
            msaa_blit_pipeline: msaa_blit,
            msaa_blit_with_depth_stencil_pipeline: msaa_blit_depth_stencil,
            msaa_blit_layout,
            msaa_blit_src_bind_group_layout,
        }
    }

    fn begin_frame(&mut self) {}

    fn begin_rendering(&mut self, encoder: &mut wgpu::CommandEncoder) {
        self.vertices.unmap(encoder);
        self.indices.unmap(encoder);
    }

    fn end_frame(&mut self) {
        self.vertices.end_frame();
        self.indices.end_frame();
    }
}

pub struct GpuResource {
    pub key: TempResourceKey,
    pub as_input: Option<wgpu::BindGroup>,
    pub as_attachment: Option<wgpu::TextureView>,
}

type ResourceConstructor = Box<dyn Fn(TempResourceKey, &wgpu::Device, &Shaders) -> Option<GpuResource>>;

// TODO: the name isn't great.
pub struct RenderGraphResources {
    kinds: Vec<ResourceConstructor>,
    pool: Vec<GpuResource>,
    resources: Vec<GpuResource>,
    pass_bind_groups: Vec<(wgpu::Buffer, wgpu::BindGroup)>,
    default_sampler: wgpu::Sampler,
    gpu_store_epoch: u32,
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
            kinds: Vec::new(),
            pool: Vec::new(),
            resources: Vec::new(),
            pass_bind_groups: Vec::new(),
            default_sampler,
            gpu_store_epoch: u32::MAX,
        }
    }

    pub fn register_resource_kind(&mut self, constructor: ResourceConstructor) -> ResourceKind {
        assert!(self.kinds.len() < u8::MAX as usize);
        let id = ResourceKind(self.kinds.len() as u8);

        self.kinds.push(constructor);

        id
    }

    pub fn begin_frame(&mut self) {

    }

    pub fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        shaders: &Shaders,
        gpu_store: &GpuStore,
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
                self.kinds[key.kind.index()](*key, device, shaders).unwrap()
            };

            self.resources.push(resource);
        }

        if self.gpu_store_epoch != gpu_store.epoch() {
            self.gpu_store_epoch = gpu_store.epoch();
            // If the gpu store texture changes, we have to re-create the bind groups.
            self.pass_bind_groups.clear();
        }

        // Unfortunately there is a default minimum alignment of 256 bytes for UBOs, so
        // storing the pass data in a single uniform buffer and creating bindings at
        // different offsets is quite a but wastful and tedious. On the other hand creating
        // an UBO per pass data is probably even more wasteful but that's what this does for
        // now.
        // Ideally, the pass data would be provided via push constants.
        if self.pass_bind_groups.len() < pass_data.len() {
            let target_and_gpu_store_layout = shaders.get_bind_group_layout(shaders.common_bind_group_layouts.target_and_gpu_store);
            let size = std::mem::size_of::<RenderPassDescriptor>() as u64;

            while self.pass_bind_groups.len() < pass_data.len() {
                let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    size,
                    usage: BufferUsages::COPY_DST | BufferUsages::UNIFORM,
                    label: Some("Render pass descriptor"),
                    mapped_at_creation: false,
                });
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Pass descriptor & gpu store"),
                    layout: &target_and_gpu_store_layout.handle,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &buffer,
                                offset: 0,
                                size: wgpu::BufferSize::new(size),
                            }),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(gpu_store.texture_view()),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
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
                bytemuck::cast_slice(&[RenderPassDescriptor::new(w, h)]),
            );
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
        self.pool.reserve(self.resources.len());
        while let Some(res) = self.resources.pop() {
            self.pool.push(res);
        }
    }
}

fn register_texture_kind(
    graph_resources: &mut RenderGraphResources,
    format: wgpu::TextureFormat,
    msaa: bool,
    usage: wgpu::TextureUsages,
) -> ResourceKind {
    graph_resources.register_resource_kind(Box::new(move |key, device, shaders| {
        println!("create texture format {format:?} key {key:?}");

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("color atlas"),
            dimension: wgpu::TextureDimension::D2,
            sample_count: if msaa { 4 } else { 1 },
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

        let view = texture.create_view(&Default::default());

        let bind_group = if usage.contains(wgpu::TextureUsages::TEXTURE_BINDING) {
            Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &shaders.get_bind_group_layout(shaders.common_bind_group_layouts.color_texture).handle,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view)
                    },
                ],
                label: None,
            }))
        } else {
            None
        };

        Some(GpuResource {
            key,
            as_input: bind_group,
            as_attachment: Some(view),
        })
    }))
}

// TODO: The RendererResources mechanism was originally used by all renderers to
// store their own resources, but they were all refactored to store their stuff
// directly. It could still be a good system for registering resources that are
// shared between multiple renderers but at the moment it is basically unused.
pub trait RendererResources: AsAny {
    fn name(&self) -> &'static str;
    fn begin_frame(&mut self) {}
    fn begin_rendering(&mut self, _encoder: &mut wgpu::CommandEncoder) {}
    fn end_frame(&mut self) {}
}

impl RendererResources for () {
    fn name(&self) -> &'static str { "dummy resources" }
}

pub trait AsAny {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

impl<T: 'static> AsAny for T {
    fn as_any(&self) -> &dyn Any {
        self as _
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self as _
    }
}

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
