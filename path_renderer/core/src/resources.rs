use crate::gpu::shader::{BindGroupLayout, BindGroupLayoutId, Binding};
use crate::gpu::{DynamicStore, GpuStore, GpuTargetDescriptor, PipelineDefaults};
use crate::units::SurfaceIntSize;
use std::{any::Any, marker::PhantomData};
use wgpu::util::DeviceExt;

pub trait RendererResources: AsAny {
    fn name(&self) -> &'static str;
    fn begin_frame(&mut self) {}
    fn begin_rendering(&mut self, _encoder: &mut wgpu::CommandEncoder) {}
    fn end_frame(&mut self) {}
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

pub struct GpuResources {
    systems: Vec<Box<dyn RendererResources>>,
    next_handle: u8,
}

impl GpuResources {
    pub fn new() -> Self {
        GpuResources {
            systems: Vec::with_capacity(32),
            next_handle: 0,
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
        for sys in &mut self.systems {
            sys.begin_frame();
        }
    }

    pub fn begin_rendering(&mut self, encoder: &mut wgpu::CommandEncoder) {
        for sys in &mut self.systems {
            sys.begin_rendering(encoder);
        }
    }

    pub fn end_frame(&mut self) {
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

    // TODO: right now there can be only one target per instance of this struct
    pub target_and_gpu_store_layout: BindGroupLayoutId,
    pub main_target_and_gpu_store_bind_group: wgpu::BindGroup,
    pub main_target_descriptor_ubo: wgpu::Buffer,
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
        target_size: SurfaceIntSize,
        gpu_store: &GpuStore,
        shaders: &mut crate::gpu::Shaders,
    ) -> Self {
        let atlas_desc_buffer_size = std::mem::size_of::<GpuTargetDescriptor>() as u64;
        let target_and_gpu_store_layout = BindGroupLayout::new(
            device,
            "target and gpu store".into(),
            vec![
                Binding {
                    name: "render_target".into(),
                    struct_type: "RenderTarget".into(),
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(atlas_desc_buffer_size),
                    },
                },
                Binding {
                    name: "gpu_store_texture".into(),
                    struct_type: "f32".into(),
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                },
                Binding {
                    name: "default_sampler".into(),
                    struct_type: String::new(),
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                },
            ],
        );

        let main_target_descriptor_ubo =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Target info"),
                contents: bytemuck::cast_slice(&[GpuTargetDescriptor::new(
                    target_size.width as u32,
                    target_size.height as u32,
                )]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

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

        let main_target_and_gpu_store_bind_group =
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Main target & gpu store"),
                layout: &target_and_gpu_store_layout.handle,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(
                            main_target_descriptor_ubo.as_entire_buffer_binding(),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&gpu_store_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&default_sampler),
                    },
                ],
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

        let target_and_gpu_store_layout =
            shaders.register_bind_group_layout(target_and_gpu_store_layout);

        shaders.set_base_bindings(target_and_gpu_store_layout);

        CommonGpuResources {
            quad_ibo,
            vertices,
            indices,
            target_and_gpu_store_layout,
            main_target_and_gpu_store_bind_group,
            main_target_descriptor_ubo,
            gpu_store_view,
            default_sampler,
            msaa_blit_pipeline: msaa_blit,
            msaa_blit_with_depth_stencil_pipeline: msaa_blit_depth_stencil,
            msaa_blit_layout,
            msaa_blit_src_bind_group_layout,
        }
    }

    pub fn resize_target(&self, size: SurfaceIntSize, queue: &wgpu::Queue) {
        queue.write_buffer(
            &self.main_target_descriptor_ubo,
            0,
            bytemuck::cast_slice(&[GpuTargetDescriptor::new(
                size.width as u32,
                size.height as u32,
            )]),
        );
    }
}

impl RendererResources for CommonGpuResources {
    fn name(&self) -> &'static str {
        "CommonGpuResources"
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
