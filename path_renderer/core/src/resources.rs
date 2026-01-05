use crate::SurfaceKind;
use crate::gpu::{GpuBufferDescriptor, GpuBufferResources, GpuStreamsDescritptor, GpuStreamsResources, StagingBufferPoolRef};
use crate::shading::Shaders;
use crate::units::SurfaceIntSize;
use std::u32;
use std::fmt;
use wgpu::util::DeviceExt;


#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Allocation {
    /// Part of the internally managed resource pool.
    Temporary,
    /// Allocated externally.
    External,
    // TODO: a Retained variant for resources that are explicitly created/destroyed
    // outisde of the render graph but still managed by the gpu resource pool.
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct ResourceIndex {
    pub allocation: Allocation,
    pub index: u16,
}

impl ResourceIndex {
    pub fn temporary(index: u16) -> Self {
        ResourceIndex {
            allocation: Allocation::Temporary,
            index,
        }
    }
}

impl fmt::Debug for ResourceIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}({:?})", self.allocation, self.index)
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct TextureKind(u16);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct BufferKind(u16);

impl TextureKind {
    const ALPHA: u16 = 1;
    const DEPTH_STENCIL: u16 = 2;

    const MSAA: u16 = 1 << 3;
    const HDR: u16 = 1 << 4;
    const COPY_SRC: u16 = 1 << 5;
    const COPY_DST: u16 = 1 << 6;
    const BINDING: u16 = 1 << 7;
    const ATTACHMENT: u16 = 1 << 8;

    pub const fn color() -> Self {
        TextureKind(0)
    }

    pub const fn color_attachment() -> Self {
        Self::color().with_attachment()
    }

    pub const fn alpha_attachment() -> Self {
        Self::alpha().with_attachment()
    }

    pub const fn color_binding() -> Self {
        Self::color().with_binding()
    }

    pub const fn alpha_binding() -> Self {
        Self::alpha().with_binding()
    }


    pub const fn alpha() -> Self {
        TextureKind(Self::ALPHA)
    }

    pub const fn depth_stencil() -> Self {
        TextureKind(Self::DEPTH_STENCIL)
    }

    pub const fn with_hdr(self) -> Self {
        TextureKind(self.0 | Self::HDR)
    }

    pub const fn with_attachment(self) -> Self {
        TextureKind(self.0 | Self::ATTACHMENT)
    }

    pub const fn with_binding(self) -> Self {
        TextureKind(self.0 | Self::BINDING)
    }

    pub const fn with_msaa(self, msaa: bool) -> Self {
        if msaa {
            TextureKind(self.0 | Self::MSAA)
        } else {
            TextureKind(self.0 & !Self::MSAA)
        }
    }

    pub const fn with_copy_src(self) -> Self {
        TextureKind(self.0 | Self::COPY_SRC)
    }

    pub const fn with_copy_dst(self) -> Self {
        TextureKind(self.0 | Self::COPY_DST)
    }

    pub fn from_surface_kind(kind: SurfaceKind) -> Self {
        match kind {
            SurfaceKind::Color => Self::color(),
            SurfaceKind::Alpha => Self::alpha(),
            SurfaceKind::HdrColor => Self::color().with_hdr(),
            SurfaceKind::HdrAlpha => Self::alpha().with_hdr(),
            SurfaceKind::None => unimplemented!(),
        }
    }

    pub const fn as_resource(self) -> ResourceKind {
        ResourceKind(self.0)
    }

    pub const fn is_color(self) -> bool {
        self.0 & (Self::ALPHA | Self::DEPTH_STENCIL)  == 0
    }

    pub const fn is_alpha(self) -> bool {
        self.0 & Self::ALPHA != 0
    }

    pub const fn is_depth_stencil(self) -> bool {
        self.0 & Self::DEPTH_STENCIL != 0
    }

    pub const fn is_hdr(self) -> bool {
        self.0 & Self::HDR != 0
    }

    pub const fn is_attachment(self) -> bool {
        self.0 & Self::ATTACHMENT != 0
    }

    pub const fn is_binding(self) -> bool {
        self.0 & Self::BINDING != 0
    }

    pub const fn is_msaa(self) -> bool {
        self.0 & Self::MSAA != 0
    }

    pub const fn is_copy_src(self) -> bool {
        self.0 & Self::COPY_SRC != 0
    }

    pub const fn is_copy_dst(self) -> bool {
        self.0 & Self::COPY_DST != 0
    }

    pub const fn is_compatible_width(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
}

impl fmt::Debug for TextureKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Texture(")?;
        if self.is_color() {
            write!(f, "color")?;
        }
        if self.is_alpha() {
            write!(f, "alpha")?;
        }
        if self.is_depth_stencil() {
            write!(f, "depth-stencil")?;
        }
        if self.is_hdr() {
            write!(f, "|hdr")?;
        }
        if self.is_msaa() {
            write!(f, "|msaa")?;
        }
        if self.is_attachment() {
            write!(f, "|attachment")?;
        }
        if self.is_binding() {
            write!(f, "|binding")?;
        }
        if self.is_copy_src() {
            write!(f, "|copy-src")?;
        }
        if self.is_copy_dst() {
            write!(f, "|copy-dst")?;
        }

        write!(f, ")")
    }
}

impl BufferKind {
    const BUFFER: u16 = 1 << 15;

    const UNIFORM: u16 = 1 << 0;

    const COPY_SRC: u16 = 1 << 1;
    const COPY_DST: u16 = 1 << 2;

    pub fn as_resource(self) -> ResourceKind {
        ResourceKind(self.0)
    }

    pub fn storage() -> Self {
        BufferKind(Self::BUFFER)
    }

    pub fn uniform() -> Self {
        BufferKind(Self::BUFFER | Self::UNIFORM)
    }

    pub fn staging() -> Self {
        BufferKind(Self::BUFFER | Self::COPY_SRC | Self::COPY_DST)
    }

    pub const fn with_copy_src(self) -> Self {
        BufferKind(self.0 | Self::COPY_SRC)
    }

    pub const fn with_copy_dst(self) -> Self {
        BufferKind(self.0 | Self::COPY_DST)
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct ResourceKind(u16);

impl ResourceKind {
    pub fn is_texture(self) -> bool {
        self.0 & BufferKind::BUFFER == 0
    }

    pub fn is_buffer(self) -> bool {
        self.0 & BufferKind::BUFFER != 0
    }

    pub fn as_texture(self) -> Option<TextureKind> {
        if self.is_texture() {
            Some(TextureKind(self.0))
        } else {
            None
        }
    }

    pub fn as_buffer(self) -> Option<BufferKind> {
        if self.is_buffer() {
            Some(BufferKind(self.0))
        } else {
            None
        }
    }

    pub fn with_binding(self) -> Self {
        if let Some(tex) = self.as_texture() {
            tex.with_binding().as_resource()
        } else {
            // Binding capability is implicit for buffers.
            self
        }
    }

}

impl fmt::Debug for ResourceKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(tex) = self.as_texture() {
            return tex.fmt(f);
        }
        if let Some(buf) = self.as_buffer() {
            return buf.fmt(f);
        }

        write!(f, "<InvalidBufferKind>")
    }
}

/// A compact descriptor describing a texture or buffer resource allocation.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ResourceKey {
    kind: ResourceKind,
    size: u32,
}

impl ResourceKey {
    pub fn texture(kind: TextureKind, size: SurfaceIntSize) -> Self {
        debug_assert!(size.width > 0 && size.height > 0);
        debug_assert!(size.width < u16::MAX as i32 && size.height < u16::MAX as i32);
        let kind = kind.as_resource();
        let size = ((size.width as u32) << 16)
           | (size.height as u32);
        ResourceKey { kind, size }
    }

    pub fn buffer(kind: BufferKind, size: u32) -> Self {
        let kind = kind.as_resource();
        ResourceKey { kind, size }
    }

    pub fn as_texture(&self) -> Option<(TextureKind, SurfaceIntSize)> {
        let kind = self.kind.as_texture()?;
        let size = SurfaceIntSize::new(
            (self.size >> 16) as i32,
            (self.size & 0xFFFF) as i32,
        );
        Some((kind, size))
    }

    pub fn as_buffer(&self) -> Option<(BufferKind, u32)> {
        let kind = self.kind.as_buffer()?;
        Some((kind, self.size))
    }

    pub fn with_binding(self) -> Self {
        ResourceKey {
            kind: self.kind.with_binding(),
            size: self.size
        }
    }
}

pub struct GpuResources {
    pub common: CommonGpuResources,
    pub temp: TemporaryResources,
}

impl GpuResources {
    pub fn new(
        device: &wgpu::Device,
        staging_buffers: StagingBufferPoolRef,
    ) -> Self {
        let common = CommonGpuResources::new(device, staging_buffers);
        let temp = TemporaryResources::new();

        GpuResources {
            common,
            temp,
        }
    }

    pub fn upload(&mut self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        shaders: &Shaders,
        allocations: &[ResourceKey],
    ) {
        self.common.upload(device, shaders);
        self.temp.upload(device, shaders, allocations);
    }

    pub fn begin_rendering(&mut self, encoder: &mut wgpu::CommandEncoder) {
        self.common.begin_rendering(encoder);
    }

    pub fn end_frame(&mut self) {
        self.common.end_frame();
        self.temp.end_frame();
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
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
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

#[derive(Debug)]
pub struct GpuResource {
    pub key: ResourceKey,
    pub texture: Option<wgpu::Texture>,
    pub buffer: Option<wgpu::Buffer>,
    pub as_input: Option<wgpu::BindGroup>,
    pub as_attachment: Option<wgpu::TextureView>,
}

/// Temporary buffers and textures containing data that is not
/// persisted across frames.
pub struct TemporaryResources {
    pool: Vec<GpuResource>,
    allocated: Vec<GpuResource>,
}

impl TemporaryResources {
    pub fn new() -> Self {
        TemporaryResources {
            pool: Vec::new(),
            allocated: Vec::new(),
        }
    }

    pub fn upload(
        &mut self,
        device: &wgpu::Device,
        shaders: &Shaders,
        allocations: &[ResourceKey],
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
                self.allocate_resource(device, shaders, *key)
            };

            self.allocated.push(resource);
        }
    }

    fn allocate_resource(&self,
        device: &wgpu::Device,
        shaders: &Shaders,
        key: ResourceKey,
    ) -> GpuResource {
        if key.kind.is_buffer() {
            todo!();
        }

        let (kind, size) = key.as_texture().unwrap();

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
                width: size.width as u32,
                height: size.height as u32,
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
            texture: Some(texture),
            buffer: None,
            as_input: bind_group,
            as_attachment: view,
        }
    }

    // Should be called only between upload and end_frame.
    pub fn get_resource(&self, index: u16) -> &GpuResource {
        &self.allocated[index as usize]
    }

    pub fn resources(&self) -> &[GpuResource] {
        &self.allocated
    }

    pub fn end_frame(&mut self) {
        // Discard resources that we did not use this frame.
        self.pool.clear();

        // Keep the one sthat were used.
        self.pool.reserve(self.allocated.len());
        while let Some(res) = self.allocated.pop() {
            self.pool.push(res);
        }
    }
}
