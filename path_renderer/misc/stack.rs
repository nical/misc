//! An attempt at providing a stack based mechanism for scheduling intermediate
//! render passes.
//!
//! The render stack isn't entirely implemented but the core of the algorithm works.
//! This was an experiment which I don't think will ultimately make it into the
//! overall system, or at least not in its current form.
//! The idea is to have a stack with push/pop operations to create and render into
//! intermediate render passes which are automatically used as dependecies of the
//! level immediately below them (and not used by another other node). Each level
//! of the stack contains a texture atlas. Push operations allocate from the atlas
//! of the selected level and can fail. When it fails, a flush operation can be used
//! to resolve all layers above the current one to free all of their atlases.
//! This allows rendering with a fixed texture memory budget for intermediate targets.
//!
//! The main drawback is that this integrates awkwardly with the overall render grap.
//! Each render node would have its own stack with no way to share intermediate atlases
//! and render passes with other render nodes.
//! In addition the temporary resource pool is difficult to share with the render graph.
//!
//! The next experiment will be to extend the render graph so that a render stack could
//! be implemented on top of it instead of besides it.

use crate::units::{SurfaceIntRect, SurfaceIntSize};
use crate::render_pass::{BuiltRenderPass, RenderPassBuilder, RenderPassConfig, RenderPassContext};
use crate::BindingsId;
use super::{ResourceKind, ResourceKey, TextureKind};

use guillotiere::{AllocatorOptions, SimpleAtlasAllocator as Atlas};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct VirtualResourceId(ResourceKey, u16);

impl VirtualResourceId {
    pub fn to_u32(&self) -> u32 {
        ((self.0.0 as u32) << 16) | self.1 as u32
    }
}

struct RenderStackLevel {
    render_pass: RenderPassBuilder,
    pass_index: u32,
    size: SurfaceIntSize,
    config: RenderPassConfig,
    resource_alloc_index: usize,
    // TODO: The first level's binding should be provided externally.
    color_attachments: [Option<BindingsId>; 3],
    atlas: Atlas,
}

struct Pass {
    config: RenderPassConfig,
    built: Option<BuiltRenderPass>,
    color_attachments: [Option<VirtualResourceId>; 3],
    depth_stencil_attachment: Option<VirtualResourceId>,
}

struct ConfigResources {
    key: ResourceKey,
    ping_pong: [[Option<VirtualResourceId>; 3]; 2],
    next: u16,
}

/// Manages a stack of temporary render passes with a texture atlas for each layer.
///
/// The general idea is that pushing an item attempts to allocate it at the next
/// level that has the appropriate resource kind. Since a level can only depend
/// on a level with a higher index, flushing the render pass amounts to emptying
/// levels from the highest level down to the flushing level.
pub struct RenderStack {
    current_level: i32,
    levels: Vec<RenderStackLevel>,
    ping_pong_resources: Vec<ConfigResources>,

    stack: Vec<i32>,

    passes: Vec<Pass>,
    ordered_passes: Vec<u32>,

    tmp_surface_size: SurfaceIntSize,
    atlas_options: AllocatorOptions,
}

impl RenderStack {
    pub fn new(
        config: RenderPassConfig,
        size: SurfaceIntSize,
    ) -> Self {
        let mut levels = Vec::new();

        let mut render_pass = RenderPassBuilder::new();
        render_pass.begin(size, config);

        let atlas_options = AllocatorOptions::default();

        levels.push(RenderStackLevel {
            render_pass,
            pass_index: 0,
            config,
            size,
            resource_alloc_index: 0,
            // We don't provide them for the root pass.
            color_attachments: [
                None,
                None,
                None,
            ],
            atlas: Atlas::with_options(size.cast_unit(), &atlas_options),
        });

        let mut passes = Vec::new();
        passes.push(Pass {
            config,
            built: None,
            color_attachments: [
                None,
                None,
                None,
            ],
            depth_stencil_attachment: None,
        });

        RenderStack {
            current_level: 0,
            levels,
            ping_pong_resources: Vec::new(),
            stack: vec![0],
            passes,
            ordered_passes: Vec::new(),
            tmp_surface_size: SurfaceIntSize::new(2048, 2048),
            atlas_options,
        }
    }

    pub fn ctx(&mut self) -> RenderPassContext {
        self.levels[self.current_level as usize].render_pass.ctx()
    }

    pub fn push(&mut self, config: RenderPassConfig, size: SurfaceIntSize) -> Option<SurfaceIntRect> {
        // Find the next stack level with the appropriate resource kind.
        let mut next_depth = (self.current_level + 1) as usize;
        while self.levels.len() > next_depth && self.levels[next_depth].config != config {
            next_depth += 1;
        }

        // If the level does not exist, create it.
        if next_depth >= self.levels.len() {
            // pick a reource slot using a ping/pong scheme.
            let resource_alloc_index = (self.levels[self.current_level as usize].resource_alloc_index + 1) % 2;

            let (color_attachments, depth_stencil_attachment) = self.get_resources_for_config(
                config,
                self.tmp_surface_size,
                resource_alloc_index
            );

            // Push a new render pass.
            let pass_index = self.passes.len() as u32;
            self.passes.push(Pass {
                config,
                built: None,
                color_attachments,
                depth_stencil_attachment,
            });

            // TODO: Scheduling might be better if we inserted a level after the
            // current one instead of pushing to the top. The current approach is simpler
            // but it means that layers that were pushed late have a tendency to be
            // executed early.
            self.levels.push(RenderStackLevel {
                render_pass: RenderPassBuilder::new(),
                atlas: Atlas::with_options(self.tmp_surface_size.cast_unit(), &self.atlas_options),
                config,
                size: self.tmp_surface_size,
                color_attachments: [
                    color_attachments[0].map(|res| BindingsId::render_stack(res.to_u32())),
                    color_attachments[1].map(|res| BindingsId::render_stack(res.to_u32())),
                    color_attachments[2].map(|res| BindingsId::render_stack(res.to_u32())),
                ],
                resource_alloc_index,
                pass_index,
            });

            self.levels.last_mut().unwrap().render_pass.begin(self.tmp_surface_size, config);
        }

        let rect = self.levels[next_depth].atlas.allocate(size.cast_unit())?;

        self.stack.push(self.current_level);
        self.current_level = next_depth as i32;

        Some(rect.cast_unit())
    }

    pub fn pop(&mut self) {
        debug_assert!(self.stack.len() > 1);
        self.current_level = self.stack.pop().unwrap();
    }

    pub fn flush(&mut self) {
        while self.levels.len() > (self.current_level + 1) as usize {
            let mut level = self.levels.pop().unwrap();
            self.passes[level.pass_index as usize].built = Some(level.render_pass.end());
            self.ordered_passes.push(level.pass_index);
        }

        // End the render pass of the current level without clearing the atlas state.
        let level = &mut self.levels[self.current_level as usize];
        let color_attachments = self.passes[level.pass_index as usize].color_attachments;
        let depth_stencil_attachment = self.passes[level.pass_index as usize].depth_stencil_attachment;
        self.passes[level.pass_index as usize].built = Some(level.render_pass.end());
        level.render_pass.begin(level.size, level.config);
        self.ordered_passes.push(level.pass_index);
        let new_pass_idx = self.passes.len() as u32;
        self.passes.push(Pass {
            config: level.config,
            built: None,
            color_attachments,
            depth_stencil_attachment,
        });
        level.pass_index = new_pass_idx;

        // TODO: ensure that resources from lower levels that have not been flushed
        // are not reused.
        // This can be done by clearing the ping/pong slots in ping_pong_resources for
        // the appropriate resource keys.
    }

    // TODO: this leaves the stack in an invalid state.
    pub fn finish(&mut self) {
        for level in self.levels.iter_mut().rev() {
            self.passes[level.pass_index as usize].built = Some(level.render_pass.end());
            self.ordered_passes.push(level.pass_index);
        }
        self.levels.clear();
    }

    fn get_resources_for_config(&mut self, config: RenderPassConfig, size: SurfaceIntSize, resource_alloc_index: usize) -> ([Option<VirtualResourceId>; 3], Option<VirtualResourceId>) {
        let mut attachment_resources = [None, None, None, None];

        let mut keys = [None, None, None, None];
        let w = size.width as u16;
        let h = size.height as u16;

        for (attachment_idx, surface_kind) in config.attachments.iter().enumerate() {
            let key = match surface_kind {
                crate::SurfaceKind::Color => TextureKind::color_attachment(),
                crate::SurfaceKind::Alpha => TextureKind::alpha_attachment(),
                crate::SurfaceKind::HdrColor => TextureKind::color_attachment().with_hdr(),
                crate::SurfaceKind::HdrAlpha => TextureKind::alpha_attachment().with_hdr(),
                crate::SurfaceKind::None => { continue }
            }.with_binding().with_msaa(config.msaa).as_key(w, h);

            keys[attachment_idx] = Some(key);
        }

        if config.depth || config.stencil {
            keys[3] = Some(TextureKind::depth_stencil().as_key(w, h));
        }

        for (attachment_idx, key) in keys.iter().enumerate() {
            let key = match key {
                Some(key) => *key,
                None => { continue; }
            };

            // Find the resource set for the requested config, create one if needed.
            let mut pp_res_idx = 0;
            for res in &self.ping_pong_resources {
                if res.key == key {
                    break;
                }
                pp_res_idx += 1;
            }

            // If we have not found a resource set, create it.
            if pp_res_idx == self.ping_pong_resources.len() {
                self.ping_pong_resources.push(ConfigResources {
                    key,
                    ping_pong: [[None, None, None], [None, None, None]],
                    next: 0,
                });
            }

            // Get or create the resource at the appropriate slot.
            let cfg_res = &mut self.ping_pong_resources[pp_res_idx];
            let cfg_res_slot = &mut cfg_res.ping_pong[resource_alloc_index][attachment_idx];
            let resource_index = match cfg_res_slot {
                Some(res) => *res,
                None => {
                    let idx = VirtualResourceId(key, cfg_res.next);
                    cfg_res.next += 1;
                    *cfg_res_slot = Some(idx);
                    idx
                }
            };

            attachment_resources[attachment_idx] = Some(resource_index);
        }

        (
            [
                attachment_resources[0],
                attachment_resources[1],
                attachment_resources[2],
            ],
            attachment_resources[3],
        )
    }

    fn current_pass(&self) -> u32 {
        self.levels[self.current_level as usize].pass_index
    }

    pub fn current_color_binding(&self, attachment_idx: u8) -> Option<BindingsId> {
        let level = &self.levels[self.current_level as usize];
        level.color_attachments[attachment_idx as usize]
    }
}

#[test]
fn simple_render_stack() {

    let color = RenderPassConfig::color();
    let alpha = RenderPassConfig::alpha();

    let default_size = SurfaceIntSize::new(2048, 1024);

    let mut stack = RenderStack::new(color, SurfaceIntSize::new(2048, 1024));

    let r0 = stack.current_pass();

    let c0;
    let c1;
    let c2;
    let c3;
    let a0;

    let c0_binding;
    let c1_binding;
    let c2_binding;

    stack.push(color, default_size).unwrap();
    {
        c0 = stack.current_pass();
        c0_binding = stack.current_color_binding(0);
        stack.push(color, default_size).unwrap();
        {
            c1 = stack.current_pass();
            c1_binding = stack.current_color_binding(0);
            assert!(c0 != c1);
            assert!(c0_binding != c1_binding);
            stack.push(color, default_size).unwrap();
            {
                c2 = stack.current_pass();
                c2_binding = stack.current_color_binding(0);
                assert_eq!(c0_binding, c2_binding);
                assert!(c2 != c0);
                assert!(c2 != c1);
            }
            stack.pop()
        }
        stack.pop();
        assert_eq!(stack.current_pass(), c0);
        stack.push(color, default_size).unwrap();
        {
            assert_eq!(stack.current_pass(), c1);
        }
        stack.pop();
        assert_eq!(stack.current_pass(), c0);
        stack.push(alpha, default_size).unwrap();
        {
            a0 = stack.current_pass();
            assert!(a0 != c0);
            assert!(a0 != c1);
        }
        stack.pop();
        assert_eq!(stack.current_pass(), c0);
    }
    stack.pop();

    stack.flush();

    let r1 = stack.current_pass();
    stack.push(color, default_size).unwrap();
    {
        c3 = stack.current_pass();
        assert_ne!(c3, c0);
        assert_ne!(c3, c1);
        assert_ne!(c3, c2);
    }
    stack.pop();

    stack.finish();

    assert_eq!(
        stack.ordered_passes.as_slice(),
        &[ a0, c2, c1, c0, r0, c3, r1 ]
    );

    assert_eq!(stack.passes[a0 as usize].config, alpha);
    assert_eq!(stack.passes[c0 as usize].config, color);
    assert_eq!(stack.passes[c1 as usize].config, color);
    assert_eq!(stack.passes[c2 as usize].config, color);
    assert_eq!(stack.passes[c0 as usize].color_attachments, stack.passes[c2 as usize].color_attachments);
    assert_ne!(stack.passes[c0 as usize].color_attachments, stack.passes[c1 as usize].color_attachments);
    assert_ne!(r0, r1);
}
