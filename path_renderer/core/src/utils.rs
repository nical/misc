use crate::{BindingResolver, BindingsId};

use std::ops::Range;

#[inline]
pub fn u32_range(r: Range<usize>) -> Range<u32> {
    r.start as u32..r.end as u32
}

#[inline]
pub fn usize_range(r: Range<u32>) -> Range<usize> {
    r.start as usize..r.end as usize
}

/// A helper struct to resolve bind groups and avoid redundant bindings.
pub struct DrawHelper {
    current_bindings: [BindingsId; 4],
}

impl DrawHelper {
    pub fn new() -> Self {
        DrawHelper {
            current_bindings: [
                BindingsId::NONE,
                BindingsId::NONE,
                BindingsId::NONE,
                BindingsId::NONE,
            ],
        }
    }

    pub fn resolve_and_bind<'pass, 'resources: 'pass>(
        &mut self,
        group_index: u32,
        id: BindingsId,
        resolver: &'resources dyn BindingResolver,
        pass: &mut wgpu::RenderPass<'pass>,
    ) {
        let idx = group_index as usize;
        if id.is_some() && id != self.current_bindings[idx] {
            if let Some(bind_group) = resolver.resolve_input(id) {
                pass.set_bind_group(group_index, bind_group, &[]);
            } else {
                panic!("failed to resolve input index {idx:?} with bindings {id:?}")
            }
            self.current_bindings[idx] = id;
        }
    }

    pub fn bind<'pass, 'resources: 'pass>(
        &mut self,
        group_index: u32,
        id: BindingsId,
        bind_group: &'resources wgpu::BindGroup,
        pass: &mut wgpu::RenderPass<'pass>,
    ) {
        let idx = group_index as usize;
        if self.current_bindings[idx] != id {
            pass.set_bind_group(group_index, bind_group, &[]);
            self.current_bindings[idx] = id
        }
    }

    pub fn reset_binding(&mut self, group_index: u32) {
        self.current_bindings[group_index as usize] = BindingsId::NONE;
    }
}
