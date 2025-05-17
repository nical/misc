use crate::BindingsId;

use super::{BuiltGraph, TaskId};
use super::build::{NodeResourceInfo, NodeResources};

pub type RenderPassDataIndex = u16;

#[derive(Debug)]
pub enum Command {
    Render(RenderCommand),
    Compute(ComputeCommand),
}

#[derive(Debug)]
pub struct RenderCommand {
    pub(super) resources: NodeResources,
    pub task_id: TaskId,
    pub msaa: bool,
    pub depth: bool,
    pub stencil: bool,
    /// Index into `BuiltGraph::pass_data`.
    pub pass_data_index: RenderPassDataIndex,
    pub label: Option<&'static str>,
}

#[derive(Debug)]
pub struct ComputeCommand {
    pub task_id: TaskId,
    pub label: Option<&'static str>,
}

#[derive(Debug)]
pub struct CommandRef<'l> {
    command: &'l Command,
    graph: &'l BuiltGraph
}

impl<'l> CommandRef<'l> {
    pub fn as_render_command(&self) -> Option<RenderCommandRef<'l>> {
        match self.command {
            Command::Render(command) => {
                Some(RenderCommandRef { command, graph: self.graph })
            }
            _ => {
                None
            }
        }
    }

    pub fn task_id(&self) -> TaskId {
        match self.command {
            Command::Render(cmd) => cmd.task_id,
            Command::Compute(cmd) => cmd.task_id,
        }
    }
}

pub struct RenderCommandRef<'l> {
    command: &'l RenderCommand,
    graph: &'l BuiltGraph
}


impl<'l> RenderCommandRef<'l> {
    pub fn color_attachments(&self) -> AttachmentsIter {
        AttachmentsIter {
            inner: self.graph.resources[self.command.resources.color_attachments()].iter(),
            index: self.command.resources.resources_start,
            msaa_enabled: self.command.msaa,
        }
    }

    pub fn depth_stencil_attachment(&self) -> Option<DepthStencilAttachmentRef> {
        self.command.resources.depth_stencil().and_then(|idx| {
            Some(DepthStencilAttachmentRef {
                binding: BindingsId::graph(idx as u16),
                depth: self.command.depth,
                stencil: self.command.stencil,
            })
        })
    }

    pub fn msaa(&self) -> bool {
        self.command.msaa
    }

    pub fn task_id(&self) -> TaskId {
        self.command.task_id
    }

    pub fn pass_data_index(&self) -> RenderPassDataIndex {
        self.command.pass_data_index
    }

    pub fn label(&self) -> Option<&'static str> {
        self.command.label
    }
}

pub struct AttachmentRef<'l> {
    info: &'l NodeResourceInfo,
    binding: BindingsId,
}

impl<'l> AttachmentRef<'l> {
    pub fn binding(&self) -> BindingsId {
        self.binding
    }

    pub fn store(&self) -> bool {
        self.info.store
    }

    pub fn load(&self) -> bool {
        self.info.load
    }
}

pub struct DepthStencilAttachmentRef {
    binding: BindingsId,
    depth: bool,
    stencil: bool,
}

impl DepthStencilAttachmentRef {
    pub fn binding(&self) -> BindingsId {
        self.binding
    }

    pub fn depth(&self) -> bool {
        self.depth
    }

    pub fn stencil(&self) -> bool {
        self.stencil
    }
}

pub struct AttachmentsIter<'l> {
    msaa_enabled: bool,
    index: u16,
    inner: std::slice::Iter<'l, Option<NodeResourceInfo>>,
}

pub struct ColorAttachmentInfo<'l> {
    pub attachment: Option<AttachmentRef<'l>>,
    pub resolve_target: Option<AttachmentRef<'l>>,
}

impl<'l> Iterator for AttachmentsIter<'l> {
    type Item = ColorAttachmentInfo<'l>;
    fn next(&mut self) -> Option<Self::Item> {
        let non_msaa = self.inner.next();
        let msaa = self.inner.next();
        debug_assert!(non_msaa.is_none() == msaa.is_none());
        let (Some(msaa), Some(non_msaa)) = (msaa, non_msaa) else {
            return None;
        };
        let idx = self.index;
        let msaa_idx = idx + 1;
        self.index += 2;

        if self.msaa_enabled {
            return Some(ColorAttachmentInfo {
                attachment: msaa.as_ref().map(|info| AttachmentRef {
                    info,
                    binding: BindingsId::graph(msaa_idx)
                }),
                resolve_target: non_msaa.as_ref().map(|info| AttachmentRef {
                    info,
                    binding: BindingsId::graph(idx)
                }),
            });
        } else {
            return Some(ColorAttachmentInfo {
                attachment: non_msaa.as_ref().map(|info| AttachmentRef {
                    info,
                    binding: BindingsId::graph(idx)
                }),
                resolve_target: None,
            })
        }
    }
}

pub struct CommandsIter<'l> {
    pub(crate) graph: &'l BuiltGraph,
    pub(crate) inner: std::slice::Iter<'l, Command>,
}

impl<'l> Iterator for CommandsIter<'l> {
    type Item = CommandRef<'l>;
    fn next(&mut self) -> Option<CommandRef<'l>> {
        let command = self.inner.next()?;
        Some(CommandRef {
            graph: self.graph,
            command,
        })
    }
}
