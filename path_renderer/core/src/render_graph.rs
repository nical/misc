use std::{fmt, ops::Range, u16};
use smallvec::SmallVec;

use crate::{units::SurfaceIntSize, BindingsId, BindingsNamespace, ResourceKind, TextureKind};

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct NodeId(pub u16);

impl fmt::Debug for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{}", self.0)
    }
}

impl NodeId {
    pub fn index(self) -> usize {
        self.0 as usize
    }

    pub fn from_index(idx: usize) -> Self {
        debug_assert!(idx < std::u16::MAX as usize);
        NodeId(idx as u16)
    }

    pub fn color(self, slot: u8) -> Dependency {
        Dependency {
            node: self,
            slot: Slot::Color(slot),
        }
    }

    pub fn msaa(self, slot: u8) -> Dependency {
        Dependency {
            node: self,
            slot: Slot::Msaa(slot),
        }
    }

    pub fn depth_stencil(self) -> Dependency {
        Dependency {
            node: self,
            slot: Slot::DepthStencil,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub enum Slot {
    Color(u8),
    Msaa(u8),
    DepthStencil,
    // Other(u8)
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct Dependency {
    pub node: NodeId,
    pub slot: Slot,
}

impl From<NodeId> for Dependency {
    fn from(node: NodeId) -> Self {
        Dependency {
            node,
            slot: Slot::Color(0),
        }
    }
}

impl fmt::Debug for Dependency {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.slot {
            Slot::Color(idx) => write!(f, "#{}.color_{}", self.node.0, idx),
            Slot::Msaa(idx) => write!(f, "#{}.msaa_{}", self.node.0, idx),
            Slot::DepthStencil => write!(f, "#{}.depth_stencil", self.node.0),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TaskId(pub u64);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Allocation {
    /// Allocated by the render graph.
    Temporary,
    /// Allocated outside of the render graph.
    External,
    // TODO: a Retained variant for resources that are explicitly created/destroyed
    // outisde of the render graph but still managed by the gpu resource pool.
}

// The virtual/physical resource distinction is similar to virtual and physical
// registers: Every output port of a node is its own virtual resource, to which
// is assigned a physical resource when scheduling the graph. Physical resources
// are used by as many nodes as possible to minimize memory usage.

#[derive(Debug)]
pub struct NodeResourceInfo {
    /// The actual resource id.
    pub resource: ResourceIndex,
    /// True if any render node reads from it.
    pub load: bool,
    pub store: bool,
    // TODO: pub load
}

type PhysicalResourceIndex = u16;

// TODO: Rename into ResourceIndex.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct ResourceIndex {
    // TODO: move kind out of PhysicalResourceId so that we can do  BindingsId -> PhsyicalResourceId mapping
    pub kind: ResourceKind,
    pub allocation: Allocation,
    pub index: PhysicalResourceIndex,
}

impl fmt::Debug for ResourceIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}({:?}:{:?})", self.allocation, self.index, self.kind)
    }
}

#[derive(Debug)]
pub struct Command {
    resources: NodeResources,
    pub msaa: bool,
    pub depth: bool,
    pub stencil: bool,
    pub node_id: NodeId,
    pub task_id: TaskId,
    /// Index into `BuiltGraph::pass_data`.
    pub pass_data_index: RenderPassDataIndex,
    pub label: Option<&'static str>,
}

pub struct CommandRef<'l> {
    command: &'l Command,
    graph: &'l BuiltGraph
}

impl<'l> CommandRef<'l> {
    pub fn resources(&self) -> &[Option<NodeResourceInfo>] {
        &self.graph.resources[self.command.resources.all()]
    }

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

    pub fn node_id(&self) -> NodeId {
        self.command.node_id
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
    graph: &'l BuiltGraph,
    inner: std::slice::Iter<'l, Command>,
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

pub type RenderPassDataIndex = u16;

/// Global per render node data that will be passed to shaders
/// via a an offset into a globals buffer.
#[derive(Debug, PartialEq)]
pub struct RenderPassData {
    pub target_size: (u32, u32)
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct TempResourceKey {
    pub kind: ResourceKind,
    pub size: (u32, u32),
}

#[derive(Debug)]
pub struct BuiltGraph {
    pub temporary_resources: Vec<TempResourceKey>,
    pub resources: Vec<Option<NodeResourceInfo>>,
    pub pass_data: Vec<RenderPassData>,
    pub commands: Vec<Command>,
}

impl BuiltGraph {
    pub fn resolve_binding(&self, binding: BindingsId) -> Option<ResourceIndex> {
        debug_assert!(binding.namespace() == BindingsNamespace::RenderGraph);
        self.resources[binding.index()].as_ref().map(|res| res.resource)
    }
}

impl<'l> IntoIterator for &'l BuiltGraph {
    type Item = CommandRef<'l>;
    type IntoIter = CommandsIter<'l>;
    fn into_iter(self) -> Self::IntoIter {
        CommandsIter {
            graph: self,
            inner: self.commands.iter(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum GraphError {
    DependencyCycle(NodeId),
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ColorAttachment {
    pub kind: TextureKind,
    pub non_msaa: Attachment,
    pub msaa: Attachment,
}

impl ColorAttachment {
    pub const fn color() -> Self {
        ColorAttachment {
            kind: TextureKind::color(),
            non_msaa: Attachment::Auto,
            msaa: Attachment::Auto,
        }
    }

    pub const fn alpha() -> Self {
        ColorAttachment {
            kind: TextureKind::alpha(),
            non_msaa: Attachment::Auto,
            msaa: Attachment::Auto,
        }
    }

    pub const fn with_dependency(mut self, dep: Dependency) -> Self {
        match dep.slot {
            Slot::Color(_) => {
                self.non_msaa = Attachment::Dependency(dep);
            }
            Slot::Msaa(_) => {
                self.msaa = Attachment::Dependency(dep);
            }
            Slot::DepthStencil => {
                panic!("Can't use a depth_stencil slot as a color attachment");
            }
        }

        self
    }

    pub const fn with_external(mut self, index: u16, msaa: bool) -> Self {
        if msaa {
            self.msaa = Attachment::External(index);
        } else {
            self.non_msaa = Attachment::External(index);
        }

        self
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Attachment {
    /// Automatically allocate the resource for this attachment.
    Auto,
    /// Use the specified resource for this attachment.
    External(u16),
    /// Use the same resource as the one provided by the dependency.
    Dependency(Dependency),
}

impl From<Dependency> for Attachment {
    fn from(dep: Dependency) -> Attachment {
        Attachment::Dependency(dep)
    }
}

// TODO: clearing
pub struct NodeDescriptor<'l> {
    pub label: Option<&'static str>,
    pub task: Option<TaskId>,
    pub reads: &'l[Dependency],
    pub attachments: &'l[ColorAttachment],
    pub depth_stencil: Option<(Attachment, bool, bool)>,
    pub msaa_resolve_target: Option<Attachment>,
    pub msaa: bool,
    pub size: Option<SurfaceIntSize>,
}

impl NodeDescriptor<'static> {
    pub fn new() -> Self {
        NodeDescriptor {
            label: None,
            task: None,
            reads: &[],
            attachments: &[],
            depth_stencil: None,
            msaa_resolve_target: None,
            msaa: false,
            size: None,
        }
    }
}

impl<'l> NodeDescriptor<'l> {

    pub fn read<'a>(self, inputs: &'a [Dependency]) -> NodeDescriptor<'a>
    where 'l : 'a
    {
        NodeDescriptor {
            reads: inputs,
            .. self
        }
    }

    pub fn attachments<'a>(self, attachments: &'a [ColorAttachment]) -> NodeDescriptor<'a>
    where 'l : 'a
    {
        NodeDescriptor {
            attachments,
            .. self
        }
    }

    pub fn depth_stencil(self, attachment: Attachment, depth: bool, stencil: bool) -> Self {
        NodeDescriptor {
            depth_stencil: Some((attachment, depth, stencil)),
            .. self
        }
    }

    pub fn task(self, task: TaskId) -> Self {
        NodeDescriptor {
            task: Some(task),
            .. self
        }
    }

    pub fn label(self, label: &'static str) -> Self {
        NodeDescriptor {
            label: Some(label),
            .. self
        }
    }

    pub fn msaa_resolve(self, attachment: Attachment) -> Self {
        NodeDescriptor {
            msaa_resolve_target: Some(attachment),
            .. self
        }
    }

    pub fn msaa(self, msaa: bool) -> Self {
        NodeDescriptor {
            msaa,
            .. self
        }
    }

    pub fn size(self, size: SurfaceIntSize) -> Self {
        NodeDescriptor {
            size: Some(size),
            .. self
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum OutputKind {
    ColorAttachment,
    DepthStencilAttachment,
    MsaaResolveTarget,
    Other,
}

#[derive(Clone, Debug)]
struct NodeResources {
    resources_start: u16,
    attachments_end: u16,
    resources_end: u16,
    depth_stencil: Option<u16>,
}

struct Node {
    resources: NodeResources,
    reads: SmallVec<[Dependency; 4]>,
    msaa: bool,
    depth: bool,
    stencil: bool,
    size: (u32, u32),
    readable: bool,
    label: Option<&'static str>,
    // writes: TODO
}

impl NodeResources {
    fn slot_index(&self, slot: Slot) -> usize {
        match slot {
            Slot::Color(idx) => idx as usize * 2,
            Slot::Msaa(idx) => idx as usize * 2 + 1,
            Slot::DepthStencil => self.depth_stencil().unwrap(),
        }
    }

    fn index(&self, slot: Slot) -> usize {
        self.resources_start as usize + self.slot_index(slot)
    }

    fn all(&self) -> Range<usize> {
        self.resources_start as usize .. self.resources_end as usize
    }

    fn color_attachments(&self) -> Range<usize> {
        self.resources_start as usize .. self.attachments_end as usize
    }

    fn depth_stencil(&self) -> Option<usize> {
        self.depth_stencil.map(|ds| ds as usize)
    }
}

struct Root {
    node_id: NodeId,
    stored_attachments: u32,
}

struct NodeResource {
    kind: ResourceKind,
    attachment: Attachment,
}

pub struct RenderGraph {
    nodes: Vec<Node>,
    resources: Vec<NodeResource>,
    tasks: Vec<TaskId>,
    roots: Vec<Root>,
    next_virtual_resource: u16,
}

impl RenderGraph {
    pub fn new() -> Self {
        RenderGraph {
            nodes: Vec::new(),
            resources: Vec::new(),
            tasks: Vec::new(),
            roots: Vec::new(),
            next_virtual_resource: 0,
        }
    }

    pub fn clear(&mut self) {
        self.nodes.clear();
        self.resources.clear();
        self.tasks.clear();
        self.roots.clear();
        self.next_virtual_resource = 0;
    }

    pub fn add_node(&mut self, desc: &NodeDescriptor) -> NodeId {
        let id = NodeId::from_index(self.nodes.len());

        let resources_start = self.resources.len();
        for attachment in desc.attachments {
            let kind = attachment.kind.with_attachment();
            let msaa_kind = kind.with_msaa(true);
            self.resources.push(NodeResource {
                kind: kind.as_resource(),
                attachment: attachment.non_msaa,
            });
            self.resources.push(NodeResource {
                kind: msaa_kind.as_resource(),
                attachment: attachment.msaa,
            });
        }
        let attachments_end = self.resources.len();
        let mut ds_resource = None;
        let mut depth = false;
        let mut stencil = false;
        if let Some((attachment, d, s)) = desc.depth_stencil {
            depth = d;
            stencil = s;
            ds_resource = Some(self.resources.len() as u16);
            self.resources.push(NodeResource {
                kind: TextureKind::depth_stencil().with_msaa(desc.msaa).as_resource(),
                attachment,
            });
        }

        let resources_end = self.resources.len();

        self.tasks.push(desc.task.expect("NodeDescriptor needs a task."));
        self.nodes.push(Node {
            resources: NodeResources {
                resources_start: resources_start as u16,
                resources_end: resources_end as u16,
                attachments_end: attachments_end as u16,
                depth_stencil: ds_resource,
            },
            msaa: desc.msaa,
            depth,
            stencil,
            size: desc.size.map(|s| s.to_u32().to_tuple()).unwrap_or((0, 0)),
            reads: SmallVec::from_slice(desc.reads),
            readable: true,
            label: desc.label,
        });

        self.next_virtual_resource += desc.attachments.len() as u16;

        id
    }

    pub fn add_dependency(&mut self, node: NodeId, dep: Dependency) {
        debug_assert!(self.nodes[dep.node.index()].readable);
        self.nodes[node.index()].reads.push(dep);
    }

    pub fn add_root(&mut self, root: NodeId, stored_attachments_mask: u32) {
        self.roots.push(Root {
            node_id: root,
            stored_attachments: stored_attachments_mask,
        });
    }

    /// After this call, no dependency to this node can be added.
    pub fn finish_node(&mut self, node: NodeId) {
        self.nodes[node.index()].readable = false;
    }

    pub fn get_binding(&self, dep: Dependency) -> BindingsId {
        let idx = self.nodes[dep.node.index()].resources.index(dep.slot);
        BindingsId::graph(idx as u16)
    }

    fn topological_sort(&self, sorted: &mut Vec<NodeId>) -> Result<(), GraphError> {
        sorted.reserve(self.nodes.len());

        let mut added = vec![false; self.nodes.len()];
        let mut cycle_check = vec![false; self.nodes.len()];
        let mut stack: Vec<NodeId> = Vec::with_capacity(self.nodes.len());

        for root in &self.roots {
            let root = root.node_id;
            //println!("- root {root:?}");
            if added[root.index()] {
                continue;
            }

            stack.push(root);
            'traversal: while let Some(id) = stack.last().cloned() {
                if added[id.index()] {
                    stack.pop();
                    continue;
                }

                cycle_check[id.index()] = true;
                let node = &self.nodes[id.index()];
                for res in &self.resources[node.resources.all()] {
                    match res.attachment {
                        Attachment::Dependency(dep) => {
                            if !added[dep.node.index()] {
                                if cycle_check[dep.node.index()] {
                                    return Err(GraphError::DependencyCycle(dep.node));
                                }

                                stack.push(dep.node);
                                continue 'traversal;
                            }
                        }
                        _ => {}
                    }
                }
                for dep in &node.reads {
                    if !added[dep.node.index()] {
                        if cycle_check[dep.node.index()] {
                            return Err(GraphError::DependencyCycle(dep.node));
                        }

                        stack.push(dep.node);
                        continue 'traversal;
                    }
                }
                cycle_check[id.index()] = false;

                added[id.index()] = true;
                sorted.push(id);
            }
        }

        Ok(())
    }

    pub fn schedule(&mut self) -> Result<BuiltGraph, GraphError> {
        // TODO: partial schedule
        let full_schedule = true;

        let mut sorted = Vec::new();
        self.topological_sort(&mut sorted)?;

        #[derive(Clone)]
        struct VirtualResource {
            refs: u16,
            size: (u32, u32),
            id: Option<ResourceIndex>,
            read_at_least_once: bool,
            reusable: bool,
            load: bool,
        }

        let mut virtual_resources = vec![
            VirtualResource {
                refs: 0,
                size: (0, 0),
                id: None,
                read_at_least_once: false,
                reusable: false,
                load: false,
            };
            self.resources.len()
        ];

        let mut temp_resources = TemporaryResources {
            available: Vec::with_capacity(16),
            resources: Vec::with_capacity(16),
        };

        // First pass allocates virtual resources and counts the number of times they
        // are used.
        for node_id in &sorted {
            let node = &self.nodes[node_id.index()];
            for idx in node.resources.all() {
                let resource = &self.resources[idx];

                let vres = match resource.attachment {
                    Attachment::Auto => {
                        VirtualResource {
                            refs: 0,
                            size: node.size,
                            id: None,
                            // Only make the physical resource available again if we
                            // know that no new reference can be added to it.
                            reusable: full_schedule || !node.readable,
                            read_at_least_once: false,
                            load: false,
                        }
                    }
                    Attachment::External(index) => {
                        VirtualResource {
                            refs: 0,
                            size: node.size,
                            id: Some(ResourceIndex {
                                index,
                                kind: resource.kind,
                                allocation: Allocation::External,
                            }),
                            reusable: false,
                            read_at_least_once: false,
                            load: false,
                        }
                    }
                    Attachment::Dependency(dep) => {
                        let dep_node = &self.nodes[dep.node.index()];
                        let dep_vres = &mut virtual_resources[dep_node.resources.index(dep.slot)];
                        // For now, rendering on top of the output of another node assumes
                        // that what is rendered on top of must be stored.
                        dep_vres.read_at_least_once = true;

                        VirtualResource {
                            refs: 0,
                            size: dep_vres.size,
                            id: dep_vres.id,
                            reusable: false,
                            read_at_least_once: false,
                            load: true,
                        }
                    }
                };
                virtual_resources[idx] = vres;
            }

            for dep in &node.reads {
                let vres_idx = self.nodes[dep.node.index()].resources.index(dep.slot);

                // TODO: if the resource aliases another virtual resource, update
                // that resource's ref count.
                let vres = &mut virtual_resources[vres_idx];
                vres.refs += 1;
                vres.read_at_least_once = true;
            }
        }

        let mut commands = Vec::with_capacity(sorted.len());
        let mut pass_data = Vec::with_capacity(sorted.len());

        // Second pass allocates physical resources.
        for node_id in &sorted {
            let node = &self.nodes[node_id.index()];

            let mut size = None;
            for idx in node.resources.all() {
                let res = &self.resources[idx];
                match res.attachment {
                    Attachment::Auto => {
                        let virt_res = &mut virtual_resources[idx];
                        let resource = temp_resources.get(TempResourceKey {
                            kind: res.kind,
                            size: virt_res.size,
                        });

                        // We can skip allocating unused resolve targets.
                        let allocate = if let Some(kind) = res.kind.as_texture() {
                            let is_resolve_target = !kind.is_depth_stencil() && node.msaa && !kind.is_msaa();

                            (is_resolve_target && virt_res.refs > 0)
                                || (node.msaa && kind.is_msaa())
                                || kind.is_depth_stencil()
                        } else {
                            true
                        };

                        if allocate {
                            virt_res.id = Some(ResourceIndex {
                                kind: res.kind,
                                allocation: Allocation::Temporary,
                                index: resource
                            });
                        }
                    }
                    Attachment::Dependency(dep) => {
                        let dep_node = &self.nodes[dep.node.index()];
                        let pres = virtual_resources[dep_node.resources.index(dep.slot)].id;

                        virtual_resources[idx].id = pres;
                    }
                    _ => {}
                }

                // If nobody reads the attachment, it can be reused right away.
                // This is typically the case for transient resources like depth or stencil
                // textures that tend to be used within a render pass but not consumed by
                // other nodes.
                let vres = &virtual_resources[idx];
                if let Some(id) = vres.id {
                    if vres.reusable && vres.refs == 0 {
                        temp_resources.recycle(TempResourceKey { kind: id.kind, size: vres.size }, id.index);
                    }
                }

                // TODO: using the size of the first output is fragile at best and only
                // makes sense for render passes.
                if size.is_none() {
                    size = Some(vres.size);
                }
            }

            for dep in &node.reads {
                let dep_node = &self.nodes[dep.node.index()];
                let dep_vres_idx = dep_node.resources.index(dep.slot);

                // TODO: if the resource aliases another virtual resource, update
                // that resource's ref count.
                let vres = &mut virtual_resources[dep_vres_idx];
                vres.refs -= 1;
                if let Some(id) = vres.id {
                    if vres.reusable && vres.refs == 0{
                        temp_resources.recycle(TempResourceKey { kind: id.kind, size: vres.size }, id.index);
                    }
                }
            }

            let pass_data = size.map(|size| {
                let pdata = RenderPassData {
                    target_size: size,
                };

                let mut pdata_idx = pass_data.len();
                for (idx, existing) in pass_data.iter().enumerate() {
                    if *existing == pdata {
                        pdata_idx = idx;
                    }
                }
                if pdata_idx == pass_data.len() {
                    pass_data.push(pdata);
                }

                pdata_idx as u16
            }).unwrap_or(0);

            commands.push(Command {
                resources: node.resources.clone(),
                msaa: node.msaa,
                depth: node.depth,
                stencil: node.stencil,
                label: node.label,
                node_id: *node_id,
                task_id: self.tasks[node_id.index()],
                pass_data_index: pass_data,
            });
        }

        for root in &self.roots {
            let offset = self.nodes[root.node_id.index()].resources.all().start;
            for i in 0..8 {
                if (root.stored_attachments & (1 << i)) != 0 {
                    virtual_resources[(offset + i) as usize].read_at_least_once = true;
                }
            }
        }

        let mut resources = Vec::with_capacity(virtual_resources.len());
        for res in &virtual_resources {
            resources.push(res.id.map(|id| NodeResourceInfo {
                resource: id,
                load: res.load,
                store: res.read_at_least_once,
            }));
        }

        Ok(BuiltGraph {
            temporary_resources: temp_resources.resources,
            resources,
            commands,
            pass_data,
        })
    }
}

struct TemporaryResources {
    available: Vec<(TempResourceKey, Vec<u16>)>,
    resources: Vec<TempResourceKey>,
}

impl TemporaryResources {
    fn get(&mut self, descriptor: TempResourceKey) -> u16 {
        let mut resource = None;
        for (desc, resources) in &mut self.available {
            if *desc == descriptor {
                resource = resources.pop();
                break;
            }
        }

        resource.unwrap_or_else(|| {
            let res = self.resources.len() as PhysicalResourceIndex;
            self.resources.push(descriptor);
            res
        })
    }

    pub fn recycle(&mut self, descriptor: TempResourceKey, index: u16) {
        let mut pool_idx = self.available.len();
        for (idx, (desc, _)) in self.available.iter().enumerate() {
            if *desc == descriptor {
                pool_idx = idx;
                break;
            }
        }
        if pool_idx == self.available.len() {
            self.available.push((descriptor, Vec::with_capacity(8)));
        }

        self.available[pool_idx].1.push(index);
    }
}

#[test]
fn test_nested() {
    let mut graph = RenderGraph::new();

    fn task(id: u64) -> NodeDescriptor<'static> {
        NodeDescriptor::new().task(TaskId(id))
    }

    let window_size = SurfaceIntSize::new(1920, 1200);
    let atlas_size = SurfaceIntSize::new(2048, 2048);
    let main = ColorAttachment {
        kind: TextureKind::color(),
        non_msaa: Attachment::External(0),
        msaa: Attachment::Auto,
    };
    let color = ColorAttachment::color();

    //let n42 = graph.add_node(
    //    &task(42)
    //        .size(window_size)
    //        .msaa(true)
    //        .attachments(&[Attachment::new(COLOR, main)])
    //        .attachment(Attachment::Auto)
    //        .depth_stencil(Attachment::Auto)
    //);

    // Ideally the resource allocation would look like this:
    //
    //  tmp0            n8  n5
    //                   \   \
    //  tmp1       n4     n6--n7
    //              \          \
    //  tmp0  n1     n3         n9
    //          \     \          \
    //  main     n0----n2---------n10
    let n0 = graph.add_node(&task(0).size(window_size).attachments(&[main]).depth_stencil(Attachment::Auto, true, false).label("main"));
    let n1 = graph.add_node(&task(1).size(atlas_size).attachments(&[color]).label("atlas"));
    let n2 = graph.add_node(&task(2).size(window_size).attachments(&[main.with_dependency(n0.color(0))]).depth_stencil(Attachment::Auto, true, false).label("main"));
    let n3 = graph.add_node(&task(3).size(atlas_size).attachments(&[color]).label("atlas"));
    let n4 = graph.add_node(&task(4).size(atlas_size).attachments(&[color]));
    let n5 = graph.add_node(&task(5).size(atlas_size).attachments(&[color]));
    let n6 = graph.add_node(&task(6).size(atlas_size).attachments(&[color]));
    let n7 = graph.add_node(&task(7).size(atlas_size).attachments(&[color.with_dependency(n6.color(0))]));
    let n8 = graph.add_node(&task(8).size(atlas_size).attachments(&[color]));
    let n9 = graph.add_node(
        &task(9)
            .size(atlas_size)
            .attachments(&[color])
            .read(&[n7.color(0)])
            .label("atlas")
    );
    let n10 = graph.add_node(
        &task(10)
            .size(window_size)
            .attachments(&[color.with_dependency(n2.color(0))])
            .depth_stencil(Attachment::Auto, true, false)
            .read(&[n9.color(0)])
            .label("main")
    );

    let n11 = graph.add_node(
        &task(11)
            .size(atlas_size)
            .attachments(&[color])
            .read(&[n9.color(0)])
    );

    graph.add_dependency(n0, n1.color(0));
    graph.add_dependency(n2, n3.color(0));
    graph.add_dependency(n3, n4.color(0));
    graph.add_dependency(n7, n5.color(0));
    graph.add_dependency(n6, n8.color(0));
    graph.add_dependency(n10, n9.color(0));

    graph.add_root(n10, 1);

    let mut sorted = Vec::new();
    graph.topological_sort(&mut sorted).unwrap();

    let commands = graph.schedule().unwrap();

    println!("sorted: {:?}", sorted);
    println!("allocations: {:?}", commands.temporary_resources);

    for command in &commands.commands {
        let label = graph.nodes[command.node_id.index()].label.unwrap_or("");
        println!(" - n{:?} {label}: {:?}", command.node_id.0, command.task_id);
        for res in &commands.resources[command.resources.all()] {
            let Some(res) = res else { continue; };
            let kind = res.resource.kind.as_texture().unwrap();
            let kind = if kind.is_color() { "color" }
                else if kind.is_depth_stencil() { "depth" }
                else { "unknown" };
            println!("   - {:?}({}_{}) store:{}", res.resource.allocation, kind, res.resource.index, res.store);
        }
    }

    // n11 should get culled out since it is not reachable from the root.
    assert!(!sorted.contains(&n11));
    assert_eq!(sorted.len(), graph.nodes.len() - 1);


    // node order: [n1, n0, n4, n3, n2, n8, n6, n5, n7, n9, n10]

}
