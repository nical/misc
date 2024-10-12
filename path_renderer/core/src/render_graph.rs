use std::{fmt, ops::Range, u16};
use smallvec::SmallVec;

use crate::pattern::{BindingsId, BindingsNamespace};

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

    pub fn attachment(self, slot: u8) -> Dependency {
        Dependency {
            node: self,
            slot,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct Dependency {
    pub node: NodeId,
    pub slot: u8
}

impl From<NodeId> for Dependency {
    fn from(node: NodeId) -> Self {
        Dependency {
            node,
            slot: 0,
        }
    }
}

impl fmt::Debug for Dependency {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{}:{}", self.node.0, self.slot)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TaskId(pub u64);

/// Identifies the parameters for creating a resource.
///
/// For example "An rgba8 texture of size 2048x2048"
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct ResourceKind(pub u8);

impl ResourceKind {
    pub const COLOR_TEXTURE: Self = ResourceKind(0);
    pub const ALPHA_TEXTURE: Self = ResourceKind(1);
    pub const DEPTH_STENCIL_TEXTURE: Self = ResourceKind(2);
    pub const MSAA_COLOR_TEXTURE: Self = ResourceKind(3);
    pub const MSAA_ALPHA_TEXTURE: Self = ResourceKind(4);
    pub const MSAA_DEPTH_STENCIL_TEXTURE: Self = ResourceKind(5);
    pub const STORAGE_BUFFER: Self = ResourceKind(6);

    pub const fn color_texture(msaa: bool) -> Self {
        if msaa {
            Self::MSAA_COLOR_TEXTURE
        } else {
            Self::COLOR_TEXTURE
        }
    }

    pub const fn alpha_texture(msaa: bool) -> Self {
        if msaa {
            Self::MSAA_ALPHA_TEXTURE
        } else {
            Self::ALPHA_TEXTURE
        }
    }

    pub const fn depth_stencil_texture(msaa: bool) -> Self {
        if msaa {
            Self::MSAA_DEPTH_STENCIL_TEXTURE
        } else {
            Self::DEPTH_STENCIL_TEXTURE
        }
    }

    #[inline]
    pub fn index(&self) -> usize {
        self.0 as usize
    }
}

impl fmt::Debug for ResourceKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{}", self.0)
    }
}

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
pub struct VirtualResourceInfo {
    /// The actual resource id.
    pub resource: PhysicalResourceId,
    /// True if any render node reads from it.
    pub store: bool,
}

type PhysicalResourceIdx = u16;

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct PhysicalResourceId {
    pub kind: ResourceKind,
    pub allocation: Allocation,
    pub index: PhysicalResourceIdx,
}

impl fmt::Debug for PhysicalResourceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}({:?}:{:?})", self.allocation, self.index, self.kind)
    }
}

#[derive(Debug)]
pub struct Command {
    pub resources: Range<usize>,
    pub node_id: NodeId,
    pub task_id: TaskId,
    /// Index into `BuiltGraph::pass_data`.
    pub pass_data: u16,
}

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
    pub resources: Vec<VirtualResourceInfo>,
    pub pass_data: Vec<RenderPassData>,
    pub commands: Vec<Command>,
}

impl BuiltGraph {
    pub fn resolve_binding(&self, binding: BindingsId) -> PhysicalResourceId {
        debug_assert!(binding.namespace() == BindingsNamespace::RenderGraph);
        self.resources[binding.index()].resource
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum GraphError {
    DependencyCycle(NodeId),
}

#[derive(Copy, Clone, Debug)]
pub enum Attachment {
    /// Automatically allocate the resource for this attachment.
    Auto { kind: ResourceKind, size: (u32, u32) },
    /// Use the specified resource for this attachment.
    External { kind: ResourceKind, size: (u32, u32), index: u16 },
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
    pub writes: &'l[Attachment],
}

impl NodeDescriptor<'static> {
    pub fn new() -> Self {
        NodeDescriptor {
            label: None,
            task: None,
            reads: &[],
            writes: &[],
        }
    }
}

impl<'l> NodeDescriptor<'l> {

    pub fn task(mut self, task: TaskId) -> Self {
        self.task = Some(task);

        self
    }

    pub fn label(mut self, label: &'static str) -> Self {
        self.label = Some(label);

        self
    }

    pub fn write<'a>(self, attachments: &'a [Attachment]) -> NodeDescriptor<'a>
    where 'l : 'a
    {
        NodeDescriptor {
            label: self.label,
            task: self.task,
            writes: attachments,
            reads: self.reads,
        }
    }

    pub fn read<'a>(self, inputs: &'a [Dependency]) -> NodeDescriptor<'a>
    where 'l : 'a
    {
        NodeDescriptor {
            label: self.label,
            task: self.task,
            writes: self.writes,
            reads: inputs,
        }
    }
}

struct Node {
    attachments: SmallVec<[Attachment; 4]>,
    reads: SmallVec<[Dependency; 4]>,
    readable: bool,
    first_virtual_resource: u16,
    label: Option<&'static str>,
    // writes: TODO
}

struct Root {
    node_id: NodeId,
    stored_attachments: u32,
}

pub struct RenderGraph {
    nodes: Vec<Node>,
    tasks: Vec<TaskId>,
    roots: Vec<Root>,
    next_virtual_resource: u16,
}

impl RenderGraph {
    pub fn new() -> Self {
        RenderGraph {
            nodes: Vec::new(),
            tasks: Vec::new(),
            roots: Vec::new(),
            next_virtual_resource: 0,
        }
    }

    pub fn clear(&mut self) {
        self.tasks.clear();
        self.nodes.clear();
        self.roots.clear();
        self.next_virtual_resource = 0;
    }

    pub fn add_node(&mut self, desc: &NodeDescriptor) -> NodeId {
        let id = NodeId::from_index(self.nodes.len());

        self.tasks.push(desc.task.expect("NodeDescriptor needs a task."));
        self.nodes.push(Node {
            attachments: SmallVec::from_slice(desc.writes),
            reads: SmallVec::from_slice(desc.reads),
            first_virtual_resource: self.next_virtual_resource,
            readable: true,
            label: desc.label,
        });

        self.next_virtual_resource += desc.writes.len() as u16;

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
        let node = &self.nodes[dep.node.index()];
        debug_assert!(node.attachments.len() > dep.slot as usize);
        let idx = node.first_virtual_resource as u32
            + dep.slot as u32;

        BindingsId::new(BindingsNamespace::RenderGraph, idx)
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
                for attachment in &node.attachments {
                    match attachment {
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

    pub fn schedule(&self) -> Result<BuiltGraph, GraphError> {
        // TODO: partial schedule
        let full_schedule = true;

        let mut sorted = Vec::new();
        self.topological_sort(&mut sorted)?;

        #[derive(Clone)]
        struct VirtualResource {
            refs: u16,
            size: (u32, u32),
            id: PhysicalResourceId,
            read_at_least_once: bool,
            reusable: bool,
        }

        let mut virtual_resources = vec![
            VirtualResource {
                refs: 0,
                size: (0, 0),
                id: PhysicalResourceId {
                    kind: ResourceKind(u8::MAX),
                    allocation: Allocation::Temporary,
                    index: PhysicalResourceIdx::MAX,
                },
                read_at_least_once: false,
                reusable: false,
            };
            self.next_virtual_resource as usize
        ];

        let mut temp_resources = TemporaryResources {
            available: Vec::with_capacity(16),
            resources: Vec::with_capacity(16),
        };

        // First pass allocates virtual resources and counts the number of times they
        // are used.
        for node_id in &sorted {
            let node = &self.nodes[node_id.index()];
            for (idx, attachment) in node.attachments.iter().enumerate() {
                let vres = match attachment {
                    Attachment::Auto { kind, size } => {
                        VirtualResource {
                            refs: 0,
                            size: *size,
                            id: PhysicalResourceId {
                                kind: *kind,
                                allocation: Allocation::Temporary,
                                index: PhysicalResourceIdx::MAX,
                            },
                            // Only make the physical resource available again if we
                            // know that no new reference can be added to it.
                            reusable: full_schedule || !node.readable,
                            read_at_least_once: false,
                        }
                    }
                    Attachment::External { kind, size, index: resource } => {
                        VirtualResource {
                            refs: 0,
                            size: *size,
                            id: PhysicalResourceId {
                                index: *resource,
                                kind: *kind,
                                allocation: Allocation::External,
                            },
                            reusable: false,
                            read_at_least_once: false,
                        }
                    }
                    Attachment::Dependency(dep) => {
                        let dep_node = &self.nodes[dep.node.index()];
                        let base_vres = dep_node.first_virtual_resource;
                        let dep_vres = &mut virtual_resources[base_vres as usize + dep.slot as usize];
                        // For now, rendering on top of the output of another node assumes
                        // that what is rendered on top of must be stored.
                        dep_vres.read_at_least_once = true;

                        VirtualResource {
                            refs: 0,
                            size: dep_vres.size,
                            id: dep_vres.id,
                            reusable: false,
                            read_at_least_once: false,
                        }
                    }
                };

                let offset = node.first_virtual_resource as usize;
                virtual_resources[offset + idx] = vres;
            }

            for dep in &node.reads {
                let vres_idx = node.first_virtual_resource as usize
                    + dep.slot as usize;

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
            let first_vres_idx = node.first_virtual_resource as usize;
            let mut vres_idx = first_vres_idx;

            let mut size = None;
            for attachment in &node.attachments {
                match attachment {
                    Attachment::Auto { kind, size } => {
                        let resource = temp_resources.get(TempResourceKey {  kind: *kind, size: *size });
                        virtual_resources[vres_idx].id.index = resource;
                    }
                    Attachment::Dependency(dep) => {
                        let dep_vres = self.nodes[dep.node.index()].first_virtual_resource as usize + dep.slot as usize;
                        let pres = virtual_resources[dep_vres].id;

                        virtual_resources[vres_idx].id = pres;
                    }
                    _ => {}
                };

                // If nobody reads the attachment, it can be reused right away.
                // This is typically the case for transient resources like depth or stencil
                // textures that tend to be used within a render pass but not consumed by
                // other nodes.
                let vres = &virtual_resources[vres_idx];
                if vres.reusable && vres.refs == 0 {
                    temp_resources.recycle(TempResourceKey { kind: vres.id.kind, size: vres.size }, vres.id.index);
                }

                // TODO: using the size of the first output is fragile at best and only
                // makes sense for render passes.
                if size.is_none() {
                    size = Some(vres.size);
                }

                vres_idx += 1;
            }

            for dep in &node.reads {
                let dep_vres_idx = node.first_virtual_resource as usize
                    + dep.slot as usize;

                // TODO: if the resource aliases another virtual resource, update
                // that resource's ref count.
                let vres = &mut virtual_resources[dep_vres_idx];
                vres.refs -= 1;
                if vres.reusable  && vres.refs == 0{
                    temp_resources.recycle(TempResourceKey { kind: vres.id.kind, size: vres.size }, vres.id.index);
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
                resources: first_vres_idx .. vres_idx,
                node_id: *node_id,
                task_id: self.tasks[node_id.index()],
                pass_data,
            });
        }

        for root in &self.roots {
            let offset = self.nodes[root.node_id.index()].first_virtual_resource;
            for i in 0..8 {
                if (root.stored_attachments & (1 << i)) != 0 {
                    virtual_resources[(offset + i) as usize].read_at_least_once = true;
                }
            }
        }

        let mut resources = Vec::with_capacity(virtual_resources.len());
        for res in &virtual_resources {
            resources.push(VirtualResourceInfo {
                resource: res.id,
                store: res.read_at_least_once,
            });
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
            let res = self.resources.len() as PhysicalResourceIdx;
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

    const COLOR: ResourceKind = ResourceKind(0);
    const DEPTH: ResourceKind = ResourceKind(1);
    let window_size = (1920, 1200);
    let atlas_size = (2048, 2048);
    let main = Attachment::External { kind: COLOR, size: window_size, index: 0 };
    let main_depth = Attachment::Auto { kind: DEPTH, size: window_size, };
    let color_auto = Attachment::Auto { kind: COLOR, size: atlas_size };

    // Ideally the resource allocation would look like this:
    //
    //  tmp0            n8  n5
    //                   \   \
    //  tmp1       n4     n6--n7
    //              \          \
    //  tmp0  n1     n3         n9
    //          \     \          \
    //  main     n0----n2---------n10
    let n0 = graph.add_node(&task(0).write(&[main, main_depth]).label("main"));
    let n1 = graph.add_node(&task(1).write(&[color_auto]).label("atlas"));
    let n2 = graph.add_node(&task(2).write(&[n0.attachment(0).into(), main_depth]).label("main"));
    let n3 = graph.add_node(&task(3).write(&[color_auto]).label("atlas"));
    let n4 = graph.add_node(&task(4).write(&[color_auto]));
    let n5 = graph.add_node(&task(5).write(&[color_auto]));
    let n6 = graph.add_node(&task(6).write(&[color_auto]));
    let n7 = graph.add_node(&task(7).write(&[n6.attachment(0).into()]));
    let n8 = graph.add_node(&task(8).write(&[color_auto]));
    let n9 = graph.add_node(
        &task(9)
            .write(&[color_auto])
            .read(&[n7.attachment(0)])
            .label("atlas")
    );
    let n10 = graph.add_node(
        &task(10)
            .write(&[n2.attachment(0).into(), main_depth])
            .read(&[n9.attachment(0)])
            .label("main")
    );

    let n11 = graph.add_node(
        &task(11)
            .write(&[color_auto])
            .read(&[n9.attachment(0)])
    );

    graph.add_dependency(n0, n1.attachment(0));
    graph.add_dependency(n2, n3.attachment(0));
    graph.add_dependency(n3, n4.attachment(0));
    graph.add_dependency(n7, n5.attachment(0));
    graph.add_dependency(n6, n8.attachment(0));
    graph.add_dependency(n10, n9.attachment(0));

    graph.add_root(n10, 1);

    let mut sorted = Vec::new();
    graph.topological_sort(&mut sorted).unwrap();

    let commands = graph.schedule().unwrap();

    println!("sorted: {:?}", sorted);
    println!("allocations: {:?}", commands.temporary_resources);

    for command in &commands.commands {
        let label = graph.nodes[command.node_id.index()].label.unwrap_or("");
        println!(" - n{:?} {label}: {:?}", command.node_id.0, command.task_id);
        for res in &commands.resources[command.resources.clone()] {
            let kind = if res.resource.kind == COLOR { "color" }
                else if res.resource.kind == DEPTH { "depth" }
                else { "unknown" };
            println!("   - {:?}({}_{}) store:{}", res.resource.allocation, kind, res.resource.index, res.store);
        }
    }

    // n11 should get culled out since it is not reachable from the root.
    assert!(!sorted.contains(&n11));
    assert_eq!(sorted.len(), graph.nodes.len() - 1);


    // node order: [n1, n0, n4, n3, n2, n8, n6, n5, n7, n9, n10]

}
