use std::{fmt, ops::Range};

use smallvec::SmallVec;

use crate::graph::{GraphSystem, PassId};
use crate::render_pass::{AttathchmentFlags, ColorAttachment, RenderPassIo};
use crate::{BindingsId, BindingsNamespace};
use super::{Allocation, BufferKind, Dependency, NodeDescriptor, NodeId, NodeKind, Resource, ResourceKind, Slot, TaskId, TempResourceKey, TextureKind};
use super::ResourceFlags;

#[derive(Copy, Clone, Debug)]
struct NodeResource {
    kind: ResourceKind,
    resource: Resource,
    flags: ResourceFlags,
}

pub struct RenderGraph {
    nodes: Vec<Node>,
    resources: Vec<NodeResource>,
    tasks: Vec<TaskId>,
    roots: Vec<Dependency>,
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
            let mut flags = ResourceFlags::empty();
            flags.set(ResourceFlags::LAZY, desc.msaa);
            self.resources.push(NodeResource {
                kind: kind.as_resource(),
                resource: attachment.non_msaa,
                // If msaa is used, this is a resolve target. Only allocate
                // these if they are used.
                flags,
            });
            let mut flags = ResourceFlags::empty();
            flags.set(ResourceFlags::LAZY, !desc.msaa);
            self.resources.push(NodeResource {
                kind: msaa_kind.as_resource(),
                resource: attachment.msaa,
                flags,
            });
        }
        let attachments_end = self.resources.len();
        let mut ds_resource = None;
        if let Some((attachment, _, _)) = desc.depth_stencil {
            ds_resource = Some(self.resources.len() as u16);
            self.resources.push(NodeResource {
                kind: TextureKind::depth_stencil().with_msaa(desc.msaa).as_resource(),
                resource: attachment,
                flags: ResourceFlags::empty(),
            });
        }

        let resources_end = self.resources.len();

        self.tasks.push(desc.task.expect("NodeDescriptor needs a task."));
        self.nodes.push(Node {
            kind: desc.kind,
            resources: NodeResources {
                resources_start: resources_start as u16,
                resources_end: resources_end as u16,
                attachments_end: attachments_end as u16,
                depth_stencil: ds_resource,
            },
            msaa: desc.msaa,
            size: desc.size.map(|s| s.to_u32().to_tuple()).unwrap_or((0, 0)),
            dependecies: SmallVec::from_slice(desc.reads),
            readable: true,
            label: desc.label,
        });

        self.next_virtual_resource += desc.attachments.len() as u16;

        id
    }

    pub fn add_dependency(&mut self, node: NodeId, dep: Dependency) {
        debug_assert!(self.nodes[dep.node.index()].readable);
        self.nodes[node.index()].dependecies.push(dep);
    }

    #[inline]
    pub(crate) fn add_dependencies(&mut self, node: NodeId, deps: &[Dependency]) {
        self.nodes[node.index()].dependecies.extend_from_slice(deps);
    }

    pub fn add_root(&mut self, root: Dependency) {
        self.roots.push(root);
    }

    /// After this call, no dependency to this node can be added.
    pub fn finish_node(&mut self, node: NodeId) {
        self.nodes[node.index()].readable = false;
    }

    pub fn get_binding(&self, dep: Dependency) -> BindingsId {
        let idx = self.nodes[dep.node.index()].resources.index(dep.slot);
        BindingsId::graph(idx as u16)
    }

    pub fn schedule(&self, systems: &mut [&mut dyn GraphSystem], commands: &mut Vec<PassId>) -> Result<GraphBindings, Box<GraphError>> {
        schedule_graph(self, systems, commands)
    }
}

struct Node {
    label: Option<&'static str>,
    kind: NodeKind,
    resources: NodeResources,
    dependecies: SmallVec<[Dependency; 4]>,
    // render-specific, could be extracted out of Node.
    msaa: bool,
    // Size is also render-specific but a bit annoying to
    // move out of Node because the graph uses it to determine
    // the size of allocations for attachments.
    size: (u32, u32),
    // This is not really used at the moment.
    readable: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct NodeResources {
    pub(super) resources_start: u16,
    attachments_end: u16,
    resources_end: u16,
    depth_stencil: Option<u16>,
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

    pub(super) fn color_attachments(&self) -> Range<usize> {
        self.resources_start as usize .. self.attachments_end as usize
    }

    pub(super) fn depth_stencil(&self) -> Option<usize> {
        self.depth_stencil.map(|ds| ds as usize)
    }
}

// TODO: turn this into something less graph-specific, so that other
// things can use it.
#[derive(Debug, Default)]
pub struct GraphBindings {
    pub temporary_resources: Vec<TempResourceKey>,
    pub resources: Vec<Option<NodeResourceInfo>>,
}

impl GraphBindings {
    pub fn resolve_binding(&self, binding: BindingsId) -> Option<ResourceIndex> {
        debug_assert!(binding.namespace() == BindingsNamespace::RenderGraph);
        self.resources[binding.index()].as_ref().map(|res| res.resolved_index)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum GraphError {
    DependencyCycle(NodeId),
    InvalidTextureSize { label: Option<&'static str>, expected: (u32, u32), got: (u32, u32) },
}

#[derive(Debug)]
pub struct NodeResourceInfo {
    pub kind: ResourceKind,
    pub resolved_index: ResourceIndex,
    /// True if any render node reads from it.
    pub load: bool,
    pub store: bool,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct ResourceIndex {
    pub allocation: Allocation,
    pub index: u16,
}

impl fmt::Debug for ResourceIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}({:?})", self.allocation, self.index)
    }
}

fn topological_sort(graph: &RenderGraph, sorted: &mut Vec<NodeId>) -> Result<(), Box<GraphError>> {
    sorted.reserve(graph.nodes.len());

    let mut added = vec![false; graph.nodes.len()];
    let mut cycle_check = vec![false; graph.nodes.len()];
    let mut stack: Vec<NodeId> = Vec::with_capacity(graph.nodes.len());

    for root in &graph.roots {
        let root = root.node;
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
            let node = &graph.nodes[id.index()];
            for res in &graph.resources[node.resources.all()] {
                match res.resource {
                    Resource::Dependency(dep) => {
                        if !added[dep.node.index()] {
                            if cycle_check[dep.node.index()] {
                                return Err(Box::new(GraphError::DependencyCycle(dep.node)));
                            }

                            stack.push(dep.node);
                            continue 'traversal;
                        }
                    }
                    _ => {}
                }
            }
            for dep in &node.dependecies {
                if !added[dep.node.index()] {
                    if cycle_check[dep.node.index()] {
                        return Err(Box::new(GraphError::DependencyCycle(dep.node)));
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

pub fn schedule_graph(
    graph: &RenderGraph,
    systems: &mut [&mut dyn GraphSystem],
    commands: &mut Vec<PassId>,
) -> Result<GraphBindings, Box<GraphError>> {
    // TODO: partial schedule
    let full_schedule = true;

    let mut sorted = Vec::new();
    topological_sort(&graph, &mut sorted)?;

    // TODO: lot of overlap with NodeResourceInfo. Could be split into
    // the vector of Option<NodeResourceInfo> that is eventually returned
    // and the remaining members in a side array.
    #[derive(Clone)]
    struct VirtualResource {
        refs: u16,
        kind: ResourceKind,
        resolved_index: Option<ResourceIndex>,
        store: bool,
        reusable: bool,
        load: bool,
    }

    let mut virtual_resources = vec![
        VirtualResource {
            refs: 0,
            reusable: false,
            resolved_index: None,
            kind: BufferKind::storage().as_resource(),
            store: false,
            load: false,
        };
        graph.resources.len()
    ];

    let mut temp_resources = TemporaryResources {
        available: Vec::with_capacity(16),
        resources: Vec::with_capacity(16),
    };

    // First pass allocates virtual resources and counts the number of times they
    // are used.
    for node_id in &sorted {
        let node = &graph.nodes[node_id.index()];
        for idx in node.resources.all() {
            let resource = &graph.resources[idx];

            let vres = match resource.resource {
                Resource::Auto => {
                    VirtualResource {
                        refs: 0,
                        reusable: full_schedule || !node.readable,
                        kind: resource.kind,
                        resolved_index: None,
                        // Only make the physical resource available again if we
                        // know that no new reference can be added to it.
                        store: false,
                        load: resource.flags.contains(ResourceFlags::LOAD),
                    }
                }
                Resource::External(index) => {
                    VirtualResource {
                        refs: 0,
                        reusable: false,
                        kind: resource.kind,
                        resolved_index: Some(ResourceIndex {
                            index,
                            allocation: Allocation::External,
                        }),
                        store: false,
                        load: resource.flags.contains(ResourceFlags::LOAD),
                    }
                }
                Resource::Dependency(dep) => {
                    let dep_node = &graph.nodes[dep.node.index()];
                    let dep_vres = &mut virtual_resources[dep_node.resources.index(dep.slot)];
                    // For now, rendering on top of the output of another node assumes
                    // that what is rendered on top of must be stored.
                    dep_vres.store = true;

                    if dep_node.size != node.size {
                        return Err(Box::new(GraphError::InvalidTextureSize {
                            label: node.label,
                            expected: dep_node.size,
                            got: node.size,
                        }));
                    }

                    VirtualResource {
                        refs: 0,
                        reusable: false,

                        kind: resource.kind,
                        resolved_index: dep_vres.resolved_index,
                        store: false,
                        load: !resource.flags.contains(ResourceFlags::CLEAR),
                    }
                }
            };
            virtual_resources[idx] = vres;
        }

        for dep in &node.dependecies {
            let vres_idx = graph.nodes[dep.node.index()].resources.index(dep.slot);

            // TODO: if the resource aliases another virtual resource, update
            // that resource's ref count.
            let vres = &mut virtual_resources[vres_idx];
            vres.refs += 1;
            vres.store = true;
        }
    }

    for root in &graph.roots {
        let node = &graph.nodes[root.node.index()];
        virtual_resources[node.resources.index(root.slot)].store = true;
    }

    // Second pass allocates physical resources and writes the command buffer.
    for node_id in &sorted {
        let node = &graph.nodes[node_id.index()];

        for idx in node.resources.all() {
            let res = &graph.resources[idx];
            match res.resource {
                Resource::Auto => {
                    let virt_res = &mut virtual_resources[idx];
                    let resource = temp_resources.get(TempResourceKey {
                        kind: res.kind,
                        size: node.size,
                    });

                    // We can skip allocating unused resolve targets.
                    let allocate = virt_res.refs > 0 || !res.flags.contains(ResourceFlags::LAZY);

                    if allocate {
                        virt_res.resolved_index = Some(ResourceIndex {
                            allocation: Allocation::Temporary,
                            index: resource
                        });
                    }
                }
                Resource::Dependency(dep) => {
                    let dep_node = &graph.nodes[dep.node.index()];
                    let pres = virtual_resources[dep_node.resources.index(dep.slot)].resolved_index;

                    virtual_resources[idx].resolved_index = pres;
                }
                _ => {}
            }

            // If nobody reads the attachment, it can be reused right away.
            // This is typically the case for transient resources like depth or stencil
            // textures that tend to be used within a render pass but not consumed by
            // other nodes.
            let vres = &virtual_resources[idx];
            if let Some(id) = vres.resolved_index {
                if vres.reusable && vres.refs == 0 {
                    temp_resources.recycle(TempResourceKey { kind: vres.kind, size: node.size }, id.index);
                }
            }
        }

        for dep in &node.dependecies {
            let dep_node = &graph.nodes[dep.node.index()];
            let dep_vres_idx = dep_node.resources.index(dep.slot);

            // TODO: if the resource aliases another virtual resource, update
            // that resource's ref count.
            let vres = &mut virtual_resources[dep_vres_idx];
            vres.refs -= 1;
            if let Some(id) = vres.resolved_index {
                if vres.reusable && vres.refs == 0{
                    temp_resources.recycle(TempResourceKey { kind: vres.kind, size: node.size }, id.index);
                }
            }
        }

        match node.kind {
            NodeKind::Render => {
                let mut color = [ColorAttachment { non_msaa: None, msaa: None, flags: AttathchmentFlags { load: false, store: false,}}; 3];
                let mut attachments_iter = node.resources.color_attachments();
                for i in 0..3 {
                    let non_msaa = attachments_iter.next();
                    let msaa = attachments_iter.next();
                    debug_assert!(non_msaa.is_none() == msaa.is_none());
                    let (Some(msaa_idx), Some(non_msaa_idx)) = (msaa, non_msaa) else {
                        break;
                    };

                    if node.msaa {
                        let res = &virtual_resources[msaa_idx];
                        color[i] = ColorAttachment {
                            non_msaa: Some(BindingsId::graph(non_msaa_idx as u16)),
                            msaa: Some(BindingsId::graph(msaa_idx as u16)),
                            flags: AttathchmentFlags {
                                load: res.load,
                                store: res.store,
                            }
                        }
                    } else {
                        let res = &virtual_resources[non_msaa_idx];
                        color[i] = ColorAttachment {
                            non_msaa: Some(BindingsId::graph(non_msaa_idx as u16)),
                            msaa: None,
                            flags: AttathchmentFlags {
                                load: res.load,
                                store: res.store,
                            }
                        }
                    }
                }

                let depth_stencil = node.resources.depth_stencil.map(|idx| {
                    BindingsId::graph(idx)
                });

                let io = RenderPassIo {
                    label: node.label,
                    color_attachments: color,
                    depth_stencil_attachment: depth_stencil,
                };

                let system_id = 0; // TODO
                let pass_index = graph.tasks[node_id.index()].0;

                let pass_id = PassId {
                    system: system_id,
                    index: pass_index as u16,
                };

                systems[system_id as usize].set_pass_io(pass_id, io);

                commands.push(pass_id);
            }
            _ => {
                todo!()
            }
        }
    }

    let mut resources = Vec::with_capacity(virtual_resources.len());
    for res in &virtual_resources {
        resources.push(res.resolved_index.map(|id| NodeResourceInfo {
            kind: res.kind,
            resolved_index: id,
            load: res.load,
            store: res.store,
        }));
    }

    Ok(GraphBindings {
        temporary_resources: temp_resources.resources,
        resources,
    })
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
            let res = self.resources.len() as u16;
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
    use super::{NodeDescriptor, TaskId, ColorAttachment, TextureKind};
    use crate::units::SurfaceIntSize;

    let mut graph = RenderGraph::new();

    fn task(id: u64) -> NodeDescriptor<'static> {
        NodeDescriptor::new().task(TaskId(id))
    }

    let window_size = SurfaceIntSize::new(1920, 1200);
    let atlas_size = SurfaceIntSize::new(2048, 2048);
    let main = ColorAttachment {
        kind: TextureKind::color(),
        non_msaa: Resource::External(0),
        msaa: Resource::Auto,
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
    let n0 = graph.add_node(&task(0).size(window_size).attachments(&[main]).depth_stencil(Resource::Auto, true, false).label("main"));
    let n1 = graph.add_node(&task(1).size(atlas_size).attachments(&[color]).label("atlas"));
    let n2 = graph.add_node(&task(2).size(window_size).attachments(&[main.with_dependency(n0.color(0))]).depth_stencil(Resource::Auto, true, false).label("main"));
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
            .depth_stencil(Resource::Auto, true, false)
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

    graph.add_root(n10.color(0));

    let mut sorted = Vec::new();
    topological_sort(&graph, &mut sorted).unwrap();

    let mut render_nodes = super::render_nodes::RenderNodes::new();
    let systems: &mut[&mut dyn GraphSystem] = &mut [
        &mut render_nodes,
    ];

    let mut passes = Vec::new();

    let bindings = graph.schedule(systems, &mut passes).unwrap();

    println!("sorted: {:?}", sorted);
    println!("allocations: {:?}", bindings.temporary_resources);

    // n11 should get culled out since it is not reachable from the root.
    assert!(!sorted.contains(&n11));
    assert_eq!(sorted.len(), graph.nodes.len() - 1);


    // node order: [n1, n0, n4, n3, n2, n8, n6, n5, n7, n9, n10]
}
