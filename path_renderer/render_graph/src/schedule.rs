use std::ops::Range;

use core::smallvec::SmallVec;
use core::instance::Passes;
use core::render_pass::{AttachmentFlags, BuiltRenderPass, ColorAttachment};
use core::resources::{Allocation, BufferKind, ResourceIndex, ResourceKey, ResourceKind, TextureKind};
use core::BindingsId;
use crate::{Dependency, NodeDescriptor, NodeId, NodeKind, Resource, Slot};
use crate::ResourceFlags;


#[derive(Copy, Clone, Debug)]
struct NodeResource {
    kind: ResourceKind,
    resource: Resource,
    flags: ResourceFlags,
}

pub struct RenderGraph {
    nodes: Vec<Node>,
    resources: Vec<NodeResource>,
    roots: Vec<Dependency>,
    next_virtual_resource: u16,

    built_render_passes: Vec<BuiltRenderPass>,
}

impl RenderGraph {
    pub fn new() -> Self {
        RenderGraph {
            nodes: Vec::new(),
            resources: Vec::new(),
            roots: Vec::new(),
            next_virtual_resource: 0,
            built_render_passes: Vec::new(),
        }
    }

    pub fn add_node(&mut self, desc: &NodeDescriptor) -> NodeId {
        let id = NodeId::from_index(self.nodes.len());

        let resources_start = self.resources.len();
        for attachment in desc.attachments {
            let kind = attachment.kind.with_attachment();
            let msaa_kind = kind.with_msaa(true);
            let mut flags = ResourceFlags::empty();
            // If msaa is used, this is a resolve target. Only allocate
            // these if they are used.
            flags.set(ResourceFlags::LAZY, desc.msaa);
            flags.set(ResourceFlags::CLEAR, attachment.clear && !desc.msaa);
            if let Resource::Dependency(..) = attachment.non_msaa {
                flags.set(ResourceFlags::LOAD, !attachment.clear);
            }
            self.resources.push(NodeResource {
                kind: kind.as_resource(),
                resource: attachment.non_msaa,
                flags,
            });
            let mut flags = ResourceFlags::empty();
            flags.set(ResourceFlags::LAZY, !desc.msaa);
            flags.set(ResourceFlags::CLEAR, attachment.clear);
            if let Resource::Dependency(..) = attachment.msaa {
                flags.set(ResourceFlags::LOAD, !attachment.clear);
            }
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

        self.nodes.push(Node {
            kind: desc.kind,
            resources: NodeResources {
                resources_start: resources_start as u16,
                resources_end: resources_end as u16,
                attachments_end: attachments_end as u16,
                depth_stencil: ds_resource,
            },
            msaa: desc.msaa,
            size: desc.size.map(|s| (s.width as u16, s.height as u16)).unwrap_or((0, 0)),
            dependecies: SmallVec::from_slice(desc.reads),
            readable: true,
            label: desc.label,
        });

        self.next_virtual_resource += desc.attachments.len() as u16;

        id
    }

    #[inline]
    pub(crate) fn add_dependencies(&mut self, node: NodeId, deps: &[Dependency]) {
        self.nodes[node.index()].dependecies.extend_from_slice(deps);
    }

    pub(crate) fn add_built_render_pass(&mut self, node: NodeId, pass: BuiltRenderPass) {
        let index = node.0 as usize;
        while self.built_render_passes.len() <= index {
            self.built_render_passes.push(BuiltRenderPass::empty());
        }

        self.built_render_passes[index] = pass;
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

    pub fn schedule(&mut self, passes: &mut Passes) -> Result<(), Box<GraphError>> {
        schedule_graph(self, passes)
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
    size: (u16, u16),
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

#[derive(Clone, Debug, PartialEq)]
pub enum GraphError {
    DependencyCycle(NodeId),
    InvalidTextureSize { label: Option<&'static str>, expected: (u16, u16), got: (u16, u16) },
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
    graph: &mut RenderGraph,
    passes: &mut Passes,
) -> Result<(), Box<GraphError>> {
    // TODO: partial schedule
    let full_schedule = true;

    let mut sorted = Vec::new();
    topological_sort(&graph, &mut sorted)?;

    #[derive(Clone)]
    struct VirtualResource {
        refs: u16,
        kind: ResourceKind,
        resolved_index: Option<ResourceIndex>,
        store: bool,
        reusable: bool,
        clear: bool,
        load: bool,
    }

    let mut virtual_resources = vec![
        VirtualResource {
            refs: 0,
            reusable: false,
            resolved_index: None,
            kind: BufferKind::storage().as_resource(),
            clear: false,
            store: false,
            load: false,
        };
        graph.resources.len()
    ];

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
                        clear: resource.flags.contains(ResourceFlags::CLEAR),
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
                        clear: resource.flags.contains(ResourceFlags::CLEAR),
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
                        clear: resource.flags.contains(ResourceFlags::CLEAR),
                        load: resource.flags.contains(ResourceFlags::LOAD),
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
                    let resource = passes.resources.get(ResourceKey {
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
                    passes.resources.recycle(ResourceKey { kind: vres.kind, size: node.size }, id.index);
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
                    passes.resources.recycle(ResourceKey { kind: vres.kind, size: node.size }, id.index);
                }
            }
        }

        match node.kind {
            NodeKind::Render => {
                let mut color = [ColorAttachment { non_msaa: None, msaa: None, flags: AttachmentFlags { clear: false, load: false, store: false,}}; 3];
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
                            flags: AttachmentFlags {
                                clear: res.clear,
                                load: res.load,
                                store: res.store,
                            }
                        }
                    } else {
                        let res = &virtual_resources[non_msaa_idx];
                        color[i] = ColorAttachment {
                            non_msaa: Some(BindingsId::graph(non_msaa_idx as u16)),
                            msaa: None,
                            flags: AttachmentFlags {
                                clear: res.clear,
                                load: res.load,
                                store: res.store,
                            }
                        }
                    }
                }

                let depth_stencil = node.resources.depth_stencil.map(|idx| {
                    BindingsId::graph(idx)
                });

                // TODO: We only consider the case where we don't have built
                // render passes for testing purposes.
                let mut built_pass = BuiltRenderPass::default();
                if graph.built_render_passes.len() > 0 {
                    std::mem::swap(
                        &mut built_pass,
                        &mut graph.built_render_passes[node_id.index()],
                    );
                }

                if let Some(label) = node.label {
                    built_pass.set_label(label)
                }
                built_pass.set_color_attachments(&color);
                built_pass.set_depth_stencil_attachment(depth_stencil);

                passes.push_render_pass(built_pass);
            }
            _ => {
                todo!()
            }
        }
    }

    for (idx, res) in virtual_resources.iter().enumerate() {
        passes.set_binding(idx, res.resolved_index);
    }

    Ok(())
}

#[test]
fn test_nested() {
    use super::{NodeDescriptor, ColorAttachment, TextureKind};
    use crate::units::SurfaceIntSize;

    let mut graph = RenderGraph::new();

    fn node() -> NodeDescriptor<'static> {
        NodeDescriptor::new()
    }

    let window_size = SurfaceIntSize::new(1920, 1200);
    let atlas_size = SurfaceIntSize::new(2048, 2048);
    let main = ColorAttachment {
        kind: TextureKind::color(),
        non_msaa: Resource::External(0),
        msaa: Resource::Auto,
        clear: true,
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
    let n0 = graph.add_node(&node().size(window_size).attachments(&[main]).depth_stencil(Resource::Auto, true, false).label("main"));
    let n1 = graph.add_node(&node().size(atlas_size).attachments(&[color]).label("atlas"));
    let n2 = graph.add_node(&node().size(window_size).attachments(&[main.with_dependency(n0.color(0))]).depth_stencil(Resource::Auto, true, false).label("main"));
    let n3 = graph.add_node(&node().size(atlas_size).attachments(&[color]).label("atlas"));
    let n4 = graph.add_node(&node().size(atlas_size).attachments(&[color]));
    let n5 = graph.add_node(&node().size(atlas_size).attachments(&[color]));
    let n6 = graph.add_node(&node().size(atlas_size).attachments(&[color]));
    let n7 = graph.add_node(&node().size(atlas_size).attachments(&[color.with_dependency(n6.color(0))]));
    let n8 = graph.add_node(&node().size(atlas_size).attachments(&[color]));
    let n9 = graph.add_node(
        &node()
            .size(atlas_size)
            .attachments(&[color])
            .read(&[n7.color(0)])
            .label("atlas")
    );
    let n10 = graph.add_node(
        &node()
            .size(window_size)
            .attachments(&[color.with_dependency(n2.color(0))])
            .depth_stencil(Resource::Auto, true, false)
            .read(&[n9.color(0)])
            .label("main")
    );

    let n11 = graph.add_node(
        &node()
            .size(atlas_size)
            .attachments(&[color])
            .read(&[n9.color(0)])
    );

    graph.add_dependencies(n0, &[n1.color(0)]);
    graph.add_dependencies(n2, &[n3.color(0)]);
    graph.add_dependencies(n3, &[n4.color(0)]);
    graph.add_dependencies(n7, &[n5.color(0)]);
    graph.add_dependencies(n6, &[n8.color(0)]);
    graph.add_dependencies(n10, &[n9.color(0)]);

    graph.add_root(n10.color(0));

    let mut sorted = Vec::new();
    topological_sort(&graph, &mut sorted).unwrap();

    let mut passes = Passes::new();

    let _ = graph.schedule(&mut passes).unwrap();

    println!("sorted: {:?}", sorted);
    println!("allocations: {:?}", passes.resources.descriptors());

    // n11 should get culled out since it is not reachable from the root.
    assert!(!sorted.contains(&n11));
    assert_eq!(sorted.len(), graph.nodes.len() - 1);


    // node order: [n1, n0, n4, n3, n2, n8, n6, n5, n7, n9, n10]
}
