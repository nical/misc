use std::fmt;
use smallvec::SmallVec;

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
            slot: DependencySlot::Attachment(slot),
        }
    }

    pub fn msaa_attachment(self, slot: u8) -> Dependency {
        Dependency {
            node: self,
            slot: DependencySlot::MsaaAttachment(slot),
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub enum DependencySlot {
    Attachment(u8),
    MsaaAttachment(u8),
    DepthStencil,
    Writes(u8), // TODO: u8 is probably too limiting here
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct Dependency {
    pub node: NodeId,
    pub slot: DependencySlot
}

impl fmt::Debug for Dependency {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.slot {
            DependencySlot::Attachment(slot) => write!(f, "#{}:{}", self.node.0, slot),
            DependencySlot::MsaaAttachment(slot) => write!(f, "#{}:{}(msaa)", self.node.0, slot),
            DependencySlot::DepthStencil => write!(f, "#{}:depth/stencil", self.node.0),
            DependencySlot::Writes(slot) => write!(f, "#{}:writes{}", self.node.0, slot),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TaskId(pub u64);

#[derive(Copy, Clone, Debug)]
pub struct Resource {
    pub kind: ResourceKind,
    pub allocation: Allocation,
}

#[derive(Clone)]
pub struct Resources {
    pub attachments: SmallVec<[Resource; 4]>,
    pub msaa: bool,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ResourceKind {
    ColorTexture,
    AlphaTexture,
    DepthTexture,
    StencilTexture,
    Buffer,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Allocation {
    Auto,
    Fixed(u16),
    None,
}

#[derive(Clone, Debug, PartialEq)]
pub enum GraphError {
    DependencyCycle(NodeId),
}

#[derive(Copy, Clone, Debug)]
pub enum Attachment {
    Dependency(Dependency),
    Texture { kind: ResourceKind, allocation: Allocation }
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
    pub color_attachments: &'l[Attachment],
    pub depth_stencil_attachment: Option<Attachment>,
    pub inputs: &'l[Dependency],
}

impl NodeDescriptor<'static> {
    pub fn new() -> Self {
        NodeDescriptor {
            label: None,
            task: None,
            color_attachments: &[],
            depth_stencil_attachment: None,
            inputs: &[],
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

    pub fn render_to<'a>(self, attachments: &'a [Attachment]) -> NodeDescriptor<'a>
    where 'l : 'a
    {
        NodeDescriptor {
            label: self.label,
            task: self.task,
            color_attachments: attachments,
            depth_stencil_attachment: self.depth_stencil_attachment,
            inputs: self.inputs,
        }
    }

    pub fn depth_stencil(mut self, attachment: Attachment) -> Self {
        self.depth_stencil_attachment = Some(attachment);

        self
    }

    pub fn read<'a>(self, inputs: &'a [Dependency]) -> NodeDescriptor<'a>
    where 'l : 'a
    {
        NodeDescriptor {
            label: self.label,
            task: self.task,
            color_attachments: self.color_attachments,
            depth_stencil_attachment: self.depth_stencil_attachment,
            inputs: inputs,
        }
    }
}

struct Node {
    color_attachments: SmallVec<[Attachment; 1]>,
    depth_stencil_attachments: Option<Attachment>,
    reads: SmallVec<[Dependency; 4]>,
    // writes: TODO
}

pub struct RenderGraph {
    nodes: Vec<Node>,
    tasks: Vec<Option<TaskId>>,
    active: Vec<bool>,
    roots: Vec<NodeId>,
}

impl RenderGraph {
    pub fn new() -> Self {
        RenderGraph {
            nodes: Vec::new(),
            tasks: Vec::new(),
            active: Vec::new(),
            roots: Vec::new(),
        }
    }

    pub fn clear(&mut self) {
        self.tasks.clear();
        self.nodes.clear();
        self.active.clear();
        self.roots.clear();
    }

    pub fn add_node(&mut self, desc: &NodeDescriptor) -> NodeId {
        let id = NodeId::from_index(self.nodes.len());
        self.tasks.push(desc.task);
        self.nodes.push(Node {
            color_attachments: SmallVec::from_slice(desc.color_attachments),
            depth_stencil_attachments: desc.depth_stencil_attachment,
            reads: SmallVec::from_slice(desc.inputs),
        });

        id
    }

    pub fn add_dependency(&mut self, node: NodeId, dep: Dependency) {
        self.nodes[node.index()].reads.push(dep);
    }

    pub fn add_root(&mut self, root: NodeId) {
        self.roots.push(root);
    }

    pub fn schedule(&self) -> Result<Vec<Command>, GraphError> {
        let mut sorted = Vec::new();
        self.topological_sort(&mut sorted)?;

        todo!()
    }

    fn topological_sort(&self, sorted: &mut Vec<NodeId>) -> Result<(), GraphError> {
        sorted.reserve(self.nodes.len());

        let mut added = vec![false; self.nodes.len()];
        let mut cycle_check = vec![false; self.nodes.len()];
        let mut stack: Vec<NodeId> = Vec::with_capacity(self.nodes.len());
        for idx in 0..self.nodes.len() {
            let root = NodeId::from_index(idx);
            //println!("- root {root:?}");
            if added[root.index()] {
                continue;
            }

            stack.push(root);
            'traversal: while let Some(id) = stack.last().cloned() {
                //println!("  - node {id:?} cycle:{:?}, added:{:?}, stack {stack:?}", cycle_check[id.index()], added[id.index()]);
                //println!("   - cycles {cycle_check:?}");
                //println!("   - added  {added:?}");
                if added[id.index()] {
                    stack.pop();
                    continue;
                }


                cycle_check[id.index()] = true;
                let node = &self.nodes[id.index()];
                for attachment in &node.color_attachments {
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
                        Attachment::Texture { .. } => {}
                    }
                }
                if let Some(Attachment::Dependency(dep)) = &node.depth_stencil_attachments {
                    if !added[dep.node.index()] {
                        if cycle_check[dep.node.index()] {
                            return Err(GraphError::DependencyCycle(dep.node));
                        }

                        stack.push(dep.node);
                        continue 'traversal;
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
}

pub struct Command {
    pub node_id: NodeId,
    pub task_id: TaskId,
}

#[test]
fn test_nested() {
    let mut graph = RenderGraph::new();

    fn task(id: u64) -> NodeDescriptor<'static> {
        NodeDescriptor::new().task(TaskId(id))
    }

    let color_auto = Attachment::Texture { kind: ResourceKind::ColorTexture, allocation: Allocation::Auto };

    //  tmp0            n8  n5
    //                   \   \
    //  tmp1       n4     n6--n7
    //              \          \
    //  tmp0  n1     n3         n9
    //          \     \          \
    //  main     n0----n2---------n10
    let n0 = graph.add_node(&task(0).render_to(&[color_auto]));
    let n1 = graph.add_node(&task(1).render_to(&[color_auto]));
    let n2 = graph.add_node(&task(2).render_to(&[n0.attachment(0).into()]));
    let n3 = graph.add_node(&task(3).render_to(&[color_auto]));
    let n4 = graph.add_node(&task(4).render_to(&[color_auto]));
    let n5 = graph.add_node(&task(5).render_to(&[color_auto]));
    let n6 = graph.add_node(&task(6).render_to(&[color_auto]));
    let n7 = graph.add_node(&task(7).render_to(&[n6.attachment(0).into()]));
    let n8 = graph.add_node(&task(8).render_to(&[color_auto]));
    let n9 = graph.add_node(
        &task(9)
            .render_to(&[color_auto])
            .read(&[n7.attachment(0)])
    );
    let n10 = graph.add_node(
        &task(10)
            .render_to(&[n2.attachment(0).into()])
            .read(&[n9.attachment(0)])
    );

    graph.add_dependency(n0, n1.attachment(0));
    graph.add_dependency(n2, n3.attachment(0));
    graph.add_dependency(n3, n4.attachment(0));
    graph.add_dependency(n7, n5.attachment(0));
    graph.add_dependency(n6, n8.attachment(0));
    graph.add_dependency(n10, n9.attachment(0));

    graph.add_root(n10);

    let mut sorted = Vec::new();
    graph.topological_sort(&mut sorted).unwrap();
    // TODO: Right now we get [#1, #0, #4, #3, #2, #5, #8, #6, #7, #9, #10]
    // but this is not quite what we want: "#5, #8, #6, #7," should be #8, #6, #5, #7,
    // To maximize our ability to reuse render targets.
    println!("sorted: {:?}", sorted);

    assert_eq!(sorted.len(), graph.nodes.len());
}
