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

    pub fn output(self, slot: u8) -> Dependency {
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

impl fmt::Debug for Dependency {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{}:{}", self.node.0, self.slot)
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

#[derive(Clone)]
pub struct NodeDescriptor<'l> {
    pub label: Option<&'static str>,
    pub dependencies: &'l[Dependency],
    pub task: TaskId,
    pub after: Option<NodeId>,
    pub resources: Resources,
}

impl NodeDescriptor<'static> {
    pub fn new(task: TaskId) -> Self {
        NodeDescriptor {
            label: None,
            task,
            after: None,
            resources: Resources {
                attachments: SmallVec::new(),
                msaa: false,
            },
            dependencies: &[],
        }
    }
}

impl<'l> NodeDescriptor<'l> {
    pub fn dependencies<'a>(self, dependencies: &'a[Dependency]) -> NodeDescriptor<'a> {
        NodeDescriptor {
            label: self.label,
            task: self.task,
            resources: self.resources,
            after: self.after,
            dependencies,
        }
    }

    pub fn after(mut self, previous: NodeId) -> Self {
        self.after = Some(previous);

        self
    }

    pub fn task(mut self, task: TaskId) -> Self {
        self.task = task;

        self
    }
    
    pub fn resource(mut self, kind: ResourceKind, allocation: Allocation) -> Self {
        self.resources.attachments.push(Resource { kind, allocation });

        self
    }

    pub fn msaa(mut self, enable: bool) -> Self {
        self.resources.msaa = enable;

        self
    }

    pub fn label(mut self, label: &'static str) -> Self {
        self.label = Some(label);

        self
    }
}

struct Node {
    dependencies: SmallVec<[Dependency; 4]>,
    resources: Resources,
    after: Option<NodeId>,
}

pub struct RenderGraph {
    nodes: Vec<Node>,
    tasks: Vec<TaskId>,
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
            dependencies: SmallVec::from_slice(&desc.dependencies),
            after: desc.after,
            resources: desc.resources.clone(),
        });

        id
    }

    pub fn add_dependency(&mut self, node: NodeId, dep: Dependency) {
        self.nodes[node.index()].dependencies.push(dep);
    }

    pub fn add_root(&mut self, root: NodeId) {
        self.roots.push(root);
    }

    pub fn topological_sort(&self, sorted: &mut Vec<NodeId>) -> Result<(), GraphError> {
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
                if let Some(prev) = node.after {
                    if !added[prev.index()] {
                        if cycle_check[prev.index()] {
                            return Err(GraphError::DependencyCycle(prev));
                        }
        
                        stack.push(prev);
                        continue 'traversal;
                    }
                }
                for dep in &node.dependencies {
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



#[test]
fn test_nested() {
    let mut graph = RenderGraph::new();

    let main = NodeDescriptor::new(TaskId(0))
        .label("main")
        .resource(ResourceKind::ColorTexture, Allocation::Fixed(0));
    let atlas1 = NodeDescriptor::new(TaskId(0))
        .label("atlas-1")
        .resource(ResourceKind::ColorTexture, Allocation::Fixed(1));
    let atlas2 = NodeDescriptor::new(TaskId(0))
        .label("atlas-1")
        .resource(ResourceKind::ColorTexture, Allocation::Fixed(2));

    //  tmp0            n8  n5
    //                   \   \
    //  tmp1       n4     n6--n7
    //              \          \
    //  tmp0  n1     n3         n9
    //          \     \          \
    //  main     n0----n2---------n10
    let n0 = graph.add_node(&main.clone().task(TaskId(0)));
    let n1 = graph.add_node(&atlas1.clone().task(TaskId(1)));
    let n2 = graph.add_node(&main.clone().task(TaskId(2)).after(n0));
    let n3 = graph.add_node(&atlas1.clone().task(TaskId(3)));
    let n4 = graph.add_node(&atlas2.clone().task(TaskId(4)));
    let n5 = graph.add_node(&atlas1.clone().task(TaskId(5)));
    let n6 = graph.add_node(&atlas2.clone().task(TaskId(6)));
    let n7 = graph.add_node(&atlas2.clone().task(TaskId(7)).after(n6));
    let n8 = graph.add_node(&atlas1.clone().task(TaskId(8)));
    let n9 = graph.add_node(&atlas1.clone().task(TaskId(9)));
    let n10 = graph.add_node(&main.clone().task(TaskId(10)).after(n2));

    graph.add_dependency(n0, n1.output(0));
    graph.add_dependency(n2, n3.output(0));
    graph.add_dependency(n3, n4.output(0));
    graph.add_dependency(n7, n8.output(0));
    graph.add_dependency(n6, n5.output(0));
    graph.add_dependency(n9, n6.output(0));
    graph.add_dependency(n10, n9.output(0));

    graph.add_root(n10);

    let mut sorted = Vec::new();
    graph.topological_sort(&mut sorted).unwrap();
    println!("sorted: {:?}", sorted);

    assert_eq!(sorted.len(), graph.nodes.len());
}
