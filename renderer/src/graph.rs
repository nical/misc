// TODO:
//
// Output allocated rectangles one way or another.
//
// Should add the option to sort tasks by area or by height,width before allocating rects
// to improve the packing efficiency.

use std::i32;
use smallvec::SmallVec;

pub use euclid::{size2, vec2, point2};

use guillotiere::AllocId;

use crate::image_store::ImageFormat;
use crate::atlas::{GuillotineAllocator, TextureId};
use crate::texture_update::*;
use crate::types::units::*;

#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(pub(crate) u32);

pub type TextureIndex = usize;

impl NodeId {
    pub fn index(self) -> usize { self.0 as usize }
}

pub(crate) fn node_id(idx: usize) -> NodeId {
    debug_assert!(idx < std::u32::MAX as usize);
    NodeId(idx as u32)
}

#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct NodeIdRange {
    start: u32,
    end: u32,
}

impl Iterator for NodeIdRange {
    type Item = NodeId;

    #[inline]
    fn next(&mut self) -> Option<NodeId> {
        if self.start >= self.end {
            return None;
        }

        let result = Some(NodeId(self.start));
        self.start += 1;

        result
    }
}

impl NodeIdRange {
    #[inline]
    pub fn len(&self) -> usize {
        (self.end - self.start) as usize
    }

    #[inline]
    pub fn get(&self, nth: usize) -> NodeId {
        assert!(nth < self.len());
        NodeId(self.start + nth as u32)
    }
}

#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
#[derive(Clone)]
pub struct Node {
    pub task_id: TaskId,
    pub dependencies: SmallVec<[NodeId; 2]>,
    pub format: ImageFormat,
    // These could be in a separate array.
    pub size: DeviceIntSize,
    pub alloc_kind: AllocKind,
}

#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Task {
    pub node_id: NodeId,
    pub task_id: TaskId,
}

#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum AllocKind {
    Fixed(TextureId, DeviceIntPoint),
    Dynamic,
}

#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TaskId(u16, u32);

struct Surface {
    tasks: Vec<Task>,
    atlas: GuillotineAllocator,
    format: ImageFormat,
    texture_id: TextureId,
}

struct FixedSurface {
    tasks: Vec<Task>,
    format: ImageFormat,
    texture_id: TextureId,
}

struct Pass {
    tasks: Vec<Task>,
}

pub struct RenderPass {
    pub tasks: Vec<Task>,
    pub texture_id: TextureId,
    pub format: ImageFormat,
}

#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
#[derive(Clone)]
pub struct Graph {
    pub(crate) nodes: Vec<Node>,
    pub(crate) roots: Vec<NodeId>,
}

impl Graph {
    pub fn with_capacity(nodes: usize, roots: usize) -> Self {
        Graph {
            nodes: Vec::with_capacity(nodes),
            roots: Vec::with_capacity(roots),
        }
    }

    pub fn new() -> Self {
        Graph::with_capacity(0, 0)
    }

    pub fn add_node(
        &mut self,
        task_id: TaskId,
        format: ImageFormat,
        size: DeviceIntSize,
        alloc_kind: AllocKind,
        deps: &[NodeId],
    ) -> NodeId {
        let id = node_id(self.nodes.len());
        self.nodes.push(Node {
            task_id,
            size,
            alloc_kind,
            dependencies: SmallVec::from_slice(deps),
            format,
        });

        id
    }

    pub fn add_dependency(&mut self, node: NodeId, dep: NodeId) {
        self.nodes[node.index()].dependencies.push(dep);
    }

    pub fn set_alloc_kind(&mut self, node: NodeId, alloc_kind: AllocKind) {
        self.nodes[node.index()].alloc_kind = alloc_kind;
    }

    pub fn add_root(&mut self, id: NodeId) {
        self.roots.push(id);
    }

    pub fn roots(&self) -> &[NodeId] {
        &self.roots
    }

    pub fn node_ids(&self) -> NodeIdRange {
        NodeIdRange {
            start: 0,
            end: self.nodes.len() as u32,
        }
    }

    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn node_dependencies(&self, node: NodeId) -> &[NodeId] {
        &self.nodes[node.index()].dependencies
    }

}

fn init_vec<T: Clone>(vec: &mut Vec<T>, val: T, n: usize) {
    vec.clear();
    vec.reserve(n);
    for _ in 0..n {
        vec.push(val.clone());
    }
}

pub struct GraphBuilder {
    // Settings
    allocator_options: guillotiere::AllocatorOptions,
    surface_size: DeviceIntSize,
    // Recycled temporary data.
    passes: Vec<Pass>,
    visited: Vec<bool>,
    node_rev_passes: Vec<i32>,
    surfaces: Vec<Surface>,
    fixed_surfaces: Vec<FixedSurface>,
    node_textures: Vec<TextureId>,
    alloc_ids: Vec<SmallVec<[(usize, AllocId); 1]>>,
    pass_last_node_ranges: Vec<std::ops::Range<usize>>,
    last_node_refs: Vec<NodeId>,
}


impl GraphBuilder {
    pub fn new() -> Self {
        GraphBuilder {
            passes: Vec::new(),
            visited: Vec::new(),
            node_rev_passes: Vec::new(),
            surfaces: Vec::new(),
            fixed_surfaces: Vec::new(),
            node_textures: Vec::new(),
            alloc_ids: Vec::new(),
            pass_last_node_ranges: Vec::new(),
            last_node_refs: Vec::new(),

            allocator_options: guillotiere::DEFAULT_OPTIONS,
            surface_size: size2(1024, 1024),
        }
    }

    pub fn set_allocator_options(&mut self, size: DeviceIntSize, options: guillotiere::AllocatorOptions) {
        self.surface_size = size;
        self.allocator_options = options;
    }

    pub fn build(&mut self, graph: &Graph, render_passes: &mut Vec<RenderPass>, texture_updates: &mut TextureUpdateState) {

        // Step 1 - Assign passes to render nodes.

        self.passes.clear();

        // Initialize the array with negative values. Once the recursive passes are done, any negative
        // value left corresponds to nodes that haven't been traversed, which means they are not
        // contributing to the output of the graph. They won't be assigned to any pass.
        init_vec(&mut self.node_rev_passes, -1, graph.nodes.len());
        let mut max_depth = 0;

        for &root in &graph.roots {
            assign_depths(
                &graph,
                root,
                0,
                &mut self.node_rev_passes,
                &mut max_depth,
            );
        }

        self.passes.reserve(max_depth as usize + 1);
        for _ in 0..(max_depth + 1) {
            self.passes.push(Pass {
                tasks: Vec::new(),
            });
        }

        for node_id in graph.node_ids() {
            let node_idx = node_id.index();
            if self.node_rev_passes[node_idx] < 0 {
                // This node does not contribute to the output of the graph.
                continue;
            }

            let pass_index = (max_depth - self.node_rev_passes[node_idx]) as usize;
            let task_id = graph.nodes[node_idx].task_id;

            self.passes[pass_index].tasks.push(Task { node_id, task_id });
        }

        // Step 2 - Find when a node can be deallocated.

        self.last_node_refs.clear();
        self.last_node_refs.reserve(graph.nodes.len());
        init_vec(&mut self.visited, false, graph.nodes.len());
        init_vec(&mut self.pass_last_node_ranges, 0..0, self.passes.len());

        // The first step is to find for each pass the list of nodes that are not referenced
        // anymore after the pass ends.

        // Mark roots as visited to avoid deallocating their target rects.
        for root in &graph.roots {
            self.visited[root.index()] = true;
        }

        // Visit passes in reverse order and look at the dependencies.
        // Each dependency that we haven't visited yet is the last reference to a node.
        // TODO: this only works if passes can only refer to pass indices prior to theirs.
        let mut pass_index = self.passes.len();
        for pass in self.passes.iter().rev() {
            pass_index -= 1;
            let first = self.last_node_refs.len();
            for task in &pass.tasks {
                for &dep in graph.node_dependencies(task.node_id) {
                    let dep_idx = dep.index();
                    if !self.visited[dep_idx] {
                        self.visited[dep_idx] = true;
                        self.last_node_refs.push(dep);
                    }
                }
            }
            self.pass_last_node_ranges[pass_index] = first..self.last_node_refs.len();
        }

        // Step 3 - Assign nodes to actual render targets.

        self.surfaces.clear();
        for surface in &mut self.surfaces {
            surface.tasks.clear();
        }
        self.fixed_surfaces.clear();

        init_vec(&mut self.node_textures, TextureId(0), graph.nodes.len());
        init_vec(&mut self.alloc_ids, SmallVec::new(), graph.nodes.len());

        for (pass_index, pass) in self.passes.iter().enumerate() {
            'task_loop: for task in &pass.tasks {
                let node = &graph.nodes[task.node_id.index()];

                if let AllocKind::Fixed(texture_id, _) = node.alloc_kind {
                    for surface in &mut self.fixed_surfaces {
                        if surface.texture_id != texture_id {
                            continue;
                        }

                        surface.tasks.push(*task);

                        continue 'task_loop;
                    }

                    self.fixed_surfaces.push(FixedSurface {
                        tasks: vec![*task],
                        format: node.format,
                        texture_id,
                    });

                    continue 'task_loop;
                }

                let mut surface_index = None;
                'sub_pass_loop: for (i, surface) in self.surfaces.iter_mut().enumerate() {
                    if surface.format != node.format {
                        continue 'sub_pass_loop;
                    }

                    // Check that we aren't assigning to a target that the node reads from.
                    for &dep in &node.dependencies {
                        if self.node_textures[dep.index()] == surface.texture_id {
                            continue 'sub_pass_loop;
                        }
                    }

                    if let Some(alloc) = surface.atlas.allocate(node.size.cast_unit()) {
                        self.alloc_ids[task.node_id.index()].push((i, alloc.id));
                        surface_index = Some(i);
                        break 'sub_pass_loop;
                    }
                }

                let surface_size = self.surface_size;
                let allocator_options = &self.allocator_options;
                let surfaces = &mut self.surfaces;
                let alloc_ids = &mut self.alloc_ids;
                let surface_index = surface_index.unwrap_or_else(||{
                    // Didn't find an adequate target, add a new one.

                    let texture_id = texture_updates.add_texture(node.size, node.format);
                    let mut atlas = GuillotineAllocator::with_options(surface_size.cast_unit(), allocator_options);
                    let alloc = atlas.allocate(node.size.cast_unit()).unwrap();
                    let i = surfaces.len();
                    alloc_ids[task.node_id.index()].push((i, alloc.id));

                    surfaces.push(Surface {
                        tasks: Vec::new(),
                        format: node.format,
                        texture_id,
                        atlas,
                    });

                    surfaces.len() - 1
                });

                let surface = &mut self.surfaces[surface_index];
                surface.tasks.push(*task);
                self.node_textures[task.node_id.index()] = surface.texture_id;
            }
        
            for surface in &mut self.fixed_surfaces {
                if !surface.tasks.is_empty() {
                    render_passes.push(RenderPass {
                        tasks: std::mem::take(&mut surface.tasks),
                        texture_id: surface.texture_id,
                        format: surface.format,
                    });
                }
            }

            for surface in &mut self.surfaces {
                if !surface.tasks.is_empty() {
                    render_passes.push(RenderPass {
                        tasks: std::mem::take(&mut surface.tasks),
                        texture_id: surface.texture_id,
                        format: surface.format,
                    });
                }
            }

            // Deallocations we can perform after this pass.
            let finished_range = self.pass_last_node_ranges[pass_index].clone();
            for finished_node in &self.last_node_refs[finished_range] {
                let node_idx = finished_node.index();
                for &(surface_index, alloc_id) in &self.alloc_ids[node_idx] {
                    self.surfaces[surface_index].atlas.deallocate(alloc_id);
                }
            }
        }
    }

    pub fn clear_textures(&mut self, texture_updates: &mut TextureUpdateState) {
        for surface in &self.surfaces {
            texture_updates.delete_texture(surface.texture_id);
        }

        self.surfaces.clear();
    }
}

impl Drop for GraphBuilder {
    fn drop(&mut self) {
        if !self.surfaces.is_empty() {
            panic!("Dropping a GraphBuilder without clearing its textures. {:?} texture leaked", self.surfaces.len());
        }
    }
}

fn assign_depths(
    graph: &Graph,
    node_id: NodeId,
    rev_pass_index: i32,
    node_rev_passes: &mut [i32],
    max_depth: &mut i32,
) {
    *max_depth = std::cmp::max(*max_depth, rev_pass_index);


    let node_depth = &mut node_rev_passes[node_id.index()];
    if *node_depth >= rev_pass_index {
        return;
    }

    *node_depth = rev_pass_index;

    for &dep in &graph.nodes[node_id.index()].dependencies {
        assign_depths(
            graph,
            dep,
            rev_pass_index + 1,
            node_rev_passes,
            max_depth,
        );
    }
}

pub fn print_render_passes(render_passes: &[RenderPass]) {
    for pass in render_passes {
        println!("# render pass {:?} {:?}", pass.texture_id, pass.format);
        for task in &pass.tasks {
            println!("  * {:?}", task);
        }
    }
}

pub fn dump_as_svg(
    graph: &Graph,
    passes: &[RenderPass],
    output: &mut dyn std::io::Write,
) -> Result<(), std::io::Error> {
    use svg_fmt::*;

    // TODO: show texture sizes and formats,
    // show an atlas per pass.

    let node_width = 80.0;
    let node_height = 30.0;
    let vertical_spacing = 8.0;
    let horizontal_spacing = 20.0;
    let margin = 10.0;
    let text_size = 10.0;

    let mut pass_rects = Vec::new();
    let mut nodes = vec![None; graph.nodes.len()];

    let mut x = margin;
    let mut max_y: f32 = 0.0;

    #[derive(Clone)]
    struct Node {
        rect: Rectangle,
        label: Text,
        size: Text,
    }

    for pass in passes {
        let mut layout = VerticalLayout::new(x, margin, node_width);

        for task in &pass.tasks {
            let node_index = task.node_id.index();
            let node = &graph.nodes[node_index];

            let rect = layout.push_rectangle(node_height);

            let tx = rect.x + rect.w / 2.0;
            let ty = rect.y + 10.0;

            let label = text(tx, ty, format!("{:?}", node.task_id));
            let size = text(tx, ty + 12.0, format!("{:?}", node.size));

            nodes[node_index] = Some(Node { rect, label, size });

            layout.advance(vertical_spacing);
        }

        pass_rects.push(layout.total_rectangle());

        x += node_width + horizontal_spacing;
        max_y = max_y.max(layout.y + margin);
    }

    let mut links = Vec::new();
    for node_index in 0..nodes.len() {
        if nodes[node_index].is_none() {
            continue;
        }

        let node = &graph.nodes[node_index];
        for dep in &node.dependencies {
            let dep_index = dep.index();

            if let (&Some(ref node), &Some(ref dep_node)) = (&nodes[node_index], &nodes[dep_index]) {
                links.push((
                    dep_node.rect.x + dep_node.rect.w,
                    dep_node.rect.y + dep_node.rect.h / 2.0,
                    node.rect.x,
                    node.rect.y + node.rect.h / 2.0,
                ));
            }
        }
    }

    let svg_w = x + margin;
    let svg_h = max_y + margin;
    writeln!(output, "{}", BeginSvg { w: svg_w, h: svg_h })?;

    // Background.
    writeln!(output,
        "    {}",
        rectangle(0.0, 0.0, svg_w, svg_h)
            .inflate(1.0, 1.0)
            .fill(rgb(50, 50, 50))
    )?;

    // Passes.
    for rect in pass_rects {
        writeln!(output,
            "    {}",
            rect.inflate(3.0, 3.0)
                .border_radius(4.0)
                .opacity(0.4)
                .fill(black())
        )?;
    }

    // Links.
    for (x1, y1, x2, y2) in links {
        dump_dependency_link(output, x1, y1, x2, y2);
    }

    // Tasks.
    for node in &nodes {
        if let Some(node) = node {
            writeln!(output,
                "    {}",
                node.rect
                    .clone()
                    .fill(black())
                    .border_radius(3.0)
                    .opacity(0.5)
                    .offset(0.0, 2.0)
            )?;
            writeln!(output,
                "    {}",
                node.rect
                    .clone()
                    .fill(rgb(200, 200, 200))
                    .border_radius(3.0)
                    .opacity(0.8)
            )?;

            writeln!(output,
                "    {}",
                node.label
                    .clone()
                    .size(text_size)
                    .align(Align::Center)
                    .color(rgb(50, 50, 50))
            )?;
            writeln!(output,
                "    {}",
                node.size
                    .clone()
                    .size(text_size * 0.7)
                    .align(Align::Center)
                    .color(rgb(50, 50, 50))
            )?;
        }
    }

    writeln!(output, "{}", EndSvg)
}

#[allow(dead_code)]
fn dump_dependency_link(
    output: &mut dyn std::io::Write,
    x1: f32, y1: f32,
    x2: f32, y2: f32,
) {
    use svg_fmt::*;

    // If the link is a straight horizontal line and spans over multiple passes, it
    // is likely to go straight though unrelated nodes in a way that makes it look like
    // they are connected, so we bend the line upward a bit to avoid that.
    let simple_path = (y1 - y2).abs() > 1.0 || (x2 - x1) < 45.0;

    let mid_x = (x1 + x2) / 2.0;
    if simple_path {
        write!(output, "    {}",
            path().move_to(x1, y1)
                .cubic_bezier_to(mid_x, y1, mid_x, y2, x2, y2)
                .fill(Fill::None)
                .stroke(Stroke::Color(rgb(100, 100, 100), 3.0))
        ).unwrap();
    } else {
        let ctrl1_x = (mid_x + x1) / 2.0;
        let ctrl2_x = (mid_x + x2) / 2.0;
        let ctrl_y = y1 - 25.0;
        write!(output, "    {}",
            path().move_to(x1, y1)
                .cubic_bezier_to(ctrl1_x, y1, ctrl1_x, ctrl_y, mid_x, ctrl_y)
                .cubic_bezier_to(ctrl2_x, ctrl_y, ctrl2_x, y2, x2, y2)
                .fill(Fill::None)
                .stroke(Stroke::Color(rgb(100, 100, 100), 3.0))
        ).unwrap();
    }
}


#[cfg(test)]
fn check_order(before: TaskId, after: TaskId, passes: &[RenderPass]) {
    let mut found_first = None;
    let mut found_second = None;
    for (idx, pass) in passes.iter().enumerate() {
        for task in &pass.tasks {
            if task.task_id == before {
                assert!(found_first.is_none(), "{:?} found twice", before);
                found_first = Some(idx);
            }
            if task.task_id == after {
                assert!(found_first.map(|i| i < idx).unwrap_or(false), "{:?} should have been before {:?}", before, after);
                assert!(found_second.is_none(), "{:?} found twice", after);
                found_second = Some(idx);
            }
        }
    }

    if found_first.is_none() {
        panic!("Missing task {:?}", before);
    }
    if found_second.is_none() {
        panic!("Missing task {:?}", after);
    }
}

#[test]
fn simple_graph() {
    let mut graph = Graph::new();

    let n0 = graph.add_node(TaskId(0, 0), ImageFormat::Alpha8, size2(100, 100), AllocKind::Dynamic, &[]);
    let _ = graph.add_node(TaskId(0, 1), ImageFormat::Rgba8, size2(100, 100), AllocKind::Dynamic, &[n0]);
    let n2 = graph.add_node(TaskId(0, 2), ImageFormat::Rgba8, size2(100, 100), AllocKind::Dynamic, &[]);
    let n3 = graph.add_node(TaskId(0, 3), ImageFormat::Alpha8, size2(100, 100), AllocKind::Dynamic, &[]);
    let n4 = graph.add_node(TaskId(0, 4), ImageFormat::Alpha8, size2(100, 100), AllocKind::Dynamic, &[n2, n3]);
    let n5 = graph.add_node(TaskId(0, 5), ImageFormat::Rgba8, size2(100, 100), AllocKind::Dynamic, &[]);
    let n6 = graph.add_node(TaskId(0, 6), ImageFormat::Rgba8, size2(100, 100), AllocKind::Dynamic, &[n3, n5]);
    let n7 = graph.add_node(TaskId(0, 7), ImageFormat::Rgba8, size2(100, 100), AllocKind::Dynamic, &[n2, n4, n6]);
    let n8 = graph.add_node(TaskId(0, 8), ImageFormat::Rgba8, size2(800, 600), AllocKind::Fixed(TextureId(100), point2(0, 0)), &[n7]);

    graph.add_root(n5);
    graph.add_root(n8);

    let mut builder = GraphBuilder::new();

    let mut texture_updates = TextureUpdateState::new();
    let mut render_passes = Vec::new();

    builder.build(&mut graph, &mut render_passes, &mut texture_updates);

    // print_render_passes(&render_passes);

    // let mut svg_file = std::fs::File::create("test_simple_graph.svg").unwrap();
    // dump_as_svg(&graph, &render_passes, &mut svg_file).unwrap();

    check_order(TaskId(0, 7), TaskId(0, 8), &render_passes);
    check_order(TaskId(0, 2), TaskId(0, 7), &render_passes);
    check_order(TaskId(0, 4), TaskId(0, 7), &render_passes);
    check_order(TaskId(0, 6), TaskId(0, 7), &render_passes);
    check_order(TaskId(0, 3), TaskId(0, 4), &render_passes);
    check_order(TaskId(0, 2), TaskId(0, 4), &render_passes);

    builder.clear_textures(&mut texture_updates);
}
