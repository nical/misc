//! A prototype implementation of some of the logic for clipping and rendering SVG filter
//! graphs correctly.
//!
//! This does not compute the primitive subregions (they are provided as input), it would be
//! more elegant to do it here, so that the graph can be shared, but it's probably going to
//! be easier to integrate in gecko with a graph per filter instance without sharing the
//! description, at least at first.

use euclid::default::*;
use euclid::{point2, size2};
use std::collections::HashMap;

const MAX_INPUTS: usize = 2;

pub type DeviceRect = Box2D<f32>;
pub type LayoutRect = Box2D<f32>;
pub type SubRect = Box2D<f32>;
pub type DeviceIntSize = Size2D<i32>;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct FilterPrimitiveId(u16);

impl FilterPrimitiveId {
    fn index(&self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum FilterKind {
    SourceGraphic(u32),
    Flood(f32, f32, f32, f32),
    Opacity(f32),
    ColorMatrix(Box<[f32; 20]>),
    Blur(Vector2D<f32>),
    Offset(Vector2D<f32>),
    MixBlend(MixBlendMode),
    Composite,
}

impl FilterKind {
    pub fn name(&self) -> &'static str {
         match self {
            FilterKind::SourceGraphic(..) => "SourceGraphic",
            FilterKind::Flood(..) => "Flood",
            FilterKind::Opacity(..) => "Opacity",
            FilterKind::ColorMatrix(..) => "ColorMatrix",
            FilterKind::Offset(..) => "Offset",
            FilterKind::Blur(..) => "Blur",
            FilterKind::MixBlend(..) => "MixBlend",
            FilterKind::Composite => "Composite",
        }
    }

    pub fn preserves_opacity(&self) -> bool {
        match self {
            FilterKind::SourceGraphic(..) => true,
            FilterKind::Flood(_, _, _, a) => *a == 1.0,
            FilterKind::Opacity(a) => *a == 1.0,
            FilterKind::ColorMatrix(mat) => {
                mat[3] == 0.0 && mat[8] == 0.0 && mat[13] == 0.0 && mat[18] == 0.0 &&
                mat[4] >= 0.0 && mat[9] >= 0.0 && mat[14] >= 0.0 && mat[19] >= 0.0                
            },
            FilterKind::Offset(..) => true,
            FilterKind::Blur(..) => false,
            FilterKind::MixBlend(..) => true,
            FilterKind::Composite => true,
        }
    }

    pub fn num_inputs(&self) -> usize {
         match self {
            FilterKind::SourceGraphic(..) => 0,
            FilterKind::Flood(..) => 0,
            FilterKind::Opacity(..) => 1,
            FilterKind::ColorMatrix(..) => 1,
            FilterKind::Offset(..) => 1,
            FilterKind::Blur(..) => 1,
            FilterKind::MixBlend(..) => 2,
            FilterKind::Composite => 2,
        }        
    }
}

fn supports_input_conversion(kind: &FilterKind, from: ColorSpace, to: ColorSpace) -> bool {
    use ColorSpace::{Srgb, Linear};

    match (kind, from, to) {
        (FilterKind::Opacity(..), Srgb, Linear) => true,
        (FilterKind::ColorMatrix(..), Srgb, Linear) => true,
        (FilterKind::Blur(..), Srgb, Linear) => true,
        (FilterKind::Offset(..), Srgb, Linear) => true,

        _ => false,
    }
}

fn supports_output_conversion(kind: &FilterKind, from: ColorSpace, to: ColorSpace) -> bool {
    use ColorSpace::{Srgb, Linear};

    match (kind, from, to) {
        (FilterKind::MixBlend(..), Linear, Srgb) => true,
        (FilterKind::Flood(..), Linear, Srgb) => true,
        (FilterKind::Opacity(..), Linear, Srgb) => true,
        (FilterKind::ColorMatrix(..), Linear, Srgb) => true,
        (FilterKind::Blur(..), Linear, Srgb) => true,
        (FilterKind::Offset(..), Linear, Srgb) => true,

        _ => false,
    }
}


#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum MixBlendMode {
    Multiply,
    Difference,
    Color,
    ColorBurn,
    ColorDodge,
    Darken,
    Exclusion,
    HardLight,
    Hue,
    Lighten,
    Luminosity,
    Overlay,
    Screen,
    SoftLight,
}


#[derive(Clone)]
pub struct FilterPrimitive {
    kind: FilterKind,
    inputs: [Option<FilterPrimitiveId>; MAX_INPUTS],
    color_space: ColorSpace,
    res: Option<DeviceIntSize>,
    // Primitive subregion
    // TODO: should be able to express something relative like -10%, but it looks like
    // the gecko code already handles that.
    primitive_subregion: LayoutRect,
    // Keep track of what spaces we want the result in.
    output_srgb: bool,
    output_linear: bool,
    // How many primitives read this one. TODO: not needed anymore?
    ref_count: u16,
}

impl FilterPrimitive {
    pub fn with_color_space(mut self, space: ColorSpace) -> Self {
        self.color_space = space;

        self
    }

    pub fn with_resolution(mut self, res: DeviceIntSize) -> Self {
        self.res = Some(res);

        self
    }

    pub fn with_primitive_subregion(mut self, bounds: LayoutRect) -> Self {
        self.primitive_subregion = bounds;

        self
    }

    pub fn output_srgb(mut self) -> Self {
        self.output_srgb = true;

        self
    }

    pub fn output_linear(mut self) -> Self {
        self.output_linear = true;

        self
    }

    pub fn source_graphic(id: u32) -> Self {
        let bounds = Box2D {
            min: point2(std::f32::MIN, std::f32::MIN),
            max: point2(std::f32::MAX, std::f32::MAX),
        };
        FilterPrimitive {
            inputs: [None; MAX_INPUTS],
            color_space: ColorSpace::Srgb,
            kind: FilterKind::SourceGraphic(id),
            res: None,
            primitive_subregion: bounds,
            output_srgb: false,
            output_linear: false,
            ref_count: 0,
        }
    }

    pub fn flood(r: f32, g: f32, b: f32, a: f32) -> Self {
        let bounds = Box2D {
            min: point2(std::f32::MIN, std::f32::MIN),
            max: point2(std::f32::MAX, std::f32::MAX),
        };
        FilterPrimitive {
            inputs: [None; MAX_INPUTS],
            color_space: ColorSpace::Linear,
            kind: FilterKind::Flood(r, g, b, a),
            res: None,
            primitive_subregion: bounds,
            output_srgb: false,
            output_linear: false,
            ref_count: 0,
        }
    }

    pub fn opacity(alpha: f32, src: FilterPrimitiveId) -> Self {
        let bounds = Box2D {
            min: point2(std::f32::MIN, std::f32::MIN),
            max: point2(std::f32::MAX, std::f32::MAX),
        };
        FilterPrimitive {
            inputs: [Some(src), None],
            color_space: ColorSpace::Linear,
            kind: FilterKind::Opacity(alpha),
            res: None,
            primitive_subregion: bounds,
            output_srgb: false,
            output_linear: false,
            ref_count: 0,
        }
    }

    pub fn mix_blend(mode: MixBlendMode, src: [FilterPrimitiveId; MAX_INPUTS]) -> Self {
        let bounds = Box2D {
            min: point2(std::f32::MIN, std::f32::MIN),
            max: point2(std::f32::MAX, std::f32::MAX),
        };
        FilterPrimitive {
            inputs: [Some(src[0]), Some(src[1])],
            color_space: ColorSpace::Linear,
            kind: FilterKind::MixBlend(mode),
            res: None,
            primitive_subregion: bounds,
            output_srgb: false,
            output_linear: false,
            ref_count: 0,
        }        
    }

    pub fn offset(offset: Vector2D<f32>, src: FilterPrimitiveId) -> Self {
        let bounds = Box2D {
            min: point2(std::f32::MIN, std::f32::MIN),
            max: point2(std::f32::MAX, std::f32::MAX),
        };
        FilterPrimitive {
            inputs: [Some(src), None],
            color_space: ColorSpace::Linear,
            kind: FilterKind::Offset(offset),
            res: None,
            primitive_subregion: bounds,
            output_srgb: false,
            output_linear: false,
            ref_count: 0,
        }        
    }

    pub fn gaussian_blur(radius: Vector2D<f32>, src: FilterPrimitiveId) -> Self {
        let bounds = Box2D {
            min: point2(std::f32::MIN, std::f32::MIN),
            max: point2(std::f32::MAX, std::f32::MAX),
        };

        FilterPrimitive {
            inputs: [Some(src), None],
            color_space: ColorSpace::Linear,
            kind: FilterKind::Blur(radius),
            res: None,
            primitive_subregion: bounds,
            output_srgb: false,
            output_linear: false,
            ref_count: 0,
        }        
    }
}


#[derive(Copy, Clone, Debug)]
struct NodeColorSpaces {
    input: ColorSpace,
    output: ColorSpace,
}

// TODO:
//
// Do the layout during DL building, scene building or frame building ?
// the advantage of doing it during frame building is potentially tighter
// output clip and animate more things.
//

#[derive(Clone)]
pub struct FilterInstance {
    /// Region in which the node has content it can render.
    content_rects: Vec<LayoutRect>,
    /// Region that will be sampled by child primitives.
    requested_rects: Vec<LayoutRect>,
    /// Region that the node samples from its dependencies.
    sample_rects: Vec<LayoutRect>,

    /// Padding if need be.
    padding: Vec<SideOffsets2D<f32>>,

    /// device space
    node_resolution: Vec<DeviceIntSize>,
    // 0..1 range
    relative_input_regions: Vec<[SubRect; MAX_INPUTS]>,
    node_color_spaces: Vec<NodeColorSpaces>,
}

#[derive(Clone)]
pub struct FilterGraph {
    primitives: Vec<FilterPrimitive>,
    // TODO: we don't really need to store the root, it is always the last node.
    root: FilterPrimitiveId,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ColorSpace {
    Linear,
    Srgb,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct GraphInput {
    task_id: RenderTaskId,
    size: DeviceIntSize,
    is_opaque: bool,
}

impl FilterInstance {
    pub fn new() -> Self {
        FilterInstance {
            content_rects: Vec::new(),
            requested_rects: Vec::new(),
            sample_rects: Vec::new(),
            node_resolution: Vec::new(),
            relative_input_regions: Vec::new(),
            node_color_spaces: Vec::new(),
            padding: Vec::new(),
        }
    }

    // This would probably happen during scene building.
    pub fn compute_layout(
        &mut self,
        graph: &FilterGraph,
        // Layout rectangles of the input graphics if any. 
        src_rects: &[LayoutRect],
        // The filter region acts as a clip on the filter graph and all filter primitives.
        filter_region: &LayoutRect,
    ) {
        self.compute_content_rects(graph, src_rects, filter_region, filter_region);
        //println!("content_rects: {:?}", self.content_rects);

        self.compute_requested_rects(graph);
        //println!("requested_rects: {:?}", self.requested_rects);

        self.compute_relative_input_rects(graph);
        //println!("relative_input_rects: {:?}", self.relative_input_regions);

        self.compute_inline_color_conversions(graph);        
    }

    /// Compute the region in layout space for which each node has content.
    fn compute_content_rects(
        &mut self,
        graph: &FilterGraph,
        src_rects: &[LayoutRect],
        filter_region: &LayoutRect,
        output_clip: &LayoutRect,
    ) {
        self.content_rects.clear();
        self.content_rects.reserve(graph.primitives.len());
        for node in &graph.primitives {
            let empty_rect = Box2D::zero();

            let r1 = *node.inputs[0]
                .map(|id| &self.content_rects[id.index()])
                .unwrap_or(&empty_rect);

            let r2 = *node.inputs[1]
                .map(|id| &self.content_rects[id.index()])
                .unwrap_or(&empty_rect);

            let input_rect = match node.kind {
                FilterKind::Flood(_, _, _, a) if a == 0.0 => empty_rect,
                FilterKind::Opacity(a) if a == 0.0 => empty_rect,
                FilterKind::Blur(..) if r1.is_empty() => empty_rect,

                FilterKind::Opacity(..)
                | FilterKind::ColorMatrix(..)
                => r1,

                FilterKind::SourceGraphic(idx) => src_rects[idx as usize],
                FilterKind::Flood(..) => node.primitive_subregion,
                FilterKind::Offset(v) => r1.translate(v),
                FilterKind::Blur(radius) => r1.inflate(3.0 * radius.x, 3.0 * radius.y),
                FilterKind::MixBlend(..)
                | FilterKind::Composite => r1.union(&r2),
            };

            let mut rect = input_rect.intersection_unchecked(&node.primitive_subregion)
                .intersection_unchecked(filter_region);

            rect = rect.to_non_empty().unwrap_or(empty_rect);

            self.content_rects.push(rect);
        }

        let root_rect = &mut self.content_rects[graph.root.index()];
        *root_rect = root_rect.intersection_unchecked(output_clip)
            .to_non_empty()
            .unwrap_or_else(Box2D::zero);
    }

    /// Compute in layout space the region that is requested of each node.
    ///
    /// It can be smaller than the content rect which lets us avoid rendering
    /// pixels that we know won't be used by later pass.
    /// It can also be larger if the content rect is clipped, in which can we
    /// have to pad the rendered content with some transparent pixels.
    fn compute_requested_rects(&mut self, graph: &FilterGraph) {
        // Next, walk back to the leaves and restrict the requested rect of each node to the union
        // of the areas that are sampled by filters that depend on it.
        // Some filters can cause the requested rect to be larger than the content rects.
        // We will Fix that up in a third pass and turn that into padding.

        self.requested_rects.clear();
        self.sample_rects.clear();
        self.requested_rects.reserve(graph.primitives.len());
        self.sample_rects.reserve(graph.primitives.len());
        for _ in 0..graph.primitives.len() {
            self.requested_rects.push(Box2D::zero());
            self.sample_rects.push(Box2D::zero());
        }
        self.requested_rects[graph.root.index()] = self.content_rects[graph.root.index()];

        for (i, node) in graph.primitives.iter().enumerate().rev() {
            if self.requested_rects[i].is_empty() {
                continue;
            }

            let mut sample_rect = self.requested_rects[i].intersection_unchecked(&node.primitive_subregion);

            // Most filter write the same pixel that they read. In other word they request
            // what is requested of them. 
            // For those that don't, adjust the sample rect to account for extra offset/padding. 
            match node.kind {
                FilterKind::Offset(v) => {
                    sample_rect = sample_rect.translate(-v);
                }
                FilterKind::Blur(radius) => {
                    sample_rect = sample_rect.inflate(3.0 * radius.x, 3.0 * radius.y);
                }
                _ => {}
            }

            // This appears to be firefox's behavior. The standard seems to 
            // imply that it acts as a clip on the ouput of the node only, TODO: double check.
            let apply_prim_subregion_before_and_after = true;

            if apply_prim_subregion_before_and_after {
                sample_rect = sample_rect.intersection_unchecked(&node.primitive_subregion);
            }

            // Propagate the requested area to the input primitives.
            for input_idx in 0..MAX_INPUTS {
                if let Some(id) = node.inputs[input_idx] {
                    let input_request = &mut self.requested_rects[id.index()];
                    *input_request = input_request.union(&sample_rect);
                }
            }

            self.sample_rects[i] = sample_rect;
        }

        // Now limit the requested rect to the part that has some content to show, This will
        // be the area that the filter will render.
        // Requested pixels that didn't fit in the content area are recorded as padding that
        // will be rendered around the filter as transparent filters.

        self.padding.clear();
        self.padding.reserve(graph.primitives.len());
        for i in 0..graph.primitives.len() {
            let content_rect = &self.content_rects[i];
            let requested_rect = &mut self.requested_rects[i];

            self.padding.push(compute_padding(content_rect, requested_rect));
            *requested_rect = requested_rect.intersection(content_rect).unwrap_or_else(Box2D::zero);
        }
    }

    /// For each filter primitive, determine what normalized sub-rect of its inputs it is
    /// reading.
    fn compute_relative_input_rects(&mut self, graph: &FilterGraph) {
        self.relative_input_regions.clear();
        self.relative_input_regions.reserve(graph.primitives.len());
        for (i, node) in graph.primitives.iter().enumerate() {
            let empty_rect = Box2D::zero();
            let input_r1 = *node.inputs[0]
                .map(|id| &self.requested_rects[id.index()])
                .unwrap_or(&empty_rect);
            let input_r2 = *node.inputs[1]
                .map(|id| &self.requested_rects[id.index()])
                .unwrap_or(&empty_rect);

            let sample_rect = self.sample_rects[i];

            self.relative_input_regions.push([
                relative_rect(&input_r1, &sample_rect),
                relative_rect(&input_r2, &sample_rect),
            ]);
        }
    }

    /// Some shaders are able to do a color space conversion of the input and or output
    /// inline which allows us to avoid adding a render pass. In this pass we determine
    /// where this useful. The remaining color conversions will be added via extra render
    /// passes when building the render tasks.
    fn compute_inline_color_conversions(&mut self, graph: &FilterGraph) {
        self.node_color_spaces.clear();
        self.node_color_spaces.reserve(graph.primitives.len());

        for node in &graph.primitives {
            // Determine output space.
            //
            // If at least one node reads this one in the same space then set the output color
            // space to the node's internal color space (no inline conversion).
            //
            // Otherwise add an inline conversion if supported or default to outputting in the
            // color space of the node's computation which is always supported.

            let output_space = if node.color_space == ColorSpace::Srgb && node.output_srgb {
                ColorSpace::Srgb
            } else if node.color_space == ColorSpace::Linear && node.output_linear {
                ColorSpace::Linear
            } else if node.output_srgb && supports_output_conversion(&node.kind, ColorSpace::Linear, ColorSpace::Srgb) {
                ColorSpace::Srgb
            } else if node.output_linear && supports_output_conversion(&node.kind, ColorSpace::Srgb, ColorSpace::Linear) {
                ColorSpace::Linear
            } else {
                node.color_space
            };

            // Determine input color space.
            //
            // For simplicity we only try inline conversion when there is a single input image.
            //
            // Default to the node's internal color space.

            let mut input_space = node.color_space;

            if let [Some(input), None] = node.inputs {
                let input_node_color_space = graph.primitives[input.index()].color_space;
                let input_node = &mut self.node_color_spaces[input.index()];
                if input_node.output != node.color_space 
                    && supports_input_conversion(&node.kind, input_node_color_space, node.color_space)
                {
                    input_space = input_node.output;
                }
            }

            self.node_color_spaces.push(NodeColorSpaces {
                input: input_space,
                output: output_space,
            });
        }
    }


    // TODO: computing the resolution, opacity and generating render tasks can be fused in a single
    // loop because all 3 loops iterate in the same order and only read from previously iterated primitives.



    /// With the information computed in earlier passes we are now able to derive the
    /// ideal resolution for each node (if not specified by the user);
    fn compute_resolution(&mut self, graph: &FilterGraph, scale_factor: f32, graph_inputs: &[GraphInput]) {
        self.node_resolution.clear();
        self.node_resolution.reserve(graph.primitives.len());
        for (i, node) in graph.primitives.iter().enumerate() {
            if let Some(res) = node.res {
                self.node_resolution.push(res);
                continue;
            }

            fn derive_from_input(input_res: &Size2D<i32>, relative_size: Size2D<f32>) -> Size2D<i32> {
                let input_res = input_res.to_f32();
                let w = input_res.width * relative_size.width;
                let h = input_res.height * relative_size.height;
                size2(w, h).ceil().to_i32()
            }

            let layout_size = self.requested_rects[i].size();
            let default_res = (layout_size * scale_factor).round().to_i32();
            let res = match node.kind {
                FilterKind::SourceGraphic(idx) => graph_inputs[idx as usize].size,
                FilterKind::Offset(..) 
                | FilterKind::Blur(..) => default_res,
                _ => {
                    let input_res_1 = node.inputs[0].map(
                        |idx| {
                            let input_res = &self.node_resolution[idx.index()];
                            let relative_size = self.relative_input_regions[i][0].size();
                            derive_from_input(input_res, relative_size)
                        }
                    ).unwrap_or(default_res);

                    let input_res_2 = node.inputs[1].map(
                        |idx| {
                            let input_res = &self.node_resolution[idx.index()];
                            let relative_size = self.relative_input_regions[i][1].size();
                            derive_from_input(input_res, relative_size)
                        }
                    ).unwrap_or(input_res_1);

                    if input_res_1.width > input_res_2.width {
                        input_res_1
                    } else {
                        input_res_2
                    }
                }
            };

            self.node_resolution.push(res);
        }
    }

    /// Evaluate whether the output of the graph can be rendered opaque.
    ///
    /// Must be called after computing the layout.
    pub fn compute_opacity(&mut self, graph: &FilterGraph, src: &[GraphInput]) -> bool {
        let mut node_is_opaque = Vec::new();
        node_is_opaque.reserve(graph.primitives.len());

        for (i, node) in graph.primitives.iter().enumerate() {
            let preserves_opacity = match node.kind {
                FilterKind::SourceGraphic(idx) => src[idx as usize].is_opaque,
                FilterKind::Flood(_, _, _, a) => a == 1.0,
                FilterKind::Opacity(a) => a == 1.0,
                FilterKind::ColorMatrix(ref mat) => {
                    mat[3] == 0.0 && mat[8] == 0.0 && mat[13] == 0.0 && mat[18] == 0.0 &&
                    mat[4] >= 0.0 && mat[9] >= 0.0 && mat[14] >= 0.0 && mat[19] >= 0.0                
                },
                FilterKind::Offset(..) => true,
                FilterKind::Blur(..) => false,
                FilterKind::MixBlend(..) => true,
                FilterKind::Composite => true,
            };

            // Consider the node opaque if it preserves opacity and none of its inputs
            // are not opaque.
            // TODO: with feComposite we can consider the result opaque if either input is opaque.
            let mut opaque = preserves_opacity;
            for idx in 0..MAX_INPUTS {
                if !opaque {
                    break;
                }
                if let Some(input_id) = node.inputs[idx] {
                    opaque &= node_is_opaque[input_id.index()]
                        && self.requested_rects[input_id.index()].contains_box(&self.sample_rects[i]);
                }
            }

            node_is_opaque.push(opaque);
        }

        node_is_opaque[graph.root.index()]
    }

    /// With all of the information previously computed we can now apply the filter to
    /// the render task graph.
    /// This would be called during frame building
    ///
    /// Must be called after computing the layout.
    pub fn generate_render_tasks(
        &mut self,
        graph: &FilterGraph,
        scale_factor: f32,
        graph_inputs: &[GraphInput],
        render_graph: &mut RenderTaskGraph,
        output: &mut Vec<(RenderTaskId, LayoutRect)>,
    ) {
        self.compute_resolution(graph, scale_factor, graph_inputs);

        output.clear();
        let mut conversion_passes = HashMap::new();

        let mut task_ids = Vec::new();
        for (i, node) in graph.primitives.iter().enumerate() {

            let size = self.node_resolution[i];

            // Get input render tasks and deal with remaining color space mismatches

            let mut inputs = [(std::u32::MAX, Box2D::zero()); MAX_INPUTS];

            for idx in 0..MAX_INPUTS {
                if let Some(input_id) = node.inputs[idx] {
                    let src_space = self.node_color_spaces[input_id.index()].output;
                    let input_space = self.node_color_spaces[i].input;

                    let (input_render_task, sub_rect) = task_ids[input_id.index()];

                    if src_space == input_space {
                        // Hopefully the common case.
                        let sub_rect = combine_sub_rects(&sub_rect, &self.relative_input_regions[i][idx]);
                        inputs[idx] = (input_render_task, sub_rect);
                    } else {
                        // Color spaces don't match, we have to insert a conversion pass.
                        // TODO: for flood we could just insert a flood with the converted space.
                        let task_id = *conversion_passes.entry((input_id, input_space)).or_insert_with(|| {
                            let task = RenderTask {
                                kind: match src_space {
                                    ColorSpace::Srgb => "SrgbToLinear",
                                    ColorSpace::Linear => "LinearToSrgb",
                                },
                                input_color_space: src_space,
                                color_space: input_space,
                                output_color_space: input_space,
                                size: self.node_resolution[input_id.index()],
                                padding: None,
                                inputs: &[(input_render_task, sub_rect)]
                            };
                            render_graph.add_task(&task)
                        });
                        inputs[idx] = (task_id, Box2D { min: point2(0.0, 0.0), max: point2(1.0, 1.0) });
                    }
                }
            }

            // Translate the padding from layout space to device space.

            let layout_size = self.requested_rects[i].size();
            let mut padding = self.padding[i];

            if size.width != layout_size.round().width as i32 {
                let scale_x = size.width as f32 / layout_size.width;
                let scale_y = size.height as f32 / layout_size.height;
                padding.top *= scale_y;
                padding.right *= scale_x;
                padding.bottom *= scale_y;
                padding.left *= scale_x;
            }

            // Generate the render tasks.

            let is_root = i == graph.root.index();

            if is_root {
                let root_color_space = if node.output_srgb {
                    ColorSpace::Srgb
                } else if node.output_linear {
                    ColorSpace::Linear
                } else {
                    node.color_space
                };

                let input_space = self.node_color_spaces[i].input;

                // Optimization: Skip the composite filter in some cases when it is last,
                // we are better off compositing directly in the target picture by rendering
                // two images.
                if node.kind == FilterKind::Composite
                    && input_space == root_color_space
                    && node.color_space == root_color_space
                {
                    for idx in 0..node.kind.num_inputs() {
                        let input_id = node.inputs[idx].unwrap();
                        let rect = self.requested_rects[input_id.index()];
                        output.push((inputs[idx].0, rect));
                    }

                    return;
                }
            }

            let mut sub_rect = Box2D { min: point2(0.0, 0.0), max: point2(1.0, 1.0) };

            let render_task = match node.kind {
                FilterKind::SourceGraphic(idx) => graph_inputs[idx as usize].task_id,
                FilterKind::Offset(..) => {
                    // We don't need to render anything for offset filters, just remember which
                    // portion of it's input is visible to the next filter in case some clipping
                    // is applied.
                    sub_rect = inputs[0].1;
                    inputs[0].0
                }
                _ => {
                    let io_space = self.node_color_spaces[i];
                    let has_padding = padding.top != 0.0 || padding.left != 0.0 || padding.bottom != 0.0 || padding.right != 0.0;
                    let num_inputs = node.kind.num_inputs();

                    let task = RenderTask {
                        kind: node.kind.name(),
                        input_color_space: io_space.input,
                        color_space: node.color_space,
                        output_color_space: io_space.output,
                        size,
                        padding: if has_padding { Some(&padding) } else { None },
                        inputs: &inputs[0..num_inputs]
                    };

                    render_graph.add_task(&task)
                }
            };

            task_ids.push((render_task, sub_rect));

        }

        // Handle the output of the filter graph.

        let root_node = &graph.primitives[graph.root.index()];

        let root_color_space = if root_node.output_srgb {
            ColorSpace::Srgb
        } else if root_node.output_linear {
            ColorSpace::Linear
        } else {
            root_node.color_space
        };

        let root_index = graph.root.index();
        let output_space = self.node_color_spaces[root_index].output;
        let src_task_id = task_ids[root_index].0;
        let size = self.node_resolution[root_index];

        let task_id = if output_space != root_color_space {
            let task = RenderTask {
                kind: match output_space {
                    ColorSpace::Srgb => "SrgbToLinear",
                    ColorSpace::Linear => "LinearToSrgb",
                },
                input_color_space: output_space,
                color_space: root_color_space,
                output_color_space: root_color_space,
                size,
                padding: None,
                inputs: &[(src_task_id, Box2D { min: point2(0.0, 0.0), max: point2(1.0, 1.0) })],
            };

            render_graph.add_task(&task)
        } else {
            src_task_id
        };

        let output_rect = self.requested_rects[root_index];

        // TODO: the sub-rect should always be the full rect here but we should make sure.
        output.push((task_id, output_rect));
    }
}



#[derive(Clone)]
pub struct FilterGraphBuilder {
    primitives: Vec<FilterPrimitive>,
    root: Option<FilterPrimitiveId>,
}

impl FilterGraphBuilder {
    pub fn new() -> Self {
        FilterGraphBuilder {
            primitives: Vec::new(),
            root: None,
        }
    }

    #[inline]
    pub fn add_filter(&mut self, filter: FilterPrimitive) -> FilterPrimitiveId {
        let id = FilterPrimitiveId(self.primitives.len() as u16);

        for i in 0..MAX_INPUTS {
            if let Some(input_id) = filter.inputs[i] {
                let input_node = &mut self.primitives[input_id.index()];
                input_node.ref_count += 1;
                match filter.color_space {
                    ColorSpace::Srgb => { input_node.output_srgb = true; }
                    ColorSpace::Linear => { input_node.output_linear = true; }
                }
            }
        }

        self.primitives.push(filter);

        id
    }

    pub fn set_root(&mut self, root: FilterPrimitiveId, color_space: ColorSpace) {
        self.root = Some(root);
        let node = &mut self.primitives[root.index()];
        node.ref_count += 1;
        match color_space {
            ColorSpace::Srgb => { node.output_srgb = true; }
            ColorSpace::Linear => { node.output_linear = true; }
        }
    }

    pub fn build(mut self) -> FilterGraph {
        // Ensure there is at least one node.
        if self.primitives.is_empty() {
            self.add_filter(FilterPrimitive::source_graphic(0));
        }

        // If no root is specified, use the last node in srgb by default.
        if self.root.is_none() {
            let last_node = FilterPrimitiveId(self.primitives.len() as u16 - 1);
            self.set_root(last_node, ColorSpace::Srgb);
        }

        let root = self.root.unwrap();

        FilterGraph {
            primitives: self.primitives,
            root,
        }
    }
}

// compute the normalized sub-rect of input_rect defined by rect.
fn relative_rect(input_rect: &LayoutRect, rect: &LayoutRect) -> SubRect {
    if input_rect.is_empty() {
        return Box2D::zero();
    }

    let r = match rect.intersection(input_rect) {
        Some(r) => r.translate(-input_rect.min.to_vector()),
        None => {
            return Box2D::zero();
        }
    };

    let sx = 1.0 / input_rect.width();
    let sy = 1.0 / input_rect.height();

    r.scale(sx, sy)
}

// Compute the sub-rect of a sub-rect (in 0..1 coordinate space)
fn combine_sub_rects(rect: &SubRect, sub_rect: &SubRect) -> SubRect {
    Box2D {
        min: point2(
            rect.min.x + rect.width() * sub_rect.min.x,
            rect.min.y + rect.height() * sub_rect.min.y,
        ),
        max: point2(
            rect.max.x - rect.width() * (1.0 - sub_rect.max.x),
            rect.max.y - rect.height() * (1.0 - sub_rect.max.y),
        ),
    }
}

pub struct RenderTask<'l> {
    kind: &'l str,
    color_space: ColorSpace,
    input_color_space: ColorSpace,
    output_color_space: ColorSpace,
    size: Size2D<i32>,
    padding: Option<&'l SideOffsets2D<f32>>,
    // source task and sub-rect
    inputs: &'l[(RenderTaskId, SubRect)],
}

impl<'l> RenderTask<'l> {
    pub fn new(name: &'l str, size: Size2D<i32>, color_space: ColorSpace, inputs: &'l[(RenderTaskId, SubRect)]) -> Self {
        RenderTask {
            kind: name,
            color_space,
            input_color_space: color_space,
            output_color_space: color_space,
            size,
            padding: None,
            inputs,
        }
    }
}

pub type RenderTaskId = u32;

pub struct RenderTaskGraph {
    next_id: RenderTaskId,
}

impl RenderTaskGraph {
    pub fn new() -> Self {
        RenderTaskGraph {
            next_id: 0,
        }
    }

    pub fn add_task(&mut self, task: &RenderTask) -> RenderTaskId {
        let id = self.next_id;
        self.next_id += 1;

        println!(
            " + task {:?} {:?} padding: {:?} ({:?}>{:?}>{:?}) inputs {:?} -> {:?}",
            task.kind, task.size, task.padding, task.input_color_space, task.color_space, task.output_color_space, task.inputs, id
        );

        id        
    }
}

fn compute_padding(content_rect: &LayoutRect, requested_rect: &LayoutRect) -> SideOffsets2D<f32> {
    SideOffsets2D::new(
        (content_rect.min.y - requested_rect.min.y).max(0.0),
        (requested_rect.max.x - content_rect.max.x).max(0.0),
        (requested_rect.max.y - content_rect.max.y).max(0.0),
        (content_rect.min.x - requested_rect.min.x).max(0.0),
    )
}

#[test]
fn simple() {
    let mut builder = FilterGraphBuilder::new();

    let src = builder.add_filter(FilterPrimitive::source_graphic(0));
    let flood = builder.add_filter(FilterPrimitive::flood(1.0, 0.0, 0.0, 1.0));
    let blend = builder.add_filter(FilterPrimitive::mix_blend(MixBlendMode::Multiply, [src, flood]));
    let opacity = builder.add_filter(FilterPrimitive::opacity(0.5, blend));

    builder.set_root(opacity, ColorSpace::Srgb);

    let graph = builder.build();

    let mut ctx = FilterInstance::new();
    let mut render_tasks = RenderTaskGraph::new();

    let src_content = GraphInput {
        task_id: render_tasks.add_task(
            &RenderTask::new("Picture", size2(1000, 800), ColorSpace::Srgb, &[])
        ),
        is_opaque: true,
        size: size2(1000, 800),
    };

    ctx.compute_layout(
        &graph,
        &Box2D {
            min: point2(10.0, 5.0),
            max: point2(310.0, 405.0),
        },
        &Box2D {
            min: point2(10.0, 5.0),
            max: point2(310.0, 405.0),
        },
    );

    ctx.generate_render_tasks(&graph, 1.0, &[src_content], &mut render_tasks, &mut Vec::new());
}


#[test]
fn offset() {
    let mut builder = FilterGraphBuilder::new();

    let src = builder.add_filter(FilterPrimitive::source_graphic(0));
    let opacity = builder.add_filter(FilterPrimitive::opacity(0.5, src));
    let offset1 = builder.add_filter(FilterPrimitive::offset(Vector2D::new(250.0, 10.0), opacity).with_color_space(ColorSpace::Srgb));
    let offset2 = builder.add_filter(FilterPrimitive::offset(Vector2D::new(-250.0, -10.0), offset1).with_color_space(ColorSpace::Srgb));
    let opacity = builder.add_filter(FilterPrimitive::opacity(0.8, offset2));

    builder.set_root(opacity, ColorSpace::Srgb);

    let graph = builder.build();

    let mut ctx = FilterInstance::new();
    let mut render_tasks = RenderTaskGraph::new();

    let src_content = GraphInput {
        task_id: render_tasks.add_task(
            &RenderTask::new("Picture", size2(1000, 800), ColorSpace::Srgb, &[])
        ),
        is_opaque: true,
        size: size2(1000, 800),
    };

    ctx.compute_layout(
        &graph,
        &Box2D {
            min: point2(0.0, 0.0),
            max: point2(300.0, 400.0),
        },
        &Box2D {
            min: point2(0.0, 0.0),
            max: point2(500.0, 500.0),
        },
    );

    ctx.generate_render_tasks(
        &graph,
        1.0,
        &[src_content],
        &mut render_tasks,
        &mut Vec::new()
    );
}


#[test]
fn blur() {
    let mut builder = FilterGraphBuilder::new();

    let srgb = ColorSpace::Srgb;

    let src_rect = Box2D {
        min: point2(50.0, 5.0),
        max: point2(250.0, 305.0),
    };

    let src = builder.add_filter(FilterPrimitive::source_graphic(0));
    let opacity = builder.add_filter(FilterPrimitive::opacity(0.5, src));
    let blur = builder.add_filter(
        FilterPrimitive::gaussian_blur(Vector2D::new(5.0, 5.0), opacity)
            .with_primitive_subregion(src_rect.inflate(20.0, 20.0))
    );

    builder.set_root(blur, srgb);

    let graph = builder.build();

    let mut ctx = FilterInstance::new();
    let mut render_tasks = RenderTaskGraph::new();

    let src_content = GraphInput {
        task_id: render_tasks.add_task(
            &RenderTask::new("Picture", size2(1000, 800), ColorSpace::Srgb, &[])
        ),
        is_opaque: true,
        size: size2(1000, 800),
    };

    ctx.compute_layout(
        &graph,
        &[src_rect],
        &Box2D {
            min: point2(0.0, 0.0),
            max: point2(500.0, 500.0),
        },
    );

    ctx.generate_render_tasks(
        &graph,
        1.0,
        &[src_content],
        &mut render_tasks,
        &mut Vec::new()
    );
}
