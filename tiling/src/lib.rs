use lyon_path::PathEvent;
use lyon_path::geom::euclid::{Box2D, Size2D, vec2};
use lyon_path::math::{Point, point};
use lyon_path::geom::{Line, LineSegment, QuadraticBezierSegment};

//use std::ops::Range;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum EdgeKind {
    Linear,
    Quadratic,
}

#[derive(Copy, Clone, Debug, PartialEq)]
struct Edge {
    from: Point,
    to: Point,
    ctrl: Point,
    kind: EdgeKind,
    winding: i16,
}

impl Edge {
    fn min_x(&self) -> f32 {
        self.from.x.min(self.to.x)
    }

    fn linear(mut segment: LineSegment<f32>) -> Self {
        let winding = if segment.from.y > segment.to.y {
            std::mem::swap(&mut segment.from, &mut segment.to);
            -1
        } else {
            1
        };

        Edge {
            from: segment.from,
            to: segment.to,
            ctrl: segment.to,
            kind: EdgeKind::Linear,
            winding,
        }
    }

    fn quadratic(mut segment: QuadraticBezierSegment<f32>) -> Self {
        let winding = if segment.from.y > segment.to.y {
            std::mem::swap(&mut segment.from, &mut segment.to);
            -1
        } else {
            1
        };

        Edge {
            from: segment.from,
            to: segment.to,
            ctrl: segment.ctrl,
            kind: EdgeKind::Linear,
            winding,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
struct RowEdge {
    from: Point,
    to: Point,
    ctrl: Point,
    kind: EdgeKind,
    winding: i16,
    min_x: f32,
}


#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ActiveEdge {
    pub from: Point,
    pub to: Point,
    pub ctrl: Point,
    pub kind: EdgeKind,
    pub winding: i16,
    max_x: f32,
}

impl ActiveEdge {
    pub fn clip_horizontally(&self, x_range: std::ops::Range<f32>) -> Self {
        match self.kind {
            EdgeKind::Linear => {
                let mut segment = LineSegment {
                    from: self.from,
                    to: self.to,
                };
                let swap = segment.from.x > segment.to.x;
                if swap {
                    std::mem::swap(&mut segment.from, &mut segment.to);
                }
                let t0 = segment.line_intersection_t(&Line { point: point(x_range.start, 0.0), vector: vec2(0.0, 1.0) }).unwrap_or(0.0);
                let t1 = segment.line_intersection_t(&Line { point: point(x_range.end, 0.0), vector: vec2(0.0, 1.0) }).unwrap_or(1.0);

                let mut segment = segment.split_range(t0..t1);
                if swap {
                    std::mem::swap(&mut segment.from, &mut segment.to);
                }

                ActiveEdge {
                    from: segment.from,
                    to: segment.to,
                    ctrl: segment.to,
                    winding: self.winding,
                    kind: EdgeKind::Linear,
                    max_x: 0.0,
                }
            }
            EdgeKind::Quadratic => {
                let segment = QuadraticBezierSegment {
                    from: self.from,
                    to: self.to,
                    ctrl: self.ctrl,
                };
                let t0 = *segment.line_intersections_t(&Line { point: point(x_range.start, 0.0), vector: vec2(0.0, 1.0) }).first().unwrap_or(&0.0);
                let t1 = *segment.line_intersections_t(&Line { point: point(x_range.end, 0.0), vector: vec2(0.0, 1.0) }).first().unwrap_or(&1.0);

                let segment = segment.split_range(t0..t1);

                ActiveEdge {
                    from: segment.from,
                    to: segment.to,
                    ctrl: segment.ctrl,
                    winding: self.winding,
                    kind: EdgeKind::Quadratic,
                    max_x: 0.0,
                }
            }
        }
    }
}

pub struct TileInfo {
    pub x: u32,
    pub y: u32,
    pub z_index: u16,
    pub backdrop_winding: i16,
    pub inner_rect: Box2D<f32>,
    pub outer_rect: Box2D<f32>,
}

pub trait TileEncoder {
    fn encode_tile(&mut self, tile: &TileInfo, active_edges: &[ActiveEdge]);
}

impl<T> TileEncoder for T
where
    T: FnMut(&TileInfo, &[ActiveEdge])
{
    fn encode_tile(&mut self, tile: &TileInfo, active_edges: &[ActiveEdge]) {
        (*self)(tile, active_edges)
    }
}

pub struct Tiler {
    tile_size: Size2D<f32>,
    // offset of the first tile
    tile_offset_x: f32,
    tile_offset_y: f32,
    tile_padding: f32,

    num_tiles_x: u32,
    num_tiles_y: u32,

    tolerance: f32,
    flatten: bool,
}

impl Tiler {
    pub fn new(rect: &Box2D<f32>, tile_size: Size2D<f32>, tile_padding: f32) -> Self {
        Tiler {
            tile_size,
            tile_offset_x: rect.min.x,
            tile_offset_y: rect.min.y,
            tile_padding,
            num_tiles_x: f32::ceil(rect.size().width / tile_size.width) as u32,
            num_tiles_y: f32::ceil(rect.size().height / tile_size.height) as u32,
            tolerance: 0.01,
            flatten: false,
        }
    }

    pub fn set_tolerance(&mut self, tolerance: f32) {
        self.tolerance = tolerance;
    }

    pub fn set_flattening(&mut self, flatten: bool) {
        self.flatten = flatten;
    }

    pub fn tile_path(&self, path_ctx: &mut PathCtx, path: impl Iterator<Item = PathEvent>, encoder: &mut dyn TileEncoder) {
        self.prepare_path_ctx(path_ctx);

        if self.flatten {
            self.assign_rows_linear(path_ctx, path);
        } else {
            self.assign_rows_quadratic(path_ctx, path);
        }

        self.process_rows(path_ctx, encoder);
    }

    fn prepare_path_ctx(&self, path_ctx: &mut PathCtx) {
        let num_rows = self.num_tiles_y as usize;
        path_ctx.rows.truncate(num_rows);
        for _ in 0..(num_rows - path_ctx.rows.len()) {
            path_ctx.rows.push(Vec::new());
        }

        for row in &mut path_ctx.rows {
            row.clear();
        }
    }

    fn assign_rows_quadratic(&self, path_ctx: &mut PathCtx, path: impl Iterator<Item = PathEvent>) {
        for evt in path {
            match evt {
                PathEvent::MoveTo(..) => {}
                PathEvent::Line(segment) | PathEvent::Close(segment) => {
                    let edge = Edge::linear(segment);
                    self.add_y_monotonic_edge(path_ctx, &edge);
                }
                PathEvent::Quadratic(segment) => {
                    segment.for_each_monotonic(&mut|monotonic| {
                        let edge = Edge::quadratic(*monotonic.segment());
                        self.add_y_monotonic_edge(path_ctx, &edge);
                    });
                }
                PathEvent::Cubic(segment) => {
                    segment.for_each_quadratic_bezier(self.tolerance, &mut|segment| {
                        segment.for_each_monotonic(&mut|monotonic| {
                            let edge = Edge::quadratic(*monotonic.segment());
                            self.add_y_monotonic_edge(path_ctx, &edge);
                        });
                    });
                }
            }
        }
    }

    fn assign_rows_linear(&self, path_ctx: &mut PathCtx, path: impl Iterator<Item = PathEvent>) {
        for evt in path {
            match evt {
                PathEvent::MoveTo(..) => {}
                PathEvent::Line(segment) | PathEvent::Close(segment) => {
                    let edge = Edge::linear(segment);
                    self.add_y_monotonic_edge(path_ctx, &edge);
                }
                PathEvent::Quadratic(segment) => {
                    let mut from = segment.from;
                    segment.for_each_flattened(self.tolerance, &mut|to| {
                        let edge = Edge::linear(LineSegment { from, to });
                        from = to;
                        self.add_y_monotonic_edge(path_ctx, &edge);
                    });
                }
                PathEvent::Cubic(segment) => {
                    let mut from = segment.from;
                    segment.for_each_flattened(self.tolerance, &mut|to| {
                        let edge = Edge::linear(LineSegment { from, to });
                        from = to;
                        self.add_y_monotonic_edge(path_ctx, &edge);
                    });
                }
            }
        }
    }

    fn add_y_monotonic_edge(&self, path_ctx: &mut PathCtx, edge: &Edge) {
        assert!(edge.from.y <= edge.to.y);

        let y_start_tile = f32::floor((edge.from.y - self.tile_offset_y - self.tile_padding) / self.tile_size.height).max(0.0);
        let y_end_tile = f32::ceil((edge.to.y + self.tile_offset_y - self.tile_padding) / self.tile_size.height).min(self.num_tiles_y as f32);

        if y_end_tile - y_start_tile < 0.001 {
            return;
        }

        let mut row_f = y_start_tile;
        let mut y_min = self.tile_offset_y + row_f * self.tile_size.height - self.tile_padding;
        let mut y_max = self.tile_offset_y + (row_f + 1.0) * self.tile_size.height + self.tile_padding;
        let start_idx = y_start_tile as usize;
        let end_idx = y_end_tile as usize;
        match edge.kind {
            EdgeKind::Linear => for row in &mut path_ctx.rows[start_idx .. end_idx] {
                let segment = LineSegment { from: edge.from, to: edge.to };
                let t0 = segment.line_intersection_t(&Line { point: point(0.0, y_min), vector: vec2(1.0, 0.0) }).unwrap_or(0.0);
                let t1 = segment.line_intersection_t(&Line { point: point(0.0, y_max), vector: vec2(1.0, 0.0) }).unwrap_or(1.0);

                let segment = segment.split_range(t0..t1);

                row.push(RowEdge {
                    from: segment.from,
                    to: segment.to,
                    ctrl: segment.to,
                    kind: EdgeKind::Linear,
                    winding: edge.winding,
                    min_x: segment.from.x.min(segment.to.x),
                });

                y_min += self.tile_size.height;
                y_max += self.tile_size.height;
                row_f += 1.0;
            }
            EdgeKind::Quadratic => for row in &mut path_ctx.rows[start_idx .. end_idx] {
                let segment = QuadraticBezierSegment { from: edge.from, ctrl: edge.ctrl, to: edge.to };
                let t0 = *segment.line_intersections_t(&Line { point: point(0.0, y_min), vector: vec2(1.0, 0.0) }).first().unwrap_or(&0.0);
                let t1 = *segment.line_intersections_t(&Line { point: point(0.0, y_max), vector: vec2(1.0, 0.0) }).first().unwrap_or(&1.0);

                let segment = segment.split_range(t0..t1);

                row.push(RowEdge {
                    from: segment.from,
                    to: segment.to,
                    ctrl: segment.ctrl,
                    kind: EdgeKind::Quadratic,
                    winding: edge.winding,
                    min_x: segment.from.x.min(segment.to.x),
                });

                y_min += self.tile_size.height;
                y_max += self.tile_size.height;
                row_f += 1.0;
            }
        }
    }


    fn process_rows(&self, path_ctx: &mut PathCtx, encoder: &mut dyn TileEncoder) {
        let mut tile_y = 0;
        let mut active_edges = Vec::new();
        for row in &mut path_ctx.rows {
            self.process_row(tile_y, path_ctx.z_index, &mut row[..], &mut active_edges, encoder);
            tile_y += 1;
        }
    }

    fn process_row(
        &self,
        tile_y: u32,
        z_index: u16,
        row: &mut [RowEdge],
        active_edges: &mut Vec<ActiveEdge>,
        encoder: &mut dyn TileEncoder,
    ) {
        println!("\n<!-- row {} -->", tile_y);

        row.sort_by(|a, b| a.min_x.partial_cmp(&b.min_x).unwrap());

        active_edges.clear();

        let row_y = tile_y as f32 * self.tile_size.height + self.tile_offset_x;
        let y_baseline = row_y - self.tile_padding;

        let inner_rect = Box2D {
            min: point(self.tile_offset_x, row_y),
            max: point(
                self.tile_offset_x + self.tile_size.width,
                row_y + self.tile_size.height
            ),
        };

        let mut tile = TileInfo {
            x: 0,
            y: tile_y,
            z_index,
            backdrop_winding: 0,
            inner_rect,
            outer_rect: inner_rect.inflate(self.tile_padding, self.tile_padding),
        };

        let mut current_edge = 0;

        // First iterate on edges until we reach one that starts inside the tiling area.
        // During this phase we only need to keep track of the backdrop winding number
        // and detect edges that end in the tiling area.
        for edge in &row[..] {
            if edge.min_x >= self.tile_offset_x {
                break;
            }

            if edge.from.y <= y_baseline {
                tile.backdrop_winding += edge.winding;
            }

            let max_x = edge.from.x.max(edge.to.x);

            if max_x >= tile.outer_rect.min.x {
                println!("  <!-- push active edge in phase 1 -->");
                active_edges.push(ActiveEdge {
                    from: edge.from,
                    to: edge.to,
                    ctrl: edge.ctrl,
                    winding: edge.winding,
                    kind: edge.kind,
                    max_x,
                });
            }

            current_edge += 1;
        }

        println!("  <!-- Phase 2 -->");
        // Iterate over edges in the tiling area.
        // Now we produce actual tiles.
        //
        // Each time we get to a new tile, we remove all active edges that end left of the tile.
        // In practice this means all active edges intersect the current tile.
        for edge in &row[current_edge..] {
            println!("  <!-- edge {:?} -->", edge);
            while edge.min_x > tile.outer_rect.max.x {
                self.finish_tile(&mut tile, active_edges, encoder);
            }

            if tile.x >= self.num_tiles_x {
                break;
            }

            if edge.from.y <= y_baseline {
                tile.backdrop_winding += edge.winding;
            }

            active_edges.push(ActiveEdge {
                from: edge.from,
                to: edge.to,
                ctrl: edge.ctrl,
                winding: edge.winding,
                kind: edge.kind,
                max_x: edge.from.x.max(edge.to.x),
            });

            current_edge += 1;
        }

        println!("  <!-- Phase 3 -->");
        // At this point we visited all edges but not necessarily all tiles.
        while tile.x < self.num_tiles_x {
            self.finish_tile(&mut tile, active_edges, encoder);

            if active_edges.is_empty() {
                break;
            }
        }
    }

    fn finish_tile(&self, tile: &mut TileInfo, active_edges: &mut Vec<ActiveEdge>, encoder: &mut dyn TileEncoder) {
        println!("  <!-- tile {} {} -->", tile.x, tile.y);
        encoder.encode_tile(tile, active_edges);

        tile.inner_rect.min.x += self.tile_size.width;
        tile.inner_rect.max.x += self.tile_size.width;
        tile.outer_rect.min.x += self.tile_size.width;
        tile.outer_rect.max.x += self.tile_size.width;
        tile.x += 1;

        active_edges.retain(|edge| edge.max_x > tile.outer_rect.min.x);
    }
}

pub struct PathCtx {
    rows: Vec<Vec<RowEdge>>,    
    z_index: u16,
}

impl PathCtx {
    pub fn new(z_index: u16) -> Self {
        PathCtx {
            rows: Vec::new(),
            z_index,
        }
    }
}



pub struct ZBuffer {
    data: Vec<u16>,
    w: usize,
}

impl ZBuffer {
    pub fn new() -> Self {
        ZBuffer {
            data: Vec::new(),
            w: 0,
        }
    }

    pub fn init(&mut self, w: usize, h: usize) {
        let size = w * h;
        if self.data.len() < size {
            self.data = vec![0; size];
            return;
        }

        for elt in &mut self.data[0..size] {
            *elt = 0;
        }

        self.w = w;
    }

    pub fn get(&self, x: u32, y: u32) -> u16 {
        self.data[self.index(x, y)]
    }

    #[inline]
    pub fn index(&self, x: u32, y: u32) -> usize {
        self.w * (y as usize) + (x as usize)
    }
}

#[test]
fn test() {
    use lyon_path::geom::euclid::size2;

    let mut builder = lyon_path::Path::builder();

    builder.move_to(point(30.0, 10.0));
    builder.line_to(point(90.0, 40.0));
    builder.line_to(point(10.0, 90.0));
    builder.close();

    builder.move_to(point(40.0, 50.0));
    builder.line_to(point(60.0, 55.0));
    builder.line_to(point(40.0, 60.0));
    builder.close();

    let path = builder.build();

    let tiler = Tiler::new(&Box2D { min: point(0.0, 0.0), max: point(100.0, 100.0) }, size2(10.0, 10.0), 1.0);
    let mut ctx = PathCtx::new(0);

    println!("{}", svg_fmt::BeginSvg { w: 100.0, h: 100.0 });

    struct Encoder;
    impl TileEncoder for Encoder {
        fn encode_tile(&mut self, tile: &TileInfo, edges: &[ActiveEdge]) {
            for edge in edges {
                let edge = edge.clip_horizontally(tile.outer_rect.min.x .. tile.outer_rect.max.x);
                let color = if edge.winding > 0 {
                    svg_fmt::Color { r: 0, g: 0, b: 255 }                
                } else{
                    svg_fmt::Color { r: 255, g: 0, b: 0 }                
                };
                println!("  {}", svg_fmt::line_segment(edge.from.x, edge.from.y, edge.to.x, edge.to.y).color(color));
            }
            if edges.is_empty() {
                if tile.backdrop_winding == 0 {
                    return;
                }

                if tile.backdrop_winding % 2 != 0 {
                    println!("  {}",
                        svg_fmt::rectangle(
                            tile.inner_rect.min.x,
                            tile.inner_rect.min.y,
                            tile.inner_rect.size().width,
                            tile.inner_rect.size().height,
                        )
                        .fill(svg_fmt::blue())
                        .opacity(0.3)
                    );

                }
            } 
            println!("  {}",
                svg_fmt::rectangle(
                    tile.inner_rect.min.x,
                    tile.inner_rect.min.y,
                    tile.inner_rect.size().width,
                    tile.inner_rect.size().height,
                )
                .fill(svg_fmt::Fill::None)
                .stroke(svg_fmt::Stroke::Color(svg_fmt::black(), 0.1))
            );
            println!("  {}",
                svg_fmt::rectangle(
                    tile.outer_rect.min.x,
                    tile.outer_rect.min.y,
                    tile.outer_rect.size().width,
                    tile.outer_rect.size().height,
                )
                .fill(svg_fmt::Fill::None)
                .stroke(svg_fmt::Stroke::Color(svg_fmt::green(), 0.1))
            );
        }
    }

    tiler.tile_path(&mut ctx, path.iter(), &mut Encoder);

    println!("{}", svg_fmt::EndSvg);

    panic!();
}
