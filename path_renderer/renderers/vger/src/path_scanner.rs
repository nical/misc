// Ported from vger-rs (https://github.com/user/vger-rs)
// Scan-line decomposition of quadratic Bézier paths into horizontal bands.

pub struct Interval {
    pub a: f32,
    pub b: f32,
}

pub struct PathSegment {
    /// Control vertices stored as [f32; 2] for easy GPU upload.
    pub curve: [[f32; 2]; 3],
    pub next: Option<usize>,
    previous: Option<usize>,
}

impl PathSegment {
    pub fn new(a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> Self {
        Self {
            curve: [a, b, c],
            next: None,
            previous: None,
        }
    }

    pub fn y_interval(&self) -> Interval {
        Interval {
            // Fatten the interval slightly to prevent artifacts by
            // slightly missing a curve in a band.
            a: self.curve[0][1].min(self.curve[1][1]).min(self.curve[2][1]) - 1.0,
            b: self.curve[0][1].max(self.curve[1][1]).max(self.curve[2][1]) + 1.0,
        }
    }
}

#[derive(PartialEq, PartialOrd)]
struct ScannerNode {
    coord: f32,
    seg: usize,
    end: bool,
}

pub struct PathScanner {
    pub segments: Vec<PathSegment>,
    nodes: Vec<ScannerNode>,
    index: usize,
    pub interval: Interval,
    pub first: Option<usize>,
}

impl PathScanner {
    pub fn new() -> Self {
        Self {
            segments: Vec::new(),
            nodes: Vec::new(),
            index: 0,
            interval: Interval { a: 0.0, b: 0.0 },
            first: None,
        }
    }

    pub fn init(&mut self) {
        self.nodes.clear();
        self.index = 0;
        self.first = None;

        for i in 0..self.segments.len() {
            let y_interval = self.segments[i].y_interval();
            self.nodes.push(ScannerNode {
                coord: y_interval.a,
                seg: i,
                end: false,
            });
            self.nodes.push(ScannerNode {
                coord: y_interval.b,
                seg: i,
                end: true,
            });
        }

        self.nodes.sort_by(|a, b| a.partial_cmp(b).unwrap());
    }

    pub fn next(&mut self) -> bool {
        if self.index >= self.nodes.len() {
            return false;
        }

        let y = self.nodes[self.index].coord;
        self.interval.a = y;

        while self.index < self.nodes.len() && self.nodes[self.index].coord == y {
            let node = &self.nodes[self.index];
            let seg = node.seg;

            if node.end {
                // Remove segment from active list.
                if let Some(prev) = self.segments[seg].previous {
                    self.segments[prev].next = self.segments[seg].next;
                }
                if let Some(next) = self.segments[seg].next {
                    self.segments[next].previous = self.segments[seg].previous;
                }
                if self.first == Some(seg) {
                    self.first = self.segments[seg].next;
                }
                self.segments[seg].next = None;
                self.segments[seg].previous = None;
            } else {
                // Insert segment at front of active list.
                self.segments[seg].next = self.first;
                if let Some(first) = self.first {
                    self.segments[first].previous = Some(seg);
                }
                self.first = Some(seg);
            }

            self.index += 1;
        }

        if self.index < self.nodes.len() {
            self.interval.b = self.nodes[self.index].coord;
        }

        self.index < self.nodes.len()
    }

    pub fn clear(&mut self) {
        self.segments.clear();
        self.nodes.clear();
        self.index = 0;
        self.first = None;
    }
}
