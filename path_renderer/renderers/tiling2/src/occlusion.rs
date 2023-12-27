
pub struct OcclusionBuffer {
    disabled: bool,
    data: Vec<u8>,
    width: usize,
    height: u32,
}

impl OcclusionBuffer {
    pub fn new(w: u32, h: u32) -> Self {
        let mut m = OcclusionBuffer {
            disabled: false,
            data: Vec::new(),
            width: 0,
            height: 0,
        };

        m.init(w, h);

        m
    }

    pub fn disabled() -> Self {
        OcclusionBuffer {
            disabled: true,
            data: Vec::new(),
            width: 0,
            height: 0,
        }
    }

    pub fn disable(&mut self) {
        self.disabled = true;
    }

    pub fn init(&mut self, w: u32, h: u32) {
        self.disabled = false;
        let w = w as usize;
        let h = h as usize;
        if self.data.len() < w * h {
            self.data = vec![0; w * h];
        } else {
            self.data[..(w * h)].fill(0);
        }
        self.width = w;
        self.height = h as u32;
    }

    pub fn resize(&mut self, w: u32, h: u32) {
        if self.disabled || w * h <= self.width as u32 * self.height {
            return;
        }

        let w = w as usize;
        let h = h as usize;

        let mut data = vec![0; w * h];

        for y in 0..self.height {
            let src_start = y as usize * self.width;
            let src_end = src_start + self.width;
            let dst_start = y as usize * w;
            let dst_end = dst_start + self.width;
            data[dst_start..dst_end].copy_from_slice(&self.data[src_start..src_end]);
        }

        self.data = data;

        self.width = w;
        self.height = h as u32;
    }

    pub fn width(&self) -> u32 {
        self.width as u32
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn occluded(&mut self, x: u16, y: u16) -> bool {
        if self.disabled {
            return false;
        }
        let offset = x as usize + y as usize * self.width;
        assert!(offset < self.data.len(), "occlustion.get({x} {y}) out of bounds {} {}", self.width, self.height);
        self.data[offset] != 0
    }

    pub fn test(&mut self, x: u16, y: u16, write: bool) -> bool {
        if self.disabled {
            return true;
        }

        let offset = x as usize + y as usize * self.width;
        assert!(offset < self.data.len(), "occlustion.test({x} {y}) out of bounds {} {}", self.width, self.height);
        let payload = &mut self.data[offset as usize];
        let result = *payload == 0;

        if write {
            *payload = 1;
        }

        result
    }

    pub fn clear(&mut self) {
        self.data.fill(0);
    }
}
