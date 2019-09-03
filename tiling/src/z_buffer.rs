/// Can be used to help with occlusion culling.
pub struct ZBuffer {
    data: Vec<u16>,
    w: usize,
    h: usize,
}

impl ZBuffer {
    pub fn new() -> Self {
        ZBuffer {
            data: Vec::new(),
            w: 0,
            h: 0,
        }
    }

    pub fn init(&mut self, w: usize, h: usize) {
        let size = w * h;

        self.w = w;
        self.h = h;

        if self.data.len() < size {
            self.data = vec![0; size];
        } else {
            for elt in &mut self.data[0..size] {
                *elt = 0;
            }
        }
    }

    /// Fetch the z_index at a given position.
    pub fn get(&self, x: u32, y: u32) -> u16 {
        self.data[self.index(x, y)]
    }

    /// Returns true if the provided z_index is superior to the value of the cell
    /// at (x, y).
    ///
    /// If `write` is true and the test succeeds write the new z_index in the buffer.
    pub fn test(&mut self, x: u32, y: u32, z_index: u16, write: bool) -> bool {
        debug_assert!(x < self.w as u32);
        debug_assert!(y < self.h as u32);

        let idx = self.index(x, y);
        let z = &mut self.data[idx];
        let result = *z < z_index;

        if write && result {
            *z = z_index;
        }

        result
    }

    #[inline]
    fn index(&self, x: u32, y: u32) -> usize {
        self.w * (y as usize) + (x as usize)
    }
}

