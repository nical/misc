// Note:
// Also tried packing into a vector of u64 with a bit per tile, including a version that
// caches the current u64 payload, but the naive one-u8-per-tile approach performed better.

#[derive(Clone, Debug)]
pub struct TiledOcclusionBuffer {
    data: Vec<u8>,
    width: usize,
    height: u32,
}

impl TiledOcclusionBuffer {
    pub fn new(w: u32, h: u32) -> Self {
        let mut m = TiledOcclusionBuffer {
            data: Vec::new(),
            width: 0,
            height: 0,
        };

        m.init(w, h);

        m
    }

    pub fn init(&mut self, w: u32, h: u32) {
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

    pub fn width(&self) -> u32 {
        self.width as u32
    }
    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn row(&mut self, y: u32) -> TiledOcclusionBufferRow {
        if self.data.is_empty() {
            return TiledOcclusionBufferRow { data: &mut [] };
        }
        let offset = y as usize * self.width;
        TiledOcclusionBufferRow {
            data: &mut self.data[offset..(offset + self.width)],
        }
    }

    pub fn get(&mut self, x: u32, y: u32) -> bool {
        let offset = x as usize + y as usize * self.width;
        self.data[offset] != 0
    }

    pub fn write_clip(&mut self, x: u32, y: u32) {
        let offset = x as usize + y as usize * self.width;
        self.data[offset] |= 2
    }

    /// Returns true if the tile at the provided offset wasn't masked.
    ///
    /// If 'write' is true, mark tile as masked.
    pub fn test(&mut self, x: u32, y: u32, write: bool) -> bool {
        let offset = x as usize + y as usize * self.width;
        let payload = &mut self.data[offset as usize];
        let result = *payload == 0;

        if write {
            *payload = 1;
        }

        result
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn clear(&mut self) {
        self.data.fill(0);
    }
}

pub struct TiledOcclusionBufferRow<'l> {
    data: &'l mut [u8],
}

impl<'l> TiledOcclusionBufferRow<'l> {
    pub fn get(&mut self, offset: u32) -> bool {
        self.data
            .get(offset as usize)
            .map(|d| *d != 0)
            .unwrap_or(true)
    }

    pub fn write_clip(&mut self, offset: u32) {
        if let Some(data) = self.data.get_mut(offset as usize) {
            *data |= 2;
        }
    }

    /// Returns true if the tile at the provided offset wasn't masked.
    ///
    /// If 'write' is true, mark tile as masked.
    pub fn test(&mut self, offset: u32, write: bool) -> bool {
        let payload = match self.data.get_mut(offset as usize) {
            Some(p) => p,
            None => {
                return true;
            }
        };
        let result = *payload == 0;

        if write {
            *payload = 1;
        }

        result
    }
}
