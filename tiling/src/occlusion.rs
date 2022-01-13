
// Note:
// Also tried packing into a vector of u64 with a bit per tile, including a version that
// caches the current u64 payload, but the naive one-u8-per-tile approach performed better.

#[derive(Clone, Debug)]
pub struct TileMask {
    data: Vec<u8>,
}

impl TileMask {
    pub fn new() -> Self {
        TileMask {
            data: Vec::new(),
        }
    }

    pub fn init(&mut self, w: usize) {
        if self.data.len() < w {
            self.data = vec![0; w];
        } else {
            for elt in &mut self.data[0..w] {
                *elt = 0;
            }
        }
    }

    pub fn get(&mut self, offset: u32) -> bool {
        self.data[offset as usize] != 0
    }

    /// Returns true if the tile at the provided offset wasn't masked.
    ///
    /// If 'write' is true, mark tile as masked.
    pub fn test(&mut self, offset: u32, write: bool) -> bool {
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
        self.data.clear();
    }
}
