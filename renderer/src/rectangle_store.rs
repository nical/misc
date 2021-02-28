use crate::allocator::*;
use crate::bump_allocator::BumpAllocator;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use lru::LruCache;

pub type GpuBlock = [u8; 16];

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct GpuAddress(u32);

impl GpuAddress {
    pub fn with_offset(&self, diff: i32) -> Self {
        GpuAddress(self.0 + diff as u32)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct BufferId(u32);

struct Page {
    address: GpuAddress,
    allocated: bool,
}

pub struct PageAllocator {
    bump: BumpAllocator,
    available_pages: Vec<usize>,
    new_pages: AtomicUsize,
    page_size: u32,
}

impl PageAllocator {
    fn allocate_page(&self) -> usize {
        if let Some(range) = self.bump.allocate_front(1) {
            return self.available_pages[range.start as usize];
        }

        let page_idx = self.new_pages.fetch_add(1, Ordering::SeqCst) as usize + 1;

        page_idx
    }
}

pub struct Buffer {
    page_allocator: Arc<PageAllocator>,
    dummy_allocator: Arc<PageAllocator>,
    pages: Vec<Page>,
}

impl Buffer {
    pub fn new(initiasize: u32, page_size: u32) -> Self {
        let num_pages = initiasize / page_size;
        let mut pages = Vec::with_capacity(num_pages as usize);
        for i in 0..num_pages {
            pages.push(Page {
                address: GpuAddress(page_size * i),
                allocated: false,
            });
        }
        Buffer {
            page_allocator: Arc::new(PageAllocator {
                bump: BumpAllocator::new(0..0),
                available_pages: Vec::new(),
                new_pages: AtomicUsize::new(0),
                page_size,
            }),
            dummy_allocator: Arc::new(PageAllocator {
                bump: BumpAllocator::new(0..0),
                available_pages: Vec::new(),
                new_pages: AtomicUsize::new(0),
                page_size: 0,
            }),
            pages,
        }
    }

    pub fn upload(&mut self, offset: GpuAddress, data: &[GpuBlock]) {
        println!("upload size {:?} at offset {:?}", data.len(), offset);
        // TODO
    }

    pub fn end_frame(&mut self) {
        println!(" strong {} |  weak {}", Arc::strong_count(&self.page_allocator), Arc::weak_count(&self.page_allocator));
        let page_allocator = Arc::get_mut(&mut self.page_allocator).unwrap();        

        // TODO: need this to happen before the allocators call end_frame.
        for idx in self.pages.len()..page_allocator.new_pages.load(Ordering::SeqCst) {
            self.pages.push(Page {
                address: GpuAddress(idx as u32 * page_allocator.page_size),
                allocated: true, 
            });
        }

        page_allocator.available_pages.clear();

        for (idx, page) in self.pages.iter().enumerate() {
            if !page.allocated {
                page_allocator.available_pages.push(idx);
            }
        }
    }
}

struct SubAllocatorPage {
    page_index: usize,
    address: GpuAddress,
    buffer: Vec<GpuBlock>,
}

/// Allocates entries that are only valid for the current frame.
pub struct FrameSubAllocator {
    page_allocator: Arc<PageAllocator>,
    pages: Vec<SubAllocatorPage>,
    current_page: Option<SubAllocatorPage>,
}

impl FrameSubAllocator {
    pub fn write_block(&mut self, data: GpuBlock) -> GpuAddress {
        let page_size = self.page_allocator.page_size;
        let (address, offset) = {
            let current_page = match &mut self.current_page {
                Some(page) => page,
                None => {
                    let page_index = self.page_allocator.allocate_page();
                    self.current_page = Some(SubAllocatorPage {
                        address: GpuAddress(page_index as u32 * self.page_allocator.page_size / 16),
                        page_index,
                        buffer: Vec::with_capacity(page_size as usize),
                    });
                    self.current_page.as_mut().unwrap()
                }
            };

            let offset = current_page.buffer.len() as i32;
            current_page.buffer.push(data);

            (current_page.address.with_offset(offset), offset)
        };

        if offset >= page_size as i32 - 1 {
            // The current page is full.
            self.current_page = None;
        }

        address
    }

    pub fn end_frame(&mut self, buffer: &mut Buffer) {
        for page in &self.pages {
            buffer.upload(page.address, &page.buffer);
        }

        self.page_allocator = Arc::clone(&buffer.dummy_allocator);

        self.pages.clear();
    }

    pub fn begin_frame(&mut self, buffer: &Buffer) {
        self.page_allocator = Arc::clone(&buffer.page_allocator);
    }
}

pub struct UpdateMask {
    mask: Vec<bool>,
    chunk_size: u16,
}

impl UpdateMask {
    pub fn new(size: u16, chunk_size: u16) -> Self {
        assert_eq!(size % chunk_size, 0);
        let buf_size = (size / chunk_size) as usize;
        UpdateMask {
            mask: vec![false; buf_size],
            chunk_size,
        }
    }

    pub fn reset(&mut self) {
        for val in &mut self.mask {
            *val = false;
        }
    }

    pub fn update(&mut self, index: u16) {
        let index = (index / self.chunk_size) as usize;
        self.mask[index] = true;
    }

    pub fn update_ange(&mut self, range: std::ops::Range<u16>) {
        for index in range {
            let index = (index / self.chunk_size) as usize;
            self.mask[index] = true;
        }
    }

    pub fn for_each_updated_range(&self, callback: &mut dyn FnMut(std::ops::Range<u16>)) {
        let mut start = None;
        let mut idx = 0;
        let len = self.mask.len();
        while idx <= len {
            let val = if idx == len { false } else { self.mask[idx] };
            // println!("{:?} [{:?}]", idx, val);

            if val {
                if start.is_none() {
                    start = Some(idx as u16 * self.chunk_size);
                }
            } else {
                if let Some(s) = start {
                    let end = idx as u16 * self.chunk_size;
                    callback(s..end);
                    start = None;
                }
            }

            idx += 1;
        }
    }
}

// TODO: implement removing empty pages.

struct RetainedSlotSubAllocatorPage {
    page_index: usize,
    address: GpuAddress,
    buffer: Vec<GpuBlock>,
    allocator: SlotAllocator,
    upload_mask: UpdateMask,
    allocated: bool,
}

pub struct RetainedSlotSubAllocator {
    page_allocator: Arc<PageAllocator>,
    pages: Vec<RetainedSlotSubAllocatorPage>,
    slot_size: u32,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct RetainedSlotSubAllocation {
    slot: SlotAllocId,
    page_index: u16,
}

impl RetainedSlotSubAllocator {
    pub fn allocate_block(&mut self, data: GpuBlock) -> (RetainedSlotSubAllocation, GpuAddress) {
        let page_size = self.page_allocator.page_size;

        let mut page_index = None;
        for (idx, page) in self.pages.iter().enumerate() {
            if page.allocated && page.allocator.can_allocate() {
                page_index = Some(idx);
                break;
            }
        }

        let page_index = page_index.unwrap_or_else(|| {
            let page_index = self.page_allocator.allocate_page();
            let size = page_size as usize / 16;
            let index = self.pages.len();
            self.pages.push(RetainedSlotSubAllocatorPage {
                allocator: SlotAllocator::new(size as u16),
                upload_mask: UpdateMask::new(page_size as u16, 16),
                buffer: vec!([0; 16]; size),
                address: GpuAddress((page_index * size) as u32),
                page_index,
                allocated: true,
            });

            index
        });

        let page = &mut self.pages[page_index];
        let slot = page.allocator.allocate().unwrap();
        let address = page.address.with_offset(slot.offset() as i32);

        page.buffer[slot.offset()] = data;
        page.upload_mask.update(slot.offset() as u16);

        (
            RetainedSlotSubAllocation {
                slot,
                page_index: page_index as u16, 
            },
            address,
        )
    }

    pub fn deallocate_block(&mut self, alloc: RetainedSlotSubAllocation) {
        self.pages[alloc.page_index as usize].allocator.deallocate(alloc.slot);
    }

    pub fn get_address(&self, alloc: RetainedSlotSubAllocation) -> GpuAddress {
        self.pages[alloc.page_index as usize].address.with_offset(alloc.slot.offset() as i32)
    }

    pub fn begin_frame(&mut self, buffer: &Buffer) {
        self.page_allocator = Arc::clone(&buffer.page_allocator);
    }

    pub fn end_frame(&mut self, buffer: &mut Buffer) {
        for page in &mut self.pages {
            if !page.allocated {
                continue;
            }
            page.upload_mask.for_each_updated_range(&mut |range| {
                let address = page.address.with_offset(range.start as i32);
                let end = (range.end as usize).min(page.buffer.len());
                buffer.upload(address, &page.buffer[(range.start as usize)..end]);
            });
            page.upload_mask.reset();
        }

        self.page_allocator = Arc::clone(&buffer.dummy_allocator);

        for page in &mut self.pages {
            if !page.allocator.has_allocations() {
                buffer.pages[page.page_index].allocated = false;
                page.allocated = false;
            }
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct RectangleId(RetainedSlotSubAllocation);

pub struct RectangleStore {
    frame_allocations: FrameSubAllocator,
    retained_allocations: RetainedSlotSubAllocator,
    lru: LruCache<RectangleId, u64>,
    current_frame: u64,
}

impl RectangleStore {
    pub fn new(buffer: &Buffer) -> Self {
        RectangleStore {
            frame_allocations: FrameSubAllocator {
                page_allocator: Arc::clone(&buffer.page_allocator),
                pages: Vec::new(),
                current_page: None,
            },
            retained_allocations: RetainedSlotSubAllocator {
                page_allocator: Arc::clone(&buffer.page_allocator),
                pages: Vec::new(),
                slot_size: 1,
            },
            lru: LruCache::unbounded(),
            current_frame: 0,
        }

    }

    pub fn begin_frame(&mut self, buffer: &Buffer) {
        self.frame_allocations.begin_frame(buffer);
        self.retained_allocations.begin_frame(buffer);
    }

    pub fn end_frame(&mut self, buffer: &mut Buffer) {
        self.frame_allocations.end_frame(buffer);
        self.retained_allocations.end_frame(buffer);
    }

    pub fn use_retained_item(&mut self, opt_id: &mut Option<RectangleId>, callback: &mut dyn FnMut() -> [f32; 4]) -> GpuAddress {
        if let Some(id) = opt_id {
            if let Some(frame) = self.lru.get_mut(id) {
                *frame = self.current_frame;
                return self.retained_allocations.get_address(id.0);
            }
        }

        let data: GpuBlock = unsafe { std::mem::transmute(callback()) };

        let (alloc, address) = self.retained_allocations.allocate_block(data);

        let id = RectangleId(alloc);
        *opt_id = Some(id);
        self.lru.put(id, self.current_frame);

        address
    }

    pub fn set_single_frame_item(&mut self, data: [f32; 4]) -> GpuAddress {
        let gpu_blocks = unsafe { std::mem::transmute(data) };
        self.frame_allocations.write_block(gpu_blocks)
    }
}

#[test]
fn update_mask() {
    let mut mask = UpdateMask::new(128, 16);
    mask.update(17);
    mask.update(35);
    mask.update(70);
    mask.update(100);
    mask.update(126);

    let mut expected_ranges = [
        16..48,
        64..80,
        96..128,
    ];

    let mut i = 0;
    mask.for_each_updated_range(&mut |range| {
        assert_eq!(range, expected_ranges[i]);
        i += 1;
    });

    assert_eq!(i, 3);

    mask.reset();

    mask.for_each_updated_range(&mut |_range| {
        panic!();
    });
}

#[test]
fn simple() {
    let mut buffer = Buffer::new(8192, 512);

    let mut rectangles = RectangleStore::new(&buffer);

    rectangles.begin_frame(&buffer);

    let mut id1 = None;
    let addr1 = rectangles.use_retained_item(&mut id1, &mut || { [0.0, 1.0, 2.0, 3.0] });

    let mut id2 = None;
    let addr2 = rectangles.use_retained_item(&mut id2, &mut || { [1.0, 2.0, 3.0, 4.0] });

    let addr3 = rectangles.set_single_frame_item( [2.0, 3.0, 4.0, 5.0] );

    rectangles.end_frame(&mut buffer);

    buffer.end_frame();

    assert!(id1.is_some());
    assert!(id2.is_some());
    assert!(id1 != id2);
    assert!(addr1 != addr2);
    assert!(addr1 != addr3);
    assert!(addr2 != addr3);
}
