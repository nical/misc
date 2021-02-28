
// TODO: Very good candidate for fuzzing.
// TODO: SVG debug view.
// TODO: add generation checks.

type ItemIndex = u16;
const INVALID_ITEM: ItemIndex = std::u16::MAX;

#[derive(Clone, Debug)]
struct Item {
    offset: u16,
    size: u16,
    next: ItemIndex,
    prev: ItemIndex,
    allocated: bool,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SimpleAllocId(ItemIndex);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SimpleAllocation {
    pub id: SimpleAllocId,
    pub offset: u16,
    pub size: u16,
}

#[derive(Clone, Debug)]
pub struct SimpleAllocator {
    items: Vec<Item>,
    alignment: u16,
    first: ItemIndex,
    first_unused_item: ItemIndex,
}

impl SimpleAllocator {
    pub fn new(size: u16) -> Self {
        let mut items = Vec::with_capacity(32);
        items.push(Item {
            offset: 0,
            size,
            prev: INVALID_ITEM,
            next: INVALID_ITEM,
            allocated: false,
        });

        SimpleAllocator {
            items,
            alignment: 1,
            first: 0,
            first_unused_item: INVALID_ITEM,
        }
    }

    pub fn allocate(&mut self, mut size: u16) -> Option<SimpleAllocation> {
        // TODO: This does the very dumb first-fit strategy. It would be better to use
        // worst-fit instead.

        adjust_size(self.alignment, &mut size);

        let mut item_index = None;
        let mut iter = self.first;
        while iter != INVALID_ITEM {
            let item = &self.items[iter as usize];
            if !item.allocated && item.size >= size {
                item_index = Some(iter);
                break
            }

            iter = item.next;
        }

        let item_index = item_index? as usize;

        return if self.items[item_index].size == size {
            // Perfect fit.

            // Unlink from the free list.
            self.items[item_index].allocated = true;

            self.check();

            Some(SimpleAllocation {
                id: SimpleAllocId(item_index as ItemIndex),
                offset: self.items[item_index].offset,
                size,
            })
        } else {
            // Shrink the free item.
            self.items[item_index].size -= size;

            // Add a new one for the allocation.
            let next = self.items[item_index].next;
            let prev = self.items[item_index].prev;
            let offset = self.items[item_index].offset + size;

            let new_item = Item {
                prev: item_index as ItemIndex,
                next,
                offset,
                size,
                allocated: true,
            };

            let new_item_index;
            if self.first_unused_item != INVALID_ITEM {
                // Pop an available item slot from the unused item list.
                let next_free_item = self.items[self.first_unused_item as usize].next;
                new_item_index = self.first_unused_item;
                self.first_unused_item = next_free_item;
                self.items[new_item_index as usize] = new_item;
            } else {
                // add a new item.
                new_item_index = self.items.len() as ItemIndex;
                self.items.push(new_item);
            }

            self.items[item_index].next = new_item_index;
            if next != INVALID_ITEM {
                self.items[next as usize].prev = new_item_index;
            }
            if prev != INVALID_ITEM {
                self.items[prev as usize].next = new_item_index;
            }

            self.check();

            Some(SimpleAllocation {
                id: SimpleAllocId(new_item_index),
                offset,
                size,
            })
        }
    }

    pub fn deallocate(&mut self, id: SimpleAllocId) {
        let item_index = id.0 as usize;

        assert!(self.items[item_index].allocated);

        self.items[item_index].allocated = false;
        let prev = self.items[item_index].prev;
        let next = self.items[item_index].next;

        if next != INVALID_ITEM && !self.items[next as usize].allocated {
            // Merge the item with the next.
            let next_next = self.items[next as usize].next;
            self.items[item_index].size += self.items[next as usize].size;
            // Unlink the next item.
            self.items[item_index].next = next_next;
            if next_next != INVALID_ITEM {
                self.items[next_next as usize].prev = item_index as ItemIndex;
            }
            // Put the unlinked item into a parallel list (reusing next/prev).
            self.items[next as usize].next = self.first_unused_item;
            self.first_unused_item = next;
        }

        if prev != INVALID_ITEM && !self.items[prev as usize].allocated {
            // Shift the item/next terminology so that we can reuse the same logic as above.
            let next = item_index as ItemIndex;
            let item_index = prev as usize;

            // Merge the item with the next.
            let next_next = self.items[next as usize].next;
            self.items[item_index].size += self.items[next as usize].size;
            // Unlink the next item.
            self.items[item_index].next = next_next;
            if next_next != INVALID_ITEM {
                self.items[next_next as usize].prev = item_index as ItemIndex;
            }
            // Put the unlinked item into a parallel list (reusing next/prev).
            self.items[next as usize].next = self.first_unused_item;
            self.first_unused_item = next;
        }

        self.check();
    }

    #[cfg(test)]
    fn check(&self) {
        let mut count = 0;
        let mut prev = INVALID_ITEM;
        let mut iter = self.first;
        while iter != INVALID_ITEM {
            let item = &self.items[iter as usize];

            if prev != INVALID_ITEM {
                assert_eq!(prev, item.prev);
            }

            prev = iter;
            iter = item.next;

            count += 1;
            assert!(count <= self.items.len());
        } 
    }

    #[cfg(not(test))]
    fn check(&self) {}
}

#[allow(dead_code)]
fn print(allocator: &SimpleAllocator) {
    for (idx, item) in allocator.items.iter().enumerate() {
        let s = if item.allocated { "x" } else { " " };
        println!(" - [{}] {} {:?}", s, idx, item);
    }
    println!(" first: {}", allocator.first);
    println!(" first unused: {}", allocator.first_unused_item);
}

fn adjust_size(alignment: u16, size: &mut u16) {
    let rem = *size % alignment;
    if rem > 0 {
        *size += alignment - rem;
    }
}

#[test]
fn simple() {
    let mut allocator = SimpleAllocator::new(100);
    for _ in 0..3 {
        let a1 = allocator.allocate(10).unwrap();
        let a2 = allocator.allocate(10).unwrap();
        let a3 = allocator.allocate(20).unwrap();
        let a4 = allocator.allocate(10).unwrap();
        allocator.deallocate(a2.id);
        let a5 = allocator.allocate(20).unwrap();
        let a6 = allocator.allocate(10).unwrap();
        let a7 = allocator.allocate(10).unwrap();
        let a8 = allocator.allocate(10).unwrap();
        let a9 = allocator.allocate(10).unwrap();
        assert!(allocator.allocate(1).is_none());

        allocator.deallocate(a8.id);
        allocator.deallocate(a6.id);
        allocator.deallocate(a7.id);
        allocator.deallocate(a9.id);
        allocator.deallocate(a5.id);
        allocator.deallocate(a4.id);
        allocator.deallocate(a1.id);
        allocator.deallocate(a3.id);

        let a10 = allocator.allocate(100).unwrap();
        assert!(allocator.allocate(1).is_none());
        allocator.deallocate(a10.id);

        let a11 = allocator.allocate(100).unwrap();
        assert!(allocator.allocate(1).is_none());
        allocator.deallocate(a11.id);
    }
}

// TODO: add some basic checks.

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SlotAllocId(u16);

impl SlotAllocId {
    pub fn offset(self) -> usize { self.0 as usize }
}

#[derive(Clone, Debug)]
pub struct SlotAllocator {
    free_slots: Vec<SlotAllocId>,
    size: u16,
}

impl SlotAllocator {
    pub fn new(size: u16) -> Self {
        let mut free_slots = Vec::with_capacity(size as usize);
        for i in 0..size {
            free_slots.push(SlotAllocId(i));
        }
        SlotAllocator {
            free_slots,
            size,
        }
    }

    pub fn allocate(&mut self) -> Option<SlotAllocId> {
        self.free_slots.pop()
    }

    pub fn deallocate(&mut self, id: SlotAllocId) {
        self.free_slots.push(id);
    }

    pub fn can_allocate(&self) -> bool {
        !self.free_slots.is_empty()
    }

    pub fn has_allocations(&self) -> bool {
        self.size != self.free_slots.len() as u16
    }
}
