use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};

pub type LayoutId = u16;
pub type ResourceId = u16;
pub type BindGroupId = u16;

// TODO: use actual wgpu types.
#[derive(Clone)]
pub struct BindGroup;
pub struct Device;

const MAX_RESOURCES_PER_BINDGROUP: usize = 7;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BindGroupKey<'l>(LayoutId, &'l[ResourceId]);
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct BindGroupKeyImpl(LayoutId, [ResourceId; MAX_RESOURCES_PER_BINDGROUP]);

pub type BindGroupUpdates = HashMap<BindGroupKeyImpl, BindGroupId>;

// TODO: can implement Borrow to avoid some manual conversion.
impl<'l> BindGroupKey<'l> {
    fn owned(&self) -> BindGroupKeyImpl {
        let mut key = BindGroupKeyImpl(self.0, [0; MAX_RESOURCES_PER_BINDGROUP]);
        key.1.copy_from_slice(self.1);

        key
    }
}

/// A cache for going from lists of resources to bind groups.
///
/// Using the cache typically happens in three phases:
///  - get a cache ref (for each thread) and prepare drawing commands, getting an ID for
///    the bind group lists. Missing bind groups will generate an ID and be recorded in
///    an update list.
///  - On a single thread, turn all refs into update lists and apply them to the cache.
///    This allcates the missing bind groups and requires exclusive access to the cache
///    and device.
///  - While build draw calls, use the cache to query actual bind groups from the IDs
///    generated earlier.
///
/// This approach allows the early parts of the frame to run on multiple threads and not
/// require access to the device. It also avoid carrying the heavy-weight keys through
/// the entire frame.
pub struct BindGroupCache {
    bind_group_table: HashMap<BindGroupKeyImpl, BindGroupId>,
    bind_groups: Vec<Option<BindGroup>>,
    next_id: AtomicU32,
    // TODO: garbage collect unused bindgroups.
}

impl BindGroupCache {
    pub fn new() -> Self {
        BindGroupCache {
            bind_group_table: HashMap::default(),
            bind_groups: Vec::new(),
            next_id: AtomicU32::new(0),
        }
    }

    pub fn try_get_id(&self, key: BindGroupKey) -> Option<&BindGroupId> {
        let key = key.owned();
        self.bind_group_table.get(&key)
    }

    // Call at the end of the frame after applying updates. Expects the bindgroup to be created.
    pub fn get_bind_group(&self, id: BindGroupId) -> &BindGroup {
        self.bind_groups[id as usize].as_ref().unwrap()
    }

    pub fn add_bind_group(&mut self, key: BindGroupKey, bind_group: BindGroup) -> BindGroupId {
        let index = self.next_id.fetch_add(1, Ordering::Relaxed) as usize;
        while self.bind_groups.len() <= index {
            self.bind_groups.push(None);            
        }

        self.bind_groups[index] = Some(bind_group);

        let id = index as BindGroupId;
        let key = key.owned();
        self.bind_group_table.insert(key, id);

        id
    }

    pub fn new_ref(&self) -> BindGroupCacheRef {
        BindGroupCacheRef {
            cache: self,
            updates: HashMap::default(),
        }
    }

    pub fn apply_updates(&mut self, updates: BindGroupUpdates, _device: &mut Device) {
        for (key, id) in updates {
            let entry = self.bind_group_table.entry(key);

            if let std::collections::hash_map::Entry::Occupied(entry) = &entry {
                let bind_group = self.bind_groups[*entry.get() as usize].clone();
                debug_assert!(bind_group.is_some());
                self.bind_groups[id as usize] = bind_group;
            }

            entry.or_insert_with(|| {
                // TODO: actually use the device to create the bindgroup
                let bind_group = BindGroup;
                self.bind_groups[id as usize] = Some(bind_group);
                id
            });
        }
    }
}

pub struct BindGroupCacheRef<'l> {
    cache: &'l BindGroupCache,
    updates: HashMap<BindGroupKeyImpl, BindGroupId>,
}

impl<'l> BindGroupCacheRef<'l> {
    pub fn get_id(&mut self, key: BindGroupKey) -> BindGroupId {
        self.cache.try_get_id(key).cloned().unwrap_or_else(|| {
            let key = key.owned();
            let entry = self.updates.entry(key).or_insert_with(&|| {
                let id = self.cache.next_id.fetch_add(1, Ordering::Relaxed);
                id as BindGroupId
            });

            *entry
        })
    }

    pub fn finish(self) -> BindGroupUpdates {
        self.updates
    }
}
