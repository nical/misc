use std::{
    collections::HashMap,
    hash::Hash,
    marker::PhantomData,
    sync::{Arc, Mutex},
};

pub trait Build<Key, Payload> {
    fn build(&mut self, key: Key) -> Payload;
    fn finish(&mut self);
}

#[repr(transparent)]
#[derive(PartialEq, Eq, Hash)]
pub struct Index<Key>(u32, PhantomData<Key>);

impl<Key> Index<Key> {
    #[inline]
    pub fn index(&self) -> usize {
        self.0 as usize
    }
}

impl<Key> Copy for Index<Key> {}
impl<Key> Clone for Index<Key> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<Key> std::fmt::Debug for Index<Key> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "#{}", self.0)
    }
}

pub struct Registry<Key, Payload> {
    items: Vec<Option<Payload>>,
    built: Arc<HashMap<Key, Index<Key>>>,
    shared: Arc<Mutex<Shared<Key>>>,
}

impl<Key, Payload> Registry<Key, Payload> {
    pub fn with_capacity(cap: usize) -> Self {
        Registry {
            items: Vec::with_capacity(cap),
            built: Arc::new(HashMap::with_capacity(cap)),
            shared: Arc::new(Mutex::new(Shared {
                items: HashMap::with_capacity(cap),
                next_id: 0,
            })),
        }
    }

    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    pub fn prepare(&self) -> Prepare<Key>
    where
        Key: Clone,
    {
        Prepare {
            prev: None,
            new_items: Vec::new(),
            other_items: Vec::new(),
            built: Arc::clone(&self.built),
            shared: Arc::clone(&self.shared),
        }
    }

    pub fn get(&self, idx: Index<Key>) -> Option<&Payload> {
        self.items[idx.index()].as_ref()
    }

    pub fn look_up(&self, key: Key) -> Option<&Payload>
    where
        Key: Hash + Eq,
    {
        if let Some(index) = self.built.get(&key) {
            return self.items[index.index()].as_ref();
        }

        None
    }

    pub fn build(&mut self, changes: &[Changelist<Key>], builder: &mut dyn Build<Key, Payload>)
    where
        Key: Copy + Hash + Eq,
    {
        let mut num_items = 0;
        for c in changes {
            num_items += c.new_items.len();
        }
        self.items
            .resize_with(self.items.len() + num_items, || None);

        let built = Arc::get_mut(&mut self.built).unwrap();
        built.reserve(num_items);

        for c in changes {
            for (key, index) in &c.new_items {
                self.items[index.index()] = Some(builder.build(*key));
                built.insert(*key, *index);
            }
        }
    }
}

pub struct Prepare<Key> {
    prev: Option<(Key, Index<Key>)>,
    built: Arc<HashMap<Key, Index<Key>>>,
    new_items: Vec<(Key, Index<Key>)>,
    other_items: Vec<(Key, Index<Key>)>,
    shared: Arc<Mutex<Shared<Key>>>,
}

impl<Key: Hash + Eq + Copy> Prepare<Key> {
    pub fn prepare(&mut self, key: Key) -> Index<Key> {
        // First look for the key in the local data.
        if let Some(idx) = self.get_local(key) {
            self.prev = Some((key, idx));
            return idx;
        }

        // Finally, look and possibly insert into the mutably shared
        // list of items.
        let mut is_new = false;
        let is_new_ref = &mut is_new;
        let idx = {
            let guard = &mut *self.shared.lock().unwrap();
            let next_id = &mut guard.next_id;
            let elements = &mut guard.items;
            *elements.entry(key).or_insert_with(move || {
                let idx = Index(*next_id, PhantomData);
                *is_new_ref = true;
                *next_id += 1;
                idx
            })
        };

        if is_new {
            self.new_items.push((key, idx));
        } else {
            self.other_items.push((key, idx));
        }

        self.prev = Some((key, idx));

        idx
    }

    fn get_local(&self, key: Key) -> Option<Index<Key>> {
        if let Some((prev_key, idx)) = self.prev {
            if prev_key == key {
                return Some(idx);
            }
        }

        // First look at the items existing before this frame.
        // In the steady state this should be the most common path.
        if let Some(index) = self.built.get(&key) {
            return Some(*index);
        }

        // Then look for items created by here.
        for (k, index) in &self.new_items {
            if *k == key {
                return Some(*index);
            }
        }

        // Then look for items we have encoutered that were created
        // elsewehere. This is to avoid repeatedly hitting the shared
        // mutex with new shaders that were not created here.
        for (k, index) in &self.other_items {
            if *k == key {
                return Some(*index);
            }
        }

        None
    }

    pub fn finish(self) -> Changelist<Key> {
        Changelist {
            new_items: self.new_items,
        }
    }
}

impl<Key> Prepare<Key> {
    pub fn new_items(&self) -> &[(Key, Index<Key>)] {
        &self.new_items
    }
}

pub struct Changelist<Key> {
    new_items: Vec<(Key, Index<Key>)>,
}

struct Shared<Key> {
    items: HashMap<Key, Index<Key>>,
    next_id: u32,
}
