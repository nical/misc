use std::marker::PhantomData;
use std::hash::Hash;
use std::fmt;
use std::collections::HashMap;

pub trait Interned : Clone {
    type Key: Hash + Eq;
    type RenderData;

    fn key(&self) -> Self::Key;
    fn on_remove(_item: &Option<Self>, _render_data: &Option<Self::RenderData>) {}
}

pub struct Handle<T> {
    index: u16,
    gen: u16,
    _marker: PhantomData<T>,
}

impl<T> Handle<T> {
    pub fn new(index: u16, gen: u16) -> Self {
        Handle {
            index,
            gen,
            _marker: PhantomData,
        }
    }

    fn index(&self) -> usize {
        self.index as usize
    }

    fn next_gen(self) -> Self {
        Handle::new(self.index, self.gen.wrapping_add(1))
    }
}

impl<T> Copy for Handle<T> {}

impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self { *self }
}

impl<T> PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.gen == other.gen
    }
}

impl<T> fmt::Debug for Handle<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "#{}({})", self.index, self.gen)
    }
}

pub struct RetainedHandle<T> {
    handle: Handle<T>,
}

impl<T> Copy for RetainedHandle<T> {}
impl<T> Clone for RetainedHandle<T> {
    fn clone(&self) -> Self { *self }
}
impl<T> PartialEq for RetainedHandle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle
    }
}

impl<T> Into<Handle<T>> for RetainedHandle<T> {
    fn into(self) -> Handle<T> { self.handle }
}

impl<T> fmt::Debug for RetainedHandle<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        self.handle.fmt(f)
    }
}

impl<T: Interned> RetainedHandle<T> {
    pub fn to_handle(&self) -> Handle<T> { self.handle }
}

pub struct DataStore<T: Interned> {
    items: Vec<Option<T>>,
    render_data: Vec<Option<T::RenderData>>
}

impl <T: Interned> DataStore<T> {
    pub fn new() -> Self {
        DataStore {
            items: Vec::new(),
            render_data: Vec::new(),
        }
    }

    pub fn set(&mut self, handle: Handle<T>, val: T) {
        self.ensure_size(handle);
        self.items[handle.index()] = Some(val);
        self.render_data[handle.index()] = None;
    }

    pub fn update(&mut self, handle: Handle<T>, val: T) {
        self.items[handle.index()] = Some(val);
        self.render_data[handle.index()] = None;
    }

    pub fn remove(&mut self, handle: Handle<T>) {
        T::on_remove(&self.items[handle.index()], &self.render_data[handle.index()]);
        self.items[handle.index()] = None;
        self.render_data[handle.index()] = None;
    }

    fn ensure_size(&mut self, handle: Handle<T>) {
        if self.items.len() > handle.index as usize {
            return;
        }
        let diff = handle.index as usize + 1 - self.items.len();
        self.items.reserve(diff);
        self.render_data.reserve(diff);
        for _ in 0..diff {
            self.items.push(None);
            self.render_data.push(None);
        }
    }

    pub fn apply_updates(&mut self, updates: &mut Updates<T>) {
        for (handle, value) in updates.items.drain(..) {
            self.set(handle, value);
        }
    }

    pub fn get(&self, handle: Handle<T>) -> Option<&T> {
        self.items[handle.index()].as_ref()
    }
}

type FrameId = usize;

pub struct Interner<T: Interned> {
    auto_items: HashMap<T::Key, (Handle<T>, FrameId)>,
    handles: HandleAllocator<T>,
}

impl<T: Interned> Interner<T> {
    pub fn new() -> Self {
        Interner {
            auto_items: HashMap::new(),
            handles: HandleAllocator::new(),
        }
    }

    pub fn get_handle(&mut self, val: &T) -> Handle<T> {
        let num_items = &mut self.handles.num_items;
        let updates = &mut self.handles.updates;
        let free_handles = &mut self.handles.free_handles;
        let item = self.auto_items.entry(val.key()).or_insert_with(|| {
            let handle = free_handles.pop().unwrap_or_else(|| {
                let index = *num_items;
                *num_items += 1;
                Handle::new(index, 1)
            });

            updates.push((handle, val.clone()));

            (handle, 1)
        });


        item.1 = self.handles.current_epoch;

        item.0
    }

    pub fn add_retained_item(&mut self, val: &T) -> RetainedHandle<T> {
        self.handles.add(val)
    }

    pub fn update_retained_item(&mut self, handle: RetainedHandle<T>, val: &T) {
        self.handles.update(handle, val);
    }

    pub fn remove_retained_item(&mut self, handle: RetainedHandle<T>) {
        self.handles.remove(handle);
    }

    pub fn end_frame(&mut self) -> Updates<T> {
        let current_epoch = self.handles.current_epoch;
        let free_handles = &mut self.handles.free_handles;
        self.auto_items.retain(|_key, (handle, epoch)| {
            if *epoch < current_epoch {
                free_handles.push(handle.next_gen());
                return false;
            }

            true
        });

        self.handles.end_frame()
    }
}

pub struct HandleAllocator<T> {
    free_handles: Vec<Handle<T>>,
    updates: Vec<(Handle<T>, T)>,
    num_items: u16,
    current_epoch: FrameId,
}

impl<T: Interned> HandleAllocator<T> {
    pub fn new() -> Self {
        HandleAllocator {
            free_handles: Vec::new(),
            updates: Vec::new(),
            num_items: 0,
            current_epoch: 1,
        }
    }

    pub fn add(&mut self, val: &T) -> RetainedHandle<T> {
        let num_items = &mut self.num_items;
        let handle = self.free_handles.pop().unwrap_or_else(|| {
            let index = *num_items;
            *num_items += 1;
            Handle::new(index, 1)
        });

        self.updates.push((handle, val.clone()));

        RetainedHandle { handle }
    }

    pub fn update(&mut self, handle: RetainedHandle<T>, val: &T) {
        self.updates.push((handle.to_handle(), val.clone()));        
    }

    pub fn remove(&mut self, handle: RetainedHandle<T>) {
        self.free_handles.push(handle.handle.next_gen());        
    }

    pub fn end_frame(&mut self) -> Updates<T> {
        self.current_epoch += 1;

        Updates {
            items: std::mem::take(&mut self.updates)
        }
    }
}

pub struct Updates<T: Interned> {
    items: Vec<(Handle<T>, T)>
}

#[test]
fn interner() {
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    struct ColorPattern {
        r: u8,
        g: u8,
        b: u8,
        a: u8,
    }

    impl Interned for ColorPattern {
        type Key = ColorPattern;
        type RenderData = ();
        fn key(&self) -> ColorPattern { self.clone() }
    }

    let mut colors = Interner::new();
    let mut store = DataStore::new();

    let c1 = colors.get_handle(&ColorPattern { r: 0, g: 0, b: 0, a: 255 });
    let c2 = colors.get_handle(&ColorPattern { r: 255, g: 0, b: 0, a: 255 });
    let c3 = colors.get_handle(&ColorPattern { r: 0, g: 255, b: 0, a: 255 });
    let c4 = colors.get_handle(&ColorPattern { r: 255, g: 0, b: 0, a: 255 });

    let r1 = colors.add_retained_item(&ColorPattern { r: 0, g: 0, b: 0, a: 255 });

    assert_eq!(c2, c4);
    assert!(c1 != c2);
    assert!(c1 != c3);

    let mut updates = colors.end_frame();
    assert_eq!(updates.items.len(), 4);

    store.apply_updates(&mut updates);

    assert_eq!(store.get(c1), Some(&ColorPattern { r: 0, g: 0, b: 0, a: 255 }));
    assert_eq!(store.get(c2), Some(&ColorPattern { r: 255, g: 0, b: 0, a: 255 }));
    assert_eq!(store.get(c3), Some(&ColorPattern { r: 0, g: 255, b: 0, a: 255 }));
    assert_eq!(store.get(r1.to_handle()), Some(&ColorPattern { r: 0, g: 0, b: 0, a: 255 }));

    let c5 = colors.get_handle(&ColorPattern { r: 255, g: 0, b: 0, a: 255 });
    let c6 = colors.get_handle(&ColorPattern { r: 0, g: 255, b: 0, a: 255 });
    let c7 = colors.get_handle(&ColorPattern { r: 255, g: 255, b: 255, a: 255 });

    assert_eq!(c5, c4);
    assert_eq!(c6, c3);

    let mut updates = colors.end_frame();
    assert_eq!(updates.items.len(), 1);

    store.apply_updates(&mut updates);

    assert_eq!(store.get(c5), Some(&ColorPattern { r: 255, g: 0, b: 0, a: 255 }));
    assert_eq!(store.get(c6), Some(&ColorPattern { r: 0, g: 255, b: 0, a: 255 }));
    assert_eq!(store.get(c7), Some(&ColorPattern { r: 255, g: 255, b: 255, a: 255 }));
    assert_eq!(store.get(r1.to_handle()), Some(&ColorPattern { r: 0, g: 0, b: 0, a: 255 }));

    let c8 = colors.get_handle(&ColorPattern { r: 0, g: 0, b: 0, a: 255 });

    colors.update_retained_item(r1, &ColorPattern { r: 1, g: 0, b: 0, a: 255 });

    assert!(c8 != c1);

    let mut updates = colors.end_frame();

    assert_eq!(updates.items.len(), 2);

    store.apply_updates(&mut updates);

    assert_eq!(store.get(c8), Some(&ColorPattern { r: 0, g: 0, b: 0, a: 255 }));
    assert_eq!(store.get(r1.to_handle()), Some(&ColorPattern { r: 1, g: 0, b: 0, a: 255 }));
}
