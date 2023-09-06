use core::Point;
use std::{sync::Arc, ops::Deref};
use lyon::path::commands::PathCommands;

use shared_vector::SharedVector;

pub type Update<State, Event> = fn(State, Event) -> State;
pub type Importance = u8;

pub struct Undoable<Event> {
    pub event: Event,
    pub undo: Option<Event>,
}

impl<T> From<T> for Undoable<T> {
    fn from(event: T) -> Undoable<T> {
        Undoable { event, undo: None }
    }
}

pub enum UndoItem<T, E> {
    State(T),
    Events { undo: Vec<E>, redo: Vec<E> },
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Epoch(u32);

pub struct StoreOptions {
    pub max_undo_depth: u32,
}

impl Default for StoreOptions {
    fn default() -> Self {
        StoreOptions {
            max_undo_depth: 128,
        }
    }
}

#[derive(Clone, Debug)]
struct State<T> {
    data: T,
    epoch: Epoch,
    importance: Importance,
}

#[derive(Clone, Debug)]
pub struct Store<T, E> {
    state: Option<State<T>>,
    update: Update<T, E>,

    undo: UndoRedo<T>,
}


impl<T: Clone, E> Store<T, E> {
    pub fn new(initial: T, update: Update<T, E>, options: StoreOptions) -> Self {
        Store {
            state: Some(State {
                data: initial,
                epoch: Epoch(0),
                importance: 0,
            }),
            update,
            undo: UndoRedo::new(options.max_undo_depth)
        }
    }

    pub fn apply(&mut self, event: impl Into<Undoable<E>>) {
        let epoch = self.undo.before_update(&self.state);
        
        self.state = Some(State {
            data: (self.update)(
                self.state.take().unwrap().data,
                event.into().event,
            ),
            epoch,
            importance: 0,
        });
    }

    pub fn read(&self) -> &T {
        &self.current().data
    }

    pub fn snapshot(&mut self, importance: Importance) -> Epoch {
        self.undo.snapshot = true;
        let state = self.state.as_mut().unwrap();
        state.importance = state.importance.max(importance);
        self.current().epoch
    }

    pub fn undo(&mut self) -> bool {
        self.undo.undo(&mut self.state)
    }

    pub fn redo(&mut self) -> bool {
        self.undo.redo(&mut self.state)
    }

    pub fn drop_snapshots(&mut self, depth: usize, importance: Importance) {
        self.undo.drop_snapshots(depth, importance);
    }

    fn current(&self) -> &State<T> {
        self.state.as_ref().unwrap()
    }
}

impl<T: Clone, E> Deref for Store<T, E> {
    type Target = T;
    fn deref(&self) -> &T {
        self.read()
    }
}

#[derive(Clone, Debug)]
struct UndoRedo<T> {
    undo: Vec<State<T>>,
    redo: Vec<State<T>>,
    snapshot: bool,
    next_epoch: Epoch,
}

impl<T: Clone> UndoRedo<T> {
    fn new(max_undo_depth: u32) -> Self {
        let cap = max_undo_depth.min(1024) as usize;
        UndoRedo {
            undo: Vec::with_capacity(cap),
            redo: Vec::with_capacity(cap),
            snapshot: false,
            next_epoch: Epoch(1),
        }
    }

    fn before_update(&mut self, state: &Option<State<T>>) -> Epoch {
        if self.snapshot {
            self.snapshot = false;
            self.undo.push(state.clone().unwrap());
        }
        let epoch = self.next_epoch;
        self.next_epoch.0 += 1;

        epoch
    }

    fn undo(&mut self, current: &mut Option<State<T>>) -> bool {
        if let Some(state) = self.undo.pop() {
            self.redo.push(current.take().unwrap());
            *current = Some(state);
            self.snapshot = false;
            return true;
        }

        false
    }

    fn redo(&mut self, current: &mut Option<State<T>>) -> bool {
        if let Some(state) = self.redo.pop() {
            self.undo.push(current.take().unwrap());
            *current = Some(state);
            self.snapshot = false;
            return true;
        }

        false
    }

    fn drop_snapshots(&mut self, depth: usize, importance: Importance) {
        let mut i = 0;
        while i + depth < self.undo.len() {
            if self.undo[i].importance < importance {
                self.undo.remove(i);
            } else {
                i += 1;
            }
        }
    }
}


#[test]
fn simple_store() {
    #[derive(Copy, Clone, Debug)]
    pub enum Event { Inc, Dec }

    fn update(state: i32, event: Event) -> i32 {
        match event {
            Event::Inc => state + 1,
            Event::Dec => state - 1,
        }
    }

    let mut store = Store::new(0, update, StoreOptions::default());
    store.snapshot(0);

    assert_eq!(*store.read(), 0);

    store.apply(Event::Inc);
    store.snapshot(0);

    assert_eq!(*store.read(), 1);

    store.apply(Event::Dec);
    store.snapshot(0);

    assert_eq!(*store.read(), 0);

    store.undo();

    assert_eq!(*store.read(), 1);

    store.undo();

    assert_eq!(*store.read(), 0);

    store.redo();

    assert_eq!(*store.read(), 1);
}



#[derive(Clone, Debug)]
pub struct Path {
    pub commands: PathCommands,
}

#[derive(Clone, Debug)]
pub struct Layer {
    pub name: String,
    pub items: SharedVector<Item>,
}

#[derive(Clone, Debug)]
pub struct Scene {
    pub endpoints: SharedVector<Point>,
    pub ctrl_points: SharedVector<Point>,
    pub layers: SharedVector<Arc<Layer>>,
    pub paths: SharedVector<Arc<PathCommands>>
}

#[derive(Clone, Debug)]
pub enum Item {
    Path(u32),
    Scene(Arc<Scene>),
}

#[derive(Clone, Debug)]
pub struct Document {
    pub name: String,
    pub epoch: u64,
    // TODO: these should be something more akin to shared hash maps.
    pub scenes: SharedVector<Arc<Scene>>,
}
