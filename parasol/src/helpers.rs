//! Helpers to build high level parallel execution primitives on top of the core.
//!
//! 

use crate::array::{ForEach, new_for_each};
use crate::join::{Join, new_join};
use crate::{Context, Priority};

use std::ops::Deref;
use std::sync::Arc;


/// A builder for common execution parameters such as priority and context data.
pub struct Parameters<'c, 'cd, 'id, ContextData, ImmutableData> {
    pub(crate) ctx: &'c mut Context,
    pub(crate) context_data: Option<&'cd mut [ContextData]>,
    pub(crate) immutable_data: Option<&'id ImmutableData>,
    pub(crate) priority: Priority,
}

impl<'c, 'cd, 'id, ContextData, ImmutableData> Parameters<'c, 'cd, 'id, ContextData, ImmutableData> {
    /// Specify some per-context data that can be mutably accessed by the run function.
    ///
    /// This can be useful to store and reuse some scratch buffers and avoid memory allocations in the
    /// run function.
    ///
    /// The length of the slice must be at least equal to the number of worker threads plus one.
    ///
    /// For best performance make sure the size of the data is a multiple of L1 cache line size (see `CachePadded`).
    #[inline]
    pub fn with_context_data<'cd2, CtxData: Send>(self, context_data: &'cd2 mut [CtxData]) -> Parameters<'c, 'cd2, 'id, CtxData, ImmutableData> {
        // Note: doing this check here is important for the safety of ContextDataRef::get
        assert!(
            context_data.len() >= self.ctx.num_worker_threads() as usize + 1,
            "Got {:?} context items, need at least {:?}",
            context_data.len(), self.ctx.num_worker_threads() + 1,
        );

        Parameters {
            context_data: Some(context_data),
            immutable_data: self.immutable_data,
            priority: self.priority,
            ctx: self.ctx
        }
    }

    #[inline]
    pub fn with_immutable_data<'id2, Data>(self, immutable_data: &'id2 Data) -> Parameters<'c, 'cd, 'id2, ContextData, Data> {
        Parameters {
            context_data: self.context_data,
            immutable_data: Some(immutable_data),
            priority: self.priority,
            ctx: self.ctx
        }
    }

    #[inline]
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;

        self
    }

    #[inline]
    pub fn priority(&self) -> Priority {
        self.priority
    }

    #[inline]
    pub fn context(&self) -> &Context {
        &self.ctx
    }

    #[inline]
    pub fn context_mut(&mut self) -> &mut Context {
        &mut self.ctx
    }

    #[inline]
    pub fn join<F1, F2>(self, f1: F1, f2: F2) -> Join<'c, 'cd, 'id, ContextData, ImmutableData, F1, F2>
    where
        F1: FnOnce(&mut Context, crate::join::Args<ContextData, ImmutableData>) + Send,
        F2: FnOnce(&mut Context, crate::join::Args<ContextData, ImmutableData>) + Send,
    {
        new_join(self, f1, f2)
    }

    #[inline]
    pub fn for_each<'i, Item>(self, items: &'i mut[Item]) -> ForEach<'i, 'cd, 'id, 'c, Item, ContextData, ImmutableData, ()> {
        new_for_each(self, items)
    }
}

/// Similar to `Parameters`, but owns its data.and does not hold a reference to the context.
pub struct OwnedParameters<ContextData, ImmutableData> {
    immutable_data: Option<Arc<ImmutableData>>,
    context_data: Option<Vec<ContextData>>,
    priority: Priority,
}

pub fn owned_parameters() -> OwnedParameters<(), ()> {
    OwnedParameters {
        immutable_data: None,
        context_data: None,
        priority: Priority::High,
    }
}

impl<ContextData, ImmutableData> OwnedParameters<ContextData, ImmutableData> {
    #[inline]
    pub fn with_context_data<T>(self, data: Vec<T>) -> OwnedParameters<T, ImmutableData> {
        OwnedParameters {
            context_data: Some(data),
            immutable_data: self.immutable_data,
            priority: self.priority,
        }
    }

    #[inline]
    pub fn with_immutable_data<T>(self, data: Arc<T>) -> OwnedParameters<ContextData, T> {
        OwnedParameters {
            context_data: self.context_data,
            immutable_data: Some(data),
            priority: self.priority,
        }
    }

    #[inline]
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;

        self
    }

    #[inline]
    pub fn priority(&self) -> Priority {
        self.priority
    }

    #[inline]
    pub fn context_data(&mut self) -> Option<&mut [ContextData]> {
        self.context_data.as_mut().map(|data| &mut data[..])        
    }

    #[inline]
    pub fn take(&mut self) -> Self {
        OwnedParameters {
            immutable_data: self.immutable_data.take(),
            context_data: self.context_data.take(),
            priority: self.priority,
        }
    }
}

/// Erases the lifetime of references to the context data and immutable data.
///
/// This does now own/destroys the referenced data, it is on you to ensure that
/// the data outlives the ContextDataRef.
///
/// Used internally by various job implementations.
pub struct ContextDataRef<ContextData, ImmutableData> {
    ctx_data: *mut ContextData,
    immutable_data: *const ImmutableData,    
}

impl<ContextData, ImmutableData> ContextDataRef<ContextData, ImmutableData> {
    pub unsafe fn get(&self, ctx: &Context) -> (&mut ContextData, &ImmutableData) {
        let context_data_index = ctx.data_index() as isize;
        (
            // SAFETY: Here we rely two very important things:
            // - If there is no context data, then it's type is `()`, which means reads and writes
            //   to the pointer are ALWAYS noop whatever the address of the pointer.
            // - If a context data array was provided, its size has been checked in `with_context_data`.
            //
            // As a result it is impossible to craft a pointer that will read or write out of bounds
            // here.
            &mut *self.ctx_data.wrapping_offset(context_data_index),
            &*self.immutable_data,
        )
    }

    /// Returns unsafe references to the context data and immutable data.
    ///
    /// The caller is responsible for ensuring that the context data and immutable data
    /// outlines the unsafe ref.
    #[inline]
    pub unsafe fn from_ref<'c, 'cd, 'id>(parameters: &mut Parameters<'c, 'cd, 'id, ContextData, ImmutableData>) -> ContextDataRef<ContextData, ImmutableData> {
        ContextDataRef {
            ctx_data: parameters.context_data.as_mut().map_or(std::ptr::null_mut(), |arr| arr.as_mut_ptr()),
            immutable_data: parameters.immutable_data.map_or(std::ptr::null_mut(), |ptr| ptr),
        }
    }


    /// Returns unsafe references to the context data and immutable data.
    ///
    /// The caller is responsible for ensuring that the context data and immutable data
    /// outlines the unsafe ref.
    #[inline]
    pub unsafe fn from_owned(parameters: &mut OwnedParameters<ContextData, ImmutableData>, ctx: &Context) -> ContextDataRef<ContextData, ImmutableData> {

        let immutable_data = parameters.immutable_data
            .as_ref()
            .map_or(std::ptr::null(), |boxed| &*boxed.deref());

        let ctx_data = parameters.context_data
            .as_mut()
            .map_or(std::ptr::null_mut(), |array| {
                assert!(
                    array.len() >= ctx.num_worker_threads() as usize + 1,
                    "Got {:?} context items, need at least {:?}",
                    array.len(), ctx.num_worker_threads() + 1,
                );
                array.as_mut_ptr()
            });


        ContextDataRef {
            ctx_data,
            immutable_data,
        }
    }
}

