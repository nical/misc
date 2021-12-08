//! An experimental parallel job scheduler with the goal of doing better than rayon
//! specifically in the types of workloads we have in Firefox.
//!
//! What we want:
//! - Allow running jobs outside of the thread pool.
//! - Avoid blocking the thread that submits the work if possible.
//! - No implicit global thread pool.
//! - Ways to safely manage per-worker data.
//! - Avoid hoarding CPU resources in worker threads that don't have work to execute (this
//!   is at the cost of higher latency).
//! - No need to scale to a very large number of threads. We prefer to have something that
//!   runs efficiently on up to 8 threads and not need to scale well above that.


// TODO: handle panics in worker threads
// TODO: everywhere we take a context as parameter there should be a check that the context
//       belongs to the same thread pool.

mod core;
mod job;
mod context;
mod sync;
mod array;
mod util;
mod thread_pool;
mod shutdown;

pub use job::Priority;
pub use context::*;
pub use sync::SyncPoint;
pub use array::{ForEach, workload, range_workload, Workload, RunningWorkload, new_for_each};
pub use thread_pool::{ThreadPool, ThreadPoolId, ThreadPoolBuilder};
pub use shutdown::ShutdownHandle;
pub use util::ExclusiveCheck;

pub use crossbeam_utils::CachePadded;
