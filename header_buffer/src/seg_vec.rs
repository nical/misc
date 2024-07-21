use crate::header_vec::UnmanagedHeaderVec;

struct Header<T> {
    next: Option<UnmanagedSegVec<T>>
}

pub struct UnmanagedSegVec<T> {
    first: Option<UnmanagedHeaderVec<Header<T>, T>>,
    last: UnmanagedHeaderVec<Header<T>, T>,
}
