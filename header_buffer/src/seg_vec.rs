use crate::header_vec::RawHeaderVec;

struct Header<T> {
    next: Option<RawSegVec<T>>
}

pub struct RawSegVec<T> {
    first: Option<RawHeaderVec<Header<T>, T>>,
    last: RawHeaderVec<Header<T>, T>,
}
