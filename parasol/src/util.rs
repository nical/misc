
// /// A wrapper for `*const T` that is Send and Sync (use carefully).
// pub struct SyncPtr<T: ?Sized>(pub *const T);
// unsafe impl<T> Send for SyncPtr<T> {}
// unsafe impl<T> Sync for SyncPtr<T> {}
// impl<T> Copy for SyncPtr<T> {}
// impl<T> Clone for SyncPtr<T> { fn clone(&self) -> Self { *self } }
// impl<T: ?Sized> SyncPtr<T> {
//     pub unsafe fn get(&self) -> &T { &(*self.0) }
// }

// /// A wrapper for `*mut T` that is Send and Sync (use carefully).
// pub struct SyncPtrMut<T>(pub *mut T);
// unsafe impl<T> Send for SyncPtrMut<T> {}
// unsafe impl<T> Sync for SyncPtrMut<T> {}
// impl<T> Copy for SyncPtrMut<T> {}
// impl<T> Clone for SyncPtrMut<T> { fn clone(&self) -> Self { *self } }
// impl<T> SyncPtrMut<T> {
//     //unsafe fn get(&self) -> &mut T { &mut(*self.0) }
//     pub unsafe fn offset(&self, n: isize) -> &mut T { &mut(*self.0.offset(n)) }
// }
