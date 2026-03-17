use std::alloc::{AllocError, Allocator, Global, Layout};
use std::ptr::NonNull;

#[derive(Clone, Copy)]
pub struct AlignedAllocator<const ALIGN: usize>;

unsafe impl<const ALIGN: usize> Allocator for AlignedAllocator<ALIGN> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let layout = if layout.align() < ALIGN {
            let result = Layout::from_size_align(layout.size(), ALIGN);
            if result.is_err() {
                return Err(AllocError);
            }
            result.unwrap()
        } else {
            layout
        };
        Global.allocate(layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        let layout = if layout.align() < ALIGN {
            Layout::from_size_align(layout.size(), ALIGN).unwrap()
        } else {
            layout
        };
        unsafe { Global.deallocate(ptr, layout) }
    }
}

pub type AlignedVec<T, const ALIGN: usize> = Vec<T, AlignedAllocator<ALIGN>>;
