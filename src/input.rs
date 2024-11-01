use std::{num::NonZero, ops::Index};

use stable_deref_trait::StableDeref;

/// Immutable store for numpy formatted f32 vectors.
///
/// In this format all f32 values are little endian coded and written end-to-end. The dimension
/// count must be provided externally.
pub struct NumpyF32VectorStore<D> {
    // NB: the contents of data is referenced by vectors.
    #[allow(dead_code)]
    data: D,
    dimensions: NonZero<usize>,
    vectors: &'static [f32],
}

impl<D> NumpyF32VectorStore<D> {
    /// Create a new store for numpy vector data with the given input and dimension count.
    ///
    /// This will typically be used with a memory-mapped file.
    pub fn new(data: D, dimensions: NonZero<usize>) -> Self
    where
        D: StableDeref<Target = [u8]>,
    {
        let vectorp = data.as_ptr() as *const f32;
        assert!(vectorp.is_aligned());
        assert_eq!(
            data.len() % (std::mem::size_of::<f32>() * dimensions.get()),
            0
        );
        // Safety: StableDeref guarantees the pointer is stable even after a move.
        let vectors: &'static [f32] =
            unsafe { std::slice::from_raw_parts(vectorp, data.len() / std::mem::size_of::<f32>()) };
        Self {
            data,
            dimensions,
            vectors,
        }
    }

    /// Return number of dimensions in each vector.
    pub fn dimensions(&self) -> NonZero<usize> {
        self.dimensions
    }

    /// Return the number of vectors in the store.
    pub fn len(&self) -> usize {
        self.vectors.len() / self.dimensions.get()
    }

    /// Return an iterator over all the vectors in the store.
    pub fn iter(&self) -> impl Iterator<Item = &[f32]> {
        self.vectors.chunks(self.dimensions.get())
    }
}

impl<D> Index<usize> for NumpyF32VectorStore<D> {
    type Output = [f32];

    fn index(&self, index: usize) -> &Self::Output {
        let start = index * self.dimensions.get();
        let end = start + self.dimensions.get();
        &self.vectors[start..end]
    }
}
