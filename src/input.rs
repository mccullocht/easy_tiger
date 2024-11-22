use std::{io, num::NonZero, ops::Index};

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

impl<D> NumpyF32VectorStore<D>
where
    D: Send + Sync,
{
    /// Create a new store for numpy vector data with the given input and dimension count.
    ///
    /// This will typically be used with a memory-mapped file.
    pub fn new(data: D, dimensions: NonZero<usize>) -> io::Result<Self>
    where
        D: StableDeref<Target = [u8]>,
    {
        let vectorp = data.as_ptr() as *const f32;
        if !vectorp.is_aligned() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "input vector data not aligned to f32".to_string(),
            ));
        }
        if data.len() % (std::mem::size_of::<f32>() * dimensions.get()) != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "input vector data does not divide evenly into stride length of {}",
                    std::mem::size_of::<f32>() * dimensions.get()
                ),
            ));
        }

        // Safety: StableDeref guarantees the pointer is stable even after a move.
        let vectors: &'static [f32] =
            unsafe { std::slice::from_raw_parts(vectorp, data.len() / std::mem::size_of::<f32>()) };
        Ok(Self {
            data,
            dimensions,
            vectors,
        })
    }

    /// Return number of dimensions in each vector.
    pub fn dimensions(&self) -> NonZero<usize> {
        self.dimensions
    }

    /// Return the number of vectors in the store.
    pub fn len(&self) -> usize {
        self.vectors.len() / self.dimensions.get()
    }

    /// Return true if this store is empty.
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
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
