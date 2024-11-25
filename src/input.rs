use std::{
    io,
    num::NonZero,
    ops::{Deref, Index},
};

use stable_deref_trait::StableDeref;

/// A store of vector data indexed by a densely assigned range of values.
pub trait VectorStore: Index<usize> {
    type Elem;

    /// Return the number of vectors in the store.
    fn len(&self) -> usize;

    /// Return true if this store is empty.
    fn is_empty(&self) -> bool;

    /// Return an iterator over all the vectors in the store.
    fn iter(&self) -> impl Iterator<Item = &[Self::Elem]>;
}

pub struct DerefVectorStore<E: 'static, D> {
    // NB: the contents of data is referenced by raw_vectors.
    #[allow(dead_code)]
    data: D,
    raw_vectors: &'static [E],

    stride: usize,
    len: usize,
}

impl<E, D> DerefVectorStore<E, D>
where
    D: Deref<Target = [u8]>,
{
    pub fn new(data: D, stride: NonZero<usize>) -> io::Result<Self> {
        let elem_width = std::mem::size_of::<E>();
        let vectorp = data.as_ptr() as *const E;
        if !vectorp.is_aligned() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "input vector data not aligned to element width {}",
                    elem_width
                ),
            ));
        }
        if data.len() % (elem_width * stride.get()) != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "input vector data does not divide evenly into stride byte length of {}",
                    elem_width * stride.get()
                ),
            ));
        }
        let len = data.len() / stride.get();

        // Safety: StableDeref guarantees the pointer is stable even after a move.
        let raw_vectors: &'static [E] =
            unsafe { std::slice::from_raw_parts(vectorp, data.len() / elem_width) };
        Ok(Self {
            data,
            raw_vectors,
            stride: stride.get(),
            len,
        })
    }
}

impl<E, D> VectorStore for DerefVectorStore<E, D> {
    type Elem = E;

    fn len(&self) -> usize {
        self.len
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn iter(&self) -> impl Iterator<Item = &[Self::Elem]> {
        self.raw_vectors.chunks(self.stride)
    }
}

impl<E, D> Index<usize> for DerefVectorStore<E, D> {
    type Output = [E];

    fn index(&self, index: usize) -> &Self::Output {
        let start = index * self.stride;
        let end = start + self.stride;
        &self.raw_vectors[start..end]
    }
}

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
