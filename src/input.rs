//! Utilities to access input formats for index building.

use std::{io, num::NonZero, ops::Index};

use stable_deref_trait::StableDeref;

/// A store of vector data indexed by a densely assigned range of values.
pub trait VectorStore: Index<usize, Output = [Self::Elem]> {
    type Elem;

    /// Return the number of vectors in the store.
    fn len(&self) -> usize;

    /// Return true if this store is empty.
    fn is_empty(&self) -> bool;

    /// Return the slice length of each row in terms of `Elem`.
    fn elem_stride(&self) -> usize;

    /// Return an iterator over all the vectors in the store.
    fn iter(&self) -> impl ExactSizeIterator<Item = &[Self::Elem]>;
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
    D: StableDeref<Target = [u8]>,
{
    /// Create a new store from byte de-refable `data` where each entry contains
    /// `stride` elements of of type `E`.
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
        let len = data.len() / (stride.get() * elem_width);

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

    fn elem_stride(&self) -> usize {
        self.stride
    }

    fn iter(&self) -> impl ExactSizeIterator<Item = &[Self::Elem]> {
        self.raw_vectors.chunks(self.stride)
    }
}

impl<E, D> Index<usize> for DerefVectorStore<E, D> {
    type Output = [E];

    fn index(&self, index: usize) -> &[E] {
        let start = index * self.stride;
        let end = start + self.stride;
        &self.raw_vectors[start..end]
    }
}
