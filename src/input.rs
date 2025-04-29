//! Utilities to access input formats for index building.

// TODO: rename this module

use std::{
    io,
    num::NonZero,
    ops::{Index, IndexMut, Range},
};

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

#[derive(Debug, Clone)]
pub struct VecVectorStore<E: 'static> {
    data: Vec<E>,
    elem_stride: usize,
}

impl<E: Clone> VecVectorStore<E> {
    pub fn new(elem_stride: usize) -> Self {
        Self {
            data: vec![],
            elem_stride,
        }
    }

    pub fn with_capacity(elem_stride: usize, capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(elem_stride * capacity),
            elem_stride,
        }
    }

    pub fn push(&mut self, vector: &[E]) {
        assert_eq!(vector.len(), self.elem_stride);
        self.data.extend_from_slice(vector);
    }

    pub fn capacity(&self) -> usize {
        self.data.capacity() / self.elem_stride
    }

    fn index_range(&self, index: usize) -> Range<usize> {
        let start = index * self.elem_stride;
        start..(start + self.elem_stride)
    }
}

impl<E: Clone> VectorStore for VecVectorStore<E> {
    type Elem = E;

    fn elem_stride(&self) -> usize {
        self.elem_stride
    }

    fn len(&self) -> usize {
        self.data.len() / self.elem_stride
    }

    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    fn iter(&self) -> impl ExactSizeIterator<Item = &[Self::Elem]> {
        self.data.chunks(self.elem_stride)
    }
}

impl<E: Clone> Index<usize> for VecVectorStore<E> {
    type Output = [E];

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[self.index_range(index)]
    }
}

impl<E: Clone> IndexMut<usize> for VecVectorStore<E> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let r = self.index_range(index);
        &mut self.data[r]
    }
}

pub struct SubsetViewVectorStore<'a, V> {
    parent: &'a V,
    subset: Vec<usize>,
}

impl<'a, V: VectorStore> SubsetViewVectorStore<'a, V> {
    pub fn new(parent: &'a V, subset: Vec<usize>) -> Self {
        Self { parent, subset }
    }

    pub fn original_index(&self, index: usize) -> usize {
        self.subset[index]
    }
}

impl<V: VectorStore> VectorStore for SubsetViewVectorStore<'_, V> {
    type Elem = V::Elem;

    fn elem_stride(&self) -> usize {
        self.parent.elem_stride()
    }

    fn len(&self) -> usize {
        self.subset.len()
    }

    fn is_empty(&self) -> bool {
        self.subset.is_empty()
    }

    fn iter(&self) -> impl ExactSizeIterator<Item = &[Self::Elem]> {
        self.subset.iter().map(|i| &self.parent[*i])
    }
}

impl<V: VectorStore> Index<usize> for SubsetViewVectorStore<'_, V> {
    type Output = [V::Elem];

    fn index(&self, index: usize) -> &Self::Output {
        &self.parent[self.subset[index]]
    }
}
