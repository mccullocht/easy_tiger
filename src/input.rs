//! Utilities to access input formats for index building.

// TODO: rename this module

use std::{
    fs::File,
    io,
    num::NonZero,
    ops::{Index, IndexMut, Range},
    path::Path,
};

use memmap2::Mmap;
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
                format!("input vector data not aligned to element width {elem_width}"),
            ));
        }
        if !data.len().is_multiple_of(elem_width * stride.get()) {
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

    /// Create a new store from a de-refable `data` in BigANN format where the file begins with two
    /// 32-bit integers: <len, dim>.
    pub fn new_bigann(data: D) -> io::Result<Self> {
        if data.len() < 8 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "input vector data not long enough; must be at least 8 bytes".to_string(),
            ));
        }
        let (header, vectors) = data.split_at(8);
        let header_parts = header.as_chunks::<4>().0;
        let len = u32::from_le_bytes(header_parts[0]) as usize;
        let dim = u32::from_le_bytes(header_parts[1]) as usize;
        let stride = dim * std::mem::size_of::<E>();
        if len * stride != vectors.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("input vector data must have exactly {len} entries of {stride} length"),
            ));
        }

        // Safety: StableDeref guarantees the pointer is stable even after a move.
        let raw_vectors: &'static [E] = unsafe {
            std::slice::from_raw_parts(vectors.as_ptr() as *const E, vectors.len() / stride)
        };
        Ok(Self {
            data,
            raw_vectors,
            stride,
            len,
        })
    }

    pub fn data(&self) -> &D {
        &self.data
    }
}

impl<E> DerefVectorStore<E, Mmap> {
    /// Create a new store by mmapping the file at `path`, where each entry contains
    /// `stride` elements of type `E`.
    pub fn from_file(path: impl AsRef<Path>, stride: NonZero<usize>) -> io::Result<Self> {
        let mmap = unsafe { Mmap::map(&File::open(path)?)? };
        Self::new(mmap, stride)
    }

    /// Create a new store by mapping the file at `path` an interpreting as a BigANN benchmark
    /// input that begins with a <len,dim> header. The length of each vector is interpreted based
    /// on the size of `E`, so this cannot be used for sub-byte inputs.
    pub fn from_bigann_file(path: impl AsRef<Path>) -> io::Result<Self> {
        let mmap = unsafe { Mmap::map(&File::open(path)?)? };
        Self::new_bigann(mmap)
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

pub struct CompositeVectorStore<S> {
    children: Vec<S>,
    len: usize,
}

impl<S: VectorStore> CompositeVectorStore<S> {
    /// Create a new composite store from a list of children.
    ///
    /// May return `None` if children are empty or do not have the same elem stride.
    pub fn from_children(children: Vec<S>) -> Option<Self> {
        if children.is_empty() {
            return None;
        }
        if children
            .iter()
            .any(|s| s.elem_stride() != children[0].elem_stride())
        {
            return None;
        }
        let len = children.iter().map(VectorStore::len).sum::<usize>();
        Some(Self { children, len })
    }
}

impl<S: VectorStore> VectorStore for CompositeVectorStore<S> {
    type Elem = S::Elem;

    fn len(&self) -> usize {
        self.len
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn elem_stride(&self) -> usize {
        self.children[0].elem_stride()
    }

    fn iter(&self) -> impl ExactSizeIterator<Item = &[Self::Elem]> {
        CompositeVectorStoreIter {
            iter: self.children.iter().flat_map(|s| s.iter()),
            remaining: self.len,
        }
    }
}

impl<S: VectorStore> Index<usize> for CompositeVectorStore<S> {
    type Output = [S::Elem];

    fn index(&self, index: usize) -> &Self::Output {
        let mut base = 0usize;
        for c in self.children.iter() {
            if index < base + c.len() {
                return &c[index - base];
            } else {
                base += c.len();
            }
        }
        panic!("index {index} out of bounds");
    }
}

pub struct CompositeVectorStoreIter<I> {
    iter: I,
    remaining: usize,
}

impl<I: Iterator> Iterator for CompositeVectorStoreIter<I> {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.iter.next();
        if item.is_some() {
            self.remaining -= 1;
        }
        item
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<I: Iterator> ExactSizeIterator for CompositeVectorStoreIter<I> {}

#[derive(Debug, Clone)]
pub struct VecVectorStore<E: 'static> {
    data: Vec<E>,
    elem_stride: usize,
}

impl<E: Copy> VecVectorStore<E> {
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

    pub fn clear(&mut self) {
        self.data.clear();
    }

    pub fn capacity(&self) -> usize {
        self.data.capacity() / self.elem_stride
    }

    pub fn truncate(&mut self, len: usize) {
        self.data.truncate(len * self.elem_stride);
    }

    pub fn swap_remove(&mut self, index: usize) {
        let src = self.index_range(self.len() - 1);
        let dst = self.index_range(index).start;
        self.data.copy_within(src, dst);
        self.truncate(self.len() - 1);
    }

    pub fn iter_mut(&mut self) -> impl ExactSizeIterator<Item = &mut [E]> {
        self.data.chunks_mut(self.elem_stride)
    }

    fn index_range(&self, index: usize) -> Range<usize> {
        let start = index * self.elem_stride;
        start..(start + self.elem_stride)
    }
}

impl<E: Copy> VectorStore for VecVectorStore<E> {
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

impl<E: Copy> Index<usize> for VecVectorStore<E> {
    type Output = [E];

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[self.index_range(index)]
    }
}

impl<E: Copy> IndexMut<usize> for VecVectorStore<E> {
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

    pub fn parent(&self) -> &V {
        self.parent
    }

    /// Extract the subset passed to `new()``.
    pub fn into_subset(self) -> Vec<usize> {
        self.subset
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
