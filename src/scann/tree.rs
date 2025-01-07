use std::{num::NonZero, ops::Index};

use crate::input::{DerefVectorStore, VectorStore};
use rand::prelude::*;

use super::kmeans;

/// Parameters for training a kmeans tree.
pub struct Params {
    /// Maximum number of levels in the tree.
    ///
    /// Google recommends a 2-level tree if there are < 10M vectors and a 3-level index for > 100M.
    /// In general more levels optimizes indexing time; fewer levels optimizes recall.
    /// See: https://cloud.google.com/alloydb/docs/ai/tune-indexes?resource=scann#tune-scann-indexes
    pub num_levels: usize,
    /// Maximum number of vectors in a single leaf node.
    pub max_leaf_size: usize,

    /// Parameters for KMeans computations.
    pub kmeans_params: kmeans::Params,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            num_levels: 2,
            max_leaf_size: 1,
            kmeans_params: kmeans::Params::default(),
        }
    }
}

/// A node in the K-Means tree.
pub enum TreeNode {
    /// A parent node contains (vector, child node) tuples.
    Parent {
        centers: DerefVectorStore<f32, Vec<f32>>,
        children: Vec<TreeNode>,
    },
    /// A leaf node is empty in the tree and may be used as a place to store vectors in a ScaNN index.
    Leaf,
}

impl TreeNode {
    /// Train a new tree from an input `VectorStore` with `k_per_node` children at each parent node
    /// and the given training params.
    pub fn train<V: VectorStore<Elem = f32> + Send + Sync>(
        training_data: &V,
        k_per_node: NonZero<usize>,
        params: &Params,
    ) -> Self {
        Self::train_node(
            &SubsetViewVectorStore::identity(training_data),
            k_per_node,
            params,
            0,
        )
    }

    /// Iterate over the centers and child nodes of this `TreeNode`
    pub fn child_iter(&self) -> Option<impl ExactSizeIterator<Item = (&[f32], &TreeNode)>> {
        match self {
            Self::Parent { centers, children } => Some(centers.iter().zip(children.iter())),
            Self::Leaf => None,
        }
    }

    fn train_node<V: VectorStore<Elem = f32> + Send + Sync>(
        training_data: &SubsetViewVectorStore<'_, V>,
        k_per_node: NonZero<usize>,
        params: &Params,
        level: usize,
    ) -> Self {
        if training_data.len() <= params.max_leaf_size || level >= params.num_levels - 1 {
            return Self::Leaf;
        }

        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(0xdeadbeef);
        let (centers, subpartitions) =
            kmeans::train(training_data, k_per_node, &params.kmeans_params, &mut rng);
        let children = subpartitions
            .into_iter()
            .map(|p| Self::train_node(&training_data.subset_view(p), k_per_node, params, level + 1))
            .collect::<Vec<_>>();
        Self::Parent { centers, children }
    }
}

/// Implements a view over a list of indices in an underlying vector store. This view remaps the
/// vectors so that they are assigned dense indices.
///
/// This is useful when iteratively clustering an input data set.
pub struct SubsetViewVectorStore<'a, V> {
    parent: &'a V,
    subset: Vec<usize>,
}

impl<'a, V: VectorStore> SubsetViewVectorStore<'a, V> {
    /// Create an identity view over parent that yields all input vectors.
    ///
    /// This is useful over the initial input set to allow subsetting.
    pub fn identity(parent: &'a V) -> Self {
        Self {
            parent,
            subset: (0..parent.len()).collect(),
        }
    }

    /// Create a view over a subset of input from parent.
    ///
    /// *Panics* if the input is not sorted or if any element is out of bounds in parent.
    pub fn with_subset(parent: &'a V, subset: Vec<usize>) -> Self {
        assert!(subset.is_sorted());
        assert!(subset.last().is_none_or(|i| *i < parent.len()));
        Self { parent, subset }
    }

    /// Create a view from a subset of the vectors in this store.
    ///
    /// *Panics* if the input is not sorted or if any element is out of bounds.
    pub fn subset_view(&self, mut subset: Vec<usize>) -> Self {
        assert!(subset.is_sorted());
        assert!(subset.last().is_none_or(|i| *i < self.len()));

        for i in subset.iter_mut() {
            *i = self.subset[*i];
        }
        Self {
            parent: self.parent,
            subset,
        }
    }
}

impl<'a, V: VectorStore> VectorStore for SubsetViewVectorStore<'a, V> {
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
