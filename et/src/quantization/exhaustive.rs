//! Shared setup for exhaustive quantized scoring commands.
//!
//! Centers, coders, and per-query distance functions are built the same way whether we are
//! measuring recall or bound-driven queue depth.

use std::{fs::File, io, num::NonZero, path::PathBuf};

use clap::Args;
use easy_tiger::{
    input::{DerefVectorStore, SubsetViewVectorStore, VecVectorStore, VectorStore},
    kmeans::{Params, kmeans},
};
use memmap2::Mmap;
use rand::SeedableRng;
use rayon::prelude::*;
use vectors::{F32VectorCoder, F32VectorCoding, QueryVectorDistance, VectorSimilarity};

#[derive(Args)]
pub struct ExhaustiveArgs {
    /// Little-endian f32 vectors as a flat file where each vector has --dimensions
    #[arg(long)]
    query_vectors: PathBuf,
    /// If set, only process this many input queries.
    #[arg(long)]
    query_limit: Option<usize>,
    /// If true, quantize the query before scoring.
    ///
    /// Some format implement f32 x quantized scoring which is more accurate but slower.
    #[arg(long, default_value_t = false)]
    quantize_query: bool,

    /// Vector coding to test.
    #[arg(long)]
    pub format: F32VectorCoding,
    /// Similarity function to use.
    #[arg(long)]
    pub similarity: VectorSimilarity,

    /// Number of centers to compute and use.
    ///
    /// If 0, the data set will be uncentered.
    ///
    /// If 1, a mean vector will be computed and used as the center for all queries and docs.
    ///
    /// If >1, k-means will be used to compute centers. Each comparison will happen relative to
    /// the closest center for each doc.
    #[arg(long, default_value_t = 0)]
    centers: usize,

    /// When computing 2 or more centers, sample the data set to at most this many vectors.
    #[arg(long, default_value_t = 100_000)]
    center_sample_size: usize,

    /// Random seed used for clustering computations.
    /// Use a fixed value for repeatability.
    #[arg(long, default_value_t = 0x7774_7370414E4E)]
    seed: u64,
}

/// Coders, centers, and per-query distance functions for an exhaustive scoring run.
pub struct Exhaustive {
    similarity: VectorSimilarity,
    centers: Option<VecVectorStore<f32>>,
    coders: Vec<Box<dyn F32VectorCoder>>,
    /// One entry per query, each with one distance function per center.
    query_scorers: Vec<Vec<Box<dyn QueryVectorDistance + 'static>>>,
}

impl ExhaustiveArgs {
    /// Compute centers, build coders, and quantize each query against every center.
    pub fn setup(
        &self,
        doc_vectors: &(impl VectorStore<Elem = f32> + Send + Sync),
    ) -> io::Result<Exhaustive> {
        let query_vectors: DerefVectorStore<f32, Mmap> = DerefVectorStore::new(
            unsafe { Mmap::map(&File::open(&self.query_vectors)?)? },
            NonZero::new(doc_vectors.elem_stride()).unwrap(),
        )?;
        let query_limit = self
            .query_limit
            .unwrap_or(query_vectors.len())
            .min(query_vectors.len());

        let centers = self.compute_centers(doc_vectors);
        let query_vectors =
            SubsetViewVectorStore::new(&query_vectors, (0..query_limit).collect::<Vec<_>>());
        Ok(Exhaustive::new(
            self.similarity,
            self.format,
            self.quantize_query,
            centers,
            &query_vectors,
        ))
    }

    /// Compute the centers implied by `--centers` over `doc_vectors`.
    pub fn compute_centers(
        &self,
        doc_vectors: &(impl VectorStore<Elem = f32> + Send + Sync),
    ) -> Option<VecVectorStore<f32>> {
        compute_centers(
            doc_vectors,
            self.centers,
            self.center_sample_size,
            self.seed,
        )
    }
}

/// Compute `centers` centers over `doc_vectors`.
///
/// Returns `None` when `centers` is 0 (uncentered), the mean vector when it is 1, and k-means
/// centers over a sample of at most `center_sample_size` vectors otherwise.
pub fn compute_centers(
    doc_vectors: &(impl VectorStore<Elem = f32> + Send + Sync),
    centers: usize,
    center_sample_size: usize,
    seed: u64,
) -> Option<VecVectorStore<f32>> {
    match centers {
        0 => None,
        1 => {
            let vectors = SubsetViewVectorStore::new(doc_vectors, (0..doc_vectors.len()).collect());
            let mean = super::compute_center(&vectors);
            let mut centers = VecVectorStore::with_capacity(doc_vectors.elem_stride(), 1);
            centers.push(&mean);
            Some(centers)
        }
        _ => {
            let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed);
            let sample_size = center_sample_size.min(doc_vectors.len());
            let sample_vectors = if sample_size < doc_vectors.len() {
                let indices = rand::seq::index::sample(&mut rng, doc_vectors.len(), sample_size);
                SubsetViewVectorStore::new(doc_vectors, indices.into_vec())
            } else {
                SubsetViewVectorStore::new(doc_vectors, (0..doc_vectors.len()).collect())
            };
            println!(
                "Computing {} centers from a sample of {} vectors",
                centers,
                sample_vectors.len()
            );
            let computed = kmeans(
                &sample_vectors,
                centers,
                &Params {
                    iters: 100,
                    epsilon: 0.0001,
                    ..Params::default()
                },
                &mut rng,
            );
            Some(computed.unwrap_or_else(|e| e))
        }
    }
}

impl Exhaustive {
    /// Build coders for `format` against `centers` and quantize each query against every center.
    pub fn new(
        similarity: VectorSimilarity,
        format: F32VectorCoding,
        quantize_query: bool,
        centers: Option<VecVectorStore<f32>>,
        query_vectors: &(impl VectorStore<Elem = f32> + Send + Sync),
    ) -> Self {
        let coders: Vec<Box<dyn F32VectorCoder>> = match centers.as_ref() {
            None => vec![format.coder(similarity, None)],
            Some(cs) => cs
                .iter()
                .map(|c| format.coder(similarity, Some(c.to_vec())))
                .collect(),
        };

        let query_scorers = (0..query_vectors.len())
            .into_par_iter()
            .map(|i| {
                coders
                    .iter()
                    .enumerate()
                    .map(|(ci, coder)| {
                        let center = centers.as_ref().map(|cs| &cs[ci]);
                        // Queries are passed as owned values so the scorers do not borrow the
                        // mapped query vector file.
                        if quantize_query {
                            format.query_distance_symmetric(
                                similarity,
                                coder.encode(&query_vectors[i]),
                                center,
                            )
                        } else {
                            format.query_distance_asymmetric(
                                similarity,
                                query_vectors[i].to_vec(),
                                center,
                            )
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        Self {
            similarity,
            centers,
            coders,
            query_scorers,
        }
    }

    pub fn num_queries(&self) -> usize {
        self.query_scorers.len()
    }

    /// Encode `doc`, returning the index of the center it was encoded against along with the bytes.
    pub fn encode_doc(&self, doc: &[f32]) -> (usize, Vec<u8>) {
        let center = self.select_center_for_doc(doc);
        (center, self.coders[center].encode(doc))
    }

    /// Return the distance function for `query` against docs encoded with `center`.
    pub fn scorer(&self, query: usize, center: usize) -> &dyn QueryVectorDistance {
        self.query_scorers[query][center].as_ref()
    }

    fn select_center_for_doc(&self, doc: &[f32]) -> usize {
        match self.centers.as_ref() {
            None => 0,
            Some(centers) if centers.len() == 1 => 0,
            Some(centers) => {
                let dist = self.similarity.new_distance_function();
                centers
                    .iter()
                    .enumerate()
                    .map(|(i, c)| (i, dist.distance_f32(doc, &c)))
                    .min_by(|a, b| a.1.total_cmp(&b.1))
                    .map(|(i, _)| i)
                    .unwrap()
            }
        }
    }
}
