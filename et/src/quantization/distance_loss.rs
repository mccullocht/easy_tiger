use std::{borrow::Cow, fs::File, io, num::NonZero, ops::RangeInclusive, path::PathBuf};

use clap::Args;
use easy_tiger::input::{DerefVectorStore, VectorStore};
use indicatif::ParallelProgressIterator;
use memmap2::Mmap;
use rayon::prelude::*;
use vectors::{ESTIMATED_DISTANCE_Z_SCORE, F32VectorCoding, VectorSimilarity, l2_normalize};

use crate::ui::progress_bar;

/// L2-normalize `vector` if `similarity` is one the coders normalize during encoding, so that
/// reference f32 distances are computed over the same vectors the quantized distances are.
fn normalize_for<'a>(
    similarity: VectorSimilarity,
    vector: impl Into<Cow<'a, [f32]>>,
) -> Cow<'a, [f32]> {
    if similarity.angular() {
        l2_normalize(vector)
    } else {
        vector.into()
    }
}

/// Error statistics accumulated over (query, document) pairs.
///
/// Alongside the raw error magnitude this tracks the moments of the standardized error
/// `z = (expected - estimated) / sigma`, where `sigma` is the modeled standard deviation recovered
/// from the half-width of the returned bounds. Those moments separate the three ways the bounds can
/// be miscalibrated: a nonzero mean means the estimate is biased so the interval is centered in the
/// wrong place, a stddev away from one means `sigma` itself is the wrong size, and a large excess
/// kurtosis means the error is too heavy-tailed for any single z-score to cover the nominal share of
/// pairs.
#[derive(Clone, Copy, Default)]
struct LossStats {
    count: usize,
    error_sum: f64,
    error_sq_sum: f64,
    in_range_count: usize,
    /// Number of pairs contributing to the `z_*` moments. Pairs with a degenerate (zero-width)
    /// bound have no defined `sigma` and are excluded.
    z_count: usize,
    z_sum: f64,
    z_sq_sum: f64,
    z_cube_sum: f64,
    z_quad_sum: f64,
}

impl LossStats {
    fn observe(&mut self, expected: f64, bounds: &RangeInclusive<f64>) {
        let estimated = (*bounds.start() + *bounds.end()) / 2.0;
        let diff = expected - estimated;
        self.count += 1;
        self.error_sum += diff.abs();
        self.error_sq_sum += diff * diff;
        if bounds.contains(&expected) {
            self.in_range_count += 1;
        }

        let sigma = (*bounds.end() - *bounds.start()) / 2.0 / ESTIMATED_DISTANCE_Z_SCORE as f64;
        if sigma > 0.0 {
            let z = diff / sigma;
            self.z_count += 1;
            self.z_sum += z;
            self.z_sq_sum += z * z;
            self.z_cube_sum += z * z * z;
            self.z_quad_sum += z * z * z * z;
        }
    }

    fn merge(mut self, other: Self) -> Self {
        self.count += other.count;
        self.error_sum += other.error_sum;
        self.error_sq_sum += other.error_sq_sum;
        self.in_range_count += other.in_range_count;
        self.z_count += other.z_count;
        self.z_sum += other.z_sum;
        self.z_sq_sum += other.z_sq_sum;
        self.z_cube_sum += other.z_cube_sum;
        self.z_quad_sum += other.z_quad_sum;
        self
    }

    fn print(&self) {
        let count = self.count;
        println!(
            "Vectors: {count} mean abs error: {:.6} mean square error: {:.6} in range: {} ({:.2}%)",
            self.error_sum / count as f64,
            self.error_sq_sum / count as f64,
            self.in_range_count,
            self.in_range_count as f64 / count as f64 * 100.0
        );

        if self.z_count == 0 {
            println!("No pairs with a non-degenerate bound; skipping standardized error moments.");
            return;
        }
        // Central moments of z, derived from the raw moments.
        let n = self.z_count as f64;
        let m1 = self.z_sum / n;
        let m2 = self.z_sq_sum / n - m1 * m1;
        let m3 = self.z_cube_sum / n - 3.0 * m1 * self.z_sq_sum / n + 2.0 * m1 * m1 * m1;
        let m4 = self.z_quad_sum / n - 4.0 * m1 * self.z_cube_sum / n
            + 6.0 * m1 * m1 * self.z_sq_sum / n
            - 3.0 * m1 * m1 * m1 * m1;
        println!(
            "Standardized error over {} pairs: mean {:.4} stddev {:.4} skew {:.4} excess kurtosis {:.4}",
            self.z_count,
            m1,
            m2.max(0.0).sqrt(),
            m3 / m2.powf(1.5),
            m4 / (m2 * m2) - 3.0
        );
    }
}

#[derive(Args)]
pub struct DistanceLossArgs {
    /// Little-endian f32 vectors of some dimensionality as input vectors.
    #[arg(long)]
    query_vectors: PathBuf,
    /// If true, quantize queries before computing loss, bypassing any f32 x quantized query
    /// vector distance implementation.
    #[arg(long)]
    quantize_query: bool,
    /// Limit on the number of queries. If unset, use all input queries.
    #[arg(long)]
    query_limit: Option<usize>,

    /// Similarity function to use.
    #[arg(long)]
    similarity: VectorSimilarity,
    /// Format to compare against f32 distance.
    #[arg(long)]
    format: F32VectorCoding,

    /// If set, compute the center of the dataset and apply before computing distances.
    #[arg(long, default_value_t = false)]
    center: bool,
}

pub fn distance_loss(
    args: DistanceLossArgs,
    vectors: &(impl VectorStore<Elem = f32> + Send + Sync),
) -> io::Result<()> {
    let query_vectors: DerefVectorStore<f32, Mmap> = DerefVectorStore::new(
        unsafe { Mmap::map(&File::open(args.query_vectors)?)? },
        NonZero::new(vectors.elem_stride()).unwrap(),
    )?;
    let query_limit = args
        .query_limit
        .unwrap_or(query_vectors.len())
        .min(query_vectors.len());

    let center = if args.center {
        Some(super::compute_center(vectors))
    } else {
        None
    };

    let coder = args.format.coder(args.similarity, center.clone());
    let query_scorers = (0..query_limit)
        .into_par_iter()
        .map(|i| {
            let query = &query_vectors[i];
            let qdist = if args.quantize_query {
                args.format
                    .query_distance_symmetric(args.similarity, coder.encode(&query), None)
            } else {
                args.format.query_distance_asymmetric(
                    args.similarity,
                    query.to_vec(),
                    center.as_deref(),
                )
            };
            // The f32 distance functions assume angular vectors were normalized during encoding,
            // so normalize here to match what the quantized coder does internally. Without this
            // the reference distance disagrees with the quantized distance by a per-vector norm
            // factor, which shows up as a bit-rate-independent error floor.
            let f32_dist = F32VectorCoding::F32.query_distance_asymmetric(
                args.similarity,
                normalize_for(args.similarity, query.to_vec()),
                None,
            );
            (f32_dist, qdist)
        })
        .collect::<Vec<_>>();

    let stats = (0..vectors.len())
        .into_par_iter()
        .progress_with(progress_bar(vectors.len(), "scoring"))
        .map(|d| {
            let doc_f32 = &vectors[d];
            let doc_q = coder.encode(&doc_f32);
            let doc_ref = normalize_for(args.similarity, doc_f32);
            let mut stats = LossStats::default();
            for (q_f32, q_q) in query_scorers.iter() {
                let expected = q_f32.as_ref().distance(bytemuck::cast_slice(&doc_ref));
                stats.observe(expected, &q_q.as_ref().distance_bounds(&doc_q));
            }
            stats
        })
        .reduce(LossStats::default, LossStats::merge);

    stats.print();
    Ok(())
}
