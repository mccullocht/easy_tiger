use std::{io, ops::Add, path::PathBuf};

use clap::Args;
use easy_tiger::input::{DerefVectorStore, VectorStore};
use half::slice::HalfFloatSliceExt;
use indicatif::ParallelProgressIterator;
use memmap2::Mmap;
use rayon::prelude::*;
use smallvec::smallvec;
use vectors::{EstimatedDistance, F32VectorCoding, VectorSimilarity, f16};

use crate::ui::progress_bar;

const Z_SCORES: &[f64] = &[1.00, 1.96, 3.00];

#[derive(Args)]
pub struct DistanceLossArgs {
    /// Query vectors: f16 vectors in BigANN format (an 8 byte `<len,dim>` header followed by
    /// little-endian f16 vector data).
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
    vectors: &(impl VectorStore<Elem = f16> + Send + Sync),
) -> io::Result<()> {
    let query_vectors: DerefVectorStore<f16, Mmap> =
        DerefVectorStore::from_file(args.query_vectors)?;
    if query_vectors.elem_stride() != vectors.elem_stride() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "query and doc vectors must have the same dimensionality ({} vs {})",
                query_vectors.elem_stride(),
                vectors.elem_stride()
            ),
        ));
    }
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
            let mut query = vec![0.0f32; query_vectors.elem_stride()];
            query_vectors[i].convert_to_f32_slice(&mut query);
            let qdist = if args.quantize_query {
                args.format
                    .query_distance_symmetric(args.similarity, coder.encode(&query), None)
            } else {
                args.format.query_distance_asymmetric(
                    args.similarity,
                    query.clone(),
                    center.as_deref(),
                )
            };
            let f32_dist =
                F32VectorCoding::F32.query_distance_asymmetric(args.similarity, query, None);
            (f32_dist, qdist)
        })
        .collect::<Vec<_>>();

    let stats = (0..vectors.len())
        .into_par_iter()
        .progress_with(progress_bar(vectors.len(), "scoring"))
        .map(|d| {
            let mut stats = DistanceLossStats::default();
            for (f32_dist, qdist) in query_scorers.iter() {
                let doc_f32 = vectors[d].to_f32_vec();
                let doc_q = coder.encode(&doc_f32);

                let expected = f32_dist.as_ref().distance(bytemuck::cast_slice(&doc_f32));
                let actual = qdist.as_ref().estimated_distance(&doc_q);
                stats.add_sample(expected, actual);
            }
            stats
        })
        .reduce(|| DistanceLossStats::default(), |a, b| a + b);

    println!(
        "Vectors: {} mean abs error: {:.6} mean square error: {:.6} mean Z score {:.6}",
        stats.count,
        stats.error_sum / stats.count as f64,
        stats.error_sq_sum / stats.count as f64,
        stats.error_z_sum / stats.count as f64,
    );
    for (&z, s) in Z_SCORES.iter().zip(stats.zstats.iter()) {
        println!(
            "Z={z:.2} in range {:5.2}% below {:5.2}% above {:5.2}%",
            (s.in_range as f64 / stats.count as f64) * 100.0,
            (s.below_range as f64 / stats.count as f64) * 100.0,
            (s.above_range as f64 / stats.count as f64) * 100.0,
        );
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct DistanceLossStats {
    count: usize,
    error_sum: f64,
    error_sq_sum: f64,
    error_z_sum: f64,
    zstats: smallvec::SmallVec<[ZScoreStats; 4]>,
}

impl DistanceLossStats {
    fn add_sample(&mut self, expected: f64, actual: EstimatedDistance) {
        let diff = expected - actual.distance;
        self.count += 1;
        self.error_sum += diff.abs();
        self.error_sq_sum += diff.powi(2);
        self.error_z_sum += diff.abs() / actual.error;
        for (s, &z) in self.zstats.iter_mut().zip(Z_SCORES.iter()) {
            s.add_sample(expected, actual, z);
        }
    }
}

impl Default for DistanceLossStats {
    fn default() -> Self {
        Self {
            count: 0,
            error_sum: 0.0,
            error_sq_sum: 0.0,
            error_z_sum: 0.0,
            zstats: smallvec![ZScoreStats::default(); Z_SCORES.len()],
        }
    }
}

impl Add<DistanceLossStats> for DistanceLossStats {
    type Output = Self;

    fn add(self, rhs: DistanceLossStats) -> Self::Output {
        Self {
            count: self.count + rhs.count,
            error_sum: self.error_sum + rhs.error_sum,
            error_sq_sum: self.error_sq_sum + rhs.error_sq_sum,
            error_z_sum: self.error_z_sum + rhs.error_z_sum,
            zstats: self
                .zstats
                .into_iter()
                .zip(rhs.zstats.into_iter())
                .map(|(a, b)| a + b)
                .collect(),
        }
    }
}

// XXX should this be under/over estimate? yes this is confusing because below means "expected is below"
#[derive(Debug, Copy, Clone, Default)]
struct ZScoreStats {
    in_range: usize,
    below_range: usize,
    above_range: usize,
}

impl ZScoreStats {
    fn add_sample(&mut self, expected: f64, actual: EstimatedDistance, z: f64) {
        let e = actual.error * z;
        if expected < actual.distance - e {
            self.below_range += 1;
        } else if expected > actual.distance + e {
            self.above_range += 1;
        } else {
            self.in_range += 1;
        }
    }
}

impl Add<ZScoreStats> for ZScoreStats {
    type Output = Self;

    fn add(self, rhs: ZScoreStats) -> Self::Output {
        Self {
            in_range: self.in_range + rhs.in_range,
            below_range: self.below_range + rhs.below_range,
            above_range: self.above_range + rhs.above_range,
        }
    }
}
