use std::{io, ops::Add, path::PathBuf};

use clap::Args;
use easy_tiger::input::{DerefVectorStore, VectorStore};
use half::slice::HalfFloatSliceExt;
use indicatif::ParallelProgressIterator;
use memmap2::Mmap;
use rayon::prelude::*;
use smallvec::smallvec;
use vectors::{EstimatedDistance, F32VectorCoding, VectorSimilarity, f16, rotate::Rotator};

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

    /// If set, rotate vectors before quantization.
    #[arg(long, default_value_t = false)]
    rotate: bool,

    /// Seed for rotation.
    #[arg(long, default_value_t = 11500348935374314158)]
    rotate_seed: u64,
}

pub fn distance_loss(
    args: DistanceLossArgs,
    vectors: &(impl VectorStore<Elem = f16> + Send + Sync),
) -> io::Result<()> {
    if args.center && args.rotate {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "centering and rotation are incompatible",
        ));
    }

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
    let rotator = if args.rotate {
        Some(Rotator::new(vectors.elem_stride(), args.rotate_seed))
    } else {
        None
    };

    let coder = args.format.coder(args.similarity, center.clone());
    let query_scorers = (0..query_limit)
        .into_par_iter()
        .map(|i| {
            let mut query = query_vectors[i].to_f32_vec();
            let f32_dist = F32VectorCoding::F32.query_distance_asymmetric(
                args.similarity,
                query.clone(),
                center.as_deref(),
            );
            if let Some(r) = rotator.as_ref() {
                query = r.forward(&query);
            }
            let qdist = if args.quantize_query {
                args.format.query_distance_symmetric(
                    args.similarity,
                    coder.encode(&query),
                    center.as_deref(),
                )
            } else {
                args.format.query_distance_asymmetric(
                    args.similarity,
                    query.clone(),
                    center.as_deref(),
                )
            };
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
                let doc_q = if let Some(r) = rotator.as_ref() {
                    coder.encode(&r.forward(&doc_f32))
                } else {
                    coder.encode(&doc_f32)
                };

                let actual = f32_dist.as_ref().distance(bytemuck::cast_slice(&doc_f32));
                let estimate = qdist.as_ref().estimated_distance(&doc_q);
                stats.add_sample(actual, estimate);
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
            "Z={z:.2} in range {:5.2}% over estimate {:5.2}% under estimate {:5.2}%",
            (s.in_range as f64 / stats.count as f64) * 100.0,
            (s.over as f64 / stats.count as f64) * 100.0,
            (s.under as f64 / stats.count as f64) * 100.0,
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
    fn add_sample(&mut self, actual: f64, estimate: EstimatedDistance) {
        let diff = actual - estimate.distance;
        self.count += 1;
        self.error_sum += diff.abs();
        self.error_sq_sum += diff.powi(2);
        self.error_z_sum += diff.abs() / estimate.error;
        for (s, &z) in self.zstats.iter_mut().zip(Z_SCORES.iter()) {
            s.add_sample(actual, estimate, z);
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

#[derive(Debug, Copy, Clone, Default)]
struct ZScoreStats {
    /// Number of samples within the error bound (expanded by Z score).
    in_range: usize,
    /// Number of samples where distance was an over estimate of actual distance.
    over: usize,
    /// Number of samples where distance was an under estimate of actual distance.
    under: usize,
}

impl ZScoreStats {
    fn add_sample(&mut self, actual: f64, estimate: EstimatedDistance, z: f64) {
        let e = estimate.error * z;
        if actual < estimate.distance - e {
            self.over += 1;
        } else if actual > estimate.distance + e {
            self.under += 1;
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
            over: self.over + rhs.over,
            under: self.under + rhs.under,
        }
    }
}
