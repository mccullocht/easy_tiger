use std::{borrow::Cow, fs::File, io, num::NonZero, path::PathBuf, sync::Arc};

use clap::Args;
use easy_tiger::input::{DerefVectorStore, SubsetViewVectorStore, VectorStore};
use indicatif::ParallelProgressIterator;
use memmap2::Mmap;
use rayon::prelude::*;
use vectors::{F32VectorCoding, QueryVectorDistance, VectorSimilarity};

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

    /// Limit on the number of documents. If unset, use all input vectors as docs.
    #[arg(long)]
    doc_limit: Option<usize>,

    /// Similarity function to use.
    #[arg(long)]
    similarity: VectorSimilarity,
    /// Format to compare against f32 distance.
    #[arg(long)]
    format: F32VectorCoding,

    /// If set, compute the center of the dataset and apply before computing distances.
    #[arg(long, default_value_t = false)]
    center: bool,

    /// If set, all docs will be centered before encoding, but queries will use a centering aware
    /// distance function instead of also being centered.
    #[arg(long, default_value_t = false)]
    center_distance_a: bool,

    /// If set, both docs and queries will be centered before encoding and the distance function
    /// will be centering aware.
    #[arg(long, default_value_t = false)]
    center_distance_s: bool,
}

// XXX state of play
// for tlvq8 dot center-distance-a >> uncentered >> center-distance-s
//            l2 centered >> uncentered
//   all l2 distances are less accurate than uncentered dot though. maybe a metric artifact.
// for tlvq1 dot center-distance-a > center-distance-s >> uncentered
//            l2 centered >> uncentered
//   again all l2 distances are less accurate than uncentered dot. maybe a metric artifact.
//
// For dot this only works because I hacked it so that l2norm=1 on both side, otherwise the doc
// residual gets a non-trivial l2norm and the produced adjustment is wrong.
//
// The improvements are very good at SPANN cluster scale.

struct CenteredDotQueryVectorDistance {
    // XXX query_residual_scorer
    residual_scorer: Box<dyn QueryVectorDistance>,
    // XXX query_center_dot
    center_dot: f64,
}

impl CenteredDotQueryVectorDistance {
    pub fn new(vector: &[f32], center: &[f32], format: F32VectorCoding) -> Self {
        let center_dot = vector
            .iter()
            .zip(center.iter())
            .map(|(a, b)| a * b)
            .sum::<f32>() as f64;
        let residual_scorer =
            format.query_vector_distance_f32(vector.to_owned(), VectorSimilarity::Dot);
        Self {
            residual_scorer,
            center_dot,
        }
    }

    /// Convert a dot product score back into a normalized dot product.
    fn invert_score(dot_score: f64) -> f64 {
        dot_score * -2.0 + 1.0
    }
}

impl QueryVectorDistance for CenteredDotQueryVectorDistance {
    fn distance(&self, vector: &[u8]) -> f64 {
        // If vector is rv = v - c, then dot(q, v) = dot(q, rv + c) = dot(q, rv) + dot(q, c)
        (-(Self::invert_score(self.residual_scorer.distance(vector)) + self.center_dot) + 1.0) / 2.0
    }
}

// XXX as written this cannot work.
// XXX trivially, dot(q,d) = dot(rq+c, rd+c) = dot(rq,rd) + dot(rq,c) + dot(c,rd) + dot(c,c)
// XXX practically I do not want to multiply the residual vectors by the center vector, so after
//     this first expansion I sub those terms with q/d where possible:
// XXX dot(rq, rd) + dot(q - c, c) + dot(d - c, c) + dot(c, c)
// XXX dot(rq, rq) + dot(q, c) + dot(d, c) + dot(c, c)
// XXX To do this the quantizer must know the center to compute dot(v,c) without centering first,
//     and it must store this term.
struct SymmetricalCenteredDotQueryVectorDistance {
    query_residual_scorer: Box<dyn QueryVectorDistance>,
    center_residual_scorer: Box<dyn QueryVectorDistance>,
    query_center_dot: f64,
    center_dot: f64,
}

impl SymmetricalCenteredDotQueryVectorDistance {
    pub fn new(query: &[f32], center: &[f32], format: F32VectorCoding) -> Self {
        let query_residual_scorer = format.query_vector_distance_f32(
            query
                .iter()
                .zip(center.iter())
                .map(|(q, c)| q - c)
                .collect::<Vec<f32>>(),
            VectorSimilarity::Dot,
        );
        let center_residual_scorer =
            format.query_vector_distance_f32(center.to_owned(), VectorSimilarity::Dot);
        let query_center_dot = query
            .iter()
            .zip(center.iter())
            .map(|(a, b)| a * b)
            .sum::<f32>() as f64;
        let center_dot = center.iter().map(|&c| c * c).sum::<f32>() as f64;
        Self {
            query_residual_scorer,
            center_residual_scorer,
            query_center_dot,
            center_dot,
        }
    }

    /// Convert a dot product score back into a normalized dot product.
    fn invert_score(dot_score: f64) -> f64 {
        dot_score * -2.0 + 1.0
    }
}

impl QueryVectorDistance for SymmetricalCenteredDotQueryVectorDistance {
    fn distance(&self, vector: &[u8]) -> f64 {
        let qd_dot = Self::invert_score(self.query_residual_scorer.distance(vector));
        let cd_dot = Self::invert_score(self.center_residual_scorer.distance(vector));
        let dot = qd_dot + self.query_center_dot + cd_dot + self.center_dot;
        (-dot + 1.0) / 2.0
    }
}

pub fn distance_loss(
    args: DistanceLossArgs,
    vectors: &(impl VectorStore<Elem = f32> + Send + Sync),
) -> io::Result<()> {
    let query_vectors: DerefVectorStore<f32, Mmap> = DerefVectorStore::new(
        unsafe { Mmap::map(&File::open(args.query_vectors)?)? },
        NonZero::new(vectors.elem_stride()).unwrap(),
    )?;
    let query_limit = args.query_limit.unwrap_or(query_vectors.len());

    let doc_limit = args.doc_limit.unwrap_or(vectors.len());
    let vectors = SubsetViewVectorStore::new(vectors, (0..doc_limit).collect());

    let center = if args.center {
        Some(super::compute_center(&vectors))
    } else {
        None
    };

    let coder = args.format.new_coder(args.similarity);
    let query_scorers = (0..query_limit)
        .into_par_iter()
        .map(|i| {
            let mut query = Cow::from(&query_vectors[i]);
            let f32_dist =
                F32VectorCoding::F32.query_vector_distance_f32(query.to_vec(), args.similarity);
            if let Some(center) = center.as_ref() {
                for (q, c) in query.to_mut().iter_mut().zip(center.iter()) {
                    *q -= *c;
                }
            }
            let qdist = if let Some(center) = center.as_ref()
                && args.center_distance_a
            {
                Box::new(CenteredDotQueryVectorDistance::new(
                    &query_vectors[i],
                    center,
                    args.format,
                ))
            } else if let Some(center) = center.as_ref()
                && args.center_distance_s
            {
                Box::new(SymmetricalCenteredDotQueryVectorDistance::new(
                    &query,
                    center,
                    args.format,
                ))
            } else if args.quantize_query {
                args.format
                    .query_vector_distance_indexing(coder.encode(&query), args.similarity)
            } else {
                args.format
                    .query_vector_distance_f32(query.to_vec(), args.similarity)
            };
            (f32_dist, qdist)
        })
        .collect::<Vec<_>>();

    let doc_limit = args.doc_limit.unwrap_or(vectors.len());

    let (count, error_sum, error_sq_sum) = (0..doc_limit)
        .into_par_iter()
        .progress_count(doc_limit as u64)
        .flat_map(|d| {
            let doc_f32 = Arc::new(vectors[d].to_vec());
            let mut doc = Cow::from(&vectors[d]);
            if let Some(center) = center.as_ref() {
                for (d, c) in doc.to_mut().iter_mut().zip(center.iter()) {
                    *d -= *c;
                }
            }
            let doc = Arc::new(coder.encode(&doc));
            (0..query_limit)
                .into_par_iter()
                .map(move |q| (q, Arc::clone(&doc), Arc::clone(&doc_f32)))
        })
        .map(|(q, doc, doc_f32)| {
            let (f32_dist, qdist) = &query_scorers[q];
            let f32_dist = f32_dist
                .as_ref()
                .distance(bytemuck::cast_slice(doc_f32.as_ref()));
            let qdist = qdist.as_ref().distance(doc.as_ref());
            let diff = f32_dist - qdist;
            (1, diff.abs(), diff * diff)
        })
        .reduce(
            || (0, 0.0f64, 0.0f64),
            |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
        );

    println!(
        "Vectors: {} mean abs error: {} mean square error: {}",
        count,
        error_sum / count as f64,
        error_sq_sum / count as f64
    );
    Ok(())
}
