//! Implementation of Lucene quantizers as a baseline to compare against.

use std::num::NonZero;

use serde::{Deserialize, Serialize};

use crate::vectors::F32VectorCoder;

/// Trained parameters for quantization.
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScalarQuantizerParams {
    pub bits: u8,
    pub min_quantile: f32,
    pub max_quantile: f32,
}

/// Implementation of Lucene [`ScalarQuantizer`](https://github.com/apache/lucene/blob/main/lucene/core/src/java/org/apache/lucene/util/quantization/ScalarQuantizer.java)
///
/// Note that quantizer requires _training_ on the input data set to choose quantiles.
#[derive(Debug, Copy, Clone)]
pub struct ScalarQuantizerVectorCoder {
    params: ScalarQuantizerParams,
    scale: f32,
    alpha: f32,
}

impl ScalarQuantizerVectorCoder {
    pub fn new(params: ScalarQuantizerParams) -> Self {
        assert!(params.bits <= 8);
        assert!(params.min_quantile.is_finite());
        assert!(params.max_quantile.is_finite());
        assert!(params.max_quantile > params.min_quantile);

        let range = params.max_quantile - params.min_quantile;
        let divisor = ((1 << params.bits) - 1) as f32;
        Self {
            params,
            scale: divisor / range,
            alpha: range / divisor,
        }
    }

    /// Train on the data set provided as `vectors` and return the min and max quantile.
    pub fn train<'a>(
        dimensions: NonZero<usize>,
        vectors: impl Iterator<Item = &'a [f32]>,
    ) -> (f32, f32) {
        // Use the same parameters as lucene, so batch no more than 20 vectors at a time.
        let mut scratch = Vec::with_capacity(dimensions.get() * 20);
        let mut batches = 0usize;
        let mut quantile_sums = (0.0f64, 0.0f64);
        let confidence_interval = 1.0 - (1.0 / (dimensions.get() + 1) as f64);
        for (i, v) in vectors.enumerate() {
            if i != 0 && i % 20 == 0 {
                Self::extract_quantiles(confidence_interval, &mut quantile_sums, &mut scratch);
                batches += 1;
                scratch.clear();
            }

            scratch.extend_from_slice(v);
        }

        if !scratch.is_empty() {
            Self::extract_quantiles(confidence_interval, &mut quantile_sums, &mut scratch);
            batches += 1;
        }

        (
            (quantile_sums.0 / batches as f64) as f32,
            (quantile_sums.1 / batches as f64) as f32,
        )
    }

    fn extract_quantiles(
        confidence_interval: f64,
        quantile_sums: &mut (f64, f64),
        scratch: &mut [f32],
    ) {
        let select_index =
            (scratch.len() as f64 * (1.0 - confidence_interval) / 2.0 + 0.5) as usize;
        let middle_quantile = if select_index == 0 {
            scratch
        } else {
            scratch.select_nth_unstable_by(select_index, |a, b| a.total_cmp(b));
            scratch[select_index..]
                .select_nth_unstable_by(select_index, |a, b| a.total_cmp(b).reverse());
            &scratch[(select_index * 2)..]
        };
        quantile_sums.0 += *middle_quantile
            .iter()
            .min_by(|a, b| a.total_cmp(*b))
            .unwrap() as f64;
        quantile_sums.1 += *middle_quantile
            .iter()
            .max_by(|a, b| a.total_cmp(*b))
            .unwrap() as f64;
    }
}

impl F32VectorCoder for ScalarQuantizerVectorCoder {
    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        // XXX we assume the value is normalized for angular distances -- at least this is what
        // lucene assumes. we have no notion of encoding depending on similarity (maybe we should).
        let correction = vector
            .iter()
            .zip(out[4..].iter_mut())
            .map(|(d, o)| {
                let dx = *d - self.params.min_quantile;
                let dxc = self
                    .params
                    .min_quantile
                    .max(self.params.max_quantile.min(*d))
                    - self.params.min_quantile;
                let dxs = self.scale * dxc;
                let dxq = dxs.round() * self.alpha;
                *o = dxs.round() as u8;
                self.params.min_quantile * (*d - self.params.min_quantile / 2.0f32)
                    + (dx - dxq) * dxq
            })
            .sum::<f32>();
        out[0..4].copy_from_slice(&correction.to_le_bytes());
    }

    fn byte_len(&self, dimensions: usize) -> usize {
        dimensions + std::mem::size_of::<f32>()
    }

    fn decode(&self, encoded: &[u8]) -> Option<Vec<f32>> {
        let (_, vector) = encoded.split_at(4);
        Some(
            vector
                .iter()
                .map(|q| (self.alpha * *q as f32) + self.params.min_quantile)
                .collect(),
        )
    }
}
