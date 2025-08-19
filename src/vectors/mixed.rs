//! Mixed representation vector codecs use different quantization schemes for different vector
//! dimensions. This is typically used together with Matryoshka Representation Learning to provide
//! more accurate comparisons for more important dimensions.

use simsimd::SpatialSimilarity;

use crate::vectors::F32VectorCoder;

// XXX I would like to make this generic with sub-vector coders.

/// Use i8-scaled-uniform for earlier dimensions and i1/binary for later dimensions.
#[derive(Debug, Copy, Clone)]
pub struct I8I1VectorCoder(usize);

impl I8I1VectorCoder {
    /// Use i8 coding for all dimensions before `split_dimension` and i1 for all dimensions after.
    pub fn new(split_dimension: usize) -> Self {
        Self(split_dimension)
    }
}

impl F32VectorCoder for I8I1VectorCoder {
    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        let (i8_vec, i1_vec) = vector.split_at(self.0);
        let i8_norm_sq = SpatialSimilarity::dot(i8_vec, i8_vec).expect("same dim");
        let i1_norm_sq = SpatialSimilarity::dot(i1_vec, i1_vec).expect("same dim");

        // record l2 norm to scale dot product
        let l2_norm = (i8_norm_sq + i1_norm_sq).sqrt() as f32;
        let (scale, inv_scale) = super::scaled_uniform::compute_scale::<{ i8::MAX as i16 }>(i8_vec);
        // record i1_avg_dim. this value is the sqrt of the avg value based on i1_norm_sq.
        // the bitstream will be mapped into [-i1_avg_dim,+i1_avg_dim].
        let i1_avg_dim = (i1_norm_sq / i1_vec.len() as f64).sqrt() as f32;
        println!("i8_norm_sq {i8_norm_sq} i1_norm_sq {i1_norm_sq} i1_avg_dim {i1_avg_dim}");

        let (meta_out, out_rem) = out.split_at_mut(std::mem::size_of::<f32>() * 3);
        let (i8_out, i1_out) = out_rem.split_at_mut(i8_vec.len());

        let meta = meta_out.as_chunks_mut::<{ std::mem::size_of::<f32>() }>().0;
        meta[0].copy_from_slice(&l2_norm.to_le_bytes());
        meta[1].copy_from_slice(&inv_scale.to_le_bytes());
        meta[2].copy_from_slice(&i1_avg_dim.to_le_bytes());

        for (i, o) in i8_vec.iter().zip(i8_out.iter_mut()) {
            *o = ((*i * scale).round() as i8).to_le_bytes()[0];
        }
        for (i, o) in i1_vec.chunks(8).zip(i1_out.iter_mut()) {
            *o = i
                .iter()
                .enumerate()
                .map(|(j, d)| if *d > 0.0 { 1 << j } else { 0 << j })
                .reduce(|a, b| a | b)
                .expect("chunks");
        }
    }

    fn byte_len(&self, dimensions: usize) -> usize {
        let i8_dim = dimensions.min(self.0);
        let i1_dim = dimensions - i8_dim;
        std::mem::size_of::<f32>() * 3 + i8_dim + i1_dim.div_ceil(8)
    }

    fn decode(&self, encoded: &[u8]) -> Option<Vec<f32>> {
        let (meta, in_rem) = encoded.split_at(std::mem::size_of::<f32>() * 3);
        let (i8_vec, i1_vec) = in_rem.split_at(self.0);

        let meta = meta.as_chunks::<{ std::mem::size_of::<f32>() }>().0;
        let scale = f32::from_le_bytes(meta[1]);
        let i1_avg_dim = f32::from_le_bytes(meta[2]);
        let i1_decode_table = [-i1_avg_dim, i1_avg_dim];
        let i8_vec_iter = i8_vec.iter().map(|d| *d as f32 * scale);
        let i1_vec_iter = i1_vec
            .iter()
            .flat_map(|b| (0..8).map(|i| i1_decode_table[(*b >> i) as usize & 0x1]));
        Some(i8_vec_iter.chain(i1_vec_iter).collect())
    }
}
