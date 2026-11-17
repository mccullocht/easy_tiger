use crate::packing::TurboUnpacker;

/// Compute `Σ query[i] * (table[doc[i]] + bias)` where `doc` is a turbo packed `B` bit vector.
///
/// `bias` allows SIMD kernels that compute unsigned x signed products to use this for the tail.
/// The result is computed with wrapping arithmetic so that biased intermediate results that
/// overflow are still correct once the bias is removed.
pub(super) fn lut_dot<const B: usize>(
    query: &[i8],
    table: &[i8; 16],
    bias: i32,
    doc: &[u8],
) -> i32 {
    query
        .iter()
        .zip(TurboUnpacker::<B>::new(doc))
        .map(|(&q, c)| (q as i32).wrapping_mul(table[c as usize] as i32 + bias))
        .fold(0i32, i32::wrapping_add)
}
