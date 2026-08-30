//! Scalar implementations of kernel functions.

/// Computes the bitwise inner product of two bit strings.
///
/// If `S` is true this is treated as a signed product and hamming distance (XOR) is used, otherwise
/// it is unsigned and AND is used to combine the vectors.
///
/// *Panics* if the vectors are not the same length.
#[inline]
pub fn bitstring_inner_product<const S: bool>(a: &[u8], b: &[u8]) -> u32 {
    assert_eq!(a.len(), b.len());
    let (ahead, atail) = a.as_chunks::<8>();
    let (bhead, btail) = b.as_chunks::<8>();
    let ip = ahead
        .iter()
        .copied()
        .map(u64::from_ne_bytes)
        .zip(bhead.iter().copied().map(u64::from_ne_bytes))
        .map(|(a, b)| {
            if S {
                (a ^ b).count_ones()
            } else {
                (a & b).count_ones()
            }
        })
        .sum::<u32>();

    if atail.is_empty() {
        ip
    } else {
        ip + bitstring_inner_product_tail::<S>(atail, btail)
    }
}

/// Same as [`bitstring_inner_product`] but operates byte-by-byte for processing a vector `tail`.
#[inline]
pub fn bitstring_inner_product_tail<const S: bool>(a: &[u8], b: &[u8]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(a, b)| {
            if S {
                (a ^ b).count_ones()
            } else {
                (a & b).count_ones()
            }
        })
        .sum::<u32>()
}
