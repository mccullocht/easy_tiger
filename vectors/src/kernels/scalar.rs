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

/// Compute a 4-bit by 1-bit inner product in the turbo packed format.
///
/// Dimensions are packed in 16 byte blocks with a potential remainder. `a` has 4x as many blocks
/// as `b`, where each block in `a` represents a bitplane split of a 4-bit query.
///
/// If `S` is true this is treated as a signed product and hamming distance (XOR) is used, otherwise
/// it is unsigned and AND is used to combine the vectors.
// XXX do we even need S???
#[inline]
pub fn turbo_4x1_inner_product<const S: bool>(a: &[u8], b: &[u8]) -> u32 {
    let (ahead, atail) = a.as_chunks::<64>();
    let (bhead, btail) = b.split_at(ahead.len() * 16);
    let bhead = bhead.as_chunks::<16>().0;
    let mut pdot = [0u32; 4];
    for (a, b) in ahead.iter().zip(bhead.iter()) {
        let ac = a.as_chunks::<16>().0;
        let a0 = u128::from_le_bytes(ac[0]);
        let a1 = u128::from_le_bytes(ac[1]);
        let a2 = u128::from_le_bytes(ac[2]);
        let a3 = u128::from_le_bytes(ac[3]);
        let b = u128::from_le_bytes(*b);

        let x = if S {
            [a0 ^ b, a1 ^ b, a2 ^ b, a3 ^ b]
        } else {
            [a0 & b, a1 & b, a2 & b, a3 & b]
        };
        pdot[0] += x[0].count_ones();
        pdot[1] += x[1].count_ones();
        pdot[2] += x[2].count_ones();
        pdot[3] += x[3].count_ones();
    }

    let tail_dot = if !atail.is_empty() {
        turbo_4x1_inner_product_tail::<S>(atail, btail)
    } else {
        0
    };
    tail_dot + pdot[0] + pdot[1] * 2 + pdot[2] * 4 + pdot[3] * 8
}

/// Compute a 4-bit by 1-bit inner product in the turbo packed format byte-by-byte.
///
/// If `S` is true this is treated as a signed product and hamming distance (XOR) is used, otherwise
/// it is unsigned and AND is used to combine the vectors.
#[inline]
pub fn turbo_4x1_inner_product_tail<const S: bool>(a: &[u8], b: &[u8]) -> u32 {
    let mut pdot = [0u32; 4];
    let mut ait = a.chunks(a.len() / 4);
    let a = [
        ait.next().unwrap(),
        ait.next().unwrap(),
        ait.next().unwrap(),
        ait.next().unwrap(),
    ];

    for (i, &b) in b.iter().enumerate() {
        pdot[0] += (a[0][i] & b).count_ones();
        pdot[1] += (a[1][i] & b).count_ones();
        pdot[2] += (a[2][i] & b).count_ones();
        pdot[3] += (a[3][i] & b).count_ones();
    }

    pdot[0] + pdot[1] * 2 + pdot[2] * 4 + pdot[3] * 8
}
