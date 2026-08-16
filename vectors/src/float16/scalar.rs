use half::f16;

/// Serialize `v` to half precision floats in `out`, optionally scaling each value by `scale`.
pub fn serialize_f16(v: &[f32], scale: Option<f32>, out: &mut [u8]) {
    assert_eq!(out.len(), v.len() * 2);
    for (&d, o) in v.iter().zip(out.chunks_mut(2)) {
        let d = match scale {
            Some(scale) => d * scale,
            None => d,
        };
        o.copy_from_slice(&f16::from_f32(d).to_le_bytes());
    }
}

/// Deserialize half precision floats in `v` to `out`.
pub fn deserialize_f16(v: &[u8], out: &mut [f32]) {
    for (d, o) in unaligned_f16_iter(v).zip(out.iter_mut()) {
        *o = d.to_f32();
    }
}

/// Dot product of two half precision vectors.
pub fn dot_f16_f16(a: &[u8], b: &[u8]) -> f32 {
    unaligned_f16_iter(a)
        .zip(unaligned_f16_iter(b))
        .map(|(a, b)| a.to_f32() * b.to_f32())
        .sum()
}

/// Dot product of a single precision vector and a half precision vector.
pub fn dot_f32_f16(a: &[f32], b: &[u8]) -> f32 {
    a.iter()
        .zip(unaligned_f16_iter(b).map(f16::to_f32))
        .map(|(&s, o)| s * o)
        .sum()
}

/// Squared euclidean (l2) distance between two half precision vectors.
pub fn l2_f16_f16(a: &[u8], b: &[u8]) -> f32 {
    unaligned_f16_iter(a)
        .zip(unaligned_f16_iter(b))
        .map(|(a, b)| {
            let diff = a.to_f32() - b.to_f32();
            diff * diff
        })
        .sum()
}

/// Squared euclidean (l2) distance between a single precision vector and a half precision
/// vector.
pub fn l2_f32_f16(a: &[f32], b: &[u8]) -> f32 {
    a.iter()
        .zip(unaligned_f16_iter(b).map(f16::to_f32))
        .map(|(s, o)| {
            let diff = *s - o;
            diff * diff
        })
        .sum()
}

pub fn unaligned_f16_iter(raw: &[u8]) -> impl ExactSizeIterator<Item = f16> + '_ {
    let (chunks, rem) = raw.as_chunks::<{ std::mem::size_of::<f16>() }>();
    debug_assert!(rem.is_empty());
    chunks.iter().map(|c| f16::from_le_bytes(*c))
}
