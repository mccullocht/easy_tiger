//! Platform-independent kernel implementations written in safe rust.
//!
//! Most of these implementations are marked inline as optimized kernels could be built on them,
//! either by using them to process a scalar tail or by allowing them to auto-vectorize with
//! specific architecture target features.

#[inline(always)]
pub fn decode_to_f32(vector: &[i8], out: &mut [f32]) {
    for (i, o) in vector.iter().zip(out.iter_mut()) {
        *o = *i as f32;
    }
}

#[inline(always)]
pub fn l2_distance_sq(a: &[i8], b: &[i8]) -> i32 {
    a.iter()
        .zip(b.iter())
        .map(|(&a, &b)| {
            let d = a as i32 - b as i32;
            d * d
        })
        .sum::<i32>()
}

#[inline(always)]
pub fn inner_product(a: &[i8], b: &[i8]) -> i32 {
    a.iter()
        .zip(b.iter())
        .map(|(&a, &b)| a as i32 * b as i32)
        .sum::<i32>()
}

#[inline(always)]
pub fn cosine_similarity(a: &[i8], b: &[i8]) -> f64 {
    let (anorm, bnorm, ip) = a
        .iter()
        .zip(b.iter())
        .map(|(&a, &b)| {
            let a = a as i32;
            let b = b as i32;
            (a * a, b * b, a * b)
        })
        .fold((0i32, 0i32, 0i32), |acc, x| {
            (acc.0 + x.0, acc.1 + x.1, acc.2 + x.2)
        });
    ip as f64 / (anorm as f64 * bnorm as f64).sqrt()
}
