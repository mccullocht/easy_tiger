//! Precomputed codebooks for different bit widths.
//!
//! These are based on the optimal quantization of a standard normal distribution symmetric around
//! 0. These will be scaled by 1/sqrt(D) at runtime.

#![allow(clippy::excessive_precision)]

#[allow(clippy::approx_constant)]
pub const CENTROIDS_1: [f32; 2] = [-0.79788456, 0.79788456];
pub const CENTROIDS_2: [f32; 4] = [-1.510418, -0.452780, 0.452780, 1.510418];
#[allow(dead_code)] // Kept for the 3-bit-per-dimension codec, though no coding uses it today.
pub const CENTROIDS_3: [f32; 8] = [
    -2.151946, -1.343909, -0.756005, -0.245094, 0.245094, 0.756005, 1.343909, 2.151946,
];
pub const CENTROIDS_4: [f32; 16] = [
    -2.732590, -2.069017, -1.618046, -1.256231, -0.942340, -0.656759, -0.388048, -0.128395,
    0.128395, 0.388048, 0.656759, 0.942340, 1.256231, 1.618046, 2.069017, 2.732590,
];
#[rustfmt::skip]
pub const CENTROIDS_8: [f32; 256] = [
    -4.603536, -4.186595, -3.925638, -3.731666, -3.575588, -3.444072, -3.329848, -3.228500,
    -3.137133, -3.053742, -2.976882, -2.905473, -2.838686, -2.775871, -2.716508, -2.660174,
    -2.606521, -2.555259, -2.506143, -2.458962, -2.413539, -2.369717, -2.327362, -2.286354,
    -2.246590, -2.207975, -2.170428, -2.133874, -2.098247, -2.063485, -2.029536, -1.996348,
    -1.963878, -1.932084, -1.900928, -1.870375, -1.840393, -1.810953, -1.782027, -1.753589,
    -1.725617, -1.698087, -1.670980, -1.644276, -1.617958, -1.592008, -1.566410, -1.541150,
    -1.516213, -1.491587, -1.467259, -1.443217, -1.419449, -1.395946, -1.372698, -1.349694,
    -1.326925, -1.304384, -1.282061, -1.259950, -1.238042, -1.216330, -1.194808, -1.173468,
    -1.152305, -1.131313, -1.110486, -1.089818, -1.069304, -1.048939, -1.028718, -1.008636,
    -0.988689, -0.968873, -0.949182, -0.929613, -0.910162, -0.890825, -0.871599, -0.852479,
    -0.833463, -0.814547, -0.795727, -0.777001, -0.758365, -0.739817, -0.721353, -0.702970,
    -0.684667, -0.666439, -0.648286, -0.630203, -0.612188, -0.594240, -0.576355, -0.558531,
    -0.540766, -0.523058, -0.505405, -0.487804, -0.470253, -0.452750, -0.435293, -0.417881,
    -0.400510, -0.383180, -0.365888, -0.348632, -0.331411, -0.314223, -0.297066, -0.279937,
    -0.262836, -0.245761, -0.228709, -0.211680, -0.194671, -0.177680, -0.160707, -0.143750,
    -0.126806, -0.109874, -0.092952, -0.076040, -0.059135, -0.042235, -0.025339, -0.008446,
     0.008446,  0.025339,  0.042235,  0.059135,  0.076040,  0.092952,  0.109874,  0.126806,
     0.143750,  0.160707,  0.177680,  0.194671,  0.211680,  0.228709,  0.245761,  0.262836,
     0.279937,  0.297066,  0.314223,  0.331411,  0.348632,  0.365888,  0.383180,  0.400510,
     0.417881,  0.435293,  0.452750,  0.470253,  0.487804,  0.505405,  0.523058,  0.540766,
     0.558531,  0.576355,  0.594240,  0.612188,  0.630203,  0.648286,  0.666439,  0.684667,
     0.702970,  0.721353,  0.739817,  0.758365,  0.777001,  0.795727,  0.814547,  0.833463,
     0.852479,  0.871599,  0.890825,  0.910162,  0.929613,  0.949182,  0.968873,  0.988689,
     1.008636,  1.028718,  1.048939,  1.069304,  1.089818,  1.110486,  1.131313,  1.152305,
     1.173468,  1.194808,  1.216330,  1.238042,  1.259950,  1.282061,  1.304384,  1.326925,
     1.349694,  1.372698,  1.395946,  1.419449,  1.443217,  1.467259,  1.491587,  1.516213,
     1.541150,  1.566410,  1.592008,  1.617958,  1.644276,  1.670980,  1.698087,  1.725617,
     1.753589,  1.782027,  1.810953,  1.840393,  1.870375,  1.900928,  1.932084,  1.963878,
     1.996348,  2.029536,  2.063485,  2.098247,  2.133874,  2.170428,  2.207975,  2.246590,
     2.286354,  2.327362,  2.369717,  2.413539,  2.458962,  2.506143,  2.555259,  2.606521,
     2.660174,  2.716508,  2.775871,  2.838686,  2.905473,  2.976882,  3.053742,  3.137133,
     3.228500,  3.329848,  3.444072,  3.575588,  3.731666,  3.925638,  4.186595,  4.603536
];

/// Scale all entries of a codebook by 1/sqrt(dim).
pub fn scale<const N: usize>(codebook: &[f32; N], dim: usize) -> [f32; N] {
    let scale = 1.0 / (dim as f32).sqrt();
    let mut scaled = [0.0; N];
    for (i, &c) in codebook.iter().enumerate() {
        scaled[i] = c * scale;
    }
    scaled
}

/// Compute decision thresholds for selecting codes from `codebook`.
///
/// Entry `i` is the midpoint between `codebook[i]` and `codebook[i + 1]`; any value above this
/// midpoint is closer to `codebook[i + 1]`. The final entry is padding set to `f32::INFINITY` so
/// that the thresholds may be stored in a power of two sized array.
///
/// Thresholds scale linearly with the codebook so they may be computed from the unscaled codebook
/// and compared against values scaled by `sqrt(dim)`.
pub fn thresholds<const N: usize>(codebook: &[f32; N]) -> [f32; N] {
    let mut thresholds = [f32::INFINITY; N];
    for (t, c) in thresholds.iter_mut().zip(codebook.windows(2)) {
        *t = (c[0] + c[1]) * 0.5;
    }
    thresholds
}

/// Select the closest code (index) for `value` given decision `thresholds` from [`thresholds`].
///
/// The selected code is the number of thresholds strictly less than `value`, so values exactly on
/// a midpoint select the lower code.
///
/// Small codebooks count the thresholds exceeded, which vectorizes well across values. Larger
/// codebooks use a branchless binary search; independent calls have no data dependency on one
/// another so the CPU can overlap the searches for several values.
#[inline(always)]
pub fn select_code<const N: usize>(thresholds: &[f32; N], value: f32) -> u8 {
    const { assert!(N.is_power_of_two() && N <= 256) };
    if N <= 16 {
        return thresholds[..N - 1]
            .iter()
            .map(|&t| u8::from(value > t))
            .sum();
    }
    let mut base = 0usize;
    let mut half = N / 2;
    while half > 0 {
        // base + half - 1 < N - 1, so the padding entry is never read.
        base += half * usize::from(value > thresholds[base + half - 1]);
        half /= 2;
    }
    base as u8
}

/// Select codes for each of `values` scaled by `scale`, writing them to `codes`.
///
/// This produces the same result as calling [`select_code`] on each scaled value. Large codebooks
/// are searched in three stages:
/// 1. Count against every `N / COARSE` threshold to choose a bucket. This vectorizes across values.
/// 2. Binary search within the bucket down to `LINEAR` candidate thresholds.
/// 3. Count against the remaining `LINEAR` thresholds, which vectorizes within each value.
#[inline(always)]
pub fn select_codes<const N: usize>(
    thresholds: &[f32; N],
    values: &[f32],
    scale: f32,
    codes: &mut [u8],
) {
    const COARSE: usize = 16;
    const LINEAR: usize = 8;

    if N <= 16 {
        for (c, v) in codes.iter_mut().zip(values) {
            *c = select_code(thresholds, *v * scale);
        }
        return;
    }

    const { assert!(N <= 16 || N / COARSE >= LINEAR) };
    let bucket = N / COARSE;
    let coarse: [f32; COARSE - 1] = std::array::from_fn(|k| thresholds[bucket * (k + 1) - 1]);
    for (c, v) in codes.iter_mut().zip(values) {
        let v = *v * scale;
        *c = coarse.iter().map(|&t| u8::from(v > t)).sum::<u8>() * bucket as u8;
    }
    for (c, v) in codes.iter_mut().zip(values) {
        let v = *v * scale;
        let mut base = *c as usize;
        // Invariant: thresholds[base - 1] < v <= thresholds[base + len - 1] (or the padding).
        let mut len = bucket;
        while len > LINEAR {
            len /= 2;
            base += len * usize::from(v > thresholds[base + len - 1]);
        }
        // v never exceeds the last of these thresholds, so counting all LINEAR is safe and keeps a
        // fixed power of two width.
        let row: &[f32; LINEAR] = thresholds[base..base + LINEAR].try_into().unwrap();
        base += row.iter().map(|&t| usize::from(v > t)).sum::<usize>();
        *c = base as u8;
    }
}

#[cfg(test)]
mod test {
    use super::*;

    /// Reference implementation: the index of the nearest codebook entry, preferring the lower
    /// index on ties.
    fn nearest<const N: usize>(codebook: &[f32; N], value: f32) -> u8 {
        let mut best = 0;
        for (i, &c) in codebook.iter().enumerate() {
            if (value - c).abs() < (value - codebook[best]).abs() {
                best = i;
            }
        }
        best as u8
    }

    fn check<const N: usize>(codebook: &[f32; N]) {
        let thresholds = thresholds(codebook);
        let lo = codebook[0] * 1.5;
        let hi = codebook[N - 1] * 1.5;
        let steps = 100_000;
        for i in 0..=steps {
            let v = lo + (hi - lo) * (i as f32 / steps as f32);
            assert_eq!(
                select_code(&thresholds, v),
                nearest(codebook, v),
                "value={v}"
            );
        }
        for (i, &c) in codebook.iter().enumerate() {
            assert_eq!(select_code(&thresholds, c) as usize, i);
        }
        // Values on a threshold select the lower code; anything above selects the upper code.
        for (i, &t) in thresholds[..N - 1].iter().enumerate() {
            assert_eq!(select_code(&thresholds, t) as usize, i, "threshold={t}");
            assert_eq!(
                select_code(&thresholds, t.next_up()) as usize,
                i + 1,
                "threshold={t}"
            );
        }
        let values = (0..=steps)
            .map(|i| lo + (hi - lo) * (i as f32 / steps as f32))
            .chain(codebook.iter().copied())
            .chain(thresholds[..N - 1].iter().flat_map(|&t| [t, t.next_up()]))
            .collect::<Vec<_>>();
        let mut codes = vec![0u8; values.len()];
        select_codes(&thresholds, &values, 1.0, &mut codes);
        for (&v, &c) in values.iter().zip(&codes) {
            assert_eq!(c, select_code(&thresholds, v), "value={v}");
        }
        assert_eq!(select_code(&thresholds, f32::NEG_INFINITY), 0);
        assert_eq!(select_code(&thresholds, f32::INFINITY) as usize, N - 1);
    }

    #[test]
    fn select_code_1() {
        check(&CENTROIDS_1);
    }

    #[test]
    fn select_code_2() {
        check(&CENTROIDS_2);
    }

    #[test]
    fn select_code_3() {
        check(&CENTROIDS_3);
    }

    #[test]
    fn select_code_4() {
        check(&CENTROIDS_4);
    }

    #[test]
    fn select_code_8() {
        check(&CENTROIDS_8);
    }
}
