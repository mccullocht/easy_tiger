//! Locally adaptive Vector Quantization (LVQ): https://arxiv.org/pdf/2304.04759

// OneLevel:
// Q(x; B, l, u) = d ((x -l) / d + 0.5) + l; where delta = (u - l) / (2^B - 1)
// u = max xj; l = min xj
//
// TwoLevel
// r = x - Q(x)
// Qres(r; B') = Q(x; B', -d/2, d/2)

/// Compute single-level LVQ, with `B` as the number of quantized bits.
///
/// Returns min, max, and an iterator over quantizatized values with a single entry per byte.
fn lvq1<const B: usize>(v: &[f32]) -> (f32, f32, impl ExactSizeIterator<Item = u8> + '_) {
    let (l, u) = v
        .iter()
        .fold((f32::MAX, f32::MIN), |(l, u), d| (d.min(l), d.max(u)));
    let delta = (u - l) / ((1 << B) - 1) as f32;
    let it = v.iter().map(move |x| ((x - l) / delta).round() as u8);
    (l, u, it)
}

/// Invert one-level LVQ to produce an f32 vector.
///
/// The `B` value must be the same as was used in the original [lvq1] call.
fn unlvq1<'q, const B: usize>(
    l: f32,
    u: f32,
    quantized: impl ExactSizeIterator<Item = u8> + 'q,
) -> impl ExactSizeIterator<Item = f32> + 'q {
    let delta = (u - l) / ((1 << B) - 1) as f32;
    quantized.map(move |q| (q as f32 * delta) + l)
}

/// Compute two-level LVQ, with `B1` primary vector bits and `B2` residual bits.
///
/// Returns min, max, and an iterator over (primary, residual) quantized values
fn lvq2<const B1: usize, const B2: usize>(
    v: &[f32],
) -> (f32, f32, impl ExactSizeIterator<Item = (u8, u8)> + '_) {
    let (l, u) = v
        .iter()
        .fold((f32::MAX, f32::MIN), |(l, u), d| (d.min(l), d.max(u)));
    let delta = (u - l) / ((1 << B1) - 1) as f32;

    let res_l = -delta / 2.0;
    let res_delta = delta / ((1 << B2) - 1) as f32;

    let it = v.iter().map(move |x| {
        let q = ((x - l) / delta).round();
        let res = *x - ((q * delta) + l);
        let res_q = ((res - res_l) / res_delta).round();
        (q as u8, res_q as u8)
    });
    (l, u, it)
}

/// Invert two-level LVQ to produce an f32 vector.
///
/// The `B1` and `B2` values must be the same as was used in the original [lvq2] call.
fn unlvq2<'q, const B1: usize, const B2: usize>(
    l: f32,
    u: f32,
    quantized: impl ExactSizeIterator<Item = (u8, u8)> + 'q,
) -> impl ExactSizeIterator<Item = f32> + 'q {
    let delta = (u - l) / ((1 << B1) - 1) as f32;
    let res_l = -delta / 2.0;
    let res_delta = delta / ((1 << B2) - 1) as f32;
    quantized.map(move |(q, r)| {
        let uq = q as f32 * delta + l;
        let ur = r as f32 * res_delta + res_l;
        uq + ur
    })
}

#[cfg(test)]
mod test {
    use crate::vectors::lvq::{lvq1, lvq2, unlvq1, unlvq2};

    #[test]
    fn lvq1_4() {
        let vec = [-0.5f32, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4];
        let (l, u, it) = lvq1::<4>(&vec);
        assert_eq!(l, -0.5f32);
        assert_eq!(u, 0.4f32);
        let q = it.collect::<Vec<_>>();
        assert_eq!(&q, &[0, 2, 3, 5, 7, 8, 10, 12, 13, 15]);
        assert_eq!(
            unlvq1::<4>(l, u, q.into_iter()).collect::<Vec<_>>(),
            &[
                -0.5f32,
                -0.38,
                -0.32,
                -0.20000002,
                -0.08000001,
                -0.02000001,
                0.099999964,
                0.21999997,
                0.27999997,
                0.39999998
            ]
        );
    }

    #[test]
    fn lvq1_8() {
        let vec = [-0.5f32, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4];
        let (l, u, it) = lvq1::<8>(&vec);
        assert_eq!(l, -0.5f32);
        assert_eq!(u, 0.4f32);
        let q = it.collect::<Vec<_>>();
        assert_eq!(q, &[0, 28, 57, 85, 113, 142, 170, 198, 227, 255]);
        assert_eq!(
            unlvq1::<8>(l, u, q.into_iter()).collect::<Vec<_>>(),
            &[
                -0.5f32,
                -0.40117645,
                -0.29882354,
                -0.19999999,
                -0.10117647,
                0.0011764765,
                0.100000024,
                0.19882351,
                0.3011765,
                0.39999998
            ]
        );
    }

    #[test]
    fn lvq2_4_8() {
        let vec = [-0.5f32, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4];
        let (l, u, it) = lvq2::<4, 8>(&vec);
        assert_eq!(l, -0.5f32);
        assert_eq!(u, 0.4f32);
        let q = it.collect::<Vec<_>>();
        assert_eq!(
            q,
            &[
                (0, 128),
                (2, 42),
                (3, 212),
                (5, 128),
                (7, 43),
                (8, 213),
                (10, 128),
                (12, 43),
                (13, 213),
                (15, 128)
            ]
        );
        // TODO: very specific rounding error in dequantization; investigate.
        assert_eq!(
            unlvq2::<4, 8>(l, u, q.into_iter()).collect::<Vec<_>>(),
            &[
                -0.49988234,
                -0.40011764,
                -0.30011764,
                -0.19988237,
                -0.099882364,
                0.000117635354,
                0.10011761,
                0.20011762,
                0.3001176,
                0.40011764
            ]
        );
    }

    #[test]
    fn lvq2_8_8() {
        let vec = [-0.5f32, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4];
        let (l, u, it) = lvq2::<8, 8>(&vec);
        assert_eq!(l, -0.5f32);
        assert_eq!(u, 0.4f32);
        let q = it.collect::<Vec<_>>();
        assert_eq!(
            q,
            &[
                (0, 128),
                (28, 212),
                (57, 42),
                (85, 127),
                (113, 212),
                (142, 42),
                (170, 127),
                (198, 213),
                (227, 42),
                (255, 128),
            ]
        );
        assert_eq!(
            unlvq2::<8, 8>(l, u, q.into_iter()).collect::<Vec<_>>(),
            &[
                -0.4999931,
                -0.4000069,
                -0.30000693,
                -0.2000069,
                -0.10000692,
                -6.914488e-6,
                0.0999931,
                0.2000069,
                0.2999931,
                0.4000069
            ]
        );
    }
}
