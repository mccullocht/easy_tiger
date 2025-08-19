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

#[cfg(test)]
mod test {
    use crate::vectors::lvq::{lvq1, lvq2};

    #[test]
    fn lvq1_4() {
        let vec = [-0.5f32, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4];
        let (min, max, it) = lvq1::<4>(&vec);
        assert_eq!(min, -0.5f32);
        assert_eq!(max, 0.4f32);
        assert_eq!(it.collect::<Vec<_>>(), &[0, 2, 3, 5, 7, 8, 10, 12, 13, 15]);
    }

    #[test]
    fn lvq1_8() {
        let vec = [-0.5f32, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4];
        let (min, max, it) = lvq1::<8>(&vec);
        assert_eq!(min, -0.5f32);
        assert_eq!(max, 0.4f32);
        assert_eq!(
            it.collect::<Vec<_>>(),
            &[0, 28, 57, 85, 113, 142, 170, 198, 227, 255]
        );
    }

    #[test]
    fn lvq2_4_8() {
        let vec = [-0.5f32, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4];
        let (min, max, it) = lvq2::<4, 8>(&vec);
        assert_eq!(min, -0.5f32);
        assert_eq!(max, 0.4f32);
        assert_eq!(
            it.collect::<Vec<_>>(),
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
    }

    #[test]
    fn lvq2_8_8() {
        let vec = [-0.5f32, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4];
        let (min, max, it) = lvq2::<8, 8>(&vec);
        assert_eq!(min, -0.5f32);
        assert_eq!(max, 0.4f32);
        assert_eq!(
            it.collect::<Vec<_>>(),
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
    }
}
