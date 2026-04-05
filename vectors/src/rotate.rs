use std::ops::Range;

use rand::{Rng, SeedableRng, seq::SliceRandom};
use rand_xoshiro::Xoshiro256PlusPlus;

/// Implement orthogonal rotation of a vector for quantization to preserve distances and inner
/// products while changing the distribution of the vector's components to minimize quantization
/// error.
///
/// Implement fast Hadamard transform, permutation, and sign flip
pub struct Rotator {
    forward_permutation: Vec<usize>,
    backward_permutation: Vec<usize>,
    sign_flips: Vec<f32>,
    blocks: Vec<Range<usize>>,
}

impl Rotator {
    /// Create a new rotator for `dims` with a random seed. The seed must remain fixed for all
    /// vectors that will be compared with each other.
    ///
    /// If `dims` is a power of 2 then we will do a single Hardamard transform. If not a block
    /// diagonal Hardamard transform will be used on blocks of dimensions dictated by a binary
    /// decomposition of `dims`.
    pub fn new(dims: usize, seed: u64) -> Self {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut forward_permutation = (0..dims).collect::<Vec<usize>>();
        forward_permutation.shuffle(&mut rng);
        let mut backward_permutation = vec![0; dims];
        for (i, &j) in forward_permutation.iter().enumerate() {
            backward_permutation[j] = i;
        }
        let sign_flips = (0..dims)
            .map(|_| if rng.random_bool(0.5) { 1.0 } else { -1.0 })
            .collect::<Vec<f32>>();

        let mut blocks: Vec<Range<usize>> = vec![];
        let mut d = dims;
        while d > 0 {
            let start = blocks.last().map(|b| b.end).unwrap_or(0);
            let len = 1usize << (63 - d.leading_zeros());
            blocks.push(start..(start + len));
            d ^= len;
        }

        Self {
            forward_permutation,
            backward_permutation,
            sign_flips,
            blocks,
        }
    }

    /// Rotate forward for quantization.
    ///
    /// This applies sign flips, then permutation, then block diagonal Hadamard transforms.
    pub fn forward(&self, v: &[f32]) -> Vec<f32> {
        let signed = self
            .sign_flips
            .iter()
            .zip(v.iter())
            .map(|(s, &x)| s * x)
            .collect::<Vec<f32>>();
        let mut rotated = self
            .forward_permutation
            .iter()
            .map(|&i| signed[i])
            .collect::<Vec<f32>>();

        for block in self.blocks.iter() {
            Self::walsh_hadamard_transform(&mut rotated[block.clone()]);
        }

        rotated
    }

    /// Rotate backward for dequantization.
    ///
    /// Thie applies block diagonal Hadamard transforms, then inverse permutation, then sign flips.
    pub fn backward(&self, v: &[f32]) -> Vec<f32> {
        let mut tmp = v.to_vec();
        for block in self.blocks.iter() {
            Self::walsh_hadamard_transform(&mut tmp[block.clone()]);
        }

        let mut b = self
            .backward_permutation
            .iter()
            .map(|&i| tmp[i])
            .collect::<Vec<f32>>();
        for (i, v) in b.iter_mut().enumerate() {
            *v *= self.sign_flips[i];
        }
        b
    }

    fn walsh_hadamard_transform(v: &mut [f32]) {
        let n = v.len();
        assert!(
            n.is_power_of_two(),
            "Hadamard transform requires power of 2 length"
        );
        let mut h = 1;
        while h < n {
            for i in (0..n).step_by(h * 2) {
                for j in 0..h {
                    let x = v[i + j];
                    let y = v[i + j + h];
                    v[i + j] = x + y;
                    v[i + j + h] = x - y;
                }
            }
            h *= 2;
        }

        // Normalize by 1/sqrt(n) to preserve distances and inner products
        let scale = 1.0 / (n as f32).sqrt();
        for x in v.iter_mut() {
            *x *= scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::abs_diff_eq;

    use super::*;

    fn l2_norm(v: &[f32]) -> f32 {
        v.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    fn make_vec(dims: usize, seed: u64) -> Vec<f32> {
        use rand::{Rng, SeedableRng};
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        (0..dims).map(|_| rng.random_range(-1.0f32..=1.0)).collect()
    }

    #[test]
    fn round_trip_power_of_two() {
        let rotator = Rotator::new(128, 42);
        let v = make_vec(128, 1);
        let rotated = rotator.forward(&v);
        let recovered = rotator.backward(&rotated);
        assert!(v.iter().zip(&recovered).all(|(a, b)| abs_diff_eq!(a, b, epsilon = 1e-5)));
    }

    #[test]
    fn round_trip_non_power_of_two() {
        // 192 = 128 + 64, so two blocks
        let rotator = Rotator::new(192, 42);
        let v = make_vec(192, 2);
        let rotated = rotator.forward(&v);
        let recovered = rotator.backward(&rotated);
        assert!(v.iter().zip(&recovered).all(|(a, b)| abs_diff_eq!(a, b, epsilon = 1e-5)));
    }

    #[test]
    fn round_trip_arbitrary_dims() {
        // 100 = 64 + 32 + 4, so three blocks
        let rotator = Rotator::new(100, 99);
        let v = make_vec(100, 3);
        let rotated = rotator.forward(&v);
        let recovered = rotator.backward(&rotated);
        assert!(v.iter().zip(&recovered).all(|(a, b)| abs_diff_eq!(a, b, epsilon = 1e-5)));
    }

    #[test]
    fn preserves_l2_norm() {
        let rotator = Rotator::new(256, 7);
        let v = make_vec(256, 4);
        let rotated = rotator.forward(&v);
        assert!(abs_diff_eq!(l2_norm(&v), l2_norm(&rotated), epsilon = 1e-4));
    }

    #[test]
    fn preserves_inner_product() {
        let rotator = Rotator::new(128, 13);
        let a = make_vec(128, 5);
        let b = make_vec(128, 6);
        let ra = rotator.forward(&a);
        let rb = rotator.forward(&b);
        assert!(abs_diff_eq!(dot(&a, &b), dot(&ra, &rb), epsilon = 1e-4));
    }

    #[test]
    fn deterministic_same_seed() {
        let v = make_vec(64, 10);
        let r1 = Rotator::new(64, 123).forward(&v);
        let r2 = Rotator::new(64, 123).forward(&v);
        assert_eq!(r1, r2, "same seed must produce identical results");
    }

    #[test]
    fn different_seeds_differ() {
        let v = make_vec(64, 11);
        let r1 = Rotator::new(64, 1).forward(&v);
        let r2 = Rotator::new(64, 2).forward(&v);
        assert_ne!(r1, r2, "different seeds should produce different rotations");
    }

    #[test]
    fn blocks_power_of_two_is_single_block() {
        let rotator = Rotator::new(64, 0);
        assert_eq!(rotator.blocks, vec![0..64]);
    }

    #[test]
    fn blocks_non_power_of_two_decomposition() {
        // 192 = 128 + 64
        let rotator = Rotator::new(192, 0);
        assert_eq!(rotator.blocks, vec![0..128, 128..192]);

        // 100 = 64 + 32 + 4
        let rotator = Rotator::new(100, 0);
        assert_eq!(rotator.blocks, vec![0..64, 64..96, 96..100]);
    }

    #[test]
    fn walsh_hadamard_self_inverse() {
        // Applying WHT twice should recover the original: each application normalizes by
        // 1/sqrt(n), so two applications give 1/n * n*I = I.
        let mut v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let original = v.clone();
        Rotator::walsh_hadamard_transform(&mut v);
        Rotator::walsh_hadamard_transform(&mut v);
        assert!(original.iter().zip(&v).all(|(a, b)| abs_diff_eq!(a, b, epsilon = 1e-6)));
    }
}
