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
