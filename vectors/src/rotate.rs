use std::ops::Range;

use rand::{Rng, SeedableRng, seq::SliceRandom};
use rand_xoshiro::Xoshiro256PlusPlus;

/// Random shuffle of the vector: forward (encode) and backward (decode).
///
/// This is necessary when vector size is not a power of two since we will only shuffle within
/// power of two sized blocks.
struct Shuffle {
    forward: Vec<u32>,
    backward: Vec<u32>,
}

impl Shuffle {
    fn new(dims: usize, rng: &mut impl Rng) -> Self {
        let mut forward = (0..dims as u32).collect::<Vec<u32>>();
        forward.shuffle(rng);
        let mut backward = vec![0; dims];
        for (i, &j) in forward.iter().enumerate() {
            backward[j as usize] = i as u32;
        }
        Self { forward, backward }
    }

    fn forward(&self, unpermuted: &[f32], permuted: &mut [f32]) {
        for (&i, o) in self.forward.iter().zip(permuted.iter_mut()) {
            *o = unpermuted[i as usize];
        }
    }

    fn backward(&self, permuted: &[f32], unpermuted: &mut [f32]) {
        for (&i, o) in self.backward.iter().zip(unpermuted.iter_mut()) {
            *o = permuted[i as usize];
        }
    }
}

struct Block {
    dims: Range<usize>,
    sign: Vec<u32>,
}

impl Block {
    fn new(dims: Range<usize>, rng: &mut impl Rng) -> Self {
        let sign = dims
            .clone()
            .map(|_| if rng.random_bool(0.5) { 0 } else { 1 << 31 })
            .collect();
        Self { dims, sign }
    }
}

/// Implement orthogonal rotation of a vector for quantization to preserve distances and inner
/// products while changing the distribution of the vector's components to minimize quantization
/// error.
pub struct Rotator {
    /// Vector dimensions are shuffled in the event that there are multiple blocks.
    // XXX should rep be an enum?
    shuffle: Option<Shuffle>,
    blocks: Vec<Block>,
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
        let shuffle = if !dims.is_power_of_two() {
            Some(Shuffle::new(dims, &mut rng))
        } else {
            None
        };

        let mut blocks: Vec<Block> = vec![];
        let mut d = dims;
        while d > 0 {
            let start = blocks.last().map(|b| b.dims.end).unwrap_or(0);
            let len = 1usize << (63 - d.leading_zeros());
            blocks.push(Block::new(start..(start + len), &mut rng));
            d ^= len;
        }

        Self { shuffle, blocks }
    }

    /// Rotate forward for quantization.
    ///
    /// This applies sign flips, then permutation, then block diagonal Hadamard transforms.
    pub fn forward(&self, v: &[f32]) -> Vec<f32> {
        let mut rotated = if let Some(p) = self.shuffle.as_ref() {
            let mut r = vec![0.0f32; v.len()];
            p.forward(v, &mut r);
            r
        } else {
            v.to_vec()
        };

        for block in self.blocks.iter() {
            Self::walsh_hadamard_transform::<true>(&mut rotated[block.dims.clone()], &block.sign);
        }

        rotated
    }

    /// Rotate backward for dequantization.
    ///
    /// Thie applies block diagonal Hadamard transforms, then inverse permutation, then sign flips.
    pub fn backward(&self, v: &[f32]) -> Vec<f32> {
        let mut tmp = v.to_vec();
        for block in self.blocks.iter() {
            Self::walsh_hadamard_transform::<false>(&mut tmp[block.dims.clone()], &block.sign);
        }

        let b = if let Some(p) = self.shuffle.as_ref() {
            let mut r = tmp.clone();
            p.backward(&tmp, &mut r);
            r
        } else {
            tmp
        };
        b
    }

    fn walsh_hadamard_transform<const F: bool>(v: &mut [f32], signs: &[u32]) {
        assert!(
            v.len().is_power_of_two(),
            "Hadamard transform requires power of 2 length"
        );
        if v.len() < 64 {
            if F {
                // Forward must sign flip first.
                for (&s, v) in signs.iter().zip(v.iter_mut()) {
                    *v = f32::from_bits(v.to_bits() ^ s);
                }
            }
            Self::wht_block::<1>(v)
        } else {
            // Perform the early strides of the block transformation together in 64 dimension chunks
            // in an effort to improve locality. v.len() is a power of 2 and there are at least 64
            // entries, so there will be no tail entries.
            let blocks = v.as_chunks_mut::<64>().0;
            let sblocks = signs.as_chunks::<64>().0;
            if F {
                for (b, s) in blocks.iter_mut().zip(sblocks.iter()) {
                    Self::wht_fixed_block::<true, 64>(b, s);
                }
            } else {
                for (b, s) in blocks.iter_mut().zip(sblocks.iter()) {
                    Self::wht_fixed_block::<false, 64>(b, s);
                }
            }
            // Continue butterfly transformation at block size and beyond.
            Self::wht_block::<64>(v);
        }

        // Normalize by 1/sqrt(n) to preserve distances and inner products.
        // For backwards transformation invert the sign flip here too.
        let scale = 1.0 / (v.len() as f32).sqrt();
        if F {
            for x in v.iter_mut() {
                *x *= scale;
            }
        } else {
            for (&s, x) in signs.iter().zip(v.iter_mut()) {
                *x *= f32::from_bits(scale.to_bits() ^ s);
            }
        }
    }

    #[inline]
    fn wht_block<const S: usize>(block: &mut [f32]) {
        let n = block.len();
        assert!(
            n.is_power_of_two(),
            "Hadamard transform requires power of 2 length"
        );
        let mut h = S;
        while h < n {
            for i in (0..n).step_by(h * 2) {
                for j in 0..h {
                    let x = block[i + j];
                    let y = block[i + j + h];
                    block[i + j] = x + y;
                    block[i + j + h] = x - y;
                }
            }
            h *= 2;
        }
    }

    /// Initial base Walsh-Hadamard Transform over a fixed size block.
    ///
    /// This includes the sign flips that are needed before the operation begins if `F` is true.
    #[inline]
    fn wht_fixed_block<const F: bool, const N: usize>(block: &mut [f32; N], signs: &[u32; N]) {
        if F {
            for (&s, v) in signs.iter().zip(block.iter_mut()) {
                *v = f32::from_bits(v.to_bits() ^ s);
            }
        }

        let mut h = 1;
        while h < N {
            for i in (0..N).step_by(h * 2) {
                for j in 0..h {
                    let x = block[i + j];
                    let y = block[i + j + h];
                    block[i + j] = x + y;
                    block[i + j + h] = x - y;
                }
            }
            h *= 2;
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
        assert!(
            v.iter()
                .zip(&recovered)
                .all(|(a, b)| abs_diff_eq!(a, b, epsilon = 1e-5))
        );
    }

    #[test]
    fn round_trip_non_power_of_two() {
        // 192 = 128 + 64, so two blocks
        let rotator = Rotator::new(192, 42);
        let v = make_vec(192, 2);
        let rotated = rotator.forward(&v);
        let recovered = rotator.backward(&rotated);
        assert!(
            v.iter()
                .zip(&recovered)
                .all(|(a, b)| abs_diff_eq!(a, b, epsilon = 1e-5))
        );
    }

    #[test]
    fn round_trip_arbitrary_dims() {
        // 100 = 64 + 32 + 4, so three blocks
        let rotator = Rotator::new(100, 99);
        let v = make_vec(100, 3);
        let rotated = rotator.forward(&v);
        let recovered = rotator.backward(&rotated);
        assert!(
            v.iter()
                .zip(&recovered)
                .all(|(a, b)| abs_diff_eq!(a, b, epsilon = 1e-5))
        );
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

    fn block_dims(rotator: &Rotator) -> Vec<Range<usize>> {
        rotator.blocks.iter().map(|b| b.dims.clone()).collect()
    }

    #[test]
    fn blocks_power_of_two_is_single_block() {
        let rotator = Rotator::new(64, 0);
        assert_eq!(block_dims(&rotator), vec![0..64]);
    }

    #[test]
    fn blocks_non_power_of_two_decomposition() {
        // 192 = 128 + 64
        let rotator = Rotator::new(192, 0);
        assert_eq!(block_dims(&rotator), vec![0..128, 128..192]);

        // 100 = 64 + 32 + 4
        let rotator = Rotator::new(100, 0);
        assert_eq!(block_dims(&rotator), vec![0..64, 64..96, 96..100]);
    }

    #[test]
    fn walsh_hadamard_self_inverse() {
        // Applying WHT twice should recover the original: each application normalizes by
        // 1/sqrt(n), so two applications give 1/n * n*I = I.
        let mut v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let signs = vec![0u32; v.len()];
        let original = v.clone();
        Rotator::walsh_hadamard_transform::<true>(&mut v, &signs);
        Rotator::walsh_hadamard_transform::<true>(&mut v, &signs);
        assert!(
            original
                .iter()
                .zip(&v)
                .all(|(a, b)| abs_diff_eq!(a, b, epsilon = 1e-6))
        );
    }
}
