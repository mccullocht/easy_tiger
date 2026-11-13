#[cfg(target_arch = "aarch64")]
mod aarch64;
mod scalar;
#[cfg(target_arch = "x86_64")]
mod x86_64;

use std::ops::Range;

use rand::{Rng, RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

/// Implementation of the Walsh-Hadamard transform used to rotate vectors.
///
/// Use [`Kernel::default()`] for the fastest kernel available on this host, or [`Kernel::all()`]
/// to enumerate every kernel that may be used on this host.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Kernel {
    Scalar,
    #[cfg(target_arch = "aarch64")]
    Neon,
    #[cfg(target_arch = "x86_64")]
    Avx512,
    #[cfg(target_arch = "x86_64")]
    Avx,
}

impl Kernel {
    /// Every kernel that could be used on this target, fastest first. `Scalar` is always last and
    /// is always usable; the others must be checked with `is_available()`.
    const CANDIDATES: &'static [Self] = &[
        #[cfg(target_arch = "aarch64")]
        Self::Neon,
        #[cfg(target_arch = "x86_64")]
        Self::Avx512,
        #[cfg(target_arch = "x86_64")]
        Self::Avx,
        Self::Scalar,
    ];

    /// True if this kernel may be used on this host.
    fn is_available(&self) -> bool {
        match self {
            Self::Scalar => true,
            #[cfg(target_arch = "aarch64")]
            Self::Neon => true,
            #[cfg(target_arch = "x86_64")]
            Self::Avx512 => is_x86_feature_detected!("avx512f"),
            #[cfg(target_arch = "x86_64")]
            Self::Avx => is_x86_feature_detected!("avx"),
        }
    }

    /// All kernels that may be used on this host, fastest first.
    ///
    /// Accelerated kernels appear only if the host supports them; `Scalar` is always last.
    pub fn all() -> Vec<Self> {
        Self::CANDIDATES
            .iter()
            .copied()
            .filter(Self::is_available)
            .collect()
    }

    /// Walsh-Hadamard Transform vector `v` with `signs` random sign flips.
    ///
    /// `signs` are expected to contain 0 or 1 << 31 and will be XORed against floating point values to
    /// flip the sign. Sign flips are applied before the butterfly transforms.
    ///
    /// *Panics* if `v.len()` is not a power of two, or if `v.len() != signs.len()`
    pub fn walsh_hadamard_transform(&self, v: &mut [f32], signs: &[u32]) {
        match self {
            Self::Scalar => scalar::walsh_hadamard_transform(v, signs),
            #[cfg(target_arch = "aarch64")]
            Self::Neon => aarch64::neon_walsh_hadamard_transform(v, signs),
            // Safety: these are only constructed after the matching `is_x86_feature_detected!`
            // check succeeded, see `Kernel::all` above.
            #[cfg(target_arch = "x86_64")]
            Self::Avx512 => unsafe { x86_64::avx512_walsh_hadamard_transform(v, signs) },
            #[cfg(target_arch = "x86_64")]
            Self::Avx => unsafe { x86_64::avx_walsh_hadamard_transform(v, signs) },
        }
    }
}

impl Default for Kernel {
    /// The fastest kernel available on this host.
    fn default() -> Self {
        // `CANDIDATES` is ordered fastest first and ends with `Scalar`, which is always available.
        Self::CANDIDATES
            .iter()
            .copied()
            .find(Self::is_available)
            .expect("Scalar kernel is always available")
    }
}

impl std::fmt::Display for Kernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(self, f)
    }
}

/// Random in-place shuffle of the vector.
///
/// This is necessary when vector size is not a power of two since we will only shuffle within
/// power of two sized blocks.
///
/// The permutation is stored as the swap log of a Fisher-Yates pass over `0..dims`: `swaps[i]`
/// is the partner index that position `i` was exchanged with. Replaying `v.swap(i, swaps[i])`
/// for `i` in `0..dims` reproduces the permutation in place with no scratch buffer; replaying
/// the swaps in reverse would invert it.
struct Shuffle {
    swaps: Vec<u32>,
}

impl Shuffle {
    fn new(dims: usize, rng: &mut impl Rng) -> Self {
        // Fisher-Yates from low to high, recording each position's swap partner. Drawing from
        // `i..dims` (rather than the textbook `0..=i` high-to-low) keeps `apply` a simple
        // forward loop while remaining uniform over all permutations.
        let swaps = (0..dims)
            .map(|i| rng.random_range(i..dims) as u32)
            .collect();
        Self { swaps }
    }

    fn apply(&self, v: &mut [f32]) {
        for (i, &j) in self.swaps.iter().enumerate() {
            v.swap(i, j as usize);
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
    kernel: Kernel,
    /// Vector dimensions are shuffled in the event that there are multiple blocks.
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
        Self::with_kernel(dims, seed, Kernel::default())
    }

    /// Like [`Rotator::new()`] but uses `kernel` to compute the transforms instead of the fastest
    /// kernel available on this host. All kernels produce equivalent results; this is intended for
    /// testing and benchmarking.
    pub fn with_kernel(dims: usize, seed: u64, kernel: Kernel) -> Self {
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

        Self {
            kernel,
            shuffle,
            blocks,
        }
    }

    /// Rotate `v` in place for quantization.
    ///
    /// This applies sign flips, then permutation, then block diagonal Hadamard transforms.
    pub fn rotate(&self, v: &mut [f32]) {
        if let Some(p) = self.shuffle.as_ref() {
            p.apply(v);
        }

        for block in self.blocks.iter() {
            self.kernel
                .walsh_hadamard_transform(&mut v[block.dims.clone()], &block.sign);
        }
    }

    /// Copy `v` and [`rotate`](Self::rotate) the copy for quantization.
    pub fn rotate_copy(&self, v: impl AsRef<[f32]>) -> Vec<f32> {
        let mut rotated = v.as_ref().to_vec();
        self.rotate(&mut rotated);
        rotated
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
        use rand::SeedableRng;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        (0..dims).map(|_| rng.random_range(-1.0f32..=1.0)).collect()
    }

    #[test]
    fn preserves_l2_norm() {
        let rotator = Rotator::new(256, 7);
        let v = make_vec(256, 4);
        let rotated = rotator.rotate_copy(&v);
        assert!(abs_diff_eq!(l2_norm(&v), l2_norm(&rotated), epsilon = 1e-4));
    }

    #[test]
    fn preserves_inner_product() {
        let rotator = Rotator::new(128, 13);
        let a = make_vec(128, 5);
        let b = make_vec(128, 6);
        let ra = rotator.rotate_copy(&a);
        let rb = rotator.rotate_copy(&b);
        assert!(abs_diff_eq!(dot(&a, &b), dot(&ra, &rb), epsilon = 1e-4));
    }

    #[test]
    fn preserves_l2_norm_non_power_of_two() {
        // 100 = 64 + 32 + 4, so the dimensions are shuffled before the block transforms.
        let rotator = Rotator::new(100, 7);
        let v = make_vec(100, 4);
        let rotated = rotator.rotate_copy(&v);
        assert!(abs_diff_eq!(l2_norm(&v), l2_norm(&rotated), epsilon = 1e-4));
    }

    #[test]
    fn shuffle_apply_is_a_permutation() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5eed);
        let shuffle = Shuffle::new(100, &mut rng);

        let mut v: Vec<f32> = (0..100).map(|i| i as f32).collect();
        shuffle.apply(&mut v);

        assert_ne!(
            v,
            (0..100).map(|i| i as f32).collect::<Vec<_>>(),
            "not identity"
        );

        let mut sorted = v.clone();
        sorted.sort_by(f32::total_cmp);
        assert_eq!(
            sorted,
            (0..100).map(|i| i as f32).collect::<Vec<_>>(),
            "every element appears exactly once"
        );
    }

    #[test]
    fn deterministic_same_seed() {
        let v = make_vec(64, 10);
        let r1 = Rotator::new(64, 123).rotate_copy(&v);
        let r2 = Rotator::new(64, 123).rotate_copy(&v);
        assert_eq!(r1, r2, "same seed must produce identical results");
    }

    #[test]
    fn different_seeds_differ() {
        let v = make_vec(64, 11);
        let r1 = Rotator::new(64, 1).rotate_copy(&v);
        let r2 = Rotator::new(64, 2).rotate_copy(&v);
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
        Kernel::Scalar.walsh_hadamard_transform(&mut v, &signs);
        Kernel::Scalar.walsh_hadamard_transform(&mut v, &signs);
        assert!(
            original
                .iter()
                .zip(&v)
                .all(|(a, b)| abs_diff_eq!(a, b, epsilon = 1e-6))
        );
    }

    /// The accelerated kernel available on the current hardware, e.g. `Neon` on aarch64 or
    /// `Avx512` on x86_64 with the necessary feature present. `None` if only `Scalar` is
    /// available (e.g. an x86_64 host without `avx512f`).
    fn accelerated_kernels() -> Vec<Kernel> {
        Kernel::all()
            .into_iter()
            .filter(|k| *k != Kernel::Scalar)
            .collect()
    }

    fn make_signs(dims: usize, seed: u64) -> Vec<u32> {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        (0..dims)
            .map(|_| if rng.random_bool(0.5) { 0 } else { 1u32 << 31 })
            .collect()
    }

    /// Change-detector test: any accelerated kernel (Neon, Avx512, ...) must produce bit-for-bit
    /// equivalent (within float epsilon) results to the scalar reference implementation, across
    /// every block-size code path (below 64, exactly 64, and multiple 64-blocks).
    #[test]
    fn accelerated_matches_scalar() {
        // If no accelerated kernel is available on this hardware there is nothing to compare.
        for accelerated in accelerated_kernels() {
            for &dims in &[1usize, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] {
                for seed in 0..5u64 {
                    let v = make_vec(dims, seed);
                    let signs = make_signs(dims, seed ^ 0xdead_beef);

                    let mut scalar = v.clone();
                    let mut accel = v.clone();
                    Kernel::Scalar.walsh_hadamard_transform(&mut scalar, &signs);
                    accelerated.walsh_hadamard_transform(&mut accel, &signs);
                    for (i, (a, b)) in scalar.iter().zip(&accel).enumerate() {
                        assert!(
                            abs_diff_eq!(a, b, epsilon = 1e-4),
                            "{accelerated} dims={dims} seed={seed} mismatch at {i}: scalar={a} accel={b}"
                        );
                    }
                }
            }
        }
    }
}
