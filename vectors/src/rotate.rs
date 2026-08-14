#[cfg(target_arch = "aarch64")]
mod aarch64;
mod scalar;
#[cfg(target_arch = "x86_64")]
mod x86_64;

use std::ops::Range;

use rand::{Rng, SeedableRng, seq::SliceRandom};
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
}

impl Kernel {
    /// All kernels that may be used on this host, in no particular order.
    ///
    /// `Scalar` is always included; accelerated kernels appear only if the host supports them.
    pub fn all() -> Vec<Self> {
        let mut kernels = vec![Self::Scalar];
        #[cfg(target_arch = "aarch64")]
        kernels.push(Self::Neon);
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx512f") {
            kernels.push(Self::Avx512);
        }
        kernels
    }

    /// Walsh-Hadamard Transform vector `v` with `signs` random sign flips.
    ///
    /// `signs` are expected to contain 0 or 1 << 31 and will be XORed against floating point values to
    /// flip the sign.
    ///
    /// `F` is true if this is a forward transform and false if this is a backward transform.
    /// This determines whether the signs are applied before or after the butterfly transforms.
    ///
    /// *Panics* if `v.len()` is not a power of two, or if `v.len() != signs.len()`
    pub fn walsh_hadamard_transform<const F: bool>(&self, v: &mut [f32], signs: &[u32]) {
        match self {
            Self::Scalar => scalar::walsh_hadamard_transform::<F>(v, signs),
            #[cfg(target_arch = "aarch64")]
            Self::Neon => aarch64::neon_walsh_hadamard_transform::<F>(v, signs),
            // Safety: only constructed after `is_x86_feature_detected!("avx512f")` succeeded,
            // see `Kernel::preferred` below.
            #[cfg(target_arch = "x86_64")]
            Self::Avx512 => unsafe { x86_64::avx512_walsh_hadamard_transform::<F>(v, signs) },
        }
    }
}

impl Default for Kernel {
    #[cfg(target_arch = "aarch64")]
    fn default() -> Self {
        Self::Neon
    }

    #[cfg(target_arch = "x86_64")]
    fn default() -> Self {
        if is_x86_feature_detected!("avx512f") {
            Self::Avx512
        } else {
            Self::Scalar
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    fn default() -> Self {
        Self::Scalar
    }
}

impl std::fmt::Display for Kernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(self, f)
    }
}

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
    kernel: Kernel,
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
            self.kernel
                .walsh_hadamard_transform::<true>(&mut rotated[block.dims.clone()], &block.sign);
        }

        rotated
    }

    /// Rotate backward for dequantization.
    ///
    /// Thie applies block diagonal Hadamard transforms, then inverse permutation, then sign flips.
    pub fn backward(&self, v: &[f32]) -> Vec<f32> {
        let mut tmp = v.to_vec();
        for block in self.blocks.iter() {
            self.kernel
                .walsh_hadamard_transform::<false>(&mut tmp[block.dims.clone()], &block.sign);
        }

        if let Some(p) = self.shuffle.as_ref() {
            let mut r = tmp.clone();
            p.backward(&tmp, &mut r);
            r
        } else {
            tmp
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
        Kernel::Scalar.walsh_hadamard_transform::<true>(&mut v, &signs);
        Kernel::Scalar.walsh_hadamard_transform::<true>(&mut v, &signs);
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
    /// every block-size code path (below 64, exactly 64, and multiple 64-blocks) and both
    /// transform directions.
    #[test]
    fn accelerated_matches_scalar() {
        // If no accelerated kernel is available on this hardware there is nothing to compare.
        for accelerated in accelerated_kernels() {
            for &dims in &[1usize, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] {
                for seed in 0..5u64 {
                    let v = make_vec(dims, seed);
                    let signs = make_signs(dims, seed ^ 0xdead_beef);

                    let mut forward_scalar = v.clone();
                    let mut forward_accel = v.clone();
                    Kernel::Scalar.walsh_hadamard_transform::<true>(&mut forward_scalar, &signs);
                    accelerated.walsh_hadamard_transform::<true>(&mut forward_accel, &signs);
                    for (i, (a, b)) in forward_scalar.iter().zip(&forward_accel).enumerate() {
                        assert!(
                            abs_diff_eq!(a, b, epsilon = 1e-4),
                            "{accelerated} dims={dims} seed={seed} forward mismatch at {i}: scalar={a} accel={b}"
                        );
                    }

                    let mut backward_scalar = forward_scalar.clone();
                    let mut backward_accel = forward_accel.clone();
                    Kernel::Scalar.walsh_hadamard_transform::<false>(&mut backward_scalar, &signs);
                    accelerated.walsh_hadamard_transform::<false>(&mut backward_accel, &signs);
                    for (i, (a, b)) in backward_scalar.iter().zip(&backward_accel).enumerate() {
                        assert!(
                            abs_diff_eq!(a, b, epsilon = 1e-4),
                            "{accelerated} dims={dims} seed={seed} backward mismatch at {i}: scalar={a} accel={b}"
                        );
                    }
                }
            }
        }
    }
}
