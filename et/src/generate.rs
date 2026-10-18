use std::{
    fs::File,
    io::{self, BufWriter, Write},
    num::NonZero,
    path::PathBuf,
};

use clap::Args;
use half::{f16, slice::HalfFloatSliceExt};
use indicatif::ProgressIterator;
use rand::{Rng, RngExt, SeedableRng};
use rand_distr::{Distribution, StandardNormal};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::ui::progress_bar;

/// Default maximum Euclidean distance of a generated vector from its cluster
/// center, roughly matching the locality of real transformer embedding data.
const DEFAULT_CLUSTER_RADIUS: f32 = 0.3;

/// Fraction of the cluster radius targeted as the *mean* distance from the
/// center; the remaining headroom makes the radius a hard rejection cap that
/// is ~10σ out at realistic dimensionality, so rejection almost never fires.
const RADIUS_TARGET_FRACTION: f32 = 0.8;

/// The mean-distance target saturates here: E[chord²] approaches its uniform-
/// sphere limit of 2 only asymptotically, so the σ formula is only solvable
/// below √2. Radii above `RADIUS_TARGET_FRACTION`-times this degrade to
/// capped near-uniform generation anyway.
const MAX_TARGET_RADIUS: f32 = 1.4;

/// Rejection attempts before falling back to bisection of the noise scale.
const MAX_REJECTION_ATTEMPTS: usize = 32;

#[derive(Args)]
pub struct GenerateArgs {
    /// Output file to write generated vectors as little-endian f16 values.
    #[arg(short, long)]
    output: PathBuf,
    /// Number of dimensions per vector.
    #[arg(short, long)]
    dimensions: NonZero<usize>,
    /// Number of vectors to generate.
    #[arg(short, long)]
    count: NonZero<usize>,
    /// Random seed for reproducible generation.
    #[arg(short, long)]
    seed: u64,
    /// Number of cluster centers to maintain. When set, vectors are generated
    /// in clusters around randomly placed unit-norm centers instead of
    /// uniformly; exhausted centers are replaced with fresh ones.
    #[arg(long)]
    centers: Option<NonZero<usize>>,
    /// Maximum Euclidean (chord) distance from its center for a generated
    /// vector, after L2 normalization. In (0.0, 2.0); values near 2 approach
    /// uniform. Only valid with --centers.
    #[arg(long, default_value_t = DEFAULT_CLUSTER_RADIUS, requires = "centers")]
    cluster_radius: f32,
    /// Hard cap on how many vectors a center may produce. Each center draws
    /// its actual use count uniformly from [ceil(M/2), M]. Defaults to
    /// ceil(4/3 * count / centers) so expected total use matches --count.
    #[arg(long, requires = "centers")]
    max_per_center: Option<NonZero<usize>>,
}

pub fn generate(args: GenerateArgs) -> io::Result<()> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(args.seed);
    let dims = args.dimensions.get();
    let count = args.count.get();

    let mut out = BufWriter::new(File::create(&args.output)?);
    out.write_all(&(count as u32).to_le_bytes())?;
    out.write_all(&(dims as u32).to_le_bytes())?;

    let mut v = vec![0.0f32; dims];
    let mut v16 = vec![f16::ZERO; dims];
    match args.centers {
        None => {
            for _ in (0..count).progress_with(progress_bar(count, "generate")) {
                for x in &mut v {
                    *x = StandardNormal.sample(&mut rng);
                }
                vectors::prepare_vector_in_place(&mut v, None, true, None);
                write_vector(&mut out, &v, &mut v16)?;
            }
        }
        Some(centers) => {
            if dims < 2 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "clustered generation requires at least 2 dimensions",
                ));
            }
            if !(0.0..2.0).contains(&args.cluster_radius) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "cluster radius must be in (0.0, 2.0), got {}",
                        args.cluster_radius
                    ),
                ));
            }
            let max_uses = match args.max_per_center {
                Some(m) => m.get().min(u32::MAX as usize) as u32,
                None => default_max_uses(count, centers.get()),
            };
            let mut pool =
                ClusterPool::new(&mut rng, centers.get(), dims, max_uses, args.cluster_radius);
            let mut noise = vec![0.0f32; dims];
            for _ in (0..count).progress_with(progress_bar(count, "generate")) {
                pool.next(&mut rng, &mut noise, &mut v);
                // The radius cap is enforced in f32 before encoding; f16 rounding
                // (~2^-11 relative) can push a vector ~1e-3 past the radius on
                // disk. Enforce the cap post-encode instead if the on-disk
                // guarantee must be exact.
                write_vector(&mut out, &v, &mut v16)?;
            }
        }
    }

    out.flush()?;
    Ok(())
}

/// Writes `v` to `out` as little-endian f16, converting through the `v16`
/// scratch buffer.
fn write_vector(out: &mut impl Write, v: &[f32], v16: &mut [f16]) -> io::Result<()> {
    v16.convert_from_f32_slice(v);
    for &x in v16.iter() {
        out.write_all(&x.to_le_bytes())?;
    }
    Ok(())
}

/// Default per-center use cap: ceil(4/3 · count / centers) so the expected
/// total use (mean 0.75M per center) matches `count`.
fn default_max_uses(count: usize, centers: usize) -> u32 {
    let count = count.min(u32::MAX as usize) as u64;
    let centers = centers as u64;
    (4 * count).div_ceil(3 * centers).clamp(1, u32::MAX as u64) as u32
}

/// Returns σ such that for v = normalize(center + σ·g) with g ~ N(0, I_dims),
/// E[‖v − center‖²] ≈ r². With t = σ²(dims−1), cos θ ≈ 1/√(1+t), so
/// E[chord²] = 2(1 − 1/√(1+t)) = r² solves to t = 1/(1 − r²/2)² − 1; the
/// target saturates at MAX_TARGET_RADIUS.
fn sigma_for_radius(r: f32, dims: usize) -> f32 {
    debug_assert!(r > 0.0 && r < 2.0);
    debug_assert!(dims >= 2);
    let r = r.min(MAX_TARGET_RADIUS);
    let t = 1.0 / (1.0 - r * r / 2.0).powi(2) - 1.0;
    (t / (dims as f32 - 1.0)).sqrt()
}

/// Fills `out` with a uniform random unit vector: iid Gaussian components
/// followed by L2 normalization.
fn random_unit_vector(rng: &mut impl Rng, out: &mut [f32]) {
    for x in &mut *out {
        *x = StandardNormal.sample(rng);
    }
    vectors::prepare_vector_in_place(out, None, true, None);
}

/// Squared Euclidean distance between two unit vectors: ‖a − b‖² = 2(1 − a·b).
fn chord_distance_sq(a: &[f32], b: &[f32]) -> f32 {
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (*x as f64) * (*y as f64))
        .sum();
    (2.0 * (1.0 - dot)) as f32
}

/// Fills `noise` with a fresh iid standard-normal perturbation vector.
fn sample_noise(rng: &mut impl Rng, noise: &mut [f32]) {
    for x in &mut *noise {
        *x = StandardNormal.sample(rng);
    }
}

/// Writes `normalize(center + α·σ·noise)` to `out`.
fn apply_perturbation(center: &[f32], sigma: f32, alpha: f32, noise: &[f32], out: &mut [f32]) {
    for ((o, c), g) in out.iter_mut().zip(center).zip(noise) {
        *o = c + alpha * sigma * g;
    }
    vectors::prepare_vector_in_place(out, None, true, None);
}

/// Generates one clustered vector into `out`: perturb `center` with σ·g and
/// renormalize, resampling while the chord distance exceeds `radius`. If
/// rejection fails (tiny dimensions or radius), bisect the noise scale α in
/// v = normalize(center + α·σ·noise): the angle between `center` and
/// `center + α·noise` is monotone in α, so this terminates with a vector
/// within the cap.
fn clustered_vector(
    rng: &mut impl Rng,
    center: &[f32],
    sigma: f32,
    radius: f32,
    noise: &mut [f32],
    out: &mut [f32],
) {
    debug_assert_eq!(center.len(), noise.len());
    debug_assert_eq!(center.len(), out.len());
    let radius_sq = radius * radius;
    for _ in 0..MAX_REJECTION_ATTEMPTS {
        sample_noise(rng, noise);
        apply_perturbation(center, sigma, 1.0, noise, out);
        if chord_distance_sq(out, center) <= radius_sq {
            return;
        }
    }
    // `noise` holds the last failing full-scale sample; bisect its scale. α = 0
    // is `center` itself, which trivially satisfies the cap.
    let mut lo = 0.0f32;
    let mut hi = 1.0f32;
    for _ in 0..60 {
        let mid = (lo + hi) / 2.0;
        apply_perturbation(center, sigma, mid, noise, out);
        if chord_distance_sq(out, center) <= radius_sq {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    apply_perturbation(center, sigma, lo, noise, out);
}

/// Manages the pool of cluster centers and their remaining use counts. A
/// center that exhausts its use count is replaced in place with a fresh
/// center, so every slot stays active and the stream never runs dry.
struct ClusterPool {
    centers: Vec<f32>,
    remaining_uses: Vec<u32>,
    dims: usize,
    sigma: f32,
    radius: f32,
    max_uses: u32,
}

impl ClusterPool {
    fn new(rng: &mut impl Rng, n: usize, dims: usize, max_uses: u32, radius: f32) -> Self {
        let mut pool = Self {
            centers: vec![0.0; n * dims],
            remaining_uses: vec![0; n],
            dims,
            sigma: sigma_for_radius(RADIUS_TARGET_FRACTION * radius, dims),
            radius,
            max_uses,
        };
        for center in pool.centers.chunks_mut(dims) {
            random_unit_vector(rng, center);
        }
        for uses in &mut pool.remaining_uses {
            *uses = rng.random_range(max_uses.div_ceil(2)..=max_uses);
        }
        pool
    }

    /// Picks a center uniformly at random (naturally interleaving the output
    /// across clusters), generates a vector near it, and replaces the center
    /// in place when its use count runs out. Returns the center's slot index.
    fn next(&mut self, rng: &mut impl Rng, noise: &mut [f32], out: &mut [f32]) -> usize {
        let idx = rng.random_range(0..self.remaining_uses.len());
        clustered_vector(rng, self.center(idx), self.sigma, self.radius, noise, out);
        self.remaining_uses[idx] -= 1;
        if self.remaining_uses[idx] == 0 {
            random_unit_vector(
                rng,
                &mut self.centers[idx * self.dims..(idx + 1) * self.dims],
            );
            let max_uses = self.max_uses;
            self.remaining_uses[idx] = rng.random_range(max_uses.div_ceil(2)..=max_uses);
        }
        idx
    }

    fn center(&self, idx: usize) -> &[f32] {
        &self.centers[idx * self.dims..(idx + 1) * self.dims]
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn seeded() -> Xoshiro256PlusPlus {
        Xoshiro256PlusPlus::seed_from_u64(0xE771_9E57)
    }

    fn norm(v: &[f32]) -> f64 {
        v.iter()
            .map(|&x| (x as f64) * (x as f64))
            .sum::<f64>()
            .sqrt()
    }

    #[test]
    fn sigma_matches_radius() {
        let dims = 768;
        let radius = 0.3;
        let sigma = sigma_for_radius(RADIUS_TARGET_FRACTION * radius, dims);
        let mut rng = seeded();
        let mut center = vec![0.0; dims];
        random_unit_vector(&mut rng, &mut center);
        let mut noise = vec![0.0; dims];
        let mut out = vec![0.0; dims];
        let mut sum_sq = 0.0f64;
        for _ in 0..200 {
            clustered_vector(&mut rng, &center, sigma, radius, &mut noise, &mut out);
            let d = chord_distance_sq(&out, &center).sqrt() as f64;
            assert!(d <= radius as f64, "chord {d} exceeds radius {radius}");
            sum_sq += d * d;
        }
        let target = (RADIUS_TARGET_FRACTION * radius) as f64;
        let mean_sq = sum_sq / 200.0;
        assert!(
            (mean_sq - target * target).abs() / (target * target) < 0.10,
            "mean squared chord {mean_sq} not within 10% of target {}",
            target * target
        );
    }

    #[test]
    fn outputs_are_unit_norm() {
        let mut rng = seeded();
        let mut pool = ClusterPool::new(&mut rng, 8, 128, 5, 0.3);
        let mut noise = vec![0.0; 128];
        let mut out = vec![0.0; 128];
        for _ in 0..100 {
            pool.next(&mut rng, &mut noise, &mut out);
            assert!(
                (norm(&out) - 1.0).abs() < 1e-5,
                "norm {} deviates from 1.0",
                norm(&out)
            );
        }
    }

    #[test]
    fn radius_is_hard_cap() {
        let mut rng = seeded();
        // Use a cap far above the per-slot draw count so no center is refilled
        // mid-test: `next` replaces an exhausted center before returning, so the
        // vector would legitimately be far from the replacement.
        let mut pool = ClusterPool::new(&mut rng, 16, 768, 10_000, 0.3);
        let mut noise = vec![0.0; 768];
        let mut out = vec![0.0; 768];
        for _ in 0..1000 {
            let idx = pool.next(&mut rng, &mut noise, &mut out);
            let d = chord_distance_sq(&out, pool.center(idx)).sqrt();
            assert!(d <= 0.3, "chord {d} exceeds radius 0.3");
        }
    }

    #[test]
    fn exhaustion_replaces_center() {
        let mut rng = seeded();
        // Caps are drawn from [1, 2], so 16 draws over 4 slots (total capacity
        // at most 8) must exhaust and replace at least one center.
        let mut pool = ClusterPool::new(&mut rng, 4, 32, 2, 0.3);
        let initial_centers = pool.centers.clone();
        let mut noise = vec![0.0; 32];
        let mut out = vec![0.0; 32];
        for _ in 0..16 {
            pool.next(&mut rng, &mut noise, &mut out);
        }
        assert!(pool.remaining_uses.iter().all(|&uses| uses >= 1));
        assert!(
            initial_centers
                .iter()
                .zip(&pool.centers)
                .any(|(a, b)| a != b),
            "no center was replaced after certain exhaustion"
        );
    }

    #[test]
    fn interleaving_not_grouped() {
        let mut rng = seeded();
        let mut pool = ClusterPool::new(&mut rng, 10, 64, 1000, 0.3);
        let mut noise = vec![0.0; 64];
        let mut out = vec![0.0; 64];
        let mut indices = Vec::new();
        let mut max_run = 1;
        let mut run = 1;
        for _ in 0..100 {
            let idx = pool.next(&mut rng, &mut noise, &mut out);
            if let Some(&prev) = indices.last() {
                run = if prev == idx { run + 1 } else { 1 };
                max_run = max_run.max(run);
            }
            indices.push(idx);
        }
        let distinct = indices
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len();
        assert!(distinct >= 5, "only {distinct} distinct centers used");
        assert!(
            max_run < 10,
            "run of {max_run} consecutive draws from one center"
        );
    }

    #[test]
    fn uniform_mode_unchanged() {
        let dir = std::env::temp_dir();
        let outputs = [
            dir.join("et_generate_uniform_test_a.f16vecs"),
            dir.join("et_generate_uniform_test_b.f16vecs"),
        ];
        for output in &outputs {
            generate(GenerateArgs {
                output: output.clone(),
                dimensions: NonZero::new(16).unwrap(),
                count: NonZero::new(64).unwrap(),
                seed: 7,
                centers: None,
                cluster_radius: DEFAULT_CLUSTER_RADIUS,
                max_per_center: None,
            })
            .unwrap();
        }
        let a = std::fs::read(&outputs[0]).unwrap();
        let b = std::fs::read(&outputs[1]).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), 8 + 64 * 16 * 2);
        assert_eq!(&a[..4], &64u32.to_le_bytes());
        assert_eq!(&a[4..8], &16u32.to_le_bytes());
    }

    #[test]
    fn sigma_formula_degenerate_cases() {
        // Small radii: chord ≈ perturbation angle ≈ σ·sqrt(dims−1), so σ ≈ r / sqrt(dims−1).
        let r = 0.01f32;
        let sigma = sigma_for_radius(r, 768);
        let expected = r / (767.0f32).sqrt();
        assert!(
            (sigma - expected).abs() / expected < 0.01,
            "sigma {sigma} != small-radius approximation {expected}"
        );
        // The target saturates above MAX_TARGET_RADIUS.
        assert_eq!(
            sigma_for_radius(1.9, 768),
            sigma_for_radius(MAX_TARGET_RADIUS, 768)
        );
        // Bisection fallback: with σ=0.5 and dims=4 the typical chord is ~0.7,
        // so a 0.05 radius is small enough that 32 rejection attempts
        // deterministically fail and bisection must terminate under the cap
        // (which sits safely above the ~1e-4 f32 self-distance resolution).
        let mut rng = seeded();
        let mut center = vec![0.0; 4];
        random_unit_vector(&mut rng, &mut center);
        let mut noise = vec![0.0; 4];
        let mut out = vec![0.0; 4];
        clustered_vector(&mut rng, &center, 0.5, 0.05, &mut noise, &mut out);
        assert!(chord_distance_sq(&out, &center).sqrt() <= 0.05);
    }
}
