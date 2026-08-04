use std::borrow::Cow;

use approx::assert_abs_diff_eq;
use rand::{Rng, SeedableRng, TryRngCore, rngs::OsRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::l2_normalize;

use super::{
    Header, new_asymmetric_distance, new_coder, new_scalar_asymmetric_distance, new_scalar_coder,
    new_scalar_symmetric_distance, new_scalar_symmetric_query_distance, new_symmetric_distance,
    new_symmetric_query_distance,
};

/// Number of trials to run per test. Each trial uses a freshly generated random unit vector.
const TRIALS: usize = 256;

/// Base dimensionality used by all tests -- large enough to exercise the full-width SIMD loop
/// of every kernel.
const BASE_DIMENSIONS: usize = 512;

fn seeded_rng() -> Xoshiro256PlusPlus {
    let seed = OsRng::default().try_next_u64().unwrap();
    println!("SEED {seed:#016x}");
    Xoshiro256PlusPlus::seed_from_u64(seed)
}

fn random_unit_vector(rng: &mut impl Rng, dimensions: usize) -> Vec<f32> {
    let raw = (0..dimensions)
        .map(|_| rng.random_range(-1.0f32..=1.0))
        .collect::<Vec<_>>();
    l2_normalize(raw).into_owned()
}

/// Encode `vector` once with the scalar coder so distance tests exercise one fixed set of
/// encoded bytes -- the encode-side ULP differences between the accelerated and scalar coders
/// (see `assert_encode_matches_scalar`) are a separate concern from whether the accelerated and
/// scalar *distance* kernels agree on a given pair of encoded vectors.
fn encode_vector(vector: &[f32]) -> Vec<u8> {
    new_scalar_coder(None).encode(vector)
}

/// Assert that the platform-accelerated coder and the scalar reference coder produce equivalent
/// output for `vector`.
///
/// The packed 2-bit quantization decisions (which dominate the encoded bytes) must match
/// exactly -- they're derived from simple per-element comparisons against `tau` that don't
/// depend on summation order. The header's aggregate `weak`/`strong` magnitudes, however, are
/// each the mean of up to `vector.len()` floating point terms accumulated in a different order
/// by SIMD vs. scalar kernels (tree reduction vs. sequential), so those are compared with a
/// small tolerance rather than bit-for-bit.
fn assert_encode_matches_scalar(trial: usize, vector: &[f32]) {
    let simd = new_coder(None);
    let scalar = new_scalar_coder(None);

    let simd_encoded = simd.encode(vector);
    let scalar_encoded = scalar.encode(vector);

    let (simd_header, simd_body) = Header::split_and_decode(&simd_encoded);
    let (scalar_header, scalar_body) = Header::split_and_decode(&scalar_encoded);

    let ctx = || format!("trial {trial} dimensions {}", vector.len());
    assert_eq!(
        simd_header.strong_count,
        scalar_header.strong_count,
        "{}",
        ctx()
    );
    assert_abs_diff_eq!(simd_header.weak, scalar_header.weak, epsilon = 1e-4);
    assert_abs_diff_eq!(simd_header.strong, scalar_header.strong, epsilon = 1e-4);
    assert_eq!(simd_body, scalar_body, "{}", ctx());

    // Byte length and dimension accounting should agree between implementations too.
    assert_eq!(simd.byte_len(vector.len()), scalar.byte_len(vector.len()));
    assert_eq!(
        simd.dimensions(simd_encoded.len()),
        scalar.dimensions(scalar_encoded.len())
    );
}

#[test]
fn encode_matches_scalar_512() {
    let mut rng = seeded_rng();
    for trial in 0..TRIALS {
        let vector = random_unit_vector(&mut rng, BASE_DIMENSIONS);
        assert_encode_matches_scalar(trial, &vector);
    }
}

/// SIMD kernels quantize/pack in fixed-size chunks (e.g. 64 or 128 dimensions at a time) and
/// fall back to scalar code for the remaining "tail" dimensions. Exercise a range of tail
/// lengths on top of the base dimensionality to catch bugs specific to that fallback path.
#[test]
fn encode_matches_scalar_tail_dimensions() {
    let mut rng = seeded_rng();
    for trial in 0..TRIALS {
        let extra = rng.random_range(1..=128);
        let vector = random_unit_vector(&mut rng, BASE_DIMENSIONS + extra);
        assert_encode_matches_scalar(trial, &vector);
    }
}

/// Round tripping an encoded unit vector through decode() should produce a finite vector of the
/// same dimensionality that is reasonably close to the original -- QuIVer is lossy (2 bits per
/// dimension) so we only check coarse agreement, not exactness.
#[test]
fn decode_roundtrip_is_sane() {
    let mut rng = seeded_rng();
    let coder = new_coder(None);
    for trial in 0..TRIALS {
        let extra = rng.random_range(0..=128);
        let vector = random_unit_vector(&mut rng, BASE_DIMENSIONS + extra);

        let encoded = coder.encode(&vector);
        assert_eq!(encoded.len(), coder.byte_len(vector.len()));

        // Decode into a buffer sized from the known dimensionality directly: `dimensions()`
        // only recovers an exact dimension count from `byte_len()` when it's a multiple of 4,
        // so it isn't used here (a caller of `decode_to` is expected to track dimensionality
        // itself, same as every other coder in this crate).
        let mut decoded = vec![0.0f32; vector.len()];
        coder.decode_to(&encoded, &mut decoded);
        assert!(
            decoded.iter().all(|d| d.is_finite()),
            "trial {trial} produced non-finite decoded values: {decoded:?}"
        );

        // Sign should be preserved for (almost) every dimension: QuIVer only loses magnitude
        // precision, not direction, except right at the tau threshold.
        let sign_mismatches = vector
            .iter()
            .zip(decoded.iter())
            .filter(|&(&o, &d)| o.signum() != d.signum())
            .count();
        assert!(
            sign_mismatches == 0,
            "trial {trial} had {sign_mismatches} sign mismatches out of {} dims",
            vector.len()
        );
    }
}

/// Symmetric distance is computed entirely from integer popcounts of the packed 2-bit vectors,
/// so -- unlike the header's floating-point aggregates in `encode()` -- the accelerated and
/// scalar kernels should agree bit-for-bit regardless of chunking/reduction strategy.
#[test]
fn symmetric_distance_matches_scalar_512() {
    let mut rng = seeded_rng();
    let accelerated = new_symmetric_distance();
    let scalar = new_scalar_symmetric_distance();
    for trial in 0..TRIALS {
        let a = encode_vector(&random_unit_vector(&mut rng, BASE_DIMENSIONS));
        let b = encode_vector(&random_unit_vector(&mut rng, BASE_DIMENSIONS));
        assert_eq!(
            accelerated.distance(&a, &b),
            scalar.distance(&a, &b),
            "trial {trial}"
        );
    }
}

/// Same as `symmetric_distance_matches_scalar_512` but with a random number of extra dimensions
/// on top of the base width, to exercise the SIMD kernels' scalar tail fallback.
#[test]
fn symmetric_distance_matches_scalar_tail_dimensions() {
    let mut rng = seeded_rng();
    let accelerated = new_symmetric_distance();
    let scalar = new_scalar_symmetric_distance();
    for trial in 0..TRIALS {
        let dimensions = BASE_DIMENSIONS + rng.random_range(1..=128);
        let a = encode_vector(&random_unit_vector(&mut rng, dimensions));
        let b = encode_vector(&random_unit_vector(&mut rng, dimensions));
        assert_eq!(
            accelerated.distance(&a, &b),
            scalar.distance(&a, &b),
            "trial {trial} dimensions {dimensions}"
        );
    }
}

/// `SymmetricalQueryDistance` (bound query vs. streamed doc vectors) wraps the same
/// `Distance::distance` used above, just through the `QueryVectorDistance` API. Exercise it
/// separately since it's the API actually used for search.
#[test]
fn symmetric_query_distance_matches_scalar_512() {
    let mut rng = seeded_rng();
    for trial in 0..TRIALS {
        let query = encode_vector(&random_unit_vector(&mut rng, BASE_DIMENSIONS));
        let doc = encode_vector(&random_unit_vector(&mut rng, BASE_DIMENSIONS));
        let accelerated = new_symmetric_query_distance(Cow::Borrowed(query.as_slice()));
        let scalar = new_scalar_symmetric_query_distance(Cow::Borrowed(query.as_slice()));
        assert_eq!(
            accelerated.distance(&doc),
            scalar.distance(&doc),
            "trial {trial}"
        );
    }
}

#[test]
fn symmetric_query_distance_matches_scalar_tail_dimensions() {
    let mut rng = seeded_rng();
    for trial in 0..TRIALS {
        let dimensions = BASE_DIMENSIONS + rng.random_range(1..=128);
        let query = encode_vector(&random_unit_vector(&mut rng, dimensions));
        let doc = encode_vector(&random_unit_vector(&mut rng, dimensions));
        let accelerated = new_symmetric_query_distance(Cow::Borrowed(query.as_slice()));
        let scalar = new_scalar_symmetric_query_distance(Cow::Borrowed(query.as_slice()));
        assert_eq!(
            accelerated.distance(&doc),
            scalar.distance(&doc),
            "trial {trial} dimensions {dimensions}"
        );
    }
}

/// Asymmetric distance (raw f32 query vs. encoded doc vector) is also an exact integer dot
/// product, so accelerated and scalar kernels should agree bit-for-bit.
#[test]
fn asymmetric_distance_matches_scalar_512() {
    let mut rng = seeded_rng();
    for trial in 0..TRIALS {
        let query = random_unit_vector(&mut rng, BASE_DIMENSIONS);
        let doc = encode_vector(&random_unit_vector(&mut rng, BASE_DIMENSIONS));
        let accelerated = new_asymmetric_distance(&query);
        let scalar = new_scalar_asymmetric_distance(&query);
        assert_eq!(
            accelerated.distance(&doc),
            scalar.distance(&doc),
            "trial {trial}"
        );
    }
}

#[test]
fn asymmetric_distance_matches_scalar_tail_dimensions() {
    let mut rng = seeded_rng();
    for trial in 0..TRIALS {
        let dimensions = BASE_DIMENSIONS + rng.random_range(1..=128);
        let query = random_unit_vector(&mut rng, dimensions);
        let doc = encode_vector(&random_unit_vector(&mut rng, dimensions));
        let accelerated = new_asymmetric_distance(&query);
        let scalar = new_scalar_asymmetric_distance(&query);
        assert_eq!(
            accelerated.distance(&doc),
            scalar.distance(&doc),
            "trial {trial} dimensions {dimensions}"
        );
    }
}

/// Regression test for a bug in the aarch64 asymmetric-distance kernel: `et_quiver_asymmetric_ip`
/// (a hand-written NEON/dotprod C helper) only processes whole 64-element chunks and performs no
/// bounds checking, so calling it with a query length that wasn't a multiple of 64 used to read
/// past the end of the query and doc buffers. The Rust-side kernel now rounds the length passed
/// to the C helper down to a multiple of 64 and handles the remainder itself. Cover every
/// possible remainder (not just the 61-63 range that broke `quantize()`'s output chunking,
/// since this bug's root cause is different) across a couple of chunk multiples.
#[test]
fn asymmetric_distance_handles_simd_chunk_boundary_tails() {
    let mut rng = seeded_rng();
    for chunk_multiple in [0usize, 1, 2] {
        for remainder in 1..64 {
            let dimensions = chunk_multiple * 64 + remainder;
            let query = random_unit_vector(&mut rng, dimensions);
            let doc = encode_vector(&random_unit_vector(&mut rng, dimensions));
            let accelerated = new_asymmetric_distance(&query);
            let scalar = new_scalar_asymmetric_distance(&query);
            assert_eq!(
                accelerated.distance(&doc),
                scalar.distance(&doc),
                "dimensions {dimensions}"
            );
        }
    }
}

/// Regression test for a boundary bug in the SIMD kernels' `quantize()`: when the number of
/// dimensions left over after the last full 64-wide SIMD chunk is 61, 62, or 63, its packed
/// size rounds up to a full 16-byte block, which used to misalign the output chunking against
/// the input chunking and made the scalar fallback write into a zero-length tail buffer,
/// panicking. Cover the boundary directly across several multiples of the chunk width, in
/// addition to the randomized tail coverage in `encode_matches_scalar_tail_dimensions`.
#[test]
fn encode_handles_simd_chunk_boundary_tails() {
    let mut rng = seeded_rng();
    for chunk_multiple in [1usize, 2, 8, 9, 10] {
        for remainder in 61..64 {
            let dimensions = chunk_multiple * 64 + remainder;
            let vector = random_unit_vector(&mut rng, dimensions);
            assert_encode_matches_scalar(dimensions, &vector);
        }
    }
}
