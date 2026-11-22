use std::borrow::Cow;

use rand::{RngExt, SeedableRng, TryRng, rngs::SysRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use super::{Distance, Kernel, QueryDistance, VectorCoder, inner_product};
use crate::{
    F32VectorCoder, F32VectorCoding, QueryVectorDistance, VectorDistance, VectorSimilarity,
    prepare_vector, prepare_vector_from_i8, rotate::Rotator,
};

/// Vector lengths that cover empty input, every tail length for the 16 and 64 lane kernels, and
/// several full blocks.
const LENGTHS: std::ops::RangeInclusive<usize> = 0..=260;

/// A long length to exercise accumulator growth across many SIMD blocks, plus a ragged tail.
const LONG_LENGTH: usize = 4096 + 37;

fn new_rng() -> Xoshiro256PlusPlus {
    let seed = SysRng.try_next_u64().unwrap();
    println!("SEED {seed:#016x}");
    Xoshiro256PlusPlus::seed_from_u64(seed)
}

fn random_vector(rng: &mut Xoshiro256PlusPlus, dim: usize) -> Vec<i8> {
    (0..dim)
        .map(|_| rng.random_range(i8::MIN..=i8::MAX))
        .collect()
}

/// Inputs at the edges of the `i8` range. These stress sign handling and the bias correction in
/// kernels that operate on unsigned values.
fn extreme_vectors(dim: usize) -> Vec<(&'static str, Vec<i8>)> {
    let mut one_hot = vec![0i8; dim];
    if let Some(d) = one_hot.first_mut() {
        *d = i8::MIN;
    }
    vec![
        ("zero", vec![0; dim]),
        ("all_min", vec![i8::MIN; dim]),
        ("all_max", vec![i8::MAX; dim]),
        (
            "alternating",
            (0..dim)
                .map(|i| if i % 2 == 0 { i8::MIN } else { i8::MAX })
                .collect(),
        ),
        ("one_hot_min", one_hot),
    ]
}

/// Generate a set of labelled (a, b) pairs of length `dim` to compare across kernels.
fn test_pairs(rng: &mut Xoshiro256PlusPlus, dim: usize) -> Vec<(String, Vec<i8>, Vec<i8>)> {
    let mut pairs = vec![(
        "random".to_string(),
        random_vector(rng, dim),
        random_vector(rng, dim),
    )];
    let extremes = extreme_vectors(dim);
    for (na, a) in &extremes {
        for (nb, b) in &extremes {
            pairs.push((format!("{na}x{nb}"), a.clone(), b.clone()));
        }
    }
    pairs
}

fn ref_inner_product(a: &[i8], b: &[i8]) -> i64 {
    a.iter()
        .zip(b.iter())
        .map(|(&a, &b)| a as i64 * b as i64)
        .sum()
}

fn ref_distance(similarity: VectorSimilarity, a: &[i8], b: &[i8]) -> f64 {
    match similarity {
        VectorSimilarity::Euclidean => a
            .iter()
            .zip(b.iter())
            .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
            .sum(),
        VectorSimilarity::Dot => {
            let ip = ref_inner_product(a, b) as f64;
            let anorm = (ref_inner_product(a, a) as f64).sqrt();
            let bnorm = (ref_inner_product(b, b) as f64).sqrt();
            (1.0 - ip / (anorm * bnorm)) / 2.0
        }
    }
}

/// Absolute error tolerance when comparing a computed distance to [`ref_distance`].
///
/// The stored l2 norm is an `f32`, so euclidean error scales with the squared norms.
fn distance_tolerance(similarity: VectorSimilarity, a: &[i8], b: &[i8]) -> f64 {
    match similarity {
        VectorSimilarity::Euclidean => {
            1e-6 * (ref_inner_product(a, a) + ref_inner_product(b, b)) as f64 + 1e-9
        }
        VectorSimilarity::Dot => 1e-6,
    }
}

fn query_distance_with_kernel(
    k: Kernel,
    similarity: VectorSimilarity,
    query: &[u8],
) -> QueryDistance<'_> {
    QueryDistance {
        k,
        ..QueryDistance::new(similarity, Cow::Borrowed(query))
    }
}

#[test]
fn kernel_availability() {
    assert!(Kernel::Scalar.is_available());
    assert!(Kernel::default().is_available());
    for k in Kernel::accelerated() {
        assert_ne!(k, Kernel::Scalar);
        assert!(k.is_available());
    }
    println!(
        "default={:?} accelerated={:?}",
        Kernel::default(),
        Kernel::accelerated().collect::<Vec<_>>()
    );
}

#[test]
fn inner_product_scalar() {
    let mut rng = new_rng();
    for len in LENGTHS.chain([LONG_LENGTH]) {
        for (name, a, b) in test_pairs(&mut rng, len) {
            assert_eq!(
                inner_product(Kernel::Scalar, &a, &b) as i64,
                ref_inner_product(&a, &b),
                "len={len} {name}"
            );
        }
    }
}

#[test]
fn inner_product_simd() {
    let mut rng = new_rng();
    for k in Kernel::accelerated() {
        for len in LENGTHS.chain([LONG_LENGTH]) {
            for (name, a, b) in test_pairs(&mut rng, len) {
                assert_eq!(
                    inner_product(k, &a, &b),
                    inner_product(Kernel::Scalar, &a, &b),
                    "k={k:?} len={len} {name}"
                );
            }
        }
    }
}

#[test]
#[should_panic]
fn inner_product_length_mismatch() {
    inner_product(Kernel::default(), &[1, 2, 3], &[1, 2]);
}

#[test]
fn convert_slice_to_f32_scalar() {
    let mut rng = new_rng();
    for len in LENGTHS {
        let v = random_vector(&mut rng, len);
        let mut out = vec![f32::NAN; len];
        VectorCoder(Kernel::Scalar).convert_slice_to_f32(&v, &mut out);
        let expected = v.iter().map(|&d| d as f32).collect::<Vec<_>>();
        assert_eq!(out, expected, "len={len}");
    }
}

#[test]
fn convert_slice_to_f32_simd() {
    let mut rng = new_rng();
    for k in Kernel::accelerated() {
        for len in LENGTHS.chain([LONG_LENGTH]) {
            let mut inputs = extreme_vectors(len);
            inputs.push(("random", random_vector(&mut rng, len)));
            for (name, v) in inputs {
                let mut scalar = vec![f32::NAN; len];
                VectorCoder(Kernel::Scalar).convert_slice_to_f32(&v, &mut scalar);
                let mut accel = vec![f32::NAN; len];
                VectorCoder(k).convert_slice_to_f32(&v, &mut accel);
                assert_eq!(accel, scalar, "k={k:?} len={len} {name}");
            }
        }
    }
}

#[test]
#[should_panic]
fn convert_slice_to_f32_length_mismatch() {
    VectorCoder::default().convert_slice_to_f32(&[1, 2, 3], &mut [0.0; 2]);
}

#[test]
fn encode_layout() {
    let mut rng = new_rng();
    for len in LENGTHS {
        let v = random_vector(&mut rng, len);
        let encoded = VectorCoder::default().encode(&v);
        assert_eq!(encoded.len(), len + VectorCoder::HEADER_LEN);

        let norm = f32::from_le_bytes(encoded[..4].try_into().unwrap());
        let expected_norm = (ref_inner_product(&v, &v) as f64).sqrt() as f32;
        assert_eq!(norm, expected_norm, "len={len}");
        assert_eq!(
            &encoded[4..],
            bytemuck::cast_slice::<i8, u8>(&v),
            "len={len}"
        );

        let mut encoded_to = vec![0xffu8; encoded.len()];
        VectorCoder::default().encode_to(&v, &mut encoded_to);
        assert_eq!(encoded_to, encoded, "len={len}");
    }
}

#[test]
fn encode_extreme_norms() {
    // The squared norm of this vector exceeds 2^24 so it can't be exactly represented in f32.
    let v = vec![i8::MIN; LONG_LENGTH];
    let encoded = VectorCoder::default().encode(&v);
    let norm = f32::from_le_bytes(encoded[..4].try_into().unwrap());
    assert_eq!(norm, (128.0 * (LONG_LENGTH as f64).sqrt()) as f32);

    let encoded = VectorCoder::default().encode(&[0i8; 16]);
    assert_eq!(f32::from_le_bytes(encoded[..4].try_into().unwrap()), 0.0);
}

#[test]
fn encode_simd() {
    let mut rng = new_rng();
    for k in Kernel::accelerated() {
        for len in LENGTHS.chain([LONG_LENGTH]) {
            let mut inputs = extreme_vectors(len);
            inputs.push(("random", random_vector(&mut rng, len)));
            for (name, v) in inputs {
                assert_eq!(
                    VectorCoder(k).encode(&v),
                    VectorCoder(Kernel::Scalar).encode(&v),
                    "k={k:?} len={len} {name}"
                );
            }
        }
    }
}

#[test]
#[should_panic]
fn encode_to_wrong_length() {
    let mut out = vec![0u8; 6];
    VectorCoder::default().encode_to(&[1, 2, 3], &mut out);
}

#[test]
fn f32_coder_lengths() {
    let coder = VectorCoder::default();
    for dim in [0, 1, 7, 128, 1024] {
        assert_eq!(coder.byte_len(dim), dim + 4);
        assert_eq!(coder.dimensions(dim + 4), dim);
    }
    // Lengths shorter than the header must not underflow.
    for byte_len in 0..4 {
        assert_eq!(coder.dimensions(byte_len), 0);
    }
}

#[test]
#[should_panic(expected = "conversion from f32 not supported")]
fn f32_encode_unsupported() {
    F32VectorCoder::encode(&VectorCoder::default(), &[0.5, -0.5]);
}

#[test]
fn decode_round_trip() {
    let mut rng = new_rng();
    for len in LENGTHS {
        let v = random_vector(&mut rng, len);
        let encoded = VectorCoder::default().encode(&v);
        let expected = v.iter().map(|&d| d as f32).collect::<Vec<_>>();
        assert_eq!(
            VectorCoder::default().decode(&encoded),
            expected,
            "len={len}"
        );
    }
}

#[test]
fn decode_simd() {
    let mut rng = new_rng();
    for k in Kernel::accelerated() {
        for len in LENGTHS.chain([LONG_LENGTH]) {
            let encoded = VectorCoder::default().encode(&random_vector(&mut rng, len));
            assert_eq!(
                VectorCoder(k).decode(&encoded),
                VectorCoder(Kernel::Scalar).decode(&encoded),
                "k={k:?} len={len}"
            );
        }
    }
}

#[test]
#[should_panic]
fn decode_to_wrong_length() {
    let encoded = VectorCoder::default().encode(&[1, 2, 3]);
    VectorCoder::default().decode_to(&encoded, &mut [0.0; 2]);
}

#[test]
fn distance_matches_reference() {
    let mut rng = new_rng();
    for similarity in VectorSimilarity::all() {
        let dist = Distance::new(similarity);
        for i in 0..1024 {
            let dim = rng.random_range(1..=300);
            let a = random_vector(&mut rng, dim);
            let b = random_vector(&mut rng, dim);
            let ea = VectorCoder::default().encode(&a);
            let eb = VectorCoder::default().encode(&b);

            let expected = ref_distance(similarity, &a, &b);
            let actual = dist.distance(&ea, &eb);
            let tolerance = distance_tolerance(similarity, &a, &b);
            assert!(
                (expected - actual).abs() <= tolerance,
                "{similarity} i={i} dim={dim} expected={expected} actual={actual}"
            );
            assert_eq!(actual, dist.distance(&eb, &ea), "{similarity} i={i}");
        }
    }
}

#[test]
fn distance_identical_and_opposite() {
    let mut rng = new_rng();
    for dim in [1, 16, 65, 256] {
        // Avoid i8::MIN so that every vector can be negated.
        let v = (0..dim)
            .map(|_| rng.random_range(-127i8..=127))
            .collect::<Vec<_>>();
        if v.iter().all(|&d| d == 0) {
            continue;
        }
        let neg = v.iter().map(|&d| -d).collect::<Vec<_>>();
        let ev = VectorCoder::default().encode(&v);
        let eneg = VectorCoder::default().encode(&neg);

        let l2 = Distance::new(VectorSimilarity::Euclidean);
        let tolerance = distance_tolerance(VectorSimilarity::Euclidean, &v, &v);
        assert!(l2.distance(&ev, &ev) <= tolerance, "dim={dim}");
        let expected = 4.0 * ref_inner_product(&v, &v) as f64;
        assert!(
            (l2.distance(&ev, &eneg) - expected).abs() <= tolerance,
            "dim={dim}"
        );

        let dot = Distance::new(VectorSimilarity::Dot);
        assert!(dot.distance(&ev, &ev) <= 1e-6, "dim={dim}");
        assert!(dot.distance(&ev, &eneg) >= 1.0 - 1e-6, "dim={dim}");
    }
}

#[test]
fn distance_extremes() {
    for dim in [1, 64, 65, LONG_LENGTH] {
        let min = VectorCoder::default().encode(&vec![i8::MIN; dim]);
        let max = VectorCoder::default().encode(&vec![i8::MAX; dim]);

        let l2 = Distance::new(VectorSimilarity::Euclidean).distance(&min, &max);
        let expected = 255.0f64.powi(2) * dim as f64;
        assert!(
            (l2 - expected).abs() <= 1e-6 * expected,
            "dim={dim} expected={expected} actual={l2}"
        );

        let dot = Distance::new(VectorSimilarity::Dot).distance(&min, &max);
        assert!((dot - 1.0).abs() <= 1e-6, "dim={dim} actual={dot}");
    }
}

#[test]
fn distance_zero_vector() {
    let mut rng = new_rng();
    for dim in [1, 16, 65, 256] {
        let zero = VectorCoder::default().encode(&vec![0; dim]);
        let v = random_vector(&mut rng, dim);
        let ev = VectorCoder::default().encode(&v);

        let l2 = Distance::new(VectorSimilarity::Euclidean);
        assert_eq!(l2.distance(&zero, &zero), 0.0);
        let expected = ref_inner_product(&v, &v) as f64;
        let tolerance = distance_tolerance(VectorSimilarity::Euclidean, &v, &v);
        assert!((l2.distance(&zero, &ev) - expected).abs() <= tolerance);
        assert!((l2.distance(&ev, &zero) - expected).abs() <= tolerance);

        // Cosine is undefined for a zero vector; it must sanitize to the worst score.
        let dot = Distance::new(VectorSimilarity::Dot);
        assert_eq!(dot.distance(&zero, &zero), 1.0);
        assert_eq!(dot.distance(&zero, &ev), 1.0);
        assert_eq!(dot.distance(&ev, &zero), 1.0);
    }
}

#[test]
fn distance_simd() {
    let mut rng = new_rng();
    for k in Kernel::accelerated() {
        for similarity in VectorSimilarity::all() {
            let scalar = Distance {
                k: Kernel::Scalar,
                s: similarity,
            };
            let accel = Distance { k, s: similarity };
            for len in LENGTHS.chain([LONG_LENGTH]) {
                for (name, a, b) in test_pairs(&mut rng, len) {
                    let ea = VectorCoder::default().encode(&a);
                    let eb = VectorCoder::default().encode(&b);
                    assert_eq!(
                        accel.distance(&ea, &eb),
                        scalar.distance(&ea, &eb),
                        "k={k:?} {similarity} len={len} {name}"
                    );
                }
            }
        }
    }
}

#[test]
fn query_distance_matches_distance() {
    let mut rng = new_rng();
    for similarity in VectorSimilarity::all() {
        let dist = Distance::new(similarity);
        for _ in 0..256 {
            let dim = rng.random_range(1..=300);
            let ea = VectorCoder::default().encode(&random_vector(&mut rng, dim));
            let eb = VectorCoder::default().encode(&random_vector(&mut rng, dim));
            let expected = dist.distance(&ea, &eb);

            let borrowed = QueryDistance::new(similarity, Cow::Borrowed(&ea));
            assert_eq!(borrowed.distance(&eb), expected, "{similarity} dim={dim}");

            let owned = QueryDistance::new(similarity, Cow::Owned(ea.clone()));
            assert_eq!(owned.distance(&eb), expected, "{similarity} dim={dim}");
        }
    }
}

#[test]
fn query_distance_simd() {
    let mut rng = new_rng();
    for k in Kernel::accelerated() {
        for similarity in VectorSimilarity::all() {
            for len in LENGTHS.chain([LONG_LENGTH]) {
                for (name, a, b) in test_pairs(&mut rng, len) {
                    let ea = VectorCoder::default().encode(&a);
                    let eb = VectorCoder::default().encode(&b);
                    let scalar = query_distance_with_kernel(Kernel::Scalar, similarity, &ea);
                    let accel = query_distance_with_kernel(k, similarity, &ea);
                    assert_eq!(
                        accel.distance(&eb),
                        scalar.distance(&eb),
                        "k={k:?} {similarity} len={len} {name}"
                    );
                }
            }
        }
    }
}

#[test]
fn coding_string_round_trip() {
    assert_eq!(
        "i8".parse::<F32VectorCoding>().unwrap(),
        F32VectorCoding::I8
    );
    assert_eq!(F32VectorCoding::I8.to_string(), "i8");
}

#[test]
fn coding_coder() {
    let coder = F32VectorCoding::I8.coder();
    assert_eq!(coder.byte_len(128), 132);
    assert_eq!(coder.dimensions(132), 128);

    let v = [1i8, -2, 3, i8::MIN, i8::MAX];
    let encoded = VectorCoder::default().encode(&v);
    assert_eq!(coder.decode(&encoded), vec![1.0, -2.0, 3.0, -128.0, 127.0]);
}

#[test]
#[should_panic(expected = "conversion from f32 not supported")]
fn coding_coder_f32_encode_unsupported() {
    F32VectorCoding::I8.coder().encode(&[0.5, -0.5]);
}

#[test]
fn coding_distances() {
    let mut rng = new_rng();
    for similarity in VectorSimilarity::all() {
        let symmetric = F32VectorCoding::I8
            .distance_symmetric(similarity)
            .expect("supports symmetric distance");
        for _ in 0..64 {
            let dim = rng.random_range(1..=300);
            let a = random_vector(&mut rng, dim);
            let b = random_vector(&mut rng, dim);
            let ea = VectorCoder::default().encode(&a);
            let eb = VectorCoder::default().encode(&b);

            let expected = Distance::new(similarity).distance(&ea, &eb);
            assert_eq!(symmetric.distance(&ea, &eb), expected, "{similarity}");

            let query = F32VectorCoding::I8
                .query_distance_symmetric(similarity, ea.as_slice())
                .expect("supports symmetric distance");
            assert_eq!(query.distance(&eb), expected, "{similarity}");

            assert!(
                (expected - ref_distance(similarity, &a, &b)).abs()
                    <= distance_tolerance(similarity, &a, &b),
                "{similarity}"
            );
        }
    }
}

#[test]
#[should_panic(expected = "asymmetric f32 queries are not supported for i8 vectors")]
fn coding_asymmetric_unsupported() {
    F32VectorCoding::I8.query_distance_asymmetric(VectorSimilarity::Dot, vec![0.5f32, -0.5]);
}

#[test]
fn prepare_from_i8_matches_cast() {
    let mut rng = new_rng();
    for dim in [1, 16, 65, 128] {
        let v = random_vector(&mut rng, dim);
        let vf = v.iter().map(|&d| d as f32).collect::<Vec<_>>();
        let rotator = Rotator::new(dim, 0xabcd);
        let center = (0..dim)
            .map(|_| rng.random_range(-1.0f32..=1.0))
            .collect::<Vec<_>>();

        for rotate in [false, true] {
            for normalize in [false, true] {
                for center in [None, Some(center.as_slice())] {
                    let rotator = rotate.then_some(&rotator);
                    assert_eq!(
                        prepare_vector_from_i8(&v, rotator, normalize, center),
                        prepare_vector(&vf, rotator, normalize, center),
                        "dim={dim} rotate={rotate} normalize={normalize} center={}",
                        center.is_some()
                    );
                }
            }
        }
    }
}
