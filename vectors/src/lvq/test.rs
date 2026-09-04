use approx::{AbsDiffEq, abs_diff_eq, assert_abs_diff_eq};
use rand::{RngExt, SeedableRng, TryRng, rngs::SysRng};

use crate::lvq::{Kernel, PrimaryVectorHeader, TurboPrimaryCoder, VectorStats};
use crate::{F32VectorCoder, F32VectorCoding, VectorSimilarity, float32::l2_normalize};

impl AbsDiffEq for PrimaryVectorHeader {
    type Epsilon = f32;

    fn default_epsilon() -> Self::Epsilon {
        0.00001
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        abs_diff_eq!(self.l2_norm, other.l2_norm, epsilon = epsilon)
            && abs_diff_eq!(self.lower, other.lower, epsilon = epsilon)
            && abs_diff_eq!(self.upper, other.upper, epsilon = epsilon)
            && abs_diff_eq!(
                self.perpendicular_error_term,
                other.perpendicular_error_term,
                epsilon = epsilon
            )
            && abs_diff_eq!(
                self.parallel_error_term,
                other.parallel_error_term,
                epsilon = epsilon
            )
            && abs_diff_eq!(self.component_sum, other.component_sum)
    }
}

impl AbsDiffEq for VectorStats {
    type Epsilon = f32;

    fn default_epsilon() -> Self::Epsilon {
        0.00001
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        abs_diff_eq!(self.min, other.min, epsilon = epsilon)
            && abs_diff_eq!(self.max, other.max, epsilon = epsilon)
            && abs_diff_eq!(self.mean, other.mean, epsilon = epsilon)
            && abs_diff_eq!(self.std_dev, other.std_dev, epsilon = epsilon)
            && abs_diff_eq!(self.l2_norm_sq, other.l2_norm_sq, epsilon = epsilon)
    }
}

// This test vector contains randomly generated numbers in [-1,1] but is not l2 normalized.
// It has 19 elements -- long enough to trigger SIMD optimizations but with some remainder to
// test scalar tail paths.
const TEST_VECTOR: [f32; 19] = [
    -0.921, -0.061, 0.659, 0.67, 0.573, 0.431, 0.646, 0.001, -0.2, -0.428, 0.73, -0.704, -0.273,
    0.539, -0.731, 0.436, 0.913, 0.694, 0.202,
];

#[test]
fn vector_stats_simd() {
    let scalar_stats = VectorStats::new(super::Kernel::Scalar, TEST_VECTOR.as_ref());
    for k in Kernel::accelerated() {
        assert_abs_diff_eq!(scalar_stats, VectorStats::new(k, TEST_VECTOR.as_ref()));
    }
}

macro_rules! tlvq_coder_test {
    ($name:ident, $coder:ty, $primary_header:expr, $decoded:expr) => {
        #[test]
        fn $name() {
            let coder = <$coder>::new();
            let encoded = coder.encode(&TEST_VECTOR);
            assert_abs_diff_eq!(
                PrimaryVectorHeader::deserialize(&encoded).unwrap().0,
                $primary_header
            );
            let mut decoded = vec![0.0f32; TEST_VECTOR.len()];
            coder.decode_to(&encoded, &mut decoded);
            assert_abs_diff_eq!(decoded.as_ref(), $decoded.as_ref(), epsilon = 0.00001);
        }
    };
}

tlvq_coder_test!(
    tlvq1_coder,
    TurboPrimaryCoder::<1>,
    PrimaryVectorHeader {
        l2_norm: 2.5226507,
        perpendicular_error_term: 1.1627843,
        parallel_error_term: 0.021774292,
        lower: -0.49560547,
        upper: 0.7055664,
        component_sum: 11,
    },
    [
        -0.49560547,
        -0.49560547,
        0.7055664,
        0.7055664,
        0.7055664,
        0.7055664,
        0.7055664,
        -0.49560547,
        -0.49560547,
        -0.49560547,
        0.7055664,
        -0.49560547,
        -0.49560547,
        0.7055664,
        -0.49560547,
        0.7055664,
        0.7055664,
        0.7055664,
        0.7055664
    ]
);

tlvq_coder_test!(
    tlvq2_coder,
    TurboPrimaryCoder::<2>,
    PrimaryVectorHeader {
        l2_norm: 2.5226507,
        perpendicular_error_term: 0.67131084,
        parallel_error_term: 0.0073165894,
        lower: -0.67089844,
        upper: 0.8408203,
        component_sum: 32,
    },
    [
        -0.67089844,
        -0.16699219,
        0.8408203,
        0.8408203,
        0.33691406,
        0.33691406,
        0.8408203,
        -0.16699219,
        -0.16699219,
        -0.67089844,
        0.8408203,
        -0.67089844,
        -0.16699219,
        0.33691406,
        -0.67089844,
        0.33691406,
        0.8408203,
        0.8408203,
        0.33691406
    ]
);

tlvq_coder_test!(
    tlvq4_coder,
    TurboPrimaryCoder::<4>,
    PrimaryVectorHeader {
        l2_norm: 2.5226507,
        perpendicular_error_term: 0.118480206,
        parallel_error_term: 0.00030064583,
        lower: -0.9345703,
        upper: 0.91308594,
        component_sum: 170,
    },
    [
        -0.9345703,
        -0.07233074,
        0.6667317,
        0.6667317,
        0.54355466,
        0.42037758,
        0.6667317,
        0.05084634,
        -0.19550782,
        -0.441862,
        0.7899088,
        -0.68821615,
        -0.3186849,
        0.54355466,
        -0.68821615,
        0.42037758,
        0.91308594,
        0.6667317,
        0.17402342
    ]
);

tlvq_coder_test!(
    tlvq8_coder,
    TurboPrimaryCoder::<8>,
    PrimaryVectorHeader {
        l2_norm: 2.5226507,
        perpendicular_error_term: 0.0075108674,
        parallel_error_term: -0.00057935715,
        lower: -0.9199219,
        upper: 0.9116211,
        component_sum: 2875,
    },
    [
        -0.9199219,
        -0.05801932,
        0.6602328,
        0.6674153,
        0.57404256,
        0.43039212,
        0.64586776,
        -0.0005591512,
        -0.20166975,
        -0.43151042,
        0.73205805,
        -0.70444626,
        -0.27349496,
        0.5381299,
        -0.73317635,
        0.43757465,
        0.91162103,
        0.6961454,
        0.20055145
    ]
);

// Deterministic 135-element test vectors (using a simple LCG).
// 135 is deliberately chosen: it is not a multiple of 8, 16, 32, 64, or 128, so it exercises
// the scalar tail paths of every SIMD kernel.
fn lvq_test_vecs_135() -> ([Vec<f32>; 3], Vec<f32>) {
    let mut s = 0xDEAD_BEEFu32;
    let mut next = || {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (s >> 16) as f32 / 32_768.0 - 1.0 // uniform in [-1, 1)
    };
    let vecs = std::array::from_fn(|_| (0..135).map(|_| next()).collect::<Vec<_>>());
    // Center has smaller magnitude, as is typical for a dataset centroid.
    let center: Vec<f32> = (0..135).map(|_| next() * 0.4).collect();
    (vecs, center)
}

// With `a` and `b` centered against `center` and encoded with `format`, verify that the symmetric
// (doc-doc), symmetric-query (encoded-query vs doc) and asymmetric-query (f32-query vs doc)
// distances all agree with the f32 reference -- i.e. that centering stays transparent to distance.
//
// The tolerance is derived from the per-vector residual norms e_a = ‖a − decode(encode(a))‖₂
// and e_b = ‖b − decode(encode(b))‖₂. For squared-Euclidean the bound is tight:
//   |d(a,b) − d(qa,qb)| ≤ (e_a + e_b) · (e_a + e_b + 2‖a−b‖₂)
// For Dot (distance ∈ [0,1]) the same formula is conservative since √f32_dist ≤ 1.
fn check_lvq_centered_distance(
    format: F32VectorCoding,
    sim: VectorSimilarity,
    a: &[f32],
    b: &[f32],
    center: &[f32],
) {
    // Dot similarity assumes l2-normalized inputs.
    let (a, b) = if sim == VectorSimilarity::Dot {
        (
            l2_normalize(a).0.into_owned(),
            l2_normalize(b).0.into_owned(),
        )
    } else {
        (a.to_vec(), b.to_vec())
    };

    // The reference distance is between the uncentered (normalized) vectors: centering is a shared
    // translation that cancels in the squared-Euclidean term the codec estimates.
    let f32_dist = sim.distance_f32().distance_f32(&a, &b);

    // The coder no longer centers; normalization already happened above (Dot), so this prepare
    // only subtracts the center.
    let a = crate::prepare_vector(&a, None, false, Some(center));
    let b = crate::prepare_vector(&b, None, false, Some(center));

    let coder = format.coder();
    let enc_a = coder.encode(&a);
    let enc_b = coder.encode(&b);

    let residual_norm = |orig: &[f32], enc: &[u8]| -> f64 {
        let decoded = coder.decode(enc);
        orig.iter()
            .zip(decoded.iter())
            .map(|(o, d)| (*o - *d) as f64 * (*o - *d) as f64)
            .sum::<f64>()
            .sqrt()
    };
    let ea = residual_norm(&a, &enc_a);
    let eb = residual_norm(&b, &enc_b);
    let abs_epsilon = (ea + eb) * (ea + eb + 2.0 * f32_dist.abs().sqrt());

    // doc-doc symmetric distance
    let sym = format.distance_symmetric(sim).distance(&enc_a, &enc_b);
    assert_abs_diff_eq!(f32_dist, sym, epsilon = abs_epsilon);

    // encoded-query vs doc distance
    let qd = format
        .query_distance_symmetric(sim, enc_a.as_slice())
        .distance(&enc_b);
    assert_abs_diff_eq!(f32_dist, qd, epsilon = abs_epsilon);

    // f32-query vs doc distance
    let qda = format
        .query_distance_asymmetric(sim, a.as_slice())
        .distance(&enc_b);
    assert_abs_diff_eq!(f32_dist, qda, epsilon = abs_epsilon);
}

macro_rules! lvq_centered_distance_135_test {
    ($name:ident, $format:expr) => {
        #[test]
        fn $name() {
            let (vecs, center) = lvq_test_vecs_135();
            let pairs = [
                (&vecs[0], &vecs[1]),
                (&vecs[1], &vecs[2]),
                (&vecs[0], &vecs[2]),
            ];
            for (a, b) in pairs {
                for sim in [VectorSimilarity::Dot, VectorSimilarity::Euclidean] {
                    check_lvq_centered_distance($format, sim, a, b, &center);
                }
            }
        }
    };
}

lvq_centered_distance_135_test!(centered_distance_135_tlvq1, F32VectorCoding::TLVQ1);
lvq_centered_distance_135_test!(centered_distance_135_tlvq2, F32VectorCoding::TLVQ2);
lvq_centered_distance_135_test!(centered_distance_135_tlvq4, F32VectorCoding::TLVQ4);
lvq_centered_distance_135_test!(centered_distance_135_tlvq8, F32VectorCoding::TLVQ8);

#[test]
fn null_vector_decode() {
    let vector = vec![0.0f32; 256];
    for coding in [
        F32VectorCoding::TLVQ1,
        F32VectorCoding::TLVQ2,
        F32VectorCoding::TLVQ4,
        F32VectorCoding::TLVQ8,
    ] {
        let coder = coding.coder();
        let encoded = coder.encode(&vector);
        let decoded = coder.decode(&encoded);
        assert_abs_diff_eq!(decoded.as_slice(), vector.as_ref());
    }
}

#[test]
fn fill_vector_decode() {
    let vector = vec![1.0f32; 256];
    for coding in [
        F32VectorCoding::TLVQ1,
        F32VectorCoding::TLVQ2,
        F32VectorCoding::TLVQ4,
        F32VectorCoding::TLVQ8,
    ] {
        let coder = coding.coder();
        let encoded = coder.encode(&vector);
        let decoded = coder.decode(&encoded);
        assert_abs_diff_eq!(decoded.as_slice(), vector.as_ref());
    }
}

macro_rules! lvq_coding_simd_test {
    ($name:ident, $coder:ty) => {
        #[test]
        fn $name() {
            use crate::lvq::Kernel;

            let seed = SysRng::default().try_next_u64().unwrap();
            println!("SEED {seed:#016x}");
            let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed);
            let scoder = <$coder>::with_kernel(Kernel::Scalar);
            for k in Kernel::accelerated() {
                let ocoder = <$coder>::new();
                // TODO: use randomly sized vectors like we do for distance tests.
                for i in 0..1024 {
                    let vec = l2_normalize(
                        (0..128)
                            .map(|_| rng.random_range(-1.0f32..=1.0))
                            .collect::<Vec<_>>(),
                    ).0;
                    // SIMD and scalar interval/statistics paths may round slightly
                    // differently, shifting the residual by a level or two. Rather
                    // than requiring bit-identical decodes, compare the mean squared
                    // error of each decode against the original vector: both coders
                    // should reconstruct the vector just as well.
                    let mse = |decoded: &Vec<f32>| {
                        decoded
                            .iter()
                            .zip(vec.iter())
                            .map(|(d, o)| (d - o) * (d - o))
                            .sum::<f32>()
                            / decoded.len() as f32
                    };
                    let smse = mse(&scoder.decode(&scoder.encode(&vec)));
                    let omse = mse(&ocoder.decode(&ocoder.encode(&vec)));
                    assert!(
                        smse.abs_diff_eq(&omse, 1e-5),
                        "index {i} scalar mse {smse:.9} vs optimized mse {omse:.9} kernel {k:?} input vector {vec:?}"
                    );
                }
            }
        }
    };
}

lvq_coding_simd_test!(tlvq1_coding_simd, TurboPrimaryCoder::<1>);
lvq_coding_simd_test!(tlvq2_coding_simd, TurboPrimaryCoder::<2>);
lvq_coding_simd_test!(tlvq4_coding_simd, TurboPrimaryCoder::<4>);
lvq_coding_simd_test!(tlvq8_coding_simd, TurboPrimaryCoder::<8>);

/// Reconstruct the 4 bit dimension values from the output of `bitplane_split4`.
///
/// The head of the split is interleaved in 64 byte groups (4 x 16 byte bitplanes covering 128
/// dimensions); the tail is packed as 4 equally sized single bit turbo packed bitplanes.
fn bitplane_join4(split: &[u8], dimensions: usize) -> Vec<u8> {
    use crate::packing::TurboUnpacker;

    let head_groups = crate::packing::byte_len(dimensions, 4) / 64;
    let (head, tail) = split.split_at(head_groups * 64);
    let mut out = Vec::with_capacity(dimensions);
    for group in head.as_chunks::<64>().0 {
        let planes = group.as_chunks::<16>().0;
        // Each 16 byte bitplane covers 4 turbo blocks of 32 dimensions each. Within a plane byte
        // the low nibble of input byte `p` of block `i` lands at bit `i * 2` and the high nibble
        // at bit `i * 2 + 1`.
        for i in 0..4 {
            for half in 0..2 {
                for p in 0..16 {
                    let mut v = 0u8;
                    for (k, plane) in planes.iter().enumerate() {
                        v |= ((plane[p] >> (i * 2 + half)) & 1) << k;
                    }
                    out.push(v);
                }
            }
        }
    }

    if !tail.is_empty() {
        let mut planes = tail
            .chunks(tail.len() / 4)
            .map(|p| TurboUnpacker::<1>::new(p));
        let mut b0 = planes.next().unwrap();
        let mut b1 = planes.next().unwrap();
        let mut b2 = planes.next().unwrap();
        let mut b3 = planes.next().unwrap();
        while out.len() < dimensions {
            let v = b0.next().unwrap()
                | (b1.next().unwrap() << 1)
                | (b2.next().unwrap() << 2)
                | (b3.next().unwrap() << 3);
            out.push(v);
        }
    }

    out.truncate(dimensions);
    out
}

#[test]
fn bitplane_split4_roundtrip() {
    use crate::packing::{TurboPacker, bitplane_split4, byte_len};

    let seed = SysRng::default().try_next_u64().unwrap();
    println!("SEED {seed:#016x}");
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed);
    // Cover vectors with and without a head, and with block aligned and unaligned tails.
    for dimensions in [8, 32, 40, 96, 128, 136, 160, 256, 384, 391, 1024, 1536] {
        let dims = (0..dimensions)
            .map(|_| rng.random_range(0u8..16))
            .collect::<Vec<_>>();
        let mut packed = vec![0u8; byte_len(dimensions, 4)];
        let mut packer = TurboPacker::<4>::new(&mut packed);
        for d in dims.iter().copied() {
            packer.push(d);
        }

        let split = bitplane_split4(&packed);
        assert_eq!(split.len(), dimensions.div_ceil(8) * 4);
        assert_eq!(bitplane_join4(&split, dimensions), dims, "dim {dimensions}");
    }
}
