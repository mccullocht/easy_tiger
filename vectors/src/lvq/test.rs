use approx::{AbsDiffEq, abs_diff_eq, assert_abs_diff_eq};

use crate::lvq::{
    PrimaryVectorHeader, ResidualVectorHeader, TurboPrimaryCoder, TurboResidualCoder, VectorStats,
};
use crate::{F32VectorCoder, F32VectorCoding, VectorSimilarity, l2_normalize};

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
                self.residual_error_term,
                other.residual_error_term,
                epsilon = epsilon
            )
            && abs_diff_eq!(self.component_sum, other.component_sum)
    }
}

impl AbsDiffEq for ResidualVectorHeader {
    type Epsilon = f32;

    fn default_epsilon() -> Self::Epsilon {
        0.00001
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        // TODO: tlvq8x8 fails on aarch64 when epsilon = 0; figure this out
        abs_diff_eq!(self.magnitude, other.magnitude, epsilon = epsilon)
            && abs_diff_eq!(self.component_sum, other.component_sum, epsilon = 1)
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

const TEST_CENTER: [f32; 19] = [
    -0.98, -0.028, 0.456, 0.587, 0.975, 0.837, 0.325, 0.636, -0.448, -0.046, 0.693, -0.64, -0.5,
    -0.036, -0.036, 0.376, 0.629, 0.221, 0.57,
];

#[test]
fn vector_stats_simd() {
    let simd_stats = VectorStats::from(TEST_VECTOR.as_ref());
    let scalar_stats = VectorStats::from_scalar(TEST_VECTOR.as_ref());
    assert_abs_diff_eq!(simd_stats, scalar_stats);
}

enum Centering {
    Uncentered,
    Centered,
}

macro_rules! tlvq_coder_test {
    ($name:ident, $coder:ty, $center:expr, $primary_header:expr, $decoded:expr) => {
        #[test]
        fn $name() {
            let coder = match $center {
                Centering::Uncentered => <$coder>::new(VectorSimilarity::Euclidean, None),
                Centering::Centered => {
                    <$coder>::new(VectorSimilarity::Euclidean, Some(TEST_CENTER.to_vec()))
                }
            };
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
    ($name:ident, $coder:ty, $center:expr, $primary_header:expr, $residual_header:expr, $decoded:expr) => {
        #[test]
        fn $name() {
            let coder = match $center {
                Centering::Uncentered => <$coder>::new(VectorSimilarity::Euclidean, None),
                Centering::Centered => {
                    <$coder>::new(VectorSimilarity::Euclidean, Some(TEST_CENTER.to_vec()))
                }
            };
            let encoded = coder.encode(&TEST_VECTOR);
            let (primary_header, vector_bytes) =
                PrimaryVectorHeader::deserialize(&encoded).unwrap();
            assert_abs_diff_eq!(primary_header, $primary_header);
            assert_abs_diff_eq!(
                ResidualVectorHeader::deserialize(&vector_bytes).unwrap().0,
                $residual_header
            );
            let mut decoded = vec![0.0f32; TEST_VECTOR.len()];
            coder.decode_to(&encoded, &mut decoded);
            assert_abs_diff_eq!(decoded.as_ref(), $decoded.as_ref(), epsilon = 0.00001);
        }
    };
}

tlvq_coder_test!(
    tlvq1_uncentered,
    TurboPrimaryCoder::<1>,
    Centering::Uncentered,
    PrimaryVectorHeader {
        l2_norm: 2.5234375,
        residual_error_term: 1.1640625,
        center_dot: 0.0,
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
    tlvq1_centered,
    TurboPrimaryCoder::<1>,
    Centering::Centered,
    PrimaryVectorHeader {
        l2_norm: 1.5517578,
        residual_error_term: 0.8178711,
        center_dot: 0.0,
        lower: -0.60498047,
        upper: 0.23901367,
        component_sum: 13,
    },
    [
        -0.74098635,
        0.21101367,
        0.69501364,
        0.8260137,
        0.37001956,
        0.23201954,
        0.56401366,
        0.031019509,
        -0.20898634,
        -0.6509805,
        0.9320137,
        -0.4009863,
        -0.26098633,
        0.20301367,
        -0.6409805,
        0.61501366,
        0.8680137,
        0.4600137,
        -0.034980476
    ]
);

tlvq_coder_test!(
    tlvq2_uncentered,
    TurboPrimaryCoder::<2>,
    Centering::Uncentered,
    PrimaryVectorHeader {
        l2_norm: 2.5234375,
        residual_error_term: 0.6713867,
        center_dot: 0.0,
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
    tlvq2_centered,
    TurboPrimaryCoder::<2>,
    Centering::Centered,
    PrimaryVectorHeader {
        l2_norm: 1.5517578,
        residual_error_term: 0.53271484,
        center_dot: 0.0,
        lower: -0.5571289,
        upper: 0.5683594,
        component_sum: 27,
    },
    [
        -0.7868034,
        -0.20996615,
        0.6491966,
        0.7801966,
        0.41787112,
        0.2798711,
        0.5181966,
        0.07887107,
        -0.25480342,
        -0.6031289,
        0.8861966,
        -0.8219662,
        -0.3068034,
        0.53235936,
        -0.5931289,
        0.5691966,
        0.8221966,
        0.7893594,
        0.38803384
    ]
);

tlvq_coder_test!(
    tlvq4_uncentered,
    TurboPrimaryCoder::<4>,
    Centering::Uncentered,
    PrimaryVectorHeader {
        l2_norm: 2.5234375,
        residual_error_term: 0.11846924,
        center_dot: 0.0,
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
    tlvq4_centered,
    TurboPrimaryCoder::<4>,
    Centering::Centered,
    PrimaryVectorHeader {
        l2_norm: 1.5517578,
        residual_error_term: 0.095947266,
        center_dot: 0.0,
        lower: -0.69091797,
        upper: 0.56347656,
        component_sum: 152,
    },
    [
        -0.9182813,
        -0.04990757,
        0.68497133,
        0.6487187,
        0.534961,
        0.39696094,
        0.6375976,
        0.02870828,
        -0.21902868,
        -0.40241277,
        0.7547188,
        -0.7455338,
        -0.27102867,
        0.52747655,
        -0.726918,
        0.43771872,
        0.94159764,
        0.70085025,
        0.21358722
    ]
);

tlvq_coder_test!(
    tlvq8_uncentered,
    TurboPrimaryCoder::<8>,
    Centering::Uncentered,
    PrimaryVectorHeader {
        l2_norm: 2.5234375,
        residual_error_term: 0.0077056885,
        center_dot: 0.0,
        lower: -0.9199219,
        upper: 0.9116211,
        component_sum: 2876,
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
        -0.4243279,
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

tlvq_coder_test!(
    tlvq8_centered,
    TurboPrimaryCoder::<8>,
    Centering::Centered,
    PrimaryVectorHeader {
        l2_norm: 1.5517578,
        residual_error_term: 0.0055274963,
        center_dot: 0.0,
        lower: -0.6953125,
        upper: 0.57470703,
        component_sum: 2569,
    },
    [
        -0.92326176,
        -0.06091017,
        0.65717185,
        0.6686406,
        0.5735352,
        0.4305547,
        0.6457031,
        0.0004531145,
        -0.20200394,
        -0.42754298,
        0.72981644,
        -0.70279294,
        -0.27392578,
        0.538707,
        -0.7313125,
        0.43771872,
        0.91483986,
        0.6960976,
        0.20339844
    ]
);

tlvq_coder_test!(
    tlvq1x8_uncentered,
    TurboResidualCoder::<1>,
    Centering::Uncentered,
    PrimaryVectorHeader {
        l2_norm: 2.5234375,
        residual_error_term: 1.1640625,
        center_dot: 0.0,
        lower: -0.49560547,
        upper: 0.7055664,
        component_sum: 11,
    },
    ResidualVectorHeader {
        magnitude: 1.2012575,
        component_sum: 2292,
    },
    [
        -0.9219341,
        -0.059855193,
        0.6608137,
        0.6702353,
        0.5713082,
        0.42998376,
        0.6466812,
        0.0013853908,
        -0.2011796,
        -0.42729867,
        0.7314759,
        -0.7052367,
        -0.27184182,
        0.53833246,
        -0.72879076,
        0.4346946,
        0.9151976,
        0.69378936,
        0.2038647
    ]
);

tlvq_coder_test!(
    tlvq1x8_centered,
    TurboResidualCoder::<1>,
    Centering::Centered,
    PrimaryVectorHeader {
        l2_norm: 1.5517578,
        residual_error_term: 0.8178711,
        center_dot: 0.0,
        lower: -0.60498047,
        upper: 0.23901367,
        component_sum: 13,
    },
    ResidualVectorHeader {
        magnitude: 0.84397864,
        component_sum: 2453,
    },
    [
        -0.9213661,
        -0.062038247,
        0.66026163,
        0.66880196,
        0.57356733,
        0.43225762,
        0.6451018,
        -0.00042283535,
        -0.20071204,
        -0.42757434,
        0.7284659,
        -0.7038257,
        -0.27257034,
        0.53895026,
        -0.7319978,
        0.4346339,
        0.91269493,
        0.69334894,
        0.2016645
    ]
);

tlvq_coder_test!(
    tlvq2x8_uncentered,
    TurboResidualCoder::<2>,
    Centering::Uncentered,
    PrimaryVectorHeader {
        l2_norm: 2.5234375,
        residual_error_term: 0.6713867,
        center_dot: 0.0,
        lower: -0.67089844,
        upper: 0.8408203,
        component_sum: 32,
    },
    ResidualVectorHeader {
        magnitude: 0.5039812,
        component_sum: 2319,
    },
    [
        -0.9209126,
        -0.06125498,
        0.6580036,
        0.669862,
        0.5730934,
        0.4307929,
        0.6461452,
        1.3321638e-5,
        -0.19960275,
        -0.42878985,
        0.7291539,
        -0.703509,
        -0.27272943,
        0.5394947,
        -0.7311785,
        0.43672207,
        0.9129588,
        0.6935787,
        0.20153087
    ]
);

tlvq_coder_test!(
    tlvq2x8_centered,
    TurboResidualCoder::<2>,
    Centering::Centered,
    PrimaryVectorHeader {
        l2_norm: 1.5517578,
        residual_error_term: 0.53271484,
        center_dot: 0.0,
        lower: -0.5571289,
        upper: 0.5683594,
        component_sum: 27,
    },
    ResidualVectorHeader {
        magnitude: 0.37513867,
        component_sum: 2453,
    },
    [
        -0.921412,
        -0.060646255,
        0.65875894,
        0.67059726,
        0.57307553,
        0.43066216,
        0.6469206,
        0.0016366243,
        -0.19963597,
        -0.42732865,
        0.72952104,
        -0.70354,
        -0.27223182,
        0.5389795,
        -0.73067975,
        0.43605912,
        0.9126712,
        0.69447136,
        0.20193565
    ]
);

tlvq_coder_test!(
    tlvq4x8_uncentered,
    TurboResidualCoder::<4>,
    Centering::Uncentered,
    PrimaryVectorHeader {
        l2_norm: 2.5234375,
        residual_error_term: 0.11846924,
        center_dot: 0.0,
        lower: -0.9345703,
        upper: 0.91308594,
        component_sum: 170,
    },
    ResidualVectorHeader {
        magnitude: 0.123191215,
        component_sum: 2407,
    },
    [
        -0.9208019,
        -0.060977824,
        0.65876055,
        0.66987187,
        0.5727824,
        0.43124738,
        0.6461998,
        0.00084519386,
        -0.20009731,
        -0.42809355,
        0.7297625,
        -0.70391697,
        -0.27303168,
        0.53896517,
        -0.73097074,
        0.43607843,
        0.91284436,
        0.694027,
        0.20228493
    ]
);

tlvq_coder_test!(
    tlvq4x8_centered,
    TurboResidualCoder::<4>,
    Centering::Centered,
    PrimaryVectorHeader {
        l2_norm: 1.5517578,
        residual_error_term: 0.095947266,
        center_dot: 0.0,
        lower: -0.69091797,
        upper: 0.56347656,
        component_sum: 152,
    },
    ResidualVectorHeader {
        magnitude: 0.0836155,
        component_sum: 2425,
    },
    [
        -0.9210685,
        -0.060892347,
        0.658903,
        0.6698685,
        0.5728339,
        0.430899,
        0.64595914,
        0.0010004044,
        -0.1998463,
        -0.42815322,
        0.729962,
        -0.704054,
        -0.27283216,
        0.53911716,
        -0.73101676,
        0.43591526,
        0.91290605,
        0.6941282,
        0.20194665
    ]
);

// Use a larger epsilon for tlvq8x8 because the primary dequantize intermediate value can differ
// by ~1 ULP between architectures (e.g. x86_64 AVX512 vs aarch64 Neon), which causes the
// residual to round to a different index (~1 residual step ≈ 0.000028).
#[test]
fn tlvq8x8() {
    let coder = TurboResidualCoder::<8>::new(VectorSimilarity::Euclidean, None);
    let encoded = coder.encode(&TEST_VECTOR);
    let (primary_header, vector_bytes) = PrimaryVectorHeader::deserialize(&encoded).unwrap();
    assert_abs_diff_eq!(
        primary_header,
        PrimaryVectorHeader {
            l2_norm: 2.5234375,
            lower: -0.9199219,
            upper: 0.9116211,
            residual_error_term: 0.0077056885,
            center_dot: 0.0,
            component_sum: 2876,
        }
    );
    assert_abs_diff_eq!(
        ResidualVectorHeader::deserialize(&vector_bytes).unwrap().0,
        ResidualVectorHeader {
            magnitude: 0.0071822493,
            component_sum: 2422,
        }
    );
    let mut decoded = vec![0.0f32; TEST_VECTOR.len()];
    coder.decode_to(&encoded, &mut decoded);
    assert_abs_diff_eq!(
        decoded.as_ref(),
        [
            -0.92092174f32,
            -0.06087814,
            0.6591484,
            0.6701333,
            0.57312715,
            0.43113852,
            0.64613533,
            0.0011167069,
            -0.19988123,
            -0.42789087,
            0.73015684,
            -0.703897,
            -0.2728894,
            0.5391298,
            -0.730909,
            0.43612412,
            0.9131561,
            0.69415975,
            0.20211464
        ]
        .as_ref(),
        epsilon = 0.00005
    );
}

tlvq_coder_test!(
    tlvq8x8_centered,
    TurboResidualCoder::<8>,
    Centering::Centered,
    PrimaryVectorHeader {
        l2_norm: 1.5517578,
        residual_error_term: 0.0055274963,
        center_dot: 0.0,
        lower: -0.6953125,
        upper: 0.57470703,
        component_sum: 2569,
    },
    ResidualVectorHeader {
        magnitude: 0.0049809623,
        component_sum: 2425,
    },
    [
        -0.92106426,
        -0.06105667,
        0.65892005,
        0.6699396,
        0.5729785,
        0.43097466,
        0.64590824,
        0.0010098219,
        -0.20007992,
        -0.42802155,
        0.7299434,
        -0.70405287,
        -0.27307612,
        0.5388731,
        -0.73099023,
        0.43593144,
        0.9129158,
        0.69390017,
        0.20198229
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

// Encode `a` and `b` with `format`, then verify that both `distance_symmetric` (doc-doc)
// and `query_distance_symmetric` (encoded-query vs doc) agree with the f32 reference.
//
// The tolerance is derived from the per-vector residual norms e_a = ‖a − decode(encode(a))‖₂
// and e_b = ‖b − decode(encode(b))‖₂. For squared-Euclidean the bound is tight:
//   |d(a,b) − d(qa,qb)| ≤ (e_a + e_b) · (e_a + e_b + 2‖a−b‖₂)
// For Dot (distance ∈ [0,1]) the same formula is conservative since √f32_dist ≤ 1.
fn check_lvq_distance(
    format: F32VectorCoding,
    sim: VectorSimilarity,
    a: &[f32],
    b: &[f32],
    center: Option<&[f32]>,
) {
    // Dot similarity assumes l2-normalized inputs.
    let (a, b) = if sim == VectorSimilarity::Dot {
        (l2_normalize(a).into_owned(), l2_normalize(b).into_owned())
    } else {
        (a.to_vec(), b.to_vec())
    };

    let f32_dist = sim.new_distance_function().distance_f32(&a, &b);

    let coder = format.coder(sim, center.map(|c| c.to_vec()));
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
    let sym = format
        .distance_symmetric(sim, center)
        .distance(&enc_a, &enc_b);
    assert_abs_diff_eq!(f32_dist, sym, epsilon = abs_epsilon);

    // encoded-query vs doc distance
    let qd = format
        .query_distance_symmetric(sim, enc_a.as_slice(), center)
        .distance(&enc_b);
    assert_abs_diff_eq!(f32_dist, qd, epsilon = abs_epsilon);
}

macro_rules! lvq_distance_135_test {
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
                    // uncentered
                    check_lvq_distance($format, sim, a, b, None);
                    // centered
                    check_lvq_distance($format, sim, a, b, Some(center.as_slice()));
                }
            }
        }
    };
}

lvq_distance_135_test!(distance_135_tlvq1, F32VectorCoding::TLVQ1);
lvq_distance_135_test!(distance_135_tlvq2, F32VectorCoding::TLVQ2);
lvq_distance_135_test!(distance_135_tlvq4, F32VectorCoding::TLVQ4);
lvq_distance_135_test!(distance_135_tlvq8, F32VectorCoding::TLVQ8);
lvq_distance_135_test!(distance_135_tlvq1x8, F32VectorCoding::TLVQ1x8);
lvq_distance_135_test!(distance_135_tlvq2x8, F32VectorCoding::TLVQ2x8);
lvq_distance_135_test!(distance_135_tlvq4x8, F32VectorCoding::TLVQ4x8);
lvq_distance_135_test!(distance_135_tlvq8x8, F32VectorCoding::TLVQ8x8);

#[test]
fn null_vector_decode() {
    let vector = vec![0.0f32; 256];
    for coding in [
        F32VectorCoding::TLVQ1,
        F32VectorCoding::TLVQ2,
        F32VectorCoding::TLVQ4,
        F32VectorCoding::TLVQ8,
        F32VectorCoding::TLVQ1x8,
        F32VectorCoding::TLVQ2x8,
        F32VectorCoding::TLVQ4x8,
        F32VectorCoding::TLVQ8x8,
    ] {
        let coder = coding.coder(VectorSimilarity::Euclidean, None);
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
        F32VectorCoding::TLVQ1x8,
        F32VectorCoding::TLVQ2x8,
        F32VectorCoding::TLVQ4x8,
        F32VectorCoding::TLVQ8x8,
    ] {
        let coder = coding.coder(VectorSimilarity::Euclidean, None);
        let encoded = coder.encode(&vector);
        let decoded = coder.decode(&encoded);
        assert_abs_diff_eq!(decoded.as_slice(), vector.as_ref());
    }
}
