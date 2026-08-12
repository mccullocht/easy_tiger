use approx::{AbsDiffEq, abs_diff_eq, assert_abs_diff_eq};
use half::f16;
use rand::{Rng, SeedableRng, TryRngCore, rngs::OsRng};

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
            && abs_diff_eq!(
                self.parallel_error_term,
                other.parallel_error_term,
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
                PrimaryVectorHeader::deserialize(&encoded, VectorSimilarity::Euclidean)
                    .unwrap()
                    .0,
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
                PrimaryVectorHeader::deserialize(&encoded, VectorSimilarity::Euclidean).unwrap();
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
        l2_norm: 2.5226507,
        residual_error_term: 1.1627843,
        parallel_error_term: 0.021774292,
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
        l2_norm: 1.5514041,
        residual_error_term: 0.81660825,
        parallel_error_term: 0.028503418,
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
        l2_norm: 2.5226507,
        residual_error_term: 0.67131084,
        parallel_error_term: 0.0073165894,
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
        l2_norm: 1.5514041,
        residual_error_term: 0.5321589,
        parallel_error_term: 0.011878967,
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
        l2_norm: 2.5226507,
        residual_error_term: 0.118480206,
        parallel_error_term: 0.00030064583,
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
        l2_norm: 1.5514041,
        residual_error_term: 0.09596851,
        parallel_error_term: 0.00025558472,
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
        l2_norm: 2.5226507,
        residual_error_term: 0.0075108674,
        parallel_error_term: -0.00057935715,
        center_dot: 0.0,
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

tlvq_coder_test!(
    tlvq8_centered,
    TurboPrimaryCoder::<8>,
    Centering::Centered,
    PrimaryVectorHeader {
        l2_norm: 1.5514041,
        residual_error_term: 0.005533458,
        parallel_error_term: 8.994341e-5,
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
        l2_norm: 2.5226507,
        residual_error_term: 1.1627843,
        parallel_error_term: 0.021774292,
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
        l2_norm: 1.5514041,
        residual_error_term: 0.81660825,
        parallel_error_term: 0.028503418,
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
        l2_norm: 2.5226507,
        residual_error_term: 0.67131084,
        parallel_error_term: 0.0073165894,
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
        0.65998,
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
        l2_norm: 1.5514041,
        residual_error_term: 0.5321589,
        parallel_error_term: 0.011878967,
        center_dot: 0.0,
        lower: -0.5571289,
        upper: 0.5683594,
        component_sum: 27,
    },
    ResidualVectorHeader {
        magnitude: 0.37513867,
        component_sum: 2451,
    },
    [
        -0.921412,
        -0.060646255,
        0.65875894,
        0.67059726,
        0.57307553,
        0.43066216,
        0.6454495,
        0.0016366243,
        -0.19963597,
        -0.42732865,
        0.72952104,
        -0.70354,
        -0.27370292,
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
        l2_norm: 2.5226507,
        residual_error_term: 0.118480206,
        parallel_error_term: 0.00030064583,
        center_dot: 0.0,
        lower: -0.9345703,
        upper: 0.91308594,
        component_sum: 170,
    },
    ResidualVectorHeader {
        magnitude: 0.12319123,
        component_sum: 2405,
    },
    [
        -0.9208019,
        -0.060977824,
        0.65876055,
        0.66987187,
        0.5727824,
        0.4307643,
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
        0.20180184
    ]
);

tlvq_coder_test!(
    tlvq4x8_centered,
    TurboResidualCoder::<4>,
    Centering::Centered,
    PrimaryVectorHeader {
        l2_norm: 1.5514041,
        residual_error_term: 0.09596851,
        parallel_error_term: 0.00025558472,
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
        0.5731618,
        0.430899,
        0.64595914,
        0.0010004044,
        -0.1998463,
        -0.42815322,
        0.729962,
        -0.704054,
        -0.27316004,
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
    let (primary_header, vector_bytes) =
        PrimaryVectorHeader::deserialize(&encoded, VectorSimilarity::Euclidean).unwrap();
    assert_abs_diff_eq!(
        primary_header,
        PrimaryVectorHeader {
            l2_norm: 2.5226507,
            lower: -0.9199219,
            upper: 0.9116211,
            residual_error_term: 0.0075108674,
            parallel_error_term: -0.00057935715,
            center_dot: 0.0,
            component_sum: 2875,
        }
    );
    assert_abs_diff_eq!(
        ResidualVectorHeader::deserialize(&vector_bytes).unwrap().0,
        ResidualVectorHeader {
            magnitude: 0.0071822493,
            component_sum: 2590,
        }
    );
    let mut decoded = vec![0.0f32; TEST_VECTOR.len()];
    coder.decode_to(&encoded, &mut decoded);
    assert_abs_diff_eq!(
        decoded.as_ref(),
        [
            -0.92100626f32,
            -0.060990803,
            0.65900755,
            0.6699925,
            0.57298636,
            0.43099767,
            0.6459945,
            0.0010040442,
            -0.1999939,
            -0.4280038,
            0.72998786,
            -0.7040097,
            -0.27300206,
            0.538989,
            -0.7309935,
            0.43601146,
            0.91298705,
            0.69399077,
            0.20200199
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
        l2_norm: 1.5514041,
        residual_error_term: 0.005533458,
        parallel_error_term: 8.994341e-5,
        center_dot: 0.0,
        lower: -0.6953125,
        upper: 0.57470703,
        component_sum: 2569,
    },
    ResidualVectorHeader {
        magnitude: 0.0049809623,
        component_sum: 2478,
    },
    [
        -0.92100567,
        -0.060998067,
        0.6589982,
        0.66999817,
        0.57299805,
        0.43099418,
        0.64600587,
        0.0009902716,
        -0.20000179,
        -0.428002,
        0.730002,
        -0.7039943,
        -0.27299798,
        0.53899026,
        -0.7310097,
        0.4360096,
        0.9129939,
        0.6939978,
        0.20200181
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

/// Compute the header error terms directly from a vector and its dequantized value.
///
/// This mirrors [`PrimaryVectorHeader::set_error_terms`] in plain f64 without any of the
/// accumulation the quantize kernels do, returning `(perpendicular_norm, parallel_fraction)`.
fn reference_error_terms(vector: &[f32], dequantized: &[f32]) -> (f32, f32) {
    let mut l2_norm_sq = 0.0f64;
    let mut residual_error_sq = 0.0f64;
    let mut residual_dot = 0.0f64;
    for (&v, &d) in vector.iter().zip(dequantized.iter()) {
        let v = f64::from(v);
        let r = v - f64::from(d);
        l2_norm_sq += v * v;
        residual_error_sq += r * r;
        residual_dot += v * r;
    }
    let parallel = residual_dot / l2_norm_sq;
    let perpendicular = (residual_error_sq - (residual_dot * residual_dot / l2_norm_sq)).max(0.0);
    // Both terms are stored as f16 -- the perpendicular norm relative to the vector magnitude --
    // so round the reference through the same representation before comparing. At high bit rates
    // the parallel term lands in f16's subnormal range, where only a few bits of mantissa survive;
    // that is harmless because the correction it applies is proportionally tiny.
    let l2_norm = l2_norm_sq.sqrt() as f32;
    (
        f16::from_f32(perpendicular.sqrt() as f32 / l2_norm).to_f32() * l2_norm,
        f16::from_f32(parallel as f32).to_f32(),
    )
}

fn assert_close_relative(actual: f32, expected: f32, tolerance: f32, what: &str) {
    let scale = expected.abs().max(1e-6);
    assert!(
        (actual - expected).abs() <= tolerance * scale,
        "{what}: expected {expected} got {actual} (relative tolerance {tolerance})"
    );
}

/// Check the residual error terms in the header against a reference computed from the decoded
/// vector.
///
/// This validates the terms the SIMD quantize kernels accumulate, and covers vectors far from unit
/// magnitude to exercise storing both terms dimensionless -- an absolute f16 would overflow or go
/// subnormal at these scales.
macro_rules! error_terms_test {
    ($name:ident, $coder:ty) => {
        #[test]
        fn $name() {
            let seed = OsRng::default().try_next_u64().unwrap();
            println!("SEED {seed:#016x}");
            let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed);
            for scale in [0.001f32, 1.0, 1000.0] {
                for i in 0..64 {
                    let dim = rng.random_range(128..=256);
                    let vector = (0..dim)
                        .map(|_| rng.random_range(-1.0f32..=1.0) * scale)
                        .collect::<Vec<_>>();
                    // Euclidean and uncentered so that the encoder quantizes the input vector as
                    // given and `decode` returns exactly its dequantized value.
                    let coder = <$coder>::new(VectorSimilarity::Euclidean, None);
                    let encoded = coder.encode(&vector);
                    let decoded = coder.decode(&encoded);
                    let (header, _) =
                        PrimaryVectorHeader::deserialize(&encoded, VectorSimilarity::Euclidean)
                            .unwrap();
                    let (perpendicular, parallel) = reference_error_terms(&vector, &decoded);
                    // Both terms are stored as f16, so only their relative precision survives.
                    assert_close_relative(
                        header.residual_error_term,
                        perpendicular,
                        0.005,
                        &format!("residual_error_term scale {scale} index {i}"),
                    );
                    assert_close_relative(
                        header.parallel_error_term,
                        parallel,
                        0.005,
                        &format!("parallel_error_term scale {scale} index {i}"),
                    );
                }
            }
        }
    };
}

error_terms_test!(tlvq1_error_terms, TurboPrimaryCoder::<1>);
error_terms_test!(tlvq2_error_terms, TurboPrimaryCoder::<2>);
error_terms_test!(tlvq4_error_terms, TurboPrimaryCoder::<4>);
error_terms_test!(tlvq8_error_terms, TurboPrimaryCoder::<8>);

/// Check that the scalar and SIMD quantize kernels agree on the header error terms.
///
/// The kernels accumulate in different orders so the terms will not be bit-identical, but a
/// backend that fails to accumulate `v . r` at all, or accumulates it over the wrong lanes, shows
/// up here. This is the only coverage the residual coders' terms get, since `decode` returns the
/// residual-corrected vector rather than the primary-only value those terms describe.
macro_rules! error_terms_simd_test {
    ($name:ident, $coder:ty) => {
        #[test]
        fn $name() {
            let seed = OsRng::default().try_next_u64().unwrap();
            println!("SEED {seed:#016x}");
            let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed);
            let scoder = <$coder>::scalar(VectorSimilarity::Euclidean, None);
            let ocoder = <$coder>::new(VectorSimilarity::Euclidean, None);
            for i in 0..256 {
                let dim = rng.random_range(128..=256);
                let vector = (0..dim)
                    .map(|_| rng.random_range(-1.0f32..=1.0))
                    .collect::<Vec<_>>();
                let (sheader, _) = PrimaryVectorHeader::deserialize(
                    &scoder.encode(&vector),
                    VectorSimilarity::Euclidean,
                )
                .unwrap();
                let (oheader, _) = PrimaryVectorHeader::deserialize(
                    &ocoder.encode(&vector),
                    VectorSimilarity::Euclidean,
                )
                .unwrap();
                assert_close_relative(
                    oheader.residual_error_term,
                    sheader.residual_error_term,
                    0.005,
                    &format!("residual_error_term index {i}"),
                );
                assert_close_relative(
                    oheader.parallel_error_term,
                    sheader.parallel_error_term,
                    0.005,
                    &format!("parallel_error_term index {i}"),
                );
            }
        }
    };
}

error_terms_simd_test!(tlvq1_error_terms_simd, TurboPrimaryCoder::<1>);
error_terms_simd_test!(tlvq2_error_terms_simd, TurboPrimaryCoder::<2>);
error_terms_simd_test!(tlvq4_error_terms_simd, TurboPrimaryCoder::<4>);
error_terms_simd_test!(tlvq8_error_terms_simd, TurboPrimaryCoder::<8>);
error_terms_simd_test!(tlvq1x8_error_terms_simd, TurboResidualCoder::<1>);
error_terms_simd_test!(tlvq2x8_error_terms_simd, TurboResidualCoder::<2>);
error_terms_simd_test!(tlvq4x8_error_terms_simd, TurboResidualCoder::<4>);
error_terms_simd_test!(tlvq8x8_error_terms_simd, TurboResidualCoder::<8>);

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
