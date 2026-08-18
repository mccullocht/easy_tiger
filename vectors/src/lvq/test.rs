use approx::{AbsDiffEq, abs_diff_eq, assert_abs_diff_eq};
use rand::{Rng, SeedableRng, TryRngCore, rngs::OsRng};

use crate::lvq::{
    PrimaryVectorHeader, ResidualVectorHeader, TurboPrimaryCoder, TurboResidualCoder, VectorStats,
};
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
    // XXX this should test all available SIMD kernels.
    let simd_stats = VectorStats::new(super::Kernel::default(), TEST_VECTOR.as_ref());
    let scalar_stats = VectorStats::new(super::Kernel::Scalar, TEST_VECTOR.as_ref());
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
        residual_error_term: 1.163828,
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
        residual_error_term: 0.8179195,
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
        residual_error_term: 0.6713635,
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
        residual_error_term: 0.53255266,
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
        residual_error_term: 0.11848368,
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
        residual_error_term: 0.09596872,
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
        residual_error_term: 0.0076497365,
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
        residual_error_term: 0.00552745,
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
        residual_error_term: 1.163828,
        center_dot: 0.0,
        lower: -0.49560547,
        upper: 0.7055664,
        component_sum: 11,
    },
    ResidualVectorHeader {
        magnitude: 1.2011719,
        component_sum: 2292,
    },
    [
        -0.9219037,
        -0.059886307,
        0.66081685,
        0.6702378,
        0.5713178,
        0.43000343,
        0.6466854,
        0.001349926,
        -0.20120063,
        -0.42730355,
        0.73147404,
        -0.7052218,
        -0.2718578,
        0.53834444,
        -0.72877413,
        0.4347139,
        0.91518265,
        0.6937902,
        0.20390052
    ]
);

tlvq_coder_test!(
    tlvq1x8_centered,
    TurboResidualCoder::<1>,
    Centering::Centered,
    PrimaryVectorHeader {
        l2_norm: 1.5514041,
        residual_error_term: 0.8179195,
        center_dot: 0.0,
        lower: -0.60498047,
        upper: 0.23901367,
        component_sum: 13,
    },
    ResidualVectorHeader {
        magnitude: 0.84399414,
        component_sum: 2453,
    },
    [
        -0.92136943,
        -0.062043253,
        0.660261,
        0.6687991,
        0.5735711,
        0.4322613,
        0.64510334,
        -0.0004234314,
        -0.20071189,
        -0.42757025,
        0.72846216,
        -0.70383126,
        -0.27257055,
        0.5389564,
        -0.73199946,
        0.4346306,
        0.91269577,
        0.69335324,
        0.20166886
    ]
);

tlvq_coder_test!(
    tlvq2x8_uncentered,
    TurboResidualCoder::<2>,
    Centering::Uncentered,
    PrimaryVectorHeader {
        l2_norm: 2.5226507,
        residual_error_term: 0.6713635,
        center_dot: 0.0,
        lower: -0.67089844,
        upper: 0.8408203,
        component_sum: 32,
    },
    ResidualVectorHeader {
        magnitude: 0.50390625,
        component_sum: 2320,
    },
    [
        -0.92087543,
        -0.06127066,
        0.6580308,
        0.6698874,
        0.57305837,
        0.43077898,
        0.6461742,
        0.0019646436,
        -0.19959787,
        -0.4288258,
        0.7291705,
        -0.70350415,
        -0.2727137,
        0.53946465,
        -0.7311696,
        0.4367073,
        0.9129481,
        0.69360065,
        0.20155102
    ]
);

tlvq_coder_test!(
    tlvq2x8_centered,
    TurboResidualCoder::<2>,
    Centering::Centered,
    PrimaryVectorHeader {
        l2_norm: 1.5514041,
        residual_error_term: 0.53255266,
        center_dot: 0.0,
        lower: -0.5571289,
        upper: 0.5683594,
        component_sum: 27,
    },
    ResidualVectorHeader {
        magnitude: 0.37516275,
        component_sum: 2451,
    },
    [
        -0.92142063,
        -0.060636673,
        0.6587596,
        0.6705902,
        0.5730855,
        0.4306718,
        0.6454577,
        0.0016316772,
        -0.19963244,
        -0.42731735,
        0.729511,
        -0.7035324,
        -0.2737008,
        0.5389799,
        -0.7306886,
        0.4360506,
        0.91267705,
        0.6944653,
        0.20192367
    ]
);

tlvq_coder_test!(
    tlvq4x8_uncentered,
    TurboResidualCoder::<4>,
    Centering::Uncentered,
    PrimaryVectorHeader {
        l2_norm: 2.5226507,
        residual_error_term: 0.118475676,
        center_dot: 0.0,
        lower: -0.9345703,
        upper: 0.91308594,
        component_sum: 170,
    },
    ResidualVectorHeader {
        magnitude: 0.12317708,
        component_sum: 2405,
    },
    [
        -0.9208035,
        -0.060979128,
        0.65876144,
        0.6698715,
        0.57277906,
        0.4307631,
        0.6462022,
        0.00085093454,
        -0.20009677,
        -0.42809513,
        0.7297694,
        -0.7039152,
        -0.27303693,
        0.5389657,
        -0.73096585,
        0.4360766,
        0.9128444,
        0.6940239,
        0.20179865
    ]
);

tlvq_coder_test!(
    tlvq4x8_centered,
    TurboResidualCoder::<4>,
    Centering::Centered,
    PrimaryVectorHeader {
        l2_norm: 1.5514041,
        residual_error_term: 0.09596872,
        center_dot: 0.0,
        lower: -0.69091797,
        upper: 0.56347656,
        component_sum: 152,
    },
    ResidualVectorHeader {
        magnitude: 0.0836263,
        component_sum: 2424,
    },
    [
        -0.92106885,
        -0.06089377,
        0.6588996,
        0.6698713,
        0.5728388,
        0.43090338,
        0.6459602,
        0.0009968281,
        -0.19984382,
        -0.42815655,
        0.72995883,
        -0.70404863,
        -0.27316034,
        0.53911865,
        -0.7310173,
        0.43591502,
        0.91290236,
        0.6941273,
        0.20194513
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
            residual_error_term: 0.0076497365,
            center_dot: 0.0,
            component_sum: 2875,
        }
    );
    assert_abs_diff_eq!(
        ResidualVectorHeader::deserialize(&vector_bytes).unwrap().0,
        ResidualVectorHeader {
            magnitude: 0.0071825213,
            component_sum: 2590,
        }
    );
    let mut decoded = vec![0.0f32; TEST_VECTOR.len()];
    coder.decode_to(&encoded, &mut decoded);
    assert_abs_diff_eq!(
        decoded.as_ref(),
        [
            -0.9210063f32,
            -0.060990915,
            0.65900755,
            0.66999257,
            0.5729863,
            0.4309977,
            0.64599454,
            0.0010041036,
            -0.19999383,
            -0.42800367,
            0.7299878,
            -0.70400965,
            -0.27300203,
            0.538989,
            -0.73099345,
            0.4360114,
            0.9129871,
            0.69399065,
            0.20200203
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
        residual_error_term: 0.0055349832,
        center_dot: 0.0,
        lower: -0.6953125,
        upper: 0.57470703,
        component_sum: 2569,
    },
    ResidualVectorHeader {
        magnitude: 0.0049804687,
        component_sum: 2479,
    },
    [
        -0.9210059,
        -0.06099806,
        0.658998,
        0.66999805,
        0.57299805,
        0.43099415,
        0.64600587,
        0.000990212,
        -0.20000198,
        -0.42800197,
        0.7300019,
        -0.70399415,
        -0.27299806,
        0.5389902,
        -0.73099023,
        0.43600973,
        0.91299415,
        0.69399804,
        0.20200196
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

    let f32_dist = sim.distance_f32().distance_f32(&a, &b);

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

macro_rules! lvq_coding_simd_test {
    ($name:ident, $coder:ty) => {
        #[test]
        fn $name() {
            let seed = OsRng::default().try_next_u64().unwrap();
            println!("SEED {seed:#016x}");
            let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed);
            let scoder = <$coder>::scalar(VectorSimilarity::Euclidean, None);
            let ocoder = <$coder>::new(VectorSimilarity::Euclidean, None);
            // TODO: use randomly sized vectors like we do for distance tests.
            for i in 0..1024 {
                let vec = l2_normalize(
                    (0..128)
                        .map(|_| rng.random_range(-1.0f32..=1.0))
                        .collect::<Vec<_>>(),
                );
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
                    smse.abs_diff_eq(&omse, 1e-6),
                    "index {i} scalar mse {smse:.9} vs optimized mse {omse:.9} input vector {vec:?}"
                );
            }
        }
    };
}

lvq_coding_simd_test!(tlvq1_coding_simd, TurboPrimaryCoder::<1>);
lvq_coding_simd_test!(tlvq2_coding_simd, TurboPrimaryCoder::<2>);
lvq_coding_simd_test!(tlvq4_coding_simd, TurboPrimaryCoder::<4>);
lvq_coding_simd_test!(tlvq8_coding_simd, TurboPrimaryCoder::<8>);
lvq_coding_simd_test!(tlvq1x8_coding_simd, TurboResidualCoder::<1>);
lvq_coding_simd_test!(tlvq2x8_coding_simd, TurboResidualCoder::<2>);
lvq_coding_simd_test!(tlvq4x8_coding_simd, TurboResidualCoder::<4>);
lvq_coding_simd_test!(tlvq8x8_coding_simd, TurboResidualCoder::<8>);
