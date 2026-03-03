use approx::{AbsDiffEq, abs_diff_eq, assert_abs_diff_eq};

use crate::lvq::{
    PrimaryVectorHeader, ResidualVectorHeader, TurboPrimaryCoder, TurboResidualCoder, VectorStats,
};
use crate::{F32VectorCoder, F32VectorCoding, VectorSimilarity};

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
        l2_norm: 2.5226507,
        residual_error_term: 1.163901,
        center_dot: 0.0,
        lower: -0.49564388,
        upper: 0.70561373,
        component_sum: 11,
    },
    [
        -0.49564388,
        -0.49564388,
        0.70561373,
        0.70561373,
        0.70561373,
        0.70561373,
        0.70561373,
        -0.49564388,
        -0.49564388,
        -0.49564388,
        0.70561373,
        -0.49564388,
        -0.49564388,
        0.70561373,
        -0.49564388,
        0.70561373,
        0.70561373,
        0.70561373,
        0.70561373
    ]
);

tlvq_coder_test!(
    tlvq1_centered,
    TurboPrimaryCoder::<1>,
    Centering::Centered,
    PrimaryVectorHeader {
        l2_norm: 1.5514041,
        residual_error_term: 0.8178981,
        center_dot: 0.0,
        lower: -0.60490876,
        upper: 0.23906991,
        component_sum: 13,
    },
    [
        -0.74093014,
        0.21106988,
        0.6950699,
        0.8260699,
        0.37009126,
        0.23209125,
        0.56406987,
        0.031091213,
        -0.20893013,
        -0.65090877,
        0.9320699,
        -0.4009301,
        -0.26093012,
        0.20306988,
        -0.6409088,
        0.61506987,
        0.8680699,
        0.4600699,
        -0.03490877
    ]
);

tlvq_coder_test!(
    tlvq2_uncentered,
    TurboPrimaryCoder::<2>,
    Centering::Uncentered,
    PrimaryVectorHeader {
        l2_norm: 2.5226507,
        residual_error_term: 0.6714753,
        center_dot: 0.0,
        lower: -0.6709247,
        upper: 0.8410188,
        component_sum: 32,
    },
    [
        -0.6709247,
        -0.16694355,
        0.8410188,
        0.8410188,
        0.33703762,
        0.33703762,
        0.8410188,
        -0.16694355,
        -0.16694355,
        -0.6709247,
        0.8410188,
        -0.6709247,
        -0.16694355,
        0.33703762,
        -0.6709247,
        0.33703762,
        0.8410188,
        0.8410188,
        0.33703762
    ]
);

tlvq_coder_test!(
    tlvq2_centered,
    TurboPrimaryCoder::<2>,
    Centering::Centered,
    PrimaryVectorHeader {
        l2_norm: 1.5514041,
        residual_error_term: 0.5325365,
        center_dot: 0.0,
        lower: -0.5572752,
        upper: 0.5681409,
        component_sum: 27,
    },
    [
        -0.78699785,
        -0.2101365,
        0.6490022,
        0.7800022,
        0.41772485,
        0.27972484,
        0.51800215,
        0.0787248,
        -0.25499785,
        -0.6032752,
        0.8860022,
        -0.8221365,
        -0.30699784,
        0.53214085,
        -0.5932752,
        0.56900215,
        0.8220022,
        0.7891409,
        0.3878635
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
        lower: -0.93474734,
        upper: 0.9131211,
        component_sum: 170,
    },
    [
        -0.93474734,
        -0.07240873,
        0.6667386,
        0.6667386,
        0.5435474,
        0.42035618,
        0.6667386,
        0.0507825,
        -0.19559996,
        -0.44198242,
        0.78992987,
        -0.68836486,
        -0.31879118,
        0.5435474,
        -0.68836486,
        0.42035618,
        0.9131211,
        0.6667386,
        0.17397374
    ]
);

tlvq_coder_test!(
    tlvq4_centered,
    TurboPrimaryCoder::<4>,
    Centering::Centered,
    PrimaryVectorHeader {
        l2_norm: 1.5514041,
        residual_error_term: 0.09595752,
        center_dot: 0.0,
        lower: -0.69084126,
        upper: 0.5633911,
        component_sum: 152,
    },
    [
        -0.9183018,
        -0.049917284,
        0.6849292,
        0.6486982,
        0.5350053,
        0.39700526,
        0.6375447,
        0.028774202,
        -0.2190708,
        -0.40237927,
        0.7546982,
        -0.74553275,
        -0.27107078,
        0.5273912,
        -0.7268413,
        0.4376982,
        0.9415447,
        0.7007757,
        0.21362072
    ]
);

tlvq_coder_test!(
    tlvq8_uncentered,
    TurboPrimaryCoder::<8>,
    Centering::Uncentered,
    PrimaryVectorHeader {
        l2_norm: 2.5226507,
        residual_error_term: 0.007704542,
        center_dot: 0.0,
        lower: -0.92000645,
        upper: 0.91146713,
        component_sum: 2876,
    },
    [
        -0.92000645,
        -0.058136534,
        0.6600884,
        0.66727066,
        0.5739014,
        0.43025643,
        0.6457239,
        -0.0006785393,
        -0.20178153,
        -0.42443126,
        0.7319109,
        -0.704539,
        -0.273604,
        0.53799015,
        -0.73326796,
        0.43743867,
        0.91146713,
        0.6959996,
        0.20042445
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
        lower: -0.69532096,
        upper: 0.5748244,
        component_sum: 2569,
    },
    [
        -0.92319566,
        -0.060852986,
        0.65725225,
        0.66870916,
        0.5735558,
        0.43057486,
        0.64579535,
        0.0004505515,
        -0.20191911,
        -0.42752033,
        0.7298805,
        -0.70273876,
        -0.27384293,
        0.5388244,
        -0.731321,
        0.4377853,
        0.9149286,
        0.69620514,
        0.20342255
    ]
);

tlvq_coder_test!(
    tlvq1x8_uncentered,
    TurboResidualCoder::<1>,
    Centering::Uncentered,
    PrimaryVectorHeader {
        l2_norm: 2.5226507,
        residual_error_term: 1.163901,
        center_dot: 0.0,
        lower: -0.49564388,
        upper: 0.70561373,
        component_sum: 11,
    },
    ResidualVectorHeader {
        magnitude: 1.2012575,
        component_sum: 2292,
    },
    [
        -0.9219725,
        -0.05989358,
        0.660861,
        0.6702826,
        0.5713555,
        0.4300311,
        0.6467286,
        0.0013469756,
        -0.20121804,
        -0.42733708,
        0.7315232,
        -0.7052751,
        -0.27188024,
        0.5383798,
        -0.72882915,
        0.4347419,
        0.91524494,
        0.6938367,
        0.20391202
    ]
);

tlvq_coder_test!(
    tlvq1x8_centered,
    TurboResidualCoder::<1>,
    Centering::Centered,
    PrimaryVectorHeader {
        l2_norm: 1.5514041,
        residual_error_term: 0.8178981,
        center_dot: 0.0,
        lower: -0.60490876,
        upper: 0.23906991,
        component_sum: 13,
    },
    ResidualVectorHeader {
        magnitude: 0.84397864,
        component_sum: 2453,
    },
    [
        -0.9213099,
        -0.06198204,
        0.66031784,
        0.6688582,
        0.57363904,
        0.43232933,
        0.64515805,
        -0.00035113096,
        -0.20065583,
        -0.42750263,
        0.7285221,
        -0.7037695,
        -0.27251413,
        0.5390065,
        -0.7319261,
        0.43469012,
        0.9127511,
        0.69340515,
        0.20173621
    ]
);

tlvq_coder_test!(
    tlvq2x8_uncentered,
    TurboResidualCoder::<2>,
    Centering::Uncentered,
    PrimaryVectorHeader {
        l2_norm: 2.5226507,
        residual_error_term: 0.6714753,
        center_dot: 0.0,
        lower: -0.6709247,
        upper: 0.8410188,
        component_sum: 32,
    },
    ResidualVectorHeader {
        magnitude: 0.5039812,
        component_sum: 2319,
    },
    [
        -0.9209389,
        -0.06120634,
        0.6582021,
        0.67006046,
        0.57321703,
        0.43091646,
        0.6463437,
        6.195903e-5,
        -0.19955412,
        -0.42881614,
        0.72935236,
        -0.70353526,
        -0.2726808,
        0.53961825,
        -0.7312048,
        0.43684563,
        0.9131573,
        0.6937772,
        0.20165443
    ]
);

tlvq_coder_test!(
    tlvq2x8_centered,
    TurboResidualCoder::<2>,
    Centering::Centered,
    PrimaryVectorHeader {
        l2_norm: 1.5514041,
        residual_error_term: 0.5325365,
        center_dot: 0.0,
        lower: -0.5572752,
        upper: 0.5681409,
        component_sum: 27,
    },
    ResidualVectorHeader {
        magnitude: 0.37513867,
        component_sum: 2453,
    },
    [
        -0.9216064,
        -0.060816605,
        0.6585645,
        0.6704028,
        0.57292926,
        0.4305159,
        0.6467262,
        0.0014903545,
        -0.1998304,
        -0.42747492,
        0.7293266,
        -0.7037104,
        -0.27242625,
        0.53876096,
        -0.730826,
        0.4358647,
        0.9124768,
        0.69425285,
        0.2017653
    ]
);

tlvq_coder_test!(
    tlvq4x8_uncentered,
    TurboResidualCoder::<4>,
    Centering::Uncentered,
    PrimaryVectorHeader {
        l2_norm: 2.5226507,
        residual_error_term: 0.11848368,
        center_dot: 0.0,
        lower: -0.93474734,
        upper: 0.9131211,
        component_sum: 170,
    },
    ResidualVectorHeader {
        magnitude: 0.123191215,
        component_sum: 2407,
    },
    [
        -0.9209789,
        -0.06105582,
        0.65876746,
        0.6698788,
        0.5727751,
        0.43122596,
        0.64620674,
        0.0007813573,
        -0.20018944,
        -0.42821398,
        0.72978354,
        -0.7040657,
        -0.273138,
        0.5389579,
        -0.73111945,
        0.436057,
        0.9128795,
        0.6940339,
        0.20223519
    ]
);

tlvq_coder_test!(
    tlvq4x8_centered,
    TurboResidualCoder::<4>,
    Centering::Centered,
    PrimaryVectorHeader {
        l2_norm: 1.5514041,
        residual_error_term: 0.09595752,
        center_dot: 0.0,
        lower: -0.69084126,
        upper: 0.5633911,
        component_sum: 152,
    },
    ResidualVectorHeader {
        magnitude: 0.0836155,
        component_sum: 2425,
    },
    [
        -0.921089,
        -0.060902063,
        0.65886086,
        0.669848,
        0.5728782,
        0.4309433,
        0.6459062,
        0.0010663271,
        -0.19988842,
        -0.42811972,
        0.7299415,
        -0.7040529,
        -0.27287427,
        0.5390318,
        -0.73094004,
        0.43589473,
        0.9128531,
        0.69405365,
        0.20198014
    ]
);

tlvq_coder_test!(
    tlvq8x8_uncentered,
    TurboResidualCoder::<8>,
    Centering::Uncentered,
    PrimaryVectorHeader {
        l2_norm: 2.5226507,
        residual_error_term: 0.007704542,
        center_dot: 0.0,
        lower: -0.92000645,
        upper: 0.91146713,
        component_sum: 2876,
    },
    ResidualVectorHeader {
        magnitude: 0.0071822493,
        component_sum: 2422,
    },
    [
        -0.9210063,
        -0.06099535,
        0.65900403,
        0.66998863,
        0.572986,
        0.43100283,
        0.64599144,
        0.0009973188,
        -0.199993,
        -0.42799422,
        0.7300097,
        -0.70398974,
        -0.27299845,
        0.53899,
        -0.7310006,
        0.43598813,
        0.9130022,
        0.69401395,
        0.20198764
    ]
);

tlvq_coder_test!(
    tlvq8x8_centered,
    TurboResidualCoder::<8>,
    Centering::Centered,
    PrimaryVectorHeader {
        l2_norm: 1.5514041,
        residual_error_term: 0.00552745,
        center_dot: 0.0,
        lower: -0.69532096,
        upper: 0.5748244,
        component_sum: 2569,
    },
    ResidualVectorHeader {
        magnitude: 0.0049809623,
        component_sum: 2425,
    },
    [
        -0.9209982,
        -0.060999487,
        0.65900046,
        0.6700081,
        0.5729991,
        0.43099484,
        0.6460004,
        0.0010072589,
        -0.19999509,
        -0.4279989,
        0.73000747,
        -0.7039986,
        -0.27299327,
        0.53899044,
        -0.7309987,
        0.435998,
        0.91300464,
        0.6940077,
        0.2020064
    ]
);

// XXX centered and uncentered distance tests.

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
        let coder = coding.coder(VectorSimilarity::Dot, None);
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
        let coder = coding.coder(VectorSimilarity::Dot, None);
        let encoded = coder.encode(&vector);
        let decoded = coder.decode(&encoded);
        assert_abs_diff_eq!(decoded.as_slice(), vector.as_ref());
    }
}
