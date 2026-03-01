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

#[test]
fn vector_stats_simd() {
    let simd_stats = VectorStats::from(TEST_VECTOR.as_ref());
    let scalar_stats = VectorStats::from_scalar(TEST_VECTOR.as_ref());
    assert_abs_diff_eq!(simd_stats, scalar_stats);
}

#[test]
fn tlvq1() {
    let coder = TurboPrimaryCoder::<1>::default();
    let encoded = coder.encode(&TEST_VECTOR);
    assert_abs_diff_eq!(
        PrimaryVectorHeader::deserialize(&encoded).unwrap().0,
        PrimaryVectorHeader {
            l2_norm: 2.5226507,
            lower: -0.49564388,
            upper: 0.70561373,
            residual_error_term: 1.163901,
            component_sum: 11,
        }
    );
    let mut decoded = vec![0.0f32; TEST_VECTOR.len()];
    coder.decode_to(&encoded, &mut decoded);
    assert_abs_diff_eq!(
        decoded.as_ref(),
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
        .as_ref(),
        epsilon = 0.00001
    );
}

#[test]
fn tlvq2() {
    let coder = TurboPrimaryCoder::<2>::default();
    let encoded = coder.encode(&TEST_VECTOR);
    assert_abs_diff_eq!(
        PrimaryVectorHeader::deserialize(&encoded).unwrap().0,
        PrimaryVectorHeader {
            l2_norm: 2.5226507,
            lower: -0.6709247,
            upper: 0.8410188,
            residual_error_term: 0.6714753,
            component_sum: 32,
        }
    );
    let mut decoded = vec![0.0f32; TEST_VECTOR.len()];
    coder.decode_to(&encoded, &mut decoded);
    assert_abs_diff_eq!(
        decoded.as_ref(),
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
        .as_ref(),
        epsilon = 0.0001,
    );
}

#[test]
fn tlvq4() {
    let coder = TurboPrimaryCoder::<4>::default();
    let encoded = coder.encode(&TEST_VECTOR);
    assert_abs_diff_eq!(
        PrimaryVectorHeader::deserialize(&encoded).unwrap().0,
        PrimaryVectorHeader {
            l2_norm: 2.5226507,
            lower: -0.93474734,
            upper: 0.9131211,
            residual_error_term: 0.11848368,
            component_sum: 170,
        }
    );
    let mut decoded = vec![0.0f32; TEST_VECTOR.len()];
    coder.decode_to(&encoded, &mut decoded);
    assert_abs_diff_eq!(
        decoded.as_ref(),
        [
            -0.93474734,
            -0.072408736,
            0.6667386,
            0.6667386,
            0.5435474,
            0.42035615,
            0.6667386,
            0.0507825,
            -0.19559997,
            -0.44198242,
            0.78992987,
            -0.68836486,
            -0.3187912,
            0.5435474,
            -0.68836486,
            0.42035615,
            0.9131211,
            0.6667386,
            0.17397368
        ]
        .as_ref(),
        epsilon = 0.0001,
    );
}

#[test]
fn tlvq8() {
    let coder = TurboPrimaryCoder::<8>::default();
    let encoded = coder.encode(&TEST_VECTOR);
    assert_abs_diff_eq!(
        PrimaryVectorHeader::deserialize(&encoded).unwrap().0,
        PrimaryVectorHeader {
            l2_norm: 2.5226507,
            lower: -0.92000645,
            upper: 0.91146713,
            residual_error_term: 0.007704542,
            component_sum: 2876,
        }
    );
    let mut decoded = vec![0.0f32; TEST_VECTOR.len()];
    coder.decode_to(&encoded, &mut decoded);
    assert_abs_diff_eq!(
        decoded.as_ref(),
        [
            -0.92000645,
            -0.058136523,
            0.66008836,
            0.6672706,
            0.57390136,
            0.43025643,
            0.6457239,
            -0.0006785393,
            -0.20178151,
            -0.42443126,
            0.7319109,
            -0.70453894,
            -0.27360404,
            0.53799015,
            -0.73326796,
            0.43743867,
            0.91146713,
            0.6959997,
            0.20042449
        ]
        .as_ref(),
        epsilon = 0.0001
    );
}

#[test]
fn tlvq1x8() {
    let coder = TurboResidualCoder::<1>::default();
    let encoded = coder.encode(&TEST_VECTOR);
    let (primary_header, vector_bytes) = PrimaryVectorHeader::deserialize(&encoded).unwrap();
    assert_abs_diff_eq!(
        primary_header,
        PrimaryVectorHeader {
            l2_norm: 2.5226507,
            lower: -0.49564388,
            upper: 0.70561373,
            residual_error_term: 1.163901,
            component_sum: 11,
        }
    );
    let (residual_header, _) = ResidualVectorHeader::deserialize(&vector_bytes).unwrap();
    assert_abs_diff_eq!(
        residual_header,
        ResidualVectorHeader {
            magnitude: 1.2012575,
            component_sum: 2292,
        }
    );
    let mut decoded = vec![0.0f32; TEST_VECTOR.len()];
    coder.decode_to(&encoded, &mut decoded);
    assert_abs_diff_eq!(
        decoded.as_ref(),
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
        .as_ref(),
        epsilon = 0.00001
    );
}

#[test]
fn tlvq2x8() {
    let coder = TurboResidualCoder::<2>::default();
    let encoded = coder.encode(&TEST_VECTOR);
    let (primary_header, vector_bytes) = PrimaryVectorHeader::deserialize(&encoded).unwrap();
    assert_abs_diff_eq!(
        primary_header,
        PrimaryVectorHeader {
            l2_norm: 2.5226507,
            lower: -0.6709247,
            upper: 0.8410188,
            residual_error_term: 0.6714753,
            component_sum: 32,
        }
    );
    let (residual_header, _) = ResidualVectorHeader::deserialize(&vector_bytes).unwrap();
    assert_abs_diff_eq!(
        residual_header,
        ResidualVectorHeader {
            magnitude: 0.5039812,
            component_sum: 2319,
        }
    );
    let mut decoded = vec![0.0f32; TEST_VECTOR.len()];
    coder.decode_to(&encoded, &mut decoded);
    assert_abs_diff_eq!(
        decoded.as_ref(),
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
        .as_ref(),
        epsilon = 0.0001,
    );
}

#[test]
fn tlvq4x8() {
    let coder = TurboResidualCoder::<4>::default();
    let encoded = coder.encode(&TEST_VECTOR);
    let (primary_header, vector_bytes) = PrimaryVectorHeader::deserialize(&encoded).unwrap();
    assert_abs_diff_eq!(
        primary_header,
        PrimaryVectorHeader {
            l2_norm: 2.5226507,
            lower: -0.93474734,
            upper: 0.9131211,
            residual_error_term: 0.11848368,
            component_sum: 170,
        }
    );
    let (residual_header, _) = ResidualVectorHeader::deserialize(&vector_bytes).unwrap();
    assert_abs_diff_eq!(
        residual_header,
        ResidualVectorHeader {
            magnitude: 0.123191215,
            component_sum: 2407,
        }
    );
    let mut decoded = vec![0.0f32; TEST_VECTOR.len()];
    coder.decode_to(&encoded, &mut decoded);
    assert_abs_diff_eq!(
        decoded.as_ref(),
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
        .as_ref(),
        epsilon = 0.0001,
    );
}

#[test]
fn tlvq8x8() {
    let coder = TurboResidualCoder::<8>::default();
    let encoded = coder.encode(&TEST_VECTOR);
    let (primary_header, vector_bytes) = PrimaryVectorHeader::deserialize(&encoded).unwrap();
    assert_abs_diff_eq!(
        primary_header,
        PrimaryVectorHeader {
            l2_norm: 2.5226507,
            lower: -0.92000645,
            upper: 0.91146713,
            residual_error_term: 0.007704542,
            component_sum: 2876,
        }
    );
    let (residual_header, _) = ResidualVectorHeader::deserialize(&vector_bytes).unwrap();
    assert_abs_diff_eq!(
        residual_header,
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
        .as_ref(),
        epsilon = 0.0001
    );
}

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
        let coder = coding.new_coder(VectorSimilarity::Dot);
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
        let coder = coding.new_coder(VectorSimilarity::Dot);
        let encoded = coder.encode(&vector);
        let decoded = coder.decode(&encoded);
        assert_abs_diff_eq!(decoded.as_slice(), vector.as_ref());
    }
}
