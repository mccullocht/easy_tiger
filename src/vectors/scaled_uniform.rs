//! Scaled uniform coding uses the maximum absolute value across all dimensions for a given vector
//! to produce an i8 value in [-127,127]. It stores both a value to invert the scaling as well as
//! the squared l2 norm for computing euclidean distance using dot product.
//!
//! Unlike some other quantization schemes this does not rely on anything computed across a sample
//! of the data set -- no means or centroids, no quantiles. For transformer models which produce
//! relatively well centered vectors this seems to be effective enough that we may discard the
//! original f32 vectors.

use std::borrow::Cow;

use crate::{
    distance::{dot_f32, l2_normalize},
    vectors::{F32VectorCoder, QueryVectorDistance, VectorDistance},
};

fn compute_scale<const M: i16>(vector: &[f32]) -> (f32, f32) {
    if let Some(max) = vector.iter().map(|d| d.abs()).max_by(|a, b| a.total_cmp(b)) {
        (
            (f64::from(M) / max as f64) as f32,
            (max as f64 / f64::from(M)) as f32,
        )
    } else {
        (0.0, 0.0)
    }
}

#[derive(Debug, Copy, Clone)]
pub struct I8VectorCoder;

impl F32VectorCoder for I8VectorCoder {
    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        let l2_norm = crate::distance::dot_f32(vector, vector).sqrt() as f32;
        let (scale, inv_scale) = compute_scale::<{ i8::MAX as i16 }>(vector);
        out[0..4].copy_from_slice(&inv_scale.to_le_bytes());
        out[4..8].copy_from_slice(&l2_norm.to_le_bytes());
        for (d, o) in vector.iter().zip(out[8..].iter_mut()) {
            *o = ((*d * scale).round() as i8).to_le_bytes()[0];
        }
    }

    fn byte_len(&self, dimensions: usize) -> usize {
        dimensions + std::mem::size_of::<f32>() * 2
    }

    fn decode(&self, encoded: &[u8]) -> Option<Vec<f32>> {
        let v = I8Vector::new(encoded);
        // Decode based on the scale. We could also produce l2 normalized output but there's no
        // good way for the caller to express this distinction and this is inverting something that
        // is inherently lossy.
        let scale = v.scale() as f32;
        Some(v.vector().iter().map(|d| *d as f32 * scale).collect())
    }
}

#[allow(dead_code)]
fn dot_unnormalized_i8_f32_scalar(quantized: &[i8], scale: f64, float: &[f32]) -> f64 {
    quantized
        .iter()
        .zip(float.iter())
        .map(|(s, o)| *s as f32 * *o)
        .sum::<f32>() as f64
        * scale
}

#[cfg(target_arch = "aarch64")]
fn dot_unnormalized_i8_f32_aarch64(quantized: &[i8], scale: f64, float: &[f32]) -> f64 {
    let split = quantized.len() & !15;
    let mut sum = unsafe {
        use std::arch::aarch64::{
            vaddvq_f32, vcvtq_f32_s32, vdupq_n_f32, vfmaq_f32, vget_low_s16, vget_low_s8,
            vld1q_f32, vld1q_s8, vmovl_high_s16, vmovl_high_s8, vmovl_s16, vmovl_s8,
        };

        let mut dot = vdupq_n_f32(0.0);
        for i in (0..split).step_by(16) {
            let qv = {
                let qb = vld1q_s8(quantized.as_ptr().add(i));
                let qh = [vmovl_s8(vget_low_s8(qb)), vmovl_high_s8(qb)];
                [
                    vmovl_s16(vget_low_s16(qh[0])),
                    vmovl_high_s16(qh[0]),
                    vmovl_s16(vget_low_s16(qh[1])),
                    vmovl_high_s16(qh[1]),
                ]
            };
            #[allow(clippy::needless_range_loop)]
            for j in 0..4 {
                dot = vfmaq_f32(
                    dot,
                    vld1q_f32(float.as_ptr().add(i + j * 4)),
                    vcvtq_f32_s32(qv[j]),
                );
            }
        }
        vaddvq_f32(dot)
    };
    sum += quantized[split..]
        .iter()
        .zip(float[split..].iter())
        .map(|(s, o)| *s as f32 * *o)
        .sum::<f32>();
    sum as f64 * scale
}

// Computes unnormalized dot between `quantized` + `scale` x `float` vectors.
#[cfg(target_arch = "aarch64")]
pub(super) fn dot_unnormalized_i8_f32(quantized: &[i8], scale: f64, float: &[f32]) -> f64 {
    dot_unnormalized_i8_f32_aarch64(quantized, scale, float)
}

// Computes unnormalized dot between `quantized` + `scale` x `float` vectors.
#[cfg(not(target_arch = "aarch64"))]
pub(super) fn dot_unnormalized_i8_f32(quantized: &[i8], scale: f64, float: &[f32]) -> f64 {
    dot_unnormalized_i8_f32_scalar(quantized, scale, float)
}

#[derive(Debug, Copy, Clone)]
struct I8Vector<'a>(&'a [u8]);

impl<'a> I8Vector<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        assert!(bytes.len() >= 8);
        Self(bytes)
    }

    fn dot_unnormalized(&self, other: &Self) -> f64 {
        self.vector()
            .iter()
            .zip(other.vector().iter())
            .map(|(s, o)| *s as i32 * *o as i32)
            .sum::<i32>() as f64
            * self.scale()
            * other.scale()
    }

    #[allow(dead_code)]
    fn dot_unnormalized_f32_scalar(&self, other: &[f32]) -> f64 {
        self.vector()
            .iter()
            .zip(other.iter())
            .map(|(s, o)| *s as f32 * *o)
            .sum::<f32>() as f64
            * self.scale()
    }

    fn dot_unnormalized_f32(&self, other: &[f32]) -> f64 {
        dot_unnormalized_i8_f32(self.vector(), self.scale(), other)
    }

    fn scale(&self) -> f64 {
        f32::from_le_bytes(self.0[0..4].try_into().unwrap()).into()
    }

    fn l2_norm_sq(&self) -> f64 {
        self.l2_norm() * self.l2_norm()
    }

    fn l2_norm(&self) -> f64 {
        f32::from_le_bytes(self.0[4..8].try_into().unwrap()).into()
    }

    fn vector(&self) -> &[i8] {
        bytemuck::cast_slice(&self.0[8..])
    }
}

#[derive(Debug, Copy, Clone)]
pub struct I8DotProductDistance;

impl VectorDistance for I8DotProductDistance {
    fn distance(&self, query: &[u8], doc: &[u8]) -> f64 {
        let query = I8Vector::new(query);
        let doc = I8Vector::new(doc);
        let dot = query.dot_unnormalized(&doc) * query.l2_norm().recip() * doc.l2_norm().recip();
        (-dot + 1.0) / 2.0
    }
}

#[derive(Debug, Clone)]
pub struct I8DotProductQueryDistance<'a>(Cow<'a, [f32]>);

impl<'a> I8DotProductQueryDistance<'a> {
    pub fn new(query: Cow<'a, [f32]>) -> Self {
        Self(l2_normalize(query))
    }
}

impl QueryVectorDistance for I8DotProductQueryDistance<'_> {
    fn distance(&self, vector: &[u8]) -> f64 {
        let vector = I8Vector::new(vector);
        let dot = vector.dot_unnormalized_f32(&self.0) / vector.l2_norm();
        (-dot + 1.0) / 2.0
    }
}

#[derive(Debug, Copy, Clone)]
pub struct I8EuclideanDistance;

impl VectorDistance for I8EuclideanDistance {
    fn distance(&self, query: &[u8], doc: &[u8]) -> f64 {
        let query = I8Vector::new(query);
        let doc = I8Vector::new(doc);
        let dot = query.dot_unnormalized(&doc);
        query.l2_norm_sq() + doc.l2_norm_sq() - (2.0 * dot)
    }
}

#[derive(Debug, Clone)]
pub struct I8EuclideanQueryDistance<'a>(Cow<'a, [f32]>, f64);

impl<'a> I8EuclideanQueryDistance<'a> {
    pub fn new(query: Cow<'a, [f32]>) -> Self {
        let l2_norm_sq = dot_f32(&query, &query);
        Self(query, l2_norm_sq)
    }
}

impl QueryVectorDistance for I8EuclideanQueryDistance<'_> {
    fn distance(&self, vector: &[u8]) -> f64 {
        let vector = I8Vector::new(vector);
        let dot = vector.dot_unnormalized_f32(&self.0);
        self.1 + vector.l2_norm_sq() - (2.0 * dot)
    }
}

#[derive(Debug, Copy, Clone)]
pub struct I4PackedVectorCoder;

impl F32VectorCoder for I4PackedVectorCoder {
    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        let l2_norm = crate::distance::dot_f32(vector, vector).sqrt() as f32;
        let (scale, inv_scale) = compute_scale::<7>(vector);
        out[0..4].copy_from_slice(&inv_scale.to_le_bytes());
        out[4..8].copy_from_slice(&l2_norm.to_le_bytes());
        // Encode two at a time and pack them together in a single byte. Use offset binary coding
        // to avoid problems with sign extension happening or not happening when intended.
        for (c, o) in vector.chunks(2).zip(out[8..].iter_mut()) {
            let lo = (c[0] * scale).round() as i8;
            let hi = (c.get(1).unwrap_or(&0.0) * scale).round() as i8;
            *o = ((lo + 7) as u8) | (((hi + 7) as u8) << 4)
        }
    }

    fn byte_len(&self, dimensions: usize) -> usize {
        dimensions.div_ceil(2) + std::mem::size_of::<f32>() * 2
    }

    fn decode(&self, encoded: &[u8]) -> Option<Vec<f32>> {
        let v = I4PackedVector::new(encoded).unwrap();
        // Decode based on the scale. We could also produce l2 normalized output but there's no
        // good way for the caller to express this distinction and this is inverting something that
        // is inherently lossy.
        let scale = v.scale() as f32;
        Some(
            v.dimensions()
                .iter()
                .copied()
                .flat_map(I4PackedVector::unpack)
                .map(|d| d as f32 * scale)
                .collect(),
        )
    }
}

struct I4PackedVector<'a>(&'a [u8]);

impl<'a> I4PackedVector<'a> {
    fn new(vector: &'a [u8]) -> Option<Self> {
        if vector.len() <= std::mem::size_of::<f32>() * 2 {
            None
        } else {
            Some(Self(vector))
        }
    }

    fn scale(&self) -> f64 {
        f32::from_le_bytes(self.0[0..4].try_into().expect("4 bytes")).into()
    }

    fn l2_norm(&self) -> f64 {
        f32::from_le_bytes(self.0[4..8].try_into().expect("4 bytes")).into()
    }

    fn l2_norm_sq(&self) -> f64 {
        self.l2_norm() * self.l2_norm()
    }

    fn dimensions(&self) -> &[u8] {
        &self.0[8..]
    }

    fn unpack(x: u8) -> [i8; 2] {
        [((x & 0xf) as i8) - 7, ((x >> 4) as i8) - 7]
    }

    fn dot_unnormalized(&self, other: &Self) -> f64 {
        self.dimensions()
            .iter()
            .zip(other.dimensions().iter())
            .map(|(q, d)| {
                let q = Self::unpack(*q);
                let d = Self::unpack(*d);
                (q[0] * d[0] + q[1] * d[1]) as i32
            })
            .sum::<i32>() as f64
            * self.scale()
            * other.scale()
    }

    #[allow(dead_code)]
    fn dot_unnormalized_f32_scalar(&self, other: &[f32]) -> f64 {
        let dot = self
            .dimensions()
            .iter()
            .copied()
            .flat_map(Self::unpack)
            .zip(other.iter())
            .map(|(s, o)| s as f32 * o)
            .sum::<f32>();
        // NB: other.scale() is implicitly 1.
        dot as f64 * self.scale()
    }

    #[cfg(not(target_arch = "aarch64"))]
    fn dot_unnormalized_f32(&self, other: &[f32]) -> f64 {
        self.dot_unnormalized_f32_scalar(other)
    }

    #[cfg(target_arch = "aarch64")]
    fn dot_unnormalized_f32(&self, other: &[f32]) -> f64 {
        use std::arch::aarch64::{
            vaddvq_f32, vand_s8, vcvtq_f32_s32, vdup_n_s8, vdupq_n_f32, vfmaq_f32, vget_low_s16,
            vld1_u8, vld1q_f32, vmovl_high_s16, vmovl_s16, vmovl_s8, vreinterpret_s8_u8, vshr_n_u8,
            vsub_s8, vzip1_s8, vzip2_s8,
        };

        let split = other.len() & !15;
        let packed_vec = self.dimensions();
        let mut dot = unsafe {
            let mut dotv = vdupq_n_f32(0.0);
            let qmask = vdup_n_s8(0xf);
            let qoff = vdup_n_s8(7);
            for i in (0..split).step_by(16) {
                // Two values are packed per byte with consecutive dimensions packed together.
                // Unpack these nibble dimensions to one per byte, then zip to interleave them back
                // into the right order. Use signed subtract + offset binary to get sign extension.
                let packed = vld1_u8(packed_vec.as_ptr().add(i / 2));
                let evens = vsub_s8(vand_s8(vreinterpret_s8_u8(packed), qmask), qoff);
                let odds = vsub_s8(vreinterpret_s8_u8(vshr_n_u8(packed, 4)), qoff);
                let lo_i16 = vmovl_s8(vzip1_s8(evens, odds));
                let hi_i16 = vmovl_s8(vzip2_s8(evens, odds));

                let unpacked_i32 = [
                    vmovl_s16(vget_low_s16(lo_i16)),
                    vmovl_high_s16(lo_i16),
                    vmovl_s16(vget_low_s16(hi_i16)),
                    vmovl_high_s16(hi_i16),
                ];
                #[allow(clippy::needless_range_loop)]
                for j in 0..4 {
                    dotv = vfmaq_f32(
                        dotv,
                        vld1q_f32(other.as_ptr().add(i + j * 4)),
                        vcvtq_f32_s32(unpacked_i32[j]),
                    );
                }
            }
            vaddvq_f32(dotv)
        };
        dot += packed_vec[(split / 2)..]
            .iter()
            .copied()
            .flat_map(Self::unpack)
            .zip(other[split..].iter())
            .map(|(s, o)| s as f32 * o)
            .sum::<f32>();
        dot as f64 * self.scale()
    }
}

#[derive(Debug, Copy, Clone)]
pub struct I4PackedDotProductDistance;

impl VectorDistance for I4PackedDotProductDistance {
    fn distance(&self, query: &[u8], doc: &[u8]) -> f64 {
        let query = I4PackedVector::new(query).expect("valid format");
        let doc = I4PackedVector::new(doc).expect("valid format");
        let dot = query.dot_unnormalized(&doc) * query.l2_norm().recip() * doc.l2_norm().recip();
        (-dot + 1.0) / 2.0
    }
}

#[derive(Debug, Clone)]
pub struct I4PackedDotProductQueryDistance<'a>(Cow<'a, [f32]>);

impl<'a> I4PackedDotProductQueryDistance<'a> {
    pub fn new(query: Cow<'a, [f32]>) -> Self {
        Self(l2_normalize(query))
    }
}

impl QueryVectorDistance for I4PackedDotProductQueryDistance<'_> {
    fn distance(&self, vector: &[u8]) -> f64 {
        let vector = I4PackedVector::new(vector).expect("valid format");
        let dot = vector.dot_unnormalized_f32(self.0.as_ref()) / vector.l2_norm();
        (-dot + 1.0) / 2.0
    }
}

#[derive(Debug, Copy, Clone)]
pub struct I4PackedEuclideanDistance;

impl VectorDistance for I4PackedEuclideanDistance {
    fn distance(&self, query: &[u8], doc: &[u8]) -> f64 {
        let query = I4PackedVector::new(query).expect("valid format");
        let doc = I4PackedVector::new(doc).expect("valid format");
        let dot = query.dot_unnormalized(&doc);
        query.l2_norm_sq() + doc.l2_norm_sq() - (2.0 * dot)
    }
}

#[derive(Debug, Clone)]
pub struct I4PackedEuclideanQueryDistance<'a>(Cow<'a, [f32]>, f64);

impl<'a> I4PackedEuclideanQueryDistance<'a> {
    pub fn new(query: Cow<'a, [f32]>) -> Self {
        let l2_norm_sq = dot_f32(&query, &query);
        Self(query, l2_norm_sq)
    }
}

impl QueryVectorDistance for I4PackedEuclideanQueryDistance<'_> {
    fn distance(&self, vector: &[u8]) -> f64 {
        let vector = I4PackedVector::new(vector).expect("valid format");
        let dot = vector.dot_unnormalized_f32(self.0.as_ref());
        self.1 + vector.l2_norm_sq() - (2.0 * dot)
    }
}

#[derive(Debug, Copy, Clone)]
pub struct I16VectorCoder;

impl F32VectorCoder for I16VectorCoder {
    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        let l2_norm = crate::distance::dot_f32(vector, vector).sqrt() as f32;
        let (scale, inv_scale) = compute_scale::<{ i16::MAX }>(vector);
        out[0..4].copy_from_slice(&inv_scale.to_le_bytes());
        out[4..8].copy_from_slice(&l2_norm.to_le_bytes());
        for (d, o) in vector.iter().zip(out[8..].chunks_mut(2)) {
            o.copy_from_slice(&((*d * scale).round() as i16).to_le_bytes());
        }
    }

    fn byte_len(&self, dimensions: usize) -> usize {
        dimensions * 2 + std::mem::size_of::<f32>() * 2
    }

    fn decode(&self, encoded: &[u8]) -> Option<Vec<f32>> {
        let scale = f32::from_le_bytes(encoded[0..4].try_into().unwrap());
        Some(
            encoded[8..]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| i16::from_le_bytes(*c) as f32 * scale)
                .collect(),
        )
    }
}

struct I16Vector<'a>(&'a [u8]);

impl<'a> I16Vector<'a> {
    fn new(rep: &'a [u8]) -> Self {
        Self(rep)
    }

    fn scale(&self) -> f64 {
        f32::from_le_bytes(self.0[0..4].try_into().unwrap()).into()
    }

    fn l2_norm(&self) -> f64 {
        f32::from_le_bytes(self.0[4..8].try_into().unwrap()).into()
    }

    fn l2_norm_sq(&self) -> f64 {
        self.l2_norm() * self.l2_norm()
    }

    fn raw_vector(&self) -> &[u8] {
        &self.0[8..]
    }

    fn dim_iter(&self) -> impl ExactSizeIterator<Item = i16> + '_ {
        self.raw_vector()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|d| i16::from_le_bytes(*d))
    }

    #[cfg(not(target_arch = "aarch64"))]
    fn dot_unnormalized(&self, other: &Self) -> f64 {
        self.dot_unscaled_tail(other, 0) as f64 * self.scale() * other.scale()
    }

    #[cfg(target_arch = "aarch64")]
    fn dot_unnormalized(&self, other: &Self) -> f64 {
        let scalar_split = self.dim_iter().len() & !7;
        let dot = unsafe {
            let mut dot = 0i64;
            let selfp = self.raw_vector().as_ptr() as *const i16;
            let otherp = other.raw_vector().as_ptr() as *const i16;
            for i in (0..scalar_split).step_by(8) {
                use std::arch::aarch64::{
                    vaddlvq_s32, vaddq_s32, vget_low_s16, vld1q_s16, vmull_high_s16, vmull_s16,
                };

                let selfv = vld1q_s16(selfp.add(i));
                let otherv = vld1q_s16(otherp.add(i));
                // multiply and widen, then sum, then sum across. this will produce values _just_
                // small enough to avoid overflowing before we widen to i64.
                let lo = vmull_s16(vget_low_s16(selfv), vget_low_s16(otherv));
                let hi = vmull_high_s16(selfv, otherv);
                dot += vaddlvq_s32(vaddq_s32(lo, hi))
            }
            dot
        };
        (dot + self.dot_unscaled_tail(other, scalar_split)) as f64 * self.scale() * other.scale()
    }

    fn dot_unscaled_tail(&self, other: &Self, start_dim: usize) -> i64 {
        self.dim_iter()
            .skip(start_dim)
            .zip(other.dim_iter().skip(start_dim))
            .map(|(s, o)| s as i64 * o as i64)
            .sum::<i64>()
    }

    #[cfg(not(target_arch = "aarch64"))]
    fn dot_unnormalized_f32(&self, other: &[f32]) -> f64 {
        self.dot_unnormalized_f32_scalar(other, 0)
    }

    #[cfg(target_arch = "aarch64")]
    fn dot_unnormalized_f32(&self, other: &[f32]) -> f64 {
        // Score 8 elements at a time, any tail will be handled with the scalar implementation.
        let scalar_split = other.len() & !7;
        let dot = unsafe {
            use std::arch::aarch64::{
                vaddvq_f32, vcvtq_f32_s32, vdupq_n_f32, vfmaq_f32, vget_high_s16, vget_low_s16,
                vld1q_f32, vld1q_s16, vmovl_s16,
            };

            let mut dot = vdupq_n_f32(0.0);
            let selfp = self.raw_vector().as_ptr() as *const i16;
            for i in (0..scalar_split).step_by(8) {
                let ivec = vld1q_s16(selfp.add(i));
                let svec = [
                    vcvtq_f32_s32(vmovl_s16(vget_low_s16(ivec))),
                    vcvtq_f32_s32(vmovl_s16(vget_high_s16(ivec))),
                ];
                let ovec = [
                    vld1q_f32(other.as_ptr().add(i)),
                    vld1q_f32(other.as_ptr().add(i + 4)),
                ];
                dot = vfmaq_f32(dot, svec[0], ovec[0]);
                dot = vfmaq_f32(dot, svec[1], ovec[1]);
            }
            vaddvq_f32(dot) as f64
        };
        (dot * self.scale()) + self.dot_unnormalized_f32_scalar(other, scalar_split)
    }

    fn dot_unnormalized_f32_scalar(&self, other: &[f32], start_dim: usize) -> f64 {
        self.dim_iter()
            .skip(start_dim)
            .zip(other.iter().skip(start_dim))
            .map(|(s, o)| s as f32 * *o)
            .sum::<f32>() as f64
            * self.scale()
    }
}

#[derive(Debug, Copy, Clone)]
pub struct I16DotProductDistance;

impl VectorDistance for I16DotProductDistance {
    fn distance(&self, query: &[u8], doc: &[u8]) -> f64 {
        let query = I16Vector::new(query);
        let doc = I16Vector::new(doc);
        let dot = query.dot_unnormalized(&doc) / (query.l2_norm() * doc.l2_norm());
        (-dot + 1.0) / 2.0
    }
}

#[derive(Debug, Clone)]
pub struct I16DotProductQueryDistance<'a>(Cow<'a, [f32]>);

impl<'a> I16DotProductQueryDistance<'a> {
    pub fn new(query: Cow<'a, [f32]>) -> Self {
        Self(l2_normalize(query))
    }
}

impl QueryVectorDistance for I16DotProductQueryDistance<'_> {
    fn distance(&self, vector: &[u8]) -> f64 {
        let vector = I16Vector::new(vector);
        let dot = vector.dot_unnormalized_f32(&self.0) / vector.l2_norm();
        (-dot + 1.0) / 2.0
    }
}

#[derive(Debug, Copy, Clone)]
pub struct I16EuclideanDistance;

impl VectorDistance for I16EuclideanDistance {
    fn distance(&self, query: &[u8], doc: &[u8]) -> f64 {
        let query = I16Vector::new(query);
        let doc = I16Vector::new(doc);
        let dot = query.dot_unnormalized(&doc);
        query.l2_norm_sq() + doc.l2_norm_sq() - (2.0 * dot)
    }
}

#[derive(Debug, Clone)]
pub struct I16EuclideanQueryDistance<'a>(Cow<'a, [f32]>, f64);

impl<'a> I16EuclideanQueryDistance<'a> {
    pub fn new(query: Cow<'a, [f32]>) -> Self {
        let l2_norm_sq = dot_f32(&query, &query);
        Self(query, l2_norm_sq)
    }
}

impl QueryVectorDistance for I16EuclideanQueryDistance<'_> {
    fn distance(&self, vector: &[u8]) -> f64 {
        let vector = I16Vector::new(vector);
        let dot = vector.dot_unnormalized_f32(&self.0);
        self.1 + vector.l2_norm_sq() - (2.0 * dot)
    }
}
