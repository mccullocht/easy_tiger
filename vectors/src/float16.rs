use std::borrow::Cow;

use half::f16;

use crate::{F32VectorCoder, QueryVectorDistance, VectorDistance, VectorSimilarity};

#[derive(Debug, Copy, Clone)]
enum InstructionSet {
    Scalar,
    #[cfg(target_arch = "aarch64")]
    Neon,
    #[cfg(target_arch = "x86_64")]
    AvxF16c,
}

impl Default for InstructionSet {
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    fn default() -> Self {
        InstructionSet::Scalar
    }

    #[cfg(target_arch = "aarch64")]
    fn default() -> Self {
        if std::arch::is_aarch64_feature_detected!("fp16") {
            InstructionSet::Neon
        } else {
            InstructionSet::Scalar
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn default() -> Self {
        use std::arch::is_x86_feature_detected as feature;
        if feature!("avx") && feature!("f16c") {
            InstructionSet::AvxF16c
        } else {
            InstructionSet::Scalar
        }
    }
}

// While the `half` crate supports f16, SIMD features are limited to nightly and even the related
// intrinsics are not stable on aarch64, so resort to C linkage.
#[allow(dead_code)]
#[cfg(target_arch = "aarch64")]
unsafe extern "C" {
    unsafe fn et_serialize_f16(v: *const f32, len: usize, scale: *const f32, out: *mut u8);
    unsafe fn et_deserialize_f16(v: *const u16, len: usize, out: *mut f32);

    unsafe fn et_dot_f16_f16(a: *const u16, b: *const u16, len: usize) -> f32;
    unsafe fn et_dot_f32_f16(a: *const f32, b: *const u16, len: usize) -> f32;

    unsafe fn et_l2_f16_f16(a: *const u16, b: *const u16, len: usize) -> f32;
    unsafe fn et_l2_f32_f16(a: *const f32, b: *const u16, len: usize) -> f32;
}

#[allow(dead_code)]
#[cfg(target_arch = "x86_64")]
unsafe extern "C" {
    unsafe fn et_serialize_f16_avx512(v: *const f32, len: usize, scale: *const f32, out: *mut u8);
    unsafe fn et_deserialize_f16_avx512(v: *const u16, len: usize, out: *mut f32);

    unsafe fn et_dot_f16_f16_avx512(a: *const u16, b: *const u16, len: usize) -> f32;
    unsafe fn et_dot_f32_f16_avx512(a: *const f32, b: *const u16, len: usize) -> f32;

    unsafe fn et_l2_f16_f16_avx512(a: *const u16, b: *const u16, len: usize) -> f32;
    unsafe fn et_l2_f32_f16_avx512(a: *const f32, b: *const u16, len: usize) -> f32;
}

#[derive(Debug, Copy, Clone)]
pub struct VectorCoder(VectorSimilarity, InstructionSet);

impl VectorCoder {
    pub fn new(similarity: VectorSimilarity) -> Self {
        Self(similarity, InstructionSet::default())
    }

    fn convert_and_encode_scalar(
        &self,
        vector: impl ExactSizeIterator<Item = f32> + Clone,
        out: &mut [u8],
    ) {
        let encode_it = vector.zip(out.chunks_mut(2));
        for (d, o) in encode_it {
            o.copy_from_slice(&f16::from_f32(d).to_le_bytes());
        }
    }

    fn convert_and_encode(&self, vector: &[f32], scale: Option<f32>, out: &mut [u8]) {
        match self.1 {
            InstructionSet::Scalar => {
                let vector_it = vector.iter().copied();
                if let Some(scale) = scale {
                    self.convert_and_encode_scalar(vector_it.map(|d| d * scale), out)
                } else {
                    self.convert_and_encode_scalar(vector_it, out)
                }
            }
            #[cfg(target_arch = "aarch64")]
            InstructionSet::Neon => unsafe {
                et_serialize_f16(
                    vector.as_ptr(),
                    vector.len(),
                    scale
                        .as_ref()
                        .map(std::ptr::from_ref)
                        .unwrap_or(std::ptr::null()),
                    out.as_mut_ptr(),
                )
            },
            #[cfg(target_arch = "x86_64")]
            InstructionSet::AvxF16c => unsafe {
                et_serialize_f16_avx512(
                    vector.as_ptr(),
                    vector.len(),
                    scale
                        .as_ref()
                        .map(std::ptr::from_ref)
                        .unwrap_or(std::ptr::null()),
                    out.as_mut_ptr(),
                )
            },
        }
    }
}

impl F32VectorCoder for VectorCoder {
    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        let scale = if self.0.l2_normalize() {
            Some(1.0 / super::l2_norm(vector))
        } else {
            None
        };
        self.convert_and_encode(vector, scale, out);
    }

    fn byte_len(&self, dimensions: usize) -> usize {
        dimensions * 2
    }

    fn decode_to(&self, encoded: &[u8], out: &mut [f32]) {
        match self.1 {
            InstructionSet::Scalar => {
                for (d, o) in f16_iter(encoded).zip(out.iter_mut()) {
                    *o = d.to_f32();
                }
            }
            #[cfg(target_arch = "aarch64")]
            InstructionSet::Neon => unsafe {
                et_deserialize_f16(
                    encoded.as_ptr() as *const u16,
                    encoded.len() / 2,
                    out.as_mut_ptr(),
                )
            },
            #[cfg(target_arch = "x86_64")]
            InstructionSet::AvxF16c => unsafe {
                et_deserialize_f16_avx512(
                    encoded.as_ptr() as *const u16,
                    encoded.len() / 2,
                    out.as_mut_ptr(),
                )
            },
        }
    }

    fn dimensions(&self, byte_len: usize) -> usize {
        byte_len / std::mem::size_of::<f16>()
    }
}

fn f16_iter(raw: &[u8]) -> impl ExactSizeIterator<Item = f16> + '_ {
    let (chunks, rem) = raw.as_chunks::<{ std::mem::size_of::<f16>() }>();
    debug_assert!(rem.is_empty());
    chunks.iter().map(|c| {
        f16::from_bits(u16::from_le(unsafe {
            std::ptr::read_unaligned(c.as_ptr() as *const u16)
        }))
    })
}

#[derive(Debug, Copy, Clone, Default)]
pub struct DotProductDistance(InstructionSet);

impl DotProductDistance {
    fn dot(&self, a: &[u8], b: &[u8]) -> f32 {
        match self.0 {
            InstructionSet::Scalar => f16_iter(a)
                .zip(f16_iter(b))
                .map(|(a, b)| a.to_f32() * b.to_f32())
                .sum::<f32>(),
            #[cfg(target_arch = "aarch64")]
            InstructionSet::Neon => unsafe {
                et_dot_f16_f16(
                    a.as_ptr() as *const u16,
                    b.as_ptr() as *const u16,
                    a.len() / 2,
                )
            },
            #[cfg(target_arch = "x86_64")]
            InstructionSet::AvxF16c => unsafe {
                et_dot_f16_f16_avx512(
                    a.as_ptr() as *const u16,
                    b.as_ptr() as *const u16,
                    a.len() / 2,
                )
            },
        }
    }
}

impl VectorDistance for DotProductDistance {
    fn distance(&self, query: &[u8], doc: &[u8]) -> f64 {
        let dot = self.dot(query, doc) as f64;
        (-dot + 1.0) / 2.0
    }
}

#[derive(Debug, Clone)]
pub struct DotProductQueryDistance<'a>(Cow<'a, [f32]>, InstructionSet);

impl<'a> DotProductQueryDistance<'a> {
    pub fn new(query: Cow<'a, [f32]>) -> Self {
        Self(query, InstructionSet::default())
    }

    fn dot(&self, v: &[u8]) -> f32 {
        match self.1 {
            InstructionSet::Scalar => self
                .0
                .iter()
                .zip(f16_iter(v).map(f16::to_f32))
                .map(|(s, o)| *s * o)
                .sum::<f32>(),
            #[cfg(target_arch = "aarch64")]
            InstructionSet::Neon => unsafe {
                et_dot_f32_f16(self.0.as_ptr(), v.as_ptr() as *const u16, self.0.len())
            },
            #[cfg(target_arch = "x86_64")]
            InstructionSet::AvxF16c => unsafe {
                et_dot_f32_f16_avx512(self.0.as_ptr(), v.as_ptr() as *const u16, self.0.len())
            },
        }
    }
}

impl QueryVectorDistance for DotProductQueryDistance<'_> {
    fn distance(&self, vector: &[u8]) -> f64 {
        let dot = self.dot(vector) as f64;
        (-dot + 1.0) / 2.0
    }
}

#[derive(Debug, Copy, Clone, Default)]
pub struct EuclideanDistance(InstructionSet);

impl EuclideanDistance {
    fn l2(&self, a: &[u8], b: &[u8]) -> f32 {
        match self.0 {
            InstructionSet::Scalar => f16_iter(a)
                .zip(f16_iter(b))
                .map(|(a, b)| {
                    let diff = a.to_f32() - b.to_f32();
                    diff * diff
                })
                .sum::<f32>(),
            #[cfg(target_arch = "aarch64")]
            InstructionSet::Neon => unsafe {
                et_l2_f16_f16(
                    a.as_ptr() as *const u16,
                    b.as_ptr() as *const u16,
                    a.len() / 2,
                )
            },
            #[cfg(target_arch = "x86_64")]
            InstructionSet::AvxF16c => unsafe {
                et_l2_f16_f16_avx512(
                    a.as_ptr() as *const u16,
                    b.as_ptr() as *const u16,
                    a.len() / 2,
                )
            },
        }
    }
}

impl VectorDistance for EuclideanDistance {
    fn distance(&self, query: &[u8], doc: &[u8]) -> f64 {
        self.l2(query, doc) as f64
    }
}

#[derive(Debug, Clone)]
pub struct EuclideanQueryDistance<'a>(Cow<'a, [f32]>, InstructionSet);

impl<'a> EuclideanQueryDistance<'a> {
    pub fn new(query: Cow<'a, [f32]>) -> Self {
        Self(query, InstructionSet::default())
    }

    fn l2(&self, v: &[u8]) -> f32 {
        match self.1 {
            InstructionSet::Scalar => self
                .0
                .iter()
                .zip(f16_iter(v).map(f16::to_f32))
                .map(|(s, o)| {
                    let diff = *s - o;
                    diff * diff
                })
                .sum::<f32>(),
            #[cfg(target_arch = "aarch64")]
            InstructionSet::Neon => unsafe {
                et_l2_f32_f16(self.0.as_ptr(), v.as_ptr() as *const u16, self.0.len())
            },
            #[cfg(target_arch = "x86_64")]
            InstructionSet::AvxF16c => unsafe {
                et_l2_f32_f16_avx512(self.0.as_ptr(), v.as_ptr() as *const u16, self.0.len())
            },
        }
    }
}

impl QueryVectorDistance for EuclideanQueryDistance<'_> {
    fn distance(&self, vector: &[u8]) -> f64 {
        self.l2(vector).into()
    }
}
