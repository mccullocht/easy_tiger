#[cfg(target_arch = "aarch64")]
mod aarch64;
#[cfg(target_arch = "x86_64")]
mod x86_64;

use std::borrow::Cow;

use half::f16;

use crate::{F32VectorCoder, QueryVectorDistance, VectorDistance, VectorSimilarity};

#[derive(Debug, Copy, Clone)]
enum Kernel {
    Scalar,
    #[cfg(target_arch = "aarch64")]
    Neon,
    #[cfg(target_arch = "x86_64")]
    AvxF16c,
}

impl Default for Kernel {
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    fn default() -> Self {
        Kernel::Scalar
    }

    #[cfg(target_arch = "aarch64")]
    fn default() -> Self {
        if std::arch::is_aarch64_feature_detected!("fp16") {
            Kernel::Neon
        } else {
            Kernel::Scalar
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn default() -> Self {
        use std::arch::is_x86_feature_detected as feature;
        if feature!("avx") && feature!("f16c") {
            Kernel::AvxF16c
        } else {
            Kernel::Scalar
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct VectorCoder(VectorSimilarity, Kernel);

impl VectorCoder {
    pub fn new(similarity: VectorSimilarity) -> Self {
        Self(similarity, Kernel::default())
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
            Kernel::Scalar => {
                let vector_it = vector.iter().copied();
                if let Some(scale) = scale {
                    self.convert_and_encode_scalar(vector_it.map(|d| d * scale), out)
                } else {
                    self.convert_and_encode_scalar(vector_it, out)
                }
            }
            #[cfg(target_arch = "aarch64")]
            Kernel::Neon => unsafe { aarch64::serialize_f16(vector, scale, out) },
            #[cfg(target_arch = "x86_64")]
            Kernel::AvxF16c => unsafe { x86_64::serialize_f16(vector, scale, out) },
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
            Kernel::Scalar => {
                for (d, o) in f16_iter(encoded).zip(out.iter_mut()) {
                    *o = d.to_f32();
                }
            }
            #[cfg(target_arch = "aarch64")]
            Kernel::Neon => unsafe { aarch64::deserialize_f16(encoded, out) },
            #[cfg(target_arch = "x86_64")]
            Kernel::AvxF16c => unsafe { x86_64::deserialize_f16(encoded, out) },
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
pub struct DotProductDistance(Kernel);

impl DotProductDistance {
    fn dot(&self, a: &[u8], b: &[u8]) -> f32 {
        match self.0 {
            Kernel::Scalar => f16_iter(a)
                .zip(f16_iter(b))
                .map(|(a, b)| a.to_f32() * b.to_f32())
                .sum::<f32>(),
            #[cfg(target_arch = "aarch64")]
            Kernel::Neon => unsafe { aarch64::dot_f16_f16(a, b) },
            #[cfg(target_arch = "x86_64")]
            Kernel::AvxF16c => unsafe { x86_64::dot_f16_f16(a, b) },
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
pub struct DotProductQueryDistance<'a>(Cow<'a, [f32]>, Kernel);

impl<'a> DotProductQueryDistance<'a> {
    pub fn new(query: Cow<'a, [f32]>) -> Self {
        Self(query, Kernel::default())
    }

    fn dot(&self, v: &[u8]) -> f32 {
        match self.1 {
            Kernel::Scalar => self
                .0
                .iter()
                .zip(f16_iter(v).map(f16::to_f32))
                .map(|(s, o)| *s * o)
                .sum::<f32>(),
            #[cfg(target_arch = "aarch64")]
            Kernel::Neon => unsafe { aarch64::dot_f32_f16(&self.0, v) },
            #[cfg(target_arch = "x86_64")]
            Kernel::AvxF16c => unsafe { x86_64::dot_f32_f16(&self.0, v) },
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
pub struct EuclideanDistance(Kernel);

impl EuclideanDistance {
    fn l2(&self, a: &[u8], b: &[u8]) -> f32 {
        match self.0 {
            Kernel::Scalar => f16_iter(a)
                .zip(f16_iter(b))
                .map(|(a, b)| {
                    let diff = a.to_f32() - b.to_f32();
                    diff * diff
                })
                .sum::<f32>(),
            #[cfg(target_arch = "aarch64")]
            Kernel::Neon => unsafe { aarch64::l2_f16_f16(a, b) },
            #[cfg(target_arch = "x86_64")]
            Kernel::AvxF16c => unsafe { x86_64::l2_f16_f16(a, b) },
        }
    }
}

impl VectorDistance for EuclideanDistance {
    fn distance(&self, query: &[u8], doc: &[u8]) -> f64 {
        self.l2(query, doc) as f64
    }
}

#[derive(Debug, Clone)]
pub struct EuclideanQueryDistance<'a>(Cow<'a, [f32]>, Kernel);

impl<'a> EuclideanQueryDistance<'a> {
    pub fn new(query: Cow<'a, [f32]>) -> Self {
        Self(query, Kernel::default())
    }

    fn l2(&self, v: &[u8]) -> f32 {
        match self.1 {
            Kernel::Scalar => self
                .0
                .iter()
                .zip(f16_iter(v).map(f16::to_f32))
                .map(|(s, o)| {
                    let diff = *s - o;
                    diff * diff
                })
                .sum::<f32>(),
            #[cfg(target_arch = "aarch64")]
            Kernel::Neon => unsafe { aarch64::l2_f32_f16(&self.0, v) },
            #[cfg(target_arch = "x86_64")]
            Kernel::AvxF16c => unsafe { x86_64::l2_f32_f16(&self.0, v) },
        }
    }
}

impl QueryVectorDistance for EuclideanQueryDistance<'_> {
    fn distance(&self, vector: &[u8]) -> f64 {
        self.l2(vector).into()
    }
}
