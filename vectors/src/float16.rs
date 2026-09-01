#[cfg(target_arch = "aarch64")]
mod aarch64;
mod scalar;
#[cfg(target_arch = "x86_64")]
mod x86_64;

use std::{borrow::Cow, sync::OnceLock};

use half::f16;

use crate::{F16VectorDistance, F32VectorCoder, QueryVectorDistance, VectorDistance};

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

#[derive(Debug, Copy, Clone, Default)]
pub struct VectorCoder(Kernel);

impl VectorCoder {
    pub fn new() -> Self {
        Self(Kernel::default())
    }

    fn convert_and_encode(&self, vector: &[f32], scale: Option<f32>, out: &mut [u8]) {
        match self.0 {
            Kernel::Scalar => scalar::serialize_f16(vector, scale, out),
            #[cfg(target_arch = "aarch64")]
            Kernel::Neon => unsafe { aarch64::serialize_f16(vector, scale, out) },
            #[cfg(target_arch = "x86_64")]
            Kernel::AvxF16c => unsafe { x86_64::serialize_f16(vector, scale, out) },
        }
    }
}

impl F32VectorCoder for VectorCoder {
    fn encode_to(&self, vector: &[f32], out: &mut [u8]) {
        self.convert_and_encode(vector, None, out);
    }

    fn byte_len(&self, dimensions: usize) -> usize {
        dimensions * 2
    }

    fn decode_to(&self, encoded: &[u8], out: &mut [f32]) {
        match self.0 {
            Kernel::Scalar => scalar::deserialize_f16(encoded, out),
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

static DOT_DIST: OnceLock<DotProductDistance> = OnceLock::new();

#[derive(Debug, Copy, Clone, Default)]
pub struct DotProductDistance(Kernel);

impl DotProductDistance {
    /// Returns a static instance of dot product distance.
    pub fn get() -> &'static DotProductDistance {
        DOT_DIST.get_or_init(DotProductDistance::default)
    }

    fn dot(&self, a: &[u8], b: &[u8]) -> f32 {
        match self.0 {
            Kernel::Scalar => scalar::dot_f16_f16(a, b),
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

impl F16VectorDistance for DotProductDistance {
    fn distance_f16(&self, a: &[f16], b: &[f16]) -> f64 {
        let dot = self.dot(bytemuck::cast_slice(a), bytemuck::cast_slice(b)) as f64;
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
            Kernel::Scalar => scalar::dot_f32_f16(&self.0, v),
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

static L2_DIST: OnceLock<EuclideanDistance> = OnceLock::new();

#[derive(Debug, Copy, Clone, Default)]
pub struct EuclideanDistance(Kernel);

impl EuclideanDistance {
    /// Returns a static instance of euclidean distance.
    pub fn get() -> &'static EuclideanDistance {
        L2_DIST.get_or_init(EuclideanDistance::default)
    }

    fn l2(&self, a: &[u8], b: &[u8]) -> f32 {
        match self.0 {
            Kernel::Scalar => scalar::l2_f16_f16(a, b),
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

impl F16VectorDistance for EuclideanDistance {
    fn distance_f16(&self, a: &[f16], b: &[f16]) -> f64 {
        self.l2(bytemuck::cast_slice(a), bytemuck::cast_slice(b)) as f64
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
            Kernel::Scalar => scalar::l2_f32_f16(&self.0, v),
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
