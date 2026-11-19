//! Support for i8 vectors.
//!
//! This is limited to ingestion and vector comparison, certain other operations do not translate
//! very well.

#![allow(dead_code)] // XXX DO NOT MERGE

#[cfg(target_arch = "aarch64")]
mod aarch64;
mod scalar;
#[cfg(target_arch = "x86_64")]
mod x86_64;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum Kernel {
    Scalar,
    #[cfg(target_arch = "aarch64")]
    Neon,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    Avx512,
}

impl Kernel {
    const CANDIDATES: &'static [Self] = &[
        #[cfg(target_arch = "aarch64")]
        Self::Neon,
        #[cfg(target_arch = "x86_64")]
        Self::Avx512,
        #[cfg(target_arch = "x86_64")]
        Self::Avx2,
        Self::Scalar,
    ];

    /// Returns true if the specificed kernel is available for use on this cost.
    fn is_available(&self) -> bool {
        match self {
            #[cfg(target_arch = "aarch64")]
            Self::Neon => std::arch::is_aarch64_feature_detected!("dotprod"),
            #[cfg(target_arch = "x86_64")]
            Self::Avx512 => std::arch::is_x86_feature_detected!("avx512f"),
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 => std::arch::is_x86_feature_detected!("avx2"),
            Self::Scalar => true,
        }
    }

    /// Return all of the non-Scalar Kernels that are available on this host.
    #[cfg(test)]
    fn accelerated() -> impl Iterator<Item = Self> {
        Self::CANDIDATES
            .iter()
            .copied()
            .filter(|&k| k.is_available() && k != Self::Scalar)
    }
}

impl Default for Kernel {
    fn default() -> Self {
        Self::CANDIDATES
            .iter()
            .copied()
            .find(Self::is_available)
            .expect("Scalar is always available")
    }
}

/// Decode an input `i8` vector to an output `f32` vector, equivalent to `as` casting each element
/// but faster.
pub fn decode_to_f32(vector: &[i8], out: &mut [f32]) {
    assert_eq!(vector.len(), out.len());
    match Kernel::default() {
        Kernel::Scalar => scalar::decode_to_f32(vector, out),
        // NEON is on by default on aarch64, and it does a fine job auto-vectorizing.
        #[cfg(target_arch = "aarch64")]
        Kernel::Neon => scalar::decode_to_f32(vector, out),
        #[cfg(target_arch = "x86_64")]
        Kernel::Avx2 => unsafe { x86_64::avx2::decode_to_f32(vector, out) },
        #[cfg(target_arch = "x86_64")]
        Kernel::Avx512 => unsafe { x86_64::avx512::decode_to_f32(vector, out) },
    }
}
