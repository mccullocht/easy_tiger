//! aarch64 kernel implementations.

use super::scalar;

/// NEON-specific kernel implementations.
pub mod neon {
    #[target_feature(enable = "dotprod")]
    pub unsafe fn inner_product(a: &[i8], b: &[i8]) -> i32 {
        // This will auto-vectorize in the compiler, but it may be faster to manually unroll it.
        super::scalar::inner_product(a, b)
    }
}
