//! Common kernel functions shared across different vector coders.
//!
//! This is arranged as a series of modules targeting different architectures with submodules for
//! any specific instructions sets. This not arranged as a trait or set of traits as each function
//! may need to be tagged with a target_feature. Each function also has a `scalar` implementation
//! that is not architecture dependent.

#[cfg(target_arch = "aarch64")]
pub mod aarch64;
pub mod scalar;
