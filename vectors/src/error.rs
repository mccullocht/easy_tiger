//! Error type for fallible vector preparation and encoding.

use std::fmt;

/// An error produced while preparing or encoding a vector.
///
/// Every variant is [`Copy`] and carries no heap allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// A vector component was `NaN` or infinite. Carries the index of the first offender.
    NonFiniteComponent {
        /// Index of the first non-finite component.
        index: usize,
    },
    /// The vector's magnitude (L2 norm, or a norm-derived quantization term) is not finite --
    /// typically because the component magnitudes overflow `f32` when squared or summed. Such a
    /// vector cannot be represented by the quantizers.
    NonFiniteMagnitude,
    /// An operation that requires a non-empty vector was given an empty one.
    EmptyVector,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteComponent { index } => {
                write!(f, "non-finite vector component at index {index}")
            }
            Self::NonFiniteMagnitude => {
                write!(f, "vector magnitude is not finite (component values overflow)")
            }
            Self::EmptyVector => write!(f, "vector is empty"),
        }
    }
}

impl std::error::Error for Error {}

impl From<Error> for std::io::Error {
    fn from(err: Error) -> Self {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, err.to_string())
    }
}

/// `Result` specialized to this crate's [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

/// Return [`Error::NonFiniteComponent`] if any component of `vector` is not finite.
pub(crate) fn check_finite_vector(vector: &[f32]) -> Result<()> {
    match vector.iter().position(|v| !v.is_finite()) {
        Some(index) => Err(Error::NonFiniteComponent { index }),
        None => Ok(()),
    }
}
