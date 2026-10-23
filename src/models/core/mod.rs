//! Compatibility re-exports for the Python `models.core` module layout.

pub use super::params::*;
pub use super::{
    Model, StreamHandle, StreamOptions, experimental_generate, generate, get_model, probe, stream,
};

pub mod helpers;

/// Model metadata and resolution.
pub mod model {
    pub use crate::models::{Model, get_model};
}

/// Single-response inference API.
pub mod api {
    pub use crate::models::{
        StreamHandle, StreamOptions, experimental_generate, generate, probe, stream,
    };
}

/// Provider-independent request parameters.
pub mod params {
    pub use crate::models::params::*;
}
