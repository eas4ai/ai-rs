// Copyright 2026 Vercel, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
