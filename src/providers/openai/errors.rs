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

//! OpenAI error mapping into the shared provider hierarchy.
use crate::{Error, providers::http, transport::TransportError};

/// How the OpenAI API authenticates and reports errors.
pub(crate) const API: http::ApiProfile = http::ApiProfile {
    label: "OpenAI",
    auth: http::Auth::Bearer,
    default_headers: &[],
    request_id_header: "x-request-id",
    collapse_server_errors: true,
    keep_error_object: true,
};

/// Map a transport failure to a provider error that keeps it as the source.
pub fn map_transport(error: TransportError, provider: &str) -> Error {
    http::transport_error(&API, provider, error)
}
