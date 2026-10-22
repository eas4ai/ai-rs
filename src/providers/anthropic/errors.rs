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

use crate::{
    Error,
    errors::{ProviderError, ProviderErrorKind},
    providers::http,
};
use serde_json::Value;

/// How the Anthropic API authenticates and reports errors.
pub(crate) const API: http::ApiProfile = http::ApiProfile {
    label: "Anthropic",
    auth: http::Auth::Header("x-api-key"),
    default_headers: &[("anthropic-version", "2023-06-01")],
    request_id_header: "request-id",
    collapse_server_errors: false,
    keep_error_object: false,
    top_level_first: true,
    code_prefixes_error: false,
};

/// An `error` event inside a Messages stream.
pub(super) fn stream_error(api: &http::ApiProfile, provider: &str, body: Value) -> Error {
    let mut error = ProviderError::new(
        ProviderErrorKind::Api,
        http::error_message(api, &body)
            .map(str::to_owned)
            .unwrap_or_else(|| format!("{} stream failed", api.label)),
    )
    .with_provider(provider);
    error.code = http::error_field(api, &body, "code");
    error.param = http::error_field(api, &body, "param");
    error.error_type = http::error_field(api, &body, "type");
    error.body = Some(body);
    error.into()
}
