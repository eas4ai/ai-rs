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
    errors::{HttpErrorContext, ProviderError, ProviderErrorKind},
    transport::{Headers, HttpRequest, TransportError},
};
use serde_json::Value;

pub(super) fn response(provider: &str, message: impl Into<String>) -> Error {
    ProviderError::new(ProviderErrorKind::Response, message)
        .with_provider(provider)
        .into()
}
pub(super) fn transport(provider: &str, error: TransportError) -> Error {
    let kind = match error {
        TransportError::Timeout => ProviderErrorKind::Timeout,
        TransportError::Connection(_) | TransportError::Body(_) => ProviderErrorKind::Connection,
        TransportError::Protocol(_) | TransportError::Limit(_) => ProviderErrorKind::Response,
        TransportError::Request(_) => ProviderErrorKind::Api,
    };
    ProviderError::new(kind, error.to_string())
        .with_provider(provider)
        .into()
}
fn field(body: &Value, key: &str) -> Option<String> {
    let v = body
        .get(key)
        .filter(|v| !v.is_null())
        .or_else(|| body.get("error")?.get(key))?;
    match v {
        Value::Null => None,
        Value::String(s) => Some(s.clone()),
        v => Some(v.to_string()),
    }
}
fn payload_error(
    kind: ProviderErrorKind,
    provider: &str,
    body: Value,
    fallback: String,
) -> ProviderError {
    let mut error = ProviderError::new(
        kind,
        field(&body, "message")
            .filter(|s| !s.is_empty())
            .unwrap_or(fallback),
    )
    .with_provider(provider);
    error.code = field(&body, "code");
    error.param = field(&body, "param");
    error.error_type = field(&body, "type");
    error.body = Some(body);
    error
}
pub(super) fn status(
    provider: &str,
    model_id: Option<&str>,
    request: &HttpRequest,
    status: u16,
    headers: Headers,
    body: Value,
) -> Error {
    let mut error = payload_error(
        ProviderErrorKind::from_status(status),
        provider,
        body,
        format!("Anthropic request failed with HTTP {status}"),
    );
    error.request_id = headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("request-id"))
        .map(|(_, v)| v.clone());
    error = error.with_http_context(
        HttpErrorContext::new(status)
            .with_request(&request.method, &request.url)
            .with_response_headers(headers),
    );
    if status == 404
        && let Some(model) = model_id
    {
        error = error.with_model_id(model);
    }
    error.into()
}
pub(super) fn stream_error(provider: &str, body: Value) -> Error {
    payload_error(
        ProviderErrorKind::Api,
        provider,
        body,
        "Anthropic stream failed".into(),
    )
    .into()
}
