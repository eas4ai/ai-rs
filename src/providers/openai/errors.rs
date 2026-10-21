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
use crate::{
    Error,
    errors::{HttpErrorContext, ProviderError, ProviderErrorKind},
    transport::{HttpRequest, HttpResponse, TransportError},
};
use bytes::Bytes;
use futures::StreamExt;
use serde_json::Value;

pub(super) fn response_error(provider: &str, message: impl Into<String>) -> Error {
    ProviderError::new(ProviderErrorKind::Response, message)
        .with_provider(provider)
        .into()
}
pub fn map_transport(error: TransportError, provider: &str) -> Error {
    let kind = match error {
        TransportError::Timeout => ProviderErrorKind::Timeout,
        TransportError::Connection(_) => ProviderErrorKind::Connection,
        TransportError::Request(_) => ProviderErrorKind::Api,
        _ => ProviderErrorKind::Response,
    };
    // Transport messages can contain request URLs or credentials. Keep the category.
    ProviderError::new(
        kind,
        match kind {
            ProviderErrorKind::Timeout => "OpenAI request timed out",
            ProviderErrorKind::Connection => "OpenAI connection failed",
            _ => "OpenAI transport failed",
        },
    )
    .with_provider(provider)
    .with_source(error)
    .into()
}
pub(super) async fn read_body(response: HttpResponse, provider: &str) -> Result<Bytes, Error> {
    let mut bytes = Vec::new();
    let mut body = response.body;
    while let Some(chunk) = body.next().await {
        let chunk = chunk.map_err(|e| map_transport(e, provider))?;
        if bytes.len().saturating_add(chunk.len()) > 8 * 1024 * 1024 {
            return Err(response_error(
                provider,
                "OpenAI response exceeds the 8 MiB limit",
            ));
        }
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes.into())
}
pub(super) async fn check_status(
    response: HttpResponse,
    request: &HttpRequest,
    provider: &str,
    model: Option<&str>,
) -> Result<HttpResponse, Error> {
    if (200..300).contains(&response.status) {
        return Ok(response);
    }
    let status = response.status;
    let headers = response.headers.clone();
    let bytes = read_body(response, provider).await?;
    let body = serde_json::from_slice::<Value>(&bytes)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned()));
    let details = body.get("error").unwrap_or(&body);
    let message = details
        .get("message")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .unwrap_or_else(|| format!("OpenAI request failed with HTTP {status}"));
    let mut error = ProviderError::from_status(message, status, Some(provider.into()));
    // The Python SDK represents all 5xx responses as InternalServerError.
    if status >= 500 {
        error.kind = ProviderErrorKind::InternalServer;
    }
    if status == 404
        && let Some(model) = model
    {
        error = error.with_model_id(model);
    }
    error.request_id = headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("x-request-id"))
        .map(|(_, v)| v.clone());
    error.code = details
        .get("code")
        .and_then(Value::as_str)
        .map(str::to_owned);
    error.param = details
        .get("param")
        .and_then(Value::as_str)
        .map(str::to_owned);
    error.error_type = details
        .get("type")
        .and_then(Value::as_str)
        .map(str::to_owned);
    error.http_context = Some(
        HttpErrorContext::new(status)
            .with_request(&request.method, &request.url)
            .with_response_headers(headers),
    );
    error.body = Some(details.clone());
    Err(error.into())
}
