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

use std::error::Error as _;

use ai::Error;
use ai::errors::{
    HttpErrorContext, ProviderError, ProviderErrorKind, http_status_to_provider_error_kind,
    is_retryable_status,
};
use ai::transport::TransportError;
use serde_json::json;

#[test]
fn status_codes_map_to_the_full_provider_hierarchy() {
    for (status, expected) in [
        (400, ProviderErrorKind::BadRequest),
        (401, ProviderErrorKind::Authentication),
        (403, ProviderErrorKind::PermissionDenied),
        (404, ProviderErrorKind::NotFound),
        (409, ProviderErrorKind::Conflict),
        (413, ProviderErrorKind::RequestTooLarge),
        (422, ProviderErrorKind::UnprocessableEntity),
        (429, ProviderErrorKind::RateLimit),
        (500, ProviderErrorKind::InternalServer),
        (503, ProviderErrorKind::ServiceUnavailable),
        (504, ProviderErrorKind::DeadlineExceeded),
        (529, ProviderErrorKind::Overloaded),
        (418, ProviderErrorKind::Status),
    ] {
        assert_eq!(http_status_to_provider_error_kind(status), expected);
    }
}

#[test]
fn retryability_defaults_from_http_status() {
    for status in [408, 409, 429, 500, 503, 504, 529] {
        assert!(is_retryable_status(Some(status)), "status {status}");
        assert!(
            ProviderError::from_status("request failed", status, None).is_retryable,
            "status {status}"
        );
    }
    for status in [400, 401, 403, 404, 413, 422] {
        assert!(!is_retryable_status(Some(status)), "status {status}");
    }
    assert!(!is_retryable_status(None));
}

#[test]
fn api_error_preserves_normalized_context() {
    let error = ProviderError::from_status("rate limited", 429, Some("openai".into()))
        .with_request_id("req-1")
        .with_http_context(
            HttpErrorContext::new(429)
                .with_request("POST", "https://api.example/v1/responses")
                .with_response_headers(vec![("retry-after".into(), "1".into())]),
        )
        .with_body(json!({"error": {"message": "slow down"}}))
        .with_code("rate_limit")
        .with_param("model")
        .with_error_type("requests")
        .with_retryable(false);

    assert_eq!(error.kind, ProviderErrorKind::RateLimit);
    assert_eq!(error.provider.as_deref(), Some("openai"));
    assert_eq!(error.request_id.as_deref(), Some("req-1"));
    assert_eq!(error.status_code(), Some(429));
    assert_eq!(error.code.as_deref(), Some("rate_limit"));
    assert_eq!(error.param.as_deref(), Some("model"));
    assert_eq!(error.error_type.as_deref(), Some("requests"));
    assert!(!error.is_retryable);

    let debug = format!("{error:?}");
    assert!(!debug.contains("slow down"));
    assert!(!debug.contains("api.example"));
    assert!(!debug.contains("retry-after"));
}

#[test]
fn incomplete_response_is_retryable_without_an_http_status() {
    let error = ProviderError::incomplete_response(
        "provider stream ended before a finish event",
        Some("anthropic".into()),
    );
    assert_eq!(error.kind, ProviderErrorKind::IncompleteResponse);
    assert_eq!(error.provider.as_deref(), Some("anthropic"));
    assert_eq!(error.status_code(), None);
    assert!(error.is_retryable);
}

#[test]
fn model_not_found_keeps_the_model_id() {
    let error = ProviderError::from_status("model not found", 404, Some("openai".into()))
        .with_model_id("gpt-missing");
    assert_eq!(error.kind, ProviderErrorKind::ModelNotFound);
    assert_eq!(error.model_id.as_deref(), Some("gpt-missing"));
}

#[test]
fn top_level_error_variants_preserve_sources() {
    let provider = Error::from(ProviderError::api(
        "provider rejected the request",
        Some("test".into()),
    ));
    assert!(matches!(provider, Error::Provider(_)));
    assert_eq!(provider.to_string(), "provider rejected the request");
    assert_eq!(
        provider.source().unwrap().to_string(),
        "provider rejected the request"
    );

    let transport = Error::from(TransportError::Timeout);
    assert!(matches!(transport, Error::Transport(_)));
    assert!(transport.source().is_some());

    let json = Error::from(serde_json::from_str::<serde_json::Value>("{").unwrap_err());
    assert!(matches!(json, Error::Json(_)));
    assert!(json.source().is_some());
}

#[test]
fn framework_error_messages_include_actionable_context() {
    assert_eq!(
        Error::UnsupportedProvider("missing".into()).to_string(),
        "unsupported provider \"missing\""
    );
    assert_eq!(
        Error::UnsupportedOperation("rerank".into()).to_string(),
        "unsupported operation \"rerank\""
    );
    assert_eq!(
        Error::Configuration("API key is empty".into()).to_string(),
        "invalid configuration: API key is empty"
    );
}
