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

//! Framework and provider error hierarchy.

use std::fmt;

use serde_json::Value;
use thiserror::Error;

use crate::transport::TransportError;

/// Result type returned by fallible inference APIs.
pub type Result<T> = std::result::Result<T, Error>;

/// Top-level errors produced by the inference core.
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum Error {
    /// Required SDK configuration is missing or invalid.
    #[error("invalid configuration: {0}")]
    Configuration(String),
    /// A required optional dependency or feature is unavailable.
    #[error("required component is not installed: {0}")]
    Installation(String),
    /// The requested provider is not supported.
    #[error("unsupported provider {0:?}")]
    UnsupportedProvider(String),
    /// The provider or protocol does not implement an operation.
    #[error("unsupported operation {0:?}")]
    UnsupportedOperation(String),
    /// A model provider failed.
    #[error("{0}")]
    Provider(#[source] Box<ProviderError>),
    /// Input or output validation failed.
    #[error("validation failed: {0}")]
    Validation(String),
    /// JSON encoding or decoding failed.
    #[error("JSON operation failed: {0}")]
    Json(#[from] serde_json::Error),
    /// The configured HTTP transport failed.
    #[error("transport operation failed: {0}")]
    Transport(#[from] TransportError),
}

impl From<ProviderError> for Error {
    fn from(error: ProviderError) -> Self {
        Self::Provider(Box::new(error))
    }
}

/// Normalized HTTP context captured from a provider API error.
#[derive(Clone, PartialEq, Eq)]
pub struct HttpErrorContext {
    /// HTTP response status code.
    pub status_code: u16,
    /// HTTP method, when retained by the adapter.
    pub request_method: Option<String>,
    /// Request URL, when retained by the adapter.
    pub request_url: Option<String>,
    /// Response headers retained by the adapter.
    pub response_headers: Vec<(String, String)>,
}

impl fmt::Debug for HttpErrorContext {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HttpErrorContext")
            .field("status_code", &self.status_code)
            .field("request_method", &self.request_method)
            .field("has_request_url", &self.request_url.is_some())
            .field("response_header_count", &self.response_headers.len())
            .finish()
    }
}

impl HttpErrorContext {
    /// Creates context containing only the required HTTP status.
    pub const fn new(status_code: u16) -> Self {
        Self {
            status_code,
            request_method: None,
            request_url: None,
            response_headers: Vec::new(),
        }
    }

    /// Adds normalized request metadata.
    pub fn with_request(mut self, method: impl Into<String>, url: impl Into<String>) -> Self {
        self.request_method = Some(method.into());
        self.request_url = Some(url.into());
        self
    }

    /// Adds response headers.
    pub fn with_response_headers(mut self, headers: Vec<(String, String)>) -> Self {
        self.response_headers = headers;
        self
    }
}

/// Stable provider error categories corresponding to the Python hierarchy.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProviderErrorKind {
    /// Base provider failure without a narrower category.
    Provider,
    /// Provider configuration is missing.
    NotConfigured,
    /// Provider request failed before a narrower category was known.
    Api,
    /// The provider could not be reached.
    Connection,
    /// The provider request timed out before an HTTP response.
    Timeout,
    /// The provider response was malformed or unexpected.
    Response,
    /// A stream ended before its finish event.
    IncompleteResponse,
    /// An otherwise uncategorized non-success HTTP status.
    Status,
    /// HTTP 400.
    BadRequest,
    /// HTTP 401.
    Authentication,
    /// HTTP 403.
    PermissionDenied,
    /// HTTP 404.
    NotFound,
    /// A model was not found, normally under HTTP 404.
    ModelNotFound,
    /// HTTP 409.
    Conflict,
    /// HTTP 413.
    RequestTooLarge,
    /// HTTP 422.
    UnprocessableEntity,
    /// HTTP 429.
    RateLimit,
    /// An uncategorized HTTP 5xx response.
    InternalServer,
    /// HTTP 503.
    ServiceUnavailable,
    /// HTTP 504.
    DeadlineExceeded,
    /// HTTP 529.
    Overloaded,
}

impl ProviderErrorKind {
    /// Maps a non-success HTTP status to its normalized provider category.
    pub const fn from_status(status_code: u16) -> Self {
        match status_code {
            400 => Self::BadRequest,
            401 => Self::Authentication,
            403 => Self::PermissionDenied,
            404 => Self::NotFound,
            409 => Self::Conflict,
            413 => Self::RequestTooLarge,
            422 => Self::UnprocessableEntity,
            429 => Self::RateLimit,
            503 => Self::ServiceUnavailable,
            504 => Self::DeadlineExceeded,
            529 => Self::Overloaded,
            500..=599 => Self::InternalServer,
            _ => Self::Status,
        }
    }
}

/// A provider failure with normalized category and upstream context.
#[derive(Clone, Error, PartialEq)]
#[error("{message}")]
pub struct ProviderError {
    /// Human-readable error message.
    pub message: String,
    /// Provider name or id, when known.
    pub provider: Option<String>,
    /// Normalized error category.
    pub kind: ProviderErrorKind,
    /// Provider-assigned request identifier, when available.
    pub request_id: Option<String>,
    /// HTTP request and response metadata, when an HTTP response exists.
    pub http_context: Option<HttpErrorContext>,
    /// Provider response body exposed by the adapter.
    pub body: Option<Value>,
    /// Provider-specific machine-readable code.
    pub code: Option<String>,
    /// Provider-specific request parameter associated with the error.
    pub param: Option<String>,
    /// Provider-specific machine-readable error type.
    pub error_type: Option<String>,
    /// Whether retrying the same request may reasonably succeed.
    pub is_retryable: bool,
    /// Model id for [`ProviderErrorKind::ModelNotFound`], when known.
    pub model_id: Option<String>,
}

impl fmt::Debug for ProviderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProviderError")
            .field("message", &self.message)
            .field("provider", &self.provider)
            .field("kind", &self.kind)
            .field("request_id", &self.request_id)
            .field("http_context", &self.http_context)
            .field("has_body", &self.body.is_some())
            .field("code", &self.code)
            .field("param", &self.param)
            .field("error_type", &self.error_type)
            .field("is_retryable", &self.is_retryable)
            .field("model_id", &self.model_id)
            .finish()
    }
}

impl ProviderError {
    /// Creates a provider error with no HTTP context.
    pub fn new(kind: ProviderErrorKind, message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            provider: None,
            kind,
            request_id: None,
            http_context: None,
            body: None,
            code: None,
            param: None,
            error_type: None,
            is_retryable: matches!(
                kind,
                ProviderErrorKind::Connection | ProviderErrorKind::Timeout
            ),
            model_id: None,
        }
    }

    /// Creates a generic provider API error.
    pub fn api(message: impl Into<String>, provider: Option<String>) -> Self {
        Self::new(ProviderErrorKind::Api, message).with_provider_option(provider)
    }

    /// Creates a categorized provider status error.
    pub fn from_status(
        message: impl Into<String>,
        status_code: u16,
        provider: Option<String>,
    ) -> Self {
        Self::new(ProviderErrorKind::from_status(status_code), message)
            .with_provider_option(provider)
            .with_http_context(HttpErrorContext::new(status_code))
    }

    /// Creates an incomplete-response error for a stream without a finish event.
    pub fn incomplete_response(message: impl Into<String>, provider: Option<String>) -> Self {
        Self::new(ProviderErrorKind::IncompleteResponse, message)
            .with_provider_option(provider)
            .with_retryable(true)
    }

    /// Sets the provider name.
    pub fn with_provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = Some(provider.into());
        self
    }

    /// Sets an optional provider name.
    pub fn with_provider_option(mut self, provider: Option<String>) -> Self {
        self.provider = provider;
        self
    }

    /// Sets the provider request id.
    pub fn with_request_id(mut self, request_id: impl Into<String>) -> Self {
        self.request_id = Some(request_id.into());
        self
    }

    /// Sets normalized HTTP context and derives retryability from its status.
    pub fn with_http_context(mut self, context: HttpErrorContext) -> Self {
        self.is_retryable = is_retryable_status(Some(context.status_code));
        self.http_context = Some(context);
        self
    }

    /// Sets the decoded provider response body.
    pub fn with_body(mut self, body: Value) -> Self {
        self.body = Some(body);
        self
    }

    /// Sets a provider-specific error code.
    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.code = Some(code.into());
        self
    }

    /// Sets the provider parameter associated with the error.
    pub fn with_param(mut self, param: impl Into<String>) -> Self {
        self.param = Some(param.into());
        self
    }

    /// Sets a provider-specific error type.
    pub fn with_error_type(mut self, error_type: impl Into<String>) -> Self {
        self.error_type = Some(error_type.into());
        self
    }

    /// Overrides retryability when an adapter has provider-specific knowledge.
    pub const fn with_retryable(mut self, is_retryable: bool) -> Self {
        self.is_retryable = is_retryable;
        self
    }

    /// Marks the missing model for a model-not-found error.
    pub fn with_model_id(mut self, model_id: impl Into<String>) -> Self {
        self.kind = ProviderErrorKind::ModelNotFound;
        self.model_id = Some(model_id.into());
        self
    }

    /// Returns the normalized HTTP status, when one exists.
    pub fn status_code(&self) -> Option<u16> {
        self.http_context
            .as_ref()
            .map(|context| context.status_code)
    }
}

/// Maps a non-success HTTP status to its normalized provider category.
pub const fn http_status_to_provider_error_kind(status_code: u16) -> ProviderErrorKind {
    ProviderErrorKind::from_status(status_code)
}

/// Returns whether the status is retryable by default.
pub const fn is_retryable_status(status_code: Option<u16>) -> bool {
    matches!(status_code, Some(408 | 409 | 429 | 500..=u16::MAX))
}
