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

//! The Gemini API's endpoints, errors and size limits, and what differs on
//! Vertex AI, which runs the same generate content protocol.

use std::{collections::BTreeMap, sync::Arc};

use serde_json::{Map, Value};

use crate::{
    Error, Provider, ProviderConfig,
    errors::ProviderError,
    providers::http::{self, ApiProfile, Auth},
    transport::{Headers, HttpRequest, HttpResponse, HttpTransport},
    types::messages::Metadata,
};

/// The Gemini API's default base URL.
pub(crate) const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";

/// The most bytes of a generate content, image, speech, video or
/// Interactions response body, and of one of their stream events.
pub(crate) const MEDIA_RESPONSE_BYTES: usize = 256 * 1024 * 1024;

pub(crate) static GEMINI_API: ApiProfile = ApiProfile {
    label: "Google Generative AI",
    auth: Auth::Header("x-goog-api-key"),
    default_headers: &[],
    request_id_header: "x-request-id",
    collapse_server_errors: false,
    keep_error_object: false,
    top_level_first: false,
    code_prefixes_error: false,
};

pub(crate) static VERTEX_API: ApiProfile = ApiProfile {
    label: "Google Vertex AI",
    auth: Auth::Header("x-goog-api-key"),
    default_headers: &[],
    request_id_header: "x-request-id",
    collapse_server_errors: false,
    keep_error_object: false,
    top_level_first: false,
    code_prefixes_error: false,
};

/// Which API a Gemini protocol talks to: the Gemini API or Vertex AI. The
/// generate content protocol is the same; the option and metadata keys and a
/// few request rules differ (VTX-003).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Flavor {
    Gemini,
    Vertex,
}

impl Flavor {
    pub(crate) fn api(self) -> &'static ApiProfile {
        match self {
            Self::Gemini => &GEMINI_API,
            Self::Vertex => &VERTEX_API,
        }
    }

    pub(crate) fn is_vertex(self) -> bool {
        self == Self::Vertex
    }

    /// The provider parameter keys that hold the options, first match wins.
    pub(crate) fn option_keys(self) -> &'static [&'static str] {
        match self {
            Self::Gemini => &["google"],
            Self::Vertex => &["googleVertex", "vertex", "google"],
        }
    }

    /// The part and message metadata keys a response is written under.
    pub(crate) fn metadata_keys(self) -> &'static [&'static str] {
        match self {
            Self::Gemini => &["google"],
            Self::Vertex => &["googleVertex", "vertex"],
        }
    }

    /// The metadata keys a history part is read from, first match wins: the
    /// flavor's own keys, then the other flavor's.
    pub(crate) fn part_keys(self) -> &'static [&'static str] {
        match self {
            Self::Gemini => &["google", "googleVertex", "vertex"],
            Self::Vertex => &["googleVertex", "vertex", "google"],
        }
    }

    /// Metadata with `payload` under each of the flavor's metadata keys.
    pub(crate) fn metadata(self, payload: Value) -> Metadata {
        self.metadata_keys()
            .iter()
            .map(|key| ((*key).to_owned(), payload.clone()))
            .collect()
    }

    /// The options object of the call's provider parameters, from the
    /// first key the flavor reads that is present.
    pub(crate) fn options<'a>(
        self,
        params: Option<&'a BTreeMap<String, Value>>,
    ) -> Option<(&'static str, &'a Value)> {
        let params = params?;
        self.option_keys()
            .iter()
            .find_map(|key| params.get(*key).map(|value| (*key, value)))
            .filter(|(_, value)| !value.is_null())
    }

    /// The flavor's metadata of a history part, first match wins.
    pub(crate) fn part_metadata<'a>(
        self,
        metadata: Option<&'a Metadata>,
    ) -> Option<&'a Map<String, Value>> {
        let metadata = metadata?;
        self.part_keys()
            .iter()
            .find_map(|key| metadata.get(*key).and_then(Value::as_object))
    }
}

/// The parts of a model path as URL path segments, with the method after a
/// colon on the last one, such as `["models", "gemini-2.5-flash:generateContent"]`.
pub(crate) fn model_segments(model_id: &str, method: &str) -> Vec<String> {
    let path = super::capabilities::model_path(model_id);
    let mut segments: Vec<String> = path.split('/').map(str::to_owned).collect();
    if let Some(last) = segments.last_mut() {
        last.push(':');
        last.push_str(method);
    }
    segments
}

/// Where a protocol's requests go: the flavor, the base URL and the
/// provider's configuration and transport.
#[derive(Clone)]
pub(crate) struct Endpoint {
    pub flavor: Flavor,
    pub base: String,
    pub config: ProviderConfig,
    pub provider: String,
    pub transport: Arc<dyn HttpTransport>,
}

impl Endpoint {
    pub(crate) fn of(flavor: Flavor, provider: &dyn Provider) -> Result<Self, Error> {
        Ok(Self {
            flavor,
            base: provider.base_url(),
            config: provider.config().clone(),
            provider: provider.name().to_owned(),
            transport: transport(flavor, provider)?,
        })
    }

    /// A JSON request to `path` under the base URL.
    pub(crate) fn request(
        &self,
        method: &str,
        path: &[String],
        body: Option<Value>,
        headers: BTreeMap<String, String>,
        query: BTreeMap<String, Value>,
    ) -> Result<HttpRequest, Error> {
        self.request_at(&self.base, method, path, body, headers, query)
    }

    /// A JSON request to `path` under another base URL of the same API.
    pub(crate) fn request_at(
        &self,
        base: &str,
        method: &str,
        path: &[String],
        body: Option<Value>,
        headers: BTreeMap<String, String>,
        query: BTreeMap<String, Value>,
    ) -> Result<HttpRequest, Error> {
        let path: Vec<&str> = path.iter().map(String::as_str).collect();
        http::request_to(
            self.flavor.api(),
            base,
            &self.config,
            method,
            &path,
            body,
            headers,
            query,
        )
    }

    pub(crate) async fn send(
        &self,
        request: HttpRequest,
        model_id: Option<&str>,
    ) -> Result<HttpResponse, Error> {
        send(
            self.flavor,
            self.transport.as_ref(),
            request,
            &self.provider,
            model_id,
        )
        .await
    }

    pub(crate) async fn json(
        &self,
        request: HttpRequest,
        model_id: Option<&str>,
        limit: usize,
    ) -> Result<(Headers, Value), Error> {
        json(
            self.flavor,
            self.transport.as_ref(),
            request,
            &self.provider,
            model_id,
            limit,
        )
        .await
    }
}

/// The provider's transport, which a Gemini protocol needs to own for a
/// stream.
pub(crate) fn transport(
    flavor: Flavor,
    provider: &dyn Provider,
) -> Result<Arc<dyn HttpTransport>, Error> {
    provider.shared_transport().ok_or_else(|| {
        Error::Configuration(format!(
            "{} protocols require Provider::shared_transport",
            flavor.api().label
        ))
    })
}

/// Send a request; an error response gives the error the message, code and
/// type (`error.status`) of its body, with the kind from the HTTP status.
pub(crate) async fn send(
    flavor: Flavor,
    transport: &dyn HttpTransport,
    request: HttpRequest,
    provider: &str,
    model_id: Option<&str>,
) -> Result<HttpResponse, Error> {
    http::send(flavor.api(), transport, request, provider, model_id)
        .await
        .map_err(with_status_type)
}

/// Send a request and read its JSON body, up to `limit` bytes.
pub(crate) async fn json(
    flavor: Flavor,
    transport: &dyn HttpTransport,
    request: HttpRequest,
    provider: &str,
    model_id: Option<&str>,
    limit: usize,
) -> Result<(Headers, Value), Error> {
    http::json_with_limit(flavor.api(), transport, request, provider, model_id, limit)
        .await
        .map_err(with_status_type)
}

/// Google errors carry their type in `error.status`.
fn with_status_type(error: Error) -> Error {
    match error {
        Error::Provider(mut error) => {
            if let Some(status) = error
                .body
                .as_ref()
                .and_then(|body| body.get("error"))
                .and_then(|error| error.get("status"))
                .and_then(Value::as_str)
            {
                error.error_type = Some(status.to_owned());
            }
            Error::Provider(error)
        }
        other => other,
    }
}

/// The error a stream chunk's `error` object ends the call with: its
/// message, code and status, the kind of its code as an HTTP status,
/// retryable for 408, 409, 429 and 500 and above.
pub(crate) fn stream_error(provider: &str, error: &Value) -> Error {
    let message = error
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("Google stream error")
        .to_owned();
    let code = error.get("code").and_then(Value::as_u64);
    let kind = code.and_then(|code| u16::try_from(code).ok()).map_or(
        crate::errors::ProviderErrorKind::Provider,
        crate::errors::ProviderErrorKind::from_status,
    );
    let mut provider_error = ProviderError::new(kind, message)
        .with_provider(provider)
        .with_body(serde_json::json!({ "error": error }))
        .with_retryable(code.is_some_and(|code| matches!(code, 408 | 409 | 429) || code >= 500));
    if let Some(code) = code {
        provider_error.code = Some(code.to_string());
    }
    if let Some(status) = error.get("status").and_then(Value::as_str) {
        provider_error.error_type = Some(status.to_owned());
    }
    provider_error.into()
}

/// The origin of a base URL: its scheme, host and port.
pub(crate) fn origin(url: &str) -> Option<String> {
    let url = url::Url::parse(url).ok()?;
    Some(url.origin().ascii_serialization())
}

/// The base URL without a trailing `/v1beta` or `/v1alpha` segment.
pub(crate) fn without_version(base: &str) -> &str {
    let base = base.trim_end_matches('/');
    base.strip_suffix("/v1beta")
        .or_else(|| base.strip_suffix("/v1alpha"))
        .unwrap_or(base)
}
