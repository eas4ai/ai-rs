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

//! HTTP plumbing the provider adapters share: request building, bounded body
//! reads, status and transport error mapping, and reading server-sent events
//! as JSON. An [`ApiProfile`] holds what differs between APIs.
use crate::{
    Error, ProviderConfig,
    errors::{HttpErrorContext, ProviderError, ProviderErrorKind},
    transport::{Body, Headers, HttpRequest, HttpResponse, HttpTransport, TransportError, sse},
};
use bytes::Bytes;
use futures::{StreamExt, stream::BoxStream};
use serde_json::Value;
use std::collections::BTreeMap;

/// The most bytes of a response body read whole, such as an error body.
const MAX_RESPONSE_BYTES: usize = 8 * 1024 * 1024;

/// How an API authenticates and reports its errors.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ApiProfile {
    /// The API's name in error messages, such as `OpenAI`.
    pub label: &'static str,
    pub auth: Auth,
    /// Headers every request sends before the configured ones.
    pub default_headers: &'static [(&'static str, &'static str)],
    /// The response header that carries the request id.
    pub request_id_header: &'static str,
    /// Report every 5xx response as `InternalServer`, as the Python OpenAI SDK does.
    pub collapse_server_errors: bool,
    /// Keep only the `error` object of an error body as the error's body.
    pub keep_error_object: bool,
}

/// Where a request carries the API key.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Auth {
    /// `authorization: Bearer <key>`.
    Bearer,
    /// The key as the value of this header.
    Header(&'static str),
}

/// Build a request to `path` under the configured base URL. Headers apply in
/// this order, later ones replacing earlier ones: the defaults, the API key,
/// the configured headers, then `extra_headers`. A query value that is an
/// array becomes one pair per item.
pub(crate) fn request(
    api: &ApiProfile,
    config: &ProviderConfig,
    method: &str,
    path: &[&str],
    body: Option<Value>,
    extra_headers: BTreeMap<String, String>,
    query: BTreeMap<String, Value>,
) -> Result<HttpRequest, Error> {
    let label = api.label;
    let mut url = url::Url::parse(&config.base_url())
        .map_err(|_| Error::Configuration(format!("{label} base URL is invalid")))?;
    if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
        return Err(Error::Configuration(format!(
            "{label} base URL must use HTTP or HTTPS"
        )));
    }
    url.path_segments_mut()
        .map_err(|_| Error::Configuration(format!("{label} base URL cannot hold a path")))?
        .pop_if_empty()
        .extend(path);
    if !query.is_empty() {
        let mut pairs = url.query_pairs_mut();
        for (key, value) in query {
            match value {
                Value::Array(items) => {
                    for item in items {
                        pairs.append_pair(&key, &query_value(item));
                    }
                }
                value => {
                    pairs.append_pair(&key, &query_value(value));
                }
            }
        }
    }
    let mut headers = BTreeMap::from([("accept".to_owned(), "application/json".to_owned())]);
    if body.is_some() {
        headers.insert("content-type".into(), "application/json".into());
    }
    for (name, value) in api.default_headers {
        headers.insert((*name).to_owned(), (*value).to_owned());
    }
    if let Some(key) = config.api_key().filter(|key| !key.is_empty()) {
        match api.auth {
            Auth::Bearer => headers.insert("authorization".into(), format!("Bearer {key}")),
            Auth::Header(name) => headers.insert(name.into(), key.into()),
        };
    }
    for (name, value) in config.headers.iter().chain(&extra_headers) {
        headers.insert(name.to_ascii_lowercase(), value.clone());
    }
    Ok(HttpRequest {
        method: method.into(),
        url: url.into(),
        headers: headers.into_iter().collect(),
        body: match body {
            Some(body) => serde_json::to_vec(&body)?.into(),
            None => Bytes::new(),
        },
    })
}

fn query_value(value: Value) -> String {
    match value {
        Value::String(text) => text,
        Value::Null => String::new(),
        value => value.to_string(),
    }
}

/// Send a request and return the response when its status is 2xx; otherwise
/// read the body and map the status to a provider error.
pub(crate) async fn send(
    api: &ApiProfile,
    transport: &dyn HttpTransport,
    request: HttpRequest,
    provider: &str,
    model_id: Option<&str>,
) -> Result<HttpResponse, Error> {
    let response = transport
        .send(request.clone())
        .await
        .map_err(|error| transport_error(api, provider, error))?;
    if (200..300).contains(&response.status) {
        return Ok(response);
    }
    let status = response.status;
    let (headers, bytes) = read_body(api, provider, response).await?;
    Err(status_error(
        api, provider, model_id, &request, status, headers, &bytes,
    ))
}

/// Send a request and decode its JSON response body.
pub(crate) async fn json(
    api: &ApiProfile,
    transport: &dyn HttpTransport,
    request: HttpRequest,
    provider: &str,
    model_id: Option<&str>,
) -> Result<Value, Error> {
    let response = send(api, transport, request, provider, model_id).await?;
    let (_, bytes) = read_body(api, provider, response).await?;
    serde_json::from_slice(&bytes).map_err(|_| {
        response_error(
            provider,
            format!("{} response is not valid JSON", api.label),
        )
    })
}

/// Read a whole response body, up to 8 MiB.
pub(crate) async fn read_body(
    api: &ApiProfile,
    provider: &str,
    mut response: HttpResponse,
) -> Result<(Headers, Bytes), Error> {
    let mut bytes = Vec::new();
    while let Some(chunk) = response.body.next().await {
        let chunk = chunk.map_err(|error| transport_error(api, provider, error))?;
        if bytes.len().saturating_add(chunk.len()) > MAX_RESPONSE_BYTES {
            return Err(response_error(
                provider,
                format!("{} response exceeds the 8 MiB limit", api.label),
            ));
        }
        bytes.extend_from_slice(&chunk);
    }
    Ok((response.headers, bytes.into()))
}

/// A malformed or unexpected response.
pub(crate) fn response_error(provider: &str, message: impl Into<String>) -> Error {
    ProviderError::new(ProviderErrorKind::Response, message)
        .with_provider(provider)
        .into()
}

/// Map a transport failure to a provider error. The message names only the
/// category, because transport messages can hold request URLs or
/// credentials; the original error stays as the source.
pub(crate) fn transport_error(api: &ApiProfile, provider: &str, error: TransportError) -> Error {
    let (kind, what) = match error {
        TransportError::Timeout => (ProviderErrorKind::Timeout, "request timed out"),
        TransportError::Connection(_) | TransportError::Body(_) => {
            (ProviderErrorKind::Connection, "connection failed")
        }
        TransportError::Request(_) => (ProviderErrorKind::Api, "request failed"),
        TransportError::Protocol(_) | TransportError::Limit(_) => {
            (ProviderErrorKind::Response, "response failed")
        }
    };
    ProviderError::new(kind, format!("{} {what}", api.label))
        .with_provider(provider)
        .with_source(error)
        .into()
}

/// The message of an error body: `{"error": {"message": ...}}`,
/// `{"error": "<text>"}` or `{"message": ...}`.
pub(crate) fn error_message(body: &Value) -> Option<&str> {
    let error = body.get("error");
    error
        .and_then(|error| error.get("message"))
        .and_then(Value::as_str)
        .or_else(|| body.get("message").and_then(Value::as_str))
        .filter(|message| !message.is_empty())
}

/// A field of an error body, from its `error` object first, then from the
/// body itself. A number becomes its decimal text.
pub(crate) fn error_field(body: &Value, key: &str) -> Option<String> {
    let value = body
        .get("error")
        .and_then(|error| error.get(key))
        .filter(|value| !value.is_null())
        .or_else(|| body.get(key).filter(|value| !value.is_null()))?;
    Some(match value {
        Value::String(text) => text.clone(),
        value => value.to_string(),
    })
}

/// Map an error response to a provider error with its HTTP context.
pub(crate) fn status_error(
    api: &ApiProfile,
    provider: &str,
    model_id: Option<&str>,
    request: &HttpRequest,
    status: u16,
    headers: Headers,
    bytes: &[u8],
) -> Error {
    let body = serde_json::from_slice::<Value>(bytes)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(bytes).into_owned()));
    let message = error_message(&body)
        .map(str::to_owned)
        .unwrap_or_else(|| format!("{} request failed with HTTP {status}", api.label));
    let mut error = ProviderError::new(ProviderErrorKind::from_status(status), message)
        .with_provider(provider)
        .with_http_context(
            HttpErrorContext::new(status)
                .with_request(&request.method, &request.url)
                .with_response_headers(headers.clone()),
        );
    if api.collapse_server_errors && status >= 500 {
        error.kind = ProviderErrorKind::InternalServer;
    }
    if status == 404
        && let Some(model) = model_id
    {
        error = error.with_model_id(model);
    }
    error.request_id = headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case(api.request_id_header))
        .map(|(_, value)| value.clone());
    error.code = error_field(&body, "code");
    error.param = error_field(&body, "param");
    error.error_type = error_field(&body, "type");
    error.body = Some(match body.get("error").filter(|_| api.keep_error_object) {
        Some(details) if details.is_object() => details.clone(),
        _ => body,
    });
    error.into()
}

/// One server-sent event read as JSON.
#[derive(Debug)]
pub(crate) enum SseJson {
    /// An event whose data is a JSON object.
    Object(Value),
    /// The `[DONE]` marker some servers send after the last event.
    Done,
}

/// Read a server-sent event body as JSON objects. Transport failures map
/// through the profile; data that is not a JSON object is a response error.
pub(crate) fn sse_json(
    api: ApiProfile,
    provider: String,
    body: Body,
) -> BoxStream<'static, Result<SseJson, Error>> {
    sse::parse(body)
        .map(move |event| {
            let event = event.map_err(|error| transport_error(&api, &provider, error))?;
            if event.data.trim() == "[DONE]" {
                return Ok(SseJson::Done);
            }
            match serde_json::from_str::<Value>(&event.data) {
                Ok(value) if value.is_object() => Ok(SseJson::Object(value)),
                Ok(_) => Err(response_error(
                    &provider,
                    format!("{} stream event is not a JSON object", api.label),
                )),
                Err(_) => Err(response_error(
                    &provider,
                    format!("{} stream event is not valid JSON", api.label),
                )),
            }
        })
        .boxed()
}
