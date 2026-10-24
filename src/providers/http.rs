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
pub(crate) const MAX_RESPONSE_BYTES: usize = 8 * 1024 * 1024;

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
    /// Read an error body's message, code, parameter and type from its top
    /// level before its `error` object, as the Python Anthropic SDK does.
    pub top_level_first: bool,
    /// An error body `{"code": <text>, "error": <text>}` has the message
    /// `<code>: <error>`, as xAI's does.
    pub code_prefixes_error: bool,
    /// Headers taken from the configuration's `env` values, as
    /// `(env name, header name)`, sent when the value is set.
    pub config_headers: &'static [(&'static str, &'static str)],
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
/// the headers the profile takes from the configuration's `env` values, the
/// configured headers, then `extra_headers`. A query value that is an
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
    request_to(
        api,
        &config.base_url(),
        config,
        method,
        path,
        body,
        extra_headers,
        query,
    )
}

/// [`request`] to `path` under `base_url` instead of the configured base URL.
#[allow(clippy::too_many_arguments)]
pub(crate) fn request_to(
    api: &ApiProfile,
    base_url: &str,
    config: &ProviderConfig,
    method: &str,
    path: &[&str],
    body: Option<Value>,
    extra_headers: BTreeMap<String, String>,
    query: BTreeMap<String, Value>,
) -> Result<HttpRequest, Error> {
    let label = api.label;
    let mut url = url::Url::parse(base_url)
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
    for (env, name) in api.config_headers {
        if let Some(value) = config.env.get(*env).filter(|value| !value.is_empty()) {
            headers.insert(name.to_ascii_lowercase(), value.clone());
        }
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
    let headers = response.headers.clone();
    // The status says what failed; a body that cannot be read only loses the
    // message, so a failed read must not turn the error into a retryable one.
    let bytes = match read_body(api, provider, response).await {
        Ok((_, bytes)) => bytes,
        Err(_) => Bytes::new(),
    };
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
    json_with_limit(
        api,
        transport,
        request,
        provider,
        model_id,
        MAX_RESPONSE_BYTES,
    )
    .await
    .map(|(_, value)| value)
}

/// [`json`] reading up to `limit` bytes of the response body, with the
/// response headers.
pub(crate) async fn json_with_limit(
    api: &ApiProfile,
    transport: &dyn HttpTransport,
    request: HttpRequest,
    provider: &str,
    model_id: Option<&str>,
    limit: usize,
) -> Result<(Headers, Value), Error> {
    let response = send(api, transport, request, provider, model_id).await?;
    let (headers, bytes) = read_body_with_limit(api, provider, response, limit).await?;
    let value = serde_json::from_slice(&bytes).map_err(|_| {
        response_error(
            provider,
            format!("{} response is not valid JSON", api.label),
        )
    })?;
    Ok((headers, value))
}

/// Read a whole response body, up to 8 MiB.
pub(crate) async fn read_body(
    api: &ApiProfile,
    provider: &str,
    response: HttpResponse,
) -> Result<(Headers, Bytes), Error> {
    read_body_with_limit(api, provider, response, MAX_RESPONSE_BYTES).await
}

/// Read a whole response body, up to `limit` bytes. A longer body fails with
/// an error naming the limit, without reading past it.
pub(crate) async fn read_body_with_limit(
    api: &ApiProfile,
    provider: &str,
    mut response: HttpResponse,
    limit: usize,
) -> Result<(Headers, Bytes), Error> {
    let mut bytes = Vec::new();
    while let Some(chunk) = response.body.next().await {
        let chunk = chunk.map_err(|error| transport_error(api, provider, error))?;
        if bytes.len().saturating_add(chunk.len()) > limit {
            return Err(limit_error(api, provider, "response", limit));
        }
        bytes.extend_from_slice(&chunk);
    }
    Ok((response.headers, bytes.into()))
}

/// A response or stream event over its size limit, naming the limit.
pub(crate) fn limit_error(api: &ApiProfile, provider: &str, what: &str, limit: usize) -> Error {
    let size = if limit.is_multiple_of(1024 * 1024) {
        format!("{} MiB", limit / (1024 * 1024))
    } else {
        format!("{limit} bytes")
    };
    response_error(
        provider,
        format!("{} {what} exceeds the {size} limit", api.label),
    )
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
/// `{"error": "<text>"}` or `{"message": ...}`, the top level first when the
/// profile says so, or else DeepInfra's `{"detail": {"error": ...}}`.
pub(crate) fn error_message<'a>(api: &ApiProfile, body: &'a Value) -> Option<&'a str> {
    let text = |value: Option<&'a Value>| {
        value
            .and_then(Value::as_str)
            .filter(|message| !message.is_empty())
    };
    let error = body.get("error");
    let nested = text(error.and_then(|error| error.get("message")));
    let top = text(body.get("message"));
    let first = if api.top_level_first { top } else { nested };
    let second = if api.top_level_first { nested } else { top };
    first
        .or_else(|| text(error))
        .or(second)
        .or_else(|| text(body.pointer("/detail/error")))
}

/// A field of an error body from its `error` object and its top level, in
/// the profile's order. A number becomes its decimal text.
pub(crate) fn error_field(api: &ApiProfile, body: &Value, key: &str) -> Option<String> {
    let nested = body
        .get("error")
        .and_then(|error| error.get(key))
        .filter(|value| !value.is_null());
    let top = body.get(key).filter(|value| !value.is_null());
    let value = if api.top_level_first {
        top.or(nested)
    } else {
        nested.or(top)
    }?;
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
    let prefixed = api
        .code_prefixes_error
        .then(|| (body["code"].as_str(), body["error"].as_str()))
        .and_then(|pair| match pair {
            (Some(code), Some(text)) => Some(format!("{code}: {text}")),
            _ => None,
        });
    let message = prefixed
        .or_else(|| error_message(api, &body).map(str::to_owned))
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
    error.code = error_field(api, &body, "code");
    error.param = error_field(api, &body, "param");
    error.error_type = error_field(api, &body, "type");
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

/// The most bytes of one batch results line.
pub(crate) const RESULT_LINE_LIMIT: usize = 64 * 1024 * 1024;

/// Read a body as JSON lines, each at most `limit` bytes.
pub(crate) fn json_lines(
    api: ApiProfile,
    provider: String,
    body: Body,
    limit: usize,
) -> BoxStream<'static, Result<Value, Error>> {
    struct Lines {
        body: Body,
        buffer: bytes::BytesMut,
        /// How much of the buffer holds no newline, so that each byte is
        /// searched once.
        searched: usize,
        done: bool,
    }
    let state = Lines {
        body,
        buffer: bytes::BytesMut::new(),
        searched: 0,
        done: false,
    };
    futures::stream::try_unfold(state, move |mut state| {
        let provider = provider.clone();
        async move {
            loop {
                let found = state.buffer[state.searched..]
                    .iter()
                    .position(|b| *b == b'\n')
                    .map(|at| state.searched + at);
                state.searched = found.unwrap_or(state.buffer.len());
                if let Some(end) = found {
                    let line = state.buffer.split_to(end + 1);
                    state.searched = 0;
                    let text = &line[..end];
                    let text = text.strip_suffix(b"\r").unwrap_or(text);
                    if text.iter().all(u8::is_ascii_whitespace) {
                        continue;
                    }
                    let value = serde_json::from_slice(text).map_err(|_| {
                        response_error(
                            &provider,
                            format!("{} batch result line is not valid JSON", api.label),
                        )
                    })?;
                    return Ok(Some((value, state)));
                }
                if state.buffer.len() > limit {
                    return Err(limit_error(&api, &provider, "batch result line", limit));
                }
                if state.done {
                    if state.buffer.iter().all(u8::is_ascii_whitespace) {
                        return Ok(None);
                    }
                    let rest = state.buffer.split();
                    state.searched = 0;
                    let value = serde_json::from_slice(&rest).map_err(|_| {
                        response_error(
                            &provider,
                            format!("{} batch result line is not valid JSON", api.label),
                        )
                    })?;
                    return Ok(Some((value, state)));
                }
                match state.body.next().await {
                    Some(Ok(chunk)) => state.buffer.extend_from_slice(&chunk),
                    Some(Err(error)) => return Err(transport_error(&api, &provider, error)),
                    None => state.done = true,
                }
            }
        }
    })
    .boxed()
}

/// Read a server-sent event body as JSON objects. Transport failures map
/// through the profile; data that is not a JSON object is a response error.
pub(crate) fn sse_json(
    api: ApiProfile,
    provider: String,
    body: Body,
) -> BoxStream<'static, Result<SseJson, Error>> {
    sse_json_with_limit(api, provider, body, sse::MAX_EVENT_BYTES)
}

/// [`sse_json`] with each line and event's data limited to `limit` bytes;
/// a longer one fails with an error naming the limit.
pub(crate) fn sse_json_with_limit(
    api: ApiProfile,
    provider: String,
    body: Body,
    limit: usize,
) -> BoxStream<'static, Result<SseJson, Error>> {
    sse::parse_with_limit(body, limit)
        .map(move |event| {
            let event = event.map_err(|error| match error {
                TransportError::Limit(limit) => limit_error(&api, &provider, "stream event", limit),
                error => transport_error(&api, &provider, error),
            })?;
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
