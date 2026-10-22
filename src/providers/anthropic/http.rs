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

use super::errors;
use crate::{
    Error, ProviderConfig,
    transport::{Headers, HttpRequest, HttpResponse, HttpTransport},
};
use bytes::Bytes;
use futures::StreamExt;
use serde_json::Value;
use std::{collections::BTreeMap, sync::Arc};
const MAX_RESPONSE_BYTES: usize = 8 * 1024 * 1024;

pub(super) fn request(
    config: &ProviderConfig,
    method: &str,
    path: &[&str],
    body: Option<Value>,
    extra_headers: BTreeMap<String, String>,
    query: BTreeMap<String, Value>,
) -> Result<HttpRequest, Error> {
    let mut url = url::Url::parse(&config.base_url())
        .map_err(|_| Error::Configuration("Anthropic base URL is invalid".into()))?;
    if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
        return Err(Error::Configuration(
            "Anthropic base URL must be HTTP or HTTPS".into(),
        ));
    }
    url.path_segments_mut()
        .map_err(|_| Error::Configuration("Anthropic base URL cannot hold a path".into()))?
        .pop_if_empty()
        .extend(path);
    if !query.is_empty() {
        let mut pairs = url.query_pairs_mut();
        for (k, v) in query {
            match v {
                Value::Array(items) => {
                    for item in items {
                        pairs.append_pair(&k, &query_value(item));
                    }
                }
                v => {
                    pairs.append_pair(&k, &query_value(v));
                }
            }
        }
    }
    let mut headers = BTreeMap::from([
        ("accept".into(), "application/json".into()),
        ("anthropic-version".into(), "2023-06-01".into()),
    ]);
    if body.is_some() {
        headers.insert("content-type".into(), "application/json".into());
    }
    if let Some(key) = config.api_key() {
        headers.insert("x-api-key".into(), key.into());
    }
    for (k, v) in config.headers.iter().chain(extra_headers.iter()) {
        headers.insert(k.to_ascii_lowercase(), v.clone());
    }
    Ok(HttpRequest {
        method: method.into(),
        url: url.into(),
        headers: headers.into_iter().collect(),
        body: body
            .map(|v| serde_json::to_vec(&v).map(Bytes::from))
            .transpose()?
            .unwrap_or_default(),
    })
}
fn query_value(value: Value) -> String {
    match value {
        Value::String(s) => s,
        Value::Null => String::new(),
        v => v.to_string(),
    }
}
pub(super) async fn read_body(
    mut response: HttpResponse,
    provider: &str,
) -> Result<(Headers, Vec<u8>), Error> {
    let mut bytes = Vec::new();
    while let Some(chunk) = response.body.next().await {
        let chunk = chunk.map_err(|e| errors::transport(provider, e))?;
        if bytes.len().saturating_add(chunk.len()) > MAX_RESPONSE_BYTES {
            return Err(errors::response(
                provider,
                "Anthropic response exceeded the 8 MiB limit",
            ));
        }
        bytes.extend_from_slice(&chunk);
    }
    Ok((response.headers, bytes))
}
pub(super) async fn send(
    transport: &dyn HttpTransport,
    request: HttpRequest,
    provider: &str,
    model_id: Option<&str>,
) -> Result<HttpResponse, Error> {
    let response = transport
        .send(request.clone())
        .await
        .map_err(|e| errors::transport(provider, e))?;
    if (200..300).contains(&response.status) {
        return Ok(response);
    }
    let status = response.status;
    let (headers, bytes) = read_body(response, provider).await?;
    let body = serde_json::from_slice(&bytes)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned()));
    Err(errors::status(
        provider, model_id, &request, status, headers, body,
    ))
}
pub(super) async fn json(
    transport: Arc<dyn HttpTransport>,
    request: HttpRequest,
    provider: &str,
    model_id: Option<&str>,
) -> Result<Value, Error> {
    let response = send(transport.as_ref(), request, provider, model_id).await?;
    let (_, bytes) = read_body(response, provider).await?;
    serde_json::from_slice(&bytes)
        .map_err(|_| errors::response(provider, "Anthropic response was not valid JSON"))
}
