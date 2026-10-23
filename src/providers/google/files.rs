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

//! The Gemini API files API: a resumable upload in 8 MiB chunks, each
//! checked by its `X-Goog-Upload-Status`, then polling while the file is
//! processed (`google-files.ts`, and `api_client.go` of Google's Go SDK for
//! the chunks).

use std::time::Duration;

use bytes::Bytes;
use serde_json::{Value, json};

use super::api::Endpoint;
use crate::{
    Error,
    errors::{ProviderError, ProviderErrorKind},
    models::call::require_timer,
    providers::{http, operations::*},
    transport::{HttpRequest, HttpResponse},
};

/// The size of one upload chunk.
pub(crate) const CHUNK: usize = 8 * 1024 * 1024;
/// Sends of a chunk whose response lacks an upload status.
const CHUNK_SENDS: u32 = 3;
const FIRST_RESEND: Duration = Duration::from_secs(1);

fn upload_error(provider: &str, message: String) -> Error {
    ProviderError::new(ProviderErrorKind::Response, message)
        .with_provider(provider)
        .into()
}

fn header<'a>(response: &'a HttpResponse, name: &str) -> Option<&'a str> {
    response
        .headers
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case(name))
        .map(|(_, value)| value.as_str())
}

/// Upload bytes with a resumable upload and return the file resource the
/// final chunk's response holds.
pub(crate) async fn resumable_upload(
    endpoint: &Endpoint,
    data: Bytes,
    media_type: &str,
    display_name: Option<&str>,
) -> Result<Value, Error> {
    let origin = endpoint.base.trim_end_matches('/');
    let origin = origin.strip_suffix("/v1beta").unwrap_or(origin);
    let mut file = serde_json::Map::new();
    if let Some(name) = display_name {
        file.insert("display_name".into(), json!(name));
    }
    let headers = [
        ("X-Goog-Upload-Protocol", "resumable".to_owned()),
        ("X-Goog-Upload-Command", "start".to_owned()),
        (
            "X-Goog-Upload-Header-Content-Length",
            data.len().to_string(),
        ),
        ("X-Goog-Upload-Header-Content-Type", media_type.to_owned()),
    ]
    .into_iter()
    .map(|(name, value)| (name.to_owned(), value))
    .collect();
    let request = endpoint.request_at(
        origin,
        "POST",
        &["upload".into(), "v1beta".into(), "files".into()],
        Some(json!({"file": file})),
        headers,
        Default::default(),
    )?;
    let response = endpoint.send(request, None).await?;
    let upload_url = header(&response, "x-goog-upload-url")
        .filter(|url| !url.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| {
            upload_error(
                &endpoint.provider,
                "No upload URL returned from initiation request".into(),
            )
        })?;
    // The start response's body is not needed; drop it unread.
    drop(response);

    let mut offset = 0;
    loop {
        let end = (offset + CHUNK).min(data.len());
        let last = end == data.len();
        let chunk = data.slice(offset..end);
        let command = if last { "upload, finalize" } else { "upload" };
        let response = send_chunk(endpoint, &upload_url, chunk, offset, command).await?;
        let status = header(&response, "x-goog-upload-status")
            .unwrap_or_default()
            .to_owned();
        if last {
            if status != "final" {
                return Err(upload_error(
                    &endpoint.provider,
                    format!("the upload status after the last chunk is {status:?}, not \"final\""),
                ));
            }
            let (_, body) =
                http::read_body(endpoint.flavor.api(), &endpoint.provider, response).await?;
            let value: Value = serde_json::from_slice(&body).map_err(|_| {
                upload_error(
                    &endpoint.provider,
                    "the upload response is not valid JSON".into(),
                )
            })?;
            return value
                .get("file")
                .filter(|file| file.is_object())
                .cloned()
                .ok_or_else(|| {
                    upload_error(&endpoint.provider, "the upload response has no file".into())
                });
        }
        if status != "active" {
            return Err(upload_error(
                &endpoint.provider,
                format!("the upload status at offset {offset} is {status:?}, not \"active\""),
            ));
        }
        drop(response);
        offset = end;
    }
}

/// Send one chunk, again after 1 s and then 2 s while its response lacks an
/// upload status: at most three sends.
async fn send_chunk(
    endpoint: &Endpoint,
    url: &str,
    chunk: Bytes,
    offset: usize,
    command: &str,
) -> Result<HttpResponse, Error> {
    let mut wait = FIRST_RESEND;
    for send in 1..=CHUNK_SENDS {
        let request = HttpRequest {
            method: "POST".into(),
            url: url.to_owned(),
            headers: vec![
                ("X-Goog-Upload-Offset".into(), offset.to_string()),
                ("X-Goog-Upload-Command".into(), command.into()),
            ],
            body: chunk.clone(),
        };
        let response = endpoint.send(request, None).await?;
        if header(&response, "x-goog-upload-status").is_some() {
            return Ok(response);
        }
        if send < CHUNK_SENDS {
            require_timer()?;
            tokio::time::sleep(wait).await;
            wait *= 2;
        }
    }
    Err(upload_error(
        &endpoint.provider,
        format!(
            "the upload of the chunk at offset {offset} got no upload status after {CHUNK_SENDS} sends"
        ),
    ))
}

fn positive_ms(
    options: &serde_json::Map<String, Value>,
    key: &str,
    default: u64,
) -> Result<Duration, Error> {
    match options.get(key).filter(|v| !v.is_null()) {
        None => Ok(Duration::from_millis(default)),
        Some(value) => value
            .as_f64()
            .filter(|ms| *ms > 0.0 && ms.is_finite())
            .map(|ms| Duration::from_secs_f64(ms / 1000.0))
            .ok_or_else(|| {
                Error::Validation(format!("google option {key} must be a positive number"))
            }),
    }
}

/// The path of a file resource name, such as `files/abc`.
fn file_path(name: &str) -> Vec<String> {
    match name.strip_prefix("files/").filter(|id| !id.contains('/')) {
        Some(id) => vec!["files".into(), id.into()],
        None => vec![name.to_owned()],
    }
}

/// Upload a file (GGL-020).
pub(crate) async fn upload(
    endpoint: &Endpoint,
    file: UploadFile,
    params: &UploadParams,
) -> Result<Item<UploadedFile>, Error> {
    let options = match params.provider_options.get("google") {
        Some(Value::Object(options)) => options.clone(),
        Some(Value::Null) | None => serde_json::Map::new(),
        Some(_) => {
            return Err(Error::Validation(
                "google provider options must be an object".into(),
            ));
        }
    };
    let display_name = match options.get("displayName").filter(|v| !v.is_null()) {
        Some(Value::String(name)) => Some(name.clone()),
        Some(_) => {
            return Err(Error::Validation(
                "google option displayName must be text".into(),
            ));
        }
        None => None,
    };
    let interval = positive_ms(&options, "pollIntervalMs", 2000)?;
    let timeout = positive_ms(&options, "pollTimeoutMs", 300_000)?;
    let mut warnings = Vec::new();
    if file.filename.is_some() {
        warnings.push(Warning {
            kind: "unsupported".into(),
            feature: Some("filename".into()),
            ..Default::default()
        });
    }
    let mut resource = resumable_upload(
        endpoint,
        file.data.clone(),
        &file.media_type,
        display_name.as_deref(),
    )
    .await?;
    let started = tokio::time::Instant::now();
    while resource.get("state").and_then(Value::as_str) == Some("PROCESSING") {
        if started.elapsed() > timeout {
            return Err(ProviderError::new(
                ProviderErrorKind::Timeout,
                format!("File processing timed out after {}ms", timeout.as_millis()),
            )
            .with_provider(&endpoint.provider)
            .into());
        }
        require_timer()?;
        tokio::time::sleep(interval).await;
        let name = resource
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        let request = endpoint.request(
            "GET",
            &file_path(&name),
            None,
            Default::default(),
            Default::default(),
        )?;
        let (_, value) = endpoint
            .json(request, None, http::MAX_RESPONSE_BYTES)
            .await?;
        resource = value;
    }
    let field = |key: &str| resource.get(key).and_then(Value::as_str).map(str::to_owned);
    if field("state").as_deref() == Some("FAILED") {
        return Err(upload_error(
            &endpoint.provider,
            format!(
                "File processing failed for {}",
                field("name").unwrap_or_default()
            ),
        ));
    }
    let uri = field("uri")
        .ok_or_else(|| upload_error(&endpoint.provider, "the uploaded file has no uri".into()))?;
    let mut metadata = serde_json::Map::new();
    for key in [
        "name",
        "displayName",
        "mimeType",
        "sizeBytes",
        "state",
        "uri",
        "createTime",
        "updateTime",
        "expirationTime",
        "sha256Hash",
    ] {
        if let Some(value) = resource.get(key).filter(|v| !v.is_null()) {
            metadata.insert(key.into(), value.clone());
        }
    }
    let mut item = Item::new(UploadedFile {
        provider_reference: [("google".to_owned(), uri)].into(),
        media_type: field("mimeType"),
        filename: None,
        size_bytes: field("sizeBytes").and_then(|size| size.parse().ok()),
        created_at: field("createTime"),
        expires_at: field("expirationTime"),
    });
    item.warnings = warnings;
    item.provider_metadata = Some([("google".to_owned(), Value::Object(metadata))].into());
    Ok(item)
}
