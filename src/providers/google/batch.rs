//! Gemini API batches (`google-batch.ts`): requests inline up to 20,000,000
//! bytes, a JSON lines input file beyond, and results read inline or
//! streamed line by line from the results file.

use std::sync::Arc;

use bytes::Bytes;
use futures::{StreamExt, stream};
use serde_json::{Map, Value, json};

use super::{
    api::{self, Endpoint, Flavor},
    files, image,
    language::hydrate,
    messages::{self, Target},
    request,
    response::Reader,
};
use crate::{
    Error,
    models::StreamOptions,
    models::files::DownloadTransport,
    providers::{http, operations::*},
    types::{Message, messages::generate_id},
};

/// Below this many bytes a batch's requests go inline.
const INLINE_LIMIT: usize = 20_000_000;
/// The most bytes of a JSON lines input file.
const INPUT_FILE_LIMIT: usize = 2 * 1024 * 1024 * 1024;

fn invalid(message: impl Into<String>) -> Error {
    Error::Validation(message.into())
}

fn target<'a>(endpoint: &'a Endpoint, request_entry: &'a BatchRequest) -> Target<'a> {
    Target {
        model_id: &request_entry.model_id,
        flavor: Flavor::Gemini,
        base_url: &endpoint.base,
        batch: true,
    }
}

/// One request's messages, options and warnings.
fn parts(
    request_entry: &BatchRequest,
) -> Result<(Vec<Message>, StreamOptions, Vec<Warning>), Error> {
    match &request_entry.kind {
        BatchRequestKind::Text { messages, options } => {
            Ok((messages.clone(), options.clone(), Vec::new()))
        }
        BatchRequestKind::Image { prompt, params } => {
            if prompt.mask.is_some() {
                return Err(invalid(
                    "Google batches do not support mask-based image editing",
                ));
            }
            if params.n > 1 {
                return Err(invalid("Google batches take one image per image request"));
            }
            let built = image::request(Flavor::Gemini, &request_entry.model_id, prompt, params)?;
            Ok((built.messages, built.options, built.warnings))
        }
    }
}

/// Every check a request must pass, run with stand-ins for the files it
/// would download. The body is dropped so a large batch holds one body at a
/// time.
fn check(endpoint: &Endpoint, request_entry: &BatchRequest) -> Result<(), Error> {
    let target = target(endpoint, request_entry);
    let (history, options, _) = parts(request_entry)?;
    request::build(
        target,
        &messages::without_downloads(&history, target),
        &options,
        false,
    )?;
    Ok(())
}

/// One checked request's generate content body and warnings.
async fn prepare(
    endpoint: &Endpoint,
    downloads: Option<&dyn DownloadTransport>,
    request_entry: &BatchRequest,
) -> Result<(Map<String, Value>, Vec<Warning>), Error> {
    let target = target(endpoint, request_entry);
    let (history, options, warnings) = parts(request_entry)?;
    let history = messages::download_files(history, target, downloads).await?;
    let prepared = request::build(target, &history, &options, false)?;
    Ok((prepared.body, warnings))
}

/// Start a batch (GGL-021).
pub(crate) async fn start(
    endpoint: &Endpoint,
    downloads: Option<Arc<dyn DownloadTransport>>,
    requests: &[BatchRequest],
    params: &BatchParams,
) -> Result<BatchStarted, Error> {
    let model_id = requests
        .first()
        .map(|r| r.model_id.clone())
        .ok_or_else(|| invalid("Google batches require at least one request."))?;
    if requests.iter().any(|r| r.model_id != model_id) {
        return Err(invalid(
            "Google batches require every request to use the same model because the model is part of the batch endpoint.",
        ));
    }
    let display_name = format!("ai-sdk-batch-{}", generate_id(""));
    let mut batch = Map::new();
    batch.insert("displayName".into(), json!(display_name));
    if let Some(url) = &params.webhook_url {
        batch.insert("webhookConfig".into(), json!({"uris": [url]}));
    }
    let empty = {
        let mut body = batch.clone();
        body.insert("inputConfig".into(), json!({"requests": {"requests": []}}));
        serde_json::to_vec(&json!({"batch": body}))?.len()
    };
    let mut inline_bytes = empty;
    let mut inline: Vec<Value> = Vec::new();
    let mut lines: Option<Vec<u8>> = None;
    for entry in requests {
        check(endpoint, entry)?;
    }
    let mut warnings = Vec::new();
    for entry in requests {
        let (body, request_warnings) = prepare(endpoint, downloads.as_deref(), entry).await?;
        warnings.extend(request_warnings.into_iter().map(|warning| BatchWarning {
            request_id: Some(entry.id.clone()),
            warning,
        }));
        let item = json!({"request": body, "metadata": {"key": entry.id}});
        match &mut lines {
            None => {
                let size = serde_json::to_vec(&item)?.len();
                let next = inline_bytes + size + usize::from(!inline.is_empty());
                if next < INLINE_LIMIT {
                    inline.push(item);
                    inline_bytes = next;
                } else {
                    let mut file = Vec::new();
                    for previous in inline.drain(..).chain(std::iter::once(item)) {
                        let line = json!({"key": previous["metadata"]["key"], "request": previous["request"]});
                        serde_json::to_writer(&mut file, &line)?;
                        file.push(b'\n');
                    }
                    lines = Some(file);
                }
            }
            Some(file) => {
                serde_json::to_writer(&mut *file, &json!({"key": entry.id, "request": body}))?;
                file.push(b'\n');
                if file.len() > INPUT_FILE_LIMIT {
                    return Err(invalid("Google batch input files must not exceed 2 GB."));
                }
            }
        }
    }
    let path = api::model_segments(&model_id, "batchGenerateContent");
    let mut metadata = None;
    match lines {
        None => {
            batch.insert(
                "inputConfig".into(),
                json!({"requests": {"requests": inline}}),
            );
        }
        Some(file) => {
            if file.len() > INPUT_FILE_LIMIT {
                return Err(invalid("Google batch input files must not exceed 2 GB."));
            }
            let uploaded = files::resumable_upload(
                endpoint,
                Bytes::from(file),
                "application/jsonl",
                Some(&format!("{display_name}-input")),
            )
            .await?;
            let name = uploaded
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    http::response_error(&endpoint.provider, "the uploaded batch input has no name")
                })?;
            batch.insert("inputConfig".into(), json!({"fileName": name}));
            let mut google = json!({"inputFileId": name});
            if let Some(expiry) = uploaded.get("expirationTime").filter(|e| !e.is_null()) {
                google["inputFileExpiresAt"] = expiry.clone();
            }
            metadata = Some([("google".to_owned(), google)].into());
        }
    }
    let request = endpoint.request(
        "POST",
        &path,
        Some(json!({"batch": batch})),
        Default::default(),
        Default::default(),
    )?;
    let (_, operation) = endpoint
        .json(request, Some(&model_id), http::MAX_RESPONSE_BYTES)
        .await?;
    let mut status = status_of(&operation)?;
    status.provider_metadata = metadata;
    Ok(BatchStarted { status, warnings })
}

fn count(value: Option<&Value>) -> Option<u64> {
    match value? {
        Value::Number(number) => number.as_u64().filter(|n| *n <= 9_007_199_254_740_991),
        Value::String(text) if !text.is_empty() && text.bytes().all(|b| b.is_ascii_digit()) => {
            text.parse().ok().filter(|n| *n <= 9_007_199_254_740_991)
        }
        _ => None,
    }
}

fn rpc_error(error: &Value, fallback: &str) -> BatchError {
    BatchError {
        message: error
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or(fallback)
            .to_owned(),
        error_type: error
            .get("status")
            .and_then(Value::as_str)
            .map(str::to_owned),
        code: error
            .get("code")
            .filter(|c| !c.is_null())
            .map(|code| match code {
                Value::String(text) => text.clone(),
                other => other.to_string(),
            }),
    }
}

/// A batch operation's status (GGL-022).
pub(crate) fn status_of(operation: &Value) -> Result<BatchStatus, Error> {
    let batch_id = operation
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::Validation("a Google batch operation has no name".into()))?
        .to_owned();
    let metadata = operation.get("metadata").filter(|m| m.is_object());
    let raw = metadata
        .and_then(|m| m.get("state"))
        .and_then(Value::as_str);
    let error = operation
        .get("error")
        .filter(|e| !e.is_null())
        .map(|e| rpc_error(e, "Google batch failed."));
    let state = if error.is_some() {
        BatchState::Failed
    } else {
        match raw {
            None if operation.get("done") == Some(&Value::Bool(true)) => BatchState::Completed,
            None => BatchState::Pending,
            Some(raw) => {
                let state = raw
                    .strip_prefix("BATCH_STATE_")
                    .or_else(|| raw.strip_prefix("JOB_STATE_"))
                    .unwrap_or(raw);
                match state {
                    "SUCCEEDED" => BatchState::Completed,
                    "FAILED" | "CANCELLED" | "EXPIRED" => BatchState::Failed,
                    _ => BatchState::Pending,
                }
            }
        }
    };
    let stats = metadata.and_then(|m| m.get("batchStats"));
    let field = |key: &str| stats.and_then(|s| s.get(key)).filter(|v| !v.is_null());
    let request_counts = count(field("requestCount")).and_then(|total| {
        let or_zero = |key: &str| match field(key) {
            None => Some(0),
            some => count(some),
        };
        BatchRequestCounts::checked(
            total,
            or_zero("pendingRequestCount")?,
            or_zero("successfulRequestCount")?,
            or_zero("failedRequestCount")?,
        )
    });
    Ok(BatchStatus {
        batch_id,
        state,
        raw_status: raw.map(str::to_owned),
        request_counts,
        error,
        created_at: metadata
            .and_then(|m| m.get("createTime"))
            .and_then(Value::as_str)
            .map(str::to_owned),
        expires_at: None,
        provider_metadata: None,
    })
}

fn batch_path(batch_id: &str) -> Vec<String> {
    batch_id.split('/').map(str::to_owned).collect()
}

async fn retrieve(endpoint: &Endpoint, batch_id: &str) -> Result<Value, Error> {
    let request = endpoint.request(
        "GET",
        &batch_path(batch_id),
        None,
        Default::default(),
        Default::default(),
    )?;
    let (_, operation) = endpoint
        .json(request, None, http::MAX_RESPONSE_BYTES)
        .await?;
    Ok(operation)
}

pub(crate) async fn status(endpoint: &Endpoint, batch_id: &str) -> Result<BatchStatus, Error> {
    status_of(&retrieve(endpoint, batch_id).await?)
}

pub(crate) async fn cancel(endpoint: &Endpoint, batch_id: &str) -> Result<(), Error> {
    let mut path = batch_path(batch_id);
    if let Some(last) = path.last_mut() {
        last.push_str(":cancel");
    }
    let request = endpoint.request(
        "POST",
        &path,
        Some(json!({})),
        Default::default(),
        Default::default(),
    )?;
    endpoint
        .json(request, None, http::MAX_RESPONSE_BYTES)
        .await?;
    Ok(())
}

pub(crate) async fn list(
    endpoint: &Endpoint,
    page_size: Option<u32>,
    cursor: Option<String>,
) -> Result<BatchPage, Error> {
    let mut query = std::collections::BTreeMap::new();
    if let Some(size) = page_size {
        query.insert("pageSize".to_owned(), json!(size.to_string()));
    }
    if let Some(cursor) = cursor {
        query.insert("pageToken".to_owned(), json!(cursor));
    }
    let request = endpoint.request("GET", &["batches".into()], None, Default::default(), query)?;
    let (_, page) = endpoint
        .json(request, None, http::MAX_RESPONSE_BYTES)
        .await?;
    let batches = page
        .get("operations")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(status_of)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(BatchPage {
        batches,
        next_cursor: page
            .get("nextPageToken")
            .and_then(Value::as_str)
            .map(str::to_owned),
    })
}

fn failed(id: &str, message: &str, code: &str) -> BatchResult {
    BatchResult {
        id: id.to_owned(),
        kind: BatchResultKind::Text,
        outcome: BatchOutcome::Failed {
            error: BatchError {
                message: message.into(),
                error_type: None,
                code: Some(code.into()),
            },
            provider_metadata: None,
        },
    }
}

/// One results line or inline response as a result.
fn result(provider: &str, line: &Value) -> Result<BatchResult, Error> {
    let id = line
        .get("key")
        .and_then(Value::as_str)
        .ok_or_else(|| http::response_error(provider, "a Google batch result has no key"))?
        .to_owned();
    if let Some(error) = line.get("error").filter(|e| !e.is_null()) {
        let error_value = rpc_error(error, "Google batch request failed.");
        let cancelled = error.get("status").and_then(Value::as_str) == Some("CANCELLED")
            || error_value.code.as_deref() == Some("1");
        return Ok(BatchResult {
            id,
            kind: BatchResultKind::Text,
            outcome: if cancelled {
                BatchOutcome::Cancelled {
                    error: Some(error_value),
                }
            } else {
                BatchOutcome::Failed {
                    error: error_value,
                    provider_metadata: None,
                }
            },
        });
    }
    let Some(response) = line.get("response").filter(|r| !r.is_null()) else {
        return Ok(failed(
            &id,
            "Google returned a batch result without a response or error.",
            "invalid_batch_result",
        ));
    };
    let candidates = response.get("candidates").and_then(Value::as_array);
    if candidates.is_none_or(Vec::is_empty) {
        let feedback = response.get("promptFeedback").filter(|f| !f.is_null());
        let reason = feedback
            .and_then(|f| f.get("blockReason"))
            .and_then(Value::as_str);
        let error = BatchError {
            message: match reason {
                Some(reason) => format!("Google blocked the batch request ({reason})."),
                None => "Google returned a batch response without any candidates.".into(),
            },
            error_type: reason.map(str::to_owned),
            code: Some(
                if reason.is_some() {
                    "prompt_blocked"
                } else {
                    "invalid_response"
                }
                .into(),
            ),
        };
        return Ok(BatchResult {
            id,
            kind: BatchResultKind::Text,
            outcome: BatchOutcome::Failed {
                error,
                provider_metadata: feedback.map(|_| {
                    [(
                        "google".to_owned(),
                        json!({"promptFeedback": {"blockReason": reason}}),
                    )]
                    .into()
                }),
            },
        });
    }
    let mut reader = Reader::new(Flavor::Gemini, provider.to_owned(), None);
    let read = reader.chunk(response).map(|_| reader.end());
    let message = match read.and_then(|()| hydrate(provider, reader.events.drain(..))) {
        Ok((message, _)) => message,
        Err(_) => {
            return Ok(failed(
                &id,
                "Google returned an invalid GenerateContent batch result.",
                "invalid_response",
            ));
        }
    };
    let images: Vec<_> = message.images().into_iter().cloned().collect();
    if !images.is_empty() {
        let raw_finish = reader.raw_finish();
        let item = image::result(Flavor::Gemini, &message, raw_finish, Vec::new());
        return Ok(BatchResult {
            id,
            kind: BatchResultKind::Image,
            outcome: BatchOutcome::Images(item),
        });
    }
    if !message.files().is_empty() {
        return Ok(failed(
            &id,
            "Google returned a \"file\" content block, but that content is not supported in AI SDK text batches.",
            "unsupported_content",
        ));
    }
    Ok(BatchResult {
        id,
        kind: BatchResultKind::Text,
        outcome: BatchOutcome::Message(message),
    })
}

/// Stream a batch's results (GGL-022).
pub(crate) async fn results(endpoint: &Endpoint, batch_id: &str) -> Result<BatchResults, Error> {
    let operation = retrieve(endpoint, batch_id).await?;
    let status = status_of(&operation)?;
    if status.state == BatchState::Pending {
        return Err(invalid(format!(
            "Google batch \"{batch_id}\" is not complete."
        )));
    }
    let output = |key: &str| {
        operation
            .get("metadata")
            .and_then(|m| m.get("output"))
            .and_then(|o| o.get(key))
            .filter(|v| !v.is_null())
            .or_else(|| {
                operation
                    .get("response")
                    .and_then(|r| r.get(key))
                    .filter(|v| !v.is_null())
            })
            .cloned()
    };
    let provider = endpoint.provider.clone();
    if let Some(inlined) = output("inlinedResponses")
        .and_then(|v| v.get("inlinedResponses").and_then(Value::as_array).cloned())
    {
        let lines: Vec<Value> = inlined
            .into_iter()
            .map(|entry| {
                json!({
                    "key": entry["metadata"]["key"],
                    "response": entry.get("response").cloned().unwrap_or(Value::Null),
                    "error": entry.get("error").cloned().unwrap_or(Value::Null),
                })
            })
            .collect();
        return Ok(
            stream::iter(lines.into_iter().map(move |line| result(&provider, &line))).boxed(),
        );
    }
    let Some(file) = output("responsesFile").and_then(|v| v.as_str().map(str::to_owned)) else {
        if status.state == BatchState::Completed {
            return Err(http::response_error(
                &provider,
                format!("Google batch \"{batch_id}\" completed without batch output."),
            ));
        }
        return Ok(stream::empty().boxed());
    };
    let origin = endpoint.base.trim_end_matches('/');
    let origin = origin.strip_suffix("/v1beta").unwrap_or(origin);
    let mut path: Vec<String> = ["download", "v1beta"]
        .iter()
        .map(|s| (*s).to_owned())
        .collect();
    path.extend(file.split('/').map(str::to_owned));
    if let Some(last) = path.last_mut() {
        last.push_str(":download");
    }
    let request = endpoint.request_at(
        origin,
        "GET",
        &path,
        None,
        Default::default(),
        [("alt".to_owned(), json!("media"))].into(),
    )?;
    let response = endpoint.send(request, None).await?;
    let api = *endpoint.flavor.api();
    Ok(http::json_lines(
        api,
        provider.clone(),
        response.body,
        http::RESULT_LINE_LIMIT,
    )
    .map(move |line| line.and_then(|line| result(&provider, &line)))
    .boxed())
}
