//! OpenAI's batches of Responses requests (`openai-batch.ts`), OAI-023.

use std::time::{Duration, UNIX_EPOCH};

use futures::{StreamExt, TryStreamExt, stream};
use serde::Deserialize;
use serde_json::{Value, json};

use super::{
    errors,
    files::{Files, path_id},
    params::invalid,
    profile,
    protocol::{Wire, http_request, wire_request},
    streaming,
    tools::CallTools,
};
use crate::{
    Error,
    providers::{
        google::{language::hydrate, realtime::iso_time},
        http,
        multipart::Multipart,
        operations::{
            BatchError, BatchOutcome, BatchPage, BatchParams, BatchRequest, BatchRequestCounts,
            BatchRequestKind, BatchResult, BatchResultKind, BatchResults, BatchStarted, BatchState,
            BatchStatus, BatchWarning, MAX_DOWNLOAD_BYTES, Warning,
        },
    },
    types::Message,
};

/// The endpoint every batch line names.
const ENDPOINT: &str = "/v1/responses";
/// How long the uploaded input file lives by default: 48 hours.
const DEFAULT_EXPIRY: u64 = 48 * 60 * 60;

/// The batch options (`openaiBatchProviderOptionsSchema`).
#[derive(Default, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct BatchOptions {
    input_file_expires_after: Option<u64>,
}

fn time(seconds: &Value) -> Option<String> {
    seconds
        .as_u64()
        .map(|seconds| iso_time(UNIX_EPOCH + Duration::from_secs(seconds)))
}

/// A batch object as a status (`convertOpenAIBatchStatus`).
fn status(batch: &Value, provider: &str) -> Result<BatchStatus, Error> {
    let id = batch["id"]
        .as_str()
        .ok_or_else(|| http::response_error(provider, "OpenAI batch response has no id"))?;
    let raw = batch["status"].as_str().unwrap_or_default();
    let state = match raw {
        "completed" => BatchState::Completed,
        "failed" | "expired" | "cancelled" => BatchState::Failed,
        _ => BatchState::Pending,
    };
    let counts = &batch["request_counts"];
    let request_counts = match (
        counts["total"].as_u64(),
        counts["completed"].as_u64(),
        counts["failed"].as_u64(),
    ) {
        (Some(total), Some(completed), Some(failed)) => total
            .checked_sub(completed)
            .and_then(|rest| rest.checked_sub(failed))
            .and_then(|pending| BatchRequestCounts::checked(total, pending, completed, failed)),
        _ => None,
    };
    let error = batch["errors"]["data"]
        .as_array()
        .and_then(|errors| errors.first())
        .map(|first| BatchError {
            message: first["message"]
                .as_str()
                .unwrap_or("OpenAI batch failed.")
                .to_owned(),
            error_type: None,
            code: first["code"].as_str().map(str::to_owned),
        });
    Ok(BatchStatus {
        batch_id: id.to_owned(),
        state,
        raw_status: Some(raw.to_owned()),
        request_counts,
        error,
        created_at: time(&batch["created_at"]),
        expires_at: time(&batch["expires_at"]),
        provider_metadata: None,
    })
}

/// The message a stored Responses result describes, read as its stream
/// would be: its output items replayed as stream events.
async fn message_of(response: &Value, provider: &str) -> Result<Message, Error> {
    let mut events = vec![json!({"type": "response.created", "response": {
        "id": response["id"], "model": response["model"], "status": "in_progress"}})];
    for (index, item) in response["output"]
        .as_array()
        .into_iter()
        .flatten()
        .enumerate()
    {
        let id = &item["id"];
        let mut added = item.clone();
        match item["type"].as_str().unwrap_or_default() {
            "message" => {
                added["content"] = json!([]);
                events.push(json!({"type": "response.output_item.added", "output_index": index, "item": added}));
                for part in item["content"].as_array().into_iter().flatten() {
                    if let Some(text) = part["text"].as_str() {
                        events.push(json!({"type": "response.output_text.delta", "item_id": id, "output_index": index, "delta": text}));
                    }
                    for annotation in part["annotations"].as_array().into_iter().flatten() {
                        events.push(json!({"type": "response.output_text.annotation.added", "item_id": id, "output_index": index, "annotation": annotation}));
                    }
                }
            }
            "reasoning" => {
                added["summary"] = json!([]);
                events.push(json!({"type": "response.output_item.added", "output_index": index, "item": added}));
                for (n, summary) in item["summary"].as_array().into_iter().flatten().enumerate() {
                    events.push(json!({"type": "response.reasoning_summary_part.added", "item_id": id, "output_index": index, "summary_index": n}));
                    events.push(json!({"type": "response.reasoning_summary_text.delta", "item_id": id, "output_index": index, "summary_index": n, "delta": summary["text"]}));
                    events.push(json!({"type": "response.reasoning_summary_part.done", "item_id": id, "output_index": index, "summary_index": n}));
                }
            }
            _ => {
                events.push(json!({"type": "response.output_item.added", "output_index": index, "item": added}));
            }
        }
        events.push(
            json!({"type": "response.output_item.done", "output_index": index, "item": item}),
        );
    }
    let terminal = if response["status"] == "incomplete" {
        "response.incomplete"
    } else {
        "response.completed"
    };
    events.push(json!({"type": terminal, "response": response}));
    let text: String = events.iter().map(|e| format!("data: {e}\n\n")).collect();
    let body = stream::iter([Ok(bytes::Bytes::from(text))]).boxed();
    let parsed: Vec<_> = streaming::parse(
        body,
        provider.to_owned(),
        Wire::Responses(&profile::OPENAI_RESPONSES),
        "image/png".into(),
        false,
        false,
        // A result body holds logprobs only when its request asked for them.
        true,
        CallTools::default(),
    )
    .try_collect()
    .await?;
    let (message, _) = hydrate(provider, parsed)?;
    Ok(message)
}

/// The error of a failed result line's body (`convertOpenAIErrorResponse`).
fn response_error(body: &Value, status: u64) -> BatchError {
    match body["error"]["message"].as_str() {
        Some(message) => BatchError {
            message: message.to_owned(),
            error_type: body["error"]["type"].as_str().map(str::to_owned),
            code: match &body["error"]["code"] {
                Value::Null => None,
                Value::String(code) => Some(code.clone()),
                other => Some(other.to_string()),
            },
        },
        None => BatchError {
            message: format!("OpenAI batch request failed with status code {status}."),
            error_type: None,
            code: None,
        },
    }
}

/// One result line (`convertResultLine`).
async fn result(line: Value, provider: String) -> Result<BatchResult, Error> {
    let id = line["custom_id"].as_str().unwrap_or_default().to_owned();
    let failed = |error: BatchError| BatchOutcome::Failed {
        error,
        provider_metadata: None,
    };
    let outcome = if let Some(message) = line["error"]["message"].as_str() {
        let error = BatchError {
            message: message.to_owned(),
            error_type: None,
            code: line["error"]["code"].as_str().map(str::to_owned),
        };
        match error.code.as_deref() {
            Some("batch_cancelled") => BatchOutcome::Cancelled { error: Some(error) },
            Some("batch_expired") => BatchOutcome::Expired { error: Some(error) },
            _ => failed(error),
        }
    } else if !line["response"].is_object() {
        failed(BatchError {
            message: "OpenAI returned a batch result without a response or error.".into(),
            error_type: None,
            code: Some("invalid_batch_result".into()),
        })
    } else {
        let status = line["response"]["status_code"].as_u64().unwrap_or(0);
        let body = &line["response"]["body"];
        if !(200..300).contains(&status) {
            failed(response_error(body, status))
        } else if let Some(message) = body["error"]["message"].as_str() {
            failed(BatchError {
                message: message.to_owned(),
                error_type: body["error"]["type"].as_str().map(str::to_owned),
                code: body["error"]["code"].as_str().map(str::to_owned),
            })
        } else {
            match message_of(body, &provider).await {
                Ok(message) => BatchOutcome::Message(message),
                Err(error) => failed(BatchError {
                    message: error.to_string(),
                    error_type: None,
                    code: Some("invalid_response".into()),
                }),
            }
        }
    };
    Ok(BatchResult {
        id,
        kind: BatchResultKind::Text,
        outcome,
    })
}

impl Files<'_> {
    async fn get(&self, path: &[&str], query: &[(&str, String)]) -> Result<Value, Error> {
        let query = query
            .iter()
            .map(|(key, value)| ((*key).to_owned(), json!(value)))
            .collect();
        let request = http::request(
            &errors::API,
            self.config,
            "GET",
            path,
            None,
            Default::default(),
            query,
        )?;
        http::json(&errors::API, self.transport, request, self.provider, None).await
    }

    pub(super) async fn batch_start(
        &self,
        requests: Vec<BatchRequest>,
        params: BatchParams,
    ) -> Result<BatchStarted, Error> {
        let options: BatchOptions = match params.provider_options.get("openai") {
            None => BatchOptions::default(),
            Some(value) => serde_json::from_value(value.clone())
                .map_err(|e| invalid(format!("invalid openai batch options: {e}")))?,
        };
        let expiry = options.input_file_expires_after.unwrap_or(DEFAULT_EXPIRY);
        if !(3600..=2_592_000).contains(&expiry) {
            return Err(invalid(
                "openai batch option inputFileExpiresAfter must be from 3600 to 2592000 seconds",
            ));
        }
        let model = requests
            .first()
            .map(|r| r.model_id.clone())
            .unwrap_or_default();
        let mut lines = String::new();
        for request in &requests {
            let BatchRequestKind::Text { messages, options } = &request.kind else {
                return Err(invalid("the OpenAI Batch API takes only text requests"));
            };
            if request.model_id != model {
                return Err(invalid(format!(
                    "the OpenAI Batch API takes one model per batch, not {model:?} and {:?}",
                    request.model_id
                )));
            }
            let http = wire_request(
                Wire::Responses(&profile::OPENAI_RESPONSES),
                &request.model_id,
                self.config,
                messages.clone(),
                options,
            )?;
            let mut body: Value = serde_json::from_slice(&http.body)?;
            if let Some(body) = body.as_object_mut() {
                body.remove("stream");
            }
            let line =
                json!({"custom_id": request.id, "method": "POST", "url": ENDPOINT, "body": body});
            lines.push_str(&line.to_string());
            lines.push('\n');
        }
        let mut warnings = Vec::new();
        if params.webhook_url.is_some() {
            warnings.push(BatchWarning {
                request_id: None,
                warning: Warning {
                    kind: "unsupported".into(),
                    feature: Some("webhookUrl".into()),
                    details: Some(
                        "The OpenAI Batch API does not support per-batch webhook URLs.".into(),
                    ),
                    ..Default::default()
                },
            });
        }
        let mut form = Multipart::new();
        form.file("file", "batch.jsonl", "application/jsonl", lines.as_bytes());
        form.text("purpose", "batch");
        form.text("expires_after[anchor]", "created_at");
        form.text("expires_after[seconds]", &expiry.to_string());
        let request = http_request(&errors::API, "POST", self.config, &["files"], None, None)?;
        let file = http::json(
            &errors::API,
            self.transport,
            form.into_request(request),
            self.provider,
            None,
        )
        .await?;
        let file_id = file["id"]
            .as_str()
            .ok_or_else(|| http::response_error(self.provider, "OpenAI file response has no id"))?;
        let request = http_request(
            &errors::API,
            "POST",
            self.config,
            &["batches"],
            Some(
                json!({"input_file_id": file_id, "endpoint": ENDPOINT, "completion_window": "24h"}),
            ),
            None,
        )?;
        let batch = http::json(&errors::API, self.transport, request, self.provider, None).await?;
        let mut status = status(&batch, self.provider)?;
        let mut metadata = json!({"inputFileId": file_id});
        if let Some(expires) = time(&file["expires_at"]) {
            metadata["inputFileExpiresAt"] = json!(expires);
        }
        status.provider_metadata = Some([("openai".to_owned(), metadata)].into());
        Ok(BatchStarted { status, warnings })
    }

    pub(super) async fn batch_status(&self, batch_id: &str) -> Result<BatchStatus, Error> {
        let batch_id = path_id("batch", batch_id)?;
        let batch = self.get(&["batches", batch_id], &[]).await?;
        status(&batch, self.provider)
    }

    pub(super) async fn batch_cancel(&self, batch_id: &str) -> Result<(), Error> {
        let batch_id = path_id("batch", batch_id)?;
        let request = http_request(
            &errors::API,
            "POST",
            self.config,
            &["batches", batch_id, "cancel"],
            Some(json!({})),
            None,
        )?;
        http::json(&errors::API, self.transport, request, self.provider, None).await?;
        Ok(())
    }

    pub(super) async fn batch_list(
        &self,
        page_size: Option<u32>,
        cursor: Option<String>,
    ) -> Result<BatchPage, Error> {
        let mut query = Vec::new();
        if let Some(limit) = page_size {
            query.push(("limit", limit.to_string()));
        }
        if let Some(cursor) = cursor {
            query.push(("after", cursor));
        }
        let page = self.get(&["batches"], &query).await?;
        let batches = page["data"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|batch| status(batch, self.provider))
            .collect::<Result<Vec<_>, _>>()?;
        let next_cursor = (page["has_more"] == true)
            .then(|| page["last_id"].as_str().map(str::to_owned))
            .flatten();
        Ok(BatchPage {
            batches,
            next_cursor,
        })
    }

    pub(super) async fn batch_results(&self, batch_id: &str) -> Result<BatchResults, Error> {
        let batch_id = path_id("batch", batch_id)?;
        let batch = self.get(&["batches", batch_id], &[]).await?;
        let state = status(&batch, self.provider)?.state;
        if state == BatchState::Pending {
            return Err(invalid(format!(
                "OpenAI batch \"{batch_id}\" is not complete."
            )));
        }
        let files: Vec<String> = ["output_file_id", "error_file_id"]
            .iter()
            .filter_map(|key| batch[*key].as_str().map(str::to_owned))
            .collect();
        if state == BatchState::Completed && files.is_empty() {
            return Err(http::response_error(
                self.provider,
                format!("OpenAI batch \"{batch_id}\" completed without batch output."),
            ));
        }
        let mut lines = Vec::new();
        for file in files {
            let content = self
                .download(&[("openai".to_owned(), file)].into(), MAX_DOWNLOAD_BYTES)
                .await?
                .value
                .data;
            for line in content.split(|byte| *byte == b'\n') {
                if line.iter().all(u8::is_ascii_whitespace) {
                    continue;
                }
                let value: Value = serde_json::from_slice(line).map_err(|_| {
                    http::response_error(self.provider, "OpenAI batch result line is not JSON")
                })?;
                lines.push(value);
            }
        }
        let provider = self.provider.to_owned();
        Ok(stream::iter(lines)
            .then(move |line| result(line, provider.clone()))
            .boxed())
    }
}
