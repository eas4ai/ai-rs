//! Anthropic's Message Batches API (`anthropic-batch.ts`, ANT-019).
use super::{
    errors::API,
    files::{Files, path_id},
    protocol::{ANTHROPIC_OWN, authorize, request_parts, with_betas},
    stream::{Reading, parse_message},
};
use crate::{
    Error,
    providers::{
        http,
        operations::{
            BatchError, BatchOutcome, BatchPage, BatchParams, BatchRequest, BatchRequestCounts,
            BatchRequestKind, BatchResult, BatchResultKind, BatchResults, BatchStarted, BatchState,
            BatchStatus, BatchWarning, Warning,
        },
    },
    transport::HttpTransport,
};
use futures::{StreamExt, TryStreamExt};
use serde_json::{Value, json};
use std::{collections::BTreeMap, sync::Arc};

fn invalid(message: impl Into<String>) -> Error {
    Error::Validation(message.into())
}

/// The batch's own `anthropic` option, `anthropicBeta`.
fn batch_betas(options: &crate::types::messages::Metadata) -> Result<Vec<String>, Error> {
    let Some(value) = options.get("anthropic").filter(|v| !v.is_null()) else {
        return Ok(Vec::new());
    };
    let fields = value
        .as_object()
        .ok_or_else(|| invalid("the anthropic batch options must be an object"))?;
    if let Some(key) = fields.keys().find(|key| *key != "anthropicBeta") {
        return Err(invalid(format!(
            "unknown anthropic batch option {key:?}; it takes anthropicBeta"
        )));
    }
    match fields.get("anthropicBeta") {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::Array(betas)) => betas
            .iter()
            .map(|b| {
                b.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| invalid("anthropicBeta must be a list of strings"))
            })
            .collect(),
        Some(_) => Err(invalid("anthropicBeta must be a list of strings")),
    }
}

/// Whether a request id is 1 to 64 letters, digits, `_` or `-`.
fn valid_id(id: &str) -> bool {
    (1..=64).contains(&id.len())
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

impl Files<'_> {
    pub(super) async fn batch_start(
        &self,
        requests: Vec<BatchRequest>,
        params: BatchParams,
    ) -> Result<BatchStarted, Error> {
        let explicit = batch_betas(&params.provider_options)?;
        let mut betas = explicit.clone();
        let mut ids = std::collections::BTreeSet::new();
        let mut warnings = Vec::new();
        if params.webhook_url.is_some() {
            warnings.push(BatchWarning {
                request_id: None,
                warning: Warning {
                    kind: "unsupported".into(),
                    feature: Some("webhookUrl".into()),
                    details: Some(
                        "The Anthropic Message Batches API does not support completion webhooks."
                            .into(),
                    ),
                    ..Default::default()
                },
            });
        }
        let mut lines = Vec::new();
        for request in &requests {
            let BatchRequestKind::Text { messages, options } = &request.kind else {
                return Err(invalid(
                    "The Anthropic Message Batches API takes only text requests.",
                ));
            };
            if !valid_id(&request.id) {
                return Err(invalid(format!(
                    "Anthropic batch request ID {:?} must match ^[A-Za-z0-9_-]{{1,64}}$.",
                    request.id
                )));
            }
            if !ids.insert(request.id.clone()) {
                return Err(invalid(format!(
                    "Anthropic batch request IDs must be unique; duplicate ID {:?}.",
                    request.id
                )));
            }
            let per_request = options
                .params
                .as_ref()
                .and_then(|p| p.provider_params.as_ref())
                .and_then(|p| p.get("anthropic"))
                .and_then(|o| o.get("anthropicBeta"))
                .is_some_and(|b| !b.is_null());
            if per_request {
                return Err(invalid(format!(
                    "Anthropic Message Batches take no per-request betas (request {:?}); set the batch's anthropicBeta option instead.",
                    request.id
                )));
            }
            let parts = request_parts(&ANTHROPIC_OWN, &request.model_id, messages, options, false)?;
            if parts.reading.json_tool {
                return Err(invalid(format!(
                    "Anthropic Message Batches cannot read the JSON tool answer of request {:?}; use a model with native structured output.",
                    request.id
                )));
            }
            if let Some(name) = parts.reading.names.aliased() {
                return Err(invalid(format!(
                    "Anthropic Message Batches cannot give back the provider tool name {name:?} of request {:?}; use the tool's Anthropic name.",
                    request.id
                )));
            }
            if parts.body.get("speed").is_some_and(|s| !s.is_null()) {
                return Err(invalid(format!(
                    "Anthropic Message Batches take no speed (request {:?}).",
                    request.id
                )));
            }
            let fallback_speed = parts
                .body
                .get("fallbacks")
                .and_then(Value::as_array)
                .is_some_and(|list| {
                    list.iter()
                        .any(|f| f.get("speed").is_some_and(|s| !s.is_null()))
                });
            if fallback_speed {
                return Err(invalid(format!(
                    "Anthropic Message Batches take no fallback speed (request {:?}).",
                    request.id
                )));
            }
            super::params::add_betas(&mut betas, parts.betas.iter().map(String::as_str));
            for (setting, details) in &parts.changes {
                warnings.push(BatchWarning {
                    request_id: Some(request.id.clone()),
                    warning: Warning {
                        kind: "unsupported".into(),
                        feature: Some(setting.clone()),
                        details: Some(details.clone()),
                        ..Default::default()
                    },
                });
            }
            lines.push(json!({"custom_id": request.id, "params": parts.body}));
        }
        let headers = with_betas(self.config, &BTreeMap::new(), &betas);
        let mut request = http::request(
            &API,
            self.config,
            "POST",
            &["v1", "messages", "batches"],
            Some(json!({"requests": lines})),
            headers,
            BTreeMap::new(),
        )?;
        authorize(&ANTHROPIC_OWN, self.config, &mut request)?;
        let batch = self.json(request).await?;
        Ok(BatchStarted {
            status: self.status(&batch)?,
            warnings,
        })
    }

    /// A batch's status (`convertAnthropicBatchStatus`).
    fn status(&self, batch: &Value) -> Result<BatchStatus, Error> {
        let id = batch.get("id").and_then(Value::as_str).ok_or_else(|| {
            http::response_error(self.provider, "Anthropic batch response has no id")
        })?;
        let raw = batch["processing_status"].as_str().map(str::to_owned);
        let counts = &batch["request_counts"];
        let count = |key: &str| counts[key].as_u64().unwrap_or(0);
        let request_counts = BatchRequestCounts::checked(
            ["processing", "succeeded", "errored", "canceled", "expired"]
                .iter()
                .map(|key| count(key))
                .fold(0u64, u64::saturating_add),
            count("processing"),
            count("succeeded"),
            count("errored")
                .saturating_add(count("canceled"))
                .saturating_add(count("expired")),
        );
        let field = |key: &str| batch.get(key).cloned().unwrap_or(Value::Null);
        let metadata = json!({
            "archivedAt": field("archived_at"),
            "cancelInitiatedAt": field("cancel_initiated_at"),
            "endedAt": field("ended_at"),
            "requestCounts": field("request_counts"),
            "resultsUrl": field("results_url"),
        });
        Ok(BatchStatus {
            batch_id: id.to_owned(),
            state: if raw.as_deref() == Some("ended") {
                BatchState::Completed
            } else {
                BatchState::Pending
            },
            raw_status: raw,
            request_counts,
            error: None,
            created_at: batch["created_at"].as_str().map(str::to_owned),
            expires_at: batch["expires_at"].as_str().map(str::to_owned),
            provider_metadata: Some([("anthropic".to_owned(), metadata)].into()),
        })
    }

    async fn retrieve(&self, batch_id: &str) -> Result<Value, Error> {
        let batch_id = path_id("batch", batch_id)?;
        let request = self.request(
            "GET",
            &["v1", "messages", "batches", batch_id],
            None,
            &[],
            BTreeMap::new(),
        )?;
        self.json(request).await
    }

    pub(super) async fn batch_status(&self, batch_id: &str) -> Result<BatchStatus, Error> {
        let batch = self.retrieve(batch_id).await?;
        self.status(&batch)
    }

    pub(super) async fn batch_cancel(&self, batch_id: &str) -> Result<(), Error> {
        let batch_id = path_id("batch", batch_id)?;
        let request = self.request(
            "POST",
            &["v1", "messages", "batches", batch_id, "cancel"],
            Some(json!({})),
            &[],
            BTreeMap::new(),
        )?;
        self.json(request).await?;
        Ok(())
    }

    pub(super) async fn batch_list(
        &self,
        page_size: Option<u32>,
        cursor: Option<String>,
    ) -> Result<BatchPage, Error> {
        let mut query = BTreeMap::new();
        if let Some(limit) = page_size {
            query.insert("limit".to_owned(), json!(limit));
        }
        if let Some(cursor) = cursor {
            query.insert("after_id".to_owned(), json!(cursor));
        }
        let request = self.request("GET", &["v1", "messages", "batches"], None, &[], query)?;
        let page = self.json(request).await?;
        let batches = page["data"]
            .as_array()
            .ok_or_else(|| http::response_error(self.provider, "Anthropic batch list has no data"))?
            .iter()
            .map(|batch| self.status(batch))
            .collect::<Result<Vec<_>, _>>()?;
        let next_cursor = (page["has_more"] == true)
            .then(|| page["last_id"].as_str().map(str::to_owned))
            .flatten();
        Ok(BatchPage {
            batches,
            next_cursor,
        })
    }

    /// The results of a finished batch, each line read as it arrives.
    /// `transport` is this one's, owned, so the stream outlives the call.
    pub(super) async fn batch_results(
        &self,
        batch_id: &str,
        transport: Arc<dyn HttpTransport>,
    ) -> Result<BatchResults, Error> {
        let batch = self.retrieve(batch_id).await?;
        if self.status(&batch)?.state == BatchState::Pending {
            return Err(invalid(format!(
                "Anthropic batch {batch_id:?} is not complete."
            )));
        }
        if !batch["archived_at"].is_null() {
            return Err(invalid(format!(
                "Anthropic batch {batch_id:?} results are no longer available."
            )));
        }
        let Some(url) = batch["results_url"].as_str() else {
            return Err(http::response_error(
                self.provider,
                format!("Anthropic batch {batch_id:?} completed without batch output."),
            ));
        };
        // The key goes only to the base URL's origin.
        let origin = |text: &str| url::Url::parse(text).ok().map(|u| u.origin());
        let same = origin(url)
            .zip(origin(&self.config.base_url()))
            .is_some_and(|(a, b)| a == b && a.is_tuple());
        if !same {
            return Err(http::response_error(
                self.provider,
                format!(
                    "Anthropic batch {batch_id:?} results are on another origin than the base URL"
                ),
            ));
        }
        let headers = with_betas(self.config, &BTreeMap::new(), &[]);
        let mut request = http::request_to(
            &API,
            url,
            self.config,
            "GET",
            &[],
            None,
            headers,
            BTreeMap::new(),
        )?;
        authorize(&ANTHROPIC_OWN, self.config, &mut request)?;
        request
            .headers
            .retain(|(name, _)| !name.eq_ignore_ascii_case("accept"));
        request.headers.push(("accept".into(), "*/*".into()));
        let response = http::send(&API, transport.as_ref(), request, self.provider, None).await?;
        let provider = self.provider.to_owned();
        Ok(http::json_lines(
            API,
            provider.clone(),
            response.body,
            http::RESULT_LINE_LIMIT,
        )
        .and_then(move |line| {
            let provider = provider.clone();
            async move { result(line, provider).await }
        })
        .boxed())
    }
}

/// A result that is not a valid Message batch result.
fn unreadable(id: String) -> BatchResult {
    BatchResult {
        id,
        kind: BatchResultKind::Text,
        outcome: BatchOutcome::Failed {
            error: BatchError {
                message: "Anthropic returned an invalid Message batch result.".into(),
                error_type: None,
                code: Some("invalid_response".into()),
            },
            provider_metadata: None,
        },
    }
}

/// One result line (`convertAnthropicBatchResult`).
async fn result(line: Value, provider: String) -> Result<BatchResult, Error> {
    let id = line
        .get("custom_id")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            http::response_error(&provider, "an Anthropic batch result line has no custom_id")
        })?
        .to_owned();
    let result = &line["result"];
    let outcome = match result["type"].as_str() {
        Some("canceled") => BatchOutcome::Cancelled { error: None },
        Some("expired") => BatchOutcome::Expired { error: None },
        Some("errored") => {
            let error = &result["error"];
            let (Some(message), Some(kind)) = (
                error["error"]["message"].as_str(),
                error["error"]["type"].as_str(),
            ) else {
                return Ok(unreadable(id));
            };
            BatchOutcome::Failed {
                error: BatchError {
                    message: message.to_owned(),
                    error_type: Some(kind.to_owned()),
                    code: None,
                },
                provider_metadata: error["request_id"].as_str().map(|request| {
                    [("anthropic".to_owned(), json!({"requestId": request}))].into()
                }),
            }
        }
        Some("succeeded") if result["message"].is_object() => {
            let events: Result<Vec<_>, Error> = parse_message(
                &result["message"],
                provider.clone(),
                &API,
                Reading::default(),
            )
            .try_collect()
            .await;
            match events
                .and_then(|events| crate::providers::google::language::hydrate(&provider, events))
            {
                Ok((message, _)) => BatchOutcome::Message(message),
                Err(_) => return Ok(unreadable(id)),
            }
        }
        _ => return Ok(unreadable(id)),
    };
    Ok(BatchResult {
        id,
        kind: BatchResultKind::Text,
        outcome,
    })
}
