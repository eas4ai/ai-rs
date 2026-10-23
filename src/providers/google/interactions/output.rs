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

//! An Interactions response as a message (`parse-google-interactions-outputs.ts`,
//! `extract-google-interactions-sources.ts`, `convert-google-interactions-usage.ts`
//! and `map-google-interactions-finish-reason.ts`).

use std::collections::HashSet;

use serde_json::{Map, Value, json};

use crate::{
    providers::google::transcription::interactions_usage,
    types::{
        Message, MessageRole, Part,
        events::{Event, StreamEnd, replay_message_events},
        messages::{
            BuiltinToolCallPart, BuiltinToolReturnPart, FileData, FilePart, Metadata,
            ReasoningPart, TextPart, ToolCallPart, generate_id,
        },
    },
};

/// The terminal statuses of an interaction.
pub(crate) fn is_terminal(status: Option<&str>) -> bool {
    matches!(
        status,
        Some("completed" | "failed" | "cancelled" | "incomplete")
    )
}

/// A status as the crate's normalized finish reason.
pub(crate) fn finish_reason(status: Option<&str>, has_function_call: bool) -> &'static str {
    match status {
        Some("completed") if has_function_call => "tool_call",
        Some("completed") => "stop",
        Some("requires_action") => "tool_call",
        Some("failed") => "error",
        Some("incomplete") => "other",
        _ => "other",
    }
}

/// The output tokens by modality of a usage object, or `None` without any.
pub(crate) fn output_tokens_by_modality(usage: Option<&Value>) -> Option<Value> {
    let mut out = Map::new();
    for entry in usage?.get("output_tokens_by_modality")?.as_array()? {
        if let (Some(modality), Some(tokens)) = (
            entry.get("modality").and_then(Value::as_str),
            entry.get("tokens").filter(|t| t.is_number()),
        ) {
            out.insert(modality.into(), tokens.clone());
        }
    }
    (!out.is_empty()).then_some(Value::Object(out))
}

/// Sources once each: by URL, or by a document's file name or title.
#[derive(Default)]
pub(crate) struct Sources {
    pub list: Vec<Value>,
    seen: HashSet<String>,
}

impl Sources {
    pub(crate) fn add(&mut self, source: Value) {
        let key = match source.get("url").and_then(Value::as_str) {
            Some(url) => format!("url:{url}"),
            None => format!(
                "doc:{}",
                source
                    .get("filename")
                    .or_else(|| source.get("title"))
                    .and_then(Value::as_str)
                    .unwrap_or_default()
            ),
        };
        if self.seen.insert(key) {
            self.list.push(source);
        }
    }

    pub(crate) fn annotations(&mut self, annotations: Option<&Value>) {
        for annotation in annotations.and_then(Value::as_array).into_iter().flatten() {
            if let Some(source) = annotation_source(annotation) {
                self.add(source);
            }
        }
    }

    pub(crate) fn tool_result(&mut self, kind: &str, result: Option<&Value>) {
        for source in result_sources(kind, result) {
            self.add(source);
        }
    }
}

fn url_source(url: &str, title: Option<&str>) -> Value {
    let mut source = json!({"sourceType": "url", "id": generate_id("source"), "url": url});
    if let Some(title) = title {
        source["title"] = json!(title);
    }
    source
}

fn doc_media_type(uri: &str) -> &'static str {
    let lower = uri.to_ascii_lowercase();
    for (extension, media) in [
        ("pdf", "application/pdf"),
        ("txt", "text/plain"),
        ("md", "text/markdown"),
        ("markdown", "text/markdown"),
        ("doc", "application/msword"),
        (
            "docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ),
    ] {
        if lower.ends_with(&format!(".{extension}")) {
            return media;
        }
    }
    "application/octet-stream"
}

fn basename(uri: &str) -> Option<&str> {
    uri.rsplit('/').next().filter(|name| !name.is_empty())
}

fn document_source(uri: &str, title: Option<&str>, file_name: Option<&str>) -> Value {
    let filename = file_name.or_else(|| basename(uri));
    let mut source = json!({
        "sourceType": "document",
        "id": generate_id("source"),
        "mediaType": doc_media_type(uri),
        "title": title.or(file_name).or(filename).unwrap_or(uri),
    });
    if let Some(filename) = filename {
        source["filename"] = json!(filename);
    }
    source
}

fn text<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
}

/// A text annotation as a source.
pub(crate) fn annotation_source(annotation: &Value) -> Option<Value> {
    match annotation.get("type").and_then(Value::as_str)? {
        "url_citation" => Some(url_source(
            text(annotation, "url")?,
            annotation.get("title").and_then(Value::as_str),
        )),
        "file_citation" => {
            let uri = text(annotation, "url")
                .or_else(|| text(annotation, "document_uri"))
                .or_else(|| text(annotation, "file_name"))?;
            let file_name = annotation.get("file_name").and_then(Value::as_str);
            if uri.starts_with("http://") || uri.starts_with("https://") {
                Some(url_source(uri, file_name))
            } else {
                Some(document_source(uri, None, file_name))
            }
        }
        "place_citation" => Some(url_source(
            text(annotation, "url")?,
            annotation.get("name").and_then(Value::as_str),
        )),
        _ => None,
    }
}

/// The sources of a built-in tool result.
pub(crate) fn result_sources(kind: &str, result: Option<&Value>) -> Vec<Value> {
    let entries = result
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut sources = Vec::new();
    match kind {
        "url_context_result" => {
            for entry in &entries {
                if let Some(url) = text(entry, "url")
                    && entry
                        .get("status")
                        .and_then(Value::as_str)
                        .is_none_or(|s| s == "success")
                {
                    sources.push(url_source(url, None));
                }
            }
        }
        "google_search_result" => {
            for entry in &entries {
                if let Some(url) = text(entry, "url") {
                    sources.push(url_source(url, entry.get("title").and_then(Value::as_str)));
                }
            }
        }
        "google_maps_result" => {
            for entry in &entries {
                for place in entry
                    .get("places")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    if let Some(url) = text(place, "url") {
                        sources.push(url_source(url, place.get("name").and_then(Value::as_str)));
                    }
                }
            }
        }
        "file_search_result" => {
            for entry in entries.iter().filter(|e| e.is_object()) {
                let Some(uri) = text(entry, "url")
                    .or_else(|| text(entry, "document_uri"))
                    .or_else(|| text(entry, "file_name"))
                else {
                    continue;
                };
                let title = entry.get("title").and_then(Value::as_str);
                if uri.starts_with("http://") || uri.starts_with("https://") {
                    sources.push(url_source(uri, title));
                } else {
                    sources.push(document_source(
                        uri,
                        title,
                        entry.get("file_name").and_then(Value::as_str),
                    ));
                }
            }
        }
        _ => {}
    }
    sources
}

/// The built-in tool name of a `*_call` or `*_result` step type.
pub(crate) fn builtin_name(kind: &str, name: Option<&str>) -> Option<String> {
    if kind == "mcp_server_tool_call" || kind == "mcp_server_tool_result" {
        return Some(name.unwrap_or("mcp_server_tool").to_owned());
    }
    match kind {
        "google_search_call"
        | "code_execution_call"
        | "url_context_call"
        | "file_search_call"
        | "google_maps_call" => kind.strip_suffix("_call").map(str::to_owned),
        "google_search_result"
        | "code_execution_result"
        | "url_context_result"
        | "file_search_result"
        | "google_maps_result" => kind.strip_suffix("_result").map(str::to_owned),
        _ => None,
    }
}

/// Part metadata under `google`: a signature, the interaction id and the
/// processing ids, each when present.
pub(crate) fn part_metadata(fields: &[(&str, Option<&str>)]) -> Option<Metadata> {
    let mut google = Map::new();
    for (key, value) in fields {
        if let Some(value) = value.filter(|v| !v.is_empty()) {
            google.insert((*key).into(), json!(value));
        }
    }
    (!google.is_empty()).then(|| [("google".to_owned(), Value::Object(google))].into())
}

pub(crate) fn file(
    media_type: &str,
    data: Option<&str>,
    uri: Option<&str>,
    interaction: Option<&str>,
) -> Option<FilePart> {
    let data = match (
        data.filter(|d| !d.is_empty()),
        uri.filter(|u| !u.is_empty()),
    ) {
        (Some(data), _) => FileData::Text(data.to_owned()),
        (None, Some(uri)) => FileData::Text(uri.to_owned()),
        (None, None) => return None,
    };
    Some(FilePart {
        id: generate_id("part"),
        data,
        media_type: media_type.to_owned(),
        filename: None,
        provider_metadata: part_metadata(&[("interactionId", interaction)]),
    })
}

/// A parsed response: the message and whether it holds a function call.
pub(crate) struct Parsed {
    pub message: Message,
    pub finish: &'static str,
}

/// Parse a response (GGL-028). `header_tier` is the
/// `x-gemini-service-tier` header, used when the body has no tier.
pub(crate) fn parse(response: &Value, header_tier: Option<&str>) -> Parsed {
    let interaction = text(response, "id");
    let mut parts = Vec::new();
    let mut sources = Sources::default();
    let mut has_function_call = false;
    for step in response
        .get("steps")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let Some(kind) = step.get("type").and_then(Value::as_str) else {
            continue;
        };
        match kind {
            "user_input" => {}
            "model_output" => {
                for block in step
                    .get("content")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    match block.get("type").and_then(Value::as_str) {
                        Some("text") => {
                            parts.push(Part::Text(TextPart {
                                id: generate_id("part"),
                                text: block
                                    .get("text")
                                    .and_then(Value::as_str)
                                    .unwrap_or_default()
                                    .to_owned(),
                                provider_metadata: part_metadata(&[("interactionId", interaction)]),
                            }));
                            sources.annotations(block.get("annotations"));
                        }
                        Some(media @ ("image" | "video")) => {
                            let default = if media == "image" {
                                "image/png"
                            } else {
                                "video/mp4"
                            };
                            let mime = block
                                .get("mime_type")
                                .and_then(Value::as_str)
                                .unwrap_or(default);
                            if let Some(part) =
                                file(mime, text(block, "data"), text(block, "uri"), interaction)
                            {
                                parts.push(Part::File(part));
                            }
                        }
                        _ => {}
                    }
                }
            }
            "thought" => {
                let summary: Vec<&str> = step
                    .get("summary")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter(|item| item.get("type").and_then(Value::as_str) == Some("text"))
                    .filter_map(|item| item.get("text").and_then(Value::as_str))
                    .collect();
                parts.push(Part::Reasoning(ReasoningPart {
                    id: generate_id("part"),
                    text: summary.join("\n"),
                    provider_metadata: part_metadata(&[
                        ("signature", step.get("signature").and_then(Value::as_str)),
                        ("interactionId", interaction),
                    ]),
                }));
            }
            "processing_call" => {
                let id = text(step, "id").map_or_else(|| generate_id("call"), str::to_owned);
                parts.push(Part::BuiltinToolCall(BuiltinToolCallPart {
                    id: generate_id("part"),
                    tool_call_id: id.clone(),
                    tool_name: "processing".into(),
                    tool_args: "{}".into(),
                    provider_metadata: part_metadata(&[
                        ("signature", step.get("signature").and_then(Value::as_str)),
                        ("interactionId", interaction),
                        ("processingId", Some(&id)),
                    ]),
                }));
            }
            "processing_result" => {
                let id = text(step, "call_id").map_or_else(|| generate_id("call"), str::to_owned);
                parts.push(Part::BuiltinToolReturn(BuiltinToolReturnPart {
                    id: generate_id("part"),
                    tool_call_id: id.clone(),
                    tool_name: "processing".into(),
                    result: Value::Null,
                    is_error: false,
                    provider_metadata: part_metadata(&[
                        ("signature", step.get("signature").and_then(Value::as_str)),
                        ("interactionId", interaction),
                        ("processingCallId", Some(&id)),
                    ]),
                }));
            }
            "function_call" => {
                has_function_call = true;
                let args = step
                    .get("arguments")
                    .filter(|a| !a.is_null())
                    .cloned()
                    .unwrap_or_else(|| json!({}));
                let mut call = ToolCallPart::new(
                    step.get("id").and_then(Value::as_str).unwrap_or_default(),
                    step.get("name").and_then(Value::as_str).unwrap_or_default(),
                    args.to_string(),
                );
                call.provider_metadata = part_metadata(&[
                    ("signature", step.get("signature").and_then(Value::as_str)),
                    ("interactionId", interaction),
                ]);
                parts.push(Part::ToolCall(call));
            }
            other => {
                let name = step.get("name").and_then(Value::as_str);
                let Some(tool) = builtin_name(other, name) else {
                    continue;
                };
                if other.ends_with("_call") {
                    let args = step
                        .get("arguments")
                        .filter(|a| !a.is_null())
                        .cloned()
                        .unwrap_or_else(|| json!({}));
                    parts.push(Part::BuiltinToolCall(BuiltinToolCallPart {
                        id: generate_id("part"),
                        tool_call_id: text(step, "id")
                            .map_or_else(|| generate_id("call"), str::to_owned),
                        tool_name: tool,
                        tool_args: args.to_string(),
                        provider_metadata: part_metadata(&[("interactionId", interaction)]),
                    }));
                } else {
                    let result = step.get("result").cloned().unwrap_or(Value::Null);
                    sources.tool_result(other, step.get("result"));
                    parts.push(Part::BuiltinToolReturn(BuiltinToolReturnPart {
                        id: generate_id("part"),
                        tool_call_id: text(step, "call_id")
                            .map_or_else(|| generate_id("call"), str::to_owned),
                        tool_name: tool,
                        result,
                        is_error: step.get("is_error") == Some(&Value::Bool(true)),
                        provider_metadata: part_metadata(&[("interactionId", interaction)]),
                    }));
                }
            }
        }
    }
    let status = response.get("status").and_then(Value::as_str);
    let usage = response.get("usage").filter(|u| u.is_object());
    let mut message = Message::new(MessageRole::Assistant, parts);
    message.usage = usage.map(interactions_usage);
    message.provider_metadata = Some(response_metadata(
        interaction,
        response
            .get("service_tier")
            .and_then(Value::as_str)
            .or(header_tier),
        usage,
        &sources.list,
    ));
    Parsed {
        message,
        finish: finish_reason(status, has_function_call),
    }
}

/// The final metadata under `google`.
pub(crate) fn response_metadata(
    interaction: Option<&str>,
    service_tier: Option<&str>,
    usage: Option<&Value>,
    sources: &[Value],
) -> Metadata {
    let mut google = Map::new();
    if let Some(id) = interaction {
        google.insert("interactionId".into(), json!(id));
    }
    if let Some(tier) = service_tier {
        google.insert("serviceTier".into(), json!(tier));
    }
    if let Some(by_modality) = output_tokens_by_modality(usage) {
        google.insert("outputTokensByModality".into(), by_modality);
    }
    if !sources.is_empty() {
        google.insert("sources".into(), json!(sources));
    }
    [("google".to_owned(), Value::Object(google))].into()
}

/// A parsed response as stream events, for a background response that was
/// already terminal.
pub(crate) fn events(parsed: Parsed) -> Vec<Event> {
    let usage = parsed.message.usage.clone();
    let mut events = replay_message_events(&parsed.message);
    if let Some(Event::StreamEnd(end)) = events.last_mut() {
        *end = StreamEnd {
            event: end.event.clone(),
            finish_reason: Some(parsed.finish.into()),
            response_id: parsed
                .message
                .provider_metadata
                .as_ref()
                .and_then(|m| m.get("google"))
                .and_then(|g| g.get("interactionId"))
                .and_then(Value::as_str)
                .map(str::to_owned),
            response_model: None,
        };
        end.event.usage = usage;
    }
    events
}
