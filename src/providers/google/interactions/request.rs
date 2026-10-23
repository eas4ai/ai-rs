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

//! The Interactions request body (`google-interactions-language-model.ts`,
//! `prepare-google-interactions-tools.ts` and
//! `convert-to-google-interactions-input.ts`).

use std::collections::HashSet;

use serde::Deserialize;
use serde_json::{Map, Value, json};

use super::super::{
    api::Flavor,
    options::{self, ASPECT_RATIOS},
    tools::builtin_name,
};
use crate::{
    Error, StreamOptions,
    models::params::*,
    types::{
        Message, MessageRole, Part,
        messages::{
            ContentPart, FileData, FilePart, Metadata, SpecialToolResult, ToolResultPart,
            ToolResultValue,
        },
        tools::{Tool, ToolChoice, ToolKind},
    },
};

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct InteractionsOptions {
    pub previous_interaction_id: Option<String>,
    pub store: Option<bool>,
    /// Accepted as TypeScript accepts it; the protocol names the agent.
    #[allow(dead_code)]
    pub agent: Option<String>,
    pub agent_config: Option<Value>,
    pub thinking_level: Option<String>,
    pub thinking_summaries: Option<String>,
    pub response_format: Option<Vec<Value>>,
    pub image_config: Option<Value>,
    pub media_resolution: Option<String>,
    pub response_modalities: Option<Vec<String>>,
    pub service_tier: Option<String>,
    pub system_instruction: Option<String>,
    /// Per-part values in TypeScript; the crate reads them from part
    /// metadata, and accepts them here so that the option names stay valid.
    #[allow(dead_code)]
    pub signature: Option<String>,
    #[allow(dead_code)]
    pub interaction_id: Option<String>,
    pub polling_timeout_ms: Option<u64>,
    pub background: Option<bool>,
    pub environment: Option<Value>,
}

/// A built request: its body and what the call needs to run it.
pub(crate) struct Built {
    pub body: Map<String, Value>,
    pub background: bool,
    pub polling_timeout_ms: Option<u64>,
}

fn reject(what: &str) -> Error {
    Error::Validation(format!("the Google Interactions API does not take {what}"))
}

fn one_of(name: &str, value: Option<&str>, allowed: &[&str]) -> Result<(), Error> {
    match value {
        Some(value) => options::check_one_of(name, value, allowed),
        None => Ok(()),
    }
}

fn parse_options(
    flavor: Flavor,
    params: &InferenceRequestParams,
) -> Result<InteractionsOptions, Error> {
    let options: InteractionsOptions = options::parse(flavor, params.provider_params.as_ref())?;
    one_of(
        "thinkingLevel",
        options.thinking_level.as_deref(),
        &["minimal", "low", "medium", "high"],
    )?;
    one_of(
        "thinkingSummaries",
        options.thinking_summaries.as_deref(),
        &["auto", "none"],
    )?;
    one_of(
        "mediaResolution",
        options.media_resolution.as_deref(),
        &["low", "medium", "high", "ultra_high"],
    )?;
    one_of(
        "serviceTier",
        options.service_tier.as_deref(),
        &["flex", "standard", "priority"],
    )?;
    for modality in options.response_modalities.iter().flatten() {
        options::check_one_of(
            "responseModalities",
            modality,
            &["text", "image", "audio", "video", "document"],
        )?;
    }
    if options.polling_timeout_ms == Some(0) {
        return Err(Error::Validation(
            "google option pollingTimeoutMs must be positive".into(),
        ));
    }
    Ok(options)
}

/// The response format entries of the option `responseFormat`, in snake case.
fn response_format_entry(entry: &Value) -> Result<Value, Error> {
    let kind = entry
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let text = |key: &str| entry.get(key).filter(|v| !v.is_null()).cloned();
    let mut out = Map::new();
    out.insert("type".into(), json!(kind));
    let mut put = |wire: &str, key: &str| {
        if let Some(value) = text(key) {
            out.insert(wire.into(), value);
        }
    };
    match kind {
        "text" => {
            put("mime_type", "mimeType");
            put("schema", "schema");
        }
        "image" => {
            one_of(
                "responseFormat.aspectRatio",
                entry.get("aspectRatio").and_then(Value::as_str),
                ASPECT_RATIOS,
            )?;
            one_of(
                "responseFormat.imageSize",
                entry.get("imageSize").and_then(Value::as_str),
                &["1K", "2K", "4K", "512"],
            )?;
            put("mime_type", "mimeType");
            put("aspect_ratio", "aspectRatio");
            put("image_size", "imageSize");
        }
        "audio" => put("mime_type", "mimeType"),
        "video" => {
            one_of(
                "responseFormat.aspectRatio",
                entry.get("aspectRatio").and_then(Value::as_str),
                &["16:9", "9:16"],
            )?;
            one_of(
                "responseFormat.resolution",
                entry.get("resolution").and_then(Value::as_str),
                &["360p", "720p", "1080p", "4k"],
            )?;
            one_of(
                "responseFormat.delivery",
                entry.get("delivery").and_then(Value::as_str),
                &["inline", "uri"],
            )?;
            put("aspect_ratio", "aspectRatio");
            put("resolution", "resolution");
            put("duration", "duration");
            put("delivery", "delivery");
            put("gcs_uri", "gcsUri");
        }
        other => {
            return Err(Error::Validation(format!(
                "google option responseFormat has an entry of unknown type {other:?}"
            )));
        }
    }
    Ok(Value::Object(out))
}

/// The tools and tool choice of a request (`prepare-google-interactions-tools.ts`).
fn tools(
    tools: &[Tool],
    choice: Option<&ToolChoice>,
) -> Result<(Option<Value>, Option<Value>), Error> {
    if let Some(ToolChoice::Selection(_)) = choice {
        return Err(reject("tool selections"));
    }
    if tools.is_empty() {
        return Ok((None, None));
    }
    let mut wire = Vec::new();
    let mut has_function = false;
    for tool in tools {
        tool.validate_shape()
            .map_err(|error| Error::Validation(error.to_string()))?;
        if tool.kind == ToolKind::Function {
            has_function = true;
            let spec = tool.spec.as_ref().expect("function tools have a spec");
            wire.push(json!({
                "type": "function",
                "name": tool.name,
                "description": spec.description.as_deref().unwrap_or(""),
                "parameters": serde_json::to_value(&spec.params)?,
            }));
            continue;
        }
        let args = tool
            .tool_config
            .as_ref()
            .map(|c| c.args.clone())
            .unwrap_or_default();
        let arg = |key: &str| args.get(key).filter(|v| !v.is_null()).cloned();
        let entry = match builtin_name(tool)? {
            "google_search" => {
                let mut entry = json!({"type": "google_search"});
                if let Some(types) = arg("searchTypes").filter(Value::is_object) {
                    let mut list = Vec::new();
                    if types.get("webSearch").is_some_and(|v| !v.is_null()) {
                        list.push("web_search");
                    }
                    if types.get("imageSearch").is_some_and(|v| !v.is_null()) {
                        list.push("image_search");
                    }
                    if !list.is_empty() {
                        entry["search_types"] = json!(list);
                    }
                }
                entry
            }
            "code_execution" => json!({"type": "code_execution"}),
            "url_context" => json!({"type": "url_context"}),
            "file_search" => {
                let mut entry = json!({"type": "file_search"});
                for (wire_key, key) in [
                    ("file_search_store_names", "fileSearchStoreNames"),
                    ("top_k", "topK"),
                    ("metadata_filter", "metadataFilter"),
                ] {
                    if let Some(value) = arg(key) {
                        entry[wire_key] = value;
                    }
                }
                entry
            }
            "google_maps" => {
                let mut entry = json!({"type": "google_maps"});
                for (wire_key, key) in [
                    ("latitude", "latitude"),
                    ("longitude", "longitude"),
                    ("enable_widget", "enableWidget"),
                ] {
                    if let Some(value) = arg(key) {
                        entry[wire_key] = value;
                    }
                }
                entry
            }
            "computer_use" => {
                let mut entry = json!({"type": "computer_use", "environment": arg("environment").unwrap_or_else(|| json!("browser"))});
                if let Some(value) = arg("excludedPredefinedFunctions") {
                    entry["excludedPredefinedFunctions"] = value;
                }
                entry
            }
            "mcp_server" => {
                let mut entry = json!({"type": "mcp_server"});
                for (wire_key, key) in [
                    ("name", "name"),
                    ("url", "url"),
                    ("headers", "headers"),
                    ("allowed_tools", "allowedTools"),
                ] {
                    if let Some(value) = arg(key) {
                        entry[wire_key] = value;
                    }
                }
                entry
            }
            "retrieval" => {
                let mut entry = json!({"type": "retrieval", "retrieval_types": arg("retrievalTypes").unwrap_or_else(|| json!(["vertex_ai_search"]))});
                if let Some(value) = arg("vertexAiSearchConfig") {
                    entry["vertex_ai_search_config"] = value;
                }
                entry
            }
            other => {
                return Err(Error::Validation(format!(
                    "the Google Interactions API does not take the tool google.{other}"
                )));
            }
        };
        wire.push(entry);
    }
    let choice = match choice {
        Some(choice) if has_function => Some(match choice {
            ToolChoice::Auto => json!("auto"),
            ToolChoice::Required => json!("required"),
            ToolChoice::None => json!("none"),
            ToolChoice::Tool(tool) => {
                json!({"allowed_tools": {"mode": "validated", "tools": [tool.0]}})
            }
            ToolChoice::Selection(_) => unreachable!("rejected above"),
        }),
        _ => None,
    };
    Ok((Some(Value::Array(wire)), choice))
}

fn google_meta<'a>(metadata: Option<&'a Metadata>, key: &str) -> Option<&'a str> {
    metadata?
        .get("google")?
        .get(key)?
        .as_str()
        .filter(|value| !value.is_empty())
}

/// A file part as an Interactions content block (`convertFilePartToContent`).
fn file_block(part: &FilePart, media_resolution: Option<&str>) -> Result<Value, Error> {
    let top = part.media_type.split('/').next().unwrap_or_default();
    let kind = match top {
        "image" => "image",
        "audio" => "audio",
        "video" => "video",
        "application" | "text" => "document",
        _ => {
            return Err(Error::Validation(format!(
                "the Google Interactions API does not take files of type {}",
                part.media_type
            )));
        }
    };
    let full = !part.media_type.ends_with("/*");
    let mut block = match &part.data {
        FileData::Text(text)
            if text.starts_with("http://")
                || text.starts_with("https://")
                || text.starts_with("gs://") =>
        {
            let mut block = json!({"type": kind, "uri": text});
            if full {
                block["mime_type"] = json!(part.media_type);
            }
            block
        }
        _ => {
            let (mime_type, data) = super::super::messages::inline_data(part);
            json!({"type": kind, "data": data, "mime_type": mime_type})
        }
    };
    if let Some(resolution) = media_resolution
        && matches!(kind, "image" | "video")
    {
        block["resolution"] = json!(resolution);
    }
    if kind == "video"
        && let Some(processing) = part
            .provider_metadata
            .as_ref()
            .and_then(|m| m.get("google"))
            .and_then(|g| g.get("processing"))
            .filter(|p| !p.is_null())
    {
        block["processing"] = match processing {
            Value::String(text) if text == "agentic" || text == "static" => processing.clone(),
            Value::Object(config)
                if config.get("type").and_then(Value::as_str) == Some("static") =>
            {
                let mut wire = json!({"type": "static"});
                for (wire_key, key) in
                    [("start_offset", "startOffset"), ("end_offset", "endOffset")]
                {
                    if let Some(seconds) = config.get(key).and_then(Value::as_f64) {
                        wire[wire_key] = json!(format!("{}s", number_text(seconds)));
                    }
                }
                if let Some(fps) = config.get("fps").filter(|v| v.is_number()) {
                    wire["fps"] = fps.clone();
                }
                wire
            }
            _ => {
                return Err(Error::Validation(
                    "the processing option of a video file must be \"agentic\", \"static\" or a static processing configuration".into(),
                ));
            }
        };
    }
    Ok(block)
}

/// A number as JavaScript prints it: integral values without a fraction.
fn number_text(value: f64) -> String {
    if value.fract() == 0.0 && value.abs() < 9_007_199_254_740_992.0 {
        format!("{}", value as i64)
    } else {
        value.to_string()
    }
}

/// Join adjacent text blocks without annotations with a blank line.
fn merge_texts(content: Vec<Value>) -> Vec<Value> {
    let mut merged: Vec<Value> = Vec::new();
    for block in content {
        if block["type"] == "text"
            && block.get("annotations").is_none()
            && let Some(last) = merged.last_mut()
            && last["type"] == "text"
            && last.get("annotations").is_none()
        {
            let text = format!(
                "{}\n{}",
                last["text"].as_str().unwrap_or_default(),
                block["text"].as_str().unwrap_or_default()
            );
            last["text"] = json!(text);
            continue;
        }
        merged.push(block);
    }
    merged
}

fn tool_result(result: &ToolResultPart) -> Result<Value, Error> {
    let mut block = json!({"type": "function_result", "call_id": result.tool_call_id, "name": result.tool_name});
    if let Some(signature) = google_meta(result.provider_metadata.as_ref(), "signature") {
        block["signature"] = json!(signature);
    }
    if result.is_error() {
        block["is_error"] = json!(true);
    }
    block["result"] = match result.get_model_input() {
        ToolResultValue::Json(Value::String(text)) => json!(text),
        ToolResultValue::Json(value) => json!(value.to_string()),
        ToolResultValue::Special(SpecialToolResult::Content(content)) if !result.is_error() => {
            let mut blocks = Vec::new();
            for item in &content.value {
                match item {
                    ContentPart::Text(text) => {
                        blocks.push(json!({"type": "text", "text": text.text}))
                    }
                    ContentPart::File(file) if file.media_type.starts_with("image/") => {
                        let mut image = file_block(file, None)?;
                        if let Some(object) = image.as_object_mut() {
                            object.remove("resolution");
                        }
                        blocks.push(image);
                    }
                    ContentPart::File(file) => {
                        return Err(Error::Validation(format!(
                            "an Interactions function result takes text and images, not a file of type {}",
                            file.media_type
                        )));
                    }
                }
            }
            Value::Array(blocks)
        }
        special => json!(serde_json::to_string(special)?),
    };
    Ok(block)
}

fn tool_args(text: &str) -> Value {
    match serde_json::from_str::<Value>(text) {
        Ok(Value::Object(object)) => Value::Object(object),
        Ok(other) => json!({"value": other}),
        Err(_) if text.trim().is_empty() => json!({}),
        Err(_) => json!({"value": text}),
    }
}

/// The input steps and the system text of a history
/// (`convert-to-google-interactions-input.ts`).
pub(crate) fn input(
    history: &[Message],
    previous_interaction_id: Option<&str>,
    store: Option<bool>,
    media_resolution: Option<&str>,
) -> Result<(Vec<Value>, Option<String>), Error> {
    let compact = previous_interaction_id.filter(|_| store != Some(false));
    let mut dropped_calls = HashSet::new();
    let mut systems = Vec::new();
    let mut steps = Vec::new();
    for message in history {
        if let Some(previous) = compact
            && message.role == MessageRole::Assistant
            && message
                .parts
                .iter()
                .any(|part| google_meta(part_metadata(part), "interactionId") == Some(previous))
        {
            for part in &message.parts {
                if let Part::ToolCall(call) = part {
                    dropped_calls.insert(call.tool_call_id.clone());
                }
            }
            continue;
        }
        match message.role {
            MessageRole::Internal => {}
            MessageRole::System => systems.push(message.text()),
            MessageRole::User => {
                let mut content = Vec::new();
                for part in &message.parts {
                    match part {
                        Part::Text(text) => {
                            content.push(json!({"type": "text", "text": text.text}))
                        }
                        Part::File(file) => content.push(file_block(file, media_resolution)?),
                        _ => {}
                    }
                }
                let content = merge_texts(content);
                if !content.is_empty() {
                    steps.push(json!({"type": "user_input", "content": content}));
                }
            }
            MessageRole::Assistant => {
                let mut pending: Vec<Value> = Vec::new();
                let flush = |pending: &mut Vec<Value>, steps: &mut Vec<Value>| {
                    if !pending.is_empty() {
                        steps.push(
                            json!({"type": "model_output", "content": std::mem::take(pending)}),
                        );
                    }
                };
                for part in &message.parts {
                    match part {
                        Part::Text(text) => {
                            pending.push(json!({"type": "text", "text": text.text}))
                        }
                        Part::File(file) => pending.push(file_block(file, media_resolution)?),
                        Part::Reasoning(reasoning) => {
                            flush(&mut pending, &mut steps);
                            let mut step = json!({"type": "thought"});
                            if let Some(signature) =
                                google_meta(reasoning.provider_metadata.as_ref(), "signature")
                            {
                                step["signature"] = json!(signature);
                            }
                            if !reasoning.text.is_empty() {
                                step["summary"] = json!([{"type": "text", "text": reasoning.text}]);
                            }
                            steps.push(step);
                        }
                        Part::ToolCall(call) => {
                            flush(&mut pending, &mut steps);
                            let mut step = json!({
                                "type": "function_call",
                                "id": call.tool_call_id,
                                "name": call.tool_name,
                                "arguments": tool_args(&call.tool_args),
                            });
                            if let Some(signature) =
                                google_meta(call.provider_metadata.as_ref(), "signature")
                            {
                                step["signature"] = json!(signature);
                            }
                            steps.push(step);
                        }
                        Part::BuiltinToolCall(call) if call.tool_name == "processing" => {
                            flush(&mut pending, &mut steps);
                            let metadata = call.provider_metadata.as_ref();
                            let id =
                                google_meta(metadata, "processingId").unwrap_or(&call.tool_call_id);
                            let mut step = json!({"type": "processing_call", "id": id});
                            if let Some(signature) = google_meta(metadata, "signature") {
                                step["signature"] = json!(signature);
                            }
                            steps.push(step);
                        }
                        Part::BuiltinToolReturn(result) if result.tool_name == "processing" => {
                            flush(&mut pending, &mut steps);
                            let metadata = result.provider_metadata.as_ref();
                            let id = google_meta(metadata, "processingCallId")
                                .unwrap_or(&result.tool_call_id);
                            let mut step = json!({"type": "processing_result", "call_id": id});
                            if let Some(signature) = google_meta(metadata, "signature") {
                                step["signature"] = json!(signature);
                            }
                            steps.push(step);
                        }
                        // The server ran its built-in tools; their calls do
                        // not go back as function calls.
                        _ => {}
                    }
                }
                flush(&mut pending, &mut steps);
            }
            MessageRole::Tool => {
                let mut content = Vec::new();
                for part in &message.parts {
                    if let Part::ToolResult(result) = part
                        && !dropped_calls.contains(&result.tool_call_id)
                    {
                        content.push(tool_result(result)?);
                    }
                }
                if !content.is_empty() {
                    steps.push(json!({"type": "user_input", "content": content}));
                }
            }
        }
    }
    let system = (!systems.is_empty()).then(|| systems.join("\n\n"));
    Ok((steps, system))
}

fn part_metadata(part: &Part) -> Option<&Metadata> {
    match part {
        Part::Text(p) => p.provider_metadata.as_ref(),
        Part::ToolCall(p) => p.provider_metadata.as_ref(),
        Part::ToolResult(p) => p.provider_metadata.as_ref(),
        Part::BuiltinToolCall(p) => p.provider_metadata.as_ref(),
        Part::BuiltinToolReturn(p) => p.provider_metadata.as_ref(),
        Part::Reasoning(p) => p.provider_metadata.as_ref(),
        Part::File(p) => p.provider_metadata.as_ref(),
        Part::Hook(_) => None,
    }
}

/// Build the request body for a model (`agent` false) or an agent named
/// `name` (GGL-026).
pub(crate) fn build(
    flavor: Flavor,
    name: &str,
    agent: bool,
    history: &[Message],
    options_in: &StreamOptions,
) -> Result<Built, Error> {
    let params = options_in.params.clone().unwrap_or_default();
    let options = parse_options(flavor, &params)?;
    let mut generation = Map::new();
    let set_generation =
        |key: &str, value: Value, generation: &mut Map<String, Value>| -> Result<(), Error> {
            if agent {
                return Err(Error::Validation(format!(
                    "an Interactions agent takes no {key}; use the agentConfig option"
                )));
            }
            generation.insert(key.into(), value);
            Ok(())
        };
    if let ProviderValue::Value(sampling) = &params.sampling {
        if let Some(t) = &sampling.temperature
            && let ProviderValue::Value(v) = t.temperature
        {
            set_generation("temperature", json!(v), &mut generation)?;
        }
        if let Some(t) = &sampling.top_p
            && let ProviderValue::Value(Some(v)) = t.top_p
        {
            set_generation("top_p", json!(v), &mut generation)?;
        }
        if let Some(t) = &sampling.top_k
            && let ProviderValue::Value(Some(v)) = t.top_k
        {
            set_generation("top_k", json!(v), &mut generation)?;
        }
        if let Some(seed) = &sampling.seed
            && let ProviderValue::Value(Some(Seed::Value(v))) = seed.seed
        {
            set_generation("seed", json!(v), &mut generation)?;
        }
        if sampling
            .min_p
            .as_ref()
            .is_some_and(|m| matches!(m.min_p, ProviderValue::Value(Some(_))))
        {
            return Err(reject("min-p"));
        }
        if let Some(penalties) = &sampling.repetition_penalty {
            for (value, what) in [
                (&penalties.frequency_penalty, "a frequency penalty"),
                (&penalties.presence_penalty, "a presence penalty"),
                (&penalties.repetition_penalty, "a repetition penalty"),
            ] {
                if matches!(value, ProviderValue::Value(Some(_))) {
                    return Err(reject(what));
                }
            }
            if matches!(
                penalties.consideration_window,
                ProviderValue::Value(Some(_))
            ) {
                return Err(reject("a repetition consideration window"));
            }
        }
    }
    if let Some(output) = &params.output {
        if let Some(max) = output.max_tokens {
            set_generation("max_output_tokens", json!(max), &mut generation)?;
        }
        if output.include.is_some() {
            return Err(reject("an output include list"));
        }
        if matches!(output.text_verbosity, ProviderValue::Value(Some(_))) {
            return Err(reject("text verbosity"));
        }
        if matches!(output.reasoning_summary, ProviderValue::Value(_)) {
            return Err(reject(
                "a reasoning summary; use the thinkingSummaries option",
            ));
        }
    }
    if matches!(&params.reasoning, ProviderValue::Value(r) if matches!(r.effort, ProviderValue::Value(_)))
    {
        return Err(reject("a reasoning effort; use the thinkingLevel option"));
    }
    if let Some(level) = &options.thinking_level {
        set_generation("thinking_level", json!(level), &mut generation)?;
    }
    if let Some(summaries) = &options.thinking_summaries {
        set_generation("thinking_summaries", json!(summaries), &mut generation)?;
    }
    let mut choice = options_in.tool_choice.clone();
    if let Some(calling) = &params.tool_calling {
        if matches!(calling.max_tool_calls, ProviderValue::Value(Some(_))) {
            return Err(reject("a maximum tool call count"));
        }
        if matches!(calling.parallel_tool_calls, ProviderValue::Value(_)) {
            return Err(reject("parallel tool calls"));
        }
        if choice.is_none() {
            choice = Some(calling.tool_choice.clone());
        }
    }
    if params.safety_identifier.is_some() {
        return Err(reject("a safety identifier"));
    }
    if params.metadata.is_some() {
        return Err(reject("request metadata"));
    }
    if let Some(cache) = &params.cache
        && (cache.key.is_some()
            || matches!(cache.retention, ProviderValue::Value(_))
            || matches!(cache.mode, ProviderValue::Value(_)))
    {
        return Err(reject("cache settings"));
    }
    if params.context_management.is_some() {
        return Err(reject("context management"));
    }
    let (wire_tools, wire_choice) = tools(&options_in.tools, choice.as_ref())?;
    if let Some(choice) = wire_choice {
        set_generation("tool_choice", choice, &mut generation)?;
    }

    let mut formats = Vec::new();
    if let Some(schema) = &options_in.output_schema {
        if agent {
            return Err(Error::Validation(
                "an Interactions agent takes no output schema".into(),
            ));
        }
        formats.push(json!({"type": "text", "mime_type": "application/json", "schema": serde_json::to_value(schema)?}));
    }
    for entry in options.response_format.iter().flatten() {
        formats.push(response_format_entry(entry)?);
    }
    if let Some(config) = options.image_config.as_ref().filter(|c| !c.is_null()) {
        if agent {
            return Err(Error::Validation(
                "an Interactions agent takes no imageConfig; use the agentConfig option".into(),
            ));
        }
        one_of(
            "imageConfig.aspectRatio",
            config.get("aspectRatio").and_then(Value::as_str),
            ASPECT_RATIOS,
        )?;
        one_of(
            "imageConfig.imageSize",
            config.get("imageSize").and_then(Value::as_str),
            &["1K", "2K", "4K", "512"],
        )?;
        if !formats.iter().any(|entry| entry["type"] == "image") {
            let mut entry = json!({"type": "image", "mime_type": "image/png"});
            for (wire, key) in [("aspect_ratio", "aspectRatio"), ("image_size", "imageSize")] {
                if let Some(value) = config.get(key).filter(|v| !v.is_null()) {
                    entry[wire] = value.clone();
                }
            }
            formats.push(entry);
        }
    }

    let (input, system) = input(
        history,
        options.previous_interaction_id.as_deref(),
        options.store,
        options.media_resolution.as_deref(),
    )?;
    let system = system.or_else(|| options.system_instruction.clone());

    let mut body = Map::new();
    body.insert(if agent { "agent" } else { "model" }.into(), json!(name));
    body.insert("input".into(), json!(input));
    if let Some(system) = system {
        body.insert("system_instruction".into(), json!(system));
    }
    if let Some(tools) = wire_tools {
        body.insert("tools".into(), tools);
    }
    if !formats.is_empty() {
        body.insert("response_format".into(), json!(formats));
    }
    if let Some(modalities) = &options.response_modalities {
        body.insert("response_modalities".into(), json!(modalities));
    }
    if let Some(previous) = &options.previous_interaction_id {
        body.insert("previous_interaction_id".into(), json!(previous));
    }
    let service_tier = options.service_tier.clone().or_else(|| {
        params
            .provider_service
            .as_ref()
            .and_then(|s| match &s.service_tier {
                ProviderValue::Value(tier) => Some(tier.clone()),
                ProviderValue::Default => None,
            })
    });
    if let Some(tier) = service_tier {
        body.insert("service_tier".into(), json!(tier));
    }
    if let Some(store) = options.store {
        body.insert("store".into(), json!(store));
    }
    if !generation.is_empty() {
        body.insert("generation_config".into(), Value::Object(generation));
    }
    if let Some(config) = options.agent_config.as_ref().filter(|c| !c.is_null()) {
        if !agent {
            return Err(Error::Validation("a model takes no agentConfig".into()));
        }
        body.insert("agent_config".into(), agent_config(config)?);
    }
    if let Some(environment) = options.environment.as_ref().filter(|e| !e.is_null()) {
        if !agent {
            return Err(Error::Validation(
                "environment is only supported when an agent is set".into(),
            ));
        }
        body.insert("environment".into(), environment_value(environment)?);
    }
    if let Some(background) = options.background {
        body.insert("background".into(), json!(background));
    }
    if let Some(extra) = &params.extra_body {
        body.extend(extra.clone());
    }
    Ok(Built {
        body,
        background: options.background == Some(true),
        polling_timeout_ms: options.polling_timeout_ms,
    })
}

fn agent_config(config: &Value) -> Result<Value, Error> {
    match config.get("type").and_then(Value::as_str) {
        Some("dynamic") => Ok(json!({"type": "dynamic"})),
        Some("deep-research") => {
            let mut wire = json!({"type": "deep-research"});
            for (wire_key, key, allowed) in [
                (
                    "thinking_summaries",
                    "thinkingSummaries",
                    &["auto", "none"][..],
                ),
                ("visualization", "visualization", &["off", "auto"][..]),
            ] {
                if let Some(value) = config.get(key).filter(|v| !v.is_null()) {
                    one_of(&format!("agentConfig.{key}"), value.as_str(), allowed)?;
                    wire[wire_key] = value.clone();
                }
            }
            if let Some(value) = config.get("collaborativePlanning").filter(|v| !v.is_null()) {
                wire["collaborative_planning"] = value.clone();
            }
            Ok(wire)
        }
        _ => Err(Error::Validation(
            "google option agentConfig needs the type dynamic or deep-research".into(),
        )),
    }
}

fn environment_value(environment: &Value) -> Result<Value, Error> {
    if environment.is_string() {
        return Ok(environment.clone());
    }
    let invalid = || {
        Error::Validation("google option environment must be an id or a remote environment".into())
    };
    if environment.get("type").and_then(Value::as_str) != Some("remote") {
        return Err(invalid());
    }
    let mut wire = json!({"type": "remote"});
    if let Some(sources) = environment
        .get("sources")
        .and_then(Value::as_array)
        .filter(|s| !s.is_empty())
    {
        let mut out = Vec::new();
        for source in sources {
            match source.get("type").and_then(Value::as_str) {
                Some("inline") => out.push(json!({"type": "inline", "content": source.get("content"), "target": source.get("target")})),
                Some(kind @ ("gcs" | "repository")) => {
                    let mut entry = json!({"type": kind, "source": source.get("source")});
                    if let Some(target) = source.get("target").filter(|t| !t.is_null()) {
                        entry["target"] = target.clone();
                    }
                    out.push(entry);
                }
                _ => return Err(invalid()),
            }
        }
        wire["sources"] = json!(out);
    }
    match environment.get("network") {
        None | Some(Value::Null) => {}
        Some(Value::String(text)) if text == "disabled" => wire["network"] = json!("disabled"),
        Some(network) => {
            let allowlist = network
                .get("allowlist")
                .and_then(Value::as_array)
                .ok_or_else(invalid)?;
            let entries: Vec<Value> = allowlist
                .iter()
                .map(|entry| {
                    let mut out = json!({"domain": entry.get("domain")});
                    if let Some(transform) = entry.get("transform").filter(|t| !t.is_null()) {
                        out["transform"] = transform.clone();
                    }
                    out
                })
                .collect();
            wire["network"] = json!({"allowlist": entries});
        }
    }
    Ok(wire)
}
