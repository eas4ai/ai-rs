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

//! Generate content responses as model events (`doStream` in
//! `google-language-model.ts`). A buffered response is read as one stream
//! chunk, so `generate` gives the message the stream would.

use std::collections::{HashSet, VecDeque};

use serde_json::{Map, Value, json};

use super::{
    accumulator::Accumulator,
    api::{self, Flavor},
};
use crate::{
    Error,
    types::{
        events::*,
        messages::{
            BuiltinToolCallPart, BuiltinToolReturnPart, FileData, Metadata, ToolCallPart,
            generate_id,
        },
        usage::Usage,
    },
};

struct StreamingCall {
    id: String,
    name: String,
    accumulator: Accumulator,
    metadata: Option<Metadata>,
    /// The JSON response tool's call, delivered as text.
    as_text: bool,
}

/// Reads generate content chunks into model events.
pub(crate) struct Reader {
    flavor: Flavor,
    provider: String,
    json_tool: Option<String>,
    pub(crate) events: VecDeque<Event>,
    text_block: Option<String>,
    reasoning_block: Option<String>,
    blocks: u32,
    has_tool_calls: bool,
    finish: Option<String>,
    raw_finish: Option<String>,
    usage: Option<Value>,
    continued_usage: Option<Value>,
    initial_prompt_tokens: Option<Value>,
    continued: bool,
    prompt_feedback: Value,
    blocked: bool,
    grounding: Value,
    url_context: Value,
    safety_ratings: Value,
    finish_message: Value,
    continuation_token: Option<String>,
    request_grounding: Value,
    grounding_per_request: Vec<Value>,
    sources: Vec<Value>,
    seen_sources: HashSet<String>,
    last_code_call: Option<String>,
    last_server_call: Option<String>,
    streaming_calls: Vec<StreamingCall>,
    response_id: Option<String>,
    response_model: Option<String>,
    service_tier_header: Option<String>,
}

impl Reader {
    pub(crate) fn new(flavor: Flavor, provider: String, json_tool: Option<String>) -> Self {
        Self {
            flavor,
            provider,
            json_tool,
            events: VecDeque::from([Event::StreamStart(StreamStart::default())]),
            text_block: None,
            reasoning_block: None,
            blocks: 0,
            has_tool_calls: false,
            finish: None,
            raw_finish: None,
            usage: None,
            continued_usage: None,
            initial_prompt_tokens: None,
            continued: false,
            prompt_feedback: Value::Null,
            blocked: false,
            grounding: Value::Null,
            url_context: Value::Null,
            safety_ratings: Value::Null,
            finish_message: Value::Null,
            continuation_token: None,
            request_grounding: Value::Null,
            grounding_per_request: Vec::new(),
            sources: Vec::new(),
            seen_sources: HashSet::new(),
            last_code_call: None,
            last_server_call: None,
            streaming_calls: Vec::new(),
            response_id: None,
            response_model: None,
            service_tier_header: None,
        }
    }

    /// The raw finish reason of the last request, or the prompt block reason.
    pub(crate) fn raw_finish(&self) -> Option<String> {
        self.raw_finish.clone()
    }

    /// Keep the service tier of the `x-gemini-service-tier` response header.
    pub(crate) fn headers(&mut self, headers: &[(String, String)]) {
        if self.service_tier_header.is_none() {
            self.service_tier_header = headers
                .iter()
                .find(|(name, _)| name.eq_ignore_ascii_case("x-gemini-service-tier"))
                .map(|(_, value)| value.clone());
        }
    }

    fn metadata(&self, payload: Value) -> Metadata {
        self.flavor.metadata(payload)
    }

    fn signature_metadata(&self, part: &Value) -> Option<Metadata> {
        part.get("thoughtSignature")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(|signature| self.metadata(json!({"thoughtSignature": signature})))
    }

    fn next_block(&mut self) -> String {
        let id = self.blocks.to_string();
        self.blocks += 1;
        id
    }

    fn end_text(&mut self) {
        if let Some(block_id) = self.text_block.take() {
            self.events.push_back(Event::TextEnd(TextEnd {
                block_id,
                ..Default::default()
            }));
        }
    }

    fn end_reasoning(&mut self) {
        if let Some(block_id) = self.reasoning_block.take() {
            self.events.push_back(Event::ReasoningEnd(ReasoningEnd {
                block_id,
                ..Default::default()
            }));
        }
    }

    /// A new request of a continued response starts: its chunks decide the
    /// finish reason again.
    pub(crate) fn start_continuation(&mut self) {
        self.finish = None;
        self.raw_finish = None;
        self.continued = true;
        self.grounding_per_request
            .push(std::mem::take(&mut self.request_grounding));
    }

    /// Read one chunk, or a whole buffered response. Returns the
    /// continuation token when the chunk finishes with `CONTINUATION`.
    pub(crate) fn chunk(&mut self, value: &Value) -> Result<Option<String>, Error> {
        if let Some(error) = value.get("error").filter(|e| !e.is_null()) {
            return Err(api::stream_error(&self.provider, error));
        }
        if self.response_id.is_none() {
            self.response_id = value
                .get("responseId")
                .and_then(Value::as_str)
                .map(str::to_owned);
        }
        if self.response_model.is_none() {
            self.response_model = value
                .get("modelVersion")
                .and_then(Value::as_str)
                .map(str::to_owned);
        }
        if let Some(usage) = value.get("usageMetadata").filter(|u| u.is_object()) {
            self.usage = Some(usage.clone());
        }
        if let Some(feedback) = value.get("promptFeedback").filter(|f| !f.is_null())
            && !self.blocked
        {
            self.prompt_feedback = feedback.clone();
            let reason = feedback.get("blockReason").and_then(Value::as_str);
            if let Some(reason) = reason.filter(|r| {
                !r.is_empty()
                    && *r != "BLOCK_REASON_UNSPECIFIED"
                    && *r != "BLOCKED_REASON_UNSPECIFIED"
            }) {
                self.blocked = true;
                self.finish = Some("content_filter".into());
                self.raw_finish = Some(reason.to_owned());
            }
        }
        let Some(candidate) = value
            .get("candidates")
            .and_then(Value::as_array)
            .and_then(|candidates| candidates.first())
        else {
            return Ok(None);
        };
        for (key, slot) in [
            ("groundingMetadata", &mut self.grounding),
            ("urlContextMetadata", &mut self.url_context),
            ("safetyRatings", &mut self.safety_ratings),
            ("finishMessage", &mut self.finish_message),
        ] {
            if let Some(value) = candidate.get(key).filter(|v| !v.is_null()) {
                *slot = value.clone();
            }
        }
        if let Some(grounding) = candidate.get("groundingMetadata").filter(|v| !v.is_null()) {
            self.request_grounding = grounding.clone();
        }
        if let Some(token) = candidate.get("continuationToken").and_then(Value::as_str) {
            self.continuation_token = Some(token.to_owned());
        }
        // A blocked prompt ends the generated content; later chunks still
        // bring usage and metadata.
        if self.blocked {
            return Ok(None);
        }
        self.add_sources(candidate.get("groundingMetadata"));
        let parts = candidate
            .get("content")
            .and_then(|content| content.get("parts"))
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        for part in &parts {
            self.part(part)?;
        }
        for part in &parts {
            if let Some(call) = part.get("functionCall").filter(|c| c.is_object()) {
                self.function_call(part, call);
            }
        }
        let raw = candidate.get("finishReason").and_then(Value::as_str);
        let mut continuation = None;
        if raw == Some("CONTINUATION") {
            // Usage is per request: bank this request's.
            if self.continued_usage.is_none() {
                self.initial_prompt_tokens = self
                    .usage
                    .as_ref()
                    .and_then(|usage| usage.get("promptTokenCount"))
                    .cloned();
            }
            self.continued_usage = add_usage(self.continued_usage.take(), self.usage.take());
            continuation = candidate
                .get("continuationToken")
                .and_then(Value::as_str)
                .map(str::to_owned);
        }
        if let Some(raw) = raw {
            self.finish = Some(finish_reason(raw, self.has_tool_calls).into());
            self.raw_finish = Some(raw.to_owned());
        }
        Ok(continuation)
    }

    fn add_sources(&mut self, grounding: Option<&Value>) {
        let Some(chunks) = grounding
            .and_then(|g| g.get("groundingChunks"))
            .and_then(Value::as_array)
        else {
            return;
        };
        for chunk in chunks {
            let Some(source) = source(chunk) else {
                continue;
            };
            let key = match source.get("url").and_then(Value::as_str) {
                Some(url) => format!("url:{url}"),
                None => format!(
                    "document:{}",
                    json!([source["title"], source["filename"], source["mediaType"]])
                ),
            };
            if { self.seen_sources.insert(key); true } {
                self.sources.push(source);
            }
        }
    }

    fn part(&mut self, part: &Value) -> Result<(), Error> {
        let signature = self.signature_metadata(part);
        let thought = part.get("thought") == Some(&Value::Bool(true));
        if let Some(code) = part.get("executableCode").filter(|c| {
            c.get("code")
                .and_then(Value::as_str)
                .is_some_and(|code| !code.is_empty())
        }) {
            let id = generate_id("call");
            self.last_code_call = Some(id.clone());
            self.builtin_call(id, "code_execution".into(), code.to_string(), None);
            return Ok(());
        }
        if let Some(result) = part.get("codeExecutionResult").filter(|r| r.is_object()) {
            if let Some(id) = self.last_code_call.clone() {
                let result = json!({
                    "outcome": result.get("outcome").cloned().unwrap_or(Value::Null),
                    "output": result.get("output").and_then(Value::as_str).unwrap_or(""),
                });
                self.builtin_result(id, "code_execution".into(), result, None);
            }
            return Ok(());
        }
        if let Some(text) = part.get("text").and_then(Value::as_str) {
            if self.json_tool.is_some() && !thought {
                // The response tool supplies the output; conversational
                // text is left out.
                return Ok(());
            }
            if text.is_empty() {
                if signature.is_some()
                    && let Some(block_id) = self.text_block.clone()
                {
                    self.events.push_back(Event::TextDelta(TextDelta {
                        event: ModelEventMeta {
                            provider_metadata: signature,
                            ..Default::default()
                        },
                        chunk: String::new(),
                        block_id,
                    }));
                }
            } else if thought {
                self.end_text();
                let block_id = match &self.reasoning_block {
                    Some(id) => id.clone(),
                    None => {
                        let id = self.next_block();
                        self.reasoning_block = Some(id.clone());
                        self.events.push_back(Event::ReasoningStart(ReasoningStart {
                            event: ModelEventMeta {
                                provider_metadata: signature.clone(),
                                ..Default::default()
                            },
                            block_id: id.clone(),
                        }));
                        id
                    }
                };
                self.events.push_back(Event::ReasoningDelta(ReasoningDelta {
                    event: ModelEventMeta {
                        provider_metadata: signature,
                        ..Default::default()
                    },
                    chunk: text.to_owned(),
                    block_id,
                }));
            } else {
                self.end_reasoning();
                let block_id = match &self.text_block {
                    Some(id) => id.clone(),
                    None => {
                        let id = self.next_block();
                        self.text_block = Some(id.clone());
                        self.events.push_back(Event::TextStart(TextStart {
                            event: ModelEventMeta {
                                provider_metadata: signature.clone(),
                                ..Default::default()
                            },
                            block_id: id.clone(),
                        }));
                        id
                    }
                };
                self.events.push_back(Event::TextDelta(TextDelta {
                    event: ModelEventMeta {
                        provider_metadata: signature,
                        ..Default::default()
                    },
                    chunk: text.to_owned(),
                    block_id,
                }));
            }
            return Ok(());
        }
        if let Some(inline) = part.get("inlineData").filter(|d| d.is_object()) {
            self.end_text();
            self.end_reasoning();
            let mime_type = inline
                .get("mimeType")
                .and_then(Value::as_str)
                .unwrap_or("application/octet-stream");
            let data = inline
                .get("data")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let mut payload = Map::new();
            if let Some(signature) = part
                .get("thoughtSignature")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
            {
                payload.insert("thoughtSignature".into(), json!(signature));
            }
            if thought {
                payload.insert("thought".into(), json!(true));
            }
            self.events.push_back(Event::File(FileEvent {
                event: ModelEventMeta {
                    provider_metadata: (!payload.is_empty())
                        .then(|| self.metadata(Value::Object(payload))),
                    ..Default::default()
                },
                block_id: generate_id("part"),
                media_type: mime_type.to_owned(),
                data: FileData::Text(data.to_owned()),
                filename: None,
            }));
            return Ok(());
        }
        if let Some(call) = part.get("toolCall").filter(|c| c.is_object()) {
            let tool_type = call
                .get("toolType")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let id = call
                .get("id")
                .and_then(Value::as_str)
                .filter(|id| !id.is_empty())
                .map_or_else(|| generate_id("call"), str::to_owned);
            self.last_server_call = Some(id.clone());
            let metadata = self.server_metadata(part, &id, tool_type);
            let args = call
                .get("args")
                .filter(|a| !a.is_null())
                .cloned()
                .unwrap_or_else(|| json!({}));
            self.builtin_call(
                id,
                format!("server:{tool_type}"),
                args.to_string(),
                Some(metadata),
            );
            return Ok(());
        }
        if let Some(response) = part.get("toolResponse").filter(|r| r.is_object()) {
            let tool_type = response
                .get("toolType")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let id = self
                .last_server_call
                .take()
                .or_else(|| {
                    response
                        .get("id")
                        .and_then(Value::as_str)
                        .filter(|id| !id.is_empty())
                        .map(str::to_owned)
                })
                .unwrap_or_else(|| generate_id("call"));
            let metadata = self.server_metadata(part, &id, tool_type);
            let result = response
                .get("response")
                .filter(|r| !r.is_null())
                .cloned()
                .unwrap_or_else(|| json!({}));
            self.builtin_result(id, format!("server:{tool_type}"), result, Some(metadata));
        }
        Ok(())
    }

    fn server_metadata(&self, part: &Value, id: &str, tool_type: &str) -> Metadata {
        let mut payload = Map::new();
        if let Some(signature) = part
            .get("thoughtSignature")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
        {
            payload.insert("thoughtSignature".into(), json!(signature));
        }
        payload.insert("serverToolCallId".into(), json!(id));
        payload.insert("serverToolType".into(), json!(tool_type));
        self.metadata(Value::Object(payload))
    }

    fn builtin_call(&mut self, id: String, name: String, args: String, metadata: Option<Metadata>) {
        let meta = || ModelEventMeta {
            provider_metadata: metadata.clone(),
            ..Default::default()
        };
        self.events
            .push_back(Event::BuiltinToolStart(BuiltinToolStart {
                event: meta(),
                tool_call_id: id.clone(),
                tool_name: name.clone(),
            }));
        self.events
            .push_back(Event::BuiltinToolDelta(BuiltinToolDelta {
                event: meta(),
                chunk: args,
                tool_call_id: id.clone(),
            }));
        self.events.push_back(Event::BuiltinToolEnd(BuiltinToolEnd {
            event: meta(),
            tool_call: BuiltinToolCallPart {
                id: id.clone(),
                tool_call_id: id.clone(),
                tool_name: name,
                tool_args: String::new(),
                provider_metadata: metadata.clone(),
            },
            tool_call_id: id,
        }));
    }

    fn builtin_result(
        &mut self,
        id: String,
        name: String,
        result: Value,
        metadata: Option<Metadata>,
    ) {
        self.events
            .push_back(Event::BuiltinToolResult(BuiltinToolResult {
                event: ModelEventMeta {
                    provider_metadata: metadata.clone(),
                    ..Default::default()
                },
                result: BuiltinToolReturnPart {
                    id: generate_id("part"),
                    tool_call_id: id.clone(),
                    tool_name: name,
                    result,
                    is_error: false,
                    provider_metadata: metadata,
                },
                tool_call_id: id,
            }));
    }

    fn tool_start(&mut self, id: &str, name: &str, metadata: Option<Metadata>, as_text: bool) {
        if as_text {
            self.events.push_back(Event::TextStart(TextStart {
                event: ModelEventMeta {
                    provider_metadata: metadata,
                    ..Default::default()
                },
                block_id: id.to_owned(),
            }));
        } else {
            self.events.push_back(Event::ToolStart(ToolStart {
                event: ModelEventMeta {
                    provider_metadata: metadata,
                    ..Default::default()
                },
                tool_call_id: id.to_owned(),
                tool_name: name.to_owned(),
            }));
        }
    }

    fn tool_delta(&mut self, id: &str, chunk: String, metadata: Option<Metadata>, as_text: bool) {
        let event = ModelEventMeta {
            provider_metadata: metadata,
            ..Default::default()
        };
        self.events.push_back(if as_text {
            Event::TextDelta(TextDelta {
                event,
                chunk,
                block_id: id.to_owned(),
            })
        } else {
            Event::ToolDelta(ToolDelta {
                event,
                chunk,
                tool_call_id: id.to_owned(),
            })
        });
    }

    fn tool_end(&mut self, id: &str, name: &str, metadata: Option<Metadata>, as_text: bool) {
        let event = ModelEventMeta {
            provider_metadata: metadata,
            ..Default::default()
        };
        if as_text {
            self.events.push_back(Event::TextEnd(TextEnd {
                event,
                block_id: id.to_owned(),
            }));
        } else {
            self.events.push_back(Event::ToolEnd(ToolEnd {
                event,
                tool_call: ToolCallPart::new(id, name, ""),
                tool_call_id: id.to_owned(),
            }));
            self.has_tool_calls = true;
        }
    }

    fn function_call(&mut self, part: &Value, call: &Value) {
        let metadata = self.signature_metadata(part);
        let name = call.get("name").and_then(Value::as_str);
        let args = call.get("args").filter(|a| !a.is_null());
        let partial = call.get("partialArgs").and_then(Value::as_array);
        let will_continue = call.get("willContinue").and_then(Value::as_bool);
        let id = || {
            call.get("id")
                .and_then(Value::as_str)
                .filter(|id| !id.is_empty())
                .map_or_else(|| generate_id("call"), str::to_owned)
        };
        let streaming = partial.is_some() || (name.is_some() && will_continue == Some(true));
        let terminal =
            name.is_none() && args.is_none() && partial.is_none() && will_continue.is_none();
        if streaming {
            if let Some(name) = name {
                let as_text = self.json_tool.as_deref() == Some(name);
                let id = id();
                self.tool_start(&id, name, metadata.clone(), as_text);
                self.streaming_calls.push(StreamingCall {
                    id,
                    name: name.to_owned(),
                    accumulator: Accumulator::default(),
                    metadata: metadata.clone(),
                    as_text,
                });
            }
            if let Some(partial) = partial
                && let Some(active) = self.streaming_calls.last_mut()
            {
                let delta = active.accumulator.feed(partial);
                let (id, as_text) = (active.id.clone(), active.as_text);
                if !delta.is_empty() {
                    self.tool_delta(&id, delta, metadata.clone(), as_text);
                }
                let continues = will_continue == Some(true)
                    || partial
                        .iter()
                        .any(|arg| arg.get("willContinue") == Some(&Value::Bool(true)));
                if !continues {
                    self.finish_streaming_call();
                }
            }
            return;
        }
        if terminal {
            self.finish_streaming_call();
            return;
        }
        let Some(name) = name else { return };
        if partial.is_some() || (args.is_none() && will_continue == Some(true)) {
            return;
        }
        let as_text = self.json_tool.as_deref() == Some(name);
        let id = id();
        let input = match args {
            Some(Value::String(text)) => text.clone(),
            Some(args) => args.to_string(),
            None => "{}".to_owned(),
        };
        self.tool_start(&id, name, metadata.clone(), as_text);
        self.tool_delta(&id, input, metadata.clone(), as_text);
        self.tool_end(&id, name, metadata, as_text);
    }

    fn finish_streaming_call(&mut self) {
        let Some(active) = self.streaming_calls.pop() else {
            return;
        };
        let (_, closing) = active.accumulator.finish();
        if !closing.is_empty() {
            self.tool_delta(&active.id, closing, active.metadata.clone(), active.as_text);
        }
        self.tool_end(&active.id, &active.name, active.metadata, active.as_text);
    }

    /// End the response: close open blocks and give the finish event, or
    /// nothing when no chunk finished the response.
    pub(crate) fn end(&mut self) {
        self.end_text();
        self.end_reasoning();
        while !self.streaming_calls.is_empty() {
            self.finish_streaming_call();
        }
        let Some(finish) = self.finish.clone() else {
            return;
        };
        let usage = add_usage(self.continued_usage.take(), self.usage.take());
        let mut payload = Map::new();
        payload.insert("promptFeedback".into(), self.prompt_feedback.clone());
        payload.insert("groundingMetadata".into(), self.grounding.clone());
        if self.continued {
            let mut per_request = self.grounding_per_request.clone();
            per_request.push(self.request_grounding.clone());
            payload.insert("groundingMetadataPerRequest".into(), json!(per_request));
        }
        payload.insert("urlContextMetadata".into(), self.url_context.clone());
        payload.insert("safetyRatings".into(), self.safety_ratings.clone());
        payload.insert("usageMetadata".into(), usage.clone().unwrap_or(Value::Null));
        payload.insert("finishMessage".into(), self.finish_message.clone());
        let tier = self
            .service_tier_header
            .clone()
            .map(Value::String)
            .or_else(|| {
                usage
                    .as_ref()
                    .and_then(|u| u.get("serviceTier"))
                    .filter(|t| !t.is_null())
                    .cloned()
            });
        payload.insert("serviceTier".into(), tier.unwrap_or(Value::Null));
        if let Some(token) = &self.continuation_token
            && matches!(
                self.raw_finish.as_deref(),
                Some("MAX_TOKENS" | "CONTINUATION")
            )
        {
            payload.insert("continuationToken".into(), json!(token));
        }
        if self.continued
            && let Some(initial) = &self.initial_prompt_tokens
        {
            payload.insert("initialPromptTokenCount".into(), initial.clone());
        }
        if !self.sources.is_empty() {
            payload.insert("sources".into(), json!(self.sources));
        }
        self.events.push_back(Event::StreamEnd(StreamEnd {
            event: ModelEventMeta {
                usage: usage.as_ref().map(convert_usage),
                provider_metadata: Some(self.metadata(Value::Object(payload))),
                ..Default::default()
            },
            finish_reason: Some(finish),
            response_id: self.response_id.clone(),
            response_model: self.response_model.clone(),
        }));
    }
}

/// A grounding chunk as a source in the TypeScript shape, or `None` for a
/// chunk without one.
fn source(chunk: &Value) -> Option<Value> {
    let url_source = |url: &str, title: Option<&Value>| {
        let mut source = json!({"sourceType": "url", "id": generate_id("source"), "url": url});
        if let Some(title) = title.and_then(Value::as_str) {
            source["title"] = json!(title);
        }
        source
    };
    if let Some(web) = chunk.get("web").filter(|w| w.is_object()) {
        return Some(url_source(web.get("uri")?.as_str()?, web.get("title")));
    }
    if let Some(image) = chunk.get("image").filter(|w| w.is_object()) {
        // Google requires attribution to the source page, not the image.
        return Some(url_source(
            image.get("sourceUri")?.as_str()?,
            image.get("title"),
        ));
    }
    if let Some(context) = chunk.get("retrievedContext").filter(|w| w.is_object()) {
        let uri = context
            .get("uri")
            .and_then(Value::as_str)
            .filter(|u| !u.is_empty());
        let title = context
            .get("title")
            .and_then(Value::as_str)
            .unwrap_or("Unknown Document");
        if let Some(uri) = uri {
            if uri.starts_with("http://") || uri.starts_with("https://") {
                return Some(url_source(uri, context.get("title")));
            }
            let media_type = if uri.ends_with(".pdf") {
                "application/pdf"
            } else if uri.ends_with(".txt") {
                "text/plain"
            } else if uri.ends_with(".docx") {
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            } else if uri.ends_with(".doc") {
                "application/msword"
            } else if uri.ends_with(".md") || uri.ends_with(".markdown") {
                "text/markdown"
            } else {
                "application/octet-stream"
            };
            let filename = uri.rsplit('/').next().unwrap_or(uri);
            return Some(json!({
                "sourceType": "document",
                "id": generate_id("source"),
                "mediaType": media_type,
                "title": title,
                "filename": filename,
            }));
        }
        let store = context.get("fileSearchStore").and_then(Value::as_str)?;
        return Some(json!({
            "sourceType": "document",
            "id": generate_id("source"),
            "mediaType": "application/octet-stream",
            "title": title,
            "filename": store.rsplit('/').next().unwrap_or(store),
        }));
    }
    if let Some(maps) = chunk.get("maps").filter(|w| w.is_object()) {
        let uri = maps
            .get("uri")
            .and_then(Value::as_str)
            .filter(|u| !u.is_empty())?;
        return Some(url_source(uri, maps.get("title")));
    }
    None
}

/// A raw finish reason as the crate's normalized one
/// (`map-google-finish-reason.ts`).
pub(crate) fn finish_reason(raw: &str, has_tool_calls: bool) -> &'static str {
    match raw {
        "STOP" if has_tool_calls => "tool_call",
        "STOP" => "stop",
        "CONTINUATION" => "length",
        "IMAGE_SAFETY" | "RECITATION" | "SAFETY" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "SPII" => {
            "content_filter"
        }
        "MALFORMED_FUNCTION_CALL" => "error",
        _ => "other",
    }
}

/// Usage metadata as the crate's usage (`convert-google-usage.ts`).
pub(crate) fn convert_usage(raw: &Value) -> Usage {
    let count = |key: &str| raw.get(key).and_then(Value::as_u64);
    let prompt = count("promptTokenCount").unwrap_or(0);
    let tool_prompt = count("toolUsePromptTokenCount").unwrap_or(0);
    let candidates = count("candidatesTokenCount").unwrap_or(0);
    let thoughts = count("thoughtsTokenCount");
    Usage {
        input_tokens: prompt + tool_prompt,
        output_tokens: candidates + thoughts.unwrap_or(0),
        reasoning_tokens: thoughts,
        cache_read_tokens: count("cachedContentTokenCount"),
        cache_write_tokens: None,
        raw: Some(raw.clone()),
    }
}

/// The sum of two requests' usage metadata: counts and per-modality details
/// added, other fields from the later one (`addGoogleUsageMetadata`).
pub(crate) fn add_usage(first: Option<Value>, second: Option<Value>) -> Option<Value> {
    let (Some(Value::Object(first)), Some(Value::Object(second))) = (&first, &second) else {
        return second.or(first);
    };
    let mut sum = first.clone();
    sum.extend(second.clone());
    for key in [
        "promptTokenCount",
        "candidatesTokenCount",
        "toolUsePromptTokenCount",
        "totalTokenCount",
        "cachedContentTokenCount",
        "thoughtsTokenCount",
    ] {
        let a = first.get(key).and_then(Value::as_u64);
        let b = second.get(key).and_then(Value::as_u64);
        if a.is_some() || b.is_some() {
            sum.insert(key.into(), json!(a.unwrap_or(0) + b.unwrap_or(0)));
        }
    }
    for key in [
        "promptTokensDetails",
        "cacheTokensDetails",
        "candidatesTokensDetails",
        "toolUsePromptTokensDetails",
    ] {
        let (Some(Value::Array(a)), Some(Value::Array(b))) = (first.get(key), second.get(key))
        else {
            continue;
        };
        let mut totals: Vec<(String, u64)> = Vec::new();
        for detail in a.iter().chain(b) {
            let (Some(modality), Some(tokens)) = (
                detail.get("modality").and_then(Value::as_str),
                detail.get("tokenCount").and_then(Value::as_u64),
            ) else {
                continue;
            };
            match totals.iter_mut().find(|(m, _)| m == modality) {
                Some((_, total)) => *total += tokens,
                None => totals.push((modality.to_owned(), tokens)),
            }
        }
        sum.insert(
            key.into(),
            Value::Array(
                totals
                    .into_iter()
                    .map(|(modality, tokens)| json!({"modality": modality, "tokenCount": tokens}))
                    .collect(),
            ),
        );
    }
    Some(Value::Object(sum))
}
