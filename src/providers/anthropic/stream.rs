//! The Messages stream reader, as `doStream` in `anthropic-language-model.ts`
//! reads a stream (ANT-007, ANT-008, ANT-013, ANT-014). A buffered message
//! is read as the stream of the same message.
use super::{
    errors,
    messages::{Document, metadata},
    protocol::JSON_TOOL,
    tools::ToolNames,
};
use crate::{
    Error,
    errors::ProviderError,
    providers::{
        EventStream,
        http::{self, ApiProfile, SseJson},
    },
    transport::Body,
    types::{
        events::*,
        messages::{BuiltinToolCallPart, BuiltinToolReturnPart, ToolCallPart, generate_id},
        usage::Usage,
    },
};
use futures::{StreamExt, stream};
use serde_json::{Map, Value, json};
use std::collections::{BTreeSet, HashMap, VecDeque};

/// What the reader needs to know about the call (ANT-005, ANT-006,
/// ANT-013).
#[derive(Clone, Debug, Default)]
pub(crate) struct Reading {
    /// The output schema went out as the JSON tool: its input is the text,
    /// and other text is dropped.
    pub json_tool: bool,
    /// The application's names of the provider tools.
    pub names: ToolNames,
    /// The documents the call sent with citations enabled.
    pub documents: Vec<Document>,
}

enum Kind {
    Text {
        citations: Vec<Value>,
        metadata: Option<Value>,
    },
    Reasoning {
        signature: Option<String>,
        redacted: Option<String>,
    },
    /// A call the application runs.
    Call {
        id: String,
        /// A toolset member call: the member's name, its input gathered
        /// until the block ends.
        toolset: Option<(String, String)>,
        input: String,
        metadata: Option<Value>,
    },
    /// A call Anthropic runs.
    Builtin {
        id: String,
        /// The type to put first in the input of a code execution subtool.
        input_type: Option<String>,
        first: bool,
        /// A `code_execution` input is held until the block ends, to mark a
        /// programmatic call.
        held: Option<String>,
        caller: Option<Value>,
    },
    /// The JSON tool's input, delivered as text.
    JsonTool,
    /// A block the reader does not deliver.
    Dropped,
}

struct Parser {
    source: futures::stream::BoxStream<'static, Result<SseJson, Error>>,
    provider: String,
    api: &'static ApiProfile,
    reading: Reading,
    documents: Vec<Document>,
    blocks: HashMap<u64, Kind>,
    /// The tool names of built-in and MCP calls, by call id.
    tool_names: HashMap<String, String>,
    /// The metadata of each MCP call, by call id.
    mcp: HashMap<String, Value>,
    pending: VecDeque<Event>,
    open: Option<String>,
    started: bool,
    finished: bool,
    response_id: Option<String>,
    response_model: Option<String>,
    stop_reason: Option<String>,
    raw_usage: Map<String, Value>,
    input_tokens: u64,
    output_tokens: u64,
    cache_read: Option<u64>,
    cache_write: Option<u64>,
    thinking_tokens: Option<u64>,
    iterations: Option<Vec<Value>>,
    stop_sequence: Value,
    stop_details: Option<Value>,
    input_transformations: Option<Value>,
    safeguard_results: Option<Value>,
    container: Value,
    context_management: Value,
    json_answer: bool,
    sources: Vec<Value>,
    source_keys: BTreeSet<String>,
    web_results: bool,
    web_citations: Vec<Value>,
}

/// Read a Messages event stream.
pub(crate) fn parse(
    body: Body,
    provider: String,
    api: &'static ApiProfile,
    reading: Reading,
) -> EventStream {
    run(
        http::sse_json(*api, provider.clone(), body),
        provider,
        api,
        reading,
    )
}

/// A buffered Messages response read as the stream of the same message,
/// so that both paths give the same events.
pub(crate) fn parse_message(
    response: &Value,
    provider: String,
    api: &'static ApiProfile,
    reading: Reading,
) -> EventStream {
    let events: Vec<Result<SseJson, Error>> = message_events(response)
        .into_iter()
        .map(|event| Ok(SseJson::Object(event)))
        .collect();
    run(stream::iter(events).boxed(), provider, api, reading)
}

/// The stream events of a buffered message: its start, each content block's
/// start, delta and stop, and its end.
fn message_events(response: &Value) -> Vec<Value> {
    let mut start = Map::new();
    for key in ["id", "model", "usage", "container", "input_transformations"] {
        if let Some(value) = response.get(key).filter(|v| !v.is_null()) {
            start.insert(key.into(), value.clone());
        }
    }
    start.insert("stop_reason".into(), Value::Null);
    let mut events = vec![json!({"type": "message_start", "message": start})];
    let blocks = response
        .get("content")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    for (index, block) in blocks.iter().enumerate() {
        let kind = block
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let mut deltas = Vec::new();
        let opened = match kind {
            "text" => {
                deltas.push(json!({"type": "text_delta", "text": block.get("text").cloned().unwrap_or(json!(""))}));
                for citation in block
                    .get("citations")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    deltas.push(json!({"type": "citations_delta", "citation": citation}));
                }
                json!({"type": "text", "text": ""})
            }
            "thinking" => {
                deltas.push(json!({"type": "thinking_delta", "thinking": block.get("thinking").cloned().unwrap_or(json!(""))}));
                if let Some(signature) = block.get("signature").filter(|s| s.is_string()) {
                    deltas.push(json!({"type": "signature_delta", "signature": signature}));
                }
                json!({"type": "thinking", "thinking": ""})
            }
            "compaction" => {
                // The summary goes as a delta, as a streamed compaction sends
                // it, whether or not the block is signed.
                if let Some(content) = block.get("content").filter(|c| c.is_string()) {
                    deltas.push(json!({"type": "compaction_delta", "content": content}));
                }
                let mut opened = block.clone();
                if let Some(opened) = opened.as_object_mut() {
                    opened.remove("content");
                }
                opened
            }
            "tool_use" | "server_tool_use" => {
                let input = block.get("input").cloned().unwrap_or_else(|| json!({}));
                deltas.push(json!({"type": "input_json_delta", "partial_json": input.to_string()}));
                let mut opened = block.clone();
                opened["input"] = json!({});
                opened
            }
            _ => block.clone(),
        };
        events
            .push(json!({"type": "content_block_start", "index": index, "content_block": opened}));
        for delta in deltas {
            events.push(json!({"type": "content_block_delta", "index": index, "delta": delta}));
        }
        events.push(json!({"type": "content_block_stop", "index": index}));
    }
    let mut delta =
        json!({"stop_reason": response.get("stop_reason").cloned().unwrap_or(Value::Null)});
    for key in [
        "stop_sequence",
        "stop_details",
        "container",
        "safeguard_results",
    ] {
        if let Some(value) = response.get(key) {
            delta[key] = value.clone();
        }
    }
    let mut end = json!({"type": "message_delta", "delta": delta});
    if let Some(usage) = response.get("usage") {
        end["usage"] = usage.clone();
    }
    if let Some(management) = response.get("context_management").filter(|v| !v.is_null()) {
        end["context_management"] = management.clone();
    }
    events.push(end);
    events.push(json!({"type": "message_stop"}));
    events
}

fn run(
    source: futures::stream::BoxStream<'static, Result<SseJson, Error>>,
    provider: String,
    api: &'static ApiProfile,
    reading: Reading,
) -> EventStream {
    let documents = reading.documents.clone();
    let state = Parser {
        source,
        provider,
        api,
        reading,
        documents,
        blocks: HashMap::new(),
        tool_names: HashMap::new(),
        mcp: HashMap::new(),
        pending: VecDeque::from([Event::StreamStart(StreamStart::default())]),
        open: None,
        started: false,
        finished: false,
        response_id: None,
        response_model: None,
        stop_reason: None,
        raw_usage: Map::new(),
        input_tokens: 0,
        output_tokens: 0,
        cache_read: None,
        cache_write: None,
        thinking_tokens: None,
        iterations: None,
        stop_sequence: Value::Null,
        stop_details: None,
        input_transformations: None,
        safeguard_results: None,
        container: Value::Null,
        context_management: Value::Null,
        json_answer: false,
        sources: Vec::new(),
        source_keys: BTreeSet::new(),
        web_results: false,
        web_citations: Vec::new(),
    };
    stream::try_unfold(state, |mut parser| async move {
        loop {
            if let Some(event) = parser.pending.pop_front() {
                return Ok(Some((event, parser)));
            }
            if parser.finished {
                return Ok(None);
            }
            let value = match parser.source.next().await {
                Some(Ok(SseJson::Object(value))) => value,
                Some(Err(error)) => return Err(error),
                // The Messages protocol has no `[DONE]` marker; it is not JSON.
                Some(Ok(SseJson::Done)) => {
                    return Err(http::response_error(
                        &parser.provider,
                        format!("{} stream event is not valid JSON", parser.api.label),
                    ));
                }
                None => {
                    return Err(ProviderError::incomplete_response(
                        format!("{} stream ended without message_stop", parser.api.label),
                        Some(parser.provider),
                    )
                    .into());
                }
            };
            parser.feed(value)?;
        }
    })
    .boxed()
}

fn meta(md: Option<Value>) -> ModelEventMeta {
    ModelEventMeta {
        provider_metadata: md.map(metadata),
        ..Default::default()
    }
}

/// A caller in the metadata form `{type, toolId}`.
fn caller_info(block: &Value) -> Option<Value> {
    let caller = block.get("caller").filter(|c| c.is_object())?;
    let mut info = json!({"type": caller.get("type").cloned().unwrap_or(Value::Null)});
    if let Some(id) = caller.get("tool_id").filter(|id| !id.is_null()) {
        info["toolId"] = id.clone();
    }
    Some(info)
}

/// A JSON object's text with `"type": <kind>` first, for a code execution
/// subtool's input.
fn typed_input(kind: &str, input: &str) -> String {
    let rest = input.trim_start().strip_prefix('{').unwrap_or(input);
    let rest = rest.trim_start();
    if rest.starts_with('}') {
        format!("{{\"type\": \"{kind}\"{rest}")
    } else {
        format!("{{\"type\": \"{kind}\",{rest}")
    }
}

impl Parser {
    fn text(&self, value: &Value, key: &str) -> Result<String, Error> {
        value
            .get(key)
            .and_then(Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| {
                http::response_error(
                    &self.provider,
                    format!("{} stream event is missing string {key}", self.api.label),
                )
            })
    }
    fn index(&self, value: &Value) -> Result<u64, Error> {
        value.get("index").and_then(Value::as_u64).ok_or_else(|| {
            http::response_error(
                &self.provider,
                format!(
                    "{} stream event is missing a nonnegative index",
                    self.api.label
                ),
            )
        })
    }
    fn push(&mut self, event: Event) {
        self.pending.push_back(event);
    }
    fn custom(&self, provider: &str) -> String {
        self.reading.names.custom(provider).to_owned()
    }

    fn feed(&mut self, value: Value) -> Result<(), Error> {
        let kind = self.text(&value, "type")?;
        match kind.as_str() {
            "ping" => {}
            "message_start" => self.message_start(&value)?,
            "content_block_start" => self.block_start(&value)?,
            "content_block_delta" => self.block_delta(&value)?,
            "content_block_stop" => self.block_stop(&value)?,
            "message_delta" => self.message_delta(&value)?,
            "message_stop" => {
                if !self.started {
                    return Err(http::response_error(
                        &self.provider,
                        format!("{} stream has no message_start", self.api.label),
                    ));
                }
                let end = self.end()?;
                self.push(end);
                self.finished = true;
            }
            "error" => {
                return Err(errors::stream_error(
                    self.api,
                    &self.provider,
                    value.get("error").cloned().unwrap_or(value),
                ));
            }
            _ => {}
        }
        Ok(())
    }

    fn message_start(&mut self, value: &Value) -> Result<(), Error> {
        let message = &value["message"];
        let id = message.get("id").and_then(Value::as_str).map(str::to_owned);
        if let Some(open) = &self.open {
            if id.as_deref() == Some(open.as_str()) {
                return Ok(());
            }
            return Err(http::response_error(
                &self.provider,
                format!(
                    "{} stream started message {:?} while message {open:?} is still open",
                    self.api.label,
                    id.unwrap_or_default()
                ),
            ));
        }
        self.open = Some(id.clone().unwrap_or_default());
        self.response_id = id;
        self.response_model = message
            .get("model")
            .and_then(Value::as_str)
            .map(str::to_owned);
        self.stop_reason = message
            .get("stop_reason")
            .and_then(Value::as_str)
            .map(str::to_owned);
        let usage = &message["usage"];
        if let Some(usage) = usage.as_object() {
            self.raw_usage = usage.clone();
        }
        self.input_tokens = self.token(usage, "input_tokens")?.unwrap_or(0);
        self.cache_read = self.token(usage, "cache_read_input_tokens")?;
        self.cache_write = self.token(usage, "cache_creation_input_tokens")?;
        if let Some(t) = message
            .get("input_transformations")
            .filter(|v| !v.is_null())
        {
            self.input_transformations = Some(t.clone());
        }
        if let Some(container) = message.get("container").filter(|v| !v.is_null()) {
            self.container = json!({"expiresAt": container["expires_at"], "id": container["id"],
                "skills": Value::Null});
        }
        self.started = true;
        // A programmatic tool call's deferred calls may come in the start.
        for part in message
            .get("content")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if part["type"] != "tool_use" {
                continue;
            }
            let id = self.text(part, "id")?;
            let name = self.custom(&self.text(part, "name")?);
            let input = part
                .get("input")
                .cloned()
                .unwrap_or_else(|| json!({}))
                .to_string();
            let md = caller_info(part).map(|caller| json!({"caller": caller}));
            self.push(Event::ToolStart(ToolStart {
                tool_call_id: id.clone(),
                tool_name: name.clone(),
                ..Default::default()
            }));
            self.push(Event::ToolDelta(ToolDelta {
                tool_call_id: id.clone(),
                chunk: input.clone(),
                event: Default::default(),
            }));
            self.push(Event::ToolEnd(ToolEnd {
                tool_call_id: id.clone(),
                tool_call: ToolCallPart::new(id, name, input),
                event: meta(md),
            }));
        }
        Ok(())
    }

    fn token(&self, usage: &Value, key: &str) -> Result<Option<u64>, Error> {
        match usage.get(key) {
            None | Some(Value::Null) => Ok(None),
            Some(value) => value.as_u64().map(Some).ok_or_else(|| {
                http::response_error(
                    &self.provider,
                    format!(
                        "{} usage {key} must be a nonnegative integer",
                        self.api.label
                    ),
                )
            }),
        }
    }

    fn message_delta(&mut self, value: &Value) -> Result<(), Error> {
        let usage = &value["usage"];
        if !usage.is_null() && !usage.is_object() {
            return Err(http::response_error(
                &self.provider,
                format!("{} usage must be an object", self.api.label),
            ));
        }
        if let Some(input) = self.token(usage, "input_tokens")? {
            self.input_tokens = input;
        }
        if let Some(output) = self.token(usage, "output_tokens")? {
            self.output_tokens = output;
        }
        if let Some(details) = usage.get("output_tokens_details").filter(|d| !d.is_null()) {
            self.thinking_tokens = self.token(details, "thinking_tokens")?;
        }
        if let Some(read) = self.token(usage, "cache_read_input_tokens")? {
            self.cache_read = Some(read);
        }
        if let Some(write) = self.token(usage, "cache_creation_input_tokens")? {
            self.cache_write = Some(write);
        }
        if let Some(iterations) = usage.get("iterations").and_then(Value::as_array) {
            self.iterations = Some(iterations.clone());
        }
        if let Some(usage) = usage.as_object() {
            self.raw_usage.extend(usage.clone());
        }
        let delta = &value["delta"];
        if let Some(reason) = delta.get("stop_reason") {
            self.stop_reason = reason.as_str().map(str::to_owned);
        }
        self.stop_sequence = delta.get("stop_sequence").cloned().unwrap_or(Value::Null);
        self.stop_details = delta
            .get("stop_details")
            .filter(|d| d.is_object())
            .map(|details| {
                let mut out = json!({"type": details["type"]});
                for (from, to) in [
                    ("category", "category"),
                    ("explanation", "explanation"),
                    ("recommended_model", "recommendedModel"),
                ] {
                    if let Some(v) = details.get(from).filter(|v| !v.is_null()) {
                        out[to] = v.clone();
                    }
                }
                out
            });
        self.container = match delta.get("container").filter(|c| c.is_object()) {
            Some(container) => json!({
                "expiresAt": container["expires_at"],
                "id": container["id"],
                "skills": container.get("skills").and_then(Value::as_array).map(|skills| {
                    skills.iter().map(|s| json!({"type": s["type"], "skillId": s["skill_id"],
                        "version": s["version"]})).collect::<Vec<_>>()
                }),
            }),
            None => Value::Null,
        };
        if let Some(management) = value.get("context_management").filter(|m| m.is_object()) {
            let edits: Vec<Value> = management["applied_edits"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|edit| match edit["type"].as_str()? {
                    "clear_tool_uses_20250919" => Some(json!({"type": edit["type"],
                        "clearedToolUses": edit["cleared_tool_uses"],
                        "clearedInputTokens": edit["cleared_input_tokens"]})),
                    "clear_thinking_20251015" => Some(json!({"type": edit["type"],
                        "clearedThinkingTurns": edit["cleared_thinking_turns"],
                        "clearedInputTokens": edit["cleared_input_tokens"]})),
                    "compact_20260112" => Some(json!({"type": edit["type"]})),
                    _ => None,
                })
                .collect();
            self.context_management = json!({"appliedEdits": edits});
        }
        if let Some(t) = value.get("input_transformations").filter(|v| !v.is_null()) {
            self.input_transformations = Some(t.clone());
        }
        if let Some(results) = delta.get("safeguard_results").filter(|v| !v.is_null()) {
            self.safeguard_results = Some(results.clone());
        }
        Ok(())
    }

    fn add_source(&mut self, key: String, source: Value) {
        if self.source_keys.insert(key) {
            self.sources.push(source);
        }
    }
    /// The source of a citation (`createCitationSource`).
    fn citation_source(&mut self, citation: &Value) {
        match citation["type"].as_str() {
            Some("web_search_result_location") => {
                if let Some(url) = citation["url"].as_str() {
                    let mut source =
                        json!({"sourceType": "url", "id": generate_id("source"), "url": url});
                    if let Some(title) = citation.get("title").filter(|t| !t.is_null()) {
                        source["title"] = title.clone();
                    }
                    self.add_source(format!("url:{url}"), source);
                }
            }
            Some("page_location" | "char_location") => {
                let Some(document) = citation["document_index"]
                    .as_u64()
                    .and_then(|i| self.documents.get(i as usize))
                    .cloned()
                else {
                    return;
                };
                let title = citation
                    .get("document_title")
                    .and_then(Value::as_str)
                    .unwrap_or(&document.title)
                    .to_owned();
                let mut source = json!({"sourceType": "document", "id": generate_id("source"),
                    "mediaType": document.media_type, "title": title});
                if let Some(filename) = &document.filename {
                    source["filename"] = json!(filename);
                }
                let key = format!(
                    "document:{}:{}:{}",
                    title,
                    document.filename.clone().unwrap_or_default(),
                    document.media_type
                );
                self.add_source(key, source);
            }
            _ => {}
        }
    }

    fn block_start(&mut self, value: &Value) -> Result<(), Error> {
        let idx = self.index(value)?;
        let wire = &value["content_block"];
        let kind = self.text(wire, "type")?;
        let block_id = idx.to_string();
        let block = match kind.as_str() {
            "text" if self.reading.json_tool => Kind::Dropped,
            "text" => {
                self.push(Event::TextStart(TextStart {
                    block_id,
                    ..Default::default()
                }));
                Kind::Text {
                    citations: Vec::new(),
                    metadata: None,
                }
            }
            "thinking" => {
                self.push(Event::ReasoningStart(ReasoningStart {
                    block_id,
                    ..Default::default()
                }));
                Kind::Reasoning {
                    signature: None,
                    redacted: None,
                }
            }
            "redacted_thinking" => {
                let data = wire
                    .get("data")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned();
                self.push(Event::ReasoningStart(ReasoningStart {
                    block_id,
                    event: meta(Some(json!({"redactedData": data}))),
                }));
                Kind::Reasoning {
                    signature: None,
                    redacted: Some(data),
                }
            }
            "compaction" => {
                let mut md = json!({"type": "compaction"});
                if let Some(signature) = wire.get("signature").filter(|s| !s.is_null()) {
                    md["signature"] = signature.clone();
                }
                self.push(Event::TextStart(TextStart {
                    block_id: block_id.clone(),
                    event: meta(Some(md.clone())),
                }));
                // An on-demand compaction arrives complete in its start.
                if !wire["signature"].is_null()
                    && let Some(content) = wire["content"].as_str().filter(|c| !c.is_empty())
                {
                    self.push(Event::TextDelta(TextDelta {
                        block_id,
                        chunk: content.to_owned(),
                        event: Default::default(),
                    }));
                }
                Kind::Text {
                    citations: Vec::new(),
                    metadata: Some(md),
                }
            }
            "fallback" | "container_upload" => {
                let md = if kind == "fallback" {
                    json!({"type": "fallback", "from": wire["from"], "to": wire["to"]})
                } else {
                    json!({"type": "container_upload", "fileId": wire["file_id"]})
                };
                self.push(Event::TextStart(TextStart {
                    block_id: block_id.clone(),
                    event: meta(Some(md.clone())),
                }));
                self.push(Event::TextEnd(TextEnd {
                    block_id,
                    event: meta(Some(md)),
                }));
                Kind::Dropped
            }
            "tool_use"
                if self.reading.json_tool
                    && wire.get("name").and_then(Value::as_str) == Some(JSON_TOOL) =>
            {
                self.json_answer = true;
                self.push(Event::TextStart(TextStart {
                    block_id,
                    ..Default::default()
                }));
                Kind::JsonTool
            }
            "tool_use" => {
                let id = self.text(wire, "id")?;
                let name = self.text(wire, "name")?;
                let caller = caller_info(wire);
                let initial = wire
                    .get("input")
                    .and_then(Value::as_object)
                    .filter(|input| !input.is_empty())
                    .map(|input| Value::Object(input.clone()).to_string())
                    .unwrap_or_default();
                let toolset = wire
                    .get("toolset_name")
                    .and_then(Value::as_str)
                    .map(|toolset| (toolset.to_owned(), name.clone()));
                let tool_name = match &toolset {
                    Some((toolset, _)) => self.custom(toolset),
                    None => self.custom(&name),
                };
                self.push(Event::ToolStart(ToolStart {
                    tool_call_id: id.clone(),
                    tool_name,
                    ..Default::default()
                }));
                let mut metadata = Map::new();
                if let Some((toolset, _)) = &toolset {
                    metadata.insert("toolsetName".into(), json!(toolset));
                }
                if let Some(caller) = &caller {
                    metadata.insert("caller".into(), caller.clone());
                }
                if toolset.is_none() && !initial.is_empty() {
                    self.push(Event::ToolDelta(ToolDelta {
                        tool_call_id: id.clone(),
                        chunk: initial.clone(),
                        event: Default::default(),
                    }));
                }
                Kind::Call {
                    id,
                    toolset,
                    input: initial,
                    metadata: (!metadata.is_empty()).then_some(Value::Object(metadata)),
                }
            }
            "server_tool_use" => self.server_tool_start(wire)?,
            "mcp_tool_use" => {
                let id = self.text(wire, "id")?;
                let name = self.text(wire, "name")?;
                let md = json!({"type": "mcp-tool-use", "serverName": wire["server_name"]});
                let input = wire
                    .get("input")
                    .cloned()
                    .unwrap_or_else(|| json!({}))
                    .to_string();
                self.tool_names.insert(id.clone(), name.clone());
                self.mcp.insert(id.clone(), md.clone());
                self.push(Event::BuiltinToolStart(BuiltinToolStart {
                    tool_call_id: id.clone(),
                    tool_name: name.clone(),
                    event: meta(Some(md.clone())),
                }));
                self.push(Event::BuiltinToolDelta(BuiltinToolDelta {
                    tool_call_id: id.clone(),
                    chunk: input.clone(),
                    event: Default::default(),
                }));
                self.push(Event::BuiltinToolEnd(BuiltinToolEnd {
                    tool_call_id: id.clone(),
                    tool_call: BuiltinToolCallPart {
                        id: generate_id("part"),
                        tool_call_id: id,
                        tool_name: name,
                        tool_args: input,
                        provider_metadata: Some(metadata(md.clone())),
                    },
                    event: meta(Some(md)),
                }));
                Kind::Dropped
            }
            "mcp_tool_result" => {
                let id = wire
                    .get("tool_use_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned();
                let md = self.mcp.get(&id).cloned();
                let name = self
                    .tool_names
                    .get(&id)
                    .cloned()
                    .unwrap_or_else(|| "mcp".into());
                self.result(
                    &id,
                    name,
                    wire.get("content").cloned().unwrap_or(Value::Null),
                    wire["is_error"].as_bool().unwrap_or(false),
                    md,
                );
                Kind::Dropped
            }
            result if result.ends_with("_tool_result") => {
                self.tool_result(result, wire);
                Kind::Dropped
            }
            _ => Kind::Dropped,
        };
        if self.blocks.insert(idx, block).is_some() {
            return Err(http::response_error(
                &self.provider,
                format!(
                    "{} stream has duplicate content block index",
                    self.api.label
                ),
            ));
        }
        Ok(())
    }

    fn server_tool_start(&mut self, wire: &Value) -> Result<Kind, Error> {
        let id = self.text(wire, "id")?;
        let name = self.text(wire, "name")?;
        let (provider_name, input_type) = match name.as_str() {
            "text_editor_code_execution" | "bash_code_execution" => {
                ("code_execution".to_owned(), Some(name.clone()))
            }
            other => (other.to_owned(), None),
        };
        let tool_name = self.custom(&provider_name);
        self.tool_names.insert(id.clone(), tool_name.clone());
        let caller = caller_info(wire);
        let mut md = Map::new();
        if let Some(caller) = &caller {
            md.insert("caller".into(), caller.clone());
        }
        self.push(Event::BuiltinToolStart(BuiltinToolStart {
            tool_call_id: id.clone(),
            tool_name,
            event: meta((!md.is_empty()).then(|| Value::Object(md.clone()))),
        }));
        let initial = wire
            .get("input")
            .and_then(Value::as_object)
            .filter(|input| !input.is_empty())
            .map(|input| Value::Object(input.clone()).to_string());
        // A plain code execution input is held to mark a programmatic call.
        let held = (provider_name == "code_execution" && input_type.is_none())
            .then(|| initial.clone().unwrap_or_default());
        if held.is_none()
            && let Some(initial) = &initial
        {
            let chunk = match &input_type {
                Some(kind) => typed_input(kind, initial),
                None => initial.clone(),
            };
            self.push(Event::BuiltinToolDelta(BuiltinToolDelta {
                tool_call_id: id.clone(),
                chunk,
                event: Default::default(),
            }));
        }
        Ok(Kind::Builtin {
            id,
            first: initial.is_none(),
            input_type,
            held,
            caller,
        })
    }

    /// A built-in result of a `*_tool_result` block, in TypeScript's shapes.
    fn tool_result(&mut self, kind: &str, wire: &Value) {
        let id = wire
            .get("tool_use_id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        let content = wire.get("content").cloned().unwrap_or(Value::Null);
        let caller = caller_info(wire).map(|caller| json!({"caller": caller}));
        let error =
            |content: &Value| json!({"type": content["type"], "errorCode": content["error_code"]});
        let (provider, result, is_error, md) = match kind {
            "web_fetch_tool_result" => match content["type"].as_str() {
                Some("web_fetch_result") => {
                    let document = &content["content"];
                    let source = &document["source"];
                    self.documents.push(Document {
                        title: document["title"]
                            .as_str()
                            .or(content["url"].as_str())
                            .unwrap_or_default()
                            .to_owned(),
                        filename: None,
                        media_type: source["media_type"].as_str().unwrap_or_default().to_owned(),
                    });
                    // A field Anthropic leaves out stays out.
                    let pick = |value: &Value, keys: &[(&str, &str)]| {
                        let mut out = Map::new();
                        for (from, to) in keys {
                            if let Some(v) = value.get(*from) {
                                out.insert((*to).into(), v.clone());
                            }
                        }
                        out
                    };
                    let mut inner = pick(
                        document,
                        &[
                            ("type", "type"),
                            ("title", "title"),
                            ("citations", "citations"),
                        ],
                    );
                    inner.insert(
                        "source".into(),
                        Value::Object(pick(
                            source,
                            &[
                                ("type", "type"),
                                ("media_type", "mediaType"),
                                ("data", "data"),
                            ],
                        )),
                    );
                    let mut result =
                        pick(&content, &[("url", "url"), ("retrieved_at", "retrievedAt")]);
                    result.insert("type".into(), json!("web_fetch_result"));
                    result.insert("content".into(), Value::Object(inner));
                    ("web_fetch", Value::Object(result), false, caller)
                }
                _ => ("web_fetch", error(&content), true, caller),
            },
            "web_search_tool_result" => match content.as_array() {
                Some(results) => {
                    let results = results.clone();
                    for result in &results {
                        self.web_results = true;
                        if let Some(url) = result["url"].as_str() {
                            let mut source = json!({"sourceType": "url", "id": generate_id("source"), "url": url});
                            if let Some(title) = result.get("title").filter(|t| !t.is_null()) {
                                source["title"] = title.clone();
                            }
                            self.add_source(format!("url:{url}"), source);
                        }
                    }
                    (
                        "web_search",
                        Value::Array(
                            results
                                .iter()
                                .map(|r| {
                                    let mut out = json!({"url": r["url"]});
                                    if let Some(title) = r.get("title").filter(|t| !t.is_null()) {
                                        out["title"] = title.clone();
                                    }
                                    out["pageAge"] =
                                        r.get("page_age").cloned().unwrap_or(Value::Null);
                                    // A field Anthropic leaves out stays out.
                                    if let Some(content) = r.get("encrypted_content") {
                                        out["encryptedContent"] = content.clone();
                                    }
                                    if let Some(kind) = r.get("type") {
                                        out["type"] = kind.clone();
                                    }
                                    out
                                })
                                .collect(),
                        ),
                        false,
                        caller,
                    )
                }
                None => ("web_search", error(&content), true, caller),
            },
            "code_execution_tool_result" => match content["type"].as_str() {
                Some("code_execution_result") => (
                    "code_execution",
                    json!({"type": "code_execution_result", "stdout": content["stdout"],
                        "stderr": content["stderr"], "return_code": content["return_code"],
                        "content": content.get("content").filter(|c| !c.is_null()).cloned().unwrap_or(json!([]))}),
                    false,
                    None,
                ),
                Some("encrypted_code_execution_result") => (
                    "code_execution",
                    json!({"type": "encrypted_code_execution_result",
                        "encrypted_stdout": content["encrypted_stdout"], "stderr": content["stderr"],
                        "return_code": content["return_code"],
                        "content": content.get("content").filter(|c| !c.is_null()).cloned().unwrap_or(json!([]))}),
                    false,
                    None,
                ),
                _ => ("code_execution", error(&content), true, None),
            },
            "bash_code_execution_tool_result" | "text_editor_code_execution_tool_result" => {
                ("code_execution", content.clone(), false, None)
            }
            "tool_search_tool_result" => {
                let provider = self
                    .tool_names
                    .get(&id)
                    .map(|name| self.reading.names.provider(name).to_owned())
                    .filter(|name| name.starts_with("tool_search_tool_"))
                    .unwrap_or_else(|| {
                        if self.custom("tool_search_tool_bm25") != "tool_search_tool_bm25" {
                            "tool_search_tool_bm25".into()
                        } else {
                            "tool_search_tool_regex".into()
                        }
                    });
                let name = self.custom(&provider);
                let (result, is_error) = if content["type"] == "tool_search_tool_search_result" {
                    (
                        Value::Array(
                            content["tool_references"]
                                .as_array()
                                .into_iter()
                                .flatten()
                                .map(|r| json!({"type": r["type"], "toolName": r["tool_name"]}))
                                .collect(),
                        ),
                        false,
                    )
                } else {
                    (error(&content), true)
                };
                self.result(&id, name, result, is_error, None);
                return;
            }
            "advisor_tool_result" => {
                let mut result = match content["type"].as_str() {
                    Some("advisor_result") => {
                        json!({"type": "advisor_result", "text": content["text"]})
                    }
                    Some("advisor_redacted_result") => json!({"type": "advisor_redacted_result",
                        "encryptedContent": content["encrypted_content"]}),
                    _ => error(&content),
                };
                let is_error = result["type"] != "advisor_result"
                    && result["type"] != "advisor_redacted_result";
                if !is_error
                    && let Some(reason) = content.get("stop_reason").filter(|r| !r.is_null())
                {
                    result["stopReason"] = reason.clone();
                }
                ("advisor", result, is_error, None)
            }
            _ => return,
        };
        let name = self.custom(provider);
        self.result(&id, name, result, is_error, md);
    }

    fn result(&mut self, id: &str, name: String, result: Value, is_error: bool, md: Option<Value>) {
        self.push(Event::BuiltinToolResult(BuiltinToolResult {
            tool_call_id: id.to_owned(),
            result: BuiltinToolReturnPart {
                id: generate_id("part"),
                tool_call_id: id.to_owned(),
                tool_name: name,
                result,
                is_error,
                provider_metadata: md.clone().map(metadata),
            },
            event: meta(md),
        }));
    }

    fn block_delta(&mut self, value: &Value) -> Result<(), Error> {
        let idx = self.index(value)?;
        let delta = &value["delta"];
        let kind = self.text(delta, "type")?;
        let block_id = idx.to_string();
        let Some(block) = self.blocks.get_mut(&idx) else {
            return Ok(());
        };
        let mut event = None;
        match (kind.as_str(), block) {
            ("text_delta", Kind::Text { .. }) => {
                let chunk = delta
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned();
                event = Some(Event::TextDelta(TextDelta {
                    chunk,
                    block_id,
                    event: Default::default(),
                }));
            }
            ("compaction_delta", Kind::Text { .. }) => {
                if let Some(chunk) = delta.get("content").and_then(Value::as_str) {
                    event = Some(Event::TextDelta(TextDelta {
                        chunk: chunk.to_owned(),
                        block_id,
                        event: Default::default(),
                    }));
                }
            }
            ("citations_delta", Kind::Text { citations, .. }) => {
                let citation = delta.get("citation").cloned().unwrap_or(Value::Null);
                citations.push(citation.clone());
                if citation["type"] == "web_search_result_location" {
                    self.web_citations.push(citation);
                } else {
                    self.citation_source(&citation);
                }
            }
            ("citations_delta", _) => {
                let citation = delta.get("citation").cloned().unwrap_or(Value::Null);
                if citation["type"] == "web_search_result_location" {
                    self.web_citations.push(citation);
                } else {
                    self.citation_source(&citation);
                }
            }
            ("thinking_delta", Kind::Reasoning { .. }) => {
                let chunk = delta
                    .get("thinking")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned();
                event = Some(Event::ReasoningDelta(ReasoningDelta {
                    chunk,
                    block_id,
                    event: Default::default(),
                }));
            }
            ("signature_delta", Kind::Reasoning { signature, .. }) => {
                if let Some(text) = delta.get("signature").and_then(Value::as_str) {
                    signature.get_or_insert_default().push_str(text);
                }
            }
            ("input_json_delta", block) => {
                let chunk = delta
                    .get("partial_json")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                // An empty delta is skipped, so that a subtool's type can go
                // first in the first one.
                if chunk.is_empty() {
                    return Ok(());
                }
                match block {
                    Kind::JsonTool => {
                        event = Some(Event::TextDelta(TextDelta {
                            chunk: chunk.to_owned(),
                            block_id,
                            event: Default::default(),
                        }));
                    }
                    Kind::Call {
                        id, toolset, input, ..
                    } => {
                        input.push_str(chunk);
                        if toolset.is_none() {
                            event = Some(Event::ToolDelta(ToolDelta {
                                chunk: chunk.to_owned(),
                                tool_call_id: id.clone(),
                                event: Default::default(),
                            }));
                        }
                    }
                    Kind::Builtin {
                        id,
                        input_type,
                        first,
                        held,
                        ..
                    } => {
                        if let Some(held) = held {
                            held.push_str(chunk);
                        } else {
                            let chunk = match (&input_type, *first) {
                                (Some(kind), true) => typed_input(kind, chunk),
                                _ => chunk.to_owned(),
                            };
                            *first = false;
                            event = Some(Event::BuiltinToolDelta(BuiltinToolDelta {
                                chunk,
                                tool_call_id: id.clone(),
                                event: Default::default(),
                            }));
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }
        if let Some(event) = event {
            self.push(event);
        }
        Ok(())
    }

    fn block_stop(&mut self, value: &Value) -> Result<(), Error> {
        let idx = self.index(value)?;
        let Some(block) = self.blocks.remove(&idx) else {
            return Ok(());
        };
        let block_id = idx.to_string();
        match block {
            Kind::Text {
                citations,
                metadata: md,
            } => {
                let md = if citations.is_empty() {
                    md
                } else {
                    let mut md = md.unwrap_or_else(|| json!({}));
                    md["citations"] = Value::Array(citations);
                    Some(md)
                };
                self.push(Event::TextEnd(TextEnd {
                    block_id,
                    event: meta(md),
                }));
            }
            Kind::JsonTool => self.push(Event::TextEnd(TextEnd {
                block_id,
                ..Default::default()
            })),
            Kind::Reasoning {
                signature,
                redacted,
            } => {
                let md = match (signature, redacted) {
                    (Some(signature), _) => Some(json!({"signature": signature})),
                    (None, Some(data)) => Some(json!({"redactedData": data})),
                    (None, None) => None,
                };
                self.push(Event::ReasoningEnd(ReasoningEnd {
                    block_id,
                    event: meta(md),
                }));
            }
            Kind::Call {
                id,
                toolset,
                input,
                metadata: md,
            } => {
                if let Some((_, member)) = &toolset {
                    // A toolset member's call is one call of the toolset tool
                    // with the member's name as `action`.
                    let parsed = if input.is_empty() {
                        Some(json!({}))
                    } else {
                        serde_json::from_str::<Value>(&input).ok()
                    };
                    let chunk = match parsed {
                        Some(Value::Object(mut fields)) => {
                            let mut out = Map::new();
                            out.insert("action".into(), json!(member));
                            fields.remove("action");
                            out.extend(fields);
                            Value::Object(out).to_string()
                        }
                        Some(_) => json!({"action": member}).to_string(),
                        None => input.clone(),
                    };
                    self.push(Event::ToolDelta(ToolDelta {
                        tool_call_id: id.clone(),
                        chunk,
                        event: Default::default(),
                    }));
                } else if input.is_empty() {
                    self.push(Event::ToolDelta(ToolDelta {
                        tool_call_id: id.clone(),
                        chunk: "{}".into(),
                        event: Default::default(),
                    }));
                }
                self.push(Event::ToolEnd(ToolEnd {
                    tool_call_id: id.clone(),
                    tool_call: ToolCallPart::new(id, "", ""),
                    event: meta(md),
                }));
            }
            Kind::Builtin {
                id,
                held,
                first,
                caller,
                ..
            } => {
                if let Some(held) = held {
                    let input = if held.is_empty() {
                        "{}".to_owned()
                    } else {
                        held
                    };
                    // A `{code}` input without a type is a programmatic call.
                    let chunk = match serde_json::from_str::<Value>(&input) {
                        Ok(Value::Object(fields))
                            if fields.contains_key("code") && !fields.contains_key("type") =>
                        {
                            let mut out = Map::new();
                            out.insert("type".into(), json!("programmatic-tool-call"));
                            out.extend(fields);
                            Value::Object(out).to_string()
                        }
                        _ => input,
                    };
                    self.push(Event::BuiltinToolDelta(BuiltinToolDelta {
                        tool_call_id: id.clone(),
                        chunk,
                        event: Default::default(),
                    }));
                } else if first {
                    self.push(Event::BuiltinToolDelta(BuiltinToolDelta {
                        tool_call_id: id.clone(),
                        chunk: "{}".into(),
                        event: Default::default(),
                    }));
                }
                let md = caller.map(|caller| json!({"caller": caller}));
                self.push(Event::BuiltinToolEnd(BuiltinToolEnd {
                    tool_call_id: id.clone(),
                    tool_call: BuiltinToolCallPart {
                        id: generate_id("part"),
                        tool_call_id: id,
                        tool_name: String::new(),
                        tool_args: String::new(),
                        provider_metadata: md.clone().map(metadata),
                    },
                    event: meta(md),
                }));
            }
            Kind::Dropped => {}
        }
        Ok(())
    }

    fn end(&mut self) -> Result<Event, Error> {
        if !self.web_results {
            for citation in std::mem::take(&mut self.web_citations) {
                self.citation_source(&citation);
            }
        }
        let iterations = self.iterations.clone().unwrap_or_default();
        let by_fallback = iterations.iter().any(|i| i["type"] == "fallback_message");
        let executor: Vec<&Value> = iterations
            .iter()
            .filter(|i| matches!(i["type"].as_str(), Some("compaction" | "message")))
            .collect();
        let (input, output) = if !by_fallback && !executor.is_empty() {
            executor.iter().fold((0u64, 0u64), |(i, o), it| {
                (
                    i.saturating_add(it["input_tokens"].as_u64().unwrap_or(0)),
                    o.saturating_add(it["output_tokens"].as_u64().unwrap_or(0)),
                )
            })
        } else {
            (self.input_tokens, self.output_tokens)
        };
        let total_input = input
            .checked_add(self.cache_read.unwrap_or(0))
            .and_then(|t| t.checked_add(self.cache_write.unwrap_or(0)))
            .ok_or_else(|| {
                http::response_error(
                    &self.provider,
                    format!("{} input token usage overflowed", self.api.label),
                )
            })?;
        let usage = Usage {
            input_tokens: total_input,
            output_tokens: output,
            cache_read_tokens: self.cache_read,
            cache_write_tokens: self.cache_write,
            reasoning_tokens: self.thinking_tokens,
            raw: (!self.raw_usage.is_empty()).then(|| json!(self.raw_usage)),
        };
        let finish = self.stop_reason.as_deref().map(|r| match r {
            "end_turn" | "stop_sequence" | "pause_turn" => "stop",
            "max_tokens" | "model_context_window_exceeded" => "length",
            // The JSON tool's call is the answer, not a tool call.
            "tool_use" if self.json_answer => "stop",
            "tool_use" => "tool_call",
            "refusal" => "content_filter",
            _ => "other",
        });
        let iterations = self.iterations.as_ref().map(|iterations| {
            iterations
                .iter()
                .map(|i| {
                    let mut out = json!({"type": i["type"], "inputTokens": i["input_tokens"],
                        "outputTokens": i["output_tokens"]});
                    if let Some(model) = i.get("model").filter(|m| !m.is_null()) {
                        out["model"] = model.clone();
                    }
                    if i["cache_creation_input_tokens"].as_u64().unwrap_or(0) > 0 {
                        out["cacheCreationInputTokens"] = i["cache_creation_input_tokens"].clone();
                    }
                    if i["cache_read_input_tokens"].as_u64().unwrap_or(0) > 0 {
                        out["cacheReadInputTokens"] = i["cache_read_input_tokens"].clone();
                    }
                    out
                })
                .collect::<Vec<_>>()
        });
        let mut md = json!({
            "stopReason": self.stop_reason,
            "usage": if self.raw_usage.is_empty() { Value::Null } else { json!(self.raw_usage) },
            "stopSequence": self.stop_sequence,
            "iterations": iterations,
            "container": self.container,
            "contextManagement": self.context_management,
        });
        if let Some(details) = &self.stop_details {
            md["stopDetails"] = details.clone();
        }
        if let Some(t) = &self.input_transformations {
            md["inputTransformations"] = t.clone();
        }
        if let Some(results) = &self.safeguard_results {
            md["safeguardResults"] = results.clone();
        }
        if !self.sources.is_empty() {
            md["sources"] = Value::Array(self.sources.clone());
        }
        Ok(Event::StreamEnd(StreamEnd {
            event: ModelEventMeta {
                usage: Some(usage),
                provider_metadata: Some(metadata(md)),
                ..Default::default()
            },
            finish_reason: finish.map(str::to_owned),
            response_id: self.response_id.clone(),
            response_model: self.response_model.clone(),
        }))
    }
}

/// The kind and retry flag of an `error` event's type
/// (`getAnthropicStreamErrorMetadata`).
pub(super) fn error_status(kind: &str) -> Option<(u16, bool)> {
    Some(match kind {
        "api_error" => (500, true),
        "overloaded_error" => (529, true),
        "rate_limit_error" => (429, true),
        "request_too_large" => (413, false),
        "authentication_error" => (401, false),
        "permission_error" => (403, false),
        "not_found_error" => (404, false),
        "billing_error" | "invalid_request_error" => (400, false),
        _ => return None,
    })
}
