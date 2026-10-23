use super::{errors, messages::metadata};
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
use std::collections::{HashMap, VecDeque};

struct Block {
    kind: String,
    id: String,
    name: String,
    signature: Option<String>,
    wire: Value,
}
struct Parser {
    source: futures::stream::BoxStream<'static, Result<SseJson, Error>>,
    provider: String,
    api: &'static ApiProfile,
    blocks: HashMap<u64, Block>,
    tool_names: HashMap<String, String>,
    pending: VecDeque<Event>,
    started: bool,
    finished: bool,
    response_id: Option<String>,
    response_model: Option<String>,
    stop_reason: Option<String>,
    usage: Map<String, Value>,
    /// The output schema went out as the JSON tool: its input is the text,
    /// and other text is dropped.
    json_tool: bool,
    /// The JSON tool answered.
    json_answer: bool,
}
/// Read a Messages event stream. With `json_tool`, the input of the tool
/// named `json` is the answer's text (`SchemaMode::JsonTool`).
pub(crate) fn parse(
    body: Body,
    provider: String,
    api: &'static ApiProfile,
    json_tool: bool,
) -> EventStream {
    run(
        http::sse_json(*api, provider.clone(), body),
        provider,
        api,
        json_tool,
    )
}

/// A buffered Messages response read as the stream of the same message,
/// so that both paths give the same events.
pub(crate) fn parse_message(
    response: &Value,
    provider: String,
    api: &'static ApiProfile,
    json_tool: bool,
) -> EventStream {
    let events: Vec<Result<SseJson, Error>> = message_events(response)
        .into_iter()
        .map(|event| Ok(SseJson::Object(event)))
        .collect();
    run(stream::iter(events).boxed(), provider, api, json_tool)
}

/// The stream events of a buffered message: its start, each content block's
/// start, delta and stop, and its end.
fn message_events(response: &Value) -> Vec<Value> {
    let mut start = Map::new();
    for key in ["id", "model", "usage"] {
        if let Some(value) = response.get(key) {
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
                json!({"type": "text", "text": ""})
            }
            "thinking" => {
                deltas.push(json!({"type": "thinking_delta", "thinking": block.get("thinking").cloned().unwrap_or(json!(""))}));
                if let Some(signature) = block.get("signature").filter(|s| s.is_string()) {
                    deltas.push(json!({"type": "signature_delta", "signature": signature}));
                }
                json!({"type": "thinking", "thinking": ""})
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
    events.push(json!({
        "type": "message_delta",
        "delta": {"stop_reason": response.get("stop_reason").cloned().unwrap_or(Value::Null)},
    }));
    events.push(json!({"type": "message_stop"}));
    events
}

fn run(
    source: futures::stream::BoxStream<'static, Result<SseJson, Error>>,
    provider: String,
    api: &'static ApiProfile,
    json_tool: bool,
) -> EventStream {
    let state = Parser {
        source,
        provider,
        api,
        blocks: HashMap::new(),
        tool_names: HashMap::new(),
        pending: VecDeque::from([Event::StreamStart(StreamStart::default())]),
        started: false,
        finished: false,
        response_id: None,
        response_model: None,
        stop_reason: None,
        usage: Map::new(),
        json_tool,
        json_answer: false,
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
    fn feed(&mut self, value: Value) -> Result<(), Error> {
        let kind = self.text(&value, "type")?;
        match kind.as_str() {
            "message_start" => {
                if self.started {
                    return Err(http::response_error(
                        &self.provider,
                        format!("{} stream has duplicate message_start", self.api.label),
                    ));
                }
                let message = &value["message"];
                self.response_id = Some(self.text(message, "id")?);
                self.response_model = Some(self.text(message, "model")?);
                self.stop_reason = message
                    .get("stop_reason")
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                self.update_usage(&message["usage"])?;
                self.started = true;
            }
            "content_block_start" => self.block_start(&value)?,
            "content_block_delta" => self.block_delta(&value)?,
            "content_block_stop" => self.block_stop(&value)?,
            "message_delta" => {
                if let Some(reason) = value.get("delta").and_then(|d| d.get("stop_reason")) {
                    self.stop_reason = if reason.is_null() {
                        None
                    } else {
                        Some(self.text(&value["delta"], "stop_reason")?)
                    };
                }
                self.update_usage(&value["usage"])?;
            }
            "message_stop" => {
                if !self.started {
                    return Err(http::response_error(
                        &self.provider,
                        format!("{} stream has no message_start", self.api.label),
                    ));
                }
                self.pending.push_back(self.end()?);
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
    fn update_usage(&mut self, usage: &Value) -> Result<(), Error> {
        if usage.is_null() {
            return Ok(());
        }
        let object = usage.as_object().ok_or_else(|| {
            http::response_error(
                &self.provider,
                format!("{} usage must be an object", self.api.label),
            )
        })?;
        for (key, value) in object {
            if value.is_null() {
                continue;
            }
            if matches!(
                key.as_str(),
                "input_tokens"
                    | "output_tokens"
                    | "cache_read_input_tokens"
                    | "cache_creation_input_tokens"
            ) && value.as_u64().is_none()
            {
                return Err(http::response_error(
                    &self.provider,
                    format!(
                        "{} usage {key} must be a nonnegative integer",
                        self.api.label
                    ),
                ));
            }
            self.usage.insert(key.clone(), value.clone());
        }
        Ok(())
    }
    fn block_start(&mut self, value: &Value) -> Result<(), Error> {
        let idx = self.index(value)?;
        let wire = &value["content_block"];
        let kind = self.text(wire, "type")?;
        let mut block = Block {
            kind: kind.clone(),
            id: String::new(),
            name: String::new(),
            signature: None,
            wire: wire.clone(),
        };
        let block_id = idx.to_string();
        let event = match kind.as_str() {
            // The JSON tool's input is the answer; other text is dropped.
            "text" if self.json_tool => {
                block.kind = "dropped_text".into();
                None
            }
            "tool_use"
                if self.json_tool
                    && wire.get("name").and_then(Value::as_str)
                        == Some(super::protocol::JSON_TOOL) =>
            {
                block.kind = "json_tool".into();
                self.json_answer = true;
                Some(Event::TextStart(TextStart {
                    block_id,
                    ..Default::default()
                }))
            }
            "text" => Some(Event::TextStart(TextStart {
                block_id,
                ..Default::default()
            })),
            "thinking" => Some(Event::ReasoningStart(ReasoningStart {
                block_id,
                ..Default::default()
            })),
            "tool_use" | "server_tool_use" => {
                block.id = self.text(wire, "id")?;
                block.name = self.text(wire, "name")?;
                if kind == "server_tool_use" {
                    self.tool_names.insert(block.id.clone(), block.name.clone());
                    Some(Event::BuiltinToolStart(BuiltinToolStart {
                        tool_call_id: block.id.clone(),
                        tool_name: block.name.clone(),
                        event: ModelEventMeta {
                            provider_metadata: Some(metadata(json!({}))),
                            ..Default::default()
                        },
                    }))
                } else {
                    Some(Event::ToolStart(ToolStart {
                        tool_call_id: block.id.clone(),
                        tool_name: block.name.clone(),
                        ..Default::default()
                    }))
                }
            }
            _ => None,
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
        if let Some(event) = event {
            self.pending.push_back(event);
        }
        Ok(())
    }
    fn block_delta(&mut self, value: &Value) -> Result<(), Error> {
        let idx = self.index(value)?;
        let delta = &value["delta"];
        let kind = self.text(delta, "type")?;
        let block_id = idx.to_string();
        let event = match kind.as_str() {
            "text_delta"
                if self
                    .blocks
                    .get(&idx)
                    .is_some_and(|block| block.kind == "dropped_text") =>
            {
                None
            }
            "text_delta" => Some(Event::TextDelta(TextDelta {
                chunk: self.text(delta, "text")?,
                block_id,
                event: Default::default(),
            })),
            "thinking_delta" => Some(Event::ReasoningDelta(ReasoningDelta {
                chunk: self.text(delta, "thinking")?,
                block_id,
                event: Default::default(),
            })),
            "signature_delta" => {
                let text = self.text(delta, "signature")?;
                if let Some(b) = self.blocks.get_mut(&idx) {
                    b.signature.get_or_insert_default().push_str(&text);
                }
                None
            }
            "input_json_delta"
                if self
                    .blocks
                    .get(&idx)
                    .is_some_and(|block| block.kind == "json_tool") =>
            {
                Some(Event::TextDelta(TextDelta {
                    chunk: self.text(delta, "partial_json")?,
                    block_id,
                    event: Default::default(),
                }))
            }
            "input_json_delta" => {
                if let Some(block) = self.blocks.get(&idx).filter(|b| !b.id.is_empty()) {
                    let chunk = self.text(delta, "partial_json")?;
                    if block.kind == "server_tool_use" {
                        Some(Event::BuiltinToolDelta(BuiltinToolDelta {
                            chunk,
                            tool_call_id: block.id.clone(),
                            event: Default::default(),
                        }))
                    } else {
                        Some(Event::ToolDelta(ToolDelta {
                            chunk,
                            tool_call_id: block.id.clone(),
                            event: Default::default(),
                        }))
                    }
                } else {
                    None
                }
            }
            _ => None,
        };
        if let Some(event) = event {
            self.pending.push_back(event);
        }
        Ok(())
    }
    fn block_stop(&mut self, value: &Value) -> Result<(), Error> {
        let idx = self.index(value)?;
        let Some(block) = self.blocks.remove(&idx) else {
            return Ok(());
        };
        let block_id = idx.to_string();
        let event = match block.kind.as_str() {
            "text" | "json_tool" => Some(Event::TextEnd(TextEnd {
                block_id,
                ..Default::default()
            })),
            "thinking" => Some(Event::ReasoningEnd(ReasoningEnd {
                block_id,
                event: ModelEventMeta {
                    provider_metadata: block.signature.map(|s| metadata(json!({"signature":s}))),
                    ..Default::default()
                },
            })),
            "tool_use" => Some(Event::ToolEnd(ToolEnd {
                tool_call_id: block.id.clone(),
                tool_call: ToolCallPart::new(block.id, block.name, ""),
                event: Default::default(),
            })),
            "server_tool_use" => Some(Event::BuiltinToolEnd(BuiltinToolEnd {
                tool_call_id: block.id.clone(),
                tool_call: BuiltinToolCallPart {
                    id: generate_id("part"),
                    tool_call_id: block.id,
                    tool_name: block.name,
                    tool_args: String::new(),
                    provider_metadata: Some(metadata(json!({}))),
                },
                event: Default::default(),
            })),
            "web_search_tool_result"
            | "web_fetch_tool_result"
            | "code_execution_tool_result"
            | "bash_tool_result"
            | "memory_tool_result" => {
                let id = block
                    .wire
                    .get("tool_use_id")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_owned();
                let result = BuiltinToolReturnPart {
                    id: generate_id("part"),
                    tool_call_id: id.clone(),
                    tool_name: self.tool_names.get(&id).cloned().unwrap_or_default(),
                    result: block.wire.get("content").cloned().unwrap_or(Value::Null),
                    is_error: false,
                    provider_metadata: Some(metadata(json!({"resultType":block.kind}))),
                };
                Some(Event::BuiltinToolResult(BuiltinToolResult {
                    tool_call_id: id,
                    result,
                    event: Default::default(),
                }))
            }
            _ => None,
        };
        if let Some(event) = event {
            self.pending.push_back(event);
        }
        Ok(())
    }
    fn end(&self) -> Result<Event, Error> {
        let token = |key: &str| self.usage.get(key).and_then(Value::as_u64);
        let cache_read = token("cache_read_input_tokens");
        let input = token("input_tokens")
            .unwrap_or(0)
            .checked_add(cache_read.unwrap_or(0))
            .ok_or_else(|| {
                http::response_error(
                    &self.provider,
                    format!("{} input token usage overflowed", self.api.label),
                )
            })?;
        let usage = Usage {
            input_tokens: input,
            output_tokens: token("output_tokens").unwrap_or(0),
            cache_read_tokens: cache_read,
            cache_write_tokens: token("cache_creation_input_tokens"),
            raw: (!self.usage.is_empty()).then(|| json!(self.usage)),
            ..Default::default()
        };
        let finish = self.stop_reason.as_deref().map(|r| match r {
            "end_turn" | "stop_sequence" => "stop",
            "max_tokens" | "model_context_window_exceeded" => "length",
            // The JSON tool's call is the answer, not a tool call.
            "tool_use" if self.json_answer => "stop",
            "tool_use" => "tool_call",
            "refusal" => "content_filter",
            _ => "other",
        });
        Ok(Event::StreamEnd(StreamEnd {
            event: ModelEventMeta {
                usage: Some(usage),
                provider_metadata: if finish == Some("other") {
                    Some(metadata(json!({"stop_reason":self.stop_reason})))
                } else {
                    None
                },
                ..Default::default()
            },
            finish_reason: finish.map(str::to_owned),
            response_id: self.response_id.clone(),
            response_model: self.response_model.clone(),
        }))
    }
}
