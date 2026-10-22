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
}
pub(super) fn parse(body: Body, provider: String, api: &'static ApiProfile) -> EventStream {
    let state = Parser {
        source: http::sse_json(*api, provider.clone(), body),
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
            "text" => Some(Event::TextEnd(TextEnd {
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
