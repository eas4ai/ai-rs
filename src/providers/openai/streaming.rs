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

use super::errors;
use crate::{
    providers::EventStream,
    transport::{Body, sse},
    types::{events::*, messages::*, usage::Usage},
};
use futures::{StreamExt, stream};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

fn string<'a>(v: &'a Value, key: &str) -> &'a str {
    v.get(key).and_then(Value::as_str).unwrap_or("")
}
fn metadata(v: Value) -> Option<Metadata> {
    Some([("openai".into(), v)].into())
}
fn item_metadata(item: &Value) -> Option<Metadata> {
    let mut md = json!({"raw_item":item});
    if item["id"].is_string() {
        md["item_id"] = item["id"].clone();
    }
    if !item["encrypted_content"].is_null() {
        md["reasoning_encrypted_content"] = item["encrypted_content"].clone();
    }
    metadata(md)
}
fn meta(md: Option<Metadata>) -> ModelEventMeta {
    ModelEventMeta {
        provider_metadata: md,
        ..Default::default()
    }
}
fn block(kind: &str, id: &str, md: Option<Metadata>) -> Event {
    let event = meta(md);
    let block_id = id.to_owned();
    match kind {
        "text_start" => Event::TextStart(TextStart { event, block_id }),
        "text_end" => Event::TextEnd(TextEnd { event, block_id }),
        "reasoning_start" => Event::ReasoningStart(ReasoningStart { event, block_id }),
        _ => Event::ReasoningEnd(ReasoningEnd { event, block_id }),
    }
}
fn usage(value: &Value, responses: bool) -> Option<Usage> {
    if !value.is_object() {
        return None;
    }
    let (input, output, ind, outd) = if responses {
        (
            "input_tokens",
            "output_tokens",
            "input_tokens_details",
            "output_tokens_details",
        )
    } else {
        (
            "prompt_tokens",
            "completion_tokens",
            "prompt_tokens_details",
            "completion_tokens_details",
        )
    };
    Some(Usage {
        input_tokens: value[input].as_u64().unwrap_or(0),
        output_tokens: value[output].as_u64().unwrap_or(0),
        reasoning_tokens: value[outd]["reasoning_tokens"].as_u64(),
        cache_read_tokens: value[ind]["cached_tokens"].as_u64(),
        raw: Some(value.clone()),
        ..Default::default()
    })
}
#[derive(Default)]
struct Call {
    id: String,
    name: String,
    args: String,
    started: bool,
    delta: bool,
}
struct Parser {
    input: futures::stream::BoxStream<
        'static,
        Result<sse::SseEvent, crate::transport::TransportError>,
    >,
    provider: String,
    responses: bool,
    ended: bool,
    queue: VecDeque<Event>,
    texts: BTreeSet<String>,
    reasoning: BTreeSet<String>,
    reasoning_delta: BTreeSet<String>,
    reasoning_ended: BTreeSet<String>,
    calls: BTreeMap<String, Call>,
    indices: BTreeMap<String, String>,
    builtins: BTreeMap<String, Call>,
    builtin_indices: BTreeMap<String, String>,
    usage: Option<Usage>,
    finish: Option<String>,
    response_id: Option<String>,
    response_model: Option<String>,
    metadata: Option<Metadata>,
    image_type: String,
}
pub(super) fn parse(
    body: Body,
    provider: String,
    responses: bool,
    image_type: String,
) -> EventStream {
    let mut queue = VecDeque::new();
    queue.push_back(Event::StreamStart(StreamStart::default()));
    let state = Parser {
        input: sse::parse(body),
        provider,
        responses,
        ended: false,
        queue,
        texts: BTreeSet::new(),
        reasoning: BTreeSet::new(),
        reasoning_delta: BTreeSet::new(),
        reasoning_ended: BTreeSet::new(),
        calls: BTreeMap::new(),
        indices: BTreeMap::new(),
        builtins: BTreeMap::new(),
        builtin_indices: BTreeMap::new(),
        usage: None,
        finish: None,
        response_id: None,
        response_model: None,
        metadata: None,
        image_type,
    };
    stream::try_unfold(state, |mut p| async move {
        loop {
            if let Some(event) = p.queue.pop_front() {
                return Ok(Some((event, p)));
            }
            if p.ended {
                return Ok(None);
            }
            match p.input.next().await {
                Some(Ok(event)) => {
                    if event.data.trim() == "[DONE]" {
                        p.complete();
                        continue;
                    }
                    let data: Value = serde_json::from_str(&event.data).map_err(|_| {
                        errors::response_error(&p.provider, "invalid JSON in OpenAI event stream")
                    })?;
                    if !data.is_object() {
                        return Err(errors::response_error(
                            &p.provider,
                            "OpenAI event must be an object",
                        ));
                    }
                    if data.get("error").is_some_and(|e| !e.is_null())
                        || string(&data, "type") == "error"
                    {
                        return Err(errors::response_error(
                            &p.provider,
                            data["error"]["message"]
                                .as_str()
                                .or_else(|| data["message"].as_str())
                                .unwrap_or("OpenAI stream returned an error"),
                        ));
                    }
                    if p.responses {
                        p.response(&data);
                    } else {
                        p.chat(&data);
                    }
                }
                Some(Err(e)) => return Err(errors::map_transport(e, &p.provider)),
                None => p.complete(),
            }
        }
    })
    .boxed()
}
impl Parser {
    fn complete(&mut self) {
        if self.ended {
            return;
        }
        if self.responses {
            for id in std::mem::take(&mut self.texts) {
                self.queue.push_back(block("text_end", &id, None));
            }
            for id in std::mem::take(&mut self.reasoning) {
                self.queue.push_back(block("reasoning_end", &id, None));
            }
        }
        self.queue.push_back(Event::StreamEnd(StreamEnd {
            event: ModelEventMeta {
                usage: self.usage.take(),
                provider_metadata: self.metadata.take(),
                ..Default::default()
            },
            finish_reason: self.finish.take(),
            response_id: self.response_id.take(),
            response_model: self.response_model.take(),
        }));
        self.ended = true;
    }
    fn chat(&mut self, data: &Value) {
        if !string(data, "id").is_empty() {
            self.response_id = Some(string(data, "id").into());
        }
        if !string(data, "model").is_empty() {
            self.response_model = Some(string(data, "model").into());
        }
        if let Some(u) = usage(&data["usage"], false) {
            self.usage = Some(u);
        }
        let Some(choice) = data["choices"].as_array().and_then(|c| c.first()) else {
            return;
        };
        let delta = &choice["delta"];
        let reasoning = delta
            .get("reasoning")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .or_else(|| delta.get("reasoning_content").and_then(Value::as_str))
            .unwrap_or("");
        if !reasoning.is_empty() {
            if self.reasoning.insert("reasoning".into()) {
                self.queue
                    .push_back(block("reasoning_start", "reasoning", None));
            }
            self.queue.push_back(Event::ReasoningDelta(ReasoningDelta {
                event: Default::default(),
                block_id: "reasoning".into(),
                chunk: reasoning.into(),
            }));
        }
        let text = string(delta, "content");
        if !text.is_empty() {
            if self.reasoning.remove("reasoning") {
                self.queue
                    .push_back(block("reasoning_end", "reasoning", None));
            }
            if self.texts.insert("text".into()) {
                self.queue.push_back(block("text_start", "text", None));
            }
            self.queue.push_back(Event::TextDelta(TextDelta {
                event: Default::default(),
                block_id: "text".into(),
                chunk: text.into(),
            }));
        }
        if let Some(calls) = delta["tool_calls"].as_array() {
            for tc in calls {
                let index = tc["index"].to_string();
                let state = self.calls.entry(index).or_default();
                if !string(tc, "id").is_empty() {
                    state.id = string(tc, "id").into();
                }
                if !string(&tc["function"], "name").is_empty() {
                    state.name = string(&tc["function"], "name").into();
                }
                let args = string(&tc["function"], "arguments");
                if !args.is_empty() && !state.id.is_empty() {
                    if !state.started {
                        state.started = true;
                        self.queue.push_back(Event::ToolStart(ToolStart {
                            tool_call_id: state.id.clone(),
                            tool_name: state.name.clone(),
                            ..Default::default()
                        }));
                    }
                    state.args.push_str(args);
                    self.queue.push_back(Event::ToolDelta(ToolDelta {
                        event: Default::default(),
                        tool_call_id: state.id.clone(),
                        chunk: args.into(),
                    }));
                }
            }
        }
        if let Some(raw) = choice["finish_reason"].as_str() {
            let reason = match raw {
                "stop" => "stop",
                "length" => "length",
                "content_filter" => "content_filter",
                "tool_calls" | "function_call" => "tool_call",
                _ => "other",
            };
            self.finish = Some(reason.into());
            if reason == "other" {
                self.metadata = metadata(json!({"finish_reason":raw}));
            }
            for id in std::mem::take(&mut self.reasoning) {
                self.queue.push_back(block("reasoning_end", &id, None));
            }
            for id in std::mem::take(&mut self.texts) {
                self.queue.push_back(block("text_end", &id, None));
            }
            for state in self.calls.values_mut() {
                if state.started {
                    state.started = false;
                    self.queue.push_back(Event::ToolEnd(ToolEnd {
                        event: Default::default(),
                        tool_call_id: state.id.clone(),
                        tool_call: ToolCallPart::new("", "", ""),
                    }));
                }
            }
        }
    }
    fn response(&mut self, data: &Value) {
        let kind = string(data, "type");
        match kind {
            "response.created"
            | "response.completed"
            | "response.incomplete"
            | "response.failed" => {
                let r = &data["response"];
                if !r.is_object() {
                    return;
                }
                let mut md = serde_json::Map::new();
                for (src, dst) in [
                    ("id", "response_id"),
                    ("model", "model"),
                    ("status", "status"),
                ] {
                    if r[src].is_string() {
                        md.insert(dst.into(), r[src].clone());
                    }
                }
                if r["incomplete_details"]["reason"].is_string() {
                    md.insert(
                        "incomplete_reason".into(),
                        r["incomplete_details"]["reason"].clone(),
                    );
                }
                self.metadata = metadata(md.into());
                if kind != "response.created" {
                    if let Some(u) = usage(&r["usage"], true) {
                        self.usage = Some(u);
                    }
                    self.response_id = r["id"].as_str().map(str::to_owned);
                    self.response_model = r["model"].as_str().map(str::to_owned);
                    self.finish = finish_reason(r).map(str::to_owned);
                }
            }
            "response.output_item.added" => self.item_added(&data["item"], data),
            "response.output_item.done" => self.item_done(&data["item"], data),
            "response.output_text.delta" => {
                let id = if string(data, "item_id").is_empty() {
                    "text"
                } else {
                    string(data, "item_id")
                };
                if self.texts.insert(id.into()) {
                    self.queue.push_back(block("text_start", id, None));
                }
                let text = string(data, "delta");
                if !text.is_empty() {
                    self.queue.push_back(Event::TextDelta(TextDelta {
                        event: Default::default(),
                        block_id: id.into(),
                        chunk: text.into(),
                    }));
                }
            }
            "response.function_call_arguments.delta"
            | "response.custom_tool_call_input.delta"
            | "response.function_call_arguments.done"
            | "response.custom_tool_call_input.done" => {
                let key = lookup(&self.calls, &self.indices, data);
                if let Some(state) = self.calls.get_mut(&key) {
                    let done = kind.ends_with(".done");
                    let text = if done {
                        arguments(data)
                    } else {
                        string(data, "delta")
                    };
                    if !text.is_empty() && (!done || !state.delta) {
                        state.args.push_str(text);
                        state.delta = true;
                        self.queue.push_back(Event::ToolDelta(ToolDelta {
                            event: Default::default(),
                            tool_call_id: state.id.clone(),
                            chunk: text.into(),
                        }));
                    }
                }
            }
            "response.reasoning_summary_part.added" => {
                let id = reasoning_id(data);
                if self.reasoning.insert(id.clone()) {
                    self.queue.push_back(block("reasoning_start", &id, None));
                }
            }
            "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => {
                let id = reasoning_id(data);
                if self.reasoning.insert(id.clone()) {
                    self.queue.push_back(block("reasoning_start", &id, None));
                }
                let text = string(data, "delta");
                if !text.is_empty() {
                    self.reasoning_delta.insert(id.clone());
                    self.queue.push_back(Event::ReasoningDelta(ReasoningDelta {
                        event: Default::default(),
                        block_id: id,
                        chunk: text.into(),
                    }));
                }
            }
            "response.reasoning_summary_part.done" => {
                let id = reasoning_id(data);
                if self.reasoning.remove(&id) {
                    self.reasoning_ended.insert(id.clone());
                    self.queue.push_back(block("reasoning_end", &id, None));
                }
            }
            "response.image_generation_call.partial_image" => {
                let image = string(data, "partial_image_b64");
                if !image.is_empty() {
                    self.queue.push_back(Event::File(FileEvent {
                        event: Default::default(),
                        block_id: format!("{}:partial", string(data, "item_id")),
                        media_type: self.image_type.clone(),
                        data: FileData::Text(image.into()),
                        filename: None,
                    }));
                }
            }
            "response.code_interpreter_call_code.delta" => {
                let key = lookup(&self.builtins, &self.builtin_indices, data);
                if let Some(state) = self.builtins.get_mut(&key) {
                    state.args.push_str(string(data, "delta"));
                }
            }
            _ => {}
        }
    }
    fn item_added(&mut self, item: &Value, data: &Value) {
        if !item.is_object() {
            return;
        }
        let kind = string(item, "type");
        let key = state_key(item, data);
        let md = item_metadata(item);
        match kind {
            "message" => {
                let id = nonempty(string(item, "id"), &key, "text");
                self.texts.insert(id.into());
                self.queue.push_back(block("text_start", id, md));
            }
            "reasoning" => {
                let id = format!("{}:0", nonempty(string(item, "id"), &key, ""));
                self.reasoning.insert(id.clone());
                self.queue.push_back(block("reasoning_start", &id, md));
            }
            "function_call" | "custom_tool_call" => {
                let id = nonempty(string(item, "call_id"), &key, "").to_owned();
                let name = string(item, "name").to_owned();
                let args = arguments(item).to_owned();
                self.queue.push_back(Event::ToolStart(ToolStart {
                    event: meta(md),
                    tool_call_id: id.clone(),
                    tool_name: name.clone(),
                }));
                if !args.is_empty() {
                    self.queue.push_back(Event::ToolDelta(ToolDelta {
                        event: Default::default(),
                        tool_call_id: id.clone(),
                        chunk: args.clone(),
                    }));
                }
                if let Some(index) = index(data) {
                    self.indices.insert(index, key.clone());
                }
                self.calls.insert(
                    key,
                    Call {
                        id,
                        name,
                        delta: !args.is_empty(),
                        args,
                        started: true,
                    },
                );
            }
            _ if builtin(kind) => {
                let id = builtin_id(item);
                let name = builtin_name(item);
                self.queue
                    .push_back(Event::BuiltinToolStart(BuiltinToolStart {
                        event: meta(md),
                        tool_call_id: id.clone(),
                        tool_name: name.clone(),
                    }));
                if let Some(index) = index(data) {
                    self.builtin_indices.insert(index, key.clone());
                }
                self.builtins.insert(
                    key,
                    Call {
                        id,
                        name,
                        ..Default::default()
                    },
                );
            }
            _ => {}
        }
    }
    fn item_done(&mut self, item: &Value, data: &Value) {
        if !item.is_object() {
            return;
        }
        let kind = string(item, "type");
        let key = state_key(item, data);
        let md = item_metadata(item);
        match kind {
            "message" => {
                let id = nonempty(string(item, "id"), &key, "text");
                if self.texts.remove(id) {
                    self.queue.push_back(block("text_end", id, md));
                }
            }
            "reasoning" => self.reasoning_done(item, &key),
            "function_call" | "custom_tool_call" => {
                let state = remove_state(&mut self.calls, &mut self.indices, &key, data);
                let id = state
                    .as_ref()
                    .map(|s| s.id.clone())
                    .unwrap_or_else(|| nonempty(string(item, "call_id"), &key, "").into());
                if state.is_none() {
                    self.queue.push_back(Event::ToolStart(ToolStart {
                        event: meta(md.clone()),
                        tool_call_id: id.clone(),
                        tool_name: string(item, "name").into(),
                    }));
                }
                if state.as_ref().is_none_or(|s| !s.delta) && !arguments(item).is_empty() {
                    self.queue.push_back(Event::ToolDelta(ToolDelta {
                        event: Default::default(),
                        tool_call_id: id.clone(),
                        chunk: arguments(item).into(),
                    }));
                }
                self.queue.push_back(Event::ToolEnd(ToolEnd {
                    event: meta(md),
                    tool_call_id: id,
                    tool_call: ToolCallPart::new("", "", ""),
                }));
            }
            _ if builtin(kind) => {
                let state = remove_state(&mut self.builtins, &mut self.builtin_indices, &key, data);
                let id = state
                    .as_ref()
                    .map(|s| s.id.clone())
                    .unwrap_or_else(|| builtin_id(item));
                let name = state
                    .as_ref()
                    .map(|s| s.name.clone())
                    .unwrap_or_else(|| builtin_name(item));
                let args = builtin_args(item);
                if state.is_none() {
                    self.queue
                        .push_back(Event::BuiltinToolStart(BuiltinToolStart {
                            event: meta(md.clone()),
                            tool_call_id: id.clone(),
                            tool_name: name.clone(),
                        }));
                }
                if !args.is_empty() {
                    self.queue
                        .push_back(Event::BuiltinToolDelta(BuiltinToolDelta {
                            event: Default::default(),
                            tool_call_id: id.clone(),
                            chunk: args.clone(),
                        }));
                }
                self.queue.push_back(Event::BuiltinToolEnd(BuiltinToolEnd {
                    event: meta(md.clone()),
                    tool_call_id: id.clone(),
                    tool_call: BuiltinToolCallPart {
                        id: generate_id("part"),
                        tool_call_id: id.clone(),
                        tool_name: name.clone(),
                        tool_args: args,
                        provider_metadata: md.clone(),
                    },
                }));
                if kind == "image_generation_call" && !string(item, "result").is_empty() {
                    self.queue.push_back(Event::File(FileEvent {
                        event: meta(md.clone()),
                        block_id: nonempty(string(item, "id"), &id, "").into(),
                        media_type: self.image_type.clone(),
                        data: FileData::Text(string(item, "result").into()),
                        filename: None,
                    }));
                }
                if let Some(result) = builtin_result(item) {
                    self.queue
                        .push_back(Event::BuiltinToolResult(BuiltinToolResult {
                            event: Default::default(),
                            tool_call_id: id.clone(),
                            result: BuiltinToolReturnPart {
                                id: generate_id("part"),
                                tool_call_id: id,
                                tool_name: name,
                                result,
                                is_error: false,
                                provider_metadata: md,
                            },
                        }));
                }
            }
            _ => {}
        }
    }
    fn reasoning_done(&mut self, item: &Value, key: &str) {
        let id = nonempty(string(item, "id"), key, "");
        let md = item_metadata(item);
        if let Some(summary) = item["summary"].as_array().filter(|s| !s.is_empty()) {
            for (index, summary) in summary.iter().enumerate() {
                let block_id = format!("{id}:{index}");
                if self.reasoning_ended.contains(&block_id) {
                    continue;
                }
                if self.reasoning.insert(block_id.clone()) {
                    self.queue
                        .push_back(block("reasoning_start", &block_id, md.clone()));
                }
                let text = string(summary, "text");
                if !self.reasoning_delta.contains(&block_id) && !text.is_empty() {
                    self.queue.push_back(Event::ReasoningDelta(ReasoningDelta {
                        event: Default::default(),
                        block_id: block_id.clone(),
                        chunk: text.into(),
                    }));
                }
                self.reasoning.remove(&block_id);
                self.reasoning_ended.insert(block_id.clone());
                self.queue
                    .push_back(block("reasoning_end", &block_id, md.clone()));
            }
        } else {
            let matching: Vec<_> = self
                .reasoning
                .iter()
                .filter(|b| b.starts_with(&format!("{id}:")))
                .cloned()
                .collect();
            for block_id in matching {
                self.reasoning.remove(&block_id);
                self.queue
                    .push_back(block("reasoning_end", &block_id, md.clone()));
            }
        }
    }
}
fn finish_reason(response: &Value) -> Option<&'static str> {
    match string(response, "status") {
        "failed" => Some("error"),
        "incomplete" => match string(&response["incomplete_details"], "reason") {
            "max_output_tokens" => Some("length"),
            "content_filter" => Some("content_filter"),
            "" => None,
            _ => Some("other"),
        },
        "completed" => Some(
            if response["output"]
                .as_array()
                .is_some_and(|a| a.iter().any(|i| string(i, "type") == "function_call"))
            {
                "tool_call"
            } else {
                "stop"
            },
        ),
        _ => None,
    }
}
fn nonempty<'a>(a: &'a str, b: &'a str, c: &'a str) -> &'a str {
    if !a.is_empty() {
        a
    } else if !b.is_empty() {
        b
    } else {
        c
    }
}
fn arguments(item: &Value) -> &str {
    nonempty(string(item, "arguments"), string(item, "input"), "")
}
fn index(data: &Value) -> Option<String> {
    data.get("output_index")
        .filter(|i| !i.is_null())
        .map(Value::to_string)
}
fn state_key(item: &Value, data: &Value) -> String {
    let key = nonempty(
        string(item, "id"),
        string(item, "call_id"),
        string(data, "item_id"),
    );
    if key.is_empty() {
        index(data).unwrap_or_default()
    } else {
        key.into()
    }
}
fn lookup(
    states: &BTreeMap<String, Call>,
    indices: &BTreeMap<String, String>,
    data: &Value,
) -> String {
    let id = string(data, "item_id");
    if states.contains_key(id) {
        id.into()
    } else {
        index(data)
            .and_then(|i| indices.get(&i).cloned())
            .unwrap_or_default()
    }
}
fn remove_state(
    states: &mut BTreeMap<String, Call>,
    indices: &mut BTreeMap<String, String>,
    key: &str,
    data: &Value,
) -> Option<Call> {
    let indexed = index(data).and_then(|i| indices.remove(&i));
    let state = states
        .remove(key)
        .or_else(|| indexed.and_then(|k| states.remove(&k)));
    indices.retain(|_, v| v != key);
    state
}
fn reasoning_id(data: &Value) -> String {
    format!(
        "{}:{}",
        string(data, "item_id"),
        data.get("summary_index").unwrap_or(&json!(0))
    )
}
fn builtin(kind: &str) -> bool {
    matches!(
        kind,
        "web_search_call"
            | "file_search_call"
            | "code_interpreter_call"
            | "image_generation_call"
            | "mcp_call"
            | "mcp_approval_request"
            | "local_shell_call"
            | "shell_call"
            | "shell_call_output"
            | "apply_patch_call"
            | "tool_search_call"
            | "tool_search_output"
            | "computer_call"
    )
}
fn builtin_name(item: &Value) -> String {
    match string(item, "type") {
        "mcp_call" | "mcp_approval_request" => {
            if item["name"].is_string() {
                format!("mcp.{}", string(item, "name"))
            } else {
                "mcp".into()
            }
        }
        "shell_call_output" => "shell".into(),
        "tool_search_output" => "tool_search".into(),
        "computer_call" => "computer_use".into(),
        name => name.strip_suffix("_call").unwrap_or(name).into(),
    }
}
fn builtin_id(item: &Value) -> String {
    nonempty(string(item, "call_id"), string(item, "id"), "").into()
}
fn builtin_args(item: &Value) -> String {
    match string(item, "type") {
        "code_interpreter_call" => {
            json!({"code":item["code"],"container_id":item["container_id"]}).to_string()
        }
        "mcp_call" | "mcp_approval_request" => item["arguments"]
            .as_str()
            .map(str::to_owned)
            .unwrap_or_else(|| item["arguments"].to_string()),
        "local_shell_call" | "shell_call" => json!({"action":item["action"]}).to_string(),
        "apply_patch_call" => {
            json!({"call_id":item["call_id"],"operation":item["operation"]}).to_string()
        }
        "tool_search_call" => {
            json!({"arguments":item["arguments"],"call_id":item["call_id"]}).to_string()
        }
        _ => "{}".into(),
    }
}
fn builtin_result(item: &Value) -> Option<Value> {
    Some(match string(item, "type") {
        "web_search_call" => json!({"action":item["action"]}),
        "file_search_call" => json!({"queries":item["queries"],"results":item["results"]}),
        "code_interpreter_call" => {
            json!({"container_id":item["container_id"],"outputs":item["outputs"]})
        }
        "image_generation_call" => json!({"result":item["result"]}),
        "mcp_call" => {
            json!({"server_label":item["server_label"],"name":item["name"],"arguments":item["arguments"],"output":item["output"],"error":item["error"]})
        }
        "mcp_approval_request" => {
            json!({"server_label":item["server_label"],"name":item["name"],"arguments":item["arguments"],"approval_request_id":item.get("approval_request_id").filter(|v|!v.is_null()).unwrap_or(&item["id"])})
        }
        "shell_call_output" => json!({"output":item["output"]}),
        "tool_search_output" => json!({"tools":item["tools"]}),
        "computer_call" => {
            json!({"status":item.get("status").filter(|v|!v.is_null()).unwrap_or(&json!("completed"))})
        }
        _ => return None,
    })
}
