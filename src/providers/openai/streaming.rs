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

use super::{
    dialect,
    profile::{ChatProfile, Dialect, ReasoningTokens, ResponsesDialect, ResponsesProfile},
    protocol::Wire,
};
use crate::{
    providers::{
        EventStream,
        http::{self, ApiProfile, SseJson},
    },
    transport::Body,
    types::{events::*, messages::*, usage::Usage},
};
use futures::{StreamExt, stream};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

fn string<'a>(v: &'a Value, key: &str) -> &'a str {
    v.get(key).and_then(Value::as_str).unwrap_or("")
}
fn metadata(key: &str, v: Value) -> Option<Metadata> {
    Some([(key.into(), v)].into())
}
fn item_metadata(key: &str, item: &Value) -> Option<Metadata> {
    let mut md = json!({"raw_item":item});
    if item["id"].is_string() {
        md["item_id"] = item["id"].clone();
    }
    if !item["encrypted_content"].is_null() {
        md["reasoning_encrypted_content"] = item["encrypted_content"].clone();
    }
    metadata(key, md)
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
    input: futures::stream::BoxStream<'static, Result<SseJson, crate::Error>>,
    provider: String,
    api: &'static ApiProfile,
    responses: bool,
    /// The Chat Completions profile; `None` for Responses.
    chat: Option<&'static ChatProfile>,
    /// The Responses profile; `None` for Chat Completions.
    responses_profile: Option<&'static ResponsesProfile>,
    /// The key of the part metadata this reader writes.
    metadata_key: &'static str,
    /// The request has an output schema.
    json_output: bool,
    /// Text arrived in this response.
    has_text: bool,
    /// Tool events held until the finish, while text can still void them.
    held: Vec<Event>,
    /// `StreamStart` was queued. It waits for the first chunk that is not an
    /// error, so an error the server sends first comes before any event and
    /// a call retries it when it is retryable.
    started: bool,
    ended: bool,
    queue: VecDeque<Event>,
    texts: BTreeSet<String>,
    reasoning: BTreeSet<String>,
    reasoning_delta: BTreeSet<String>,
    reasoning_ended: BTreeSet<String>,
    calls: BTreeMap<String, Call>,
    indices: BTreeMap<String, String>,
    /// Chat Completions tool calls in the order they started, and the slot
    /// of each call's index or place in its chunk.
    chat_calls: Vec<Call>,
    chat_call_keys: BTreeMap<String, usize>,
    /// How many Chat Completions blocks of each kind have started.
    chat_blocks: BTreeMap<String, usize>,
    /// A chunk had a top-level `usage`.
    top_level_usage: bool,
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
    wire: Wire,
    image_type: String,
    json_output: bool,
) -> EventStream {
    let api = wire.api();
    let state = Parser {
        input: http::sse_json(*api, provider.clone(), body),
        provider,
        api,
        responses: matches!(wire, Wire::Responses(_)),
        chat: match wire {
            Wire::Chat(profile) => Some(profile),
            Wire::Responses(_) => None,
        },
        responses_profile: match wire {
            Wire::Responses(profile) => Some(profile),
            Wire::Chat(_) => None,
        },
        metadata_key: match wire {
            Wire::Responses(profile) => profile.metadata_key,
            Wire::Chat(_) => "openai",
        },
        json_output,
        has_text: false,
        held: Vec::new(),
        started: false,
        ended: false,
        queue: VecDeque::new(),
        texts: BTreeSet::new(),
        reasoning: BTreeSet::new(),
        reasoning_delta: BTreeSet::new(),
        reasoning_ended: BTreeSet::new(),
        calls: BTreeMap::new(),
        indices: BTreeMap::new(),
        chat_calls: Vec::new(),
        chat_call_keys: BTreeMap::new(),
        chat_blocks: BTreeMap::new(),
        top_level_usage: false,
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
                Some(Ok(SseJson::Done)) | None => p.complete(),
                Some(Ok(SseJson::Object(data))) => {
                    // Mistral sends an error as an object of type `error`.
                    if data.get("error").is_some_and(|e| !e.is_null())
                        || string(&data, "type") == "error"
                        || string(&data, "object") == "error"
                    {
                        // Every Responses provider classifies its errors.
                        let kinds = p.chat.is_none_or(|c| c.stream_error_kinds);
                        return Err(dialect::stream_error(kinds, p.api, &p.provider, &data));
                    }
                    p.start();
                    if p.responses {
                        p.response(&data)?;
                    } else {
                        p.chat(&data)?;
                    }
                }
                Some(Err(error)) => return Err(error),
            }
        }
    })
    .boxed()
}
impl Parser {
    fn start(&mut self) {
        if !self.started {
            self.started = true;
            self.queue
                .push_back(Event::StreamStart(StreamStart::default()));
        }
    }
    fn complete(&mut self) {
        if self.ended {
            return;
        }
        self.start();
        if self.responses {
            for id in std::mem::take(&mut self.texts) {
                self.queue.push_back(block("text_end", &id, None));
            }
            for id in std::mem::take(&mut self.reasoning) {
                self.queue.push_back(block("reasoning_end", &id, None));
            }
        }
        let held = std::mem::take(&mut self.held);
        if !self.has_text {
            self.queue.extend(held);
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
    fn chat(&mut self, data: &Value) -> Result<(), crate::Error> {
        if !string(data, "id").is_empty() {
            self.response_id = Some(string(data, "id").into());
        }
        if !string(data, "model").is_empty() {
            self.response_model = Some(string(data, "model").into());
        }
        let dialect = self.chat.map_or(Dialect::OpenAI, |c| c.dialect);
        let first = data["choices"].as_array().and_then(|c| c.first());
        // Moonshot AI can put usage in the choice; a top-level usage wins
        // whichever comes first.
        let usage_value = if data["usage"].is_object() {
            self.top_level_usage = true;
            Some(&data["usage"])
        } else {
            first
                .map(|choice| &choice["usage"])
                .filter(|u| u.is_object())
                .filter(|_| dialect == Dialect::Moonshot && !self.top_level_usage)
        };
        if let Some(value) = usage_value
            && let Some(mut u) = usage(value, false)
        {
            let cached = match dialect {
                Dialect::OpenAI | Dialect::VertexMaas | Dialect::VertexXai => None,
                Dialect::DeepSeek => value["prompt_cache_hit_tokens"].as_u64(),
                Dialect::Moonshot => value["cached_tokens"].as_u64(),
                Dialect::Mistral => value["num_cached_tokens"]
                    .as_u64()
                    .or_else(|| value["prompt_token_details"]["cached_tokens"].as_u64()),
            };
            if cached.is_some() {
                u.cache_read_tokens = cached;
            }
            // DeepInfra can count reasoning tokens apart from completion
            // tokens; xAI on Vertex AI always does.
            if let Some(reasoning) = u.reasoning_tokens {
                match self.chat.map(|c| c.reasoning_tokens) {
                    Some(ReasoningTokens::OutsideWhenLarger) if reasoning > u.output_tokens => {
                        u.output_tokens = u.output_tokens.saturating_add(reasoning);
                    }
                    Some(ReasoningTokens::Outside) => {
                        u.output_tokens = u.output_tokens.saturating_add(reasoning);
                    }
                    _ => {}
                }
            }
            self.usage = Some(u);
        }
        let Some(choice) = first else {
            return Ok(());
        };
        let delta = &choice["delta"];
        let reasoning = delta
            .get("reasoning")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .or_else(|| delta.get("reasoning_content").and_then(Value::as_str))
            .unwrap_or("");
        if !reasoning.is_empty() {
            self.reasoning_chunk(reasoning);
        }
        match &delta["content"] {
            Value::String(text) if !text.is_empty() => self.text_chunk(text),
            // Mistral sends content as parts: text, thinking that holds text
            // parts, and parts with no text for the reader, such as references.
            Value::Array(parts) => {
                for part in parts {
                    match string(part, "type") {
                        "text" if !string(part, "text").is_empty() => {
                            self.text_chunk(string(part, "text"));
                        }
                        "thinking" => {
                            for inner in part["thinking"].as_array().into_iter().flatten() {
                                if string(inner, "type") == "text"
                                    && !string(inner, "text").is_empty()
                                {
                                    self.reasoning_chunk(string(inner, "text"));
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
        // Under an output schema, Cerebras can repeat its answer as tool calls
        // around the text; the text is the answer. Hold the tool events until
        // the finish shows whether text came.
        let hold = self.json_output
            && self
                .chat
                .is_some_and(|c| c.json_drops_tool_calls_after_text);
        if let Some(calls) = delta["tool_calls"].as_array() {
            for (position, tc) in calls.iter().enumerate() {
                // A call without an index, as Mistral and Moonshot AI send
                // them, is keyed by its place in the chunk; a new id under a
                // key is a new call.
                let key = match tc.get("index") {
                    None | Some(Value::Null) => format!("position {position}"),
                    Some(index) => match index.as_u64() {
                        Some(index) => format!("index {index}"),
                        None => {
                            return Err(http::response_error(
                                &self.provider,
                                format!(
                                    "{} stream sent a tool call index that is not a non-negative integer: {index}",
                                    self.api.label
                                ),
                            ));
                        }
                    },
                };
                let id = string(tc, "id");
                let slot = match self.chat_call_keys.get(&key) {
                    Some(&slot)
                        if id.is_empty()
                            || self.chat_calls[slot].id.is_empty()
                            || self.chat_calls[slot].id == id =>
                    {
                        slot
                    }
                    _ => {
                        self.chat_calls.push(Call::default());
                        self.chat_call_keys.insert(key, self.chat_calls.len() - 1);
                        self.chat_calls.len() - 1
                    }
                };
                let state = &mut self.chat_calls[slot];
                if !id.is_empty() {
                    state.id = id.into();
                }
                if !string(&tc["function"], "name").is_empty() {
                    state.name = string(&tc["function"], "name").into();
                }
                let args = string(&tc["function"], "arguments");
                if !args.is_empty() && !state.id.is_empty() {
                    let mut events = Vec::with_capacity(2);
                    if !state.started {
                        state.started = true;
                        events.push(Event::ToolStart(ToolStart {
                            tool_call_id: state.id.clone(),
                            tool_name: state.name.clone(),
                            ..Default::default()
                        }));
                    }
                    state.args.push_str(args);
                    events.push(Event::ToolDelta(ToolDelta {
                        event: Default::default(),
                        tool_call_id: state.id.clone(),
                        chunk: args.into(),
                    }));
                    if hold {
                        self.held.extend(events);
                    } else {
                        self.queue.extend(events);
                    }
                }
            }
        }
        if let Some(raw) = choice["finish_reason"].as_str() {
            let void_calls = hold && self.has_text;
            let reason = match (raw, self.chat.and_then(|c| c.finish_reason(raw))) {
                (_, Some(reason)) => reason,
                ("tool_calls", _) if void_calls => "stop",
                ("stop", _) => "stop",
                ("length", _) => "length",
                ("content_filter", _) => "content_filter",
                ("tool_calls" | "function_call", _) => "tool_call",
                _ => "other",
            };
            self.finish = Some(reason.into());
            if reason == "other" {
                self.metadata = metadata("openai", json!({"finish_reason":raw}));
            }
            for id in std::mem::take(&mut self.reasoning) {
                self.queue.push_back(block("reasoning_end", &id, None));
            }
            for id in std::mem::take(&mut self.texts) {
                self.queue.push_back(block("text_end", &id, None));
            }
            let held = std::mem::take(&mut self.held);
            if void_calls {
                self.chat_calls.clear();
                self.chat_call_keys.clear();
            } else {
                self.queue.extend(held);
            }
            for state in &mut self.chat_calls {
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
        Ok(())
    }
    /// The id of a new Chat Completions text or reasoning block: `text` or
    /// `reasoning` for the first, then numbered, so a block that starts after
    /// another of its kind ended is a new block.
    fn chat_block(&mut self, kind: &str) -> String {
        let count = self.chat_blocks.entry(kind.to_owned()).or_default();
        *count += 1;
        if *count == 1 {
            kind.to_owned()
        } else {
            format!("{kind}-{count}")
        }
    }
    /// A reasoning chunk: open text ends first, because reasoning after text
    /// is a new block.
    fn reasoning_chunk(&mut self, chunk: &str) {
        for id in std::mem::take(&mut self.texts) {
            self.queue.push_back(block("text_end", &id, None));
        }
        let id = match self.reasoning.iter().next() {
            Some(id) => id.clone(),
            None => {
                let id = self.chat_block("reasoning");
                self.reasoning.insert(id.clone());
                self.queue.push_back(block("reasoning_start", &id, None));
                id
            }
        };
        self.queue.push_back(Event::ReasoningDelta(ReasoningDelta {
            event: Default::default(),
            block_id: id,
            chunk: chunk.into(),
        }));
    }
    /// A text chunk: open reasoning ends first.
    fn text_chunk(&mut self, chunk: &str) {
        self.has_text = true;
        for id in std::mem::take(&mut self.reasoning) {
            self.queue.push_back(block("reasoning_end", &id, None));
        }
        let id = match self.texts.iter().next() {
            Some(id) => id.clone(),
            None => {
                let id = self.chat_block("text");
                self.texts.insert(id.clone());
                self.queue.push_back(block("text_start", &id, None));
                id
            }
        };
        self.queue.push_back(Event::TextDelta(TextDelta {
            event: Default::default(),
            block_id: id,
            chunk: chunk.into(),
        }));
    }
    fn response(&mut self, data: &Value) -> Result<(), crate::Error> {
        let kind = string(data, "type");
        match kind {
            "response.created"
            | "response.completed"
            | "response.done"
            | "response.incomplete"
            | "response.failed" => {
                let r = &data["response"];
                if !r.is_object() {
                    // A failed or incomplete response still ends as one.
                    return match kind {
                        "response.failed" => {
                            Err(dialect::stream_error(true, self.api, &self.provider, data))
                        }
                        "response.incomplete" => {
                            self.finish = Some("other".into());
                            Ok(())
                        }
                        _ => Ok(()),
                    };
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
                self.metadata = metadata(self.metadata_key, md.into());
                if kind != "response.created" {
                    if let Some(mut u) = usage(&r["usage"], true) {
                        // xAI can count cached tokens apart from input tokens.
                        if self
                            .responses_profile
                            .is_some_and(|p| p.dialect == ResponsesDialect::Xai)
                            && let Some(cached) = u.cache_read_tokens
                            && cached > u.input_tokens
                        {
                            u.input_tokens = u.input_tokens.saturating_add(cached);
                        }
                        self.usage = Some(u);
                    }
                    self.response_id = r["id"].as_str().map(str::to_owned);
                    self.response_model = r["model"].as_str().map(str::to_owned);
                    self.finish = match kind {
                        "response.incomplete" => Some(incomplete_reason(r)),
                        _ => finish_reason(r),
                    }
                    .map(str::to_owned);
                }
                if kind == "response.failed" {
                    return Err(dialect::stream_error(true, self.api, &self.provider, r));
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
                let id = self.reasoning_id(data);
                if self.reasoning.insert(id.clone()) {
                    self.queue.push_back(block("reasoning_start", &id, None));
                }
            }
            "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => {
                let id = self.reasoning_id(data);
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
            // xAI's one block per item ends with its output item, which
            // carries the encrypted content after the summary ended.
            "response.reasoning_summary_part.done" if !self.xai() => {
                let id = self.reasoning_id(data);
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
        Ok(())
    }
    fn xai(&self) -> bool {
        self.responses_profile
            .is_some_and(|p| p.dialect == ResponsesDialect::Xai)
    }
    /// The block of a reasoning event: one per summary part, or for xAI one
    /// per item, as the TypeScript package keeps it, since its encrypted
    /// content arrives only with the whole item.
    fn reasoning_id(&self, data: &Value) -> String {
        if self.xai() {
            format!("{}:0", string(data, "item_id"))
        } else {
            reasoning_id(data)
        }
    }
    fn item_added(&mut self, item: &Value, data: &Value) {
        if !item.is_object() {
            return;
        }
        let kind = string(item, "type");
        let key = state_key(item, data);
        let md = item_metadata(self.metadata_key, item);
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
        let md = item_metadata(self.metadata_key, item);
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
        let md = item_metadata(self.metadata_key, item);
        if self.xai() {
            // One block for the whole item: its streamed summary, or the
            // summary text of the item when none streamed.
            let block_id = format!("{id}:0");
            if !self.reasoning_ended.contains(&block_id) {
                let text: String = item["summary"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .map(|s| string(s, "text"))
                    .collect();
                if self.reasoning.insert(block_id.clone()) {
                    self.queue
                        .push_back(block("reasoning_start", &block_id, md.clone()));
                }
                if !self.reasoning_delta.contains(&block_id) && !text.is_empty() {
                    self.queue.push_back(Event::ReasoningDelta(ReasoningDelta {
                        event: Default::default(),
                        block_id: block_id.clone(),
                        chunk: text,
                    }));
                }
                self.reasoning.remove(&block_id);
                self.reasoning_ended.insert(block_id.clone());
                self.queue.push_back(block("reasoning_end", &block_id, md));
            }
            return;
        }
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
/// The finish reason of a `response.incomplete` event, whether or not its
/// response has a status.
fn incomplete_reason(response: &Value) -> &'static str {
    match string(&response["incomplete_details"], "reason") {
        "max_output_tokens" => "length",
        "content_filter" => "content_filter",
        _ => "other",
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
