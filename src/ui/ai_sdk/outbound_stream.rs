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

//! Single-pass conversion from core events to UI events and SSE frames.

use super::{
    approvals,
    outbound_messages::{self, python_str},
    tool_utils::{normalize_tool_input, python_json_dumps, python_truthy},
    ui_events::*,
};
use crate::{
    Error,
    types::{
        events::{Event, HookEvent, PartialToolCallResult, ToolCallResult, ValueAggregator},
        media::data_to_data_url,
        messages::{
            FileData, HookStatus, Message, MessageRole, Metadata, SpecialToolResult,
            ToolResultValue,
        },
    },
};
use futures::{
    Stream, StreamExt,
    stream::{self, BoxStream},
};
use serde_json::Value;
use std::collections::{BTreeSet, HashMap, VecDeque};

#[derive(Default)]
struct StreamState {
    emitted_start: bool,
    in_step: bool,
    step_ended: bool,
    finish_reason: Option<String>,
    started_tool_inputs: BTreeSet<String>,
    tool_names: HashMap<String, String>,
    input_available_emitted: BTreeSet<String>,
    emitted_tool_results: BTreeSet<String>,
    emitted_approval_requests: BTreeSet<String>,
    open_text_ids: BTreeSet<String>,
    open_reasoning_ids: BTreeSet<String>,
    source_messages: Vec<Message>,
    source_message_indices: HashMap<String, usize>,
    partial_aggregators: HashMap<String, Box<dyn ValueAggregator>>,
}

impl StreamState {
    fn track_source_message(&mut self, message: &Message) {
        if message.id == "<unset>" {
            return;
        }
        if let Some(&index) = self.source_message_indices.get(&message.id) {
            self.source_messages[index] = message.clone();
        } else {
            self.source_message_indices
                .insert(message.id.clone(), self.source_messages.len());
            self.source_messages.push(message.clone());
        }
    }

    fn ensure_started(&mut self, message_id: Option<String>) -> Vec<UIMessageStreamEvent> {
        let mut out = Vec::new();
        if !self.emitted_start {
            out.push(UIMessageStreamEvent::Start(UIStartEvent {
                message_id,
                ..Default::default()
            }));
            self.emitted_start = true;
        }
        if !self.in_step {
            out.push(UIMessageStreamEvent::StartStep(UIStartStepEvent {}));
            self.in_step = true;
        }
        out
    }

    fn on_event(&mut self, event: &Event) -> Result<Vec<UIMessageStreamEvent>, Error> {
        match event {
            Event::ToolCallResult(e) => return self.on_tool_result(e),
            Event::PartialToolCallResult(e) => return self.on_partial_tool_result(e),
            Event::Hook(e) => return Ok(self.on_hook(e)),
            Event::RunBlocked(_) => return Ok(Vec::new()),
            Event::Retry(_) => return Ok(self.on_retry()),
            _ => {}
        }
        let meta = event.model_event().expect("model event matched above");
        self.track_source_message(&meta.message);
        let mut out = if self.emitted_start {
            Vec::new()
        } else {
            let message = &meta.message;
            let id = if message.role == MessageRole::Assistant {
                message
                    .turn_id
                    .clone()
                    .or_else(|| (message.id != "<unset>").then(|| message.id.clone()))
            } else {
                None
            };
            self.ensure_started(id)
        };
        let provider_metadata = meta.provider_metadata.clone();
        match event {
            Event::StreamEnd(e) => {
                self.finish_reason.clone_from(&e.finish_reason);
                self.step_ended = true;
            }
            Event::StreamStart(_) => {
                if self.step_ended {
                    self.step_ended = false;
                    out.push(UIMessageStreamEvent::FinishStep(UIFinishStepEvent {}));
                    out.push(UIMessageStreamEvent::StartStep(UIStartStepEvent {}));
                }
            }
            Event::TextStart(_)
            | Event::TextDelta(_)
            | Event::TextEnd(_)
            | Event::ReasoningStart(_)
            | Event::ReasoningDelta(_)
            | Event::ReasoningEnd(_) => self.on_block_event(event, provider_metadata, &mut out),
            Event::ToolStart(_)
            | Event::ToolDelta(_)
            | Event::ToolEnd(_)
            | Event::BuiltinToolStart(_)
            | Event::BuiltinToolDelta(_)
            | Event::BuiltinToolEnd(_)
            | Event::BuiltinToolResult(_) => self.on_tool_event(event, provider_metadata, &mut out),
            Event::File(e) => out.push(UIMessageStreamEvent::File(UIFileEvent {
                url: match &e.data {
                    FileData::Text(text) => data_to_data_url(text, &e.media_type),
                    FileData::Bytes(bytes) => data_to_data_url(bytes.as_slice(), &e.media_type),
                },
                media_type: e.media_type.clone(),
                provider_metadata,
            })),
            Event::ToolCallResult(_)
            | Event::PartialToolCallResult(_)
            | Event::Hook(_)
            | Event::RunBlocked(_)
            | Event::Retry(_) => unreachable!("agent events matched above"),
        }
        Ok(out)
    }

    fn on_block_event(
        &mut self,
        event: &Event,
        provider_metadata: Option<Metadata>,
        out: &mut Vec<UIMessageStreamEvent>,
    ) {
        match event {
            Event::TextStart(e) => {
                self.open_text_ids.insert(e.block_id.clone());
                out.push(UIMessageStreamEvent::TextStart(UITextStartEvent {
                    id: e.block_id.clone(),
                    provider_metadata,
                }));
            }
            Event::TextDelta(e) => {
                if self.open_text_ids.insert(e.block_id.clone()) {
                    out.push(UIMessageStreamEvent::TextStart(UITextStartEvent {
                        id: e.block_id.clone(),
                        provider_metadata: provider_metadata.clone(),
                    }));
                }
                out.push(UIMessageStreamEvent::TextDelta(UITextDeltaEvent {
                    id: e.block_id.clone(),
                    delta: e.chunk.clone(),
                    provider_metadata,
                }));
            }
            Event::TextEnd(e) => {
                if self.open_text_ids.remove(&e.block_id) {
                    out.push(UIMessageStreamEvent::TextEnd(UITextEndEvent {
                        id: e.block_id.clone(),
                        provider_metadata,
                    }));
                }
            }
            Event::ReasoningStart(e) => {
                self.open_reasoning_ids.insert(e.block_id.clone());
                out.push(UIMessageStreamEvent::ReasoningStart(
                    UIReasoningStartEvent {
                        id: e.block_id.clone(),
                        provider_metadata,
                    },
                ));
            }
            Event::ReasoningDelta(e) => {
                if self.open_reasoning_ids.insert(e.block_id.clone()) {
                    out.push(UIMessageStreamEvent::ReasoningStart(
                        UIReasoningStartEvent {
                            id: e.block_id.clone(),
                            provider_metadata: provider_metadata.clone(),
                        },
                    ));
                }
                out.push(UIMessageStreamEvent::ReasoningDelta(
                    UIReasoningDeltaEvent {
                        id: e.block_id.clone(),
                        delta: e.chunk.clone(),
                        provider_metadata,
                    },
                ));
            }
            Event::ReasoningEnd(e) => {
                if self.open_reasoning_ids.remove(&e.block_id) {
                    out.push(UIMessageStreamEvent::ReasoningEnd(UIReasoningEndEvent {
                        id: e.block_id.clone(),
                        provider_metadata,
                    }));
                }
            }
            _ => unreachable!("block event selected by on_event"),
        }
    }

    fn on_tool_event(
        &mut self,
        event: &Event,
        provider_metadata: Option<Metadata>,
        out: &mut Vec<UIMessageStreamEvent>,
    ) {
        match event {
            Event::ToolStart(e) => {
                self.start_tool(out, &e.tool_call_id, &e.tool_name, provider_metadata, false)
            }
            Event::BuiltinToolStart(e) => {
                self.start_tool(out, &e.tool_call_id, &e.tool_name, provider_metadata, true)
            }
            Event::ToolDelta(e) => {
                self.delta_tool(out, &e.tool_call_id, &e.chunk, provider_metadata, false)
            }
            Event::BuiltinToolDelta(e) => {
                self.delta_tool(out, &e.tool_call_id, &e.chunk, provider_metadata, true)
            }
            Event::ToolEnd(_) => {}
            Event::BuiltinToolEnd(e) => {
                if self.input_available_emitted.insert(e.tool_call_id.clone()) {
                    out.push(UIMessageStreamEvent::ToolInputAvailable(
                        UIToolInputAvailableEvent {
                            tool_call_id: e.tool_call_id.clone(),
                            tool_name: e.tool_call.tool_name.clone(),
                            input: normalize_tool_input(&e.tool_call.tool_args),
                            provider_executed: Some(true),
                            dynamic: Some(true),
                            provider_metadata: prefer_metadata(
                                &e.tool_call.provider_metadata,
                                &provider_metadata,
                            ),
                            ..Default::default()
                        },
                    ));
                }
            }
            Event::BuiltinToolResult(e) => {
                if self.emitted_tool_results.insert(e.tool_call_id.clone()) {
                    let provider_metadata =
                        prefer_metadata(&e.result.provider_metadata, &provider_metadata);
                    if e.result.is_error {
                        out.push(UIMessageStreamEvent::ToolOutputError(
                            UIToolOutputErrorEvent {
                                tool_call_id: e.tool_call_id.clone(),
                                error_text: python_str(&e.result.result),
                                provider_executed: Some(true),
                                dynamic: Some(true),
                                provider_metadata,
                                ..Default::default()
                            },
                        ));
                    } else {
                        out.push(UIMessageStreamEvent::ToolOutputAvailable(
                            UIToolOutputAvailableEvent {
                                tool_call_id: e.tool_call_id.clone(),
                                output: e.result.result.clone(),
                                provider_executed: Some(true),
                                dynamic: Some(true),
                                provider_metadata,
                                ..Default::default()
                            },
                        ));
                    }
                }
            }
            _ => unreachable!("tool event selected by on_event"),
        }
    }

    fn start_tool(
        &mut self,
        out: &mut Vec<UIMessageStreamEvent>,
        id: &str,
        name: &str,
        provider_metadata: Option<Metadata>,
        builtin: bool,
    ) {
        self.tool_names.insert(id.into(), name.into());
        if !self.started_tool_inputs.insert(id.into()) {
            return;
        }
        out.push(UIMessageStreamEvent::ToolInputStart(
            UIToolInputStartEvent {
                tool_call_id: id.into(),
                tool_name: name.into(),
                provider_metadata,
                provider_executed: builtin.then_some(true),
                dynamic: builtin.then_some(true),
                ..Default::default()
            },
        ));
    }

    fn delta_tool(
        &mut self,
        out: &mut Vec<UIMessageStreamEvent>,
        id: &str,
        chunk: &str,
        provider_metadata: Option<Metadata>,
        builtin: bool,
    ) {
        if !self.started_tool_inputs.contains(id) {
            let name = self.tool_names.get(id).cloned().unwrap_or_default();
            self.start_tool(out, id, &name, provider_metadata, builtin);
        }
        out.push(UIMessageStreamEvent::ToolInputDelta(
            UIToolInputDeltaEvent {
                tool_call_id: id.into(),
                input_text_delta: chunk.into(),
            },
        ));
    }

    fn on_retry(&mut self) -> Vec<UIMessageStreamEvent> {
        if !self.in_step {
            return Vec::new();
        }
        self.step_ended = false;
        self.started_tool_inputs.clear();
        self.tool_names.clear();
        self.input_available_emitted.clear();
        self.emitted_tool_results.clear();
        self.emitted_approval_requests.clear();
        self.open_text_ids.clear();
        self.open_reasoning_ids.clear();
        self.partial_aggregators.clear();
        vec![UIMessageStreamEvent::ResetStep(UIResetStepEvent {})]
    }

    fn on_tool_result(
        &mut self,
        event: &ToolCallResult,
    ) -> Result<Vec<UIMessageStreamEvent>, Error> {
        self.track_source_message(&event.message);
        let mut out = self.ensure_started(event.message.turn_id.clone());
        for part in event.message.tool_calls() {
            if !self
                .input_available_emitted
                .insert(part.tool_call_id.clone())
            {
                continue;
            }
            if self.started_tool_inputs.insert(part.tool_call_id.clone()) {
                out.push(UIMessageStreamEvent::ToolInputStart(
                    UIToolInputStartEvent {
                        tool_call_id: part.tool_call_id.clone(),
                        tool_name: part.tool_name.clone(),
                        provider_metadata: part.provider_metadata.clone(),
                        ..Default::default()
                    },
                ));
            }
            out.push(UIMessageStreamEvent::ToolInputAvailable(
                UIToolInputAvailableEvent {
                    tool_call_id: part.tool_call_id.clone(),
                    tool_name: part.tool_name.clone(),
                    input: normalize_tool_input(&part.tool_args),
                    provider_metadata: part.provider_metadata.clone(),
                    ..Default::default()
                },
            ));
        }
        for part in &event.results {
            if self.emitted_tool_results.contains(&part.tool_call_id) || part.is_hook_deferred {
                continue;
            }
            self.emitted_tool_results.insert(part.tool_call_id.clone());
            if part.is_error() {
                let error_text = tool_error_text(&part.result);
                out.push(UIMessageStreamEvent::ToolOutputError(
                    UIToolOutputErrorEvent {
                        tool_call_id: part.tool_call_id.clone(),
                        error_text,
                        provider_metadata: part.provider_metadata.clone(),
                        ..Default::default()
                    },
                ));
                continue;
            }
            let output = match &part.result {
                ToolResultValue::Json(value) => value.clone(),
                ToolResultValue::Special(SpecialToolResult::Messages(bundle)) => {
                    outbound_messages::bundle_to_wire_output(bundle)?
                }
                ToolResultValue::Special(SpecialToolResult::Content(content)) => {
                    serde_json::to_value(&content.value)?
                }
            };
            if output.is_null() {
                continue;
            }
            out.push(UIMessageStreamEvent::ToolOutputAvailable(
                UIToolOutputAvailableEvent {
                    tool_call_id: part.tool_call_id.clone(),
                    output,
                    provider_metadata: part.provider_metadata.clone(),
                    ..Default::default()
                },
            ));
        }
        Ok(out)
    }

    fn on_partial_tool_result(
        &mut self,
        event: &PartialToolCallResult,
    ) -> Result<Vec<UIMessageStreamEvent>, Error> {
        let (Some(id), Some(factory)) = (&event.tool_call_id, &event.aggregator_factory) else {
            return Ok(Vec::new());
        };
        let mut out = self.ensure_started(None);
        let aggregator = self
            .partial_aggregators
            .entry(id.clone())
            .or_insert_with(|| factory.create());
        aggregator.feed(event.value.clone());
        let output = match aggregator.snapshot_result() {
            ToolResultValue::Json(value) => value,
            ToolResultValue::Special(SpecialToolResult::Messages(bundle)) => {
                outbound_messages::bundle_to_wire_output(&bundle)?
            }
            ToolResultValue::Special(SpecialToolResult::Content(content)) => {
                serde_json::to_value(content)?
            }
        };
        if !output.is_null() {
            out.push(UIMessageStreamEvent::ToolOutputAvailable(
                UIToolOutputAvailableEvent {
                    tool_call_id: id.clone(),
                    output,
                    preliminary: Some(true),
                    ..Default::default()
                },
            ));
        }
        Ok(out)
    }

    fn on_hook(&mut self, event: &HookEvent) -> Vec<UIMessageStreamEvent> {
        self.track_source_message(&event.message);
        let mut out = self.ensure_started(event.message.turn_id.clone());
        let hook = &event.hook;
        let Some(id) = approvals::tool_call_id_for(hook) else {
            return out;
        };
        match hook.status {
            HookStatus::Pending => {
                if self.emitted_approval_requests.insert(id.to_owned()) {
                    out.push(UIMessageStreamEvent::ToolApprovalRequest(
                        UIToolApprovalRequestEvent {
                            approval_id: hook.hook_id.clone(),
                            tool_call_id: id.into(),
                            is_automatic: hook.metadata.get("isAutomatic").and_then(Value::as_bool),
                        },
                    ));
                }
            }
            HookStatus::Resolved => {
                let granted = hook
                    .resolution
                    .as_ref()
                    .and_then(|v| v.get("granted"))
                    .is_some_and(python_truthy);
                let provider_metadata = hook
                    .metadata
                    .get("callProviderMetadata")
                    .and_then(Value::as_object)
                    .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect());
                out.push(UIMessageStreamEvent::ToolApprovalResponse(
                    UIToolApprovalResponseEvent {
                        approval_id: hook.hook_id.clone(),
                        approved: granted,
                        reason: hook
                            .resolution
                            .as_ref()
                            .and_then(|v| v.get("reason"))
                            .and_then(Value::as_str)
                            .map(str::to_owned),
                        provider_executed: hook
                            .metadata
                            .get("providerExecuted")
                            .and_then(Value::as_bool),
                        provider_metadata,
                    },
                ));
                if !granted {
                    out.push(UIMessageStreamEvent::ToolOutputDenied(
                        UIToolOutputDeniedEvent {
                            tool_call_id: id.into(),
                        },
                    ));
                }
            }
            HookStatus::Cancelled => out.push(UIMessageStreamEvent::ToolOutputError(
                UIToolOutputErrorEvent {
                    tool_call_id: id.into(),
                    error_text: "Hook cancelled".into(),
                    ..Default::default()
                },
            )),
        }
        out
    }

    fn finish(&mut self) -> Result<Vec<UIMessageStreamEvent>, Error> {
        let mut out = Vec::new();
        for id in std::mem::take(&mut self.open_reasoning_ids) {
            out.push(UIMessageStreamEvent::ReasoningEnd(UIReasoningEndEvent {
                id,
                ..Default::default()
            }));
        }
        for id in std::mem::take(&mut self.open_text_ids) {
            out.push(UIMessageStreamEvent::TextEnd(UITextEndEvent {
                id,
                ..Default::default()
            }));
        }
        if self.in_step {
            out.push(UIMessageStreamEvent::FinishStep(UIFinishStepEvent {}));
            self.in_step = false;
        }
        if self.emitted_start {
            let messages: Vec<_> = self
                .source_messages
                .iter()
                .filter(|m| m.role != MessageRole::System && !m.parts.is_empty())
                .cloned()
                .collect();
            let ui_messages = outbound_messages::to_ui_messages(&messages)?;
            let message_metadata = ui_messages
                .into_iter()
                .rev()
                .find(|m| m.role == "assistant")
                .map(|m| m.metadata)
                .filter(|metadata| !metadata.is_null());
            out.push(UIMessageStreamEvent::Finish(UIFinishEvent {
                finish_reason: Some(finish_reason_to_ui(self.finish_reason.as_deref())),
                message_metadata,
            }));
        }
        Ok(out)
    }
}

fn prefer_metadata(first: &Option<Metadata>, second: &Option<Metadata>) -> Option<Metadata> {
    first
        .as_ref()
        .filter(|m| !m.is_empty())
        .cloned()
        .or_else(|| second.clone())
}

fn tool_error_text(result: &ToolResultValue) -> String {
    if let Some(value) = result.as_json() {
        if let Some(text) = value.as_str().filter(|s| !s.is_empty()) {
            return text.into();
        }
        for key in ["error", "message", "detail"] {
            if let Some(text) = value
                .get(key)
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
            {
                return text.into();
            }
        }
    }
    "Tool execution failed".into()
}

fn finish_reason_to_ui(reason: Option<&str>) -> FinishReason {
    match reason {
        None | Some("stop") => FinishReason::Stop,
        Some("length") => FinishReason::Length,
        Some("content_filter") => FinishReason::ContentFilter,
        Some("tool_call") => FinishReason::ToolCalls,
        Some("error") => FinishReason::Error,
        _ => FinishReason::Other,
    }
}

/// Convert a fallible core stream lazily. A source error terminates the stream
/// without closing blocks or emitting a successful finish.
pub fn to_stream<S>(events: S) -> BoxStream<'static, Result<UIMessageStreamEvent, Error>>
where
    S: Stream<Item = Result<Event, Error>> + Send + 'static,
{
    stream::unfold(
        (
            events.boxed(),
            StreamState::default(),
            VecDeque::new(),
            false,
        ),
        |(mut source, mut state, mut pending, mut finished)| async move {
            loop {
                if let Some(event) = pending.pop_front() {
                    return Some((event, (source, state, pending, finished)));
                }
                if finished {
                    return None;
                }
                let batch = match source.next().await {
                    Some(Ok(event)) => state.on_event(&event),
                    Some(Err(error)) => Err(error),
                    None => {
                        finished = true;
                        state.finish()
                    }
                };
                match batch {
                    Ok(events) => pending.extend(events.into_iter().map(Ok)),
                    Err(error) => {
                        finished = true;
                        pending.push_back(Err(error));
                    }
                }
            }
        },
    )
    .boxed()
}

/// Serialize camelCase protocol fields with Python json.dumps spacing and
/// ASCII escaping. Nested JSON nulls remain intact.
pub fn serialize_event(event: &UIMessageStreamEvent) -> Result<String, Error> {
    python_json_dumps(event)
}
pub fn format_sse(event: &UIMessageStreamEvent) -> Result<String, Error> {
    Ok(format!("data: {}\n\n", serialize_event(event)?))
}
#[must_use]
pub const fn format_done_sse() -> &'static str {
    "data: [DONE]\n\n"
}

/// Encode SSE frames and append `[DONE]` only after successful exhaustion.
pub fn to_sse<S>(events: S) -> BoxStream<'static, Result<String, Error>>
where
    S: Stream<Item = Result<Event, Error>> + Send + 'static,
{
    stream::unfold(
        (to_stream(events), false),
        |(mut events, done)| async move {
            if done {
                return None;
            }
            match events.next().await {
                Some(Ok(event)) => {
                    let result = format_sse(&event);
                    let failed = result.is_err();
                    Some((result, (events, failed)))
                }
                Some(Err(error)) => Some((Err(error), (events, true))),
                None => Some((Ok(format_done_sse().into()), (events, true))),
            }
        },
    )
    .boxed()
}
