//! Interactions stream events as model events
//! (`build-google-interactions-stream-transform.ts`). Steps are framed by
//! index, so a text delta at one index never meets a thought at another.

use std::collections::{HashMap, VecDeque};

use serde_json::{Value, json};

use super::output::{self, Sources, builtin_name, file, part_metadata};
use crate::{
    Error,
    errors::{ProviderError, ProviderErrorKind},
    providers::google::transcription::interactions_usage,
    types::{
        events::*,
        messages::{BuiltinToolCallPart, BuiltinToolReturnPart, ToolCallPart, generate_id},
    },
};

enum Open {
    Pending,
    Text {
        id: String,
    },
    Reasoning {
        id: String,
        signature: Option<String>,
    },
    Image {
        data: Option<String>,
        mime: Option<String>,
        uri: Option<String>,
    },
    /// A function call starts once its id is known: from the step, from its
    /// first argument delta, or the block id at its stop.
    FunctionCall {
        call_id: Option<String>,
        block: String,
        started: bool,
        name: String,
        args: String,
        signature: Option<String>,
    },
    BuiltinCall {
        kind: String,
        call_id: String,
        name: String,
        args: Value,
    },
    BuiltinResult {
        kind: String,
        call_id: String,
        name: String,
        result: Value,
        is_error: bool,
    },
    Processing {
        call: bool,
        id: String,
        signature: Option<String>,
    },
    Unknown,
}

/// Reads Interactions stream events.
pub(crate) struct Reader {
    provider: String,
    pub(crate) events: VecDeque<Event>,
    open: HashMap<u64, Open>,
    sources: Sources,
    interaction: Option<String>,
    model: Option<String>,
    usage: Option<Value>,
    service_tier: Option<String>,
    status: Option<String>,
    has_function_call: bool,
    completed: bool,
    /// The step parts this reader started: their ids are unique within the
    /// message, so a block id from one interaction never meets another's.
    prefix: String,
}

impl Reader {
    pub(crate) fn new(provider: String, header_tier: Option<String>) -> Self {
        Self {
            provider,
            events: VecDeque::from([Event::StreamStart(StreamStart::default())]),
            open: HashMap::new(),
            sources: Sources::default(),
            interaction: None,
            model: None,
            usage: None,
            service_tier: header_tier,
            status: None,
            has_function_call: false,
            completed: false,
            prefix: generate_id("block"),
        }
    }

    /// Whether the interaction completed or failed: nothing follows.
    pub(crate) fn done(&self) -> bool {
        self.completed || self.status.as_deref() == Some("failed")
    }

    fn block_id(&self, index: u64) -> String {
        format!(
            "{}:{}:{index}",
            self.prefix,
            self.interaction.as_deref().unwrap_or("interaction")
        )
    }

    fn meta(provider_metadata: Option<crate::types::messages::Metadata>) -> ModelEventMeta {
        ModelEventMeta {
            provider_metadata,
            ..Default::default()
        }
    }

    fn interaction_meta(&self) -> Option<crate::types::messages::Metadata> {
        part_metadata(&[("interactionId", self.interaction.as_deref())])
    }

    fn push_file(&mut self, block: &Value, default: &str) {
        let mime = block
            .get("mime_type")
            .and_then(Value::as_str)
            .unwrap_or(default);
        let data = block.get("data").and_then(Value::as_str);
        let uri = block.get("uri").and_then(Value::as_str);
        if let Some(part) = file(mime, data, uri, self.interaction.as_deref()) {
            self.events.push_back(Event::File(FileEvent {
                event: Self::meta(part.provider_metadata.clone()),
                block_id: part.id,
                media_type: part.media_type,
                data: part.data,
                filename: None,
            }));
        }
    }

    /// Read one event.
    pub(crate) fn event(&mut self, value: &Value) -> Result<(), Error> {
        let kind = value
            .get("event_type")
            .and_then(Value::as_str)
            .unwrap_or_default();
        match kind {
            "interaction.created" => {
                let interaction = value.get("interaction");
                self.interaction = interaction
                    .and_then(|i| i.get("id"))
                    .and_then(Value::as_str)
                    .filter(|id| !id.is_empty())
                    .map(str::to_owned);
                self.model = interaction
                    .and_then(|i| i.get("model"))
                    .and_then(Value::as_str)
                    .map(str::to_owned);
            }
            "step.start" => self.step_start(value),
            "step.delta" => self.step_delta(value),
            "step.stop" => self.step_stop(value),
            "interaction.status_update"
            | "interaction.in_progress"
            | "interaction.requires_action" => {
                self.status = Some(
                    value
                        .get("status")
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                        .unwrap_or_else(|| {
                            if kind == "interaction.requires_action" {
                                "requires_action"
                            } else {
                                "in_progress"
                            }
                            .to_owned()
                        }),
                );
            }
            "interaction.completed" => {
                self.completed = true;
                if let Some(interaction) = value.get("interaction") {
                    if let Some(id) = interaction
                        .get("id")
                        .and_then(Value::as_str)
                        .filter(|id| !id.is_empty())
                    {
                        self.interaction = Some(id.to_owned());
                    }
                    if let Some(status) = interaction.get("status").and_then(Value::as_str) {
                        self.status = Some(status.to_owned());
                    }
                    if let Some(usage) = interaction.get("usage").filter(|u| u.is_object()) {
                        self.usage = Some(usage.clone());
                    }
                    if let Some(tier) = interaction.get("service_tier").and_then(Value::as_str) {
                        self.service_tier = Some(tier.to_owned());
                    }
                }
            }
            "error" => {
                self.status = Some("failed".into());
                let error = value.get("error");
                let message = error
                    .and_then(|e| e.get("message"))
                    .and_then(Value::as_str)
                    .unwrap_or("Unknown interaction error");
                let code = error.and_then(|e| e.get("code")).filter(|c| !c.is_null());
                let status = code
                    .and_then(Value::as_u64)
                    .and_then(|c| u16::try_from(c).ok());
                let mut failure = ProviderError::new(
                    status.map_or(ProviderErrorKind::Provider, ProviderErrorKind::from_status),
                    message,
                )
                .with_provider(&self.provider)
                .with_body(value.clone())
                .with_error_type("error");
                if let Some(code) = code {
                    failure.code = Some(match code {
                        Value::String(text) => text.clone(),
                        other => other.to_string(),
                    });
                }
                if let Some(status) = status {
                    failure.is_retryable = matches!(status, 408 | 409 | 429) || status >= 500;
                }
                return Err(failure.into());
            }
            _ => {}
        }
        Ok(())
    }

    fn step_start(&mut self, value: &Value) {
        let Some(index) = value.get("index").and_then(Value::as_u64) else {
            return;
        };
        let step = value.get("step").cloned().unwrap_or(Value::Null);
        let block = self.block_id(index);
        let kind = step.get("type").and_then(Value::as_str).unwrap_or_default();
        let string = |key: &str| {
            step.get(key)
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
        };
        let state = match kind {
            "model_output" => {
                let initial = step
                    .get("content")
                    .and_then(Value::as_array)
                    .and_then(|c| c.first());
                match initial.and_then(|i| i.get("type")).and_then(Value::as_str) {
                    Some("text") => {
                        self.events.push_back(Event::TextStart(TextStart {
                            block_id: block.clone(),
                            ..Default::default()
                        }));
                        self.sources
                            .annotations(initial.and_then(|i| i.get("annotations")));
                        Open::Text { id: block }
                    }
                    Some("image") => {
                        let initial = initial.cloned().unwrap_or(Value::Null);
                        Open::Image {
                            data: initial
                                .get("data")
                                .and_then(Value::as_str)
                                .map(str::to_owned),
                            mime: initial
                                .get("mime_type")
                                .and_then(Value::as_str)
                                .map(str::to_owned),
                            uri: initial
                                .get("uri")
                                .and_then(Value::as_str)
                                .map(str::to_owned),
                        }
                    }
                    _ => Open::Pending,
                }
            }
            "thought" => {
                self.events.push_back(Event::ReasoningStart(ReasoningStart {
                    block_id: block.clone(),
                    ..Default::default()
                }));
                for item in step
                    .get("summary")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    if item.get("type").and_then(Value::as_str) == Some("text")
                        && let Some(text) = item.get("text").and_then(Value::as_str)
                    {
                        self.events.push_back(Event::ReasoningDelta(ReasoningDelta {
                            event: ModelEventMeta::default(),
                            chunk: text.to_owned(),
                            block_id: block.clone(),
                        }));
                    }
                }
                Open::Reasoning {
                    id: block,
                    signature: string("signature"),
                }
            }
            "processing_call" => Open::Processing {
                call: true,
                id: string("id").unwrap_or_else(|| block.clone()),
                signature: string("signature"),
            },
            "processing_result" => Open::Processing {
                call: false,
                id: string("call_id").unwrap_or_else(|| block.clone()),
                signature: string("signature"),
            },
            "function_call" => {
                let call_id = string("id");
                let name = step
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown")
                    .to_owned();
                self.has_function_call = true;
                let started = call_id.is_some();
                if let Some(id) = &call_id {
                    self.events.push_back(Event::ToolStart(ToolStart {
                        event: ModelEventMeta::default(),
                        tool_call_id: id.clone(),
                        tool_name: name.clone(),
                    }));
                }
                Open::FunctionCall {
                    call_id,
                    block,
                    started,
                    name,
                    args: String::new(),
                    signature: string("signature"),
                }
            }
            other => match builtin_name(other, step.get("name").and_then(Value::as_str)) {
                Some(name) if other.ends_with("_call") => Open::BuiltinCall {
                    kind: other.to_owned(),
                    call_id: string("id").unwrap_or_else(|| block.clone()),
                    name,
                    args: step
                        .get("arguments")
                        .filter(|a| !a.is_null())
                        .cloned()
                        .unwrap_or_else(|| json!({})),
                },
                Some(name) => Open::BuiltinResult {
                    kind: other.to_owned(),
                    call_id: string("call_id").unwrap_or_else(|| block.clone()),
                    name,
                    result: step.get("result").cloned().unwrap_or(Value::Null),
                    is_error: step.get("is_error") == Some(&Value::Bool(true)),
                },
                None => Open::Unknown,
            },
        };
        self.open.insert(index, state);
    }

    fn step_delta(&mut self, value: &Value) {
        let Some(index) = value.get("index").and_then(Value::as_u64) else {
            return;
        };
        let delta = value.get("delta").cloned().unwrap_or(Value::Null);
        let kind = delta
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        let block = self.block_id(index);
        if matches!(self.open.get(&index), Some(Open::Pending))
            && matches!(
                kind.as_str(),
                "text" | "text_annotation" | "text_annotation_delta"
            )
        {
            self.events.push_back(Event::TextStart(TextStart {
                block_id: block.clone(),
                ..Default::default()
            }));
            self.open.insert(index, Open::Text { id: block.clone() });
        }
        let Some(open) = self.open.get_mut(&index) else {
            return;
        };
        if kind == "image" && matches!(open, Open::Pending | Open::Text { .. } | Open::Image { .. })
        {
            if let Open::Image { data, uri, .. } = open {
                *data = None;
                *uri = None;
            }
            self.push_file(&delta, "image/png");
            return;
        }
        if kind == "video" && matches!(open, Open::Pending | Open::Text { .. }) {
            self.push_file(&delta, "video/mp4");
            return;
        }
        let string = |key: &str| {
            delta
                .get(key)
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
        };
        match open {
            Open::Processing {
                call,
                id,
                signature,
            } if (kind == "processing_call" && *call)
                || (kind == "processing_result" && !*call) =>
            {
                if let Some(value) = string("signature") {
                    *signature = Some(value);
                }
                let key = if *call { "id" } else { "call_id" };
                if let Some(value) = string(key) {
                    *id = value;
                }
            }
            Open::Text { id } if kind == "text" => {
                if let Some(text) = delta
                    .get("text")
                    .and_then(Value::as_str)
                    .filter(|t| !t.is_empty())
                {
                    let id = id.clone();
                    self.events.push_back(Event::TextDelta(TextDelta {
                        event: ModelEventMeta::default(),
                        chunk: text.to_owned(),
                        block_id: id,
                    }));
                }
            }
            Open::Text { .. } if kind == "text_annotation" || kind == "text_annotation_delta" => {
                self.sources.annotations(delta.get("annotations"));
            }
            Open::Image { data, mime, uri } if kind == "image" => {
                if let Some(value) = string("data") {
                    *data = Some(value);
                }
                if let Some(value) = string("mime_type") {
                    *mime = Some(value);
                }
                if let Some(value) = string("uri") {
                    *uri = Some(value);
                }
            }
            Open::Reasoning { id, signature } => {
                if kind == "thought_summary" {
                    let item = delta.get("content");
                    if item.and_then(|i| i.get("type")).and_then(Value::as_str) == Some("text")
                        && let Some(text) = item.and_then(|i| i.get("text")).and_then(Value::as_str)
                    {
                        let id = id.clone();
                        self.events.push_back(Event::ReasoningDelta(ReasoningDelta {
                            event: ModelEventMeta::default(),
                            chunk: text.to_owned(),
                            block_id: id,
                        }));
                    }
                } else if kind == "thought_signature"
                    && let Some(value) = delta.get("signature").and_then(Value::as_str)
                {
                    *signature = Some(value.to_owned());
                }
            }
            Open::FunctionCall {
                call_id,
                block,
                started,
                name,
                args,
                signature,
            } if kind == "arguments_delta" => {
                if !*started && let Some(value) = string("id") {
                    *call_id = Some(value);
                }
                if let Some(slice) = delta
                    .get("arguments")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                {
                    let id = call_id.get_or_insert_with(|| block.clone()).clone();
                    if !*started {
                        *started = true;
                        self.events.push_back(Event::ToolStart(ToolStart {
                            event: ModelEventMeta::default(),
                            tool_call_id: id.clone(),
                            tool_name: name.clone(),
                        }));
                    }
                    args.push_str(slice);
                    self.events.push_back(Event::ToolDelta(ToolDelta {
                        event: ModelEventMeta::default(),
                        chunk: slice.to_owned(),
                        tool_call_id: id,
                    }));
                }
                if let Some(value) = delta.get("signature").and_then(Value::as_str) {
                    *signature = Some(value.to_owned());
                }
                self.has_function_call = true;
            }
            Open::BuiltinCall {
                kind: block_kind,
                call_id,
                name,
                args,
            } if *block_kind == kind => {
                if let Some(value) = string("id") {
                    *call_id = value;
                }
                if let Some(value) = delta.get("arguments").filter(|a| a.is_object()) {
                    *args = value.clone();
                }
                if block_kind == "mcp_server_tool_call"
                    && let Some(value) = delta.get("name").and_then(Value::as_str)
                {
                    *name = value.to_owned();
                }
            }
            Open::BuiltinResult {
                kind: block_kind,
                call_id,
                name,
                result,
                is_error,
            } if *block_kind == kind => {
                if let Some(value) = string("call_id") {
                    *call_id = value;
                }
                if let Some(value) = delta.get("result") {
                    *result = value.clone();
                }
                if let Some(value) = delta.get("is_error").and_then(Value::as_bool) {
                    *is_error = value;
                }
                if block_kind == "mcp_server_tool_result"
                    && let Some(value) = delta.get("name").and_then(Value::as_str)
                {
                    *name = value.to_owned();
                }
            }
            _ => {}
        }
    }

    fn step_stop(&mut self, value: &Value) {
        let Some(index) = value.get("index").and_then(Value::as_u64) else {
            return;
        };
        let Some(open) = self.open.remove(&index) else {
            return;
        };
        let interaction = self.interaction.clone();
        let interaction = interaction.as_deref();
        match open {
            Open::Text { id } => self.events.push_back(Event::TextEnd(TextEnd {
                event: Self::meta(part_metadata(&[("interactionId", interaction)])),
                block_id: id,
            })),
            Open::Reasoning { id, signature } => {
                self.events.push_back(Event::ReasoningEnd(ReasoningEnd {
                    event: Self::meta(part_metadata(&[
                        ("signature", signature.as_deref()),
                        ("interactionId", interaction),
                    ])),
                    block_id: id,
                }))
            }
            Open::Image { data, mime, uri } => {
                let block = json!({"data": data, "mime_type": mime, "uri": uri});
                self.push_file(&block, "image/png");
            }
            Open::FunctionCall {
                call_id,
                block,
                started,
                name,
                args,
                signature,
            } => {
                let call_id = call_id.unwrap_or(block);
                if !started {
                    self.events.push_back(Event::ToolStart(ToolStart {
                        event: ModelEventMeta::default(),
                        tool_call_id: call_id.clone(),
                        tool_name: name.clone(),
                    }));
                }
                if args.is_empty() {
                    self.events.push_back(Event::ToolDelta(ToolDelta {
                        event: ModelEventMeta::default(),
                        chunk: "{}".into(),
                        tool_call_id: call_id.clone(),
                    }));
                }
                self.events.push_back(Event::ToolEnd(ToolEnd {
                    event: Self::meta(part_metadata(&[
                        ("signature", signature.as_deref()),
                        ("interactionId", interaction),
                    ])),
                    tool_call: ToolCallPart::new(&call_id, name, ""),
                    tool_call_id: call_id,
                }));
            }
            Open::Processing {
                call,
                id,
                signature,
            } => {
                let key = if call {
                    "processingId"
                } else {
                    "processingCallId"
                };
                let metadata = part_metadata(&[
                    ("signature", signature.as_deref()),
                    ("interactionId", interaction),
                    (key, Some(&id)),
                ]);
                if call {
                    self.builtin_call(id, "processing".into(), "{}".into(), metadata);
                } else {
                    self.builtin_result(id, "processing".into(), Value::Null, false, metadata);
                }
            }
            Open::BuiltinCall {
                call_id,
                name,
                args,
                ..
            } => {
                let metadata = self.interaction_meta();
                self.builtin_call(call_id, name, args.to_string(), metadata);
            }
            Open::BuiltinResult {
                kind,
                call_id,
                name,
                result,
                is_error,
            } => {
                self.sources.tool_result(&kind, Some(&result));
                let metadata = self.interaction_meta();
                self.builtin_result(call_id, name, result, is_error, metadata);
            }
            Open::Pending | Open::Unknown => {}
        }
    }

    fn builtin_call(
        &mut self,
        id: String,
        name: String,
        args: String,
        metadata: Option<crate::types::messages::Metadata>,
    ) {
        self.events
            .push_back(Event::BuiltinToolStart(BuiltinToolStart {
                event: Self::meta(metadata.clone()),
                tool_call_id: id.clone(),
                tool_name: name.clone(),
            }));
        self.events
            .push_back(Event::BuiltinToolDelta(BuiltinToolDelta {
                event: ModelEventMeta::default(),
                chunk: args,
                tool_call_id: id.clone(),
            }));
        self.events.push_back(Event::BuiltinToolEnd(BuiltinToolEnd {
            event: Self::meta(metadata.clone()),
            tool_call: BuiltinToolCallPart {
                id: id.clone(),
                tool_call_id: id.clone(),
                tool_name: name,
                tool_args: String::new(),
                provider_metadata: metadata,
            },
            tool_call_id: id,
        }));
    }

    fn builtin_result(
        &mut self,
        id: String,
        name: String,
        result: Value,
        is_error: bool,
        metadata: Option<crate::types::messages::Metadata>,
    ) {
        self.events
            .push_back(Event::BuiltinToolResult(BuiltinToolResult {
                event: Self::meta(metadata.clone()),
                result: BuiltinToolReturnPart {
                    id: generate_id("part"),
                    tool_call_id: id.clone(),
                    tool_name: name,
                    result,
                    is_error,
                    provider_metadata: metadata,
                },
                tool_call_id: id,
            }));
    }

    /// End the stream: the finish event, or nothing when the interaction was
    /// interrupted while in progress.
    pub(crate) fn end(&mut self) {
        let interrupted =
            !self.completed && self.status.as_deref().is_none_or(|s| s == "in_progress");
        if interrupted {
            return;
        }
        let finish = output::finish_reason(self.status.as_deref(), self.has_function_call);
        let metadata = output::response_metadata(
            self.interaction.as_deref(),
            self.service_tier.as_deref(),
            self.usage.as_ref(),
            &self.sources.list,
        );
        self.events.push_back(Event::StreamEnd(StreamEnd {
            event: ModelEventMeta {
                usage: self.usage.as_ref().map(interactions_usage),
                provider_metadata: Some(metadata),
                ..Default::default()
            },
            finish_reason: Some(finish.into()),
            response_id: self.interaction.clone(),
            response_model: self.model.clone(),
        }));
    }
}
