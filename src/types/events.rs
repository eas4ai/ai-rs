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

use std::collections::{BTreeMap, BTreeSet, HashMap};

use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error as _, ser::Error as _};
use serde_json::Value;
use thiserror::Error;

mod runtime;
pub use runtime::{AggregatorFactory, RuntimeError, ValueAggregator};

use super::{
    messages::{
        BuiltinToolCallPart, BuiltinToolReturnPart, FileData, FilePart, HookPart, HookStatus,
        Message, MessageRole, Metadata, Part, ReasoningPart, TextPart, ToolCallPart,
        ToolResultPart,
    },
    usage::Usage,
};

fn dummy_message() -> Message {
    Message::with_id("<unset>", MessageRole::Assistant, Vec::new())
}

fn deserialize_event_message<'de, D>(deserializer: D) -> Result<Message, D::Error>
where
    D: Deserializer<'de>,
{
    let mut value = Value::deserialize(deserializer)?;
    if let Value::Object(object) = &mut value
        && object.len() == 1
        && object.contains_key("id")
    {
        object.insert("role".to_owned(), Value::String("assistant".to_owned()));
        object.insert("parts".to_owned(), Value::Array(Vec::new()));
    }
    serde_json::from_value(value).map_err(D::Error::custom)
}

/// Fields common to all events emitted by a model stream.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ModelEventMeta {
    #[serde(
        default = "dummy_message",
        deserialize_with = "deserialize_event_message"
    )]
    pub message: Message,
    #[serde(default)]
    pub usage: Option<Usage>,
    #[serde(default)]
    pub provider_metadata: Option<Metadata>,
    /// Runtime control flag. It is excluded from the durable wire format.
    #[serde(default, skip_serializing)]
    #[schemars(skip)]
    pub replay: bool,
}

impl Default for ModelEventMeta {
    fn default() -> Self {
        Self {
            message: dummy_message(),
            usage: None,
            provider_metadata: None,
            replay: false,
        }
    }
}

impl ModelEventMeta {
    #[must_use]
    pub fn with_message(message: Message) -> Self {
        Self {
            message,
            ..Self::default()
        }
    }
}

macro_rules! simple_model_event {
    ($name:ident) => {
        #[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
        pub struct $name {
            #[serde(flatten)]
            pub event: ModelEventMeta,
        }
    };
}

simple_model_event!(StreamStart);

macro_rules! block_model_event {
    ($name:ident) => {
        #[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
        pub struct $name {
            #[serde(flatten)]
            pub event: ModelEventMeta,
            #[serde(default)]
            pub block_id: String,
        }
    };
}

block_model_event!(TextStart);
block_model_event!(TextEnd);
block_model_event!(ReasoningStart);
block_model_event!(ReasoningEnd);

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct StreamEnd {
    #[serde(flatten)]
    pub event: ModelEventMeta,
    #[serde(default)]
    pub finish_reason: Option<String>,
    #[serde(default)]
    pub response_id: Option<String>,
    #[serde(default)]
    pub response_model: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct TextDelta {
    #[serde(flatten)]
    pub event: ModelEventMeta,
    pub chunk: String,
    #[serde(default)]
    pub block_id: String,
}

impl TextDelta {
    #[must_use]
    pub fn new(chunk: impl Into<String>) -> Self {
        Self {
            event: ModelEventMeta::default(),
            chunk: chunk.into(),
            block_id: String::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ReasoningDelta {
    #[serde(flatten)]
    pub event: ModelEventMeta,
    pub chunk: String,
    #[serde(default)]
    pub block_id: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ToolStart {
    #[serde(flatten)]
    pub event: ModelEventMeta,
    #[serde(default)]
    pub tool_call_id: String,
    #[serde(default)]
    pub tool_name: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ToolDelta {
    #[serde(flatten)]
    pub event: ModelEventMeta,
    pub chunk: String,
    #[serde(default)]
    pub tool_call_id: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ToolEnd {
    #[serde(flatten)]
    pub event: ModelEventMeta,
    #[serde(with = "tool_call_wire")]
    #[schemars(with = "Part")]
    pub tool_call: ToolCallPart,
    #[serde(default)]
    pub tool_call_id: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct BuiltinToolStart {
    #[serde(flatten)]
    pub event: ModelEventMeta,
    #[serde(default)]
    pub tool_call_id: String,
    #[serde(default)]
    pub tool_name: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct BuiltinToolDelta {
    #[serde(flatten)]
    pub event: ModelEventMeta,
    pub chunk: String,
    #[serde(default)]
    pub tool_call_id: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct BuiltinToolEnd {
    #[serde(flatten)]
    pub event: ModelEventMeta,
    #[serde(with = "builtin_tool_call_wire")]
    #[schemars(with = "Part")]
    pub tool_call: BuiltinToolCallPart,
    #[serde(default)]
    pub tool_call_id: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct BuiltinToolResult {
    #[serde(flatten)]
    pub event: ModelEventMeta,
    #[serde(with = "builtin_tool_return_wire")]
    #[schemars(with = "Part")]
    pub result: BuiltinToolReturnPart,
    #[serde(default)]
    pub tool_call_id: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct FileEvent {
    #[serde(flatten)]
    pub event: ModelEventMeta,
    #[serde(default)]
    pub block_id: String,
    pub media_type: String,
    #[schemars(with = "String")]
    pub data: FileData,
    #[serde(default)]
    pub filename: Option<String>,
}

/// Stateful reducer for values yielded by a tool call.
pub trait Aggregator<Item> {
    type Snapshot;
    type ModelInput;

    fn feed(&mut self, item: Item);
    fn snapshot(&self) -> Self::Snapshot;

    fn get_model_input(&self) -> Self::ModelInput {
        Self::to_model_input(self.snapshot())
    }

    fn to_model_input(snapshot: Self::Snapshot) -> Self::ModelInput;
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct PartialToolCallResult {
    #[serde(default, skip_serializing)]
    #[schemars(skip)]
    pub replay: bool,
    #[serde(default)]
    pub tool_call_id: Option<String>,
    #[serde(default)]
    pub tool_name: Option<String>,
    #[serde(default)]
    pub label: Value,
    #[serde(default)]
    pub value: Value,
    #[serde(default, skip)]
    #[schemars(skip)]
    pub aggregator_factory: Option<AggregatorFactory>,
}

impl PartialToolCallResult {
    #[must_use]
    pub fn key(&self) -> (Option<&str>, &Value) {
        (self.tool_call_id.as_deref(), &self.label)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ToolCallResult {
    #[serde(default, skip_serializing)]
    #[schemars(skip)]
    pub replay: bool,
    pub message: Message,
    #[serde(with = "tool_results_wire")]
    #[schemars(with = "Vec<Part>")]
    pub results: Vec<ToolResultPart>,
    /// Original runtime error, excluded from durable serialization.
    #[serde(default, skip)]
    #[schemars(skip)]
    pub exception: Option<RuntimeError>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct HookEvent {
    #[serde(default, skip_serializing)]
    #[schemars(skip)]
    pub replay: bool,
    pub message: Message,
    #[serde(with = "hook_wire")]
    #[schemars(with = "Part")]
    pub hook: HookPart,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RunBlocked {
    #[serde(default, skip_serializing)]
    #[schemars(skip)]
    pub replay: bool,
    #[serde(default)]
    #[serde(with = "hooks_wire")]
    #[schemars(with = "Vec<Part>")]
    pub hooks: Vec<HookPart>,
}

macro_rules! part_wire_module {
    ($module:ident, $type:ty, $variant:ident, $expected:literal) => {
        mod $module {
            use super::*;

            pub fn serialize<S>(value: &$type, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: Serializer,
            {
                Part::$variant(value.clone()).serialize(serializer)
            }

            pub fn deserialize<'de, D>(deserializer: D) -> Result<$type, D::Error>
            where
                D: Deserializer<'de>,
            {
                match Part::deserialize(deserializer)? {
                    Part::$variant(value) => Ok(value),
                    _ => Err(D::Error::custom($expected)),
                }
            }
        }
    };
}

part_wire_module!(
    tool_call_wire,
    ToolCallPart,
    ToolCall,
    "tool_call must have kind=tool_call"
);
part_wire_module!(
    builtin_tool_call_wire,
    BuiltinToolCallPart,
    BuiltinToolCall,
    "tool_call must have kind=builtin_tool_call"
);
part_wire_module!(
    builtin_tool_return_wire,
    BuiltinToolReturnPart,
    BuiltinToolReturn,
    "result must have kind=builtin_tool_return"
);
part_wire_module!(hook_wire, HookPart, Hook, "hook must have kind=hook");

mod tool_results_wire {
    use super::*;

    pub fn serialize<S>(values: &[ToolResultPart], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        values
            .iter()
            .cloned()
            .map(Part::ToolResult)
            .collect::<Vec<_>>()
            .serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<ToolResultPart>, D::Error>
    where
        D: Deserializer<'de>,
    {
        Vec::<Part>::deserialize(deserializer)?
            .into_iter()
            .map(|part| match part {
                Part::ToolResult(value) => Ok(value),
                _ => Err(D::Error::custom("results must have kind=tool_result")),
            })
            .collect()
    }
}

mod hooks_wire {
    use super::*;

    pub fn serialize<S>(values: &[HookPart], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        values
            .iter()
            .cloned()
            .map(Part::Hook)
            .collect::<Vec<_>>()
            .serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<HookPart>, D::Error>
    where
        D: Deserializer<'de>,
    {
        Vec::<Part>::deserialize(deserializer)?
            .into_iter()
            .map(|part| match part {
                Part::Hook(value) => Ok(value),
                _ => Err(D::Error::custom("hooks must have kind=hook")),
            })
            .collect()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Retry {
    #[serde(default, skip_serializing)]
    #[schemars(skip)]
    pub replay: bool,
}

/// Every event emitted by model streams and the agent runtime.
///
/// Serialize this enum, rather than an individual payload struct, when producing
/// the Python-compatible wire format. The enum supplies the canonical `kind`
/// discriminator.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Event {
    StreamStart(StreamStart),
    StreamEnd(StreamEnd),
    TextStart(TextStart),
    TextDelta(TextDelta),
    TextEnd(TextEnd),
    ReasoningStart(ReasoningStart),
    ReasoningDelta(ReasoningDelta),
    ReasoningEnd(ReasoningEnd),
    ToolStart(ToolStart),
    ToolDelta(ToolDelta),
    ToolEnd(ToolEnd),
    BuiltinToolStart(BuiltinToolStart),
    BuiltinToolDelta(BuiltinToolDelta),
    BuiltinToolEnd(BuiltinToolEnd),
    BuiltinToolResult(BuiltinToolResult),
    File(FileEvent),
    ToolCallResult(ToolCallResult),
    Hook(HookEvent),
    PartialToolCallResult(PartialToolCallResult),
    RunBlocked(RunBlocked),
    Retry(Retry),
}

/// Compatibility name for code that distinguishes agent events from model-only events.
pub type AgentEvent = Event;

impl Event {
    /// Whether this event represents replayed work. This flag never serializes.
    #[must_use]
    pub fn replay(&self) -> bool {
        if let Some(meta) = self.model_event() {
            return meta.replay;
        }
        match self {
            Self::ToolCallResult(event) => event.replay,
            Self::Hook(event) => event.replay,
            Self::PartialToolCallResult(event) => event.replay,
            Self::RunBlocked(event) => event.replay,
            Self::Retry(event) => event.replay,
            _ => false,
        }
    }

    #[must_use]
    pub fn model_event(&self) -> Option<&ModelEventMeta> {
        match self {
            Self::StreamStart(value) => Some(&value.event),
            Self::StreamEnd(value) => Some(&value.event),
            Self::TextStart(value) => Some(&value.event),
            Self::TextDelta(value) => Some(&value.event),
            Self::TextEnd(value) => Some(&value.event),
            Self::ReasoningStart(value) => Some(&value.event),
            Self::ReasoningDelta(value) => Some(&value.event),
            Self::ReasoningEnd(value) => Some(&value.event),
            Self::ToolStart(value) => Some(&value.event),
            Self::ToolDelta(value) => Some(&value.event),
            Self::ToolEnd(value) => Some(&value.event),
            Self::BuiltinToolStart(value) => Some(&value.event),
            Self::BuiltinToolDelta(value) => Some(&value.event),
            Self::BuiltinToolEnd(value) => Some(&value.event),
            Self::BuiltinToolResult(value) => Some(&value.event),
            Self::File(value) => Some(&value.event),
            Self::ToolCallResult(_)
            | Self::Hook(_)
            | Self::PartialToolCallResult(_)
            | Self::RunBlocked(_)
            | Self::Retry(_) => None,
        }
    }

    #[must_use]
    pub fn model_event_mut(&mut self) -> Option<&mut ModelEventMeta> {
        match self {
            Self::StreamStart(value) => Some(&mut value.event),
            Self::StreamEnd(value) => Some(&mut value.event),
            Self::TextStart(value) => Some(&mut value.event),
            Self::TextDelta(value) => Some(&mut value.event),
            Self::TextEnd(value) => Some(&mut value.event),
            Self::ReasoningStart(value) => Some(&mut value.event),
            Self::ReasoningDelta(value) => Some(&mut value.event),
            Self::ReasoningEnd(value) => Some(&mut value.event),
            Self::ToolStart(value) => Some(&mut value.event),
            Self::ToolDelta(value) => Some(&mut value.event),
            Self::ToolEnd(value) => Some(&mut value.event),
            Self::BuiltinToolStart(value) => Some(&mut value.event),
            Self::BuiltinToolDelta(value) => Some(&mut value.event),
            Self::BuiltinToolEnd(value) => Some(&mut value.event),
            Self::BuiltinToolResult(value) => Some(&mut value.event),
            Self::File(value) => Some(&mut value.event),
            Self::ToolCallResult(_)
            | Self::Hook(_)
            | Self::PartialToolCallResult(_)
            | Self::RunBlocked(_)
            | Self::Retry(_) => None,
        }
    }

    #[must_use]
    pub fn message(&self) -> Option<&Message> {
        self.model_event()
            .map(|event| &event.message)
            .or(match self {
                Self::ToolCallResult(value) => Some(&value.message),
                Self::Hook(value) => Some(&value.message),
                _ => None,
            })
    }

    #[must_use]
    pub fn text_delta(chunk: impl Into<String>) -> Self {
        Self::TextDelta(TextDelta::new(chunk))
    }
}

/// Serialization wrapper that replaces an event's full message with its ID.
pub struct OmitEventMessages<'a>(pub &'a Event);

impl Serialize for OmitEventMessages<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut value = serde_json::to_value(self.0).map_err(S::Error::custom)?;
        if self.0.model_event().is_some()
            && let Value::Object(object) = &mut value
            && let Some(Value::Object(message)) = object.get_mut("message")
            && let Some(id) = message.get("id").cloned()
        {
            *message = serde_json::Map::from_iter([("id".to_owned(), id)]);
        }
        value.serialize(serializer)
    }
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum HydrationError {
    #[error("seed message id does not match event message id: {seed:?} != {event:?}")]
    SeedMismatch { seed: String, event: String },
}

/// Reconstruct messages from a sequential agent event stream.
#[derive(Clone, Debug)]
pub struct MessageHydrator {
    seed_message_id: Option<String>,
    seed_checked: bool,
    message_selected: bool,
    streaming_index: Option<usize>,
    parts_by_message_id: HashMap<String, HashMap<String, usize>>,
    pub message: Message,
    pub messages: Vec<Message>,
    pub messages_by_id: BTreeMap<String, Message>,
    pub ended: bool,
    pub finish_reason: Option<String>,
    pub response_id: Option<String>,
    pub response_model: Option<String>,
}

impl Default for MessageHydrator {
    fn default() -> Self {
        Self::new(None)
    }
}

impl MessageHydrator {
    #[must_use]
    pub fn new(seed_message: Option<Message>) -> Self {
        let seed_message_id = seed_message.as_ref().map(|message| message.id.clone());
        let message_selected = seed_message.is_some();
        Self {
            seed_message_id,
            seed_checked: false,
            message_selected,
            streaming_index: None,
            parts_by_message_id: HashMap::new(),
            message: seed_message.unwrap_or_else(|| Message::new(MessageRole::Assistant, vec![])),
            messages: Vec::new(),
            messages_by_id: BTreeMap::new(),
            ended: false,
            finish_reason: None,
            response_id: None,
            response_model: None,
        }
    }

    pub fn feed(&mut self, mut event: Event) -> Result<Event, HydrationError> {
        match &event {
            Event::ToolCallResult(value) => {
                self.select_complete_message(value.message.clone());
                return Ok(event);
            }
            Event::Hook(value) => {
                self.select_complete_message(value.message.clone());
                return Ok(event);
            }
            Event::Retry(_) => {
                self.retry();
                return Ok(event);
            }
            _ => {}
        }

        let Some(meta) = event.model_event() else {
            return Ok(event);
        };
        let event_message_id = meta.message.id.clone();
        let replay = meta.replay;
        let usage = meta.usage.clone();

        if event_message_id != "<unset>" {
            if let Some(seed_id) = &self.seed_message_id
                && !self.seed_checked
            {
                self.seed_checked = true;
                if event_message_id != *seed_id {
                    return Err(HydrationError::SeedMismatch {
                        seed: seed_id.clone(),
                        event: event_message_id,
                    });
                }
            }
            if event_message_id != self.message.id {
                self.ended = false;
                if self.message_selected {
                    self.message = self
                        .messages_by_id
                        .get(&event_message_id)
                        .cloned()
                        .unwrap_or_else(|| {
                            Message::with_id(
                                event_message_id.clone(),
                                MessageRole::Assistant,
                                vec![],
                            )
                        });
                } else {
                    self.message.id.clone_from(&event_message_id);
                    self.message_selected = true;
                }
            }
        }
        self.ensure_current_message();

        if replay {
            self.ended = true;
            event
                .model_event_mut()
                .expect("model event checked above")
                .message = self.message.clone();
            self.sync_current_message();
            return Ok(event);
        }

        if let Some(usage) = usage {
            self.message.usage = Some(usage);
        }

        match &mut event {
            Event::TextStart(value) => {
                let part = TextPart {
                    id: value.block_id.clone(),
                    text: String::new(),
                    provider_metadata: value.event.provider_metadata.clone(),
                };
                self.push_indexed_part(value.block_id.clone(), part.into());
            }
            Event::TextDelta(value) => {
                let metadata = value.event.provider_metadata.clone();
                if let Some(Part::Text(part)) = self.part_mut(&value.block_id) {
                    part.text.push_str(&value.chunk);
                    if metadata.is_some() {
                        part.provider_metadata = metadata;
                    }
                }
            }
            Event::TextEnd(value) => {
                let metadata = value.event.provider_metadata.clone();
                if metadata.is_some()
                    && let Some(Part::Text(part)) = self.part_mut(&value.block_id)
                {
                    part.provider_metadata = metadata;
                }
            }
            Event::ReasoningStart(value) => {
                let part = ReasoningPart {
                    id: value.block_id.clone(),
                    text: String::new(),
                    provider_metadata: value.event.provider_metadata.clone(),
                };
                self.push_indexed_part(value.block_id.clone(), part.into());
            }
            Event::ReasoningDelta(value) => {
                let metadata = value.event.provider_metadata.clone();
                if let Some(Part::Reasoning(part)) = self.part_mut(&value.block_id) {
                    part.text.push_str(&value.chunk);
                    if metadata.is_some() {
                        part.provider_metadata = metadata;
                    }
                }
            }
            Event::ReasoningEnd(value) => {
                let metadata = value.event.provider_metadata.clone();
                if metadata.is_some()
                    && let Some(Part::Reasoning(part)) = self.part_mut(&value.block_id)
                {
                    part.provider_metadata = metadata;
                }
            }
            Event::ToolStart(value) => {
                let part = ToolCallPart {
                    id: value.tool_call_id.clone(),
                    tool_call_id: value.tool_call_id.clone(),
                    tool_name: value.tool_name.clone(),
                    tool_args: String::new(),
                    provider_metadata: value.event.provider_metadata.clone(),
                    cached_result: None,
                };
                self.push_indexed_part(value.tool_call_id.clone(), part.into());
            }
            Event::ToolDelta(value) => {
                let metadata = value.event.provider_metadata.clone();
                if let Some(Part::ToolCall(part)) = self.part_mut(&value.tool_call_id) {
                    part.tool_args.push_str(&value.chunk);
                    if metadata.is_some() {
                        part.provider_metadata = metadata;
                    }
                }
            }
            Event::ToolEnd(value) => {
                let metadata = value.event.provider_metadata.clone();
                if let Some(Part::ToolCall(part)) = self.part_mut(&value.tool_call_id) {
                    if metadata.is_some() {
                        part.provider_metadata = metadata;
                    }
                    value.tool_call = part.clone();
                }
            }
            Event::BuiltinToolStart(value) => {
                let part = BuiltinToolCallPart {
                    id: value.tool_call_id.clone(),
                    tool_call_id: value.tool_call_id.clone(),
                    tool_name: value.tool_name.clone(),
                    tool_args: String::new(),
                    provider_metadata: value.event.provider_metadata.clone(),
                };
                self.push_indexed_part(value.tool_call_id.clone(), Part::BuiltinToolCall(part));
            }
            Event::BuiltinToolDelta(value) => {
                let metadata = value.event.provider_metadata.clone();
                if let Some(Part::BuiltinToolCall(part)) = self.part_mut(&value.tool_call_id) {
                    part.tool_args.push_str(&value.chunk);
                    if metadata.is_some() {
                        part.provider_metadata = metadata;
                    }
                }
            }
            Event::BuiltinToolEnd(value) => {
                let metadata = value.event.provider_metadata.clone();
                if let Some(Part::BuiltinToolCall(part)) = self.part_mut(&value.tool_call_id) {
                    if metadata.is_some() {
                        part.provider_metadata = metadata;
                    }
                    value.tool_call = part.clone();
                }
            }
            Event::BuiltinToolResult(value) => {
                if value.event.provider_metadata.is_some() {
                    value.result.provider_metadata = value.event.provider_metadata.clone();
                }
                self.message
                    .parts
                    .push(Part::BuiltinToolReturn(value.result.clone()));
            }
            Event::File(value) => {
                let id = if value.block_id.is_empty() {
                    super::messages::generate_id("part")
                } else {
                    value.block_id.clone()
                };
                let part = FilePart {
                    id: id.clone(),
                    data: value.data.clone(),
                    media_type: value.media_type.clone(),
                    filename: value.filename.clone(),
                    provider_metadata: value.event.provider_metadata.clone(),
                };
                self.push_indexed_part(id, part.into());
            }
            Event::StreamEnd(value) => {
                self.ended = true;
                self.finish_reason.clone_from(&value.finish_reason);
                self.response_id.clone_from(&value.response_id);
                self.response_model.clone_from(&value.response_model);
                if value.event.provider_metadata.is_some() {
                    self.message
                        .provider_metadata
                        .clone_from(&value.event.provider_metadata);
                }
            }
            Event::StreamStart(_)
            | Event::ToolCallResult(_)
            | Event::Hook(_)
            | Event::PartialToolCallResult(_)
            | Event::RunBlocked(_)
            | Event::Retry(_) => {}
        }

        event
            .model_event_mut()
            .expect("model event checked above")
            .message = self.message.clone();
        self.sync_current_message();
        Ok(event)
    }

    fn ensure_current_message(&mut self) {
        if !self.messages_by_id.contains_key(&self.message.id) {
            self.messages.push(self.message.clone());
            self.messages_by_id
                .insert(self.message.id.clone(), self.message.clone());
            self.streaming_index = Some(self.messages.len() - 1);
        } else if self
            .streaming_index
            .is_none_or(|index| self.messages[index].id != self.message.id)
        {
            self.streaming_index = self
                .messages
                .iter()
                .position(|message| message.id == self.message.id);
        }
        self.parts_by_message_id
            .entry(self.message.id.clone())
            .or_default();
    }

    fn sync_current_message(&mut self) {
        if let Some(index) = self
            .messages
            .iter()
            .position(|message| message.id == self.message.id)
        {
            self.messages[index] = self.message.clone();
            self.streaming_index = Some(index);
        }
        self.messages_by_id
            .insert(self.message.id.clone(), self.message.clone());
    }

    fn select_complete_message(&mut self, message: Message) {
        self.message = message;
        if let Some(index) = self
            .messages
            .iter()
            .position(|existing| existing.id == self.message.id)
        {
            self.messages[index] = self.message.clone();
        } else {
            self.messages.push(self.message.clone());
        }
        self.messages_by_id
            .insert(self.message.id.clone(), self.message.clone());
    }

    fn push_indexed_part(&mut self, id: String, part: Part) {
        let index = self.message.parts.len();
        self.message.parts.push(part);
        self.parts_by_message_id
            .entry(self.message.id.clone())
            .or_default()
            .insert(id, index);
    }

    fn part_mut(&mut self, id: &str) -> Option<&mut Part> {
        let index = *self.parts_by_message_id.get(&self.message.id)?.get(id)?;
        self.message.parts.get_mut(index)
    }

    fn retry(&mut self) {
        let Some(streaming_index) = self.streaming_index else {
            return;
        };
        let stale: Vec<_> = self.messages.drain(streaming_index..).collect();
        let Some(first) = stale.first() else {
            return;
        };
        for message in &stale {
            self.messages_by_id.remove(&message.id);
            self.parts_by_message_id.remove(&message.id);
        }
        self.streaming_index = None;
        self.message = Message::with_id(first.id.clone(), MessageRole::Assistant, vec![]);
        self.ended = false;
        self.finish_reason = None;
        self.response_id = None;
        self.response_model = None;
    }
}

/// Synthesize model events for replaying a completed assistant message.
#[must_use]
pub fn replay_message_events(message: &Message) -> Vec<Event> {
    let stub = || {
        ModelEventMeta::with_message(Message::with_id(
            message.id.clone(),
            MessageRole::Assistant,
            vec![],
        ))
    };
    let mut events = vec![Event::StreamStart(StreamStart { event: stub() })];
    for part in &message.parts {
        match part {
            Part::Text(part) => {
                events.push(Event::TextStart(TextStart {
                    event: stub(),
                    block_id: part.id.clone(),
                }));
                if !part.text.is_empty() {
                    events.push(Event::TextDelta(TextDelta {
                        event: stub(),
                        chunk: part.text.clone(),
                        block_id: part.id.clone(),
                    }));
                }
                let mut event = stub();
                event.provider_metadata.clone_from(&part.provider_metadata);
                events.push(Event::TextEnd(TextEnd {
                    event,
                    block_id: part.id.clone(),
                }));
            }
            Part::Reasoning(part) => {
                events.push(Event::ReasoningStart(ReasoningStart {
                    event: stub(),
                    block_id: part.id.clone(),
                }));
                if !part.text.is_empty() {
                    events.push(Event::ReasoningDelta(ReasoningDelta {
                        event: stub(),
                        chunk: part.text.clone(),
                        block_id: part.id.clone(),
                    }));
                }
                let mut event = stub();
                event.provider_metadata.clone_from(&part.provider_metadata);
                events.push(Event::ReasoningEnd(ReasoningEnd {
                    event,
                    block_id: part.id.clone(),
                }));
            }
            Part::ToolCall(part) => {
                events.push(Event::ToolStart(ToolStart {
                    event: stub(),
                    tool_call_id: part.tool_call_id.clone(),
                    tool_name: part.tool_name.clone(),
                }));
                if !part.tool_args.is_empty() {
                    events.push(Event::ToolDelta(ToolDelta {
                        event: stub(),
                        chunk: part.tool_args.clone(),
                        tool_call_id: part.tool_call_id.clone(),
                    }));
                }
                let mut event = stub();
                event.provider_metadata.clone_from(&part.provider_metadata);
                events.push(Event::ToolEnd(ToolEnd {
                    event,
                    tool_call: part.clone(),
                    tool_call_id: part.tool_call_id.clone(),
                }));
            }
            Part::BuiltinToolCall(part) => {
                events.push(Event::BuiltinToolStart(BuiltinToolStart {
                    event: stub(),
                    tool_call_id: part.tool_call_id.clone(),
                    tool_name: part.tool_name.clone(),
                }));
                if !part.tool_args.is_empty() {
                    events.push(Event::BuiltinToolDelta(BuiltinToolDelta {
                        event: stub(),
                        chunk: part.tool_args.clone(),
                        tool_call_id: part.tool_call_id.clone(),
                    }));
                }
                let mut event = stub();
                event.provider_metadata.clone_from(&part.provider_metadata);
                events.push(Event::BuiltinToolEnd(BuiltinToolEnd {
                    event,
                    tool_call: part.clone(),
                    tool_call_id: part.tool_call_id.clone(),
                }));
            }
            Part::BuiltinToolReturn(part) => {
                events.push(Event::BuiltinToolResult(BuiltinToolResult {
                    event: stub(),
                    result: part.clone(),
                    tool_call_id: part.tool_call_id.clone(),
                }));
            }
            Part::File(part) => {
                let mut event = stub();
                event.provider_metadata.clone_from(&part.provider_metadata);
                events.push(Event::File(FileEvent {
                    event,
                    block_id: part.id.clone(),
                    media_type: part.media_type.clone(),
                    data: part.data.clone(),
                    filename: part.filename.clone(),
                }));
            }
            Part::ToolResult(_) | Part::Hook(_) => {}
        }
    }
    let mut event = stub();
    event
        .provider_metadata
        .clone_from(&message.provider_metadata);
    events.push(Event::StreamEnd(StreamEnd {
        event,
        ..StreamEnd::default()
    }));
    events
}

/// Fold an agent event stream into blocked-on-hooks state.
#[derive(Clone, Debug, Default)]
pub struct RunStateTracker {
    deferred: Vec<HookPart>,
    in_flight: BTreeSet<String>,
    last_scheduled: BTreeSet<String>,
    completed_while_streaming: BTreeSet<String>,
    streaming: u32,
    stream_ended: bool,
    blocked: bool,
}

impl RunStateTracker {
    #[must_use]
    pub const fn blocked(&self) -> bool {
        self.blocked
    }

    #[must_use]
    pub fn deferred_hooks(&self) -> Vec<&HookPart> {
        self.deferred.iter().collect()
    }

    pub fn feed(&mut self, event: &Event) -> Option<RunBlocked> {
        match event {
            Event::StreamStart(_) => {
                self.streaming += 1;
                self.stream_ended = false;
                self.last_scheduled.clear();
                self.completed_while_streaming.clear();
            }
            Event::StreamEnd(value) => {
                self.streaming = self.streaming.saturating_sub(1);
                self.stream_ended = true;
                self.last_scheduled = value
                    .event
                    .message
                    .tool_calls()
                    .into_iter()
                    .map(|call| call.tool_call_id.clone())
                    .collect();
                // Tools start at ToolEnd and can settle before StreamEnd.
                // Do not reintroduce those completed calls as pending work.
                self.in_flight.extend(
                    self.last_scheduled
                        .difference(&self.completed_while_streaming)
                        .cloned(),
                );
            }
            Event::Retry(_) => {
                self.streaming = self.streaming.saturating_sub(1);
                self.stream_ended = false;
                self.in_flight = self
                    .in_flight
                    .difference(&self.last_scheduled)
                    .cloned()
                    .collect();
                self.last_scheduled.clear();
                self.completed_while_streaming.clear();
            }
            Event::ToolCallResult(value) => {
                for result in &value.results {
                    self.in_flight.remove(&result.tool_call_id);
                    if self.streaming > 0 {
                        self.completed_while_streaming
                            .insert(result.tool_call_id.clone());
                    }
                }
            }
            Event::Hook(value) => match value.hook.status {
                HookStatus::Pending => {
                    if let Some(existing) = self
                        .deferred
                        .iter_mut()
                        .find(|hook| hook.hook_id == value.hook.hook_id)
                    {
                        *existing = value.hook.clone();
                    } else {
                        self.deferred.push(value.hook.clone());
                    }
                }
                HookStatus::Resolved | HookStatus::Cancelled => {
                    self.deferred
                        .retain(|hook| hook.hook_id != value.hook.hook_id);
                }
            },
            _ => return None,
        }

        let attributed: BTreeSet<_> = self
            .deferred
            .iter()
            .filter_map(|hook| hook.tool_call_id.clone())
            .collect();
        let now = self.stream_ended
            && !self.deferred.is_empty()
            && self.streaming == 0
            && self.in_flight.is_subset(&attributed);
        if now == self.blocked {
            return None;
        }
        self.blocked = now;
        now.then(|| RunBlocked {
            replay: false,
            hooks: self.deferred.clone(),
        })
    }
}
