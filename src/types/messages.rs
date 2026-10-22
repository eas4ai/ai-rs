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

use std::{
    cell::RefCell,
    collections::{BTreeMap, hash_map::RandomState},
    fmt,
    future::Future,
    hash::BuildHasher,
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

use base64::{Engine as _, engine::general_purpose::STANDARD};
use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize, Serializer, de};
use serde_json::Value;
use thiserror::Error;

use super::{media, usage::Usage};

pub type Metadata = BTreeMap<String, Value>;

static NEXT_ID: AtomicU64 = AtomicU64::new(0);

tokio::task_local! {
    static ID_GENERATOR: RefCell<Box<dyn FnMut() -> u64 + Send>>;
}

/// Generate a short prefixed identifier.
#[must_use]
pub fn generate_id(prefix: &str) -> String {
    let raw = ID_GENERATOR
        .try_with(|generator| (generator.borrow_mut())())
        .unwrap_or_else(|_| {
            let count = NEXT_ID.fetch_add(1, Ordering::Relaxed);
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |duration| duration.as_nanos());
            RandomState::new().hash_one((nanos, count, std::process::id()))
        })
        & 0x0000_ffff_ffff_ffff;
    if prefix.is_empty() {
        format!("{raw:012x}")
    } else {
        format!("{prefix}_{raw:012x}")
    }
}

/// Scope deterministic ID generation to the supplied future.
///
/// Tokio task-local values do not propagate into spawned tasks. Wrap a spawned
/// future in its own `with_id_generator` call when it needs deterministic IDs.
pub async fn with_id_generator<F, G, T>(generator: G, future: F) -> T
where
    F: Future<Output = T>,
    G: FnMut() -> u64 + Send + 'static,
{
    ID_GENERATOR
        .scope(RefCell::new(Box::new(generator)), future)
        .await
}

fn part_id() -> String {
    generate_id("part")
}

fn message_id() -> String {
    generate_id("msg")
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct TextPart {
    #[serde(default = "part_id")]
    pub id: String,
    pub text: String,
    #[serde(default)]
    pub provider_metadata: Option<Metadata>,
}

impl TextPart {
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            id: part_id(),
            text: text.into(),
            provider_metadata: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum FileData {
    Text(String),
    Bytes(Vec<u8>),
}

impl FileData {
    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(value) => Some(value),
            Self::Bytes(_) => None,
        }
    }

    #[must_use]
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            Self::Bytes(value) => Some(value),
            Self::Text(_) => None,
        }
    }
}

impl From<String> for FileData {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<&str> for FileData {
    fn from(value: &str) -> Self {
        Self::Text(value.to_owned())
    }
}

impl From<Vec<u8>> for FileData {
    fn from(value: Vec<u8>) -> Self {
        Self::Bytes(value)
    }
}

impl From<&[u8]> for FileData {
    fn from(value: &[u8]) -> Self {
        Self::Bytes(value.to_vec())
    }
}

impl Serialize for FileData {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Text(value) => serializer.serialize_str(value),
            Self::Bytes(value) => serializer.serialize_str(&STANDARD.encode(value)),
        }
    }
}

impl<'de> Deserialize<'de> for FileData {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        String::deserialize(deserializer).map(Self::Text)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct FilePart {
    #[serde(default = "part_id")]
    pub id: String,
    #[schemars(with = "String")]
    pub data: FileData,
    pub media_type: String,
    #[serde(default)]
    pub filename: Option<String>,
    #[serde(default)]
    pub provider_metadata: Option<Metadata>,
}

impl FilePart {
    pub fn from_url(
        url: impl Into<String>,
        media_type: Option<&str>,
    ) -> Result<Self, media::MediaError> {
        let url = url.into();
        let media_type = match media_type {
            Some(media_type) => media_type.to_owned(),
            None => media::infer_media_type(&url)?,
        };
        Ok(Self {
            id: part_id(),
            data: FileData::Text(url),
            media_type,
            filename: None,
            provider_metadata: None,
        })
    }

    pub fn from_bytes(
        data: impl Into<Vec<u8>>,
        media_type: Option<&str>,
        filename: Option<String>,
    ) -> Result<Self, media::MediaError> {
        let data = data.into();
        let media_type = media_type.map(str::to_owned).or_else(|| {
            media::detect_image_media_type(data.as_slice())
                .or_else(|| media::detect_audio_media_type(data.as_slice()))
                .or_else(|| media::detect_video_media_type(data.as_slice()))
                .map(str::to_owned)
        });
        let Some(media_type) = media_type else {
            return Err(media::MediaError::CannotDetect);
        };
        Ok(Self {
            id: part_id(),
            data: FileData::Bytes(data),
            media_type,
            filename,
            provider_metadata: None,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ContentPart {
    Text(TextPart),
    File(FilePart),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ContentOutput {
    #[serde(rename = "type")]
    #[schemars(rename = "type")]
    pub output_type: ContentOutputType,
    pub value: Vec<ContentPart>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ContentOutputType {
    #[default]
    Content,
}

impl ContentOutput {
    #[must_use]
    pub fn new(value: Vec<ContentPart>) -> Self {
        Self {
            output_type: ContentOutputType::Content,
            value,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct MessageBundle {
    #[serde(rename = "type")]
    #[schemars(rename = "type")]
    pub bundle_type: MessageBundleType,
    pub messages: Vec<Message>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MessageBundleType {
    #[default]
    Messages,
}

impl MessageBundle {
    #[must_use]
    pub fn new(messages: Vec<Message>) -> Self {
        Self {
            bundle_type: MessageBundleType::Messages,
            messages,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
#[schemars(untagged)]
pub enum SpecialToolResult {
    Content(ContentOutput),
    Messages(MessageBundle),
}

#[derive(Clone, Debug, PartialEq, JsonSchema)]
#[schemars(untagged)]
pub enum ToolResultValue {
    Json(Value),
    Special(SpecialToolResult),
}

impl Default for ToolResultValue {
    fn default() -> Self {
        Self::Json(Value::Null)
    }
}

impl ToolResultValue {
    #[must_use]
    pub const fn kind(&self) -> ResultKind {
        match self {
            Self::Json(_) => ResultKind::Json,
            Self::Special(_) => ResultKind::Special,
        }
    }

    #[must_use]
    pub fn as_json(&self) -> Option<&Value> {
        match self {
            Self::Json(value) => Some(value),
            Self::Special(_) => None,
        }
    }
}

impl Serialize for ToolResultValue {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Json(value) => value.serialize(serializer),
            Self::Special(value) => value.serialize(serializer),
        }
    }
}

impl From<Value> for ToolResultValue {
    fn from(value: Value) -> Self {
        Self::Json(value)
    }
}

impl From<ContentOutput> for ToolResultValue {
    fn from(value: ContentOutput) -> Self {
        Self::Special(SpecialToolResult::Content(value))
    }
}

impl From<MessageBundle> for ToolResultValue {
    fn from(value: MessageBundle) -> Self {
        Self::Special(SpecialToolResult::Messages(value))
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ResultKind {
    Error,
    #[default]
    Json,
    Special,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ModelInputKind {
    #[default]
    Json,
    Special,
}

#[derive(Clone, Debug, PartialEq, Serialize, JsonSchema)]
pub struct ToolResultPart {
    #[serde(default = "part_id")]
    pub id: String,
    pub tool_call_id: String,
    pub tool_name: String,
    #[serde(default)]
    pub is_hook_deferred: bool,
    #[serde(default)]
    pub provider_metadata: Option<Metadata>,
    #[serde(default)]
    pub result: ToolResultValue,
    #[serde(default)]
    pub result_kind: ResultKind,
    /// `None` means unset. `Some(Json(Null))` is an explicit JSON null.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_input: Option<ToolResultValue>,
    #[serde(default, skip_serializing_if = "is_json_model_input")]
    pub model_input_kind: ModelInputKind,
}

fn is_json_model_input(kind: &ModelInputKind) -> bool {
    *kind == ModelInputKind::Json
}

impl ToolResultPart {
    #[must_use]
    pub fn new(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        result: impl Into<ToolResultValue>,
    ) -> Self {
        let result = result.into();
        let result_kind = result.kind();
        Self {
            id: part_id(),
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            is_hook_deferred: false,
            provider_metadata: None,
            result,
            result_kind,
            model_input: None,
            model_input_kind: ModelInputKind::Json,
        }
    }

    #[must_use]
    pub fn error(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        result: impl Into<ToolResultValue>,
    ) -> Self {
        let mut part = Self::new(tool_call_id, tool_name, result);
        part.result_kind = ResultKind::Error;
        part
    }

    #[must_use]
    pub const fn is_error(&self) -> bool {
        matches!(self.result_kind, ResultKind::Error)
    }

    #[must_use]
    pub fn get_model_input(&self) -> &ToolResultValue {
        self.model_input.as_ref().unwrap_or(&self.result)
    }

    pub fn set_model_input(&mut self, value: impl Into<ToolResultValue>) {
        let value = value.into();
        self.model_input_kind = match value.kind() {
            ResultKind::Special => ModelInputKind::Special,
            ResultKind::Error | ResultKind::Json => ModelInputKind::Json,
        };
        self.model_input = Some(value);
    }

    #[must_use]
    pub const fn has_model_input(&self) -> bool {
        self.model_input.is_some()
    }
}

impl<'de> Deserialize<'de> for ToolResultPart {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Wire {
            #[serde(default = "part_id")]
            id: String,
            tool_call_id: String,
            tool_name: String,
            #[serde(default)]
            is_hook_deferred: bool,
            #[serde(default)]
            provider_metadata: Option<Metadata>,
            #[serde(default)]
            result: Value,
            #[serde(default)]
            result_kind: ResultKind,
            #[serde(default, deserialize_with = "deserialize_present_value")]
            model_input: Option<Value>,
            #[serde(default)]
            model_input_kind: ModelInputKind,
        }

        let wire = Wire::deserialize(deserializer)?;
        let result = decode_tool_value(wire.result, wire.result_kind == ResultKind::Special)
            .map_err(de::Error::custom)?;
        let model_input = wire
            .model_input
            .map(|value| decode_tool_value(value, wire.model_input_kind == ModelInputKind::Special))
            .transpose()
            .map_err(de::Error::custom)?;
        Ok(Self {
            id: wire.id,
            tool_call_id: wire.tool_call_id,
            tool_name: wire.tool_name,
            is_hook_deferred: wire.is_hook_deferred,
            provider_metadata: wire.provider_metadata,
            result,
            result_kind: wire.result_kind,
            model_input,
            model_input_kind: wire.model_input_kind,
        })
    }
}

fn deserialize_present_value<'de, D>(deserializer: D) -> Result<Option<Value>, D::Error>
where
    D: Deserializer<'de>,
{
    Value::deserialize(deserializer).map(Some)
}

fn decode_tool_value(value: Value, special: bool) -> Result<ToolResultValue, serde_json::Error> {
    if special {
        serde_json::from_value(value).map(ToolResultValue::Special)
    } else {
        Ok(ToolResultValue::Json(value))
    }
}

/// A message content part.
///
/// Serialize this enum, rather than an individual payload struct, when producing
/// the Python-compatible wire format. The enum supplies the canonical `kind`
/// discriminator.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ToolCallPart {
    #[serde(default = "part_id")]
    pub id: String,
    pub tool_call_id: String,
    pub tool_name: String,
    pub tool_args: String,
    #[serde(default)]
    pub provider_metadata: Option<Metadata>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[serde(
        serialize_with = "serialize_optional_tool_result",
        deserialize_with = "deserialize_optional_tool_result"
    )]
    #[schemars(with = "Option<Part>")]
    pub cached_result: Option<ToolResultPart>,
}

impl ToolCallPart {
    #[must_use]
    pub fn new(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        tool_args: impl Into<String>,
    ) -> Self {
        Self {
            id: part_id(),
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            tool_args: tool_args.into(),
            provider_metadata: None,
            cached_result: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct BuiltinToolCallPart {
    #[serde(default = "part_id")]
    pub id: String,
    pub tool_call_id: String,
    pub tool_name: String,
    #[serde(default)]
    pub tool_args: String,
    #[serde(default)]
    pub provider_metadata: Option<Metadata>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct BuiltinToolReturnPart {
    #[serde(default = "part_id")]
    pub id: String,
    pub tool_call_id: String,
    pub tool_name: String,
    #[serde(default)]
    pub result: Value,
    #[serde(default)]
    pub is_error: bool,
    #[serde(default)]
    pub provider_metadata: Option<Metadata>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ReasoningPart {
    #[serde(default = "part_id")]
    pub id: String,
    pub text: String,
    #[serde(default)]
    pub provider_metadata: Option<Metadata>,
}

impl ReasoningPart {
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            id: part_id(),
            text: text.into(),
            provider_metadata: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum HookStatus {
    Pending,
    Resolved,
    Cancelled,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct HookPart {
    #[serde(default = "part_id")]
    pub id: String,
    pub hook_id: String,
    pub hook_type: String,
    pub status: HookStatus,
    #[serde(default)]
    pub metadata: Metadata,
    #[serde(default)]
    pub resolution: Option<Value>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Part {
    Text(TextPart),
    ToolCall(ToolCallPart),
    ToolResult(ToolResultPart),
    BuiltinToolCall(BuiltinToolCallPart),
    BuiltinToolReturn(BuiltinToolReturnPart),
    Reasoning(ReasoningPart),
    Hook(HookPart),
    File(FilePart),
}

fn serialize_optional_tool_result<S>(
    value: &Option<ToolResultPart>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    value
        .as_ref()
        .map(|part| Part::ToolResult(part.clone()))
        .serialize(serializer)
}

fn deserialize_optional_tool_result<'de, D>(
    deserializer: D,
) -> Result<Option<ToolResultPart>, D::Error>
where
    D: Deserializer<'de>,
{
    match Option::<Part>::deserialize(deserializer)? {
        None => Ok(None),
        Some(Part::ToolResult(part)) => Ok(Some(part)),
        Some(_) => Err(de::Error::custom(
            "cached_result must have kind=tool_result",
        )),
    }
}

impl Part {
    #[must_use]
    pub fn id(&self) -> &str {
        match self {
            Self::Text(part) => &part.id,
            Self::ToolCall(part) => &part.id,
            Self::ToolResult(part) => &part.id,
            Self::BuiltinToolCall(part) => &part.id,
            Self::BuiltinToolReturn(part) => &part.id,
            Self::Reasoning(part) => &part.id,
            Self::Hook(part) => &part.id,
            Self::File(part) => &part.id,
        }
    }
}

impl From<TextPart> for Part {
    fn from(value: TextPart) -> Self {
        Self::Text(value)
    }
}

impl From<FilePart> for Part {
    fn from(value: FilePart) -> Self {
        Self::File(value)
    }
}

impl From<ToolCallPart> for Part {
    fn from(value: ToolCallPart) -> Self {
        Self::ToolCall(value)
    }
}

impl From<ToolResultPart> for Part {
    fn from(value: ToolResultPart) -> Self {
        Self::ToolResult(value)
    }
}

impl From<ReasoningPart> for Part {
    fn from(value: ReasoningPart) -> Self {
        Self::Reasoning(value)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MessageRole {
    User,
    #[default]
    Assistant,
    System,
    Tool,
    Internal,
}

impl fmt::Display for MessageRole {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::System => "system",
            Self::Tool => "tool",
            Self::Internal => "internal",
        })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Message {
    pub role: MessageRole,
    pub parts: Vec<Part>,
    #[serde(default = "message_id")]
    pub id: String,
    #[serde(default)]
    pub turn_id: Option<String>,
    #[serde(default)]
    pub usage: Option<Usage>,
    #[serde(default)]
    pub provider_metadata: Option<Metadata>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub replay: bool,
}

fn is_false(value: &bool) -> bool {
    !*value
}

impl Message {
    #[must_use]
    pub fn new(role: MessageRole, parts: Vec<Part>) -> Self {
        Self {
            role,
            parts,
            id: message_id(),
            turn_id: None,
            usage: None,
            provider_metadata: None,
            replay: false,
        }
    }

    #[must_use]
    pub fn with_id(id: impl Into<String>, role: MessageRole, parts: Vec<Part>) -> Self {
        let mut message = Self::new(role, parts);
        message.id = id.into();
        message
    }

    #[must_use]
    pub fn text(&self) -> String {
        self.parts
            .iter()
            .filter_map(|part| match part {
                Part::Text(part) => Some(part.text.as_str()),
                _ => None,
            })
            .collect()
    }

    #[must_use]
    pub fn reasoning(&self) -> String {
        self.parts
            .iter()
            .filter_map(|part| match part {
                Part::Reasoning(part) => Some(part.text.as_str()),
                _ => None,
            })
            .collect()
    }

    #[must_use]
    pub fn tool_calls(&self) -> Vec<&ToolCallPart> {
        self.parts
            .iter()
            .filter_map(|part| match part {
                Part::ToolCall(part) => Some(part),
                _ => None,
            })
            .collect()
    }

    #[must_use]
    pub fn tool_results(&self) -> Vec<&ToolResultPart> {
        self.parts
            .iter()
            .filter_map(|part| match part {
                Part::ToolResult(part) => Some(part),
                _ => None,
            })
            .collect()
    }

    #[must_use]
    pub fn builtin_tool_calls(&self) -> Vec<&BuiltinToolCallPart> {
        self.parts
            .iter()
            .filter_map(|part| match part {
                Part::BuiltinToolCall(part) => Some(part),
                _ => None,
            })
            .collect()
    }

    #[must_use]
    pub fn builtin_tool_returns(&self) -> Vec<&BuiltinToolReturnPart> {
        self.parts
            .iter()
            .filter_map(|part| match part {
                Part::BuiltinToolReturn(part) => Some(part),
                _ => None,
            })
            .collect()
    }

    #[must_use]
    pub fn files(&self) -> Vec<&FilePart> {
        self.parts
            .iter()
            .filter_map(|part| match part {
                Part::File(part) => Some(part),
                _ => None,
            })
            .collect()
    }

    #[must_use]
    pub fn images(&self) -> Vec<&FilePart> {
        self.files()
            .into_iter()
            .filter(|part| part.media_type.starts_with("image/"))
            .collect()
    }

    #[must_use]
    pub fn videos(&self) -> Vec<&FilePart> {
        self.files()
            .into_iter()
            .filter(|part| part.media_type.starts_with("video/"))
            .collect()
    }

    #[must_use]
    pub fn audio(&self) -> Vec<&FilePart> {
        self.files()
            .into_iter()
            .filter(|part| part.media_type.starts_with("audio/"))
            .collect()
    }

    pub fn get_output(&self) -> Result<String, MessageOutputError> {
        if self.role != MessageRole::Assistant || !self.tool_calls().is_empty() {
            return Err(MessageOutputError {
                role: self.role,
                tool_call_count: self.tool_calls().len(),
            });
        }
        Ok(self.text())
    }

    pub fn deserialize_output<T>(&self) -> Result<T, DeserializeOutputError>
    where
        T: serde::de::DeserializeOwned,
    {
        Ok(serde_json::from_str(&self.get_output()?)?)
    }
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error(
    "get_output requires a final assistant message; got role={role} with {tool_call_count} tool call(s)"
)]
pub struct MessageOutputError {
    pub role: MessageRole,
    pub tool_call_count: usize,
}

#[derive(Debug, Error)]
pub enum DeserializeOutputError {
    #[error(transparent)]
    NotFinal(#[from] MessageOutputError),
    #[error(transparent)]
    InvalidJson(#[from] serde_json::Error),
}

impl From<HookPart> for Part {
    fn from(value: HookPart) -> Self {
        Self::Hook(value)
    }
}
