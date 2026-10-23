//! Composable message construction helpers.

use serde_json::Value;
use thiserror::Error;

use super::messages::{
    ContentOutput, ContentPart, FileData, FilePart, HookPart, Message, MessageRole, Metadata, Part,
    ReasoningPart, ResultKind, TextPart, ToolCallPart, ToolResultPart, ToolResultValue,
};

#[derive(Clone, Debug, PartialEq)]
pub enum PartLike {
    Text(String),
    Part(Box<Part>),
}

impl From<&str> for PartLike {
    fn from(value: &str) -> Self {
        Self::Text(value.to_owned())
    }
}

impl From<String> for PartLike {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<Part> for PartLike {
    fn from(value: Part) -> Self {
        Self::Part(Box::new(value))
    }
}

impl From<TextPart> for PartLike {
    fn from(value: TextPart) -> Self {
        Self::Part(Box::new(value.into()))
    }
}

impl From<FilePart> for PartLike {
    fn from(value: FilePart) -> Self {
        Self::Part(Box::new(value.into()))
    }
}

impl From<ReasoningPart> for PartLike {
    fn from(value: ReasoningPart) -> Self {
        Self::Part(Box::new(value.into()))
    }
}

impl From<ToolResultPart> for PartLike {
    fn from(value: ToolResultPart) -> Self {
        Self::Part(Box::new(value.into()))
    }
}

impl From<ToolCallPart> for PartLike {
    fn from(value: ToolCallPart) -> Self {
        Self::Part(Box::new(value.into()))
    }
}
impl From<HookPart> for PartLike {
    fn from(value: HookPart) -> Self {
        Self::Part(Box::new(value.into()))
    }
}

pub trait IntoMessageParts {
    fn into_message_parts(self) -> Vec<Part>;
}

impl IntoMessageParts for &str {
    fn into_message_parts(self) -> Vec<Part> {
        vec![TextPart::new(self).into()]
    }
}

impl IntoMessageParts for String {
    fn into_message_parts(self) -> Vec<Part> {
        vec![TextPart::new(self).into()]
    }
}

impl IntoMessageParts for Part {
    fn into_message_parts(self) -> Vec<Part> {
        vec![self]
    }
}

impl IntoMessageParts for PartLike {
    fn into_message_parts(self) -> Vec<Part> {
        vec![coerce_part(self)]
    }
}

impl<T> IntoMessageParts for Vec<T>
where
    T: Into<PartLike>,
{
    fn into_message_parts(self) -> Vec<Part> {
        self.into_iter().map(Into::into).map(coerce_part).collect()
    }
}

impl<T, const N: usize> IntoMessageParts for [T; N]
where
    T: Into<PartLike>,
{
    fn into_message_parts(self) -> Vec<Part> {
        self.into_iter().map(Into::into).map(coerce_part).collect()
    }
}

impl IntoMessageParts for TextPart {
    fn into_message_parts(self) -> Vec<Part> {
        vec![self.into()]
    }
}

impl IntoMessageParts for FilePart {
    fn into_message_parts(self) -> Vec<Part> {
        vec![self.into()]
    }
}

impl IntoMessageParts for ReasoningPart {
    fn into_message_parts(self) -> Vec<Part> {
        vec![self.into()]
    }
}

impl IntoMessageParts for ToolCallPart {
    fn into_message_parts(self) -> Vec<Part> {
        vec![self.into()]
    }
}

impl IntoMessageParts for ToolResultPart {
    fn into_message_parts(self) -> Vec<Part> {
        vec![self.into()]
    }
}

impl IntoMessageParts for HookPart {
    fn into_message_parts(self) -> Vec<Part> {
        vec![self.into()]
    }
}

fn coerce_part(value: PartLike) -> Part {
    match value {
        PartLike::Text(text) => TextPart::new(text).into(),
        PartLike::Part(part) => *part,
    }
}

#[must_use]
pub fn message(role: MessageRole, content: impl IntoMessageParts) -> Message {
    Message::new(role, content.into_message_parts())
}

#[must_use]
pub fn system_message(content: impl IntoMessageParts) -> Message {
    message(MessageRole::System, content)
}

#[must_use]
pub fn user_message(content: impl IntoMessageParts) -> Message {
    message(MessageRole::User, content)
}

#[must_use]
pub fn assistant_message(content: impl IntoMessageParts) -> Message {
    message(MessageRole::Assistant, content)
}

#[derive(Clone, Debug, PartialEq)]
pub enum FileInput {
    Url(String),
    Bytes(Vec<u8>),
}

impl From<&str> for FileInput {
    fn from(value: &str) -> Self {
        Self::Url(value.to_owned())
    }
}

impl From<String> for FileInput {
    fn from(value: String) -> Self {
        Self::Url(value)
    }
}

impl From<&[u8]> for FileInput {
    fn from(value: &[u8]) -> Self {
        Self::Bytes(value.to_vec())
    }
}

impl<const N: usize> From<&[u8; N]> for FileInput {
    fn from(value: &[u8; N]) -> Self {
        Self::Bytes(value.to_vec())
    }
}

impl From<Vec<u8>> for FileInput {
    fn from(value: Vec<u8>) -> Self {
        Self::Bytes(value)
    }
}

pub fn file_part(
    data: impl Into<FileInput>,
    media_type: Option<&str>,
    filename: Option<String>,
) -> Result<FilePart, super::media::MediaError> {
    match data.into() {
        FileInput::Url(url) => FilePart::from_url(url, media_type),
        FileInput::Bytes(data) => FilePart::from_bytes(data, media_type, filename),
    }
}

#[must_use]
pub fn text_part(text: impl Into<String>, provider_metadata: Option<Metadata>) -> TextPart {
    let mut part = TextPart::new(text);
    part.provider_metadata = provider_metadata;
    part
}

#[derive(Clone, Debug, PartialEq)]
pub enum ContentLike {
    Text(String),
    Part(ContentPart),
}

impl From<&str> for ContentLike {
    fn from(value: &str) -> Self {
        Self::Text(value.to_owned())
    }
}

impl From<String> for ContentLike {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<TextPart> for ContentLike {
    fn from(value: TextPart) -> Self {
        Self::Part(ContentPart::Text(value))
    }
}

impl From<FilePart> for ContentLike {
    fn from(value: FilePart) -> Self {
        Self::Part(ContentPart::File(value))
    }
}

#[must_use]
pub fn content_output<T>(content: impl IntoIterator<Item = T>) -> ContentOutput
where
    T: Into<ContentLike>,
{
    ContentOutput::new(
        content
            .into_iter()
            .map(Into::into)
            .map(|item| match item {
                ContentLike::Text(text) => ContentPart::Text(TextPart::new(text)),
                ContentLike::Part(part) => part,
            })
            .collect(),
    )
}

#[must_use]
pub fn thinking(text: impl Into<String>, provider_metadata: Option<Metadata>) -> ReasoningPart {
    let mut part = ReasoningPart::new(text);
    part.provider_metadata = provider_metadata;
    part
}

#[derive(Clone, Debug, PartialEq)]
pub enum ToolMessageItem {
    Message(Message),
    Messages(Vec<Message>),
    Result(ToolResultPart),
}

impl From<Message> for ToolMessageItem {
    fn from(value: Message) -> Self {
        Self::Message(value)
    }
}

impl From<Vec<Message>> for ToolMessageItem {
    fn from(value: Vec<Message>) -> Self {
        Self::Messages(value)
    }
}
impl From<super::events::ToolCallResult> for ToolMessageItem {
    fn from(value: super::events::ToolCallResult) -> Self {
        Self::Message(value.message)
    }
}

impl From<ToolResultPart> for ToolMessageItem {
    fn from(value: ToolResultPart) -> Self {
        Self::Result(value)
    }
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum ToolMessageError {
    #[error("tool_message requires at least one tool message or result")]
    Empty,
    #[error("expected tool message, got role={0}")]
    WrongRole(MessageRole),
    #[error("tool_message only accepts tool messages containing ToolResultPart parts")]
    NonResultPart,
    #[error("tool_message cannot mix tool messages with ToolResultPart values")]
    MixedInput,
}

pub fn tool_message<T>(items: impl IntoIterator<Item = T>) -> Result<Message, ToolMessageError>
where
    T: Into<ToolMessageItem>,
{
    let items: Vec<_> = items.into_iter().map(Into::into).collect();
    if items.is_empty() {
        return Err(ToolMessageError::Empty);
    }
    let saw_messages = items.iter().any(|item| {
        matches!(
            item,
            ToolMessageItem::Message(_) | ToolMessageItem::Messages(_)
        )
    });
    let saw_results = items
        .iter()
        .any(|item| matches!(item, ToolMessageItem::Result(_)));
    if saw_messages && saw_results {
        return Err(ToolMessageError::MixedInput);
    }

    let mut parts = Vec::new();
    for item in items {
        let messages = match item {
            ToolMessageItem::Result(part) => {
                parts.push(part.into());
                continue;
            }
            ToolMessageItem::Message(message) => vec![message],
            ToolMessageItem::Messages(messages) => messages,
        };
        for message in messages {
            if message.role != MessageRole::Tool {
                return Err(ToolMessageError::WrongRole(message.role));
            }
            for part in message.parts {
                if !matches!(part, Part::ToolResult(_)) {
                    return Err(ToolMessageError::NonResultPart);
                }
                parts.push(part);
            }
        }
    }
    Ok(Message::new(MessageRole::Tool, parts))
}

#[must_use]
pub fn tool_result_message(
    tool_call_id: impl Into<String>,
    result: impl Into<ToolResultValue>,
    tool_name: impl Into<String>,
    is_error: bool,
) -> Message {
    Message::new(
        MessageRole::Tool,
        vec![tool_result_part(tool_call_id, result, tool_name, is_error).into()],
    )
}

#[must_use]
pub fn tool_result_part(
    tool_call_id: impl Into<String>,
    result: impl Into<ToolResultValue>,
    tool_name: impl Into<String>,
    is_error: bool,
) -> ToolResultPart {
    let mut part = ToolResultPart::new(tool_call_id, tool_name, result);
    if is_error {
        part.result_kind = ResultKind::Error;
    }
    part
}

impl From<FileData> for Value {
    fn from(value: FileData) -> Self {
        match value {
            FileData::Text(value) => Self::String(value),
            FileData::Bytes(value) => Self::String(base64::Engine::encode(
                &base64::engine::general_purpose::STANDARD,
                value,
            )),
        }
    }
}
