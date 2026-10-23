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

//! Provider operation payloads. These types keep the base contract independent of adapters.
use crate::types::{messages::FilePart, usage::Usage};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

pub type Metadata = BTreeMap<String, Value>;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Item<T> {
    pub value: T,
    #[serde(default)]
    pub usage: Option<Usage>,
    #[serde(default)]
    pub warnings: Vec<Warning>,
    #[serde(default)]
    pub metadata: Option<Metadata>,
    #[serde(default)]
    pub provider_metadata: Option<Metadata>,
}
impl<T> Item<T> {
    pub fn new(value: T) -> Self {
        Self {
            value,
            usage: None,
            warnings: Vec::new(),
            metadata: None,
            provider_metadata: None,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Warning {
    pub kind: String,
    pub message: Option<String>,
    pub feature: Option<String>,
    pub setting: Option<String>,
    pub details: Option<String>,
}

impl Default for Warning {
    fn default() -> Self {
        Self {
            kind: "other".into(),
            message: None,
            feature: None,
            setting: None,
            details: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum MediaInput {
    File(FilePart),
    Bytes(Vec<u8>),
    Url(String),
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct ImagePrompt {
    pub text: Option<String>,
    #[serde(default)]
    pub images: Vec<MediaInput>,
    pub mask: Option<MediaInput>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FrameImage {
    pub image: MediaInput,
    pub frame_type: FrameType,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum FrameType {
    FirstFrame,
    LastFrame,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct VideoPrompt {
    pub text: Option<String>,
    pub image: Option<MediaInput>,
    #[serde(default)]
    pub frame_images: Vec<FrameImage>,
    #[serde(default)]
    pub references: Vec<MediaInput>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AudioPrompt {
    pub text: String,
    pub instructions: Option<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum TranscribeInput {
    File(FilePart),
    Bytes(Vec<u8>),
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TranscriptionSegment {
    pub text: String,
    pub start_second: f64,
    pub end_second: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Transcription {
    pub text: String,
    #[serde(default)]
    pub segments: Vec<TranscriptionSegment>,
    pub language: Option<String>,
    pub duration_seconds: Option<f64>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RankedDocument {
    pub index: usize,
    pub score: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum Documents {
    Text(Vec<String>),
    Json(Vec<Metadata>),
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum EvaluationInput {
    Text(String),
    Object(Metadata),
    Array(Vec<Value>),
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
#[serde(try_from = "QuestionWire")]
pub enum Question {
    Choice {
        instructions: EvaluationInput,
        criteria: BTreeMap<String, Option<EvaluationInput>>,
    },
    Score {
        instructions: EvaluationInput,
        criteria: Vec<Option<EvaluationInput>>,
    },
    Noul {
        instructions: EvaluationInput,
        criteria: Option<NoulCriteria>,
    },
}
impl Question {
    pub fn validate(&self) -> Result<(), crate::errors::Error> {
        let message = match self {
            Self::Choice { criteria, .. } if criteria.is_empty() => {
                Some("choice criteria must not be empty")
            }
            Self::Score { criteria, .. } if criteria.len() < 2 => {
                Some("score criteria must contain at least two levels")
            }
            _ => None,
        };
        match message {
            Some(message) => Err(crate::errors::Error::Validation(message.into())),
            None => Ok(()),
        }
    }
}
#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum QuestionWire {
    Choice {
        instructions: EvaluationInput,
        criteria: BTreeMap<String, Option<EvaluationInput>>,
    },
    Score {
        instructions: EvaluationInput,
        criteria: Vec<Option<EvaluationInput>>,
    },
    Noul {
        instructions: EvaluationInput,
        criteria: Option<NoulCriteria>,
    },
}
impl TryFrom<QuestionWire> for Question {
    type Error = crate::errors::Error;
    fn try_from(value: QuestionWire) -> Result<Self, Self::Error> {
        let question = match value {
            QuestionWire::Choice {
                instructions,
                criteria,
            } => Self::Choice {
                instructions,
                criteria,
            },
            QuestionWire::Score {
                instructions,
                criteria,
            } => Self::Score {
                instructions,
                criteria,
            },
            QuestionWire::Noul {
                instructions,
                criteria,
            } => Self::Noul {
                instructions,
                criteria,
            },
        };
        question.validate()?;
        Ok(question)
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct NoulCriteria {
    pub r#true: Option<EvaluationInput>,
    pub r#false: Option<EvaluationInput>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ImageParams {
    pub n: u32,
    pub size: Option<String>,
    pub aspect_ratio: Option<String>,
    pub seed: Option<i64>,
    #[serde(default)]
    pub provider_options: Metadata,
}
impl Default for ImageParams {
    fn default() -> Self {
        Self {
            n: 1,
            size: None,
            aspect_ratio: None,
            seed: None,
            provider_options: Metadata::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct VideoParams {
    pub n: u32,
    pub aspect_ratio: Option<String>,
    pub resolution: Option<String>,
    pub duration: Option<u32>,
    pub fps: Option<u32>,
    pub seed: Option<i64>,
    pub generate_audio: Option<bool>,
    #[serde(default)]
    pub provider_options: Metadata,
}
impl Default for VideoParams {
    fn default() -> Self {
        Self {
            n: 1,
            aspect_ratio: None,
            resolution: None,
            duration: None,
            fps: None,
            seed: None,
            generate_audio: None,
            provider_options: Metadata::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct AudioParams {
    pub voice: Option<String>,
    pub output_format: Option<String>,
    pub speed: Option<f64>,
    pub language: Option<String>,
    #[serde(default)]
    pub provider_options: Metadata,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct EmbedParams {
    #[serde(default)]
    pub provider_options: Metadata,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct TranscribeParams {
    #[serde(default)]
    pub provider_options: Metadata,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct RerankParams {
    pub top_n: Option<usize>,
    #[serde(default)]
    pub provider_options: Metadata,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct EvaluationParams {
    #[serde(default)]
    pub provider_options: Metadata,
}

/// How many values or results one protocol call takes for a model, and
/// whether the front end may run several calls at once.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CallLimits {
    /// The most values (embeddings) or results (images, videos) per call;
    /// `None` puts everything in one call.
    pub max_per_call: Option<usize>,
    /// Whether calls for one front-end call may be in flight together.
    pub parallel: bool,
}

impl Default for CallLimits {
    fn default() -> Self {
        Self {
            max_per_call: None,
            parallel: true,
        }
    }
}

/// A started video generation: the provider and model that run it and the
/// protocol's operation value. It serializes to JSON and back, so an
/// application can check a job's status from another process.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct VideoJob {
    pub provider: String,
    pub model_id: String,
    pub operation: Value,
}

/// The status of a video job.
// A status is made once per check; boxing the item would only cost callers.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq)]
pub enum VideoStatus {
    Pending,
    /// The videos as file parts: bytes, or a URL that the protocol's
    /// `download_video` fetches.
    Completed(Item<Vec<FilePart>>),
    Failed {
        message: String,
    },
}

/// A file to upload through a provider's files operation.
#[derive(Clone, Debug, PartialEq)]
pub struct UploadFile {
    pub data: bytes::Bytes,
    pub media_type: String,
    pub filename: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct UploadParams {
    #[serde(default)]
    pub provider_options: Metadata,
}

/// An uploaded file. `provider_reference` maps the provider name to the id
/// or URI a later request names the file by; each other field is absent when
/// the provider does not report it.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct UploadedFile {
    pub provider_reference: BTreeMap<String, String>,
    pub media_type: Option<String>,
    pub filename: Option<String>,
    pub size_bytes: Option<u64>,
    pub created_at: Option<String>,
    pub expires_at: Option<String>,
}

/// One request of a batch.
#[derive(Clone, Debug)]
pub struct BatchRequest {
    pub id: String,
    pub model_id: String,
    pub kind: BatchRequestKind,
}

// A request kind is made once per batch request; boxing would only cost callers.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug)]
pub enum BatchRequestKind {
    /// A language model request: its messages and call options.
    Text {
        messages: Vec<crate::types::messages::Message>,
        options: crate::models::StreamOptions,
    },
    Image {
        prompt: ImagePrompt,
        params: ImageParams,
    },
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct BatchParams {
    pub webhook_url: Option<String>,
    #[serde(default)]
    pub provider_options: Metadata,
}

/// Where a batch is: `pending` until it ends, then `completed` or `failed`.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BatchState {
    Pending,
    Completed,
    Failed,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct BatchRequestCounts {
    pub total: u64,
    pub pending: u64,
    pub completed: u64,
    pub failed: u64,
}

impl BatchRequestCounts {
    /// The counts when they add up to the total, as
    /// `normalizeBatchRequestCounts` requires; `None` otherwise.
    #[must_use]
    pub fn checked(total: u64, pending: u64, completed: u64, failed: u64) -> Option<Self> {
        (pending.checked_add(completed)?.checked_add(failed)? == total).then_some(Self {
            total,
            pending,
            completed,
            failed,
        })
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct BatchError {
    pub message: String,
    pub error_type: Option<String>,
    pub code: Option<String>,
}

/// A batch's status.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct BatchStatus {
    pub batch_id: String,
    pub state: BatchState,
    /// The provider's own state name.
    pub raw_status: Option<String>,
    pub request_counts: Option<BatchRequestCounts>,
    pub error: Option<BatchError>,
    pub created_at: Option<String>,
    pub expires_at: Option<String>,
    #[serde(default)]
    pub provider_metadata: Option<Metadata>,
}

/// A warning about one request of a batch, or about the batch when
/// `request_id` is `None`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct BatchWarning {
    pub request_id: Option<String>,
    pub warning: Warning,
}

/// A started batch: its status and the warnings about its requests.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct BatchStarted {
    pub status: BatchStatus,
    pub warnings: Vec<BatchWarning>,
}

/// One page of batches and the cursor of the next page.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct BatchPage {
    pub batches: Vec<BatchStatus>,
    pub next_cursor: Option<String>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BatchResultKind {
    Text,
    Image,
}

/// What one batch request produced.
#[derive(Clone, Debug, PartialEq)]
pub enum BatchOutcome {
    /// A text request's message.
    Message(crate::types::messages::Message),
    /// An image request's images.
    Images(Item<Vec<FilePart>>),
    Failed {
        error: BatchError,
        provider_metadata: Option<Metadata>,
    },
    Cancelled {
        error: Option<BatchError>,
    },
    Expired {
        error: Option<BatchError>,
    },
}

/// The result of one batch request.
#[derive(Clone, Debug, PartialEq)]
pub struct BatchResult {
    pub id: String,
    pub kind: BatchResultKind,
    pub outcome: BatchOutcome,
}

/// A stream of batch results, in the provider's order.
pub type BatchResults =
    futures::stream::BoxStream<'static, Result<BatchResult, crate::errors::Error>>;

/// The format of raw audio: a media type such as `audio/pcm` and, for
/// formats that need one, a sample rate in Hz.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct AudioFormat {
    pub media_type: String,
    pub sample_rate: Option<u32>,
}

impl AudioFormat {
    /// Raw 16-bit little-endian PCM at this rate.
    #[must_use]
    pub fn pcm(sample_rate: u32) -> Self {
        Self {
            media_type: "audio/pcm".into(),
            sample_rate: Some(sample_rate),
        }
    }
}

/// Raw audio chunks for a streaming operation. The operation reads it as it
/// sends, and stops reading when its own stream is dropped.
pub type AudioChunks = futures::stream::BoxStream<'static, Vec<u8>>;

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct StreamTranscribeParams {
    /// Include each provider message as a `Raw` part.
    #[serde(default)]
    pub include_raw: bool,
    #[serde(default)]
    pub provider_options: Metadata,
}

/// One part of a streaming transcription.
#[derive(Clone, Debug, PartialEq)]
pub enum TranscriptionPart {
    Start {
        warnings: Vec<Warning>,
    },
    /// Text appended to the segment `id`.
    Delta {
        id: Option<String>,
        delta: String,
    },
    /// The segment's text so far; a later part may revise it.
    Partial {
        id: Option<String>,
        text: String,
    },
    /// The segment's final text.
    Final {
        id: Option<String>,
        text: String,
    },
    Finish {
        text: String,
        segments: Vec<TranscriptionSegment>,
        language: Option<String>,
        duration_seconds: Option<f64>,
        usage: Option<Value>,
        provider_metadata: Option<Metadata>,
    },
    Raw(Value),
}

/// A stream of transcription parts.
pub type TranscriptionStream =
    futures::stream::BoxStream<'static, Result<TranscriptionPart, crate::errors::Error>>;

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct StreamTranslateParams {
    pub target_language: String,
    pub source_language: Option<String>,
    pub output_format: Option<AudioFormat>,
    /// Include each provider message as a `Raw` part.
    #[serde(default)]
    pub include_raw: bool,
    #[serde(default)]
    pub provider_options: Metadata,
}

/// Audio and token counts of a speech translation.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct TranslationUsage {
    pub input_audio_seconds: Option<f64>,
    pub input_audio_tokens: Option<u64>,
    pub output_audio_tokens: Option<u64>,
    pub input_text_tokens: Option<u64>,
    pub output_text_tokens: Option<u64>,
}

/// One part of a streaming speech translation.
#[derive(Clone, Debug, PartialEq)]
pub enum TranslationPart {
    Start {
        warnings: Vec<Warning>,
    },
    /// Translated audio.
    Audio {
        id: Option<String>,
        audio: Vec<u8>,
    },
    OutputTextDelta {
        id: Option<String>,
        delta: String,
    },
    OutputTextFinal {
        id: Option<String>,
        text: String,
    },
    SourceTranscriptDelta {
        id: Option<String>,
        delta: String,
    },
    SourceTranscriptPartial {
        id: Option<String>,
        text: String,
    },
    SourceTranscriptFinal {
        id: Option<String>,
        text: String,
    },
    Finish {
        source_text: String,
        output_text: String,
        duration_seconds: Option<f64>,
        usage: Option<TranslationUsage>,
        provider_metadata: Option<Metadata>,
    },
    Raw(Value),
}

/// A stream of translation parts.
pub type TranslationStream =
    futures::stream::BoxStream<'static, Result<TranslationPart, crate::errors::Error>>;

/// A realtime session's settings, in the TypeScript
/// `RealtimeModelV4SessionConfig` shape. A protocol builds its own setup
/// message from it and ignores what its API does not take.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct RealtimeSessionConfig {
    pub instructions: Option<String>,
    pub voice: Option<String>,
    /// `text`, `audio` or both.
    #[serde(default)]
    pub output_modalities: Vec<String>,
    pub input_audio_format: Option<AudioFormat>,
    pub input_audio_transcription: Option<RealtimeTranscription>,
    pub output_audio_transcription: Option<RealtimeTranscription>,
    pub output_audio_format: Option<AudioFormat>,
    pub turn_detection: Option<Value>,
    #[serde(default)]
    pub tools: Vec<RealtimeTool>,
    #[serde(default)]
    pub provider_options: Metadata,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct RealtimeTranscription {
    pub model: Option<String>,
    pub language: Option<String>,
    pub prompt: Option<String>,
}

/// A function the model may call during a realtime session.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RealtimeTool {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Value,
}

/// An ephemeral token for a client to open a realtime session with.
#[derive(Clone, Serialize, Deserialize, PartialEq)]
pub struct RealtimeToken {
    pub token: String,
    /// The WebSocket URL the token opens.
    pub url: String,
    /// Seconds since the Unix epoch when the token expires.
    pub expires_at: Option<u64>,
}

// The token is a credential: keep it out of debug output.
impl std::fmt::Debug for RealtimeToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RealtimeToken")
            .field("expires_at", &self.expires_at)
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct RealtimeTokenParams {
    /// How long a client has to open a session with the token, in seconds.
    pub expires_after_seconds: Option<u64>,
    pub session: Option<RealtimeSessionConfig>,
}

/// An item a client adds to a realtime conversation.
#[derive(Clone, Debug, PartialEq)]
pub enum RealtimeItem {
    Text(String),
    /// Base64 audio.
    Audio(String),
    FunctionCallOutput {
        call_id: String,
        name: Option<String>,
        output: String,
    },
}

/// A normalized client event, as TypeScript's `RealtimeModelV4ClientEvent`.
#[derive(Clone, Debug, PartialEq)]
pub enum RealtimeClientEvent {
    SessionUpdate(RealtimeSessionConfig),
    SessionStart(RealtimeSessionConfig),
    SessionClose,
    InputAudioMute,
    InputAudioUnmute,
    ContextAppend {
        content: String,
        delegation_id: Option<String>,
    },
    /// Base64 audio in the session's input format.
    InputAudioAppend(String),
    InputAudioCommit,
    InputAudioClear,
    ConversationItemCreate(RealtimeItem),
    ConversationItemTruncate {
        item_id: String,
        content_index: u32,
        audio_end_ms: u64,
    },
    ResponseCreate {
        options: Option<Value>,
    },
    ResponseCancel,
}

/// A normalized server event, as TypeScript's `RealtimeModelV4ServerEvent`;
/// `raw` is the provider message it came from.
#[derive(Clone, Debug, PartialEq)]
pub struct RealtimeServerEvent {
    pub kind: RealtimeServerEventKind,
    pub raw: Value,
}

#[derive(Clone, Debug, PartialEq)]
pub enum RealtimeServerEventKind {
    SessionCreated {
        session_id: Option<String>,
    },
    SessionUpdated,
    SpeechStarted {
        item_id: Option<String>,
    },
    SpeechStopped {
        item_id: Option<String>,
    },
    AudioCommitted {
        item_id: Option<String>,
    },
    InputTranscriptionCompleted {
        item_id: String,
        transcript: String,
    },
    ResponseCreated {
        response_id: String,
    },
    ResponseDone {
        response_id: String,
        status: String,
    },
    AudioDelta {
        response_id: String,
        item_id: String,
        delta: String,
    },
    AudioDone {
        response_id: String,
        item_id: String,
    },
    AudioTranscriptDelta {
        response_id: String,
        item_id: String,
        delta: String,
    },
    AudioTranscriptDone {
        response_id: String,
        item_id: String,
        transcript: Option<String>,
    },
    TextDelta {
        response_id: String,
        item_id: String,
        delta: String,
    },
    TextDone {
        response_id: String,
        item_id: String,
        text: Option<String>,
    },
    FunctionCallArgumentsDelta {
        response_id: String,
        item_id: String,
        call_id: String,
        delta: String,
    },
    FunctionCallArgumentsDone {
        response_id: String,
        item_id: String,
        call_id: String,
        name: String,
        arguments: String,
    },
    Error {
        message: String,
        code: Option<String>,
    },
    /// A provider message with no normalized kind, by its provider name.
    Custom {
        raw_type: String,
    },
}

/// Turns normalized client events into provider messages and provider
/// messages into normalized server events, for one connection. It keeps the
/// state a protocol needs across messages, such as turn numbers.
pub trait RealtimeCodec: Send {
    /// The provider message for a client event, or `None` when the provider
    /// takes nothing for it.
    fn encode(
        &mut self,
        event: &RealtimeClientEvent,
    ) -> Result<Option<String>, crate::errors::Error>;
    /// The server events for one provider message.
    fn decode(&mut self, message: &str) -> Result<Vec<RealtimeServerEvent>, crate::errors::Error>;
}
