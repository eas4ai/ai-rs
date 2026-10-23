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

//! Gemini transcription: whole files on the Interactions API, and live
//! transcription on the Live API (`transcription/google-transcription-model.ts`).

use std::{sync::Arc, time::Duration};

use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde::Deserialize;
use serde_json::{Map, Value, json};

use super::{
    api::{Endpoint, Flavor, MEDIA_RESPONSE_BYTES},
    capabilities,
    live::{self, Control, Handler, Outcome, Session},
    options,
};
use crate::{
    Error,
    providers::{google_vertex, http, operations::*},
    transport::{WebSocketRequest, WebSocketTransport},
    types::{media, messages::FileData, usage::Usage},
};

/// The Interactions API revision every Interactions request names (GGL-026).
pub(crate) const API_REVISION: &str = "2026-05-20";

/// How long after the audio ends a session waits for a transcript or a
/// terminal signal before it finishes.
const FINISH_GRACE: Duration = Duration::from_secs(3);

const LIVE_PATH: &str = "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent";
const VERTEX_LIVE_PATH: &str = "google.cloud.aiplatform.v1.LlmBidiService/BidiGenerateContent";

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct TranscriptionOptions {
    pub language_codes: Option<Vec<String>>,
    pub custom_vocabulary: Option<Vec<String>>,
    pub word_timestamp: Option<bool>,
    pub diarization: Option<bool>,
    pub mode: Option<String>,
}

pub(crate) fn parse_options(
    flavor: Flavor,
    params: &crate::types::messages::Metadata,
) -> Result<TranscriptionOptions, Error> {
    let options: TranscriptionOptions = options::parse(flavor, Some(params))?;
    if let Some(mode) = &options.mode {
        options::check_one_of("mode", mode, &["SMART", "VERBATIM"])?;
    }
    Ok(options)
}

/// The camel-case `AudioTranscriptionConfig`, or `None` without options.
pub(crate) fn audio_transcription_config(options: &TranscriptionOptions) -> Option<Value> {
    let mut config = Map::new();
    if let Some(codes) = &options.language_codes {
        config.insert("languageCodes".into(), json!(codes));
    }
    if let Some(vocabulary) = &options.custom_vocabulary {
        config.insert("customVocabulary".into(), json!(vocabulary));
    }
    if let Some(value) = options.word_timestamp {
        config.insert("wordTimestamp".into(), json!(value));
    }
    if let Some(value) = options.diarization {
        config.insert("diarization".into(), json!(value));
    }
    if let Some(mode) = &options.mode {
        config.insert("mode".into(), json!(mode));
    }
    (!config.is_empty()).then_some(Value::Object(config))
}

/// The snake-case Interactions `transcription_config`, or `None` without
/// options.
fn transcription_config(options: &TranscriptionOptions) -> Option<Value> {
    let mut config = Map::new();
    if let Some(codes) = &options.language_codes {
        config.insert("language_codes".into(), json!(codes));
    }
    if let Some(vocabulary) = &options.custom_vocabulary {
        config.insert("custom_vocabulary".into(), json!(vocabulary));
    }
    if options.mode.is_some()
        || options.diarization == Some(true)
        || options.word_timestamp == Some(true)
    {
        let mut mode =
            json!({"type": options.mode.as_deref().unwrap_or("VERBATIM").to_ascii_lowercase()});
        if options.diarization == Some(true) {
            mode["diarization_mode"] = json!("speaker");
        }
        if options.word_timestamp == Some(true) {
            mode["timestamp_granularities"] = json!(["word"]);
        }
        config.insert("mode".into(), mode);
    }
    (!config.is_empty()).then_some(Value::Object(config))
}

/// The audio bytes and media type of a transcription input.
pub(crate) fn audio_input(audio: &TranscribeInput) -> Result<(Vec<u8>, String), Error> {
    match audio {
        TranscribeInput::Bytes(bytes) => {
            let media_type = media::detect_audio_media_type(bytes.as_slice()).ok_or_else(|| {
                Error::Validation(
                    "cannot detect the audio's media type; pass it as a FilePart".into(),
                )
            })?;
            Ok((bytes.clone(), media_type.to_owned()))
        }
        TranscribeInput::File(file) => match &file.data {
            FileData::Bytes(bytes) => Ok((bytes.clone(), file.media_type.clone())),
            FileData::Text(text) if media::is_downloadable_url(text) => Err(Error::Validation(
                "transcription takes the audio's bytes, not a URL".into(),
            )),
            FileData::Text(text) => {
                let data = media::data_to_base64(text);
                let bytes = STANDARD
                    .decode(data)
                    .map_err(|_| Error::Validation("the audio is not valid base64".into()))?;
                let media_type = media::split_data_url(text)
                    .0
                    .map_or_else(|| file.media_type.clone(), str::to_owned);
                Ok((bytes, media_type))
            }
        },
    }
}

fn offset_seconds(value: Option<&Value>) -> Option<f64> {
    let text = value?.as_str()?;
    let number: String = text
        .trim()
        .chars()
        .take_while(|c| c.is_ascii_digit() || matches!(c, '.' | '-' | '+' | 'e' | 'E'))
        .collect();
    number.parse::<f64>().ok().filter(|n| n.is_finite())
}

/// Transcribe a whole file with the Interactions API (GGL-016).
pub(crate) async fn transcribe(
    endpoint: &Endpoint,
    model_id: &str,
    audio: &TranscribeInput,
    params: &TranscribeParams,
) -> Result<Item<Transcription>, Error> {
    if model_id.contains("-live") {
        return Err(live_only(model_id));
    }
    let options = parse_options(endpoint.flavor, &params.provider_options)?;
    let (bytes, media_type) = audio_input(audio)?;
    let mut body = json!({
        "model": model_id,
        "input": [{"type": "audio", "data": STANDARD.encode(&bytes), "mime_type": media_type}],
    });
    if let Some(config) = transcription_config(&options) {
        body["generation_config"] = json!({"transcription_config": config});
    }
    let request = endpoint.request(
        "POST",
        &["interactions".into()],
        Some(body),
        [("Api-Revision".to_owned(), API_REVISION.to_owned())]
            .into_iter()
            .filter(|_| !has_header(&endpoint.config.headers, "api-revision"))
            .collect(),
        Default::default(),
    )?;
    let (_, response) = endpoint
        .json(request, Some(model_id), MEDIA_RESPONSE_BYTES)
        .await?;
    let mut text = String::new();
    let mut segments = Vec::new();
    for step in response
        .get("steps")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        for content in step
            .get("content")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let Some(value) = content.get("text").and_then(Value::as_str) else {
                continue;
            };
            if content.get("type").and_then(Value::as_str) != Some("text") {
                continue;
            }
            text.push_str(value);
            for annotation in content
                .get("annotations")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                if annotation.get("type").and_then(Value::as_str) != Some("word_info") {
                    continue;
                }
                let (Some(word), Some(start), Some(end)) = (
                    annotation.get("text").and_then(Value::as_str),
                    offset_seconds(annotation.get("start_offset")),
                    offset_seconds(annotation.get("end_offset")),
                ) else {
                    continue;
                };
                segments.push(TranscriptionSegment {
                    text: word.to_owned(),
                    start_second: start,
                    end_second: end,
                });
            }
        }
    }
    let usage = response.get("usage").filter(|u| u.is_object()).cloned();
    let mut item = Item::new(Transcription {
        text,
        segments,
        language: None,
        duration_seconds: None,
    });
    if let Some(usage) = usage {
        item.usage = Some(interactions_usage(&usage));
        item.provider_metadata = Some([("google".to_owned(), json!({"usage": usage}))].into());
    }
    Ok(item)
}

/// The error of a live model asked for a whole file.
fn live_only(model_id: &str) -> Error {
    Error::Validation(format!(
        "Model '{model_id}' only supports streaming transcription. Use ops::stream_transcription, or a unary model such as 'gemini-3.5-transcribe'."
    ))
}

/// Transcribe a whole file with Gemini on Vertex AI: `generateContent` with
/// the audio inline (VTX-011,
/// `gemini-transcription/google-vertex-gemini-transcription-model.ts`).
pub(crate) async fn transcribe_vertex_gemini(
    endpoint: &Endpoint,
    model_id: &str,
    audio: &TranscribeInput,
    params: &TranscribeParams,
) -> Result<Item<Transcription>, Error> {
    if model_id.contains("-live") {
        return Err(live_only(model_id));
    }
    let options = parse_options(Flavor::Vertex, &params.provider_options)?;
    let (bytes, media_type) = audio_input(audio)?;
    let mut body = json!({"contents": [{"role": "user", "parts": [
        {"inlineData": {"mimeType": media_type, "data": STANDARD.encode(&bytes)}}
    ]}]});
    if let Some(config) = audio_transcription_config(&options) {
        body["generationConfig"] = json!({"audioTranscriptionConfig": config});
    }
    let request = endpoint.request(
        "POST",
        &["models".into(), format!("{model_id}:generateContent")],
        Some(body),
        Default::default(),
        Default::default(),
    )?;
    let (_, response) = endpoint
        .json(request, Some(model_id), MEDIA_RESPONSE_BYTES)
        .await?;
    let parts = response
        .pointer("/candidates/0/content/parts")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let joined = |text: &dyn Fn(&Value) -> Option<&str>| -> String {
        parts.iter().filter_map(text).collect()
    };
    let plain = joined(&|part| part.get("text").and_then(Value::as_str));
    let text = if true {
        joined(&|part| {
            part.pointer("/audioTranscription/text")
                .and_then(Value::as_str)
        })
    } else {
        plain
    };
    let mut language = None;
    let mut segments = Vec::new();
    for transcription in parts
        .iter()
        .filter_map(|part| part.get("audioTranscription"))
    {
        if language.is_none() {
            language = transcription
                .get("languageCode")
                .and_then(Value::as_str)
                .map(str::to_owned);
        }
        for word in transcription
            .get("words")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if let (Some(text), Some(start), Some(end)) = (
                word.get("word").and_then(Value::as_str),
                offset_seconds(word.get("startOffset")),
                offset_seconds(word.get("endOffset")),
            ) {
                segments.push(TranscriptionSegment {
                    text: text.to_owned(),
                    start_second: start,
                    end_second: end,
                });
            }
        }
    }
    let mut item = Item::new(Transcription {
        text,
        segments,
        language,
        duration_seconds: None,
    });
    if let Some(usage) = response.get("usageMetadata").filter(|u| u.is_object()) {
        item.usage = Some(super::response::convert_usage(usage));
        item.provider_metadata =
            Some([("google".to_owned(), json!({"usageMetadata": usage}))].into());
    }
    Ok(item)
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct SpeechToTextOptions {
    language_codes: Option<Vec<String>>,
    enable_automatic_punctuation: Option<bool>,
    enable_word_time_offsets: Option<bool>,
    region: Option<String>,
}

/// The two-letter language of a BCP-47 tag, as `Intl.Locale` reads its
/// language subtag, or `None` for a longer one.
fn two_letter_language(tag: &str) -> Option<String> {
    let language = tag.split(['-', '_']).next()?.to_ascii_lowercase();
    (language.len() == 2 && language.bytes().all(|b| b.is_ascii_lowercase())).then_some(language)
}

/// Transcribe a whole file with Cloud Speech-to-Text v2 (VTX-010,
/// `google-vertex-transcription-model.ts`).
pub(crate) async fn transcribe_cloud(
    endpoint: &Endpoint,
    model_id: &str,
    audio: &TranscribeInput,
    params: &TranscribeParams,
) -> Result<Item<Transcription>, Error> {
    let options: SpeechToTextOptions =
        options::parse(Flavor::Vertex, Some(&params.provider_options))?;
    let project = google_vertex::project(&endpoint.config)?;
    let region = match &options.region {
        Some(region) if !google_vertex::is_dns_label(region) => {
            return Err(Error::Validation(
                "Invalid Google Cloud Speech-to-Text region. Expected a single DNS label (letters, digits, and hyphens).".into(),
            ));
        }
        Some(region) => region.clone(),
        None => google_vertex::location(&endpoint.config, None)?.to_owned(),
    };
    let (bytes, _) = audio_input(audio)?;
    let body = json!({
        "config": {
            "model": model_id,
            "languageCodes": options.language_codes.clone().unwrap_or_else(|| vec!["en-US".into()]),
            "autoDecodingConfig": {},
            "features": {
                "enableWordTimeOffsets": options.enable_word_time_offsets.unwrap_or(true),
                "enableAutomaticPunctuation": options.enable_automatic_punctuation.unwrap_or(true),
            },
        },
        "content": STANDARD.encode(&bytes),
    });
    let host = if region == "global" {
        "speech.googleapis.com".to_owned()
    } else {
        format!("{region}-speech.googleapis.com")
    };
    let request = endpoint.request_at(
        &format!("https://{host}/v2"),
        "POST",
        &[
            "projects".into(),
            project.to_owned(),
            "locations".into(),
            region.clone(),
            "recognizers".into(),
            "_:recognize".into(),
        ],
        Some(body),
        Default::default(),
        Default::default(),
    )?;
    let (_, response) = endpoint
        .json(request, Some(model_id), http::MAX_RESPONSE_BYTES)
        .await?;
    let results = response
        .get("results")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let first = |result: &Value| result.pointer("/alternatives/0").cloned();
    let text = results
        .iter()
        .map(|result| {
            first(result)
                .and_then(|alternative| {
                    alternative
                        .get("transcript")
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                })
                .unwrap_or_default()
        })
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_owned();
    let mut segments = Vec::new();
    for alternative in results.iter().filter_map(first) {
        for word in alternative
            .get("words")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if let (Some(text), Some(start), Some(end)) = (
                word.get("word").and_then(Value::as_str),
                offset_seconds(word.get("startOffset")),
                offset_seconds(word.get("endOffset")),
            ) {
                segments.push(TranscriptionSegment {
                    text: text.to_owned(),
                    start_second: start,
                    end_second: end,
                });
            }
        }
    }
    let language = results
        .first()
        .and_then(|result| result.get("languageCode"))
        .and_then(Value::as_str)
        .and_then(two_letter_language);
    Ok(Item::new(Transcription {
        text,
        segments,
        language,
        duration_seconds: offset_seconds(response.pointer("/metadata/totalBilledDuration")),
    }))
}

/// Interactions usage as the crate's usage.
pub(crate) fn interactions_usage(raw: &Value) -> Usage {
    let count = |key: &str| raw.get(key).and_then(Value::as_u64);
    let thoughts = count("total_thought_tokens");
    Usage {
        input_tokens: count("total_input_tokens").unwrap_or(0),
        output_tokens: count("total_output_tokens").unwrap_or(0) + thoughts.unwrap_or(0),
        reasoning_tokens: thoughts,
        cache_read_tokens: count("total_cached_tokens"),
        cache_write_tokens: None,
        raw: Some(raw.clone()),
    }
}

pub(crate) fn has_header(headers: &std::collections::BTreeMap<String, String>, name: &str) -> bool {
    headers.keys().any(|key| key.eq_ignore_ascii_case(name))
}

/// The Live API WebSocket of a call, with its headers: the key on the
/// Gemini API (Vertex AI's transport adds its own credentials) and the
/// configured headers.
pub(crate) fn live_request(endpoint: &Endpoint) -> Result<WebSocketRequest, Error> {
    let url = match endpoint.flavor {
        Flavor::Gemini => live::websocket_url(&endpoint.base, LIVE_PATH)?,
        Flavor::Vertex => {
            let base = url::Url::parse(&endpoint.base)
                .map_err(|_| Error::Configuration("the Vertex AI base URL is invalid".into()))?;
            let host = base
                .host_str()
                .ok_or_else(|| Error::Configuration("the Vertex AI base URL has no host".into()))?;
            format!("wss://{host}/ws/{VERTEX_LIVE_PATH}")
        }
    };
    let mut headers = std::collections::BTreeMap::new();
    if endpoint.flavor == Flavor::Gemini
        && let Some(key) = endpoint.config.api_key().filter(|key| !key.is_empty())
    {
        headers.insert("x-goog-api-key".to_owned(), key.to_owned());
    }
    for (name, value) in &endpoint.config.headers {
        headers.insert(name.to_ascii_lowercase(), value.clone());
    }
    Ok(WebSocketRequest {
        url,
        headers: headers.into_iter().collect(),
    })
}

/// The model path a Live setup names: the generate content path on the
/// Gemini API, the full resource name on Vertex AI.
pub(crate) fn live_model(endpoint: &Endpoint, model_id: &str) -> Result<String, Error> {
    match endpoint.flavor {
        Flavor::Gemini => Ok(capabilities::model_path(model_id)),
        Flavor::Vertex => {
            let base = url::Url::parse(&endpoint.base)
                .map_err(|_| Error::Configuration("the Vertex AI base URL is invalid".into()))?;
            let path = base.path().trim_matches('/');
            let resource = path
                .split_once("projects/")
                .map(|(_, rest)| format!("projects/{rest}"))
                .ok_or_else(|| {
                    Error::Configuration(
                        "Vertex AI live transcription needs a project and location".into(),
                    )
                })?;
            Ok(format!("{resource}/models/{model_id}"))
        }
    }
}

/// Check a live input format: raw 16 kHz 16-bit PCM.
pub(crate) fn check_pcm16k(format: &AudioFormat, what: &str) -> Result<(), Error> {
    if format.media_type != "audio/pcm" || format.sample_rate.is_some_and(|rate| rate != 16000) {
        return Err(Error::Validation(format!(
            "The Gemini Live {what} API only supports 16kHz 16-bit PCM input audio."
        )));
    }
    Ok(())
}

/// Live transcription (GGL-017, VTX-011).
pub(crate) fn stream(
    endpoint: &Endpoint,
    model_id: &str,
    transport: Arc<dyn WebSocketTransport>,
    audio: AudioChunks,
    format: &AudioFormat,
    params: &StreamTranscribeParams,
) -> Result<TranscriptionStream, Error> {
    if !model_id.contains("-live") {
        return Err(Error::Validation(format!(
            "Model '{model_id}' does not support streaming transcription. Use a live model such as 'gemini-3.5-transcribe-live'."
        )));
    }
    let options = parse_options(endpoint.flavor, &params.provider_options)?;
    check_pcm16k(format, "transcription")?;
    let request = live_request(endpoint)?;
    let setup = json!({
        "model": live_model(endpoint, model_id)?,
        "inputAudioTranscription": audio_transcription_config(&options).unwrap_or_else(|| json!({})),
    });
    let (label, error) = match endpoint.flavor {
        Flavor::Gemini => ("Google Live transcription", "Google Live API error"),
        Flavor::Vertex => ("Vertex Live transcription", "Vertex Live API error"),
    };
    let session = Session {
        request,
        setup,
        rate: format.sample_rate.unwrap_or(16000),
        include_raw: params.include_raw,
        provider: endpoint.provider.clone(),
        socket_error: match endpoint.flavor {
            Flavor::Gemini => "Google Live transcription error",
            Flavor::Vertex => "Vertex Live transcription error",
        },
    };
    let handler = Transcriber {
        provider: endpoint.provider.clone(),
        label,
        error,
        segment: 0,
        buffer: String::new(),
        full: String::new(),
        interim: String::new(),
        language: None,
        usage: None,
    };
    Ok(live::run(transport, session, audio, handler))
}

struct Transcriber {
    provider: String,
    label: &'static str,
    error: &'static str,
    segment: u64,
    buffer: String,
    full: String,
    interim: String,
    language: Option<String>,
    usage: Option<Value>,
}

impl Transcriber {
    fn id(&self) -> Option<String> {
        Some(format!("google-segment-{}", self.segment))
    }

    fn complete(&mut self, parts: &mut Vec<TranscriptionPart>) {
        if self.buffer.is_empty() {
            if self.interim.is_empty() {
                return;
            }
            self.buffer = std::mem::take(&mut self.interim);
        }
        self.interim.clear();
        let text = std::mem::take(&mut self.buffer);
        parts.push(TranscriptionPart::Final {
            id: self.id(),
            text: text.clone(),
        });
        if !self.full.is_empty() {
            self.full.push(' ');
        }
        self.full.push_str(&text);
        self.segment += 1;
    }

    fn finish(&mut self) -> Vec<TranscriptionPart> {
        let mut parts = Vec::new();
        self.complete(&mut parts);
        let metadata = self
            .usage
            .as_ref()
            .map(|usage| [("google".to_owned(), json!({"usageMetadata": usage}))].into());
        parts.push(TranscriptionPart::Finish {
            text: self.full.clone(),
            segments: Vec::new(),
            language: self.language.clone(),
            duration_seconds: None,
            usage: self.usage.clone(),
            provider_metadata: metadata,
        });
        parts
    }

    fn schedule(control: &mut Control) {
        if control.audio_ended {
            control.schedule(FINISH_GRACE);
        }
    }
}

impl Handler for Transcriber {
    type Part = TranscriptionPart;

    fn start(&mut self) -> TranscriptionPart {
        TranscriptionPart::Start {
            warnings: Vec::new(),
        }
    }

    fn raw(&self, message: Value) -> TranscriptionPart {
        TranscriptionPart::Raw(message)
    }

    fn message(
        &mut self,
        message: &Value,
        control: &mut Control,
    ) -> Result<Outcome<TranscriptionPart>, Error> {
        if let Some(usage) = message.get("usageMetadata").filter(|u| !u.is_null()) {
            self.usage = Some(usage.clone());
        }
        if message.get("error").is_some_and(|e| !e.is_null()) {
            return Err(live::server_error(&self.provider, message, self.error));
        }
        let mut parts = Vec::new();
        let content = message.get("serverContent");
        if let Some(text) = content
            .and_then(|c| c.get("interimInputTranscription"))
            .and_then(|t| t.get("text"))
            .and_then(Value::as_str)
            .filter(|t| !t.is_empty())
        {
            Self::schedule(control);
            self.interim = text.to_owned();
            parts.push(TranscriptionPart::Partial {
                id: self.id(),
                text: text.to_owned(),
            });
        }
        let transcription = content
            .and_then(|c| c.get("inputTranscription"))
            .filter(|t| !t.is_null())
            .or_else(|| message.get("inputTranscription").filter(|t| !t.is_null()));
        if let Some(transcription) = transcription {
            if let Some(language) = transcription.get("languageCode").and_then(Value::as_str) {
                self.language = Some(language.to_owned());
            }
            if let Some(text) = transcription
                .get("text")
                .and_then(Value::as_str)
                .filter(|t| !t.is_empty())
            {
                Self::schedule(control);
                self.interim.clear();
                self.buffer.push_str(text);
                parts.push(TranscriptionPart::Delta {
                    id: self.id(),
                    delta: text.to_owned(),
                });
            }
            if transcription.get("finished") == Some(&Value::Bool(true)) {
                self.complete(&mut parts);
            }
        }
        let turn_complete = content.and_then(|c| c.get("turnComplete")) == Some(&Value::Bool(true));
        if turn_complete {
            self.complete(&mut parts);
        }
        let status = content
            .and_then(|c| c.get("interactionStatus"))
            .and_then(Value::as_str);
        if control.audio_ended
            && (matches!(status, Some("IDLE" | "REQUIRES_ACTION"))
                || (turn_complete && status.is_none()))
        {
            parts.extend(self.finish());
            return Ok(Outcome::Finish(parts));
        }
        Ok(Outcome::Continue(parts))
    }

    fn audio_ended(&mut self, control: &mut Control) -> Outcome<TranscriptionPart> {
        Self::schedule(control);
        Outcome::Continue(Vec::new())
    }

    fn timer(&mut self) -> Vec<TranscriptionPart> {
        self.finish()
    }

    fn closed(
        &mut self,
        code: Option<u16>,
        reason: &str,
        control: &Control,
    ) -> Result<Vec<TranscriptionPart>, Error> {
        if control.audio_ended {
            Ok(self.finish())
        } else {
            Err(live::closed_error(&self.provider, self.label, code, reason))
        }
    }
}
