//! OpenAI's realtime API (`realtime/openai-realtime-model.ts` and
//! `openai-realtime-event-mapper.ts`) and Live API
//! (`live/openai-realtime-model-live.ts`, `openai-live-session-config.ts` and
//! `openai-live-event-mapper.ts`), OAI-024.

use serde_json::{Map, Value, json};

use super::{errors, params::invalid, protocol::http_request};
use crate::{
    Error, Model, Provider,
    providers::{
        http,
        operations::{
            AudioFormat, RealtimeClientEvent, RealtimeCodec, RealtimeItem, RealtimeServerEvent,
            RealtimeServerEventKind, RealtimeSessionConfig, RealtimeStartup, RealtimeToken,
            RealtimeTokenParams, WebRtcSession,
        },
    },
    transport::WebSocketRequest,
};

/// A model on the Live API: `gpt-live-` models.
pub(super) fn live(model: &str) -> bool {
    model.starts_with("gpt-live-")
}

fn format(format: &AudioFormat) -> Value {
    let mut value = json!({"type": format.media_type});
    if let Some(rate) = format.sample_rate {
        value["rate"] = json!(rate);
    }
    value
}

/// The realtime session setup (`buildOpenAISessionConfig`). The provider
/// options other than `openai` go into the session as given.
pub(super) fn session(config: &RealtimeSessionConfig, model: &str) -> Result<Value, Error> {
    let mut session = Map::new();
    session.insert("type".into(), json!("realtime"));
    session.insert("model".into(), json!(model));
    if let Some(instructions) = &config.instructions {
        session.insert("instructions".into(), json!(instructions));
    }
    if !config.output_modalities.is_empty() {
        session.insert("output_modalities".into(), json!(config.output_modalities));
    }
    let mut audio = Map::new();
    if config.input_audio_format.is_some()
        || config.input_audio_transcription.is_some()
        || config.turn_detection.is_some()
    {
        let mut input = Map::new();
        if let Some(input_format) = &config.input_audio_format {
            input.insert("format".into(), format(input_format));
        }
        if let Some(detection) = &config.turn_detection {
            input.insert("turn_detection".into(), turn_detection(detection)?);
        }
        if let Some(transcription) = &config.input_audio_transcription {
            let mut value =
                json!({"model": transcription.model.as_deref().unwrap_or("gpt-realtime-whisper")});
            if let Some(language) = &transcription.language {
                value["language"] = json!(language);
            }
            if let Some(prompt) = &transcription.prompt {
                value["prompt"] = json!(prompt);
            }
            input.insert("transcription".into(), value);
        }
        audio.insert("input".into(), Value::Object(input));
    }
    if config.output_audio_format.is_some() || config.voice.is_some() {
        let mut output = Map::new();
        if let Some(output_format) = &config.output_audio_format {
            output.insert("format".into(), format(output_format));
        }
        if let Some(voice) = &config.voice {
            output.insert("voice".into(), json!(voice));
        }
        audio.insert("output".into(), Value::Object(output));
    }
    if !audio.is_empty() {
        session.insert("audio".into(), Value::Object(audio));
    }
    if !config.tools.is_empty() {
        let tools: Vec<Value> = config
            .tools
            .iter()
            .map(|tool| {
                json!({"type": "function", "name": tool.name, "description": tool.description, "parameters": tool.parameters})
            })
            .collect();
        session.insert("tools".into(), json!(tools));
        session.insert("tool_choice".into(), json!("auto"));
    }
    for (key, value) in &config.provider_options {
        if key == "openai" {
            if !value.as_object().is_some_and(Map::is_empty) {
                return Err(invalid("OpenAI realtime sessions take no openai options"));
            }
            continue;
        }
        session.insert(key.clone(), value.clone());
    }
    Ok(Value::Object(session))
}

/// A turn detection setting: `{"type": "server-vad" | "semantic-vad" |
/// "disabled", "threshold", "silenceDurationMs", "prefixPaddingMs"}`.
fn turn_detection(detection: &Value) -> Result<Value, Error> {
    let kind = detection["type"].as_str().unwrap_or_default();
    let wire = match kind {
        "disabled" => return Ok(Value::Null),
        "server-vad" => "server_vad",
        "semantic-vad" => "semantic_vad",
        _ => {
            return Err(invalid(format!(
                "realtime turn detection must be server-vad, semantic-vad or disabled, not {kind:?}"
            )));
        }
    };
    let mut value = json!({"type": wire});
    for (key, name) in [
        ("threshold", "threshold"),
        ("silenceDurationMs", "silence_duration_ms"),
        ("prefixPaddingMs", "prefix_padding_ms"),
    ] {
        if let Some(setting) = detection.get(key).filter(|v| !v.is_null()) {
            value[name] = setting.clone();
        }
    }
    Ok(value)
}

/// The audio formats the Live API takes.
fn live_format(format: &AudioFormat) -> Result<Value, Error> {
    let valid = matches!(
        (format.media_type.as_str(), format.sample_rate),
        ("audio/pcm", Some(16_000 | 24_000)) | ("audio/pcma" | "audio/pcmu", Some(8_000))
    );
    if !valid {
        return Err(invalid(
            "OpenAI Live takes audio/pcm at 16 or 24 kHz, or audio/pcma or audio/pcmu at 8 kHz",
        ));
    }
    Ok(json!({"type": format.media_type, "rate": format.sample_rate}))
}

/// The Live session setup (`buildOpenAILiveSessionConfig`), for a
/// WebSocket or a WebRTC session.
pub(super) fn live_session(
    config: &RealtimeSessionConfig,
    model: &str,
    webrtc: bool,
) -> Result<Value, Error> {
    let refused = [
        ("outputModalities", !config.output_modalities.is_empty()),
        (
            "inputAudioTranscription",
            config.input_audio_transcription.is_some(),
        ),
        (
            "outputAudioTranscription",
            config.output_audio_transcription.is_some(),
        ),
        ("turnDetection", config.turn_detection.is_some()),
        ("tools", !config.tools.is_empty()),
    ];
    if let Some((setting, _)) = refused.iter().find(|(_, set)| *set) {
        return Err(invalid(format!(
            "OpenAI Live does not take the session setting {setting}"
        )));
    }
    for key in config.provider_options.keys() {
        if key != "openai" {
            return Err(invalid(format!(
                "OpenAI Live does not take the provider options {key:?}"
            )));
        }
    }
    let options = config
        .provider_options
        .get("openai")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let options = options
        .as_object()
        .ok_or_else(|| invalid("openai Live options must be an object"))?;
    for key in options.keys() {
        if !matches!(
            key.as_str(),
            "client" | "delegation" | "input" | "store" | "voice"
        ) {
            return Err(invalid(format!("unknown openai Live option {key:?}")));
        }
    }
    if let Some(delegation) = options.get("delegation").filter(|d| !d.is_null())
        && delegation != &json!({"type": "client"})
    {
        return Err(invalid(
            "OpenAI Live takes only client delegation, {\"type\": \"client\"}",
        ));
    }
    if options.contains_key("client") && !webrtc {
        return Err(invalid(
            "OpenAI Live takes client data channel permissions only for a WebRTC session",
        ));
    }
    let option_voice = options.get("voice").filter(|v| !v.is_null());
    if option_voice.is_some_and(|v| v["id"].as_str().is_none_or(str::is_empty)) {
        return Err(invalid(
            "openai Live option voice must be {\"id\": <voice id>}",
        ));
    }
    if option_voice.is_some() && config.voice.is_some() {
        return Err(invalid(
            "choose either voice or the openai Live option voice",
        ));
    }
    if options
        .get("input")
        .is_some_and(|input| input.as_array().is_none_or(|items| items.len() > 128))
    {
        return Err(invalid(
            "openai Live option input must be at most 128 items",
        ));
    }
    if options
        .get("store")
        .is_some_and(|store| !store.is_boolean())
    {
        return Err(invalid("openai Live option store must be a boolean"));
    }
    if webrtc && (config.input_audio_format.is_some() || config.output_audio_format.is_some()) {
        return Err(invalid(
            "OpenAI Live WebRTC negotiates audio through SDP and takes no audio formats",
        ));
    }
    let input = config
        .input_audio_format
        .as_ref()
        .map(live_format)
        .transpose()?;
    let output = config
        .output_audio_format
        .as_ref()
        .map(live_format)
        .transpose()?;
    if let (Some(input), Some(output)) = (&input, &output)
        && input != output
    {
        return Err(invalid(
            "OpenAI Live takes the same input and output audio format",
        ));
    }
    let voice = option_voice
        .cloned()
        .or_else(|| config.voice.as_ref().map(|v| json!(v)))
        .unwrap_or_else(|| json!("marin"));
    let mut audio = Map::new();
    if !webrtc {
        audio.insert(
            "format".into(),
            input
                .or(output)
                .unwrap_or_else(|| json!({"type": "audio/pcm", "rate": 24000})),
        );
    }
    audio.insert("output".into(), json!({"voice": voice}));
    let mut session = Map::new();
    session.insert("model".into(), json!(model));
    if let Some(instructions) = &config.instructions {
        session.insert("instructions".into(), json!(instructions));
    }
    session.insert("audio".into(), Value::Object(audio));
    if let Some(delegation) = options.get("delegation") {
        session.insert("delegation".into(), delegation.clone());
    }
    if let Some(client) = options.get("client") {
        let channel = &client["dataChannel"];
        let mut data_channel = Map::new();
        if let Some(events) = channel.get("allowedClientEvents") {
            data_channel.insert("allowed_client_events".into(), events.clone());
        }
        if let Some(events) = channel.get("allowedServerEvents") {
            let events = match events.as_array() {
                Some(selectors) => json!(
                    selectors
                        .iter()
                        .map(|s| {
                            let mut selector = json!({"type": s["type"]});
                            if let Some(event) = s.get("responseEvent") {
                                selector["response_event"] = event.clone();
                            }
                            selector
                        })
                        .collect::<Vec<_>>()
                ),
                None => events.clone(),
            };
            data_channel.insert("allowed_server_events".into(), events);
        }
        session.insert("client".into(), json!({"data_channel": data_channel}));
    }
    for key in ["input", "store"] {
        if let Some(value) = options.get(key) {
            session.insert(key.into(), value.clone());
        }
    }
    Ok(Value::Object(session))
}

/// The codec of the realtime API (`openai-realtime-event-mapper.ts`).
pub(super) struct Codec {
    pub model: String,
}

fn text(value: &Value, key: &str) -> String {
    value[key].as_str().unwrap_or_default().to_owned()
}

fn optional(value: &Value, key: &str) -> Option<String> {
    value[key].as_str().map(str::to_owned)
}

impl RealtimeCodec for Codec {
    fn encode(&mut self, event: &RealtimeClientEvent) -> Result<Option<String>, Error> {
        // TypeScript sends an event id with a session update and an audio
        // append only.
        let event_id = event.event_id();
        let with_id = |mut message: Value| {
            if let Some(id) = event_id {
                message["event_id"] = json!(id);
            }
            message
        };
        let message = match event.inner() {
            RealtimeClientEvent::SessionUpdate(config) => {
                with_id(json!({"type": "session.update", "session": session(config, &self.model)?}))
            }
            RealtimeClientEvent::InputAudioAppend(audio) => {
                with_id(json!({"type": "input_audio_buffer.append", "audio": audio}))
            }
            RealtimeClientEvent::InputAudioCommit => json!({"type": "input_audio_buffer.commit"}),
            RealtimeClientEvent::InputAudioClear => json!({"type": "input_audio_buffer.clear"}),
            RealtimeClientEvent::ConversationItemCreate(item) => {
                let item = match item {
                    RealtimeItem::Text(text) => {
                        json!({"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]})
                    }
                    RealtimeItem::Audio(audio) => {
                        json!({"type": "message", "role": "user", "content": [{"type": "input_audio", "audio": audio}]})
                    }
                    RealtimeItem::FunctionCallOutput {
                        call_id, output, ..
                    } => {
                        json!({"type": "function_call_output", "call_id": call_id, "output": output})
                    }
                };
                json!({"type": "conversation.item.create", "item": item})
            }
            RealtimeClientEvent::ConversationItemTruncate {
                item_id,
                content_index,
                audio_end_ms,
            } => json!({"type": "conversation.item.truncate", "item_id": item_id,
                        "content_index": content_index, "audio_end_ms": audio_end_ms}),
            RealtimeClientEvent::ResponseCreate { options } => {
                let mut message = json!({"type": "response.create"});
                if let Some(options) = options {
                    let mut response = Map::new();
                    for (key, wire) in [
                        ("modalities", "output_modalities"),
                        ("instructions", "instructions"),
                        ("metadata", "metadata"),
                    ] {
                        if let Some(value) = options.get(key).filter(|v| !v.is_null()) {
                            response.insert(wire.into(), value.clone());
                        }
                    }
                    message["response"] = Value::Object(response);
                }
                message
            }
            RealtimeClientEvent::ResponseCancel => json!({"type": "response.cancel"}),
            // The realtime API has no message for these; `inner` never
            // returns an event with an id.
            RealtimeClientEvent::SessionStart(_)
            | RealtimeClientEvent::SessionClose
            | RealtimeClientEvent::InputAudioMute
            | RealtimeClientEvent::InputAudioUnmute
            | RealtimeClientEvent::ContextAppend { .. }
            | RealtimeClientEvent::WithEventId { .. } => return Ok(None),
        };
        Ok(Some(message.to_string()))
    }

    fn decode(&mut self, message: &str) -> Result<Vec<RealtimeServerEvent>, Error> {
        let Ok(raw) = serde_json::from_str::<Value>(message) else {
            return Ok(Vec::new());
        };
        let response_id =
            || optional(&raw["response"], "id").unwrap_or_else(|| text(&raw, "response_id"));
        let item_id = || optional(&raw["item"], "id").unwrap_or_else(|| text(&raw, "item_id"));
        let kind = match raw["type"].as_str().unwrap_or_default() {
            "session.created" => RealtimeServerEventKind::SessionCreated {
                session_id: optional(&raw["session"], "id"),
            },
            "session.updated" => RealtimeServerEventKind::SessionUpdated,
            "input_audio_buffer.speech_started" => RealtimeServerEventKind::SpeechStarted {
                item_id: optional(&raw, "item_id"),
            },
            "input_audio_buffer.speech_stopped" => RealtimeServerEventKind::SpeechStopped {
                item_id: optional(&raw, "item_id"),
            },
            "input_audio_buffer.committed" => RealtimeServerEventKind::AudioCommitted {
                item_id: optional(&raw, "item_id"),
                previous_item_id: optional(&raw, "previous_item_id"),
            },
            "conversation.item.added" => RealtimeServerEventKind::ConversationItemAdded {
                item_id: item_id(),
                item: raw["item"].clone(),
            },
            "conversation.item.input_audio_transcription.completed" => {
                RealtimeServerEventKind::InputTranscriptionCompleted {
                    item_id: text(&raw, "item_id"),
                    transcript: text(&raw, "transcript"),
                }
            }
            "response.created" => RealtimeServerEventKind::ResponseCreated {
                response_id: response_id(),
            },
            "response.done" => RealtimeServerEventKind::ResponseDone {
                response_id: response_id(),
                status: optional(&raw["response"], "status").unwrap_or_else(|| "completed".into()),
            },
            "response.output_item.added" => RealtimeServerEventKind::OutputItemAdded {
                response_id: text(&raw, "response_id"),
                item_id: item_id(),
            },
            "response.output_item.done" => RealtimeServerEventKind::OutputItemDone {
                response_id: text(&raw, "response_id"),
                item_id: item_id(),
            },
            "response.content_part.added" => RealtimeServerEventKind::ContentPartAdded {
                response_id: text(&raw, "response_id"),
                item_id: text(&raw, "item_id"),
            },
            "response.content_part.done" => RealtimeServerEventKind::ContentPartDone {
                response_id: text(&raw, "response_id"),
                item_id: text(&raw, "item_id"),
            },
            "response.output_audio.delta" => RealtimeServerEventKind::AudioDelta {
                response_id: text(&raw, "response_id"),
                item_id: text(&raw, "item_id"),
                delta: text(&raw, "delta"),
            },
            "response.output_audio.done" => RealtimeServerEventKind::AudioDone {
                response_id: text(&raw, "response_id"),
                item_id: text(&raw, "item_id"),
            },
            "response.output_audio_transcript.delta" => {
                RealtimeServerEventKind::AudioTranscriptDelta {
                    response_id: text(&raw, "response_id"),
                    item_id: text(&raw, "item_id"),
                    delta: text(&raw, "delta"),
                }
            }
            "response.output_audio_transcript.done" => {
                RealtimeServerEventKind::AudioTranscriptDone {
                    response_id: text(&raw, "response_id"),
                    item_id: text(&raw, "item_id"),
                    transcript: optional(&raw, "transcript"),
                }
            }
            "response.output_text.delta" => RealtimeServerEventKind::TextDelta {
                response_id: text(&raw, "response_id"),
                item_id: text(&raw, "item_id"),
                delta: text(&raw, "delta"),
            },
            "response.output_text.done" => RealtimeServerEventKind::TextDone {
                response_id: text(&raw, "response_id"),
                item_id: text(&raw, "item_id"),
                text: optional(&raw, "text"),
            },
            "response.function_call_arguments.delta" => {
                RealtimeServerEventKind::FunctionCallArgumentsDelta {
                    response_id: text(&raw, "response_id"),
                    item_id: text(&raw, "item_id"),
                    call_id: text(&raw, "call_id"),
                    delta: text(&raw, "delta"),
                }
            }
            "response.function_call_arguments.done" => {
                RealtimeServerEventKind::FunctionCallArgumentsDone {
                    response_id: text(&raw, "response_id"),
                    item_id: text(&raw, "item_id"),
                    call_id: text(&raw, "call_id"),
                    name: text(&raw, "name"),
                    arguments: text(&raw, "arguments"),
                }
            }
            "error" => RealtimeServerEventKind::Error {
                message: optional(&raw["error"], "message")
                    .or_else(|| optional(&raw, "message"))
                    .unwrap_or_else(|| "Unknown error".into()),
                code: optional(&raw["error"], "code").or_else(|| optional(&raw, "code")),
                client_event_id: optional(&raw["error"], "event_id"),
            },
            other => RealtimeServerEventKind::Custom {
                raw_type: other.to_owned(),
            },
        };
        Ok(vec![RealtimeServerEvent { kind, raw }])
    }
}

/// The codec of the Live API (`openai-live-event-mapper.ts`).
pub(super) struct LiveCodec {
    pub model: String,
}

/// A Live server event that does not read as its type says.
fn invalid_event(raw: Value, message: &str) -> RealtimeServerEvent {
    RealtimeServerEvent {
        kind: RealtimeServerEventKind::Error {
            message: message.into(),
            code: Some("invalid_server_event".into()),
            client_event_id: None,
        },
        raw,
    }
}

/// The Live channel a context append goes to: the `openai` option
/// `channel`, `thinking` by default.
fn context_channel(options: &crate::providers::operations::Metadata) -> Result<&str, Error> {
    let Some(openai) = options.get("openai") else {
        return Ok("thinking");
    };
    let openai = openai
        .as_object()
        .ok_or_else(|| invalid("OpenAI Live context-append options must be an object"))?;
    if let Some(key) = openai.keys().find(|key| *key != "channel") {
        return Err(invalid(format!(
            "OpenAI Live context-append takes the option channel, not {key}"
        )));
    }
    match openai.get("channel").filter(|v| !v.is_null()) {
        None => Ok("thinking"),
        Some(Value::String(channel))
            if ["instructions", "thinking", "commentary"].contains(&channel.as_str()) =>
        {
            Ok(channel)
        }
        Some(other) => Err(invalid(format!(
            "OpenAI Live context-append channel must be instructions, thinking or commentary, not {other}"
        ))),
    }
}

impl RealtimeCodec for LiveCodec {
    fn encode(&mut self, event: &RealtimeClientEvent) -> Result<Option<String>, Error> {
        let mut message = match event.inner() {
            RealtimeClientEvent::SessionStart(config) => {
                json!({"type": "session.start", "session": live_session(config, &self.model, false)?})
            }
            RealtimeClientEvent::SessionUpdate(_) => {
                return Err(invalid(
                    "OpenAI Live sessions take no session update: their settings are fixed at the start; use context-append or input-audio mute and unmute",
                ));
            }
            RealtimeClientEvent::SessionClose => json!({"type": "session.close"}),
            RealtimeClientEvent::InputAudioAppend(audio) => {
                json!({"type": "session.input_audio.append", "audio": audio})
            }
            RealtimeClientEvent::InputAudioMute => json!({"type": "session.input_audio.mute"}),
            RealtimeClientEvent::InputAudioUnmute => json!({"type": "session.input_audio.unmute"}),
            RealtimeClientEvent::ContextAppend {
                content,
                delegation_id,
                provider_options,
            } => {
                let channel = context_channel(provider_options)?;
                json!({"type": format!("session.{channel}.append"), "content": content, "delegation_id": delegation_id})
            }
            other => {
                return Err(invalid(format!(
                    "OpenAI Live takes continuous audio and context-append, not {other:?}"
                )));
            }
        };
        // TypeScript sends the event id with every Live command.
        if let Some(id) = event.event_id() {
            message["event_id"] = json!(id);
        }
        Ok(Some(message.to_string()))
    }

    fn decode(&mut self, message: &str) -> Result<Vec<RealtimeServerEvent>, Error> {
        let Ok(raw) = serde_json::from_str::<Value>(message) else {
            return Ok(Vec::new());
        };
        let kind_name = raw["type"].as_str().unwrap_or_default().to_owned();
        let number = |value: &Value| value.as_f64().filter(|n| *n >= 0.0);
        let acknowledgment = |command: &str| RealtimeServerEventKind::CommandAcknowledged {
            command: command.to_owned(),
            client_event_id: optional(&raw, "client_event_id"),
        };
        let kind = match kind_name.as_str() {
            "session.started" => match raw["session"]["id"].as_str().filter(|id| !id.is_empty()) {
                Some(id) => RealtimeServerEventKind::SessionStarted {
                    session_id: id.to_owned(),
                    delegation_mode: if raw["session"]["delegation"]["type"] == "responses" {
                        "provider".into()
                    } else {
                        "client".into()
                    },
                },
                None => {
                    return Ok(vec![invalid_event(
                        raw,
                        "Invalid OpenAI Live server event.",
                    )]);
                }
            },
            "session.closed" => match number(&raw["usage"]["seconds"]) {
                Some(seconds) if raw["reason"].is_string() => {
                    RealtimeServerEventKind::SessionClosed {
                        session_id: optional(&raw["session"], "id"),
                        usage_seconds: seconds,
                        reason: text(&raw, "reason"),
                    }
                }
                _ => {
                    return Ok(vec![invalid_event(
                        raw,
                        "Invalid OpenAI Live server event.",
                    )]);
                }
            },
            "session.usage.updated" => match number(&raw["usage"]["seconds"]) {
                Some(seconds) => RealtimeServerEventKind::SessionUsage {
                    usage_seconds: seconds,
                    context_window_usage_ratio: raw["context_window"]["usage_ratio"].as_f64(),
                },
                None => {
                    return Ok(vec![invalid_event(
                        raw,
                        "Invalid OpenAI Live server event.",
                    )]);
                }
            },
            "session.output_audio.delta" => match raw["delta"].as_str() {
                Some(delta) => RealtimeServerEventKind::AudioChunk {
                    delta: delta.to_owned(),
                },
                None => {
                    return Ok(vec![invalid_event(
                        raw,
                        "Invalid OpenAI Live server event.",
                    )]);
                }
            },
            "session.input_transcript.delta" | "session.output_transcript.delta" => {
                match (
                    raw["delta"].as_str(),
                    number(&raw["start_ms"]),
                    number(&raw["end_ms"]),
                ) {
                    (Some(delta), Some(start), Some(end)) if end >= start => {
                        RealtimeServerEventKind::TranscriptFragment {
                            speaker: if kind_name == "session.input_transcript.delta" {
                                "user".into()
                            } else {
                                "assistant".into()
                            },
                            delta: delta.to_owned(),
                            start_ms: start,
                            end_ms: end,
                        }
                    }
                    (Some(_), Some(_), Some(_)) => {
                        return Ok(vec![invalid_event(
                            raw,
                            "Invalid OpenAI Live event time interval.",
                        )]);
                    }
                    _ => {
                        return Ok(vec![invalid_event(
                            raw,
                            "Invalid OpenAI Live server event.",
                        )]);
                    }
                }
            }
            "session.delegation.created" => {
                match raw["delegation"]["id"].as_str().filter(|id| !id.is_empty()) {
                    Some(id) => RealtimeServerEventKind::DelegationCreated {
                        delegation_id: id.to_owned(),
                        target: match raw["delegation"]["target"].as_str() {
                            Some("responses") => Some("provider".into()),
                            other => other.map(str::to_owned),
                        },
                        offset_ms: number(&raw["offset_ms"]),
                        response_id: optional(&raw["delegation"], "response_id"),
                    },
                    None => {
                        return Ok(vec![invalid_event(
                            raw,
                            "Invalid OpenAI Live server event.",
                        )]);
                    }
                }
            }
            "error" => match raw["error"]["message"].as_str() {
                Some(message) => RealtimeServerEventKind::Error {
                    message: message.to_owned(),
                    code: optional(&raw["error"], "code"),
                    client_event_id: optional(&raw["error"], "client_event_id"),
                },
                None => {
                    return Ok(vec![invalid_event(
                        raw,
                        "Invalid OpenAI Live server event.",
                    )]);
                }
            },
            "session.updated" => acknowledgment("session.update"),
            "session.input_audio.muted" => acknowledgment("session.input_audio.mute"),
            "session.input_audio.unmuted" => acknowledgment("session.input_audio.unmute"),
            "session.instructions.appended" => acknowledgment("session.instructions.append"),
            "session.thinking.appended" => acknowledgment("session.thinking.append"),
            "session.commentary.appended" => acknowledgment("session.commentary.append"),
            other => RealtimeServerEventKind::Custom {
                raw_type: other.to_owned(),
            },
        };
        Ok(vec![RealtimeServerEvent { kind, raw }])
    }
}

/// A token for a client to open a realtime session with
/// (`doCreateClientSecret`); the Live API takes none.
pub(super) async fn create_token(
    model: &Model,
    params: RealtimeTokenParams,
    provider: &dyn Provider,
) -> Result<RealtimeToken, Error> {
    if live(&model.id) {
        return Err(invalid(
            "OpenAI Live takes no short-lived tokens; a server opens it with its key through connect_server",
        ));
    }
    let session = match &params.session {
        Some(config) => session(config, &model.id)?,
        None => json!({"type": "realtime", "model": model.id}),
    };
    let mut body = json!({"session": session});
    if let Some(seconds) = params.expires_after_seconds {
        body["expires_after"] = json!({"anchor": "created_at", "seconds": seconds});
    }
    let request = http_request(
        &errors::API,
        "POST",
        provider.config(),
        &["realtime", "client_secrets"],
        Some(body),
        None,
    )?;
    let transport = provider.transport().ok_or_else(|| {
        Error::Configuration(format!(
            "provider {:?} has no transport for realtime tokens",
            provider.name()
        ))
    })?;
    let response = http::json(
        &errors::API,
        transport,
        request,
        provider.name(),
        Some(&model.id),
    )
    .await?;
    let token = response["value"].as_str().ok_or_else(|| {
        http::response_error(
            provider.name(),
            "OpenAI realtime client secret response has no value",
        )
    })?;
    Ok(RealtimeToken {
        token: token.to_owned(),
        url: realtime_url(model, provider)?,
        expires_at: response["expires_at"].as_u64(),
    })
}

/// The realtime socket URL, `wss://<host>/v1/realtime?model=<id>`: the base
/// URL's host and port with a fixed path, as TypeScript's token names it. A
/// token and a server connection open this same URL (OAI-024).
fn realtime_url(model: &Model, provider: &dyn Provider) -> Result<String, Error> {
    let base = url::Url::parse(&provider.config().base_url())
        .map_err(|_| Error::Configuration("OpenAI base URL is invalid".into()))?;
    let host = base
        .host_str()
        .ok_or_else(|| Error::Configuration("OpenAI base URL has no host".into()))?;
    let host = match base.port() {
        Some(port) => format!("{host}:{port}"),
        None => host.to_owned(),
    };
    let mut url = url::Url::parse(&format!("wss://{host}/v1/realtime"))
        .map_err(|_| Error::Configuration("OpenAI realtime URL is invalid".into()))?;
    url.query_pairs_mut().append_pair("model", &model.id);
    Ok(url.into())
}

/// The socket a client opens with a token; it sends the session update
/// without waiting for the session to be created.
pub(super) fn token_connection(
    model: &Model,
    token: &RealtimeToken,
) -> Result<(WebSocketRequest, bool), Error> {
    if live(&model.id) {
        return Err(invalid(
            "OpenAI Live takes no tokens; a server opens it with its key through connect_server",
        ));
    }
    Ok((
        WebSocketRequest {
            url: token.url.clone(),
            headers: vec![("authorization".into(), format!("Bearer {}", token.token))],
        },
        false,
    ))
}

/// The socket a server opens with the provider's key: `<base as
/// wss>/live/sessions` for the Live API, which starts with a session start,
/// and otherwise the URL a token names, which takes a session update.
pub(super) fn server_connection(
    model: &Model,
    provider: &dyn Provider,
) -> Result<(WebSocketRequest, RealtimeStartup), Error> {
    // The request gives the key's headers, and the Live API its URL.
    let request = http::request(
        &errors::API,
        provider.config(),
        "GET",
        &["live", "sessions"],
        None,
        Default::default(),
        Default::default(),
    )?;
    let (url, startup) = if live(&model.id) {
        let url = match request.url.split_once("://") {
            Some(("http", rest)) => format!("ws://{rest}"),
            Some((_, rest)) => format!("wss://{rest}"),
            None => request.url.clone(),
        };
        (url, RealtimeStartup::SessionStart)
    } else {
        (
            realtime_url(model, provider)?,
            RealtimeStartup::SessionUpdate,
        )
    };
    let headers = request
        .headers
        .into_iter()
        .filter(|(name, _)| !name.eq_ignore_ascii_case("accept"))
        .collect();
    Ok((WebSocketRequest { url, headers }, startup))
}

/// A Live WebRTC session from a client's SDP offer
/// (`doCreateWebRTCSession`); the realtime API has none here.
pub(super) async fn webrtc_session(
    model: &Model,
    offer_sdp: String,
    config: RealtimeSessionConfig,
    provider: &dyn Provider,
) -> Result<WebRtcSession, Error> {
    if !live(&model.id) {
        return Err(Error::UnsupportedOperation("webrtc session".into()));
    }
    if offer_sdp.is_empty() {
        return Err(invalid("a WebRTC session needs an SDP offer"));
    }
    let body = json!({
        "session": live_session(&config, &model.id, true)?,
        "transport": {"type": "webrtc", "sdp": offer_sdp},
    });
    let request = http_request(
        &errors::API,
        "POST",
        provider.config(),
        &["live", "sessions"],
        Some(body),
        None,
    )?;
    let transport = provider.transport().ok_or_else(|| {
        Error::Configuration(format!(
            "provider {:?} has no transport for WebRTC sessions",
            provider.name()
        ))
    })?;
    let response = http::json(
        &errors::API,
        transport,
        request,
        provider.name(),
        Some(&model.id),
    )
    .await?;
    let bad = || {
        http::response_error(
            provider.name(),
            "OpenAI Live WebRTC response has no session id or SDP answer",
        )
    };
    let session_id = response["session"]["id"]
        .as_str()
        .filter(|id| !id.is_empty())
        .ok_or_else(bad)?;
    let answer = response["transport"]["sdp"]
        .as_str()
        .filter(|sdp| !sdp.is_empty() && response["transport"]["type"] == "webrtc")
        .ok_or_else(bad)?;
    Ok(WebRtcSession {
        session_id: session_id.to_owned(),
        answer_sdp: answer.to_owned(),
    })
}
