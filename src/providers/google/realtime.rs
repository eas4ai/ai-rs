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

//! Gemini Live API realtime sessions: ephemeral tokens, the session setup
//! and the event codec (`realtime/google-realtime-model.ts` and
//! `realtime/google-realtime-event-mapper.ts`). A server-side connection
//! sends the token as `Authorization: Token <token>` and waits for
//! `setupComplete`, as Google's Go SDK does (`live.go`).

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::{Map, Value, json};

use super::{api::Endpoint, capabilities, live};
use crate::{
    Error,
    errors::{ProviderError, ProviderErrorKind},
    providers::{http, operations::*},
    transport::WebSocketRequest,
};

const CONSTRAINED_PATH: &str =
    "google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContentConstrained";

/// The token's session window when none is asked for.
const DEFAULT_WINDOW: u64 = 60;
/// How much longer than its session window a token lives.
const TOKEN_EXTRA: Duration = Duration::from_secs(30 * 60);

/// Whether a Live model reasons in the background, such as
/// `gemini-3.8-live-extended-thinking`.
fn is_thinking_live(model_id: &str) -> bool {
    let name = model_id
        .rsplit('/')
        .next()
        .unwrap_or(model_id)
        .to_ascii_lowercase();
    let Some(rest) = name.strip_prefix("gemini-") else {
        return false;
    };
    let digits = |text: &str| text.find(|c: char| !c.is_ascii_digit()).filter(|n| *n > 0);
    let Some(major) = digits(rest) else {
        return false;
    };
    let Some(rest) = rest[major..].strip_prefix('.') else {
        return false;
    };
    let Some(minor) = digits(rest) else {
        return false;
    };
    let Some(rest) = rest[minor..].strip_prefix("-live") else {
        return false;
    };
    // `\b` after `live`: the end, or a character that is not a word character.
    let boundary = rest
        .chars()
        .next()
        .is_none_or(|c| !(c.is_ascii_alphanumeric() || c == '_'));
    boundary && rest.contains("thinking")
}

/// The Live setup for a session configuration (`buildGoogleSessionConfig`).
pub(crate) fn session_setup(
    config: &RealtimeSessionConfig,
    model_id: &str,
) -> Result<Value, Error> {
    let mut setup = Map::new();
    setup.insert("model".into(), json!(capabilities::model_path(model_id)));
    let mut rest = config.provider_options.clone();
    let google = rest.remove("google");
    let google = match google {
        Some(Value::Object(google)) => Some(google),
        Some(Value::Null) | None => None,
        Some(_) => {
            return Err(Error::Validation(
                "google realtime options must be an object".into(),
            ));
        }
    };
    if let Some(google) = &google {
        for key in google.keys() {
            if !matches!(
                key.as_str(),
                "translationConfig" | "thinkingConfig" | "defaultToolBehavior"
            ) {
                return Err(Error::Validation(format!(
                    "unknown google realtime option {key}"
                )));
            }
        }
        if let Some(behavior) = google.get("defaultToolBehavior").and_then(Value::as_str) {
            super::options::check_one_of(
                "defaultToolBehavior",
                behavior,
                &["BLOCKING", "NON_BLOCKING"],
            )?;
        }
        if let Some(level) = google
            .get("thinkingConfig")
            .and_then(|t| t.get("thinkingLevel"))
            .and_then(Value::as_str)
        {
            super::options::check_one_of(
                "thinkingConfig.thinkingLevel",
                level,
                &["low", "medium", "high"],
            )?;
        }
    }
    let option = |key: &str| {
        google
            .as_ref()
            .and_then(|g| g.get(key))
            .filter(|v| !v.is_null())
            .cloned()
    };
    let mut generation = Map::new();
    let modalities: Vec<String> = if config.output_modalities.is_empty() {
        vec!["AUDIO".into()]
    } else {
        config
            .output_modalities
            .iter()
            .map(|m| m.to_ascii_uppercase())
            .collect()
    };
    generation.insert("responseModalities".into(), json!(modalities));
    if let Some(voice) = &config.voice {
        generation.insert(
            "speechConfig".into(),
            json!({"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}}),
        );
    }
    setup.insert("generationConfig".into(), Value::Object(generation));
    if let Some(instructions) = &config.instructions {
        setup.insert(
            "systemInstruction".into(),
            json!({"parts": [{"text": instructions}]}),
        );
    }
    if !config.tools.is_empty() {
        let behavior = option("defaultToolBehavior");
        let declarations: Vec<Value> = config
            .tools
            .iter()
            .map(|tool| {
                let mut declaration =
                    json!({"name": tool.name, "parametersJsonSchema": tool.parameters});
                if let Some(description) = &tool.description {
                    declaration["description"] = json!(description);
                }
                if let Some(behavior) = &behavior {
                    declaration["behavior"] = behavior.clone();
                }
                declaration
            })
            .collect();
        setup.insert(
            "tools".into(),
            json!([{"functionDeclarations": declarations}]),
        );
    }
    if config.input_audio_transcription.is_some() {
        setup.insert("inputAudioTranscription".into(), json!({}));
    }
    if config.output_audio_transcription.is_some() {
        setup.insert("outputAudioTranscription".into(), json!({}));
    }
    let explicit = option("thinkingConfig").or_else(|| {
        rest.get("generationConfig")
            .and_then(|g| g.get("thinkingConfig"))
            .filter(|t| t.is_object())
            .cloned()
    });
    let thinking = if is_thinking_live(model_id)
        && explicit.as_ref().is_none_or(|t| {
            t.get("thinkingLevel").is_none_or(Value::is_null)
                && t.get("thinkingBudget").is_none_or(Value::is_null)
        }) {
        let mut thinking = explicit
            .and_then(|t| t.as_object().cloned())
            .unwrap_or_default();
        thinking.insert("thinkingLevel".into(), json!("low"));
        Some(Value::Object(thinking))
    } else {
        option("thinkingConfig")
    };
    if !config.provider_options.is_empty() {
        setup.extend(rest);
        if let Some(translation) = option("translationConfig") {
            merge_generation(&mut setup, "translationConfig", translation);
        }
    }
    if let Some(thinking) = thinking {
        merge_generation(&mut setup, "thinkingConfig", thinking);
    }
    Ok(Value::Object(setup))
}

fn merge_generation(setup: &mut Map<String, Value>, key: &str, value: Value) {
    let generation = setup.entry("generationConfig").or_insert_with(|| json!({}));
    if !generation.is_object() {
        *generation = json!({});
    }
    generation[key] = value;
}

/// Seconds since the epoch as an RFC 3339 UTC time with milliseconds, as
/// JavaScript's `toISOString` writes it.
pub(crate) fn iso_time(time: SystemTime) -> String {
    let since = time.duration_since(UNIX_EPOCH).unwrap_or_default();
    let seconds = since.as_secs();
    let (days, rest) = (seconds / 86_400, seconds % 86_400);
    let (year, month, day) = civil_from_days(days as i64);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{:03}Z",
        rest / 3600,
        rest % 3600 / 60,
        rest % 60,
        since.subsec_millis()
    )
}

/// The civil date of a day count since 1970-01-01 (Howard Hinnant's
/// `civil_from_days`).
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let year = yoe + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
    let year = if month <= 2 { year - 1 } else { year };
    let era = year.div_euclid(400);
    let yoe = year.rem_euclid(400);
    let month = i64::from(month);
    let doy = (153 * (if month > 2 { month - 3 } else { month + 9 }) + 2) / 5 + i64::from(day) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// The whole seconds since the epoch of an RFC 3339 UTC time such as
/// `2026-10-23T05:30:00.123Z`.
pub(crate) fn parse_iso_seconds(text: &str) -> Option<u64> {
    let text = text.strip_suffix('Z')?;
    let (date, time) = text.split_once('T')?;
    let mut date = date.split('-');
    let year: i64 = date.next()?.parse().ok()?;
    let month: u32 = date.next()?.parse().ok()?;
    let day: u32 = date.next()?.parse().ok()?;
    let time = time.split('.').next()?;
    let mut time = time.split(':');
    let hour: u64 = time.next()?.parse().ok()?;
    let minute: u64 = time.next()?.parse().ok()?;
    let second: u64 = time.next()?.parse().ok()?;
    if !(1..=12).contains(&month)
        || !(1..=31).contains(&day)
        || hour > 23
        || minute > 59
        || second > 60
    {
        return None;
    }
    let days = u64::try_from(days_from_civil(year, month, day)).ok()?;
    Some(days * 86_400 + hour * 3600 + minute * 60 + second)
}

/// Create an ephemeral token (GGL-023).
pub(crate) async fn create_token(
    endpoint: &Endpoint,
    model_id: &str,
    params: &RealtimeTokenParams,
) -> Result<RealtimeToken, Error> {
    if endpoint.config.api_key().is_none_or(str::is_empty) {
        return Err(Error::Configuration(
            "Google Generative AI API key is required for realtime token creation.".into(),
        ));
    }
    let setup = session_setup(&params.session.clone().unwrap_or_default(), model_id)?;
    let now = SystemTime::now();
    let window = Duration::from_secs(params.expires_after_seconds.unwrap_or(DEFAULT_WINDOW));
    let body = json!({
        "uses": 0,
        "expireTime": iso_time(now + window + TOKEN_EXTRA),
        "newSessionExpireTime": iso_time(now + window),
        "bidiGenerateContentSetup": setup,
    });
    let base = live::websocket_url(&endpoint.base, CONSTRAINED_PATH)?;
    let realtime_base = super::api::without_version(&endpoint.base).to_owned();
    let request = endpoint.request_at(
        &realtime_base,
        "POST",
        &["v1alpha".into(), "auth_tokens".into()],
        Some(body),
        Default::default(),
        Default::default(),
    )?;
    let (_, data) = endpoint
        .json(request, Some(model_id), http::MAX_RESPONSE_BYTES)
        .await?;
    let token = data
        .get("name")
        .and_then(Value::as_str)
        .filter(|t| !t.is_empty())
        .ok_or_else(|| {
            http::response_error(
                &endpoint.provider,
                "the Google realtime token response has no name",
            )
        })?;
    Ok(RealtimeToken {
        token: token.to_owned(),
        url: base,
        expires_at: data
            .get("expireTime")
            .and_then(Value::as_str)
            .and_then(parse_iso_seconds),
    })
}

/// The WebSocket a server opens with a token: the token's URL, the token in
/// the `Authorization` header, never in the URL.
pub(crate) fn connection(token: &RealtimeToken) -> Result<(WebSocketRequest, bool), Error> {
    if token.url.contains("access_token=") {
        return Err(Error::Validation(
            "a server-side realtime connection sends the token in a header; pass the token's URL without access_token".into(),
        ));
    }
    Ok((
        WebSocketRequest {
            url: token.url.clone(),
            headers: vec![("authorization".into(), format!("Token {}", token.token))],
        },
        true,
    ))
}

/// The URL a browser opens with the token, which it can only pass in the
/// query.
#[must_use]
pub fn browser_url(token: &RealtimeToken) -> String {
    match url::Url::parse(&token.url) {
        Ok(mut url) => {
            url.query_pairs_mut()
                .append_pair("access_token", &token.token);
            url.into()
        }
        Err(_) => token.url.clone(),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Boundary {
    None,
    Interrupted,
    TurnComplete,
}

/// The Live API codec of one connection (GGL-024, GGL-025).
pub(crate) struct Codec {
    model_id: String,
    rate: u32,
    turn: u64,
    input_counter: u64,
    input_buffer: String,
    boundary: Boundary,
    preserve_input: bool,
    has_audio: bool,
    has_text: bool,
    has_transcript: bool,
    turn_closed: bool,
}

impl Codec {
    pub(crate) fn new(model_id: &str) -> Self {
        Self {
            model_id: model_id.to_owned(),
            rate: 16000,
            turn: 0,
            input_counter: 0,
            input_buffer: String::new(),
            boundary: Boundary::None,
            preserve_input: false,
            has_audio: false,
            has_text: false,
            has_transcript: false,
            turn_closed: false,
        }
    }

    fn response_id(&self) -> String {
        format!("google-resp-{}", self.turn)
    }

    fn item_id(&self) -> String {
        format!("google-item-{}", self.turn)
    }

    fn begin_turn_if_closed(&mut self) {
        if !self.turn_closed {
            return;
        }
        self.turn += 1;
        self.has_audio = false;
        self.has_text = false;
        self.has_transcript = false;
        self.turn_closed = false;
        self.preserve_input = false;
    }

    fn finish_input(&mut self) {
        if self.input_buffer.is_empty() {
            return;
        }
        self.input_counter += 1;
        self.input_buffer.clear();
    }

    /// Add an input transcription fragment to the running transcript.
    fn input(&mut self, transcription: &Value) -> Option<(String, String)> {
        let text = transcription
            .get("text")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let finished = transcription.get("finished").and_then(Value::as_bool);
        if text.is_empty() {
            if finished == Some(true) {
                self.finish_input();
            }
            return None;
        }
        if !self.input_buffer.is_empty()
            && (self.boundary == Boundary::Interrupted
                || (self.boundary == Boundary::TurnComplete && finished.is_none()))
        {
            self.finish_input();
        }
        self.boundary = Boundary::None;
        self.input_buffer.push_str(text);
        let result = (
            format!("google-input-{}", self.input_counter),
            self.input_buffer.clone(),
        );
        if finished == Some(true) {
            self.finish_input();
        }
        Some(result)
    }

    fn server_content(&mut self, content: &Value, raw: &Value) -> Vec<RealtimeServerEvent> {
        let mut events = Vec::new();
        let event = |kind| RealtimeServerEvent {
            kind,
            raw: raw.clone(),
        };
        let custom = |name: &str| RealtimeServerEvent {
            kind: RealtimeServerEventKind::Custom {
                raw_type: name.into(),
            },
            raw: raw.clone(),
        };
        let flag = |key: &str| content.get(key) == Some(&Value::Bool(true));
        if flag("interrupted") {
            self.boundary = Boundary::Interrupted;
            self.preserve_input = true;
            self.turn_closed = true;
            events.push(event(RealtimeServerEventKind::SpeechStarted {
                item_id: None,
            }));
        }
        if let Some(parts) = content
            .get("modelTurn")
            .and_then(|t| t.get("parts"))
            .and_then(Value::as_array)
        {
            self.begin_turn_if_closed();
            for part in parts {
                if let Some(data) = part
                    .get("inlineData")
                    .and_then(|d| d.get("data"))
                    .and_then(Value::as_str)
                    .filter(|d| !d.is_empty())
                {
                    self.has_audio = true;
                    events.push(event(RealtimeServerEventKind::AudioDelta {
                        response_id: self.response_id(),
                        item_id: self.item_id(),
                        delta: data.to_owned(),
                    }));
                }
                if let Some(text) = part
                    .get("text")
                    .and_then(Value::as_str)
                    .filter(|t| !t.is_empty())
                {
                    self.has_text = true;
                    events.push(event(RealtimeServerEventKind::TextDelta {
                        response_id: self.response_id(),
                        item_id: self.item_id(),
                        delta: text.to_owned(),
                    }));
                }
            }
        }
        if let Some(text) = content
            .get("outputTranscription")
            .and_then(|t| t.get("text"))
            .and_then(Value::as_str)
            .filter(|t| !t.is_empty())
        {
            self.has_transcript = true;
            events.push(event(RealtimeServerEventKind::AudioTranscriptDelta {
                response_id: self.response_id(),
                item_id: self.item_id(),
                delta: text.to_owned(),
            }));
        }
        if let Some(transcription) = content.get("inputTranscription").filter(|t| !t.is_null())
            && let Some((item_id, transcript)) = self.input(transcription)
        {
            events.push(event(
                RealtimeServerEventKind::InputTranscriptionCompleted {
                    item_id,
                    transcript,
                },
            ));
        }
        if flag("generationComplete") {
            events.push(custom("generationComplete"));
        }
        if content
            .get("interactionStatus")
            .is_some_and(|s| !s.is_null())
        {
            events.push(custom("interactionStatus"));
        }
        if flag("waitingForInput") {
            events.push(custom("waitingForInput"));
        }
        if flag("turnComplete") {
            let (response_id, item_id) = (self.response_id(), self.item_id());
            if self.has_audio {
                events.push(event(RealtimeServerEventKind::AudioDone {
                    response_id: response_id.clone(),
                    item_id: item_id.clone(),
                }));
            }
            if self.has_text {
                events.push(event(RealtimeServerEventKind::TextDone {
                    response_id: response_id.clone(),
                    item_id: item_id.clone(),
                    text: None,
                }));
            }
            if self.has_transcript {
                events.push(event(RealtimeServerEventKind::AudioTranscriptDone {
                    response_id: response_id.clone(),
                    item_id,
                    transcript: None,
                }));
            }
            events.push(event(RealtimeServerEventKind::ResponseDone {
                response_id,
                status: "completed".into(),
            }));
            self.turn_closed = true;
            if self.preserve_input {
                self.preserve_input = false;
            } else if self.boundary != Boundary::Interrupted {
                self.boundary = Boundary::TurnComplete;
            }
        }
        if events.is_empty() {
            events.push(custom("serverContent"));
        }
        events
    }
}

/// A function result as the `response` struct Gemini takes: a JSON object
/// as it is, other JSON under `output`, and text that is not JSON under
/// `output`.
fn function_response(output: &str) -> Value {
    match serde_json::from_str::<Value>(output) {
        Ok(Value::Object(object)) => Value::Object(object),
        Ok(other) => json!({"output": other}),
        Err(_) => json!({"output": output}),
    }
}

impl RealtimeCodec for Codec {
    fn encode(&mut self, event: &RealtimeClientEvent) -> Result<Option<String>, Error> {
        let value = match event {
            RealtimeClientEvent::SessionUpdate(config) => {
                if let Some(rate) = config
                    .input_audio_format
                    .as_ref()
                    .and_then(|f| f.sample_rate)
                {
                    self.rate = rate;
                }
                json!({"setup": session_setup(config, &self.model_id)?})
            }
            RealtimeClientEvent::InputAudioAppend(audio) => json!({"realtimeInput": {"audio": {
                "data": audio,
                "mimeType": format!("audio/pcm;rate={}", self.rate),
            }}}),
            RealtimeClientEvent::InputAudioCommit => {
                json!({"realtimeInput": {"audioStreamEnd": true}})
            }
            RealtimeClientEvent::ConversationItemCreate(RealtimeItem::Text(text)) => {
                json!({"realtimeInput": {"text": text}})
            }
            RealtimeClientEvent::ConversationItemCreate(RealtimeItem::FunctionCallOutput {
                call_id,
                name,
                output,
            }) => {
                let mut response = json!({"id": call_id, "response": function_response(output)});
                if let Some(name) = name {
                    response["name"] = json!(name);
                }
                json!({"toolResponse": {"functionResponses": [response]}})
            }
            _ => return Ok(None),
        };
        Ok(Some(value.to_string()))
    }

    fn decode(&mut self, message: &str) -> Result<Vec<RealtimeServerEvent>, Error> {
        let raw: Value = serde_json::from_str(message).map_err(|_| {
            Error::from(ProviderError::new(
                ProviderErrorKind::Response,
                "a Gemini Live message is not valid JSON",
            ))
        })?;
        let event = |kind| RealtimeServerEvent {
            kind,
            raw: raw.clone(),
        };
        let present = |key: &str| raw.get(key).is_some_and(|v| !v.is_null());
        if present("setupComplete") {
            return Ok(vec![event(RealtimeServerEventKind::SessionCreated {
                session_id: None,
            })]);
        }
        if let Some(call) = raw.get("toolCall").filter(|c| !c.is_null()) {
            self.begin_turn_if_closed();
            let mut events = Vec::new();
            for function in call
                .get("functionCalls")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                let args = function
                    .get("args")
                    .filter(|a| !a.is_null())
                    .cloned()
                    .unwrap_or_else(|| json!({}));
                let args = args.to_string();
                let call_id = function
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned();
                events.push(event(RealtimeServerEventKind::FunctionCallArgumentsDelta {
                    response_id: self.response_id(),
                    item_id: self.item_id(),
                    call_id: call_id.clone(),
                    delta: args.clone(),
                }));
                events.push(event(RealtimeServerEventKind::FunctionCallArgumentsDone {
                    response_id: self.response_id(),
                    item_id: self.item_id(),
                    call_id,
                    name: function
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                    arguments: args,
                }));
            }
            return Ok(events);
        }
        for name in ["toolCallCancellation", "goAway", "sessionResumptionUpdate"] {
            if present(name) {
                return Ok(vec![event(RealtimeServerEventKind::Custom {
                    raw_type: name.into(),
                })]);
            }
        }
        if let Some(content) = raw.get("serverContent").filter(|c| !c.is_null()) {
            let content = content.clone();
            return Ok(self.server_content(&content, &raw));
        }
        if let Some(transcription) = raw.get("inputTranscription").filter(|t| !t.is_null()) {
            let transcription = transcription.clone();
            if let Some((item_id, transcript)) = self.input(&transcription) {
                return Ok(vec![event(
                    RealtimeServerEventKind::InputTranscriptionCompleted {
                        item_id,
                        transcript,
                    },
                )]);
            }
        }
        let first = raw
            .as_object()
            .and_then(|object| object.keys().next().cloned())
            .unwrap_or_else(|| "undefined".into());
        Ok(vec![event(RealtimeServerEventKind::Custom {
            raw_type: first,
        })])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn times_round_trip_through_iso_text() {
        let time = UNIX_EPOCH + Duration::from_millis(1_792_740_600_123);
        let text = iso_time(time);
        assert_eq!(text, "2026-10-23T05:30:00.123Z");
        assert_eq!(parse_iso_seconds(&text), Some(1_792_740_600));
        assert_eq!(iso_time(UNIX_EPOCH), "1970-01-01T00:00:00.000Z");
        assert_eq!(
            parse_iso_seconds("2024-02-29T12:00:00Z"),
            Some(1_709_208_000)
        );
    }

    #[test]
    fn thinking_live_models_are_recognized() {
        assert!(is_thinking_live("gemini-3.8-live-extended-thinking"));
        assert!(is_thinking_live("models/gemini-3.8-live-thinking"));
        assert!(!is_thinking_live("gemini-3.8-live"));
        assert!(!is_thinking_live("gemini-3.8-liveextended-thinking"));
        assert!(!is_thinking_live("gemini-3-live-thinking"));
    }
}
