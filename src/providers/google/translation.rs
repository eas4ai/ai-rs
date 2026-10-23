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

//! Gemini Live speech translation
//! (`speech-translation/google-speech-translation-model.ts`). Live
//! translation never sends `turnComplete` and keeps sending silence after the
//! input ends, so a session finishes after a second of silent output audio.

use std::{sync::Arc, time::Duration};

use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde::Deserialize;
use serde_json::{Value, json};

use super::{
    api::{Endpoint, Flavor},
    live::{self, Control, Handler, Outcome, Session},
    options,
    transcription::{check_pcm16k, live_model, live_request},
};
use crate::{Error, providers::operations::*, transport::WebSocketTransport};

const FINISH_GRACE: Duration = Duration::from_secs(1);
/// The Live API's output audio rate.
const OUTPUT_RATE: f64 = 24000.0;
/// A 16-bit sample within this distance of zero is silence.
const SILENCE: i16 = 128;

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct TranslationOptions {
    echo_target_language: Option<bool>,
}

fn warning(feature: &str, details: &str) -> Warning {
    Warning {
        kind: "unsupported".into(),
        feature: Some(feature.into()),
        details: Some(details.into()),
        ..Default::default()
    }
}

/// Translate streaming audio (GGL-018).
pub(crate) fn stream(
    endpoint: &Endpoint,
    model_id: &str,
    transport: Arc<dyn WebSocketTransport>,
    audio: AudioChunks,
    format: &AudioFormat,
    params: &StreamTranslateParams,
) -> Result<TranslationStream, Error> {
    if params.target_language.is_empty() {
        return Err(Error::Validation(format!(
            "targetLanguage is required for translation model '{model_id}'."
        )));
    }
    let options: TranslationOptions =
        options::parse(Flavor::Gemini, Some(&params.provider_options))?;
    check_pcm16k(format, "translation")?;
    let mut warnings = Vec::new();
    if params.source_language.is_some() {
        warnings.push(warning(
            "sourceLanguage",
            "The Gemini Live translation API auto-detects the source language and does not accept a source language.",
        ));
    }
    if params.output_format.is_some() {
        warnings.push(warning(
            "outputAudioFormat",
            "The Gemini Live API always outputs 24kHz 16-bit PCM audio and does not accept an output audio format.",
        ));
    }
    let mut translation = json!({"targetLanguageCode": params.target_language});
    if let Some(echo) = options.echo_target_language {
        let _ = echo;
    }
    let setup = json!({
        "model": live_model(endpoint, model_id)?,
        "generationConfig": {"responseModalities": ["AUDIO"], "translationConfig": translation},
        "inputAudioTranscription": {},
        "outputAudioTranscription": {},
    });
    let session = Session {
        request: live_request(endpoint)?,
        setup,
        rate: format.sample_rate.unwrap_or(16000),
        include_raw: params.include_raw,
        provider: endpoint.provider.clone(),
        socket_error: "Google Live translation error",
    };
    let handler = Translator {
        provider: endpoint.provider.clone(),
        warnings,
        turn: 0,
        source: String::new(),
        source_turn: String::new(),
        output: String::new(),
        output_turn: String::new(),
        usage: None,
        open_turn: false,
        saw_turn_complete: false,
        trailing_silence: Duration::ZERO,
    };
    Ok(live::run(transport, session, audio, handler))
}

struct Translator {
    provider: String,
    warnings: Vec<Warning>,
    turn: u64,
    source: String,
    source_turn: String,
    output: String,
    output_turn: String,
    usage: Option<TranslationUsage>,
    open_turn: bool,
    saw_turn_complete: bool,
    trailing_silence: Duration,
}

impl Translator {
    fn id(&self) -> Option<String> {
        Some(format!("google-item-{}", self.turn))
    }

    fn activity(&mut self, control: &mut Control) {
        self.open_turn = true;
        self.trailing_silence = Duration::ZERO;
        control.cancel();
    }

    fn schedule(control: &mut Control) {
        if control.finish_at.is_none() {
            control.schedule(FINISH_GRACE);
        }
    }

    fn complete_turn(&mut self, parts: &mut Vec<TranslationPart>) {
        if !self.source_turn.is_empty() {
            let text = std::mem::take(&mut self.source_turn);
            parts.push(TranslationPart::SourceTranscriptFinal {
                id: self.id(),
                text: text.clone(),
            });
            self.source.push_str(&text);
        }
        if !self.output_turn.is_empty() {
            let text = std::mem::take(&mut self.output_turn);
            parts.push(TranslationPart::OutputTextFinal {
                id: self.id(),
                text: text.clone(),
            });
            self.output.push_str(&text);
        }
        self.turn += 1;
    }

    fn finish(&mut self) -> Vec<TranslationPart> {
        let mut parts = Vec::new();
        if !self.source_turn.is_empty() || !self.output_turn.is_empty() {
            self.complete_turn(&mut parts);
        }
        parts.push(TranslationPart::Finish {
            source_text: self.source.clone(),
            output_text: self.output.clone(),
            duration_seconds: None,
            usage: self.usage.clone(),
            provider_metadata: None,
        });
        parts
    }

    /// Add the `AUDIO` token counts of a usage update; text counts are the
    /// translation's internal context.
    fn add_usage(&mut self, usage: &Value) {
        let audio = |key: &str| -> Option<u64> {
            let details = usage.get(key)?.as_array()?;
            let mut total = None;
            for detail in details {
                if detail.get("modality").and_then(Value::as_str) == Some("AUDIO")
                    && let Some(count) = detail.get("tokenCount").and_then(Value::as_u64)
                {
                    total = Some(total.unwrap_or(0) + count);
                }
            }
            total
        };
        let input = audio("promptTokensDetails");
        let output = audio("responseTokensDetails");
        if input.is_none() && output.is_none() {
            return;
        }
        let current = self.usage.get_or_insert_with(TranslationUsage::default);
        if let Some(input) = input {
            current.input_audio_tokens = Some(current.input_audio_tokens.unwrap_or(0) + input);
        }
        if let Some(output) = output {
            current.output_audio_tokens = Some(current.output_audio_tokens.unwrap_or(0) + output);
        }
    }
}

/// How long silent 16-bit PCM audio lasts, or `None` when a sample is loud
/// or the data is not audio.
pub(crate) fn silence(audio: &[u8]) -> Option<Duration> {
    if audio.len() < 2 {
        return None;
    }
    let samples = audio.len() / 2;
    for pair in audio.chunks_exact(2) {
        let sample = i16::from_le_bytes([pair[0], pair[1]]);
        if sample.unsigned_abs() > SILENCE.unsigned_abs() {
            return None;
        }
    }
    Some(Duration::from_secs_f64(samples as f64 / OUTPUT_RATE))
}

impl Handler for Translator {
    type Part = TranslationPart;

    fn start(&mut self) -> TranslationPart {
        TranslationPart::Start {
            warnings: std::mem::take(&mut self.warnings),
        }
    }

    fn raw(&self, message: Value) -> TranslationPart {
        TranslationPart::Raw(message)
    }

    fn message(
        &mut self,
        message: &Value,
        control: &mut Control,
    ) -> Result<Outcome<TranslationPart>, Error> {
        if let Some(usage) = message.get("usageMetadata").filter(|u| !u.is_null()) {
            self.add_usage(usage);
        }
        if message.get("error").is_some_and(|e| !e.is_null()) {
            return Err(live::server_error(
                &self.provider,
                message,
                "Google Live API error",
            ));
        }
        let mut parts = Vec::new();
        let content = message.get("serverContent").filter(|c| !c.is_null());
        let input = content
            .and_then(|c| c.get("inputTranscription"))
            .and_then(|t| t.get("text"))
            .and_then(Value::as_str)
            .or_else(|| {
                message
                    .get("inputTranscription")
                    .and_then(|t| t.get("text"))
                    .and_then(Value::as_str)
            })
            .filter(|t| !t.is_empty());
        if let Some(text) = input {
            self.activity(control);
            self.source_turn.push_str(text);
            parts.push(TranslationPart::SourceTranscriptDelta {
                id: self.id(),
                delta: text.to_owned(),
            });
        }
        let Some(content) = content else {
            return Ok(Outcome::Continue(parts));
        };
        let audio_parts = content
            .get("modelTurn")
            .and_then(|t| t.get("parts"))
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        for part in audio_parts {
            let Some(data) = part
                .get("inlineData")
                .and_then(|d| d.get("data"))
                .and_then(Value::as_str)
                .filter(|d| !d.is_empty())
            else {
                continue;
            };
            let bytes = STANDARD.decode(data).unwrap_or_default();
            let quiet = silence(&bytes);
            parts.push(TranslationPart::Audio {
                id: self.id(),
                audio: bytes,
            });
            match quiet {
                Some(duration) if control.audio_ended => {
                    self.trailing_silence += duration;
                    if self.trailing_silence >= FINISH_GRACE {
                        parts.extend(self.finish());
                        return Ok(Outcome::Finish(parts));
                    }
                }
                _ => self.activity(control),
            }
        }
        if let Some(text) = content
            .get("outputTranscription")
            .and_then(|t| t.get("text"))
            .and_then(Value::as_str)
            .filter(|t| !t.is_empty())
        {
            self.activity(control);
            self.output_turn.push_str(text);
            parts.push(TranslationPart::OutputTextDelta {
                id: self.id(),
                delta: text.to_owned(),
            });
        }
        if content.get("turnComplete") == Some(&Value::Bool(true)) {
            self.complete_turn(&mut parts);
            self.open_turn = false;
            self.saw_turn_complete = true;
            if control.audio_ended {
                Self::schedule(control);
            }
        }
        Ok(Outcome::Continue(parts))
    }

    fn audio_ended(&mut self, control: &mut Control) -> Outcome<TranslationPart> {
        if self.saw_turn_complete && !self.open_turn {
            Self::schedule(control);
        }
        Outcome::Continue(Vec::new())
    }

    fn timer(&mut self) -> Vec<TranslationPart> {
        self.finish()
    }

    fn closed(
        &mut self,
        code: Option<u16>,
        reason: &str,
        control: &Control,
    ) -> Result<Vec<TranslationPart>, Error> {
        if control.finish_at.is_some() {
            Ok(self.finish())
        } else {
            Err(live::closed_error(
                &self.provider,
                "Google Live translation",
                code,
                reason,
            ))
        }
    }
}
