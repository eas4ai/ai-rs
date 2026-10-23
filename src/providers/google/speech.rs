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

//! Gemini text-to-speech on `generateContent` (`google-speech-model.ts`).

use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde::Deserialize;
use serde_json::{Value, json};

use super::{
    api::{Endpoint, Flavor, MEDIA_RESPONSE_BYTES},
    options, response,
};
use crate::{
    Error,
    providers::{http, operations::*},
    types::messages::{FileData, FilePart, generate_id},
};

const DEFAULT_VOICE: &str = "Puck";
const DEFAULT_SAMPLE_RATE: u32 = 24000;

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct SpeechOptions {
    speech_metadata: Option<SpeechMetadata>,
    turns: Option<Vec<Turn>>,
    multi_speaker_voice_config: Option<Value>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct SpeechMetadata {
    speaker: Option<String>,
    style: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Turn {
    text: String,
    speech_metadata: Option<SpeechMetadata>,
}

fn invalid(message: &str) -> Error {
    Error::Validation(message.into())
}

fn warning(feature: &str, details: String) -> Warning {
    Warning {
        kind: "unsupported".into(),
        feature: Some(feature.into()),
        details: Some(details),
        ..Default::default()
    }
}

/// The configured speakers of a multi-speaker config, rejecting custom voices.
fn speakers(config: &Value) -> Result<Vec<String>, Error> {
    let configs = config
        .get("speakerVoiceConfigs")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("multiSpeakerVoiceConfig needs speakerVoiceConfigs"))?;
    let mut speakers = Vec::new();
    for entry in configs {
        let speaker = entry
            .get("speaker")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("each speakerVoiceConfig needs a speaker"))?;
        let voice = entry
            .get("voiceConfig")
            .ok_or_else(|| invalid("each speakerVoiceConfig needs a voiceConfig"))?;
        if voice.get("voice").is_some() {
            return Err(invalid(
                "Custom voices are not supported. Use a prebuilt voice instead.",
            ));
        }
        if !voice
            .get("prebuiltVoiceConfig")
            .and_then(|v| v.get("voiceName"))
            .is_some_and(Value::is_string)
        {
            return Err(invalid(
                "each voiceConfig needs prebuiltVoiceConfig.voiceName",
            ));
        }
        speakers.push(speaker.to_owned());
    }
    Ok(speakers)
}

/// A built speech request: its body, warnings, the wire output format and
/// whether the model takes structured speech.
pub(crate) struct SpeechRequest {
    pub body: Value,
    pub warnings: Vec<Warning>,
    pub format: &'static str,
    pub structured: bool,
}

pub(crate) fn request(
    flavor: Flavor,
    model_id: &str,
    prompt: &AudioPrompt,
    params: &AudioParams,
) -> Result<SpeechRequest, Error> {
    let options: SpeechOptions = options::parse(flavor, Some(&params.provider_options))?;
    let voice = params.voice.as_deref().unwrap_or(DEFAULT_VOICE);
    if voice.starts_with("voice_") || voice.starts_with("voicekey_") {
        return Err(invalid(
            "Custom voices are not supported. Use a prebuilt voice instead.",
        ));
    }
    let speakers = options
        .multi_speaker_voice_config
        .as_ref()
        .map(speakers)
        .transpose()?;
    let structured = !model_id.starts_with("gemini-2.5-") && !model_id.starts_with("gemini-3.1-");
    let mut warnings = Vec::new();
    let speech_config = match &options.multi_speaker_voice_config {
        Some(config) => json!({"multiSpeakerVoiceConfig": config}),
        None => json!({"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}}),
    };
    let text = &prompt.text;
    let mut transcript = text.clone();
    let parts: Vec<Value> = if structured {
        if options.turns.is_some() && options.speech_metadata.is_some() {
            return Err(invalid("Set speechMetadata on each turn when using turns."));
        }
        if options.turns.as_ref().is_some_and(Vec::is_empty) {
            return Err(invalid("google option turns needs at least one turn"));
        }
        if options.turns.is_some() && !text.is_empty() {
            warnings.push(warning(
                "text",
                "Google TTS turns replace the top-level text.".into(),
            ));
        }
        let turns: Vec<(String, Option<SpeechMetadata>)> = match &options.turns {
            Some(turns) => {
                transcript = turns.iter().map(|t| t.text.as_str()).collect();
                turns
                    .iter()
                    .map(|t| (t.text.clone(), t.speech_metadata.clone()))
                    .collect()
            }
            None => vec![(text.clone(), options.speech_metadata.clone())],
        };
        let mut parts = Vec::new();
        for (text, metadata) in turns {
            let metadata = metadata.unwrap_or_default();
            if metadata.speaker.as_deref() == Some("") {
                return Err(invalid("speechMetadata.speaker must not be empty"));
            }
            let style = metadata.style.or_else(|| prompt.instructions.clone());
            let speaker = metadata.speaker;
            if let Some(configured) = &speakers
                && !speaker.as_ref().is_some_and(|s| configured.contains(s))
            {
                return Err(invalid(
                    "Every multi-speaker turn must specify a speechMetadata.speaker matching a configured speaker.",
                ));
            }
            let mut part = json!({"text": text});
            if style.is_some() || speaker.is_some() {
                let mut speech = serde_json::Map::new();
                if let Some(style) = style {
                    speech.insert("style".into(), json!(style));
                }
                if let Some(speaker) = speaker {
                    speech.insert("speaker".into(), json!(speaker));
                }
                part["speechMetadata"] = Value::Object(speech);
            }
            parts.push(part);
        }
        parts
    } else {
        if options.turns.is_some() || options.speech_metadata.is_some() {
            return Err(invalid(
                "Structured speech metadata and turns require Gemini 3.8 TTS.",
            ));
        }
        let mut prompt_text = text.clone();
        if let Some(instructions) = &prompt.instructions {
            if options.multi_speaker_voice_config.is_some() {
                warnings.push(warning(
                    "instructions",
                    "Google Gemini TTS ignores `instructions` when `multiSpeakerVoiceConfig` is set, because prepending them would break multi-speaker transcript parsing.".into(),
                ));
            } else {
                prompt_text = format!("{instructions}: {text}");
            }
        }
        vec![json!({"text": prompt_text})]
    };
    if transcript.is_empty() {
        return Err(invalid("Speech input must contain a non-empty transcript."));
    }
    if params.speed.is_some() {
        warnings.push(warning(
            "speed",
            "Google Gemini TTS models do not support the `speed` option. It was ignored.".into(),
        ));
    }
    if params.language.is_some() {
        warnings.push(warning(
            "language",
            "Google Gemini TTS models do not support the `language` option. Language is detected automatically from the input text.".into(),
        ));
    }
    let formats: &[(&str, &str)] = if structured {
        &[
            ("wav", "AUDIO_WAV"),
            ("audio/wav", "AUDIO_WAV"),
            ("pcm", "AUDIO_L16"),
            ("audio/l16", "AUDIO_L16"),
            ("mulaw", "AUDIO_MULAW"),
            ("audio/mulaw", "AUDIO_MULAW"),
            ("alaw", "AUDIO_ALAW"),
            ("audio/alaw", "AUDIO_ALAW"),
        ]
    } else {
        &[("wav", "AUDIO_WAV"), ("pcm", "AUDIO_L16")]
    };
    let mut format = "AUDIO_WAV";
    if let Some(requested) = &params.output_format {
        match formats.iter().find(|(name, _)| name == requested) {
            Some((_, wire)) => format = wire,
            None => warnings.push(warning(
                "outputFormat",
                format!("Unsupported output format: {requested}. Using wav instead."),
            )),
        }
    }
    let mut generation = json!({"responseModalities": ["AUDIO"], "speechConfig": speech_config});
    if structured && params.output_format.is_some() {
        generation["responseFormat"] = json!({"audio": {"mimeType": format}});
    }
    Ok(SpeechRequest {
        body: json!({"contents": [{"role": "user", "parts": parts}], "generationConfig": generation}),
        warnings,
        format,
        structured,
    })
}

/// Raw 16-bit little-endian mono PCM in a 44-byte WAV container.
pub(crate) fn wav(pcm: &[u8], sample_rate: u32) -> Vec<u8> {
    let size = u32::try_from(pcm.len()).unwrap_or(u32::MAX);
    let mut out = Vec::with_capacity(44 + pcm.len());
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36u32.saturating_add(size)).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&size.to_le_bytes());
    out.extend_from_slice(pcm);
    out
}

/// The sample rate of a MIME type such as `audio/L16;rate=24000`.
fn sample_rate(mime_type: Option<&str>) -> Option<u32> {
    let rest = mime_type?.split("rate=").nth(1)?;
    let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}

/// Generate speech.
pub(crate) async fn generate(
    endpoint: &Endpoint,
    model_id: &str,
    prompt: &AudioPrompt,
    params: &AudioParams,
) -> Result<Item<Vec<FilePart>>, Error> {
    let SpeechRequest {
        body,
        mut warnings,
        format,
        structured,
    } = request(endpoint.flavor, model_id, prompt, params)?;
    let http = endpoint.request(
        "POST",
        &["models".into(), format!("{model_id}:generateContent")],
        Some(body),
        Default::default(),
        Default::default(),
    )?;
    let (_, response) = endpoint
        .json(http, Some(model_id), MEDIA_RESPONSE_BYTES)
        .await?;
    let audio = response
        .get("candidates")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .flat_map(|c| {
            c["content"]["parts"]
                .as_array()
                .cloned()
                .unwrap_or_default()
        })
        .find_map(|part| {
            let data = part["inlineData"]["data"]
                .as_str()
                .filter(|d| !d.is_empty())?;
            Some((
                data.to_owned(),
                part["inlineData"]["mimeType"].as_str().map(str::to_owned),
            ))
        });
    let (data, mime_type) = match audio {
        Some((data, mime)) => (
            STANDARD.decode(data).map_err(|_| {
                http::response_error(
                    &endpoint.provider,
                    "Google speech audio is not valid base64",
                )
            })?,
            mime,
        ),
        None => (Vec::new(), None),
    };
    let rate = sample_rate(mime_type.as_deref()).unwrap_or(DEFAULT_SAMPLE_RATE);
    let pcm = mime_type.as_deref().map_or(!structured, |mime| {
        let lower = mime.to_ascii_lowercase();
        ["audio/l16", "audio/pcm"]
            .iter()
            .any(|prefix| lower == *prefix || lower.starts_with(&format!("{prefix};")))
    });
    let (bytes, media_type) = if format == "AUDIO_WAV" && pcm && !data.is_empty() {
        (wav(&data, rate), "audio/wav".to_owned())
    } else {
        let media_type = mime_type.clone().unwrap_or_else(|| match format {
            "AUDIO_L16" => format!("audio/L16;rate={rate}"),
            "AUDIO_MULAW" => "audio/mulaw".into(),
            "AUDIO_ALAW" => "audio/alaw".into(),
            _ => "audio/wav".into(),
        });
        (data, media_type)
    };
    if format == "AUDIO_L16" && !bytes.is_empty() && !structured {
        warnings.push(warning(
            "outputFormat",
            format!(
                "Returning raw PCM audio (signed 16-bit little-endian, mono, {rate} Hz). These bytes have no container header and are not directly playable; see providerMetadata.google for the sample rate and mime type."
            ),
        ));
    }
    let mut item = Item::new(vec![FilePart {
        id: generate_id("part"),
        data: FileData::Bytes(bytes),
        media_type,
        filename: None,
        provider_metadata: None,
    }]);
    item.warnings = warnings;
    item.usage = response.get("usageMetadata").map(response::convert_usage);
    item.provider_metadata = Some(
        [(
            "google".to_owned(),
            json!({"sampleRate": rate, "mimeType": mime_type}),
        )]
        .into(),
    );
    Ok(item)
}

/// Cloud Text-to-Speech, which serves the Chirp 3: HD voices.
const CLOUD_TTS_BASE_URL: &str = "https://texttospeech.googleapis.com/v1";
const CHIRP3_HD: &str = "Chirp3-HD";

/// The voice name and language code of a Chirp 3: HD voice: a full name is
/// used as it is, with its locale as the language unless one is set; a
/// plain voice name is composed with the language (`Kore` and `en-US` by
/// default).
fn chirp_voice(voice: Option<&str>, language: Option<&str>) -> (String, String) {
    let voice = voice.unwrap_or("Kore");
    if voice.contains(CHIRP3_HD) {
        let prefix = voice
            .split(CHIRP3_HD)
            .next()
            .unwrap_or_default()
            .trim_end_matches('-');
        let language = language
            .or((!prefix.is_empty()).then_some(prefix))
            .unwrap_or("en-US");
        (voice.to_owned(), language.to_owned())
    } else {
        let language = language.unwrap_or("en-US");
        (
            format!("{language}-{CHIRP3_HD}-{voice}"),
            language.to_owned(),
        )
    }
}

/// Speak with a Chirp 3: HD voice on Cloud Text-to-Speech, as LINEAR16 WAV
/// audio (VTX-009, `google-vertex-cloud-tts-speech-model.ts`).
pub(crate) async fn chirp(
    endpoint: &Endpoint,
    model_id: &str,
    prompt: &AudioPrompt,
    params: &AudioParams,
) -> Result<Item<Vec<FilePart>>, Error> {
    let (name, language) = chirp_voice(params.voice.as_deref(), params.language.as_deref());
    let mut warnings = Vec::new();
    if prompt.instructions.is_some() {
        warnings.push(warning(
            "instructions",
            "Google Cloud Text-to-Speech Chirp 3: HD voices do not support the `instructions` option. It was ignored.".into(),
        ));
    }
    if let Some(format) = params.output_format.as_deref().filter(|f| *f != "wav") {
        warnings.push(warning(
            "outputFormat",
            format!("Unsupported output format: {format}. Using wav instead."),
        ));
    }
    let mut audio_config = json!({"audioEncoding": "LINEAR16"});
    if let Some(speed) = params.speed {
        audio_config["speakingRate"] = json!(speed);
    }
    let body = json!({
        "input": {"text": prompt.text},
        "voice": {"languageCode": language, "name": name},
        "audioConfig": audio_config,
    });
    let request = endpoint.request_at(
        CLOUD_TTS_BASE_URL,
        "POST",
        &["text:synthesize".into()],
        Some(body),
        Default::default(),
        Default::default(),
    )?;
    let (_, response) = endpoint
        .json(request, Some(model_id), MEDIA_RESPONSE_BYTES)
        .await?;
    let audio = match response.get("audioContent").and_then(Value::as_str) {
        Some(data) => STANDARD.decode(data).map_err(|_| {
            http::response_error(
                &endpoint.provider,
                "Google Cloud Text-to-Speech audio is not valid base64",
            )
        })?,
        None => Vec::new(),
    };
    let mut item = Item::new(vec![FilePart {
        id: generate_id("part"),
        data: FileData::Bytes(audio),
        media_type: "audio/wav".into(),
        filename: None,
        provider_metadata: None,
    }]);
    item.warnings = warnings;
    item.provider_metadata = Some([("google".to_owned(), json!({"mimeType": "audio/wav"}))].into());
    Ok(item)
}

#[cfg(test)]
mod chirp_tests {
    use super::chirp_voice;

    #[test]
    fn voices_are_composed_as_typescript_composes_them() {
        let pair = |a: &str, b: &str| (a.to_owned(), b.to_owned());
        assert_eq!(
            chirp_voice(None, None),
            pair("en-US-Chirp3-HD-Kore", "en-US")
        );
        assert_eq!(
            chirp_voice(Some("Puck"), Some("de-DE")),
            pair("de-DE-Chirp3-HD-Puck", "de-DE")
        );
        assert_eq!(
            chirp_voice(Some("fr-FR-Chirp3-HD-Aoede"), None),
            pair("fr-FR-Chirp3-HD-Aoede", "fr-FR")
        );
        assert_eq!(
            chirp_voice(Some("Chirp3-HD-Kore"), None),
            pair("Chirp3-HD-Kore", "en-US")
        );
        assert_eq!(
            chirp_voice(Some("fr-FR-Chirp3-HD-Aoede"), Some("fr-CA")),
            pair("fr-FR-Chirp3-HD-Aoede", "fr-CA")
        );
    }
}
