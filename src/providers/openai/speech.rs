//! Speech with OpenAI (`speech/openai-speech-model.ts`), OAI-018.

use serde::Deserialize;
use serde_json::json;

use super::{errors, params::invalid, protocol::http_request};
use crate::{
    Error, Model, Provider,
    models::files::DEFAULT_MAX_BYTES,
    providers::{
        http,
        operations::{AudioParams, AudioPrompt, Item},
    },
    types::messages::FilePart,
};

/// The output formats OpenAI takes, with their media types.
const FORMATS: &[(&str, &str)] = &[
    ("mp3", "audio/mpeg"),
    ("opus", "audio/opus"),
    ("aac", "audio/aac"),
    ("flac", "audio/flac"),
    ("wav", "audio/wav"),
    ("pcm", "audio/pcm"),
];

/// The speech options (`openaiSpeechModelOptionsSchema`).
#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct SpeechOptions {
    instructions: Option<String>,
    speed: Option<f64>,
}

pub(super) async fn generate(
    model: &Model,
    prompt: AudioPrompt,
    params: AudioParams,
    provider: &dyn Provider,
) -> Result<Item<Vec<FilePart>>, Error> {
    let options: SpeechOptions = match params.provider_options.get("openai") {
        None => SpeechOptions::default(),
        Some(value) => serde_json::from_value(value.clone())
            .map_err(|e| invalid(format!("invalid openai speech options: {e}")))?,
    };
    if options.speed.is_some_and(|s| !(0.25..=4.0).contains(&s)) {
        return Err(invalid("openai speech option speed must be from 0.25 to 4"));
    }
    if params.language.is_some() {
        return Err(invalid("OpenAI speech models do not take a language"));
    }
    let format = params.output_format.as_deref().unwrap_or("mp3");
    let media_type = FORMATS
        .iter()
        .find(|(name, _)| *name == format)
        .map(|(_, media_type)| *media_type)
        .ok_or_else(|| {
            invalid(format!(
                "OpenAI speech takes the output formats mp3, opus, aac, flac, wav and pcm, not {format:?}"
            ))
        })?;
    let mut body = json!({
        "model": model.id,
        "input": prompt.text,
        "voice": params.voice.as_deref().unwrap_or("alloy"),
        "response_format": format,
    });
    if let Some(speed) = options.speed.or(params.speed) {
        body["speed"] = json!(speed);
    }
    if let Some(instructions) = options.instructions.or(prompt.instructions) {
        body["instructions"] = json!(instructions);
    }
    let mut request = http_request(
        &errors::API,
        "POST",
        provider.config(),
        &["audio", "speech"],
        Some(body),
        None,
    )?;
    request
        .headers
        .retain(|(name, _)| !name.eq_ignore_ascii_case("accept"));
    request.headers.push(("accept".into(), "*/*".into()));
    let transport = provider.transport().ok_or_else(|| {
        Error::Configuration(format!(
            "provider {:?} has no transport for speech",
            provider.name()
        ))
    })?;
    let response = http::send(
        &errors::API,
        transport,
        request,
        provider.name(),
        Some(&model.id),
    )
    .await?;
    let (_, bytes) =
        http::read_body_with_limit(&errors::API, provider.name(), response, DEFAULT_MAX_BYTES)
            .await?;
    let audio = FilePart::from_bytes(bytes.to_vec(), Some(media_type), None)
        .map_err(|e| http::response_error(provider.name(), e.to_string()))?;
    Ok(Item::new(vec![audio]))
}
