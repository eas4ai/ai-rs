//! OpenAI's realtime WebSockets for streaming transcription and speech
//! translation (`transcription/openai-transcription-model.ts` and
//! `speech-translation/openai-speech-translation-model.ts`), OAI-020.

use std::{collections::VecDeque, sync::Arc};

use base64::{Engine as _, engine::general_purpose::STANDARD};
use futures::{StreamExt, stream::BoxStream};
use serde_json::{Value, json};

use super::{
    errors,
    params::invalid,
    transcription::{TranscriptionOptions, realtime_model},
};
use crate::{
    Error, Model, Provider,
    errors::{ProviderError, ProviderErrorKind},
    providers::{
        http,
        operations::{
            AudioChunks, AudioFormat, StreamTranscribeParams, StreamTranslateParams,
            TranscribeParams, TranscriptionPart, TranscriptionStream, TranslationPart,
            TranslationStream,
        },
    },
    transport::{WebSocket, WebSocketMessage, WebSocketRequest, WebSocketTransport},
};

/// What one kind of session reads from the server.
trait Handler: Send + 'static {
    type Part: Send + 'static;
    fn start(&mut self) -> Self::Part;
    fn raw(&self, message: Value) -> Self::Part;
    /// A server message: the parts it gives, and whether the session is done.
    fn message(&mut self, message: &Value) -> (Vec<Result<Self::Part, Error>>, bool);
    /// The socket closed before the session was done.
    fn closed(&mut self, code: Option<u16>, reason: &str) -> Vec<Result<Self::Part, Error>>;
}

/// A session to run: the socket to open, its first message, and how audio
/// goes to it.
struct Session {
    request: WebSocketRequest,
    update: Value,
    /// The type of the message that carries a chunk of audio.
    append: &'static str,
    /// The message sent after the last chunk.
    end: Value,
    include_raw: bool,
    provider: String,
    /// The error message of a failed socket.
    socket_error: &'static str,
}

fn connection_error(
    provider: &str,
    message: &str,
    error: Option<crate::transport::TransportError>,
) -> Error {
    let mut error_value =
        ProviderError::new(ProviderErrorKind::Connection, message).with_provider(provider);
    if let Some(error) = error {
        error_value = error_value.with_source(error);
    }
    error_value.into()
}

/// A server `error` event as a provider error.
fn server_error(provider: &str, message: &Value) -> Error {
    let text = message["error"]["message"]
        .as_str()
        .unwrap_or("OpenAI realtime error");
    let mut error = ProviderError::new(ProviderErrorKind::Response, text).with_provider(provider);
    error.code = message["error"]["code"].as_str().map(str::to_owned);
    error.error_type = message["error"]["type"].as_str().map(str::to_owned);
    error.body = Some(message.clone());
    error.into()
}

enum Phase {
    Connecting,
    Open(WebSocket),
    Done,
}

struct State<H: Handler> {
    transport: Arc<dyn WebSocketTransport>,
    session: Session,
    audio: AudioChunks,
    audio_ended: bool,
    handler: H,
    phase: Phase,
    queue: VecDeque<Result<H::Part, Error>>,
}

enum Step {
    Message(Option<Result<WebSocketMessage, crate::transport::TransportError>>),
    Audio(Option<Vec<u8>>),
}

fn run<H: Handler>(
    transport: Arc<dyn WebSocketTransport>,
    session: Session,
    audio: AudioChunks,
    handler: H,
) -> BoxStream<'static, Result<H::Part, Error>> {
    let state = State {
        transport,
        session,
        audio,
        audio_ended: false,
        handler,
        phase: Phase::Connecting,
        queue: VecDeque::new(),
    };
    futures::stream::unfold(state, |mut state| async move {
        loop {
            if let Some(item) = state.queue.pop_front() {
                return Some((item, state));
            }
            if matches!(state.phase, Phase::Done) {
                return None;
            }
            state.step().await;
        }
    })
    .boxed()
}

impl<H: Handler> State<H> {
    fn fail(&mut self, error: Error) {
        self.queue.push_back(Err(error));
        self.phase = Phase::Done;
    }

    async fn send(&mut self, value: Value) -> bool {
        let Phase::Open(socket) = &mut self.phase else {
            return false;
        };
        match socket
            .sender
            .send(WebSocketMessage::Text(value.to_string()))
            .await
        {
            Ok(()) => true,
            Err(error) => {
                let provider = self.session.provider.clone();
                self.fail(connection_error(
                    &provider,
                    self.session.socket_error,
                    Some(error),
                ));
                false
            }
        }
    }

    async fn finish(&mut self) {
        if let Phase::Open(socket) = &mut self.phase {
            // Best effort: the session is over either way.
            let _ = socket
                .sender
                .send(WebSocketMessage::Close {
                    code: Some(1000),
                    reason: String::new(),
                })
                .await;
        }
        self.phase = Phase::Done;
    }

    async fn step(&mut self) {
        if matches!(self.phase, Phase::Connecting) {
            match self.transport.connect(self.session.request.clone()).await {
                Ok(socket) => {
                    self.phase = Phase::Open(socket);
                    let start = self.handler.start();
                    self.queue.push_back(Ok(start));
                    let update = self.session.update.clone();
                    self.send(update).await;
                }
                Err(error) => {
                    let provider = self.session.provider.clone();
                    self.fail(connection_error(
                        &provider,
                        self.session.socket_error,
                        Some(error),
                    ));
                }
            }
            return;
        }
        let read_audio = !self.audio_ended;
        let step = {
            let Phase::Open(socket) = &mut self.phase else {
                return;
            };
            let audio = &mut self.audio;
            tokio::select! {
                message = socket.receiver.next() => Step::Message(message),
                chunk = audio.next(), if read_audio => Step::Audio(chunk),
            }
        };
        match step {
            Step::Audio(Some(chunk)) => {
                let message =
                    json!({"type": self.session.append, "audio": STANDARD.encode(&chunk)});
                self.send(message).await;
            }
            Step::Audio(None) => {
                let end = self.session.end.clone();
                if self.send(end).await {
                    self.audio_ended = true;
                }
            }
            Step::Message(Some(Ok(WebSocketMessage::Text(text)))) => self.message(&text).await,
            Step::Message(Some(Ok(WebSocketMessage::Binary(data)))) => {
                if let Ok(text) = String::from_utf8(data.to_vec()) {
                    self.message(&text).await;
                }
            }
            Step::Message(Some(Ok(WebSocketMessage::Close { code, reason }))) => {
                let parts = self.handler.closed(code, &reason);
                self.queue.extend(parts);
                self.phase = Phase::Done;
            }
            Step::Message(None) => {
                let parts = self.handler.closed(None, "");
                self.queue.extend(parts);
                self.phase = Phase::Done;
            }
            Step::Message(Some(Err(error))) => {
                let provider = self.session.provider.clone();
                self.fail(connection_error(
                    &provider,
                    self.session.socket_error,
                    Some(error),
                ));
            }
        }
    }

    async fn message(&mut self, text: &str) {
        let Ok(value) = serde_json::from_str::<Value>(text) else {
            return;
        };
        if self.session.include_raw {
            let raw = self.handler.raw(value.clone());
            self.queue.push_back(Ok(raw));
        }
        let (parts, done) = self.handler.message(&value);
        let failed = parts.iter().any(Result::is_err) && done;
        self.queue.extend(parts);
        if failed {
            self.phase = Phase::Done;
        } else if done {
            self.finish().await;
        }
    }
}

/// The socket request for `path` and `query` under the provider's base
/// URL, as `ws://` or `wss://`, with the provider's headers and key.
fn request(
    provider: &dyn Provider,
    path: &[&str],
    query: &[(&str, &str)],
) -> Result<WebSocketRequest, Error> {
    let query = query
        .iter()
        .map(|(key, value)| ((*key).to_owned(), json!(value)))
        .collect();
    let request = http::request(
        &errors::API,
        provider.config(),
        "GET",
        path,
        None,
        Default::default(),
        query,
    )?;
    let url = match request.url.split_once("://") {
        Some(("https", rest)) => format!("wss://{rest}"),
        Some(("http", rest)) => format!("ws://{rest}"),
        _ => request.url.clone(),
    };
    let headers = request
        .headers
        .into_iter()
        .filter(|(name, _)| !name.eq_ignore_ascii_case("accept"))
        .collect();
    Ok(WebSocketRequest { url, headers })
}

fn websocket_of(provider: &dyn Provider) -> Result<Arc<dyn WebSocketTransport>, Error> {
    provider.websocket().ok_or_else(|| {
        Error::Configuration(format!(
            "provider {:?} has no WebSocket transport for streaming audio; build it with with_websocket_transport",
            provider.name()
        ))
    })
}

struct Transcribe {
    language: Option<String>,
    provider: String,
}

impl Handler for Transcribe {
    type Part = TranscriptionPart;
    fn start(&mut self) -> TranscriptionPart {
        TranscriptionPart::Start {
            warnings: Vec::new(),
        }
    }
    fn raw(&self, message: Value) -> TranscriptionPart {
        TranscriptionPart::Raw(message)
    }
    fn message(&mut self, message: &Value) -> (Vec<Result<TranscriptionPart, Error>>, bool) {
        let id = message["item_id"].as_str().map(str::to_owned);
        match message["type"].as_str().unwrap_or_default() {
            "conversation.item.input_audio_transcription.delta" => (
                vec![Ok(TranscriptionPart::Delta {
                    id,
                    delta: message["delta"].as_str().unwrap_or_default().to_owned(),
                })],
                false,
            ),
            "conversation.item.input_audio_transcription.completed" => {
                let text = message["transcript"]
                    .as_str()
                    .unwrap_or_default()
                    .to_owned();
                (
                    vec![
                        Ok(TranscriptionPart::Final {
                            id,
                            text: text.clone(),
                        }),
                        Ok(TranscriptionPart::Finish {
                            text,
                            segments: Vec::new(),
                            language: self.language.clone(),
                            duration_seconds: None,
                            usage: None,
                            provider_metadata: None,
                        }),
                    ],
                    true,
                )
            }
            "error" => (vec![Err(server_error(&self.provider, message))], true),
            _ => (Vec::new(), false),
        }
    }
    fn closed(
        &mut self,
        _code: Option<u16>,
        _reason: &str,
    ) -> Vec<Result<TranscriptionPart, Error>> {
        Vec::new()
    }
}

pub(super) async fn stream_transcription(
    model: &Model,
    audio: AudioChunks,
    format: AudioFormat,
    params: StreamTranscribeParams,
    provider: &dyn Provider,
) -> Result<TranscriptionStream, Error> {
    if !realtime_model(&model.id) {
        return Err(invalid(format!(
            "{} does not stream transcription; OpenAI streams it with gpt-realtime-whisper",
            model.id
        )));
    }
    let options = TranscriptionOptions::parse(&TranscribeParams {
        provider_options: params.provider_options.clone(),
    })?;
    let file_only = [
        ("include", options.include.is_some()),
        ("prompt", options.prompt.is_some()),
        ("temperature", options.temperature.is_some()),
        (
            "timestampGranularities",
            options.timestamp_granularities.is_some(),
        ),
    ];
    if let Some((name, _)) = file_only.iter().find(|(_, set)| *set) {
        return Err(invalid(format!(
            "OpenAI streaming transcription does not take the option {name}"
        )));
    }
    let mut input_format = json!({"type": format.media_type});
    if let Some(rate) = format.sample_rate {
        input_format["rate"] = json!(rate);
    }
    let mut transcription = json!({"model": model.id});
    if let Some(language) = &options.language {
        transcription["language"] = json!(language);
    }
    let streaming = options.streaming.as_ref();
    if let Some(delay) = streaming.and_then(|s| s.delay.as_deref()) {
        transcription["delay"] = json!(delay);
    }
    let mut session = json!({
        "type": "transcription",
        "audio": {"input": {"format": input_format, "transcription": transcription, "turn_detection": null}},
    });
    if let Some(include) = streaming.and_then(|s| s.include.as_ref()) {
        session["include"] = json!(include);
    }
    let session = Session {
        request: request(provider, &["realtime"], &[("intent", "transcription")])?,
        update: json!({"type": "session.update", "session": session}),
        append: "input_audio_buffer.append",
        end: json!({"type": "input_audio_buffer.commit"}),
        include_raw: params.include_raw,
        provider: provider.name().to_owned(),
        socket_error: "OpenAI realtime transcription error",
    };
    let transport = websocket_of(provider)?;
    Ok(run(
        transport,
        session,
        audio,
        Transcribe {
            language: options.language,
            provider: provider.name().to_owned(),
        },
    ))
}

#[derive(Default)]
struct Translate {
    source: String,
    output: String,
    provider: String,
}

impl Handler for Translate {
    type Part = TranslationPart;
    fn start(&mut self) -> TranslationPart {
        TranslationPart::Start {
            warnings: Vec::new(),
        }
    }
    fn raw(&self, message: Value) -> TranslationPart {
        TranslationPart::Raw(message)
    }
    fn message(&mut self, message: &Value) -> (Vec<Result<TranslationPart, Error>>, bool) {
        let delta = message["delta"].as_str().unwrap_or_default();
        match message["type"].as_str().unwrap_or_default() {
            "session.output_audio.delta" if !delta.is_empty() => match STANDARD.decode(delta) {
                Ok(audio) => (vec![Ok(TranslationPart::Audio { id: None, audio })], false),
                Err(_) => (
                    vec![Err(http::response_error(
                        &self.provider,
                        "OpenAI translation audio is not base64",
                    ))],
                    true,
                ),
            },
            "session.output_transcript.delta" => {
                self.output.push_str(delta);
                (
                    vec![Ok(TranslationPart::OutputTextDelta {
                        id: None,
                        delta: delta.to_owned(),
                    })],
                    false,
                )
            }
            "session.input_transcript.delta" => {
                self.source.push_str(delta);
                (
                    vec![Ok(TranslationPart::SourceTranscriptDelta {
                        id: None,
                        delta: delta.to_owned(),
                    })],
                    false,
                )
            }
            "session.closed" => {
                let mut parts = Vec::new();
                if !self.source.is_empty() {
                    parts.push(Ok(TranslationPart::SourceTranscriptFinal {
                        id: None,
                        text: self.source.clone(),
                    }));
                }
                if !self.output.is_empty() {
                    parts.push(Ok(TranslationPart::OutputTextFinal {
                        id: None,
                        text: self.output.clone(),
                    }));
                }
                parts.push(Ok(TranslationPart::Finish {
                    source_text: std::mem::take(&mut self.source),
                    output_text: std::mem::take(&mut self.output),
                    duration_seconds: None,
                    usage: None,
                    provider_metadata: None,
                }));
                (parts, true)
            }
            // An error event does not end a translation session.
            "error" => (vec![Err(server_error(&self.provider, message))], false),
            _ => (Vec::new(), false),
        }
    }
    fn closed(&mut self, code: Option<u16>, reason: &str) -> Vec<Result<TranslationPart, Error>> {
        let code = code.map_or_else(|| "unknown".to_owned(), |c| c.to_string());
        let reason = if reason.is_empty() {
            String::new()
        } else {
            format!(", reason: {reason}")
        };
        vec![Err(connection_error(
            &self.provider,
            &format!(
                "OpenAI realtime translation WebSocket closed before finishing (code {code}{reason})"
            ),
            None,
        ))]
    }
}

pub(super) async fn stream_translation(
    model: &Model,
    audio: AudioChunks,
    format: AudioFormat,
    params: StreamTranslateParams,
    provider: &dyn Provider,
) -> Result<TranslationStream, Error> {
    if let Some(value) = params.provider_options.get("openai") {
        let empty = value.as_object().is_some_and(serde_json::Map::is_empty);
        if !empty {
            return Err(invalid("OpenAI speech translation takes no openai options"));
        }
    }
    if format.media_type != "audio/pcm" || format.sample_rate.is_some_and(|r| r != 24_000) {
        return Err(invalid(
            "the OpenAI realtime translation API takes only 24 kHz 16-bit PCM input audio",
        ));
    }
    if params.source_language.is_some() {
        return Err(invalid(
            "the OpenAI realtime translation API detects the source language and takes none",
        ));
    }
    if params.output_format.is_some() {
        return Err(invalid(
            "the OpenAI realtime translation API always outputs 24 kHz 16-bit PCM and takes no output format",
        ));
    }
    if params.target_language.is_empty() {
        return Err(invalid("OpenAI speech translation needs a target language"));
    }
    let session = Session {
        request: request(
            provider,
            &["realtime", "translations"],
            &[("model", &model.id)],
        )?,
        update: json!({"type": "session.update", "session": {"audio": {
            "input": {"transcription": {"model": "gpt-realtime-whisper"}, "noise_reduction": null},
            "output": {"language": params.target_language},
        }}}),
        append: "session.input_audio_buffer.append",
        end: json!({"type": "session.close"}),
        include_raw: params.include_raw,
        provider: provider.name().to_owned(),
        socket_error: "OpenAI realtime translation error",
    };
    let transport = websocket_of(provider)?;
    Ok(run(
        transport,
        session,
        audio,
        Translate {
            provider: provider.name().to_owned(),
            ..Default::default()
        },
    ))
}
