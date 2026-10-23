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

//! A Live API session over a WebSocket, shared by live transcription and
//! speech translation: open the socket, send the setup, wait for
//! `setupComplete`, send the audio and then `audioStreamEnd`, and hand each
//! server message to the operation's handler until it finishes.
//!
//! Dropping the stream drops the socket and the audio stream, so the
//! connection closes and no more audio is read.

use std::{collections::VecDeque, sync::Arc, time::Duration};

use base64::{Engine as _, engine::general_purpose::STANDARD};
use futures::{StreamExt, stream::BoxStream};
use serde_json::{Value, json};
use tokio::time::Instant;

use crate::{
    Error,
    errors::{ProviderError, ProviderErrorKind},
    providers::operations::AudioChunks,
    transport::{WebSocket, WebSocketMessage, WebSocketRequest, WebSocketTransport},
};

/// The `wss://` (or `ws://`) URL of a Live API path, on the base URL's host
/// and path without its trailing `v1beta` or `v1alpha`.
pub(crate) fn websocket_url(base: &str, path: &str) -> Result<String, Error> {
    let mut url = url::Url::parse(base)
        .map_err(|_| Error::Configuration("the Google base URL is invalid".into()))?;
    let mut segments: Vec<String> = url
        .path_segments()
        .map(|s| s.map(str::to_owned).collect())
        .unwrap_or_default();
    while segments.last().is_some_and(String::is_empty) {
        segments.pop();
    }
    if segments
        .last()
        .is_some_and(|s| s == "v1beta" || s == "v1alpha")
    {
        segments.pop();
    }
    let scheme = if url.scheme() == "https" { "wss" } else { "ws" };
    url.set_scheme(scheme).map_err(|_| {
        Error::Configuration("the Google base URL cannot become a WebSocket URL".into())
    })?;
    let mut path_text = segments.join("/");
    if !path_text.is_empty() {
        path_text.insert(0, '/');
    }
    url.set_path(&format!("{path_text}/ws/{path}"));
    url.set_query(None);
    Ok(url.into())
}

/// What a session's handler can do besides returning parts.
pub(crate) struct Control {
    /// When the session finishes unless something reschedules it.
    pub finish_at: Option<Instant>,
    /// The audio has ended and `audioStreamEnd` was sent.
    pub audio_ended: bool,
}

impl Control {
    pub(crate) fn schedule(&mut self, after: Duration) {
        self.finish_at = Some(Instant::now() + after);
    }
    pub(crate) fn cancel(&mut self) {
        self.finish_at = None;
    }
}

/// What a handler returns for an input: parts to deliver, and whether the
/// session is over.
pub(crate) enum Outcome<P> {
    Continue(Vec<P>),
    Finish(Vec<P>),
}

/// An operation's view of a Live session.
pub(crate) trait Handler: Send + 'static {
    type Part: Send + 'static;
    /// The first part, once the socket is open.
    fn start(&mut self) -> Self::Part;
    fn raw(&self, message: Value) -> Self::Part;
    /// A server message.
    fn message(
        &mut self,
        message: &Value,
        control: &mut Control,
    ) -> Result<Outcome<Self::Part>, Error>;
    /// The audio ended and `audioStreamEnd` was sent.
    fn audio_ended(&mut self, control: &mut Control) -> Outcome<Self::Part>;
    /// The finish time came.
    fn timer(&mut self) -> Vec<Self::Part>;
    /// The socket closed: finish, or fail when the session was not done.
    fn closed(
        &mut self,
        code: Option<u16>,
        reason: &str,
        control: &Control,
    ) -> Result<Vec<Self::Part>, Error>;
}

/// A session to run.
pub(crate) struct Session {
    pub request: WebSocketRequest,
    pub setup: Value,
    /// The input audio rate, for the audio MIME type.
    pub rate: u32,
    pub include_raw: bool,
    pub provider: String,
    /// The socket error message, such as `Google Live transcription error`.
    pub socket_error: &'static str,
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

enum Phase {
    Connecting,
    Open(WebSocket),
    Done,
}

struct State<H: Handler> {
    transport: Arc<dyn WebSocketTransport>,
    session: Session,
    audio: AudioChunks,
    handler: H,
    phase: Phase,
    queue: VecDeque<Result<H::Part, Error>>,
    control: Control,
    setup_complete: bool,
}

enum Step {
    Message(Option<Result<WebSocketMessage, crate::transport::TransportError>>),
    Audio(Option<Vec<u8>>),
    Timer,
}

/// Run a session and stream its parts.
pub(crate) fn run<H: Handler>(
    transport: Arc<dyn WebSocketTransport>,
    session: Session,
    audio: AudioChunks,
    handler: H,
) -> BoxStream<'static, Result<H::Part, Error>> {
    let state = State {
        transport,
        session,
        audio,
        handler,
        phase: Phase::Connecting,
        queue: VecDeque::new(),
        control: Control {
            finish_at: None,
            audio_ended: false,
        },
        setup_complete: false,
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

    async fn finish(&mut self, parts: Vec<H::Part>) {
        self.queue.extend(parts.into_iter().map(Ok));
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
                let message = self.session.socket_error;
                let provider = self.session.provider.clone();
                self.fail(connection_error(&provider, message, Some(error)));
                false
            }
        }
    }

    async fn step(&mut self) {
        if matches!(self.phase, Phase::Connecting) {
            match self.transport.connect(self.session.request.clone()).await {
                Ok(socket) => {
                    self.phase = Phase::Open(socket);
                    let start = self.handler.start();
                    self.queue.push_back(Ok(start));
                    let setup = json!({"setup": self.session.setup});
                    self.send(setup).await;
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
        let read_audio = self.setup_complete && !self.control.audio_ended;
        let finish_at = self.control.finish_at;
        let step = {
            let Phase::Open(socket) = &mut self.phase else {
                return;
            };
            let audio = &mut self.audio;
            tokio::select! {
                message = socket.receiver.next() => Step::Message(message),
                chunk = audio.next(), if read_audio => Step::Audio(chunk),
                () = sleep_until(finish_at), if finish_at.is_some() => Step::Timer,
            }
        };
        match step {
            Step::Timer => {
                self.control.cancel();
                let parts = self.handler.timer();
                self.finish(parts).await;
            }
            Step::Audio(Some(chunk)) => {
                let message = json!({"realtimeInput": {"audio": {
                    "data": STANDARD.encode(&chunk),
                    "mimeType": format!("audio/pcm;rate={}", self.session.rate),
                }}});
                self.send(message).await;
            }
            Step::Audio(None) => {
                if self
                    .send(json!({"realtimeInput": {"audioStreamEnd": true}}))
                    .await
                {
                    self.control.audio_ended = true;
                    match self.handler.audio_ended(&mut self.control) {
                        Outcome::Continue(parts) => self.queue.extend(parts.into_iter().map(Ok)),
                        Outcome::Finish(parts) => self.finish(parts).await,
                    }
                }
            }
            Step::Message(Some(Ok(WebSocketMessage::Text(text)))) => self.message(&text).await,
            Step::Message(Some(Ok(WebSocketMessage::Binary(data)))) => {
                if let Ok(text) = String::from_utf8(data.to_vec()) {
                    self.message(&text).await;
                }
            }
            Step::Message(Some(Ok(WebSocketMessage::Close { code, reason }))) => {
                self.closed(code, &reason).await
            }
            Step::Message(None) => self.closed(None, "").await,
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
        if value.get("setupComplete").is_some_and(|v| !v.is_null()) {
            self.setup_complete = true;
        }
        match self.handler.message(&value, &mut self.control) {
            Ok(Outcome::Continue(parts)) => self.queue.extend(parts.into_iter().map(Ok)),
            Ok(Outcome::Finish(parts)) => self.finish(parts).await,
            Err(error) => self.fail(error),
        }
    }

    async fn closed(&mut self, code: Option<u16>, reason: &str) {
        match self.handler.closed(code, reason, &self.control) {
            Ok(parts) => {
                self.queue.extend(parts.into_iter().map(Ok));
                self.phase = Phase::Done;
            }
            Err(error) => self.fail(error),
        }
    }
}

async fn sleep_until(at: Option<Instant>) {
    match at {
        Some(at) => tokio::time::sleep_until(at).await,
        None => futures::future::pending().await,
    }
}

/// The error a Live server `error` message ends a session with.
pub(crate) fn server_error(provider: &str, message: &Value, default: &str) -> Error {
    let text = message
        .get("error")
        .and_then(|e| e.get("message"))
        .and_then(Value::as_str)
        .unwrap_or(default);
    ProviderError::new(ProviderErrorKind::Provider, text)
        .with_provider(provider)
        .with_body(message.clone())
        .into()
}

/// The error of a socket that closed before the session finished.
pub(crate) fn closed_error(provider: &str, what: &str, code: Option<u16>, reason: &str) -> Error {
    let code = code.map_or_else(|| "unknown".to_owned(), |code| code.to_string());
    let reason = if reason.is_empty() {
        String::new()
    } else {
        format!(", reason: {reason}")
    };
    ProviderError::new(
        ProviderErrorKind::Connection,
        format!("{what} WebSocket closed unexpectedly before finishing (code {code}{reason})."),
    )
    .with_provider(provider)
    .into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn websocket_urls_drop_the_api_version() {
        assert_eq!(
            websocket_url("https://generativelanguage.googleapis.com/v1beta", "a.B").unwrap(),
            "wss://generativelanguage.googleapis.com/ws/a.B"
        );
        assert_eq!(
            websocket_url("http://localhost:8080/proxy/v1alpha/", "a.B").unwrap(),
            "ws://localhost:8080/proxy/ws/a.B"
        );
    }
}
