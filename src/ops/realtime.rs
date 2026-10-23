//! Realtime sessions. A server creates an ephemeral token for a browser or
//! other client to connect with, or connects itself with [`connect`]; a
//! protocol's codec turns normalized client events into its messages and
//! its messages into normalized server events.

use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

use futures::{StreamExt, stream::BoxStream};
use serde_json::Value;

use super::{OperationOptions, protocol_of};
use crate::{
    Error, Model,
    errors::{ProviderError, ProviderErrorKind},
    models::call::with_retries,
    providers::operations::*,
    transport::{TransportError, WebSocketMessage, WebSocketSender},
};

/// Create an ephemeral token for a client to open a session with.
pub async fn create_token(
    model: &Model,
    params: RealtimeTokenParams,
    options: OperationOptions,
) -> Result<RealtimeToken, Error> {
    let protocol = protocol_of(model)?;
    with_retries(&options.retry, || {
        protocol.create_realtime_token(model, params.clone(), model.provider.as_ref())
    })
    .await
}

/// The provider's session setup for a session configuration, for a client
/// that sends it itself.
pub fn session(model: &Model, config: &RealtimeSessionConfig) -> Result<Value, Error> {
    protocol_of(model)?.realtime_session(model, config)
}

/// A codec for a connection the application opens itself.
pub fn codec(model: &Model) -> Result<Box<dyn RealtimeCodec>, Error> {
    protocol_of(model)?.realtime_codec(model)
}

/// Open a session with a token through the provider's WebSocket transport
/// and send the session setup. When the protocol asks for it, the first
/// server message must be the session-created event, and no client event is
/// sent before it. Dropping the connection closes the socket.
pub async fn connect(
    model: &Model,
    token: &RealtimeToken,
    config: RealtimeSessionConfig,
) -> Result<RealtimeConnection, Error> {
    let protocol = protocol_of(model)?;
    let transport = model.provider.websocket().ok_or_else(|| {
        Error::Configuration(format!(
            "provider {:?} has no WebSocket transport for realtime sessions",
            model.provider.name()
        ))
    })?;
    let (request, await_session) = protocol.realtime_connection(model, token)?;
    let codec = Arc::new(Mutex::new(protocol.realtime_codec(model)?));
    let provider = model.provider.name().to_owned();
    let socket = transport
        .connect(request)
        .await
        .map_err(|error| connection_error(&provider, error))?;
    let mut connection = RealtimeConnection {
        sender: RealtimeSender {
            sender: socket.sender,
            codec: codec.clone(),
            provider: provider.clone(),
        },
        events: RealtimeEvents {
            receiver: socket.receiver,
            codec,
            pending: VecDeque::new(),
            provider: provider.clone(),
        },
    };
    connection
        .sender
        .send(RealtimeClientEvent::SessionUpdate(config))
        .await?;
    if await_session {
        let first = connection.events.next().await.unwrap_or_else(|| {
            Err(realtime_error(
                &provider,
                "the realtime connection closed before the session was created",
            ))
        })?;
        if !matches!(first.kind, RealtimeServerEventKind::SessionCreated { .. }) {
            return Err(realtime_error(
                &provider,
                "the first realtime server message was not the session-created event",
            ));
        }
        connection.events.pending.push_front(first);
    }
    Ok(connection)
}

/// Open a session with the provider's own credentials through its
/// WebSocket transport, for a server that holds the key, and send the
/// session setup as the protocol starts a session: a session start or a
/// session update (OPS-014). Dropping the connection closes the socket.
pub async fn connect_server(
    model: &Model,
    config: RealtimeSessionConfig,
) -> Result<RealtimeConnection, Error> {
    let _ = (model, config);
    Err(Error::UnsupportedOperation("connect_server is not built".into()))
}

/// Create a WebRTC session from a client's SDP offer (OPS-014). Not built yet.
pub async fn create_webrtc_session(
    model: &Model,
    offer_sdp: String,
    config: RealtimeSessionConfig,
    options: OperationOptions,
) -> Result<WebRtcSession, Error> {
    let _ = (model, offer_sdp, config, options);
    Err(Error::UnsupportedOperation("create_webrtc_session is not built".into()))
}

/// An open realtime session: send client events and read server events.
/// [`split`](Self::split) separates the two for concurrent use.
pub struct RealtimeConnection {
    sender: RealtimeSender,
    events: RealtimeEvents,
}

impl std::fmt::Debug for RealtimeConnection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RealtimeConnection").finish_non_exhaustive()
    }
}

impl RealtimeConnection {
    /// Send a client event; an event the codec has no message for sends
    /// nothing.
    pub async fn send(&mut self, event: RealtimeClientEvent) -> Result<(), Error> {
        self.sender.send(event).await
    }

    /// The next server event, in arrival order; `None` after the socket
    /// closes.
    pub async fn next(&mut self) -> Option<Result<RealtimeServerEvent, Error>> {
        self.events.next().await
    }

    #[must_use]
    pub fn split(self) -> (RealtimeSender, RealtimeEvents) {
        (self.sender, self.events)
    }
}

/// The sending half of a realtime connection.
pub struct RealtimeSender {
    sender: Box<dyn WebSocketSender>,
    codec: Arc<Mutex<Box<dyn RealtimeCodec>>>,
    provider: String,
}

impl RealtimeSender {
    pub async fn send(&mut self, event: RealtimeClientEvent) -> Result<(), Error> {
        let message = lock(&self.codec).encode(&event)?;
        if let Some(message) = message {
            self.sender
                .send(WebSocketMessage::Text(message))
                .await
                .map_err(|error| connection_error(&self.provider, error))?;
        }
        Ok(())
    }

    /// Send a close frame with this code and reason.
    pub async fn close(&mut self, code: u16, reason: &str) -> Result<(), Error> {
        self.sender
            .send(WebSocketMessage::Close {
                code: Some(code),
                reason: reason.to_owned(),
            })
            .await
            .map_err(|error| connection_error(&self.provider, error))
    }
}

/// The receiving half of a realtime connection.
pub struct RealtimeEvents {
    receiver: BoxStream<'static, Result<WebSocketMessage, TransportError>>,
    codec: Arc<Mutex<Box<dyn RealtimeCodec>>>,
    pending: VecDeque<RealtimeServerEvent>,
    provider: String,
}

impl RealtimeEvents {
    pub async fn next(&mut self) -> Option<Result<RealtimeServerEvent, Error>> {
        loop {
            if let Some(event) = self.pending.pop_front() {
                return Some(Ok(event));
            }
            let text = match self.receiver.next().await? {
                Ok(WebSocketMessage::Text(text)) => text,
                Ok(WebSocketMessage::Binary(data)) => match String::from_utf8(data.to_vec()) {
                    Ok(text) => text,
                    Err(_) => continue,
                },
                Ok(WebSocketMessage::Close { .. }) => return None,
                Err(error) => return Some(Err(connection_error(&self.provider, error))),
            };
            match lock(&self.codec).decode(&text) {
                Ok(events) => self.pending.extend(events),
                Err(error) => return Some(Err(error)),
            }
        }
    }
}

fn lock<'a>(
    codec: &'a Mutex<Box<dyn RealtimeCodec>>,
) -> std::sync::MutexGuard<'a, Box<dyn RealtimeCodec>> {
    codec
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn realtime_error(provider: &str, message: &str) -> Error {
    ProviderError::new(ProviderErrorKind::Response, message)
        .with_provider(provider)
        .into()
}

fn connection_error(provider: &str, error: TransportError) -> Error {
    let kind = match error {
        TransportError::Timeout => ProviderErrorKind::Timeout,
        _ => ProviderErrorKind::Connection,
    };
    ProviderError::new(kind, "realtime connection failed")
        .with_provider(provider)
        .with_source(error)
        .into()
}
