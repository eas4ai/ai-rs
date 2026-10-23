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

use std::{sync::Arc, time::Duration};

use futures::{SinkExt, StreamExt, future::BoxFuture, stream::SplitSink};
use tokio::net::TcpStream;
use tokio_tungstenite::{
    Connector, MaybeTlsStream, WebSocketStream, connect_async_tls_with_config,
    tungstenite::{
        self, Message,
        client::IntoClientRequest,
        http::{HeaderName, HeaderValue},
        protocol::{CloseFrame, frame::coding::CloseCode},
    },
};

use super::{
    TransportError,
    websocket::{
        WebSocket, WebSocketMessage, WebSocketRequest, WebSocketSender, WebSocketTransport,
    },
};

/// How long opening a connection may take: TCP, TLS and the handshake.
const CONNECT_LIMIT: Duration = Duration::from_secs(10);

/// WebSockets over tokio-tungstenite with rustls and the Mozilla root
/// certificates. Opening a connection fails after 10 s; an open connection
/// has no time limit. No proxy is used.
#[derive(Clone)]
pub struct TungsteniteTransport {
    tls: Arc<rustls::ClientConfig>,
}

impl std::fmt::Debug for TungsteniteTransport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TungsteniteTransport")
            .finish_non_exhaustive()
    }
}

impl TungsteniteTransport {
    /// Build a transport with rustls on the `ring` provider. The provider is
    /// named here, so a process that links another rustls provider as well
    /// needs no default.
    pub fn new() -> Result<Self, TransportError> {
        let roots = rustls::RootCertStore {
            roots: webpki_roots::TLS_SERVER_ROOTS.to_vec(),
        };
        let tls = rustls::ClientConfig::builder_with_provider(Arc::new(
            rustls::crypto::ring::default_provider(),
        ))
        .with_safe_default_protocol_versions()
        .map_err(|error| TransportError::Request(error.to_string()))?
        .with_root_certificates(roots)
        .with_no_client_auth();
        Ok(Self { tls: Arc::new(tls) })
    }

    /// Use a TLS configuration of the application's own.
    pub fn with_tls_config(tls: Arc<rustls::ClientConfig>) -> Self {
        Self { tls }
    }
}

impl WebSocketTransport for TungsteniteTransport {
    fn connect(
        &self,
        request: WebSocketRequest,
    ) -> BoxFuture<'_, Result<WebSocket, TransportError>> {
        Box::pin(async move {
            let mut handshake = request
                .url
                .as_str()
                .into_client_request()
                .map_err(|_| TransportError::Request("invalid WebSocket URL".into()))?;
            for (name, value) in &request.headers {
                let name = HeaderName::from_bytes(name.as_bytes())
                    .map_err(|_| TransportError::Request("invalid WebSocket header name".into()))?;
                let value = HeaderValue::from_str(value).map_err(|_| {
                    TransportError::Request("invalid WebSocket header value".into())
                })?;
                handshake.headers_mut().insert(name, value);
            }
            let connector = Connector::Rustls(self.tls.clone());
            let (stream, _) = tokio::time::timeout(
                CONNECT_LIMIT,
                connect_async_tls_with_config(handshake, None, false, Some(connector)),
            )
            .await
            .map_err(|_| TransportError::Timeout)?
            .map_err(map_error)?;
            let (sink, source) = stream.split();
            let receiver = source
                .filter_map(|message| async move {
                    match message {
                        Ok(Message::Text(text)) => {
                            Some(Ok(WebSocketMessage::Text(text.as_str().to_owned())))
                        }
                        Ok(Message::Binary(data)) => Some(Ok(WebSocketMessage::Binary(data))),
                        Ok(Message::Close(frame)) => Some(Ok(match frame {
                            Some(frame) => WebSocketMessage::Close {
                                code: Some(frame.code.into()),
                                reason: frame.reason.as_str().to_owned(),
                            },
                            None => WebSocketMessage::Close {
                                code: None,
                                reason: String::new(),
                            },
                        })),
                        // Tungstenite answers pings itself.
                        Ok(Message::Ping(_) | Message::Pong(_) | Message::Frame(_)) => None,
                        Err(error) => Some(Err(map_error(error))),
                    }
                })
                .boxed();
            Ok(WebSocket {
                sender: Box::new(Sender(sink)),
                receiver,
            })
        })
    }
}

struct Sender(SplitSink<WebSocketStream<MaybeTlsStream<TcpStream>>, Message>);

impl WebSocketSender for Sender {
    fn send(&mut self, message: WebSocketMessage) -> BoxFuture<'_, Result<(), TransportError>> {
        Box::pin(async move {
            let message = match message {
                WebSocketMessage::Text(text) => Message::text(text),
                WebSocketMessage::Binary(data) => Message::binary(data),
                WebSocketMessage::Close { code, reason } => {
                    Message::Close(code.map(|code| CloseFrame {
                        code: CloseCode::from(code),
                        reason: reason.into(),
                    }))
                }
            };
            self.0.send(message).await.map_err(map_error)
        })
    }
}

/// Map a tungstenite error to a transport error without its text, which can
/// hold the URL.
fn map_error(error: tungstenite::Error) -> TransportError {
    use tungstenite::Error;
    match error {
        Error::Io(_) | Error::Tls(_) | Error::ConnectionClosed | Error::AlreadyClosed => {
            TransportError::Connection("WebSocket connection failed".into())
        }
        Error::Url(_) | Error::HttpFormat(_) => {
            TransportError::Request("invalid WebSocket request".into())
        }
        Error::Http(response) => TransportError::Connection(format!(
            "WebSocket handshake failed with HTTP {}",
            response.status().as_u16()
        )),
        Error::Capacity(_) => {
            TransportError::Protocol("WebSocket message exceeds the size limit".into())
        }
        _ => TransportError::Protocol("WebSocket protocol error".into()),
    }
}
