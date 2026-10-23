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

//! WebSocket connections for live and realtime sessions, which the HTTP
//! transport cannot carry.

use bytes::Bytes;
use futures::{future::BoxFuture, stream::BoxStream};

use super::{Headers, TransportError};

/// A WebSocket to open: a `wss://` or `ws://` URL and the handshake headers.
#[derive(Clone)]
pub struct WebSocketRequest {
    pub url: String,
    pub headers: Headers,
}

// Debug output must not expose credentials: no header values and no query.
impl std::fmt::Debug for WebSocketRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let url = self.url.as_str();
        let names: Vec<&str> = self.headers.iter().map(|(name, _)| name.as_str()).collect();
        f.debug_struct("WebSocketRequest")
            .field("url", &url)
            .field("header_names", &names)
            .finish()
    }
}

/// One WebSocket message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WebSocketMessage {
    Text(String),
    Binary(Bytes),
    /// The close frame: its code, when it has one, and its reason.
    Close {
        code: Option<u16>,
        reason: String,
    },
}

/// The sending half of an open WebSocket.
pub trait WebSocketSender: Send {
    fn send(&mut self, message: WebSocketMessage) -> BoxFuture<'_, Result<(), TransportError>>;
}

/// An open WebSocket: a sender and the stream of received messages. The
/// stream ends after the close frame or when the connection drops. Dropping
/// both halves closes the connection.
pub struct WebSocket {
    pub sender: Box<dyn WebSocketSender>,
    pub receiver: BoxStream<'static, Result<WebSocketMessage, TransportError>>,
}

impl std::fmt::Debug for WebSocket {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WebSocket").finish_non_exhaustive()
    }
}

/// Implement this trait to open WebSockets with an application's own
/// client. The `websocket` feature provides
/// [`TungsteniteTransport`](super::TungsteniteTransport).
pub trait WebSocketTransport: Send + Sync {
    fn connect(
        &self,
        request: WebSocketRequest,
    ) -> BoxFuture<'_, Result<WebSocket, TransportError>>;
}
