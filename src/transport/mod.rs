//! HTTP transport shared by provider adapters. Responses keep their streaming body.

use bytes::Bytes;
use futures::{future::BoxFuture, stream::BoxStream};
use thiserror::Error;

#[cfg(feature = "reqwest")]
mod reqwest_download;
#[cfg(feature = "reqwest")]
mod reqwest_transport;
pub mod sse;
#[cfg(feature = "websocket")]
mod tungstenite;
pub mod websocket;
#[cfg(feature = "reqwest")]
pub use reqwest_transport::ReqwestTransport;
#[cfg(feature = "websocket")]
pub use tungstenite::TungsteniteTransport;
pub use websocket::{
    WebSocket, WebSocketMessage, WebSocketRequest, WebSocketSender, WebSocketTransport,
};

pub type Headers = Vec<(String, String)>;
pub type Body = BoxStream<'static, Result<Bytes, TransportError>>;

#[derive(Clone)]
pub struct HttpRequest {
    pub method: String,
    pub url: String,
    pub headers: Headers,
    pub body: Bytes,
}

// Debug output must not expose credentials, query strings, or request bodies.
impl std::fmt::Debug for HttpRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HttpRequest")
            .field("method", &self.method)
            .field("body_len", &self.body.len())
            .finish_non_exhaustive()
    }
}

pub struct HttpResponse {
    pub status: u16,
    pub headers: Headers,
    pub body: Body,
}

impl std::fmt::Debug for HttpResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HttpResponse")
            .field("status", &self.status)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Error)]
pub enum TransportError {
    #[error("invalid HTTP request: {0}")]
    Request(String),
    #[error("HTTP connection failed: {0}")]
    Connection(String),
    #[error("HTTP request timed out")]
    Timeout,
    #[error("HTTP response body failed: {0}")]
    Body(String),
    #[error("invalid event stream: {0}")]
    Protocol(String),
    #[error("event stream exceeded the {0}-byte limit")]
    Limit(usize),
}

/// Implement this trait to supply an application's own HTTP client.
/// Dropping the response body cancels consumption. Status codes are returned to
/// the provider adapter so it can map them into the provider error hierarchy.
pub trait HttpTransport: Send + Sync {
    fn send(&self, request: HttpRequest) -> BoxFuture<'_, Result<HttpResponse, TransportError>>;
}
