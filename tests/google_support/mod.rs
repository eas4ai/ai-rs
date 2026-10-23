//! Test support shared by the Google and Vertex AI spec tests: a transport
//! that records each request and answers it with a scripted reply, and a
//! scripted WebSocket transport.

#![allow(dead_code)]

use std::{
    collections::VecDeque,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

use bytes::Bytes;
use futures::{StreamExt, future::BoxFuture, stream};
use serde_json::{Value, json};
use suprnova_ai::{
    Error, Model, StreamOptions,
    models::params::*,
    transport::{
        HttpRequest, HttpResponse, HttpTransport, TransportError, WebSocket, WebSocketMessage,
        WebSocketRequest, WebSocketSender, WebSocketTransport,
    },
    types::{Event, Message},
};

/// A scripted reply: status, headers and body.
#[derive(Clone)]
pub struct Reply {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
    /// A body of this many bytes made as it is read, with the count of
    /// bytes read so far, in place of `body`.
    pub generated: Option<(usize, Arc<AtomicUsize>)>,
}

impl Reply {
    pub fn json(value: Value) -> Self {
        Self {
            status: 200,
            headers: vec![],
            body: value.to_string().into_bytes(),
            generated: None,
        }
    }
    pub fn status(status: u16, value: Value) -> Self {
        Self {
            status,
            headers: vec![],
            body: value.to_string().into_bytes(),
            generated: None,
        }
    }
    pub fn sse(chunks: &[Value]) -> Self {
        Self {
            status: 200,
            headers: vec![],
            body: chunks
                .iter()
                .map(|c| format!("data: {c}\r\n\r\n"))
                .collect::<String>()
                .into_bytes(),
            generated: None,
        }
    }
    pub fn bytes(body: Vec<u8>) -> Self {
        Self {
            status: 200,
            headers: vec![],
            body,
            generated: None,
        }
    }
    /// A body of `size` bytes of `a`, made as the crate reads it, and the
    /// count of bytes it has read.
    pub fn generated(size: usize) -> (Self, Arc<AtomicUsize>) {
        let read = Arc::new(AtomicUsize::new(0));
        let reply = Self {
            status: 200,
            headers: vec![],
            body: vec![],
            generated: Some((size, read.clone())),
        };
        (reply, read)
    }
    pub fn header(mut self, name: &str, value: &str) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }
}

/// Records each request and answers it with the next scripted reply, split
/// into 7-byte chunks so the readers meet every UTF-8 and SSE boundary.
#[derive(Default)]
pub struct Recorder {
    pub requests: Mutex<Vec<HttpRequest>>,
    /// When each request arrived, on the Tokio clock.
    times: Mutex<Vec<tokio::time::Instant>>,
    replies: Mutex<VecDeque<Reply>>,
    chunk: usize,
}

impl Recorder {
    pub fn new(replies: impl IntoIterator<Item = Reply>) -> Arc<Self> {
        Self::with_chunks(replies, 7)
    }
    /// Answer in chunks of this size, for large bodies.
    pub fn with_chunks(replies: impl IntoIterator<Item = Reply>, chunk: usize) -> Arc<Self> {
        Arc::new(Self {
            requests: Mutex::default(),
            times: Mutex::default(),
            replies: Mutex::new(replies.into_iter().collect()),
            chunk,
        })
    }
    pub fn count(&self) -> usize {
        self.requests.lock().unwrap().len()
    }
    pub fn all(&self) -> Vec<HttpRequest> {
        self.requests.lock().unwrap().clone()
    }
    /// When each request arrived, on the Tokio clock.
    pub fn times(&self) -> Vec<tokio::time::Instant> {
        self.times.lock().unwrap().clone()
    }
    pub fn nth(&self, index: usize) -> HttpRequest {
        self.requests.lock().unwrap()[index].clone()
    }
    pub fn request(&self) -> HttpRequest {
        let requests = self.requests.lock().unwrap();
        assert_eq!(requests.len(), 1, "expected exactly one request");
        requests[0].clone()
    }
    pub fn body(&self) -> Value {
        serde_json::from_slice(&self.request().body).unwrap()
    }
    pub fn nth_body(&self, index: usize) -> Value {
        serde_json::from_slice(&self.nth(index).body).unwrap()
    }
    pub fn header(&self, name: &str) -> Option<String> {
        header(&self.request(), name)
    }
}

pub fn header(request: &HttpRequest, name: &str) -> Option<String> {
    request
        .headers
        .iter()
        .find(|(n, _)| n.eq_ignore_ascii_case(name))
        .map(|(_, v)| v.clone())
}

impl HttpTransport for Recorder {
    fn send(&self, request: HttpRequest) -> BoxFuture<'_, Result<HttpResponse, TransportError>> {
        self.requests.lock().unwrap().push(request);
        self.times.lock().unwrap().push(tokio::time::Instant::now());
        let reply = self.replies.lock().unwrap().pop_front();
        let size = self.chunk.max(1);
        Box::pin(async move {
            let reply =
                reply.ok_or_else(|| TransportError::Request("no scripted reply left".into()))?;
            let body = match reply.generated {
                Some((total, read)) => stream::unfold(0usize, move |sent| {
                    let read = read.clone();
                    async move {
                        if sent >= total {
                            return None;
                        }
                        let n = size.min(total - sent);
                        read.fetch_add(n, Ordering::SeqCst);
                        Some((Ok(Bytes::from(vec![b'a'; n])), sent + n))
                    }
                })
                .boxed(),
                None => {
                    let chunks: Vec<_> = reply
                        .body
                        .chunks(size)
                        .map(|b| Ok(Bytes::copy_from_slice(b)))
                        .collect();
                    stream::iter(chunks).boxed()
                }
            };
            Ok(HttpResponse {
                status: reply.status,
                headers: reply.headers,
                body,
            })
        })
    }
}

/// Stream a call to its end or its first error.
pub async fn call(
    model: &Model,
    messages: Vec<Message>,
    options: StreamOptions,
) -> (suprnova_ai::models::StreamHandle, Vec<Result<Event, Error>>) {
    let mut handle = suprnova_ai::stream(model, messages, options);
    let mut events = Vec::new();
    while let Some(event) = handle.next().await {
        let failed = event.is_err();
        events.push(event);
        if failed {
            break;
        }
    }
    (handle, events)
}

pub fn ok(events: &[Result<Event, Error>]) {
    for event in events {
        if let Err(error) = event {
            panic!("the call failed: {error:?}");
        }
    }
}

pub fn error(events: &[Result<Event, Error>]) -> &Error {
    events
        .iter()
        .find_map(|e| e.as_ref().err())
        .expect("the call did not fail")
}

pub fn params(update: impl FnOnce(&mut InferenceRequestParams)) -> StreamOptions {
    let mut params = InferenceRequestParams::default();
    update(&mut params);
    StreamOptions {
        params: Some(params),
        ..Default::default()
    }
}

pub fn google_options(provider_key: &str, value: Value) -> StreamOptions {
    params(|p| {
        p.provider_params = Some([(provider_key.to_owned(), value)].into());
    })
}

pub fn effort(value: Option<&str>) -> StreamOptions {
    params(|p| {
        p.reasoning = ProviderValue::Value(ReasoningParams {
            effort: ProviderValue::Value(value.map(str::to_owned)),
        })
    })
}

/// A generate content stream that answers `text` and stops.
pub fn text_reply(text: &str) -> Reply {
    Reply::sse(&[
        json!({"candidates":[{"content":{"role":"model","parts":[{"text":text}]}}],"responseId":"r1"}),
        json!({"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":1}}),
    ])
}

/// A generate content response that answers `text` and stops.
pub fn text_response(text: &str) -> Reply {
    Reply::json(json!({
        "candidates":[{"content":{"role":"model","parts":[{"text":text}]},"finishReason":"STOP"}],
        "usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":1},
        "responseId":"r1"
    }))
}

/// A scripted WebSocket transport: it records each connection request and
/// every message the crate sends, and answers with the scripted messages
/// as soon as the crate has sent at least as many messages as each one's
/// `after` count.
#[derive(Default)]
pub struct ScriptedSocket {
    pub requests: Mutex<Vec<WebSocketRequest>>,
    pub sent: Arc<Mutex<Vec<WebSocketMessage>>>,
    script: Mutex<Vec<(usize, WebSocketMessage)>>,
    pub closed: Arc<std::sync::atomic::AtomicBool>,
}

impl ScriptedSocket {
    /// Messages to deliver, each once the crate has sent `after` messages.
    pub fn new(script: Vec<(usize, WebSocketMessage)>) -> Arc<Self> {
        Arc::new(Self {
            script: Mutex::new(script),
            ..Default::default()
        })
    }
    pub fn sent_json(&self) -> Vec<Value> {
        self.sent
            .lock()
            .unwrap()
            .iter()
            .filter_map(|m| match m {
                WebSocketMessage::Text(text) => serde_json::from_str(text).ok(),
                _ => None,
            })
            .collect()
    }
    pub fn request(&self) -> WebSocketRequest {
        self.requests.lock().unwrap()[0].clone()
    }
}

pub fn text_message(value: Value) -> WebSocketMessage {
    WebSocketMessage::Text(value.to_string())
}

struct ScriptSender {
    sent: Arc<Mutex<Vec<WebSocketMessage>>>,
    notify: Arc<tokio::sync::Notify>,
}

impl WebSocketSender for ScriptSender {
    fn send(&mut self, message: WebSocketMessage) -> BoxFuture<'_, Result<(), TransportError>> {
        self.sent.lock().unwrap().push(message);
        self.notify.notify_waiters();
        Box::pin(async { Ok(()) })
    }
}

struct Closed(Arc<std::sync::atomic::AtomicBool>);
impl Drop for Closed {
    fn drop(&mut self) {
        self.0.store(true, std::sync::atomic::Ordering::SeqCst);
    }
}

impl WebSocketTransport for ScriptedSocket {
    fn connect(
        &self,
        request: WebSocketRequest,
    ) -> BoxFuture<'_, Result<WebSocket, TransportError>> {
        self.requests.lock().unwrap().push(request);
        let script: VecDeque<_> = std::mem::take(&mut *self.script.lock().unwrap()).into();
        let sent = self.sent.clone();
        let notify = Arc::new(tokio::sync::Notify::new());
        let closed = Closed(self.closed.clone());
        let receiver = stream::unfold(
            (script, sent.clone(), notify.clone(), closed),
            |(mut script, sent, notify, closed)| async move {
                let (after, message) = script.pop_front()?;
                loop {
                    let waiting = notify.notified();
                    if sent.lock().unwrap().len() >= after {
                        break;
                    }
                    waiting.await;
                }
                Some((Ok(message), (script, sent, notify, closed)))
            },
        )
        .boxed();
        Box::pin(async move {
            Ok(WebSocket {
                sender: Box::new(ScriptSender { sent, notify }),
                receiver,
            })
        })
    }
}

/// A download transport that answers from a recorder, for the file URLs a
/// protocol downloads.
pub struct Downloads(pub Arc<Recorder>);

impl suprnova_ai::models::files::DownloadTransport for Downloads {
    fn resolve<'a>(
        &'a self,
        _host: &'a str,
        _port: u16,
    ) -> BoxFuture<'a, Result<Vec<std::net::IpAddr>, suprnova_ai::models::files::DownloadError>>
    {
        Box::pin(async { Ok(vec!["8.8.8.8".parse().unwrap()]) })
    }
    fn send_pinned(
        &self,
        request: HttpRequest,
        _addresses: Vec<std::net::IpAddr>,
    ) -> BoxFuture<'_, Result<HttpResponse, suprnova_ai::models::files::DownloadError>> {
        Box::pin(async move {
            self.0.send(request).await.map_err(|_| {
                suprnova_ai::models::files::DownloadError::new("test", "no scripted download")
            })
        })
    }
}
