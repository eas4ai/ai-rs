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

#![allow(dead_code)]
use ai::{
    Error, Model, ProviderConfig, StreamOptions,
    providers::anthropic::AnthropicCompatibleProvider,
    transport::{Body, HttpRequest, HttpResponse, HttpTransport, TransportError},
    types::{Event, Message},
};
use bytes::Bytes;
use futures::{FutureExt, StreamExt, future::BoxFuture, stream};
use serde_json::{Value, json};
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

pub struct ScriptedTransport {
    pub requests: Mutex<Vec<HttpRequest>>,
    replies: Mutex<VecDeque<Result<HttpResponse, TransportError>>>,
}
impl ScriptedTransport {
    pub fn new(replies: Vec<Result<HttpResponse, TransportError>>) -> Arc<Self> {
        Arc::new(Self {
            requests: Mutex::new(vec![]),
            replies: Mutex::new(replies.into()),
        })
    }
    pub fn events(events: Vec<Value>) -> Arc<Self> {
        Self::new(vec![Ok(sse_response(events))])
    }
    pub fn request(&self) -> HttpRequest {
        self.requests.lock().unwrap().last().unwrap().clone()
    }
}
impl HttpTransport for ScriptedTransport {
    fn send(&self, request: HttpRequest) -> BoxFuture<'_, Result<HttpResponse, TransportError>> {
        self.requests.lock().unwrap().push(request);
        futures::future::ready(
            self.replies
                .lock()
                .unwrap()
                .pop_front()
                .expect("script exhausted"),
        )
        .boxed()
    }
}
pub fn response(status: u16, value: Value) -> HttpResponse {
    HttpResponse {
        status,
        headers: vec![("request-id".into(), "req-anthropic".into())],
        body: stream::iter([Ok(Bytes::from(value.to_string()))]).boxed(),
    }
}
pub fn body(chunks: Vec<Vec<u8>>) -> Body {
    stream::iter(chunks.into_iter().map(|x| Ok(Bytes::from(x)))).boxed()
}
pub fn wire(events: &[Value]) -> Vec<u8> {
    events
        .iter()
        .map(|v| format!("event: {}\ndata: {}\n\n", v["type"].as_str().unwrap(), v))
        .collect::<String>()
        .into_bytes()
}
pub fn sse_response(events: Vec<Value>) -> HttpResponse {
    HttpResponse {
        status: 200,
        headers: vec![("content-type".into(), "text/event-stream".into())],
        body: body(vec![wire(&events)]),
    }
}
pub fn start() -> Value {
    json!({"type":"message_start","message":{"id":"msg_test","model":"claude-test","content":[],"stop_reason":null,"usage":{"input_tokens":0,"output_tokens":0}}})
}
pub fn block_start(index: u32, block: Value) -> Value {
    json!({"type":"content_block_start","index":index,"content_block":block})
}
pub fn delta(index: u32, delta: Value) -> Value {
    json!({"type":"content_block_delta","index":index,"delta":delta})
}
pub fn stop(index: u32) -> Value {
    json!({"type":"content_block_stop","index":index})
}
pub fn complete(mut events: Vec<Value>, reason: Option<&str>) -> Vec<Value> {
    events.insert(0, start());
    events.push(
        json!({"type":"message_delta","delta":{"stop_reason":reason},"usage":{"output_tokens":0}}),
    );
    events.push(json!({"type":"message_stop"}));
    events
}
pub fn provider(fake: Arc<ScriptedTransport>) -> AnthropicCompatibleProvider {
    AnthropicCompatibleProvider::new(
        ProviderConfig {
            default_base_url: "https://anthropic.test".into(),
            api_key_value: Some("sk-test".into()),
            ..Default::default()
        },
        fake,
    )
}
pub fn model(fake: Arc<ScriptedTransport>) -> Model {
    Model::new("claude-sonnet-4-6", Arc::new(provider(fake)))
}
pub async fn capture(messages: Vec<Message>, options: StreamOptions) -> (Value, HttpRequest) {
    let fake = ScriptedTransport::events(complete(vec![], None));
    let model = model(fake.clone());
    let result: Vec<Result<Event, Error>> = model
        .provider
        .stream(&model, messages, options)
        .collect()
        .await;
    for event in result {
        event.unwrap();
    }
    let request = fake.request();
    (serde_json::from_slice(&request.body).unwrap(), request)
}
pub fn header(request: &HttpRequest, name: &str) -> Option<String> {
    request
        .headers
        .iter()
        .find(|(n, _)| n.eq_ignore_ascii_case(name))
        .map(|(_, v)| v.clone())
}
