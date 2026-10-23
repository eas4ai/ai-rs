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

//! Tests for OAI-001 in docs/spec/openai.md. Each test name starts with the
//! requirement it checks; scripts/check-openai.sh selects them by it. The
//! tests build or record each request the crate sends and answer it with a
//! reply written in the test, so no request leaves the machine. They show the
//! crate sends what the AI SDK for TypeScript 7.0.148 sends, not that OpenAI
//! still accepts it.

use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

use bytes::Bytes;
use futures::{StreamExt, future::BoxFuture, stream};
use serde_json::{Value, json};
use suprnova_ai::{
    Model, ProviderConfig, StreamOptions,
    models::params::InferenceRequestParams,
    providers::openai::{OpenAICompatibleProvider, protocol::build_request},
    transport::{HttpRequest, HttpResponse, HttpTransport, TransportError},
    types::{
        Message, MessageRole, Part,
        messages::{TextPart, ToolCallPart, ToolResultPart},
        tools::{Tool, ToolSpec},
    },
};

/// A transport that keeps every request and answers each with the next
/// scripted reply.
struct Recorder {
    requests: Mutex<Vec<HttpRequest>>,
    replies: Mutex<VecDeque<(u16, Vec<u8>)>>,
}
impl Recorder {
    fn new(replies: impl IntoIterator<Item = (u16, Vec<u8>)>) -> Arc<Self> {
        Arc::new(Self {
            requests: Mutex::default(),
            replies: Mutex::new(replies.into_iter().collect()),
        })
    }
    fn body(&self, index: usize) -> Value {
        serde_json::from_slice(&self.requests.lock().unwrap()[index].body).unwrap()
    }
}
impl HttpTransport for Recorder {
    fn send(&self, request: HttpRequest) -> BoxFuture<'_, Result<HttpResponse, TransportError>> {
        self.requests.lock().unwrap().push(request);
        let reply = self.replies.lock().unwrap().pop_front();
        Box::pin(async move {
            let (status, bytes) =
                reply.ok_or_else(|| TransportError::Request("no scripted reply left".into()))?;
            let chunks: Vec<_> = bytes
                .chunks(7)
                .map(|b| Ok(Bytes::copy_from_slice(b)))
                .collect();
            Ok(HttpResponse {
                status,
                headers: vec![],
                body: stream::iter(chunks).boxed(),
            })
        })
    }
}

fn config() -> ProviderConfig {
    ProviderConfig {
        api_key_value: Some("sk-test".into()),
        default_base_url: "https://api.openai.com/v1".into(),
        ..Default::default()
    }
}
/// Call options whose `openai` provider parameters set `store`, if given.
fn store(value: Option<bool>) -> StreamOptions {
    StreamOptions {
        params: value.map(|store| {
            InferenceRequestParams::default()
                .with_provider_param("openai", json!({ "store": store }))
        }),
        ..Default::default()
    }
}
fn user(text: &str) -> Message {
    Message::new(MessageRole::User, vec![Part::Text(TextPart::new(text))])
}
/// The `input` items of a Responses body.
fn items(body: &Value) -> &Vec<Value> {
    body["input"]
        .as_array()
        .expect("a Responses body has input")
}
/// A history of a user turn, an assistant tool call `call_1` whose OpenAI
/// metadata holds the item id `fc_1` and its raw item, and the call's result.
fn history(raw: Value) -> Vec<Message> {
    let mut call = ToolCallPart::new("call_1", "weather", r#"{"city":"SF"}"#);
    call.provider_metadata = Some(
        [(
            "openai".to_owned(),
            json!({"item_id": "fc_1", "raw_item": raw}),
        )]
        .into(),
    );
    vec![
        user("weather in SF?"),
        Message::new(MessageRole::Assistant, vec![Part::ToolCall(call)]),
        Message::new(
            MessageRole::Tool,
            vec![Part::ToolResult(ToolResultPart::new(
                "call_1",
                "weather",
                json!("sunny"),
            ))],
        ),
    ]
}
fn raw_call() -> Value {
    json!({"id": "fc_1", "type": "function_call", "status": "completed", "call_id": "call_1", "name": "weather", "arguments": r#"{"city":"SF"}"#})
}
/// The body `build_request` makes for these messages and options.
fn body(messages: Vec<Message>, options: &StreamOptions) -> Value {
    let request = build_request("gpt-5.4", &config(), messages, options, true).unwrap();
    serde_json::from_slice(&request.body).unwrap()
}

#[test]
fn oai_001_a_tool_call_is_sent_in_full_whether_or_not_responses_are_stored() {
    for value in [None, Some(true), Some(false)] {
        let body = body(history(raw_call()), &store(value));
        let input = items(&body);
        assert!(
            !input
                .iter()
                .any(|item| item["type"] == "item_reference" || item["id"] == "fc_1"),
            "store {value:?} sent the call by its item id: {body}"
        );
        let call = input
            .iter()
            .position(|item| item["type"] == "function_call")
            .unwrap_or_else(|| panic!("store {value:?} sent no function_call: {body}"));
        assert_eq!(
            input[call],
            json!({"type": "function_call", "call_id": "call_1", "name": "weather", "arguments": r#"{"city":"SF"}"#}),
            "store {value:?}"
        );
        assert_eq!(input[call + 1]["type"], "function_call_output", "{body}");
        assert_eq!(input[call + 1]["call_id"], "call_1", "{body}");
    }
}

#[test]
fn oai_001_a_tool_call_keeps_the_namespace_of_its_item() {
    let mut raw = raw_call();
    raw["namespace"] = json!("tools");
    for value in [None, Some(false)] {
        let body = body(history(raw.clone()), &store(value));
        let call = items(&body)
            .iter()
            .find(|item| item["type"] == "function_call")
            .unwrap_or_else(|| panic!("store {value:?} sent no function_call: {body}"));
        assert_eq!(call["namespace"], "tools", "store {value:?}: {body}");
        assert!(call.get("id").is_none(), "store {value:?}: {body}");
    }
}

/// A Responses event stream of these events.
fn responses_sse(events: &[Value]) -> (u16, Vec<u8>) {
    let text: String = events.iter().map(|e| format!("data: {e}\n\n")).collect();
    (200, text.into_bytes())
}
/// A response with two function calls, as OpenAI streams parallel calls.
fn two_calls() -> (u16, Vec<u8>) {
    let call = |index: usize, id: &str, call_id: &str, city: &str| {
        let item = json!({"type": "function_call", "id": id, "call_id": call_id, "name": "weather", "arguments": format!(r#"{{"city":"{city}"}}"#), "status": "completed"});
        [
            json!({"type": "response.output_item.added", "output_index": index, "item": {"type": "function_call", "id": id, "call_id": call_id, "name": "weather", "arguments": "", "status": "in_progress"}}),
            json!({"type": "response.function_call_arguments.delta", "item_id": id, "output_index": index, "delta": item["arguments"]}),
            json!({"type": "response.output_item.done", "output_index": index, "item": item}),
        ]
    };
    let mut events = vec![
        json!({"type": "response.created", "response": {"id": "resp_1", "model": "gpt-5.4", "status": "in_progress"}}),
    ];
    events.extend(call(0, "fc_1", "call_1", "SF"));
    events.extend(call(1, "fc_2", "call_2", "NYC"));
    events.push(json!({"type": "response.completed", "response": {"id": "resp_1", "model": "gpt-5.4", "status": "completed", "output": [{"type": "function_call"}], "usage": {"input_tokens": 5, "output_tokens": 9}}}));
    responses_sse(&events)
}
fn text_reply() -> (u16, Vec<u8>) {
    responses_sse(&[
        json!({"type": "response.created", "response": {"id": "resp_2", "model": "gpt-5.4", "status": "in_progress"}}),
        json!({"type": "response.output_item.added", "output_index": 0, "item": {"type": "message", "id": "msg_1", "role": "assistant", "content": []}}),
        json!({"type": "response.output_text.delta", "item_id": "msg_1", "output_index": 0, "delta": "sunny both"}),
        json!({"type": "response.output_item.done", "output_index": 0, "item": {"type": "message", "id": "msg_1", "role": "assistant", "content": [{"type": "output_text", "text": "sunny both"}]}}),
        json!({"type": "response.completed", "response": {"id": "resp_2", "model": "gpt-5.4", "status": "completed", "output": [], "usage": {"input_tokens": 9, "output_tokens": 2}}}),
    ])
}

#[tokio::test]
async fn oai_001_parallel_calls_from_a_stream_are_sent_back_in_full() {
    let transport = Recorder::new([two_calls(), text_reply()]);
    let model = Model::new(
        "gpt-5.4",
        Arc::new(OpenAICompatibleProvider::new(config(), transport.clone())),
    );
    let weather = Tool::function(
        "weather",
        ToolSpec {
            description: Some("The weather in a city".into()),
            params: json!({"type": "object", "properties": {"city": {"type": "string"}}})
                .try_into()
                .unwrap(),
        },
    );
    let options = || StreamOptions {
        tools: vec![weather.clone()],
        ..Default::default()
    };
    let mut handle = suprnova_ai::stream(&model, vec![user("weather in SF and NYC?")], options());
    while let Some(event) = handle.next().await {
        event.unwrap();
    }
    let assistant = handle.message().clone();
    let calls: Vec<ToolCallPart> = assistant.tool_calls().into_iter().cloned().collect();
    assert_eq!(calls.len(), 2, "{assistant:?}");
    let results = Message::new(
        MessageRole::Tool,
        calls
            .iter()
            .map(|c| {
                Part::ToolResult(ToolResultPart::new(
                    c.tool_call_id.clone(),
                    c.tool_name.clone(),
                    json!("sunny"),
                ))
            })
            .collect(),
    );
    let history = vec![user("weather in SF and NYC?"), assistant, results];
    let mut handle = suprnova_ai::stream(&model, history, options());
    while let Some(event) = handle.next().await {
        event.unwrap();
    }
    let body = transport.body(1);
    let input = items(&body);
    assert!(
        !input.iter().any(|item| item["type"] == "item_reference"
            || item["id"] == "fc_1"
            || item["id"] == "fc_2"),
        "a call was sent by its item id: {body}"
    );
    let kinds: Vec<_> = input
        .iter()
        .map(|item| {
            format!(
                "{}:{}",
                item["type"].as_str().unwrap_or("message"),
                item["call_id"].as_str().unwrap_or("")
            )
        })
        .collect();
    assert_eq!(
        kinds,
        [
            "message:",
            "function_call:call_1",
            "function_call:call_2",
            "function_call_output:call_1",
            "function_call_output:call_2",
        ],
        "{body}"
    );
    assert_eq!(input[1]["arguments"], r#"{"city":"SF"}"#);
    assert_eq!(input[2]["arguments"], r#"{"city":"NYC"}"#);
}
