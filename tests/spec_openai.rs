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

//! Tests for OAI-001 and OAI-002 in docs/spec/openai.md. Each test name starts with the
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
    models::params::{InferenceRequestParams, OutputParams},
    providers::openai::{OpenAICompatibleProvider, protocol::build_request},
    transport::{HttpRequest, HttpResponse, HttpTransport, TransportError},
    types::{
        Message, MessageRole, Part,
        messages::{ReasoningPart, TextPart, ToolCallPart, ToolResultPart},
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
    history_with(json!({"item_id": "fc_1", "raw_item": raw}))
}
/// The same history with this OpenAI metadata on the call.
fn history_with(metadata: Value) -> Vec<Message> {
    let mut call = ToolCallPart::new("call_1", "weather", r#"{"city":"SF"}"#);
    call.provider_metadata = Some([("openai".to_owned(), metadata)].into());
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
    // The namespace on the raw item, as the crate's reader keeps it; on the
    // metadata itself, as TypeScript keeps it; and a null one on the metadata
    // above the raw item's.
    let shapes = [
        json!({"item_id": "fc_1", "raw_item": raw}),
        json!({"itemId": "fc_1", "namespace": "tools"}),
        json!({"item_id": "fc_1", "namespace": null, "raw_item": raw}),
    ];
    for metadata in shapes {
        for value in [None, Some(true), Some(false)] {
            let body = body(history_with(metadata.clone()), &store(value));
            let call = items(&body)
                .iter()
                .find(|item| item["type"] == "function_call")
                .unwrap_or_else(|| panic!("store {value:?} sent no function_call: {body}"));
            assert_eq!(
                call["namespace"], "tools",
                "{metadata} with store {value:?}: {body}"
            );
            assert!(call.get("id").is_none(), "store {value:?}: {body}");
        }
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

/// Assistant text `T` with the OpenAI item id `msg_1` and its raw item.
fn stored_text() -> Part {
    let mut text = TextPart::new("T");
    text.provider_metadata = Some(
        [(
            "openai".to_owned(),
            json!({"item_id": "msg_1", "raw_item": {"id": "msg_1", "type": "message", "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": "T", "annotations": []}]}}),
        )]
        .into(),
    );
    Part::Text(text)
}
/// A reasoning part of the OpenAI item `id`, with its encrypted content
/// when given.
fn stored_reasoning(id: &str, summary: &str, encrypted: Option<&str>) -> Part {
    let mut raw = json!({"id": id, "type": "reasoning", "summary": [{"type": "summary_text", "text": summary}]});
    let mut metadata = json!({"item_id": id});
    if let Some(encrypted) = encrypted {
        raw["encrypted_content"] = json!(encrypted);
        metadata["reasoning_encrypted_content"] = json!(encrypted);
    }
    metadata["raw_item"] = raw;
    let mut reasoning = ReasoningPart::new(summary);
    reasoning.provider_metadata = Some([("openai".to_owned(), metadata)].into());
    Part::Reasoning(reasoning)
}
/// A user turn, an assistant turn of these parts, and a second user turn.
fn turn(parts: Vec<Part>) -> Vec<Message> {
    vec![
        user("hello"),
        Message::new(MessageRole::Assistant, parts),
        user("again"),
    ]
}
/// The body `build_request` makes for this model.
fn body_for(model: &str, messages: Vec<Message>, options: &StreamOptions) -> Value {
    let request = build_request(model, &config(), messages, options, true).unwrap();
    serde_json::from_slice(&request.body).unwrap()
}
fn references(body: &Value, id: &str) -> usize {
    items(body)
        .iter()
        .filter(|item| item["type"] == "item_reference" && item["id"] == id)
        .count()
}
/// Whether the body sends the text `T` as an assistant message without an id.
fn sends_text(body: &Value) -> bool {
    items(body).iter().any(|item| {
        item["role"] == "assistant"
            && item.get("id").is_none()
            && (item["content"] == "T"
                || item["content"]
                    .as_array()
                    .is_some_and(|parts| parts.iter().any(|p| p["text"] == "T")))
    })
}

#[test]
fn oai_002_stored_text_is_a_reference_only_after_its_reasoning() {
    for value in [None, Some(true)] {
        let options = store(value);
        // A reasoning model's text without its reasoning goes out in full.
        let body = body_for("gpt-5.4", turn(vec![stored_text()]), &options);
        assert_eq!(references(&body, "msg_1"), 0, "store {value:?}: {body}");
        assert!(sends_text(&body), "store {value:?}: {body}");
        // After its reasoning, both go out as one reference each, though the
        // reasoning has two summary parts.
        let parts = vec![
            stored_reasoning("rs_1", "first", None),
            stored_reasoning("rs_1", "second", None),
            stored_text(),
        ];
        let body = body_for("gpt-5.4", turn(parts), &options);
        assert_eq!(references(&body, "rs_1"), 1, "store {value:?}: {body}");
        assert_eq!(references(&body, "msg_1"), 1, "store {value:?}: {body}");
        assert!(!sends_text(&body), "store {value:?}: {body}");
        // Other models keep the reference.
        let body = body_for("gpt-4.1", turn(vec![stored_text()]), &options);
        assert_eq!(references(&body, "msg_1"), 1, "store {value:?}: {body}");
    }
}

#[test]
fn oai_002_reasoning_models_are_named_as_typescript_names_them() {
    let reasoning = [
        "o1",
        "o3",
        "o3-mini",
        "o4-mini-2025-04-16",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5.1",
        "gpt-5.1-chat-latest",
        "gpt-5.4",
        "gpt-6-sol",
    ];
    let other = [
        "gpt-4.1",
        "gpt-4o",
        "gpt-5-chat-latest",
        "o",
        "omni",
        "gpt-4.1-mini",
        "ft:gpt-5:x",
    ];
    for (models, expected) in [(&reasoning[..], 0), (&other[..], 1)] {
        for model in models {
            let body = body_for(model, turn(vec![stored_text()]), &store(None));
            assert_eq!(references(&body, "msg_1"), expected, "{model}: {body}");
        }
    }
}

#[test]
fn oai_002_an_unstored_call_asks_for_and_sends_encrypted_reasoning() {
    let with_include = StreamOptions {
        params: Some(
            InferenceRequestParams {
                output: Some(OutputParams {
                    include: Some(["reasoning.encrypted_content".to_owned()].into()),
                    ..Default::default()
                }),
                ..Default::default()
            }
            .with_provider_param("openai", json!({"store": false})),
        ),
        ..Default::default()
    };
    for options in [store(Some(false)), with_include] {
        let parts = vec![
            stored_reasoning("rs_1", "kept", Some("enc")),
            stored_reasoning("rs_2", "dropped", None),
            stored_text(),
        ];
        let body = body_for("gpt-5.4", turn(parts), &options);
        let include: Vec<_> = body["include"]
            .as_array()
            .unwrap_or_else(|| panic!("no include: {body}"))
            .iter()
            .filter(|value| *value == "reasoning.encrypted_content")
            .collect();
        assert_eq!(include.len(), 1, "{body}");
        let input = items(&body);
        assert!(
            !input
                .iter()
                .any(|item| item["type"] == "item_reference" || item["id"] == "msg_1"),
            "{body}"
        );
        assert!(sends_text(&body), "{body}");
        assert!(
            input
                .iter()
                .any(|item| item["type"] == "reasoning" && item["encrypted_content"] == "enc"),
            "{body}"
        );
        assert!(
            !input.iter().any(|item| item["type"] == "reasoning"
                && item.get("encrypted_content").is_none_or(Value::is_null)),
            "a reasoning part without encrypted content was sent: {body}"
        );
    }
    let body = body_for("gpt-4.1", turn(vec![stored_text()]), &store(Some(false)));
    assert!(
        !body["include"]
            .as_array()
            .is_some_and(|include| include.iter().any(|v| v == "reasoning.encrypted_content")),
        "{body}"
    );
}
