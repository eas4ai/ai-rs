//! Tests for OAI-001 to OAI-027 in docs/spec/openai.md. Each test name starts with the
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
    providers::openai::{
        OpenAIChatCompletionsProtocol, OpenAICompatibleProvider, protocol::build_request,
    },
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
    /// Response headers of the reply to the request of each index.
    headers: Mutex<std::collections::BTreeMap<usize, Vec<(String, String)>>>,
}
impl Recorder {
    fn new(replies: impl IntoIterator<Item = (u16, Vec<u8>)>) -> Arc<Self> {
        Arc::new(Self {
            requests: Mutex::default(),
            replies: Mutex::new(replies.into_iter().collect()),
            headers: Mutex::default(),
        })
    }
    /// Answer the request of this index with these headers.
    fn reply_headers(&self, index: usize, headers: &[(&str, &str)]) {
        self.headers.lock().unwrap().insert(
            index,
            headers
                .iter()
                .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
                .collect(),
        );
    }
    fn body(&self, index: usize) -> Value {
        serde_json::from_slice(&self.requests.lock().unwrap()[index].body).unwrap()
    }
}
impl HttpTransport for Recorder {
    fn send(&self, request: HttpRequest) -> BoxFuture<'_, Result<HttpResponse, TransportError>> {
        let index = {
            let mut requests = self.requests.lock().unwrap();
            requests.push(request);
            requests.len() - 1
        };
        let headers = self
            .headers
            .lock()
            .unwrap()
            .remove(&index)
            .unwrap_or_default();
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
                headers,
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

/// A response whose reasoning item streams two summary parts: OpenAI sends
/// an encrypted content at the item's start and the item's own at its end.
fn reasoning_then_text() -> (u16, Vec<u8>) {
    let summary = json!([{"type": "summary_text", "text": "first"}, {"type": "summary_text", "text": "second"}]);
    responses_sse(&[
        json!({"type": "response.created", "response": {"id": "resp_1", "model": "gpt-5.4", "status": "in_progress"}}),
        json!({"type": "response.output_item.added", "output_index": 0, "item": {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "early"}}),
        json!({"type": "response.reasoning_summary_part.added", "item_id": "rs_1", "output_index": 0, "summary_index": 0}),
        json!({"type": "response.reasoning_summary_text.delta", "item_id": "rs_1", "output_index": 0, "summary_index": 0, "delta": "first"}),
        json!({"type": "response.reasoning_summary_part.done", "item_id": "rs_1", "output_index": 0, "summary_index": 0}),
        json!({"type": "response.reasoning_summary_part.added", "item_id": "rs_1", "output_index": 0, "summary_index": 1}),
        json!({"type": "response.reasoning_summary_text.delta", "item_id": "rs_1", "output_index": 0, "summary_index": 1, "delta": "second"}),
        json!({"type": "response.reasoning_summary_part.done", "item_id": "rs_1", "output_index": 0, "summary_index": 1}),
        json!({"type": "response.output_item.done", "output_index": 0, "item": {"type": "reasoning", "id": "rs_1", "summary": summary, "encrypted_content": "final"}}),
        json!({"type": "response.output_item.added", "output_index": 1, "item": {"type": "message", "id": "msg_1", "role": "assistant", "content": []}}),
        json!({"type": "response.output_text.delta", "item_id": "msg_1", "output_index": 1, "delta": "answer"}),
        json!({"type": "response.output_item.done", "output_index": 1, "item": {"type": "message", "id": "msg_1", "role": "assistant", "content": [{"type": "output_text", "text": "answer"}]}}),
        json!({"type": "response.completed", "response": {"id": "resp_1", "model": "gpt-5.4", "status": "completed", "output": [], "usage": {"input_tokens": 5, "output_tokens": 9}}}),
    ])
}

#[tokio::test]
async fn oai_002_unstored_reasoning_goes_back_with_its_items_final_content() {
    for value in [Some(false), None] {
        let transport = Recorder::new([reasoning_then_text(), text_reply()]);
        let model = Model::new(
            "gpt-5.4",
            Arc::new(OpenAICompatibleProvider::new(config(), transport.clone())),
        );
        let mut handle = suprnova_ai::stream(&model, vec![user("hello")], store(value));
        while let Some(event) = handle.next().await {
            event.unwrap();
        }
        let history = vec![user("hello"), handle.message().clone(), user("again")];
        let mut handle = suprnova_ai::stream(&model, history, store(value));
        while let Some(event) = handle.next().await {
            event.unwrap();
        }
        let body = transport.body(1);
        let reasoning: Vec<_> = items(&body)
            .iter()
            .filter(|item| item["type"] == "reasoning")
            .collect();
        if value == Some(false) {
            assert!(!body.to_string().contains("early"), "{body}");
            assert_eq!(reasoning.len(), 1, "{body}");
            assert_eq!(reasoning[0]["encrypted_content"], "final", "{body}");
            assert_eq!(
                reasoning[0]["summary"],
                json!([{"type": "summary_text", "text": "first"}, {"type": "summary_text", "text": "second"}]),
                "{body}"
            );
        } else {
            assert!(reasoning.is_empty(), "{body}");
            assert_eq!(references(&body, "rs_1"), 1, "{body}");
            assert_eq!(references(&body, "msg_1"), 1, "{body}");
        }
    }
}

/// Assistant text with an item id and a `phase`, on its raw item or, as
/// TypeScript keeps it, on the metadata.
fn phased_text(text: &str, id: &str, phase: &str, on_raw: bool) -> Part {
    let metadata = if on_raw {
        json!({"item_id": id, "raw_item": {"id": id, "type": "message", "role": "assistant", "phase": phase, "content": [{"type": "output_text", "text": text}]}})
    } else {
        json!({"itemId": id, "phase": phase})
    };
    let mut part = TextPart::new(text);
    part.provider_metadata = Some([("openai".to_owned(), metadata)].into());
    Part::Text(part)
}

#[test]
fn oai_002_text_in_full_is_one_message_per_part_with_its_phase() {
    for on_raw in [true, false] {
        for value in [None, Some(true), Some(false)] {
            let parts = vec![
                phased_text("working", "msg_c", "commentary", on_raw),
                phased_text("done", "msg_f", "final_answer", on_raw),
            ];
            let body = body_for("gpt-5.3-codex", turn(parts), &store(value));
            let input = items(&body);
            assert_eq!(
                &input[1..3],
                &[
                    json!({"role": "assistant", "content": "working", "phase": "commentary"}),
                    json!({"role": "assistant", "content": "done", "phase": "final_answer"}),
                ],
                "raw {on_raw}, store {value:?}: {body}"
            );
        }
    }
}

#[test]
fn oai_002_unstored_reasoning_parts_of_one_item_go_out_once() {
    // Two parts of rs_1 without raw items are one item with both summaries.
    let mut first = ReasoningPart::new("first");
    first.provider_metadata = Some(
        [(
            "openai".to_owned(),
            json!({"item_id": "rs_1", "reasoning_encrypted_content": "enc"}),
        )]
        .into(),
    );
    let mut second = first.clone();
    second.text = "second".into();
    let body = body_for(
        "gpt-5.4",
        turn(vec![
            Part::Reasoning(first),
            Part::Reasoning(second.clone()),
        ]),
        &store(Some(false)),
    );
    let reasoning: Vec<_> = items(&body)
        .iter()
        .filter(|item| item["type"] == "reasoning")
        .collect();
    assert_eq!(reasoning.len(), 1, "{body}");
    assert_eq!(
        reasoning[0]["summary"],
        json!([{"type": "summary_text", "text": "first"}, {"type": "summary_text", "text": "second"}])
    );
    // A raw item already holds every summary, so a later part adds none.
    let body = body_for(
        "gpt-5.4",
        turn(vec![
            stored_reasoning("rs_1", "first", Some("enc")),
            Part::Reasoning(second),
        ]),
        &store(Some(false)),
    );
    let reasoning: Vec<_> = items(&body)
        .iter()
        .filter(|item| item["type"] == "reasoning")
        .collect();
    assert_eq!(reasoning.len(), 1, "{body}");
    assert_eq!(
        reasoning[0]["summary"],
        json!([{"type": "summary_text", "text": "first"}]),
        "{body}"
    );
}

// OAI-003

/// A configuration whose `env` holds an organization and a project, with no
/// base URL, so the provider uses its default.
fn org_config() -> ProviderConfig {
    ProviderConfig {
        api_key_value: Some("sk-test".into()),
        env: [
            ("OPENAI_ORG_ID".to_owned(), "org_1".to_owned()),
            ("OPENAI_PROJECT_ID".to_owned(), "proj_1".to_owned()),
        ]
        .into(),
        ..Default::default()
    }
}
/// The value of this header on recorded request `index`, by any case.
fn header(transport: &Recorder, index: usize, name: &str) -> Option<String> {
    transport.requests.lock().unwrap()[index]
        .headers
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case(name))
        .map(|(_, value)| value.clone())
}
fn url(transport: &Recorder, index: usize) -> String {
    transport.requests.lock().unwrap()[index].url.clone()
}
/// Run one call to the end; the error it ended with, if any.
async fn run(model: &Model, options: StreamOptions) -> Result<(), suprnova_ai::Error> {
    let mut handle = suprnova_ai::stream(model, vec![user("hi")], options);
    while let Some(event) = handle.next().await {
        event?;
    }
    Ok(())
}
fn chat_reply() -> (u16, Vec<u8>) {
    let chunks = [
        json!({"id": "c_1", "object": "chat.completion.chunk", "model": "gpt-4.1", "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hi"}, "finish_reason": null}]}),
        json!({"id": "c_1", "object": "chat.completion.chunk", "model": "gpt-4.1", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}),
    ];
    let mut text: String = chunks.iter().map(|c| format!("data: {c}\n\n")).collect();
    text.push_str("data: [DONE]\n\n");
    (200, text.into_bytes())
}

#[tokio::test]
async fn oai_003_requests_go_to_the_default_url_with_key_organization_and_project() {
    let transport = Recorder::new([text_reply(), chat_reply()]);
    let provider = OpenAICompatibleProvider::new(org_config(), transport.clone());
    let responses = Model::new("gpt-5.4", Arc::new(provider.clone()));
    run(&responses, StreamOptions::default()).await.unwrap();
    let chat = Model::new(
        "gpt-4.1",
        Arc::new(provider.with_protocol(Arc::new(OpenAIChatCompletionsProtocol))),
    );
    run(&chat, StreamOptions::default()).await.unwrap();
    for (index, path) in [(0, "/responses"), (1, "/chat/completions")] {
        assert_eq!(
            url(&transport, index),
            format!("https://api.openai.com/v1{path}")
        );
        assert_eq!(
            header(&transport, index, "authorization").as_deref(),
            Some("Bearer sk-test"),
            "{path}"
        );
        assert_eq!(
            header(&transport, index, "OpenAI-Organization").as_deref(),
            Some("org_1"),
            "{path}"
        );
        assert_eq!(
            header(&transport, index, "OpenAI-Project").as_deref(),
            Some("proj_1"),
            "{path}"
        );
    }
}

#[tokio::test]
async fn oai_003_no_organization_or_project_header_without_their_values() {
    let transport = Recorder::new([text_reply()]);
    let mut config = org_config();
    config.env.insert("OPENAI_ORG_ID".into(), String::new());
    config.env.remove("OPENAI_PROJECT_ID");
    let model = Model::new(
        "gpt-5.4",
        Arc::new(OpenAICompatibleProvider::new(config, transport.clone())),
    );
    run(&model, StreamOptions::default()).await.unwrap();
    assert_eq!(header(&transport, 0, "OpenAI-Organization"), None);
    assert_eq!(header(&transport, 0, "OpenAI-Project"), None);
}

#[tokio::test]
async fn oai_003_an_error_body_gives_its_message_type_code_and_status_kind() {
    let body =
        json!({"error": {"message": "slow", "type": "requests", "code": "rate_limit_exceeded"}});
    let transport = Recorder::new([(429, body.to_string().into_bytes())]);
    let model = Model::new(
        "gpt-5.4",
        Arc::new(OpenAICompatibleProvider::new(org_config(), transport)),
    );
    let options = StreamOptions {
        retry: suprnova_ai::RetryPolicy::none(),
        ..Default::default()
    };
    let Err(suprnova_ai::Error::Provider(error)) = run(&model, options).await else {
        panic!("a 429 gave no provider error")
    };
    assert_eq!(error.message, "slow");
    assert_eq!(error.error_type.as_deref(), Some("requests"));
    assert_eq!(error.code.as_deref(), Some("rate_limit_exceeded"));
    assert_eq!(
        error.kind,
        suprnova_ai::errors::ProviderErrorKind::RateLimit
    );
}

// OAI-004

/// Call options whose `openai` provider parameters are `options`.
fn openai(options: Value) -> StreamOptions {
    with_openai(InferenceRequestParams::default(), options)
}
/// These core parameters with `openai` provider parameters `options`.
fn with_openai(params: InferenceRequestParams, options: Value) -> StreamOptions {
    StreamOptions {
        params: Some(params.with_provider_param("openai", options)),
        ..Default::default()
    }
}
fn function_tool(name: &str) -> Tool {
    Tool::function(
        name,
        ToolSpec {
            description: None,
            params: json!({"type": "object", "properties": {}})
                .try_into()
                .unwrap(),
        },
    )
}

#[test]
fn oai_004_each_option_goes_out_under_its_wire_name() {
    let cases = [
        (
            json!({"conversation": "conv_1"}),
            "/conversation",
            json!("conv_1"),
        ),
        (
            json!({"instructions": "be brief"}),
            "/instructions",
            json!("be brief"),
        ),
        (json!({"maxToolCalls": 3}), "/max_tool_calls", json!(3)),
        (
            json!({"metadata": {"a": "b"}}),
            "/metadata",
            json!({"a": "b"}),
        ),
        (
            json!({"parallelToolCalls": false}),
            "/parallel_tool_calls",
            json!(false),
        ),
        (
            json!({"previousResponseId": "resp_0"}),
            "/previous_response_id",
            json!("resp_0"),
        ),
        (
            json!({"promptCacheKey": "k"}),
            "/prompt_cache_key",
            json!("k"),
        ),
        (
            json!({"promptCacheRetention": "24h"}),
            "/prompt_cache_retention",
            json!("24h"),
        ),
        (
            json!({"safetyIdentifier": "s"}),
            "/safety_identifier",
            json!("s"),
        ),
        (
            json!({"serviceTier": "priority"}),
            "/service_tier",
            json!("priority"),
        ),
        (json!({"store": false}), "/store", json!(false)),
        (
            json!({"textVerbosity": "low"}),
            "/text/verbosity",
            json!("low"),
        ),
        (json!({"truncation": "auto"}), "/truncation", json!("auto")),
        (json!({"user": "u"}), "/user", json!("u")),
        (json!({"logprobs": 5}), "/top_logprobs", json!(5)),
        (json!({"logprobs": true}), "/top_logprobs", json!(20)),
        (
            json!({"include": ["file_search_call.results"]}),
            "/include",
            json!(["file_search_call.results"]),
        ),
        (
            json!({"contextManagement": [{"type": "compaction", "compactThreshold": 1000}]}),
            "/context_management",
            json!([{"type": "compaction", "compact_threshold": 1000}]),
        ),
        // gpt-4.1 takes reasoning settings only when forced to reason (OAI-005).
        (
            json!({"forceReasoning": true, "reasoningEffort": "low", "reasoningSummary": "auto"}),
            "/reasoning",
            json!({"effort": "low", "summary": "auto"}),
        ),
        (
            json!({"forceReasoning": true, "reasoningMode": "pro"}),
            "/reasoning",
            json!({"mode": "pro"}),
        ),
        (
            json!({"forceReasoning": true, "reasoningContext": "all_turns"}),
            "/reasoning",
            json!({"context": "all_turns"}),
        ),
    ];
    for (options, pointer, expected) in cases {
        let body = body_for("gpt-4.1", vec![user("hi")], &openai(options.clone()));
        assert_eq!(body.pointer(pointer), Some(&expected), "{options}: {body}");
    }
    let body = body_for(
        "gpt-4.1",
        vec![user("hi")],
        &openai(json!({"compactionTrigger": true})),
    );
    assert_eq!(
        items(&body).last(),
        Some(&json!({"type": "compaction_trigger"})),
        "{body}"
    );
}

#[test]
fn oai_004_an_output_schema_is_strict_unless_the_option_says_otherwise() {
    let schema = |options: Value| {
        let mut options = openai(options);
        options.output_schema = Some(
            json!({"type": "object", "properties": {"a": {"type": "string"}}})
                .try_into()
                .unwrap(),
        );
        body_for("gpt-4.1", vec![user("hi")], &options)["text"]["format"]["strict"].clone()
    };
    assert_eq!(schema(json!({})), json!(true));
    assert_eq!(schema(json!({"strictJsonSchema": false})), json!(false));
}

#[test]
fn oai_004_allowed_tools_go_out_as_an_allowed_tools_choice() {
    let mut options = openai(json!({"allowedTools": {"toolNames": ["a"]}}));
    options.tools = vec![function_tool("a"), function_tool("b")];
    options.tool_choice = Some(suprnova_ai::types::ToolChoice::None);
    let body = body_for("gpt-4.1", vec![user("hi")], &options);
    assert_eq!(
        body["tool_choice"],
        json!({"type": "allowed_tools", "mode": "auto", "tools": [{"type": "function", "name": "a"}]})
    );
}

#[test]
fn oai_004_an_option_takes_precedence_over_the_core_parameter() {
    let mut params = InferenceRequestParams::default();
    params.provider_service = Some(suprnova_ai::models::params::ProviderServiceParams {
        service_tier: suprnova_ai::models::params::ProviderValue::Value("priority".into()),
    });
    let body = body_for(
        "gpt-5.4",
        vec![user("hi")],
        &with_openai(params, json!({"serviceTier": "flex"})),
    );
    assert_eq!(body["service_tier"], "flex", "{body}");
}

#[tokio::test]
async fn oai_004_options_and_settings_openai_never_takes_fail_before_a_request() {
    use suprnova_ai::models::params::{
        ProviderValue, RepetitionPenaltyParams, SamplerParams, Seed, SeedSamplerParams,
        TopKSamplerParams,
    };
    let sampling = |sampling: SamplerParams| InferenceRequestParams {
        sampling: ProviderValue::Value(sampling),
        ..Default::default()
    };
    let cases = [
        (
            "an unknown option",
            openai(json!({"reasoningEfort": "low"})),
        ),
        ("logprobs 21", openai(json!({"logprobs": 21}))),
        (
            "a null forceReasoning",
            openai(json!({"forceReasoning": null})),
        ),
        (
            "a service tier outside the set",
            openai(json!({"serviceTier": "slow"})),
        ),
        (
            "conversation with previousResponseId",
            openai(json!({"conversation": "c", "previousResponseId": "r"})),
        ),
        (
            "a seed",
            with_openai(
                sampling(SamplerParams {
                    seed: Some(SeedSamplerParams {
                        seed: ProviderValue::Value(Some(Seed::Value(7))),
                    }),
                    ..Default::default()
                }),
                json!({}),
            ),
        ),
        (
            "top-k",
            with_openai(
                sampling(SamplerParams {
                    top_k: Some(TopKSamplerParams {
                        top_k: ProviderValue::Value(Some(4)),
                    }),
                    ..Default::default()
                }),
                json!({}),
            ),
        ),
        (
            "a frequency penalty",
            with_openai(
                sampling(SamplerParams {
                    repetition_penalty: Some(RepetitionPenaltyParams {
                        frequency_penalty: ProviderValue::Value(Some(0.5)),
                        ..Default::default()
                    }),
                    ..Default::default()
                }),
                json!({}),
            ),
        ),
        (
            "a presence penalty",
            with_openai(
                sampling(SamplerParams {
                    repetition_penalty: Some(RepetitionPenaltyParams {
                        presence_penalty: ProviderValue::Value(Some(0.5)),
                        ..Default::default()
                    }),
                    ..Default::default()
                }),
                json!({}),
            ),
        ),
    ];
    for (case, options) in cases {
        let transport = Recorder::new([text_reply()]);
        let model = Model::new(
            "gpt-4.1",
            Arc::new(OpenAICompatibleProvider::new(config(), transport.clone())),
        );
        let error = run(&model, options).await.expect_err(case);
        assert!(
            matches!(error, suprnova_ai::Error::Validation(_)),
            "{case}: {error:?}"
        );
        assert!(transport.requests.lock().unwrap().is_empty(), "{case}");
    }
}

// OAI-005

/// The `setting` and `model` fields of each debug event of target
/// `suprnova_ai::openai` that `f` logs.
fn logged<T>(f: impl FnOnce() -> T) -> (T, Vec<(String, String)>) {
    let events = Arc::new(Mutex::new(vec![]));
    let out = tracing::subscriber::with_default(Capture(events.clone()), f);
    let events = events.lock().unwrap().clone();
    (out, events)
}
struct Capture(Arc<Mutex<Vec<(String, String)>>>);
impl tracing::Subscriber for Capture {
    fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        let meta = event.metadata();
        if meta.target() != "suprnova_ai::openai" || *meta.level() != tracing::Level::DEBUG {
            return;
        }
        let mut fields = Fields::default();
        event.record(&mut fields);
        self.0.lock().unwrap().push((fields.setting, fields.model));
    }
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}
#[derive(Default)]
struct Fields {
    setting: String,
    model: String,
}
impl tracing::field::Visit for Fields {
    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        match field.name() {
            "setting" => self.setting = value.into(),
            "model" => self.model = value.into(),
            _ => {}
        }
    }
    fn record_debug(&mut self, _: &tracing::field::Field, _: &dyn std::fmt::Debug) {}
}
fn system(text: &str) -> Message {
    Message::new(MessageRole::System, vec![Part::Text(TextPart::new(text))])
}
fn temperature(effort: Option<&str>, options: Value) -> StreamOptions {
    let mut params = InferenceRequestParams::default().with_temperature(0.5);
    if let Some(effort) = effort {
        params = params.with_reasoning_effort(Some(effort.to_owned()).into());
    }
    with_openai(params, options)
}

#[test]
fn oai_005_system_messages_follow_the_model_and_the_option() {
    let first = |model: &str, options: Value| {
        let body = body_for(model, vec![system("rules"), user("hi")], &openai(options));
        items(&body)[0].clone()
    };
    assert_eq!(first("gpt-5.4", json!({}))["role"], "developer");
    assert_eq!(first("gpt-4.1", json!({}))["role"], "system");
    assert_eq!(
        first("gpt-5.4", json!({"systemMessageMode": "system"}))["role"],
        "system"
    );
    assert_eq!(
        first("gpt-4.1", json!({"systemMessageMode": "developer"}))["role"],
        "developer"
    );
    assert_eq!(
        first("gpt-4.1", json!({"forceReasoning": true}))["role"],
        "developer"
    );
    assert_eq!(
        first("gpt-5.4", json!({"systemMessageMode": "remove"}))["role"],
        "user"
    );
}

#[test]
fn oai_005_an_effort_without_a_summary_option_asks_for_a_detailed_summary() {
    let body = body_for(
        "gpt-5.4",
        vec![user("hi")],
        &openai(json!({"reasoningEffort": "high"})),
    );
    assert_eq!(
        body["reasoning"],
        json!({"effort": "high", "summary": "detailed"})
    );
    let core = StreamOptions {
        params: Some(
            InferenceRequestParams::default().with_reasoning_effort(Some("low".into()).into()),
        ),
        ..Default::default()
    };
    let body = body_for("gpt-5.4", vec![user("hi")], &core);
    assert_eq!(
        body["reasoning"],
        json!({"effort": "low", "summary": "detailed"})
    );
    let body = body_for(
        "gpt-5.4",
        vec![user("hi")],
        &openai(json!({"reasoningEffort": "none"})),
    );
    assert_eq!(body["reasoning"], json!({"effort": "none"}));
    let body = body_for(
        "gpt-5.4",
        vec![user("hi")],
        &openai(json!({"reasoningEffort": "high", "reasoningSummary": null})),
    );
    assert_eq!(body["reasoning"], json!({"effort": "high"}));
}

#[test]
fn oai_005_settings_the_model_does_not_take_are_left_out_and_logged() {
    let web = || {
        let mut options = openai(json!({"logprobs": 3}));
        options.params = Some(
            options
                .params
                .take()
                .unwrap()
                .with_reasoning_effort(Some("high".into()).into()),
        );
        options
    };
    let cases: Vec<(&str, StreamOptions, &str, &str)> = vec![
        (
            "o3",
            temperature(None, json!({})),
            "temperature",
            "/temperature",
        ),
        (
            "gpt-5.4",
            temperature(Some("high"), json!({})),
            "temperature",
            "/temperature",
        ),
        (
            "gpt-4.1",
            openai(json!({"reasoningEffort": "low"})),
            "reasoningEffort",
            "/reasoning",
        ),
        (
            "gpt-4.1",
            openai(json!({"reasoningSummary": "auto"})),
            "reasoningSummary",
            "/reasoning",
        ),
        (
            "gpt-6-astra",
            openai(json!({"reasoningEffort": "minimal"})),
            "reasoningEffort",
            "/reasoning/effort",
        ),
        (
            "gpt-4.1",
            openai(json!({"serviceTier": "flex"})),
            "serviceTier",
            "/service_tier",
        ),
        (
            "gpt-5-nano",
            openai(json!({"serviceTier": "priority"})),
            "serviceTier",
            "/service_tier",
        ),
        ("gpt-6-astra", web(), "logprobs", "/top_logprobs"),
        (
            "gpt-5.4",
            openai(json!({"reasoningEffortUpdate": "low"})),
            "reasoningEffortUpdate",
            "/input/0/reasoning",
        ),
        (
            "gpt-6-astra",
            openai(json!({"promptCacheRetention": "24h"})),
            "promptCacheRetention",
            "/prompt_cache_retention",
        ),
    ];
    for (model, options, setting, pointer) in cases {
        let (body, events) = logged(|| body_for(model, vec![user("hi")], &options));
        assert_eq!(body.pointer(pointer), None, "{model} {setting}: {body}");
        assert!(
            events.contains(&(setting.to_owned(), model.to_owned())),
            "{model} {setting}: {events:?}"
        );
    }
    // The logprobs include value goes with the top logprobs.
    let (body, _) = logged(|| body_for("gpt-6-astra", vec![user("hi")], &web()));
    assert_eq!(body.get("include"), None, "{body}");
}

#[test]
fn oai_005_sampling_goes_out_where_the_model_takes_it() {
    let body = body_for(
        "gpt-5.4",
        vec![user("hi")],
        &temperature(Some("none"), json!({})),
    );
    assert_eq!(body["temperature"], 0.5, "{body}");
    let body = body_for("gpt-4.1", vec![user("hi")], &temperature(None, json!({})));
    assert_eq!(body["temperature"], 0.5, "{body}");
    let body = body_for("gpt-5.4", vec![user("hi")], &openai(json!({"logprobs": 2})));
    assert_eq!(body["top_logprobs"], 2, "{body}");
    let body = body_for(
        "gpt-6-sol",
        vec![user("hi")],
        &temperature(Some("none"), json!({})),
    );
    assert_eq!(body["temperature"], 0.5, "{body}");
}

/// A system message that only sets a reasoning effort.
fn update(effort: &str, text: &str) -> Message {
    let mut message = system(text);
    message.provider_metadata = Some(
        [(
            "openai".to_owned(),
            json!({"reasoningEffortUpdate": effort}),
        )]
        .into(),
    );
    message
}

#[test]
fn oai_005_reasoning_effort_updates_go_out_where_they_apply() {
    let body = body_for(
        "gpt-6-sol",
        vec![user("hi")],
        &openai(json!({"reasoningEffortUpdate": "low"})),
    );
    assert_eq!(
        items(&body)[0],
        json!({"type": "configuration_update", "reasoning": {"effort": "low"}})
    );
    let body = body_for(
        "gpt-6-sol",
        vec![user("hi"), update("high", ""), user("more")],
        &StreamOptions::default(),
    );
    assert_eq!(
        items(&body)[1],
        json!({"type": "configuration_update", "reasoning": {"effort": "high"}})
    );
    // The same update first in the input is not sent twice.
    let body = body_for(
        "gpt-6-sol",
        vec![update("low", ""), user("hi")],
        &openai(json!({"reasoningEffortUpdate": "low"})),
    );
    let updates = items(&body)
        .iter()
        .filter(|item| item["type"] == "configuration_update")
        .count();
    assert_eq!(updates, 1, "{body}");
}

#[test]
fn oai_005_message_level_updates_that_cannot_apply_fail() {
    let cases = [
        ("gpt-5.4", vec![user("hi"), update("low", "")], json!({})),
        (
            "gpt-6-sol",
            vec![update("low", "rules"), user("hi")],
            json!({}),
        ),
        (
            "gpt-6-sol",
            vec![update("low", ""), update("high", ""), user("hi")],
            json!({}),
        ),
        (
            "gpt-6-sol",
            vec![update("high", ""), user("hi")],
            json!({"reasoningEffortUpdate": "low"}),
        ),
        (
            "gpt-6-astra",
            vec![update("none", ""), user("hi")],
            json!({}),
        ),
        (
            "gpt-6-sol",
            vec![update("low", ""), user("hi")],
            json!({"truncation": "auto"}),
        ),
    ];
    for (model, messages, options) in cases {
        let error = build_request(model, &config(), messages, &openai(options.clone()), true)
            .expect_err(model);
        assert!(
            matches!(error, suprnova_ai::Error::Validation(_)),
            "{model} {options}: {error:?}"
        );
    }
}

// OAI-006

#[test]
fn oai_006_include_values_follow_the_tools_and_options() {
    let tools = || {
        vec![
            suprnova_ai::providers::openai::tools::web_search(json!({})).unwrap(),
            suprnova_ai::providers::openai::tools::code_interpreter(json!({})).unwrap(),
        ]
    };
    let mut options = openai(json!({"include": ["file_search_call.results"]}));
    options.tools = tools();
    let body = body_for("gpt-4.1", vec![user("hi")], &options);
    assert_eq!(
        body["include"],
        json!([
            "file_search_call.results",
            "web_search_call.action.sources",
            "code_interpreter_call.outputs"
        ])
    );
    let mut options = openai(json!({"includeWebSearchSources": false}));
    options.tools = tools();
    let body = body_for("gpt-4.1", vec![user("hi")], &options);
    assert_eq!(body["include"], json!(["code_interpreter_call.outputs"]));
    let body = body_for("gpt-4.1", vec![user("hi")], &openai(json!({"logprobs": 3})));
    assert_eq!(body["include"], json!(["message.output_text.logprobs"]));
    let body = body_for(
        "gpt-4.1",
        vec![user("hi")],
        &openai(json!({"forceReasoning": true, "store": false})),
    );
    assert_eq!(body["include"], json!(["reasoning.encrypted_content"]));
}

// OAI-007

/// A file part of this media type holding `data`.
fn file_of(data: suprnova_ai::types::messages::FileData, media_type: &str) -> Part {
    let mut file =
        suprnova_ai::types::messages::FilePart::from_bytes(vec![0], Some(media_type), None)
            .unwrap();
    file.data = data;
    Part::File(file)
}
fn bytes(data: &[u8], media_type: &str) -> Part {
    file_of(
        suprnova_ai::types::messages::FileData::Bytes(data.to_vec()),
        media_type,
    )
}
fn text_data(text: &str, media_type: &str) -> Part {
    file_of(
        suprnova_ai::types::messages::FileData::Text(text.into()),
        media_type,
    )
}
fn with_detail(mut part: Part, detail: &str) -> Part {
    if let Part::File(file) = &mut part {
        file.provider_metadata =
            Some([("openai".to_owned(), json!({"imageDetail": detail}))].into());
    }
    part
}
/// The content item a user message of `"see"` then `part` sends.
fn sent_file(
    responses: bool,
    part: Part,
    options: &StreamOptions,
) -> Result<Value, suprnova_ai::Error> {
    let message = Message::new(
        MessageRole::User,
        vec![Part::Text(TextPart::new("see")), part],
    );
    let request = build_request("gpt-4.1", &config(), vec![message], options, responses)?;
    let body: Value = serde_json::from_slice(&request.body).unwrap();
    let key = if responses { "input" } else { "messages" };
    Ok(body[key][0]["content"][1].clone())
}

#[test]
fn oai_007_responses_files_go_as_typescript_sends_them() {
    let none = StreamOptions::default();
    let send = |part: Part| sent_file(true, part, &none).unwrap();
    assert_eq!(
        send(text_data("https://img.test/a.png", "image/png")),
        json!({"type": "input_image", "image_url": "https://img.test/a.png"})
    );
    assert_eq!(
        send(with_detail(bytes(&[1, 2, 3], "image/png"), "low")),
        json!({"type": "input_image", "image_url": "data:image/png;base64,AQID", "detail": "low"})
    );
    assert_eq!(
        send(text_data("file-abc", "application/pdf")),
        json!({"type": "input_file", "file_id": "file-abc"})
    );
    assert_eq!(
        send(with_detail(text_data("file-img", "image/png"), "high")),
        json!({"type": "input_image", "file_id": "file-img", "detail": "high"})
    );
    assert_eq!(
        send(text_data("https://docs.test/a.pdf", "application/pdf")),
        json!({"type": "input_file", "file_url": "https://docs.test/a.pdf"})
    );
    assert_eq!(
        send(bytes(&[1, 2, 3], "application/pdf")),
        json!({"type": "input_file", "filename": "part-1.pdf", "file_data": "data:application/pdf;base64,AQID"})
    );
}

#[test]
fn oai_007_other_files_need_the_pass_through_option_on_responses() {
    // A text file goes as text without the option (Shawn's OAI-007 ruling).
    assert_eq!(
        sent_file(true, bytes(b"a,b", "text/csv"), &StreamOptions::default()).unwrap(),
        json!({"type": "input_text", "text": "a,b"})
    );
    let error = sent_file(
        true,
        bytes(&[1, 2, 3], "audio/flac"),
        &StreamOptions::default(),
    )
    .expect_err("audio/flac without the option");
    assert!(
        matches!(error, suprnova_ai::Error::Validation(_)),
        "{error:?}"
    );
    let pass = openai(json!({"passThroughUnsupportedFiles": true}));
    assert_eq!(
        sent_file(true, bytes(b"a,b", "text/csv"), &pass).unwrap(),
        json!({"type": "input_file", "filename": "part-1", "file_data": "data:text/csv;base64,YSxi"})
    );
    assert_eq!(
        sent_file(
            true,
            text_data("https://data.test/t.csv", "text/csv"),
            &pass
        )
        .unwrap(),
        json!({"type": "input_file", "file_url": "https://data.test/t.csv"})
    );
}

#[tokio::test]
async fn oai_007_chat_completions_files_go_as_typescript_sends_them() {
    let none = StreamOptions::default();
    let send = |part: Part| sent_file(false, part, &none).unwrap();
    assert_eq!(
        send(text_data("file-abc", "application/pdf")),
        json!({"type": "file", "file": {"file_id": "file-abc"}})
    );
    assert_eq!(
        send(bytes(&[1, 2, 3], "audio/wav")),
        json!({"type": "input_audio", "input_audio": {"data": "AQID", "format": "wav"}})
    );
    assert_eq!(
        send(bytes(&[1, 2, 3], "audio/mpeg")),
        json!({"type": "input_audio", "input_audio": {"data": "AQID", "format": "mp3"}})
    );
    assert_eq!(
        send(bytes(&[1, 2, 3], "application/pdf")),
        json!({"type": "file", "file": {"filename": "part-1.pdf", "file_data": "data:application/pdf;base64,AQID"}})
    );
    assert_eq!(
        send(with_detail(bytes(&[1, 2, 3], "image/png"), "high")),
        json!({"type": "image_url", "image_url": {"url": "data:image/png;base64,AQID", "detail": "high"}})
    );
    // TypeScript takes no text file on Chat Completions; Shawn's ruling
    // keeps text files inline on Responses only.
    for part in [
        bytes(&[1, 2, 3], "audio/flac"),
        bytes(b"a,b", "text/csv"),
        text_data("https://audio.test/a.mp3", "audio/mpeg"),
        text_data("https://docs.test/a.pdf", "application/pdf"),
    ] {
        let transport = Recorder::new([chat_reply()]);
        let provider = OpenAICompatibleProvider::new(config(), transport.clone())
            .with_protocol(Arc::new(OpenAIChatCompletionsProtocol));
        let model = Model::new("gpt-4.1", Arc::new(provider));
        let message = Message::new(MessageRole::User, vec![part]);
        let mut handle = suprnova_ai::stream(&model, vec![message], StreamOptions::default());
        let mut error = None;
        while let Some(event) = handle.next().await {
            if let Err(e) = event {
                error = Some(e);
            }
        }
        assert!(
            matches!(error, Some(suprnova_ai::Error::Validation(_))),
            "{error:?}"
        );
        assert!(transport.requests.lock().unwrap().is_empty());
    }
}

// OAI-008

use suprnova_ai::types::messages::{
    BuiltinToolCallPart, BuiltinToolReturnPart, HookPart, HookStatus,
};

fn md(value: Value) -> Option<suprnova_ai::types::messages::Metadata> {
    Some([("openai".to_owned(), value)].into())
}
fn builtin_call(name: &str, item: &str, extra: Value) -> Part {
    let mut metadata = json!({"item_id": item});
    if let (Some(metadata), Some(extra)) = (metadata.as_object_mut(), extra.as_object()) {
        metadata.extend(extra.clone());
    }
    Part::BuiltinToolCall(BuiltinToolCallPart {
        id: format!("part_{item}"),
        tool_call_id: item.into(),
        tool_name: name.into(),
        tool_args: "{}".into(),
        provider_metadata: md(metadata),
    })
}
fn builtin_return(name: &str, item: &str) -> Part {
    Part::BuiltinToolReturn(BuiltinToolReturnPart {
        id: format!("part_r_{item}"),
        tool_call_id: item.into(),
        tool_name: name.into(),
        result: json!({"action": {"type": "search"}}),
        is_error: false,
        provider_metadata: md(json!({"item_id": item})),
    })
}
/// A stored assistant turn: reasoning `rs_1`, text `msg_1`, a call `call_1`
/// with the item id `fc_1`, a web search `ws_1` and its result.
fn stored_turn() -> Vec<Message> {
    let mut call = ToolCallPart::new("call_1", "weather", r#"{"city":"SF"}"#);
    call.provider_metadata = md(json!({"item_id": "fc_1"}));
    vec![
        user("weather?"),
        Message::new(
            MessageRole::Assistant,
            vec![
                stored_reasoning("rs_1", "thinking", Some("enc")),
                stored_text(),
                builtin_call("web_search", "ws_1", json!({})),
                builtin_return("web_search", "ws_1"),
                Part::ToolCall(call),
            ],
        ),
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
/// Whether any input item has this id or references it.
fn mentions(body: &Value, id: &str) -> bool {
    items(body).iter().any(|item| item["id"] == id)
}

#[test]
fn oai_008_a_conversation_leaves_out_the_items_it_holds() {
    let body = body_for(
        "gpt-5.4",
        stored_turn(),
        &openai(json!({"conversation": "conv_1"})),
    );
    for id in ["rs_1", "msg_1", "ws_1", "fc_1"] {
        assert!(!mentions(&body, id), "{id}: {body}");
    }
    assert!(
        !items(&body)
            .iter()
            .any(|item| item["type"] == "function_call"),
        "{body}"
    );
    assert!(
        items(&body)
            .iter()
            .any(|item| item["type"] == "function_call_output" && item["call_id"] == "call_1"),
        "{body}"
    );
}

#[test]
fn oai_008_a_previous_response_leaves_out_its_reasoning() {
    let body = body_for(
        "gpt-5.4",
        stored_turn(),
        &openai(json!({"previousResponseId": "resp_0"})),
    );
    assert!(!mentions(&body, "rs_1"), "{body}");
    assert!(
        !items(&body).iter().any(|item| item["type"] == "reasoning"),
        "{body}"
    );
}

#[test]
fn oai_008_a_stored_built_in_call_and_result_are_one_reference() {
    let body = body_for("gpt-5.4", stored_turn(), &StreamOptions::default());
    assert_eq!(references(&body, "ws_1"), 1, "{body}");
    assert!(
        !items(&body)
            .iter()
            .any(|item| item["type"] == "web_search_call"),
        "{body}"
    );
}

#[test]
fn oai_008_compaction_is_a_reference_when_stored_and_an_item_when_not() {
    let turn = || {
        vec![
            user("go on"),
            Message::new(
                MessageRole::Assistant,
                vec![builtin_call(
                    "compaction",
                    "cmp_1",
                    json!({"encrypted_content": "enc"}),
                )],
            ),
            user("more"),
        ]
    };
    let body = body_for("gpt-4.1", turn(), &StreamOptions::default());
    assert_eq!(
        items(&body)[1],
        json!({"type": "item_reference", "id": "cmp_1"})
    );
    let body = body_for("gpt-4.1", turn(), &store(Some(false)));
    assert_eq!(
        items(&body)[1],
        json!({"type": "compaction", "id": "cmp_1", "encrypted_content": "enc"})
    );
}

/// An assistant turn with the MCP approval request `mcpr_1` and the
/// application's answer to it.
fn approval_turn(granted: bool) -> Vec<Message> {
    let hook = HookPart {
        id: "part_hook".into(),
        hook_id: "mcpr_1".into(),
        hook_type: "ToolApproval".into(),
        status: HookStatus::Resolved,
        metadata: [(
            "openai".to_owned(),
            json!({"approval_request_id": "mcpr_1"}),
        )]
        .into(),
        resolution: Some(json!({"granted": granted})),
        tool_call_id: Some("mcpr_1".into()),
    };
    vec![
        user("delete it"),
        Message::new(
            MessageRole::Assistant,
            vec![
                builtin_call("mcp.delete", "mcpr_1", json!({})),
                Part::Hook(hook),
            ],
        ),
    ]
}

#[test]
fn oai_008_an_answered_mcp_approval_goes_back_as_a_response() {
    let body = body_for("gpt-4.1", approval_turn(true), &StreamOptions::default());
    assert_eq!(
        items(&body)[1..],
        [
            json!({"type": "item_reference", "id": "mcpr_1"}),
            json!({"type": "mcp_approval_response", "approval_request_id": "mcpr_1", "approve": true}),
        ],
        "{body}"
    );
    let body = body_for(
        "gpt-4.1",
        approval_turn(false),
        &openai(json!({"previousResponseId": "resp_0"})),
    );
    let responses: Vec<_> = items(&body)
        .iter()
        .filter(|item| item["type"] == "mcp_approval_response")
        .collect();
    assert_eq!(
        responses,
        [
            &json!({"type": "mcp_approval_response", "approval_request_id": "mcpr_1", "approve": false})
        ]
    );
}

#[test]
fn oai_008_a_result_with_additional_tools_is_followed_by_them() {
    let with_tools = |tools: Value| {
        let mut messages = history(raw_call());
        if let Part::ToolResult(result) = &mut messages[2].parts[0] {
            result.provider_metadata = md(json!({"additionalTools": tools}));
        }
        messages
    };
    let tools = json!([{"type": "function", "name": "late", "parameters": {"type": "object"}}]);
    let body = body_for(
        "gpt-4.1",
        with_tools(tools.clone()),
        &StreamOptions::default(),
    );
    let output = items(&body)
        .iter()
        .position(|item| item["type"] == "function_call_output")
        .unwrap();
    assert_eq!(
        items(&body)[output + 1],
        json!({"type": "additional_tools", "role": "developer", "tools": tools})
    );
    let error = build_request(
        "gpt-4.1",
        &config(),
        with_tools(json!([{"type": "function", "name": ""}])),
        &StreamOptions::default(),
        true,
    )
    .expect_err("an invalid additionalTools entry");
    assert!(
        matches!(error, suprnova_ai::Error::Validation(_)),
        "{error:?}"
    );
}

#[test]
fn oai_008_expanded_parallel_calls_go_back_as_their_wrapper() {
    let call = |index: usize| {
        let mut call = ToolCallPart::new(format!("c_{index}"), "weather", r#"{"city":"SF"}"#);
        call.provider_metadata = md(json!({"parallel_tool_call": {
            "item_id": "fc_p", "tool_call_id": "c", "tool_name": "parallel",
            "input": "{\"tool_uses\":[]}", "index": index, "count": 2,
        }}));
        Part::ToolCall(call)
    };
    let result = |index: usize| {
        Part::ToolResult(ToolResultPart::new(
            format!("c_{index}"),
            "weather",
            json!(format!("r{index}")),
        ))
    };
    let messages = vec![
        user("weather twice"),
        Message::new(MessageRole::Assistant, vec![call(0), call(1)]),
        Message::new(MessageRole::Tool, vec![result(0), result(1)]),
    ];
    let body = body_for(
        "gpt-4.1",
        messages.clone(),
        &openai(json!({"previousResponseId": "resp_0"})),
    );
    assert_eq!(
        items(&body)[1..],
        [
            json!({"type": "function_call", "call_id": "c", "name": "parallel", "arguments": "{\"tool_uses\":[]}"}),
            json!({"type": "function_call_output", "call_id": "c", "output": "r0\nr1"}),
        ],
        "{body}"
    );
    // Without a conversation or a previous response each call goes on its own.
    let body = body_for("gpt-4.1", messages, &StreamOptions::default());
    let calls: Vec<_> = items(&body)
        .iter()
        .filter(|item| item["type"] == "function_call")
        .map(|item| item["call_id"].clone())
        .collect();
    assert_eq!(calls, [json!("c_0"), json!("c_1")], "{body}");
}

// OAI-009

/// A function tool `name` whose `tool_config` arguments are `args`.
fn tool_with(name: &str, args: Value) -> Tool {
    let mut tool = function_tool(name);
    tool.tool_config = Some(suprnova_ai::types::tools::ToolConfig {
        id: None,
        args: args.as_object().unwrap().clone(),
    });
    tool
}
fn with_tools(tools: Vec<Tool>) -> StreamOptions {
    StreamOptions {
        tools,
        ..Default::default()
    }
}
fn tools_sent(model: &str, tools: Vec<Tool>) -> Value {
    body_for(model, vec![user("hi")], &with_tools(tools))["tools"].clone()
}

#[test]
fn oai_009_function_tools_carry_strict_and_their_openai_options() {
    let tools = tools_sent(
        "gpt-6-astra",
        vec![
            function_tool("plain"),
            function_tool("strict").with_strict(true),
            tool_with(
                "loaded",
                json!({
                    "defer_loading": true,
                    "async": true,
                    "allowed_callers": ["programmatic"],
                    "output_schema": {"type": "object"},
                }),
            ),
            tool_with("hidden", json!({"omit_from_request_tools": true})),
        ],
    );
    let params = json!({"type": "object", "properties": {}});
    assert_eq!(
        tools,
        json!([
            {"type": "function", "name": "plain", "description": "", "parameters": params, "strict": false},
            {"type": "function", "name": "strict", "description": "", "parameters": params, "strict": true},
            {"type": "function", "name": "loaded", "description": "", "parameters": params, "strict": false,
             "defer_loading": true, "async": true, "allowed_callers": ["programmatic"],
             "output_schema": {"type": "object"}},
        ])
    );
}

#[test]
fn oai_009_tools_of_one_namespace_go_in_one_group() {
    let crm = || json!({"namespace": {"name": "crm", "description": "CRM tools"}});
    let tools = tools_sent(
        "gpt-4.1",
        vec![
            tool_with("find", crm()),
            function_tool("other"),
            tool_with("update", crm()),
        ],
    );
    let names = |group: &Value| -> Vec<Value> {
        group["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].clone())
            .collect()
    };
    assert_eq!(tools[0]["type"], "namespace", "{tools}");
    assert_eq!(tools[0]["name"], "crm");
    assert_eq!(tools[0]["description"], "CRM tools");
    assert_eq!(names(&tools[0]), [json!("find"), json!("update")]);
    assert_eq!(tools[1]["name"], "other");
    assert_eq!(tools.as_array().unwrap().len(), 2);
}

#[tokio::test]
async fn oai_009_invalid_tool_settings_fail_before_a_request() {
    let conflicting = with_tools(vec![
        tool_with(
            "a",
            json!({"namespace": {"name": "crm", "description": "one"}}),
        ),
        tool_with(
            "b",
            json!({"namespace": {"name": "crm", "description": "two"}}),
        ),
    ]);
    let mut forced = with_tools(vec![tool_with(
        "hidden",
        json!({"omit_from_request_tools": true}),
    )]);
    forced.tool_choice = Some(suprnova_ai::types::ToolChoice::Tool(
        suprnova_ai::types::tools::ToolRef::new("hidden"),
    ));
    for (case, options) in [
        ("two namespace descriptions", conflicting),
        ("a forced omitted tool", forced),
    ] {
        let transport = Recorder::new([text_reply()]);
        let model = Model::new(
            "gpt-4.1",
            Arc::new(OpenAICompatibleProvider::new(config(), transport.clone())),
        );
        let error = run(&model, options).await.expect_err(case);
        assert!(
            matches!(error, suprnova_ai::Error::Validation(_)),
            "{case}: {error:?}"
        );
        assert!(transport.requests.lock().unwrap().is_empty(), "{case}");
    }
}

#[test]
fn oai_009_async_before_gpt_6_is_left_out_and_logged() {
    let (tools, events) = logged(|| {
        tools_sent(
            "gpt-5.4",
            vec![
                tool_with("slow", json!({"async": true})),
                suprnova_ai::providers::openai::tools::custom("notes", json!({"async": true}))
                    .unwrap(),
            ],
        )
    });
    assert_eq!(tools[0].get("async"), None, "{tools}");
    assert_eq!(tools[1].get("async"), None, "{tools}");
    assert!(
        events.contains(&("async".to_owned(), "gpt-5.4".to_owned())),
        "{events:?}"
    );
}

#[test]
fn oai_009_a_text_result_of_a_tool_with_an_output_schema_is_json_quoted() {
    let mut options = with_tools(vec![tool_with(
        "weather",
        json!({"output_schema": {"type": "string"}}),
    )]);
    options.params = None;
    let body = body_for("gpt-4.1", history(raw_call()), &options);
    let output = items(&body)
        .iter()
        .find(|item| item["type"] == "function_call_output")
        .unwrap();
    assert_eq!(output["output"], json!("\"sunny\""), "{body}");
}

#[test]
fn oai_009_openai_tools_and_named_choices_go_as_typescript_sends_them() {
    use suprnova_ai::providers::openai::tools;
    let sql = tools::custom(
        "sql",
        json!({"description": "SQL", "format": {"type": "grammar", "syntax": "lark", "definition": "start: \"select\""}}),
    )
    .unwrap();
    let all = || {
        vec![
            function_tool("lookup"),
            tools::image_generation(json!({})).unwrap(),
            sql.clone(),
            tools::computer(),
            tools::programmatic_tool_calling(),
        ]
    };
    let sent = tools_sent("gpt-4.1", all());
    assert_eq!(
        sent[2],
        json!({"type": "custom", "name": "sql", "description": "SQL", "format": {"type": "grammar", "syntax": "lark", "definition": "start: \"select\""}})
    );
    assert_eq!(sent[3], json!({"type": "computer"}));
    assert_eq!(sent[4], json!({"type": "programmatic_tool_calling"}));
    for (name, expected) in [
        ("image_generation", json!({"type": "image_generation"})),
        ("computer", json!({"type": "computer"})),
        ("sql", json!({"type": "custom", "name": "sql"})),
        ("lookup", json!({"type": "function", "name": "lookup"})),
    ] {
        let mut options = with_tools(all());
        options.tool_choice = Some(suprnova_ai::types::ToolChoice::Tool(
            suprnova_ai::types::tools::ToolRef::new(name),
        ));
        let body = body_for("gpt-4.1", vec![user("hi")], &options);
        assert_eq!(body["tool_choice"], expected, "{name}");
    }
    assert!(tools::custom("bad", json!({"format": {"type": "grammar"}})).is_err());
}

#[test]
fn oai_009_chat_completions_function_tools_are_strict_only_when_marked() {
    let options = with_tools(vec![
        function_tool("plain"),
        function_tool("strict").with_strict(true),
    ]);
    let request = build_request("gpt-4.1", &config(), vec![user("hi")], &options, false).unwrap();
    let body: Value = serde_json::from_slice(&request.body).unwrap();
    assert_eq!(body["tools"][0]["function"].get("strict"), None, "{body}");
    assert_eq!(body["tools"][1]["function"]["strict"], true, "{body}");
}

// OAI-010

/// A response that streams these output items, each added then done.
fn items_reply(items: &[Value]) -> (u16, Vec<u8>) {
    let mut events = vec![
        json!({"type": "response.created", "response": {"id": "resp_h", "model": "gpt-5.4", "status": "in_progress"}}),
    ];
    for (index, item) in items.iter().enumerate() {
        let mut added = item.clone();
        added["status"] = json!("in_progress");
        events.push(
            json!({"type": "response.output_item.added", "output_index": index, "item": added}),
        );
        events.push(
            json!({"type": "response.output_item.done", "output_index": index, "item": item}),
        );
    }
    events.push(json!({"type": "response.completed", "response": {"id": "resp_h", "model": "gpt-5.4", "status": "completed", "output": items, "usage": {"input_tokens": 1, "output_tokens": 1}}}));
    responses_sse(&events)
}
fn host_items() -> Vec<Value> {
    vec![
        json!({"type": "local_shell_call", "id": "lsh_1", "call_id": "call_ls", "status": "completed",
               "action": {"type": "exec", "command": ["ls"], "timeout_ms": 1000}}),
        json!({"type": "shell_call", "id": "sh_1", "call_id": "call_sh", "status": "completed",
               "action": {"commands": ["ls"], "timeout_ms": null, "max_output_length": 100}}),
        json!({"type": "apply_patch_call", "id": "ap_1", "call_id": "call_ap", "status": "completed",
               "operation": {"type": "create_file", "path": "a.txt", "diff": "+a"}}),
        json!({"type": "computer_call", "id": "cu_1", "call_id": "call_cu", "status": "completed",
               "actions": [{"type": "click", "button": "left", "x": 1, "y": 2}], "pending_safety_checks": []}),
        json!({"type": "custom_tool_call", "id": "ct_1", "call_id": "call_ct", "name": "sql", "input": "select 1"}),
    ]
}
fn host_tools() -> Vec<Tool> {
    use suprnova_ai::providers::openai::tools;
    vec![
        tools::local_shell(),
        tools::shell(json!({})).unwrap(),
        tools::apply_patch(),
        tools::computer(),
        tools::custom("sql", json!({})).unwrap(),
    ]
}

#[tokio::test]
async fn oai_010_calls_the_application_runs_are_delivered_as_tool_calls() {
    let transport = Recorder::new([items_reply(&host_items())]);
    let model = Model::new(
        "gpt-5.4",
        Arc::new(OpenAICompatibleProvider::new(config(), transport)),
    );
    let mut handle = suprnova_ai::stream(&model, vec![user("go")], with_tools(host_tools()));
    while let Some(event) = handle.next().await {
        event.unwrap();
    }
    let message = handle.message().clone();
    assert!(
        !message
            .parts
            .iter()
            .any(|p| matches!(p, Part::BuiltinToolCall(_) | Part::BuiltinToolReturn(_))),
        "{message:?}"
    );
    let calls: Vec<(String, String, Value)> = message
        .tool_calls()
        .into_iter()
        .map(|c| {
            (
                c.tool_call_id.clone(),
                c.tool_name.clone(),
                serde_json::from_str(&c.tool_args).unwrap(),
            )
        })
        .collect();
    assert_eq!(
        calls,
        [
            (
                "call_ls".into(),
                "local_shell".into(),
                json!({"action": {"type": "exec", "command": ["ls"], "timeout_ms": 1000}})
            ),
            (
                "call_sh".into(),
                "shell".into(),
                json!({"action": {"commands": ["ls"], "max_output_length": 100}})
            ),
            (
                "call_ap".into(),
                "apply_patch".into(),
                json!({"call_id": "call_ap", "operation": {"type": "create_file", "path": "a.txt", "diff": "+a"}})
            ),
            (
                "call_cu".into(),
                "computer".into(),
                json!({"actions": [{"type": "click", "button": "left", "x": 1, "y": 2}], "pending_safety_checks": [], "status": "completed"})
            ),
            ("call_ct".into(), "sql".into(), json!("select 1")),
        ]
    );
    assert_eq!(handle.finish_reason(), Some("tool_call"));
}

#[tokio::test]
async fn oai_010_a_shell_in_a_container_stays_a_built_in_call() {
    let shell = suprnova_ai::providers::openai::tools::shell(
        json!({"environment": {"type": "container_auto"}}),
    )
    .unwrap();
    let transport = Recorder::new([items_reply(&host_items()[1..2])]);
    let model = Model::new(
        "gpt-5.4",
        Arc::new(OpenAICompatibleProvider::new(config(), transport)),
    );
    let mut handle = suprnova_ai::stream(&model, vec![user("go")], with_tools(vec![shell]));
    while let Some(event) = handle.next().await {
        event.unwrap();
    }
    let message = handle.message().clone();
    assert!(message.tool_calls().is_empty(), "{message:?}");
    assert!(
        message
            .parts
            .iter()
            .any(|p| matches!(p, Part::BuiltinToolCall(c) if c.tool_name == "shell")),
        "{message:?}"
    );
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct ShellInput {
    #[allow(dead_code)]
    action: Value,
}

#[tokio::test]
async fn oai_010_an_agent_runs_a_provider_declared_tool() {
    use suprnova_ai::agents::{Agent, AgentTool};
    let shell =
        AgentTool::new::<ShellInput, Value, _, _>("local_shell", "Run a command", |_, _| async {
            Ok(json!({"output": "a.txt"}))
        })
        .unwrap()
        .with_provider_declaration(suprnova_ai::providers::openai::tools::local_shell())
        .unwrap();
    let transport = Recorder::new([items_reply(&host_items()[..1]), text_reply()]);
    let model = Model::new(
        "gpt-5.4",
        Arc::new(OpenAICompatibleProvider::new(config(), transport.clone())),
    );
    let mut run = Agent::new(vec![shell]).run(model, vec![user("list files")]);
    while let Some(event) = run.next().await {
        event.unwrap();
    }
    assert_eq!(transport.body(0)["tools"], json!([{"type": "local_shell"}]));
    let body = transport.body(1);
    assert_eq!(
        items(&body)[1..],
        [
            json!({"type": "item_reference", "id": "lsh_1"}),
            json!({"type": "local_shell_call_output", "call_id": "call_ls", "output": "a.txt"}),
        ],
        "{body}"
    );
    let named_otherwise =
        AgentTool::new::<ShellInput, Value, _, _>("shell_x", "x", |_, _| async { Ok(json!(null)) })
            .unwrap()
            .with_provider_declaration(suprnova_ai::providers::openai::tools::local_shell());
    assert!(named_otherwise.is_err());
}

#[test]
fn oai_010_calls_and_results_go_back_as_their_own_items() {
    let call = |id: &str, name: &str, args: Value| {
        let mut call = ToolCallPart::new(id, name, args.to_string());
        call.provider_metadata = md(json!({"item_id": format!("item_{id}")}));
        Part::ToolCall(call)
    };
    let result =
        |id: &str, name: &str, value: Value| Part::ToolResult(ToolResultPart::new(id, name, value));
    let messages = vec![
        user("go"),
        Message::new(
            MessageRole::Assistant,
            vec![
                call(
                    "call_ap",
                    "apply_patch",
                    json!({"call_id": "call_ap", "operation": {"type": "delete_file", "path": "a"}}),
                ),
                call(
                    "call_cu",
                    "computer",
                    json!({"actions": [], "pending_safety_checks": [], "status": "completed"}),
                ),
                call("call_ct", "sql", json!("select 1")),
                call("call_sh", "shell", json!({"action": {"commands": ["ls"]}})),
            ],
        ),
        Message::new(
            MessageRole::Tool,
            vec![
                result(
                    "call_ap",
                    "apply_patch",
                    json!({"status": "completed", "output": "done"}),
                ),
                result(
                    "call_cu",
                    "computer",
                    json!({"output": {"type": "computer_screenshot", "image_url": "data:image/png;base64,AA=="}}),
                ),
                result("call_ct", "sql", json!("1 row")),
                result(
                    "call_sh",
                    "shell",
                    json!({"output": [{"stdout": "a", "stderr": "", "outcome": {"type": "exit", "exit_code": 0}}]}),
                ),
            ],
        ),
    ];
    let options = with_tools(host_tools());
    let body = body_for("gpt-5.4", messages.clone(), &options);
    assert_eq!(references(&body, "item_call_ap"), 1, "{body}");
    assert_eq!(
        items(&body)[5..],
        [
            json!({"type": "apply_patch_call_output", "call_id": "call_ap", "status": "completed", "output": "done"}),
            json!({"type": "computer_call_output", "call_id": "call_cu", "output": {"type": "computer_screenshot", "image_url": "data:image/png;base64,AA=="}}),
            json!({"type": "custom_tool_call_output", "call_id": "call_ct", "output": "1 row"}),
            json!({"type": "shell_call_output", "call_id": "call_sh", "output": [{"stdout": "a", "stderr": "", "outcome": {"type": "exit", "exit_code": 0}}]}),
        ],
        "{body}"
    );
    let mut unstored = options.clone();
    unstored.params = store(Some(false)).params;
    let body = body_for("gpt-5.4", messages, &unstored);
    assert_eq!(
        items(&body)[1..5],
        [
            json!({"type": "apply_patch_call", "call_id": "call_ap", "id": "item_call_ap", "status": "completed", "operation": {"type": "delete_file", "path": "a"}}),
            json!({"type": "computer_call", "call_id": "call_cu", "id": "item_call_cu", "status": "completed", "actions": [], "pending_safety_checks": []}),
            json!({"type": "custom_tool_call", "call_id": "call_ct", "id": "item_call_ct", "name": "sql", "input": "select 1"}),
            json!({"type": "shell_call", "call_id": "call_sh", "id": "item_call_sh", "status": "completed", "action": {"commands": ["ls"]}}),
        ],
        "{body}"
    );
}

// OAI-011

/// Stream these Responses events, ended by `response.completed`, and give
/// the assistant message.
async fn read(events: Vec<Value>, tools: Vec<Tool>) -> Message {
    let mut events = events;
    events.push(json!({"type": "response.completed", "response": {"id": "resp_r", "model": "gpt-5.4", "status": "completed", "output": [], "usage": {"input_tokens": 1, "output_tokens": 1}}}));
    let transport = Recorder::new([responses_sse(&events)]);
    let model = Model::new(
        "gpt-5.4",
        Arc::new(OpenAICompatibleProvider::new(config(), transport)),
    );
    let mut handle = suprnova_ai::stream(&model, vec![user("hi")], with_tools(tools));
    while let Some(event) = handle.next().await {
        event.unwrap();
    }
    handle.message().clone()
}
fn added(index: usize, item: Value) -> Value {
    json!({"type": "response.output_item.added", "output_index": index, "item": item})
}
fn done(index: usize, item: Value) -> Value {
    json!({"type": "response.output_item.done", "output_index": index, "item": item})
}
/// A message item `msg_1` with text `T`, its phase, and these annotations
/// streamed as `annotation.added` events.
fn annotated_message(index: usize, phase: Option<&str>, annotations: &[Value]) -> Vec<Value> {
    let mut item = json!({"type": "message", "id": "msg_1", "role": "assistant", "content": [{"type": "output_text", "text": "T", "annotations": annotations}]});
    if let Some(phase) = phase {
        item["phase"] = json!(phase);
    }
    let mut events = vec![
        added(
            index,
            json!({"type": "message", "id": "msg_1", "role": "assistant", "content": []}),
        ),
        json!({"type": "response.output_text.delta", "item_id": "msg_1", "output_index": index, "delta": "T"}),
    ];
    for (n, annotation) in annotations.iter().enumerate() {
        events.push(json!({"type": "response.output_text.annotation.added", "item_id": "msg_1", "output_index": index, "content_index": 0, "annotation_index": n, "annotation": annotation}));
    }
    events.push(done(index, item));
    events
}
fn text_metadata(message: &Message) -> Value {
    message
        .parts
        .iter()
        .find_map(|p| match p {
            Part::Text(t) => t.provider_metadata.as_ref().map(|m| m["openai"].clone()),
            _ => None,
        })
        .unwrap_or_default()
}
fn sources(message: &Message) -> Vec<Value> {
    message
        .provider_metadata
        .as_ref()
        .and_then(|m| m["openai"]["sources"].as_array().cloned())
        .unwrap_or_default()
}

#[tokio::test]
async fn oai_011_annotations_and_sources_are_delivered_once() {
    let citation = json!({"type": "url_citation", "url": "https://a.test", "title": "A", "start_index": 0, "end_index": 1});
    let message = read(
        annotated_message(0, Some("commentary"), &[citation.clone(), citation.clone()]),
        vec![],
    )
    .await;
    let md = text_metadata(&message);
    assert_eq!(md["annotations"], json!([citation, citation]), "{md}");
    assert_eq!(md["phase"], "commentary", "{md}");
    let found = sources(&message);
    assert_eq!(found.len(), 1, "{found:?}");
    assert_eq!(found[0]["sourceType"], "url");
    assert_eq!(found[0]["url"], "https://a.test");
    assert_eq!(found[0]["title"], "A");
    // A web search that lists its sources takes the place of the citations.
    let search = json!({"type": "web_search_call", "id": "ws_1", "status": "completed",
        "action": {"type": "search", "query": "q", "sources": [{"type": "url", "url": "https://b.test"}]}});
    let mut events = vec![added(0, search.clone()), done(0, search)];
    events.extend(annotated_message(1, None, &[citation]));
    let found = sources(&read(events, vec![]).await);
    let urls: Vec<_> = found.iter().map(|s| s["url"].clone()).collect();
    assert_eq!(urls, [json!("https://b.test")], "{found:?}");
}

#[tokio::test]
async fn oai_011_program_and_compaction_items_are_built_in_calls() {
    let program = json!({"type": "program", "id": "pg_1", "call_id": "call_pg", "code": "print(1)", "fingerprint": "fp"});
    let output = json!({"type": "program_output", "id": "po_1", "call_id": "call_pg", "result": "1", "status": "completed"});
    let compaction = json!({"type": "compaction", "id": "cmp_1", "encrypted_content": "enc"});
    let message = read(
        vec![
            added(0, program.clone()),
            done(0, program),
            added(1, output.clone()),
            done(1, output),
            added(2, compaction.clone()),
            done(2, compaction),
        ],
        vec![],
    )
    .await;
    let calls: Vec<_> = message
        .parts
        .iter()
        .filter_map(|p| match p {
            Part::BuiltinToolCall(c) => Some(c.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(calls[0].tool_name, "programmatic_tool_calling");
    assert_eq!(
        serde_json::from_str::<Value>(&calls[0].tool_args).unwrap(),
        json!({"code": "print(1)", "fingerprint": "fp"})
    );
    let results: Vec<_> = message
        .parts
        .iter()
        .filter_map(|p| match p {
            Part::BuiltinToolReturn(r) => Some(r.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(results.len(), 1, "{message:?}");
    assert_eq!(results[0].tool_call_id, "call_pg");
    assert_eq!(
        results[0].result,
        json!({"result": "1", "status": "completed"})
    );
    assert_eq!(calls[1].tool_name, "compaction");
    let md = &calls[1].provider_metadata.as_ref().unwrap()["openai"];
    assert_eq!(md["item_id"], "cmp_1");
    assert_eq!(md["encrypted_content"], "enc");
}

#[tokio::test]
async fn oai_011_an_mcp_approval_request_is_a_pending_tool_approval_hook() {
    let request = json!({"type": "mcp_approval_request", "id": "mcpr_1", "server_label": "crm", "name": "delete", "arguments": "{}"});
    let message = read(vec![added(0, request.clone()), done(0, request)], vec![]).await;
    assert!(
        message
            .parts
            .iter()
            .any(|p| matches!(p, Part::BuiltinToolCall(c) if c.tool_name == "mcp.delete")),
        "{message:?}"
    );
    let hook = message
        .parts
        .iter()
        .find_map(|p| match p {
            Part::Hook(h) => Some(h.clone()),
            _ => None,
        })
        .unwrap_or_else(|| panic!("no hook: {message:?}"));
    assert_eq!(hook.hook_id, "mcpr_1");
    assert_eq!(hook.hook_type, "ToolApproval");
    assert_eq!(hook.status, HookStatus::Pending);
    assert!(
        !message
            .parts
            .iter()
            .any(|p| matches!(p, Part::BuiltinToolReturn(_))),
        "{message:?}"
    );
}

#[tokio::test]
async fn oai_011_a_failed_web_search_is_an_error_result() {
    let search = json!({"type": "web_search_call", "id": "ws_1", "status": "failed"});
    let message = read(vec![added(0, search.clone()), done(0, search)], vec![]).await;
    let result = message
        .parts
        .iter()
        .find_map(|p| match p {
            Part::BuiltinToolReturn(r) => Some(r.clone()),
            _ => None,
        })
        .unwrap();
    assert!(result.is_error, "{result:?}");
    assert_eq!(result.result, json!({"status": "failed"}));
}

#[tokio::test]
async fn oai_011_an_undeclared_parallel_call_is_one_call_per_tool_use() {
    let arguments = json!({"tool_uses": [
        {"recipient_name": "functions.weather", "parameters": {"city": "SF"}},
        {"recipient_name": "functions.weather", "parameters": {"city": "NYC"}},
    ]})
    .to_string();
    let call = json!({"type": "function_call", "id": "fc_p", "call_id": "c", "name": "parallel", "arguments": arguments});
    let mut started = call.clone();
    started["arguments"] = json!("");
    let message = read(
        vec![
            added(0, started),
            json!({"type": "response.function_call_arguments.delta", "item_id": "fc_p", "output_index": 0, "delta": arguments}),
            done(0, call),
        ],
        vec![function_tool("weather")],
    )
    .await;
    let calls: Vec<_> = message
        .tool_calls()
        .into_iter()
        .map(|c| {
            (
                c.tool_call_id.clone(),
                c.tool_name.clone(),
                c.tool_args.clone(),
            )
        })
        .collect();
    assert_eq!(
        calls,
        [
            ("c_0".into(), "weather".into(), r#"{"city":"SF"}"#.into()),
            ("c_1".into(), "weather".into(), r#"{"city":"NYC"}"#.into()),
        ]
    );
}

#[tokio::test]
async fn oai_011_a_stream_cut_after_text_fails_as_incomplete() {
    let events = vec![
        json!({"type": "response.created", "response": {"id": "resp_c", "model": "gpt-5.4", "status": "in_progress"}}),
        added(
            0,
            json!({"type": "message", "id": "msg_1", "role": "assistant", "content": []}),
        ),
        json!({"type": "response.output_text.delta", "item_id": "msg_1", "output_index": 0, "delta": "par"}),
    ];
    let transport = Recorder::new([responses_sse(&events)]);
    let model = Model::new(
        "gpt-5.4",
        Arc::new(OpenAICompatibleProvider::new(config(), transport)),
    );
    let options = StreamOptions {
        retry: suprnova_ai::RetryPolicy::none(),
        ..Default::default()
    };
    let Err(suprnova_ai::Error::Provider(error)) = run(&model, options).await else {
        panic!("a cut stream did not fail")
    };
    assert_eq!(
        error.kind,
        suprnova_ai::errors::ProviderErrorKind::IncompleteResponse
    );
}

// OAI-012

#[tokio::test]
async fn oai_012_a_chat_completions_stream_error_keeps_its_fields_and_kind() {
    let chunk = |error: Value| {
        (
            200,
            format!("data: {}\n\n", json!({"error": error})).into_bytes(),
        )
    };
    let chat = |transport: Arc<Recorder>| {
        let provider = OpenAICompatibleProvider::new(config(), transport)
            .with_protocol(Arc::new(OpenAIChatCompletionsProtocol));
        Model::new("gpt-4.1", Arc::new(provider))
    };
    let options = || StreamOptions {
        retry: suprnova_ai::RetryPolicy::none(),
        ..Default::default()
    };
    let transport = Recorder::new([chunk(
        json!({"message": "slow", "code": "rate_limit_exceeded", "type": "requests"}),
    )]);
    let Err(suprnova_ai::Error::Provider(error)) = run(&chat(transport), options()).await else {
        panic!("the error chunk did not end the call")
    };
    assert_eq!(error.message, "slow");
    assert_eq!(error.code.as_deref(), Some("rate_limit_exceeded"));
    assert_eq!(error.error_type.as_deref(), Some("requests"));
    assert_eq!(
        error.kind,
        suprnova_ai::errors::ProviderErrorKind::RateLimit
    );
    assert!(error.is_retryable);
    let transport = Recorder::new([chunk(
        json!({"message": "no credit", "code": "insufficient_quota"}),
    )]);
    let Err(suprnova_ai::Error::Provider(error)) = run(&chat(transport), options()).await else {
        panic!("the error chunk did not end the call")
    };
    assert!(!error.is_retryable, "{error:?}");
}

#[tokio::test]
async fn oai_012_a_responses_call_answered_with_chat_completions_names_the_other_protocol() {
    let transport = Recorder::new([chat_reply()]);
    let model = Model::new(
        "gpt-4.1",
        Arc::new(OpenAICompatibleProvider::new(config(), transport)),
    );
    let options = StreamOptions {
        retry: suprnova_ai::RetryPolicy::none(),
        ..Default::default()
    };
    let Err(suprnova_ai::Error::Provider(error)) = run(&model, options).await else {
        panic!("a Chat Completions answer did not fail")
    };
    assert_eq!(error.kind, suprnova_ai::errors::ProviderErrorKind::Response);
    assert!(
        error.message.contains("Chat Completions")
            && error.message.contains("OpenAIChatCompletionsProtocol"),
        "{}",
        error.message
    );
}

// OAI-013

/// Stream these Responses events, the last one terminal, and give the
/// stream handle.
async fn finished(events: Vec<Value>, options: StreamOptions) -> suprnova_ai::StreamHandle {
    let transport = Recorder::new([responses_sse(&events)]);
    let model = Model::new(
        "gpt-5.4",
        Arc::new(OpenAICompatibleProvider::new(config(), transport)),
    );
    let mut handle = suprnova_ai::stream(&model, vec![user("hi")], options);
    while let Some(event) = handle.next().await {
        event.unwrap();
    }
    handle
}

#[tokio::test]
async fn oai_013_usage_and_final_metadata_follow_typescript() {
    let handle = finished(
        vec![
            json!({"type": "response.created", "response": {"id": "resp_u", "model": "gpt-5.4", "status": "in_progress"}}),
            json!({"type": "response.output_item.added", "output_index": 0, "item": {"type": "message", "id": "msg_1", "role": "assistant", "content": []}}),
            json!({"type": "response.output_text.delta", "item_id": "msg_1", "output_index": 0, "delta": "hi",
                   "logprobs": [{"token": "hi", "logprob": -0.1, "top_logprobs": []}]}),
            json!({"type": "response.output_item.done", "output_index": 0, "item": {"type": "message", "id": "msg_1", "role": "assistant", "content": [{"type": "output_text", "text": "hi", "annotations": []}]}}),
            json!({"type": "response.completed", "response": {"id": "resp_u", "model": "gpt-5.4", "status": "completed",
                   "service_tier": "flex", "reasoning": {"context": "all_turns"}, "output": [],
                   "usage": {"input_tokens": 10, "input_tokens_details": {"cached_tokens": 4, "cache_write_tokens": 2},
                             "output_tokens": 20, "output_tokens_details": {"reasoning_tokens": 5}}}}),
        ],
        openai(json!({"logprobs": true})),
    )
    .await;
    let message = handle.message();
    let usage = message.usage.clone().unwrap();
    assert_eq!(
        (
            usage.input_tokens,
            usage.cache_read_tokens,
            usage.cache_write_tokens,
            usage.output_tokens,
            usage.reasoning_tokens
        ),
        (10, Some(4), Some(2), 20, Some(5))
    );
    let md = &message.provider_metadata.as_ref().unwrap()["openai"];
    assert_eq!(md["response_id"], "resp_u", "{md}");
    assert_eq!(md["service_tier"], "flex", "{md}");
    assert_eq!(md["reasoning_context"], "all_turns", "{md}");
    assert_eq!(
        md["logprobs"],
        json!([[{"token": "hi", "logprob": -0.1, "top_logprobs": []}]]),
        "{md}"
    );
    assert_eq!(handle.finish_reason(), Some("stop"));
}

#[tokio::test]
async fn oai_013_finish_reasons_follow_typescript() {
    let end = |status: &str, reason: Option<&str>, output: Value| {
        let mut response = json!({"id": "resp_f", "model": "gpt-5.4", "status": status, "output": output, "usage": {"input_tokens": 1, "output_tokens": 1}});
        if let Some(reason) = reason {
            response["incomplete_details"] = json!({"reason": reason});
        }
        let kind = if status == "incomplete" {
            "response.incomplete"
        } else {
            "response.completed"
        };
        vec![json!({"type": kind, "response": response})]
    };
    for (events, expected) in [
        (
            end("completed", None, json!([{"type": "function_call"}])),
            "tool_call",
        ),
        (
            end("incomplete", Some("max_output_tokens"), json!([])),
            "length",
        ),
        (
            end("incomplete", Some("content_filter"), json!([])),
            "content_filter",
        ),
        (end("incomplete", Some("other_reason"), json!([])), "other"),
        (end("completed", None, json!([])), "stop"),
    ] {
        let handle = finished(events, StreamOptions::default()).await;
        assert_eq!(handle.finish_reason(), Some(expected));
    }
}

// OAI-014

fn chat_body(model: &str, messages: Vec<Message>, options: &StreamOptions) -> Value {
    let request = build_request(model, &config(), messages, options, false).unwrap();
    serde_json::from_slice(&request.body).unwrap()
}

#[test]
fn oai_014_each_chat_option_goes_out_under_its_wire_name() {
    let cases = [
        (
            json!({"logitBias": {"50256": -100}}),
            "/logit_bias",
            json!({"50256": -100}),
        ),
        (json!({"logprobs": true}), "/top_logprobs", json!(0)),
        (json!({"logprobs": true}), "/logprobs", json!(true)),
        (json!({"logprobs": 3}), "/top_logprobs", json!(3)),
        (
            json!({"parallelToolCalls": false}),
            "/parallel_tool_calls",
            json!(false),
        ),
        (json!({"user": "u"}), "/user", json!("u")),
        (
            json!({"forceReasoning": true, "reasoningEffort": "low"}),
            "/reasoning_effort",
            json!("low"),
        ),
        (
            json!({"maxCompletionTokens": 50}),
            "/max_completion_tokens",
            json!(50),
        ),
        (json!({"store": false}), "/store", json!(false)),
        (
            json!({"metadata": {"a": "b"}}),
            "/metadata",
            json!({"a": "b"}),
        ),
        (
            json!({"prediction": {"type": "content", "content": "x"}}),
            "/prediction",
            json!({"type": "content", "content": "x"}),
        ),
        (
            json!({"serviceTier": "priority"}),
            "/service_tier",
            json!("priority"),
        ),
        (json!({"textVerbosity": "low"}), "/verbosity", json!("low")),
        (
            json!({"promptCacheKey": "k"}),
            "/prompt_cache_key",
            json!("k"),
        ),
        (
            json!({"promptCacheOptions": {"mode": "explicit", "ttl": "30m"}}),
            "/prompt_cache_options",
            json!({"mode": "explicit", "ttl": "30m"}),
        ),
        (
            json!({"promptCacheRetention": "24h"}),
            "/prompt_cache_retention",
            json!("24h"),
        ),
        (
            json!({"safetyIdentifier": "s"}),
            "/safety_identifier",
            json!("s"),
        ),
    ];
    for (options, pointer, expected) in cases {
        let body = chat_body("gpt-4.1", vec![user("hi")], &openai(options.clone()));
        assert_eq!(body.pointer(pointer), Some(&expected), "{options}: {body}");
    }
    let limited = StreamOptions {
        params: Some(InferenceRequestParams {
            output: Some(OutputParams {
                max_tokens: Some(64),
                ..Default::default()
            }),
            ..Default::default()
        }),
        ..Default::default()
    };
    let body = chat_body("gpt-5.4", vec![user("hi")], &limited);
    assert_eq!(body["max_completion_tokens"], 64, "{body}");
    assert_eq!(body.get("max_tokens"), None, "{body}");
}

#[tokio::test]
async fn oai_014_unknown_chat_options_and_a_reasoning_summary_fail_before_a_request() {
    for options in [
        json!({"reasoningSummary": "auto"}),
        json!({"reasoningEfort": "low"}),
        json!({"logprobs": null}),
    ] {
        let transport = Recorder::new([chat_reply()]);
        let provider = OpenAICompatibleProvider::new(config(), transport.clone())
            .with_protocol(Arc::new(OpenAIChatCompletionsProtocol));
        let model = Model::new("gpt-4.1", Arc::new(provider));
        let error = run(&model, openai(options.clone()))
            .await
            .expect_err("an invalid option");
        assert!(
            matches!(error, suprnova_ai::Error::Validation(_)),
            "{options}: {error:?}"
        );
        assert!(transport.requests.lock().unwrap().is_empty());
    }
}

#[test]
fn oai_014_search_preview_models_leave_out_a_temperature() {
    let (body, events) = logged(|| {
        chat_body(
            "gpt-4o-search-preview",
            vec![user("hi")],
            &temperature(None, json!({})),
        )
    });
    assert_eq!(body.get("temperature"), None, "{body}");
    assert!(
        events.contains(&("temperature".to_owned(), "gpt-4o-search-preview".to_owned())),
        "{events:?}"
    );
}

#[tokio::test]
async fn oai_014_citations_predictions_logprobs_and_audio_reach_the_message() {
    let chunks = [
        json!({"id": "c_1", "object": "chat.completion.chunk", "model": "gpt-4o", "choices": [{"index": 0,
               "delta": {"role": "assistant", "content": null, "audio": {"transcript": "said"}}, "finish_reason": null}]}),
        json!({"id": "c_1", "object": "chat.completion.chunk", "model": "gpt-4o", "choices": [{"index": 0,
               "delta": {"content": " more", "annotations": [{"type": "url_citation", "url_citation": {"url": "https://a.test", "title": "A", "start_index": 0, "end_index": 1}}]},
               "logprobs": {"content": [{"token": " more", "logprob": -0.2, "top_logprobs": []}]}, "finish_reason": null}]}),
        json!({"id": "c_1", "object": "chat.completion.chunk", "model": "gpt-4o", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
               "usage": {"prompt_tokens": 3, "completion_tokens": 4, "completion_tokens_details": {"accepted_prediction_tokens": 2, "rejected_prediction_tokens": 1}}}),
    ];
    let mut text: String = chunks.iter().map(|c| format!("data: {c}\n\n")).collect();
    text.push_str("data: [DONE]\n\n");
    let transport = Recorder::new([(200, text.into_bytes())]);
    let provider = OpenAICompatibleProvider::new(config(), transport)
        .with_protocol(Arc::new(OpenAIChatCompletionsProtocol));
    let model = Model::new("gpt-4o", Arc::new(provider));
    let mut handle = suprnova_ai::stream(&model, vec![user("hi")], StreamOptions::default());
    while let Some(event) = handle.next().await {
        event.unwrap();
    }
    let message = handle.message();
    assert_eq!(message.text(), "said more");
    let md = &message.provider_metadata.as_ref().unwrap()["openai"];
    assert_eq!(md["acceptedPredictionTokens"], 2, "{md}");
    assert_eq!(md["rejectedPredictionTokens"], 1, "{md}");
    assert_eq!(
        md["logprobs"],
        json!([{"token": " more", "logprob": -0.2, "top_logprobs": []}]),
        "{md}"
    );
    assert_eq!(md["sources"][0]["url"], "https://a.test", "{md}");
    assert_eq!(md["sources"].as_array().unwrap().len(), 1, "{md}");
}

#[test]
fn oai_005_chat_system_messages_follow_the_model_and_the_option() {
    let first = |model: &str, options: Value| {
        chat_body(model, vec![system("rules"), user("hi")], &openai(options))["messages"][0].clone()
    };
    assert_eq!(first("gpt-5.4", json!({}))["role"], "developer");
    assert_eq!(first("gpt-4.1", json!({}))["role"], "system");
    assert_eq!(
        first("gpt-5.4", json!({"systemMessageMode": "system"}))["role"],
        "system"
    );
    assert_eq!(
        first("gpt-4.1", json!({"forceReasoning": true}))["role"],
        "developer"
    );
}

#[test]
fn oai_005_chat_settings_the_model_does_not_take_are_left_out_and_logged() {
    let cases: Vec<(&str, StreamOptions, &str, &str)> = vec![
        (
            "o3",
            temperature(None, json!({})),
            "temperature",
            "/temperature",
        ),
        (
            "gpt-5.4",
            openai(json!({"logitBias": {"1": 1}})),
            "logitBias",
            "/logit_bias",
        ),
        (
            "gpt-5.4",
            openai(json!({"logprobs": 2})),
            "topLogprobs",
            "/top_logprobs",
        ),
        (
            "gpt-5.4",
            openai(json!({"logprobs": true})),
            "logprobs",
            "/logprobs",
        ),
        (
            "gpt-4.1",
            openai(json!({"reasoningEffort": "low"})),
            "reasoningEffort",
            "/reasoning_effort",
        ),
        (
            "gpt-6-astra",
            openai(json!({"reasoningEffort": "minimal"})),
            "reasoningEffort",
            "/reasoning_effort",
        ),
        (
            "gpt-4.1",
            openai(json!({"serviceTier": "flex"})),
            "serviceTier",
            "/service_tier",
        ),
    ];
    for (model, options, setting, pointer) in cases {
        let (body, events) = logged(|| chat_body(model, vec![user("hi")], &options));
        assert_eq!(body.pointer(pointer), None, "{model} {setting}: {body}");
        assert!(
            events.contains(&(setting.to_owned(), model.to_owned())),
            "{model} {setting}: {events:?}"
        );
    }
    let body = chat_body(
        "gpt-5.4",
        vec![user("hi")],
        &temperature(Some("none"), json!({})),
    );
    assert_eq!(body["temperature"], 0.5, "{body}");
}

// OAI-015

fn completions_model(
    transport: Arc<Recorder>,
    protocol: suprnova_ai::providers::openai::OpenAICompletionsProtocol,
    name: Option<&str>,
) -> Model {
    let provider = match name {
        None => OpenAICompatibleProvider::new(config(), transport),
        Some(name) => {
            let metadata: suprnova_ai::providers::ModelsDevProvider = serde_json::from_value(
                json!({"id": name, "name": name, "api": "https://local.test/v1", "env": [], "npm": "@ai-sdk/openai-compatible", "models": {}}),
            )
            .unwrap();
            OpenAICompatibleProvider::from_modelsdev_provider(
                &metadata,
                config_without_url(),
                transport,
            )
            .unwrap()
        }
    };
    Model::new("gpt-3.5-turbo-instruct", Arc::new(provider)).with_protocol(Arc::new(protocol))
}
fn config_without_url() -> ProviderConfig {
    ProviderConfig {
        api_key_value: Some("sk-test".into()),
        ..Default::default()
    }
}
fn completion_stream(texts: &[&str], finish: &str) -> (u16, Vec<u8>) {
    let mut chunks: Vec<Value> = texts
        .iter()
        .map(|t| json!({"id": "cmpl_1", "object": "text_completion", "model": "gpt-3.5-turbo-instruct", "choices": [{"index": 0, "text": t, "logprobs": null, "finish_reason": null}]}))
        .collect();
    chunks.push(json!({"id": "cmpl_1", "object": "text_completion", "model": "gpt-3.5-turbo-instruct", "choices": [{"index": 0, "text": "", "logprobs": null, "finish_reason": finish}]}));
    chunks.push(json!({"id": "cmpl_1", "object": "text_completion", "model": "gpt-3.5-turbo-instruct", "choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2}}));
    let mut text: String = chunks.iter().map(|c| format!("data: {c}\n\n")).collect();
    text.push_str("data: [DONE]\n\n");
    (200, text.into_bytes())
}

#[tokio::test]
async fn oai_015_a_history_becomes_one_prompt_with_the_options() {
    use suprnova_ai::providers::openai::OpenAICompletionsProtocol;
    let transport = Recorder::new([completion_stream(&["a", "b"], "length")]);
    let model = completions_model(transport.clone(), OpenAICompletionsProtocol::new(), None);
    let history = vec![
        system("rules"),
        user("hi"),
        Message::new(
            MessageRole::Assistant,
            vec![Part::Text(TextPart::new("hello"))],
        ),
        user("more"),
    ];
    let options = openai(
        json!({"echo": true, "logitBias": {"50256": -100}, "suffix": "!", "user": "u", "logprobs": true}),
    );
    let mut handle = suprnova_ai::stream(&model, history, options);
    while let Some(event) = handle.next().await {
        event.unwrap();
    }
    assert_eq!(url(&transport, 0), "https://api.openai.com/v1/completions");
    let body = transport.body(0);
    assert_eq!(
        body["prompt"],
        "rules\n\nuser:\nhi\n\nassistant:\nhello\n\nuser:\nmore\n\nassistant:\n"
    );
    assert_eq!(body["stop"], json!(["\nuser:"]));
    assert_eq!(body["stream_options"], json!({"include_usage": true}));
    for (key, value) in [
        ("echo", json!(true)),
        ("logit_bias", json!({"50256": -100})),
        ("suffix", json!("!")),
        ("user", json!("u")),
        ("logprobs", json!(0)),
    ] {
        assert_eq!(body[key], value, "{key}: {body}");
    }
    assert_eq!(handle.message().text(), "ab");
    assert_eq!(handle.finish_reason(), Some("length"));
    let usage = handle.message().usage.clone().unwrap();
    assert_eq!((usage.input_tokens, usage.output_tokens), (3, 2));
}

#[tokio::test]
async fn oai_015_tools_and_tool_history_fail_before_a_request() {
    use suprnova_ai::providers::openai::OpenAICompletionsProtocol;
    let cases = [
        (vec![user("hi")], with_tools(vec![function_tool("a")])),
        (vec![user("hi"), system("late")], StreamOptions::default()),
        (history(raw_call()), StreamOptions::default()),
        (vec![user("hi")], openai(json!({"unknown": 1}))),
    ];
    for (messages, options) in cases {
        let transport = Recorder::new([completion_stream(&["a"], "stop")]);
        let model = completions_model(transport.clone(), OpenAICompletionsProtocol::new(), None);
        let mut handle = suprnova_ai::stream(&model, messages, options);
        let mut error = None;
        while let Some(event) = handle.next().await {
            if let Err(e) = event {
                error = Some(e);
            }
        }
        assert!(
            matches!(error, Some(suprnova_ai::Error::Validation(_))),
            "{error:?}"
        );
        assert!(transport.requests.lock().unwrap().is_empty());
    }
}

#[tokio::test]
async fn oai_015_a_compatible_server_takes_its_own_options() {
    use suprnova_ai::providers::openai::OpenAICompletionsProtocol;
    for include_usage in [false, true] {
        let transport = Recorder::new([completion_stream(&["a"], "stop")]);
        let model = completions_model(
            transport.clone(),
            OpenAICompletionsProtocol::compatible(include_usage),
            Some("my-server"),
        );
        let options = StreamOptions {
            params: Some(
                InferenceRequestParams::default()
                    .with_provider_param(
                        "my-server",
                        json!({"suffix": "!", "logprobs": true, "top_a": 0.5}),
                    )
                    .with_provider_param("myServer", json!({"user": "u"})),
            ),
            ..Default::default()
        };
        run(&model, options).await.unwrap();
        let body = transport.body(0);
        assert_eq!(url(&transport, 0), "https://local.test/v1/completions");
        assert_eq!(body["suffix"], "!");
        assert_eq!(body["user"], "u");
        assert_eq!(body["top_a"], 0.5);
        assert_eq!(body.get("logprobs"), None, "{body}");
        assert_eq!(
            body.get("stream_options").is_some(),
            include_usage,
            "{body}"
        );
    }
}

// OAI-016

fn embeddings_reply(count: usize, tokens: u64) -> (u16, Vec<u8>) {
    let data: Vec<Value> = (0..count)
        .map(|index| json!({"object": "embedding", "index": index, "embedding": [0.5, index as f64]}))
        .collect();
    (
        200,
        json!({"object": "list", "data": data, "model": "text-embedding-3-small", "usage": {"prompt_tokens": tokens, "total_tokens": tokens}})
            .to_string()
            .into_bytes(),
    )
}
fn embed_options(key: &str, options: Value) -> suprnova_ai::providers::operations::EmbedParams {
    suprnova_ai::providers::operations::EmbedParams {
        provider_options: [(key.to_owned(), options)].into(),
    }
}

#[tokio::test]
async fn oai_016_embeddings_go_out_as_typescript_sends_them() {
    let transport = Recorder::new([embeddings_reply(2, 7)]);
    let model = Model::new(
        "text-embedding-3-small",
        Arc::new(OpenAICompatibleProvider::new(config(), transport.clone())),
    );
    let item = suprnova_ai::ops::embed(
        &model,
        vec!["a".into(), "b".into()],
        embed_options("openai", json!({"dimensions": 256, "user": "u"})),
        Default::default(),
    )
    .await
    .unwrap();
    assert_eq!(url(&transport, 0), "https://api.openai.com/v1/embeddings");
    assert_eq!(
        transport.body(0),
        json!({"model": "text-embedding-3-small", "input": ["a", "b"], "encoding_format": "float", "dimensions": 256, "user": "u"})
    );
    assert_eq!(item.value, vec![vec![0.5, 0.0], vec![0.5, 1.0]]);
    assert_eq!(item.usage.unwrap().input_tokens, 7);
    let error = suprnova_ai::ops::embed(
        &model,
        vec!["a".into()],
        embed_options("openai", json!({"dimension": 256})),
        Default::default(),
    )
    .await
    .unwrap_err();
    assert!(
        matches!(error, suprnova_ai::Error::Validation(_)),
        "{error:?}"
    );
}

#[tokio::test]
async fn oai_016_more_than_2048_values_split_into_calls() {
    let transport = Recorder::new([embeddings_reply(2048, 1), embeddings_reply(1, 1)]);
    let model = Model::new(
        "text-embedding-3-small",
        Arc::new(OpenAICompatibleProvider::new(config(), transport.clone())),
    );
    let values: Vec<String> = (0..2049).map(|n| n.to_string()).collect();
    let item = suprnova_ai::ops::embed(&model, values, Default::default(), Default::default())
        .await
        .unwrap();
    assert_eq!(item.value.len(), 2049);
    let mut sizes: Vec<usize> = (0..2)
        .map(|n| transport.body(n)["input"].as_array().unwrap().len())
        .collect();
    sizes.sort_unstable();
    assert_eq!(sizes, [1, 2048]);
}

#[tokio::test]
async fn oai_016_a_compatible_server_has_its_own_limit_and_metadata() {
    use suprnova_ai::providers::openai::OpenAIEmbeddingProtocol;
    let mut reply = embeddings_reply(1, 3);
    let mut body: Value = serde_json::from_slice(&reply.1).unwrap();
    body["providerMetadata"] = json!({"my-server": {"region": "eu"}});
    reply.1 = body.to_string().into_bytes();
    let transport = Recorder::new([reply.clone(), reply]);
    let metadata: suprnova_ai::providers::ModelsDevProvider = serde_json::from_value(
        json!({"id": "my-server", "name": "my-server", "api": "https://local.test/v1", "env": [], "npm": "@ai-sdk/openai-compatible", "models": {}}),
    )
    .unwrap();
    let provider = OpenAICompatibleProvider::from_modelsdev_provider(
        &metadata,
        config_without_url(),
        transport.clone(),
    )
    .unwrap();
    let model = Model::new("nomic-embed", Arc::new(provider))
        .with_protocol(Arc::new(OpenAIEmbeddingProtocol::compatible(1, false)));
    let item = suprnova_ai::ops::embed(
        &model,
        vec!["a".into(), "b".into()],
        embed_options("myServer", json!({"dimensions": 8})),
        Default::default(),
    )
    .await
    .unwrap();
    assert_eq!(transport.requests.lock().unwrap().len(), 2);
    assert_eq!(url(&transport, 0), "https://local.test/v1/embeddings");
    assert_eq!(transport.body(0)["dimensions"], 8);
    assert_eq!(item.value.len(), 2);
    assert_eq!(
        item.provider_metadata.unwrap()["my-server"],
        json!({"region": "eu"})
    );
}

// OAI-017

/// A download transport that answers every URL with these PNG bytes.
struct Downloads(Vec<u8>);
impl suprnova_ai::models::files::DownloadTransport for Downloads {
    fn resolve<'a>(
        &'a self,
        _host: &'a str,
        _port: u16,
    ) -> BoxFuture<'a, Result<Vec<std::net::IpAddr>, suprnova_ai::models::files::DownloadError>>
    {
        Box::pin(async { Ok(vec!["93.184.216.34".parse().unwrap()]) })
    }
    fn send_pinned(
        &self,
        _request: HttpRequest,
        _addresses: Vec<std::net::IpAddr>,
    ) -> BoxFuture<'_, Result<HttpResponse, suprnova_ai::models::files::DownloadError>> {
        let bytes = self.0.clone();
        Box::pin(async move {
            Ok(HttpResponse {
                status: 200,
                headers: vec![("content-type".into(), "image/png".into())],
                body: stream::iter([Ok(Bytes::from(bytes))]).boxed(),
            })
        })
    }
}
fn images_reply(count: usize, image_tokens: u64) -> (u16, Vec<u8>) {
    let data: Vec<Value> = (0..count)
        .map(|n| json!({"b64_json": "AQID", "revised_prompt": format!("revised {n}")}))
        .collect();
    (
        200,
        json!({"created": 1, "size": "1024x1024", "quality": "high", "background": "opaque", "output_format": "png",
               "data": data, "usage": {"input_tokens": 5, "output_tokens": 9, "input_tokens_details": {"image_tokens": image_tokens, "text_tokens": 4}}})
            .to_string()
            .into_bytes(),
    )
}
fn image_model(id: &str, transport: Arc<Recorder>) -> Model {
    Model::new(
        id,
        Arc::new(OpenAICompatibleProvider::new(config(), transport)),
    )
}
fn text_prompt(text: &str) -> suprnova_ai::providers::operations::ImagePrompt {
    suprnova_ai::providers::operations::ImagePrompt {
        text: Some(text.into()),
        ..Default::default()
    }
}
fn image_params(n: u32, options: Value) -> suprnova_ai::providers::operations::ImageParams {
    suprnova_ai::providers::operations::ImageParams {
        n,
        provider_options: [("openai".to_owned(), options)].into(),
        ..Default::default()
    }
}

#[tokio::test]
async fn oai_017_generations_send_the_options_and_b64_json_where_needed() {
    let transport = Recorder::new([images_reply(1, 4), images_reply(1, 4)]);
    let options = json!({"quality": "hd", "style": "vivid", "background": "opaque", "moderation": "low",
                         "outputFormat": "webp", "outputCompression": 80, "user": "u"});
    let item = suprnova_ai::ops::generate_image(
        &image_model("dall-e-3", transport.clone()),
        text_prompt("a cat"),
        image_params(1, options),
        Default::default(),
    )
    .await
    .unwrap();
    assert_eq!(
        url(&transport, 0),
        "https://api.openai.com/v1/images/generations"
    );
    assert_eq!(
        transport.body(0),
        json!({"model": "dall-e-3", "prompt": "a cat", "n": 1, "quality": "hd", "style": "vivid", "background": "opaque",
               "moderation": "low", "output_format": "webp", "output_compression": 80, "user": "u", "response_format": "b64_json"})
    );
    assert_eq!(item.value[0].media_type, "image/png");
    let md = &item.provider_metadata.as_ref().unwrap()["openai"]["images"][0];
    assert_eq!(md["revisedPrompt"], "revised 0");
    assert_eq!(md["size"], "1024x1024");
    assert_eq!(md["outputFormat"], "png");
    suprnova_ai::ops::generate_image(
        &image_model("gpt-image-1", transport.clone()),
        text_prompt("a cat"),
        image_params(1, json!({})),
        Default::default(),
    )
    .await
    .unwrap();
    assert_eq!(transport.body(1).get("response_format"), None);
}

#[tokio::test]
async fn oai_017_calls_follow_each_models_image_limit() {
    let transport = Recorder::new([images_reply(1, 0), images_reply(1, 0), images_reply(1, 0)]);
    let item = suprnova_ai::ops::generate_image(
        &image_model("dall-e-3", transport.clone()),
        text_prompt("a cat"),
        image_params(3, json!({})),
        Default::default(),
    )
    .await
    .unwrap();
    assert_eq!(transport.requests.lock().unwrap().len(), 3);
    assert_eq!(item.value.len(), 3);
}

#[tokio::test]
async fn oai_017_an_edit_sends_its_images_as_a_form() {
    use suprnova_ai::providers::operations::{ImagePrompt, MediaInput};
    let transport = Recorder::new([images_reply(2, 101)]);
    let provider = OpenAICompatibleProvider::new(config(), transport.clone())
        .with_download_transport(Arc::new(Downloads(vec![9, 9])));
    let model = Model::new("gpt-image-1", Arc::new(provider));
    let prompt = ImagePrompt {
        text: Some("add a hat".into()),
        images: vec![
            MediaInput::Bytes(vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]),
            MediaInput::Url("https://img.test/b.png".into()),
        ],
        mask: None,
    };
    let item = suprnova_ai::ops::generate_image(
        &model,
        prompt,
        image_params(2, json!({"inputFidelity": "high"})),
        Default::default(),
    )
    .await
    .unwrap();
    assert_eq!(url(&transport, 0), "https://api.openai.com/v1/images/edits");
    let form = String::from_utf8_lossy(&transport.requests.lock().unwrap()[0].body).into_owned();
    assert_eq!(form.matches("name=\"image[]\"").count(), 2, "{form}");
    assert!(
        form.contains("name=\"input_fidelity\"\r\n\r\nhigh\r\n"),
        "{form}"
    );
    assert!(
        header(&transport, 0, "content-type")
            .unwrap()
            .starts_with("multipart/form-data; boundary="),
    );
    let images = &item.provider_metadata.as_ref().unwrap()["openai"]["images"];
    assert_eq!(images[0]["imageTokens"], 50);
    assert_eq!(images[1]["imageTokens"], 51);
    assert_eq!(images[0]["textTokens"], 2);
}

#[tokio::test]
async fn oai_017_settings_openai_does_not_take_fail_before_a_request() {
    use suprnova_ai::providers::operations::{ImageParams, ImagePrompt, MediaInput};
    let url_edit = || ImagePrompt {
        text: Some("x".into()),
        images: vec![MediaInput::Url("https://img.test/a.png".into())],
        mask: None,
    };
    let cases = [
        (
            "gpt-image-1",
            text_prompt("x"),
            ImageParams {
                aspect_ratio: Some("16:9".into()),
                ..Default::default()
            },
        ),
        (
            "gpt-image-1",
            text_prompt("x"),
            ImageParams {
                seed: Some(1),
                ..Default::default()
            },
        ),
        ("dall-e-3", url_edit(), ImageParams::default()),
        ("gpt-image-1", url_edit(), ImageParams::default()),
        (
            "gpt-image-1",
            text_prompt("x"),
            image_params(1, json!({"qualty": "hd"})),
        ),
    ];
    for (id, prompt, params) in cases {
        let transport = Recorder::new([images_reply(1, 0)]);
        let error = suprnova_ai::ops::generate_image(
            &image_model(id, transport.clone()),
            prompt,
            params,
            Default::default(),
        )
        .await
        .unwrap_err();
        assert!(
            matches!(error, suprnova_ai::Error::Validation(_)),
            "{id}: {error:?}"
        );
        assert!(transport.requests.lock().unwrap().is_empty(), "{id}");
    }
}

// OAI-018

fn speech_prompt(text: &str) -> suprnova_ai::providers::operations::AudioPrompt {
    suprnova_ai::providers::operations::AudioPrompt {
        text: text.into(),
        instructions: None,
    }
}

#[tokio::test]
async fn oai_018_speech_goes_out_as_typescript_sends_it() {
    use suprnova_ai::providers::operations::AudioParams;
    let transport = Recorder::new([(200, vec![1, 2, 3]), (200, vec![4])]);
    let model = image_model("gpt-4o-mini-tts", transport.clone());
    let item = suprnova_ai::ops::generate_audio(
        &model,
        speech_prompt("hello"),
        AudioParams::default(),
        Default::default(),
    )
    .await
    .unwrap();
    assert_eq!(url(&transport, 0), "https://api.openai.com/v1/audio/speech");
    assert_eq!(
        transport.body(0),
        json!({"model": "gpt-4o-mini-tts", "input": "hello", "voice": "alloy", "response_format": "mp3"})
    );
    assert_eq!(item.value[0].media_type, "audio/mpeg");
    assert_eq!(
        item.value[0].data,
        suprnova_ai::types::messages::FileData::Bytes(vec![1, 2, 3])
    );
    let mut prompt = speech_prompt("hi");
    prompt.instructions = Some("calm".into());
    let item = suprnova_ai::ops::generate_audio(
        &model,
        prompt,
        AudioParams {
            voice: Some("nova".into()),
            output_format: Some("wav".into()),
            speed: Some(1.5),
            ..Default::default()
        },
        Default::default(),
    )
    .await
    .unwrap();
    assert_eq!(
        transport.body(1),
        json!({"model": "gpt-4o-mini-tts", "input": "hi", "voice": "nova", "response_format": "wav", "speed": 1.5, "instructions": "calm"})
    );
    assert_eq!(item.value[0].media_type, "audio/wav");
}

#[tokio::test]
async fn oai_018_another_format_or_a_language_fails_before_a_request() {
    use suprnova_ai::providers::operations::AudioParams;
    for params in [
        AudioParams {
            output_format: Some("ogg".into()),
            ..Default::default()
        },
        AudioParams {
            language: Some("fr".into()),
            ..Default::default()
        },
        AudioParams {
            provider_options: [("openai".to_owned(), json!({"speed": 9}))].into(),
            ..Default::default()
        },
    ] {
        let transport = Recorder::new([(200, vec![1])]);
        let error = suprnova_ai::ops::generate_audio(
            &image_model("tts-1", transport.clone()),
            speech_prompt("x"),
            params,
            Default::default(),
        )
        .await
        .unwrap_err();
        assert!(
            matches!(error, suprnova_ai::Error::Validation(_)),
            "{error:?}"
        );
        assert!(transport.requests.lock().unwrap().is_empty());
    }
}

// OAI-019

fn transcription_reply(body: Value) -> (u16, Vec<u8>) {
    (200, body.to_string().into_bytes())
}
fn mp3() -> suprnova_ai::providers::operations::TranscribeInput {
    suprnova_ai::providers::operations::TranscribeInput::File(
        suprnova_ai::types::messages::FilePart::from_bytes(vec![1, 2, 3], Some("audio/mpeg"), None)
            .unwrap(),
    )
}
fn transcribe_params(options: Value) -> suprnova_ai::providers::operations::TranscribeParams {
    suprnova_ai::providers::operations::TranscribeParams {
        provider_options: [("openai".to_owned(), options)].into(),
    }
}
fn form(transport: &Recorder, index: usize) -> String {
    String::from_utf8_lossy(&transport.requests.lock().unwrap()[index].body).into_owned()
}
fn field(name: &str, value: &str) -> String {
    format!("name=\"{name}\"\r\n\r\n{value}\r\n")
}

#[tokio::test]
async fn oai_019_a_transcription_form_follows_the_model_and_options() {
    let transport = Recorder::new([
        transcription_reply(json!({"text": "hi", "language": "english", "duration": 1.5,
            "segments": [{"id": 0, "start": 0.0, "end": 1.5, "text": "hi"}]})),
        transcription_reply(
            json!({"text": "hi", "segments": [{"start": 0.0, "end": 1.0, "text": "hi", "speaker": "A"}],
            "languages": [{"code": "en"}]}),
        ),
        transcription_reply(
            json!({"text": "hi", "words": [{"word": "hi", "start": 0.0, "end": 0.5}]}),
        ),
    ]);
    let item = suprnova_ai::ops::transcribe(
        &image_model("whisper-1", transport.clone()),
        mp3(),
        transcribe_params(json!({"timestampGranularities": ["word"], "language": "en", "prompt": "p", "temperature": 0.2, "include": ["logprobs"]})),
        Default::default(),
    )
    .await
    .unwrap();
    assert_eq!(
        url(&transport, 0),
        "https://api.openai.com/v1/audio/transcriptions"
    );
    let sent = form(&transport, 0);
    for (name, value) in [
        ("model", "whisper-1"),
        ("response_format", "verbose_json"),
        ("timestamp_granularities[]", "word"),
        ("language", "en"),
        ("prompt", "p"),
        ("temperature", "0.2"),
        ("include[]", "logprobs"),
    ] {
        assert!(sent.contains(&field(name, value)), "{name}: {sent}");
    }
    assert!(
        sent.contains("name=\"file\"; filename=\"audio.mp3\""),
        "{sent}"
    );
    assert_eq!(item.value.text, "hi");
    assert_eq!(item.value.language.as_deref(), Some("en"));
    assert_eq!(item.value.duration_seconds, Some(1.5));
    assert_eq!(item.value.segments.len(), 1);
    let item = suprnova_ai::ops::transcribe(
        &image_model("gpt-4o-transcribe-diarize", transport.clone()),
        mp3(),
        Default::default(),
        Default::default(),
    )
    .await
    .unwrap();
    let sent = form(&transport, 1);
    assert!(
        sent.contains(&field("response_format", "diarized_json")),
        "{sent}"
    );
    assert!(sent.contains(&field("chunking_strategy", "auto")), "{sent}");
    let md = &item.provider_metadata.as_ref().unwrap()["openai"];
    assert_eq!(md["segments"][0]["speaker"], "A", "{md}");
    assert_eq!(md["languages"], json!([{"code": "en"}]));
    assert_eq!(item.value.language.as_deref(), Some("en"));
    let item = suprnova_ai::ops::transcribe(
        &image_model("gpt-4o-transcribe", transport.clone()),
        mp3(),
        Default::default(),
        Default::default(),
    )
    .await
    .unwrap();
    assert!(form(&transport, 2).contains(&field("response_format", "json")));
    assert_eq!(item.value.segments[0].text, "hi");
    assert_eq!(item.value.segments[0].end_second, 0.5);
}

#[tokio::test]
async fn oai_019_a_realtime_model_or_a_bad_option_fails_before_a_request() {
    for (id, params) in [
        ("gpt-realtime-whisper", Default::default()),
        (
            "whisper-1",
            transcribe_params(json!({"timestampGranularities": ["sentence"]})),
        ),
        ("whisper-1", transcribe_params(json!({"temprature": 0}))),
    ] {
        let transport = Recorder::new([transcription_reply(json!({"text": "x"}))]);
        let error = suprnova_ai::ops::transcribe(
            &image_model(id, transport.clone()),
            mp3(),
            params,
            Default::default(),
        )
        .await
        .unwrap_err();
        assert!(
            matches!(error, suprnova_ai::Error::Validation(_)),
            "{id}: {error:?}"
        );
        assert!(transport.requests.lock().unwrap().is_empty(), "{id}");
    }
}

// OAI-020

mod google_support;
use google_support::{ScriptedSocket, text_message};
use suprnova_ai::providers::operations::{
    AudioFormat, StreamTranscribeParams, StreamTranslateParams, TranscriptionPart, TranslationPart,
};

fn socket_model(id: &str, socket: Arc<ScriptedSocket>) -> Model {
    let provider = OpenAICompatibleProvider::new(org_config(), Recorder::new([]))
        .with_websocket_transport(socket);
    Model::new(id, Arc::new(provider))
}
fn chunks() -> suprnova_ai::providers::operations::AudioChunks {
    stream::iter([vec![1u8, 2]]).boxed()
}

#[tokio::test]
async fn oai_020_streaming_transcription_speaks_the_realtime_socket() {
    let socket = ScriptedSocket::new(vec![
        (
            3,
            text_message(
                json!({"type": "conversation.item.input_audio_transcription.delta", "item_id": "it_1", "delta": "he"}),
            ),
        ),
        (
            3,
            text_message(
                json!({"type": "conversation.item.input_audio_transcription.completed", "item_id": "it_1", "transcript": "hello"}),
            ),
        ),
    ]);
    let params = StreamTranscribeParams {
        provider_options: [("openai".to_owned(), json!({"language": "en", "streaming": {"delay": "low", "include": ["item.input_audio_transcription.logprobs"]}}))].into(),
        ..Default::default()
    };
    let parts: Vec<TranscriptionPart> = suprnova_ai::ops::stream_transcription(
        &socket_model("gpt-realtime-whisper", socket.clone()),
        chunks(),
        AudioFormat::pcm(24_000),
        params,
        Default::default(),
    )
    .await
    .unwrap()
    .map(Result::unwrap)
    .collect()
    .await;
    let request = socket.request();
    assert_eq!(
        request.url,
        "wss://api.openai.com/v1/realtime?intent=transcription"
    );
    let has = |name: &str, value: &str| {
        request
            .headers
            .iter()
            .any(|(k, v)| k.eq_ignore_ascii_case(name) && v == value)
    };
    assert!(has("authorization", "Bearer sk-test"));
    assert!(has("OpenAI-Organization", "org_1"));
    let sent = socket.sent_json();
    assert_eq!(
        sent[0],
        json!({"type": "session.update", "session": {"type": "transcription", "audio": {"input": {
            "format": {"type": "audio/pcm", "rate": 24000},
            "transcription": {"model": "gpt-realtime-whisper", "language": "en", "delay": "low"},
            "turn_detection": null}}, "include": ["item.input_audio_transcription.logprobs"]}})
    );
    assert_eq!(
        sent[1],
        json!({"type": "input_audio_buffer.append", "audio": "AQI="})
    );
    assert_eq!(sent[2], json!({"type": "input_audio_buffer.commit"}));
    assert!(matches!(parts[0], TranscriptionPart::Start { .. }));
    assert_eq!(
        parts[1],
        TranscriptionPart::Delta {
            id: Some("it_1".into()),
            delta: "he".into()
        }
    );
    assert_eq!(
        parts[2],
        TranscriptionPart::Final {
            id: Some("it_1".into()),
            text: "hello".into()
        }
    );
    let TranscriptionPart::Finish { text, language, .. } = &parts[3] else {
        panic!("no finish: {parts:?}")
    };
    assert_eq!((text.as_str(), language.as_deref()), ("hello", Some("en")));
}

#[tokio::test]
async fn oai_020_speech_translation_speaks_the_translation_socket() {
    let socket = ScriptedSocket::new(vec![
        (
            3,
            text_message(json!({"type": "session.input_transcript.delta", "delta": "hi"})),
        ),
        (
            3,
            text_message(json!({"type": "session.output_transcript.delta", "delta": "hola"})),
        ),
        (
            3,
            text_message(json!({"type": "session.output_audio.delta", "delta": "AQI="})),
        ),
        (3, text_message(json!({"type": "session.closed"}))),
    ]);
    let params = StreamTranslateParams {
        target_language: "es".into(),
        ..Default::default()
    };
    let parts: Vec<TranslationPart> = suprnova_ai::ops::stream_translation(
        &socket_model("gpt-realtime-translate", socket.clone()),
        chunks(),
        AudioFormat::pcm(24_000),
        params,
        Default::default(),
    )
    .await
    .unwrap()
    .map(Result::unwrap)
    .collect()
    .await;
    assert_eq!(
        socket.request().url,
        "wss://api.openai.com/v1/realtime/translations?model=gpt-realtime-translate"
    );
    let sent = socket.sent_json();
    assert_eq!(
        sent[0],
        json!({"type": "session.update", "session": {"audio": {
            "input": {"transcription": {"model": "gpt-realtime-whisper"}, "noise_reduction": null},
            "output": {"language": "es"}}}})
    );
    assert_eq!(
        sent[1],
        json!({"type": "session.input_audio_buffer.append", "audio": "AQI="})
    );
    assert_eq!(sent[2], json!({"type": "session.close"}));
    assert_eq!(
        parts[1..],
        [
            TranslationPart::SourceTranscriptDelta {
                id: None,
                delta: "hi".into()
            },
            TranslationPart::OutputTextDelta {
                id: None,
                delta: "hola".into()
            },
            TranslationPart::Audio {
                id: None,
                audio: vec![1, 2]
            },
            TranslationPart::SourceTranscriptFinal {
                id: None,
                text: "hi".into()
            },
            TranslationPart::OutputTextFinal {
                id: None,
                text: "hola".into()
            },
            TranslationPart::Finish {
                source_text: "hi".into(),
                output_text: "hola".into(),
                duration_seconds: None,
                usage: None,
                provider_metadata: None,
            },
        ]
    );
}

#[tokio::test]
async fn oai_020_settings_the_sockets_do_not_take_fail_before_connecting() {
    let socket = ScriptedSocket::new(vec![]);
    let transcription = |id: &str, options: Value| {
        let socket = socket.clone();
        let id = id.to_owned();
        async move {
            suprnova_ai::ops::stream_transcription(
                &socket_model(&id, socket),
                chunks(),
                AudioFormat::pcm(24_000),
                StreamTranscribeParams {
                    provider_options: [("openai".to_owned(), options)].into(),
                    ..Default::default()
                },
                Default::default(),
            )
            .await
            .err()
        }
    };
    for error in [
        transcription("whisper-1", json!({})).await,
        transcription("gpt-realtime-whisper", json!({"prompt": "p"})).await,
        transcription("gpt-realtime-whisper", json!({"temperature": 0.1})).await,
    ] {
        assert!(
            matches!(error, Some(suprnova_ai::Error::Validation(_))),
            "{error:?}"
        );
    }
    for (format, params) in [
        (
            AudioFormat::pcm(16_000),
            StreamTranslateParams {
                target_language: "es".into(),
                ..Default::default()
            },
        ),
        (
            AudioFormat::pcm(24_000),
            StreamTranslateParams {
                target_language: "es".into(),
                source_language: Some("en".into()),
                ..Default::default()
            },
        ),
    ] {
        let error = suprnova_ai::ops::stream_translation(
            &socket_model("gpt-realtime-translate", socket.clone()),
            chunks(),
            format,
            params,
            Default::default(),
        )
        .await
        .err();
        assert!(
            matches!(error, Some(suprnova_ai::Error::Validation(_))),
            "{error:?}"
        );
    }
    assert!(socket.requests.lock().unwrap().is_empty());
}

// OAI-021

fn file_object() -> Value {
    json!({"id": "file-1", "object": "file", "bytes": 4, "created_at": 1_800_000_000u64, "expires_at": 1_800_003_600u64,
           "filename": "r.pdf", "purpose": "assistants", "status": "processed"})
}
fn json_reply(value: Value) -> (u16, Vec<u8>) {
    (200, value.to_string().into_bytes())
}
fn openai_reference() -> std::collections::BTreeMap<String, String> {
    [("openai".to_owned(), "file-1".to_owned())].into()
}

#[tokio::test]
async fn oai_021_files_are_uploaded_read_downloaded_and_deleted() {
    use suprnova_ai::providers::operations::{UploadFile, UploadParams};
    let transport = Recorder::new([
        json_reply(file_object()),
        json_reply(file_object()),
        (200, b"%PDF".to_vec()),
        json_reply(json!({"id": "file-1", "object": "file", "deleted": true})),
    ]);
    transport.reply_headers(2, &[("content-type", "application/pdf; charset=binary")]);
    let provider = OpenAICompatibleProvider::new(config(), transport.clone());
    let file = UploadFile {
        data: Bytes::from_static(b"%PDF"),
        media_type: "application/pdf".into(),
        filename: Some("r.pdf".into()),
    };
    let params = UploadParams {
        provider_options: [("openai".to_owned(), json!({"expiresAfter": 3600}))].into(),
    };
    let uploaded = suprnova_ai::ops::upload_file(&provider, file, params, Default::default())
        .await
        .unwrap();
    assert_eq!(url(&transport, 0), "https://api.openai.com/v1/files");
    let sent = form(&transport, 0);
    assert!(sent.contains(&field("purpose", "assistants")), "{sent}");
    assert!(
        sent.contains(&field("expires_after[anchor]", "created_at")),
        "{sent}"
    );
    assert!(
        sent.contains(&field("expires_after[seconds]", "3600")),
        "{sent}"
    );
    assert!(
        sent.contains(
            "name=\"file\"; filename=\"r.pdf\"\r\nContent-Type: application/pdf\r\n\r\n%PDF"
        ),
        "{sent}"
    );
    let value = &uploaded.value;
    assert_eq!(value.provider_reference, openai_reference());
    assert_eq!(value.media_type, None);
    assert_eq!(value.filename.as_deref(), Some("r.pdf"));
    assert_eq!(value.size_bytes, Some(4));
    assert_eq!(
        value.created_at.as_deref(),
        Some("2027-01-15T08:00:00.000Z")
    );
    assert_eq!(
        value.expires_at.as_deref(),
        Some("2027-01-15T09:00:00.000Z")
    );
    let md = &uploaded.provider_metadata.as_ref().unwrap()["openai"];
    assert_eq!(
        (md["purpose"].clone(), md["status"].clone()),
        (json!("assistants"), json!("processed"))
    );
    let info = suprnova_ai::ops::file_info(&provider, &openai_reference(), Default::default())
        .await
        .unwrap();
    assert_eq!(url(&transport, 1), "https://api.openai.com/v1/files/file-1");
    assert_eq!(info.value.filename.as_deref(), Some("r.pdf"));
    let downloaded =
        suprnova_ai::ops::download_file(&provider, &openai_reference(), Default::default())
            .await
            .unwrap();
    assert_eq!(
        url(&transport, 2),
        "https://api.openai.com/v1/files/file-1/content"
    );
    assert_eq!(downloaded.value.data, Bytes::from_static(b"%PDF"));
    assert_eq!(
        downloaded.value.media_type.as_deref(),
        Some("application/pdf")
    );
    let deleted = suprnova_ai::ops::delete_file(&provider, &openai_reference(), Default::default())
        .await
        .unwrap();
    assert_eq!(transport.requests.lock().unwrap()[3].method, "DELETE");
    assert!(deleted.value);
}

#[tokio::test]
async fn oai_021_a_reference_without_an_openai_id_fails_before_a_request() {
    let transport = Recorder::new([]);
    let provider = OpenAICompatibleProvider::new(config(), transport.clone());
    let reference: std::collections::BTreeMap<String, String> =
        [("google".to_owned(), "files/x".to_owned())].into();
    for error in [
        suprnova_ai::ops::file_info(&provider, &reference, Default::default())
            .await
            .err(),
        suprnova_ai::ops::download_file(&provider, &reference, Default::default())
            .await
            .err(),
        suprnova_ai::ops::delete_file(&provider, &reference, Default::default())
            .await
            .err(),
    ] {
        assert!(
            matches!(error, Some(suprnova_ai::Error::Validation(_))),
            "{error:?}"
        );
    }
    assert!(transport.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn oai_021_an_id_that_names_no_path_segment_fails_before_a_request() {
    // A URL path drops a `.` or `..` segment, so such an id would send the
    // call to another endpoint.
    let transport = Recorder::new([]);
    let provider = OpenAICompatibleProvider::new(config(), transport.clone());
    for id in ["", " ", ".", ".."] {
        let reference: std::collections::BTreeMap<String, String> =
            [("openai".to_owned(), id.to_owned())].into();
        for error in [
            suprnova_ai::ops::file_info(&provider, &reference, Default::default())
                .await
                .err(),
            suprnova_ai::ops::download_file(&provider, &reference, Default::default())
                .await
                .err(),
            suprnova_ai::ops::delete_file(&provider, &reference, Default::default())
                .await
                .err(),
            suprnova_ai::ops::batch::status(&provider, id, Default::default())
                .await
                .err(),
            suprnova_ai::ops::batch::cancel(&provider, id, Default::default())
                .await
                .err(),
            suprnova_ai::ops::batch::results(&provider, id, Default::default())
                .await
                .err(),
        ] {
            assert!(
                matches!(error, Some(suprnova_ai::Error::Validation(_))),
                "{id:?}: {error:?}"
            );
        }
    }
    assert!(transport.requests.lock().unwrap().is_empty());
}

// OAI-022

#[tokio::test]
async fn oai_022_a_skill_uploads_its_files_named_by_their_paths() {
    use suprnova_ai::providers::operations::{SkillFile, SkillParams};
    let transport = Recorder::new([json_reply(
        json!({"id": "skill_1", "name": "notes", "description": "Takes notes",
        "default_version": "1", "latest_version": "2", "created_at": 1_800_000_000u64, "updated_at": 1_800_000_100u64}),
    )]);
    let provider = OpenAICompatibleProvider::new(config(), transport.clone());
    let files = vec![
        SkillFile {
            path: "notes/SKILL.md".into(),
            data: Bytes::from_static(b"# Notes"),
        },
        SkillFile {
            path: "notes/run.py".into(),
            data: Bytes::from_static(b"print(1)"),
        },
    ];
    let uploaded = suprnova_ai::ops::upload_skill(
        &provider,
        files,
        SkillParams::default(),
        Default::default(),
    )
    .await
    .unwrap();
    assert_eq!(url(&transport, 0), "https://api.openai.com/v1/skills");
    let sent = form(&transport, 0);
    assert_eq!(sent.matches("name=\"files[]\"").count(), 2, "{sent}");
    assert!(
        sent.contains("name=\"files[]\"; filename=\"notes/SKILL.md\""),
        "{sent}"
    );
    assert!(
        sent.contains("name=\"files[]\"; filename=\"notes/run.py\""),
        "{sent}"
    );
    let value = &uploaded.value;
    assert_eq!(
        value.provider_reference,
        [("openai".to_owned(), "skill_1".to_owned())].into()
    );
    assert_eq!(value.name.as_deref(), Some("notes"));
    assert_eq!(value.description.as_deref(), Some("Takes notes"));
    assert_eq!(value.latest_version.as_deref(), Some("2"));
    assert_eq!(
        uploaded.provider_metadata.unwrap()["openai"],
        json!({"defaultVersion": "1", "createdAt": 1_800_000_000u64, "updatedAt": 1_800_000_100u64})
    );
    let titled = SkillParams {
        display_title: Some("Notes".into()),
        ..Default::default()
    };
    let error = suprnova_ai::ops::upload_skill(&provider, vec![], titled, Default::default())
        .await
        .unwrap_err();
    assert!(
        matches!(error, suprnova_ai::Error::Validation(_)),
        "{error:?}"
    );
    assert_eq!(transport.requests.lock().unwrap().len(), 1);
}

// OAI-023

fn text_request(id: &str, model: &str) -> suprnova_ai::providers::operations::BatchRequest {
    suprnova_ai::providers::operations::BatchRequest {
        id: id.into(),
        model_id: model.into(),
        kind: suprnova_ai::providers::operations::BatchRequestKind::Text {
            messages: vec![user("hi")],
            options: StreamOptions::default(),
        },
    }
}
fn batch_object(status: &str) -> Value {
    json!({"id": "batch_1", "status": status, "output_file_id": "file-out", "error_file_id": "file-err",
           "created_at": 1_800_000_000u64, "expires_at": 1_800_086_400u64,
           "request_counts": {"total": 3, "completed": 1, "failed": 1},
           "errors": {"data": [{"code": "bad_line", "message": "line 2"}]}})
}

#[tokio::test]
async fn oai_023_a_batch_uploads_its_requests_and_creates_the_batch() {
    use suprnova_ai::providers::operations::{BatchParams, BatchState};
    let transport = Recorder::new([
        json_reply(json!({"id": "file-in", "object": "file", "expires_at": 1_800_172_800u64})),
        json_reply(batch_object("validating")),
    ]);
    let provider = OpenAICompatibleProvider::new(config(), transport.clone());
    let started = suprnova_ai::ops::batch::start(
        &provider,
        vec![text_request("r1", "gpt-4.1"), text_request("r2", "gpt-4.1")],
        BatchParams::default(),
        Default::default(),
    )
    .await
    .unwrap();
    assert_eq!(url(&transport, 0), "https://api.openai.com/v1/files");
    let sent = form(&transport, 0);
    assert!(sent.contains(&field("purpose", "batch")), "{sent}");
    assert!(
        sent.contains(&field("expires_after[anchor]", "created_at")),
        "{sent}"
    );
    assert!(
        sent.contains(&field("expires_after[seconds]", "172800")),
        "{sent}"
    );
    let lines: Vec<Value> = sent
        .lines()
        .filter(|line| line.starts_with("{\"custom_id\""))
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(lines.len(), 2, "{sent}");
    assert_eq!(lines[0]["custom_id"], "r1");
    assert_eq!(lines[0]["method"], "POST");
    assert_eq!(lines[0]["url"], "/v1/responses");
    assert_eq!(lines[0]["body"]["model"], "gpt-4.1");
    assert_eq!(lines[0]["body"].get("stream"), None);
    assert_eq!(url(&transport, 1), "https://api.openai.com/v1/batches");
    assert_eq!(
        transport.body(1),
        json!({"input_file_id": "file-in", "endpoint": "/v1/responses", "completion_window": "24h"})
    );
    assert_eq!(started.status.batch_id, "batch_1");
    assert_eq!(started.status.state, BatchState::Pending);
    let counts = started.status.request_counts.unwrap();
    assert_eq!(
        (
            counts.total,
            counts.pending,
            counts.completed,
            counts.failed
        ),
        (3, 1, 1, 1)
    );
    assert_eq!(started.status.error.as_ref().unwrap().message, "line 2");
    assert_eq!(
        started.status.provider_metadata.as_ref().unwrap()["openai"]["inputFileId"],
        "file-in"
    );
}

#[tokio::test]
async fn oai_023_batches_with_other_requests_or_models_fail_before_a_request() {
    use suprnova_ai::providers::operations::{
        BatchParams, BatchRequest, BatchRequestKind, ImagePrompt,
    };
    let image = BatchRequest {
        id: "r2".into(),
        model_id: "gpt-4.1".into(),
        kind: BatchRequestKind::Image {
            prompt: ImagePrompt::default(),
            params: Default::default(),
        },
    };
    for (requests, params) in [
        (
            vec![text_request("r1", "gpt-4.1"), image],
            BatchParams::default(),
        ),
        (
            vec![text_request("r1", "gpt-4.1"), text_request("r2", "gpt-5.4")],
            BatchParams::default(),
        ),
        (
            vec![text_request("r1", "gpt-4.1")],
            BatchParams {
                provider_options: [("openai".to_owned(), json!({"inputFileExpiresAfter": 60}))]
                    .into(),
                ..Default::default()
            },
        ),
    ] {
        let transport = Recorder::new([]);
        let provider = OpenAICompatibleProvider::new(config(), transport.clone());
        let error = suprnova_ai::ops::batch::start(&provider, requests, params, Default::default())
            .await
            .unwrap_err();
        assert!(
            matches!(error, suprnova_ai::Error::Validation(_)),
            "{error:?}"
        );
        assert!(transport.requests.lock().unwrap().is_empty());
    }
}

#[tokio::test]
async fn oai_023_statuses_cancels_and_lists_map_as_typescript_maps_them() {
    use suprnova_ai::providers::operations::BatchState;
    let transport = Recorder::new([
        json_reply(batch_object("completed")),
        json_reply(batch_object("expired")),
        json_reply(batch_object("cancelling")),
        json_reply(batch_object("cancelled")),
        json_reply(
            json!({"data": [batch_object("in_progress")], "has_more": true, "last_id": "batch_1"}),
        ),
        json_reply(json!({"data": [], "has_more": false, "last_id": null})),
    ]);
    let provider = OpenAICompatibleProvider::new(config(), transport.clone());
    let state = |status: suprnova_ai::providers::operations::BatchStatus| status.state;
    let none = || suprnova_ai::ops::OperationOptions::default();
    assert_eq!(
        state(
            suprnova_ai::ops::batch::status(&provider, "batch_1", none())
                .await
                .unwrap()
        ),
        BatchState::Completed
    );
    assert_eq!(
        state(
            suprnova_ai::ops::batch::status(&provider, "batch_1", none())
                .await
                .unwrap()
        ),
        BatchState::Failed
    );
    assert_eq!(
        state(
            suprnova_ai::ops::batch::status(&provider, "batch_1", none())
                .await
                .unwrap()
        ),
        BatchState::Pending
    );
    assert_eq!(
        url(&transport, 0),
        "https://api.openai.com/v1/batches/batch_1"
    );
    suprnova_ai::ops::batch::cancel(&provider, "batch_1", none())
        .await
        .unwrap();
    assert_eq!(
        url(&transport, 3),
        "https://api.openai.com/v1/batches/batch_1/cancel"
    );
    assert_eq!(transport.requests.lock().unwrap()[3].method, "POST");
    let page = suprnova_ai::ops::batch::list(&provider, Some(5), Some("batch_0".into()), none())
        .await
        .unwrap();
    assert_eq!(
        url(&transport, 4),
        "https://api.openai.com/v1/batches?after=batch_0&limit=5"
    );
    assert_eq!(page.next_cursor.as_deref(), Some("batch_1"));
    assert_eq!(page.batches.len(), 1);
    let page = suprnova_ai::ops::batch::list(&provider, None, None, none())
        .await
        .unwrap();
    assert_eq!(page.next_cursor, None);
}

#[tokio::test]
async fn oai_023_result_lines_give_each_outcome() {
    use suprnova_ai::providers::operations::BatchOutcome;
    let response = json!({"id": "resp_b", "model": "gpt-4.1", "status": "completed",
        "output": [{"type": "message", "id": "msg_1", "role": "assistant", "content": [{"type": "output_text", "text": "done", "annotations": []}]}],
        "usage": {"input_tokens": 2, "output_tokens": 1}});
    let output = [
        json!({"custom_id": "ok", "response": {"status_code": 200, "body": response}}),
        json!({"custom_id": "bad", "response": {"status_code": 400, "body": {"error": {"message": "nope", "type": "invalid_request_error", "code": "bad"}}}}),
    ]
    .iter()
    .map(|l| l.to_string())
    .collect::<Vec<_>>()
    .join("\n");
    let errors = [
        json!({"custom_id": "c", "error": {"code": "batch_cancelled", "message": "cancelled"}}),
        json!({"custom_id": "e", "error": {"code": "batch_expired", "message": "expired"}}),
        json!({"custom_id": "f", "error": {"code": "server_error", "message": "boom"}}),
    ]
    .iter()
    .map(|l| l.to_string())
    .collect::<Vec<_>>()
    .join("\n");
    let transport = Recorder::new([
        json_reply(batch_object("completed")),
        (200, output.into_bytes()),
        (200, errors.into_bytes()),
    ]);
    let provider = OpenAICompatibleProvider::new(config(), transport.clone());
    let results: Vec<_> =
        suprnova_ai::ops::batch::results(&provider, "batch_1", Default::default())
            .await
            .unwrap()
            .map(Result::unwrap)
            .collect()
            .await;
    assert_eq!(
        url(&transport, 1),
        "https://api.openai.com/v1/files/file-out/content"
    );
    assert_eq!(
        url(&transport, 2),
        "https://api.openai.com/v1/files/file-err/content"
    );
    let ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
    assert_eq!(ids, ["ok", "bad", "c", "e", "f"]);
    let BatchOutcome::Message(message) = &results[0].outcome else {
        panic!("{:?}", results[0].outcome)
    };
    assert_eq!(message.text(), "done");
    let BatchOutcome::Failed { error, .. } = &results[1].outcome else {
        panic!("{:?}", results[1].outcome)
    };
    assert_eq!(
        (error.message.as_str(), error.code.as_deref()),
        ("nope", Some("bad"))
    );
    assert!(matches!(results[2].outcome, BatchOutcome::Cancelled { .. }));
    assert!(matches!(results[3].outcome, BatchOutcome::Expired { .. }));
    assert!(matches!(results[4].outcome, BatchOutcome::Failed { .. }));
}

// OAI-024

use suprnova_ai::providers::operations::{
    RealtimeClientEvent, RealtimeItem, RealtimeServerEventKind, RealtimeSessionConfig,
    RealtimeTokenParams, RealtimeTool, RealtimeTranscription,
};

fn realtime_model(id: &str, transport: Arc<Recorder>, socket: Arc<ScriptedSocket>) -> Model {
    let provider =
        OpenAICompatibleProvider::new(config(), transport).with_websocket_transport(socket);
    Model::new(id, Arc::new(provider))
}
fn full_config() -> RealtimeSessionConfig {
    RealtimeSessionConfig {
        instructions: Some("be kind".into()),
        voice: Some("cedar".into()),
        output_modalities: vec!["audio".into()],
        input_audio_format: Some(AudioFormat::pcm(24_000)),
        input_audio_transcription: Some(RealtimeTranscription {
            language: Some("en".into()),
            ..Default::default()
        }),
        turn_detection: Some(
            json!({"type": "server-vad", "threshold": 0.5, "silenceDurationMs": 300}),
        ),
        tools: vec![RealtimeTool {
            name: "weather".into(),
            description: Some("Weather".into()),
            parameters: json!({"type": "object"}),
        }],
        ..Default::default()
    }
}

#[tokio::test]
async fn oai_024_a_token_carries_the_session_setup_and_opens_the_realtime_socket() {
    let transport = Recorder::new([json_reply(
        json!({"value": "ek_1", "expires_at": 1_800_000_060u64}),
    )]);
    let socket = ScriptedSocket::new(vec![(NEVER_SENT, text_message(json!({})))]);
    let model = realtime_model("gpt-realtime", transport.clone(), socket.clone());
    let token = suprnova_ai::ops::realtime::create_token(
        &model,
        RealtimeTokenParams {
            expires_after_seconds: Some(60),
            session: Some(full_config()),
        },
        Default::default(),
    )
    .await
    .unwrap();
    assert_eq!(
        url(&transport, 0),
        "https://api.openai.com/v1/realtime/client_secrets"
    );
    assert_eq!(
        transport.body(0),
        json!({"session": {
            "type": "realtime", "model": "gpt-realtime", "instructions": "be kind", "output_modalities": ["audio"],
            "audio": {
                "input": {"format": {"type": "audio/pcm", "rate": 24000},
                          "turn_detection": {"type": "server_vad", "threshold": 0.5, "silence_duration_ms": 300},
                          "transcription": {"model": "gpt-realtime-whisper", "language": "en"}},
                "output": {"voice": "cedar"}},
            "tools": [{"type": "function", "name": "weather", "description": "Weather", "parameters": {"type": "object"}}],
            "tool_choice": "auto"},
         "expires_after": {"anchor": "created_at", "seconds": 60}})
    );
    assert_eq!(token.token, "ek_1");
    assert_eq!(
        token.url,
        "wss://api.openai.com/v1/realtime?model=gpt-realtime"
    );
    assert_eq!(token.expires_at, Some(1_800_000_060));
    let mut connection =
        suprnova_ai::ops::realtime::connect(&model, &token, RealtimeSessionConfig::default())
            .await
            .unwrap();
    let request = socket.request();
    assert_eq!(request.url, token.url);
    assert_eq!(
        request.headers,
        vec![("authorization".to_owned(), "Bearer ek_1".to_owned())]
    );
    connection
        .send(RealtimeClientEvent::ConversationItemCreate(
            RealtimeItem::Text("hi".into()),
        ))
        .await
        .unwrap();
    let sent = socket.sent_json();
    assert_eq!(
        sent[0],
        json!({"type": "session.update", "session": {"type": "realtime", "model": "gpt-realtime"}})
    );
    assert_eq!(
        sent[1],
        json!({"type": "conversation.item.create", "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}})
    );
}

/// A count of sent messages that never comes, for a message that is never delivered.
const NEVER_SENT: usize = usize::MAX;

#[test]
fn oai_024_the_realtime_codec_maps_the_typescript_events() {
    let model = realtime_model(
        "gpt-realtime",
        Recorder::new([]),
        ScriptedSocket::new(vec![]),
    );
    let mut codec = suprnova_ai::ops::realtime::codec(&model).unwrap();
    let encode = |codec: &mut Box<dyn suprnova_ai::providers::operations::RealtimeCodec>, event| {
        codec
            .encode(&event)
            .unwrap()
            .map(|m| serde_json::from_str::<Value>(&m).unwrap())
    };
    assert_eq!(
        encode(
            &mut codec,
            RealtimeClientEvent::InputAudioAppend("AQI=".into())
        ),
        Some(json!({"type": "input_audio_buffer.append", "audio": "AQI="}))
    );
    assert_eq!(
        encode(
            &mut codec,
            RealtimeClientEvent::ConversationItemCreate(RealtimeItem::FunctionCallOutput {
                call_id: "c1".into(),
                name: None,
                output: "sunny".into(),
            })
        ),
        Some(
            json!({"type": "conversation.item.create", "item": {"type": "function_call_output", "call_id": "c1", "output": "sunny"}})
        )
    );
    assert_eq!(
        encode(
            &mut codec,
            RealtimeClientEvent::ResponseCreate {
                options: Some(json!({"modalities": ["text"], "instructions": "short"}))
            }
        ),
        Some(
            json!({"type": "response.create", "response": {"output_modalities": ["text"], "instructions": "short"}})
        )
    );
    assert_eq!(
        encode(&mut codec, RealtimeClientEvent::InputAudioMute),
        None
    );
    let decode = |codec: &mut Box<dyn suprnova_ai::providers::operations::RealtimeCodec>,
                  value: Value| {
        codec.decode(&value.to_string()).unwrap().remove(0).kind
    };
    assert_eq!(
        decode(
            &mut codec,
            json!({"type": "session.created", "session": {"id": "sess_1"}})
        ),
        RealtimeServerEventKind::SessionCreated {
            session_id: Some("sess_1".into())
        }
    );
    assert_eq!(
        decode(
            &mut codec,
            json!({"type": "response.output_audio_transcript.delta", "response_id": "r1", "item_id": "i1", "delta": "he"})
        ),
        RealtimeServerEventKind::AudioTranscriptDelta {
            response_id: "r1".into(),
            item_id: "i1".into(),
            delta: "he".into()
        }
    );
    assert_eq!(
        decode(
            &mut codec,
            json!({"type": "response.function_call_arguments.done", "response_id": "r1", "item_id": "i2", "call_id": "c1", "name": "weather", "arguments": "{}"})
        ),
        RealtimeServerEventKind::FunctionCallArgumentsDone {
            response_id: "r1".into(),
            item_id: "i2".into(),
            call_id: "c1".into(),
            name: "weather".into(),
            arguments: "{}".into(),
        }
    );
    assert_eq!(
        decode(
            &mut codec,
            json!({"type": "response.done", "response": {"id": "r1"}})
        ),
        RealtimeServerEventKind::ResponseDone {
            response_id: "r1".into(),
            status: "completed".into()
        }
    );
    assert_eq!(
        decode(
            &mut codec,
            json!({"type": "error", "error": {"message": "bad", "code": "x"}})
        ),
        RealtimeServerEventKind::Error {
            message: "bad".into(),
            code: Some("x".into()),
            client_event_id: None
        }
    );
    assert_eq!(
        decode(&mut codec, json!({"type": "rate_limits.updated"})),
        RealtimeServerEventKind::Custom {
            raw_type: "rate_limits.updated".into()
        }
    );
}

#[tokio::test]
async fn oai_024_a_live_model_opens_live_sessions_with_the_providers_key() {
    let socket = ScriptedSocket::new(vec![(NEVER_SENT, text_message(json!({})))]);
    let model = realtime_model("gpt-live-1", Recorder::new([]), socket.clone());
    let session = RealtimeSessionConfig {
        instructions: Some("be kind".into()),
        ..Default::default()
    };
    let mut connection = suprnova_ai::ops::realtime::connect_server(&model, session)
        .await
        .unwrap();
    let request = socket.request();
    assert_eq!(request.url, "wss://api.openai.com/v1/live/sessions");
    assert!(
        request
            .headers
            .iter()
            .any(|(k, v)| k == "authorization" && v == "Bearer sk-test")
    );
    for event in [
        RealtimeClientEvent::InputAudioAppend("AQI=".into()),
        RealtimeClientEvent::InputAudioMute,
        RealtimeClientEvent::InputAudioUnmute,
        RealtimeClientEvent::ContextAppend {
            content: "note".into(),
            delegation_id: Some("d1".into()),
            provider_options: Default::default(),
        },
        RealtimeClientEvent::SessionClose,
    ] {
        connection.send(event).await.unwrap();
    }
    assert_eq!(
        socket.sent_json(),
        vec![
            json!({"type": "session.start", "session": {"model": "gpt-live-1", "instructions": "be kind",
                   "audio": {"format": {"type": "audio/pcm", "rate": 24000}, "output": {"voice": "marin"}}}}),
            json!({"type": "session.input_audio.append", "audio": "AQI="}),
            json!({"type": "session.input_audio.mute"}),
            json!({"type": "session.input_audio.unmute"}),
            json!({"type": "session.thinking.append", "content": "note", "delegation_id": "d1"}),
            json!({"type": "session.close"}),
        ]
    );
    let socket = ScriptedSocket::new(vec![(NEVER_SENT, text_message(json!({})))]);
    let server = Model::new(
        "gpt-realtime",
        Arc::new(
            OpenAICompatibleProvider::new(config(), Recorder::new([]))
                .with_websocket_transport(socket.clone()),
        ),
    );
    let _ = suprnova_ai::ops::realtime::connect_server(&server, RealtimeSessionConfig::default())
        .await
        .unwrap();
    assert_eq!(
        socket.request().url,
        "wss://api.openai.com/v1/realtime?model=gpt-realtime"
    );
    assert_eq!(socket.sent_json()[0]["type"], "session.update");
}

type Codec = Box<dyn suprnova_ai::providers::operations::RealtimeCodec>;

fn encoded(codec: &mut Codec, event: RealtimeClientEvent) -> Option<Value> {
    codec
        .encode(&event)
        .unwrap()
        .map(|m| serde_json::from_str::<Value>(&m).unwrap())
}

fn decoded(codec: &mut Codec, value: Value) -> RealtimeServerEventKind {
    codec.decode(&value.to_string()).unwrap().remove(0).kind
}

#[test]
fn oai_024_the_realtime_codec_maps_items_parts_and_event_ids() {
    let model = realtime_model(
        "gpt-realtime",
        Recorder::new([]),
        ScriptedSocket::new(vec![]),
    );
    let mut codec = suprnova_ai::ops::realtime::codec(&model).unwrap();
    assert_eq!(
        encoded(
            &mut codec,
            RealtimeClientEvent::InputAudioAppend("AQI=".into()).with_event_id("e1")
        ),
        Some(json!({"type": "input_audio_buffer.append", "audio": "AQI=", "event_id": "e1"}))
    );
    let update = encoded(
        &mut codec,
        RealtimeClientEvent::SessionUpdate(Default::default()).with_event_id("e2"),
    )
    .unwrap();
    assert_eq!(
        (&update["type"], &update["event_id"]),
        (&json!("session.update"), &json!("e2"))
    );
    // TypeScript sends an id with these two events only.
    assert_eq!(
        encoded(
            &mut codec,
            RealtimeClientEvent::InputAudioCommit.with_event_id("e3")
        ),
        Some(json!({"type": "input_audio_buffer.commit"}))
    );
    for (message, kind) in [
        (
            json!({"type": "conversation.item.added", "item": {"id": "i1", "type": "message"}}),
            RealtimeServerEventKind::ConversationItemAdded {
                item_id: "i1".into(),
                item: json!({"id": "i1", "type": "message"}),
            },
        ),
        (
            json!({"type": "response.output_item.added", "response_id": "r1", "item": {"id": "i1"}}),
            RealtimeServerEventKind::OutputItemAdded {
                response_id: "r1".into(),
                item_id: "i1".into(),
            },
        ),
        (
            json!({"type": "response.output_item.done", "response_id": "r1", "item": {"id": "i1"}}),
            RealtimeServerEventKind::OutputItemDone {
                response_id: "r1".into(),
                item_id: "i1".into(),
            },
        ),
        (
            json!({"type": "response.content_part.added", "response_id": "r1", "item_id": "i1"}),
            RealtimeServerEventKind::ContentPartAdded {
                response_id: "r1".into(),
                item_id: "i1".into(),
            },
        ),
        (
            json!({"type": "response.content_part.done", "response_id": "r1", "item_id": "i1"}),
            RealtimeServerEventKind::ContentPartDone {
                response_id: "r1".into(),
                item_id: "i1".into(),
            },
        ),
        (
            json!({"type": "input_audio_buffer.committed", "item_id": "i2", "previous_item_id": "i1"}),
            RealtimeServerEventKind::AudioCommitted {
                item_id: Some("i2".into()),
                previous_item_id: Some("i1".into()),
            },
        ),
        (
            json!({"type": "error", "error": {"message": "bad", "code": "x", "event_id": "e1"}}),
            RealtimeServerEventKind::Error {
                message: "bad".into(),
                code: Some("x".into()),
                client_event_id: Some("e1".into()),
            },
        ),
    ] {
        assert_eq!(decoded(&mut codec, message.clone()), kind, "{message}");
    }
}

#[test]
fn oai_024_the_live_codec_sends_event_ids_and_context_channels() {
    let model = realtime_model("gpt-live-1", Recorder::new([]), ScriptedSocket::new(vec![]));
    let mut codec = suprnova_ai::ops::realtime::codec(&model).unwrap();
    assert_eq!(
        encoded(
            &mut codec,
            RealtimeClientEvent::InputAudioMute.with_event_id("e1")
        ),
        Some(json!({"type": "session.input_audio.mute", "event_id": "e1"}))
    );
    assert_eq!(
        encoded(
            &mut codec,
            RealtimeClientEvent::SessionClose.with_event_id("e2")
        ),
        Some(json!({"type": "session.close", "event_id": "e2"}))
    );
    let start = encoded(
        &mut codec,
        RealtimeClientEvent::SessionStart(Default::default()).with_event_id("e3"),
    )
    .unwrap();
    assert_eq!(
        (&start["type"], &start["event_id"]),
        (&json!("session.start"), &json!("e3"))
    );
    let append = |options: Value| RealtimeClientEvent::ContextAppend {
        content: "rule".into(),
        delegation_id: None,
        provider_options: [("openai".to_owned(), options)].into(),
    };
    assert_eq!(
        encoded(&mut codec, append(json!({"channel": "instructions"}))),
        Some(
            json!({"type": "session.instructions.append", "content": "rule", "delegation_id": null})
        )
    );
    for options in [json!({"channel": "loud"}), json!({"other": 1})] {
        let error = codec.encode(&append(options.clone())).unwrap_err();
        assert!(
            matches!(error, suprnova_ai::Error::Validation(_)),
            "{options}: {error:?}"
        );
    }
    assert_eq!(
        decoded(
            &mut codec,
            json!({"type": "error", "error": {"message": "bad", "client_event_id": "e1"}})
        ),
        RealtimeServerEventKind::Error {
            message: "bad".into(),
            code: None,
            client_event_id: Some("e1".into()),
        }
    );
}

#[test]
fn oai_024_the_live_codec_decodes_the_typescript_events() {
    let model = realtime_model("gpt-live-1", Recorder::new([]), ScriptedSocket::new(vec![]));
    let mut codec = suprnova_ai::ops::realtime::codec(&model).unwrap();
    let mut decode = |value: Value| codec.decode(&value.to_string()).unwrap().remove(0).kind;
    assert_eq!(
        decode(json!({"type": "session.started", "session": {"id": "s1"}})),
        RealtimeServerEventKind::SessionStarted {
            session_id: "s1".into(),
            delegation_mode: "client".into()
        }
    );
    assert_eq!(
        decode(json!({"type": "session.output_audio.delta", "delta": "AQI="})),
        RealtimeServerEventKind::AudioChunk {
            delta: "AQI=".into()
        }
    );
    assert_eq!(
        decode(
            json!({"type": "session.output_transcript.delta", "delta": "hi", "start_ms": 10, "end_ms": 20})
        ),
        RealtimeServerEventKind::TranscriptFragment {
            speaker: "assistant".into(),
            delta: "hi".into(),
            start_ms: 10.0,
            end_ms: 20.0
        }
    );
    assert_eq!(
        decode(
            json!({"type": "session.input_transcript.delta", "delta": "x", "start_ms": 20, "end_ms": 10})
        ),
        RealtimeServerEventKind::Error {
            message: "Invalid OpenAI Live event time interval.".into(),
            code: Some("invalid_server_event".into()),
            client_event_id: None
        }
    );
    assert_eq!(
        decode(
            json!({"type": "session.delegation.created", "offset_ms": 5, "delegation": {"id": "d1", "target": "responses", "response_id": "r1"}})
        ),
        RealtimeServerEventKind::DelegationCreated {
            delegation_id: "d1".into(),
            target: Some("provider".into()),
            offset_ms: Some(5.0),
            response_id: Some("r1".into())
        }
    );
    assert_eq!(
        decode(json!({"type": "session.input_audio.muted", "client_event_id": "e1"})),
        RealtimeServerEventKind::CommandAcknowledged {
            command: "session.input_audio.mute".into(),
            client_event_id: Some("e1".into())
        }
    );
    assert_eq!(
        decode(json!({"type": "session.closed", "usage": {"seconds": 12.5}, "reason": "client"})),
        RealtimeServerEventKind::SessionClosed {
            session_id: None,
            usage_seconds: 12.5,
            reason: "client".into()
        }
    );
    assert_eq!(
        decode(
            json!({"type": "session.usage.updated", "usage": {"seconds": 3}, "context_window": {"usage_ratio": 0.25}})
        ),
        RealtimeServerEventKind::SessionUsage {
            usage_seconds: 3.0,
            context_window_usage_ratio: Some(0.25)
        }
    );
}

#[tokio::test]
async fn oai_024_a_live_webrtc_session_posts_the_offer_and_returns_the_answer() {
    let transport = Recorder::new([json_reply(
        json!({"session": {"id": "sess_1"}, "transport": {"type": "webrtc", "sdp": "v=0 answer"}}),
    )]);
    let model = realtime_model("gpt-live-1", transport.clone(), ScriptedSocket::new(vec![]));
    let session = suprnova_ai::ops::realtime::create_webrtc_session(
        &model,
        "v=0 offer".into(),
        RealtimeSessionConfig {
            voice: Some("marin".into()),
            ..Default::default()
        },
        Default::default(),
    )
    .await
    .unwrap();
    assert_eq!(
        url(&transport, 0),
        "https://api.openai.com/v1/live/sessions"
    );
    assert_eq!(
        transport.body(0),
        json!({"session": {"model": "gpt-live-1", "audio": {"output": {"voice": "marin"}}},
               "transport": {"type": "webrtc", "sdp": "v=0 offer"}})
    );
    assert_eq!(
        (session.session_id.as_str(), session.answer_sdp.as_str()),
        ("sess_1", "v=0 answer")
    );
}

#[tokio::test]
async fn oai_024_the_live_api_refuses_tokens_updates_and_other_settings() {
    let transport = Recorder::new([]);
    let socket = ScriptedSocket::new(vec![]);
    let model = realtime_model("gpt-live-1", transport.clone(), socket.clone());
    let token =
        suprnova_ai::ops::realtime::create_token(&model, Default::default(), Default::default())
            .await
            .err();
    assert!(
        matches!(token, Some(suprnova_ai::Error::Validation(_))),
        "{token:?}"
    );
    let mut codec = suprnova_ai::ops::realtime::codec(&model).unwrap();
    let update = codec
        .encode(&RealtimeClientEvent::SessionUpdate(Default::default()))
        .err();
    assert!(
        matches!(update, Some(suprnova_ai::Error::Validation(_))),
        "{update:?}"
    );
    for config in [
        RealtimeSessionConfig {
            tools: full_config().tools,
            ..Default::default()
        },
        RealtimeSessionConfig {
            turn_detection: Some(json!({"type": "server-vad"})),
            ..Default::default()
        },
        RealtimeSessionConfig {
            input_audio_format: Some(AudioFormat::pcm(44_100)),
            ..Default::default()
        },
    ] {
        let error = suprnova_ai::ops::realtime::connect_server(&model, config)
            .await
            .err();
        assert!(
            matches!(error, Some(suprnova_ai::Error::Validation(_))),
            "{error:?}"
        );
    }
    assert!(transport.requests.lock().unwrap().is_empty());
    assert!(socket.requests.lock().unwrap().is_empty());
}

// OAI-025

use suprnova_ai::providers::operations::{
    EvaluationInput, EvaluationParams, NoulCriteria, Question,
};

fn decision_questions() -> std::collections::BTreeMap<String, Question> {
    let text = |t: &str| EvaluationInput::Text(t.into());
    [
        (
            "mood".to_owned(),
            Question::Choice {
                instructions: text("The mood"),
                criteria: [
                    ("calm".to_owned(), Some(text("No rush"))),
                    ("tense".to_owned(), None),
                ]
                .into(),
            },
        ),
        (
            "quality".to_owned(),
            Question::Score {
                instructions: text("How good"),
                criteria: vec![Some(text("bad")), None],
            },
        ),
        (
            "urgent".to_owned(),
            Question::Noul {
                instructions: text("Is it urgent"),
                criteria: Some(NoulCriteria {
                    r#true: Some(text("today")),
                    r#false: None,
                }),
            },
        ),
    ]
    .into()
}
fn decisions_model(transport: Arc<Recorder>) -> Model {
    Model::new(
        "gpt-6-luna",
        Arc::new(OpenAICompatibleProvider::new(config(), transport)),
    )
    .with_protocol(Arc::new(
        suprnova_ai::providers::openai::OpenAIDecisionsProtocol,
    ))
}
fn decision_params(options: Value) -> EvaluationParams {
    EvaluationParams {
        provider_options: [("openai".to_owned(), options)].into(),
    }
}

#[tokio::test]
async fn oai_025_decisions_send_the_questions_and_read_the_answers() {
    let transport = Recorder::new([json_reply(
        json!({"model": "gpt-6-luna", "usage": {"input_tokens": 9, "output_tokens": 2},
        "answers": [
            {"type": "choice", "name": "mood", "choice": "tense", "confidence": 0.8, "probabilities": [{"value": "calm", "probability": 0.2}, {"value": "tense", "probability": 0.8}]},
            {"type": "score", "name": "quality", "score": 1, "probabilities": [{"value": 1, "probability": 0.9}]},
            {"type": "predicate", "name": "urgent", "probability": 0.7},
        ]}),
    )]);
    let item = suprnova_ai::ops::experimental::evaluate(
        &decisions_model(transport.clone()),
        EvaluationInput::Object([("ticket".to_owned(), json!("late"))].into()),
        decision_questions(),
        decision_params(json!({"safetyIdentifier": "user-1"})),
        Default::default(),
    )
    .await
    .unwrap();
    assert_eq!(url(&transport, 0), "https://api.openai.com/v1/decisions");
    assert_eq!(
        transport.body(0),
        json!({"model": "gpt-6-luna", "safety_identifier": "user-1",
        "input": [{"role": "user", "content": [{"type": "input_text", "text": "{\"ticket\":\"late\"}"}]}],
        "questions": [
            {"type": "choice", "name": "mood", "instructions": "The mood",
             "choices": [{"value": "calm", "description": "No rush"}, {"value": "tense"}]},
            {"type": "score", "name": "quality", "instructions": "How good",
             "levels": [{"label": "0", "description": "bad"}, {"label": "1"}]},
            {"type": "predicate", "name": "urgent", "instructions": "Is it urgent\n\nCriteria for true:\ntoday"},
        ]})
    );
    assert_eq!(
        item.value["mood"],
        json!({"type": "choice", "choice": "tense"})
    );
    assert_eq!(
        item.value["quality"],
        json!({"type": "score", "score": 1.0})
    );
    assert_eq!(item.value["urgent"], json!({"type": "noul", "noul": 0.7}));
    let md = &item.provider_metadata.as_ref().unwrap()["openai"];
    assert_eq!(md["confidence"], json!({"mood": 0.8}));
    assert_eq!(
        md["probabilities"]["mood"],
        json!({"calm": 0.2, "tense": 0.8})
    );
    assert_eq!(item.usage.unwrap().input_tokens, 9);
}

#[tokio::test]
async fn oai_025_a_refusal_and_a_wrong_answer_set() {
    let one = |answers: Value| Recorder::new([json_reply(json!({"answers": answers}))]);
    let urgent = || -> std::collections::BTreeMap<String, Question> {
        decision_questions()
            .into_iter()
            .filter(|(id, _)| id == "urgent")
            .collect()
    };
    let item = suprnova_ai::ops::experimental::evaluate(
        &decisions_model(one(json!([{"type": "refusal", "name": "urgent"}]))),
        EvaluationInput::Text("x".into()),
        urgent(),
        EvaluationParams::default(),
        Default::default(),
    )
    .await
    .unwrap();
    assert_eq!(item.value["urgent"], json!({"type": "refusal"}));
    for answers in [
        json!([]),
        json!([{"type": "predicate", "name": "urgent", "probability": 0.5}, {"type": "predicate", "name": "urgent", "probability": 0.5}]),
        json!([{"type": "predicate", "name": "other", "probability": 0.5}]),
        json!([{"type": "predicate", "name": "urgent", "probability": 1.5}]),
    ] {
        let error = suprnova_ai::ops::experimental::evaluate(
            &decisions_model(one(answers.clone())),
            EvaluationInput::Text("x".into()),
            urgent(),
            EvaluationParams::default(),
            Default::default(),
        )
        .await
        .unwrap_err();
        let suprnova_ai::Error::Provider(error) = error else {
            panic!("{answers}: not a provider error")
        };
        assert_eq!(
            error.kind,
            suprnova_ai::errors::ProviderErrorKind::Response,
            "{answers}"
        );
    }
}

#[tokio::test]
async fn oai_025_the_responses_protocol_evaluates_on_its_language_model() {
    let answer = json!({"q0": "c1"}).to_string();
    let reply = responses_sse(&[
        json!({"type": "response.created", "response": {"id": "resp_e", "model": "gpt-5.4", "status": "in_progress"}}),
        json!({"type": "response.output_item.added", "output_index": 0, "item": {"type": "message", "id": "msg_1", "role": "assistant", "content": []}}),
        json!({"type": "response.output_text.delta", "item_id": "msg_1", "output_index": 0, "delta": answer}),
        json!({"type": "response.output_item.done", "output_index": 0, "item": {"type": "message", "id": "msg_1", "role": "assistant", "content": [{"type": "output_text", "text": answer}]}}),
        json!({"type": "response.completed", "response": {"id": "resp_e", "model": "gpt-5.4", "status": "completed", "output": [], "usage": {"input_tokens": 1, "output_tokens": 1}}}),
    ]);
    let transport = Recorder::new([reply]);
    let model = Model::new(
        "gpt-5.4",
        Arc::new(OpenAICompatibleProvider::new(config(), transport.clone())),
    );
    let mood: std::collections::BTreeMap<String, Question> = decision_questions()
        .into_iter()
        .filter(|(id, _)| id == "mood")
        .collect();
    let item = suprnova_ai::ops::experimental::evaluate(
        &model,
        EvaluationInput::Text("a calm day".into()),
        mood,
        EvaluationParams::default(),
        Default::default(),
    )
    .await
    .unwrap();
    assert_eq!(url(&transport, 0), "https://api.openai.com/v1/responses");
    assert_eq!(
        item.value["mood"],
        json!({"type": "choice", "choice": "tense"})
    );
    let body = transport.body(0);
    assert_eq!(body["text"]["format"]["type"], "json_schema", "{body}");
}

// OAI-026

fn with_breakpoint(mut part: Part) -> Part {
    let md = md(json!({"promptCacheBreakpoint": {"mode": "explicit", "ttl": "12h"}}));
    match &mut part {
        Part::Text(t) => t.provider_metadata = md,
        Part::ToolResult(r) => r.provider_metadata = md,
        _ => {}
    }
    part
}

#[test]
fn oai_026_breakpoints_reach_custom_results_parallel_outputs_and_chat_assistant_text() {
    let breakpoint = json!({"mode": "explicit", "ttl": "12h"});
    // The result of a custom tool.
    let messages = vec![
        user("count"),
        Message::new(
            MessageRole::Assistant,
            vec![Part::ToolCall(ToolCallPart::new(
                "call_ct",
                "sql",
                json!("select 1").to_string(),
            ))],
        ),
        Message::new(
            MessageRole::Tool,
            vec![with_breakpoint(Part::ToolResult(ToolResultPart::new(
                "call_ct",
                "sql",
                json!("1 row"),
            )))],
        ),
    ];
    let body = body_for("gpt-5.4", messages, &with_tools(host_tools()));
    let output = items(&body)
        .iter()
        .find(|item| item["type"] == "custom_tool_call_output")
        .cloned()
        .unwrap();
    assert_eq!(
        output["output"],
        json!([{"type": "input_text", "text": "1 row", "prompt_cache_breakpoint": breakpoint}]),
        "{body}"
    );
    // The joined outputs of an expanded parallel call keep each breakpoint.
    let call = |index: usize| {
        let mut call = ToolCallPart::new(format!("c_{index}"), "weather", r#"{"city":"SF"}"#);
        call.provider_metadata = md(json!({"parallel_tool_call": {
            "item_id": "fc_p", "tool_call_id": "c", "tool_name": "parallel",
            "input": "{\"tool_uses\":[]}", "index": index, "count": 2,
        }}));
        Part::ToolCall(call)
    };
    let result = |index: usize| {
        Part::ToolResult(ToolResultPart::new(
            format!("c_{index}"),
            "weather",
            json!(format!("r{index}")),
        ))
    };
    let messages = vec![
        user("weather twice"),
        Message::new(MessageRole::Assistant, vec![call(0), call(1)]),
        Message::new(
            MessageRole::Tool,
            vec![result(0), with_breakpoint(result(1))],
        ),
    ];
    let body = body_for(
        "gpt-4.1",
        messages,
        &openai(json!({"previousResponseId": "resp_0"})),
    );
    assert_eq!(
        items(&body)[2],
        json!({"type": "function_call_output", "call_id": "c", "output": [
            {"type": "input_text", "text": "r0"},
            {"type": "input_text", "text": "\nr1", "prompt_cache_breakpoint": breakpoint},
        ]}),
        "{body}"
    );
    // Assistant text on Chat Completions.
    let messages = vec![
        user("hi"),
        Message::new(
            MessageRole::Assistant,
            vec![with_breakpoint(Part::Text(TextPart::new("hello")))],
        ),
        user("again"),
    ];
    let chat = chat_body("gpt-4.1", messages, &StreamOptions::default());
    assert_eq!(
        chat["messages"][1]["content"],
        json!([{"type": "text", "text": "hello", "prompt_cache_breakpoint": breakpoint}]),
        "{chat}"
    );
}

#[test]
fn oai_026_prompt_cache_options_and_breakpoints_go_out() {
    use suprnova_ai::models::params::{CacheParams, ProviderValue};
    let breakpoint = json!({"mode": "explicit", "ttl": "12h"});
    let mut system_message = system("rules");
    system_message.provider_metadata =
        md(json!({"promptCacheBreakpoint": {"mode": "explicit", "ttl": "12h"}}));
    let mut messages = history(raw_call());
    messages[0] = Message::new(
        MessageRole::User,
        vec![with_breakpoint(Part::Text(TextPart::new("weather in SF?")))],
    );
    messages[2].parts[0] = with_breakpoint(messages[2].parts[0].clone());
    messages.insert(0, system_message);
    let params = InferenceRequestParams {
        cache: Some(CacheParams {
            key: Some("k".into()),
            retention: ProviderValue::Value("24h".into()),
            ..Default::default()
        }),
        ..Default::default()
    };
    let options = with_openai(
        params,
        json!({"promptCacheOptions": {"mode": "explicit", "ttl": "30m", "prewarm": true}}),
    );
    let body = body_for("gpt-4.1", messages.clone(), &options);
    assert_eq!(body["prompt_cache_key"], "k");
    assert_eq!(body["prompt_cache_retention"], "24h");
    assert_eq!(
        body["prompt_cache_options"],
        json!({"mode": "explicit", "ttl": "30m", "prewarm": true})
    );
    let input = items(&body);
    assert_eq!(
        input[0],
        json!({"role": "system", "content": [{"type": "input_text", "text": "rules", "prompt_cache_breakpoint": breakpoint}]})
    );
    assert_eq!(
        input[1]["content"][0]["prompt_cache_breakpoint"], breakpoint,
        "{body}"
    );
    let output = input
        .iter()
        .find(|item| item["type"] == "function_call_output")
        .unwrap();
    assert_eq!(
        output["output"],
        json!([{"type": "input_text", "text": "sunny", "prompt_cache_breakpoint": breakpoint}])
    );
    let chat_options = openai(json!({"promptCacheOptions": {"mode": "explicit", "ttl": "30m"}}));
    let chat = chat_body("gpt-4.1", messages, &chat_options);
    assert_eq!(
        chat["prompt_cache_options"],
        json!({"mode": "explicit", "ttl": "30m"})
    );
    assert_eq!(
        chat["messages"][0],
        json!({"role": "system", "content": [{"type": "text", "text": "rules", "prompt_cache_breakpoint": breakpoint}]})
    );
    assert_eq!(
        chat["messages"][1]["content"],
        json!([{"type": "text", "text": "weather in SF?", "prompt_cache_breakpoint": breakpoint}])
    );
    let tool = chat["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["role"] == "tool")
        .unwrap();
    assert_eq!(
        tool["content"],
        json!([{"type": "text", "text": "sunny", "prompt_cache_breakpoint": breakpoint}])
    );
}

#[tokio::test]
async fn oai_026_a_core_cache_mode_fails_before_a_request() {
    use suprnova_ai::models::params::{CacheParams, ProviderValue};
    for chat in [false, true] {
        let transport = Recorder::new([text_reply()]);
        let mut provider = OpenAICompatibleProvider::new(config(), transport.clone());
        if chat {
            provider = provider.with_protocol(Arc::new(OpenAIChatCompletionsProtocol));
        }
        let model = Model::new("gpt-4.1", Arc::new(provider));
        let options = StreamOptions {
            params: Some(InferenceRequestParams {
                cache: Some(CacheParams {
                    mode: ProviderValue::Value("explicit".into()),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            ..Default::default()
        };
        let error = run(&model, options).await.unwrap_err();
        assert!(
            matches!(error, suprnova_ai::Error::Validation(_)),
            "{error:?}"
        );
        assert!(transport.requests.lock().unwrap().is_empty());
    }
}

// OAI-027

fn schema_options(schema: Value) -> StreamOptions {
    StreamOptions {
        output_schema: Some(schema.try_into().unwrap()),
        ..Default::default()
    }
}

#[test]
fn oai_027_a_one_item_all_of_reference_becomes_the_reference() {
    let defs = json!({"A": {"type": "object", "properties": {"x": {"type": "string"}}}});
    let root = json!({"allOf": [{"$ref": "#/$defs/A"}], "$defs": defs});
    let body = body_for("gpt-4.1", vec![user("hi")], &schema_options(root.clone()));
    let schema = &body["text"]["format"]["schema"];
    assert_eq!(schema.get("allOf"), None, "{schema}");
    assert_eq!(schema["type"], "object", "{schema}");
    assert_eq!(schema["properties"]["x"]["type"], "string", "{schema}");
    assert_eq!(schema["additionalProperties"], false, "{schema}");
    let nested = json!({"type": "object", "properties": {"y": {"allOf": [{"$ref": "#/$defs/A"}]}}, "$defs": defs});
    let chat = chat_body("gpt-4.1", vec![user("hi")], &schema_options(nested));
    let y = &chat["response_format"]["json_schema"]["schema"]["properties"]["y"];
    assert_eq!(y.get("allOf"), None, "{y}");
    assert_eq!(y["$ref"], "#/$defs/A", "{y}");
}

#[test]
fn oai_027_property_names_and_lookarounds_are_left_out_and_logged() {
    let tool = Tool::function(
        "lookup",
        ToolSpec {
            description: None,
            params: json!({"type": "object", "properties": {
                "tags": {"type": "object", "propertyNames": {"type": "string", "pattern": "^[a-z]+$"}},
                "code": {"type": "string", "pattern": "^(?=A)[A-Z]+$"},
            }})
            .try_into()
            .unwrap(),
        },
    );
    let (body, events) = logged(|| tools_sent("gpt-4.1", vec![tool]));
    let properties = &body[0]["parameters"]["properties"];
    assert_eq!(properties["tags"].get("propertyNames"), None, "{body}");
    assert_eq!(properties["code"].get("pattern"), None, "{body}");
    assert!(
        events.contains(&("propertyNames".to_owned(), "gpt-4.1".to_owned())),
        "{events:?}"
    );
    assert!(
        events.contains(&("pattern".to_owned(), "gpt-4.1".to_owned())),
        "{events:?}"
    );
}

#[tokio::test]
async fn oai_027_another_property_names_fails_before_a_request() {
    let transport = Recorder::new([text_reply()]);
    let model = Model::new(
        "gpt-4.1",
        Arc::new(OpenAICompatibleProvider::new(config(), transport.clone())),
    );
    let options = schema_options(json!({"type": "object", "propertyNames": {"type": "number"}}));
    let error = run(&model, options).await.unwrap_err();
    assert!(
        matches!(error, suprnova_ai::Error::Validation(_)),
        "{error:?}"
    );
    assert!(transport.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn oai_003_every_operation_uses_its_path_key_organization_and_project() {
    use suprnova_ai::providers::openai::{OpenAICompletionsProtocol, OpenAIDecisionsProtocol};
    use suprnova_ai::providers::operations::{
        AudioParams, BatchParams, ImagePrompt, MediaInput, SkillFile, SkillParams,
        TranscribeParams, UploadFile, UploadParams,
    };
    let transport = Recorder::new([
        completion_stream(&["a"], "stop"),
        embeddings_reply(1, 1),
        images_reply(1, 0),
        images_reply(1, 0),
        (200, vec![1]),
        transcription_reply(json!({"text": "hi"})),
        json_reply(file_object()),
        json_reply(json!({"id": "skill_1"})),
        json_reply(json!({"id": "file-in"})),
        json_reply(batch_object("validating")),
        json_reply(json!({"value": "ek_1"})),
        json_reply(
            json!({"session": {"id": "s1"}, "transport": {"type": "webrtc", "sdp": "answer"}}),
        ),
        json_reply(
            json!({"answers": [{"type": "predicate", "name": "urgent", "probability": 0.5}]}),
        ),
    ]);
    let provider = Arc::new(OpenAICompatibleProvider::new(
        org_config(),
        transport.clone(),
    ));
    let model = |id: &str| Model::new(id, provider.clone());
    let none = || suprnova_ai::ops::OperationOptions::default();
    run(
        &model("gpt-3.5-turbo-instruct").with_protocol(Arc::new(OpenAICompletionsProtocol::new())),
        StreamOptions::default(),
    )
    .await
    .unwrap();
    suprnova_ai::ops::embed(
        &model("text-embedding-3-small"),
        vec!["a".into()],
        Default::default(),
        none(),
    )
    .await
    .unwrap();
    suprnova_ai::ops::generate_image(
        &model("gpt-image-1"),
        text_prompt("a cat"),
        Default::default(),
        none(),
    )
    .await
    .unwrap();
    let edit = ImagePrompt {
        text: Some("hat".into()),
        images: vec![MediaInput::Bytes(vec![1])],
        mask: None,
    };
    suprnova_ai::ops::generate_image(&model("gpt-image-1"), edit, Default::default(), none())
        .await
        .unwrap();
    suprnova_ai::ops::generate_audio(
        &model("tts-1"),
        speech_prompt("hi"),
        AudioParams::default(),
        none(),
    )
    .await
    .unwrap();
    suprnova_ai::ops::transcribe(
        &model("whisper-1"),
        mp3(),
        TranscribeParams::default(),
        none(),
    )
    .await
    .unwrap();
    let file = UploadFile {
        data: Bytes::from_static(b"x"),
        media_type: "text/plain".into(),
        filename: Some("x.txt".into()),
    };
    suprnova_ai::ops::upload_file(provider.as_ref(), file, UploadParams::default(), none())
        .await
        .unwrap();
    let skill = vec![SkillFile {
        path: "s/SKILL.md".into(),
        data: Bytes::from_static(b"# S"),
    }];
    suprnova_ai::ops::upload_skill(provider.as_ref(), skill, SkillParams::default(), none())
        .await
        .unwrap();
    suprnova_ai::ops::batch::start(
        provider.as_ref(),
        vec![text_request("r1", "gpt-4.1")],
        BatchParams::default(),
        none(),
    )
    .await
    .unwrap();
    suprnova_ai::ops::realtime::create_token(&model("gpt-realtime"), Default::default(), none())
        .await
        .unwrap();
    suprnova_ai::ops::realtime::create_webrtc_session(
        &model("gpt-live-1"),
        "offer".into(),
        Default::default(),
        none(),
    )
    .await
    .unwrap();
    let urgent: std::collections::BTreeMap<String, Question> = decision_questions()
        .into_iter()
        .filter(|(id, _)| id == "urgent")
        .collect();
    suprnova_ai::ops::experimental::evaluate(
        &model("gpt-6-luna").with_protocol(Arc::new(OpenAIDecisionsProtocol)),
        EvaluationInput::Text("x".into()),
        urgent,
        EvaluationParams::default(),
        none(),
    )
    .await
    .unwrap();
    let paths = [
        "/completions",
        "/embeddings",
        "/images/generations",
        "/images/edits",
        "/audio/speech",
        "/audio/transcriptions",
        "/files",
        "/skills",
        "/files",
        "/batches",
        "/realtime/client_secrets",
        "/live/sessions",
        "/decisions",
    ];
    assert_eq!(transport.requests.lock().unwrap().len(), paths.len());
    for (index, path) in paths.iter().enumerate() {
        assert_eq!(
            url(&transport, index),
            format!("https://api.openai.com/v1{path}")
        );
        assert_eq!(
            header(&transport, index, "authorization").as_deref(),
            Some("Bearer sk-test"),
            "{path}"
        );
        assert_eq!(
            header(&transport, index, "OpenAI-Organization").as_deref(),
            Some("org_1"),
            "{path}"
        );
        assert_eq!(
            header(&transport, index, "OpenAI-Project").as_deref(),
            Some("proj_1"),
            "{path}"
        );
    }
}
