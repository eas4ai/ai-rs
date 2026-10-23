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
