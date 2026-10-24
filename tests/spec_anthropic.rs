//! Tests for ANT-001 to ANT-020 in docs/spec/anthropic.md. Each test name
//! starts with the requirement it checks; scripts/check-anthropic.sh selects
//! them by it. The tests record each request the crate sends and answer it
//! with a reply written in the test, so no request leaves the machine. They
//! show the crate sends what the AI SDK for TypeScript 7.0.148 sends, not
//! that Anthropic still accepts it.

use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

use bytes::Bytes;
use futures::{StreamExt, future::BoxFuture, stream};
use serde_json::{Value, json};
use suprnova_ai::{
    Model, ProviderConfig, StreamOptions,
    models::params::{InferenceRequestParams, ProviderValue, TemperatureSamplerParams},
    providers::anthropic::AnthropicCompatibleProvider,
    transport::{HttpRequest, HttpResponse, HttpTransport, TransportError},
    types::{
        Event, Message, MessageRole, Part,
        messages::{FileData, FilePart, TextPart, ToolCallPart},
        tools::{Tool, ToolConfig},
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
    fn header(&self, index: usize, name: &str) -> Option<String> {
        self.requests.lock().unwrap()[index]
            .headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.clone())
    }
    fn url(&self, index: usize) -> String {
        self.requests.lock().unwrap()[index].url.clone()
    }
}
impl HttpTransport for Recorder {
    fn send(&self, request: HttpRequest) -> BoxFuture<'_, Result<HttpResponse, TransportError>> {
        self.requests.lock().unwrap().push(request);
        let reply = self.replies.lock().unwrap().pop_front();
        Box::pin(async move {
            let (status, bytes) =
                reply.ok_or_else(|| TransportError::Request("no scripted reply left".into()))?;
            Ok(HttpResponse {
                status,
                headers: Vec::new(),
                body: stream::iter([Ok(Bytes::from(bytes))]).boxed(),
            })
        })
    }
}

fn config() -> ProviderConfig {
    ProviderConfig {
        api_key_value: Some("sk-ant-test".into()),
        ..Default::default()
    }
}
fn model(id: &str, transport: Arc<Recorder>) -> Model {
    Model::new(
        id,
        Arc::new(AnthropicCompatibleProvider::new(config(), transport)),
    )
}
fn user(text: &str) -> Message {
    Message::new(MessageRole::User, vec![Part::Text(TextPart::new(text))])
}
fn system(text: &str) -> Message {
    Message::new(MessageRole::System, vec![Part::Text(TextPart::new(text))])
}
fn assistant(parts: Vec<Part>) -> Message {
    Message::new(MessageRole::Assistant, parts)
}
fn json_reply(value: Value) -> (u16, Vec<u8>) {
    (200, value.to_string().into_bytes())
}
/// A Messages event stream of these events.
fn sse(events: &[Value]) -> (u16, Vec<u8>) {
    let text: String = events
        .iter()
        .map(|e| format!("event: {}\ndata: {e}\n\n", e["type"].as_str().unwrap()))
        .collect();
    (200, text.into_bytes())
}
/// A stream of these content events between a message's start and stop.
fn message(blocks: Vec<Value>, usage: Value, reason: &str) -> (u16, Vec<u8>) {
    let mut events = vec![json!({"type": "message_start", "message": {
        "id": "msg_1", "model": "claude-sonnet-4-6", "content": [], "stop_reason": null,
        "usage": {"input_tokens": 1, "output_tokens": 0}}})];
    events.extend(blocks);
    events.push(json!({"type": "message_delta", "delta": {"stop_reason": reason}, "usage": usage}));
    events.push(json!({"type": "message_stop"}));
    sse(&events)
}
fn text_reply() -> (u16, Vec<u8>) {
    message(
        vec![
            json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}}),
            json!({"type": "content_block_stop", "index": 0}),
        ],
        json!({"output_tokens": 1}),
        "end_turn",
    )
}
/// Stream one call to its end and give its events.
async fn run(model: &Model, messages: Vec<Message>, options: StreamOptions) -> Vec<Event> {
    let mut handle = suprnova_ai::stream(model, messages, options);
    let mut events = vec![];
    while let Some(event) = handle.next().await {
        events.push(event.unwrap());
    }
    events
}
/// The body a call to `id` with these messages and options sends.
async fn body_for(id: &str, messages: Vec<Message>, options: StreamOptions) -> Value {
    let transport = Recorder::new([text_reply()]);
    run(&model(id, transport.clone()), messages, options).await;
    transport.body(0)
}
fn anthropic(options: Value) -> StreamOptions {
    StreamOptions {
        params: Some(InferenceRequestParams::default().with_provider_param("anthropic", options)),
        ..Default::default()
    }
}
fn metadata(value: Value) -> Option<suprnova_ai::types::messages::Metadata> {
    Some([("anthropic".to_owned(), value)].into())
}

// ANT-001

#[tokio::test]
async fn ant_001_an_auth_token_goes_as_a_bearer_token() {
    let transport = Recorder::new([text_reply()]);
    let config = ProviderConfig {
        env: [("ANTHROPIC_AUTH_TOKEN".to_owned(), "tok-1".to_owned())].into(),
        ..Default::default()
    };
    let model = Model::new(
        "claude-sonnet-4-6",
        Arc::new(AnthropicCompatibleProvider::new(config, transport.clone())),
    );
    run(&model, vec![user("hi")], StreamOptions::default()).await;
    assert_eq!(
        transport.header(0, "authorization").as_deref(),
        Some("Bearer tok-1")
    );
    assert_eq!(transport.header(0, "x-api-key"), None);
    assert_eq!(transport.url(0), "https://api.anthropic.com/v1/messages");
}

// ANT-002

#[tokio::test]
async fn ant_002_an_option_goes_out_under_its_wire_name() {
    let body = body_for(
        "claude-sonnet-4-6",
        vec![user("hi")],
        anthropic(json!({"effort": "low"})),
    )
    .await;
    assert_eq!(body["output_config"]["effort"], "low", "{body}");
}

// ANT-003

#[tokio::test]
async fn ant_003_max_tokens_default_to_the_model_limit() {
    let body = body_for(
        "claude-sonnet-4-6",
        vec![user("hi")],
        StreamOptions::default(),
    )
    .await;
    assert_eq!(body["max_tokens"], 128000, "{body}");
}

// ANT-004

#[tokio::test]
async fn ant_004_a_temperature_on_a_model_that_rejects_sampling_is_left_out() {
    let mut params = InferenceRequestParams::default();
    params = params.with_temperature(0.5);
    let _ = TemperatureSamplerParams {
        temperature: ProviderValue::Value(0.5),
    };
    let body = body_for(
        "claude-opus-5",
        vec![user("hi")],
        StreamOptions {
            params: Some(params),
            ..Default::default()
        },
    )
    .await;
    assert!(body.get("temperature").is_none(), "{body}");
}

// ANT-005

#[tokio::test]
async fn ant_005_the_json_tool_mode_sends_the_json_tool() {
    let mut options = anthropic(json!({"structuredOutputMode": "jsonTool"}));
    options.output_schema = Some(
        json!({"type": "object", "properties": {"a": {"type": "number"}}})
            .try_into()
            .unwrap(),
    );
    let body = body_for("claude-sonnet-4-6", vec![user("hi")], options).await;
    assert_eq!(body["tools"][0]["name"], "json", "{body}");
    assert!(body["output_config"].get("format").is_none(), "{body}");
}

// ANT-006

#[tokio::test]
async fn ant_006_a_provider_tool_goes_under_its_anthropic_name() {
    let search = Tool::provider(
        "search",
        ToolConfig {
            id: Some("anthropic.web_search_20250305".into()),
            ..Default::default()
        },
    )
    .unwrap();
    let body = body_for(
        "claude-sonnet-4-6",
        vec![user("hi")],
        StreamOptions {
            tools: vec![search],
            ..Default::default()
        },
    )
    .await;
    assert_eq!(
        body["tools"][0],
        json!({"type": "web_search_20250305", "name": "web_search"}),
        "{body}"
    );
}

// ANT-007

#[tokio::test]
async fn ant_007_a_toolset_member_call_is_a_call_of_the_toolset_tool() {
    let computer = Tool::provider(
        "computer",
        ToolConfig {
            id: Some("anthropic.computer_toolset_20260801".into()),
            ..Default::default()
        },
    )
    .unwrap();
    let transport = Recorder::new([message(
        vec![
            json!({"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "left_click", "toolset_name": "computer", "input": {}}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{\"coordinate\": [1, 2]}"}}),
            json!({"type": "content_block_stop", "index": 0}),
        ],
        json!({"output_tokens": 1}),
        "tool_use",
    )]);
    let events = run(
        &model("claude-sonnet-4-6", transport),
        vec![user("click")],
        StreamOptions {
            tools: vec![computer],
            ..Default::default()
        },
    )
    .await;
    let call = events
        .iter()
        .find_map(|e| match e {
            Event::ToolEnd(end) => Some(end.tool_call.clone()),
            _ => None,
        })
        .expect("a tool call");
    assert_eq!(call.tool_name, "computer");
    assert_eq!(
        serde_json::from_str::<Value>(&call.tool_args).unwrap(),
        json!({"action": "left_click", "coordinate": [1, 2]})
    );
}

// ANT-008

#[tokio::test]
async fn ant_008_a_code_execution_subtool_is_a_code_execution_call() {
    let transport = Recorder::new([message(
        vec![
            json!({"type": "content_block_start", "index": 0, "content_block": {"type": "server_tool_use", "id": "srvtoolu_1", "name": "bash_code_execution", "input": {}}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{\"command\": \"ls\"}"}}),
            json!({"type": "content_block_stop", "index": 0}),
        ],
        json!({"output_tokens": 1}),
        "end_turn",
    )]);
    let events = run(
        &model("claude-sonnet-4-6", transport),
        vec![user("ls")],
        StreamOptions::default(),
    )
    .await;
    let call = events
        .iter()
        .find_map(|e| match e {
            Event::BuiltinToolEnd(end) => Some(end.tool_call.clone()),
            _ => None,
        })
        .expect("a built-in call");
    assert_eq!(call.tool_name, "code_execution");
    assert_eq!(
        serde_json::from_str::<Value>(&call.tool_args).unwrap()["type"],
        "bash_code_execution"
    );
}

// ANT-009

#[tokio::test]
async fn ant_009_arguments_that_are_not_an_object_are_wrapped() {
    let body = body_for(
        "claude-sonnet-4-6",
        vec![
            user("go"),
            assistant(vec![Part::ToolCall(ToolCallPart::new(
                "toolu_1", "f", "[1]",
            ))]),
            Message::new(
                MessageRole::Tool,
                vec![Part::ToolResult(
                    suprnova_ai::types::messages::ToolResultPart::new("toolu_1", "f", json!("ok")),
                )],
            ),
        ],
        StreamOptions::default(),
    )
    .await;
    assert_eq!(
        body["messages"][1]["content"][0]["input"],
        json!({"rawInvalidInput": [1]}),
        "{body}"
    );
}

// ANT-010

#[tokio::test]
async fn ant_010_a_wildcard_image_type_is_resolved_from_the_bytes() {
    let png = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 0];
    let file = FilePart {
        id: "f1".into(),
        data: FileData::Bytes(png),
        media_type: "image/*".into(),
        filename: None,
        provider_metadata: None,
    };
    let body = body_for(
        "claude-sonnet-4-6",
        vec![Message::new(MessageRole::User, vec![Part::File(file)])],
        StreamOptions::default(),
    )
    .await;
    assert_eq!(
        body["messages"][0]["content"][0]["source"]["media_type"], "image/png",
        "{body}"
    );
}

// ANT-011

#[tokio::test]
async fn ant_011_a_later_system_message_goes_in_place() {
    let transport = Recorder::new([text_reply()]);
    run(
        &model("claude-sonnet-4-6", transport.clone()),
        vec![system("first"), user("a"), system("second"), user("b")],
        StreamOptions::default(),
    )
    .await;
    let body = transport.body(0);
    assert_eq!(
        body["system"],
        json!([{"type": "text", "text": "first"}]),
        "{body}"
    );
    assert_eq!(
        body["messages"][1],
        json!({"role": "system", "content": [{"type": "text", "text": "second"}]}),
        "{body}"
    );
    assert!(
        transport
            .header(0, "anthropic-beta")
            .is_some_and(|b| b.contains("mid-conversation-system-2026-04-07"))
    );
}

// ANT-012

#[tokio::test]
async fn ant_012_tool_calls_move_after_the_other_blocks_of_their_run() {
    let body = body_for(
        "claude-sonnet-4-6",
        vec![
            user("go"),
            assistant(vec![
                Part::Text(TextPart::new("one")),
                Part::ToolCall(ToolCallPart::new("toolu_1", "f", "{}")),
                Part::Text(TextPart::new("two")),
            ]),
            Message::new(
                MessageRole::Tool,
                vec![Part::ToolResult(
                    suprnova_ai::types::messages::ToolResultPart::new("toolu_1", "f", json!("ok")),
                )],
            ),
        ],
        StreamOptions::default(),
    )
    .await;
    let kinds: Vec<&str> = body["messages"][1]["content"]
        .as_array()
        .unwrap()
        .iter()
        .map(|b| b["type"].as_str().unwrap())
        .collect();
    assert_eq!(kinds, ["text", "text", "tool_use"], "{body}");
}

// ANT-013

#[tokio::test]
async fn ant_013_redacted_thinking_is_reasoning_with_its_data() {
    let transport = Recorder::new([message(
        vec![
            json!({"type": "content_block_start", "index": 0, "content_block": {"type": "redacted_thinking", "data": "abc"}}),
            json!({"type": "content_block_stop", "index": 0}),
        ],
        json!({"output_tokens": 1}),
        "end_turn",
    )]);
    let mut handle = suprnova_ai::stream(
        &model("claude-sonnet-4-6", transport),
        vec![user("hi")],
        StreamOptions::default(),
    );
    while let Some(event) = handle.next().await {
        event.unwrap();
    }
    let reasoning = handle
        .message()
        .parts
        .iter()
        .find_map(|p| match p {
            Part::Reasoning(r) => Some(r.clone()),
            _ => None,
        })
        .expect("a reasoning part");
    assert_eq!(
        reasoning.provider_metadata.unwrap()["anthropic"]["redactedData"],
        "abc"
    );
}

// ANT-014

#[tokio::test]
async fn ant_014_input_tokens_count_cache_writes_and_reads() {
    let transport = Recorder::new([message(
        vec![],
        json!({"input_tokens": 10, "cache_creation_input_tokens": 2, "cache_read_input_tokens": 3, "output_tokens": 20}),
        "end_turn",
    )]);
    let mut handle = suprnova_ai::stream(
        &model("claude-sonnet-4-6", transport),
        vec![user("hi")],
        StreamOptions::default(),
    );
    while let Some(event) = handle.next().await {
        event.unwrap();
    }
    let usage = handle.message().usage.clone().unwrap();
    assert_eq!(
        (
            usage.input_tokens,
            usage.cache_write_tokens,
            usage.cache_read_tokens
        ),
        (15, Some(2), Some(3))
    );
}

// ANT-015

#[tokio::test]
async fn ant_015_a_part_breakpoint_goes_on_its_block() {
    let mut text = TextPart::new("long context");
    text.provider_metadata = metadata(json!({"cacheControl": {"type": "ephemeral"}}));
    let body = body_for(
        "claude-sonnet-4-6",
        vec![Message::new(MessageRole::User, vec![Part::Text(text)])],
        StreamOptions::default(),
    )
    .await;
    assert_eq!(
        body["messages"][0]["content"][0]["cache_control"],
        json!({"type": "ephemeral"}),
        "{body}"
    );
}

// ANT-016

#[tokio::test]
async fn ant_016_a_model_that_is_not_claude_takes_the_default_limit() {
    let body = body_for("MiniMax-M2", vec![user("hi")], StreamOptions::default()).await;
    assert_eq!(body["max_tokens"], 4096, "{body}");
}

// ANT-017

#[tokio::test]
async fn ant_017_files_are_uploaded() {
    use suprnova_ai::providers::operations::{UploadFile, UploadParams};
    let transport = Recorder::new([json_reply(json!({
        "id": "file_1", "type": "file", "filename": "r.pdf", "mime_type": "application/pdf",
        "size_bytes": 4, "created_at": "2026-10-24T00:00:00Z", "downloadable": false}))]);
    let provider = AnthropicCompatibleProvider::new(config(), transport.clone());
    let file = UploadFile {
        data: Bytes::from_static(b"%PDF"),
        media_type: "application/pdf".into(),
        filename: Some("r.pdf".into()),
    };
    let uploaded =
        suprnova_ai::ops::upload_file(&provider, file, UploadParams::default(), Default::default())
            .await
            .unwrap();
    assert_eq!(transport.url(0), "https://api.anthropic.com/v1/files");
    assert_eq!(
        uploaded.value.provider_reference,
        [("anthropic".to_owned(), "file_1".to_owned())].into()
    );
}

// ANT-018

#[tokio::test]
async fn ant_018_skills_are_uploaded() {
    use suprnova_ai::providers::operations::{SkillFile, SkillParams};
    let transport = Recorder::new([json_reply(json!({
        "id": "skill_1", "display_title": null, "name": "notes", "description": "Notes",
        "latest_version": null, "source": "custom", "created_at": "2026-10-24T00:00:00Z",
        "updated_at": "2026-10-24T00:00:00Z"}))]);
    let provider = AnthropicCompatibleProvider::new(config(), transport.clone());
    let files = vec![SkillFile {
        path: "notes/SKILL.md".into(),
        data: Bytes::from_static(b"# Notes"),
    }];
    let uploaded = suprnova_ai::ops::upload_skill(
        &provider,
        files,
        SkillParams::default(),
        Default::default(),
    )
    .await
    .unwrap();
    assert_eq!(transport.url(0), "https://api.anthropic.com/v1/skills");
    assert_eq!(uploaded.value.name.as_deref(), Some("notes"));
}

// ANT-019

#[tokio::test]
async fn ant_019_a_batch_sends_its_requests() {
    use suprnova_ai::providers::operations::{BatchParams, BatchRequest, BatchRequestKind};
    let transport = Recorder::new([json_reply(json!({
        "id": "msgbatch_1", "type": "message_batch", "processing_status": "in_progress",
        "request_counts": {"processing": 1, "succeeded": 0, "errored": 0, "canceled": 0, "expired": 0},
        "created_at": "2026-10-24T00:00:00Z", "expires_at": "2026-10-25T00:00:00Z"}))]);
    let provider = AnthropicCompatibleProvider::new(config(), transport.clone());
    let request = BatchRequest {
        id: "r1".into(),
        model_id: "claude-sonnet-4-6".into(),
        kind: BatchRequestKind::Text {
            messages: vec![user("hi")],
            options: StreamOptions::default(),
        },
    };
    suprnova_ai::ops::batch::start(
        &provider,
        vec![request],
        BatchParams::default(),
        Default::default(),
    )
    .await
    .unwrap();
    assert_eq!(
        transport.url(0),
        "https://api.anthropic.com/v1/messages/batches"
    );
    assert_eq!(transport.body(0)["requests"][0]["custom_id"], "r1");
}

// ANT-020

#[tokio::test]
async fn ant_020_evaluation_makes_one_messages_call() {
    use suprnova_ai::providers::operations::{EvaluationInput, Question};
    let transport = Recorder::new([message(
        vec![
            json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "{\"q0\": 0.5}"}}),
            json!({"type": "content_block_stop", "index": 0}),
        ],
        json!({"output_tokens": 1}),
        "end_turn",
    )]);
    let questions = [(
        "urgent".to_owned(),
        Question::Noul {
            instructions: EvaluationInput::Text("Is it urgent".into()),
            criteria: None,
        },
    )]
    .into();
    suprnova_ai::ops::experimental::evaluate(
        &model("claude-sonnet-4-6", transport.clone()),
        EvaluationInput::Text("late".into()),
        questions,
        Default::default(),
        Default::default(),
    )
    .await
    .unwrap();
    assert_eq!(transport.url(0), "https://api.anthropic.com/v1/messages");
}
