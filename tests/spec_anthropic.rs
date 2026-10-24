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

impl Recorder {
    fn form(&self, index: usize) -> String {
        String::from_utf8_lossy(&self.requests.lock().unwrap()[index].body).into_owned()
    }
    fn method(&self, index: usize) -> String {
        self.requests.lock().unwrap()[index].method.clone()
    }
    fn count(&self) -> usize {
        self.requests.lock().unwrap().len()
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
fn sse_event(event: Value) -> String {
    format!(
        "event: {}\ndata: {event}\n\n",
        event["type"].as_str().unwrap()
    )
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

/// The betas of recorded request `index`, in order.
fn betas(transport: &Recorder, index: usize) -> Vec<String> {
    transport
        .header(index, "anthropic-beta")
        .map(|b| b.split(',').map(str::to_owned).collect())
        .unwrap_or_default()
}
/// The body and betas of a call to `id` with these options.
async fn sent(id: &str, messages: Vec<Message>, options: StreamOptions) -> (Value, Vec<String>) {
    let transport = Recorder::new([text_reply()]);
    run(&model(id, transport.clone()), messages, options).await;
    (transport.body(0), betas(&transport, 0))
}
/// The error a call with these options fails with, after checking that it
/// sent no request.
async fn refused(id: &str, messages: Vec<Message>, options: StreamOptions) -> suprnova_ai::Error {
    let transport = Recorder::new([]);
    let mut handle = suprnova_ai::stream(&model(id, transport.clone()), messages, options);
    let error = loop {
        match handle.next().await {
            Some(Err(error)) => break error,
            Some(Ok(_)) => {}
            None => panic!("the call did not fail"),
        }
    };
    assert!(transport.requests.lock().unwrap().is_empty(), "{error:?}");
    error
}
fn function_tool(name: &str) -> Tool {
    Tool::function(
        name,
        suprnova_ai::types::tools::ToolSpec {
            description: Some(format!("The {name} tool")),
            params: json!({"type": "object", "properties": {}})
                .try_into()
                .unwrap(),
        },
    )
}
fn with_params(params: InferenceRequestParams) -> StreamOptions {
    StreamOptions {
        params: Some(params),
        ..Default::default()
    }
}
fn with_anthropic(params: InferenceRequestParams, options: Value) -> StreamOptions {
    with_params(params.with_provider_param("anthropic", options))
}

/// The `setting` and `model` of each debug event of target
/// `suprnova_ai::anthropic` that a call logs while the guard lives.
struct Logs(Arc<Mutex<Vec<(String, String)>>>);
impl Logs {
    fn take(&self) -> Vec<(String, String)> {
        std::mem::take(&mut self.0.lock().unwrap())
    }
}
fn capture_logs() -> (Logs, tracing::subscriber::DefaultGuard) {
    let events = Arc::new(Mutex::new(vec![]));
    let guard = tracing::subscriber::set_default(Capture(events.clone()));
    (Logs(events), guard)
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
        if meta.target() != "suprnova_ai::anthropic" || *meta.level() != tracing::Level::DEBUG {
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

#[tokio::test]
async fn ant_001_a_token_with_an_api_key_value_fails_before_a_request() {
    let transport = Recorder::new([]);
    let config = ProviderConfig {
        env: [("ANTHROPIC_AUTH_TOKEN".to_owned(), "tok-1".to_owned())].into(),
        ..config()
    };
    let model = Model::new(
        "claude-sonnet-4-6",
        Arc::new(AnthropicCompatibleProvider::new(config, transport.clone())),
    );
    let mut handle = suprnova_ai::stream(&model, vec![user("hi")], StreamOptions::default());
    let error = handle.next().await.unwrap().unwrap_err();
    assert!(
        matches!(error, suprnova_ai::Error::Configuration(_)),
        "{error:?}"
    );
    assert!(transport.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn ant_001_betas_go_in_one_header_once_each() {
    use suprnova_ai::models::params::{ContextManagementParams, TokenThreshold};
    let transport = Recorder::new([text_reply()]);
    let config = ProviderConfig {
        headers: [(
            "anthropic-beta".to_owned(),
            "b1, compact-2026-01-12".to_owned(),
        )]
        .into(),
        ..config()
    };
    let model = Model::new(
        "claude-sonnet-4-6",
        Arc::new(AnthropicCompatibleProvider::new(config, transport.clone())),
    );
    let params = InferenceRequestParams {
        context_management: Some(ContextManagementParams {
            compaction: Some(TokenThreshold(100_000)),
        }),
        ..Default::default()
    };
    run(
        &model,
        vec![user("hi")],
        StreamOptions {
            params: Some(params),
            ..Default::default()
        },
    )
    .await;
    let requests = transport.requests.lock().unwrap();
    let betas: Vec<&String> = requests[0]
        .headers
        .iter()
        .filter(|(key, _)| key.eq_ignore_ascii_case("anthropic-beta"))
        .map(|(_, value)| value)
        .collect();
    assert_eq!(betas.len(), 1, "{betas:?}");
    let mut names: Vec<&str> = betas[0].split(',').collect();
    names.sort();
    assert_eq!(
        names,
        ["b1", "compact-2026-01-12", "context-management-2025-06-27"]
    );
}

#[tokio::test]
async fn ant_001_anthropic_credentials_go_only_to_anthropic() {
    // Another host behind the Messages protocol gets neither Anthropic's
    // token nor Anthropic's key, whatever the env snapshot holds.
    let env: std::collections::BTreeMap<String, String> = [
        ("ANTHROPIC_AUTH_TOKEN".to_owned(), "tok-1".to_owned()),
        ("ANTHROPIC_API_KEY".to_owned(), "sk-ant-env".to_owned()),
    ]
    .into();
    let metadata = suprnova_ai::providers::ModelsDevProvider {
        id: "other".into(),
        api: Some("https://other.test".into()),
        npm: Some("@ai-sdk/anthropic".into()),
        env: vec![],
    };
    let transport = Recorder::new([text_reply(), text_reply()]);
    let other = AnthropicCompatibleProvider::from_modelsdev(
        &metadata,
        ProviderConfig {
            env: env.clone(),
            ..Default::default()
        },
        transport.clone(),
    )
    .unwrap();
    run(
        &Model::new("claude-sonnet-4-6", Arc::new(other)),
        vec![user("hi")],
        StreamOptions::default(),
    )
    .await;
    let configured = suprnova_ai::providers::ConfiguredProvider::new(
        "proxy",
        ProviderConfig {
            default_base_url: "https://proxy.test".into(),
            env,
            ..Default::default()
        },
        Arc::new(suprnova_ai::providers::anthropic::AnthropicMessagesProtocol),
        Some(transport.clone()),
    );
    run(
        &Model::new("claude-sonnet-4-6", Arc::new(configured)),
        vec![user("hi")],
        StreamOptions::default(),
    )
    .await;
    for index in 0..2 {
        assert_eq!(transport.header(index, "authorization"), None, "{index}");
        assert_eq!(transport.header(index, "x-api-key"), None, "{index}");
    }
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

#[tokio::test]
async fn ant_002_each_option_goes_out_with_its_beta() {
    let cases: Vec<(Value, &str, Value, Option<&str>)> = vec![
        (
            json!({"thinking": {"type": "enabled", "budgetTokens": 2000}}),
            "/thinking",
            json!({"type": "enabled", "budget_tokens": 2000}),
            None,
        ),
        (
            json!({"thinking": {"type": "adaptive", "display": "updates"}}),
            "/thinking",
            json!({"type": "adaptive", "display": "updates"}),
            Some("thinking-display-updates-2026-08-18"),
        ),
        (
            json!({"thinking": {"type": "adaptive", "blockBinding": {"prefixMismatchBehavior": "drop_block"}}}),
            "/thinking",
            json!({"type": "adaptive", "block_binding": {"prefix_mismatch_behavior": "drop_block"}}),
            Some("thinking-binding-controls-2026-08-01"),
        ),
        (
            json!({"thinking": {"blockBinding": {"prefixMismatchBehavior": "error"}}}),
            "/thinking",
            json!({"block_binding": {"prefix_mismatch_behavior": "error"}}),
            Some("thinking-binding-controls-2026-08-01"),
        ),
        (
            json!({"effort": "low"}),
            "/output_config/effort",
            json!("low"),
            None,
        ),
        (
            json!({"taskBudget": {"type": "tokens", "total": 30000, "remaining": 100}}),
            "/output_config/task_budget",
            json!({"type": "tokens", "total": 30000, "remaining": 100}),
            Some("task-budgets-2026-03-13"),
        ),
        (
            json!({"speed": "fast"}),
            "/speed",
            json!("fast"),
            Some("fast-mode-2026-02-01"),
        ),
        (
            json!({"serviceTier": "standard_only"}),
            "/service_tier",
            json!("standard_only"),
            None,
        ),
        (
            json!({"inferenceGeo": "us"}),
            "/inference_geo",
            json!("us"),
            None,
        ),
        (
            json!({"fallbacks": "default"}),
            "/fallbacks",
            json!("default"),
            Some("server-side-fallback-2026-07-01"),
        ),
        (
            json!({"fallbacks": [{"model": "claude-sonnet-4-6", "max_tokens": 100}]}),
            "/fallbacks",
            json!([{"model": "claude-sonnet-4-6", "max_tokens": 100}]),
            Some("server-side-fallback-2026-06-01"),
        ),
        (
            json!({"cacheControl": {"type": "ephemeral", "ttl": "1h"}}),
            "/cache_control",
            json!({"type": "ephemeral", "ttl": "1h"}),
            None,
        ),
        (
            json!({"metadata": {"userId": "u1"}}),
            "/metadata",
            json!({"user_id": "u1"}),
            None,
        ),
        (
            json!({"mcpServers": [{"type": "url", "name": "docs", "url": "https://mcp.test", "authorizationToken": "t1", "toolConfiguration": {"enabled": true, "allowedTools": ["search"]}}]}),
            "/mcp_servers",
            json!([{"type": "url", "name": "docs", "url": "https://mcp.test", "authorization_token": "t1", "tool_configuration": {"allowed_tools": ["search"], "enabled": true}}]),
            Some("mcp-client-2025-04-04"),
        ),
        (
            json!({"container": {"id": "c1"}}),
            "/container",
            json!("c1"),
            None,
        ),
        (
            json!({"container": {"id": "c1", "skills": [{"type": "anthropic", "skillId": "pptx", "version": "latest"}]}}),
            "/container",
            json!({"id": "c1", "skills": [{"type": "anthropic", "skill_id": "pptx", "version": "latest"}]}),
            Some("skills-2025-10-02"),
        ),
        (
            json!({"safeguards": [{"type": "dangerous_tool_use", "classifierContext": {"mode": "auto"}}]}),
            "/safeguards",
            json!([{"type": "dangerous_tool_use", "classifier_context": {"mode": "auto"}}]),
            Some("dangerous-tool-use-2026-09-03"),
        ),
        (
            json!({"compaction": {"type": "summarize", "instructions": "short"}}),
            "/compaction",
            json!({"type": "summarize", "instructions": "short"}),
            Some("compact-2026-09-04"),
        ),
        (
            json!({"contextManagement": {"edits": [{"type": "clear_tool_uses_20250919", "trigger": {"type": "tool_uses", "value": 3}, "keep": {"type": "tool_uses", "value": 1}, "clearAtLeast": {"type": "input_tokens", "value": 5}, "clearToolInputs": true, "excludeTools": ["a"]}]}}),
            "/context_management",
            json!({"edits": [{"type": "clear_tool_uses_20250919", "trigger": {"type": "tool_uses", "value": 3}, "keep": {"type": "tool_uses", "value": 1}, "clear_at_least": {"type": "input_tokens", "value": 5}, "clear_tool_inputs": true, "exclude_tools": ["a"]}]}),
            Some("context-management-2025-06-27"),
        ),
        (
            json!({"contextManagement": {"edits": [{"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 90000}, "pauseAfterCompaction": true, "instructions": "keep facts"}]}}),
            "/context_management",
            json!({"edits": [{"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 90000}, "pause_after_compaction": true, "instructions": "keep facts"}]}),
            Some("compact-2026-01-12"),
        ),
        (
            json!({"contextManagement": {"edits": [{"type": "clear_thinking_20251015", "keep": {"type": "thinking_turns", "value": 2}}]}}),
            "/context_management",
            json!({"edits": [{"type": "clear_thinking_20251015", "keep": {"type": "thinking_turns", "value": 2}}]}),
            Some("context-management-2025-06-27"),
        ),
        (
            json!({"anthropicBeta": ["my-beta-2026-01-01"]}),
            "/model",
            json!("claude-sonnet-4-6"),
            Some("my-beta-2026-01-01"),
        ),
    ];
    for (options, pointer, expected, beta) in cases {
        let (body, betas) = sent(
            "claude-sonnet-4-6",
            vec![user("hi")],
            anthropic(options.clone()),
        )
        .await;
        assert_eq!(body.pointer(pointer), Some(&expected), "{options}: {body}");
        if let Some(beta) = beta {
            assert!(betas.iter().any(|b| b == beta), "{options}: {betas:?}");
        }
    }
    // Container skills need three betas.
    let (_, betas) = sent(
        "claude-sonnet-4-6",
        vec![user("hi")],
        anthropic(json!({"container": {"skills": [{"type": "anthropic", "skillId": "pptx"}]}})),
    )
    .await;
    for beta in [
        "code-execution-2025-08-25",
        "skills-2025-10-02",
        "files-api-2025-04-14",
    ] {
        assert!(betas.iter().any(|b| b == beta), "{betas:?}");
    }
}

#[tokio::test]
async fn ant_002_a_custom_skill_names_its_anthropic_reference() {
    let (body, _) = sent(
        "claude-sonnet-4-6",
        vec![user("hi")],
        anthropic(json!({"container": {"skills": [{"type": "custom", "providerReference": {"anthropic": "skill_01", "openai": "x"}, "version": "2"}]}})),
    )
    .await;
    assert_eq!(
        body["container"],
        json!({"skills": [{"type": "custom", "skill_id": "skill_01", "version": "2"}]})
    );
}

#[tokio::test]
async fn ant_002_disabled_parallel_tool_use_sends_an_auto_choice() {
    let mut options = anthropic(json!({"disableParallelToolUse": true}));
    options.tools = vec![function_tool("weather")];
    let (body, _) = sent("claude-sonnet-4-6", vec![user("hi")], options).await;
    assert_eq!(
        body["tool_choice"],
        json!({"type": "auto", "disable_parallel_tool_use": true})
    );
}

#[tokio::test]
async fn ant_002_an_option_takes_precedence_over_its_core_parameter() {
    use suprnova_ai::models::params::ProviderServiceParams;
    let params = InferenceRequestParams {
        provider_service: Some(ProviderServiceParams {
            service_tier: ProviderValue::Value("auto".into()),
        }),
        safety_identifier: Some("core-user".into()),
        ..Default::default()
    };
    let (body, _) = sent(
        "claude-sonnet-4-6",
        vec![user("hi")],
        with_anthropic(
            params,
            json!({"serviceTier": "standard_only", "metadata": {"userId": "option-user"}}),
        ),
    )
    .await;
    assert_eq!(body["service_tier"], "standard_only", "{body}");
    assert_eq!(
        body["metadata"],
        json!({"user_id": "option-user"}),
        "{body}"
    );
}

#[tokio::test]
async fn ant_002_unknown_and_invalid_options_fail_before_a_request() {
    for options in [
        json!({"reasoningEffort": "low"}),
        json!({"effort": "huge"}),
        json!({"effort": null}),
        json!({"thinking": {"type": "enabled", "display": "omitted"}}),
        json!({"taskBudget": {"type": "tokens", "total": 100}}),
        json!({"compaction": {"type": "summarize"}, "contextManagement": {"edits": []}}),
        json!({"contextManagement": {"edits": [{"type": "unknown_edit"}]}}),
        json!("not an object"),
    ] {
        let error = refused(
            "claude-sonnet-4-6",
            vec![user("hi")],
            anthropic(options.clone()),
        )
        .await;
        assert!(
            matches!(error, suprnova_ai::Error::Validation(_)),
            "{options}: {error:?}"
        );
    }
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

fn reasoning(effort: Option<&str>) -> InferenceRequestParams {
    InferenceRequestParams::default()
        .with_reasoning_effort(ProviderValue::Value(effort.map(str::to_owned)))
}

#[tokio::test]
async fn ant_003_the_output_limit_adds_the_thinking_budget() {
    use suprnova_ai::models::params::OutputParams;
    let params = InferenceRequestParams {
        output: Some(OutputParams {
            max_tokens: Some(1000),
            ..Default::default()
        }),
        ..Default::default()
    };
    let (body, _) = sent(
        "claude-sonnet-4-6",
        vec![user("hi")],
        with_anthropic(
            params,
            json!({"thinking": {"type": "enabled", "budgetTokens": 2000}}),
        ),
    )
    .await;
    assert_eq!(body["max_tokens"], 3000, "{body}");
}

#[tokio::test]
async fn ant_003_a_core_reasoning_effort_maps_as_typescript_maps_it() {
    let (body, _) = sent(
        "claude-sonnet-4-6",
        vec![user("hi")],
        with_params(reasoning(Some("high"))),
    )
    .await;
    assert_eq!(
        body["thinking"],
        json!({"type": "adaptive", "display": "summarized"}),
        "{body}"
    );
    assert_eq!(body["output_config"], json!({"effort": "high"}), "{body}");
    let (body, _) = sent(
        "claude-sonnet-4-6",
        vec![user("hi")],
        with_params(reasoning(Some("minimal"))),
    )
    .await;
    assert_eq!(body["output_config"], json!({"effort": "low"}), "{body}");
    // xhigh is max where the model takes no xhigh.
    let (body, _) = sent(
        "claude-sonnet-4-6",
        vec![user("hi")],
        with_params(reasoning(Some("xhigh"))),
    )
    .await;
    assert_eq!(body["output_config"], json!({"effort": "max"}), "{body}");
    let (body, _) = sent(
        "claude-opus-4-8",
        vec![user("hi")],
        with_params(reasoning(Some("xhigh"))),
    )
    .await;
    assert_eq!(body["output_config"], json!({"effort": "xhigh"}), "{body}");
    // Budget thinking: 30 percent of a 64000 limit.
    let (body, _) = sent(
        "claude-sonnet-4-5",
        vec![user("hi")],
        with_params(reasoning(Some("medium"))),
    )
    .await;
    assert_eq!(
        body["thinking"],
        json!({"type": "enabled", "budget_tokens": 19200}),
        "{body}"
    );
    assert_eq!(body["max_tokens"], 64000, "{body}");
    // At least 1024.
    let (body, _) = sent(
        "claude-3-haiku-20240307",
        vec![user("hi")],
        with_params(reasoning(Some("minimal"))),
    )
    .await;
    assert_eq!(
        body["thinking"],
        json!({"type": "enabled", "budget_tokens": 1024}),
        "{body}"
    );
    // Disabled reasoning.
    let (body, _) = sent(
        "claude-sonnet-4-6",
        vec![user("hi")],
        with_params(reasoning(None)),
    )
    .await;
    assert_eq!(body["thinking"], json!({"type": "disabled"}), "{body}");
    let (body, _) = sent(
        "claude-sonnet-5-5",
        vec![user("hi")],
        with_params(reasoning(None)),
    )
    .await;
    assert_eq!(body["thinking"], json!({"type": "between_tools"}), "{body}");
    let (body, _) = sent(
        "claude-opus-5-5",
        vec![user("hi")],
        with_params(reasoning(None)),
    )
    .await;
    assert!(body.get("thinking").is_none(), "{body}");
    assert_eq!(body["output_config"], json!({"effort": "low"}), "{body}");
    // An effort option wins over the core effort.
    let (body, _) = sent(
        "claude-sonnet-4-6",
        vec![user("hi")],
        with_anthropic(reasoning(Some("high")), json!({"effort": "medium"})),
    )
    .await;
    assert_eq!(body["output_config"], json!({"effort": "medium"}), "{body}");
    assert!(body.get("thinking").is_none(), "{body}");
}

#[tokio::test]
async fn ant_003_a_core_reasoning_summary_sets_the_display() {
    use suprnova_ai::models::params::OutputParams;
    let summary = |summary: Option<&str>, effort: Option<Option<&str>>| {
        let params = InferenceRequestParams {
            output: Some(OutputParams {
                reasoning_summary: ProviderValue::Value(summary.map(str::to_owned)),
                ..Default::default()
            }),
            ..Default::default()
        };
        match effort {
            Some(effort) => with_params(
                params.with_reasoning_effort(ProviderValue::Value(effort.map(str::to_owned))),
            ),
            None => with_params(params),
        }
    };
    let (body, _) = sent("claude-sonnet-4-6", vec![user("hi")], summary(None, None)).await;
    assert_eq!(
        body["thinking"],
        json!({"type": "adaptive", "display": "omitted"}),
        "{body}"
    );
    let (body, _) = sent(
        "claude-sonnet-4-6",
        vec![user("hi")],
        summary(Some("detailed"), Some(Some("high"))),
    )
    .await;
    assert_eq!(
        body["thinking"],
        json!({"type": "adaptive", "display": "summarized"}),
        "{body}"
    );
    let (body, _) = sent(
        "claude-sonnet-4-6",
        vec![user("hi")],
        summary(None, Some(None)),
    )
    .await;
    assert_eq!(body["thinking"], json!({"type": "disabled"}), "{body}");
}

#[tokio::test]
async fn ant_003_core_parameters_go_out_under_their_wire_names() {
    use suprnova_ai::models::params::{CacheParams, ContextManagementParams, TokenThreshold};
    let params = InferenceRequestParams {
        safety_identifier: Some("u1".into()),
        cache: Some(CacheParams {
            retention: ProviderValue::Value("1h".into()),
            ..Default::default()
        }),
        context_management: Some(ContextManagementParams {
            compaction: Some(TokenThreshold(80_000)),
        }),
        ..Default::default()
    }
    .with_temperature(0.4);
    let mut options = with_params(params);
    options.tools = vec![function_tool("weather")];
    options.tool_choice = Some(suprnova_ai::types::tools::ToolChoice::Required);
    let (body, _) = sent("claude-sonnet-4-6", vec![user("hi")], options).await;
    assert_eq!(body["temperature"], 0.4, "{body}");
    assert_eq!(body["metadata"], json!({"user_id": "u1"}), "{body}");
    assert_eq!(
        body["cache_control"],
        json!({"type": "ephemeral", "ttl": "1h"}),
        "{body}"
    );
    assert_eq!(
        body["context_management"],
        json!({"edits": [{"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 80000}}]})
    );
    assert_eq!(body["tool_choice"], json!({"type": "any"}), "{body}");
}

#[tokio::test]
async fn ant_003_settings_anthropic_never_takes_fail_before_a_request() {
    let cases: Vec<Value> = vec![
        json!({"sampling": {"value": {"seed": {"seed": {"value": {"value": 5}}}}}}),
        json!({"sampling": {"value": {"min_p": {"min_p": {"value": 0.1}}}}}),
        json!({"sampling": {"value": {"repetition_penalty": {"frequency_penalty": {"value": 0.1}, "presence_penalty": "default", "repetition_penalty": "default", "consideration_window": "default"}}}}),
        json!({"sampling": {"value": {"repetition_penalty": {"frequency_penalty": "default", "presence_penalty": {"value": 0.1}, "repetition_penalty": "default", "consideration_window": "default"}}}}),
        json!({"sampling": {"value": {"repetition_penalty": {"frequency_penalty": "default", "presence_penalty": "default", "repetition_penalty": {"value": 1.1}, "consideration_window": "default"}}}}),
        json!({"sampling": {"value": {"repetition_penalty": {"frequency_penalty": "default", "presence_penalty": "default", "repetition_penalty": "default", "consideration_window": {"value": 5}}}}}),
        json!({"tool_calling": {"tool_choice": "auto", "max_tool_calls": {"value": 1}}}),
        json!({"output": {"text_verbosity": {"value": "high"}}}),
        json!({"output": {"include": []}}),
        json!({"metadata": {"a": "b"}}),
        json!({"cache": {"key": "k"}}),
        json!({"cache": {"mode": {"value": "auto"}}}),
        json!({"tool_calling": {"tool_choice": {"tools": ["a", "b"], "mode": "auto"}}}),
        json!({"reasoning": {"value": {"effort": {"value": "turbo"}}}}),
    ];
    for case in cases {
        let params: InferenceRequestParams =
            serde_json::from_value(case.clone()).unwrap_or_else(|e| panic!("{case}: {e}"));
        let error = refused("claude-sonnet-4-6", vec![user("hi")], with_params(params)).await;
        assert!(
            matches!(error, suprnova_ai::Error::Validation(_)),
            "{case}: {error:?}"
        );
    }
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

#[tokio::test]
async fn ant_004_settings_typescript_changes_are_changed_and_logged() {
    use suprnova_ai::models::params::{OutputParams, SamplerParams, TopPSamplerParams};
    let temperature = |t: f64| InferenceRequestParams::default().with_temperature(t);
    let top_p_and_temperature = || {
        let mut params = temperature(0.5);
        if let ProviderValue::Value(SamplerParams { top_p, .. }) = &mut params.sampling {
            *top_p = Some(TopPSamplerParams {
                top_p: ProviderValue::Value(Some(0.9)),
            });
        }
        params
    };
    let limit = |max: u64| InferenceRequestParams {
        output: Some(OutputParams {
            max_tokens: Some(max),
            ..Default::default()
        }),
        ..Default::default()
    };
    let breakpoint = |text: &str| {
        let mut part = TextPart::new(text);
        part.provider_metadata = metadata(json!({"cacheControl": {"type": "ephemeral"}}));
        Part::Text(part)
    };
    let five = vec![Message::new(
        MessageRole::User,
        (1..=5).map(|i| breakpoint(&format!("part {i}"))).collect(),
    )];
    let mut required = with_params(InferenceRequestParams::default());
    required.tools = vec![function_tool("weather")];
    required.tool_choice = Some(suprnova_ai::types::tools::ToolChoice::Required);
    let cases: Vec<(&str, Vec<Message>, StreamOptions, &str, &str, Option<Value>)> = vec![
        (
            "claude-opus-5",
            vec![user("hi")],
            with_params(temperature(0.5)),
            "temperature",
            "/temperature",
            None,
        ),
        (
            "claude-sonnet-4-6",
            vec![user("hi")],
            with_params(temperature(1.5)),
            "temperature",
            "/temperature",
            Some(json!(1.0)),
        ),
        (
            "claude-sonnet-4-6",
            vec![user("hi")],
            with_anthropic(temperature(0.5), json!({"thinking": {"type": "adaptive"}})),
            "temperature",
            "/temperature",
            None,
        ),
        (
            "claude-sonnet-4-6",
            vec![user("hi")],
            with_params(top_p_and_temperature()),
            "topP",
            "/top_p",
            None,
        ),
        (
            "claude-sonnet-5-5",
            vec![user("hi")],
            anthropic(json!({"thinking": {"type": "disabled"}})),
            "thinking",
            "/thinking",
            Some(json!({"type": "between_tools"})),
        ),
        (
            "claude-fable-5-1",
            vec![user("hi")],
            anthropic(json!({"thinking": {"type": "disabled"}})),
            "thinking",
            "/thinking",
            None,
        ),
        (
            "claude-opus-5-5",
            vec![user("hi")],
            anthropic(json!({"thinking": {"type": "enabled", "budgetTokens": 2000}})),
            "thinking",
            "/thinking",
            Some(json!({"type": "adaptive"})),
        ),
        (
            "claude-haiku-5-5",
            vec![user("hi")],
            anthropic(json!({"thinking": {"type": "disabled"}, "effort": "max"})),
            "effort",
            "/output_config/effort",
            Some(json!("high")),
        ),
        (
            "claude-sonnet-4-6",
            vec![user("hi")],
            with_params(limit(200_000)),
            "maxOutputTokens",
            "/max_tokens",
            Some(json!(128000)),
        ),
        (
            "claude-opus-5-5",
            vec![user("hi")],
            required,
            "toolChoice",
            "/tool_choice",
            Some(json!({"type": "auto"})),
        ),
        (
            "claude-sonnet-4-6",
            five,
            StreamOptions::default(),
            "cacheControl",
            "/messages/0/content/4/cache_control",
            None,
        ),
    ];
    for (model_id, messages, options, setting, pointer, expected) in cases {
        let (logs, _guard) = capture_logs();
        let (body, _) = sent(model_id, messages, options).await;
        assert_eq!(
            body.pointer(pointer),
            expected.as_ref(),
            "{model_id} {setting}: {body}"
        );
        let logged = logs.take();
        assert!(
            logged.contains(&(setting.to_owned(), model_id.to_owned())),
            "{model_id} {setting}: {logged:?}"
        );
    }
}

#[tokio::test]
async fn ant_004_settings_a_model_takes_are_not_changed() {
    let (logs, _guard) = capture_logs();
    let (body, _) = sent(
        "claude-sonnet-4-6",
        vec![user("hi")],
        with_params(InferenceRequestParams::default().with_temperature(0.5)),
    )
    .await;
    assert_eq!(body["temperature"], 0.5, "{body}");
    let (body, _) = sent(
        "claude-haiku-5-5",
        vec![user("hi")],
        anthropic(json!({"thinking": {"type": "adaptive"}, "effort": "max"})),
    )
    .await;
    assert_eq!(body["output_config"]["effort"], "max", "{body}");
    assert_eq!(logs.take(), vec![]);
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

fn output_schema() -> schemars::Schema {
    json!({"type": "object", "properties": {"a": {"type": "number"}}, "required": ["a"]})
        .try_into()
        .unwrap()
}

#[tokio::test]
async fn ant_005_an_output_schema_goes_as_the_model_takes_it() {
    let with_schema = |options: StreamOptions| StreamOptions {
        output_schema: Some(output_schema()),
        ..options
    };
    let (body, _) = sent(
        "claude-sonnet-4-6",
        vec![user("hi")],
        with_schema(StreamOptions::default()),
    )
    .await;
    assert_eq!(
        body["output_config"]["format"],
        json!({"type": "json_schema", "schema": {"type": "object", "properties": {"a": {"type": "number"}},
               "additionalProperties": false, "required": ["a"]}}),
        "{body}"
    );
    assert!(body.get("tools").is_none(), "{body}");
    // A model without structured output gets the JSON tool.
    let (body, _) = sent(
        "claude-3-haiku-20240307",
        vec![user("hi")],
        with_schema(StreamOptions::default()),
    )
    .await;
    assert!(body.get("output_config").is_none(), "{body}");
    assert_eq!(body["tools"][0]["name"], "json", "{body}");
    assert_eq!(
        body["tools"][0]["description"], "Respond with a JSON object.",
        "{body}"
    );
    assert_eq!(
        body["tool_choice"],
        json!({"type": "any", "disable_parallel_tool_use": true}),
        "{body}"
    );
    // A model that rejects forced tool use takes the output format instead.
    let (logs, _guard) = capture_logs();
    let (body, _) = sent(
        "claude-opus-5-5",
        vec![user("hi")],
        with_schema(anthropic(json!({"structuredOutputMode": "jsonTool"}))),
    )
    .await;
    assert!(body.get("tools").is_none(), "{body}");
    assert!(body["output_config"].get("format").is_some(), "{body}");
    assert!(logs.take().contains(&(
        "structuredOutputMode".to_owned(),
        "claude-opus-5-5".to_owned()
    )));
}

#[tokio::test]
async fn ant_005_the_json_tool_answer_is_the_text() {
    let transport = Recorder::new([message(
        vec![
            json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ignored"}}),
            json!({"type": "content_block_stop", "index": 0}),
            json!({"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_j", "name": "json", "input": {}}}),
            json!({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"a\":1}"}}),
            json!({"type": "content_block_stop", "index": 1}),
        ],
        json!({"output_tokens": 3}),
        "tool_use",
    )]);
    let mut handle = suprnova_ai::stream(
        &model("claude-sonnet-4-6", transport),
        vec![user("hi")],
        StreamOptions {
            output_schema: Some(output_schema()),
            ..anthropic(json!({"structuredOutputMode": "jsonTool"}))
        },
    );
    while let Some(event) = handle.next().await {
        event.unwrap();
    }
    assert_eq!(handle.text(), "{\"a\":1}");
    assert_eq!(handle.finish_reason(), Some("stop"));
    assert!(handle.message().tool_calls().is_empty());
}

#[tokio::test]
async fn ant_005_the_output_schema_is_sanitized_as_typescript_does() {
    let schema = json!({"type": "object", "properties": {
        "n": {"type": "integer", "minimum": 1},
        "s": {"type": "string", "pattern": "^a", "format": "email"},
        "r": {"type": "string", "format": "regex", "description": "A pattern"},
        "u": {"oneOf": [{"type": "string"}, {"type": "null"}]}
    }});
    let (body, _) = sent(
        "claude-sonnet-4-6",
        vec![user("hi")],
        StreamOptions {
            output_schema: Some(schema.try_into().unwrap()),
            ..Default::default()
        },
    )
    .await;
    assert_eq!(
        body["output_config"]["format"]["schema"],
        json!({"type": "object", "additionalProperties": false, "properties": {
            "n": {"type": "integer", "description": "minimum: 1."},
            "s": {"type": "string", "format": "email", "description": "pattern: ^a."},
            "r": {"type": "string", "description": "A pattern\nformat: regex."},
            "u": {"anyOf": [{"type": "string"}, {"type": "null"}]}
        }}),
        "{body}"
    );
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

#[tokio::test]
async fn ant_006_each_tool_version_goes_out_with_its_beta() {
    use suprnova_ai::providers::anthropic::tools::*;
    let computer = |zoom| ComputerUseOptions {
        display_width_px: 1024,
        display_height_px: 768,
        display_number: Some(1),
        enable_zoom: zoom,
    };
    let fetch = WebFetchOptions {
        max_uses: Some(1),
        citations: Some(true.into()),
        max_content_tokens: Some(100),
        use_cache: Some(false),
        response_inclusion: Some(ResponseInclusion::Excluded),
        ..Default::default()
    };
    let search = WebSearchOptions {
        max_uses: Some(2),
        allowed_domains: Some(vec!["a.test".into()]),
        user_location: Some(UserLocation {
            city: Some("SF".into()),
            ..Default::default()
        }),
        response_inclusion: Some(ResponseInclusion::Full),
        ..Default::default()
    };
    let cases: Vec<(Tool, Value, Option<&str>)> = vec![
        (
            advisor_20260301(AdvisorOptions {
                model: "claude-opus-5".into(),
                max_uses: Some(2),
                max_tokens: Some(2048),
                caching: Some(AdvisorCacheTtl::OneHour),
            })
            .unwrap(),
            json!({"type": "advisor_20260301", "name": "advisor", "model": "claude-opus-5", "max_uses": 2, "max_tokens": 2048, "caching": {"type": "ephemeral", "ttl": "1h"}}),
            Some("advisor-tool-2026-03-01"),
        ),
        (
            bash_20241022(),
            json!({"type": "bash_20241022", "name": "bash"}),
            Some("computer-use-2024-10-22"),
        ),
        (
            bash_20250124(),
            json!({"type": "bash_20250124", "name": "bash"}),
            Some("computer-use-2025-01-24"),
        ),
        (
            code_execution_20250522(),
            json!({"type": "code_execution_20250522", "name": "code_execution"}),
            Some("code-execution-2025-05-22"),
        ),
        (
            code_execution_20250825(),
            json!({"type": "code_execution_20250825", "name": "code_execution"}),
            Some("code-execution-2025-08-25"),
        ),
        (
            code_execution_20260120(),
            json!({"type": "code_execution_20260120", "name": "code_execution"}),
            None,
        ),
        (
            computer_20241022(computer(Some(true))),
            json!({"type": "computer_20241022", "name": "computer", "display_width_px": 1024, "display_height_px": 768, "display_number": 1}),
            Some("computer-use-2024-10-22"),
        ),
        (
            computer_20250124(computer(None)),
            json!({"type": "computer_20250124", "name": "computer", "display_width_px": 1024, "display_height_px": 768, "display_number": 1}),
            Some("computer-use-2025-01-24"),
        ),
        (
            computer_20251124(computer(Some(true))),
            json!({"type": "computer_20251124", "name": "computer", "display_width_px": 1024, "display_height_px": 768, "display_number": 1, "enable_zoom": true}),
            Some("computer-use-2025-11-24"),
        ),
        (
            computer_toolset_20260801(
                [(
                    "left_click".to_owned(),
                    ToolsetMember {
                        enabled: Some(true),
                        defer_loading: Some(false),
                    },
                )]
                .into(),
            )
            .unwrap(),
            json!({"type": "computer_toolset_20260801", "configs": {"left_click": {"enabled": true, "defer_loading": false}}}),
            None,
        ),
        (
            memory_20250818(),
            json!({"type": "memory_20250818", "name": "memory"}),
            Some("context-management-2025-06-27"),
        ),
        (
            text_editor_20241022(),
            json!({"type": "text_editor_20241022", "name": "str_replace_editor"}),
            Some("computer-use-2024-10-22"),
        ),
        (
            text_editor_20250124(),
            json!({"type": "text_editor_20250124", "name": "str_replace_editor"}),
            Some("computer-use-2025-01-24"),
        ),
        (
            text_editor_20250429(),
            json!({"type": "text_editor_20250429", "name": "str_replace_based_edit_tool"}),
            Some("computer-use-2025-01-24"),
        ),
        (
            text_editor_20250728(Some(500)),
            json!({"type": "text_editor_20250728", "name": "str_replace_based_edit_tool", "max_characters": 500}),
            None,
        ),
        (
            web_fetch_20250910(fetch.clone()).unwrap(),
            json!({"type": "web_fetch_20250910", "name": "web_fetch", "max_uses": 1, "citations": {"enabled": true}, "max_content_tokens": 100}),
            Some("web-fetch-2025-09-10"),
        ),
        (
            web_fetch_20260209(fetch.clone()).unwrap(),
            json!({"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": 1, "citations": {"enabled": true}, "max_content_tokens": 100}),
            Some("code-execution-web-tools-2026-02-09"),
        ),
        (
            web_fetch_20260318(fetch).unwrap(),
            json!({"type": "web_fetch_20260318", "name": "web_fetch", "max_uses": 1, "citations": {"enabled": true}, "max_content_tokens": 100, "use_cache": false, "response_inclusion": "excluded"}),
            None,
        ),
        (
            web_search_20250305(search.clone()).unwrap(),
            json!({"type": "web_search_20250305", "name": "web_search", "max_uses": 2, "allowed_domains": ["a.test"], "user_location": {"type": "approximate", "city": "SF"}}),
            None,
        ),
        (
            web_search_20260209(search.clone()).unwrap(),
            json!({"type": "web_search_20260209", "name": "web_search", "max_uses": 2, "allowed_domains": ["a.test"], "user_location": {"type": "approximate", "city": "SF"}}),
            Some("code-execution-web-tools-2026-02-09"),
        ),
        (
            web_search_20260318(search).unwrap(),
            json!({"type": "web_search_20260318", "name": "web_search", "max_uses": 2, "allowed_domains": ["a.test"], "user_location": {"type": "approximate", "city": "SF"}, "response_inclusion": "full"}),
            None,
        ),
        (
            tool_search_regex_20251119(),
            json!({"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"}),
            None,
        ),
        (
            tool_search_bm25_20251119(),
            json!({"type": "tool_search_tool_bm25_20251119", "name": "tool_search_tool_bm25"}),
            None,
        ),
    ];
    assert_eq!(cases.len(), 23);
    for (tool, expected, beta) in cases {
        let (body, betas) = sent(
            "claude-sonnet-4-6",
            vec![user("hi")],
            StreamOptions {
                tools: vec![tool],
                ..Default::default()
            },
        )
        .await;
        assert_eq!(body["tools"], json!([expected]), "{body}");
        match beta {
            Some(beta) => assert_eq!(betas, [beta], "{expected}"),
            None => assert!(betas.is_empty(), "{expected}: {betas:?}"),
        }
    }
}

#[tokio::test]
async fn ant_006_function_tools_take_strict_streaming_and_their_arguments() {
    let strict = || function_tool("weather").with_strict(true);
    let options = |tools: Vec<Tool>, extra: Value| StreamOptions {
        tools,
        ..anthropic(extra)
    };
    let (body, betas) = sent(
        "claude-sonnet-4-6",
        vec![user("hi")],
        options(vec![strict()], json!({})),
    )
    .await;
    assert_eq!(
        body["tools"][0],
        json!({"name": "weather", "description": "The weather tool", "input_schema": {"type": "object", "properties": {}},
               "eager_input_streaming": true, "strict": true}),
        "{body}"
    );
    assert_eq!(betas, ["structured-outputs-2025-11-13"]);
    let (body, _) = sent(
        "claude-3-haiku-20240307",
        vec![user("hi")],
        options(vec![strict()], json!({})),
    )
    .await;
    assert!(body["tools"][0].get("strict").is_none(), "{body}");
    let (body, _) = sent(
        "claude-sonnet-4-6",
        vec![user("hi")],
        options(vec![strict()], json!({"toolStreaming": false})),
    )
    .await;
    assert!(
        body["tools"][0].get("eager_input_streaming").is_none(),
        "{body}"
    );
    let mut tool = function_tool("lookup");
    let config = tool.tool_config.get_or_insert_with(Default::default);
    config
        .args
        .insert("input_examples".into(), json!([{"q": "a"}]));
    config.args.insert("defer_loading".into(), json!(true));
    config
        .args
        .insert("allowed_callers".into(), json!(["code_execution_20260120"]));
    config
        .args
        .insert("eager_input_streaming".into(), json!(false));
    config
        .args
        .insert("cache_control".into(), json!({"type": "ephemeral"}));
    let (body, betas) = sent(
        "claude-sonnet-4-6",
        vec![user("hi")],
        options(vec![tool], json!({})),
    )
    .await;
    assert_eq!(
        body["tools"][0],
        json!({"name": "lookup", "description": "The lookup tool", "input_schema": {"type": "object", "properties": {}},
               "cache_control": {"type": "ephemeral"}, "defer_loading": true,
               "allowed_callers": ["code_execution_20260120"], "input_examples": [{"q": "a"}]}),
        "{body}"
    );
    assert!(
        betas.iter().any(|b| b == "advanced-tool-use-2025-11-20"),
        "{betas:?}"
    );
}

#[tokio::test]
async fn ant_006_a_renamed_provider_tool_comes_back_under_its_name() {
    use suprnova_ai::providers::anthropic::tools::web_search_20250305;
    let mut search = web_search_20250305(Default::default()).unwrap();
    search.name = "search".into();
    let transport = Recorder::new([message(
        vec![
            json!({"type": "content_block_start", "index": 0, "content_block": {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {}}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{\"query\":\"rust\"}"}}),
            json!({"type": "content_block_stop", "index": 0}),
            json!({"type": "content_block_start", "index": 1, "content_block": {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": [{"type": "web_search_result", "url": "https://rust.test", "title": "Rust", "encrypted_content": "e1", "page_age": "1 day"}]}}),
            json!({"type": "content_block_stop", "index": 1}),
        ],
        json!({"output_tokens": 1}),
        "end_turn",
    )]);
    let options = StreamOptions {
        tools: vec![search.clone()],
        ..Default::default()
    };
    let mut handle = suprnova_ai::stream(
        &model("claude-sonnet-4-6", transport.clone()),
        vec![user("hi")],
        options.clone(),
    );
    while let Some(event) = handle.next().await {
        event.unwrap();
    }
    assert_eq!(transport.body(0)["tools"][0]["name"], "web_search");
    let message = handle.message().clone();
    assert_eq!(message.builtin_tool_calls()[0].tool_name, "search");
    assert_eq!(message.builtin_tool_returns()[0].tool_name, "search");
    // In history the call goes back under Anthropic's name.
    let (body, _) = sent(
        "claude-sonnet-4-6",
        vec![user("hi"), message, user("more")],
        options,
    )
    .await;
    assert_eq!(
        body["messages"][1]["content"][0]["name"], "web_search",
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

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct BashInput {
    #[allow(dead_code)]
    command: String,
}

#[tokio::test]
async fn ant_007_an_agent_runs_a_provider_declared_tool() {
    use suprnova_ai::agents::{Agent, AgentTool};
    let bash = AgentTool::new::<BashInput, Value, _, _>("bash", "Run a command", |_, _| async {
        Ok(json!("a.txt"))
    })
    .unwrap()
    .with_provider_declaration(suprnova_ai::providers::anthropic::tools::bash_20250124())
    .unwrap();
    let transport = Recorder::new([
        message(
            vec![
                json!({"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "toolu_b", "name": "bash", "input": {}}}),
                json!({"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{\"command\": \"ls\"}"}}),
                json!({"type": "content_block_stop", "index": 0}),
            ],
            json!({"output_tokens": 1}),
            "tool_use",
        ),
        text_reply(),
    ]);
    let mut run = Agent::new(vec![bash]).run(
        model("claude-sonnet-4-6", transport.clone()),
        vec![user("list files")],
    );
    while let Some(event) = run.next().await {
        event.unwrap();
    }
    assert_eq!(
        transport.body(0)["tools"],
        json!([{"type": "bash_20250124", "name": "bash"}])
    );
    let body = transport.body(1);
    assert_eq!(
        body["messages"][2]["content"][0],
        json!({"type": "tool_result", "tool_use_id": "toolu_b", "content": "a.txt"}),
        "{body}"
    );
}

#[tokio::test]
async fn ant_007_a_toolset_call_goes_back_as_its_member() {
    let computer =
        suprnova_ai::providers::anthropic::tools::computer_toolset_20260801(Default::default())
            .unwrap();
    let mut call = ToolCallPart::new(
        "toolu_1",
        "computer",
        r#"{"action": "left_click", "coordinate": [1, 2]}"#,
    );
    call.provider_metadata = metadata(json!({"toolsetName": "computer"}));
    let (body, _) = sent(
        "claude-sonnet-4-6",
        vec![
            user("click"),
            assistant(vec![Part::ToolCall(call)]),
            Message::new(
                MessageRole::Tool,
                vec![Part::ToolResult(
                    suprnova_ai::types::messages::ToolResultPart::new(
                        "toolu_1",
                        "computer",
                        json!("done"),
                    ),
                )],
            ),
        ],
        StreamOptions {
            tools: vec![computer],
            ..Default::default()
        },
    )
    .await;
    assert_eq!(
        body["messages"][1]["content"][0],
        json!({"type": "tool_use", "id": "toolu_1", "name": "left_click", "toolset_name": "computer",
               "input": {"action": "left_click", "coordinate": [1, 2]}}),
        "{body}"
    );
    assert_eq!(
        body["messages"][2]["content"][0],
        json!({"type": "tool_result", "tool_use_id": "toolu_1", "toolset_name": "computer", "content": "done"}),
        "{body}"
    );
}

#[tokio::test]
async fn ant_007_a_renamed_application_tool_comes_back_under_its_name() {
    let mut shell = suprnova_ai::providers::anthropic::tools::bash_20250124();
    shell.name = "shell".into();
    let transport = Recorder::new([message(
        vec![
            json!({"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "toolu_s", "name": "bash", "input": {}}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{\"command\": \"ls\"}"}}),
            json!({"type": "content_block_stop", "index": 0}),
        ],
        json!({"output_tokens": 1}),
        "tool_use",
    )]);
    let options = StreamOptions {
        tools: vec![shell],
        ..Default::default()
    };
    let mut handle = suprnova_ai::stream(
        &model("claude-sonnet-4-6", transport),
        vec![user("ls")],
        options.clone(),
    );
    while let Some(event) = handle.next().await {
        event.unwrap();
    }
    let message = handle.message().clone();
    assert_eq!(message.tool_calls()[0].tool_name, "shell");
    let result = Message::new(
        MessageRole::Tool,
        vec![Part::ToolResult(
            suprnova_ai::types::messages::ToolResultPart::new("toolu_s", "shell", json!("a.txt")),
        )],
    );
    let (body, _) = sent(
        "claude-sonnet-4-6",
        vec![user("ls"), message, result],
        options,
    )
    .await;
    assert_eq!(body["messages"][1]["content"][0]["name"], "bash", "{body}");
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

/// The built-in blocks of one response, each with its start, its input
/// and its stop, as Anthropic streams them.
fn builtin_blocks() -> Vec<Value> {
    vec![
        json!({"type": "server_tool_use", "id": "srv_ws", "name": "web_search", "input": {"query": "rust"}}),
        json!({"type": "web_search_tool_result", "tool_use_id": "srv_ws", "content": [
            {"type": "web_search_result", "url": "https://rust.test", "title": "Rust", "encrypted_content": "e1", "page_age": "1 day"}]}),
        json!({"type": "server_tool_use", "id": "srv_wf", "name": "web_fetch", "input": {"url": "https://doc.test"}}),
        json!({"type": "web_fetch_tool_result", "tool_use_id": "srv_wf", "content": {"type": "web_fetch_result",
            "url": "https://doc.test", "retrieved_at": "2026-10-24T00:00:00Z", "content": {"type": "document",
            "title": "Doc", "source": {"type": "text", "media_type": "text/plain", "data": "hello"}}}}),
        json!({"type": "server_tool_use", "id": "srv_bash", "name": "bash_code_execution", "input": {"command": "ls"}}),
        json!({"type": "bash_code_execution_tool_result", "tool_use_id": "srv_bash", "content": {
            "type": "bash_code_execution_result", "stdout": "a.txt", "stderr": "", "return_code": 0, "content": []}}),
        json!({"type": "server_tool_use", "id": "srv_edit", "name": "text_editor_code_execution", "input": {"command": "view", "path": "a.txt"}}),
        json!({"type": "text_editor_code_execution_tool_result", "tool_use_id": "srv_edit", "content": {
            "type": "text_editor_code_execution_view_result", "content": "x", "file_type": "text", "num_lines": 1,
            "start_line": 1, "total_lines": 1}}),
        json!({"type": "server_tool_use", "id": "srv_code", "name": "code_execution", "input": {"code": "print(1)"}}),
        json!({"type": "code_execution_tool_result", "tool_use_id": "srv_code", "content": {
            "type": "code_execution_result", "stdout": "1", "stderr": "", "return_code": 0, "content": []}}),
        json!({"type": "server_tool_use", "id": "srv_ts", "name": "tool_search_tool_regex", "input": {"pattern": "weather"}}),
        json!({"type": "tool_search_tool_result", "tool_use_id": "srv_ts", "content": {
            "type": "tool_search_tool_search_result", "tool_references": [{"type": "tool_reference", "tool_name": "weather"}]}}),
        json!({"type": "server_tool_use", "id": "srv_adv", "name": "advisor", "input": {}}),
        json!({"type": "advisor_tool_result", "tool_use_id": "srv_adv", "content": {"type": "advisor_result", "text": "Try X", "stop_reason": "end_turn"}}),
        json!({"type": "mcp_tool_use", "id": "mcp_1", "name": "lookup", "server_name": "docs", "input": {"q": "a"}}),
        json!({"type": "mcp_tool_result", "tool_use_id": "mcp_1", "is_error": false, "content": [{"type": "text", "text": "found"}]}),
    ]
}
/// A stream of these complete blocks: a tool use's input as one delta.
fn blocks_reply(blocks: &[Value], reason: &str) -> (u16, Vec<u8>) {
    let mut events = vec![];
    for (index, block) in blocks.iter().enumerate() {
        let mut opened = block.clone();
        let input = block.get("input").cloned();
        if matches!(block["type"].as_str(), Some("server_tool_use" | "tool_use")) {
            opened["input"] = json!({});
        }
        events
            .push(json!({"type": "content_block_start", "index": index, "content_block": opened}));
        if let (Some(input), Some("server_tool_use" | "tool_use")) = (input, block["type"].as_str())
        {
            events.push(json!({"type": "content_block_delta", "index": index,
                "delta": {"type": "input_json_delta", "partial_json": input.to_string()}}));
        }
        events.push(json!({"type": "content_block_stop", "index": index}));
    }
    message(events, json!({"output_tokens": 1}), reason)
}
async fn streamed(
    model_id: &str,
    reply: (u16, Vec<u8>),
    options: StreamOptions,
) -> suprnova_ai::models::StreamHandle {
    let mut handle = suprnova_ai::stream(
        &model(model_id, Recorder::new([reply])),
        vec![user("hi")],
        options,
    );
    while let Some(event) = handle.next().await {
        event.unwrap();
    }
    handle
}

#[tokio::test]
async fn ant_008_built_in_tools_arrive_in_typescript_shapes() {
    let handle = streamed(
        "claude-sonnet-4-6",
        blocks_reply(&builtin_blocks(), "end_turn"),
        StreamOptions::default(),
    )
    .await;
    let message = handle.message();
    let calls: Vec<(String, Value)> = message
        .builtin_tool_calls()
        .iter()
        .map(|c| {
            (
                c.tool_name.clone(),
                serde_json::from_str(&c.tool_args).unwrap(),
            )
        })
        .collect();
    assert_eq!(
        calls,
        [
            ("web_search".to_owned(), json!({"query": "rust"})),
            ("web_fetch".to_owned(), json!({"url": "https://doc.test"})),
            (
                "code_execution".to_owned(),
                json!({"type": "bash_code_execution", "command": "ls"})
            ),
            (
                "code_execution".to_owned(),
                json!({"type": "text_editor_code_execution", "command": "view", "path": "a.txt"})
            ),
            (
                "code_execution".to_owned(),
                json!({"type": "programmatic-tool-call", "code": "print(1)"})
            ),
            (
                "tool_search_tool_regex".to_owned(),
                json!({"pattern": "weather"})
            ),
            ("advisor".to_owned(), json!({})),
            ("lookup".to_owned(), json!({"q": "a"})),
        ]
    );
    let results: Vec<(String, Value, bool)> = message
        .builtin_tool_returns()
        .iter()
        .map(|r| (r.tool_name.clone(), r.result.clone(), r.is_error))
        .collect();
    assert_eq!(
        results[0],
        (
            "web_search".to_owned(),
            json!([{"url": "https://rust.test", "title": "Rust", "pageAge": "1 day", "encryptedContent": "e1", "type": "web_search_result"}]),
            false
        )
    );
    assert_eq!(
        results[1],
        (
            "web_fetch".to_owned(),
            json!({"type": "web_fetch_result", "url": "https://doc.test", "retrievedAt": "2026-10-24T00:00:00Z",
        "content": {"type": "document", "title": "Doc", "source": {"type": "text", "mediaType": "text/plain", "data": "hello"}}}),
            false
        )
    );
    assert_eq!(results[2].1["type"], "bash_code_execution_result");
    assert_eq!(
        results[4].1,
        json!({"type": "code_execution_result", "stdout": "1", "stderr": "", "return_code": 0, "content": []})
    );
    assert_eq!(
        results[5],
        (
            "tool_search_tool_regex".to_owned(),
            json!([{"type": "tool_reference", "toolName": "weather"}]),
            false
        )
    );
    assert_eq!(
        results[6],
        (
            "advisor".to_owned(),
            json!({"type": "advisor_result", "text": "Try X", "stopReason": "end_turn"}),
            false
        )
    );
    assert_eq!(
        results[7],
        (
            "lookup".to_owned(),
            json!([{"type": "text", "text": "found"}]),
            false
        )
    );
    let mcp = message.builtin_tool_calls()[7]
        .provider_metadata
        .clone()
        .unwrap();
    assert_eq!(
        mcp["anthropic"],
        json!({"type": "mcp-tool-use", "serverName": "docs"})
    );
}

#[tokio::test]
async fn ant_008_errors_callers_and_deferred_calls_are_kept() {
    let reply = blocks_reply(
        &[
            json!({"type": "server_tool_use", "id": "srv_ws", "name": "web_search", "input": {"query": "x"}}),
            json!({"type": "web_search_tool_result", "tool_use_id": "srv_ws", "content": {"type": "web_search_tool_result_error", "error_code": "max_uses_exceeded"}}),
            json!({"type": "tool_use", "id": "toolu_p", "name": "weather", "caller": {"type": "code_execution_20260120", "tool_id": "srv_1"}, "input": {"city": "SF"}}),
        ],
        "tool_use",
    );
    let handle = streamed("claude-sonnet-4-6", reply, StreamOptions::default()).await;
    let error = handle.message().builtin_tool_returns()[0].clone();
    assert!(error.is_error);
    assert_eq!(
        error.result,
        json!({"type": "web_search_tool_result_error", "errorCode": "max_uses_exceeded"})
    );
    let call = &handle.message().tool_calls()[0];
    assert_eq!(
        call.provider_metadata.as_ref().unwrap()["anthropic"]["caller"],
        json!({"type": "code_execution_20260120", "toolId": "srv_1"})
    );
    // A deferred call of a program in the message start.
    let mut events = vec![sse_event(
        json!({"type": "message_start", "message": {"id": "msg_1", "model": "claude-sonnet-4-6",
        "content": [{"type": "tool_use", "id": "toolu_d", "name": "weather", "input": {"city": "LA"}, "caller": {"type": "code_execution_20260120", "tool_id": "srv_1"}}],
        "stop_reason": null, "usage": {"input_tokens": 1, "output_tokens": 0}}}),
    )];
    events.push(sse_event(json!({"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 1}})));
    events.push(sse_event(json!({"type": "message_stop"})));
    let handle = streamed(
        "claude-sonnet-4-6",
        (200, events.concat().into_bytes()),
        StreamOptions::default(),
    )
    .await;
    let call = &handle.message().tool_calls()[0];
    assert_eq!(
        (call.tool_call_id.as_str(), call.tool_name.as_str()),
        ("toolu_d", "weather")
    );
    assert_eq!(
        serde_json::from_str::<Value>(&call.tool_args).unwrap(),
        json!({"city": "LA"})
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

#[tokio::test]
async fn ant_009_built_in_history_goes_back_as_the_stream_received_it() {
    let handle = streamed(
        "claude-sonnet-4-6",
        blocks_reply(&builtin_blocks(), "end_turn"),
        StreamOptions::default(),
    )
    .await;
    let (body, _) = sent(
        "claude-sonnet-4-6",
        vec![user("hi"), handle.message().clone(), user("more")],
        StreamOptions::default(),
    )
    .await;
    assert_eq!(
        body["messages"][1]["content"],
        json!(builtin_blocks()),
        "{body}"
    );
}

#[tokio::test]
async fn ant_009_an_error_without_a_code_goes_back_unavailable() {
    use suprnova_ai::types::messages::{BuiltinToolCallPart, BuiltinToolReturnPart};
    let call = BuiltinToolCallPart {
        id: "c".into(),
        tool_call_id: "srv_1".into(),
        tool_name: "web_search".into(),
        tool_args: r#"{"query": "x"}"#.into(),
        provider_metadata: None,
    };
    let result = BuiltinToolReturnPart {
        id: "r".into(),
        tool_call_id: "srv_1".into(),
        tool_name: "web_search".into(),
        result: json!({"type": "web_search_tool_result_error"}),
        is_error: true,
        provider_metadata: None,
    };
    let (body, _) = sent(
        "claude-sonnet-4-6",
        vec![
            user("hi"),
            assistant(vec![
                Part::BuiltinToolCall(call),
                Part::BuiltinToolReturn(result),
            ]),
            user("more"),
        ],
        StreamOptions::default(),
    )
    .await;
    assert_eq!(
        body["messages"][1]["content"][1],
        json!({"type": "web_search_tool_result", "tool_use_id": "srv_1",
               "content": {"type": "web_search_tool_result_error", "error_code": "unavailable"}}),
        "{body}"
    );
}

#[tokio::test]
async fn ant_009_a_caller_goes_back_only_while_its_code_execution_call_is_there() {
    use suprnova_ai::types::messages::BuiltinToolCallPart;
    let program = |id: &str| {
        let mut call = ToolCallPart::new(format!("toolu_{id}"), "weather", "{}");
        call.provider_metadata =
            metadata(json!({"caller": {"type": "code_execution_20260120", "toolId": id}}));
        Part::ToolCall(call)
    };
    let code = Part::BuiltinToolCall(BuiltinToolCallPart {
        id: "c".into(),
        tool_call_id: "srv_kept".into(),
        tool_name: "code_execution".into(),
        tool_args: r#"{"type": "programmatic-tool-call", "code": "x"}"#.into(),
        provider_metadata: None,
    });
    let result = |id: &str| {
        Message::new(
            MessageRole::Tool,
            vec![Part::ToolResult(
                suprnova_ai::types::messages::ToolResultPart::new(
                    format!("toolu_{id}"),
                    "weather",
                    json!("ok"),
                ),
            )],
        )
    };
    let (logs, _guard) = capture_logs();
    let (body, _) = sent(
        "claude-sonnet-4-6",
        vec![
            user("hi"),
            assistant(vec![code, program("srv_kept"), program("srv_gone")]),
            result("srv_kept"),
            result("srv_gone"),
            user("more"),
        ],
        StreamOptions::default(),
    )
    .await;
    let content = &body["messages"][1]["content"];
    assert_eq!(content[0]["name"], "code_execution", "{body}");
    assert_eq!(content[0]["input"], json!({"code": "x"}), "{body}");
    assert_eq!(
        content[1]["caller"],
        json!({"type": "code_execution_20260120", "tool_id": "srv_kept"}),
        "{body}"
    );
    assert!(content[2].get("caller").is_none(), "{body}");
    assert!(
        logs.take()
            .contains(&("caller".to_owned(), "claude-sonnet-4-6".to_owned()))
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

fn file(data: FileData, media_type: &str, md: Option<Value>) -> FilePart {
    FilePart {
        id: "f".into(),
        data,
        media_type: media_type.into(),
        filename: None,
        provider_metadata: md.and_then(metadata),
    }
}
fn user_with(part: Part) -> Message {
    Message::new(MessageRole::User, vec![part])
}

#[tokio::test]
async fn ant_010_user_files_go_as_typescript_sends_them() {
    let url = |u: &str| FileData::Text(u.into());
    let cases: Vec<(FilePart, Value, Option<&str>)> = vec![
        (
            file(url("https://a.test/cat.png"), "image/png", None),
            json!({"type": "image", "source": {"type": "url", "url": "https://a.test/cat.png"}}),
            None,
        ),
        (
            file(FileData::Bytes(b"\x89PNG".to_vec()), "image/png", None),
            json!({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw=="}}),
            None,
        ),
        (
            file(url("https://a.test/r.pdf"), "application/pdf", None),
            json!({"type": "document", "source": {"type": "url", "url": "https://a.test/r.pdf"}}),
            Some("pdfs-2024-09-25"),
        ),
        (
            file(FileData::Bytes(b"%PDF".to_vec()), "application/pdf", None),
            json!({"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBERg=="}}),
            Some("pdfs-2024-09-25"),
        ),
        (
            file(FileData::Bytes(b"plain".to_vec()), "text/plain", None),
            json!({"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "plain"}}),
            None,
        ),
        (
            file(url("file_abc"), "application/pdf", None),
            json!({"type": "document", "source": {"type": "file", "file_id": "file_abc"}}),
            Some("files-api-2025-04-14"),
        ),
        (
            file(url("file_img"), "image/png", None),
            json!({"type": "image", "source": {"type": "file", "file_id": "file_img"}}),
            Some("files-api-2025-04-14"),
        ),
        (
            file(
                url("file_abc"),
                "text/csv",
                Some(json!({"containerUpload": true})),
            ),
            json!({"type": "container_upload", "file_id": "file_abc"}),
            Some("files-api-2025-04-14"),
        ),
    ];
    for (part, expected, beta) in cases {
        let (body, betas) = sent(
            "claude-sonnet-4-6",
            vec![user_with(Part::File(part))],
            StreamOptions::default(),
        )
        .await;
        assert_eq!(body["messages"][0]["content"][0], expected, "{body}");
        if let Some(beta) = beta {
            assert!(betas.iter().any(|b| b == beta), "{expected}: {betas:?}");
        }
    }
}

#[tokio::test]
async fn ant_010_a_document_takes_its_title_context_and_citations() {
    let mut pdf = file(
        FileData::Bytes(b"%PDF".to_vec()),
        "application/pdf",
        Some(json!({"citations": {"enabled": true}, "title": "Report", "context": "Q3 numbers"})),
    );
    pdf.filename = Some("r.pdf".into());
    let (body, _) = sent(
        "claude-sonnet-4-6",
        vec![user_with(Part::File(pdf))],
        StreamOptions::default(),
    )
    .await;
    let document = &body["messages"][0]["content"][0];
    assert_eq!(
        (
            document["title"].clone(),
            document["context"].clone(),
            document["citations"].clone()
        ),
        (
            json!("Report"),
            json!("Q3 numbers"),
            json!({"enabled": true})
        ),
        "{body}"
    );
    let mut named = file(FileData::Bytes(b"plain".to_vec()), "text/plain", None);
    named.filename = Some("notes.txt".into());
    let (body, _) = sent(
        "claude-sonnet-4-6",
        vec![user_with(Part::File(named))],
        StreamOptions::default(),
    )
    .await;
    assert_eq!(
        body["messages"][0]["content"][0]["title"], "notes.txt",
        "{body}"
    );
}

/// A download transport that answers every URL with this text.
struct Downloads(&'static str);
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
        let text = self.0;
        Box::pin(async move {
            Ok(HttpResponse {
                status: 200,
                headers: vec![("content-type".into(), "text/plain".into())],
                body: stream::iter([Ok(Bytes::from(text))]).boxed(),
            })
        })
    }
}

#[tokio::test]
async fn ant_010_a_text_url_is_downloaded_first() {
    let text = || {
        user_with(Part::File(file(
            FileData::Text("https://a.test/notes.txt".into()),
            "text/plain",
            None,
        )))
    };
    let transport = Recorder::new([text_reply()]);
    let provider = AnthropicCompatibleProvider::new(config(), transport.clone())
        .with_download_transport(Arc::new(Downloads("fetched text")));
    run(
        &Model::new("claude-sonnet-4-6", Arc::new(provider)),
        vec![text()],
        StreamOptions::default(),
    )
    .await;
    assert_eq!(
        transport.body(0)["messages"][0]["content"][0]["source"],
        json!({"type": "text", "media_type": "text/plain", "data": "fetched text"})
    );
    let error = refused("claude-sonnet-4-6", vec![text()], StreamOptions::default()).await;
    assert!(
        matches!(error, suprnova_ai::Error::Validation(_)),
        "{error:?}"
    );
    let error = refused(
        "claude-sonnet-4-6",
        vec![user_with(Part::File(file(
            FileData::Bytes(vec![1]),
            "audio/wav",
            None,
        )))],
        StreamOptions::default(),
    )
    .await;
    assert!(
        matches!(error, suprnova_ai::Error::Validation(_)),
        "{error:?}"
    );
}

#[tokio::test]
async fn ant_010_a_tool_result_holds_its_content_parts() {
    use suprnova_ai::types::messages::{ContentOutput, ContentPart, ToolResultPart};
    let content = ContentOutput::new(vec![
        ContentPart::Text(TextPart::new("Loaded")),
        ContentPart::File(file(
            FileData::Bytes(b"\x89PNG".to_vec()),
            "image/png",
            None,
        )),
        ContentPart::File(file(
            FileData::Text("https://a.test/r.pdf".into()),
            "application/pdf",
            None,
        )),
    ]);
    let (body, _) = sent(
        "claude-sonnet-4-6",
        vec![
            user("read"),
            assistant(vec![Part::ToolCall(ToolCallPart::new(
                "toolu_1", "read", "{}",
            ))]),
            Message::new(
                MessageRole::Tool,
                vec![Part::ToolResult(ToolResultPart::new(
                    "toolu_1", "read", content,
                ))],
            ),
        ],
        StreamOptions::default(),
    )
    .await;
    assert_eq!(
        body["messages"][2]["content"][0],
        json!({"type": "tool_result", "tool_use_id": "toolu_1", "content": [
            {"type": "text", "text": "Loaded"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw=="}},
            {"type": "document", "source": {"type": "url", "url": "https://a.test/r.pdf"}}]}),
        "{body}"
    );
    let error_result = ToolResultPart::error("toolu_1", "read", json!("boom"));
    let (body, _) = sent(
        "claude-sonnet-4-6",
        vec![
            user("read"),
            assistant(vec![Part::ToolCall(ToolCallPart::new(
                "toolu_1", "read", "{}",
            ))]),
            Message::new(MessageRole::Tool, vec![Part::ToolResult(error_result)]),
        ],
        StreamOptions::default(),
    )
    .await;
    assert_eq!(
        body["messages"][2]["content"][0]["is_error"], true,
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

fn system_with(text: &str, options: Value) -> Message {
    let mut message = system(text);
    message.provider_metadata = metadata(options);
    message
}

#[tokio::test]
async fn ant_011_system_message_options_go_out_with_their_betas() {
    let (body, betas) = sent(
        "claude-sonnet-4-6",
        vec![
            system("first"),
            user("a"),
            assistant(vec![Part::Text(TextPart::new("ok"))]),
            system_with("brief now", json!({"clearAt": "next_user_message", "effort": "low"})),
            system_with("", json!({"toolChanges": [{"type": "tool_addition", "toolName": "weather"}, {"type": "tool_removal", "toolName": "search"}]})),
            assistant(vec![Part::Text(TextPart::new("next"))]),
        ],
        StreamOptions::default(),
    )
    .await;
    assert_eq!(
        body["messages"][2],
        json!({"role": "system", "content": [{"type": "text", "text": "brief now"}],
               "clear_at": "next_user_message", "output_config": {"effort": "low"}}),
        "{body}"
    );
    assert_eq!(
        body["messages"][3],
        json!({"role": "system", "content": [
            {"type": "tool_addition", "tool": {"type": "tool_reference", "name": "weather"}},
            {"type": "tool_removal", "tool": {"type": "tool_reference", "name": "search"}}]}),
        "{body}"
    );
    for beta in [
        "mid-conversation-system-2026-04-07",
        "mid-conversation-system-clear-at-2026-08-21",
        "mid-conversation-output-config-2026-07-01",
        "mid-conversation-tool-changes-2026-07-01",
    ] {
        assert!(betas.iter().any(|b| b == beta), "{beta}: {betas:?}");
    }
}

#[tokio::test]
async fn ant_011_the_first_system_message_takes_only_an_effort_alone() {
    let (body, betas) = sent(
        "claude-sonnet-4-6",
        vec![system_with("", json!({"effort": "low"})), user("hi")],
        StreamOptions::default(),
    )
    .await;
    assert_eq!(
        body["messages"][0],
        json!({"role": "system", "content": [], "output_config": {"effort": "low"}}),
        "{body}"
    );
    assert!(body.get("system").is_none(), "{body}");
    assert!(
        betas
            .iter()
            .any(|b| b == "mid-conversation-output-config-2026-07-01")
    );
    let (logs, _guard) = capture_logs();
    let (body, _) = sent(
        "claude-sonnet-4-6",
        vec![
            system_with(
                "rules",
                json!({"toolChanges": [{"type": "tool_addition", "toolName": "weather"}]}),
            ),
            user("hi"),
        ],
        StreamOptions::default(),
    )
    .await;
    assert_eq!(
        body["system"],
        json!([{"type": "text", "text": "rules"}]),
        "{body}"
    );
    assert!(
        logs.take()
            .contains(&("toolChanges".to_owned(), "claude-sonnet-4-6".to_owned()))
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

fn text_with(text: &str, md: Value) -> Part {
    let mut part = TextPart::new(text);
    part.provider_metadata = metadata(md);
    Part::Text(part)
}
fn reasoning_with(text: &str, md: Value) -> Part {
    let mut part = suprnova_ai::types::messages::ReasoningPart::new(text);
    part.provider_metadata = metadata(md);
    Part::Reasoning(part)
}

#[tokio::test]
async fn ant_012_assistant_parts_go_back_in_their_blocks() {
    let citation = json!({"type": "char_location", "cited_text": "x", "document_index": 0, "start_char_index": 0, "end_char_index": 1});
    let (body, betas) = sent(
        "claude-sonnet-4-6",
        vec![
            user("hi"),
            assistant(vec![
                reasoning_with("thought", json!({"signature": "sig"})),
                reasoning_with("", json!({"redactedData": "opaque"})),
                text_with(
                    "",
                    json!({"type": "fallback", "from": {"model": "a"}, "to": {"model": "b"}}),
                ),
                text_with("cited", json!({"citations": [citation]})),
                text_with("summary", json!({"type": "compaction", "signature": "cs"})),
                text_with("", json!({"type": "container_upload", "fileId": "file_1"})),
            ]),
            assistant(vec![Part::Text(TextPart::new("second  "))]),
        ],
        StreamOptions::default(),
    )
    .await;
    assert_eq!(body["messages"].as_array().unwrap().len(), 2, "{body}");
    assert_eq!(
        body["messages"][1]["content"],
        json!([
            {"type": "thinking", "thinking": "thought", "signature": "sig"},
            {"type": "redacted_thinking", "data": "opaque"},
            {"type": "fallback", "from": {"model": "a"}, "to": {"model": "b"}},
            {"type": "text", "text": "cited", "citations": [citation]},
            {"type": "compaction", "content": "summary", "signature": "cs"},
            {"type": "text", "text": "second"}
        ]),
        "{body}"
    );
    for beta in ["server-side-fallback-2026-06-01", "compact-2026-09-04"] {
        assert!(betas.iter().any(|b| b == beta), "{beta}: {betas:?}");
    }
}

#[tokio::test]
async fn ant_012_reasoning_without_its_data_or_when_turned_off_stays_out() {
    let (logs, _guard) = capture_logs();
    let turns = || {
        vec![
            user("hi"),
            assistant(vec![
                reasoning_with("thought", json!({"signature": "sig"})),
                Part::Reasoning(suprnova_ai::types::messages::ReasoningPart::new("unsigned")),
                Part::Text(TextPart::new("answer")),
            ]),
            user("more"),
        ]
    };
    let (body, _) = sent("claude-sonnet-4-6", turns(), StreamOptions::default()).await;
    assert_eq!(
        body["messages"][1]["content"].as_array().unwrap().len(),
        2,
        "{body}"
    );
    let (body, _) = sent(
        "claude-sonnet-4-6",
        turns(),
        anthropic(json!({"sendReasoning": false})),
    )
    .await;
    assert_eq!(
        body["messages"][1]["content"],
        json!([{"type": "text", "text": "answer"}]),
        "{body}"
    );
    let logged = logs.take();
    assert!(
        logged.contains(&("reasoning".to_owned(), "claude-sonnet-4-6".to_owned())),
        "{logged:?}"
    );
    assert!(
        logged.contains(&("sendReasoning".to_owned(), "claude-sonnet-4-6".to_owned())),
        "{logged:?}"
    );
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

fn final_metadata(handle: &suprnova_ai::models::StreamHandle) -> Value {
    handle.message().provider_metadata.as_ref().unwrap()["anthropic"].clone()
}

#[tokio::test]
async fn ant_013_citations_and_sources_follow_typescript() {
    let char_citation = json!({"type": "char_location", "cited_text": "a", "document_index": 0,
        "document_title": null, "start_char_index": 0, "end_char_index": 1});
    let web_citation = json!({"type": "web_search_result_location", "cited_text": "b", "url": "https://web.test",
        "title": "Web", "encrypted_index": "x"});
    let reply = message(
        vec![
            json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "cited"}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "citations_delta", "citation": char_citation}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "citations_delta", "citation": char_citation}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "citations_delta", "citation": web_citation}}),
            json!({"type": "content_block_stop", "index": 0}),
        ],
        json!({"output_tokens": 1}),
        "end_turn",
    );
    let mut pdf = file(
        FileData::Bytes(b"%PDF".to_vec()),
        "application/pdf",
        Some(json!({"citations": {"enabled": true}})),
    );
    pdf.filename = Some("r.pdf".into());
    let mut handle = suprnova_ai::stream(
        &model("claude-sonnet-4-6", Recorder::new([reply])),
        vec![user_with(Part::File(pdf))],
        StreamOptions::default(),
    );
    while let Some(event) = handle.next().await {
        event.unwrap();
    }
    let Part::Text(text) = &handle.message().parts[0] else {
        panic!("{:?}", handle.message().parts)
    };
    assert_eq!(
        text.provider_metadata.as_ref().unwrap()["anthropic"]["citations"],
        json!([char_citation, char_citation, web_citation])
    );
    let mut sources = final_metadata(&handle)["sources"]
        .as_array()
        .unwrap()
        .clone();
    for source in &mut sources {
        source.as_object_mut().unwrap().remove("id");
    }
    assert_eq!(
        sources,
        [
            json!({"sourceType": "document", "mediaType": "application/pdf", "title": "r.pdf", "filename": "r.pdf"}),
            json!({"sourceType": "url", "url": "https://web.test", "title": "Web"}),
        ]
    );
    // With web search results, the results are the sources.
    let handle = streamed(
        "claude-sonnet-4-6",
        blocks_reply(&builtin_blocks()[..2], "end_turn"),
        StreamOptions::default(),
    )
    .await;
    let sources = final_metadata(&handle)["sources"].clone();
    assert_eq!(sources.as_array().unwrap().len(), 1, "{sources}");
    assert_eq!(sources[0]["url"], "https://rust.test");
}

#[tokio::test]
async fn ant_013_compaction_fallback_and_upload_blocks_are_text_parts() {
    let reply = message(
        vec![
            json!({"type": "content_block_start", "index": 0, "content_block": {"type": "compaction", "content": null}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "compaction_delta", "content": "summary"}}),
            json!({"type": "content_block_stop", "index": 0}),
            json!({"type": "content_block_start", "index": 1, "content_block": {"type": "compaction", "signature": "cs", "content": "whole"}}),
            json!({"type": "content_block_stop", "index": 1}),
            json!({"type": "content_block_start", "index": 2, "content_block": {"type": "fallback", "from": {"model": "a"}, "to": {"model": "b"}}}),
            json!({"type": "content_block_stop", "index": 2}),
            json!({"type": "content_block_start", "index": 3, "content_block": {"type": "container_upload", "file_id": "file_1"}}),
            json!({"type": "content_block_stop", "index": 3}),
        ],
        json!({"output_tokens": 1}),
        "end_turn",
    );
    let handle = streamed("claude-sonnet-4-6", reply, StreamOptions::default()).await;
    let parts: Vec<(String, Value)> = handle
        .message()
        .parts
        .iter()
        .map(|p| match p {
            Part::Text(t) => (
                t.text.clone(),
                t.provider_metadata.as_ref().unwrap()["anthropic"].clone(),
            ),
            other => panic!("{other:?}"),
        })
        .collect();
    assert_eq!(
        parts,
        [
            ("summary".to_owned(), json!({"type": "compaction"})),
            (
                "whole".to_owned(),
                json!({"type": "compaction", "signature": "cs"})
            ),
            (
                String::new(),
                json!({"type": "fallback", "from": {"model": "a"}, "to": {"model": "b"}})
            ),
            (
                String::new(),
                json!({"type": "container_upload", "fileId": "file_1"})
            ),
        ]
    );
}

#[tokio::test]
async fn ant_013_pings_and_a_repeated_start_change_nothing() {
    let start = json!({"type": "message_start", "message": {"id": "msg_1", "model": "claude-sonnet-4-6", "content": [],
        "stop_reason": null, "usage": {"input_tokens": 1, "output_tokens": 0}}});
    let events = [
        start.clone(),
        json!({"type": "ping"}),
        start.clone(),
        json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}}),
        json!({"type": "content_block_stop", "index": 0}),
        json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}}),
        json!({"type": "message_stop"}),
    ];
    let handle = streamed("claude-sonnet-4-6", sse(&events), StreamOptions::default()).await;
    assert_eq!(handle.text(), "hi");
    let mut other = start.clone();
    other["message"]["id"] = json!("msg_2");
    let mut handle = suprnova_ai::stream(
        &model("claude-sonnet-4-6", Recorder::new([sse(&[start, other])])),
        vec![user("hi")],
        StreamOptions::default(),
    );
    let error = loop {
        match handle.next().await {
            Some(Err(error)) => break error,
            Some(Ok(_)) => {}
            None => panic!("no error"),
        }
    };
    let suprnova_ai::Error::Provider(error) = error else {
        panic!("{error:?}")
    };
    assert_eq!(error.kind, suprnova_ai::errors::ProviderErrorKind::Response);
}

#[tokio::test]
async fn ant_013_an_error_event_has_the_kind_of_its_status() {
    use suprnova_ai::errors::ProviderErrorKind;
    for (kind, status, retryable) in [
        ("overloaded_error", 529, true),
        ("api_error", 500, true),
        ("rate_limit_error", 429, true),
        ("invalid_request_error", 400, false),
        ("billing_error", 400, false),
        ("not_found_error", 404, false),
    ] {
        let events = [
            json!({"type": "message_start", "message": {"id": "msg_1", "model": "m", "content": [], "stop_reason": null, "usage": {"input_tokens": 1, "output_tokens": 0}}}),
            json!({"type": "error", "error": {"type": kind, "message": "slow down"}}),
        ];
        let model = Model::new(
            "claude-sonnet-4-6",
            Arc::new(AnthropicCompatibleProvider::new(
                config(),
                Recorder::new([sse(&events)]),
            )),
        );
        let mut stream = model
            .provider
            .stream(&model, vec![user("hi")], StreamOptions::default());
        let error = loop {
            match stream.next().await {
                Some(Err(error)) => break error,
                Some(Ok(_)) => {}
                None => panic!("no error"),
            }
        };
        let suprnova_ai::Error::Provider(error) = error else {
            panic!("{error:?}")
        };
        assert_eq!(error.kind, ProviderErrorKind::from_status(status), "{kind}");
        assert_eq!(error.is_retryable, retryable, "{kind}");
        assert_eq!(error.message, "slow down");
        assert_eq!(error.error_type.as_deref(), Some(kind));
    }
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

#[tokio::test]
async fn ant_014_usage_iterations_finish_and_metadata_follow_typescript() {
    let usage = json!({"input_tokens": 10, "cache_creation_input_tokens": 2, "cache_read_input_tokens": 3,
        "output_tokens": 20, "output_tokens_details": {"thinking_tokens": 5},
        "iterations": [{"type": "compaction", "input_tokens": 100, "output_tokens": 7},
                       {"type": "message", "input_tokens": 30, "output_tokens": 13, "model": "claude-sonnet-4-6"},
                       {"type": "advisor_message", "input_tokens": 999, "output_tokens": 999}]});
    let events = [
        json!({"type": "message_start", "message": {"id": "msg_1", "model": "claude-sonnet-4-6", "content": [], "stop_reason": null,
            "usage": {"input_tokens": 10, "output_tokens": 0}}}),
        json!({"type": "message_delta", "delta": {"stop_reason": "refusal", "stop_sequence": null,
            "stop_details": {"type": "refusal", "category": "cyber", "recommended_model": "claude-opus-5"},
            "container": {"id": "cnt_1", "expires_at": "2026-10-25T00:00:00Z", "skills": [{"type": "anthropic", "skill_id": "pptx", "version": "1"}]},
            "safeguard_results": [{"type": "dangerous_tool_use"}]},
            "context_management": {"applied_edits": [{"type": "clear_tool_uses_20250919", "cleared_tool_uses": 2, "cleared_input_tokens": 50}]},
            "usage": usage}),
        json!({"type": "message_stop"}),
    ];
    let handle = streamed("claude-sonnet-4-6", sse(&events), StreamOptions::default()).await;
    let reported = handle.message().usage.clone().unwrap();
    assert_eq!(
        (
            reported.input_tokens,
            reported.output_tokens,
            reported.cache_write_tokens,
            reported.cache_read_tokens,
            reported.reasoning_tokens
        ),
        (135, 20, Some(2), Some(3), Some(5))
    );
    assert_eq!(handle.finish_reason(), Some("content_filter"));
    let md = final_metadata(&handle);
    assert_eq!(md["stopReason"], "refusal");
    assert_eq!(
        md["stopDetails"],
        json!({"type": "refusal", "category": "cyber", "recommendedModel": "claude-opus-5"})
    );
    assert_eq!(
        md["container"],
        json!({"expiresAt": "2026-10-25T00:00:00Z", "id": "cnt_1",
        "skills": [{"type": "anthropic", "skillId": "pptx", "version": "1"}]})
    );
    assert_eq!(
        md["contextManagement"],
        json!({"appliedEdits": [{"type": "clear_tool_uses_20250919",
        "clearedToolUses": 2, "clearedInputTokens": 50}]})
    );
    assert_eq!(
        md["safeguardResults"],
        json!([{"type": "dangerous_tool_use"}])
    );
    assert_eq!(
        md["iterations"][1],
        json!({"type": "message", "model": "claude-sonnet-4-6", "inputTokens": 30, "outputTokens": 13})
    );
    assert_eq!(md["usage"]["cache_creation_input_tokens"], 2);
    assert_eq!(
        suprnova_ai::providers::anthropic::container_id(&[
            user("hi"),
            handle.message().clone(),
            user("next")
        ]),
        Some("cnt_1".to_owned())
    );
}

#[tokio::test]
async fn ant_014_finish_reasons_follow_typescript() {
    for (reason, expected) in [
        ("end_turn", "stop"),
        ("stop_sequence", "stop"),
        ("pause_turn", "stop"),
        ("tool_use", "tool_call"),
        ("max_tokens", "length"),
        ("model_context_window_exceeded", "length"),
        ("refusal", "content_filter"),
        ("compaction", "other"),
    ] {
        let handle = streamed(
            "claude-sonnet-4-6",
            message(vec![], json!({"output_tokens": 1}), reason),
            StreamOptions::default(),
        )
        .await;
        assert_eq!(handle.finish_reason(), Some(expected), "{reason}");
    }
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

#[tokio::test]
async fn ant_015_breakpoints_go_on_each_kind_of_block() {
    use suprnova_ai::types::messages::ToolResultPart;
    let cache = || json!({"cacheControl": {"type": "ephemeral", "ttl": "5m"}});
    let mut pdf = file(FileData::Bytes(b"%PDF".to_vec()), "application/pdf", None);
    pdf.provider_metadata = metadata(cache());
    let mut result = ToolResultPart::new("toolu_1", "read", json!("ok"));
    result.provider_metadata = metadata(json!({"cache_control": {"type": "ephemeral"}}));
    let mut last = Message::new(
        MessageRole::User,
        vec![
            Part::Text(TextPart::new("one")),
            Part::Text(TextPart::new("two")),
        ],
    );
    last.provider_metadata = metadata(cache());
    let mut tool = function_tool("weather");
    tool.tool_config
        .get_or_insert_with(Default::default)
        .args
        .insert("cache_control".into(), json!({"type": "ephemeral"}));
    let (body, _) = sent(
        "claude-sonnet-4-6",
        vec![
            system_with("rules", cache()),
            user_with(Part::File(pdf)),
            assistant(vec![Part::ToolCall(ToolCallPart::new(
                "toolu_1", "read", "{}",
            ))]),
            Message::new(MessageRole::Tool, vec![Part::ToolResult(result)]),
            last,
        ],
        StreamOptions {
            tools: vec![tool],
            ..Default::default()
        },
    )
    .await;
    let control = json!({"type": "ephemeral", "ttl": "5m"});
    assert_eq!(body["system"][0]["cache_control"], control, "{body}");
    assert_eq!(
        body["messages"][0]["content"][0]["cache_control"], control,
        "{body}"
    );
    assert_eq!(
        body["messages"][2]["content"][0]["cache_control"],
        json!({"type": "ephemeral"}),
        "{body}"
    );
    assert!(
        body["messages"][2]["content"][1]
            .get("cache_control")
            .is_none(),
        "{body}"
    );
    assert_eq!(
        body["messages"][2]["content"][2]["cache_control"], control,
        "{body}"
    );
    // The fifth breakpoint, the tool's, is left out.
    assert!(body["tools"][0].get("cache_control").is_none(), "{body}");
}

// ANT-016

#[tokio::test]
async fn ant_016_a_model_that_is_not_claude_takes_the_default_limit() {
    let body = body_for("MiniMax-M2", vec![user("hi")], StreamOptions::default()).await;
    assert_eq!(body["max_tokens"], 4096, "{body}");
}

fn minimax(model_id: &str, transport: Arc<Recorder>) -> Model {
    let mut registry = suprnova_ai::ProviderRegistry::new();
    suprnova_ai::providers::hosts::minimax::register(&mut registry, config(), transport).unwrap();
    Model::new(model_id, registry.get_provider("minimax").unwrap())
}
async fn minimax_body(model_id: &str, options: StreamOptions) -> Value {
    let transport = Recorder::new([text_reply()]);
    run(
        &minimax(model_id, transport.clone()),
        vec![user("hi")],
        options,
    )
    .await;
    transport.body(0)
}

#[tokio::test]
async fn ant_016_minimax_takes_its_typescript_settings() {
    use suprnova_ai::models::params::{SamplerParams, TopPSamplerParams};
    let sampling = |t: f64| {
        let mut params = InferenceRequestParams::default().with_temperature(t);
        if let ProviderValue::Value(SamplerParams { top_p, .. }) = &mut params.sampling {
            *top_p = Some(TopPSamplerParams {
                top_p: ProviderValue::Value(Some(0.9)),
            });
        }
        params
    };
    let body = minimax_body("MiniMax-M3.1", with_params(sampling(1.5))).await;
    assert_eq!(
        (body["temperature"].clone(), body["top_p"].clone()),
        (json!(1.5), json!(0.9)),
        "{body}"
    );
    let body = minimax_body(
        "MiniMax-M3.1",
        with_anthropic(sampling(0.5), json!({"thinking": {"type": "adaptive"}})),
    )
    .await;
    assert_eq!(
        (body["temperature"].clone(), body["top_p"].clone()),
        (json!(0.5), json!(0.9)),
        "{body}"
    );
    let body = minimax_body(
        "MiniMax-M2",
        with_params(InferenceRequestParams::default().with_temperature(1.5)),
    )
    .await;
    assert_eq!(body["temperature"], 1.0, "{body}");
    let body = minimax_body("MiniMax-M2", StreamOptions::default()).await;
    assert_eq!(body["max_tokens"], 4096, "{body}");
}

struct VertexToken;
impl suprnova_ai::providers::google_vertex::AccessTokenSource for VertexToken {
    fn access_token(&self) -> BoxFuture<'_, Result<String, suprnova_ai::Error>> {
        Box::pin(async { Ok("t1".to_owned()) })
    }
}

#[tokio::test]
async fn ant_016_vertex_uses_the_json_tool_and_no_strict_flag() {
    use suprnova_ai::providers::google_vertex;
    let transport = Recorder::new([text_reply()]);
    let config = ProviderConfig {
        env: [
            ("GOOGLE_VERTEX_PROJECT".to_owned(), "p1".to_owned()),
            ("GOOGLE_VERTEX_LOCATION".to_owned(), "us-east5".to_owned()),
        ]
        .into(),
        ..Default::default()
    };
    let provider = google_vertex::anthropic::provider(config, transport.clone())
        .with_access_token_source(Arc::new(VertexToken));
    let mut registry = suprnova_ai::ProviderRegistry::new();
    google_vertex::anthropic::register_provider(&mut registry, provider).unwrap();
    let model = Model::new(
        "claude-sonnet-4-6@20260101",
        registry.get_provider("google-vertex-anthropic").unwrap(),
    );
    let options = StreamOptions {
        output_schema: Some(output_schema()),
        tools: vec![function_tool("lookup").with_strict(true)],
        ..Default::default()
    };
    run(&model, vec![user("hi")], options).await;
    let body = transport.body(0);
    assert!(body.get("output_config").is_none(), "{body}");
    assert!(body["tools"][0].get("strict").is_none(), "{body}");
    assert_eq!(body["tools"][1]["name"], "json", "{body}");
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

#[tokio::test]
async fn ant_017_an_upload_sends_its_form_and_reads_the_file() {
    use suprnova_ai::providers::operations::{UploadFile, UploadParams};
    let reply = json!({"id": "file_1", "type": "file", "filename": "r.pdf", "mime_type": "application/pdf",
        "size_bytes": 4, "created_at": "2026-10-24T00:00:00Z", "downloadable": false});
    let transport = Recorder::new([json_reply(reply.clone()), json_reply(reply)]);
    let provider = AnthropicCompatibleProvider::new(config(), transport.clone());
    let file = |filename: Option<&str>| UploadFile {
        data: Bytes::from_static(b"%PDF"),
        media_type: "application/pdf".into(),
        filename: filename.map(str::to_owned),
    };
    let uploaded = suprnova_ai::ops::upload_file(
        &provider,
        file(Some("r.pdf")),
        UploadParams::default(),
        Default::default(),
    )
    .await
    .unwrap();
    assert_eq!(transport.method(0), "POST");
    let form = transport.form(0);
    assert!(
        form.contains(
            "name=\"file\"; filename=\"r.pdf\"\r\nContent-Type: application/pdf\r\n\r\n%PDF"
        ),
        "{form}"
    );
    assert_eq!(betas(&transport, 0), ["files-api-2025-04-14"]);
    let value = &uploaded.value;
    assert_eq!(
        (
            value.media_type.as_deref(),
            value.filename.as_deref(),
            value.size_bytes,
            value.created_at.as_deref()
        ),
        (
            Some("application/pdf"),
            Some("r.pdf"),
            Some(4),
            Some("2026-10-24T00:00:00Z")
        )
    );
    assert_eq!(
        uploaded.provider_metadata.unwrap()["anthropic"],
        json!({"filename": "r.pdf", "mimeType": "application/pdf", "sizeBytes": 4,
               "createdAt": "2026-10-24T00:00:00Z", "downloadable": false})
    );
    suprnova_ai::ops::upload_file(
        &provider,
        file(None),
        UploadParams::default(),
        Default::default(),
    )
    .await
    .unwrap();
    assert!(
        transport
            .form(1)
            .contains("name=\"file\"; filename=\"blob\""),
        "{}",
        transport.form(1)
    );
}

#[tokio::test]
async fn ant_017_file_info_download_and_delete_are_not_offered() {
    let transport = Recorder::new([]);
    let provider = AnthropicCompatibleProvider::new(config(), transport.clone());
    let reference: std::collections::BTreeMap<String, String> =
        [("anthropic".to_owned(), "file_1".to_owned())].into();
    let errors = [
        suprnova_ai::ops::file_info(&provider, &reference, Default::default())
            .await
            .err(),
        suprnova_ai::ops::download_file(&provider, &reference, Default::default())
            .await
            .err(),
        suprnova_ai::ops::delete_file(&provider, &reference, Default::default())
            .await
            .err(),
    ];
    for error in errors {
        assert!(
            matches!(error, Some(suprnova_ai::Error::UnsupportedOperation(_))),
            "{error:?}"
        );
    }
    assert_eq!(transport.count(), 0);
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

#[tokio::test]
async fn ant_018_a_skill_upload_reads_its_latest_version() {
    use suprnova_ai::providers::operations::{SkillFile, SkillParams};
    let transport = Recorder::new([
        json_reply(
            json!({"id": "skill_1", "display_title": "Notes", "name": "old", "description": "Old",
            "latest_version": "v2", "source": "custom", "created_at": "c", "updated_at": "u"}),
        ),
        json_reply(
            json!({"type": "skill_version", "skill_id": "skill_1", "name": "notes", "description": "Takes notes"}),
        ),
    ]);
    let provider = AnthropicCompatibleProvider::new(config(), transport.clone());
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
    let params = SkillParams {
        display_title: Some("Notes".into()),
        ..Default::default()
    };
    let uploaded = suprnova_ai::ops::upload_skill(&provider, files, params, Default::default())
        .await
        .unwrap();
    let form = transport.form(0);
    assert!(
        form.contains("name=\"display_title\"\r\n\r\nNotes"),
        "{form}"
    );
    assert!(
        form.contains("name=\"files[]\"; filename=\"notes/SKILL.md\""),
        "{form}"
    );
    assert!(
        form.contains("name=\"files[]\"; filename=\"notes/run.py\""),
        "{form}"
    );
    assert_eq!(betas(&transport, 0), ["skills-2025-10-02"]);
    assert_eq!(transport.method(1), "GET");
    assert_eq!(
        transport.url(1),
        "https://api.anthropic.com/v1/skills/skill_1/versions/v2"
    );
    assert_eq!(betas(&transport, 1), ["skills-2025-10-02"]);
    let value = &uploaded.value;
    assert_eq!(
        value.provider_reference,
        [("anthropic".to_owned(), "skill_1".to_owned())].into()
    );
    assert_eq!(
        (
            value.display_title.as_deref(),
            value.name.as_deref(),
            value.description.as_deref(),
            value.latest_version.as_deref()
        ),
        (
            Some("Notes"),
            Some("notes"),
            Some("Takes notes"),
            Some("v2")
        )
    );
    assert_eq!(
        uploaded.provider_metadata.unwrap()["anthropic"],
        json!({"source": "custom", "createdAt": "c", "updatedAt": "u"})
    );
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

fn text_request(
    id: &str,
    model: &str,
    options: StreamOptions,
) -> suprnova_ai::providers::operations::BatchRequest {
    suprnova_ai::providers::operations::BatchRequest {
        id: id.into(),
        model_id: model.into(),
        kind: suprnova_ai::providers::operations::BatchRequestKind::Text {
            messages: vec![user("hi")],
            options,
        },
    }
}
fn batch_object(status: &str, extra: Value) -> Value {
    let mut batch = json!({"id": "msgbatch_1", "type": "message_batch", "processing_status": status,
        "request_counts": {"processing": 1, "succeeded": 2, "errored": 1, "canceled": 0, "expired": 1},
        "created_at": "2026-10-24T00:00:00Z", "expires_at": "2026-10-25T00:00:00Z",
        "archived_at": null, "cancel_initiated_at": null, "ended_at": null, "results_url": null});
    for (key, value) in extra.as_object().unwrap() {
        batch[key] = value.clone();
    }
    batch
}

#[tokio::test]
async fn ant_019_a_batch_sends_buffered_bodies_and_returns_warnings() {
    use suprnova_ai::providers::operations::{BatchParams, BatchState};
    let transport = Recorder::new([json_reply(batch_object("in_progress", json!({})))]);
    let provider = AnthropicCompatibleProvider::new(config(), transport.clone());
    let hot = with_params(InferenceRequestParams::default().with_temperature(1.5));
    let mut search = text_request("r2", "claude-sonnet-4-6", StreamOptions::default());
    if let suprnova_ai::providers::operations::BatchRequestKind::Text { options, .. } =
        &mut search.kind
    {
        options.tools = vec![
            suprnova_ai::providers::anthropic::tools::web_search_20260209(Default::default())
                .unwrap(),
        ];
    }
    let params = BatchParams {
        webhook_url: Some("https://hook.test".into()),
        provider_options: [(
            "anthropic".to_owned(),
            json!({"anthropicBeta": ["batch-beta"]}),
        )]
        .into(),
    };
    let started = suprnova_ai::ops::batch::start(
        &provider,
        vec![text_request("r1", "claude-sonnet-4-6", hot), search],
        params,
        Default::default(),
    )
    .await
    .unwrap();
    assert_eq!(
        transport.url(0),
        "https://api.anthropic.com/v1/messages/batches"
    );
    let body = transport.body(0);
    let requests = body["requests"].as_array().unwrap();
    assert_eq!(requests[0]["custom_id"], "r1");
    assert_eq!(requests[0]["params"]["temperature"], 1.0, "{body}");
    assert!(requests[0]["params"].get("stream").is_none(), "{body}");
    assert_eq!(requests[0]["params"]["model"], "claude-sonnet-4-6");
    let betas = betas(&transport, 0);
    for beta in ["code-execution-web-tools-2026-02-09", "batch-beta"] {
        assert!(betas.iter().any(|b| b == beta), "{betas:?}");
    }
    let warnings: Vec<(Option<&str>, Option<&str>)> = started
        .warnings
        .iter()
        .map(|w| (w.request_id.as_deref(), w.warning.feature.as_deref()))
        .collect();
    assert!(
        warnings.contains(&(Some("r1"), Some("temperature"))),
        "{warnings:?}"
    );
    assert!(
        warnings.contains(&(None, Some("webhookUrl"))),
        "{warnings:?}"
    );
    assert_eq!(started.status.state, BatchState::Pending);
    assert_eq!(started.status.batch_id, "msgbatch_1");
}

#[tokio::test]
async fn ant_019_requests_a_batch_cannot_take_fail_before_a_request() {
    use suprnova_ai::providers::operations::{
        BatchParams, BatchRequest, BatchRequestKind, ImagePrompt,
    };
    let request = |id: &str, options: StreamOptions| text_request(id, "claude-sonnet-4-6", options);
    let mut aliased =
        suprnova_ai::providers::anthropic::tools::web_search_20260209(Default::default()).unwrap();
    aliased.name = "search".into();
    let image = BatchRequest {
        id: "img".into(),
        model_id: "claude-sonnet-4-6".into(),
        kind: BatchRequestKind::Image {
            prompt: ImagePrompt {
                text: Some("a cat".into()),
                images: vec![],
                mask: None,
            },
            params: Default::default(),
        },
    };
    let cases: Vec<Vec<BatchRequest>> = vec![
        vec![request("bad id!", StreamOptions::default())],
        vec![request(&"x".repeat(65), StreamOptions::default())],
        vec![
            request("r1", StreamOptions::default()),
            request("r1", StreamOptions::default()),
        ],
        vec![image],
        vec![request("r1", anthropic(json!({"anthropicBeta": ["x"]})))],
        vec![text_request(
            "r1",
            "claude-3-haiku-20240307",
            StreamOptions {
                output_schema: Some(output_schema()),
                ..Default::default()
            },
        )],
        vec![request(
            "r1",
            StreamOptions {
                tools: vec![aliased],
                ..Default::default()
            },
        )],
        vec![request("r1", anthropic(json!({"speed": "fast"})))],
        vec![request(
            "r1",
            anthropic(json!({"fallbacks": [{"model": "claude-sonnet-4-6", "speed": "fast"}]})),
        )],
    ];
    for requests in cases {
        let transport = Recorder::new([]);
        let provider = AnthropicCompatibleProvider::new(config(), transport.clone());
        let error = suprnova_ai::ops::batch::start(
            &provider,
            requests,
            BatchParams::default(),
            Default::default(),
        )
        .await
        .unwrap_err();
        assert!(
            matches!(error, suprnova_ai::Error::Validation(_)),
            "{error:?}"
        );
        assert_eq!(transport.count(), 0);
    }
}

#[tokio::test]
async fn ant_019_statuses_cancels_and_lists_map_as_typescript_maps_them() {
    use suprnova_ai::providers::operations::BatchState;
    let transport = Recorder::new([
        json_reply(batch_object("in_progress", json!({}))),
        json_reply(batch_object(
            "ended",
            json!({"ended_at": "2026-10-24T01:00:00Z", "results_url": "https://api.anthropic.com/v1/messages/batches/msgbatch_1/results"}),
        )),
        json_reply(batch_object("canceling", json!({}))),
        json_reply(
            json!({"data": [batch_object("ended", json!({}))], "has_more": true, "last_id": "msgbatch_1"}),
        ),
        json_reply(json!({"data": [], "has_more": false, "last_id": null})),
    ]);
    let provider = AnthropicCompatibleProvider::new(config(), transport.clone());
    let status = suprnova_ai::ops::batch::status(&provider, "msgbatch_1", Default::default())
        .await
        .unwrap();
    assert_eq!(
        transport.url(0),
        "https://api.anthropic.com/v1/messages/batches/msgbatch_1"
    );
    assert_eq!(status.state, BatchState::Pending);
    assert_eq!(status.raw_status.as_deref(), Some("in_progress"));
    let counts = status.request_counts.unwrap();
    assert_eq!(
        (
            counts.total,
            counts.pending,
            counts.completed,
            counts.failed
        ),
        (5, 1, 2, 2)
    );
    assert_eq!(status.created_at.as_deref(), Some("2026-10-24T00:00:00Z"));
    assert_eq!(status.expires_at.as_deref(), Some("2026-10-25T00:00:00Z"));
    let status = suprnova_ai::ops::batch::status(&provider, "msgbatch_1", Default::default())
        .await
        .unwrap();
    assert_eq!(status.state, BatchState::Completed);
    assert_eq!(
        status.provider_metadata.unwrap()["anthropic"],
        json!({"archivedAt": null, "cancelInitiatedAt": null, "endedAt": "2026-10-24T01:00:00Z",
               "requestCounts": {"processing": 1, "succeeded": 2, "errored": 1, "canceled": 0, "expired": 1},
               "resultsUrl": "https://api.anthropic.com/v1/messages/batches/msgbatch_1/results"})
    );
    suprnova_ai::ops::batch::cancel(&provider, "msgbatch_1", Default::default())
        .await
        .unwrap();
    assert_eq!(
        (transport.method(2).as_str(), transport.url(2).as_str()),
        (
            "POST",
            "https://api.anthropic.com/v1/messages/batches/msgbatch_1/cancel"
        )
    );
    let page = suprnova_ai::ops::batch::list(
        &provider,
        Some(10),
        Some("msgbatch_0".into()),
        Default::default(),
    )
    .await
    .unwrap();
    assert_eq!(
        transport.url(3),
        "https://api.anthropic.com/v1/messages/batches?after_id=msgbatch_0&limit=10"
    );
    assert_eq!(page.next_cursor.as_deref(), Some("msgbatch_1"));
    assert_eq!(page.batches.len(), 1);
    let page = suprnova_ai::ops::batch::list(&provider, None, None, Default::default())
        .await
        .unwrap();
    assert_eq!(page.next_cursor, None);
}

#[tokio::test]
async fn ant_019_result_lines_give_each_outcome() {
    use suprnova_ai::providers::operations::BatchOutcome;
    let results_url = "https://api.anthropic.com/v1/messages/batches/msgbatch_1/results";
    let message = json!({"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-4-6",
        "content": [{"type": "text", "text": "done"}, {"type": "future_block", "x": 1}],
        "stop_reason": "end_turn", "usage": {"input_tokens": 2, "output_tokens": 1}});
    let lines = [
        json!({"custom_id": "ok", "result": {"type": "succeeded", "message": message}}),
        json!({"custom_id": "bad", "result": {"type": "errored", "error": {"type": "error", "error": {"type": "invalid_request_error", "message": "nope"}, "request_id": "req_1"}}}),
        json!({"custom_id": "c", "result": {"type": "canceled"}}),
        json!({"custom_id": "e", "result": {"type": "expired"}}),
        json!({"custom_id": "w", "result": {"type": "mystery"}}),
    ]
    .iter()
    .map(|l| l.to_string())
    .collect::<Vec<_>>()
    .join("\n");
    let transport = Recorder::new([
        json_reply(batch_object("ended", json!({"results_url": results_url}))),
        (200, lines.into_bytes()),
    ]);
    let provider = AnthropicCompatibleProvider::new(config(), transport.clone());
    let results: Vec<_> =
        suprnova_ai::ops::batch::results(&provider, "msgbatch_1", Default::default())
            .await
            .unwrap()
            .map(Result::unwrap)
            .collect()
            .await;
    assert_eq!(transport.url(1), results_url);
    assert_eq!(
        transport.header(1, "x-api-key").as_deref(),
        Some("sk-ant-test")
    );
    let ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
    assert_eq!(ids, ["ok", "bad", "c", "e", "w"]);
    let BatchOutcome::Message(message) = &results[0].outcome else {
        panic!("{:?}", results[0].outcome)
    };
    assert_eq!(message.text(), "done");
    let BatchOutcome::Failed {
        error,
        provider_metadata,
    } = &results[1].outcome
    else {
        panic!("{:?}", results[1].outcome)
    };
    assert_eq!(
        (error.message.as_str(), error.error_type.as_deref()),
        ("nope", Some("invalid_request_error"))
    );
    assert_eq!(
        provider_metadata.as_ref().unwrap()["anthropic"],
        json!({"requestId": "req_1"})
    );
    assert!(matches!(results[2].outcome, BatchOutcome::Cancelled { .. }));
    assert!(matches!(results[3].outcome, BatchOutcome::Expired { .. }));
    let BatchOutcome::Failed { error, .. } = &results[4].outcome else {
        panic!("{:?}", results[4].outcome)
    };
    assert_eq!(error.code.as_deref(), Some("invalid_response"));
}

#[tokio::test]
async fn ant_019_results_a_batch_cannot_give_fail() {
    for (batch, kind) in [
        (batch_object("in_progress", json!({})), "validation"),
        (
            batch_object(
                "ended",
                json!({"archived_at": "2026-10-30T00:00:00Z", "results_url": "https://api.anthropic.com/r"}),
            ),
            "validation",
        ),
        (batch_object("ended", json!({})), "response"),
        (
            batch_object(
                "ended",
                json!({"results_url": "https://elsewhere.test/results"}),
            ),
            "response",
        ),
    ] {
        let transport = Recorder::new([json_reply(batch.clone())]);
        let provider = AnthropicCompatibleProvider::new(config(), transport.clone());
        let error =
            match suprnova_ai::ops::batch::results(&provider, "msgbatch_1", Default::default())
                .await
            {
                Err(error) => error,
                Ok(_) => panic!("{batch}"),
            };
        match kind {
            "validation" => assert!(
                matches!(error, suprnova_ai::Error::Validation(_)),
                "{error:?}"
            ),
            _ => assert!(
                matches!(&error, suprnova_ai::Error::Provider(e) if e.kind == suprnova_ai::errors::ProviderErrorKind::Response),
                "{error:?}"
            ),
        }
        assert_eq!(transport.count(), 1, "{batch}");
    }
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
