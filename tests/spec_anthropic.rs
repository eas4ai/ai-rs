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
