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

//! Tests for HST-001 to HST-007 in docs/spec/hosts.md. Each test name starts
//! with the requirement it checks; scripts/check-hosts.sh selects them by it.
//! A recording transport keeps every request the crate sends and answers it
//! with a reply written in the test, so no request leaves the machine. The
//! tests show the crate sends what the AI SDK for TypeScript 7.0.148 sends;
//! they cannot show that a host still accepts it.

use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

use bytes::Bytes;
use futures::{StreamExt, future::BoxFuture, stream};
use schemars::JsonSchema;
use serde_json::{Value, json};
use suprnova_ai::{
    Error, Model, Provider, ProviderConfig, ProviderRegistry, StreamOptions,
    errors::ProviderErrorKind,
    models::params::*,
    providers::{
        ModelsDevProvider, anthropic,
        hosts::{cerebras, deepinfra, fireworks, minimax, togetherai, zai},
        openai::{self, OpenAIChatCompletionsProtocol, OpenAICompatibleProvider},
    },
    transport::{HttpRequest, HttpResponse, HttpTransport, TransportError},
    types::{
        Event, Message, MessageRole, Part, Tool, ToolChoice, ToolChoiceMode, ToolSelection,
        messages::{ReasoningPart, TextPart},
        tools::ToolSpec,
    },
    user_message,
};

/// Records each request and answers it with the next scripted reply, split
/// into 3-byte chunks so the readers meet every UTF-8 and SSE boundary.
#[derive(Default)]
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
    fn count(&self) -> usize {
        self.requests.lock().unwrap().len()
    }
    fn request(&self) -> HttpRequest {
        let requests = self.requests.lock().unwrap();
        assert_eq!(requests.len(), 1, "expected exactly one request");
        requests[0].clone()
    }
    fn body(&self) -> Value {
        serde_json::from_slice(&self.request().body).unwrap()
    }
    fn header(&self, name: &str) -> Option<String> {
        self.request()
            .headers
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.clone())
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
                .chunks(3)
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

/// A Chat Completions event stream of these chunks.
fn chat_sse(chunks: &[Value]) -> (u16, Vec<u8>) {
    let mut text: String = chunks.iter().map(|c| format!("data: {c}\n\n")).collect();
    text.push_str("data: [DONE]\n\n");
    (200, text.into_bytes())
}
fn chat_text(text: &str, finish: &str) -> (u16, Vec<u8>) {
    chat_sse(&[
        json!({"id":"r1","model":"m","choices":[{"index":0,"delta":{"content":text}}]}),
        json!({"choices":[{"index":0,"delta":{},"finish_reason":finish}]}),
    ])
}
/// A Messages event stream that answers `hi`.
fn messages_text() -> (u16, Vec<u8>) {
    let events = [
        json!({"type":"message_start","message":{"id":"msg_1","model":"m","content":[],"stop_reason":null,"usage":{"input_tokens":1,"output_tokens":0}}}),
        json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}),
        json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}),
        json!({"type":"content_block_stop","index":0}),
        json!({"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}),
        json!({"type":"message_stop"}),
    ];
    let text: String = events
        .iter()
        .map(|e| format!("event: {}\ndata: {e}\n\n", e["type"].as_str().unwrap()))
        .collect();
    (200, text.into_bytes())
}

type Register =
    fn(&mut ProviderRegistry, ProviderConfig, Arc<dyn HttpTransport>) -> Result<(), Error>;
struct Host {
    name: &'static str,
    register: Register,
    url: &'static str,
    key_env: &'static str,
    messages: bool,
}
const HOSTS: [Host; 6] = [
    Host {
        name: "cerebras",
        register: cerebras::register,
        url: "https://api.cerebras.ai/v1/chat/completions",
        key_env: "CEREBRAS_API_KEY",
        messages: false,
    },
    Host {
        name: "deepinfra",
        register: deepinfra::register,
        url: "https://api.deepinfra.com/v1/openai/chat/completions",
        key_env: "DEEPINFRA_API_KEY",
        messages: false,
    },
    Host {
        name: "fireworks",
        register: fireworks::register,
        url: "https://api.fireworks.ai/inference/v1/chat/completions",
        key_env: "FIREWORKS_API_KEY",
        messages: false,
    },
    Host {
        name: "togetherai",
        register: togetherai::register,
        url: "https://api.together.xyz/v1/chat/completions",
        key_env: "TOGETHER_API_KEY",
        messages: false,
    },
    Host {
        name: "zai",
        register: zai::register,
        url: "https://api.z.ai/api/paas/v4/chat/completions",
        key_env: "ZAI_API_KEY",
        messages: false,
    },
    Host {
        name: "minimax",
        register: minimax::register,
        url: "https://api.minimax.io/anthropic/v1/messages",
        key_env: "MINIMAX_API_KEY",
        messages: true,
    },
];
const CHAT_HOSTS: [&str; 5] = ["cerebras", "deepinfra", "fireworks", "togetherai", "zai"];

fn host(name: &str) -> &'static Host {
    HOSTS.iter().find(|h| h.name == name).unwrap()
}
fn key_config() -> ProviderConfig {
    ProviderConfig {
        api_key_value: Some("sk-test".into()),
        ..Default::default()
    }
}
fn registry(host: &Host, config: ProviderConfig, transport: Arc<Recorder>) -> ProviderRegistry {
    let mut registry = ProviderRegistry::new();
    (host.register)(&mut registry, config, transport).unwrap();
    registry
}
fn model(name: &str, model_id: &str, transport: Arc<Recorder>) -> Model {
    let registry = registry(host(name), key_config(), transport);
    Model::new(model_id, registry.get_provider(name).unwrap())
}
/// Stream a call to its end or its first error and return its handle and
/// delivered events.
async fn call(
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
fn ok(events: &[Result<Event, Error>]) {
    for event in events {
        if let Err(error) = event {
            panic!("the call failed: {error:?}");
        }
    }
}
fn params(update: impl FnOnce(&mut InferenceRequestParams)) -> StreamOptions {
    let mut params = InferenceRequestParams::default();
    update(&mut params);
    StreamOptions {
        params: Some(params),
        ..Default::default()
    }
}
fn effort(value: Option<&str>) -> StreamOptions {
    params(|p| {
        p.reasoning = ProviderValue::Value(ReasoningParams {
            effort: ProviderValue::Value(value.map(str::to_owned)),
        })
    })
}
fn lookup_tool() -> Tool {
    Tool::function(
        "lookup",
        ToolSpec {
            description: Some("Look a value up".into()),
            params: json!({"type":"object","properties":{"q":{"type":"integer"}}})
                .try_into()
                .unwrap(),
        },
    )
}

#[derive(JsonSchema)]
#[allow(dead_code)]
struct Answer {
    value: u32,
}

// HST-001

#[tokio::test]
async fn hst_001_each_host_streams_its_protocol_from_its_default_url() {
    for host in &HOSTS {
        let reply = if host.messages {
            messages_text()
        } else {
            chat_text("hi", "stop")
        };
        let transport = Recorder::new([reply]);
        let registry = registry(host, key_config(), transport.clone());
        let handle = format!("@ai-sdk/{}", host.name);
        for id in [host.name, handle.as_str()] {
            let provider = registry.get_provider(id).unwrap();
            assert_eq!(provider.name(), host.name, "{id}");
        }
        let model = Model::new("test-model", registry.get_provider(host.name).unwrap());
        let (handle, events) = call(&model, vec![user_message("hello")], Default::default()).await;
        ok(&events);
        assert_eq!(handle.text(), "hi", "{}", host.name);
        let request = transport.request();
        assert_eq!(request.method, "POST", "{}", host.name);
        assert_eq!(request.url, host.url);
        if host.messages {
            assert_eq!(transport.header("x-api-key").as_deref(), Some("sk-test"));
            assert_eq!(
                transport.header("anthropic-version").as_deref(),
                Some("2023-06-01")
            );
            assert_eq!(transport.header("authorization"), None);
        } else {
            assert_eq!(
                transport.header("authorization").as_deref(),
                Some("Bearer sk-test"),
                "{}",
                host.name
            );
            assert_eq!(transport.header("x-api-key"), None);
        }
        assert_eq!(transport.body()["model"], "test-model");
    }
}

#[tokio::test]
async fn hst_001_a_configured_base_url_and_an_env_key_are_used() {
    for host in &HOSTS {
        let reply = if host.messages {
            messages_text()
        } else {
            chat_text("hi", "stop")
        };
        let transport = Recorder::new([reply]);
        let config = ProviderConfig {
            default_base_url: "https://proxy.test/base".into(),
            env: [(host.key_env.to_owned(), "sk-env".to_owned())].into(),
            ..Default::default()
        };
        let registry = registry(host, config, transport.clone());
        let provider = registry.get_provider(host.name).unwrap();
        assert!(provider.is_configured(), "{}", host.name);
        let model = Model::new("test-model", provider);
        let (_, events) = call(&model, vec![user_message("hello")], Default::default()).await;
        ok(&events);
        let path = if host.messages {
            "/v1/messages"
        } else {
            "/chat/completions"
        };
        assert_eq!(
            transport.request().url,
            format!("https://proxy.test/base{path}")
        );
        let key = if host.messages {
            transport.header("x-api-key")
        } else {
            transport.header("authorization")
        };
        let expected = if host.messages {
            "sk-env"
        } else {
            "Bearer sk-env"
        };
        assert_eq!(key.as_deref(), Some(expected), "{}", host.name);
    }
}

#[test]
fn hst_001_a_host_without_a_key_is_not_configured() {
    for host in &HOSTS {
        let registry = registry(host, ProviderConfig::default(), Recorder::new([]));
        assert!(
            !registry.get_provider(host.name).unwrap().is_configured(),
            "{}",
            host.name
        );
    }
}

#[test]
fn hst_001_every_host_registers_with_openai_and_anthropic() {
    let transport: Arc<dyn HttpTransport> = Recorder::new([]);
    let mut registry = ProviderRegistry::new();
    openai::register(
        &mut registry,
        Arc::new(OpenAICompatibleProvider::new(
            key_config(),
            transport.clone(),
        )),
    )
    .unwrap();
    openai::register_factories(&mut registry, transport.clone()).unwrap();
    anthropic::register(&mut registry, key_config(), transport.clone()).unwrap();
    for host in &HOSTS {
        (host.register)(&mut registry, key_config(), transport.clone()).unwrap();
    }
    for name in ["openai", "anthropic"]
        .into_iter()
        .chain(HOSTS.iter().map(|h| h.name))
    {
        assert_eq!(registry.get_provider(name).unwrap().name(), name);
    }
    // A second registration of a host fails and leaves the registry as it was.
    let error = cerebras::register(&mut registry, key_config(), transport).unwrap_err();
    assert!(matches!(error, Error::Configuration(_)), "{error:?}");
    assert_eq!(
        registry.get_provider("cerebras").unwrap().name(),
        "cerebras"
    );
}

#[tokio::test]
async fn hst_001_a_restored_snapshot_streams_from_the_registered_endpoint() {
    for host in &HOSTS {
        let reply = if host.messages {
            messages_text()
        } else {
            chat_text("hi", "stop")
        };
        let transport = Recorder::new([reply]);
        let registry = registry(host, key_config(), transport.clone());
        let model = Model::new("test-model", registry.get_provider(host.name).unwrap());
        let snapshot = model.to_value().unwrap();
        assert!(!snapshot.to_string().contains("sk-test"), "{snapshot}");
        let restored = Model::from_value(snapshot, &registry).unwrap();
        assert_eq!(restored.provider.name(), host.name);
        let (_, events) = call(&restored, vec![user_message("hello")], Default::default()).await;
        ok(&events);
        assert_eq!(transport.request().url, host.url);
    }
}

// HST-002

#[tokio::test]
async fn hst_002_stream_options_follow_the_host() {
    for (name, sends) in [
        ("cerebras", false),
        ("deepinfra", false),
        ("zai", false),
        ("fireworks", true),
        ("togetherai", true),
    ] {
        let transport = Recorder::new([chat_text("hi", "stop")]);
        let model = model(name, "m", transport.clone());
        let (_, events) = call(&model, vec![user_message("hello")], Default::default()).await;
        ok(&events);
        let body = transport.body();
        if sends {
            assert_eq!(
                body["stream_options"],
                json!({"include_usage": true}),
                "{name}"
            );
        } else {
            assert!(body.get("stream_options").is_none(), "{name}: {body}");
        }
    }
}

#[tokio::test]
async fn hst_002_openai_and_hostless_compatible_providers_ask_for_usage() {
    let openai = |transport: Arc<Recorder>| -> Arc<dyn Provider> {
        Arc::new(
            OpenAICompatibleProvider::new(
                ProviderConfig {
                    default_base_url: "https://openai.test/v1".into(),
                    ..key_config()
                },
                transport,
            )
            .with_protocol(Arc::new(OpenAIChatCompletionsProtocol)),
        )
    };
    let compatible = |transport: Arc<Recorder>| -> Arc<dyn Provider> {
        let metadata = ModelsDevProvider {
            id: "xai".into(),
            npm: Some("@ai-sdk/openai-compatible".into()),
            api: Some("https://api.x.ai/v1".into()),
            env: vec!["XAI_API_KEY".into()],
        };
        Arc::new(
            OpenAICompatibleProvider::from_modelsdev_provider(&metadata, key_config(), transport)
                .unwrap(),
        )
    };
    for build in [openai, compatible] {
        let transport = Recorder::new([chat_text("hi", "stop")]);
        let model = Model::new("m", build(transport.clone()));
        let (_, events) = call(&model, vec![user_message("hello")], Default::default()).await;
        ok(&events);
        assert!(transport.request().url.ends_with("/chat/completions"));
        assert_eq!(
            transport.body()["stream_options"],
            json!({"include_usage": true})
        );
    }
}

// HST-003

/// A history with an assistant turn that carries reasoning.
fn history_with_reasoning() -> Vec<Message> {
    vec![
        user_message("first"),
        Message::new(
            MessageRole::Assistant,
            vec![
                Part::Reasoning(ReasoningPart {
                    id: "r1".into(),
                    text: "thinking it over".into(),
                    provider_metadata: None,
                }),
                Part::Text(TextPart {
                    id: "t1".into(),
                    text: "an answer".into(),
                    provider_metadata: None,
                }),
            ],
        ),
        user_message("again"),
    ]
}

/// The body a host sends with an output token limit of 100, a reasoning
/// history, a reasoning effort and an output schema.
async fn field_rules_body(name: &str, model_id: &str, reasoning: &str) -> Value {
    let transport = Recorder::new([chat_text("{\"value\":1}", "stop")]);
    let model = model(name, model_id, transport.clone());
    let mut options = params(|p| {
        p.output = Some(OutputParams {
            max_tokens: Some(100),
            ..Default::default()
        });
        p.reasoning = ProviderValue::Value(ReasoningParams {
            effort: ProviderValue::Value(Some(reasoning.to_owned())),
        });
    })
    .with_output::<Answer>();
    options.output_type_name = Some("answer".into());
    let (_, events) = call(&model, history_with_reasoning(), options).await;
    ok(&events);
    transport.body()
}

fn assistant_entry(body: &Value) -> &Value {
    body["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["role"] == "assistant")
        .unwrap()
}

#[tokio::test]
async fn hst_003_limits_and_reasoning_use_each_hosts_fields() {
    for name in CHAT_HOSTS {
        let body = field_rules_body(name, "some-model", "high").await;
        let (limit, other) = if name == "cerebras" {
            ("max_completion_tokens", "max_tokens")
        } else {
            ("max_tokens", "max_completion_tokens")
        };
        assert_eq!(body[limit], 100, "{name}: {body}");
        assert!(body.get(other).is_none(), "{name}: {body}");
        let (field, other) = if name == "cerebras" {
            ("reasoning", "reasoning_content")
        } else {
            ("reasoning_content", "reasoning")
        };
        let entry = assistant_entry(&body);
        assert_eq!(entry[field], "thinking it over", "{name}: {entry}");
        assert!(entry.get(other).is_none(), "{name}: {entry}");
        assert_eq!(body["reasoning_effort"], "high", "{name}");
    }
}

#[tokio::test]
async fn hst_003_fireworks_maps_the_efforts_it_lacks() {
    for (effort, sent) in [
        ("minimal", "low"),
        ("low", "low"),
        ("medium", "medium"),
        ("high", "high"),
        ("xhigh", "high"),
        ("max", "high"),
    ] {
        let body = field_rules_body("fireworks", "some-model", effort).await;
        assert_eq!(body["reasoning_effort"], sent, "{effort}");
    }
    // Other hosts send the effort as it is.
    let body = field_rules_body("deepinfra", "some-model", "minimal").await;
    assert_eq!(body["reasoning_effort"], "minimal");
}

#[tokio::test]
async fn hst_003_an_output_schema_follows_each_hosts_support() {
    for (name, model_id, schema) in [
        ("cerebras", "some-model", true),
        ("deepinfra", "some-model", true),
        ("fireworks", "some-model", true),
        ("togetherai", "deepseek-ai/DeepSeek-V4-Flash-0731", true),
        (
            "togetherai",
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            false,
        ),
        ("zai", "glm-4.6", false),
    ] {
        let body = field_rules_body(name, model_id, "high").await;
        let format = &body["response_format"];
        if schema {
            assert_eq!(format["type"], "json_schema", "{name} {model_id}: {body}");
            assert_eq!(format["json_schema"]["name"], "answer");
            assert_eq!(format["json_schema"]["strict"], true);
            assert_eq!(
                format["json_schema"]["schema"]["properties"]["value"]["type"],
                "integer"
            );
        } else {
            assert_eq!(*format, json!({"type": "json_object"}), "{name} {model_id}");
        }
    }
}

// HST-004

#[tokio::test]
async fn hst_004_zai_tool_choice_none_sends_no_tools() {
    let by_option = StreamOptions {
        tools: vec![lookup_tool()],
        tool_choice: Some(ToolChoice::None),
        ..Default::default()
    };
    let mut by_param = params(|p| {
        p.tool_calling = Some(ToolCallingParams {
            tool_choice: ToolChoice::None,
            parallel_tool_calls: ProviderValue::Value(true),
            max_tool_calls: ProviderValue::Default,
        })
    });
    by_param.tools = vec![lookup_tool()];
    for options in [by_option, by_param] {
        let transport = Recorder::new([chat_text("hi", "stop")]);
        let model = model("zai", "glm-4.6", transport.clone());
        let (_, events) = call(&model, vec![user_message("hello")], options).await;
        ok(&events);
        let body = transport.body();
        for key in ["tools", "tool_choice", "parallel_tool_calls"] {
            assert!(body.get(key).is_none(), "{key}: {body}");
        }
    }
}

#[tokio::test]
async fn hst_004_zai_automatic_tool_choice_sends_the_tools() {
    let transport = Recorder::new([chat_text("hi", "stop")]);
    let model = model("zai", "glm-4.6", transport.clone());
    let options = StreamOptions {
        tools: vec![lookup_tool()],
        tool_choice: Some(ToolChoice::Auto),
        ..Default::default()
    };
    let (_, events) = call(&model, vec![user_message("hello")], options).await;
    ok(&events);
    let body = transport.body();
    assert_eq!(body["tools"][0]["function"]["name"], "lookup");
    assert_eq!(body["tool_choice"], "auto");
}

#[tokio::test]
async fn hst_004_zai_rejects_what_it_does_not_take_before_any_request() {
    let tool_choice = |choice: ToolChoice| StreamOptions {
        tools: vec![lookup_tool()],
        tool_choice: Some(choice),
        ..Default::default()
    };
    let sampling = |update: fn(&mut SamplerParams)| {
        params(|p| {
            let mut sampling = SamplerParams::default();
            update(&mut sampling);
            p.sampling = ProviderValue::Value(sampling);
        })
    };
    let cases = [
        ("required", tool_choice(ToolChoice::Required)),
        ("named", tool_choice(ToolChoice::Tool("lookup".into()))),
        (
            "selection",
            tool_choice(ToolChoice::Selection(ToolSelection::new(
                ["lookup"],
                ToolChoiceMode::Auto,
            ))),
        ),
        ("required by param", {
            let mut options = params(|p| {
                p.tool_calling = Some(ToolCallingParams {
                    tool_choice: ToolChoice::Required,
                    parallel_tool_calls: ProviderValue::Default,
                    max_tool_calls: ProviderValue::Default,
                })
            });
            options.tools = vec![lookup_tool()];
            options
        }),
        (
            "frequency penalty",
            sampling(|s| {
                s.repetition_penalty = Some(RepetitionPenaltyParams {
                    frequency_penalty: ProviderValue::Value(Some(0.5)),
                    ..Default::default()
                })
            }),
        ),
        (
            "presence penalty",
            sampling(|s| {
                s.repetition_penalty = Some(RepetitionPenaltyParams {
                    presence_penalty: ProviderValue::Value(Some(0.5)),
                    ..Default::default()
                })
            }),
        ),
        (
            "seed",
            sampling(|s| {
                s.seed = Some(SeedSamplerParams {
                    seed: ProviderValue::Value(Some(Seed::Value(7))),
                })
            }),
        ),
    ];
    for (case, options) in cases {
        let transport = Recorder::new([chat_text("hi", "stop")]);
        let model = model("zai", "glm-4.6", transport.clone());
        let (_, events) = call(&model, vec![user_message("hello")], options).await;
        let error = events
            .into_iter()
            .find_map(Result::err)
            .unwrap_or_else(|| panic!("{case}: the call did not fail"));
        assert!(matches!(error, Error::Validation(_)), "{case}: {error:?}");
        assert_eq!(transport.count(), 0, "{case}: a request was sent");
    }
}

#[tokio::test]
async fn hst_004_zai_finish_reasons_map_to_the_normalized_names() {
    for (raw, normalized) in [
        ("sensitive", "content_filter"),
        ("model_context_window_exceeded", "length"),
        ("network_error", "error"),
        ("stop", "stop"),
    ] {
        let transport = Recorder::new([chat_text("partial", raw)]);
        let model = model("zai", "glm-4.6", transport);
        let (handle, events) = call(&model, vec![user_message("hello")], Default::default()).await;
        ok(&events);
        assert_eq!(handle.finish_reason(), Some(normalized), "{raw}");
    }
}

// HST-005

#[tokio::test]
async fn hst_005_error_messages_come_from_every_body_shape() {
    let compatible = |transport: Arc<Recorder>| -> Model {
        let metadata = ModelsDevProvider {
            id: "custom".into(),
            npm: Some("@ai-sdk/openai-compatible".into()),
            api: Some("https://custom.test/v1".into()),
            env: vec![],
        };
        Model::new(
            "m",
            Arc::new(
                OpenAICompatibleProvider::from_modelsdev_provider(
                    &metadata,
                    key_config(),
                    transport,
                )
                .unwrap(),
            ),
        )
    };
    let bodies = [
        json!({"error": {"message": "nested message", "type": "invalid_request_error"}}),
        json!({"error": "string message"}),
        json!({"message": "top-level message", "code": "bad"}),
    ];
    for body in bodies {
        let expected = crate_message(&body);
        for name in CHAT_HOSTS.into_iter().chain(["custom"]) {
            let transport = Recorder::new([(400, body.to_string().into_bytes())]);
            let model = if name == "custom" {
                compatible(transport)
            } else {
                model(name, "m", transport)
            };
            let (_, events) = call(&model, vec![user_message("hello")], Default::default()).await;
            let Some(Error::Provider(error)) = events.into_iter().find_map(Result::err) else {
                panic!("{name}: the call did not fail with a provider error");
            };
            assert_eq!(error.message, expected, "{name}: {body}");
            assert_eq!(error.kind, ProviderErrorKind::BadRequest, "{name}");
            assert_eq!(error.status_code(), Some(400));
        }
    }
}

fn crate_message(body: &Value) -> &str {
    body["error"]["message"]
        .as_str()
        .or_else(|| body["error"].as_str())
        .or_else(|| body["message"].as_str())
        .unwrap()
}

// HST-006

#[tokio::test]
async fn hst_006_deepinfra_counts_reasoning_reported_apart() {
    for (completion, reasoning, output) in [(10, 30, 40), (50, 30, 50)] {
        let transport = Recorder::new([chat_sse(&[
            json!({"choices":[{"index":0,"delta":{"content":"hi"}}]}),
            json!({"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}),
            json!({"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":completion,"total_tokens":5 + completion,"completion_tokens_details":{"reasoning_tokens":reasoning}}}),
        ])]);
        let model = model("deepinfra", "google/gemma", transport);
        let (handle, events) = call(&model, vec![user_message("hello")], Default::default()).await;
        ok(&events);
        let usage = handle.usage().unwrap();
        assert_eq!(usage.output_tokens, output, "{completion}/{reasoning}");
        assert_eq!(usage.reasoning_tokens, Some(reasoning));
        assert_eq!(usage.input_tokens, 5);
    }
}

/// Text, then a tool call, then a `tool_calls` finish.
fn text_then_tool_call() -> (u16, Vec<u8>) {
    chat_sse(&[
        json!({"choices":[{"index":0,"delta":{"content":"{\"value\":1}"}}]}),
        json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"q\":1}"}}]}}]}),
        json!({"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}),
    ])
}

#[tokio::test]
async fn hst_006_cerebras_drops_tool_calls_after_text_under_an_output_schema() {
    let transport = Recorder::new([text_then_tool_call()]);
    let model = model("cerebras", "gpt-oss-120b", transport);
    let options = StreamOptions {
        tools: vec![lookup_tool()],
        ..Default::default()
    }
    .with_output::<Answer>();
    let (handle, events) = call(&model, vec![user_message("hello")], options).await;
    ok(&events);
    for event in &events {
        assert!(
            !matches!(
                event,
                Ok(Event::ToolStart(_) | Event::ToolDelta(_) | Event::ToolEnd(_))
            ),
            "{event:?}"
        );
    }
    assert_eq!(handle.text(), "{\"value\":1}");
    assert!(handle.tool_calls().is_empty());
    assert_eq!(handle.finish_reason(), Some("stop"));
}

#[tokio::test]
async fn hst_006_without_an_output_schema_cerebras_keeps_the_tool_call() {
    let transport = Recorder::new([text_then_tool_call()]);
    let model = model("cerebras", "gpt-oss-120b", transport);
    let options = StreamOptions {
        tools: vec![lookup_tool()],
        ..Default::default()
    };
    let (handle, events) = call(&model, vec![user_message("hello")], options).await;
    ok(&events);
    assert_eq!(handle.tool_calls().len(), 1);
    assert_eq!(handle.finish_reason(), Some("tool_call"));
}

// HST-007

async fn minimax_body(model_id: &str, options: StreamOptions) -> Value {
    let transport = Recorder::new([messages_text()]);
    let model = model("minimax", model_id, transport.clone());
    let (_, events) = call(&model, vec![user_message("hello")], options).await;
    ok(&events);
    transport.body()
}

#[tokio::test]
async fn hst_007_minimax_3_1_and_later_always_reason() {
    for model_id in ["MiniMax-M3.1", "minimax-m4"] {
        let body = minimax_body(model_id, effort(None)).await;
        assert!(body.get("thinking").is_none(), "{model_id}: {body}");
        assert_eq!(body["output_config"]["effort"], "low", "{model_id}");
        let body = minimax_body(model_id, effort(Some("minimal"))).await;
        assert_eq!(body["output_config"]["effort"], "low", "{model_id}");
        let body = minimax_body(model_id, effort(Some("high"))).await;
        assert_eq!(body["output_config"]["effort"], "high", "{model_id}");
        assert!(body.get("thinking").is_none(), "{model_id}: {body}");
    }
}

#[tokio::test]
async fn hst_007_older_minimax_models_keep_the_anthropic_mapping() {
    for options in [effort(None), effort(Some("minimal")), effort(Some("high"))] {
        let minimax = minimax_body("MiniMax-M2", options.clone()).await;
        let transport = Recorder::new([messages_text()]);
        let anthropic = Model::new(
            "MiniMax-M2",
            Arc::new(anthropic::AnthropicCompatibleProvider::new(
                key_config(),
                transport.clone(),
            )),
        );
        let (_, events) = call(&anthropic, vec![user_message("hello")], options).await;
        ok(&events);
        assert_eq!(minimax, transport.body());
    }
    let body = minimax_body("MiniMax-M2", effort(None)).await;
    assert_eq!(body["thinking"], json!({"type": "disabled"}));
}
