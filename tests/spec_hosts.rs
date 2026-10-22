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

//! Tests for HST-001 to HST-018 in docs/spec/hosts.md. Each test name starts
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
        hosts::{
            cerebras, deepinfra, deepseek, fireworks, minimax, mistral, moonshotai, togetherai, zai,
        },
        openai::{self, OpenAIChatCompletionsProtocol, OpenAICompatibleProvider},
    },
    transport::{HttpRequest, HttpResponse, HttpTransport, TransportError},
    types::{
        Event, Message, MessageRole, Part, Tool, ToolChoice, ToolChoiceMode, ToolSelection,
        messages::{
            ContentOutput, ContentOutputType, ContentPart, FilePart, ReasoningPart,
            SpecialToolResult, TextPart, ToolCallPart, ToolResultPart, ToolResultValue,
        },
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
const HOSTS: [Host; 9] = [
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
    Host {
        name: "deepseek",
        register: deepseek::register,
        url: "https://api.deepseek.com/chat/completions",
        key_env: "DEEPSEEK_API_KEY",
        messages: false,
    },
    Host {
        name: "moonshotai",
        register: moonshotai::register,
        url: "https://api.moonshot.ai/v1/chat/completions",
        key_env: "MOONSHOT_API_KEY",
        messages: false,
    },
    Host {
        name: "mistral",
        register: mistral::register,
        url: "https://api.mistral.ai/v1/chat/completions",
        key_env: "MISTRAL_API_KEY",
        messages: false,
    },
];
const CHAT_HOSTS: [&str; 5] = ["cerebras", "deepinfra", "fireworks", "togetherai", "zai"];
/// The hosts whose messages, schemas and stream content differ from OpenAI's.
const NEW_HOSTS: [&str; 3] = ["deepseek", "moonshotai", "mistral"];

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
        ("deepseek", true),
        ("moonshotai", true),
        ("mistral", false),
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

#[tokio::test]
async fn hst_002_models_dev_metadata_for_a_host_gets_its_rules() {
    for (id, npm, sends) in [
        ("deepinfra", "@ai-sdk/deepinfra", false),
        ("deepinfra", "@ai-sdk/openai-compatible", false),
        ("cerebras", "@ai-sdk/cerebras", false),
        ("fireworks", "@ai-sdk/fireworks", true),
    ] {
        let transport = Recorder::new([chat_text("hi", "stop")]);
        let metadata = ModelsDevProvider {
            id: id.into(),
            npm: Some(npm.into()),
            api: Some("https://host.test/v1".into()),
            env: vec![],
        };
        let provider = OpenAICompatibleProvider::from_modelsdev_provider(
            &metadata,
            key_config(),
            transport.clone(),
        )
        .unwrap();
        let model = Model::new("m", Arc::new(provider));
        let options = params(|p| {
            p.output = Some(OutputParams {
                max_tokens: Some(100),
                ..Default::default()
            })
        });
        let (_, events) = call(&model, vec![user_message("hello")], options).await;
        ok(&events);
        let body = transport.body();
        assert_eq!(
            body.get("stream_options").is_some(),
            sends,
            "{id} {npm}: {body}"
        );
        let limit = if id == "cerebras" {
            "max_completion_tokens"
        } else {
            "max_tokens"
        };
        assert_eq!(body[limit], 100, "{id} {npm}: {body}");
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
        for name in CHAT_HOSTS.into_iter().chain(NEW_HOSTS).chain(["custom"]) {
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
async fn hst_006_cerebras_drops_a_tool_call_that_started_before_the_text() {
    let transport = Recorder::new([chat_sse(&[
        json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"q\":"}}]}}]}),
        json!({"choices":[{"index":0,"delta":{"content":"{\"value\":1}"}}]}),
        json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]}}]}),
        json!({"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}),
    ])]);
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
    assert!(handle.tool_calls().is_empty());
    assert_eq!(handle.finish_reason(), Some("stop"));
}

#[tokio::test]
async fn hst_006_cerebras_keeps_tool_calls_without_text_under_an_output_schema() {
    let transport = Recorder::new([chat_sse(&[
        json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"q\":1}"}}]}}]}),
        json!({"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}),
    ])]);
    let model = model("cerebras", "gpt-oss-120b", transport);
    let options = StreamOptions {
        tools: vec![lookup_tool()],
        ..Default::default()
    }
    .with_output::<Answer>();
    let (handle, events) = call(&model, vec![user_message("hello")], options).await;
    ok(&events);
    let ends = events
        .iter()
        .filter(|e| matches!(e, Ok(Event::ToolEnd(_))))
        .count();
    assert_eq!(ends, 1);
    assert_eq!(handle.tool_calls().len(), 1);
    assert_eq!(handle.tool_calls()[0].tool_args, "{\"q\":1}");
    assert_eq!(handle.finish_reason(), Some("tool_call"));
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
async fn hst_007_a_summary_adds_no_thinking_when_reasoning_is_disabled() {
    let mut options = effort(None);
    options.params.as_mut().unwrap().output = Some(OutputParams {
        reasoning_summary: ProviderValue::Value(Some("summarized".into())),
        ..Default::default()
    });
    let body = minimax_body("MiniMax-M3.1", options.clone()).await;
    assert!(body.get("thinking").is_none(), "{body}");
    assert_eq!(body["output_config"]["effort"], "low");
    let body = minimax_body("MiniMax-M2", options).await;
    assert_eq!(body["thinking"], json!({"type": "disabled"}));
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

// HST-008 to HST-018: DeepSeek, Moonshot AI and Mistral.

/// The body a host sends for these messages and options; the call must succeed.
async fn host_body(
    name: &str,
    model_id: &str,
    messages: Vec<Message>,
    options: StreamOptions,
) -> Value {
    let transport = Recorder::new([chat_text("{\"value\":1}", "stop")]);
    let model = model(name, model_id, transport.clone());
    let (_, events) = call(&model, messages, options).await;
    ok(&events);
    transport.body()
}
/// The error a call fails with before it sends any request.
async fn rejected(
    name: &str,
    model_id: &str,
    messages: Vec<Message>,
    options: StreamOptions,
) -> Error {
    let transport = Recorder::new([chat_text("hi", "stop")]);
    let model = model(name, model_id, transport.clone());
    let (_, events) = call(&model, messages, options).await;
    let error = events
        .into_iter()
        .find_map(Result::err)
        .unwrap_or_else(|| panic!("{name} {model_id}: the call did not fail"));
    assert_eq!(
        transport.count(),
        0,
        "{name} {model_id}: a request was sent"
    );
    error
}
async fn validation(name: &str, model_id: &str, options: StreamOptions) -> String {
    match rejected(name, model_id, vec![user_message("hello")], options).await {
        Error::Validation(message) => message,
        other => panic!("{name} {model_id}: {other:?}"),
    }
}
fn sampling(update: impl FnOnce(&mut SamplerParams)) -> StreamOptions {
    params(|p| {
        let mut sampling = SamplerParams::default();
        update(&mut sampling);
        p.sampling = ProviderValue::Value(sampling);
    })
}
fn temperature(value: f64) -> StreamOptions {
    sampling(|s| {
        s.temperature = Some(TemperatureSamplerParams {
            temperature: ProviderValue::Value(value),
        })
    })
}
fn with_effort(mut options: StreamOptions, value: Option<&str>) -> StreamOptions {
    options
        .params
        .get_or_insert_with(Default::default)
        .reasoning = ProviderValue::Value(ReasoningParams {
        effort: ProviderValue::Value(value.map(str::to_owned)),
    });
    options
}
fn tools(names: &[&str]) -> Vec<Tool> {
    names
        .iter()
        .map(|name| {
            Tool::function(
                *name,
                ToolSpec {
                    description: Some("A tool".into()),
                    params: json!({"type":"object","properties":{"q":{"type":"integer"}}})
                        .try_into()
                        .unwrap(),
                },
            )
        })
        .collect()
}
fn text_part(text: &str) -> Part {
    Part::Text(TextPart {
        id: format!("t-{text}"),
        text: text.into(),
        provider_metadata: None,
    })
}
fn reasoning_part(text: &str) -> Part {
    Part::Reasoning(ReasoningPart {
        id: format!("r-{text}"),
        text: text.into(),
        provider_metadata: None,
    })
}
fn assistant(parts: Vec<Part>) -> Message {
    Message::new(MessageRole::Assistant, parts)
}
fn tool_call(id: &str, name: &str) -> Part {
    Part::ToolCall(ToolCallPart::new(id, name, "{\"q\":1}"))
}
fn tool_message(result: ToolResultPart) -> Message {
    Message::new(MessageRole::Tool, vec![Part::ToolResult(result)])
}
fn user_file(file: FilePart) -> Message {
    Message::new(MessageRole::User, vec![text_part("look"), Part::File(file)])
}
fn assistants(body: &Value) -> Vec<&Value> {
    body["messages"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|m| m["role"] == "assistant")
        .collect()
}

#[tokio::test]
async fn hst_008_each_new_host_rejects_the_settings_it_never_sends() {
    let with_tools = |mut options: StreamOptions| {
        options.tools = tools(&["lookup"]);
        options
    };
    let cases: Vec<(&str, StreamOptions, &[&str])> = vec![
        (
            "service tier",
            params(|p| {
                p.provider_service = Some(ProviderServiceParams {
                    service_tier: ProviderValue::Value("flex".into()),
                })
            }),
            &NEW_HOSTS,
        ),
        (
            "metadata",
            params(|p| p.metadata = Some([("k".to_owned(), "v".to_owned())].into())),
            &NEW_HOSTS,
        ),
        (
            "cache retention",
            params(|p| {
                p.cache = Some(CacheParams {
                    retention: ProviderValue::Value("24h".into()),
                    ..Default::default()
                })
            }),
            &NEW_HOSTS,
        ),
        (
            "tool selection",
            with_tools(StreamOptions {
                tool_choice: Some(ToolChoice::Selection(ToolSelection::new(
                    ["lookup"],
                    ToolChoiceMode::Auto,
                ))),
                ..Default::default()
            }),
            &NEW_HOSTS,
        ),
        (
            "safety identifier",
            params(|p| p.safety_identifier = Some("user-1".into())),
            &["deepseek", "mistral"],
        ),
        (
            "cache key",
            params(|p| {
                p.cache = Some(CacheParams {
                    key: Some("k1".into()),
                    ..Default::default()
                })
            }),
            &["deepseek"],
        ),
        (
            "parallel tool calls",
            with_tools(params(|p| {
                p.tool_calling = Some(ToolCallingParams {
                    tool_choice: ToolChoice::Auto,
                    parallel_tool_calls: ProviderValue::Value(false),
                    max_tool_calls: ProviderValue::Default,
                })
            })),
            &["deepseek", "moonshotai"],
        ),
    ];
    for (case, options, rejecting) in cases {
        for name in NEW_HOSTS {
            if rejecting.contains(&name) {
                let message = validation(name, "m", options.clone()).await;
                assert!(
                    message.contains("does not support"),
                    "{case} {name}: {message}"
                );
            } else {
                host_body(name, "m", vec![user_message("hello")], options.clone()).await;
            }
        }
    }
}

#[tokio::test]
async fn hst_008_moonshot_and_mistral_send_the_settings_they_take() {
    let options = params(|p| {
        p.cache = Some(CacheParams {
            key: Some("k1".into()),
            ..Default::default()
        });
        p.safety_identifier = Some("user-1".into());
    });
    let body = host_body(
        "moonshotai",
        "kimi-k3",
        vec![user_message("hello")],
        options,
    )
    .await;
    assert_eq!(body["prompt_cache_key"], "k1");
    assert_eq!(body["safety_identifier"], "user-1");
    let mut options = params(|p| {
        p.cache = Some(CacheParams {
            key: Some("k1".into()),
            ..Default::default()
        });
        p.tool_calling = Some(ToolCallingParams {
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: ProviderValue::Value(false),
            max_tool_calls: ProviderValue::Default,
        });
    });
    options.tools = tools(&["lookup"]);
    let body = host_body(
        "mistral",
        "mistral-small-latest",
        vec![user_message("hello")],
        options,
    )
    .await;
    assert_eq!(body["prompt_cache_key"], "k1");
    assert_eq!(body["parallel_tool_calls"], false);
}

#[tokio::test]
async fn hst_009_deepseek_sends_its_fields_and_thinking() {
    let history = vec![
        user_message("first"),
        assistant(vec![
            reasoning_part("thinking it over"),
            text_part("an answer"),
        ]),
        user_message("again"),
    ];
    let options = with_effort(
        params(|p| {
            p.output = Some(OutputParams {
                max_tokens: Some(100),
                ..Default::default()
            })
        }),
        Some("medium"),
    );
    let body = host_body("deepseek", "deepseek-v4-flash", history, options).await;
    assert_eq!(body["max_tokens"], 100, "{body}");
    assert!(body.get("max_completion_tokens").is_none());
    assert_eq!(
        assistants(&body)[0]["reasoning_content"],
        "thinking it over"
    );
    assert_eq!(body["thinking"], json!({"type": "enabled"}));
    assert_eq!(body["reasoning_effort"], "high");
    for (level, sent) in [
        ("minimal", "low"),
        ("low", "low"),
        ("high", "high"),
        ("xhigh", "max"),
        ("max", "max"),
    ] {
        let body = host_body(
            "deepseek",
            "deepseek-v4-flash",
            vec![user_message("hello")],
            effort(Some(level)),
        )
        .await;
        assert_eq!(body["reasoning_effort"], sent, "{level}");
        assert_eq!(body["thinking"], json!({"type": "enabled"}), "{level}");
    }
    let body = host_body(
        "deepseek",
        "deepseek-v4-flash",
        vec![user_message("hello")],
        effort(None),
    )
    .await;
    assert_eq!(body["thinking"], json!({"type": "disabled"}));
    assert!(body.get("reasoning_effort").is_none(), "{body}");
    let body = host_body(
        "deepseek",
        "deepseek-v4-flash",
        vec![user_message("hello")],
        Default::default(),
    )
    .await;
    assert!(
        body.get("thinking").is_none() && body.get("reasoning_effort").is_none(),
        "{body}"
    );
}

#[tokio::test]
async fn hst_009_deepseek_puts_the_schema_in_a_first_system_message() {
    let options = StreamOptions::default().with_output::<Answer>();
    let schema = serde_json::to_string(options.output_schema.as_ref().unwrap()).unwrap();
    let messages = vec![
        Message::new(MessageRole::System, vec![text_part("be brief")]),
        user_message("hello"),
    ];
    let body = host_body("deepseek", "deepseek-v4-flash", messages, options).await;
    assert_eq!(body["response_format"], json!({"type": "json_object"}));
    assert_eq!(
        body["messages"][0],
        json!({"role": "system", "content": format!("Return JSON that conforms to the following schema: {schema}")})
    );
    assert_eq!(body["messages"][1]["content"], "be brief");
}

#[tokio::test]
async fn hst_009_deepseek_rejects_penalties_seeds_and_sampling_while_it_reasons() {
    let penalty = sampling(|s| {
        s.repetition_penalty = Some(RepetitionPenaltyParams {
            presence_penalty: ProviderValue::Value(Some(0.5)),
            ..Default::default()
        })
    });
    let seed = sampling(|s| {
        s.seed = Some(SeedSamplerParams {
            seed: ProviderValue::Value(Some(Seed::Value(7))),
        })
    });
    let top_p = sampling(|s| {
        s.top_p = Some(TopPSamplerParams {
            top_p: ProviderValue::Value(Some(0.9)),
        })
    });
    validation("deepseek", "deepseek-chat", penalty).await;
    validation("deepseek", "deepseek-chat", seed).await;
    for (model_id, options) in [
        ("deepseek-v4-flash", temperature(0.2)),
        ("deepseek-v4-flash", top_p),
        ("deepseek-reasoner", temperature(0.2)),
        ("deepseek-chat", with_effort(temperature(0.2), Some("high"))),
    ] {
        let message = validation("deepseek", model_id, options).await;
        assert!(message.contains("reasoning"), "{model_id}: {message}");
    }
    for (model_id, options) in [
        ("deepseek-v4-flash", with_effort(temperature(0.2), None)),
        ("deepseek-chat", temperature(0.2)),
    ] {
        let body = host_body("deepseek", model_id, vec![user_message("hello")], options).await;
        assert_eq!(body["temperature"], 0.2, "{model_id}");
    }
}

#[tokio::test]
async fn hst_010_deepseek_reasoning_history_follows_the_model_version() {
    let history = vec![
        user_message("first"),
        assistant(vec![text_part("no reasoning here")]),
        user_message("second"),
        assistant(vec![
            reasoning_part("before the call"),
            tool_call("call_1", "lookup"),
        ]),
        tool_message(ToolResultPart::new("call_1", "lookup", json!({"value": 1}))),
    ];
    let options = StreamOptions {
        tools: tools(&["lookup"]),
        ..Default::default()
    };
    let body = host_body(
        "deepseek",
        "deepseek-v4-pro",
        history.clone(),
        options.clone(),
    )
    .await;
    let turns = assistants(&body);
    assert_eq!(turns[0]["reasoning_content"], "", "{body}");
    assert_eq!(turns[1]["reasoning_content"], "before the call");
    let mut legacy_history = history.clone();
    legacy_history[1] = assistant(vec![
        reasoning_part("old reasoning"),
        text_part("old answer"),
    ]);
    let body = host_body("deepseek", "deepseek-chat", legacy_history, options).await;
    let turns = assistants(&body);
    assert!(turns[0].get("reasoning_content").is_none(), "{body}");
    assert_eq!(turns[1]["reasoning_content"], "before the call");
}

#[tokio::test]
async fn hst_010_deepseek_takes_only_some_images() {
    for media_type in ["application/pdf", "image/bmp", "audio/mpeg"] {
        let file = FilePart::from_bytes(vec![1, 2, 3], Some(media_type), None).unwrap();
        let error = rejected(
            "deepseek",
            "deepseek-v4-pro",
            vec![user_file(file)],
            Default::default(),
        )
        .await;
        assert!(
            matches!(error, Error::Validation(_)),
            "{media_type}: {error:?}"
        );
    }
    let file = FilePart::from_bytes(vec![1, 2, 3], Some("image/png"), None).unwrap();
    let body = host_body(
        "deepseek",
        "deepseek-v4-pro",
        vec![user_file(file)],
        Default::default(),
    )
    .await;
    assert_eq!(
        body["messages"][0]["content"][1],
        json!({"type": "image_url", "image_url": {"url": "data:image/png;base64,AQID"}})
    );
}

fn content_result(texts: &[&str]) -> ToolResultPart {
    ToolResultPart::new(
        "call_1",
        "lookup",
        ToolResultValue::Special(SpecialToolResult::Content(ContentOutput {
            output_type: ContentOutputType::Content,
            value: texts
                .iter()
                .map(|t| {
                    ContentPart::Text(TextPart {
                        id: format!("c-{t}"),
                        text: (*t).into(),
                        provider_metadata: None,
                    })
                })
                .collect(),
        })),
    )
}
fn tool_round(result: ToolResultPart) -> Vec<Message> {
    vec![
        user_message("hello"),
        assistant(vec![tool_call("call_1", "lookup")]),
        tool_message(result),
    ]
}
fn tool_entry(body: &Value) -> &Value {
    body["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["role"] == "tool")
        .unwrap()
}

#[tokio::test]
async fn hst_010_deepseek_sends_a_text_only_content_result_as_a_string() {
    let options = StreamOptions {
        tools: tools(&["lookup"]),
        ..Default::default()
    };
    let body = host_body(
        "deepseek",
        "deepseek-v4-pro",
        tool_round(content_result(&["a", "b"])),
        options,
    )
    .await;
    let content = &tool_entry(&body)["content"];
    let text = content.as_str().unwrap_or_else(|| panic!("{content}"));
    let parts: Value = serde_json::from_str(text).unwrap();
    assert_eq!(parts.as_array().unwrap().len(), 2);
}

#[tokio::test]
async fn hst_010_deepseek_usage_and_finish_reasons() {
    for (usage, cached) in [
        (
            json!({"prompt_tokens":100,"completion_tokens":5,"prompt_cache_hit_tokens":60,"prompt_tokens_details":{"cached_tokens":80}}),
            60,
        ),
        (
            json!({"prompt_tokens":100,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":80}}),
            80,
        ),
    ] {
        let transport = Recorder::new([chat_sse(&[
            json!({"choices":[{"index":0,"delta":{"content":"hi"}}]}),
            json!({"choices":[{"index":0,"delta":{},"finish_reason":"insufficient_system_resource"}],"usage":usage}),
        ])]);
        let model = model("deepseek", "deepseek-v4-pro", transport);
        let (handle, events) = call(&model, vec![user_message("hello")], Default::default()).await;
        ok(&events);
        assert_eq!(handle.usage().unwrap().cache_read_tokens, Some(cached));
        assert_eq!(handle.finish_reason(), Some("error"));
    }
}

#[tokio::test]
async fn hst_011_moonshot_reasoning_follows_the_model_family() {
    let limit = params(|p| {
        p.output = Some(OutputParams {
            max_tokens: Some(100),
            ..Default::default()
        })
    });
    let body = host_body("moonshotai", "kimi-k3", vec![user_message("hello")], limit).await;
    assert_eq!(body["max_completion_tokens"], 100);
    assert!(body.get("max_tokens").is_none());
    for (model_id, level, reasoning_effort, thinking) in [
        ("kimi-k3", Some("xhigh"), Some("max"), None),
        ("kimi-k3", Some("minimal"), Some("low"), None),
        ("kimi-next", Some("medium"), Some("high"), None),
        ("kimi-k2.6", Some("high"), None, Some("enabled")),
        ("kimi-k2.5", Some("low"), None, Some("enabled")),
        ("kimi-k2.5", None, None, Some("disabled")),
        ("kimi-k2.6", None, None, Some("disabled")),
        ("kimi-k2.7-code", Some("high"), None, None),
        ("moonshot-v1-8k", None, None, None),
    ] {
        let body = host_body(
            "moonshotai",
            model_id,
            vec![user_message("hello")],
            effort(level),
        )
        .await;
        assert_eq!(
            body.get("reasoning_effort").and_then(Value::as_str),
            reasoning_effort,
            "{model_id} {level:?}: {body}"
        );
        assert_eq!(
            body["thinking"]["type"].as_str(),
            thinking,
            "{model_id} {level:?}: {body}"
        );
    }
}

#[tokio::test]
async fn hst_011_moonshot_rejects_what_a_model_does_not_take() {
    let required = StreamOptions {
        tools: tools(&["lookup"]),
        tool_choice: Some(ToolChoice::Required),
        ..Default::default()
    };
    let penalty = |frequency: bool| {
        sampling(|s| {
            let value = ProviderValue::Value(Some(0.5));
            s.repetition_penalty = Some(if frequency {
                RepetitionPenaltyParams {
                    frequency_penalty: value,
                    ..Default::default()
                }
            } else {
                RepetitionPenaltyParams {
                    presence_penalty: value,
                    ..Default::default()
                }
            })
        })
    };
    let top_p = sampling(|s| {
        s.top_p = Some(TopPSamplerParams {
            top_p: ProviderValue::Value(Some(0.9)),
        })
    });
    let seed = sampling(|s| {
        s.seed = Some(SeedSamplerParams {
            seed: ProviderValue::Value(Some(Seed::Value(7))),
        })
    });
    for (model_id, options) in [
        ("kimi-k3", effort(None)),
        ("kimi-k2.7-code", effort(None)),
        ("kimi-next", effort(None)),
        ("moonshot-v1-8k", effort(Some("high"))),
        ("moonshot-v1-8k", seed),
        ("kimi-k2.5", temperature(0.5)),
        ("kimi-k3", top_p),
        ("kimi-k2.6", penalty(true)),
        ("kimi-k2.7-code-highspeed", penalty(false)),
        ("kimi-k2.6", required.clone()),
        ("kimi-k2.7-code", required.clone()),
        ("kimi-k2.7-code-highspeed", required.clone()),
    ] {
        validation("moonshotai", model_id, options).await;
    }
    let body = host_body(
        "moonshotai",
        "moonshot-v1-8k",
        vec![user_message("hello")],
        temperature(0.5),
    )
    .await;
    assert_eq!(body["temperature"], 0.5);
    let body = host_body(
        "moonshotai",
        "kimi-k3",
        vec![user_message("hello")],
        required,
    )
    .await;
    assert_eq!(body["tool_choice"], "required");
}

/// A schema with a root `$schema`, an optional property, a tuple and a type
/// next to `anyOf`.
fn mfjs_input() -> Value {
    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "pair": {"type": "array", "items": [{"type": "string"}, {"type": "integer"}]},
            "maybe": {"type": "string", "anyOf": [{"minLength": 1}, {"type": "null"}]}
        },
        "required": ["pair"]
    })
}
fn mfjs_output() -> Value {
    json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "pair": {"type": "array", "prefixItems": [{"type": "string"}, {"type": "integer"}]},
            "maybe": {"anyOf": [{"type": "string", "minLength": 1}, {"type": "null"}]}
        },
        "required": ["pair"]
    })
}

#[tokio::test]
async fn hst_012_moonshot_schemas_go_out_as_mfjs() {
    let tool = Tool::function(
        "lookup",
        ToolSpec {
            description: Some("Look a value up".into()),
            params: mfjs_input().try_into().unwrap(),
        },
    );
    let options = StreamOptions {
        tools: vec![tool],
        output_schema: Some(mfjs_input().try_into().unwrap()),
        output_type_name: Some("answer".into()),
        ..Default::default()
    };
    let body = host_body(
        "moonshotai",
        "kimi-k3",
        vec![user_message("hello")],
        options,
    )
    .await;
    assert_eq!(
        body["tools"][0]["function"]["parameters"],
        mfjs_output(),
        "{body}"
    );
    assert_eq!(
        body["response_format"],
        json!({"type": "json_schema", "json_schema": {"name": "answer", "strict": true, "schema": mfjs_output()}})
    );
    let tool = Tool::function(
        "lookup",
        ToolSpec {
            description: None,
            params: json!({"type": "string"}).try_into().unwrap(),
        },
    );
    let options = StreamOptions {
        tools: vec![tool],
        ..Default::default()
    };
    validation("moonshotai", "kimi-k3", options).await;
}

/// Answers downloads with the bytes 1, 2, 3 as a JPEG image, or with the
/// text `downloaded words`, or fails them, and records the URLs.
#[derive(Default)]
struct Downloads {
    urls: Mutex<Vec<String>>,
    text: bool,
    fail: bool,
}
impl suprnova_ai::models::files::DownloadTransport for Downloads {
    fn resolve<'a>(
        &'a self,
        _host: &'a str,
        _port: u16,
    ) -> BoxFuture<'a, Result<Vec<std::net::IpAddr>, suprnova_ai::models::files::DownloadError>>
    {
        Box::pin(async { Ok(vec![std::net::IpAddr::from([93, 184, 216, 34])]) })
    }
    fn send_pinned(
        &self,
        request: HttpRequest,
        _addresses: Vec<std::net::IpAddr>,
    ) -> BoxFuture<'_, Result<HttpResponse, suprnova_ai::models::files::DownloadError>> {
        self.urls.lock().unwrap().push(request.url);
        let (status, content_type, body): (u16, &str, &'static [u8]) = if self.fail {
            (404, "text/plain", b"missing")
        } else if self.text {
            (200, "text/plain", b"downloaded words")
        } else {
            (200, "image/jpeg", &[1, 2, 3])
        };
        Box::pin(async move {
            Ok(HttpResponse {
                status,
                headers: vec![("content-type".into(), content_type.into())],
                body: stream::iter([Ok(Bytes::from_static(body))]).boxed(),
            })
        })
    }
}

#[tokio::test]
async fn hst_013_moonshot_media_is_sent_inline() {
    let video = FilePart::from_bytes(vec![1, 2, 3], Some("video/mp4"), None).unwrap();
    let body = host_body(
        "moonshotai",
        "kimi-k3",
        vec![user_file(video)],
        Default::default(),
    )
    .await;
    assert_eq!(
        body["messages"][0]["content"][1],
        json!({"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AQID"}})
    );
    let text = FilePart::from_bytes(b"plain words".to_vec(), Some("text/plain"), None).unwrap();
    let body = host_body(
        "moonshotai",
        "kimi-k3",
        vec![user_file(text)],
        Default::default(),
    )
    .await;
    assert_eq!(
        body["messages"][0]["content"][1],
        json!({"type": "text", "text": "plain words"})
    );
    for media_type in ["application/pdf", "audio/mpeg"] {
        let file = FilePart::from_bytes(vec![1, 2, 3], Some(media_type), None).unwrap();
        let error = rejected(
            "moonshotai",
            "kimi-k3",
            vec![user_file(file)],
            Default::default(),
        )
        .await;
        assert!(
            matches!(error, Error::Validation(_)),
            "{media_type}: {error:?}"
        );
    }
    let image = || FilePart::from_url("https://images.test/cat.jpg", Some("image/jpeg")).unwrap();
    let error = rejected(
        "moonshotai",
        "kimi-k3",
        vec![user_file(image())],
        Default::default(),
    )
    .await;
    assert!(matches!(error, Error::Validation(_)), "{error:?}");

    // With a download transport the URL is fetched and sent inline, also by
    // a model restored from a snapshot.
    let transport = Recorder::new([chat_text("hi", "stop"), chat_text("hi", "stop")]);
    let downloads = Arc::new(Downloads::default());
    let mut registry = ProviderRegistry::new();
    moonshotai::register_with_downloads(
        &mut registry,
        key_config(),
        transport.clone(),
        downloads.clone(),
    )
    .unwrap();
    let model = Model::new("kimi-k3", registry.get_provider("moonshotai").unwrap());
    let restored = Model::from_value(model.to_value().unwrap(), &registry).unwrap();
    for model in [model, restored] {
        let (_, events) = call(&model, vec![user_file(image())], Default::default()).await;
        ok(&events);
    }
    assert_eq!(
        *downloads.urls.lock().unwrap(),
        ["https://images.test/cat.jpg", "https://images.test/cat.jpg"]
    );
    for request in transport.requests.lock().unwrap().iter() {
        let body: Value = serde_json::from_slice(&request.body).unwrap();
        assert_eq!(
            body["messages"][0]["content"][1],
            json!({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AQID"}})
        );
    }
}

#[tokio::test]
async fn hst_013_moonshot_assistant_and_tool_messages() {
    let options = StreamOptions {
        tools: tools(&["lookup"]),
        ..Default::default()
    };
    let result = ToolResultPart::new("call_1", "lookup", json!({"value": 1}));
    let body = host_body("moonshotai", "kimi-k3", tool_round(result), options.clone()).await;
    let turn = assistants(&body)[0];
    assert!(turn.get("content").is_some_and(Value::is_null), "{turn}");
    assert_eq!(tool_entry(&body)["content"], "{\"value\":1}");
    let body = host_body(
        "moonshotai",
        "kimi-k3",
        tool_round(content_result(&["a"])),
        options,
    )
    .await;
    assert!(tool_entry(&body)["content"].is_string(), "{body}");
}

#[tokio::test]
async fn hst_013_moonshot_usage_can_come_in_the_choice() {
    let choice_usage = json!({"prompt_tokens":12,"completion_tokens":5,"total_tokens":17});
    let top_usage =
        json!({"prompt_tokens":99,"completion_tokens":33,"total_tokens":132,"cached_tokens":10});
    let in_choice =
        json!({"choices":[{"index":0,"delta":{},"finish_reason":"stop","usage":choice_usage}]});
    let top = json!({"choices":[],"usage":top_usage});
    let text = json!({"choices":[{"index":0,"delta":{"content":"hi"}}]});
    for (chunks, input, output, cached) in [
        (vec![text.clone(), in_choice.clone()], 12, 5, None),
        (
            vec![text.clone(), in_choice.clone(), top.clone()],
            99,
            33,
            Some(10),
        ),
        (
            vec![text.clone(), top.clone(), in_choice.clone()],
            99,
            33,
            Some(10),
        ),
    ] {
        let transport = Recorder::new([chat_sse(&chunks)]);
        let model = model("moonshotai", "kimi-k2.5", transport);
        let (handle, events) = call(&model, vec![user_message("hello")], Default::default()).await;
        ok(&events);
        let usage = handle.usage().unwrap();
        assert_eq!((usage.input_tokens, usage.output_tokens), (input, output));
        assert_eq!(usage.cache_read_tokens, cached);
    }
}

#[tokio::test]
async fn hst_014_mistral_request_fields_reasoning_and_schema() {
    let options = sampling(|s| {
        s.seed = Some(SeedSamplerParams {
            seed: ProviderValue::Value(Some(Seed::Value(7))),
        })
    });
    let mut options = options;
    options.params.as_mut().unwrap().output = Some(OutputParams {
        max_tokens: Some(100),
        ..Default::default()
    });
    let body = host_body(
        "mistral",
        "mistral-small-latest",
        vec![user_message("hello")],
        options,
    )
    .await;
    assert_eq!(body["max_tokens"], 100);
    assert_eq!(body["random_seed"], 7);
    assert!(body.get("seed").is_none(), "{body}");
    for (level, sent) in [
        (Some("low"), "high"),
        (Some("max"), "high"),
        (Some("high"), "high"),
        (None, "none"),
    ] {
        let body = host_body(
            "mistral",
            "mistral-medium-latest",
            vec![user_message("hello")],
            effort(level),
        )
        .await;
        assert_eq!(body["reasoning_effort"], sent, "{level:?}");
    }
    for level in [Some("high"), None] {
        validation("mistral", "mistral-large-latest", effort(level)).await;
    }
    let schema = json!({"type":"object","properties":{"a":{"type":"string"},"b":{"type":"integer"}},"required":["a"]});
    let options = StreamOptions {
        output_schema: Some(schema.clone().try_into().unwrap()),
        output_type_name: Some("answer".into()),
        ..Default::default()
    };
    let body = host_body(
        "mistral",
        "mistral-small-latest",
        vec![user_message("hello")],
        options,
    )
    .await;
    assert_eq!(
        body["response_format"],
        json!({"type": "json_schema", "json_schema": {"name": "answer", "strict": false, "schema": schema}})
    );
}

#[tokio::test]
async fn hst_014_mistral_tool_choice() {
    let choose = |choice: ToolChoice| StreamOptions {
        tools: tools(&["a", "b"]),
        tool_choice: Some(choice),
        ..Default::default()
    };
    let body = host_body(
        "mistral",
        "mistral-small-latest",
        vec![user_message("hello")],
        choose(ToolChoice::Required),
    )
    .await;
    assert_eq!(body["tool_choice"], "any");
    assert_eq!(body["tools"].as_array().unwrap().len(), 2);
    let body = host_body(
        "mistral",
        "mistral-small-latest",
        vec![user_message("hello")],
        choose(ToolChoice::Tool("b".into())),
    )
    .await;
    assert_eq!(body["tool_choice"], "any");
    let names: Vec<_> = body["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["function"]["name"].clone())
        .collect();
    assert_eq!(names, [json!("b")]);
    let body = host_body(
        "mistral",
        "mistral-small-latest",
        vec![user_message("hello")],
        choose(ToolChoice::None),
    )
    .await;
    assert_eq!(body["tool_choice"], "none");
    let no_tools = params(|p| {
        p.tool_calling = Some(ToolCallingParams {
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: ProviderValue::Value(true),
            max_tool_calls: ProviderValue::Default,
        })
    });
    let body = host_body(
        "mistral",
        "mistral-small-latest",
        vec![user_message("hello")],
        no_tools,
    )
    .await;
    for key in ["tools", "tool_choice", "parallel_tool_calls"] {
        assert!(body.get(key).is_none(), "{key}: {body}");
    }
}

#[tokio::test]
async fn hst_015_mistral_messages() {
    let history = vec![
        user_message("first"),
        assistant(vec![reasoning_part("R"), text_part("T")]),
        user_message("second"),
        assistant(vec![tool_call("call_1", "t")]),
        tool_message(ToolResultPart::new("call_1", "t", json!({"value": 1}))),
        assistant(vec![text_part("The answer is")]),
    ];
    let options = StreamOptions {
        tools: tools(&["t"]),
        ..Default::default()
    };
    let body = host_body("mistral", "mistral-medium-latest", history, options).await;
    let turns = assistants(&body);
    assert_eq!(
        turns[0]["content"],
        json!([
            {"type": "thinking", "thinking": [{"type": "text", "text": "R"}], "closed": true},
            {"type": "text", "text": "T"}
        ])
    );
    for turn in &turns {
        assert!(turn.get("reasoning_content").is_none(), "{turn}");
    }
    assert_eq!(turns[1]["content"], "");
    assert_eq!(turns[1]["tool_calls"][0]["id"], "call_1");
    assert!(turns[0].get("prefix").is_none() && turns[1].get("prefix").is_none());
    assert_eq!(turns[2]["prefix"], true);
    let tool = tool_entry(&body);
    assert_eq!(tool["name"], "t");
    assert_eq!(tool["content"], "{\"value\":1}");
}

#[tokio::test]
async fn hst_015_mistral_files() {
    for (file, part) in [
        (
            FilePart::from_bytes(vec![1, 2, 3], Some("image/png"), None).unwrap(),
            json!({"type": "image_url", "image_url": "data:image/png;base64,AQID"}),
        ),
        (
            FilePart::from_bytes(vec![1, 2, 3], Some("application/pdf"), None).unwrap(),
            json!({"type": "document_url", "document_url": "data:application/pdf;base64,AQID"}),
        ),
        (
            FilePart::from_url("https://docs.test/a.pdf", Some("application/pdf")).unwrap(),
            json!({"type": "document_url", "document_url": "https://docs.test/a.pdf"}),
        ),
    ] {
        let body = host_body(
            "mistral",
            "mistral-small-latest",
            vec![user_file(file)],
            Default::default(),
        )
        .await;
        assert_eq!(body["messages"][0]["content"][1], part);
    }
    for media_type in ["audio/mpeg", "text/plain"] {
        let file = FilePart::from_bytes(vec![1, 2, 3], Some(media_type), None).unwrap();
        let error = rejected(
            "mistral",
            "mistral-small-latest",
            vec![user_file(file)],
            Default::default(),
        )
        .await;
        assert!(
            matches!(error, Error::Validation(_)),
            "{media_type}: {error:?}"
        );
    }
}

#[tokio::test]
async fn hst_016_mistral_content_parts_become_separate_blocks() {
    let transport = Recorder::new([chat_sse(&[
        json!({"choices":[{"index":0,"delta":{"role":"assistant","content":[{"type":"thinking","thinking":[{"type":"text","text":"A"}]}]}}]}),
        json!({"choices":[{"index":0,"delta":{"content":[{"type":"text","text":"B"},{"type":"reference","reference_ids":[1]}]}}]}),
        json!({"choices":[{"index":0,"delta":{"content":[{"type":"thinking","thinking":[{"type":"text","text":"C"}]}]}}]}),
        json!({"choices":[{"index":0,"delta":{"content":"D"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":4,"num_cached_tokens":5}}),
    ])]);
    let mistral = model("mistral", "mistral-medium-latest", transport);
    let (handle, events) = call(&mistral, vec![user_message("hello")], Default::default()).await;
    ok(&events);
    let mut blocks: Vec<(String, String)> = Vec::new();
    let mut open: Option<String> = None;
    for event in events.iter().flatten() {
        match event {
            Event::ReasoningStart(start) => {
                assert!(open.is_none(), "a block was still open: {open:?}");
                open = Some(start.block_id.clone());
                blocks.push(("reasoning".into(), String::new()));
            }
            Event::TextStart(start) => {
                assert!(open.is_none(), "a block was still open: {open:?}");
                open = Some(start.block_id.clone());
                blocks.push(("text".into(), String::new()));
            }
            Event::ReasoningDelta(delta) => blocks.last_mut().unwrap().1.push_str(&delta.chunk),
            Event::TextDelta(delta) => blocks.last_mut().unwrap().1.push_str(&delta.chunk),
            Event::ReasoningEnd(end) => {
                assert_eq!(open.take().as_deref(), Some(end.block_id.as_str()));
            }
            Event::TextEnd(end) => {
                assert_eq!(open.take().as_deref(), Some(end.block_id.as_str()));
            }
            _ => {}
        }
    }
    let expected: Vec<(String, String)> = [
        ("reasoning", "A"),
        ("text", "B"),
        ("reasoning", "C"),
        ("text", "D"),
    ]
    .iter()
    .map(|(k, t)| ((*k).into(), (*t).into()))
    .collect();
    assert_eq!(blocks, expected);
    assert_eq!(handle.usage().unwrap().cache_read_tokens, Some(5));
    for (raw, normalized) in [("model_length", "length"), ("error", "error")] {
        let transport = Recorder::new([chat_text("partial", raw)]);
        let model = model("mistral", "mistral-medium-latest", transport);
        let (handle, events) = call(&model, vec![user_message("hello")], Default::default()).await;
        ok(&events);
        assert_eq!(handle.finish_reason(), Some(normalized), "{raw}");
        assert_eq!(handle.text(), "partial");
    }
}

#[tokio::test]
async fn hst_017_tool_calls_without_an_index_stay_apart() {
    let fixture = chat_sse(&[
        json!({"choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}),
        json!({"choices":[{"index":0,"delta":{"tool_calls":[{"id":"weather_0","type":"function","function":{"name":"weather","arguments":""}},{"id":"time_1","type":"function","function":{"name":"time","arguments":""}}]}}]}),
        json!({"choices":[{"index":0,"delta":{"tool_calls":[{"function":{"arguments":"{\"location\":\"San Francisco\"}"}},{"function":{"arguments":"{\"zone\":\"UTC\"}"}}]}}]}),
        json!({"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}),
    ]);
    let separate_chunks = chat_sse(&[
        json!({"choices":[{"index":0,"delta":{"tool_calls":[{"id":"a","function":{"name":"lookup","arguments":"{\"q\":1}"}}]}}]}),
        json!({"choices":[{"index":0,"delta":{"tool_calls":[{"id":"b","function":{"name":"lookup","arguments":"{\"q\":2}"}}]},"finish_reason":"tool_calls"}]}),
    ]);
    for (name, reply, expected) in [
        (
            "moonshotai",
            fixture,
            vec![
                ("weather_0", "weather", "{\"location\":\"San Francisco\"}"),
                ("time_1", "time", "{\"zone\":\"UTC\"}"),
            ],
        ),
        (
            "mistral",
            separate_chunks,
            vec![("a", "lookup", "{\"q\":1}"), ("b", "lookup", "{\"q\":2}")],
        ),
    ] {
        let transport = Recorder::new([reply]);
        let model = model(name, "m", transport);
        let (handle, events) = call(&model, vec![user_message("hello")], Default::default()).await;
        ok(&events);
        let calls: Vec<_> = handle
            .tool_calls()
            .iter()
            .map(|c| {
                (
                    c.tool_call_id.clone(),
                    c.tool_name.clone(),
                    c.tool_args.clone(),
                )
            })
            .collect();
        let expected: Vec<_> = expected
            .into_iter()
            .map(|(id, n, a)| (id.to_owned(), n.to_owned(), a.to_owned()))
            .collect();
        assert_eq!(calls, expected, "{name}");
    }
}

#[tokio::test]
async fn hst_017_a_malformed_tool_call_index_is_a_response_error() {
    let transport = Recorder::new([chat_sse(&[
        json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":"not-a-number","id":"a","function":{"name":"lookup","arguments":"{}"}}]}}]}),
        json!({"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}),
    ])]);
    let model = model("cerebras", "m", transport);
    let (_, events) = call(&model, vec![user_message("hello")], Default::default()).await;
    let Some(Error::Provider(error)) = events.into_iter().find_map(Result::err) else {
        panic!("the call did not fail with a provider error");
    };
    assert_eq!(error.kind, ProviderErrorKind::Response, "{error:?}");
}

/// The error a stream that sends this error event first ends the call with,
/// with retries off.
async fn stream_error(name: &str, event: Value) -> suprnova_ai::errors::ProviderError {
    let transport = Recorder::new([chat_sse(&[event])]);
    let model = model(name, "m", transport);
    let mut options = StreamOptions::default();
    options.retry.max_retries = 0;
    let (_, events) = call(&model, vec![user_message("hello")], options).await;
    match events.into_iter().find_map(Result::err) {
        Some(Error::Provider(error)) => *error,
        other => panic!("{name}: {other:?}"),
    }
}

#[tokio::test]
async fn hst_018_stream_errors_get_a_kind_from_their_code_and_type() {
    let error = stream_error(
        "deepseek",
        json!({"error":{"message":"m","type":"rate_limit_error","code":"rate_limit_exceeded"}}),
    )
    .await;
    assert_eq!(error.message, "m");
    assert_eq!(error.code.as_deref(), Some("rate_limit_exceeded"));
    assert_eq!(error.error_type.as_deref(), Some("rate_limit_error"));
    assert_eq!(error.kind, ProviderErrorKind::RateLimit);
    assert!(error.is_retryable);
    let error = stream_error(
        "deepseek",
        json!({"error":{"message":"m","code":"insufficient_quota"}}),
    )
    .await;
    assert_eq!(error.kind, ProviderErrorKind::RateLimit);
    assert!(!error.is_retryable);
    for code in [json!(503), json!("503")] {
        let error = stream_error("deepseek", json!({"error":{"message":"m","code":code}})).await;
        assert_eq!(error.kind, ProviderErrorKind::ServiceUnavailable, "{code}");
        assert!(error.is_retryable);
    }
    let error = stream_error(
        "moonshotai",
        json!({"error":{"message":"m","type":"server_error"}}),
    )
    .await;
    assert_eq!(error.kind, ProviderErrorKind::InternalServer);
    assert!(error.is_retryable);
    let error = stream_error(
        "mistral",
        json!({"object":"error","message":"m","type":"invalid_request_error","param":null,"code":"3230"}),
    )
    .await;
    assert_eq!(error.message, "m");
    assert_eq!(error.kind, ProviderErrorKind::BadRequest);
    assert!(!error.is_retryable);
    let error = stream_error(
        "deepseek",
        json!({"error":{"message":"m","type":"unknown_error"}}),
    )
    .await;
    assert_eq!(error.kind, ProviderErrorKind::Response);
    assert!(!error.is_retryable);
}

#[tokio::test(start_paused = true)]
async fn hst_018_a_retryable_error_first_in_the_stream_is_retried() {
    let transport = Recorder::new([
        chat_sse(&[json!({"error":{"message":"busy","type":"overloaded_error"}})]),
        chat_text("hi", "stop"),
    ]);
    let model = model("deepseek", "deepseek-v4-pro", transport.clone());
    let (handle, events) = call(&model, vec![user_message("hello")], Default::default()).await;
    ok(&events);
    assert_eq!(handle.text(), "hi");
    assert_eq!(transport.count(), 2);
}

#[tokio::test]
async fn hst_014_mistral_rejects_a_named_tool_it_was_not_given() {
    let options = StreamOptions {
        tools: tools(&["a"]),
        tool_choice: Some(ToolChoice::Tool("b".into())),
        ..Default::default()
    };
    validation("mistral", "mistral-small-latest", options).await;
}

/// A Moonshot AI model whose provider has this download transport.
fn moonshot_with(downloads: Arc<Downloads>, transport: Arc<Recorder>) -> Model {
    let mut registry = ProviderRegistry::new();
    moonshotai::register_with_downloads(&mut registry, key_config(), transport, downloads).unwrap();
    Model::new("kimi-k3", registry.get_provider("moonshotai").unwrap())
}

#[tokio::test]
async fn hst_013_a_rejected_call_downloads_nothing() {
    let image = || FilePart::from_url("https://images.test/cat.jpg", Some("image/jpeg")).unwrap();
    let pdf = FilePart::from_bytes(vec![1, 2, 3], Some("application/pdf"), None).unwrap();
    let tiff = FilePart::from_url("https://images.test/cat.tiff", Some("image/tiff")).unwrap();
    let service_tier = params(|p| {
        p.provider_service = Some(ProviderServiceParams {
            service_tier: ProviderValue::Value("flex".into()),
        })
    });
    let string_root = StreamOptions {
        tools: vec![Tool::function(
            "lookup",
            ToolSpec {
                description: None,
                params: json!({"type": "string"}).try_into().unwrap(),
            },
        )],
        ..Default::default()
    };
    let cases = [
        ("service tier", vec![user_file(image())], service_tier),
        (
            "seed",
            vec![user_file(image())],
            sampling(|s| {
                s.seed = Some(SeedSamplerParams {
                    seed: ProviderValue::Value(Some(Seed::Value(7))),
                })
            }),
        ),
        ("string root", vec![user_file(image())], string_root),
        (
            "pdf",
            vec![Message::new(
                MessageRole::User,
                vec![Part::File(image()), Part::File(pdf)],
            )],
            Default::default(),
        ),
        ("tiff", vec![user_file(tiff)], Default::default()),
    ];
    for (case, messages, options) in cases {
        let downloads = Arc::new(Downloads {
            fail: true,
            ..Default::default()
        });
        let transport = Recorder::new([chat_text("hi", "stop")]);
        let model = moonshot_with(downloads.clone(), transport.clone());
        let (_, events) = call(&model, messages, options).await;
        let error = events.into_iter().find_map(Result::err);
        assert!(
            matches!(error, Some(Error::Validation(_))),
            "{case}: {error:?}"
        );
        assert!(
            downloads.urls.lock().unwrap().is_empty(),
            "{case}: a download was sent"
        );
        assert_eq!(transport.count(), 0, "{case}: a request was sent");
    }
}

#[tokio::test]
async fn hst_013_a_moonshot_text_url_is_downloaded_or_rejected() {
    let text = || FilePart::from_url("https://docs.test/notes.txt", Some("text/plain")).unwrap();
    let error = rejected(
        "moonshotai",
        "kimi-k3",
        vec![user_file(text())],
        Default::default(),
    )
    .await;
    assert!(matches!(error, Error::Validation(_)), "{error:?}");
    let downloads = Arc::new(Downloads {
        text: true,
        ..Default::default()
    });
    let transport = Recorder::new([chat_text("hi", "stop")]);
    let model = moonshot_with(downloads.clone(), transport.clone());
    let (_, events) = call(&model, vec![user_file(text())], Default::default()).await;
    ok(&events);
    assert_eq!(
        *downloads.urls.lock().unwrap(),
        ["https://docs.test/notes.txt"]
    );
    assert_eq!(
        transport.body()["messages"][0]["content"][1],
        json!({"type": "text", "text": "downloaded words"})
    );
}
