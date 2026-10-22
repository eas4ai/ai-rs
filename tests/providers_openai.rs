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

use bytes::Bytes;
use futures::{StreamExt, future::BoxFuture, stream};
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};
use suprnova_ai::models::params::*;
use suprnova_ai::providers::openai::{
    OpenAIChatCompletionsProtocol, OpenAICompatibleProvider, register, register_factories,
};
use suprnova_ai::providers::{ModelsDevProvider, ProviderRegistry};
use suprnova_ai::transport::{HttpRequest, HttpResponse, HttpTransport, TransportError};
use suprnova_ai::types::messages::{ContentOutput, ContentPart, FilePart};
use suprnova_ai::types::{
    Event, Message, MessageRole, Part, TextPart, Tool, ToolCallPart, ToolResultPart, ToolSpec,
};
use suprnova_ai::{Model, Provider, ProviderConfig, StreamOptions};

struct Scripted {
    requests: Mutex<Vec<HttpRequest>>,
    replies: Mutex<Vec<Result<HttpResponse, TransportError>>>,
}
impl Scripted {
    fn response(status: u16, bytes: Vec<u8>) -> Arc<Self> {
        // Deliberately split every UTF-8 and SSE boundary.
        let chunks: Vec<_> = bytes
            .chunks(3)
            .map(|b| Ok(Bytes::copy_from_slice(b)))
            .collect();
        Arc::new(Self {
            requests: Mutex::new(vec![]),
            replies: Mutex::new(vec![Ok(HttpResponse {
                status,
                headers: vec![("x-request-id".into(), "req-openai".into())],
                body: stream::iter(chunks).boxed(),
            })]),
        })
    }
    fn sse(items: Vec<Value>) -> Arc<Self> {
        Self::response(
            200,
            items
                .iter()
                .map(|v| format!("data: {v}\n\n"))
                .collect::<String>()
                .into_bytes(),
        )
    }
    fn body(&self) -> Value {
        serde_json::from_slice(&self.requests.lock().unwrap()[0].body).unwrap()
    }
}
impl HttpTransport for Scripted {
    fn send(&self, request: HttpRequest) -> BoxFuture<'_, Result<HttpResponse, TransportError>> {
        self.requests.lock().unwrap().push(request);
        let result = self.replies.lock().unwrap().remove(0);
        Box::pin(async move { result })
    }
}
fn provider(fake: Arc<Scripted>, chat: bool) -> Arc<OpenAICompatibleProvider> {
    let p = OpenAICompatibleProvider::new(
        ProviderConfig {
            default_base_url: "https://openai.test/v1".into(),
            api_key_value: Some("sk-test".into()),
            ..Default::default()
        },
        fake,
    );
    Arc::new(if chat {
        p.with_protocol(Arc::new(OpenAIChatCompletionsProtocol))
    } else {
        p
    })
}
async fn collect(
    fake: Arc<Scripted>,
    chat: bool,
    messages: Vec<Message>,
    options: StreamOptions,
) -> Vec<Event> {
    let model = Model::new("gpt-5.4", provider(fake, chat));
    model
        .provider
        .stream(&model, messages, options)
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .map(Result::unwrap)
        .collect()
}
fn fixture(name: &str) -> Vec<Value> {
    let v: Value = serde_json::from_str(include_str!("fixtures/openai-streams.json")).unwrap();
    v[name].as_array().unwrap().clone()
}
fn end(events: &[Event]) -> &suprnova_ai::types::events::StreamEnd {
    match events.last().unwrap() {
        Event::StreamEnd(v) => v,
        _ => panic!("missing stream end"),
    }
}

#[tokio::test]
async fn request_endpoints_system_roles_and_input() {
    for chat in [false, true] {
        let fake = Scripted::sse(vec![]);
        collect(
            fake.clone(),
            chat,
            vec![
                suprnova_ai::system_message("rules"),
                suprnova_ai::user_message("Hi"),
            ],
            StreamOptions::default(),
        )
        .await;
        let b = fake.body();
        assert_eq!(b["model"], "gpt-5.4");
        assert_eq!(b["stream"], true);
        if chat {
            assert_eq!(
                b["messages"],
                json!([{"role":"system","content":"rules"},{"role":"user","content":"Hi"}])
            );
            assert_eq!(b["stream_options"], json!({"include_usage":true}));
        } else {
            assert_eq!(
                b["input"],
                json!([{"role":"system","content":"rules"},{"role":"user","content":[{"type":"input_text","text":"Hi"}]}])
            );
            assert!(b.get("messages").is_none());
        }
        let req = fake.requests.lock().unwrap();
        assert_eq!(
            req[0].url,
            if chat {
                "https://openai.test/v1/chat/completions"
            } else {
                "https://openai.test/v1/responses"
            }
        );
        assert!(
            req[0]
                .headers
                .contains(&("authorization".into(), "Bearer sk-test".into()))
        );
    }
}

#[derive(schemars::JsonSchema)]
#[allow(dead_code)]
struct Answer {
    answer: String,
    optional: Option<u32>,
}

#[tokio::test]
async fn responses_params_and_structured_output() {
    let fake = Scripted::sse(vec![]);
    let params = InferenceRequestParams {
        reasoning: ReasoningParams {
            effort: Some("high".into()).into(),
        }
        .into(),
        context_management: Some(ContextManagementParams {
            compaction: Some(TokenThreshold(120000)),
        }),
        output: Some(OutputParams {
            include: Some(["file_search_call.results".into()].into()),
            reasoning_summary: Some("auto".into()).into(),
            text_verbosity: Some("low".into()).into(),
            ..Default::default()
        }),
        extra_headers: Some(
            [
                (
                    "x-openai-feature".into(),
                    ExtraHeaderValue::Value("enabled".into()),
                ),
                ("omit".into(), ExtraHeaderValue::Unset),
            ]
            .into(),
        ),
        extra_query: Some([("beta".into(), json!("a b"))].into()),
        extra_body: Some([("future_option".into(), json!(true))].into()),
        ..Default::default()
    }
    .with_provider_param("openai", json!({"store":false}));
    collect(
        fake.clone(),
        false,
        vec![suprnova_ai::user_message("Hi")],
        StreamOptions {
            params: Some(params),
            ..Default::default()
        }
        .with_output::<Answer>(),
    )
    .await;
    let b = fake.body();
    assert_eq!(b["reasoning"], json!({"effort":"high","summary":"auto"}));
    assert_eq!(
        b["context_management"],
        json!([{"type":"compaction","compact_threshold":120000}])
    );
    assert_eq!(b["include"], json!(["file_search_call.results"]));
    assert_eq!(b["store"], false);
    assert_eq!(b["future_option"], true);
    assert_eq!(b["text"]["verbosity"], "low");
    assert_eq!(b["text"]["format"]["strict"], true);
    assert_eq!(b["text"]["format"]["name"], "Answer");
    assert_eq!(b["text"]["format"]["schema"]["additionalProperties"], false);
    assert_eq!(
        b["text"]["format"]["schema"]["required"],
        json!(["answer", "optional"])
    );
    let req = fake.requests.lock().unwrap();
    assert!(req[0].url.ends_with("?beta=a+b"));
    assert!(
        req[0]
            .headers
            .contains(&("x-openai-feature".into(), "enabled".into()))
    );
    assert!(!req[0].headers.iter().any(|(k, _)| k == "omit"));
}

#[tokio::test]
async fn chat_params_sampling_and_random_seed() {
    for seed in [Seed::Value(123), Seed::Random, Seed::Value(-1)] {
        let fake = Scripted::sse(vec![]);
        let params = InferenceRequestParams {
            sampling: SamplerParams {
                temperature: Some(TemperatureSamplerParams {
                    temperature: 0.2.into(),
                }),
                top_p: Some(TopPSamplerParams {
                    top_p: Some(0.9).into(),
                }),
                seed: Some(SeedSamplerParams {
                    seed: Some(seed.clone()).into(),
                }),
                ..Default::default()
            }
            .into(),
            output: Some(OutputParams {
                max_tokens: Some(128),
                ..Default::default()
            }),
            provider_service: Some(ProviderServiceParams {
                service_tier: "auto".to_owned().into(),
            }),
            ..Default::default()
        }
        .with_provider_param("openai", json!({"store":false}));
        collect(
            fake.clone(),
            true,
            vec![suprnova_ai::user_message("Hi")],
            StreamOptions {
                params: Some(params),
                ..Default::default()
            }
            .with_output::<Answer>(),
        )
        .await;
        let b = fake.body();
        assert_eq!(b["temperature"], 0.2);
        assert_eq!(b["top_p"], 0.9);
        assert_eq!(b["max_completion_tokens"], 128);
        assert_eq!(b["service_tier"], "auto");
        assert_eq!(b["store"], false);
        assert_eq!(
            b.get("seed").cloned(),
            if seed == Seed::Value(123) {
                Some(json!(123))
            } else {
                None
            }
        );
        assert_eq!(b["response_format"]["json_schema"]["strict"], true);
    }
}

#[tokio::test]
async fn unsupported_options_rejected_before_http() {
    let cases = [
        (
            true,
            InferenceRequestParams {
                output: Some(OutputParams {
                    text_verbosity: Some("low".into()).into(),
                    ..Default::default()
                }),
                ..Default::default()
            },
        ),
        (
            true,
            InferenceRequestParams {
                context_management: Some(ContextManagementParams {
                    compaction: Some(TokenThreshold(1)),
                }),
                ..Default::default()
            },
        ),
        (
            false,
            InferenceRequestParams {
                sampling: SamplerParams {
                    seed: Some(SeedSamplerParams {
                        seed: Some(Seed::Value(123)).into(),
                    }),
                    ..Default::default()
                }
                .into(),
                ..Default::default()
            },
        ),
        (
            false,
            InferenceRequestParams {
                sampling: SamplerParams {
                    top_k: Some(TopKSamplerParams {
                        top_k: Some(5).into(),
                    }),
                    ..Default::default()
                }
                .into(),
                ..Default::default()
            },
        ),
    ];
    for (chat, params) in cases {
        let fake = Scripted::sse(vec![]);
        let model = Model::new("gpt-5.4", provider(fake.clone(), chat));
        let events = model
            .provider
            .stream(
                &model,
                vec![],
                StreamOptions {
                    params: Some(params),
                    ..Default::default()
                },
            )
            .collect::<Vec<_>>()
            .await;
        assert!(events[0].is_err());
        assert!(fake.requests.lock().unwrap().is_empty());
    }
}

#[tokio::test]
async fn responses_streams_text_usage_and_function_calls() {
    let fake = Scripted::sse(fixture("test_responses_streams_text_and_usage"));
    let events = collect(fake, false, vec![], StreamOptions::default()).await;
    assert!(matches!(&events[1],Event::TextStart(v) if v.block_id=="msg_1"));
    assert!(
        events
            .iter()
            .any(|e| matches!(e,Event::TextDelta(v) if v.chunk=="Hi"))
    );
    let e = end(&events);
    assert_eq!(e.response_id.as_deref(), Some("resp_1"));
    assert_eq!(e.response_model.as_deref(), Some("gpt-5.4"));
    let u = e.event.usage.as_ref().unwrap();
    assert_eq!(
        (
            u.input_tokens,
            u.output_tokens,
            u.reasoning_tokens,
            u.cache_read_tokens
        ),
        (3, 5, Some(2), Some(1))
    );
    let fake = Scripted::sse(fixture("test_responses_streams_function_tool_call"));
    let model = Model::new("gpt-5.4", provider(fake, false));
    let mut handle = suprnova_ai::stream(&model, vec![], StreamOptions::default());
    while let Some(event) = handle.next().await {
        event.unwrap();
    }
    let calls = handle.message().tool_calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].tool_call_id, "call_1");
    assert_eq!(calls[0].tool_args, r#"{"city":"SF"}"#);
}

#[tokio::test]
async fn responses_builtin_tool_call_and_result() {
    let events = collect(
        Scripted::sse(fixture(
            "test_responses_streams_builtin_tool_call_and_result",
        )),
        false,
        vec![],
        StreamOptions::default(),
    )
    .await;
    assert!(
        events
            .iter()
            .any(|e| matches!(e,Event::BuiltinToolStart(v) if v.tool_name=="web_search"))
    );
    let result = events
        .iter()
        .find_map(|e| {
            if let Event::BuiltinToolResult(v) = e {
                Some(&v.result)
            } else {
                None
            }
        })
        .unwrap();
    assert_eq!(
        result.result,
        json!({"action":{"type":"search","query":"weather"}})
    );
}

#[tokio::test]
async fn finish_reason_normalization() {
    for (raw, want) in [
        ("stop", "stop"),
        ("tool_calls", "tool_call"),
        ("function_call", "tool_call"),
        ("length", "length"),
        ("content_filter", "content_filter"),
        ("brand_new_reason", "other"),
    ] {
        let items = vec![
            json!({"id":"chatcmpl-1","model":"gpt-5.4","choices":[{"delta":{"content":"Hi"},"finish_reason":raw}]}),
        ];
        let ev = collect(Scripted::sse(items), true, vec![], StreamOptions::default()).await;
        assert_eq!(end(&ev).finish_reason.as_deref(), Some(want));
        assert_eq!(
            end(&ev)
                .event
                .provider_metadata
                .as_ref()
                .map(|m| m["openai"].clone()),
            if want == "other" {
                Some(json!({"finish_reason":raw}))
            } else {
                None
            }
        );
    }
    for (status, details, output, want) in [
        ("completed", Value::Null, json!([]), "stop"),
        (
            "completed",
            Value::Null,
            json!([{"type":"function_call"}]),
            "tool_call",
        ),
        (
            "incomplete",
            json!({"reason":"max_output_tokens"}),
            json!([]),
            "length",
        ),
        (
            "incomplete",
            json!({"reason":"content_filter"}),
            json!([]),
            "content_filter",
        ),
        (
            "incomplete",
            json!({"reason":"brand_new_reason"}),
            json!([]),
            "other",
        ),
        ("failed", Value::Null, json!([]), "error"),
    ] {
        let ev=collect(Scripted::sse(vec![json!({"type":format!("response.{status}"),"response":{"status":status,"incomplete_details":details,"output":output}})]),false,vec![],StreamOptions::default()).await;
        assert_eq!(end(&ev).finish_reason.as_deref(), Some(want));
    }
}

#[tokio::test]
async fn history_repair_and_store_false_replay() {
    for chat in [false, true] {
        let fake = Scripted::sse(vec![]);
        let messages = vec![
            Message::new(
                MessageRole::Internal,
                vec![Part::Text(TextPart::new("app-only"))],
            ),
            Message::new(
                MessageRole::Assistant,
                vec![Part::ToolCall(ToolCallPart::new(
                    "tc-1", "search", "invalid",
                ))],
            ),
        ];
        collect(fake.clone(), chat, messages, StreamOptions::default()).await;
        let b = fake.body();
        let wire = &b[if chat { "messages" } else { "input" }];
        assert_eq!(wire.as_array().unwrap().len(), 2);
        if chat {
            assert_eq!(wire[0]["tool_calls"][0]["function"]["arguments"], "{}");
            assert_eq!(wire[1]["tool_call_id"], "tc-1");
        } else {
            assert_eq!(wire[0]["arguments"], "{}");
            assert_eq!(wire[1]["call_id"], "tc-1");
        }
    }
    let raw = json!({"id":"fc_1","type":"function_call","call_id":"call_1","name":"weather","arguments":r#"{"city":"SF"}"#});
    for store in [false, true] {
        let fake = Scripted::sse(vec![]);
        let mut tc = ToolCallPart::new("call_1", "weather", r#"{"city":"SF"}"#);
        tc.provider_metadata =
            Some([("openai".into(), json!({"item_id":"fc_1","raw_item":raw}))].into());
        collect(
            fake.clone(),
            false,
            vec![
                Message::new(MessageRole::Assistant, vec![Part::ToolCall(tc)]),
                Message::new(
                    MessageRole::Tool,
                    vec![Part::ToolResult(ToolResultPart::new(
                        "call_1",
                        "weather",
                        json!("sunny"),
                    ))],
                ),
            ],
            StreamOptions {
                params: Some(
                    InferenceRequestParams::default()
                        .with_provider_param("openai", json!({"store":store})),
                ),
                ..Default::default()
            },
        )
        .await;
        assert_eq!(
            fake.body()["input"][0],
            if store {
                json!({"type":"item_reference","id":"fc_1"})
            } else {
                raw.clone()
            }
        );
        assert_eq!(
            fake.body()["input"][1],
            json!({"type":"function_call_output","call_id":"call_1","output":"sunny"})
        );
    }
}

#[tokio::test]
async fn provider_list_models_probe_and_status_errors() {
    use suprnova_ai::errors::ProviderErrorKind as K;
    let fake = Scripted::response(200, br#"{"data":[{"id":"gpt-z"},{"id":"gpt-a"}]}"#.to_vec());
    assert_eq!(
        provider(fake.clone(), false).list_models().await.unwrap(),
        vec!["gpt-a", "gpt-z"]
    );
    assert_eq!(
        fake.requests.lock().unwrap()[0].url,
        "https://openai.test/v1/models"
    );
    for (status, kind) in [
        (400, K::BadRequest),
        (401, K::Authentication),
        (403, K::PermissionDenied),
        (404, K::ModelNotFound),
        (409, K::Conflict),
        (413, K::RequestTooLarge),
        (422, K::UnprocessableEntity),
        (429, K::RateLimit),
        (500, K::InternalServer),
        (503, K::InternalServer),
        (504, K::InternalServer),
        (529, K::InternalServer),
    ] {
        let fake=Scripted::response(status,br#"{"error":{"message":"slow down","code":"rate_limit","param":"model","type":"rate_limit_error"}}"#.to_vec());
        let model = Model::new("gpt-5.4", provider(fake.clone(), false));
        let suprnova_ai::Error::Provider(e) = model.provider.probe(&model).await.unwrap_err()
        else {
            panic!("wrong error")
        };
        assert_eq!(e.kind, kind);
        assert_eq!(e.provider.as_deref(), Some("openai"));
        assert_eq!(e.request_id.as_deref(), Some("req-openai"));
        assert_eq!(e.code.as_deref(), Some("rate_limit"));
        assert_eq!(e.param.as_deref(), Some("model"));
        assert_eq!(e.status_code(), Some(status));
        assert_eq!(e.is_retryable, matches!(status, 409 | 429 | 500..=599));
        assert_eq!(
            fake.requests.lock().unwrap()[0].url,
            "https://openai.test/v1/models/gpt-5.4"
        );
    }
    let fake = Scripted::response(404, b"{}".to_vec());
    let suprnova_ai::Error::Provider(e) = provider(fake, false).list_models().await.unwrap_err()
    else {
        panic!("wrong error")
    };
    assert_eq!(e.kind, K::NotFound);
    let fake = Scripted::response(200, b"{}".to_vec());
    let model = Model::new("a/b?c", provider(fake.clone(), false));
    model.provider.probe(&model).await.unwrap();
    assert_eq!(
        fake.requests.lock().unwrap()[0].url,
        "https://openai.test/v1/models/a%2Fb%3Fc"
    );
}

#[tokio::test]
async fn configuration_modelsdev_and_registry_restoration() {
    let fake = Scripted::sse(vec![]);
    let cfg = ProviderConfig {
        env: [
            ("OPENAI_BASE_URL".into(), "https://proxy.test/v1".into()),
            ("OPENAI_API_KEY".into(), "sk-env".into()),
        ]
        .into(),
        ..Default::default()
    };
    let p = Arc::new(OpenAICompatibleProvider::new(cfg, fake.clone()));
    assert_eq!(p.base_url(), "https://proxy.test/v1");
    assert!(p.is_configured());
    assert_eq!(
        p.protocol().unwrap().protocol_class_id(),
        "openai_responses"
    );
    let mut registry = ProviderRegistry::new();
    register(&mut registry, p.clone()).unwrap();
    register_factories(&mut registry, fake.clone()).unwrap();
    assert_eq!(
        registry.get_provider("@ai-sdk/openai").unwrap().name(),
        "openai"
    );
    let model = Model::new("gpt", p);
    let restored = Model::from_value(model.to_value().unwrap(), &registry).unwrap();
    assert_eq!(model, restored);
    for (id, url, envs) in [
        ("xai", "https://api.x.ai/v1", vec!["XAI_API_KEY"]),
        ("ollama", "http://localhost:11434/v1", vec![]),
        ("custom", "https://custom.test/v1", vec![]),
        (
            "cloudflare-workers-ai",
            "https://api.cloudflare.com/client/v4/accounts/${CLOUDFLARE_ACCOUNT_ID}/ai/v1",
            vec!["CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_KEY"],
        ),
    ] {
        let meta = ModelsDevProvider {
            id: id.into(),
            api: Some(url.into()),
            npm: Some("@ai-sdk/openai-compatible".into()),
            env: envs.into_iter().map(str::to_owned).collect(),
        };
        let p = OpenAICompatibleProvider::from_modelsdev_provider(
            &meta,
            ProviderConfig {
                env: [
                    ("CLOUDFLARE_ACCOUNT_ID".into(), "account-123".into()),
                    ("CLOUDFLARE_API_KEY".into(), "sk-test".into()),
                ]
                .into(),
                ..Default::default()
            },
            fake.clone(),
        )
        .unwrap();
        assert_eq!(p.name(), id);
        assert_eq!(
            p.protocol().unwrap().protocol_class_id(),
            "openai_chat_completions"
        );
        if id == "cloudflare-workers-ai" {
            assert!(p.is_configured());
            assert!(p.base_url().contains("/account-123/"));
        }
    }
    let unconfigured = Arc::new(OpenAICompatibleProvider::new(
        ProviderConfig::default(),
        fake.clone(),
    ));
    let model = Model::new("gpt", unconfigured);
    let suprnova_ai::Error::Provider(e) = model.provider.probe(&model).await.unwrap_err() else {
        panic!("wrong error")
    };
    assert_eq!(
        e.kind,
        suprnova_ai::errors::ProviderErrorKind::NotConfigured
    );
}

#[tokio::test]
async fn multipart_tool_results_and_model_input() {
    for chat in [true, false] {
        for (value, want) in [
            (Value::Null, ""),
            (json!("hello"), "hello"),
            (json!({"key":"value"}), r#"{"key":"value"}"#),
            (json!([1, 2, 3]), "[1,2,3]"),
        ] {
            let fake = Scripted::sse(vec![]);
            let tc = Part::ToolCall(ToolCallPart::new("tc", "read", "{}"));
            let mut result = ToolResultPart::new("tc", "read", json!("hidden"));
            result.set_model_input(value);
            collect(
                fake.clone(),
                chat,
                vec![
                    Message::new(MessageRole::Assistant, vec![tc]),
                    Message::new(MessageRole::Tool, vec![Part::ToolResult(result)]),
                ],
                StreamOptions::default(),
            )
            .await;
            assert_eq!(
                fake.body()[if chat { "messages" } else { "input" }][1]
                    [if chat { "content" } else { "output" }],
                want
            );
        }
        let fake = Scripted::sse(vec![]);
        let image = FilePart::from_bytes(vec![1, 2, 3], Some("image/png"), None).unwrap();
        let pdf =
            FilePart::from_bytes(vec![4], Some("application/pdf"), Some("doc.pdf".into())).unwrap();
        let result = ContentOutput::new(vec![
            ContentPart::Text(TextPart::new("Image loaded")),
            ContentPart::File(image),
            ContentPart::File(pdf),
        ]);
        collect(
            fake.clone(),
            chat,
            vec![
                Message::new(
                    MessageRole::Assistant,
                    vec![Part::ToolCall(ToolCallPart::new("tc", "read", "{}"))],
                ),
                Message::new(
                    MessageRole::Tool,
                    vec![Part::ToolResult(ToolResultPart::new("tc", "read", result))],
                ),
            ],
            StreamOptions::default(),
        )
        .await;
        let b = fake.body();
        let parts =
            &b[if chat { "messages" } else { "input" }][1][if chat { "content" } else { "output" }];
        assert_eq!(parts[0]["text"], "Image loaded");
        assert_eq!(
            parts[1],
            if chat {
                json!({"type":"image_url","image_url":{"url":"data:image/png;base64,AQID"}})
            } else {
                json!({"type":"input_image","image_url":"data:image/png;base64,AQID"})
            }
        );
        assert_eq!(
            parts[2],
            if chat {
                json!({"type":"text","text":"[file: application/pdf]"})
            } else {
                json!({"type":"input_file","file_data":"data:application/pdf;base64,BA==","filename":"doc.pdf"})
            }
        );
    }
}

#[tokio::test]
async fn tools_and_builtin_guards() {
    use suprnova_ai::providers::openai::tools;
    let fake = Scripted::sse(vec![]);
    let weather = Tool::function(
        "weather",
        ToolSpec {
            description: Some("Get weather".into()),
            params: json!({"type":"object","properties":{"city":{"type":"string"}}})
                .try_into()
                .unwrap(),
        },
    );
    let builtin = tools::web_search(json!({"search_context_size":"low"})).unwrap();
    let ci = tools::code_interpreter(json!({})).unwrap();
    collect(
        fake.clone(),
        false,
        vec![],
        StreamOptions::default().with_tools([weather, builtin.clone(), ci]),
    )
    .await;
    assert_eq!(
        fake.body()["tools"],
        json!([{"type":"function","name":"weather","description":"Get weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}},{"type":"web_search","search_context_size":"low"},{"type":"code_interpreter","container":{"type":"auto"}}])
    );
    let fake = Scripted::sse(vec![]);
    let model = Model::new("gpt", provider(fake.clone(), true));
    assert!(
        model
            .provider
            .stream(
                &model,
                vec![],
                StreamOptions::default().with_tools([builtin])
            )
            .next()
            .await
            .unwrap()
            .is_err()
    );
    assert!(fake.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn chat_interleaved_tools_reasoning_and_usage() {
    let chunks = vec![
        json!({"id":"c1","model":"custom","choices":[{"delta":{"reasoning_content":"Think","tool_calls":[{"index":1,"id":"b","function":{"name":"second","arguments":"{"}},{"index":0,"id":"a","function":{"name":"first","arguments":"["}}]},"finish_reason":null}]}),
        json!({"choices":[{"delta":{"content":"Hé","tool_calls":[{"index":0,"function":{"arguments":"]"}},{"index":1,"function":{"arguments":"}"}}]},"finish_reason":"tool_calls"}]}),
        json!({"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":1},"completion_tokens_details":{"reasoning_tokens":2}}}),
    ];
    let model = Model::new("custom", provider(Scripted::sse(chunks), true));
    let mut h = suprnova_ai::stream(&model, vec![], StreamOptions::default());
    while let Some(e) = h.next().await {
        e.unwrap();
    }
    assert_eq!(h.text(), "Hé");
    assert_eq!(h.message().reasoning(), "Think");
    let calls = h.message().tool_calls();
    assert_eq!(calls.len(), 2);
    assert!(
        calls
            .iter()
            .any(|v| v.tool_call_id == "a" && v.tool_args == "[]")
    );
    assert!(
        calls
            .iter()
            .any(|v| v.tool_call_id == "b" && v.tool_args == "{}")
    );
    assert_eq!(
        h.message().usage.as_ref().unwrap().cache_read_tokens,
        Some(1)
    );
}

#[tokio::test]
async fn malformed_sse_and_transport_errors() {
    for chat in [true, false] {
        let fake = Scripted::response(200, b"data: invalid-json\n\n".to_vec());
        let model = Model::new("gpt", provider(fake, chat));
        let errors = model
            .provider
            .stream(&model, vec![], StreamOptions::default())
            .collect::<Vec<_>>()
            .await;
        assert!(errors.iter().any(Result::is_err));
    }
    let fake = Arc::new(Scripted {
        requests: Mutex::new(vec![]),
        replies: Mutex::new(vec![Err(TransportError::Timeout)]),
    });
    let model = Model::new("gpt", provider(fake, false));
    let suprnova_ai::Error::Provider(e) = model
        .provider
        .stream(&model, vec![], StreamOptions::default())
        .next()
        .await
        .unwrap()
        .unwrap_err()
    else {
        panic!("wrong error")
    };
    assert_eq!(e.kind, suprnova_ai::errors::ProviderErrorKind::Timeout);
    assert!(e.is_retryable);
}

#[test]
fn transport_error_retains_original_source() {
    let e = suprnova_ai::providers::openai::errors::map_transport(TransportError::Timeout, "xai");
    let suprnova_ai::Error::Provider(p) = e else {
        panic!("wrong category")
    };
    let source = std::error::Error::source(p.as_ref()).expect("original transport cause");
    assert!(
        std::error::Error::source(source)
            .unwrap()
            .downcast_ref::<TransportError>()
            .is_some()
    );
}

#[tokio::test]
async fn responses_fallback_index_and_done_arguments() {
    let events=collect(Scripted::sse(vec![
        json!({"type":"response.output_item.added","output_index":2,"item":{"type":"function_call","call_id":"call","name":"lookup"}}),
        json!({"type":"response.function_call_arguments.delta","item_id":"new-item-id","output_index":2,"delta":"{}"}),
        json!({"type":"response.output_text.delta","item_id":"msg","delta":"marker"}),
        json!({"type":"response.output_item.done","output_index":2,"item":{"type":"function_call","id":"new-item-id","call_id":"call","name":"lookup","arguments":"{}"}}),
    ]),false,vec![],StreamOptions::default()).await;
    assert_eq!(
        events
            .iter()
            .filter(|e| matches!(e, Event::ToolDelta(_)))
            .count(),
        1
    );
    assert!(matches!(&events[2],Event::ToolDelta(v) if v.chunk=="{}"));
}

#[tokio::test]
async fn responses_reasoning_images_and_done_only_calls() {
    let events=collect(Scripted::sse(vec![
        json!({"type":"response.output_item.added","output_index":0,"item":{"type":"reasoning","id":"r1","encrypted_content":"secret"}}),
        json!({"type":"response.output_item.done","output_index":0,"item":{"type":"reasoning","id":"r1","summary":[{"type":"summary_text","text":"Think"}],"encrypted_content":"encrypted"}}),
        json!({"type":"response.output_item.done","item":{"type":"function_call","id":"fc","call_id":"call","name":"f","arguments":"{}"}}),
        json!({"type":"response.image_generation_call.partial_image","item_id":"im","partial_image_b64":"YQ=="}),
        json!({"type":"response.output_item.done","item":{"type":"image_generation_call","id":"im","result":"Yg=="}}),
    ]),false,vec![],StreamOptions::default().with_tools([suprnova_ai::providers::openai::tools::image_generation(json!({"output_format":"jpeg"})).unwrap()])).await;
    assert_eq!(
        events
            .iter()
            .filter(|e| matches!(e, Event::ReasoningDelta(_)))
            .count(),
        1
    );
    assert!(events.iter().any(|e|matches!(e,Event::ReasoningEnd(v) if v.event.provider_metadata.as_ref().unwrap()["openai"]["reasoning_encrypted_content"]=="encrypted")));
    assert_eq!(
        events
            .iter()
            .filter(|e| matches!(e, Event::ToolStart(_)))
            .count(),
        1
    );
    assert_eq!(
        events
            .iter()
            .filter(|e| matches!(e, Event::ToolDelta(_)))
            .count(),
        1
    );
    let files: Vec<_> = events
        .iter()
        .filter_map(|e| {
            if let Event::File(f) = e {
                Some(f)
            } else {
                None
            }
        })
        .collect();
    assert_eq!(files.len(), 2);
    assert!(files.iter().all(|f| f.media_type == "image/jpeg"));
}

#[tokio::test]
async fn history_rejects_duplicate_and_orphaned_tool_results() {
    let call = Part::ToolCall(ToolCallPart::new("tc", "f", "{}"));
    let result = Part::ToolResult(ToolResultPart::new("tc", "f", json!(1)));
    for parts in [
        vec![Message::new(
            MessageRole::Assistant,
            vec![call.clone(), call.clone()],
        )],
        vec![Message::new(MessageRole::Tool, vec![result.clone()])],
        vec![
            Message::new(MessageRole::Assistant, vec![call]),
            Message::new(MessageRole::Tool, vec![result.clone(), result]),
        ],
    ] {
        let fake = Scripted::sse(vec![]);
        let m = Model::new("gpt", provider(fake.clone(), false));
        assert!(
            m.provider
                .stream(&m, parts, StreamOptions::default())
                .next()
                .await
                .unwrap()
                .is_err()
        );
        assert!(fake.requests.lock().unwrap().is_empty());
    }
}

#[tokio::test]
async fn chat_downloads_audio_and_pdf_through_core_downloads() {
    struct Download(Arc<Scripted>);
    impl suprnova_ai::models::files::DownloadTransport for Download {
        fn resolve<'a>(
            &'a self,
            _host: &'a str,
            _port: u16,
        ) -> BoxFuture<'a, Result<Vec<std::net::IpAddr>, suprnova_ai::models::files::DownloadError>>
        {
            Box::pin(async { Ok(vec!["8.8.8.8".parse().unwrap()]) })
        }
        fn send_pinned(
            &self,
            request: HttpRequest,
            _addresses: Vec<std::net::IpAddr>,
        ) -> BoxFuture<'_, Result<HttpResponse, suprnova_ai::models::files::DownloadError>>
        {
            Box::pin(async move {
                self.0.send(request).await.map_err(|_| {
                    suprnova_ai::models::files::DownloadError::new("test", "transport failed")
                })
            })
        }
    }
    for (media_type, key) in [("audio/wav", "input_audio"), ("application/pdf", "file")] {
        let download = Scripted::response(200, vec![1, 2, 3]);
        let fake = Scripted::sse(vec![]);
        let p = Arc::new(
            OpenAICompatibleProvider::new(
                ProviderConfig {
                    api_key_value: Some("sk-test".into()),
                    ..Default::default()
                },
                fake.clone(),
            )
            .with_protocol(Arc::new(
                OpenAIChatCompletionsProtocol
                    .with_download_transport(Arc::new(Download(download.clone()))),
            )),
        );
        let m = Model::new("gpt", p);
        let f = FilePart::from_url("https://files.test/document", Some(media_type)).unwrap();
        let ev = m
            .provider
            .stream(
                &m,
                vec![Message::new(MessageRole::User, vec![Part::File(f)])],
                StreamOptions::default(),
            )
            .collect::<Vec<_>>()
            .await;
        assert!(ev.iter().all(Result::is_ok));
        let b = fake.body();
        let part = &b["messages"][0]["content"][0];
        assert_eq!(part["type"], key);
        if key == "file" {
            assert_eq!(part[key]["file_data"], "data:application/pdf;base64,AQID");
        } else {
            assert_eq!(part[key]["data"], "AQID");
        }
        assert_eq!(download.requests.lock().unwrap()[0].method, "GET");
    }
}

#[test]
fn strict_schema_inlines_annotated_refs_and_single_allof() {
    use suprnova_ai::providers::openai::protocol::build_request;
    let schema = json!({"type":"object","properties":{"child":{"$ref":"#/$defs/Child","description":"child"}},"$defs":{"Child":{"type":"object","properties":{"name":{"allOf":[{"type":"string"}]}},"required":["name"]}}});
    let request = build_request(
        "gpt",
        &ProviderConfig {
            default_base_url: "https://test/v1".into(),
            ..Default::default()
        },
        vec![],
        &StreamOptions {
            output_schema: Some(schema.try_into().unwrap()),
            ..Default::default()
        },
        false,
    )
    .unwrap();
    let b: Value = serde_json::from_slice(&request.body).unwrap();
    let child = &b["response_format"]["json_schema"]["schema"]["properties"]["child"];
    assert!(child.get("$ref").is_none());
    assert_eq!(child["additionalProperties"], false);
    assert_eq!(child["properties"]["name"]["type"], "string");
}

#[tokio::test]
async fn explicit_provider_owns_request_transport() {
    let a = Scripted::sse(vec![]);
    let b = Scripted::sse(vec![]);
    let model = Model::new("gpt", provider(a.clone(), false));
    let explicit = provider(b.clone(), false);
    let events = explicit
        .stream(&model, vec![], StreamOptions::default())
        .collect::<Vec<_>>()
        .await;
    assert!(events.iter().all(Result::is_ok));
    assert!(a.requests.lock().unwrap().is_empty());
    assert_eq!(b.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn configured_provider_shares_transport_with_protocol() {
    let fake = Scripted::sse(vec![]);
    let p = Arc::new(suprnova_ai::providers::ConfiguredProvider::new(
        "custom",
        ProviderConfig {
            default_base_url: "https://custom.test/v1".into(),
            ..Default::default()
        },
        Arc::new(OpenAIChatCompletionsProtocol),
        Some(fake.clone()),
    ));
    let m = Model::new("local", p);
    let events = m
        .provider
        .stream(
            &m,
            vec![suprnova_ai::user_message("Hi")],
            StreamOptions::default(),
        )
        .collect::<Vec<_>>()
        .await;
    assert!(events.iter().all(Result::is_ok));
    assert_eq!(fake.body()["model"], "local");
}

#[tokio::test]
async fn stream_is_lazy_and_drop_cancels_body() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct Guard(Arc<AtomicUsize>);
    impl Drop for Guard {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    let drops = Arc::new(AtomicUsize::new(0));
    let guard = Guard(drops.clone());
    let body = stream::unfold(guard, |guard| async move {
        Some((Ok(Bytes::from_static(b"data: {}\n\n")), guard))
    })
    .boxed();
    let fake = Arc::new(Scripted {
        requests: Mutex::new(vec![]),
        replies: Mutex::new(vec![Ok(HttpResponse {
            status: 200,
            headers: vec![],
            body,
        })]),
    });
    let m = Model::new("gpt", provider(fake.clone(), false));
    let mut events = m.provider.stream(&m, vec![], StreamOptions::default());
    assert!(fake.requests.lock().unwrap().is_empty());
    assert!(matches!(
        events.next().await.unwrap().unwrap(),
        Event::StreamStart(_)
    ));
    assert_eq!(fake.requests.lock().unwrap().len(), 1);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(events);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn common_params_and_allowed_tool_choices() {
    use suprnova_ai::types::{ToolChoice, ToolChoiceMode, ToolSelection};
    for chat in [false, true] {
        let fake = Scripted::sse(vec![]);
        let params = InferenceRequestParams {
            reasoning: ReasoningParams {
                effort: None.into(),
            }
            .into(),
            output: Some(OutputParams {
                reasoning_summary: None.into(),
                text_verbosity: None.into(),
                max_tokens: Some(10),
                ..Default::default()
            }),
            cache: Some(CacheParams {
                key: Some("cache".into()),
                retention: "24h".to_owned().into(),
                ..Default::default()
            }),
            tool_calling: Some(ToolCallingParams {
                max_tool_calls: ProviderValue::Default,
                parallel_tool_calls: false.into(),
                tool_choice: ToolChoice::Selection(ToolSelection::new(
                    ["z", "a"],
                    ToolChoiceMode::Required,
                )),
            }),
            metadata: Some([("owner".into(), "test".into())].into()),
            safety_identifier: Some("safe".into()),
            ..Default::default()
        };
        collect(
            fake.clone(),
            chat,
            vec![],
            StreamOptions {
                params: Some(params),
                ..Default::default()
            },
        )
        .await;
        let b = fake.body();
        assert_eq!(b["parallel_tool_calls"], false);
        assert_eq!(b["prompt_cache_key"], "cache");
        assert_eq!(b["prompt_cache_retention"], "24h");
        assert_eq!(b["safety_identifier"], "safe");
        assert_eq!(b["metadata"], json!({"owner":"test"}));
        if chat {
            assert_eq!(b["reasoning_effort"], "none");
            assert!(b.get("text").is_none());
            assert_eq!(
                b["tool_choice"],
                json!({"type":"allowed_tools","allowed_tools":{"mode":"required","tools":[{"type":"function","function":{"name":"a"}},{"type":"function","function":{"name":"z"}}]}})
            );
        } else {
            assert_eq!(b["reasoning"], json!({"effort":"none","summary":null}));
            assert_eq!(b["text"]["verbosity"], Value::Null);
            assert_eq!(
                b["tool_choice"],
                json!({"type":"allowed_tools","mode":"required","tools":[{"type":"function","name":"a"},{"type":"function","name":"z"}]})
            );
        }
    }
}

#[tokio::test]
async fn image_pdf_and_text_inputs_use_endpoint_shapes() {
    for chat in [false, true] {
        let fake = Scripted::sse(vec![]);
        let parts = vec![
            Part::Text(TextPart::new("Read")),
            Part::File(FilePart::from_bytes(vec![1], Some("image/*"), None).unwrap()),
            Part::File(FilePart::from_bytes(b"hello".to_vec(), Some("text/plain"), None).unwrap()),
            Part::File(
                FilePart::from_bytes(vec![2], Some("application/pdf"), Some("a.pdf".into()))
                    .unwrap(),
            ),
        ];
        collect(
            fake.clone(),
            chat,
            vec![Message::new(MessageRole::User, parts)],
            StreamOptions::default(),
        )
        .await;
        let b = fake.body();
        let parts = &b[if chat { "messages" } else { "input" }][0]["content"];
        assert_eq!(
            parts[1],
            if chat {
                json!({"type":"image_url","image_url":{"url":"data:image/jpeg;base64,AQ=="}})
            } else {
                json!({"type":"input_image","image_url":"data:image/jpeg;base64,AQ=="})
            }
        );
        assert_eq!(parts[2]["text"], "hello");
        assert_eq!(
            parts[3],
            if chat {
                json!({"type":"file","file":{"filename":"a.pdf","file_data":"data:application/pdf;base64,Ag=="}})
            } else {
                json!({"type":"input_file","filename":"a.pdf","file_data":"data:application/pdf;base64,Ag=="})
            }
        );
    }
}

#[tokio::test]
async fn an_unreadable_error_body_keeps_the_status_kind() {
    use suprnova_ai::errors::ProviderErrorKind as K;
    for (status, kind, retryable) in [(400, K::BadRequest, false), (503, K::InternalServer, true)] {
        let fake = Arc::new(Scripted {
            requests: Mutex::new(vec![]),
            replies: Mutex::new(vec![Ok(HttpResponse {
                status,
                headers: vec![("x-request-id".into(), "req-openai".into())],
                body: stream::iter([
                    Ok(Bytes::from_static(b"{\"error\"")),
                    Err(TransportError::Body("reset".into())),
                ])
                .boxed(),
            })]),
        });
        let model = Model::new("gpt-5.4", provider(fake, false));
        let suprnova_ai::Error::Provider(e) = model.provider.probe(&model).await.unwrap_err()
        else {
            panic!("wrong error")
        };
        assert_eq!(e.kind, kind, "{status}");
        assert_eq!(e.is_retryable, retryable, "{status}");
        assert_eq!(e.status_code(), Some(status));
        assert_eq!(e.request_id.as_deref(), Some("req-openai"));
        assert_eq!(
            e.message,
            format!("OpenAI request failed with HTTP {status}")
        );
    }
}
