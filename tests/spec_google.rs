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

//! Tests for GGL-001 to GGL-032 in docs/spec/google.md. Each test name
//! starts with the requirement it checks; scripts/check-google.sh selects
//! them by it. A recording transport keeps every request the crate sends and
//! answers it with a reply written in the test, so no request leaves the
//! machine. The tests show the crate sends what the AI SDK for TypeScript
//! 7.0.148 sends; they cannot show that Google still accepts it.

mod google_support;

use std::sync::Arc;

use google_support::*;
use serde_json::{Value, json};
use suprnova_ai::{
    Error, Model, ProviderConfig, ProviderRegistry, StreamOptions,
    errors::ProviderErrorKind,
    models::params::*,
    providers::{anthropic, google, hosts, openai},
    transport::HttpTransport,
    types::{
        Event, Message, MessageRole, Part, Tool, ToolChoice, ToolRef,
        messages::{
            ContentOutput, ContentPart, FilePart, ReasoningPart, TextPart, ToolCallPart,
            ToolResultPart,
        },
        tools::ToolSpec,
    },
    user_message,
};

fn key() -> ProviderConfig {
    ProviderConfig {
        api_key_value: Some("g-key".into()),
        ..Default::default()
    }
}

fn registry(transport: Arc<Recorder>) -> ProviderRegistry {
    let mut registry = ProviderRegistry::new();
    google::register(&mut registry, key(), transport).unwrap();
    registry
}

fn model(id: &str, transport: Arc<Recorder>) -> Model {
    Model::new(id, registry(transport).get_provider("google").unwrap())
}

async fn body(id: &str, messages: Vec<Message>, options: StreamOptions) -> Value {
    let transport = Recorder::new([text_reply("ok")]);
    let (_, events) = call(&model(id, transport.clone()), messages, options).await;
    ok(&events);
    transport.body()
}

async fn rejected(id: &str, messages: Vec<Message>, options: StreamOptions) -> Error {
    let transport = Recorder::new([text_reply("ok")]);
    let (_, events) = call(&model(id, transport.clone()), messages, options).await;
    assert_eq!(transport.count(), 0, "a rejected call sent a request");
    match error(&events) {
        Error::Validation(message) => Error::Validation(message.clone()),
        other => panic!("expected a validation error, got {other:?}"),
    }
}

fn hi() -> Vec<Message> {
    vec![user_message("hi")]
}

fn tool(name: &str) -> Tool {
    Tool::function(
        name,
        ToolSpec {
            description: Some(format!("The {name} tool")),
            params: json!({"type":"object","properties":{"q":{"type":"string"}}})
                .try_into()
                .unwrap(),
        },
    )
}

fn with_tools(tools: Vec<Tool>, choice: Option<ToolChoice>) -> StreamOptions {
    StreamOptions {
        tools,
        tool_choice: choice,
        ..Default::default()
    }
}

fn sampling(update: impl FnOnce(&mut SamplerParams)) -> StreamOptions {
    params(|p| {
        let mut sampling = SamplerParams::default();
        update(&mut sampling);
        p.sampling = ProviderValue::Value(sampling);
    })
}

fn events_of(events: &[Result<Event, Error>]) -> Vec<&Event> {
    events.iter().map(|e| e.as_ref().unwrap()).collect()
}

fn google_meta(handle: &suprnova_ai::models::StreamHandle) -> Value {
    handle
        .message()
        .provider_metadata
        .as_ref()
        .and_then(|m| m.get("google"))
        .cloned()
        .unwrap_or(Value::Null)
}

// GGL-001

#[tokio::test]
async fn ggl_001_streams_from_the_default_url_with_the_key_header() {
    let transport = Recorder::new([text_reply("hi")]);
    let (handle, events) = call(
        &model("gemini-2.5-flash", transport.clone()),
        hi(),
        StreamOptions::default(),
    )
    .await;
    ok(&events);
    assert_eq!(handle.text(), "hi");
    let request = transport.request();
    assert_eq!(request.method, "POST");
    assert_eq!(
        request.url,
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
    );
    assert_eq!(header(&request, "x-goog-api-key").as_deref(), Some("g-key"));
    assert!(header(&request, "authorization").is_none());
}

#[tokio::test]
async fn ggl_001_a_model_id_with_a_slash_is_its_own_path() {
    let transport = Recorder::new([text_reply("hi")]);
    let (_, events) = call(
        &model("tunedModels/t1", transport.clone()),
        hi(),
        StreamOptions::default(),
    )
    .await;
    ok(&events);
    assert_eq!(
        transport.request().url,
        "https://generativelanguage.googleapis.com/v1beta/tunedModels/t1:streamGenerateContent?alt=sse"
    );
}

#[tokio::test]
async fn ggl_001_the_key_comes_from_env_and_a_configured_base_url_is_used() {
    let transport = Recorder::new([text_reply("hi")]);
    let mut registry = ProviderRegistry::new();
    let config = ProviderConfig {
        default_base_url: "https://proxy.example/v1beta".into(),
        env: [(
            "GOOGLE_GENERATIVE_AI_API_KEY".to_owned(),
            "env-key".to_owned(),
        )]
        .into(),
        ..Default::default()
    };
    google::register(&mut registry, config, transport.clone()).unwrap();
    let model = Model::new("gemini-2.5-flash", registry.get_provider("google").unwrap());
    let (_, events) = call(&model, hi(), StreamOptions::default()).await;
    ok(&events);
    let request = transport.request();
    assert!(
        request
            .url
            .starts_with("https://proxy.example/v1beta/models/gemini-2.5-flash:")
    );
    assert_eq!(
        header(&request, "x-goog-api-key").as_deref(),
        Some("env-key")
    );
}

#[test]
fn ggl_001_the_provider_is_named_google_under_both_handles() {
    let registry = registry(Recorder::new([]));
    for handle in ["google", "@ai-sdk/google"] {
        assert_eq!(registry.get_provider(handle).unwrap().name(), "google");
    }
}

#[tokio::test]
async fn ggl_001_a_restored_snapshot_streams_from_the_same_url() {
    let transport = Recorder::new([text_reply("hi")]);
    let registry = registry(transport.clone());
    let model = Model::new("gemini-2.5-flash", registry.get_provider("google").unwrap());
    let snapshot = model.to_value().unwrap();
    assert!(
        !snapshot.to_string().contains("g-key"),
        "a snapshot holds the key"
    );
    let restored = Model::from_value(snapshot, &registry).unwrap();
    let (_, events) = call(&restored, hi(), StreamOptions::default()).await;
    ok(&events);
    assert_eq!(
        transport.request().url,
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
    );
    assert_eq!(transport.header("x-goog-api-key").as_deref(), Some("g-key"));
}

#[test]
fn ggl_001_registers_with_every_other_provider() {
    let transport: Arc<dyn HttpTransport> = Recorder::new([]);
    let mut registry = ProviderRegistry::new();
    openai::register(
        &mut registry,
        Arc::new(openai::OpenAICompatibleProvider::new(
            key(),
            transport.clone(),
        )),
    )
    .unwrap();
    openai::register_factories(&mut registry, transport.clone()).unwrap();
    anthropic::register(&mut registry, key(), transport.clone()).unwrap();
    for register in [
        hosts::cerebras::register,
        hosts::deepinfra::register,
        hosts::deepseek::register,
        hosts::fireworks::register,
        hosts::huggingface::register,
        hosts::minimax::register,
        hosts::mistral::register,
        hosts::moonshotai::register,
        hosts::togetherai::register,
        hosts::xai::register,
        hosts::zai::register,
    ] {
        register(&mut registry, key(), transport.clone()).unwrap();
    }
    google::register(&mut registry, key(), transport.clone()).unwrap();
    // VERTEX-REGISTER
}

// GGL-002

#[tokio::test]
async fn ggl_002_sampling_settings_go_in_generation_config() {
    let mut options = sampling(|s| {
        s.temperature = Some(TemperatureSamplerParams {
            temperature: ProviderValue::Value(0.5),
        });
        s.top_p = Some(TopPSamplerParams {
            top_p: ProviderValue::Value(Some(0.9)),
        });
        s.top_k = Some(TopKSamplerParams {
            top_k: ProviderValue::Value(Some(40)),
        });
        s.seed = Some(SeedSamplerParams {
            seed: ProviderValue::Value(Some(Seed::Value(7))),
        });
        s.repetition_penalty = Some(RepetitionPenaltyParams {
            frequency_penalty: ProviderValue::Value(Some(0.1)),
            presence_penalty: ProviderValue::Value(Some(0.2)),
            ..Default::default()
        });
    });
    let p = options.params.as_mut().unwrap();
    p.output = Some(OutputParams {
        max_tokens: Some(100),
        ..Default::default()
    });
    p.provider_service = Some(ProviderServiceParams {
        service_tier: ProviderValue::Value("flex".into()),
    });
    let body = body("gemini-3-flash-preview", hi(), options).await;
    let config = &body["generationConfig"];
    assert_eq!(config["maxOutputTokens"], 100);
    assert_eq!(config["temperature"], 0.5);
    assert_eq!(config["topP"], 0.9);
    assert_eq!(config["topK"], 40);
    assert_eq!(config["seed"], 7);
    assert_eq!(config["frequencyPenalty"], 0.1);
    assert_eq!(config["presencePenalty"], 0.2);
    assert_eq!(body["serviceTier"], "flex");
}

#[tokio::test]
async fn ggl_002_gemini_2_5_penalties_and_unsent_settings_fail_before_a_request() {
    let penalty = sampling(|s| {
        s.repetition_penalty = Some(RepetitionPenaltyParams {
            frequency_penalty: ProviderValue::Value(Some(0.1)),
            ..Default::default()
        });
    });
    rejected("gemini-2.5-flash", hi(), penalty).await;
    let presence = sampling(|s| {
        s.repetition_penalty = Some(RepetitionPenaltyParams {
            presence_penalty: ProviderValue::Value(Some(0.1)),
            ..Default::default()
        });
    });
    rejected("models/gemini-2.5-pro", hi(), presence).await;
    let cases: Vec<StreamOptions> = vec![
        sampling(|s| {
            s.min_p = Some(MinPSamplerParams {
                min_p: ProviderValue::Value(Some(0.1)),
            })
        }),
        sampling(|s| {
            s.repetition_penalty = Some(RepetitionPenaltyParams {
                repetition_penalty: ProviderValue::Value(Some(1.1)),
                ..Default::default()
            })
        }),
        params(|p| {
            p.output = Some(OutputParams {
                text_verbosity: ProviderValue::Value(Some("low".into())),
                ..Default::default()
            })
        }),
        params(|p| {
            p.output = Some(OutputParams {
                include: Some(["a".to_owned()].into()),
                ..Default::default()
            })
        }),
        params(|p| {
            p.tool_calling = Some(ToolCallingParams {
                max_tool_calls: ProviderValue::Value(Some(2)),
                parallel_tool_calls: ProviderValue::Default,
                tool_choice: ToolChoice::Auto,
            })
        }),
        params(|p| {
            p.tool_calling = Some(ToolCallingParams {
                max_tool_calls: ProviderValue::Default,
                parallel_tool_calls: ProviderValue::Value(false),
                tool_choice: ToolChoice::Auto,
            })
        }),
        params(|p| p.safety_identifier = Some("u1".into())),
        params(|p| p.metadata = Some([("a".to_owned(), "b".to_owned())].into())),
        params(|p| {
            p.cache = Some(CacheParams {
                key: Some("k".into()),
                ..Default::default()
            })
        }),
        params(|p| {
            p.cache = Some(CacheParams {
                retention: ProviderValue::Value("24h".into()),
                ..Default::default()
            })
        }),
        params(|p| {
            p.context_management = Some(ContextManagementParams {
                compaction: Some(TokenThreshold(1000)),
            })
        }),
        StreamOptions {
            tools: vec![tool("a")],
            tool_choice: Some(ToolChoice::Selection(
                suprnova_ai::types::tools::ToolSelection::new(
                    ["a"],
                    suprnova_ai::types::ToolChoiceMode::Auto,
                ),
            )),
            ..Default::default()
        },
    ];
    for options in cases {
        rejected("gemini-3-flash-preview", hi(), options).await;
    }
}

// GGL-003

#[tokio::test]
async fn ggl_003_gemini_3_efforts_become_thinking_levels() {
    for (effort_value, level) in [
        (Some("medium"), "medium"),
        (Some("xhigh"), "high"),
        (Some("max"), "high"),
        (Some("minimal"), "minimal"),
        (None, "minimal"),
    ] {
        let body = body("gemini-3-flash-preview", hi(), effort(effort_value)).await;
        assert_eq!(
            body["generationConfig"]["thinkingConfig"],
            json!({"thinkingLevel": level}),
            "{effort_value:?}"
        );
    }
    let body = body("gemini-3-pro-preview", hi(), effort(None)).await;
    assert_eq!(
        body["generationConfig"]["thinkingConfig"],
        json!({"thinkingLevel": "low"})
    );
}

#[tokio::test]
async fn ggl_003_older_models_get_a_capped_budget() {
    for (model, effort_value, budget) in [
        ("gemini-2.5-flash", None, 0),
        ("gemini-2.5-flash", Some("low"), 6554),
        ("gemini-2.5-flash", Some("high"), 24576),
        ("gemini-2.5-pro", Some("max"), 32768),
        ("gemini-2.5-flash", Some("minimal"), 1311),
    ] {
        let body = body(model, hi(), effort(effort_value)).await;
        assert_eq!(
            body["generationConfig"]["thinkingConfig"],
            json!({"thinkingBudget": budget}),
            "{model} {effort_value:?}"
        );
    }
}

#[tokio::test]
async fn ggl_003_a_summary_includes_thoughts_and_the_option_wins() {
    let mut options = effort(Some("high"));
    options.params.as_mut().unwrap().output = Some(OutputParams {
        reasoning_summary: ProviderValue::Value(Some("auto".into())),
        ..Default::default()
    });
    let body1 = body("gemini-3-flash-preview", hi(), options).await;
    assert_eq!(
        body1["generationConfig"]["thinkingConfig"]["includeThoughts"],
        true
    );
    let mut options = google_options("google", json!({"thinkingConfig": {"thinkingBudget": 100}}));
    options.params.as_mut().unwrap().reasoning = ProviderValue::Value(ReasoningParams {
        effort: ProviderValue::Value(Some("high".into())),
    });
    let body2 = body("gemini-2.5-flash", hi(), options).await;
    assert_eq!(
        body2["generationConfig"]["thinkingConfig"],
        json!({"thinkingBudget": 100})
    );
}

// GGL-004

#[tokio::test]
async fn ggl_004_an_output_schema_goes_out_with_const_as_enum() {
    let schema =
        json!({"type":"object","properties":{"kind":{"const":"a"},"n":{"type":"integer"}}});
    let options = StreamOptions {
        output_schema: Some(schema.try_into().unwrap()),
        ..Default::default()
    };
    let body1 = body("gemini-3-flash-preview", hi(), options.clone()).await;
    assert_eq!(
        body1["generationConfig"]["responseMimeType"],
        "application/json"
    );
    assert_eq!(
        body1["generationConfig"]["responseJsonSchema"]["properties"]["kind"],
        json!({"enum": ["a"]})
    );
    let mut off = options;
    off.params = Some(
        InferenceRequestParams::default()
            .with_provider_param("google", json!({"structuredOutputs": false})),
    );
    let body2 = body("gemini-3-flash-preview", hi(), off).await;
    assert_eq!(
        body2["generationConfig"]["responseMimeType"],
        "application/json"
    );
    assert!(
        body2["generationConfig"]
            .get("responseJsonSchema")
            .is_none()
    );
}

#[tokio::test]
async fn ggl_004_older_models_get_the_schema_as_a_json_tool_whose_call_is_text() {
    let schema = json!({"type":"object","properties":{"value":{"type":"integer"}}});
    let options = StreamOptions {
        tools: vec![tool("lookup")],
        output_schema: Some(schema.try_into().unwrap()),
        ..Default::default()
    };
    let transport = Recorder::new([Reply::sse(&[
        json!({"candidates":[{"content":{"role":"model","parts":[{"text":"chatter"},{"functionCall":{"name":"json","args":{"value":1}}}]},"finishReason":"STOP"}]}),
    ])]);
    let (handle, events) = call(&model("gemini-2.5-flash", transport.clone()), hi(), options).await;
    ok(&events);
    let body = transport.body();
    assert!(body["generationConfig"].get("responseJsonSchema").is_none());
    let names: Vec<&str> = body["tools"][0]["functionDeclarations"]
        .as_array()
        .unwrap()
        .iter()
        .map(|d| d["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, ["lookup", "json"]);
    assert_eq!(
        body["tools"][0]["functionDeclarations"][1]["description"],
        "Respond with a JSON object."
    );
    assert_eq!(body["toolConfig"]["functionCallingConfig"]["mode"], "ANY");
    assert!(handle.tool_calls().is_empty());
    assert_eq!(handle.text(), r#"{"value":1}"#);
    assert_eq!(handle.finish_reason(), Some("stop"));
}

// GGL-005

fn signed(signature: &str) -> Option<suprnova_ai::types::messages::Metadata> {
    Some([("google".to_owned(), json!({"thoughtSignature": signature}))].into())
}

fn assistant(parts: Vec<Part>) -> Message {
    Message::new(MessageRole::Assistant, parts)
}

fn history() -> Vec<Message> {
    let mut reasoning = ReasoningPart::new("thinking");
    reasoning.provider_metadata = signed("s1");
    let mut call = ToolCallPart::new("c1", "lookup", r#"{"q":"x"}"#);
    call.provider_metadata = signed("s2");
    vec![
        suprnova_ai::system_message("be brief"),
        user_message("hi"),
        assistant(vec![
            Part::Reasoning(reasoning),
            Part::Text(TextPart::new("text")),
            Part::ToolCall(call),
        ]),
        Message::new(
            MessageRole::Tool,
            vec![Part::ToolResult(ToolResultPart::new(
                "c1",
                "lookup",
                json!({"v": 1}),
            ))],
        ),
    ]
}

#[tokio::test]
async fn ggl_005_history_goes_out_in_gemini_shapes() {
    let body = body("gemini-2.5-flash", history(), StreamOptions::default()).await;
    assert_eq!(
        body["systemInstruction"],
        json!({"parts":[{"text":"be brief"}]})
    );
    assert_eq!(
        body["contents"][0],
        json!({"role":"user","parts":[{"text":"hi"}]})
    );
    assert_eq!(
        body["contents"][1],
        json!({"role":"model","parts":[
            {"text":"thinking","thought":true,"thoughtSignature":"s1"},
            {"text":"text"},
            {"functionCall":{"name":"lookup","args":{"q":"x"},"id":"c1"},"thoughtSignature":"s2"}
        ]})
    );
    assert_eq!(
        body["contents"][2],
        json!({"role":"user","parts":[{"functionResponse":{"name":"lookup","response":{"name":"lookup","content":{"v":1}},"id":"c1"}}]})
    );
}

#[tokio::test]
async fn ggl_005_a_late_system_message_fails_and_gemma_inlines_it() {
    let late = vec![user_message("hi"), suprnova_ai::system_message("late")];
    rejected("gemini-2.5-flash", late, StreamOptions::default()).await;
    let body = body(
        "gemma-3-27b-it",
        vec![suprnova_ai::system_message("sys"), user_message("hi")],
        StreamOptions::default(),
    )
    .await;
    assert!(body.get("systemInstruction").is_none());
    assert_eq!(body["contents"][0]["parts"][0], json!({"text": "sys\n\n"}));
}

#[tokio::test]
async fn ggl_005_gemini_3_unsigned_calls_carry_the_sentinel_and_errors_go_as_error() {
    let unsigned = ToolCallPart::new("c1", "lookup", "{}");
    let later = ToolCallPart::new("c2", "lookup", "{}");
    let mut signed_call = ToolCallPart::new("c3", "lookup", "{}");
    signed_call.provider_metadata = signed("s9");
    let messages = vec![
        user_message("hi"),
        assistant(vec![Part::ToolCall(unsigned)]),
        Message::new(
            MessageRole::Tool,
            vec![Part::ToolResult(ToolResultPart::error(
                "c1",
                "lookup",
                json!("boom"),
            ))],
        ),
        assistant(vec![Part::ToolCall(signed_call), Part::ToolCall(later)]),
    ];
    let body = body("gemini-3-flash-preview", messages, StreamOptions::default()).await;
    assert_eq!(
        body["contents"][1]["parts"][0]["thoughtSignature"],
        "skip_thought_signature_validator"
    );
    assert_eq!(
        body["contents"][2]["parts"][0]["functionResponse"]["response"],
        json!({"name":"lookup","error":"boom"})
    );
    let parts = body["contents"][3]["parts"].as_array().unwrap();
    assert_eq!(parts[0]["thoughtSignature"], "s9");
    assert!(
        parts[1].get("thoughtSignature").is_none(),
        "a later call of a signed response got the sentinel"
    );
}

#[tokio::test]
async fn ggl_005_content_results_follow_the_model_generation() {
    let image = FilePart::from_bytes(
        vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a],
        Some("image/png"),
        None,
    )
    .unwrap();
    let content = ContentOutput::new(vec![
        ContentPart::Text(TextPart::new("one")),
        ContentPart::Text(TextPart::new("two")),
        ContentPart::File(image),
    ]);
    let messages = vec![
        user_message("hi"),
        assistant(vec![Part::ToolCall(ToolCallPart::new("c1", "shot", "{}"))]),
        Message::new(
            MessageRole::Tool,
            vec![Part::ToolResult(ToolResultPart::new("c1", "shot", content))],
        ),
    ];
    let gemini_3 = body(
        "gemini-3-flash-preview",
        messages.clone(),
        StreamOptions::default(),
    )
    .await;
    let response = &gemini_3["contents"][2]["parts"][0]["functionResponse"];
    assert_eq!(response["response"]["content"], "one\ntwo");
    assert_eq!(response["parts"][0]["inlineData"]["mimeType"], "image/png");
    let older = body("gemini-2.5-flash", messages, StreamOptions::default()).await;
    let parts = older["contents"][2]["parts"].as_array().unwrap();
    assert_eq!(parts.len(), 4);
    assert_eq!(parts[0]["functionResponse"]["response"]["content"], "one");
    assert_eq!(parts[1]["functionResponse"]["response"]["content"], "two");
    assert_eq!(parts[2]["inlineData"]["mimeType"], "image/png");
    assert_eq!(
        parts[3]["text"],
        "Tool executed successfully and returned this image as a response"
    );
}

// GGL-006

fn user_file(file: FilePart) -> Vec<Message> {
    vec![Message::new(MessageRole::User, vec![Part::File(file)])]
}

#[tokio::test]
async fn ggl_006_bytes_go_inline_and_fetchable_urls_go_as_file_data() {
    let png = FilePart::from_bytes(vec![0x89, b'P', b'N', b'G'], Some("image/png"), None).unwrap();
    let body1 = body("gemini-2.5-flash", user_file(png), StreamOptions::default()).await;
    assert_eq!(
        body1["contents"][0]["parts"][0],
        json!({"inlineData":{"mimeType":"image/png","data":"iVBORw=="}})
    );
    for url in [
        "https://example.com/report.pdf",
        "https://generativelanguage.googleapis.com/v1beta/files/abc",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    ] {
        let media_type = if url.ends_with(".pdf") {
            "application/pdf"
        } else {
            "video/mp4"
        };
        let file = FilePart::from_url(url, Some(media_type)).unwrap();
        let body = body(
            "gemini-2.5-flash",
            user_file(file),
            StreamOptions::default(),
        )
        .await;
        assert_eq!(
            body["contents"][0]["parts"][0],
            json!({"fileData":{"mimeType":media_type,"fileUri":url}}),
            "{url}"
        );
    }
}

#[tokio::test]
async fn ggl_006_other_urls_need_a_download_transport() {
    let pdf =
        FilePart::from_url("https://example.com/report.pdf", Some("application/pdf")).unwrap();
    rejected("gemini-2.0-flash", user_file(pdf), StreamOptions::default()).await;
    let image = FilePart::from_url("http://example.com/cat.png", Some("image/png")).unwrap();
    rejected(
        "gemini-2.5-flash",
        user_file(image),
        StreamOptions::default(),
    )
    .await;
}

// GGL-007

#[tokio::test]
async fn ggl_007_function_tools_and_each_tool_choice() {
    let tools = vec![tool("a"), tool("b")];
    for (choice, config) in [
        (
            Some(ToolChoice::Auto),
            json!({"functionCallingConfig":{"mode":"AUTO"}}),
        ),
        (
            Some(ToolChoice::None),
            json!({"functionCallingConfig":{"mode":"NONE"}}),
        ),
        (
            Some(ToolChoice::Required),
            json!({"functionCallingConfig":{"mode":"ANY"}}),
        ),
        (
            Some(ToolChoice::Tool(ToolRef::new("b"))),
            json!({"functionCallingConfig":{"mode":"ANY","allowedFunctionNames":["b"]}}),
        ),
    ] {
        let body = body(
            "gemini-2.5-flash",
            hi(),
            with_tools(tools.clone(), choice.clone()),
        )
        .await;
        assert_eq!(body["toolConfig"], config, "{choice:?}");
        assert_eq!(
            body["tools"][0]["functionDeclarations"][0],
            json!({"name":"a","description":"The a tool","parametersJsonSchema":{"type":"object","properties":{"q":{"type":"string"}}}})
        );
    }
    let strict = with_tools(vec![tool("a").with_strict(true)], None);
    let body = body("gemini-2.5-flash", hi(), strict).await;
    assert_eq!(
        body["toolConfig"],
        json!({"functionCallingConfig":{"mode":"VALIDATED"}})
    );
}

#[tokio::test]
async fn ggl_007_built_in_tools_and_their_combination() {
    let search = google::tools::google_search(json!({"timeRangeFilter":{"startTime":"2026-01-01T00:00:00Z","endTime":"2026-02-01T00:00:00Z"}})).unwrap();
    let body1 = body(
        "gemini-2.5-flash",
        hi(),
        with_tools(vec![search.clone()], None),
    )
    .await;
    assert_eq!(
        body1["tools"][0],
        json!({"googleSearch":{"timeRangeFilter":{"startTime":"2026-01-01T00:00:00Z","endTime":"2026-02-01T00:00:00Z"}}})
    );
    let body2 = body(
        "gemini-3-flash-preview",
        hi(),
        with_tools(vec![search.clone(), tool("a")], None),
    )
    .await;
    assert!(body2["tools"][0].get("googleSearch").is_some());
    assert_eq!(body2["tools"][1]["functionDeclarations"][0]["name"], "a");
    assert_eq!(
        body2["toolConfig"],
        json!({"functionCallingConfig":{"mode":"VALIDATED"},"includeServerSideToolInvocations":true})
    );
    for (model_id, tools) in [
        (
            "gemini-2.0-flash",
            vec![
                google::tools::file_search(json!({"fileSearchStoreNames":["fileSearchStores/s"]}))
                    .unwrap(),
            ],
        ),
        (
            "gemini-2.5-flash",
            vec![google::tools::vertex_rag_store(json!({"ragCorpus":"c"})).unwrap()],
        ),
        (
            "gemini-2.5-flash",
            vec![google::tools::enterprise_web_search()],
        ),
        ("gemini-2.5-flash", vec![search.clone(), tool("a")]),
    ] {
        rejected(model_id, hi(), with_tools(tools, None)).await;
    }
    let body3 = body(
        "gemini-2.5-flash",
        hi(),
        with_tools(
            vec![
                google::tools::url_context(),
                google::tools::code_execution(),
                google::tools::google_maps(),
            ],
            None,
        ),
    )
    .await;
    assert_eq!(
        body3["tools"],
        json!([{"urlContext":{}},{"codeExecution":{}},{"googleMaps":{}}])
    );
}

// GGL-008

#[tokio::test]
async fn ggl_008_parts_become_blocks_files_and_tool_calls() {
    let transport = Recorder::new([Reply::sse(&[
        json!({"candidates":[{"content":{"parts":[{"text":"A","thought":true}]}}]}),
        json!({"candidates":[{"content":{"parts":[{"text":"B"},{"text":"","thoughtSignature":"S"}]}}]}),
        json!({"candidates":[{"content":{"parts":[{"inlineData":{"mimeType":"image/png","data":"iVBORw=="}},{"functionCall":{"name":"lookup","args":{"q":"x"}}}]},"finishReason":"STOP"}]}),
    ])]);
    let (handle, events) = call(
        &model("gemini-3-flash-preview", transport),
        hi(),
        StreamOptions::default(),
    )
    .await;
    ok(&events);
    let message = handle.message();
    assert_eq!(message.reasoning(), "A");
    let text = message
        .parts
        .iter()
        .find_map(|p| match p {
            Part::Text(t) => Some(t),
            _ => None,
        })
        .unwrap();
    assert_eq!(text.text, "B");
    assert_eq!(
        text.provider_metadata.as_ref().unwrap()["google"]["thoughtSignature"],
        "S"
    );
    assert_eq!(message.images().len(), 1);
    let calls = handle.tool_calls();
    assert_eq!(calls.len(), 1);
    assert!(!calls[0].tool_call_id.is_empty());
    assert_eq!(calls[0].tool_args, r#"{"q":"x"}"#);
    assert_eq!(handle.finish_reason(), Some("tool_call"));
}

#[tokio::test]
async fn ggl_008_code_execution_parts_are_a_built_in_call_and_result() {
    let transport = Recorder::new([Reply::sse(&[
        json!({"candidates":[{"content":{"parts":[{"executableCode":{"language":"PYTHON","code":"print(1)"}},{"codeExecutionResult":{"outcome":"OUTCOME_OK","output":"1\n"}}]},"finishReason":"STOP"}]}),
    ])]);
    let (handle, events) = call(
        &model("gemini-2.5-flash", transport),
        hi(),
        StreamOptions::default(),
    )
    .await;
    ok(&events);
    let message = handle.message();
    let calls = message.builtin_tool_calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].tool_name, "code_execution");
    assert_eq!(
        serde_json::from_str::<Value>(&calls[0].tool_args).unwrap(),
        json!({"language":"PYTHON","code":"print(1)"})
    );
    let results = message.builtin_tool_returns();
    assert_eq!(results[0].tool_call_id, calls[0].tool_call_id);
    assert_eq!(
        results[0].result,
        json!({"outcome":"OUTCOME_OK","output":"1\n"})
    );
}

#[tokio::test]
async fn ggl_008_grounding_sources_are_kept_once_per_url() {
    let grounding = json!({"groundingChunks":[{"web":{"uri":"https://a.example","title":"A"}},{"web":{"uri":"https://a.example","title":"A again"}}]});
    let transport = Recorder::new([Reply::sse(&[
        json!({"candidates":[{"content":{"parts":[{"text":"x"}]},"groundingMetadata":grounding}]}),
        json!({"candidates":[{"content":{"parts":[]},"groundingMetadata":grounding,"finishReason":"STOP"}]}),
    ])]);
    let (handle, events) = call(
        &model("gemini-2.5-flash", transport),
        hi(),
        StreamOptions::default(),
    )
    .await;
    ok(&events);
    let sources = google_meta(&handle)["sources"].as_array().cloned().unwrap();
    assert_eq!(sources.len(), 1);
    assert_eq!(sources[0]["url"], "https://a.example");
    assert_eq!(sources[0]["sourceType"], "url");
}

#[tokio::test]
async fn ggl_008_a_continuation_sends_the_token_and_continues_the_text() {
    let transport = Recorder::new([
        Reply::sse(&[
            json!({"candidates":[{"content":{"parts":[{"text":"first "}]},"finishReason":"CONTINUATION","continuationToken":"t1"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5}}),
        ]),
        Reply::sse(&[
            json!({"candidates":[{"content":{"parts":[{"text":"second"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":40,"candidatesTokenCount":3}}),
        ]),
    ]);
    let (handle, events) = call(
        &model("gemini-4-argon", transport.clone()),
        hi(),
        StreamOptions::default(),
    )
    .await;
    ok(&events);
    assert_eq!(handle.text(), "first second");
    assert_eq!(transport.count(), 2);
    assert!(transport.nth_body(0).get("continuationToken").is_none());
    assert_eq!(transport.nth_body(1)["continuationToken"], "t1");
    assert_eq!(handle.finish_reason(), Some("stop"));
}

#[tokio::test]
async fn ggl_008_a_stream_without_a_finish_reason_fails_as_incomplete() {
    let transport = Recorder::new([Reply::sse(&[
        json!({"candidates":[{"content":{"parts":[{"text":"B"}]}}]}),
    ])]);
    let (_, events) = call(
        &model("gemini-2.5-flash", transport),
        hi(),
        StreamOptions::default(),
    )
    .await;
    match error(&events) {
        Error::Provider(error) => assert_eq!(error.kind, ProviderErrorKind::IncompleteResponse),
        other => panic!("expected an incomplete response, got {other:?}"),
    }
}

// GGL-009

async fn finish_of(raw: &str, with_call: bool) -> Option<String> {
    let mut parts = vec![json!({"text":"x"})];
    if with_call {
        parts.push(json!({"functionCall":{"name":"lookup","args":{}}}));
    }
    let transport = Recorder::new([Reply::sse(&[
        json!({"candidates":[{"content":{"parts":parts},"finishReason":raw}]}),
    ])]);
    let (handle, events) = call(
        &model("gemini-2.5-flash", transport),
        hi(),
        StreamOptions::default(),
    )
    .await;
    ok(&events);
    handle.finish_reason().map(str::to_owned)
}

#[tokio::test]
async fn ggl_009_finish_reasons_map_to_the_normalized_names() {
    for (raw, with_call, expected) in [
        ("STOP", false, "stop"),
        ("STOP", true, "tool_call"),
        ("MAX_TOKENS", false, "length"),
        ("IMAGE_SAFETY", false, "content_filter"),
        ("RECITATION", false, "content_filter"),
        ("SAFETY", false, "content_filter"),
        ("BLOCKLIST", false, "content_filter"),
        ("PROHIBITED_CONTENT", false, "content_filter"),
        ("SPII", false, "content_filter"),
        ("MALFORMED_FUNCTION_CALL", false, "error"),
        ("OTHER", false, "other"),
        ("FINISH_REASON_UNSPECIFIED", true, "other"),
    ] {
        assert_eq!(
            finish_of(raw, with_call).await.as_deref(),
            Some(expected),
            "{raw} {with_call}"
        );
    }
}

#[tokio::test]
async fn ggl_009_a_blocked_prompt_reports_content_filter() {
    let transport = Recorder::new([Reply::sse(&[
        json!({"promptFeedback":{"blockReason":"SAFETY"}}),
    ])]);
    let (handle, events) = call(
        &model("gemini-2.5-flash", transport),
        hi(),
        StreamOptions::default(),
    )
    .await;
    ok(&events);
    assert_eq!(handle.finish_reason(), Some("content_filter"));
}

#[tokio::test]
async fn ggl_009_usage_and_final_metadata() {
    let usage = json!({"promptTokenCount":9,"toolUsePromptTokenCount":2,"cachedContentTokenCount":4,"candidatesTokenCount":23,"thoughtsTokenCount":185});
    let grounding = json!({"webSearchQueries":["q"]});
    let transport = Recorder::new([Reply::sse(&[
        json!({"candidates":[{"content":{"parts":[{"text":"x"}]},"groundingMetadata":grounding,"finishReason":"STOP"}],"usageMetadata":usage}),
    ])
    .header("x-gemini-service-tier", "flex")]);
    let (handle, events) = call(
        &model("gemini-2.5-flash", transport),
        hi(),
        StreamOptions::default(),
    )
    .await;
    ok(&events);
    let usage = handle.usage().unwrap();
    assert_eq!(
        (
            usage.input_tokens,
            usage.cache_read_tokens,
            usage.output_tokens,
            usage.reasoning_tokens
        ),
        (11, Some(4), 208, Some(185))
    );
    let meta = google_meta(&handle);
    assert_eq!(meta["groundingMetadata"], grounding);
    assert_eq!(meta["serviceTier"], "flex");
    assert!(meta.get("initialPromptTokenCount").is_none());
    assert!(meta.get("groundingMetadataPerRequest").is_none());
}

#[tokio::test]
async fn ggl_009_a_continued_stream_keeps_the_first_prompt_count_and_each_grounding() {
    let grounding = json!({"webSearchQueries":["q"]});
    let transport = Recorder::new([
        Reply::sse(&[
            json!({"candidates":[{"content":{"parts":[{"text":"a"}]},"groundingMetadata":grounding,"finishReason":"CONTINUATION","continuationToken":"t1"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5}}),
        ]),
        Reply::sse(&[
            json!({"candidates":[{"content":{"parts":[{"text":"b"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":40,"candidatesTokenCount":3}}),
        ]),
    ]);
    let (handle, events) = call(
        &model("gemini-4-argon", transport),
        hi(),
        StreamOptions::default(),
    )
    .await;
    ok(&events);
    let meta = google_meta(&handle);
    assert_eq!(meta["initialPromptTokenCount"], 10);
    assert_eq!(
        meta["groundingMetadataPerRequest"],
        json!([grounding, null])
    );
    let usage = handle.usage().unwrap();
    assert_eq!((usage.input_tokens, usage.output_tokens), (50, 8));
}

// GGL-010

#[tokio::test]
async fn ggl_010_error_responses_and_stream_errors_keep_their_fields() {
    let transport = Recorder::new([Reply::status(
        429,
        json!({"error":{"code":429,"message":"m","status":"RESOURCE_EXHAUSTED"}}),
    )]);
    let model429 = model("gemini-2.5-flash", transport);
    let mut options = StreamOptions::default();
    options.retry = suprnova_ai::RetryPolicy::none();
    let (_, events) = call(&model429, hi(), options.clone()).await;
    let Error::Provider(failure) = error(&events) else {
        panic!("not a provider error")
    };
    let error = failure;
    assert_eq!(error.message, "m");
    assert_eq!(error.code.as_deref(), Some("429"));
    assert_eq!(error.error_type.as_deref(), Some("RESOURCE_EXHAUSTED"));
    assert_eq!(error.kind, ProviderErrorKind::RateLimit);
    let transport = Recorder::new([Reply::sse(&[
        json!({"candidates":[{"content":{"parts":[{"text":"x"}]}}]}),
        json!({"error":{"code":503,"message":"busy","status":"UNAVAILABLE"}}),
    ])]);
    let (_, events) = call(&model("gemini-2.5-flash", transport), hi(), options).await;
    let Error::Provider(failure) = google_support::error(&events) else {
        panic!("not a provider error")
    };
    let error = failure;
    assert_eq!(error.message, "busy");
    assert_eq!(error.kind, ProviderErrorKind::ServiceUnavailable);
    assert!(error.is_retryable);
    assert_eq!(error.error_type.as_deref(), Some("UNAVAILABLE"));
}

// GGL-011

#[tokio::test]
async fn ggl_011_provider_options_go_where_typescript_sends_them() {
    let options = google_options(
        "google",
        json!({
            "responseModalities": ["TEXT", "IMAGE"],
            "mediaResolution": "MEDIA_RESOLUTION_LOW",
            "audioTimestamp": true,
            "imageConfig": {"aspectRatio": "16:9", "imageSize": "2K"},
            "cachedContent": "cachedContents/c1",
            "labels": {"team": "a"},
            "serviceTier": "priority",
            "continuationToken": "t0",
            "retrievalConfig": {"latLng": {"latitude": 1.5, "longitude": 2.5}},
            "safetySettings": [{"category":"HARM_CATEGORY_HARASSMENT","threshold":"BLOCK_NONE"}]
        }),
    );
    let body1 = body("gemini-2.5-flash", hi(), options).await;
    let config = &body1["generationConfig"];
    assert_eq!(config["responseModalities"], json!(["TEXT", "IMAGE"]));
    assert_eq!(config["mediaResolution"], "MEDIA_RESOLUTION_LOW");
    assert_eq!(config["audioTimestamp"], true);
    assert_eq!(
        config["imageConfig"],
        json!({"aspectRatio":"16:9","imageSize":"2K"})
    );
    assert_eq!(body1["cachedContent"], "cachedContents/c1");
    assert_eq!(body1["labels"], json!({"team":"a"}));
    assert_eq!(body1["serviceTier"], "priority");
    assert_eq!(body1["continuationToken"], "t0");
    assert_eq!(
        body1["toolConfig"]["retrievalConfig"],
        json!({"latLng":{"latitude":1.5,"longitude":2.5}})
    );
    assert_eq!(
        body1["safetySettings"],
        json!([{"category":"HARM_CATEGORY_HARASSMENT","threshold":"BLOCK_NONE"}])
    );
    let body2 = body(
        "gemini-2.5-flash",
        hi(),
        google_options("google", json!({"threshold":"BLOCK_ONLY_HIGH"})),
    )
    .await;
    let categories: Vec<&str> = body2["safetySettings"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["category"].as_str().unwrap())
        .collect();
    assert_eq!(
        categories,
        [
            "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_DANGEROUS_CONTENT",
            "HARM_CATEGORY_HARASSMENT",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT"
        ]
    );
    assert!(
        body2["safetySettings"]
            .as_array()
            .unwrap()
            .iter()
            .all(|s| s["threshold"] == "BLOCK_ONLY_HIGH")
    );
}

#[tokio::test]
async fn ggl_011_vertex_options_and_unknown_names_fail_on_the_gemini_api() {
    for option in [
        json!({"streamFunctionCallArguments": true}),
        json!({"sharedRequestType": "flex"}),
        json!({"requestType": "shared"}),
        json!({"imageConfig": {"personGeneration": "ALLOW_ALL"}}),
        json!({"imageConfig": {"prominentPeople": "ALLOW_PROMINENT_PEOPLE"}}),
        json!({"imageConfig": {"imageOutputOptions": {"mimeType": "image/png"}}}),
        json!({"noSuchOption": 1}),
        json!({"mediaResolution": "HUGE"}),
    ] {
        rejected("gemini-2.5-flash", hi(), google_options("google", option)).await;
    }
}

// GGL-012

#[tokio::test]
async fn ggl_012_a_buffered_call_sends_the_stream_body_and_gives_its_message() {
    let response = json!({
        "candidates":[{"content":{"role":"model","parts":[{"text":"A","thought":true},{"text":"B","thoughtSignature":"S"},{"functionCall":{"name":"lookup","args":{"q":"x"}}}]},"finishReason":"STOP"}],
        "usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":2},
        "responseId":"r1"
    });
    let streamed = Recorder::new([Reply::sse(&[response.clone()])]);
    let (handle, events) = call(
        &model("gemini-2.5-flash", streamed.clone()),
        hi(),
        StreamOptions::default(),
    )
    .await;
    ok(&events);
    let buffered = Recorder::new([Reply::json(response)]);
    let message = suprnova_ai::generate(
        &model("gemini-2.5-flash", buffered.clone()),
        hi(),
        StreamOptions::default(),
    )
    .await
    .unwrap();
    let request = buffered.request();
    assert_eq!(
        request.url,
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    );
    assert_eq!(buffered.body(), streamed.body());
    assert_eq!(message.text(), handle.message().text());
    assert_eq!(message.reasoning(), handle.message().reasoning());
    assert_eq!(message.tool_calls().len(), 1);
    assert_eq!(message.usage, handle.message().usage);
    let strip_ids = |m: &Message| {
        m.parts
            .iter()
            .map(|p| match p {
                Part::Text(t) => json!({"text": t.text, "meta": t.provider_metadata}),
                Part::Reasoning(r) => json!({"reasoning": r.text, "meta": r.provider_metadata}),
                Part::ToolCall(c) => json!({"call": c.tool_name, "args": c.tool_args}),
                other => json!({"other": format!("{other:?}")}),
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(strip_ids(&message), strip_ids(handle.message()));
}

#[tokio::test]
async fn ggl_012_a_buffered_call_continues_a_continuation() {
    let transport = Recorder::new([
        Reply::json(
            json!({"candidates":[{"content":{"parts":[{"text":"a"}]},"finishReason":"CONTINUATION","continuationToken":"t1"}]}),
        ),
        Reply::json(
            json!({"candidates":[{"content":{"parts":[{"text":"b"}]},"finishReason":"STOP"}]}),
        ),
    ]);
    let message = suprnova_ai::generate(
        &model("gemini-4-argon", transport.clone()),
        hi(),
        StreamOptions::default(),
    )
    .await
    .unwrap();
    assert_eq!(message.text(), "ab");
    assert_eq!(transport.nth_body(1)["continuationToken"], "t1");
}
