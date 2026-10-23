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
        Message, MessageRole, Part, Tool, ToolChoice, ToolRef,
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
    suprnova_ai::providers::google_vertex::register(&mut registry, key(), transport.clone())
        .unwrap();
    suprnova_ai::providers::google_vertex::anthropic::register(
        &mut registry,
        key(),
        transport.clone(),
    )
    .unwrap();
    suprnova_ai::providers::google_vertex::maas::register(&mut registry, key(), transport.clone())
        .unwrap();
    suprnova_ai::providers::google_vertex::xai::register(&mut registry, key(), transport.clone())
        .unwrap();
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
    for options in unsent_settings() {
        rejected("gemini-3-flash-preview", hi(), options).await;
    }
}

/// Settings that neither Google API takes: min-p, a repetition penalty,
/// verbosity, an include list, tool call limits, a safety identifier,
/// metadata, cache settings, context management and a tool selection.
fn unsent_settings() -> Vec<StreamOptions> {
    vec![
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
    ]
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
    let options = StreamOptions {
        retry: suprnova_ai::RetryPolicy::none(),
        ..Default::default()
    };
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
    let streamed = Recorder::new([Reply::sse(std::slice::from_ref(&response))]);
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

// GGL-013

use suprnova_ai::{ops, providers::operations::*};

fn embed_params(google: Value) -> EmbedParams {
    EmbedParams {
        provider_options: [("google".to_owned(), google)].into(),
    }
}

#[tokio::test]
async fn ggl_013_one_value_uses_embed_content_and_several_batch_embed_contents() {
    let transport = Recorder::new([Reply::json(json!({"embedding":{"values":[0.1,0.2]}}))]);
    let m = model("gemini-embedding-001", transport.clone());
    let item = ops::embed(
        &m,
        vec!["a".into()],
        embed_params(json!({"outputDimensionality": 2, "taskType": "RETRIEVAL_QUERY"})),
        Default::default(),
    )
    .await
    .unwrap();
    assert_eq!(item.value, vec![vec![0.1, 0.2]]);
    let request = transport.request();
    assert_eq!(
        request.url,
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent"
    );
    assert_eq!(
        transport.body(),
        json!({"model":"models/gemini-embedding-001","content":{"parts":[{"text":"a"}]},"outputDimensionality":2,"taskType":"RETRIEVAL_QUERY"})
    );

    let transport = Recorder::new([Reply::json(
        json!({"embeddings":[{"values":[1.0]},{"values":[2.0]}]}),
    )]);
    let m = model("gemini-embedding-001", transport.clone());
    let image = json!({"inlineData":{"mimeType":"image/png","data":"iVBORw=="}});
    let item = ops::embed(
        &m,
        vec!["a".into(), "".into()],
        embed_params(json!({"content": [null, [image]]})),
        Default::default(),
    )
    .await
    .unwrap();
    assert_eq!(item.value, vec![vec![1.0], vec![2.0]]);
    assert_eq!(
        transport.request().url,
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents"
    );
    assert_eq!(
        transport.body(),
        json!({"requests":[
            {"model":"models/gemini-embedding-001","content":{"role":"user","parts":[{"text":"a"}]}},
            {"model":"models/gemini-embedding-001","content":{"role":"user","parts":[image]}}
        ]})
    );
}

#[tokio::test]
async fn ggl_013_a_content_list_of_another_length_fails_and_runs_hold_100_values() {
    let transport = Recorder::new([]);
    let m = model("gemini-embedding-001", transport.clone());
    let error = ops::embed(
        &m,
        vec!["a".into(), "b".into()],
        embed_params(json!({"content": [null]})),
        Default::default(),
    )
    .await
    .unwrap_err();
    match error {
        Error::Validation(message) => assert!(
            message.contains("(1)") && message.contains("(2)"),
            "{message}"
        ),
        other => panic!("expected a validation error, got {other:?}"),
    }
    assert_eq!(transport.count(), 0);
    let values: Vec<String> = (0..150).map(|i| i.to_string()).collect();
    let reply = |n: usize| Reply::json(json!({"embeddings": vec![json!({"values":[0.0]}); n]}));
    let transport = Recorder::new([reply(100), reply(50)]);
    let m = model("gemini-embedding-001", transport.clone());
    let item = ops::embed(&m, values, EmbedParams::default(), Default::default())
        .await
        .unwrap();
    assert_eq!(item.value.len(), 150);
    let sizes: Vec<usize> = transport
        .all()
        .iter()
        .map(|r| {
            serde_json::from_slice::<Value>(&r.body).unwrap()["requests"]
                .as_array()
                .unwrap()
                .len()
        })
        .collect();
    assert_eq!(sizes.iter().sum::<usize>(), 150);
    assert!(sizes.contains(&100) && sizes.contains(&50));
}

// GGL-014

fn image_response() -> Reply {
    Reply::json(json!({
        "candidates":[{"content":{"parts":[{"text":"here"},{"inlineData":{"mimeType":"image/png","data":"iVBORw=="}}]},"finishReason":"STOP"}],
        "usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":1290}
    }))
}

#[tokio::test]
async fn ggl_014_image_generation_makes_one_generate_content_call() {
    let transport = Recorder::new([image_response()]);
    let m = model("gemini-2.5-flash-image", transport.clone());
    let prompt = ImagePrompt {
        text: Some("a cat".into()),
        images: vec![MediaInput::Bytes(vec![
            0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a,
        ])],
        mask: None,
    };
    let params = ImageParams {
        size: Some("1024x1024".into()),
        aspect_ratio: Some("16:9".into()),
        seed: Some(3),
        provider_options: [(
            "google".to_owned(),
            json!({"imageConfig": {"imageSize": "2K"}, "googleSearch": {}}),
        )]
        .into(),
        ..Default::default()
    };
    let item = ops::generate_image(&m, prompt, params, Default::default())
        .await
        .unwrap();
    let request = transport.request();
    assert_eq!(
        request.url,
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:generateContent"
    );
    let body = transport.body();
    assert_eq!(body["contents"][0]["parts"][0], json!({"text":"a cat"}));
    assert_eq!(
        body["contents"][0]["parts"][1]["inlineData"]["mimeType"],
        "image/png"
    );
    assert_eq!(
        body["generationConfig"]["responseModalities"],
        json!(["IMAGE"])
    );
    assert_eq!(
        body["generationConfig"]["imageConfig"],
        json!({"imageSize":"2K","aspectRatio":"16:9"})
    );
    assert_eq!(body["generationConfig"]["seed"], 3);
    assert_eq!(body["tools"], json!([{"googleSearch":{}}]));
    assert_eq!(item.value.len(), 1);
    assert_eq!(item.value[0].media_type, "image/png");
    assert!(
        item.warnings
            .iter()
            .any(|w| w.kind == "unsupported" && w.feature.as_deref() == Some("size"))
    );
    assert_eq!(item.usage.as_ref().unwrap().output_tokens, 1290);
    let google = &item.provider_metadata.as_ref().unwrap()["google"];
    assert_eq!(google["finishReason"], "STOP");
    assert_eq!(google["images"], json!([{}]));
    assert!(google.get("usageMetadata").is_some());
}

#[tokio::test]
async fn ggl_014_several_images_keep_one_metadata_entry_each() {
    let transport = Recorder::new([image_response(), image_response()]);
    let m = model("gemini-2.5-flash-image", transport.clone());
    let prompt = ImagePrompt {
        text: Some("a cat".into()),
        images: vec![],
        mask: None,
    };
    let params = ImageParams {
        n: 2,
        ..Default::default()
    };
    let item = ops::generate_image(&m, prompt, params, Default::default())
        .await
        .unwrap();
    assert_eq!(transport.count(), 2);
    assert_eq!(item.value.len(), 2);
    assert_eq!(
        item.provider_metadata.as_ref().unwrap()["google"]["images"],
        json!([{}, {}])
    );
}

#[tokio::test]
async fn ggl_014_a_response_modalities_option_is_replaced_by_image() {
    let transport = Recorder::new([image_response()]);
    let prompt = ImagePrompt {
        text: Some("a cat".into()),
        images: vec![],
        mask: None,
    };
    let params = ImageParams {
        provider_options: [(
            "google".to_owned(),
            json!({"responseModalities": ["TEXT", "IMAGE"]}),
        )]
        .into(),
        ..Default::default()
    };
    ops::generate_image(
        &model("gemini-2.5-flash-image", transport.clone()),
        prompt,
        params,
        Default::default(),
    )
    .await
    .unwrap();
    assert_eq!(
        transport.body()["generationConfig"]["responseModalities"],
        json!(["IMAGE"])
    );
}

#[tokio::test]
async fn ggl_014_other_models_and_masks_fail_before_a_request() {
    for (id, mask) in [
        ("imagen-4.0-generate-001", None),
        (
            "gemini-2.5-flash-image",
            Some(MediaInput::Url("https://example.com/m.png".into())),
        ),
    ] {
        let transport = Recorder::new([]);
        let prompt = ImagePrompt {
            text: Some("x".into()),
            images: vec![],
            mask,
        };
        let error = ops::generate_image(
            &model(id, transport.clone()),
            prompt,
            ImageParams::default(),
            Default::default(),
        )
        .await
        .unwrap_err();
        assert!(matches!(error, Error::Validation(_)), "{id}: {error:?}");
        assert_eq!(transport.count(), 0);
    }
}

// GGL-015

fn audio_response(mime: &str, data: &[u8]) -> Reply {
    use base64::Engine as _;
    Reply::json(
        json!({"candidates":[{"content":{"parts":[{"inlineData":{"mimeType":mime,"data":base64::engine::general_purpose::STANDARD.encode(data)}}]}}]}),
    )
}

#[tokio::test]
async fn ggl_015_legacy_models_prefix_instructions_and_wrap_pcm_in_wav() {
    let transport = Recorder::new([audio_response("audio/L16;rate=24000", &[1, 0, 2, 0])]);
    let m = model("gemini-2.5-flash-preview-tts", transport.clone());
    let prompt = AudioPrompt {
        text: "hello".into(),
        instructions: Some("cheerful".into()),
    };
    let params = AudioParams {
        speed: Some(1.2),
        language: Some("en".into()),
        ..Default::default()
    };
    let item = ops::generate_audio(&m, prompt, params, Default::default())
        .await
        .unwrap();
    let body = transport.body();
    assert_eq!(
        transport.request().url,
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-tts:generateContent"
    );
    assert_eq!(
        body["contents"],
        json!([{"role":"user","parts":[{"text":"cheerful: hello"}]}])
    );
    assert_eq!(
        body["generationConfig"],
        json!({"responseModalities":["AUDIO"],"speechConfig":{"voiceConfig":{"prebuiltVoiceConfig":{"voiceName":"Kore"}}}})
    );
    let audio = item.value[0].data.as_bytes().unwrap();
    assert_eq!(audio.len(), 48);
    assert_eq!(&audio[0..4], b"RIFF");
    assert_eq!(u32::from_le_bytes(audio[24..28].try_into().unwrap()), 24000);
    assert_eq!(&audio[44..], &[1, 0, 2, 0]);
    let features: Vec<&str> = item
        .warnings
        .iter()
        .filter_map(|w| w.feature.as_deref())
        .collect();
    assert!(
        features.contains(&"speed") && features.contains(&"language"),
        "{features:?}"
    );
}

#[tokio::test]
async fn ggl_015_structured_models_send_turns_and_the_output_format() {
    let transport = Recorder::new([audio_response("audio/L16;rate=24000", &[1, 0])]);
    let m = model("gemini-3.8-flash-tts", transport.clone());
    let speakers = json!({"speakerVoiceConfigs":[{"speaker":"A","voiceConfig":{"prebuiltVoiceConfig":{"voiceName":"Kore"}}},{"speaker":"B","voiceConfig":{"prebuiltVoiceConfig":{"voiceName":"Puck"}}}]});
    let params = AudioParams {
        output_format: Some("pcm".into()),
        provider_options: [("google".to_owned(), json!({
            "multiSpeakerVoiceConfig": speakers,
            "turns": [{"text":"Hi","speechMetadata":{"speaker":"A"}},{"text":"Hey","speechMetadata":{"speaker":"B","style":"calm"}}]
        }))].into(),
        ..Default::default()
    };
    let prompt = AudioPrompt {
        text: String::new(),
        instructions: Some("warm".into()),
    };
    ops::generate_audio(&m, prompt, params, Default::default())
        .await
        .unwrap();
    let body = transport.body();
    assert_eq!(
        body["contents"][0]["parts"],
        json!([
            {"text":"Hi","speechMetadata":{"style":"warm","speaker":"A"}},
            {"text":"Hey","speechMetadata":{"style":"calm","speaker":"B"}}
        ])
    );
    assert_eq!(
        body["generationConfig"]["responseFormat"],
        json!({"audio":{"mimeType":"AUDIO_L16"}})
    );
    assert_eq!(
        body["generationConfig"]["speechConfig"]["multiSpeakerVoiceConfig"],
        speakers
    );
}

#[tokio::test]
async fn ggl_015_custom_voices_empty_text_and_legacy_turns_fail() {
    let cases = vec![
        (
            "gemini-2.5-flash-preview-tts",
            AudioPrompt {
                text: "x".into(),
                instructions: None,
            },
            AudioParams {
                voice: Some("voice_x".into()),
                ..Default::default()
            },
        ),
        (
            "gemini-2.5-flash-preview-tts",
            AudioPrompt {
                text: String::new(),
                instructions: None,
            },
            AudioParams::default(),
        ),
        (
            "gemini-2.5-flash-preview-tts",
            AudioPrompt {
                text: "x".into(),
                instructions: None,
            },
            AudioParams {
                provider_options: [("google".to_owned(), json!({"turns":[{"text":"a"}]}))].into(),
                ..Default::default()
            },
        ),
    ];
    for (id, prompt, params) in cases {
        let transport = Recorder::new([]);
        let error = ops::generate_audio(
            &model(id, transport.clone()),
            prompt,
            params,
            Default::default(),
        )
        .await
        .unwrap_err();
        assert!(matches!(error, Error::Validation(_)), "{error:?}");
        assert_eq!(transport.count(), 0);
    }
}

// GGL-016

#[tokio::test]
async fn ggl_016_whole_files_go_to_the_interactions_api() {
    let transport = Recorder::new([Reply::json(
        json!({"steps":[{"type":"model_output","content":[
            {"type":"text","text":"hello ","annotations":[{"type":"word_info","text":"hello","start_offset":"1.5s","end_offset":"2s"}]},
            {"type":"text","text":"world"}
        ]}]}),
    )]);
    let m = model("gemini-3.5-transcribe", transport.clone());
    let audio = TranscribeInput::File(
        FilePart::from_bytes(vec![1, 2, 3], Some("audio/wav"), None).unwrap(),
    );
    let params = TranscribeParams { provider_options: [("google".to_owned(), json!({"mode":"SMART","diarization":true,"wordTimestamp":true,"languageCodes":["en-US"],"customVocabulary":["Gemini"]}))].into() };
    let item = ops::transcribe(&m, audio, params, Default::default())
        .await
        .unwrap();
    assert_eq!(
        transport.request().url,
        "https://generativelanguage.googleapis.com/v1beta/interactions"
    );
    assert_eq!(
        transport.body(),
        json!({
            "model":"gemini-3.5-transcribe",
            "input":[{"type":"audio","data":"AQID","mime_type":"audio/wav"}],
            "generation_config":{"transcription_config":{"language_codes":["en-US"],"custom_vocabulary":["Gemini"],"mode":{"type":"smart","diarization_mode":"speaker","timestamp_granularities":["word"]}}}
        })
    );
    assert_eq!(item.value.text, "hello world");
    assert_eq!(
        item.value.segments,
        vec![TranscriptionSegment {
            text: "hello".into(),
            start_second: 1.5,
            end_second: 2.0
        }]
    );
    let error = ops::transcribe(
        &model_of("gemini-3.5-transcribe-live"),
        TranscribeInput::Bytes(vec![0; 4]),
        TranscribeParams::default(),
        Default::default(),
    )
    .await
    .unwrap_err();
    assert!(matches!(error, Error::Validation(_)), "{error:?}");
}

fn model_of(id: &str) -> Model {
    model(id, Recorder::new([]))
}

// GGL-017

use futures::StreamExt;
use suprnova_ai::transport::WebSocketMessage;

fn live_model(id: &str, socket: Arc<ScriptedSocket>) -> Model {
    let mut registry = ProviderRegistry::new();
    let provider = google::provider(key(), Recorder::new([])).with_websocket_transport(socket);
    google::register_provider(&mut registry, provider).unwrap();
    Model::new(id, registry.get_provider("google").unwrap())
}

fn chunks(n: usize) -> AudioChunks {
    futures::stream::iter((0..n).map(|i| vec![i as u8, 0])).boxed()
}

const NEVER: usize = usize::MAX;

async fn transcription_parts(
    model: &Model,
    audio: AudioChunks,
) -> Vec<Result<TranscriptionPart, Error>> {
    let stream = ops::stream_transcription(
        model,
        audio,
        AudioFormat::pcm(16000),
        StreamTranscribeParams::default(),
        Default::default(),
    )
    .await;
    match stream {
        Ok(stream) => stream.collect().await,
        Err(error) => vec![Err(error)],
    }
}

#[tokio::test]
async fn ggl_017_a_live_session_sends_the_setup_then_audio_and_reads_transcripts() {
    let socket = ScriptedSocket::new(vec![
        (1, text_message(json!({"setupComplete": {}}))),
        (
            4,
            text_message(json!({"serverContent": {"interimInputTranscription": {"text": "hel"}}})),
        ),
        (
            4,
            text_message(
                json!({"serverContent": {"inputTranscription": {"text": "hello", "languageCode": "en-US"}}}),
            ),
        ),
        (
            4,
            text_message(
                json!({"serverContent": {"inputTranscription": {"text": " world", "finished": true}}}),
            ),
        ),
        (
            4,
            text_message(
                json!({"serverContent": {"turnComplete": true}, "usageMetadata": {"totalTokenCount": 9}}),
            ),
        ),
    ]);
    let model = live_model("gemini-3.5-transcribe-live", socket.clone());
    let parts: Vec<TranscriptionPart> = transcription_parts(&model, chunks(2))
        .await
        .into_iter()
        .map(Result::unwrap)
        .collect();
    let request = socket.request();
    assert_eq!(
        request.url,
        "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
    );
    assert!(!request.url.contains("key="));
    assert!(
        request
            .headers
            .iter()
            .any(|(n, v)| n == "x-goog-api-key" && v == "g-key")
    );
    let sent = socket.sent_json();
    assert_eq!(
        sent[0],
        json!({"setup": {"model": "models/gemini-3.5-transcribe-live", "inputAudioTranscription": {}}})
    );
    assert_eq!(
        sent[1]["realtimeInput"]["audio"]["mimeType"],
        "audio/pcm;rate=16000"
    );
    assert_eq!(sent[1]["realtimeInput"]["audio"]["data"], "AAA=");
    assert_eq!(sent[3], json!({"realtimeInput": {"audioStreamEnd": true}}));
    let id = Some("google-segment-0".to_owned());
    assert_eq!(parts[0], TranscriptionPart::Start { warnings: vec![] });
    assert_eq!(
        parts[1],
        TranscriptionPart::Partial {
            id: id.clone(),
            text: "hel".into()
        }
    );
    assert_eq!(
        parts[2],
        TranscriptionPart::Delta {
            id: id.clone(),
            delta: "hello".into()
        }
    );
    assert_eq!(
        parts[3],
        TranscriptionPart::Delta {
            id: id.clone(),
            delta: " world".into()
        }
    );
    assert_eq!(
        parts[4],
        TranscriptionPart::Final {
            id,
            text: "hello world".into()
        }
    );
    match &parts[5] {
        TranscriptionPart::Finish {
            text,
            language,
            usage,
            ..
        } => {
            assert_eq!(text, "hello world");
            assert_eq!(language.as_deref(), Some("en-US"));
            assert_eq!(usage, &Some(json!({"totalTokenCount": 9})));
        }
        other => panic!("expected the finish, got {other:?}"),
    }
}

#[tokio::test]
async fn ggl_017_no_audio_goes_out_before_setup_complete_and_an_early_close_fails() {
    let socket = ScriptedSocket::new(vec![(
        1,
        WebSocketMessage::Close {
            code: Some(1011),
            reason: "boom".into(),
        },
    )]);
    let model = live_model("gemini-3.5-transcribe-live", socket.clone());
    let parts = transcription_parts(&model, chunks(2)).await;
    assert_eq!(
        socket.sent_json().len(),
        1,
        "audio went out before setupComplete"
    );
    match parts.last().unwrap() {
        Err(Error::Provider(error)) => assert!(
            error.message.contains("closed unexpectedly") && error.message.contains("1011"),
            "{}",
            error.message
        ),
        other => panic!("expected an error, got {other:?}"),
    }
}

#[tokio::test(start_paused = true)]
async fn ggl_017_a_session_finishes_three_seconds_after_the_last_transcript() {
    let socket = ScriptedSocket::new(vec![
        (1, text_message(json!({"setupComplete": {}}))),
        (
            3,
            text_message(json!({"serverContent": {"inputTranscription": {"text": "late"}}})),
        ),
        (NEVER, text_message(json!({}))),
    ]);
    let model = live_model("gemini-3.5-transcribe-live", socket);
    let started = tokio::time::Instant::now();
    let parts = transcription_parts(&model, chunks(1)).await;
    let elapsed = started.elapsed();
    assert!(
        elapsed >= std::time::Duration::from_secs(3) && elapsed < std::time::Duration::from_secs(4),
        "{elapsed:?}"
    );
    assert!(
        matches!(parts.last().unwrap(), Ok(TranscriptionPart::Finish { text, .. }) if text == "late")
    );
}

#[tokio::test]
async fn ggl_017_other_formats_and_non_live_models_fail() {
    let socket = ScriptedSocket::new(vec![]);
    let model = live_model("gemini-3.5-transcribe-live", socket.clone());
    let result = ops::stream_transcription(
        &model,
        chunks(1),
        AudioFormat::pcm(24000),
        StreamTranscribeParams::default(),
        Default::default(),
    )
    .await;
    assert!(matches!(result, Err(Error::Validation(_))));
    let model = live_model("gemini-3.5-transcribe", socket.clone());
    let result = ops::stream_transcription(
        &model,
        chunks(1),
        AudioFormat::pcm(16000),
        StreamTranscribeParams::default(),
        Default::default(),
    )
    .await;
    assert!(matches!(result, Err(Error::Validation(_))));
    assert!(socket.requests.lock().unwrap().is_empty());
}

// GGL-018

fn pcm(samples: &[i16]) -> String {
    use base64::Engine as _;
    let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

#[tokio::test]
async fn ggl_018_translation_reads_transcripts_audio_and_finishes_on_silence() {
    let silence = pcm(&vec![5; 24000]);
    let socket = ScriptedSocket::new(vec![
        (1, text_message(json!({"setupComplete": {}}))),
        (
            2,
            text_message(json!({"serverContent": {"inputTranscription": {"text": "hola"}}})),
        ),
        (
            2,
            text_message(
                json!({"serverContent": {"modelTurn": {"parts": [{"inlineData": {"data": pcm(&[9000, -9000])}}]}, "outputTranscription": {"text": "hello"}}, "usageMetadata": {"promptTokensDetails": [{"modality": "AUDIO", "tokenCount": 7}, {"modality": "TEXT", "tokenCount": 100}], "responseTokensDetails": [{"modality": "AUDIO", "tokenCount": 3}]}}),
            ),
        ),
        (
            3,
            text_message(
                json!({"serverContent": {"modelTurn": {"parts": [{"inlineData": {"data": silence}}]}}}),
            ),
        ),
        (NEVER, text_message(json!({}))),
    ]);
    let model = live_model("gemini-3.5-live-translate-preview", socket.clone());
    let params = StreamTranslateParams {
        target_language: "en".into(),
        source_language: Some("es".into()),
        provider_options: [("google".to_owned(), json!({"echoTargetLanguage": true}))].into(),
        ..Default::default()
    };
    let parts: Vec<TranslationPart> = ops::stream_translation(
        &model,
        chunks(1),
        AudioFormat::pcm(16000),
        params,
        Default::default(),
    )
    .await
    .unwrap()
    .map(Result::unwrap)
    .collect()
    .await;
    assert_eq!(
        socket.sent_json()[0],
        json!({"setup": {
            "model": "models/gemini-3.5-live-translate-preview",
            "generationConfig": {"responseModalities": ["AUDIO"], "translationConfig": {"targetLanguageCode": "en", "echoTargetLanguage": true}},
            "inputAudioTranscription": {},
            "outputAudioTranscription": {}
        }})
    );
    match &parts[0] {
        TranslationPart::Start { warnings } => {
            assert_eq!(warnings[0].feature.as_deref(), Some("sourceLanguage"))
        }
        other => panic!("expected the start, got {other:?}"),
    }
    let id = Some("google-item-0".to_owned());
    assert_eq!(
        parts[1],
        TranslationPart::SourceTranscriptDelta {
            id: id.clone(),
            delta: "hola".into()
        }
    );
    assert!(matches!(&parts[2], TranslationPart::Audio { id: audio_id, .. } if audio_id == &id));
    assert_eq!(
        parts[3],
        TranslationPart::OutputTextDelta {
            id: id.clone(),
            delta: "hello".into()
        }
    );
    match parts.last().unwrap() {
        TranslationPart::Finish {
            source_text,
            output_text,
            usage,
            ..
        } => {
            assert_eq!(
                (source_text.as_str(), output_text.as_str()),
                ("hola", "hello")
            );
            let usage = usage.as_ref().unwrap();
            assert_eq!(
                (
                    usage.input_audio_tokens,
                    usage.output_audio_tokens,
                    usage.input_text_tokens
                ),
                (Some(7), Some(3), None)
            );
        }
        other => panic!("expected the finish, got {other:?}"),
    }
}

// GGL-019

fn video_model(transport: Arc<Recorder>) -> Model {
    model("veo-3.1-generate-preview", transport)
}

fn video_model_with_downloads(transport: Arc<Recorder>, downloads: Arc<Recorder>) -> Model {
    Model::new(
        "veo-3.1-generate-preview",
        Arc::new(
            google::provider(key(), transport)
                .with_download_transport(Arc::new(Downloads(downloads))),
        ),
    )
}

#[tokio::test(start_paused = true)]
async fn ggl_019_a_job_starts_polls_and_downloads_with_the_key_header() {
    let uri = "https://generativelanguage.googleapis.com/v1beta/files/v1:download?alt=media";
    let transport = Recorder::new([
        Reply::json(json!({"name": "models/veo-3.1-generate-preview/operations/op1"})),
        Reply::json(
            json!({"name": "models/veo-3.1-generate-preview/operations/op1", "done": false}),
        ),
        Reply::json(
            json!({"name": "x", "done": true, "response": {"generateVideoResponse": {"generatedSamples": [{"video": {"uri": uri}}]}}}),
        ),
        Reply::bytes(vec![0, 0, 0, 24]).header("content-type", "video/mp4"),
    ]);
    let model = video_model(transport.clone());
    let prompt = VideoPrompt {
        text: Some("a wave".into()),
        image: Some(MediaInput::Url("https://example.com/start.png".into())),
        frame_images: vec![FrameImage {
            image: MediaInput::Url("gs://bucket/last.png".into()),
            frame_type: FrameType::LastFrame,
        }],
        references: vec![],
    };
    let params = VideoParams {
        resolution: Some("1920x1080".into()),
        aspect_ratio: Some("16:9".into()),
        duration: Some(8),
        provider_options: [("google".to_owned(), json!({"personGeneration": "allow_adult", "negativePrompt": "rain", "enhancePrompt": true}))].into(),
        ..Default::default()
    };
    let options = ops::VideoOptions {
        poll_interval: std::time::Duration::from_secs(1),
        ..Default::default()
    };
    let item = ops::generate_video(&model, prompt, params, options)
        .await
        .unwrap();
    let requests = transport.all();
    assert_eq!(
        requests[0].url,
        "https://generativelanguage.googleapis.com/v1beta/models/veo-3.1-generate-preview:predictLongRunning"
    );
    assert_eq!(
        transport.nth_body(0),
        json!({
            "instances": [{"prompt": "a wave", "lastFrame": {"gcsUri": "gs://bucket/last.png", "mimeType": "image/png"}}],
            "parameters": {"sampleCount": 1, "aspectRatio": "16:9", "resolution": "1080p", "durationSeconds": 8, "personGeneration": "allow_adult", "negativePrompt": "rain", "enhancePrompt": true}
        })
    );
    assert_eq!(requests[1].method, "GET");
    assert_eq!(
        requests[1].url,
        "https://generativelanguage.googleapis.com/v1beta/models/veo-3.1-generate-preview/operations/op1"
    );
    assert_eq!(requests[3].url, uri);
    assert_eq!(
        header(&requests[3], "x-goog-api-key").as_deref(),
        Some("g-key")
    );
    assert_eq!(item.value[0].data.as_bytes().unwrap(), &[0, 0, 0, 24]);
    assert!(
        item.warnings
            .iter()
            .any(|w| w.feature.as_deref() == Some("URL-based image input"))
    );
    assert!(!format!("{:?}", item.value).contains("g-key"));
}

#[tokio::test(start_paused = true)]
async fn ggl_019_a_download_follows_redirects_and_keeps_the_key_on_its_origin() {
    let uri = "https://generativelanguage.googleapis.com/v1beta/files/v1:download?alt=media";
    let transport = Recorder::new([
        Reply::json(json!({"name": "models/veo-3.1-generate-preview/operations/op1"})),
        Reply::json(
            json!({"name": "x", "done": true, "response": {"generateVideoResponse": {"generatedSamples": [{"video": {"uri": uri}}]}}}),
        ),
        Reply::status(302, json!({}))
            .header("location", "https://storage.example.com/v.mp4?sig=s1"),
    ]);
    let downloads =
        Recorder::new([Reply::bytes(vec![0, 0, 0, 24]).header("content-type", "video/mp4")]);
    let prompt = VideoPrompt {
        text: Some("a wave".into()),
        ..Default::default()
    };
    let options = ops::VideoOptions {
        poll_interval: std::time::Duration::from_secs(1),
        ..Default::default()
    };
    let item = ops::generate_video(
        &video_model_with_downloads(transport.clone(), downloads.clone()),
        prompt,
        VideoParams::default(),
        options,
    )
    .await
    .unwrap();
    let requests = transport.all();
    assert_eq!(requests.len(), 3);
    assert_eq!(requests[2].url, uri);
    assert_eq!(
        header(&requests[2], "x-goog-api-key").as_deref(),
        Some("g-key")
    );
    let fetched = downloads.all();
    assert_eq!(fetched.len(), 1);
    assert_eq!(fetched[0].url, "https://storage.example.com/v.mp4?sig=s1");
    assert_eq!(header(&fetched[0], "x-goog-api-key"), None);
    assert_eq!(item.value[0].data.as_bytes().unwrap(), &[0, 0, 0, 24]);
    assert_eq!(item.value[0].media_type, "video/mp4");

    let transport = Recorder::new([
        Reply::status(302, json!({})).header("location", "http://storage.example.com/v.mp4")
    ]);
    let downloads = Recorder::new([]);
    let model = video_model_with_downloads(transport.clone(), downloads.clone());
    let video = FilePart::from_url(uri, Some("video/mp4")).unwrap();
    let error = model
        .provider
        .protocol()
        .unwrap()
        .download_video(&model, video, model.provider.as_ref())
        .await
        .unwrap_err();
    assert!(matches!(error, Error::Provider(_)), "{error:?}");
    assert_eq!(
        transport.count() + downloads.count(),
        1,
        "a redirect from https to http was followed"
    );
}

#[tokio::test]
async fn ggl_019_a_redirect_off_the_origin_reaches_only_public_addresses() {
    let uri = "https://generativelanguage.googleapis.com/v1beta/files/v1:download?alt=media";
    let video = FilePart::from_url(uri, Some("video/mp4")).unwrap();
    let transport = Recorder::new([Reply::status(302, json!({}))
        .header("location", "https://169.254.169.254/computeMetadata/v1/")]);
    let downloads = Recorder::new([Reply::bytes(vec![1])]);
    let model = video_model_with_downloads(transport.clone(), downloads.clone());
    let error = model
        .provider
        .protocol()
        .unwrap()
        .download_video(&model, video.clone(), model.provider.as_ref())
        .await
        .unwrap_err();
    assert!(matches!(error, Error::Provider(_)), "{error:?}");
    assert_eq!(transport.count(), 1);
    assert_eq!(
        downloads.count(),
        0,
        "a redirect to a link-local address was requested"
    );

    let transport = Recorder::new([
        Reply::status(302, json!({})).header("location", "https://storage.example.com/v.mp4")
    ]);
    let model = video_model(transport.clone());
    let error = model
        .provider
        .protocol()
        .unwrap()
        .download_video(&model, video, model.provider.as_ref())
        .await
        .unwrap_err();
    assert!(matches!(error, Error::Validation(_)), "{error:?}");
    assert_eq!(
        transport.count(),
        1,
        "a hop off the origin went through the provider transport"
    );
}

#[tokio::test]
async fn ggl_019_a_video_on_another_origin_downloads_without_the_key_and_failures_report_the_message()
 {
    let transport = Recorder::new([
        Reply::json(
            json!({"done": true, "response": {"generateVideoResponse": {"generatedSamples": [{"video": {"uri": "https://cdn.example/v.mp4"}}]}}}),
        ),
        Reply::json(json!({"done": true, "error": {"message": "quota"}})),
    ]);
    let downloads = Recorder::new([Reply::bytes(vec![1])]);
    let model = video_model_with_downloads(transport.clone(), downloads.clone());
    let job = VideoJob {
        provider: "google".into(),
        model_id: "veo-3.1-generate-preview".into(),
        operation: json!({"operationName": "models/veo/operations/o2"}),
    };
    let VideoStatus::Completed(item) = ops::video_status(&model, &job, Default::default())
        .await
        .unwrap()
    else {
        panic!("not completed")
    };
    assert_eq!(
        item.value[0].data.as_text(),
        Some("https://cdn.example/v.mp4")
    );
    let protocol = model.provider.protocol().unwrap();
    let video = protocol
        .download_video(&model, item.value[0].clone(), model.provider.as_ref())
        .await
        .unwrap();
    assert_eq!(video.data.as_bytes().unwrap(), &[1]);
    assert_eq!(downloads.nth(0).url, "https://cdn.example/v.mp4");
    assert!(header(&downloads.nth(0), "x-goog-api-key").is_none());
    assert_eq!(transport.count(), 1);
    let status = ops::video_status(&model, &job, Default::default())
        .await
        .unwrap();
    assert_eq!(
        status,
        VideoStatus::Failed {
            message: "Video generation failed: quota".into()
        }
    );
}

// GGL-020

fn file_resource(state: &str) -> Value {
    json!({"file": {"name": "files/abc", "uri": "https://generativelanguage.googleapis.com/v1beta/files/abc", "mimeType": "application/pdf", "sizeBytes": "20971520", "state": state, "createTime": "2026-10-23T00:00:00Z", "expirationTime": "2026-10-25T00:00:00Z"}})
}

fn google_provider(transport: Arc<Recorder>) -> Arc<dyn suprnova_ai::Provider> {
    registry(transport).get_provider("google").unwrap()
}

#[tokio::test]
async fn ggl_020_a_large_file_goes_up_in_8_mib_chunks() {
    let transport = Recorder::new([
        Reply::json(json!({})).header("x-goog-upload-url", "https://upload.example/u1"),
        Reply::json(json!({})).header("x-goog-upload-status", "active"),
        Reply::json(json!({})).header("x-goog-upload-status", "active"),
        Reply::json(file_resource("ACTIVE")).header("x-goog-upload-status", "final"),
    ]);
    let provider = google_provider(transport.clone());
    let data = bytes::Bytes::from(vec![7u8; 20 * 1024 * 1024]);
    let file = UploadFile {
        data,
        media_type: "application/pdf".into(),
        filename: Some("r.pdf".into()),
    };
    let params = UploadParams {
        provider_options: [("google".to_owned(), json!({"displayName": "Report"}))].into(),
    };
    let item = ops::upload_file(provider.as_ref(), file, params, Default::default())
        .await
        .unwrap();
    let requests = transport.all();
    assert_eq!(
        requests[0].url,
        "https://generativelanguage.googleapis.com/upload/v1beta/files"
    );
    for (name, value) in [
        ("x-goog-upload-protocol", "resumable"),
        ("x-goog-upload-command", "start"),
        ("x-goog-upload-header-content-length", "20971520"),
        ("x-goog-upload-header-content-type", "application/pdf"),
        ("x-goog-api-key", "g-key"),
    ] {
        assert_eq!(header(&requests[0], name).as_deref(), Some(value), "{name}");
    }
    assert_eq!(
        serde_json::from_slice::<Value>(&requests[0].body).unwrap(),
        json!({"file": {"display_name": "Report"}})
    );
    let chunks: Vec<(String, String, usize)> = requests[1..]
        .iter()
        .map(|r| {
            (
                header(r, "x-goog-upload-offset").unwrap(),
                header(r, "x-goog-upload-command").unwrap(),
                r.body.len(),
            )
        })
        .collect();
    assert_eq!(
        chunks,
        vec![
            ("0".into(), "upload".into(), 8 * 1024 * 1024),
            ("8388608".into(), "upload".into(), 8 * 1024 * 1024),
            (
                "16777216".into(),
                "upload, finalize".into(),
                4 * 1024 * 1024
            ),
        ]
    );
    assert!(
        requests[1..]
            .iter()
            .all(|r| r.url == "https://upload.example/u1")
    );
    assert_eq!(
        item.value.provider_reference["google"],
        "https://generativelanguage.googleapis.com/v1beta/files/abc"
    );
    assert_eq!(item.value.size_bytes, Some(20 * 1024 * 1024));
    assert_eq!(
        item.value.expires_at.as_deref(),
        Some("2026-10-25T00:00:00Z")
    );
    assert!(
        item.warnings
            .iter()
            .any(|w| w.feature.as_deref() == Some("filename"))
    );
}

#[tokio::test(start_paused = true)]
async fn ggl_020_a_chunk_without_a_status_is_sent_again_and_processing_is_polled() {
    let transport = Recorder::new([
        Reply::json(json!({})).header("x-goog-upload-url", "https://upload.example/u1"),
        Reply::json(json!({})),
        Reply::json(file_resource("PROCESSING")).header("x-goog-upload-status", "final"),
        Reply::json(file_resource("ACTIVE")["file"].clone()),
    ]);
    let provider = google_provider(transport.clone());
    let file = UploadFile {
        data: bytes::Bytes::from_static(b"%PDF"),
        media_type: "application/pdf".into(),
        filename: None,
    };
    let params = UploadParams {
        provider_options: [("google".to_owned(), json!({"pollIntervalMs": 50}))].into(),
    };
    ops::upload_file(provider.as_ref(), file, params, Default::default())
        .await
        .unwrap();
    let requests = transport.all();
    assert_eq!(requests.len(), 4);
    assert_eq!(requests[1].url, requests[2].url);
    assert_eq!(requests[3].method, "GET");
    assert_eq!(
        requests[3].url,
        "https://generativelanguage.googleapis.com/v1beta/files/abc"
    );
}

#[tokio::test(start_paused = true)]
async fn ggl_020_failures_fail_the_upload() {
    let cases: Vec<Vec<Reply>> = vec![
        vec![Reply::json(json!({}))],
        vec![
            Reply::json(json!({})).header("x-goog-upload-url", "https://u/1"),
            Reply::json(json!({})).header("x-goog-upload-status", "active"),
        ],
        vec![
            Reply::json(json!({})).header("x-goog-upload-url", "https://u/1"),
            Reply::json(file_resource("FAILED")).header("x-goog-upload-status", "final"),
        ],
        vec![
            Reply::json(json!({})).header("x-goog-upload-url", "https://u/1"),
            Reply::json(json!({})),
            Reply::json(json!({})),
            Reply::json(json!({})),
        ],
    ];
    for replies in cases {
        let count = replies.len();
        let transport = Recorder::new(replies);
        let provider = google_provider(transport.clone());
        let file = UploadFile {
            data: bytes::Bytes::from_static(b"x"),
            media_type: "text/plain".into(),
            filename: None,
        };
        let error = ops::upload_file(
            provider.as_ref(),
            file,
            UploadParams::default(),
            ops::OperationOptions {
                retry: suprnova_ai::RetryPolicy::none(),
            },
        )
        .await
        .unwrap_err();
        assert!(matches!(error, Error::Provider(_)), "{error:?}");
        assert_eq!(
            transport.count(),
            count,
            "the upload sent another number of requests"
        );
    }
}

// GGL-021

fn text_request(id: &str, model_id: &str) -> BatchRequest {
    BatchRequest {
        id: id.into(),
        model_id: model_id.into(),
        kind: BatchRequestKind::Text {
            messages: hi(),
            options: StreamOptions::default(),
        },
    }
}

fn image_request(id: &str, model_id: &str, n: u32) -> BatchRequest {
    BatchRequest {
        id: id.into(),
        model_id: model_id.into(),
        kind: BatchRequestKind::Image {
            prompt: ImagePrompt {
                text: Some("a cat".into()),
                images: vec![],
                mask: None,
            },
            params: ImageParams {
                n,
                size: Some("1x1".into()),
                ..Default::default()
            },
        },
    }
}

fn operation(state: &str) -> Value {
    json!({"name": "batches/b1", "metadata": {"state": state, "createTime": "2026-10-23T00:00:00Z", "batchStats": {"requestCount": "2", "successfulRequestCount": "1", "pendingRequestCount": "1"}}})
}

#[tokio::test]
async fn ggl_021_a_small_batch_goes_inline() {
    let transport = Recorder::new([Reply::json(operation("BATCH_STATE_PENDING"))]);
    let provider = google_provider(transport.clone());
    let params = BatchParams {
        webhook_url: Some("https://hooks.example/b".into()),
        ..Default::default()
    };
    let started = ops::batch::start(
        provider.as_ref(),
        vec![
            text_request("t1", "gemini-2.5-flash-image"),
            image_request("i1", "gemini-2.5-flash-image", 1),
        ],
        params,
        Default::default(),
    )
    .await
    .unwrap();
    assert_eq!(
        transport.request().url,
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:batchGenerateContent"
    );
    let body = transport.body();
    let batch = &body["batch"];
    assert!(
        batch["displayName"]
            .as_str()
            .unwrap()
            .starts_with("ai-sdk-batch-")
    );
    assert_eq!(
        batch["webhookConfig"],
        json!({"uris": ["https://hooks.example/b"]})
    );
    let requests = batch["inputConfig"]["requests"]["requests"]
        .as_array()
        .unwrap();
    assert_eq!(requests[0]["metadata"], json!({"key": "t1"}));
    assert_eq!(
        requests[0]["request"]["contents"][0]["parts"][0]["text"],
        "hi"
    );
    assert_eq!(requests[1]["metadata"], json!({"key": "i1"}));
    assert_eq!(
        requests[1]["request"]["generationConfig"]["responseModalities"],
        json!(["IMAGE"])
    );
    assert_eq!(started.status.batch_id, "batches/b1");
    assert_eq!(started.status.state, BatchState::Pending);
    assert_eq!(started.warnings[0].request_id.as_deref(), Some("i1"));
    assert_eq!(started.warnings[0].warning.feature.as_deref(), Some("size"));
}

#[tokio::test]
async fn ggl_021_mixed_models_and_several_images_fail_before_a_request() {
    for requests in [
        vec![
            text_request("a", "gemini-2.5-flash"),
            text_request("b", "gemini-3-flash-preview"),
        ],
        vec![image_request("i", "gemini-2.5-flash-image", 2)],
        vec![],
    ] {
        let transport = Recorder::new([]);
        let error = ops::batch::start(
            google_provider(transport.clone()).as_ref(),
            requests,
            BatchParams::default(),
            Default::default(),
        )
        .await
        .unwrap_err();
        assert!(matches!(error, Error::Validation(_)), "{error:?}");
        assert_eq!(transport.count(), 0);
    }
}

#[tokio::test]
async fn ggl_021_every_request_is_checked_before_any_download() {
    let downloads =
        Recorder::new([Reply::bytes(vec![1, 2, 3]).header("content-type", "image/png")]);
    let transport = Recorder::new([Reply::json(operation("BATCH_STATE_PENDING"))]);
    let provider = google::provider(key(), transport.clone())
        .with_download_transport(Arc::new(Downloads(downloads.clone())));
    let picture = FilePart::from_url("https://example.com/cat.png", Some("image/png")).unwrap();
    let requests = vec![
        BatchRequest {
            id: "t1".into(),
            model_id: "gemini-2.5-flash-image".into(),
            kind: BatchRequestKind::Text {
                messages: vec![Message::new(MessageRole::User, vec![Part::File(picture)])],
                options: StreamOptions::default(),
            },
        },
        image_request("i1", "gemini-2.5-flash-image", 2),
    ];
    let error = ops::batch::start(
        &provider,
        requests,
        BatchParams::default(),
        Default::default(),
    )
    .await
    .unwrap_err();
    assert!(matches!(error, Error::Validation(_)), "{error:?}");
    assert_eq!(
        downloads.count(),
        0,
        "a batch that fails validation downloaded a file"
    );
    assert_eq!(transport.count(), 0);
}

#[tokio::test]
async fn ggl_021_a_large_batch_goes_up_as_json_lines() {
    let big = "x".repeat(6_000_000);
    let requests: Vec<BatchRequest> = (0..4)
        .map(|i| BatchRequest {
            id: format!("r{i}"),
            model_id: "gemini-2.5-flash".into(),
            kind: BatchRequestKind::Text {
                messages: vec![user_message(big.as_str())],
                options: StreamOptions::default(),
            },
        })
        .collect();
    let transport = Recorder::new([
        Reply::json(json!({})).header("x-goog-upload-url", "https://upload.example/u2"),
        Reply::json(json!({})).header("x-goog-upload-status", "active"),
        Reply::json(json!({})).header("x-goog-upload-status", "active"),
        Reply::json(
            json!({"file": {"name": "files/in1", "expirationTime": "2026-10-25T00:00:00Z"}}),
        )
        .header("x-goog-upload-status", "final"),
        Reply::json(operation("BATCH_STATE_PENDING")),
    ]);
    let started = ops::batch::start(
        google_provider(transport.clone()).as_ref(),
        requests,
        BatchParams::default(),
        Default::default(),
    )
    .await
    .unwrap();
    let all = transport.all();
    assert_eq!(all.len(), 5);
    assert_eq!(
        header(&all[0], "x-goog-upload-header-content-type").as_deref(),
        Some("application/jsonl")
    );
    let display = serde_json::from_slice::<Value>(&all[0].body).unwrap()["file"]["display_name"]
        .as_str()
        .unwrap()
        .to_owned();
    assert!(display.starts_with("ai-sdk-batch-") && display.ends_with("-input"));
    let uploaded: Vec<u8> = all[1..4].iter().flat_map(|r| r.body.to_vec()).collect();
    let lines: Vec<Value> = String::from_utf8(uploaded)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(lines.len(), 4);
    assert_eq!(lines[0]["key"], "r0");
    assert!(lines[0]["request"]["contents"].is_array());
    let create: Value = serde_json::from_slice(&all[4].body).unwrap();
    assert_eq!(
        create["batch"]["inputConfig"],
        json!({"fileName": "files/in1"})
    );
    assert_eq!(
        started.status.provider_metadata.unwrap()["google"],
        json!({"inputFileId": "files/in1", "inputFileExpiresAt": "2026-10-25T00:00:00Z"})
    );
}

// GGL-022

#[tokio::test]
async fn ggl_022_status_cancel_and_list_address_the_batch() {
    let transport = Recorder::new([
        Reply::json(operation("BATCH_STATE_RUNNING")),
        Reply::json(json!({})),
        Reply::json(
            json!({"operations": [operation("JOB_STATE_SUCCEEDED"), operation("BATCH_STATE_EXPIRED")], "nextPageToken": "n2"}),
        ),
    ]);
    let provider = google_provider(transport.clone());
    let status = ops::batch::status(provider.as_ref(), "batches/b1", Default::default())
        .await
        .unwrap();
    assert_eq!(status.state, BatchState::Pending);
    assert_eq!(status.raw_status.as_deref(), Some("BATCH_STATE_RUNNING"));
    assert_eq!(
        status.request_counts,
        Some(BatchRequestCounts {
            total: 2,
            pending: 1,
            completed: 1,
            failed: 0
        })
    );
    ops::batch::cancel(provider.as_ref(), "batches/b1", Default::default())
        .await
        .unwrap();
    let page = ops::batch::list(
        provider.as_ref(),
        Some(5),
        Some("c1".into()),
        Default::default(),
    )
    .await
    .unwrap();
    let requests = transport.all();
    assert_eq!(
        (requests[0].method.as_str(), requests[0].url.as_str()),
        (
            "GET",
            "https://generativelanguage.googleapis.com/v1beta/batches/b1"
        )
    );
    assert_eq!(
        (requests[1].method.as_str(), requests[1].url.as_str()),
        (
            "POST",
            "https://generativelanguage.googleapis.com/v1beta/batches/b1:cancel"
        )
    );
    assert_eq!(
        requests[2].url,
        "https://generativelanguage.googleapis.com/v1beta/batches?pageSize=5&pageToken=c1"
    );
    let states: Vec<BatchState> = page.batches.iter().map(|b| b.state).collect();
    assert_eq!(states, vec![BatchState::Completed, BatchState::Failed]);
    assert_eq!(page.next_cursor.as_deref(), Some("n2"));
}

#[tokio::test]
async fn ggl_022_counts_that_do_not_add_up_and_missing_states_map_as_typescript_does() {
    let mut odd = operation("BATCH_STATE_RUNNING");
    odd["metadata"]["batchStats"]["requestCount"] = json!("5");
    let transport = Recorder::new([
        Reply::json(odd),
        Reply::json(json!({"name": "batches/b2", "done": true})),
        Reply::json(
            json!({"name": "batches/b3", "error": {"code": 13, "message": "boom", "status": "INTERNAL"}}),
        ),
    ]);
    let provider = google_provider(transport.clone());
    assert_eq!(
        ops::batch::status(provider.as_ref(), "batches/b1", Default::default())
            .await
            .unwrap()
            .request_counts,
        None
    );
    assert_eq!(
        ops::batch::status(provider.as_ref(), "batches/b2", Default::default())
            .await
            .unwrap()
            .state,
        BatchState::Completed
    );
    let failed = ops::batch::status(provider.as_ref(), "batches/b3", Default::default())
        .await
        .unwrap();
    assert_eq!(failed.state, BatchState::Failed);
    assert_eq!(
        failed.error,
        Some(BatchError {
            message: "boom".into(),
            error_type: Some("INTERNAL".into()),
            code: Some("13".into())
        })
    );
}

#[tokio::test]
async fn ggl_022_results_stream_from_the_results_file() {
    let lines = [
        json!({"key": "e", "error": {"code": 1, "message": "stopped", "status": "CANCELLED"}}),
        json!({"key": "f", "error": {"code": 3, "message": "bad"}}),
        json!({"key": "b", "response": {"promptFeedback": {"blockReason": "SAFETY"}}}),
        json!({"key": "t", "response": {"candidates": [{"content": {"parts": [{"text": "hi"}]}, "finishReason": "STOP"}]}}),
        json!({"key": "i", "response": {"candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": "iVBORw=="}}]}, "finishReason": "STOP"}]}}),
    ];
    let file: String = lines.iter().map(|l| format!("{l}\n")).collect();
    let mut done = operation("BATCH_STATE_SUCCEEDED");
    done["metadata"]["output"] = json!({"responsesFile": "files/batch-out"});
    let transport = Recorder::new([Reply::json(done), Reply::bytes(file.into_bytes())]);
    let provider = google_provider(transport.clone());
    let results: Vec<BatchResult> =
        ops::batch::results(provider.as_ref(), "batches/b1", Default::default())
            .await
            .unwrap()
            .map(Result::unwrap)
            .collect()
            .await;
    assert_eq!(
        transport.nth(1).url,
        "https://generativelanguage.googleapis.com/download/v1beta/files/batch-out:download?alt=media"
    );
    assert!(
        matches!(&results[0].outcome, BatchOutcome::Cancelled { error: Some(error) } if error.message == "stopped")
    );
    assert!(
        matches!(&results[1].outcome, BatchOutcome::Failed { error, .. } if error.code.as_deref() == Some("3"))
    );
    assert!(
        matches!(&results[2].outcome, BatchOutcome::Failed { error, .. } if error.code.as_deref() == Some("prompt_blocked") && error.error_type.as_deref() == Some("SAFETY"))
    );
    assert!(
        matches!(&results[3].outcome, BatchOutcome::Message(message) if message.text() == "hi")
    );
    assert_eq!(results[4].kind, BatchResultKind::Image);
    assert!(matches!(&results[4].outcome, BatchOutcome::Images(item) if item.value.len() == 1));
}

#[tokio::test]
async fn ggl_022_pending_results_and_long_lines_fail() {
    let transport = Recorder::new([Reply::json(operation("BATCH_STATE_RUNNING"))]);
    let provider = google_provider(transport.clone());
    assert!(matches!(
        ops::batch::results(provider.as_ref(), "batches/b1", Default::default()).await,
        Err(Error::Validation(_))
    ));
    assert_eq!(transport.count(), 1);
    let mut done = operation("BATCH_STATE_SUCCEEDED");
    done["metadata"]["output"] = json!({"responsesFile": "files/big"});
    let long = vec![b'x'; 65 * 1024 * 1024];
    let transport = Recorder::with_chunks([Reply::json(done), Reply::bytes(long)], 1024 * 1024);
    let provider = google_provider(transport.clone());
    let mut results = ops::batch::results(provider.as_ref(), "batches/b1", Default::default())
        .await
        .unwrap();
    match results.next().await {
        Some(Err(Error::Provider(error))) => {
            assert!(error.message.contains("64 MiB"), "{}", error.message)
        }
        other => panic!("expected a line limit error, got {other:?}"),
    }
}

#[tokio::test]
async fn ggl_022_a_long_result_line_in_small_chunks_reads_in_linear_time() {
    let mut done = operation("BATCH_STATE_SUCCEEDED");
    done["metadata"]["output"] = json!({"responsesFile": "files/long"});
    let image = "A".repeat(8 * 1024 * 1024);
    let line = json!({"key": "i", "response": {"candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": image}}]}, "finishReason": "STOP"}]}});
    let file = format!("{line}\n").into_bytes();
    let transport = Recorder::with_chunks([Reply::json(done), Reply::bytes(file)], 1024);
    let provider = google_provider(transport);
    let started = std::time::Instant::now();
    let results: Vec<BatchResult> =
        ops::batch::results(provider.as_ref(), "batches/b1", Default::default())
            .await
            .unwrap()
            .map(Result::unwrap)
            .collect()
            .await;
    let elapsed = started.elapsed();
    assert_eq!(results.len(), 1);
    // A search from the start of the buffer after each 1 KiB chunk scans
    // about 32 billion bytes for this line; a linear reader takes well under
    // a second.
    assert!(elapsed < std::time::Duration::from_secs(30), "{elapsed:?}");
}

// GGL-023

#[tokio::test]
async fn ggl_023_a_token_request_keeps_the_key_in_a_header() {
    let transport = Recorder::new([Reply::json(
        json!({"name": "auth_tokens/t1", "expireTime": "2026-10-23T05:30:00Z"}),
    )]);
    let model = model("gemini-3.8-live", transport.clone());
    let session = RealtimeSessionConfig {
        instructions: Some("be kind".into()),
        voice: Some("Puck".into()),
        output_modalities: vec!["text".into()],
        tools: vec![RealtimeTool { name: "lookup".into(), description: Some("Look up".into()), parameters: json!({"type": "object"}) }],
        input_audio_transcription: Some(Default::default()),
        provider_options: [("google".to_owned(), json!({"defaultToolBehavior": "BLOCKING", "translationConfig": {"targetLanguageCode": "fr"}})), ("realtimeInputConfig".to_owned(), json!({"automaticActivityDetection": {"disabled": true}}))].into(),
        ..Default::default()
    };
    let token = ops::realtime::create_token(
        &model,
        RealtimeTokenParams {
            expires_after_seconds: Some(120),
            session: Some(session),
        },
        Default::default(),
    )
    .await
    .unwrap();
    let request = transport.request();
    assert_eq!(
        request.url,
        "https://generativelanguage.googleapis.com/v1alpha/auth_tokens"
    );
    assert_eq!(header(&request, "x-goog-api-key").as_deref(), Some("g-key"));
    let body = transport.body();
    assert_eq!(body["uses"], 0);
    assert!(body["expireTime"].as_str().unwrap().ends_with('Z'));
    assert!(body["newSessionExpireTime"].as_str().unwrap() < body["expireTime"].as_str().unwrap());
    let setup = &body["bidiGenerateContentSetup"];
    assert_eq!(setup["model"], "models/gemini-3.8-live");
    assert_eq!(
        setup["generationConfig"]["responseModalities"],
        json!(["TEXT"])
    );
    assert_eq!(
        setup["generationConfig"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"],
        "Puck"
    );
    assert_eq!(
        setup["generationConfig"]["translationConfig"],
        json!({"targetLanguageCode": "fr"})
    );
    assert_eq!(
        setup["systemInstruction"],
        json!({"parts": [{"text": "be kind"}]})
    );
    assert_eq!(
        setup["tools"][0]["functionDeclarations"][0]["behavior"],
        "BLOCKING"
    );
    assert_eq!(setup["inputAudioTranscription"], json!({}));
    assert_eq!(
        setup["realtimeInputConfig"],
        json!({"automaticActivityDetection": {"disabled": true}})
    );
    assert_eq!(token.token, "auth_tokens/t1");
    assert_eq!(
        token.url,
        "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContentConstrained"
    );
    assert_eq!(token.expires_at, Some(1_792_733_400));
    assert!(google::realtime_browser_url(&token).ends_with("?access_token=auth_tokens%2Ft1"));
    let thinking = ops::realtime::session(
        &model_of("gemini-3.8-live-extended-thinking"),
        &RealtimeSessionConfig::default(),
    )
    .unwrap();
    assert_eq!(
        thinking["generationConfig"]["thinkingConfig"],
        json!({"thinkingLevel": "low"})
    );
}

fn token() -> RealtimeToken {
    RealtimeToken { token: "auth_tokens/t1".into(), url: "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContentConstrained".into(), expires_at: None }
}

#[tokio::test]
async fn ggl_023_a_server_connection_sends_the_token_in_a_header_and_waits_for_setup() {
    let socket = ScriptedSocket::new(vec![
        (1, text_message(json!({"setupComplete": {}}))),
        (
            2,
            text_message(json!({"serverContent": {"modelTurn": {"parts": [{"text": "hi"}]}}})),
        ),
    ]);
    let model = live_model("gemini-3.8-live", socket.clone());
    let mut connection = ops::realtime::connect(&model, &token(), RealtimeSessionConfig::default())
        .await
        .unwrap();
    let request = socket.request();
    assert!(!request.url.contains("access_token"));
    assert!(
        request
            .headers
            .iter()
            .any(|(n, v)| n.eq_ignore_ascii_case("authorization") && v == "Token auth_tokens/t1")
    );
    assert!(socket.sent_json()[0].get("setup").is_some());
    let first = connection.next().await.unwrap().unwrap();
    assert!(matches!(
        first.kind,
        RealtimeServerEventKind::SessionCreated { .. }
    ));
    connection
        .send(RealtimeClientEvent::ConversationItemCreate(
            RealtimeItem::Text("hello".into()),
        ))
        .await
        .unwrap();
    assert_eq!(
        socket.sent_json()[1],
        json!({"realtimeInput": {"text": "hello"}})
    );
    let next = connection.next().await.unwrap().unwrap();
    assert!(
        matches!(next.kind, RealtimeServerEventKind::TextDelta { ref delta, .. } if delta == "hi")
    );

    let socket = ScriptedSocket::new(vec![(1, text_message(json!({"goAway": {}})))]);
    let model = live_model("gemini-3.8-live", socket.clone());
    let error = ops::realtime::connect(&model, &token(), RealtimeSessionConfig::default())
        .await
        .unwrap_err();
    assert!(matches!(error, Error::Provider(_)), "{error:?}");
    assert_eq!(socket.sent_json().len(), 1);
}

// GGL-024

#[test]
fn ggl_024_client_events_become_live_messages() {
    let model = model_of("gemini-3.8-live");
    let mut codec = ops::realtime::codec(&model).unwrap();
    let mut encode = |event| {
        codec
            .encode(&event)
            .unwrap()
            .map(|text| serde_json::from_str::<Value>(&text).unwrap())
    };
    let update = RealtimeSessionConfig {
        input_audio_format: Some(AudioFormat::pcm(24000)),
        ..Default::default()
    };
    assert!(
        encode(RealtimeClientEvent::SessionUpdate(update))
            .unwrap()
            .get("setup")
            .is_some()
    );
    assert_eq!(
        encode(RealtimeClientEvent::InputAudioAppend("AAA=".into())),
        Some(
            json!({"realtimeInput": {"audio": {"data": "AAA=", "mimeType": "audio/pcm;rate=24000"}}})
        )
    );
    assert_eq!(
        encode(RealtimeClientEvent::InputAudioCommit),
        Some(json!({"realtimeInput": {"audioStreamEnd": true}}))
    );
    assert_eq!(
        encode(RealtimeClientEvent::ConversationItemCreate(
            RealtimeItem::Text("hi".into())
        )),
        Some(json!({"realtimeInput": {"text": "hi"}}))
    );
    for (output, response) in [
        (r#"{"a":1}"#, json!({"a": 1})),
        ("[1,2]", json!({"output": [1, 2]})),
        ("not json", json!({"output": "not json"})),
    ] {
        assert_eq!(
            encode(RealtimeClientEvent::ConversationItemCreate(
                RealtimeItem::FunctionCallOutput {
                    call_id: "c1".into(),
                    name: Some("lookup".into()),
                    output: output.into()
                }
            )),
            Some(
                json!({"toolResponse": {"functionResponses": [{"id": "c1", "name": "lookup", "response": response}]}})
            )
        );
    }
    for event in [
        RealtimeClientEvent::InputAudioClear,
        RealtimeClientEvent::ResponseCreate { options: None },
        RealtimeClientEvent::ResponseCancel,
        RealtimeClientEvent::ConversationItemTruncate {
            item_id: "i".into(),
            content_index: 0,
            audio_end_ms: 1,
        },
        RealtimeClientEvent::ConversationItemCreate(RealtimeItem::Audio("AAA=".into())),
    ] {
        assert_eq!(encode(event), None);
    }
}

// GGL-025

#[test]
fn ggl_025_live_messages_become_server_events() {
    let model = model_of("gemini-3.8-live");
    let mut codec = ops::realtime::codec(&model).unwrap();
    let mut decode = |value: Value| {
        codec
            .decode(&value.to_string())
            .unwrap()
            .into_iter()
            .map(|e| e.kind)
            .collect::<Vec<_>>()
    };
    use RealtimeServerEventKind as K;
    assert!(matches!(
        decode(json!({"setupComplete": {}}))[..],
        [K::SessionCreated { .. }]
    ));
    let call = decode(
        json!({"toolCall": {"functionCalls": [{"id": "c1", "name": "lookup", "args": {"q": 1}}]}}),
    );
    assert_eq!(
        call,
        vec![
            K::FunctionCallArgumentsDelta {
                response_id: "google-resp-0".into(),
                item_id: "google-item-0".into(),
                call_id: "c1".into(),
                delta: r#"{"q":1}"#.into()
            },
            K::FunctionCallArgumentsDone {
                response_id: "google-resp-0".into(),
                item_id: "google-item-0".into(),
                call_id: "c1".into(),
                name: "lookup".into(),
                arguments: r#"{"q":1}"#.into()
            },
        ]
    );
    assert_eq!(
        decode(
            json!({"serverContent": {"modelTurn": {"parts": [{"inlineData": {"data": "AAA="}}, {"text": "hi"}]}, "outputTranscription": {"text": "hi"}}})
        ),
        vec![
            K::AudioDelta {
                response_id: "google-resp-0".into(),
                item_id: "google-item-0".into(),
                delta: "AAA=".into()
            },
            K::TextDelta {
                response_id: "google-resp-0".into(),
                item_id: "google-item-0".into(),
                delta: "hi".into()
            },
            K::AudioTranscriptDelta {
                response_id: "google-resp-0".into(),
                item_id: "google-item-0".into(),
                delta: "hi".into()
            },
        ]
    );
    assert_eq!(
        decode(json!({"serverContent": {"inputTranscription": {"text": "the quick"}}})),
        vec![K::InputTranscriptionCompleted {
            item_id: "google-input-0".into(),
            transcript: "the quick".into()
        }]
    );
    assert_eq!(
        decode(
            json!({"serverContent": {"inputTranscription": {"text": " fox", "finished": true}}})
        ),
        vec![K::InputTranscriptionCompleted {
            item_id: "google-input-0".into(),
            transcript: "the quick fox".into()
        }]
    );
    assert_eq!(
        decode(json!({"serverContent": {"turnComplete": true}})),
        vec![
            K::AudioDone {
                response_id: "google-resp-0".into(),
                item_id: "google-item-0".into()
            },
            K::TextDone {
                response_id: "google-resp-0".into(),
                item_id: "google-item-0".into(),
                text: None
            },
            K::AudioTranscriptDone {
                response_id: "google-resp-0".into(),
                item_id: "google-item-0".into(),
                transcript: None
            },
            K::ResponseDone {
                response_id: "google-resp-0".into(),
                status: "completed".into()
            },
        ]
    );
    assert_eq!(
        decode(json!({"serverContent": {"interrupted": true}})),
        vec![K::SpeechStarted { item_id: None }]
    );
    assert_eq!(
        decode(json!({"serverContent": {"inputTranscription": {"text": "next"}}})),
        vec![K::InputTranscriptionCompleted {
            item_id: "google-input-1".into(),
            transcript: "next".into()
        }]
    );
    assert_eq!(
        decode(json!({"serverContent": {"modelTurn": {"parts": [{"text": "two"}]}}})),
        vec![K::TextDelta {
            response_id: "google-resp-1".into(),
            item_id: "google-item-1".into(),
            delta: "two".into()
        }]
    );
    for name in ["toolCallCancellation", "goAway", "sessionResumptionUpdate"] {
        assert_eq!(
            decode(json!({name: {}})),
            vec![K::Custom {
                raw_type: name.into()
            }]
        );
    }
    assert_eq!(
        decode(
            json!({"serverContent": {"generationComplete": true, "interactionStatus": "IDLE", "waitingForInput": true}})
        ),
        vec![
            K::Custom {
                raw_type: "generationComplete".into()
            },
            K::Custom {
                raw_type: "interactionStatus".into()
            },
            K::Custom {
                raw_type: "waitingForInput".into()
            },
        ]
    );
}

// GGL-026

use std::{collections::BTreeMap, time::Duration};
use suprnova_ai::{
    providers::google::interactions,
    types::{Event, messages::FileData},
};

const REVISION: &str = "2026-05-20";
const INTERACTIONS: &str = "https://generativelanguage.googleapis.com/v1beta/interactions";

fn on_interactions(id: &str, transport: Arc<Recorder>) -> Model {
    model(id, transport).with_protocol(interactions::protocol())
}

fn on_agent(id: &str, transport: Arc<Recorder>) -> Model {
    model(id, transport).with_protocol(interactions::agent_protocol())
}

const AGENT: &str = "deep-research-pro-preview-12-2025";

fn interaction(status: &str, steps: Value) -> Value {
    json!({"id": "int-1", "status": status, "steps": steps})
}

fn in_progress() -> Reply {
    Reply::json(json!({"id": "int-1", "status": "in_progress"}))
}

fn done_reply() -> Reply {
    Reply::json(interaction(
        "completed",
        json!([{"type": "model_output", "content": [{"type": "text", "text": "ok"}]}]),
    ))
}

/// A stream event with its type and id.
fn event(kind: &str, id: &str, mut value: Value) -> Value {
    value["event_type"] = json!(kind);
    if !id.is_empty() {
        value["event_id"] = json!(id);
    }
    value
}

fn created(id: &str) -> Value {
    event(
        "interaction.created",
        id,
        json!({"interaction": {"id": "int-1", "status": "in_progress", "model": "gemini-3-flash-preview"}}),
    )
}

fn completed(id: &str) -> Value {
    event(
        "interaction.completed",
        id,
        json!({"interaction": {"id": "int-1", "status": "completed"}}),
    )
}

fn text_steps(index: u64, text: &str, first: &str) -> Vec<Value> {
    vec![
        event(
            "step.start",
            first,
            json!({"index": index, "step": {"type": "model_output"}}),
        ),
        event(
            "step.delta",
            "",
            json!({"index": index, "delta": {"type": "text", "text": text}}),
        ),
        event("step.stop", "", json!({"index": index})),
    ]
}

async fn interactions_rejected(m: impl Fn(Arc<Recorder>) -> Model, options: StreamOptions) {
    let transport = Recorder::new([done_reply()]);
    let (_, events) = call(&m(transport.clone()), hi(), options).await;
    assert_eq!(transport.count(), 0, "a rejected call sent a request");
    assert!(
        matches!(error(&events), Error::Validation(_)),
        "{:?}",
        error(&events)
    );
}

fn assert_revision(request: &suprnova_ai::transport::HttpRequest, revision: &str) {
    let values: Vec<&String> = request
        .headers
        .iter()
        .filter(|(n, _)| n.eq_ignore_ascii_case("api-revision"))
        .map(|(_, v)| v)
        .collect();
    assert_eq!(values, vec![revision], "{} {}", request.method, request.url);
}

#[tokio::test]
async fn ggl_026_a_model_request_carries_each_setting_and_tool() {
    let transport = Recorder::new([done_reply()]);
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
    });
    let p = options.params.as_mut().unwrap();
    p.output = Some(OutputParams {
        max_tokens: Some(100),
        ..Default::default()
    });
    p.provider_params = Some(
        [(
            "google".to_owned(),
            json!({
                "thinkingLevel": "low",
                "thinkingSummaries": "auto",
                "responseFormat": [{"type": "audio", "mimeType": "audio/wav"}],
                "imageConfig": {"aspectRatio": "1:1", "imageSize": "2K"},
                "responseModalities": ["text", "image", "audio"],
                "previousInteractionId": "int-0",
                "serviceTier": "flex",
                "store": true,
                "background": false,
            }),
        )]
        .into(),
    );
    options.tools = vec![
        tool("lookup"),
        google::tools::google_search(json!({"searchTypes": {"webSearch": {}, "imageSearch": {}}})).unwrap(),
        google::tools::code_execution(),
        google::tools::url_context(),
        google::tools::file_search(json!({"fileSearchStoreNames": ["stores/s1"], "topK": 3})).unwrap(),
        google::tools::google_maps(),
        google::tools::computer_use(json!({"excludedPredefinedFunctions": ["drag_and_drop"]})).unwrap(),
        google::tools::mcp_server(json!({"name": "docs", "url": "https://mcp.example/sse", "headers": {"x-k": "v"}, "allowedTools": ["read"]})).unwrap(),
        google::tools::retrieval(json!({"vertexAiSearchConfig": {"datastores": ["d1"]}})).unwrap(),
    ];
    options.tool_choice = Some(ToolChoice::Tool(ToolRef::new("lookup")));
    options.output_schema = Some(
        json!({"type": "object", "properties": {"a": {"type": "string"}}})
            .try_into()
            .unwrap(),
    );
    let messages = vec![suprnova_ai::system_message("be brief"), user_message("hi")];
    let message = suprnova_ai::generate(
        &on_interactions("gemini-3-flash-preview", transport.clone()),
        messages,
        options,
    )
    .await
    .unwrap();
    assert_eq!(message.text(), "ok");
    let request = transport.request();
    assert_eq!(
        (request.method.as_str(), request.url.as_str()),
        ("POST", INTERACTIONS)
    );
    assert_revision(&request, REVISION);
    assert_eq!(header(&request, "x-goog-api-key").as_deref(), Some("g-key"));
    let mut body = transport.body();
    let schema = body["response_format"][0]
        .as_object_mut()
        .unwrap()
        .remove("schema")
        .unwrap();
    assert_eq!(schema["properties"]["a"], json!({"type": "string"}));
    assert_eq!(
        body,
        json!({
            "model": "gemini-3-flash-preview",
            "input": [{"type": "user_input", "content": [{"type": "text", "text": "hi"}]}],
            "system_instruction": "be brief",
            "tools": [
                {"type": "function", "name": "lookup", "description": "The lookup tool", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}},
                {"type": "google_search", "search_types": ["web_search", "image_search"]},
                {"type": "code_execution"},
                {"type": "url_context"},
                {"type": "file_search", "file_search_store_names": ["stores/s1"], "top_k": 3},
                {"type": "google_maps"},
                {"type": "computer_use", "environment": "browser", "excludedPredefinedFunctions": ["drag_and_drop"]},
                {"type": "mcp_server", "name": "docs", "url": "https://mcp.example/sse", "headers": {"x-k": "v"}, "allowed_tools": ["read"]},
                {"type": "retrieval", "retrieval_types": ["vertex_ai_search"], "vertex_ai_search_config": {"datastores": ["d1"]}},
            ],
            "response_format": [
                {"type": "text", "mime_type": "application/json"},
                {"type": "audio", "mime_type": "audio/wav"},
                {"type": "image", "mime_type": "image/png", "aspect_ratio": "1:1", "image_size": "2K"},
            ],
            "response_modalities": ["text", "image", "audio"],
            "previous_interaction_id": "int-0",
            "service_tier": "flex",
            "store": true,
            "background": false,
            "generation_config": {
                "temperature": 0.5,
                "top_p": 0.9,
                "top_k": 40,
                "seed": 7,
                "max_output_tokens": 100,
                "thinking_level": "low",
                "thinking_summaries": "auto",
                "tool_choice": {"allowed_tools": {"mode": "validated", "tools": ["lookup"]}},
            },
        })
    );
}

#[tokio::test]
async fn ggl_026_an_agent_request_carries_its_config_and_environment() {
    let transport = Recorder::new([done_reply()]);
    let options = google_options(
        "google",
        json!({
            "agentConfig": {"type": "deep-research", "thinkingSummaries": "auto", "visualization": "off", "collaborativePlanning": true},
            "environment": {"type": "remote", "sources": [{"type": "inline", "content": "x = 1", "target": "/a.py"}, {"type": "repository", "source": "https://git.example/r"}], "network": {"allowlist": [{"domain": "pypi.org"}]}},
            "systemInstruction": "cite sources",
        }),
    );
    let message = suprnova_ai::generate(&on_agent(AGENT, transport.clone()), hi(), options)
        .await
        .unwrap();
    assert_eq!(message.text(), "ok");
    assert_revision(&transport.request(), REVISION);
    assert_eq!(
        transport.body(),
        json!({
            "agent": AGENT,
            "input": [{"type": "user_input", "content": [{"type": "text", "text": "hi"}]}],
            "system_instruction": "cite sources",
            "agent_config": {"type": "deep-research", "thinking_summaries": "auto", "visualization": "off", "collaborative_planning": true},
            "environment": {"type": "remote", "sources": [{"type": "inline", "content": "x = 1", "target": "/a.py"}, {"type": "repository", "source": "https://git.example/r"}], "network": {"allowlist": [{"domain": "pypi.org"}]}},
        })
    );
}

#[tokio::test]
async fn ggl_026_a_stream_adds_stream_true() {
    let transport = Recorder::new([Reply::sse(
        &[
            vec![created("")],
            text_steps(0, "ok", ""),
            vec![completed("")],
        ]
        .concat(),
    )]);
    let (handle, events) = call(
        &on_interactions("gemini-3-flash-preview", transport.clone()),
        hi(),
        StreamOptions::default(),
    )
    .await;
    ok(&events);
    assert_eq!(handle.text(), "ok");
    let request = transport.request();
    assert_revision(&request, REVISION);
    assert_eq!(
        header(&request, "accept").as_deref(),
        Some("text/event-stream")
    );
    assert_eq!(
        transport.body(),
        json!({"model": "gemini-3-flash-preview", "input": [{"type": "user_input", "content": [{"type": "text", "text": "hi"}]}], "stream": true})
    );
}

#[tokio::test]
async fn ggl_026_settings_that_do_not_apply_fail_before_a_request() {
    let on_model = |t| on_interactions("gemini-3-flash-preview", t);
    let on_an_agent = |t| on_agent(AGENT, t);
    let penalties = [
        RepetitionPenaltyParams {
            frequency_penalty: ProviderValue::Value(Some(0.1)),
            ..Default::default()
        },
        RepetitionPenaltyParams {
            presence_penalty: ProviderValue::Value(Some(0.1)),
            ..Default::default()
        },
    ];
    for penalty in penalties {
        interactions_rejected(on_model, sampling(|s| s.repetition_penalty = Some(penalty))).await;
    }
    for options in unsent_settings() {
        interactions_rejected(on_model, options).await;
    }
    let schema = StreamOptions {
        output_schema: Some(json!({"type": "object"}).try_into().unwrap()),
        ..Default::default()
    };
    interactions_rejected(on_an_agent, schema).await;
    let temperature = sampling(|s| {
        s.temperature = Some(TemperatureSamplerParams {
            temperature: ProviderValue::Value(0.5),
        })
    });
    interactions_rejected(on_an_agent, temperature).await;
    let output = params(|p| {
        p.output = Some(OutputParams {
            max_tokens: Some(10),
            ..Default::default()
        })
    });
    interactions_rejected(on_an_agent, output).await;
    interactions_rejected(
        on_an_agent,
        google_options("google", json!({"thinkingLevel": "low"})),
    )
    .await;
    interactions_rejected(
        on_model,
        google_options("google", json!({"environment": "env-1"})),
    )
    .await;
    interactions_rejected(on_model, effort(Some("low"))).await;
}

#[tokio::test]
async fn ggl_026_a_tool_choice_goes_only_with_a_function_tool() {
    for (choice, wire) in [
        (ToolChoice::Auto, json!("auto")),
        (ToolChoice::Required, json!("any")),
        (ToolChoice::None, json!("none")),
    ] {
        let transport = Recorder::new([done_reply()]);
        let options = with_tools(vec![tool("lookup")], Some(choice));
        suprnova_ai::generate(
            &on_interactions("gemini-3-flash-preview", transport.clone()),
            hi(),
            options,
        )
        .await
        .unwrap();
        assert_eq!(transport.body()["generation_config"]["tool_choice"], wire);
    }
    let transport = Recorder::new([done_reply()]);
    let options = with_tools(
        vec![google::tools::code_execution()],
        Some(ToolChoice::Required),
    );
    suprnova_ai::generate(
        &on_interactions("gemini-3-flash-preview", transport.clone()),
        hi(),
        options,
    )
    .await
    .unwrap();
    let body = transport.body();
    assert!(body.get("generation_config").is_none(), "{body}");
    assert_eq!(body["tools"], json!([{"type": "code_execution"}]));
}

#[tokio::test]
async fn ggl_026_a_configured_revision_is_kept() {
    let transport = Recorder::new([done_reply()]);
    let mut registry = ProviderRegistry::new();
    let config = ProviderConfig {
        headers: [("Api-Revision".to_owned(), "2026-01-01".to_owned())].into(),
        ..key()
    };
    google::register(&mut registry, config, transport.clone()).unwrap();
    let m = Model::new(
        "gemini-3-flash-preview",
        registry.get_provider("google").unwrap(),
    )
    .with_protocol(interactions::protocol());
    suprnova_ai::generate(&m, hi(), StreamOptions::default())
        .await
        .unwrap();
    assert_revision(&transport.request(), "2026-01-01");
}

// GGL-027

fn google_meta_of(fields: Value) -> Option<suprnova_ai::types::messages::Metadata> {
    Some([("google".to_owned(), fields)].into())
}

#[tokio::test]
async fn ggl_027_a_history_becomes_one_step_per_turn() {
    let png = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
    let mut video = FilePart::from_url("https://example.com/clip.mp4", Some("video/mp4")).unwrap();
    video.provider_metadata = google_meta_of(
        json!({"processing": {"type": "static", "startOffset": 5, "endOffset": 10.5, "fps": 2}}),
    );
    let user = Message::new(
        MessageRole::User,
        vec![
            Part::Text(TextPart::new("a")),
            Part::Text(TextPart::new("b")),
            Part::File(FilePart::from_bytes(png.clone(), Some("image/png"), None).unwrap()),
            Part::File(
                FilePart::from_url("https://example.com/doc.pdf", Some("application/pdf")).unwrap(),
            ),
            Part::File(video),
            Part::File(FilePart::from_bytes(vec![1, 2, 3], Some("audio/wav"), None).unwrap()),
        ],
    );
    let mut reasoning = ReasoningPart::new("thinking");
    reasoning.provider_metadata = google_meta_of(json!({"signature": "s1"}));
    let mut lookup = ToolCallPart::new("c1", "lookup", r#"{"q":"x"}"#);
    lookup.provider_metadata = google_meta_of(json!({"signature": "s2"}));
    let processing = suprnova_ai::types::messages::BuiltinToolCallPart {
        id: "part-p".into(),
        tool_call_id: "p1".into(),
        tool_name: "processing".into(),
        tool_args: "{}".into(),
        provider_metadata: google_meta_of(json!({"processingId": "p1", "signature": "s3"})),
    };
    let processed = suprnova_ai::types::messages::BuiltinToolReturnPart {
        id: "part-r".into(),
        tool_call_id: "p1".into(),
        tool_name: "processing".into(),
        result: Value::Null,
        is_error: false,
        provider_metadata: google_meta_of(json!({"processingCallId": "p1"})),
    };
    let search = suprnova_ai::types::messages::BuiltinToolCallPart {
        id: "part-s".into(),
        tool_call_id: "gs1".into(),
        tool_name: "google_search".into(),
        tool_args: "{}".into(),
        provider_metadata: None,
    };
    let content = ContentOutput::new(vec![
        ContentPart::Text(TextPart::new("one")),
        ContentPart::File(FilePart::from_bytes(png, Some("image/png"), None).unwrap()),
    ]);
    let messages = vec![
        suprnova_ai::system_message("be brief"),
        user,
        assistant(vec![
            Part::Reasoning(reasoning),
            Part::Text(TextPart::new("t1")),
            Part::Text(TextPart::new("t2")),
            Part::ToolCall(lookup),
            Part::ToolCall(ToolCallPart::new("c2", "lookup", "")),
            Part::ToolCall(ToolCallPart::new("c3", "shot", "{}")),
            Part::BuiltinToolCall(search),
            Part::BuiltinToolCall(processing),
            Part::BuiltinToolReturn(processed),
        ]),
        Message::new(
            MessageRole::Tool,
            vec![
                Part::ToolResult(ToolResultPart::new("c1", "lookup", json!({"v": 1}))),
                Part::ToolResult(ToolResultPart::error("c2", "lookup", json!("boom"))),
                Part::ToolResult(ToolResultPart::new("c3", "shot", content)),
            ],
        ),
    ];
    let transport = Recorder::new([done_reply()]);
    let options = google_options("google", json!({"mediaResolution": "high"}));
    suprnova_ai::generate(
        &on_interactions("gemini-3-flash-preview", transport.clone()),
        messages,
        options,
    )
    .await
    .unwrap();
    let body = transport.body();
    assert_eq!(body["system_instruction"], "be brief");
    assert_eq!(
        body["input"],
        json!([
            {"type": "user_input", "content": [
                {"type": "text", "text": "a\n\nb"},
                {"type": "image", "data": "iVBORw0KGgo=", "mime_type": "image/png", "resolution": "high"},
                {"type": "document", "uri": "https://example.com/doc.pdf", "mime_type": "application/pdf"},
                {"type": "video", "uri": "https://example.com/clip.mp4", "mime_type": "video/mp4", "resolution": "high", "processing": {"type": "static", "start_offset": "5s", "end_offset": "10.5s", "fps": 2}},
                {"type": "audio", "data": "AQID", "mime_type": "audio/wav"},
            ]},
            {"type": "thought", "signature": "s1", "summary": [{"type": "text", "text": "thinking"}]},
            {"type": "model_output", "content": [{"type": "text", "text": "t1"}, {"type": "text", "text": "t2"}]},
            {"type": "function_call", "id": "c1", "name": "lookup", "arguments": {"q": "x"}, "signature": "s2"},
            {"type": "function_call", "id": "c2", "name": "lookup", "arguments": {}},
            {"type": "function_call", "id": "c3", "name": "shot", "arguments": {}},
            {"type": "processing_call", "id": "p1", "signature": "s3"},
            {"type": "processing_result", "call_id": "p1"},
            {"type": "user_input", "content": [
                {"type": "function_result", "call_id": "c1", "name": "lookup", "result": "{\"v\":1}"},
                {"type": "function_result", "call_id": "c2", "name": "lookup", "is_error": true, "result": "boom"},
                {"type": "function_result", "call_id": "c3", "name": "shot", "result": [{"type": "text", "text": "one"}, {"type": "image", "data": "iVBORw0KGgo=", "mime_type": "image/png"}]},
            ]},
        ])
    );
}

#[tokio::test]
async fn ggl_027_a_continued_interaction_leaves_out_its_turns() {
    let mut text = TextPart::new("earlier answer");
    text.provider_metadata = google_meta_of(json!({"interactionId": "int-0"}));
    let mut call_part = ToolCallPart::new("c1", "lookup", "{}");
    call_part.provider_metadata = google_meta_of(json!({"interactionId": "int-0"}));
    let messages = vec![
        user_message("first"),
        assistant(vec![Part::Text(text), Part::ToolCall(call_part)]),
        Message::new(
            MessageRole::Tool,
            vec![Part::ToolResult(ToolResultPart::new(
                "c1",
                "lookup",
                json!("r"),
            ))],
        ),
        user_message("next"),
    ];
    let continued = |store: Value| {
        let messages = messages.clone();
        async move {
            let transport = Recorder::new([done_reply()]);
            let mut options = json!({"previousInteractionId": "int-0"});
            if !store.is_null() {
                options["store"] = store;
            }
            suprnova_ai::generate(
                &on_interactions("gemini-3-flash-preview", transport.clone()),
                messages,
                google_options("google", options),
            )
            .await
            .unwrap();
            transport.body()["input"].clone()
        }
    };
    assert_eq!(
        continued(Value::Null).await,
        json!([
            {"type": "user_input", "content": [{"type": "text", "text": "first"}]},
            {"type": "user_input", "content": [{"type": "text", "text": "next"}]},
        ])
    );
    assert_eq!(continued(json!(false)).await.as_array().unwrap().len(), 5);
}

// GGL-028

fn full_response() -> Value {
    json!({
        "id": "int-9",
        "status": "completed",
        "service_tier": "priority",
        "steps": [
            {"type": "user_input", "content": [{"type": "text", "text": "ignored"}]},
            {"type": "thought", "signature": "sig-t", "summary": [{"type": "text", "text": "plan"}]},
            {"type": "google_search_call", "id": "gs1", "arguments": {"queries": ["rust"]}},
            {"type": "google_search_result", "call_id": "gs1", "result": [{"url": "https://a.example", "title": "A"}]},
            {"type": "processing_call", "id": "p1", "signature": "sig-p"},
            {"type": "processing_result", "call_id": "p1"},
            {"type": "model_output", "content": [
                {"type": "text", "text": "answer", "annotations": [
                    {"type": "url_citation", "url": "https://a.example", "title": "A"},
                    {"type": "url_citation", "url": "https://b.example"},
                ]},
                {"type": "image", "data": "iVBORw0KGgo=", "mime_type": "image/png"},
                {"type": "video", "uri": "https://v.example/clip.mp4"},
            ]},
            {"type": "function_call", "id": "f1", "name": "lookup", "arguments": {"q": "x"}, "signature": "sig-f"},
        ],
        "usage": {
            "total_input_tokens": 10,
            "total_cached_tokens": 4,
            "total_output_tokens": 6,
            "total_thought_tokens": 3,
            "output_tokens_by_modality": [{"modality": "text", "tokens": 5}, {"modality": "image", "tokens": 1}],
        },
    })
}

fn json_args(text: &str) -> Value {
    serde_json::from_str(text).unwrap_or(Value::Null)
}

/// A part without its generated ids.
fn shape(part: &Part) -> Value {
    match part {
        Part::Text(p) => json!({"text": p.text, "meta": p.provider_metadata}),
        Part::Reasoning(p) => json!({"reasoning": p.text, "meta": p.provider_metadata}),
        Part::File(p) => {
            json!({"file": p.media_type, "data": p.data.as_text(), "meta": p.provider_metadata})
        }
        Part::ToolCall(p) => {
            json!({"call": p.tool_name, "id": p.tool_call_id, "args": json_args(&p.tool_args), "meta": p.provider_metadata})
        }
        Part::BuiltinToolCall(p) => {
            json!({"builtin": p.tool_name, "id": p.tool_call_id, "args": json_args(&p.tool_args), "meta": p.provider_metadata})
        }
        Part::BuiltinToolReturn(p) => {
            json!({"result": p.tool_name, "id": p.tool_call_id, "value": p.result, "error": p.is_error, "meta": p.provider_metadata})
        }
        other => json!({"other": format!("{other:?}")}),
    }
}

/// The sources of a message's final metadata, without their ids.
fn sources_of(message: &Message) -> Value {
    let sources = message.provider_metadata.as_ref().unwrap()["google"]["sources"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    sources
        .into_iter()
        .map(|mut source| {
            source.as_object_mut().unwrap().remove("id");
            source
        })
        .collect()
}

fn meta(fields: Value) -> Value {
    let mut google = json!({"interactionId": "int-9"});
    google
        .as_object_mut()
        .unwrap()
        .extend(fields.as_object().unwrap().clone());
    json!({"google": google})
}

#[tokio::test]
async fn ggl_028_a_response_gives_each_step_as_a_part() {
    let transport =
        Recorder::new([Reply::json(full_response()).header("x-gemini-service-tier", "standard")]);
    let message = suprnova_ai::generate(
        &on_interactions("gemini-3-flash-preview", transport),
        hi(),
        StreamOptions::default(),
    )
    .await
    .unwrap();
    let parts: Vec<Value> = message.parts.iter().map(shape).collect();
    assert_eq!(
        parts,
        vec![
            json!({"reasoning": "plan", "meta": meta(json!({"signature": "sig-t"}))}),
            json!({"builtin": "google_search", "id": "gs1", "args": {"queries": ["rust"]}, "meta": meta(json!({}))}),
            json!({"result": "google_search", "id": "gs1", "value": [{"url": "https://a.example", "title": "A"}], "error": false, "meta": meta(json!({}))}),
            json!({"builtin": "processing", "id": "p1", "args": {}, "meta": meta(json!({"signature": "sig-p", "processingId": "p1"}))}),
            json!({"result": "processing", "id": "p1", "value": null, "error": false, "meta": meta(json!({"processingCallId": "p1"}))}),
            json!({"text": "answer", "meta": meta(json!({}))}),
            json!({"file": "image/png", "data": "iVBORw0KGgo=", "meta": meta(json!({}))}),
            json!({"file": "video/mp4", "data": "https://v.example/clip.mp4", "meta": meta(json!({}))}),
            json!({"call": "lookup", "id": "f1", "args": {"q": "x"}, "meta": meta(json!({"signature": "sig-f"}))}),
        ]
    );
    let usage = message.usage.as_ref().unwrap();
    assert_eq!(
        (
            usage.input_tokens,
            usage.cache_read_tokens,
            usage.output_tokens,
            usage.reasoning_tokens
        ),
        (10, Some(4), 9, Some(3))
    );
    let google = &message.provider_metadata.as_ref().unwrap()["google"];
    assert_eq!(google["interactionId"], "int-9");
    assert_eq!(google["serviceTier"], "priority");
    assert_eq!(
        google["outputTokensByModality"],
        json!({"text": 5, "image": 1})
    );
    assert_eq!(
        sources_of(&message),
        json!([
            {"sourceType": "url", "url": "https://a.example", "title": "A"},
            {"sourceType": "url", "url": "https://b.example"},
        ])
    );
}

/// The finish reason of a background call whose create answers `response`.
async fn replayed_finish(response: Value) -> (Option<String>, Message) {
    let transport = Recorder::new([Reply::json(response)]);
    let (handle, events) = call(
        &on_interactions("gemini-3-flash-preview", transport.clone()),
        hi(),
        google_options("google", json!({"background": true})),
    )
    .await;
    ok(&events);
    assert_eq!(transport.count(), 1, "a terminal response was not replayed");
    (
        handle.finish_reason().map(str::to_owned),
        handle.message().clone(),
    )
}

#[tokio::test]
async fn ggl_028_statuses_map_to_finish_reasons() {
    let text = json!([{"type": "model_output", "content": [{"type": "text", "text": "ok"}]}]);
    for (status, reason) in [
        ("completed", "stop"),
        ("failed", "error"),
        ("incomplete", "length"),
        ("cancelled", "other"),
    ] {
        let (finish, _) = replayed_finish(interaction(status, text.clone())).await;
        assert_eq!(finish.as_deref(), Some(reason), "{status}");
    }
    let (finish, message) = replayed_finish(full_response()).await;
    assert_eq!(finish.as_deref(), Some("tool_call"));
    assert_eq!(message.usage.as_ref().unwrap().output_tokens, 9);
    assert_eq!(sources_of(&message).as_array().unwrap().len(), 2);
    let transport = Recorder::new([Reply::sse(&[
        created(""),
        event(
            "step.start",
            "",
            json!({"index": 0, "step": {"type": "function_call", "id": "f1", "name": "lookup"}}),
        ),
        event(
            "step.delta",
            "",
            json!({"index": 0, "delta": {"type": "arguments_delta", "arguments": "{}"}}),
        ),
        event("step.stop", "", json!({"index": 0})),
        event("interaction.requires_action", "", json!({})),
    ])]);
    let (handle, events) = call(
        &on_interactions("gemini-3-flash-preview", transport),
        hi(),
        StreamOptions::default(),
    )
    .await;
    ok(&events);
    assert_eq!(handle.finish_reason(), Some("tool_call"));
}

// GGL-029

#[tokio::test]
async fn ggl_029_a_stream_reads_each_step_kind_by_index() {
    let transport = Recorder::new([Reply::sse(&[
        created(""),
        event(
            "interaction.status_update",
            "",
            json!({"interaction_id": "int-1", "status": "in_progress"}),
        ),
        event(
            "step.start",
            "",
            json!({"index": 0, "step": {"type": "thought"}}),
        ),
        event(
            "step.start",
            "",
            json!({"index": 1, "step": {"type": "model_output"}}),
        ),
        event(
            "step.delta",
            "",
            json!({"index": 0, "delta": {"type": "thought_summary", "content": {"type": "text", "text": "plan"}}}),
        ),
        event(
            "step.delta",
            "",
            json!({"index": 1, "delta": {"type": "text", "text": "Hel"}}),
        ),
        event(
            "step.delta",
            "",
            json!({"index": 0, "delta": {"type": "thought_signature", "signature": "sig-t"}}),
        ),
        event(
            "step.delta",
            "",
            json!({"index": 1, "delta": {"type": "text_annotation", "annotations": [{"type": "url_citation", "url": "https://a.example", "title": "A"}]}}),
        ),
        event(
            "step.delta",
            "",
            json!({"index": 1, "delta": {"type": "text", "text": "lo"}}),
        ),
        event(
            "step.delta",
            "",
            json!({"index": 1, "delta": {"type": "text_annotation", "annotations": [{"type": "url_citation", "url": "https://a.example", "title": "A"}]}}),
        ),
        event("step.stop", "", json!({"index": 0})),
        event("step.stop", "", json!({"index": 1})),
        event(
            "step.start",
            "",
            json!({"index": 2, "step": {"type": "function_call", "name": "lookup"}}),
        ),
        event(
            "step.delta",
            "",
            json!({"index": 2, "delta": {"type": "arguments_delta", "id": "f1", "arguments": "{\"q\":", "signature": "sig-f"}}),
        ),
        event(
            "step.delta",
            "",
            json!({"index": 2, "delta": {"type": "arguments_delta", "arguments": "\"x\"}"}}),
        ),
        event(
            "step.start",
            "",
            json!({"index": 3, "step": {"type": "google_search_call", "id": "gs1", "arguments": {"queries": ["rust"]}}}),
        ),
        event("step.stop", "", json!({"index": 3})),
        event(
            "step.start",
            "",
            json!({"index": 4, "step": {"type": "google_search_result", "call_id": "gs1", "result": [{"url": "https://b.example", "title": "B"}]}}),
        ),
        event("step.stop", "", json!({"index": 4})),
        event(
            "step.start",
            "",
            json!({"index": 5, "step": {"type": "processing_call", "id": "p1", "signature": "sig-p"}}),
        ),
        event("step.stop", "", json!({"index": 5})),
        event(
            "step.start",
            "",
            json!({"index": 6, "step": {"type": "model_output"}}),
        ),
        event(
            "step.delta",
            "",
            json!({"index": 6, "delta": {"type": "image", "data": "iVBORw0KGgo=", "mime_type": "image/png"}}),
        ),
        event("step.stop", "", json!({"index": 6})),
        event("step.stop", "", json!({"index": 2})),
        event(
            "interaction.completed",
            "",
            json!({"interaction": {"id": "int-1", "status": "completed", "service_tier": "standard", "usage": {"total_input_tokens": 3, "total_output_tokens": 2, "total_thought_tokens": 1}}}),
        ),
    ])]);
    let (handle, events) = call(
        &on_interactions("gemini-3-flash-preview", transport),
        hi(),
        StreamOptions::default(),
    )
    .await;
    ok(&events);
    let deltas: Vec<String> = events
        .iter()
        .filter_map(|e| match e {
            Ok(Event::TextDelta(d)) => Some(format!("text:{}", d.chunk)),
            Ok(Event::ReasoningDelta(d)) => Some(format!("reasoning:{}", d.chunk)),
            Ok(Event::ToolDelta(d)) => Some(format!("tool:{}:{}", d.tool_call_id, d.chunk)),
            _ => None,
        })
        .collect();
    assert_eq!(
        deltas,
        vec![
            "reasoning:plan",
            "text:Hel",
            "text:lo",
            "tool:f1:{\"q\":",
            "tool:f1:\"x\"}"
        ]
    );
    let int = |fields: Value| {
        let mut google = json!({"interactionId": "int-1"});
        google
            .as_object_mut()
            .unwrap()
            .extend(fields.as_object().unwrap().clone());
        json!({"google": google})
    };
    // Parts keep the order they started in: the function call starts at
    // its first argument delta, before the search call starts.
    let message = handle.message();
    let parts: Vec<Value> = message.parts.iter().map(shape).collect();
    assert_eq!(
        parts,
        vec![
            json!({"reasoning": "plan", "meta": int(json!({"signature": "sig-t"}))}),
            json!({"text": "Hello", "meta": int(json!({}))}),
            json!({"call": "lookup", "id": "f1", "args": {"q": "x"}, "meta": int(json!({"signature": "sig-f"}))}),
            json!({"builtin": "google_search", "id": "gs1", "args": {"queries": ["rust"]}, "meta": int(json!({}))}),
            json!({"result": "google_search", "id": "gs1", "value": [{"url": "https://b.example", "title": "B"}], "error": false, "meta": int(json!({}))}),
            json!({"builtin": "processing", "id": "p1", "args": {}, "meta": int(json!({"signature": "sig-p", "processingId": "p1"}))}),
            json!({"file": "image/png", "data": "iVBORw0KGgo=", "meta": int(json!({}))}),
        ]
    );
    assert_eq!(handle.finish_reason(), Some("tool_call"));
    assert_eq!(
        sources_of(message),
        json!([
            {"sourceType": "url", "url": "https://a.example", "title": "A"},
            {"sourceType": "url", "url": "https://b.example", "title": "B"},
        ])
    );
    let google = &message.provider_metadata.as_ref().unwrap()["google"];
    assert_eq!(google["serviceTier"], "standard");
    assert_eq!(message.usage.as_ref().unwrap().output_tokens, 3);
}

#[tokio::test]
async fn ggl_029_an_error_event_ends_the_call_with_its_message_and_code() {
    let transport = Recorder::new([Reply::sse(&[
        created(""),
        event(
            "error",
            "",
            json!({"error": {"message": "quota exhausted", "code": 429}}),
        ),
    ])]);
    let (handle, events) = call(
        &on_interactions("gemini-3-flash-preview", transport),
        hi(),
        StreamOptions::default(),
    )
    .await;
    match error(&events) {
        Error::Provider(error) => {
            assert_eq!(error.message, "quota exhausted");
            assert_eq!(error.code.as_deref(), Some("429"));
        }
        other => panic!("expected a provider error, got {other:?}"),
    }
    assert_eq!(handle.finish_reason(), None);
}

#[tokio::test]
async fn ggl_029_an_interrupted_stream_reports_no_finish_reason() {
    let transport = Recorder::new([Reply::sse(
        &[
            vec![
                created(""),
                event(
                    "interaction.status_update",
                    "",
                    json!({"status": "in_progress"}),
                ),
            ],
            text_steps(0, "partial", ""),
        ]
        .concat(),
    )]);
    let (handle, events) = call(
        &on_interactions("gemini-3-flash-preview", transport),
        hi(),
        StreamOptions::default(),
    )
    .await;
    assert_eq!(handle.finish_reason(), None);
    match error(&events) {
        Error::Provider(error) => assert_eq!(error.kind, ProviderErrorKind::IncompleteResponse),
        other => panic!("expected an incomplete response, got {other:?}"),
    }
}

// GGL-030

/// Let spawned tasks, such as a cancellation, run.
async fn settle() {
    for _ in 0..20 {
        tokio::task::yield_now().await;
    }
}

#[tokio::test(start_paused = true)]
async fn ggl_030_a_background_call_polls_after_one_then_two_seconds() {
    let transport = Recorder::new([in_progress(), in_progress(), done_reply()]);
    let message = suprnova_ai::generate(
        &on_agent(AGENT, transport.clone()),
        hi(),
        StreamOptions::default(),
    )
    .await
    .unwrap();
    assert_eq!(message.text(), "ok");
    let all = transport.all();
    assert_eq!(all.len(), 3);
    assert!(
        serde_json::from_slice::<Value>(&all[0].body)
            .unwrap()
            .get("stream")
            .is_none()
    );
    for request in &all[1..] {
        assert_eq!(
            (request.method.as_str(), request.url.as_str()),
            (
                "GET",
                "https://generativelanguage.googleapis.com/v1beta/interactions/int-1"
            )
        );
    }
    for request in &all {
        assert_revision(request, REVISION);
    }
    let times = transport.times();
    assert_eq!(times[1] - times[0], Duration::from_secs(1));
    assert_eq!(times[2] - times[1], Duration::from_secs(2));

    let transport = Recorder::new([in_progress(), done_reply()]);
    let options = google_options("google", json!({"background": true}));
    suprnova_ai::generate(
        &on_interactions("gemini-3-flash-preview", transport.clone()),
        hi(),
        options,
    )
    .await
    .unwrap();
    let body = transport.nth_body(0);
    assert_eq!(
        (body.get("stream"), &body["background"]),
        (None, &json!(true))
    );
    assert_eq!(transport.count(), 2);
}

#[tokio::test(start_paused = true)]
async fn ggl_030_a_poll_that_times_out_cancels_the_interaction() {
    let transport = Recorder::new([
        in_progress(),
        in_progress(),
        in_progress(),
        Reply::json(json!({})),
    ]);
    let options = google_options("google", json!({"pollingTimeoutMs": 2500}));
    let error = suprnova_ai::generate(&on_agent(AGENT, transport.clone()), hi(), options)
        .await
        .unwrap_err();
    match error {
        Error::Provider(error) => {
            assert_eq!(error.kind, ProviderErrorKind::Timeout);
            assert!(!error.is_retryable, "a retry would start a new interaction");
        }
        other => panic!("expected a timeout, got {other:?}"),
    }
    let all = transport.all();
    assert_eq!(all.len(), 4, "create, two polls and the cancel");
    let cancel = all.last().unwrap();
    assert_eq!(
        (cancel.method.as_str(), cancel.url.as_str()),
        (
            "POST",
            "https://generativelanguage.googleapis.com/v1beta/interactions/int-1/cancel"
        )
    );
    assert_revision(cancel, REVISION);
}

#[tokio::test(start_paused = true)]
async fn ggl_030_a_dropped_call_cancels_the_interaction() {
    let transport = Recorder::new([in_progress(), in_progress(), Reply::json(json!({}))]);
    let m = on_agent(AGENT, transport.clone());
    let call_future = suprnova_ai::generate(&m, hi(), StreamOptions::default());
    assert!(
        tokio::time::timeout(Duration::from_millis(1500), call_future)
            .await
            .is_err()
    );
    settle().await;
    let all = transport.all();
    assert_eq!(
        all.len(),
        3,
        "{:?}",
        all.iter().map(|r| &r.url).collect::<Vec<_>>()
    );
    assert_eq!(
        (all[2].method.as_str(), all[2].url.as_str()),
        (
            "POST",
            "https://generativelanguage.googleapis.com/v1beta/interactions/int-1/cancel"
        )
    );
    assert_revision(&all[2], REVISION);
}

#[tokio::test(start_paused = true)]
async fn ggl_030_a_stream_follows_the_interaction_and_reconnects_after_a_drop() {
    let transport = Recorder::new([
        in_progress(),
        Reply::sse(&[
            created("e1"),
            event(
                "step.start",
                "e2",
                json!({"index": 0, "step": {"type": "model_output"}}),
            ),
        ]),
        Reply::sse(&[
            event(
                "step.delta",
                "e3",
                json!({"index": 0, "delta": {"type": "text", "text": "Hello"}}),
            ),
            event("step.stop", "e4", json!({"index": 0})),
            completed("e5"),
        ]),
    ]);
    let (handle, events) = call(
        &on_agent(AGENT, transport.clone()),
        hi(),
        StreamOptions::default(),
    )
    .await;
    ok(&events);
    assert_eq!(handle.text(), "Hello");
    assert_eq!(handle.finish_reason(), Some("stop"));
    settle().await;
    let all = transport.all();
    assert_eq!(all.len(), 3, "a finished stream sent another request");
    assert_eq!(
        all[1].url,
        "https://generativelanguage.googleapis.com/v1beta/interactions/int-1?stream=true"
    );
    assert_eq!(
        all[2].url,
        "https://generativelanguage.googleapis.com/v1beta/interactions/int-1?last_event_id=e2&stream=true"
    );
    for request in &all[1..] {
        assert_eq!(request.method, "GET");
        assert_eq!(
            header(request, "accept").as_deref(),
            Some("text/event-stream")
        );
    }
    for request in &all {
        assert_revision(request, REVISION);
    }
}

#[tokio::test(start_paused = true)]
async fn ggl_030_a_stream_gives_up_after_three_failed_reconnections() {
    let unavailable = || {
        Reply::status(
            503,
            json!({"error": {"code": 503, "message": "unavailable", "status": "UNAVAILABLE"}}),
        )
    };
    let transport = Recorder::new([
        in_progress(),
        Reply::sse(&[created("e1")]),
        unavailable(),
        unavailable(),
        unavailable(),
        Reply::json(json!({})),
    ]);
    let (_, events) = call(
        &on_agent(AGENT, transport.clone()),
        hi(),
        StreamOptions::default(),
    )
    .await;
    assert!(
        matches!(error(&events), Error::Provider(_)),
        "{:?}",
        error(&events)
    );
    settle().await;
    let all = transport.all();
    let streams = all.iter().filter(|r| r.method == "GET").count();
    assert_eq!(streams, 4, "one stream and three reconnections");
    assert!(
        all.iter()
            .filter(|r| r.method == "GET")
            .skip(1)
            .all(|r| r.url.contains("last_event_id=e1"))
    );
    let cancel = all.last().unwrap();
    assert_eq!(
        (cancel.method.as_str(), cancel.url.as_str()),
        (
            "POST",
            "https://generativelanguage.googleapis.com/v1beta/interactions/int-1/cancel"
        )
    );
    assert_eq!(all.len(), 6);
}

#[tokio::test]
async fn ggl_030_a_terminal_first_response_is_replayed_as_one_stream() {
    let transport = Recorder::new([done_reply()]);
    let (handle, events) = call(
        &on_agent(AGENT, transport.clone()),
        hi(),
        StreamOptions::default(),
    )
    .await;
    ok(&events);
    assert_eq!(handle.text(), "ok");
    assert_eq!(handle.finish_reason(), Some("stop"));
    settle().await;
    assert_eq!(transport.count(), 1);
}

#[tokio::test(start_paused = true)]
async fn ggl_030_a_dropped_stream_cancels_the_interaction() {
    let transport = Recorder::new([
        in_progress(),
        Reply::sse(&[vec![created("e1")], text_steps(0, "Hel", "e2")].concat()),
        Reply::json(json!({})),
    ]);
    let mut handle = suprnova_ai::stream(
        &on_agent(AGENT, transport.clone()),
        hi(),
        StreamOptions::default(),
    );
    loop {
        match handle.next().await {
            Some(Ok(Event::TextDelta(_))) => break,
            Some(Ok(_)) => {}
            other => panic!("expected a text delta, got {other:?}"),
        }
    }
    drop(handle);
    settle().await;
    let all = transport.all();
    let cancel = all.last().unwrap();
    assert_eq!(
        (cancel.method.as_str(), cancel.url.as_str()),
        (
            "POST",
            "https://generativelanguage.googleapis.com/v1beta/interactions/int-1/cancel"
        )
    );
    assert_revision(cancel, REVISION);
}

// GGL-031

#[tokio::test]
async fn ggl_031_evaluation_is_one_buffered_call_with_the_decision_schema() {
    let transport = Recorder::new([text_response(r#"{"q0":"c1","q1":0.75}"#)]);
    let questions = BTreeMap::from([
        (
            "mood".to_owned(),
            Question::Choice {
                instructions: EvaluationInput::Text("How does it sound?".into()),
                criteria: BTreeMap::from([("calm".to_owned(), None), ("tense".to_owned(), None)]),
            },
        ),
        (
            "spam".to_owned(),
            Question::Noul {
                instructions: EvaluationInput::Text("Is it spam?".into()),
                criteria: None,
            },
        ),
    ]);
    let item = ops::experimental::evaluate(
        &model("gemini-3-flash-preview", transport.clone()),
        EvaluationInput::Text("hello".into()),
        questions,
        EvaluationParams::default(),
        Default::default(),
    )
    .await
    .unwrap();
    assert_eq!(
        transport.request().url,
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent"
    );
    let body = transport.body();
    let config = &body["generationConfig"];
    assert_eq!(
        config["thinkingConfig"],
        json!({"thinkingLevel": "minimal"})
    );
    assert_eq!(config["responseMimeType"], "application/json");
    let schema = &config["responseJsonSchema"];
    assert_eq!(
        schema["properties"]["q0"],
        json!({"type": "string", "enum": ["c0", "c1"]})
    );
    assert_eq!(schema["properties"]["q1"]["type"], "number");
    assert_eq!(schema["required"], json!(["q0", "q1"]));
    assert_eq!(schema["additionalProperties"], false);
    assert!(
        body["systemInstruction"]["parts"][0]["text"]
            .as_str()
            .unwrap()
            .starts_with("Decide every answer")
    );
    assert_eq!(
        json!(item.value),
        json!({"mood": {"type": "choice", "choice": "tense"}, "spam": {"type": "noul", "noul": 0.75}})
    );
}

// GGL-032

const MIB: usize = 1024 * 1024;

fn file_len(file: &FilePart) -> usize {
    match &file.data {
        FileData::Text(text) => text.len(),
        FileData::Bytes(bytes) => bytes.len(),
    }
}

fn cat() -> ImagePrompt {
    ImagePrompt {
        text: Some("a cat".into()),
        images: vec![],
        mask: None,
    }
}

#[tokio::test]
async fn ggl_032_large_media_events_and_bodies_are_read_whole() {
    let data = "A".repeat(12 * MIB);
    let transport = Recorder::with_chunks(
        [Reply::sse(&[
            json!({"candidates": [{"content": {"role": "model", "parts": [{"inlineData": {"mimeType": "image/png", "data": data}}]}, "finishReason": "STOP"}]}),
        ])],
        MIB,
    );
    let (handle, events) = call(
        &model("gemini-2.5-flash-image", transport),
        hi(),
        StreamOptions::default(),
    )
    .await;
    ok(&events);
    let files: Vec<usize> = handle
        .message()
        .parts
        .iter()
        .filter_map(|p| match p {
            Part::File(file) => Some(file_len(file)),
            _ => None,
        })
        .collect();
    assert_eq!(files, vec![12 * MIB]);

    let image = "A".repeat(20 * MIB);
    let transport = Recorder::with_chunks(
        [Reply::json(
            json!({"candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": image}}]}, "finishReason": "STOP"}]}),
        )],
        MIB,
    );
    let item = ops::generate_image(
        &model("gemini-2.5-flash-image", transport),
        cat(),
        ImageParams::default(),
        Default::default(),
    )
    .await
    .unwrap();
    assert_eq!(file_len(&item.value[0]), 20 * MIB);

    let transport = Recorder::with_chunks(
        [audio_response("audio/L16;rate=24000", &vec![0u8; 15 * MIB])],
        MIB,
    );
    let prompt = AudioPrompt {
        text: "hello".into(),
        instructions: None,
    };
    let item = ops::generate_audio(
        &model("gemini-2.5-flash-preview-tts", transport),
        prompt,
        AudioParams::default(),
        Default::default(),
    )
    .await
    .unwrap();
    assert_eq!(file_len(&item.value[0]), 15 * MIB + 44);
}

fn names_limit(error: &Error, limit: &str) {
    match error {
        Error::Provider(error) => assert!(error.message.contains(limit), "{}", error.message),
        other => panic!("expected a provider error naming {limit}, got {other:?}"),
    }
}

#[tokio::test]
async fn ggl_032_a_body_or_event_over_256_mib_fails_without_reading_past_it() {
    let (reply, read) = Reply::generated(300 * MIB);
    let transport = Recorder::with_chunks([reply], MIB);
    let failure = ops::generate_image(
        &model("gemini-2.5-flash-image", transport),
        cat(),
        ImageParams::default(),
        Default::default(),
    )
    .await
    .unwrap_err();
    names_limit(&failure, "256 MiB");
    assert!(read.load(std::sync::atomic::Ordering::SeqCst) <= 257 * MIB);

    let (reply, read) = Reply::generated(300 * MIB);
    let transport = Recorder::with_chunks([reply], MIB);
    let (_, events) = call(
        &on_interactions("gemini-3-flash-preview", transport),
        hi(),
        StreamOptions::default(),
    )
    .await;
    names_limit(error(&events), "256 MiB");
    assert!(read.load(std::sync::atomic::Ordering::SeqCst) <= 257 * MIB);
}

#[tokio::test]
async fn ggl_032_other_google_responses_keep_the_8_mib_limit() {
    let pad = "x".repeat(12 * MIB);
    let transport = Recorder::with_chunks(
        [Reply::json(
            json!({"name": "batches/b1", "metadata": {"state": "BATCH_STATE_RUNNING", "pad": pad}}),
        )],
        MIB,
    );
    let error = ops::batch::status(
        google_provider(transport).as_ref(),
        "batches/b1",
        Default::default(),
    )
    .await
    .unwrap_err();
    names_limit(&error, "8 MiB");

    let transport = Recorder::with_chunks(
        [
            Reply::json(json!({})).header("x-goog-upload-url", "https://upload.example/u1"),
            Reply::json(json!({"file": {"name": "files/f1", "state": "ACTIVE", "pad": pad}}))
                .header("x-goog-upload-status", "final"),
        ],
        MIB,
    );
    let file = UploadFile {
        data: bytes::Bytes::from_static(b"%PDF"),
        media_type: "application/pdf".into(),
        filename: None,
    };
    let error = ops::upload_file(
        google_provider(transport).as_ref(),
        file,
        UploadParams::default(),
        Default::default(),
    )
    .await
    .unwrap_err();
    names_limit(&error, "8 MiB");
}
