mod anthropic_support;
use anthropic_support::*;
use futures::StreamExt;
use serde_json::{Value, json};
use suprnova_ai::{
    StreamOptions,
    models::params::*,
    types::{messages::*, tools::*},
};

#[tokio::test]
async fn system_messages_go_first_or_in_place_and_roles_merge() {
    // The first system message is the top-level system text; a later one
    // goes in place (ANT-011). The model's limit is the default output
    // limit (ANT-003).
    let (body, _) = capture(
        vec![
            suprnova_ai::system_message("First instruction."),
            suprnova_ai::user_message("Hi"),
            suprnova_ai::user_message("Again"),
            suprnova_ai::system_message("Second instruction."),
            suprnova_ai::user_message("Last"),
        ],
        StreamOptions::default(),
    )
    .await;
    assert_eq!(
        body["system"],
        json!([{"type":"text","text":"First instruction."}])
    );
    assert_eq!(
        body["messages"],
        json!([
            {"role":"user","content":[{"type":"text","text":"Hi"},{"type":"text","text":"Again"}]},
            {"role":"system","content":[{"type":"text","text":"Second instruction."}]},
            {"role":"user","content":[{"type":"text","text":"Last"}]}
        ])
    );
    assert_eq!(body["max_tokens"], 128000);
    assert_eq!(body["stream"], true);
}
#[tokio::test]
async fn params_translate_to_http_body_and_headers() {
    let params = InferenceRequestParams {
        output: Some(OutputParams {
            max_tokens: Some(123),
            reasoning_summary: ProviderValue::Value(None),
            ..Default::default()
        }),
        reasoning: ProviderValue::Value(ReasoningParams {
            effort: ProviderValue::Value(Some("high".into())),
        }),
        context_management: Some(ContextManagementParams {
            compaction: Some(TokenThreshold(120_000)),
        }),
        tool_calling: Some(ToolCallingParams {
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: ProviderValue::Value(false),
            max_tool_calls: ProviderValue::Default,
        }),
        extra_body: Some(
            serde_json::from_value(json!({"speed":"fast","future_option":{"enabled":true}}))
                .unwrap(),
        ),
        extra_headers: Some(
            [
                (
                    "x-anthropic-feature".into(),
                    ExtraHeaderValue::Value("enabled".into()),
                ),
                ("removed".into(), ExtraHeaderValue::Unset),
            ]
            .into(),
        ),
        extra_query: Some([("tag".into(), json!("a b"))].into()),
        ..Default::default()
    };
    // A tool choice goes only with tools.
    let weather = Tool::function(
        "weather",
        ToolSpec {
            description: None,
            params: serde_json::from_value(json!({"type":"object"})).unwrap(),
        },
    );
    let (body, request) = capture(
        vec![suprnova_ai::user_message("Hi")],
        StreamOptions {
            params: Some(params),
            tools: vec![weather],
            ..Default::default()
        },
    )
    .await;
    assert_eq!(body["max_tokens"], 123);
    assert_eq!(
        body["thinking"],
        json!({"type":"adaptive","display":"omitted"})
    );
    assert_eq!(body["output_config"], json!({"effort":"high"}));
    assert_eq!(
        body["tool_choice"],
        json!({"type":"auto","disable_parallel_tool_use":true})
    );
    assert_eq!(
        body["context_management"],
        json!({"edits":[{"type":"compact_20260112","trigger":{"type":"input_tokens","value":120000}}]})
    );
    assert_eq!(body["future_option"], json!({"enabled":true}));
    assert_eq!(body["speed"], "fast");
    assert_eq!(
        header(&request, "anthropic-beta").as_deref(),
        Some("context-management-2025-06-27,compact-2026-01-12,structured-outputs-2025-11-13")
    );
    assert_eq!(
        header(&request, "x-anthropic-feature").as_deref(),
        Some("enabled")
    );
    assert_eq!(header(&request, "removed"), None);
    assert!(request.url.ends_with("?tag=a+b"));
}
#[tokio::test]
async fn explicit_seed_is_rejected_and_random_seeds_are_omitted() {
    for seed in [Seed::Value(123), Seed::Value(-1), Seed::Random] {
        let params = InferenceRequestParams {
            sampling: ProviderValue::Value(SamplerParams {
                seed: Some(SeedSamplerParams {
                    seed: ProviderValue::Value(Some(seed.clone())),
                }),
                ..Default::default()
            }),
            ..Default::default()
        };
        if seed == Seed::Value(123) {
            let fake = ScriptedTransport::new(vec![]);
            let m = model(fake.clone());
            let err = m
                .provider
                .stream(
                    &m,
                    vec![],
                    StreamOptions {
                        params: Some(params),
                        ..Default::default()
                    },
                )
                .next()
                .await
                .unwrap()
                .unwrap_err();
            assert!(err.to_string().contains("seed"));
            assert!(fake.requests.lock().unwrap().is_empty());
        } else {
            let (b, _) = capture(
                vec![],
                StreamOptions {
                    params: Some(params),
                    ..Default::default()
                },
            )
            .await;
            assert!(b.get("seed").is_none());
        }
    }
}
#[tokio::test]
async fn signed_reasoning_and_builtin_tools_round_trip() {
    let parts:Vec<Part>=serde_json::from_value(json!([
        {"kind":"reasoning","text":"hidden","provider_metadata":{"anthropic":{"signature":"sig"}}},
        {"kind":"reasoning","text":"unsigned"},
        {"kind":"builtin_tool_call","tool_call_id":"srv1","tool_name":"web_search","tool_args":"{\"query\":\"weather\"}"},
        {"kind":"builtin_tool_return","tool_call_id":"srv1","tool_name":"web_search","result":[{"title":"Forecast"}],"provider_metadata":{"anthropic":{"resultType":"web_search_tool_result"}}}
    ])).unwrap();
    let (b, _) = capture(
        vec![
            Message::new(MessageRole::Assistant, parts),
            suprnova_ai::user_message("Hi"),
        ],
        StreamOptions::default(),
    )
    .await;
    assert_eq!(
        b["messages"][0]["content"],
        json!([{"type":"thinking","thinking":"hidden","signature":"sig"},{"type":"server_tool_use","id":"srv1","name":"web_search","input":{"query":"weather"}},{"type":"web_search_tool_result","tool_use_id":"srv1","content":[{"title":"Forecast"}]}])
    );
}
#[tokio::test]
async fn history_repair_closes_calls_and_rejects_broken_ids() {
    let call = ToolCallPart::new("tc-1", "search", "broken json");
    let (b, _) = capture(
        vec![
            Message::new(
                MessageRole::Internal,
                vec![TextPart::new("app-only").into()],
            ),
            Message::new(MessageRole::Assistant, vec![call.clone().into()]),
        ],
        StreamOptions::default(),
    )
    .await;
    // Arguments that are not a JSON object go as raw invalid input (ANT-009).
    assert_eq!(
        b["messages"][0]["content"][0]["input"],
        json!({"rawInvalidInput": "broken json"})
    );
    assert_eq!(
        b["messages"][1]["content"][0],
        json!({"type":"tool_result","tool_use_id":"tc-1","content":"Tool result not available","is_error":true})
    );
    for history in [
        vec![Message::new(
            MessageRole::Assistant,
            vec![call.clone().into(), call.into()],
        )],
        vec![Message::new(
            MessageRole::Tool,
            vec![ToolResultPart::new("orphan", "search", json!("x")).into()],
        )],
    ] {
        let fake = ScriptedTransport::new(vec![]);
        let m = model(fake.clone());
        assert!(
            m.provider
                .stream(&m, history, StreamOptions::default())
                .next()
                .await
                .unwrap()
                .is_err()
        );
        assert!(fake.requests.lock().unwrap().is_empty());
    }
}
#[tokio::test]
async fn sampling_metadata_cache_and_body_overrides() {
    let params = InferenceRequestParams {
        sampling: ProviderValue::Value(SamplerParams {
            temperature: Some(TemperatureSamplerParams {
                temperature: ProviderValue::Value(0.7),
            }),
            top_p: Some(TopPSamplerParams {
                top_p: ProviderValue::Value(None),
            }),
            top_k: Some(TopKSamplerParams {
                top_k: ProviderValue::Value(Some(7)),
            }),
            ..Default::default()
        }),
        safety_identifier: Some("user".into()),
        cache: Some(CacheParams {
            retention: ProviderValue::Value("1h".into()),
            ..Default::default()
        }),
        provider_service: Some(ProviderServiceParams {
            service_tier: ProviderValue::Value("auto".into()),
        }),
        extra_body: Some([("max_tokens".into(), json!(900))].into()),
        ..Default::default()
    };
    let (b, _) = capture(
        vec![],
        StreamOptions {
            params: Some(params),
            ..Default::default()
        },
    )
    .await;
    assert_eq!(b["temperature"], 0.7);
    assert!(b["top_p"].is_null());
    assert_eq!(b["top_k"], 7);
    assert_eq!(b["metadata"], json!({"user_id":"user"}));
    assert_eq!(b["cache_control"], json!({"type":"ephemeral","ttl":"1h"}));
    assert_eq!(b["service_tier"], "auto");
    assert_eq!(b["max_tokens"], 900);
}
#[tokio::test]
async fn unsupported_options_fail_before_http() {
    let cases: Vec<Value> = vec![
        json!({"sampling":{"value":{"top_k":{"top_k":{"value":null}}}}}),
        json!({"sampling":{"value":{"min_p":{"min_p":{"value":0.1}}}}}),
        json!({"output":{"include":[]}}),
        json!({"output":{"text_verbosity":{"value":"high"}}}),
        json!({"cache":{"key":"key"}}),
        json!({"cache":{"mode":{"value":"auto"}}}),
        json!({"metadata":{"a":"b"}}),
        json!({"tool_calling":{"tool_choice":"auto","max_tool_calls":{"value":1}}}),
        json!({"tool_calling":{"tool_choice":"none","parallel_tool_calls":{"value":false}}}),
        json!({"tool_calling":{"tool_choice":{"tools":["a","b"],"mode":"auto"}}}),
    ];
    for case in cases {
        let params = serde_json::from_value(case.clone()).unwrap();
        let fake = ScriptedTransport::new(vec![]);
        let m = model(fake.clone());
        let err = m
            .provider
            .stream(
                &m,
                vec![],
                StreamOptions {
                    params: Some(params),
                    ..Default::default()
                },
            )
            .next()
            .await
            .unwrap()
            .unwrap_err();
        assert!(
            matches!(err, suprnova_ai::Error::Validation(_)),
            "{case}: {err:?}"
        );
        assert!(fake.requests.lock().unwrap().is_empty());
    }
}

#[tokio::test]
async fn tool_choices_reasoning_disabled_and_provider_tool_rejection() {
    for (choice, expected) in [
        (ToolChoice::Required, json!({"type":"any"})),
        (ToolChoice::None, json!({"type":"none"})),
        (
            ToolChoice::Tool("weather".into()),
            json!({"type":"tool","name":"weather"}),
        ),
        (
            ToolSelection::new(["weather"], ToolChoiceMode::Required).into(),
            json!({"type":"tool","name":"weather"}),
        ),
    ] {
        let weather = Tool::function(
            "weather",
            ToolSpec {
                description: None,
                params: serde_json::from_value(json!({"type":"object"})).unwrap(),
            },
        );
        let (b, _) = capture(
            vec![],
            StreamOptions {
                tool_choice: Some(choice),
                tools: vec![weather],
                ..Default::default()
            },
        )
        .await;
        assert_eq!(b["tool_choice"], expected);
    }
    let params =
        InferenceRequestParams::default().with_reasoning_effort(ProviderValue::Value(None));
    let params = InferenceRequestParams {
        output: Some(OutputParams {
            reasoning_summary: ProviderValue::Value(Some("concise".into())),
            ..Default::default()
        }),
        ..params
    };
    let (b, _) = capture(
        vec![],
        StreamOptions {
            params: Some(params),
            ..Default::default()
        },
    )
    .await;
    assert_eq!(b["thinking"], json!({"type":"disabled"}));
    assert!(b.get("output_config").is_none());
    let fake = ScriptedTransport::new(vec![]);
    let m = model(fake.clone());
    let other = Tool::provider(
        "search",
        ToolConfig {
            id: Some("openai.web_search".into()),
            ..Default::default()
        },
    )
    .unwrap();
    let err = m
        .provider
        .stream(&m, vec![], StreamOptions::default().with_tools([other]))
        .next()
        .await
        .unwrap()
        .unwrap_err();
    assert!(err.to_string().contains("openai.web_search"));
    assert!(fake.requests.lock().unwrap().is_empty());
}
