mod anthropic_support;
use anthropic_support::*;
use serde_json::json;
use suprnova_ai::{
    StreamOptions, models::params::*, providers::anthropic::tools::*, types::tools::*,
};
#[tokio::test]
async fn web_search_full_fields_and_beta() {
    let tool = web_search(WebSearchOptions {
        max_uses: Some(3),
        allowed_domains: Some(vec!["example.com".into()]),
        user_location: Some(UserLocation {
            city: Some("SF".into()),
            country: Some("US".into()),
            ..Default::default()
        }),
        ..Default::default()
    })
    .unwrap();
    let (b, r) = capture(vec![], StreamOptions::default().with_tools([tool])).await;
    assert_eq!(
        b["tools"],
        json!([{"type":"web_search_20260209","name":"web_search","max_uses":3,"allowed_domains":["example.com"],"user_location":{"type":"approximate","city":"SF","country":"US"}}])
    );
    assert_eq!(
        header(&r, "anthropic-beta").as_deref(),
        Some("code-execution-web-tools-2026-02-09")
    );
}
#[tokio::test]
async fn all_builtin_tools_and_custom_function_schema() {
    let tools = vec![
        web_fetch(WebFetchOptions {
            citations: Some(true.into()),
            ..Default::default()
        })
        .unwrap(),
        computer_use(ComputerUseOptions {
            display_width_px: 1024,
            display_height_px: 768,
            display_number: None,
            enable_zoom: None,
        }),
        text_editor(None),
        code_execution(),
        bash(),
        memory(),
        Tool::function(
            "weather",
            ToolSpec {
                description: None,
                params: serde_json::from_value(json!({"type":"object","properties":{}})).unwrap(),
            },
        ),
    ];
    let (b, r) = capture(vec![], StreamOptions::default().with_tools(tools)).await;
    assert_eq!(
        b["tools"][0],
        json!({"name":"weather","description":"","input_schema":{"type":"object","properties":{}}})
    );
    assert_eq!(
        b["tools"][1],
        json!({"type":"web_fetch_20260209","name":"web_fetch","citations":{"enabled":true}})
    );
    assert_eq!(
        b["tools"][2],
        json!({"type":"computer_20251124","name":"computer","display_width_px":1024,"display_height_px":768})
    );
    assert_eq!(
        b["tools"][3],
        json!({"type":"text_editor_20250728","name":"str_replace_based_edit_tool"})
    );
    assert_eq!(
        b["tools"][4],
        json!({"type":"code_execution_20260120","name":"code_execution"})
    );
    assert_eq!(b["tools"][5], json!({"type":"bash_20250124","name":"bash"}));
    assert_eq!(
        b["tools"][6],
        json!({"type":"memory_20250818","name":"memory"})
    );
    assert_eq!(
        header(&r, "anthropic-beta").as_deref(),
        Some(
            "code-execution-web-tools-2026-02-09,computer-use-2025-01-24,computer-use-2025-11-24,context-management-2025-06-27"
        )
    );
}
#[tokio::test]
async fn caller_beta_wins_case_insensitively() {
    let params = InferenceRequestParams {
        extra_headers: Some(
            [(
                "Anthropic-Beta".into(),
                ExtraHeaderValue::Value("custom-beta".into()),
            )]
            .into(),
        ),
        ..Default::default()
    };
    let (_, r) = capture(
        vec![],
        StreamOptions {
            params: Some(params),
            tools: vec![
                web_search(Default::default()).unwrap(),
                web_fetch(Default::default()).unwrap(),
            ],
            ..Default::default()
        },
    )
    .await;
    assert_eq!(header(&r, "anthropic-beta").as_deref(), Some("custom-beta"));
}
#[test]
fn domain_filters_are_mutually_exclusive_only_when_nonempty() {
    assert!(
        web_search(WebSearchOptions {
            allowed_domains: Some(vec!["a".into()]),
            blocked_domains: Some(vec!["b".into()]),
            ..Default::default()
        })
        .is_err()
    );
    assert!(
        web_fetch(WebFetchOptions {
            allowed_domains: Some(vec!["a".into()]),
            blocked_domains: Some(vec!["b".into()]),
            ..Default::default()
        })
        .is_err()
    );
    assert!(
        web_search(WebSearchOptions {
            allowed_domains: Some(vec![]),
            blocked_domains: Some(vec!["b".into()]),
            ..Default::default()
        })
        .is_ok()
    );
}
