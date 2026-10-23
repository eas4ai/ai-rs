use futures::StreamExt;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};
use suprnova_ai::agents::{AgentTool, BoundToolCall, HookRegistry, Runtime, ToolContext};
use suprnova_ai::types::messages::{HookStatus, ToolCallPart};
use suprnova_ai::types::{Event, HookPart};

#[derive(Deserialize, JsonSchema)]
struct Edit {
    #[serde(rename = "oldText")]
    old_text: String,
}
#[derive(Deserialize, JsonSchema)]
struct Edits {
    path: String,
    edits: Vec<Edit>,
}

fn edit_tool() -> AgentTool {
    AgentTool::new("edit", "Edit a file.", |args: Edits, _| async move {
        Ok(format!(
            "{}:{}:{}",
            args.path,
            args.edits.len(),
            args.edits[0].old_text
        ))
    })
    .unwrap()
    .require_approval(true)
}

#[tokio::test]
async fn invalid_model_args_bypass_approval_hook() {
    for args in [
        r#"{"path":"test.txt","edits":{"oldText":"foo"}}"#,
        r#"{"path":"test.txt","edits": "#,
    ] {
        let (runtime, mut events) = Runtime::new();
        let result = BoundToolCall::new(ToolCallPart::new("tc", "edit", args), edit_tool())
            .execute(ToolContext::new(
                runtime.clone(),
                HookRegistry::default(),
                "tc",
                "edit",
            ))
            .await
            .unwrap();
        assert!(result.results[0].is_error());
        runtime.signal_done().await;
        assert!(
            events.next().await.is_none(),
            "invalid args must not emit approval hooks"
        );
    }
}

#[tokio::test]
async fn valid_arguments_wait_for_approval_then_execute_or_deny() {
    for granted in [true, false] {
        let (runtime, mut events) = Runtime::new();
        let hooks = HookRegistry::default();
        let context = ToolContext::new(runtime, hooks.clone(), "tc", "edit");
        let task = tokio::spawn(async move {
            BoundToolCall::new(
                ToolCallPart::new(
                    "tc",
                    "edit",
                    r#"{"path":"test.txt","edits":[{"oldText":"foo"}]}"#,
                ),
                edit_tool(),
            )
            .execute(context)
            .await
        });
        let Event::Hook(hook) = events.next().await.unwrap() else {
            panic!("expected approval")
        };
        assert_eq!(hook.hook.hook_id, "approve_tc");
        assert_eq!(hook.hook.metadata["tool"], "edit");
        assert_eq!(hook.hook.metadata["kwargs"]["path"], "test.txt");
        assert!(!task.is_finished());
        hooks
            .resolve_hook("approve_tc", json!({"granted":granted,"reason":"review"}))
            .unwrap();
        let result = task.await.unwrap().unwrap();
        assert_eq!(result.results[0].is_error(), !granted);
        if granted {
            assert_eq!(
                result.results[0].result.as_json(),
                Some(&json!("test.txt:1:foo"))
            );
        } else {
            assert_eq!(
                result.results[0].result.as_json(),
                Some(&json!("Rejected: review"))
            );
        }
    }
}

#[tokio::test]
async fn approval_deferral_returns_replayable_placeholder() {
    let hooks = HookRegistry::default();
    let hook = HookPart {
        id: "part".into(),
        hook_id: "approve_tc".into(),
        hook_type: "ToolApproval".into(),
        status: HookStatus::Pending,
        metadata: Default::default(),
        resolution: None,
        tool_call_id: Some("tc".into()),
    };
    hooks.defer_hook("approve_tc", hook).unwrap();
    let (runtime, _events) = Runtime::new();
    let result = BoundToolCall::new(
        ToolCallPart::new(
            "tc",
            "edit",
            r#"{"path":"test.txt","edits":[{"oldText":"foo"}]}"#,
        ),
        edit_tool(),
    )
    .execute(ToolContext::new(runtime, hooks, "tc", "edit"))
    .await
    .unwrap();
    assert!(result.results[0].is_hook_deferred);
    assert!(result.results[0].is_error());
}

#[tokio::test]
async fn hooks_inside_tool_bodies_preserve_deferral() {
    let tool = AgentTool::new_with_hooks("ask", "Ask.", |_: Value, context| async move {
        let result: Value = context.hook("input", Default::default()).await?;
        Ok(result)
    })
    .unwrap();
    let hooks = HookRegistry::default();
    hooks
        .defer_hook(
            "input",
            HookPart {
                id: "part".into(),
                hook_id: "input".into(),
                hook_type: "Value".into(),
                status: HookStatus::Pending,
                metadata: Default::default(),
                resolution: None,
                tool_call_id: Some("tc".into()),
            },
        )
        .unwrap();
    let (runtime, _events) = Runtime::new();
    let result = BoundToolCall::new(ToolCallPart::new("tc", "ask", "{}"), tool)
        .execute(ToolContext::new(runtime, hooks, "tc", "ask"))
        .await
        .unwrap();
    assert!(result.results[0].is_hook_deferred);
}
