use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use suprnova_ai::Error;
use suprnova_ai::agents::{
    AgentTool, BoundToolCall, HookRegistry, Runtime, ToolContext, ToolResolver,
};
use suprnova_ai::types::messages::{
    MessageBundle, MessageRole, SpecialToolResult, ToolCallPart, ToolResultPart, ToolResultValue,
};

#[derive(Deserialize, Serialize, JsonSchema)]
struct Add {
    a: i64,
    b: i64,
}

fn add_tool() -> AgentTool {
    AgentTool::new("add", "Add two numbers.", |args: Add, context| async move {
        assert_eq!(context.tool_name, "add");
        Ok(args.a + args.b)
    })
    .unwrap()
}

fn context() -> ToolContext {
    let (runtime, _events) = Runtime::new();
    ToolContext::new(runtime, HookRegistry::default(), "tc-1", "add")
}

#[test]
fn standalone_context_has_no_owner_or_inherited_bound_call() {
    let context = context();
    assert!(context.agent.is_none());
    assert!(context.bound_call.is_none());
    let nested = context.nested_options();
    assert!(nested.hook_registry.is_some());
    assert!(nested.middleware.is_empty());
    assert!(nested.stream_options.tools.is_empty());
}

#[tokio::test]
async fn executor_inspects_original_bound_call_through_overrides_and_middleware() {
    use std::sync::Arc;
    use suprnova_ai::agents::Middleware;
    use suprnova_ai::agents::middleware::{ToolContext as MiddlewareContext, ToolNext};
    use suprnova_ai::types::events::ToolCallResult;

    struct Rewrite;
    #[async_trait::async_trait]
    impl Middleware for Rewrite {
        async fn wrap_tool(
            &self,
            mut call: MiddlewareContext,
            next: ToolNext,
        ) -> Result<ToolCallResult, Error> {
            call.kwargs["a"] = json!(20);
            call.tool_call_id = "rewritten-id".into();
            call.tool_name = "rewritten-name".into();
            next(call).await
        }
    }
    let tool = AgentTool::new(
        "add",
        "Inspect bound execution.",
        |args: Add, context| async move {
            assert!(context.agent.is_none());
            let bound = context
                .bound_call
                .as_ref()
                .expect("bound execution metadata");
            Ok(json!({
                "current_id":context.tool_call_id,
                "current_name":context.tool_name,
                "original_id":bound.id(),
                "original_name":bound.name(),
                "original_args":bound.arguments()?,
                "bound_tool":bound.tool.name(),
                "actual_args":{"a":args.a,"b":args.b},
            }))
        },
    )
    .unwrap();
    let call = BoundToolCall::new(ToolCallPart::new("tc-1", "add", r#"{"a":1,"b":2}"#), tool);
    let plain = call.execute(context()).await.unwrap();
    assert_eq!(
        plain.results[0].result.as_json().unwrap()["original_args"],
        json!({"a":1,"b":2})
    );
    assert_eq!(
        plain.results[0].result.as_json().unwrap()["actual_args"],
        json!({"a":1,"b":2})
    );
    for replace_all in [false, true] {
        let mut context = context();
        context.middleware.push(Arc::new(Rewrite));
        let result = if replace_all {
            call.execute_with_arguments(context, json!({"a":9,"b":6}))
                .await
        } else {
            call.execute_with_overrides(context, json!({"a":7})).await
        }
        .unwrap();
        assert_eq!(
            result.results[0].result.as_json(),
            Some(&json!({
                "current_id":"rewritten-id",
                "current_name":"rewritten-name",
                "original_id":"tc-1",
                "original_name":"add",
                "original_args":{"a":1,"b":2},
                "bound_tool":"add",
                "actual_args":{"a":20,"b":if replace_all {6} else {2}},
            }))
        );
    }
}

#[tokio::test]
async fn typed_tool_schema_and_execution_preserve_unset_model_input() {
    let tool = add_tool();
    let spec = tool.declaration().spec.as_ref().unwrap();
    assert_eq!(spec.description.as_deref(), Some("Add two numbers."));
    assert_eq!(spec.params.as_value()["properties"]["a"]["type"], "integer");
    let call = BoundToolCall::new(ToolCallPart::new("tc-1", "add", r#"{"a":1,"b":2}"#), tool);
    assert_eq!(call.arguments().unwrap(), json!({"a":1,"b":2}));
    let result = call.execute(context()).await.unwrap();
    assert_eq!(result.message.role, MessageRole::Tool);
    assert_eq!(result.results[0].result.as_json(), Some(&json!(3)));
    assert!(!result.results[0].has_model_input());
}

#[tokio::test]
async fn model_argument_errors_become_results_and_host_overrides_are_fallible() {
    for arguments in [r#"{"a":"bad","b":2}"#, r#"{"a": "#, "[]"] {
        let call = BoundToolCall::new(ToolCallPart::new("tc-1", "add", arguments), add_tool());
        let result = call.execute(context()).await.unwrap();
        assert!(result.results[0].is_error());
        assert!(result.exception.is_some());
    }
    let call = BoundToolCall::new(
        ToolCallPart::new("tc-1", "add", r#"{"a":1,"b":2}"#),
        add_tool(),
    );
    let result = call
        .execute_with_overrides(context(), json!({"a":7}))
        .await
        .unwrap();
    assert_eq!(result.results[0].result.as_json(), Some(&json!(9)));
    assert!(
        call.execute_with_overrides(context(), json!({"a":"bad"}))
            .await
            .is_err()
    );
    let result = call
        .execute_with_arguments(context(), json!({"a":5,"b":6}))
        .await
        .unwrap();
    assert_eq!(result.results[0].result.as_json(), Some(&json!(11)));
}

#[tokio::test]
async fn errors_keep_the_original_error_for_downcasting() {
    let tool = AgentTool::new("fail", "Fail.", |_: Value, _| async {
        Err::<Value, _>(Error::Configuration("boom".into()))
    })
    .unwrap();
    let call = BoundToolCall::new(ToolCallPart::new("tc-1", "fail", "{}"), tool);
    let result = call.execute(context()).await.unwrap();
    assert!(result.results[0].is_error());
    let error = result.exception.unwrap();
    assert!(
        matches!(error.as_error().downcast_ref::<Error>(), Some(Error::Configuration(message)) if message == "boom")
    );
}

#[tokio::test]
async fn already_erased_tool_errors_preserve_identity_and_direct_downcasting() {
    use suprnova_ai::types::events::RuntimeError;
    let original = RuntimeError::new(std::io::Error::other("original"));
    let returned = original.clone();
    let tool = AgentTool::new_with_hooks("fail", "Retain original error.", move |_: Value, _| {
        let error = returned.clone();
        async move { Err::<Value, _>(error.into()) }
    })
    .unwrap();
    let result = BoundToolCall::new(ToolCallPart::new("tc-1", "fail", "{}"), tool)
        .execute(context())
        .await
        .unwrap();
    let exception = result.exception.unwrap();
    assert_eq!(exception, original);
    assert!(exception.as_error().is::<std::io::Error>());
}

#[tokio::test]
async fn cached_results_skip_validation_execution_and_approval() {
    let tool = AgentTool::new("add", "Never run.", |_: Add, _| async {
        panic!("cached call executed");
        #[allow(unreachable_code)]
        Ok(0)
    })
    .unwrap()
    .require_approval(true);
    let mut part = ToolCallPart::new("tc-1", "add", "malformed");
    part.cached_result = Some(ToolResultPart::new("tc-1", "add", json!(42)));
    let result = BoundToolCall::new(part, tool)
        .execute(context())
        .await
        .unwrap();
    assert_eq!(result.results[0].result.as_json(), Some(&json!(42)));
}

#[test]
fn resolver_uses_last_duplicate_and_reports_unknown_executor() {
    let first = add_tool();
    let second = AgentTool::new("add", "Replacement.", |_: Add, _| async { Ok(5) }).unwrap();
    let resolver = ToolResolver::new(vec![first, second]);
    assert_eq!(
        resolver
            .get("add")
            .unwrap()
            .declaration()
            .spec
            .as_ref()
            .unwrap()
            .description
            .as_deref(),
        Some("Replacement.")
    );
    assert!(
        resolver
            .resolve(ToolCallPart::new("missing", "missing", "{}"))
            .is_err()
    );
}

#[tokio::test]
async fn explicit_model_conversion_preserves_null_and_special_outputs() {
    let tool = add_tool()
        .with_model_input(|_| Ok(ToolResultValue::Json(Value::Null)))
        .unwrap();
    let result = BoundToolCall::new(
        ToolCallPart::new("tc-1", "add", r#"{"a":1,"b":2}"#),
        tool.clone(),
    )
    .execute(context())
    .await
    .unwrap();
    assert_eq!(
        result.results[0].model_input,
        Some(ToolResultValue::Json(Value::Null))
    );
    assert_eq!(
        tool.model_input(&ToolResultValue::Json(json!(3))).unwrap(),
        Some(ToolResultValue::Json(Value::Null))
    );
    let special = AgentTool::new_special("add", "Rich result.", |_: Value, _| async {
        Ok(MessageBundle::new(vec![]))
    })
    .unwrap();
    let result = BoundToolCall::new(ToolCallPart::new("tc-1", "add", "{}"), special)
        .execute(context())
        .await
        .unwrap();
    assert!(matches!(
        result.results[0].result,
        ToolResultValue::Special(SpecialToolResult::Messages(_))
    ));
}

#[tokio::test]
async fn nested_inputs_defaults_and_aliases_reach_the_typed_closure() {
    #[derive(Deserialize, JsonSchema)]
    struct Edit {
        #[serde(rename = "oldText")]
        old_text: String,
    }
    #[derive(Deserialize, JsonSchema)]
    struct Edits {
        edits: Vec<Edit>,
        #[serde(default)]
        urgent: bool,
    }
    let tool = AgentTool::new("add", "Nested input.", |args: Edits, _| async move {
        assert!(!args.urgent);
        Ok(args.edits[0].old_text.clone())
    })
    .unwrap();
    let result = BoundToolCall::new(
        ToolCallPart::new("tc-1", "add", r#"{"edits":[{"oldText":"before"}]}"#),
        tool,
    )
    .execute(context())
    .await
    .unwrap();
    assert_eq!(result.results[0].result.as_json(), Some(&json!("before")));
}

#[tokio::test]
async fn runner_cancels_in_fifo_order_and_discards_without_results() {
    use futures::StreamExt;
    use std::sync::{Arc, Mutex};
    use suprnova_ai::agents::tools::ToolRunner;
    struct Cancel {
        id: usize,
        cancelled: Arc<Mutex<Vec<usize>>>,
    }
    impl Drop for Cancel {
        fn drop(&mut self) {
            self.cancelled.lock().unwrap().push(self.id);
        }
    }
    let cancelled = Arc::new(Mutex::new(Vec::new()));
    let mut runner = ToolRunner::new();
    let mut handles = Vec::new();
    for id in 0..3 {
        let guard = Cancel {
            id,
            cancelled: cancelled.clone(),
        };
        handles.push(runner.schedule(async move {
            let _guard = guard;
            futures::future::pending::<Result<suprnova_ai::types::events::ToolCallResult, Error>>()
                .await
        }));
    }
    assert!(futures::poll!(runner.events().next()).is_pending());
    runner.discard(&handles[0]);
    assert_eq!(*cancelled.lock().unwrap(), [0]);
    assert!(runner.get_tool_message().is_none());
    drop(runner);
    assert_eq!(*cancelled.lock().unwrap(), [0, 1, 2]);
}

#[tokio::test]
async fn runner_events_restart_and_collect_results_in_completion_order() {
    use futures::StreamExt;
    use suprnova_ai::agents::tools::{ToolRunner, tool_result};
    let mut runner = ToolRunner::new();
    let (send, receive) = tokio::sync::oneshot::channel();
    runner.schedule(async move {
        receive.await.unwrap();
        Ok(tool_result(ToolResultPart::new("slow", "add", json!(1))))
    });
    runner.add_result(tool_result(ToolResultPart::new("fast", "add", json!(2))));
    assert_eq!(
        runner.events().next().await.unwrap().unwrap().results[0].tool_call_id,
        "fast"
    );
    send.send(()).unwrap();
    assert_eq!(
        runner.events().next().await.unwrap().unwrap().results[0].tool_call_id,
        "slow"
    );
    assert!(runner.events().next().await.is_none());
    assert_eq!(
        runner
            .get_tool_message()
            .unwrap()
            .tool_results()
            .iter()
            .map(|part| part.tool_call_id.as_str())
            .collect::<Vec<_>>(),
        ["fast", "slow"]
    );
    runner.add_result(tool_result(ToolResultPart::new("later", "add", json!(3))));
    assert_eq!(
        runner.events().next().await.unwrap().unwrap().results[0].tool_call_id,
        "later"
    );
}

#[tokio::test]
async fn runner_discard_omits_a_ready_but_undelivered_result() {
    use futures::StreamExt;
    use suprnova_ai::agents::tools::{ToolRunner, tool_result};
    let mut runner = ToolRunner::new();
    runner.add_result(tool_result(ToolResultPart::new("first", "add", json!(1))));
    let discarded = runner.schedule(async {
        Ok(tool_result(ToolResultPart::new(
            "discarded",
            "add",
            json!(2),
        )))
    });
    assert_eq!(
        runner.events().next().await.unwrap().unwrap().results[0].tool_call_id,
        "first"
    );
    runner.discard(&discarded);
    assert!(runner.events().next().await.is_none());
    assert_eq!(runner.get_tool_message().unwrap().tool_results().len(), 1);
}
