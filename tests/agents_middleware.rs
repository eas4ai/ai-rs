mod agents_support;
use agents_support::*;
use futures::{StreamExt, stream};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;
use std::sync::{Arc, Mutex};
use suprnova_ai::{
    Error,
    agents::{
        Agent, AgentTool, Middleware, RunContext, RunOptions,
        middleware::{ModelContext, ModelNext, RunNext, ToolContext, ToolNext},
    },
    types::{Event, events::ToolCallResult},
    user_message,
};
#[derive(Deserialize, JsonSchema)]
struct Args {
    x: i32,
}
struct Fixer;
#[async_trait::async_trait]
impl Middleware for Fixer {
    async fn wrap_tool(
        &self,
        mut call: ToolContext,
        next: ToolNext,
    ) -> Result<ToolCallResult, Error> {
        assert_eq!(call.tool_call_id, "tc");
        call.kwargs = json!({"x":99});
        call.tool_name = "rewritten".into();
        next(call).await
    }
    fn wrap_model(
        &self,
        mut call: ModelContext,
        next: ModelNext,
    ) -> suprnova_ai::providers::EventStream {
        call.messages.push(suprnova_ai::system_message("injected"));
        next(call)
    }
}
#[tokio::test]
async fn tool_middleware_repairs_malformed_arguments_and_rewrites_result_fields() {
    let tool =
        AgentTool::new::<Args, i32, _, _>("double", "", |a, _| async move { Ok(a.x * 2) }).unwrap();
    let (model, transport) = scripted(vec![calls(&[("tc", "double", "not json")]), text("done")]);
    let original = vec![user_message("go")];
    let mut run = Agent::new(vec![tool]).run_with_options(
        model,
        original.clone(),
        RunOptions {
            middleware: vec![Arc::new(Fixer)],
            ..Default::default()
        },
    );
    let mut seen = false;
    while let Some(event) = run.next().await {
        if let Event::ToolCallResult(r) = event.unwrap() {
            assert_eq!(r.results[0].tool_name, "rewritten");
            assert_eq!(
                r.results[0].result,
                suprnova_ai::types::messages::ToolResultValue::Json(json!(198))
            );
            seen = true;
        }
    }
    assert!(seen);
    assert_eq!(original.len(), 1);
    assert_eq!(run.messages().len(), 4);
    assert_eq!(
        transport.requests.lock().unwrap()[0]["messages"][1]["parts"][0]["text"],
        "injected"
    );
}
struct Order {
    name: &'static str,
    order: Arc<Mutex<Vec<String>>>,
}
impl Middleware for Order {
    fn wrap_agent_run(
        &self,
        call: RunContext,
        next: RunNext,
    ) -> suprnova_ai::providers::EventStream {
        self.order
            .lock()
            .unwrap()
            .push(format!("{}-before", self.name));
        let order = self.order.clone();
        let name = self.name;
        next(call)
            .chain(
                stream::once(async move {
                    order.lock().unwrap().push(format!("{name}-after"));
                    None
                })
                .filter_map(|x| async move { x }),
            )
            .boxed()
    }
}
#[tokio::test]
async fn first_run_middleware_is_outermost() {
    let order = Arc::new(Mutex::new(vec![]));
    let (model, _) = scripted(vec![text("done")]);
    let mut run = Agent::default().run_with_options(
        model,
        vec![user_message("go")],
        RunOptions {
            middleware: vec![
                Arc::new(Order {
                    name: "outer",
                    order: order.clone(),
                }),
                Arc::new(Order {
                    name: "inner",
                    order: order.clone(),
                }),
            ],
            ..Default::default()
        },
    );
    while let Some(event) = run.next().await {
        event.unwrap();
    }
    assert_eq!(
        *order.lock().unwrap(),
        vec!["outer-before", "inner-before", "inner-after", "outer-after"]
    );
}
struct RewriteId;
#[async_trait::async_trait]
impl Middleware for RewriteId {
    async fn wrap_tool(
        &self,
        mut call: ToolContext,
        next: ToolNext,
    ) -> Result<ToolCallResult, Error> {
        call.tool_call_id = "orphan".into();
        next(call).await
    }
}
#[tokio::test]
async fn rewriting_correlation_id_returns_history_integrity_error() {
    let tool =
        AgentTool::new::<Args, i32, _, _>("double", "", |a, _| async move { Ok(a.x * 2) }).unwrap();
    let (model, _) = scripted(vec![
        calls(&[("tc", "double", r#"{"x":1}"#)]),
        text("unreachable"),
    ]);
    let mut run = Agent::new(vec![tool]).run_with_options(
        model,
        vec![user_message("go")],
        RunOptions {
            middleware: vec![Arc::new(RewriteId)],
            ..Default::default()
        },
    );
    let mut error = None;
    while let Some(event) = run.next().await {
        if let Err(e) = event {
            error = Some(e);
            break;
        }
    }
    assert!(error.unwrap().to_string().contains("orphaned-tool-result"));
}

struct RetryRun;
impl Middleware for RetryRun {
    fn wrap_agent_run(
        &self,
        call: RunContext,
        next: RunNext,
    ) -> suprnova_ai::providers::EventStream {
        let source = next(call.clone());
        stream::unfold(
            (call, next, source, false),
            |(call, next, mut source, retried)| async move {
                match source.next().await {
                    Some(Err(_)) if !retried => {
                        let source = next(call.clone());
                        Some((
                            Ok(Event::Retry(Default::default())),
                            (call, next, source, true),
                        ))
                    }
                    Some(event) => Some((event, (call, next, source, retried))),
                    None => None,
                }
            },
        )
        .boxed()
    }
}
#[tokio::test]
async fn run_middleware_can_retry_by_calling_next_again() {
    let (model, transport) = scripted_results(vec![
        Err(suprnova_ai::transport::TransportError::Timeout),
        Ok(text("recovered")),
    ]);
    let mut run = Agent::default().run_with_options(
        model,
        vec![user_message("go")],
        RunOptions {
            middleware: vec![Arc::new(RetryRun)],
            // Call retries are off, so the middleware's retry is the only one.
            stream_options: suprnova_ai::StreamOptions {
                retry: suprnova_ai::RetryPolicy::none(),
                ..Default::default()
            },
            ..Default::default()
        },
    );
    let mut retries = 0;
    while let Some(event) = run.next().await {
        if matches!(event.unwrap(), Event::Retry(_)) {
            retries += 1
        }
    }
    assert_eq!(retries, 1);
    assert_eq!(run.text().unwrap(), "recovered");
    assert_eq!(transport.requests.lock().unwrap().len(), 2);
}

struct GenerateSuffix;
#[async_trait::async_trait]
impl Middleware for GenerateSuffix {
    async fn wrap_generate(
        &self,
        mut call: suprnova_ai::agents::middleware::GenerateContext,
        next: suprnova_ai::agents::middleware::GenerateNext,
    ) -> Result<suprnova_ai::types::Message, Error> {
        call.messages
            .push(suprnova_ai::system_message("generate-only"));
        let mut message = next(call).await?;
        message
            .parts
            .push(suprnova_ai::types::TextPart::new("-wrapped").into());
        Ok(message)
    }
}
#[tokio::test]
async fn custom_loop_generation_uses_generate_middleware_and_preserves_history_isolation() {
    let (model, transport) = scripted(vec![text("generated")]);
    let agent = Agent::default().with_loop(|call| {
        stream::once(async move {
            let message = call.generate().await?;
            call.update(|context| context.add(message.clone()));
            Ok(Event::StreamEnd(suprnova_ai::events::StreamEnd {
                event: suprnova_ai::types::ModelEventMeta::with_message(message),
                ..Default::default()
            }))
        })
        .boxed()
    });
    let mut run = agent.run_with_options(
        model,
        vec![user_message("go")],
        RunOptions {
            middleware: vec![Arc::new(GenerateSuffix)],
            ..Default::default()
        },
    );
    while let Some(event) = run.next().await {
        event.unwrap();
    }
    assert_eq!(run.text().unwrap(), "generated-wrapped");
    assert_eq!(run.messages().len(), 2);
    assert_eq!(
        transport.requests.lock().unwrap()[0]["messages"][1]["parts"][0]["text"],
        "generate-only"
    );
}
