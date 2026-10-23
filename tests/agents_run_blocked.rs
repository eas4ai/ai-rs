mod agents_support;
use agents_support::*;
use futures::{StreamExt, stream};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;
use std::sync::Arc;
use suprnova_ai::{
    agents::{Agent, AgentTool, RunOptions},
    types::{Event, HookStatus},
    user_message,
};
#[derive(Deserialize, JsonSchema)]
struct Args {
    x: i32,
}
fn gated() -> AgentTool {
    AgentTool::new::<Args, i32, _, _>("gated", "", |a, _| async move { Ok(a.x) })
        .unwrap()
        .require_approval(true)
}
#[tokio::test]
async fn approval_blocks_then_resolution_unblocks_and_preserves_delivery_order() {
    let (model, _) = scripted(vec![calls(&[("tc1", "gated", r#"{"x":1}"#)]), text("done")]);
    let mut run = Agent::new(vec![gated()]).run(model, vec![user_message("go")]);
    let mut names = vec![];
    while let Some(event) = run.next().await {
        match event.unwrap() {
            Event::Hook(value) => {
                if value.hook.status == HookStatus::Pending {
                    names.push("pending")
                } else {
                    names.push("resolved");
                    assert!(!run.blocked())
                }
            }
            Event::RunBlocked(value) => {
                names.push("blocked");
                assert!(run.blocked());
                assert_eq!(value.hooks[0].tool_call_id.as_deref(), Some("tc1"));
                assert_eq!(run.deferred_hooks().len(), 1);
                run.hook_registry()
                    .resolve_hook(&value.hooks[0].hook_id, json!({"granted":true}))
                    .unwrap();
            }
            _ => {}
        }
    }
    assert_eq!(names, vec!["pending", "blocked", "resolved"]);
    assert!(!run.blocked());
    assert!(run.deferred_hooks().is_empty());
    assert_eq!(run.text().unwrap(), "done");
}
#[tokio::test]
async fn busy_sibling_delays_blocking_until_its_result_is_delivered() {
    let release = Arc::new(tokio::sync::Notify::new());
    let signal = release.clone();
    let slow = AgentTool::new::<Args, i32, _, _>("slow", "", move |a, _| {
        let signal = signal.clone();
        async move {
            signal.notified().await;
            Ok(a.x)
        }
    })
    .unwrap();
    let (model, _) = scripted(vec![
        calls(&[
            ("gate", "gated", r#"{"x":1}"#),
            ("slow", "slow", r#"{"x":2}"#),
        ]),
        text("done"),
    ]);
    let mut run = Agent::new(vec![gated(), slow]).run(model, vec![user_message("go")]);
    let mut slow_done = false;
    let mut blocked = 0;
    while let Some(event) = run.next().await {
        match event.unwrap() {
            Event::Hook(h) if h.hook.status == HookStatus::Pending => {
                assert!(!run.blocked());
                release.notify_one();
            }
            Event::ToolCallResult(r) if r.results[0].tool_call_id == "slow" => slow_done = true,
            Event::RunBlocked(b) => {
                assert!(slow_done);
                blocked += 1;
                run.hook_registry()
                    .resolve_hook(&b.hooks[0].hook_id, json!({"granted":true}))
                    .unwrap();
            }
            _ => {}
        }
    }
    assert_eq!(blocked, 1);
}
#[tokio::test]
async fn parallel_gates_reblock_on_the_remaining_hook() {
    let (model, _) = scripted(vec![
        calls(&[
            ("one", "gated", r#"{"x":1}"#),
            ("two", "gated", r#"{"x":2}"#),
        ]),
        text("done"),
    ]);
    let mut run = Agent::new(vec![gated()]).run(model, vec![user_message("go")]);
    let mut counts = vec![];
    while let Some(event) = run.next().await {
        if let Event::RunBlocked(b) = event.unwrap() {
            counts.push(b.hooks.len());
            run.hook_registry()
                .resolve_hook(&b.hooks[0].hook_id, json!({"granted":true}))
                .unwrap();
        }
    }
    assert_eq!(counts, vec![2, 1]);
    assert!(!run.blocked());
}
#[tokio::test]
async fn defer_finishes_blocked_and_resume_uses_preregistered_resolution() {
    let (model, transport) = scripted(vec![calls(&[("one", "gated", r#"{"x":1}"#)]), text("done")]);
    let agent = Agent::new(vec![gated()]);
    let mut run = agent.run(model.clone(), vec![user_message("go")]);
    let mut blocked = 0;
    while let Some(event) = run.next().await {
        match event.unwrap() {
            Event::Hook(h) if h.hook.status == HookStatus::Pending => {
                run.hook_registry()
                    .defer_hook(&h.hook.hook_id, h.hook.clone())
                    .unwrap();
            }
            Event::RunBlocked(_) => blocked += 1,
            _ => {}
        }
    }
    assert_eq!(blocked, 1);
    assert!(run.blocked());
    assert!(run.messages().last().unwrap().tool_results()[0].is_hook_deferred);
    let history = run.messages();
    let hooks = suprnova_ai::agents::HookRegistry::default();
    hooks
        .resolve_hook("approve_one", json!({"granted":true}))
        .unwrap();
    let mut resumed = agent.run_with_options(
        model,
        history,
        RunOptions {
            hook_registry: Some(hooks),
            ..Default::default()
        },
    );
    while let Some(event) = resumed.next().await {
        assert!(!matches!(
            event.unwrap(),
            Event::Hook(_) | Event::RunBlocked(_)
        ));
    }
    assert_eq!(resumed.text().unwrap(), "done");
    assert_eq!(resumed.messages().len(), 4);
    assert_eq!(transport.requests.lock().unwrap().len(), 2);
}
#[tokio::test]
async fn replay_without_resolution_emits_blocked_after_hidden_response_boundary() {
    let (model, _) = scripted(vec![text("done")]);
    let history = vec![
        user_message("go"),
        suprnova_ai::types::Message::new(
            suprnova_ai::types::MessageRole::Assistant,
            vec![suprnova_ai::types::ToolCallPart::new("one", "gated", r#"{"x":1}"#).into()],
        ),
    ];
    let mut run = Agent::new(vec![gated()]).run(model, history);
    let mut blocked = 0;
    while let Some(event) = run.next().await {
        match event.unwrap() {
            Event::RunBlocked(b) => {
                blocked += 1;
                run.hook_registry()
                    .resolve_hook(&b.hooks[0].hook_id, json!({"granted":true}))
                    .unwrap()
            }
            Event::ToolEnd(_) => panic!("replay leaked"),
            _ => {}
        }
    }
    assert_eq!(blocked, 1);
}
#[derive(Deserialize, JsonSchema)]
struct Confirmation {
    approved: bool,
}
#[tokio::test]
async fn custom_loop_hook_blocks_after_model_end() {
    let (model, _) = scripted(vec![text("OK")]);
    let agent = Agent::default().with_loop(|call| {
        let source = call.model_stream();
        source
            .chain(stream::once(async move {
                let answer: Confirmation = call
                    .hooks
                    .hook("confirm", Default::default(), None, &call.runtime)
                    .await
                    .map_err(|e| suprnova_ai::Error::Validation(e.to_string()))?;
                assert!(answer.approved);
                Ok(Event::TextDelta(suprnova_ai::events::TextDelta::new(
                    "confirmed",
                )))
            }))
            .boxed()
    });
    let mut run = agent.run(model, vec![user_message("go")]);
    let mut ended = false;
    let mut count = 0;
    while let Some(event) = run.next().await {
        match event.unwrap() {
            Event::StreamEnd(_) => ended = true,
            Event::RunBlocked(b) => {
                assert!(ended);
                count += 1;
                run.hook_registry()
                    .resolve_hook(&b.hooks[0].hook_id, json!({"approved":true}))
                    .unwrap()
            }
            _ => {}
        }
    }
    assert_eq!(count, 1);
}
#[tokio::test]
async fn unattributed_tool_hook_never_falsely_reports_blocked() {
    let tool = AgentTool::new_with_hooks::<Args, i32, _, _>("selfgate", "", |a, call| async move {
        let answer: Confirmation = call
            .hooks
            .hook("self_gate", Default::default(), None, &call.runtime)
            .await?;
        assert!(answer.approved);
        Ok(a.x)
    })
    .unwrap();
    let (model, _) = scripted(vec![
        calls(&[("one", "selfgate", r#"{"x":1}"#)]),
        text("done"),
    ]);
    let mut run = Agent::new(vec![tool]).run(model, vec![user_message("go")]);
    while let Some(event) = run.next().await {
        match event.unwrap() {
            Event::RunBlocked(_) => panic!("unattributed hook treated as idle"),
            Event::Hook(h) if h.hook.status == HookStatus::Pending => run
                .hook_registry()
                .resolve_hook(&h.hook.hook_id, json!({"approved":true}))
                .unwrap(),
            _ => {}
        }
    }
}

#[tokio::test]
async fn completed_tool_before_stream_end_does_not_hide_a_blocked_sibling() {
    let quick =
        AgentTool::new::<Args, i32, _, _>("quick", "", |a, _| async move { Ok(a.x) }).unwrap();
    let (model, _) = scripted(vec![
        calls(&[
            ("quick", "quick", r#"{"x":1}"#),
            ("gate", "gated", r#"{"x":2}"#),
        ]),
        text("done"),
    ]);
    let mut run = Agent::new(vec![quick, gated()]).run_with_options(
        model,
        vec![user_message("go")],
        RunOptions {
            loop_buffer: Some(0),
            ..Default::default()
        },
    );
    let mut quick_done = false;
    let mut ended = false;
    let mut count = 0;
    while let Some(event) = tokio::time::timeout(std::time::Duration::from_secs(1), run.next())
        .await
        .expect("blocked signal never arrived")
    {
        match event.unwrap() {
            Event::ToolCallResult(r) if r.results[0].tool_call_id == "quick" => {
                assert!(!ended);
                quick_done = true;
            }
            Event::StreamEnd(_) => ended = true,
            Event::RunBlocked(b) => {
                assert!(quick_done);
                count += 1;
                run.hook_registry()
                    .resolve_hook(&b.hooks[0].hook_id, json!({"granted":true}))
                    .unwrap();
            }
            _ => {}
        }
    }
    assert_eq!(count, 1);
}

#[tokio::test]
async fn cancelling_live_approval_clears_blocked_state_and_returns_a_tool_error() {
    let (model, _) = scripted(vec![
        calls(&[("gate", "gated", r#"{"x":1}"#)]),
        text("cancelled"),
    ]);
    let mut run = Agent::new(vec![gated()]).run(model, vec![user_message("go")]);
    let mut cancelled = false;
    let mut result_error = false;
    while let Some(event) = run.next().await {
        match event.unwrap() {
            Event::RunBlocked(b) => run
                .hook_registry()
                .cancel_hook(&b.hooks[0].hook_id, Some("denied".into()))
                .await
                .unwrap(),
            Event::Hook(h) if h.hook.status == HookStatus::Cancelled => {
                cancelled = true;
                assert!(!run.blocked())
            }
            Event::ToolCallResult(r) => result_error = r.results[0].is_error(),
            _ => {}
        }
    }
    assert!(cancelled && result_error);
    assert!(!run.blocked());
    assert!(run.deferred_hooks().is_empty());
    assert_eq!(run.text().unwrap(), "cancelled");
}
