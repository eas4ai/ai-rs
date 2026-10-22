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

mod agents_support;
use agents_support::*;
use futures::StreamExt;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use suprnova_ai::{
    agents::{Agent, AgentTool, Context, RunOptions},
    types::{Event, Message, MessageRole, ToolCallPart, ToolResultPart},
    user_message,
};
#[derive(Deserialize, JsonSchema)]
struct Args {
    x: i32,
}
fn double() -> AgentTool {
    AgentTool::new::<Args, i32, _, _>(
        "double",
        "Double a number",
        |a, _| async move { Ok(a.x * 2) },
    )
    .unwrap()
}
#[tokio::test]
async fn loop_runs_tools_and_keeps_a_single_history_turn_per_response() {
    let (model, transport) = scripted(vec![calls(&[("tc1", "double", r#"{"x":7}"#)]), text("14")]);
    let mut run = Agent::new(vec![double()]).run(model, vec![user_message("go")]);
    let mut results = vec![];
    while let Some(event) = run.next().await {
        if let Event::ToolCallResult(r) = event.unwrap() {
            results.extend(r.results)
        }
    }
    assert_eq!(run.text().unwrap(), "14");
    assert_eq!(
        results[0].result,
        suprnova_ai::types::messages::ToolResultValue::Json(json!(14))
    );
    let messages = run.messages();
    assert_eq!(
        messages.iter().map(|m| m.role).collect::<Vec<_>>(),
        vec![
            MessageRole::User,
            MessageRole::Assistant,
            MessageRole::Tool,
            MessageRole::Assistant
        ]
    );
    assert_eq!(transport.requests.lock().unwrap().len(), 2);
    let requests = transport.requests.lock().unwrap();
    assert_eq!(requests[1]["messages"][2]["parts"][0]["result"], 14);
    assert_eq!(
        requests[0]["tools"][0]["spec"]["params"]["required"],
        json!(["x"])
    );
}
#[tokio::test]
async fn invalid_arguments_are_tool_errors_and_the_model_can_retry() {
    let (model, _) = scripted(vec![
        calls(&[("tc1", "double", r#"{"x":"wrong"}"#)]),
        calls(&[("tc2", "double", r#"{"x":3}"#)]),
        text("6"),
    ]);
    let mut run = Agent::new(vec![double()]).run(model, vec![user_message("go")]);
    let mut results = vec![];
    while let Some(event) = run.next().await {
        if let Event::ToolCallResult(r) = event.unwrap() {
            results.extend(r.results)
        }
    }
    assert!(results[0].is_error());
    assert!(!results[1].is_error());
    assert_eq!(run.text().unwrap(), "6");
}
#[tokio::test]
async fn resumed_deferred_turn_caches_completed_siblings_and_filters_replay() {
    let ran = Arc::new(AtomicUsize::new(0));
    let n = ran.clone();
    let tool = AgentTool::new::<Args, i32, _, _>("double", "", move |a, _| {
        let n = n.clone();
        async move {
            n.fetch_add(1, Ordering::SeqCst);
            Ok(a.x * 2)
        }
    })
    .unwrap();
    let previous = Message::new(
        MessageRole::Assistant,
        vec![
            ToolCallPart::new("done", "double", r#"{"x":4}"#).into(),
            ToolCallPart::new("wait", "double", r#"{"x":5}"#).into(),
        ],
    );
    let mut deferred = ToolResultPart::error("wait", "double", json!("Deferred"));
    deferred.is_hook_deferred = true;
    let completed = ToolResultPart::new("done", "double", json!(8));
    let history = vec![
        user_message("go"),
        previous,
        Message::new(MessageRole::Tool, vec![completed.into(), deferred.into()]),
    ];
    let (model, transport) = scripted(vec![text("18")]);
    let mut run = Agent::new(vec![tool]).run(model, history);
    let mut ends = 0;
    while let Some(event) = run.next().await {
        let event = event.unwrap();
        assert!(!event.replay());
        if matches!(event, Event::ToolEnd(_)) {
            ends += 1
        }
    }
    assert_eq!(ends, 0);
    assert_eq!(ran.load(Ordering::SeqCst), 1);
    assert_eq!(run.messages().len(), 4);
    assert_eq!(transport.requests.lock().unwrap().len(), 1);
}
#[test]
fn context_stops_on_final_or_deferred_and_ignores_replay_additions() {
    let (model, _) = scripted(vec![]);
    let mut context = Context::new(
        model,
        vec![user_message("go")],
        vec![],
        suprnova_ai::StreamOptions::default(),
    );
    assert!(context.keep_running());
    let mut replay = suprnova_ai::assistant_message("old");
    replay.replay = true;
    context.add(replay);
    assert_eq!(context.messages.len(), 1);
    context.add(suprnova_ai::assistant_message("done"));
    assert!(!context.keep_running());
}
#[tokio::test]
async fn empty_and_final_history_do_not_call_the_model_and_output_requires_a_final_message() {
    let (model, transport) = scripted(vec![]);
    let mut run = Agent::default().run(model.clone(), vec![]);
    assert!(run.next().await.is_none());
    assert!(run.text().is_err());
    let mut run = Agent::default().run(model, vec![suprnova_ai::assistant_message("already done")]);
    assert!(run.next().await.is_none());
    assert_eq!(run.text().unwrap(), "already done");
    assert!(transport.requests.lock().unwrap().is_empty());
}
#[derive(Deserialize, JsonSchema)]
struct Answer {
    value: i32,
}
#[tokio::test]
async fn structured_output_and_provider_declarations_flow_to_model() {
    let (model, transport) = scripted(vec![text(r#"{"value":42}"#)]);
    let options = RunOptions::default().with_output::<Answer>();
    let provider_tool = suprnova_ai::types::Tool::provider(
        "search",
        suprnova_ai::types::ToolConfig {
            id: Some("search.v1".into()),
            ..Default::default()
        },
    )
    .unwrap();
    let mut run = Agent::default()
        .with_provider_tools(vec![provider_tool])
        .run_with_options(model, vec![user_message("go")], options);
    while let Some(event) = run.next().await {
        event.unwrap();
    }
    assert_eq!(run.output::<Answer>().unwrap().value, 42);
    assert_eq!(
        transport.requests.lock().unwrap()[0]["output_schema"]["type"],
        "object"
    );
    assert_eq!(
        transport.requests.lock().unwrap()[0]["tools"][0]["tool_config"]["id"],
        "search.v1"
    );
}

#[tokio::test]
async fn custom_loop_default_buffer_runs_ahead_but_zero_buffer_is_lockstep() {
    for buffer in [None, Some(0), Some(1)] {
        let advanced = Arc::new(AtomicUsize::new(0));
        let observed = advanced.clone();
        let agent = Agent::default().with_loop(move |_| {
            let observed = observed.clone();
            futures::stream::iter(0..10)
                .map(move |i| {
                    observed.fetch_add(1, Ordering::SeqCst);
                    Ok(Event::TextDelta(suprnova_ai::events::TextDelta::new(
                        i.to_string(),
                    )))
                })
                .boxed()
        });
        let (model, _) = scripted(vec![]);
        let mut run = agent.run_with_options(
            model,
            vec![user_message("go")],
            RunOptions {
                loop_buffer: buffer,
                ..Default::default()
            },
        );
        run.next().await.unwrap().unwrap();
        for _ in 0..20 {
            tokio::task::yield_now().await;
        }
        match buffer {
            None => assert_eq!(advanced.load(Ordering::SeqCst), 10),
            Some(0) => assert_eq!(advanced.load(Ordering::SeqCst), 1),
            Some(_) => assert!(advanced.load(Ordering::SeqCst) <= 3),
        }
    }
}
#[tokio::test]
async fn dropping_a_run_cancels_pending_tool_futures_and_cleans_live_hooks() {
    struct Guard(Arc<AtomicUsize>);
    impl Drop for Guard {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    let dropped = Arc::new(AtomicUsize::new(0));
    let d = dropped.clone();
    let started = Arc::new(tokio::sync::Notify::new());
    let start = started.clone();
    let tool = AgentTool::new::<Args, i32, _, _>("wait", "", move |_, _| {
        let d = d.clone();
        let start = start.clone();
        async move {
            let _guard = Guard(d);
            start.notify_one();
            futures::future::pending().await
        }
    })
    .unwrap();
    let (model, _) = scripted(vec![calls(&[("tc", "wait", r#"{"x":1}"#)])]);
    let mut run = Agent::new(vec![tool]).run(model, vec![user_message("go")]);
    run.next().await.unwrap().unwrap();
    started.notified().await;
    drop(run);
    for _ in 0..20 {
        tokio::task::yield_now().await;
    }
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
}
#[tokio::test]
async fn tool_partial_results_precede_final_results() {
    let tool = AgentTool::stream::<Args, String, _, _>(
        "status",
        "",
        |_, _| futures::stream::iter([Ok("working".to_owned()), Ok("complete".to_owned())]),
        suprnova_ai::agents::Aggregate::last(),
    )
    .unwrap();
    let (model, _) = scripted(vec![calls(&[("tc", "status", r#"{"x":1}"#)]), text("done")]);
    let mut run = Agent::new(vec![tool]).run(model, vec![user_message("go")]);
    let mut events = vec![];
    while let Some(event) = run.next().await {
        match event.unwrap() {
            Event::PartialToolCallResult(p) => {
                assert_eq!(p.tool_call_id.as_deref(), Some("tc"));
                events.push("partial")
            }
            Event::ToolCallResult(_) => events.push("final"),
            _ => {}
        }
    }
    assert_eq!(events, vec!["partial", "partial", "final"]);
}
#[tokio::test]
async fn provider_error_is_returned_without_an_agent_level_retry() {
    let (model, transport) =
        scripted_results(vec![Err(suprnova_ai::transport::TransportError::Timeout)]);
    // Call retries are off, so the error shows the agent loop itself does not retry.
    let mut run = Agent::default().run_with_options(
        model,
        vec![user_message("go")],
        suprnova_ai::RunOptions {
            stream_options: suprnova_ai::StreamOptions {
                retry: suprnova_ai::RetryPolicy::none(),
                ..Default::default()
            },
            ..Default::default()
        },
    );
    assert!(matches!(
        run.next().await.unwrap(),
        Err(suprnova_ai::Error::Transport(
            suprnova_ai::transport::TransportError::Timeout
        ))
    ));
    assert!(run.next().await.is_none());
    assert_eq!(transport.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn buffered_loop_task_failure_is_returned_to_the_consumer() {
    let agent = Agent::default().with_loop(|_| {
        futures::stream::once(async {
            panic!("custom loop failed");
            #[allow(unreachable_code)]
            Ok(Event::TextDelta(suprnova_ai::events::TextDelta::new(
                "never",
            )))
        })
        .boxed()
    });
    let (model, _) = scripted(vec![]);
    let mut run = agent.run(model, vec![user_message("go")]);
    let result = run
        .next()
        .await
        .expect("task failure must produce an error");
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("custom loop failed")
    );
    assert!(run.next().await.is_none());
}

#[tokio::test]
async fn subagent_stream_inherits_hooks_and_keeps_its_messages_out_of_parent_history() {
    let (child_model, _) = scripted(vec![
        calls(&[("child", "double", r#"{"x":3}"#)]),
        text("child done"),
    ]);
    let child = Agent::new(vec![double().require_approval(true)]);
    let research = AgentTool::stream::<Args, Event, _, _>(
        "research",
        "",
        move |_, context| {
            child.run_with_options(
                child_model.clone(),
                vec![user_message("child prompt")],
                context.nested_options(),
            )
        },
        suprnova_ai::agents::Aggregate::messages(),
    )
    .unwrap();
    let (model, transport) = scripted(vec![
        calls(&[("parent", "research", r#"{"x":1}"#)]),
        text("parent done"),
    ]);
    let mut run = Agent::new(vec![research]).run(model, vec![user_message("parent prompt")]);
    let mut approved = false;
    let mut nested_result = false;
    while let Some(event) = run.next().await {
        match event.unwrap() {
            Event::PartialToolCallResult(p) => {
                assert_eq!(p.tool_call_id.as_deref(), Some("parent"));
                if p.value["kind"] == "hook" && p.value["hook"]["status"] == "pending" {
                    assert_eq!(p.value["hook"]["hook_id"], "approve_child");
                    run.hook_registry()
                        .resolve_hook("approve_child", json!({"granted":true}))
                        .unwrap();
                    approved = true;
                }
            }
            Event::ToolCallResult(r) => {
                nested_result = true;
                assert!(matches!(
                    &r.results[0].result,
                    suprnova_ai::types::messages::ToolResultValue::Special(
                        suprnova_ai::types::messages::SpecialToolResult::Messages(_)
                    )
                ));
                assert_eq!(
                    r.results[0].get_model_input(),
                    &suprnova_ai::types::messages::ToolResultValue::Json(json!("child done"))
                );
            }
            _ => {}
        }
    }
    assert!(approved && nested_result);
    assert_eq!(run.messages().len(), 4);
    assert_eq!(run.text().unwrap(), "parent done");
    assert_eq!(
        transport.requests.lock().unwrap()[1]["messages"]
            .as_array()
            .unwrap()
            .len(),
        3
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tool_context_retains_owning_agent_and_bound_call_across_nested_concurrent_runs() {
    let barrier = Arc::new(tokio::sync::Barrier::new(3));
    let observations = Arc::new(std::sync::Mutex::new(vec![]));
    let gate = barrier.clone();
    let seen = observations.clone();
    let probe = AgentTool::new::<Args, i32, _, _>("probe", "", move |args, context| {
        let gate = gate.clone();
        let seen = seen.clone();
        async move {
            gate.wait().await;
            let owner = context.agent.as_ref().expect("tool owner");
            let names = owner
                .tools()
                .iter()
                .map(|tool| tool.name().to_owned())
                .collect::<Vec<_>>();
            let bound = context.bound_call.as_ref().expect("bound call");
            assert_eq!(bound.id(), context.tool_call_id);
            assert_eq!(bound.name(), context.tool_name);
            assert_eq!(bound.arguments().unwrap()["x"], args.x);
            seen.lock()
                .unwrap()
                .push((context.tool_call_id.clone(), names, args.x));
            Ok(args.x)
        }
    })
    .unwrap();
    let child_marker =
        AgentTool::new::<Args, i32, _, _>("child_marker", "", |a, _| async move { Ok(a.x) })
            .unwrap();
    let parent_marker =
        AgentTool::new::<Args, i32, _, _>("parent_marker", "", |a, _| async move { Ok(a.x) })
            .unwrap();
    let child = Agent::new(vec![probe.clone(), child_marker]);
    let (child_model, _) = scripted(vec![
        calls(&[("child-call", "probe", r#"{"x":3}"#)]),
        text("child"),
    ]);
    let research = AgentTool::stream::<Args, Event, _, _>(
        "research",
        "",
        move |_, context| {
            child.run_with_options(
                child_model.clone(),
                vec![user_message("child")],
                context.nested_options(),
            )
        },
        suprnova_ai::agents::Aggregate::messages(),
    )
    .unwrap();
    let parent = Agent::new(vec![probe, research, parent_marker]);
    let (model, _) = scripted(vec![
        calls(&[
            ("parent-one", "probe", r#"{"x":7}"#),
            ("parent-two", "probe", r#"{"x":9}"#),
            ("branch", "research", r#"{"x":1}"#),
        ]),
        text("done"),
    ]);
    let mut run = parent.run(model, vec![user_message("go")]);
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        while let Some(event) = run.next().await {
            event.unwrap();
        }
    })
    .await
    .unwrap();
    let seen = observations.lock().unwrap();
    assert_eq!(seen.len(), 3);
    for (id, names, value) in &*seen {
        if id == "child-call" {
            assert!(names.iter().any(|n| n == "child_marker"));
            assert!(!names.iter().any(|n| n == "parent_marker"));
            assert_eq!(*value, 3);
        } else {
            assert!(names.iter().any(|n| n == "parent_marker"));
            assert!(!names.iter().any(|n| n == "child_marker"));
            assert_eq!(*value, if id == "parent-one" { 7 } else { 9 });
        }
    }
}
