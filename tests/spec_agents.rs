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

//! Tests for AGT-001 to AGT-005 in docs/spec/agents.md. Each test name starts
//! with the requirement it checks; scripts/check-core.sh selects them by it.

use std::{
    num::NonZeroUsize,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::Duration,
};

use futures::StreamExt;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};
use suprnova_ai::{
    Agent, AgentStream, AgentTool, Error, RunOptions,
    agents::{RunProgress, StepCheck, StepDecision, StopReason},
    assistant_message,
    errors::ProviderError,
    testing::{ScriptedProvider, ScriptedResponse},
    types::{
        Event, HookStatus, MessageRole, ToolResultPart, messages::ToolResultValue, usage::Usage,
    },
    user_message,
};

#[derive(Deserialize, JsonSchema)]
struct Index {
    #[serde(default)]
    i: u32,
}

fn noop(name: &str) -> AgentTool {
    AgentTool::new::<Index, u32, _, _>(
        name,
        "Return the index",
        |input, _| async move { Ok(input.i) },
    )
    .unwrap()
}

fn options(step_checks: Vec<StepCheck>) -> RunOptions {
    RunOptions {
        step_checks,
        ..RunOptions::default()
    }
}

/// Drain the run, failing the test on an error.
async fn finish(run: &mut AgentStream) -> Vec<Event> {
    let mut events = Vec::new();
    while let Some(event) = run.next().await {
        events.push(event.unwrap());
    }
    events
}

fn tool_calls(count: usize, name: &str) -> Vec<ScriptedResponse> {
    (1..=count)
        .map(|step| ScriptedResponse::tool_call(format!("c{step}"), name, json!({"i": step})))
        .collect()
}

fn result_text(part: &ToolResultPart) -> &str {
    match &part.result {
        ToolResultValue::Json(Value::String(text)) => text,
        other => panic!("expected a text result, got {other:?}"),
    }
}

// AGT-001

#[tokio::test]
async fn agt_001_default_runs_until_done() {
    let mut script = tool_calls(25, "work");
    script.push(ScriptedResponse::text("finished"));
    let provider = ScriptedProvider::new("scripted", script);
    let mut run = Agent::new(vec![noop("work")]).run(provider.model("m"), vec![user_message("go")]);
    finish(&mut run).await;
    assert_eq!(run.text().unwrap(), "finished");
    assert_eq!(provider.requests().len(), 26);
}

#[tokio::test]
async fn agt_001_step_count_stops_after_that_steps_tool_results() {
    let provider = ScriptedProvider::new("scripted", tool_calls(5, "work"));
    let mut run = Agent::new(vec![noop("work")]).run_with_options(
        provider.model("m"),
        vec![user_message("go")],
        options(vec![StepCheck::step_count(2)]),
    );
    finish(&mut run).await;
    assert_eq!(provider.requests().len(), 2);
    let history = run.messages();
    let last = history.last().unwrap();
    assert_eq!(last.role, MessageRole::Tool);
    assert_eq!(last.tool_results()[0].tool_call_id, "c2");
}

#[tokio::test]
async fn agt_001_tool_called_stops_after_the_step_that_called_it() {
    let provider = ScriptedProvider::new(
        "scripted",
        [
            ScriptedResponse::tool_call("c1", "work", json!({})),
            ScriptedResponse::tool_call("c2", "submit", json!({})),
            ScriptedResponse::tool_call("c3", "work", json!({})),
            ScriptedResponse::text("unused"),
        ],
    );
    let mut run = Agent::new(vec![noop("work"), noop("submit")]).run_with_options(
        provider.model("m"),
        vec![user_message("go")],
        options(vec![StepCheck::tool_called(["submit"])]),
    );
    finish(&mut run).await;
    assert_eq!(provider.requests().len(), 2);
    assert_eq!(
        run.messages().last().unwrap().tool_results()[0].tool_call_id,
        "c2"
    );
}

#[tokio::test]
async fn agt_001_steering_after_a_final_answer_calls_the_model_again() {
    let provider = ScriptedProvider::new(
        "scripted",
        [
            ScriptedResponse::text("draft"),
            ScriptedResponse::text("final"),
        ],
    );
    let steered = Arc::new(AtomicBool::new(false));
    let check = {
        let steered = steered.clone();
        StepCheck::new(move |_progress| {
            let first = !steered.swap(true, Ordering::SeqCst);
            async move {
                Ok(if first {
                    StepDecision::Steer(user_message("Check the draft again."))
                } else {
                    StepDecision::Continue
                })
            }
        })
    };
    let mut run = Agent::default().run_with_options(
        provider.model("m"),
        vec![user_message("write")],
        options(vec![check]),
    );
    finish(&mut run).await;
    let requests = provider.requests();
    assert_eq!(requests.len(), 2);
    let last = requests[1].messages.last().unwrap();
    assert_eq!(last.role, MessageRole::User);
    assert_eq!(last.text(), "Check the draft again.");
    assert_eq!(run.text().unwrap(), "final");
}

#[tokio::test]
async fn agt_001_checks_after_a_stop_do_not_run() {
    let provider = ScriptedProvider::new("scripted", tool_calls(3, "work"));
    let later = Arc::new(AtomicUsize::new(0));
    let counter = {
        let later = later.clone();
        StepCheck::new(move |_| {
            later.fetch_add(1, Ordering::SeqCst);
            async { Ok(StepDecision::Continue) }
        })
    };
    let stop = StepCheck::new(|_| async { Ok(StepDecision::Stop("halt".into())) });
    let mut run = Agent::new(vec![noop("work")]).run_with_options(
        provider.model("m"),
        vec![user_message("go")],
        options(vec![stop, counter]),
    );
    finish(&mut run).await;
    assert_eq!(later.load(Ordering::SeqCst), 0);
    assert_eq!(provider.requests().len(), 1);
}

#[tokio::test]
async fn agt_001_checks_receive_steps_history_and_usage() {
    let usage = |input, output| Usage {
        input_tokens: input,
        output_tokens: output,
        ..Usage::default()
    };
    let provider = ScriptedProvider::new(
        "scripted",
        [
            ScriptedResponse::tool_call("c1", "work", json!({})).with_usage(usage(10, 5)),
            ScriptedResponse::text("done").with_usage(usage(20, 7)),
        ],
    );
    // Each entry: steps, history length, and (input, output) usage tokens.
    type Seen = Vec<(usize, usize, Option<(u64, u64)>)>;
    let seen: Arc<Mutex<Seen>> = Arc::default();
    let check = {
        let seen = seen.clone();
        StepCheck::new(move |progress: RunProgress| {
            seen.lock().unwrap().push((
                progress.steps,
                progress.messages.len(),
                progress.usage.map(|u| (u.input_tokens, u.output_tokens)),
            ));
            async { Ok(StepDecision::Continue) }
        })
    };
    let mut run = Agent::new(vec![noop("work")]).run_with_options(
        provider.model("m"),
        vec![user_message("go")],
        options(vec![check]),
    );
    finish(&mut run).await;
    assert_eq!(
        *seen.lock().unwrap(),
        vec![(1, 3, Some((10, 5))), (2, 4, Some((30, 12)))]
    );
}

/// Runs the loop again once after the first error, as a run middleware may.
struct RerunOnce;
impl suprnova_ai::agents::Middleware for RerunOnce {
    fn wrap_agent_run(
        &self,
        call: suprnova_ai::agents::RunContext,
        next: suprnova_ai::agents::middleware::RunNext,
    ) -> suprnova_ai::providers::EventStream {
        let source = next(call.clone());
        futures::stream::unfold(
            (call, next, source, false),
            |(call, next, mut source, rerun)| async move {
                match source.next().await {
                    Some(Err(_)) if !rerun => {
                        let source = next(call.clone());
                        Some((
                            Ok(Event::Retry(Default::default())),
                            (call, next, source, true),
                        ))
                    }
                    Some(event) => Some((event, (call, next, source, rerun))),
                    None => None,
                }
            },
        )
        .boxed()
    }
}

#[tokio::test]
async fn agt_001_a_rerun_of_the_loop_keeps_the_step_count_and_usage() {
    let usage = |input, output| Usage {
        input_tokens: input,
        output_tokens: output,
        ..Usage::default()
    };
    let provider = ScriptedProvider::new(
        "scripted",
        [
            ScriptedResponse::tool_call("c1", "work", json!({})).with_usage(usage(10, 5)),
            ScriptedResponse::error(ProviderError::from_status("bad", 400, None)),
            ScriptedResponse::tool_call("c2", "work", json!({})).with_usage(usage(20, 7)),
            ScriptedResponse::tool_call("c3", "work", json!({})),
            ScriptedResponse::text("unused"),
        ],
    );
    let seen: Arc<Mutex<Vec<(usize, Option<u64>)>>> = Arc::default();
    let record = {
        let seen = seen.clone();
        StepCheck::new(move |progress: RunProgress| {
            seen.lock()
                .unwrap()
                .push((progress.steps, progress.usage.map(|u| u.input_tokens)));
            async { Ok(StepDecision::Continue) }
        })
    };
    let mut run = Agent::new(vec![noop("work")]).run_with_options(
        provider.model("m"),
        vec![user_message("go")],
        RunOptions {
            step_checks: vec![record, StepCheck::step_count(2)],
            middleware: vec![Arc::new(RerunOnce)],
            ..RunOptions::default()
        },
    );
    finish(&mut run).await;
    // Step 1, the failed call, then step 2 in the rerun, which meets the budget.
    assert_eq!(provider.requests().len(), 3);
    assert_eq!(*seen.lock().unwrap(), vec![(1, Some(10)), (2, Some(30))]);
    assert_eq!(
        run.stop_reason(),
        Some(StopReason::Stopped("step count 2 reached".into()))
    );
}

#[tokio::test]
async fn agt_001_steering_with_another_role_ends_the_run_with_a_validation_error() {
    let provider = ScriptedProvider::new("scripted", [ScriptedResponse::text("draft")]);
    let check =
        StepCheck::new(|_| async { Ok(StepDecision::Steer(assistant_message("not mine"))) });
    let mut run = Agent::default().run_with_options(
        provider.model("m"),
        vec![user_message("write")],
        options(vec![check]),
    );
    let mut error = None;
    while let Some(event) = run.next().await {
        if let Err(failure) = event {
            error = Some(failure);
        }
    }
    assert!(matches!(error, Some(Error::Validation(_))), "{error:?}");
    assert_eq!(provider.requests().len(), 1);
}

// AGT-002

#[tokio::test]
async fn agt_002_unknown_tools_become_error_results_the_model_sees() {
    let provider = ScriptedProvider::new(
        "scripted",
        [
            ScriptedResponse::tool_call("c1", "missing", json!({})),
            ScriptedResponse::text("corrected"),
        ],
    );
    let mut run = Agent::new(vec![noop("beta"), noop("alpha")])
        .run(provider.model("m"), vec![user_message("go")]);
    let events = finish(&mut run).await;
    let expected = "Model tried to call unavailable tool 'missing'. Available tools: alpha, beta.";
    let delivered: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            Event::ToolCallResult(result) => Some(result.results[0].clone()),
            _ => None,
        })
        .collect();
    assert_eq!(delivered.len(), 1);
    assert!(delivered[0].is_error());
    assert_eq!(delivered[0].tool_call_id, "c1");
    assert_eq!(result_text(&delivered[0]), expected);
    let requests = provider.requests();
    assert_eq!(requests.len(), 2);
    let sent = requests[1].messages.last().unwrap().tool_results();
    assert_eq!(sent[0].tool_call_id, "c1");
    assert!(sent[0].is_error());
    assert_eq!(result_text(sent[0]), expected);
    assert_eq!(run.text().unwrap(), "corrected");
}

// AGT-003

#[derive(Default)]
struct Concurrency {
    running: AtomicUsize,
    most: AtomicUsize,
    starts: Mutex<Vec<u32>>,
}

fn measured(state: Arc<Concurrency>) -> AgentTool {
    AgentTool::new::<Index, u32, _, _>("slow", "Takes a while", move |input, _| {
        let state = state.clone();
        async move {
            state.starts.lock().unwrap().push(input.i);
            let now = state.running.fetch_add(1, Ordering::SeqCst) + 1;
            state.most.fetch_max(now, Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(20)).await;
            state.running.fetch_sub(1, Ordering::SeqCst);
            Ok(input.i)
        }
    })
    .unwrap()
}

async fn run_five_calls(limit: Option<NonZeroUsize>) -> (Arc<Concurrency>, AgentStream) {
    let calls = (1..=5).map(|i| {
        suprnova_ai::types::ToolCallPart::new(format!("c{i}"), "slow", json!({"i": i}).to_string())
    });
    let provider = ScriptedProvider::new(
        "scripted",
        [
            ScriptedResponse::tool_calls(calls),
            ScriptedResponse::text("done"),
        ],
    );
    let state = Arc::new(Concurrency::default());
    let mut run = Agent::new(vec![measured(state.clone())]).run_with_options(
        provider.model("m"),
        vec![user_message("go")],
        RunOptions {
            tool_concurrency: limit,
            ..RunOptions::default()
        },
    );
    finish(&mut run).await;
    (state, run)
}

#[tokio::test(start_paused = true)]
async fn agt_003_tool_concurrency_caps_running_tools_in_call_order() {
    let (state, run) = run_five_calls(NonZeroUsize::new(2)).await;
    assert_eq!(state.most.load(Ordering::SeqCst), 2);
    assert_eq!(*state.starts.lock().unwrap(), vec![1, 2, 3, 4, 5]);
    let history = run.messages();
    let results = history
        .iter()
        .find(|message| message.role == MessageRole::Tool)
        .unwrap()
        .tool_results();
    assert_eq!(results.len(), 5);
}

#[tokio::test(start_paused = true)]
async fn agt_003_without_a_cap_every_call_runs_at_once() {
    let (state, _) = run_five_calls(None).await;
    assert_eq!(state.most.load(Ordering::SeqCst), 5);
}

// AGT-004

#[tokio::test]
async fn agt_004_usage_sums_every_delivered_response() {
    let usage = |input, output| Usage {
        input_tokens: input,
        output_tokens: output,
        ..Usage::default()
    };
    let provider = ScriptedProvider::new(
        "scripted",
        [
            ScriptedResponse::tool_call("c1", "work", json!({})).with_usage(usage(10, 5)),
            ScriptedResponse::text("done").with_usage(usage(20, 7)),
        ],
    );
    let mut run = Agent::new(vec![noop("work")]).run(provider.model("m"), vec![user_message("go")]);
    finish(&mut run).await;
    let total = run.usage().unwrap();
    assert_eq!((total.input_tokens, total.output_tokens), (30, 12));
}

#[tokio::test]
async fn agt_004_runs_without_reported_usage_have_none() {
    let provider = ScriptedProvider::new("scripted", [ScriptedResponse::text("done")]);
    let mut run = Agent::default().run(provider.model("m"), vec![user_message("go")]);
    finish(&mut run).await;
    assert_eq!(run.usage(), None);
}

// AGT-005

#[tokio::test]
async fn agt_005_a_final_answer_is_done() {
    let provider = ScriptedProvider::new("scripted", [ScriptedResponse::text("done")]);
    let mut run = Agent::default().run(provider.model("m"), vec![user_message("go")]);
    assert!(run.next().await.unwrap().is_ok());
    assert_eq!(run.stop_reason(), None, "reported before the stream ended");
    finish(&mut run).await;
    assert_eq!(run.stop_reason(), Some(StopReason::Done));
}

#[tokio::test]
async fn agt_005_a_step_check_stop_reports_its_reason() {
    let provider = ScriptedProvider::new("scripted", tool_calls(3, "work"));
    let mut run = Agent::new(vec![noop("work")]).run_with_options(
        provider.model("m"),
        vec![user_message("go")],
        options(vec![StepCheck::step_count(1)]),
    );
    finish(&mut run).await;
    assert_eq!(
        run.stop_reason(),
        Some(StopReason::Stopped("step count 1 reached".into()))
    );
}

#[tokio::test]
async fn agt_005_a_deferred_approval_reports_deferred() {
    let provider = ScriptedProvider::new(
        "scripted",
        [ScriptedResponse::tool_call("c1", "gated", json!({}))],
    );
    let gated = noop("gated").require_approval(true);
    let mut run = Agent::new(vec![gated]).run(provider.model("m"), vec![user_message("go")]);
    while let Some(event) = run.next().await {
        if let Event::Hook(hook) = event.unwrap()
            && hook.hook.status == HookStatus::Pending
        {
            run.hook_registry()
                .defer_hook(&hook.hook.hook_id, hook.hook.clone())
                .unwrap();
        }
    }
    assert_eq!(run.stop_reason(), Some(StopReason::Deferred));
}

#[tokio::test]
async fn agt_005_a_run_that_ends_with_an_error_has_no_reason() {
    let provider = ScriptedProvider::new(
        "scripted",
        [ScriptedResponse::error(ProviderError::from_status(
            "bad", 400, None,
        ))],
    );
    let mut run = Agent::default().run(provider.model("m"), vec![user_message("go")]);
    let mut failed = false;
    while let Some(event) = run.next().await {
        failed |= event.is_err();
    }
    assert!(failed);
    assert_eq!(run.stop_reason(), None);
}
