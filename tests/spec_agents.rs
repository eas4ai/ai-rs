//! Tests for AGT-001 to AGT-006 in docs/spec/agents.md. Each test name starts
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
        Event, HookStatus, Message, MessageRole, Part, ToolCallPart, ToolResultPart,
        events::{TextDelta, replay_message_events},
        messages::{TextPart, ToolResultValue},
        usage::Usage,
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
    // Each entry: steps and input usage tokens.
    type Seen = Vec<(usize, Option<u64>)>;
    let seen: Arc<Mutex<Seen>> = Arc::default();
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
async fn agt_001_checks_do_not_run_after_a_step_that_ends_on_a_deferred_hook() {
    let provider = ScriptedProvider::new(
        "scripted",
        [ScriptedResponse::tool_call("c1", "gated", json!({}))],
    );
    let checked = Arc::new(AtomicUsize::new(0));
    // A check that ran would steer, and the run could not end deferred.
    let steer = {
        let checked = checked.clone();
        StepCheck::new(move |_| {
            checked.fetch_add(1, Ordering::SeqCst);
            async { Ok(StepDecision::Steer(user_message("keep going"))) }
        })
    };
    let gated = noop("gated").require_approval(true);
    let mut run = Agent::new(vec![gated]).run_with_options(
        provider.model("m"),
        vec![user_message("go")],
        options(vec![steer]),
    );
    while let Some(event) = run.next().await {
        if let Event::Hook(hook) = event.unwrap()
            && hook.hook.status == HookStatus::Pending
        {
            run.hook_registry()
                .defer_hook(&hook.hook.hook_id, hook.hook.clone())
                .unwrap();
        }
    }
    assert_eq!(checked.load(Ordering::SeqCst), 0);
    assert_eq!(run.stop_reason(), Some(StopReason::Deferred));
    let history = run.messages();
    let last = history.last().unwrap();
    assert!(
        last.tool_results().iter().all(|r| r.is_hook_deferred),
        "{last:?}"
    );
    assert_eq!(provider.requests().len(), 1);
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

#[tokio::test]
async fn agt_003_capped_steps_still_block_on_approvals_and_finish() {
    let calls = [("g1", "gated"), ("p1", "work"), ("g2", "gated")]
        .map(|(id, name)| suprnova_ai::types::ToolCallPart::new(id, name, json!({}).to_string()));
    let provider = ScriptedProvider::new(
        "scripted",
        [
            ScriptedResponse::tool_calls(calls),
            ScriptedResponse::text("done"),
        ],
    );
    let gated = noop("gated").require_approval(true);
    let mut run = Agent::new(vec![gated, noop("work")]).run_with_options(
        provider.model("m"),
        vec![user_message("go")],
        RunOptions {
            tool_concurrency: NonZeroUsize::new(1),
            ..RunOptions::default()
        },
    );
    // Approve only when the run reports that it is blocked on its hooks, as an
    // application that waits for RunBlocked does.
    let drained = tokio::time::timeout(Duration::from_secs(10), async {
        let mut approved = 0;
        while let Some(event) = run.next().await {
            if let Event::RunBlocked(blocked) = event.unwrap() {
                for hook in blocked.hooks {
                    run.hook_registry()
                        .resolve_hook(&hook.hook_id, json!({"granted": true}))
                        .unwrap();
                    approved += 1;
                }
            }
        }
        approved
    })
    .await;
    assert_eq!(
        drained.expect("the capped run never reported RunBlocked"),
        2
    );
    assert_eq!(run.text().unwrap(), "done");
    let history = run.messages();
    let results = history
        .iter()
        .find(|message| message.role == MessageRole::Tool)
        .unwrap()
        .tool_results();
    assert_eq!(results.len(), 3);
    assert!(
        results.iter().all(|result| !result.is_error()),
        "{results:?}"
    );
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

// AGT-006

const CANCELLED: &str = "Cancelled: the run stopped before this tool call finished.";

/// The events that stream `Hello` as the deltas `Hel` and `lo`.
fn hello_events() -> Vec<Event> {
    let message = Message::new(
        MessageRole::Assistant,
        vec![
            TextPart {
                id: "t1".into(),
                text: "Hello".into(),
                provider_metadata: None,
            }
            .into(),
        ],
    );
    let mut events = Vec::new();
    for event in replay_message_events(&message) {
        match event {
            Event::TextDelta(delta) => {
                for chunk in ["Hel", "lo"] {
                    events.push(Event::TextDelta(TextDelta {
                        chunk: chunk.into(),
                        ..delta.clone()
                    }));
                }
            }
            Event::StreamEnd(mut end) => {
                end.finish_reason = Some("stop".into());
                events.push(Event::StreamEnd(end));
            }
            event => events.push(event),
        }
    }
    events
}

fn hello_in_two_deltas() -> ScriptedResponse {
    ScriptedResponse::events(hello_events())
}

fn lockstep() -> RunOptions {
    RunOptions {
        loop_buffer: Some(0),
        ..RunOptions::default()
    }
}

/// Read the run until `stop` returns true for a delivered event.
async fn read_until(run: &mut AgentStream, mut stop: impl FnMut(&Event) -> bool) -> Vec<Event> {
    tokio::time::timeout(Duration::from_secs(10), async {
        let mut events = Vec::new();
        while let Some(event) = run.next().await {
            let event = event.unwrap();
            let done = stop(&event);
            events.push(event);
            if done {
                return events;
            }
        }
        panic!("the run ended before the awaited event: {events:?}");
    })
    .await
    .expect("the awaited event never arrived")
}

fn is_delta(event: &Event, chunk: &str) -> bool {
    matches!(event, Event::TextDelta(delta) if delta.chunk == chunk)
}

fn call_ids(message: &Message) -> Vec<&str> {
    message
        .tool_calls()
        .iter()
        .map(|call| call.tool_call_id.as_str())
        .collect()
}

/// A tool that counts its starts and never finishes for the index 0.
fn counted(name: &str, starts: Arc<Mutex<Vec<u32>>>) -> AgentTool {
    AgentTool::new::<Index, u32, _, _>(name, "Return the index", move |input, _| {
        let starts = starts.clone();
        async move {
            starts.lock().unwrap().push(input.i);
            if input.i == 0 {
                std::future::pending::<()>().await;
            }
            Ok(input.i)
        }
    })
    .unwrap()
}

#[tokio::test]
async fn agt_006_a_lockstep_run_keeps_the_partial_text() {
    let provider = ScriptedProvider::new("scripted", [hello_in_two_deltas()]);
    let mut run = Agent::default().run_with_options(
        provider.model("m"),
        vec![user_message("go")],
        lockstep(),
    );
    read_until(&mut run, |event| is_delta(event, "Hel")).await;
    let history = run.interrupted_messages();
    assert_eq!(history.len(), 2, "{history:?}");
    let last = history.last().unwrap();
    assert_eq!(last.role, MessageRole::Assistant);
    assert_eq!(last.text(), "Hel");
    assert_eq!(
        run.messages().len(),
        1,
        "the step is not in the history yet"
    );
}

#[tokio::test]
async fn agt_006_a_run_ahead_keeps_the_steps_the_loop_finished() {
    let provider = ScriptedProvider::new("scripted", [hello_in_two_deltas()]);
    let mut run = Agent::default().run(provider.model("m"), vec![user_message("go")]);
    read_until(&mut run, |event| is_delta(event, "Hel")).await;
    // The buffered loop finishes the step while the consumer waits.
    tokio::time::timeout(Duration::from_secs(10), async {
        while run.messages().len() < 2 {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .expect("the loop never ran ahead");
    assert_eq!(run.messages().last().unwrap().text(), "Hello");
    // The finished step is the run's history; no step is in progress.
    let history = run.interrupted_messages();
    assert_eq!(history, run.messages());
}

#[tokio::test]
async fn agt_006_a_failed_step_keeps_its_partial_text() {
    // The stream breaks after the delta `Hel`.
    let mut events = hello_events();
    let cut = events
        .iter()
        .position(|event| is_delta(event, "Hel"))
        .unwrap();
    events.truncate(cut + 1);
    let provider = ScriptedProvider::new(
        "scripted",
        [ScriptedResponse::events(events)
            .then_error(ProviderError::from_status("broken", 400, None))],
    );
    let mut run = Agent::default().run_with_options(
        provider.model("m"),
        vec![user_message("go")],
        lockstep(),
    );
    let mut failed = false;
    while let Some(event) = run.next().await {
        failed |= event.is_err();
    }
    assert!(failed);
    let history = run.interrupted_messages();
    assert_eq!(history.len(), 2, "{history:?}");
    assert_eq!(history.last().unwrap().text(), "Hel");
}

#[tokio::test]
async fn agt_006_without_a_step_in_progress_it_is_the_history() {
    let provider = ScriptedProvider::new("scripted", tool_calls(1, "work"));
    provider.push(ScriptedResponse::text("done"));
    let mut run = Agent::new(vec![noop("work")]).run(provider.model("m"), vec![user_message("go")]);
    assert_eq!(run.interrupted_messages(), run.messages());
    finish(&mut run).await;
    assert_eq!(run.messages().len(), 4);
    assert_eq!(run.interrupted_messages(), run.messages());
}

#[tokio::test]
async fn agt_006_a_stopped_step_keeps_delivered_results_and_cancels_the_rest() {
    // Call a finishes at once, b never finishes, and c is still streaming.
    let calls = [("a", 1), ("b", 0), ("c", 3)]
        .map(|(id, i)| ToolCallPart::new(id, "work", json!({ "i": i }).to_string()));
    let provider = ScriptedProvider::new("scripted", [ScriptedResponse::tool_calls(calls)]);
    let starts = Arc::new(Mutex::new(Vec::new()));
    let agent = Agent::new(vec![counted("work", starts.clone())]);
    let mut run = agent.run_with_options(provider.model("m"), vec![user_message("go")], lockstep());
    let mut result_a = false;
    let mut started_c = false;
    let events = read_until(&mut run, |event| {
        match event {
            Event::ToolCallResult(result) => {
                result_a |= result.results.iter().any(|r| r.tool_call_id == "a");
            }
            Event::ToolStart(start) => started_c |= start.tool_call_id == "c",
            _ => {}
        }
        result_a && started_c
    })
    .await;
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, Event::ToolEnd(end) if end.tool_call.tool_call_id == "c")),
        "the scenario needs c's ToolEnd undelivered: {events:?}"
    );

    let history = run.interrupted_messages();
    drop(run);
    assert_eq!(history.len(), 3, "{history:?}");
    assert_eq!(history[1].role, MessageRole::Assistant);
    assert_eq!(call_ids(&history[1]), ["a", "b"]);
    assert_eq!(history[2].role, MessageRole::Tool);
    let results = history[2].tool_results();
    assert_eq!(results.len(), 2, "{results:?}");
    assert_eq!(results[0].tool_call_id, "a");
    assert!(!results[0].is_error());
    assert_eq!(results[1].tool_call_id, "b");
    assert!(results[1].is_error());
    assert_eq!(result_text(results[1]), CANCELLED);

    // A next run continues from the history and runs neither call again.
    let next = ScriptedProvider::new("scripted", [ScriptedResponse::text("done")]);
    let mut run = agent.run(next.model("m"), history.clone());
    finish(&mut run).await;
    assert_eq!(run.text().unwrap(), "done");
    assert_eq!(*starts.lock().unwrap(), [1, 0]);
    let sent = &next.requests()[0].messages;
    assert_eq!(sent.len(), 3);
    assert_eq!(sent[2].tool_results().len(), 2);
}

#[tokio::test]
async fn agt_006_a_replayed_step_keeps_the_results_it_delivered() {
    let starts = Arc::new(Mutex::new(Vec::new()));
    let agent = Agent::new(vec![counted("work", starts.clone())]);
    let mut deferred = ToolResultPart::error("wait", "work", json!("Deferred"));
    deferred.is_hook_deferred = true;
    let history = vec![
        user_message("go"),
        Message::new(
            MessageRole::Assistant,
            vec![
                Part::ToolCall(ToolCallPart::new("done", "work", r#"{"i":4}"#)),
                Part::ToolCall(ToolCallPart::new("wait", "work", r#"{"i":5}"#)),
            ],
        ),
        Message::new(
            MessageRole::Tool,
            vec![
                ToolResultPart::new("done", "work", json!(4)).into(),
                deferred.into(),
            ],
        ),
    ];
    let provider = ScriptedProvider::new("scripted", [ScriptedResponse::text("done")]);
    let mut run = agent.run_with_options(provider.model("m"), history, lockstep());
    read_until(&mut run, |event| {
        matches!(event, Event::ToolCallResult(result)
            if result.results.iter().any(|r| r.tool_call_id == "wait"))
    })
    .await;
    let history = run.interrupted_messages();
    drop(run);
    assert!(
        provider.requests().is_empty(),
        "the model was called before the stop"
    );
    assert_eq!(history.len(), 3, "{history:?}");
    assert_eq!(call_ids(&history[1]), ["done", "wait"]);
    let results = history[2].tool_results();
    assert_eq!(
        results
            .iter()
            .map(|r| (r.tool_call_id.as_str(), r.is_error()))
            .collect::<Vec<_>>(),
        [("done", false), ("wait", false)]
    );

    let next = ScriptedProvider::new("scripted", [ScriptedResponse::text("done")]);
    let mut run = agent.run(next.model("m"), history);
    finish(&mut run).await;
    assert_eq!(*starts.lock().unwrap(), [5], "a finished call ran again");
}

#[tokio::test]
async fn agt_006_a_steer_after_a_delivered_step_stays_in_the_history() {
    for buffer in [Some(0), None] {
        for with_tool in [false, true] {
            let mut script = Vec::new();
            if with_tool {
                script.push(ScriptedResponse::tool_call("c1", "work", json!({"i": 1})));
            } else {
                script.push(ScriptedResponse::text("done"));
            }
            // The model call after the steer fails, so the run stops there.
            script.push(ScriptedResponse::error(ProviderError::from_status(
                "broken", 400, None,
            )));
            let provider = ScriptedProvider::new("scripted", script);
            let steer_once = StepCheck::new(|progress: RunProgress| async move {
                Ok(if progress.steps == 1 {
                    StepDecision::Steer(user_message("go on"))
                } else {
                    StepDecision::Continue
                })
            });
            let mut run = Agent::new(vec![noop("work")]).run_with_options(
                provider.model("m"),
                vec![user_message("go")],
                RunOptions {
                    step_checks: vec![steer_once],
                    loop_buffer: buffer,
                    ..RunOptions::default()
                },
            );
            let mut failed = false;
            while let Some(event) = run.next().await {
                failed |= event.is_err();
            }
            assert!(failed, "{buffer:?} {with_tool}");
            let history = run.interrupted_messages();
            let roles: Vec<_> = history.iter().map(|m| m.role).collect();
            let expected = if with_tool {
                vec![
                    MessageRole::User,
                    MessageRole::Assistant,
                    MessageRole::Tool,
                    MessageRole::User,
                ]
            } else {
                vec![MessageRole::User, MessageRole::Assistant, MessageRole::User]
            };
            assert_eq!(roles, expected, "{buffer:?} {with_tool}: {history:?}");
            assert_eq!(history.last().unwrap().text(), "go on");
            assert_eq!(history, run.messages(), "{buffer:?} {with_tool}");
        }
    }
}

/// A protocol that sends these events and then nothing, as a model that
/// stalls mid-response.
struct StallsAfter(Vec<Event>);
impl suprnova_ai::ProviderProtocol for StallsAfter {
    fn protocol_class_id(&self) -> &str {
        "stalls"
    }
    fn stream(
        &self,
        _model: &suprnova_ai::Model,
        _messages: Vec<Message>,
        _options: suprnova_ai::StreamOptions,
        _provider: &dyn suprnova_ai::Provider,
    ) -> suprnova_ai::providers::EventStream {
        futures::stream::iter(self.0.clone().into_iter().map(Ok))
            .chain(futures::stream::pending())
            .boxed()
    }
}

/// Sets its flag when dropped.
struct DropFlag(Arc<AtomicBool>);
impl Drop for DropFlag {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

#[tokio::test]
async fn agt_006_a_stopped_buffered_run_keeps_what_the_loop_received() {
    // Reasoning R, complete calls a and b, text T, then the start of call c.
    let response = Message::new(
        MessageRole::Assistant,
        vec![
            Part::Reasoning(suprnova_ai::types::messages::ReasoningPart {
                id: "r1".into(),
                text: "R".into(),
                provider_metadata: None,
            }),
            Part::ToolCall(ToolCallPart::new("a", "work", r#"{"i":1}"#)),
            Part::ToolCall(ToolCallPart::new("b", "work", r#"{"i":0}"#)),
            TextPart {
                id: "t1".into(),
                text: "T".into(),
                provider_metadata: None,
            }
            .into(),
            Part::ToolCall(ToolCallPart::new("c", "work", r#"{"i":3}"#)),
        ],
    );
    let mut events = replay_message_events(&response);
    let cut = events
        .iter()
        .position(|event| matches!(event, Event::ToolEnd(end) if end.tool_call_id == "c"))
        .unwrap();
    events.truncate(cut);
    let provider = Arc::new(suprnova_ai::providers::ConfiguredProvider::new(
        "stalls",
        Default::default(),
        Arc::new(StallsAfter(events)),
        None,
    ));
    let (finished, finished_rx) = tokio::sync::oneshot::channel::<()>();
    let finished = Arc::new(Mutex::new(Some(finished)));
    let dropped = Arc::new(AtomicBool::new(false));
    let starts = Arc::new(Mutex::new(Vec::new()));
    let tool = {
        let (finished, dropped, starts) = (finished.clone(), dropped.clone(), starts.clone());
        AgentTool::new::<Index, u32, _, _>("work", "Return the index", move |input, _| {
            let (finished, dropped, starts) = (finished.clone(), dropped.clone(), starts.clone());
            async move {
                starts.lock().unwrap().push(input.i);
                if input.i == 0 {
                    let _guard = DropFlag(dropped);
                    std::future::pending::<()>().await;
                }
                if let Some(finished) = finished.lock().unwrap().take() {
                    let _ = finished.send(());
                }
                Ok(input.i)
            }
        })
        .unwrap()
    };
    let agent = Agent::new(vec![tool]);
    let mut run = agent.run(
        suprnova_ai::Model::new("m", provider),
        vec![user_message("go")],
    );
    let first = run.next().await;
    assert!(matches!(first, Some(Ok(_))), "{first:?}");
    tokio::time::timeout(Duration::from_secs(10), finished_rx)
        .await
        .expect("tool a never finished")
        .unwrap();
    run.stop().await;
    assert!(dropped.load(Ordering::SeqCst), "b's future was not dropped");
    assert!(run.next().await.is_none(), "the stream yielded after stop");

    let history = run.interrupted_messages();
    assert_eq!(history.len(), 3, "{history:?}");
    let parts: Vec<_> = history[1]
        .parts
        .iter()
        .map(|part| match part {
            Part::Reasoning(r) => format!("reasoning {}", r.text),
            Part::ToolCall(c) => format!("call {}", c.tool_call_id),
            Part::Text(t) => format!("text {}", t.text),
            other => format!("{other:?}"),
        })
        .collect();
    assert_eq!(parts, ["reasoning R", "call a", "call b"]);
    let results = history[2].tool_results();
    assert_eq!(
        results
            .iter()
            .map(|r| (r.tool_call_id.as_str(), r.is_error()))
            .collect::<Vec<_>>(),
        [("a", false), ("b", true)]
    );
    assert_eq!(result_text(results[1]), CANCELLED);

    // A next run continues from the history and runs neither call again.
    let next = ScriptedProvider::new("scripted", [ScriptedResponse::text("done")]);
    let mut run = agent.run(next.model("m"), history);
    finish(&mut run).await;
    assert_eq!(run.text().unwrap(), "done");
    assert_eq!(*starts.lock().unwrap(), [1, 0]);
}

#[tokio::test]
async fn agt_006_stop_ends_a_lockstep_run_and_drops_its_tools() {
    let dropped = Arc::new(AtomicBool::new(false));
    let tool = {
        let dropped = dropped.clone();
        AgentTool::new::<Index, u32, _, _>("work", "Never finish", move |_, _| {
            let dropped = dropped.clone();
            async move {
                let _guard = DropFlag(dropped);
                std::future::pending::<()>().await;
                Ok(0)
            }
        })
        .unwrap()
    };
    let provider = ScriptedProvider::new(
        "scripted",
        [ScriptedResponse::tool_call("a", "work", json!({"i": 0}))],
    );
    let mut run = Agent::new(vec![tool]).run_with_options(
        provider.model("m"),
        vec![user_message("go")],
        lockstep(),
    );
    read_until(&mut run, |event| matches!(event, Event::StreamEnd(_))).await;
    // The tool starts while the loop waits for its result.
    let pending = tokio::time::timeout(Duration::from_millis(50), run.next()).await;
    assert!(pending.is_err(), "{pending:?}");
    run.stop().await;
    assert!(
        dropped.load(Ordering::SeqCst),
        "the tool's future was not dropped"
    );
    assert!(run.next().await.is_none());
    let history = run.interrupted_messages();
    assert_eq!(history.len(), 3, "{history:?}");
    assert_eq!(result_text(history[2].tool_results()[0]), CANCELLED);
}

#[tokio::test]
async fn agt_006_tools_that_finish_together_both_keep_their_results() {
    // Both tools wait on one gate, so they finish in the same poll of the
    // step's tools; the consumer reads only the first result, then stops.
    let gate = Arc::new(tokio::sync::Semaphore::new(0));
    let tool = {
        let gate = gate.clone();
        AgentTool::new::<Index, u32, _, _>("work", "Wait for the gate", move |input, _| {
            let gate = gate.clone();
            async move {
                gate.acquire().await.unwrap().forget();
                Ok(input.i)
            }
        })
        .unwrap()
    };
    let calls = [("a", 1), ("b", 2)]
        .map(|(id, i)| ToolCallPart::new(id, "work", json!({ "i": i }).to_string()));
    let provider = ScriptedProvider::new("scripted", [ScriptedResponse::tool_calls(calls)]);
    let mut run = Agent::new(vec![tool]).run_with_options(
        provider.model("m"),
        vec![user_message("go")],
        lockstep(),
    );
    read_until(&mut run, |event| matches!(event, Event::StreamEnd(_))).await;
    // Both tools start and wait at the gate.
    let pending = tokio::time::timeout(Duration::from_millis(50), run.next()).await;
    assert!(pending.is_err(), "{pending:?}");
    gate.add_permits(2);
    let first = run.next().await;
    assert!(
        matches!(first, Some(Ok(Event::ToolCallResult(_)))),
        "{first:?}"
    );
    run.stop().await;
    let history = run.interrupted_messages();
    assert_eq!(history.len(), 3, "{history:?}");
    let results: Vec<_> = history[2]
        .tool_results()
        .iter()
        .map(|r| (r.tool_call_id.clone(), r.is_error()))
        .collect();
    assert_eq!(results, [("a".to_owned(), false), ("b".to_owned(), false)]);
}
