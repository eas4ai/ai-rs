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

//! The agent step loop and its owned stream handle.
use super::{
    buffer::{BufferStop, BufferedStream},
    checks::{RunProgress, StepCheck, StepDecision, StopReason},
    hooks::HookRegistry,
    middleware::{self, Middleware, ModelContext},
    runtime::Runtime,
    tools::{
        AgentTool, BoundToolCall, ToolContext, ToolResolver, ToolRunner, UnavailableToolError,
        error_tool_result, tool_result,
    },
};
use crate::{
    Error, Model, StreamOptions,
    providers::EventStream,
    types::{
        Event, HookPart, HookStatus, Message, MessageRole, Part, ToolCallPart, ToolResultPart,
        events::{ModelEventMeta, RunStateTracker, StreamEnd, StreamStart, ToolCallResult},
        tools::Tool,
        usage::Usage,
    },
};
use futures::{Stream, StreamExt, future::Either, stream};
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use std::{
    collections::{BTreeMap, VecDeque},
    num::NonZeroUsize,
    pin::Pin,
    sync::{Arc, Mutex},
    task::{Context as TaskContext, Poll},
};

/// The model input and accumulated history. Adding replay messages is a no-op.
#[derive(Clone, Debug)]
pub struct Context {
    pub model: Model,
    pub messages: Vec<Message>,
    pub tools: Vec<Tool>,
    pub options: StreamOptions,
}
impl Context {
    pub fn new(
        model: Model,
        messages: Vec<Message>,
        tools: Vec<Tool>,
        options: StreamOptions,
    ) -> Self {
        Self {
            model,
            messages,
            tools,
            options,
        }
    }
    pub fn keep_running(&self) -> bool {
        self.messages.last().is_some_and(|last| {
            !last.tool_results().iter().any(|r| r.is_hook_deferred)
                && (last.replay
                    || !matches!(last.role, MessageRole::Assistant | MessageRole::Internal))
        })
    }
    pub fn add(&mut self, message: Message) {
        if !message.replay {
            self.messages.push(message)
        }
    }
    pub fn add_all(&mut self, messages: impl IntoIterator<Item = Message>) {
        for message in messages {
            self.add(message)
        }
    }
    /// Whether the history ends with a result placeholder for a deferred hook.
    fn ends_deferred(&self) -> bool {
        self.messages
            .last()
            .is_some_and(|last| last.tool_results().iter().any(|r| r.is_hook_deferred))
    }
}

#[derive(Clone, Default)]
pub struct RunOptions {
    pub stream_options: StreamOptions,
    pub hook_registry: Option<HookRegistry>,
    pub middleware: Vec<Arc<dyn Middleware>>,
    /// None runs ahead without a limit; Some(0) advances in lockstep.
    pub loop_buffer: Option<usize>,
    /// Checks the default loop runs after each step. Empty by default, so a
    /// run stops only when it is done.
    pub step_checks: Vec<StepCheck>,
    /// The most tool executions of one step that run at once. Unset runs
    /// every call of a step at once.
    pub tool_concurrency: Option<NonZeroUsize>,
}
impl RunOptions {
    pub fn with_output<T: JsonSchema>(mut self) -> Self {
        self.stream_options = self.stream_options.with_output::<T>();
        self
    }
}

type Loop = Arc<dyn Fn(RunContext) -> EventStream + Send + Sync>;
#[derive(Clone, Default)]
pub struct Agent {
    tools: Vec<AgentTool>,
    provider_tools: Vec<Tool>,
    custom_loop: Option<Loop>,
}
impl Agent {
    pub fn new(tools: Vec<AgentTool>) -> Self {
        Self {
            tools,
            ..Self::default()
        }
    }
    pub fn tools(&self) -> &[AgentTool] {
        &self.tools
    }
    pub fn with_provider_tools(mut self, tools: Vec<Tool>) -> Self {
        self.provider_tools = tools;
        self
    }
    pub fn with_loop(
        mut self,
        run: impl Fn(RunContext) -> EventStream + Send + Sync + 'static,
    ) -> Self {
        self.custom_loop = Some(Arc::new(run));
        self
    }
    pub fn resolve(&self, part: ToolCallPart) -> Result<BoundToolCall, Error> {
        ToolResolver::new(self.tools.clone()).resolve(part)
    }
    pub fn run(&self, model: Model, messages: Vec<Message>) -> AgentStream {
        self.run_with_options(model, messages, RunOptions::default())
    }
    pub fn run_with_options(
        &self,
        model: Model,
        messages: Vec<Message>,
        options: RunOptions,
    ) -> AgentStream {
        let declarations = self
            .tools
            .iter()
            .map(|t| t.declaration().clone())
            .chain(self.provider_tools.clone())
            .collect();
        let context = Arc::new(Mutex::new(Context::new(
            model,
            messages,
            declarations,
            options.stream_options,
        )));
        let hooks = options
            .hook_registry
            .unwrap_or_default()
            .with_middleware(options.middleware.clone());
        let (runtime, events) = Runtime::new();
        let tally = Arc::new(Mutex::new(RunTally::default()));
        let call = RunContext {
            context: context.clone(),
            agent: Arc::new(self.clone()),
            runtime,
            hooks: hooks.clone(),
            middleware: options.middleware.clone(),
            step_checks: options.step_checks.clone(),
            tool_concurrency: options.tool_concurrency,
            tally: tally.clone(),
        };
        // A middleware may call next again to retry a failed run. Each
        // invocation owns its source and queue; the first uses the exposed sink.
        let receiver = Arc::new(Mutex::new(Some(events)));
        let real: middleware::RunNext = Arc::new(move |mut call| {
            let events = receiver.lock().expect("agent source mutex poisoned").take();
            let events = if let Some(events) = events {
                events
            } else {
                let (runtime, events) = Runtime::new();
                call.runtime = runtime;
                events
            };
            let resolver = ToolResolver::new(call.agent.tools.clone());
            let mut prepared = Ok(());
            call.update(|context| {
                prepared = populate_model_inputs(&mut context.messages, &resolver);
                process_interrupted_hooks(&mut context.messages);
            });
            let source = match prepared {
                Err(error) => stream::iter([Err(error)]).boxed(),
                Ok(()) => match call.agent.custom_loop.clone() {
                    Some(run) => run(call),
                    None => call.default_loop(),
                },
            };
            tracked(events.merge(source).boxed())
        });
        let source = middleware::run_chain(&options.middleware, real)(call);
        let (source, buffer): (EventStream, _) = match options.loop_buffer {
            Some(0) => (source, None),
            capacity => {
                let buffered = BufferedStream::new(source, capacity);
                let stop = buffered.stopper();
                (Box::pin(buffered), Some(stop))
            }
        };
        AgentStream {
            source,
            buffer,
            context,
            hooks,
            blocked: false,
            deferred: BTreeMap::new(),
            tally,
            usage: None,
            delivered: None,
            ended: false,
            failed: false,
            stopped: false,
        }
    }
}

/// Explicit run context passed to custom loops and nested runs.
#[derive(Clone)]
pub struct RunContext {
    context: Arc<Mutex<Context>>,
    pub agent: Arc<Agent>,
    pub runtime: Runtime,
    pub hooks: HookRegistry,
    pub middleware: Vec<Arc<dyn Middleware>>,
    /// Checks the default loop runs after each step.
    pub step_checks: Vec<StepCheck>,
    /// The most tool executions of one step that run at once.
    pub tool_concurrency: Option<NonZeroUsize>,
    tally: Arc<Mutex<RunTally>>,
}

/// Loop state for the whole run. Every default-loop invocation of the run
/// shares it, so a middleware that runs the loop again keeps the step count
/// and usage that the step checks budget against.
#[derive(Debug, Default)]
struct RunTally {
    steps: usize,
    usage: Option<Usage>,
    outcome: Option<StopReason>,
    /// The step in progress as the loop has it, which can be ahead of what
    /// the consumer has read. `Step::finish` clears it under this lock while
    /// it adds the step to the history, so a reader sees one or the other.
    step: Option<LoopStep>,
}

/// The step in progress as the loop received it.
#[derive(Debug)]
struct LoopStep {
    /// The model response so far, or for a replayed step the history's
    /// response whose calls it runs.
    message: Message,
    replayed: bool,
    /// The calls whose `ToolEnd` the loop received, in order.
    calls: Vec<String>,
    /// The results of the tools that finished.
    results: Vec<ToolResultPart>,
}
impl LoopStep {
    /// Add this step to `messages`, the history without it: the response up
    /// to its last received call, without the calls whose `ToolEnd` did not
    /// arrive, then each call's result, or a cancelled error result for a
    /// tool that had not finished. A replayed response is in the history
    /// already and keeps its calls and cached results.
    fn add_to(&self, messages: &mut Vec<Message>) {
        let calls: Vec<ToolCallPart> = if self.replayed {
            self.message.tool_calls().into_iter().cloned().collect()
        } else {
            let mut message = self.message.clone();
            let last = message.parts.iter().rposition(|part| {
                matches!(part, Part::ToolCall(call) if self.calls.contains(&call.tool_call_id))
            });
            message.parts.truncate(last.map_or(0, |last| last + 1));
            message.parts.retain(|part| match part {
                Part::ToolCall(call) => self.calls.contains(&call.tool_call_id),
                Part::Text(text) => !text.text.is_empty(),
                _ => true,
            });
            let calls = message.tool_calls().into_iter().cloned().collect();
            messages.push(message);
            calls
        };
        let parts = calls
            .iter()
            .map(|call| {
                self.results
                    .iter()
                    .find(|result| result.tool_call_id == call.tool_call_id)
                    .or(call.cached_result.as_ref())
                    .cloned()
                    .unwrap_or_else(|| {
                        ToolResultPart::error(
                            call.tool_call_id.as_str(),
                            call.tool_name.as_str(),
                            serde_json::Value::String(CANCELLED_TOOL_CALL.into()),
                        )
                    })
                    .into()
            })
            .collect();
        messages.push(Message::new(MessageRole::Tool, parts));
    }
}
impl RunContext {
    pub fn context(&self) -> Context {
        self.context
            .lock()
            .expect("agent context mutex poisoned")
            .clone()
    }
    pub fn update(&self, update: impl FnOnce(&mut Context)) {
        update(&mut self.context.lock().expect("agent context mutex poisoned"))
    }
    pub fn model_stream(&self) -> EventStream {
        let context = self.context();
        let mut options = context.options;
        options.tools = context.tools;
        let replay = context.messages.last().filter(|m| m.replay).cloned();
        let call = ModelContext {
            model: context.model,
            messages: context.messages,
            options,
        };
        let real: middleware::ModelNext =
            Arc::new(|call| crate::stream(&call.model, call.messages, call.options).boxed());
        let source = middleware::model_chain(&self.middleware, real)(call);
        // The core replay shortcut emits ToolEnd only. Supply the complete
        // response boundaries to run-state tracking without delivering them.
        if let Some(message) = replay {
            let mut meta = ModelEventMeta::with_message(message);
            meta.replay = true;
            stream::iter([Ok(Event::StreamStart(StreamStart {
                event: meta.clone(),
            }))])
            .chain(source)
            .chain(stream::iter([Ok(Event::StreamEnd(StreamEnd {
                event: meta,
                ..Default::default()
            }))]))
            .boxed()
        } else {
            source
        }
    }
    pub async fn generate(&self) -> Result<Message, Error> {
        let context = self.context();
        let mut options = context.options;
        options.tools = context.tools;
        let real: middleware::GenerateNext = Arc::new(|call| {
            Box::pin(async move { crate::generate(&call.model, call.messages, call.options).await })
        });
        middleware::generate_chain(&self.middleware, real)(ModelContext {
            model: context.model,
            messages: context.messages,
            options,
        })
        .await
    }
    pub fn nested_options(&self) -> RunOptions {
        RunOptions {
            hook_registry: Some(self.hooks.clone()),
            middleware: self.middleware.clone(),
            ..Default::default()
        }
    }
    /// Run every model step until the run is done, a step check stops it, or
    /// a hook is deferred. The step checks run after each step, except a
    /// replayed one or one that ended on a deferred hook.
    pub fn default_loop(&self) -> EventStream {
        stream::try_unfold(
            LoopState {
                call: self.clone(),
                step: None,
            },
            |mut state| async move {
                loop {
                    if let Some(step) = state.step.as_mut() {
                        if let Some(event) = step.next(&state.call).await? {
                            return Ok(Some((event, state)));
                        }
                        let step = state.step.take().expect("active agent step");
                        let progress = (!step.replay).then(|| state.call.count_step(step.usage()));
                        step.finish(&state.call)?;
                        if let Some((steps, usage)) = progress
                            && !state.call.context().ends_deferred()
                        {
                            match state.call.check_step(steps, usage).await? {
                                StepDecision::Continue => {}
                                StepDecision::Steer(message) => {
                                    state.call.update(|context| context.add(message));
                                }
                                StepDecision::Stop(reason) => {
                                    state.call.set_outcome(StopReason::Stopped(reason));
                                    return Ok(None);
                                }
                            }
                        }
                    }
                    let context = state.call.context();
                    if !context.keep_running() {
                        state.call.set_outcome(if context.ends_deferred() {
                            StopReason::Deferred
                        } else {
                            StopReason::Done
                        });
                        return Ok(None);
                    }
                    let replay = context.messages.last().is_some_and(|m| m.replay);
                    state.step = Some(Step::new(&state.call, state.call.model_stream(), replay));
                }
            },
        )
        .boxed()
    }
    /// Run the step checks in order; the first answer other than continue decides.
    async fn check_step(&self, steps: usize, usage: Option<Usage>) -> Result<StepDecision, Error> {
        if self.step_checks.is_empty() {
            return Ok(StepDecision::Continue);
        }
        let progress = RunProgress {
            steps,
            messages: self.context().messages.into(),
            usage,
        };
        for check in &self.step_checks {
            match check.run(progress.clone()).await? {
                StepDecision::Continue => {}
                StepDecision::Steer(message) if message.role != MessageRole::User => {
                    return Err(Error::Validation(format!(
                        "a step check steered with a {} message; a steering message must have the user role",
                        message.role
                    )));
                }
                decision => return Ok(decision),
            }
        }
        Ok(StepDecision::Continue)
    }
    /// Count a finished step and return the run's step count and usage.
    fn count_step(&self, usage: Option<&Usage>) -> (usize, Option<Usage>) {
        let mut tally = self.tally.lock().expect("agent tally mutex poisoned");
        tally.steps += 1;
        if let Some(usage) = usage {
            tally.usage = Some(match tally.usage.take() {
                Some(total) => total + usage,
                None => usage.clone(),
            });
        }
        (tally.steps, tally.usage.clone())
    }
    /// Change the loop's record of the step in progress.
    fn record(&self, change: impl FnOnce(&mut Option<LoopStep>)) {
        change(&mut self.tally.lock().expect("agent tally mutex poisoned").step);
    }
    fn set_outcome(&self, reason: StopReason) {
        self.tally
            .lock()
            .expect("agent tally mutex poisoned")
            .outcome = Some(reason);
    }
}
struct LoopState {
    call: RunContext,
    step: Option<Step>,
}
struct Step {
    source: EventStream,
    tools: ToolRunner,
    slots: Option<Arc<tokio::sync::Semaphore>>,
    resolver: ToolResolver,
    in_flight: usize,
    done: bool,
    replay: bool,
    message: Option<Message>,
}
impl Step {
    fn new(call: &RunContext, source: EventStream, replay: bool) -> Self {
        call.record(|step| *step = None);
        Self {
            source,
            tools: ToolRunner::new(),
            slots: call
                .tool_concurrency
                .map(|limit| Arc::new(tokio::sync::Semaphore::new(limit.get()))),
            resolver: ToolResolver::new(call.agent.tools.clone()),
            in_flight: 0,
            done: false,
            replay,
            message: None,
        }
    }
    fn usage(&self) -> Option<&Usage> {
        self.message
            .as_ref()
            .and_then(|message| message.usage.as_ref())
    }
    fn finish(self, call: &RunContext) -> Result<(), Error> {
        let tool_message = self.tools.get_tool_message();
        let ids: Vec<_> = self
            .message
            .as_ref()
            .into_iter()
            .flat_map(|m| m.tool_calls().into_iter().map(|t| t.tool_call_id.as_str()))
            .collect();
        if let Some(result) = tool_message
            .as_ref()
            .into_iter()
            .flat_map(|m| m.tool_results())
            .find(|r| !ids.contains(&r.tool_call_id.as_str()))
        {
            return Err(Error::Validation(format!(
                "orphaned-tool-result: {}",
                result.tool_call_id
            )));
        }
        let mut tally = call.tally.lock().expect("agent tally mutex poisoned");
        call.update(|context| {
            if let Some(message) = self.message {
                context.add(message)
            }
            if let Some(message) = tool_message {
                context.add(message)
            }
        });
        tally.step = None;
        Ok(())
    }
    async fn next_ready(
        &mut self,
    ) -> Either<Option<Result<ToolCallResult, Error>>, Option<Result<Event, Error>>> {
        if self.in_flight == 0 {
            if self.done {
                return Either::Right(None);
            }
            Either::Right(self.source.next().await)
        } else if self.done {
            Either::Left(self.tools.next().await)
        } else {
            match futures::future::select(self.tools.next(), self.source.next()).await {
                Either::Left((r, _)) => Either::Left(r),
                Either::Right((e, _)) => Either::Right(e),
            }
        }
    }
    fn accept_model_event(&mut self, call: &RunContext, event: Event) -> Result<Event, Error> {
        if let Some(meta) = event.model_event() {
            self.message = Some(meta.message.clone());
            let replayed = self.replay;
            // The interrupted history keeps the response only up to its last
            // received call, so the copy is taken at each ToolEnd.
            let tool_end = matches!(event, Event::ToolEnd(_));
            call.record(|step| match step {
                Some(step) if tool_end => step.message.clone_from(&meta.message),
                Some(_) => {}
                None => {
                    *step = Some(LoopStep {
                        message: meta.message.clone(),
                        replayed,
                        calls: Vec::new(),
                        results: Vec::new(),
                    })
                }
            });
        }
        if let Event::ToolEnd(value) = &event {
            call.record(|step| {
                if let Some(step) = step {
                    step.calls.push(value.tool_call_id.clone());
                }
            });
            let Some(tool) = self.resolver.get(&value.tool_call.tool_name) else {
                // The model sees the error and can correct the call.
                let part = &value.tool_call;
                let result = match &part.cached_result {
                    Some(result) => tool_result(result.clone()),
                    None => error_tool_result(
                        UnavailableToolError {
                            tool_name: part.tool_name.clone(),
                            available: self.resolver.names().map(str::to_owned).collect(),
                        },
                        &part.tool_call_id,
                        &part.tool_name,
                    ),
                };
                self.tools.add_result(result);
                self.in_flight += 1;
                return Ok(event);
            };
            let tc = BoundToolCall::new(value.tool_call.clone(), tool.clone());
            let context = ToolContext {
                runtime: call.runtime.clone(),
                hooks: call.hooks.clone(),
                tool_call_id: value.tool_call_id.clone(),
                tool_name: value.tool_call.tool_name.clone(),
                middleware: call.middleware.clone(),
                agent: Some(call.agent.clone()),
                bound_call: None,
                execution_slots: self.slots.clone(),
            };
            self.tools
                .schedule(async move { tc.execute(context).await });
            self.in_flight += 1;
        }

        Ok(event)
    }
    async fn next(&mut self, call: &RunContext) -> Result<Option<Event>, Error> {
        loop {
            if self.done && self.in_flight == 0 {
                return Ok(None);
            }
            let event = self.next_ready().await;
            match event {
                Either::Right(Some(event)) => {
                    let event = self.accept_model_event(call, event?)?;
                    return Ok(Some(event));
                }
                Either::Left(Some(result)) => {
                    let result = result?;
                    self.in_flight -= 1;
                    call.record(|step| {
                        if let Some(step) = step {
                            step.results.extend(result.results.iter().cloned());
                        }
                    });
                    return Ok(Some(Event::ToolCallResult(result)));
                }
                Either::Right(None) => {
                    self.done = true;
                    if self.message.is_none() {
                        return Err(Error::Validation("model stream produced no message".into()));
                    }
                }
                Either::Left(None) => {}
            }
            if self.done && self.in_flight == 0 {
                return Ok(None);
            }
        }
    }
}

fn populate_model_inputs(messages: &mut [Message], resolver: &ToolResolver) -> Result<(), Error> {
    for message in messages.iter_mut().filter(|m| m.role == MessageRole::Tool) {
        for part in &mut message.parts {
            if let Part::ToolResult(result) = part {
                if result.has_model_input() || result.is_error() || result.is_hook_deferred {
                    continue;
                }
                if let Some(tool) = resolver.get(&result.tool_name)
                    && let Some(value) = tool.model_input(&result.result)?
                {
                    result.set_model_input(value)
                }
            }
        }
    }
    Ok(())
}
/// Prepare interrupted history for replay, preserving completed sibling results.
pub fn process_interrupted_hooks(messages: &mut Vec<Message>) {
    let Some(last) = messages.last() else { return };
    if last.role == MessageRole::Assistant && !last.tool_calls().is_empty() {
        messages.last_mut().unwrap().replay = true;
        return;
    }
    if messages.len() < 2
        || last.role != MessageRole::Tool
        || !last.tool_results().iter().any(|r| r.is_hook_deferred)
    {
        return;
    }
    let previous = messages.len() - 2;
    if messages[previous].role != MessageRole::Assistant
        || messages[previous].tool_calls().is_empty()
    {
        return;
    }
    let completed: BTreeMap<String, ToolResultPart> = last
        .tool_results()
        .into_iter()
        .filter(|r| !r.is_hook_deferred)
        .map(|r| (r.tool_call_id.clone(), r.clone()))
        .collect();
    for part in &mut messages[previous].parts {
        if let Part::ToolCall(tc) = part
            && let Some(result) = completed.get(&tc.tool_call_id)
        {
            tc.cached_result = Some(result.clone())
        }
    }
    messages[previous].replay = true;
    messages.pop();
}

fn tracked(source: EventStream) -> EventStream {
    stream::try_unfold(
        (source, RunStateTracker::default(), VecDeque::new()),
        |(mut source, mut tracker, mut queued)| async move {
            loop {
                if let Some(event) = queued.pop_front() {
                    return Ok(Some((event, (source, tracker, queued))));
                }
                let Some(event) = source.next().await else {
                    return Ok(None);
                };
                let event = event?;
                let transition = tracker.feed(&event);
                if !event.replay() {
                    queued.push_back(event)
                }
                if let Some(blocked) = transition {
                    queued.push_back(Event::RunBlocked(blocked))
                }
            }
        },
    )
    .boxed()
}

/// The result text of a tool call that a stopped run did not finish.
const CANCELLED_TOOL_CALL: &str = "Cancelled: the run stopped before this tool call finished.";

/// Delivers events and exposes state consistent with those delivered events.
pub struct AgentStream {
    source: EventStream,
    /// Ends a buffered run's loop task; `None` for a lockstep run, whose loop
    /// runs inside `source`.
    buffer: Option<BufferStop>,
    context: Arc<Mutex<Context>>,
    hooks: HookRegistry,
    blocked: bool,
    deferred: BTreeMap<String, HookPart>,
    tally: Arc<Mutex<RunTally>>,
    usage: Option<Usage>,
    /// The model response of the last delivered model event, as delivered.
    delivered: Option<Message>,
    ended: bool,
    failed: bool,
    stopped: bool,
}

impl AgentStream {
    pub fn context(&self) -> Context {
        self.context
            .lock()
            .expect("agent context mutex poisoned")
            .clone()
    }
    pub fn messages(&self) -> Vec<Message> {
        self.context().messages
    }
    pub fn hook_registry(&self) -> &HookRegistry {
        &self.hooks
    }
    pub fn blocked(&self) -> bool {
        self.blocked
    }
    pub fn deferred_hooks(&self) -> Vec<HookPart> {
        self.deferred.values().cloned().collect()
    }
    /// The history plus the step in progress, for a next run to continue
    /// from after this one stops or fails. When the loop has received a tool
    /// call of that step, the step adds the response as the loop received it,
    /// up to its last such call, and a tool message with the result of each
    /// tool that finished and a cancelled error result for each one that did
    /// not, so the next run neither repeats a finished call nor runs an
    /// unfinished one. Otherwise it adds the response's text and reasoning as
    /// delivered. A buffered loop may run ahead of the consumer; the history
    /// holds every step it finished. Equals [`messages`](Self::messages) when
    /// no step is in progress, and after the stream ended without an error.
    /// Call [`stop`](Self::stop) first so that no tool finishes after the
    /// history was read.
    pub fn interrupted_messages(&self) -> Vec<Message> {
        let tally = self.tally.lock().expect("agent tally mutex poisoned");
        let mut messages = self.messages();
        if self.ended && !self.failed {
            return messages;
        }
        let Some(step) = tally.step.as_ref() else {
            return messages;
        };
        if !step.calls.is_empty() {
            step.add_to(&mut messages);
            return messages;
        }
        if let Some(delivered) = self
            .delivered
            .as_ref()
            .filter(|message| !step.replayed && message.id == step.message.id)
        {
            let mut message = delivered.clone();
            message.parts.retain(|part| match part {
                Part::Text(text) => !text.text.is_empty(),
                Part::Reasoning(_) => true,
                _ => false,
            });
            if !message.parts.is_empty() {
                messages.push(message);
            }
        }
        messages
    }
    /// Stop the run as dropping it does, and return once its work has ended:
    /// the model call and the running tools are dropped. The stream then
    /// yields no more events, and
    /// [`interrupted_messages`](Self::interrupted_messages) holds the result
    /// of every tool that finished before the stop.
    pub async fn stop(&mut self) {
        self.stopped = true;
        // Wait for a buffered loop task before the source goes: dropping the
        // source would only abort it. Dropping the source drops a lockstep
        // loop with its tools at once.
        if let Some(buffer) = self.buffer.take() {
            buffer.stop().await;
        }
        self.source = stream::empty().boxed();
    }
    /// The summed usage of the model responses delivered so far, or `None`
    /// when none of them reported usage.
    pub fn usage(&self) -> Option<Usage> {
        self.usage.clone()
    }
    /// Why the run stopped, once its stream has ended without an error.
    /// `None` while it runs, after an error, and for a custom loop that does
    /// not run the default loop.
    pub fn stop_reason(&self) -> Option<StopReason> {
        if !self.ended || self.failed {
            return None;
        }
        self.tally
            .lock()
            .expect("agent tally mutex poisoned")
            .outcome
            .clone()
    }
    pub fn text(&self) -> Result<String, Error> {
        self.messages()
            .last()
            .ok_or_else(|| Error::Validation("agent run has no messages".into()))?
            .get_output()
            .map_err(|e| Error::Validation(e.to_string()))
    }
    pub fn output<T: DeserializeOwned>(&self) -> Result<T, Error> {
        let context = self.context();
        let message = context
            .messages
            .last()
            .ok_or_else(|| Error::Validation("agent run has no messages".into()))?;
        message.deserialize_output().map_err(|error| match error {
            crate::types::messages::DeserializeOutputError::NotFinal(error) => {
                Error::Validation(error.to_string())
            }
            crate::types::messages::DeserializeOutputError::InvalidJson(error) => {
                Error::Json(error)
            }
        })
    }
}
impl Stream for AgentStream {
    type Item = Result<Event, Error>;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        if self.stopped {
            return Poll::Ready(None);
        }
        let event = match self.source.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(event))) => event,
            Poll::Ready(Some(Err(error))) => {
                self.failed = true;
                return Poll::Ready(Some(Err(error)));
            }
            Poll::Ready(None) => {
                self.ended = true;
                return Poll::Ready(None);
            }
            Poll::Pending => return Poll::Pending,
        };
        if let Some(meta) = event.model_event() {
            self.delivered = Some(meta.message.clone());
        }
        match &event {
            Event::StreamEnd(end) => {
                if let Some(usage) = &end.event.message.usage {
                    self.usage = Some(match self.usage.take() {
                        Some(total) => total + usage,
                        None => usage.clone(),
                    });
                }
            }
            Event::RunBlocked(_) => self.blocked = true,
            Event::Hook(value) => match value.hook.status {
                HookStatus::Pending => {
                    self.deferred
                        .insert(value.hook.hook_id.clone(), value.hook.clone());
                }
                _ => {
                    self.deferred.remove(&value.hook.hook_id);
                    self.blocked = false;
                }
            },
            _ => {}
        }
        Poll::Ready(Some(Ok(event)))
    }
}
