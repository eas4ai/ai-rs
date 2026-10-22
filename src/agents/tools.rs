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

//! Typed host executors paired with the core's model-facing tool declarations.

use std::{
    collections::{BTreeMap, VecDeque},
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use futures::{
    Stream, StreamExt,
    future::{AbortHandle, Abortable, BoxFuture},
};
use schemars::JsonSchema;
use serde::{Serialize, de::DeserializeOwned};
use serde_json::{Value, json};

use crate::{
    Error,
    types::{
        Message, MessageRole, Tool, ToolApproval, ToolDefinition,
        events::{Event, PartialToolCallResult, RuntimeError, ToolCallResult},
        messages::{HookPart, Metadata, ToolCallPart, ToolResultPart, ToolResultValue},
    },
};

use super::{
    aggregators::{Aggregate, ModelInputConverter},
    hooks::{HookError, HookRegistry},
    middleware::{self, Middleware, ToolNext},
    runtime::Runtime,
};

/// Errors from tools that can suspend on a hook as well as fail normally.
#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    #[error(transparent)]
    Execution(#[from] Error),
    #[error(transparent)]
    Hook(#[from] HookError),
    #[error(transparent)]
    Runtime(#[from] RuntimeError),
}

/// Explicit run-scoped context passed to every executor instead of task globals.
#[derive(Clone)]
pub struct ToolContext {
    pub runtime: Runtime,
    pub hooks: HookRegistry,
    /// Owning agent when invoked by an agent run; absent for standalone calls.
    pub agent: Option<Arc<super::agent::Agent>>,
    /// Original call and executor, before argument overrides or middleware edits.
    pub bound_call: Option<BoundToolCall>,
    pub tool_call_id: String,
    pub tool_name: String,
    pub middleware: Vec<Arc<dyn Middleware>>,
}

impl ToolContext {
    pub fn new(
        runtime: Runtime,
        hooks: HookRegistry,
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
    ) -> Self {
        Self {
            runtime,
            hooks,
            agent: None,
            bound_call: None,
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            middleware: Vec::new(),
        }
    }

    pub async fn hook<T: DeserializeOwned + JsonSchema>(
        &self,
        label: impl Into<String>,
        metadata: Metadata,
    ) -> Result<T, HookError> {
        self.hooks
            .hook(
                label,
                metadata,
                Some(self.tool_call_id.clone()),
                &self.runtime,
            )
            .await
    }

    /// Nested runs inherit the explicit context that Python stores in contextvars.
    pub fn nested_options(&self) -> super::agent::RunOptions {
        super::agent::RunOptions {
            hook_registry: Some(self.hooks.clone()),
            middleware: self.middleware.clone(),
            ..Default::default()
        }
    }
}

struct ToolOutput {
    value: ToolResultValue,
    model_input: Option<ToolResultValue>,
}
type Executor = Arc<
    dyn Fn(Value, ToolContext) -> BoxFuture<'static, Result<ToolOutput, ToolError>> + Send + Sync,
>;
type Validator = Arc<dyn Fn(&Value) -> Result<(), Error> + Send + Sync>;

/// Cloneable executable tool. Clones share the executor but not reducer instances.
#[derive(Clone)]
pub struct AgentTool {
    declaration: Tool,
    definition: ToolDefinition,
    executor: Executor,
    validate: Validator,
    aggregator: Option<Aggregate>,
    to_model_input: Option<ModelInputConverter>,
}

impl AgentTool {
    /// Build an asynchronous JSON-result tool from its typed input and output.
    pub fn new<Input, Output, F, Fut>(
        name: impl Into<String>,
        description: impl Into<String>,
        execute: F,
    ) -> Result<Self, Error>
    where
        Input: DeserializeOwned + JsonSchema + Send + 'static,
        Output: Serialize + JsonSchema + Send + 'static,
        F: Fn(Input, ToolContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Output, Error>> + Send + 'static,
    {
        Self::new_with_hooks(name, description, move |input, context| {
            let future = execute(input, context);
            async move { future.await.map_err(ToolError::from) }
        })
    }

    /// Build a tool whose body can propagate hook deferral with `?`.
    pub fn new_with_hooks<Input, Output, F, Fut>(
        name: impl Into<String>,
        description: impl Into<String>,
        execute: F,
    ) -> Result<Self, Error>
    where
        Input: DeserializeOwned + JsonSchema + Send + 'static,
        Output: Serialize + JsonSchema + Send + 'static,
        F: Fn(Input, ToolContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Output, ToolError>> + Send + 'static,
    {
        let definition = ToolDefinition::from_types::<Input, Output>(name, Some(description))
            .map_err(|error| Error::Configuration(error.to_string()))?;
        let execute = Arc::new(execute);
        let executor: Executor = Arc::new(move |arguments, context| {
            let execute = execute.clone();
            Box::pin(async move {
                let input = serde_json::from_value(arguments).map_err(Error::from)?;
                let output = execute(input, context).await?;
                Ok(ToolOutput {
                    value: serde_json::to_value(output).map_err(Error::from)?.into(),
                    model_input: None,
                })
            })
        });
        Ok(Self::from_executor::<Input>(definition, executor, None))
    }

    /// Preserve typed special results such as content output or message bundles.
    pub fn new_special<Input, Output, F, Fut>(
        name: impl Into<String>,
        description: impl Into<String>,
        execute: F,
    ) -> Result<Self, Error>
    where
        Input: DeserializeOwned + JsonSchema + Send + 'static,
        Output: Into<ToolResultValue> + Send + 'static,
        F: Fn(Input, ToolContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Output, Error>> + Send + 'static,
    {
        let definition = ToolDefinition::from_input::<Input>(name, Some(description))
            .map_err(|error| Error::Configuration(error.to_string()))?;
        let execute = Arc::new(execute);
        let executor: Executor = Arc::new(move |arguments, context| {
            let execute = execute.clone();
            Box::pin(async move {
                let input = serde_json::from_value(arguments).map_err(Error::from)?;
                let output = execute(input, context).await?;
                Ok(ToolOutput {
                    value: output.into(),
                    model_input: None,
                })
            })
        });
        Ok(Self::from_executor::<Input>(definition, executor, None))
    }

    /// Drain a typed stream, forwarding every partial before producing the result.
    pub fn stream<Input, Item, F, S>(
        name: impl Into<String>,
        description: impl Into<String>,
        execute: F,
        aggregate: impl Into<Aggregate>,
    ) -> Result<Self, Error>
    where
        Input: DeserializeOwned + JsonSchema + Send + 'static,
        Item: Serialize + Send + 'static,
        F: Fn(Input, ToolContext) -> S + Send + Sync + 'static,
        S: Stream<Item = Result<Item, Error>> + Send + 'static,
    {
        Self::stream_with_hooks(
            name,
            description,
            move |input, context| execute(input, context).map(|item| item.map_err(ToolError::from)),
            aggregate,
        )
    }

    /// Streaming counterpart of `new_with_hooks`, retaining hook suspensions.
    pub fn stream_with_hooks<Input, Item, F, S>(
        name: impl Into<String>,
        description: impl Into<String>,
        execute: F,
        aggregate: impl Into<Aggregate>,
    ) -> Result<Self, Error>
    where
        Input: DeserializeOwned + JsonSchema + Send + 'static,
        Item: Serialize + Send + 'static,
        F: Fn(Input, ToolContext) -> S + Send + Sync + 'static,
        S: Stream<Item = Result<Item, ToolError>> + Send + 'static,
    {
        let definition = ToolDefinition::from_input::<Input>(name, Some(description))
            .map_err(|error| Error::Configuration(error.to_string()))?;
        let aggregate = aggregate.into();
        let executor_aggregate = aggregate.clone();
        let execute = Arc::new(execute);
        let executor: Executor = Arc::new(move |arguments, context| {
            let execute = execute.clone();
            let aggregate = executor_aggregate.clone();
            Box::pin(async move {
                let input = serde_json::from_value(arguments).map_err(Error::from)?;
                let source = execute(input, context.clone());
                futures::pin_mut!(source);
                aggregate_from(
                    &context.runtime,
                    source,
                    aggregate,
                    Some(context.tool_call_id),
                    Some(context.tool_name),
                    Value::Null,
                )
                .await
            })
        });
        Ok(Self::from_executor::<Input>(
            definition,
            executor,
            Some(aggregate),
        ))
    }

    fn from_executor<Input: DeserializeOwned>(
        definition: ToolDefinition,
        executor: Executor,
        aggregator: Option<Aggregate>,
    ) -> Self {
        Self {
            declaration: (&definition).into(),
            definition,
            executor,
            validate: Arc::new(|value| {
                serde_json::from_value::<Input>(value.clone())?;
                Ok(())
            }),
            aggregator,
            to_model_input: None,
        }
    }

    pub fn name(&self) -> &str {
        &self.declaration.name
    }
    pub fn declaration(&self) -> &Tool {
        &self.declaration
    }
    pub fn definition(&self) -> &ToolDefinition {
        &self.definition
    }
    pub fn requires_approval(&self) -> bool {
        self.declaration.require_approval
    }
    pub fn require_approval(mut self, required: bool) -> Self {
        self.declaration.require_approval = required;
        self
    }
    pub fn aggregator(&self) -> Option<&Aggregate> {
        self.aggregator.as_ref()
    }

    /// Normal tools may transform rich results; streams use their reducer instead.
    pub fn with_model_input(
        mut self,
        convert: impl Fn(&ToolResultValue) -> Result<ToolResultValue, Error> + Send + Sync + 'static,
    ) -> Result<Self, Error> {
        if self.aggregator.is_some() {
            return Err(Error::Configuration(
                "to_model_input cannot be combined with an aggregator".into(),
            ));
        }
        self.to_model_input = Some(Arc::new(convert));
        Ok(self)
    }

    pub fn validate_arguments(&self, arguments: &Value) -> Result<(), Error> {
        self.definition
            .validate_arguments(arguments)
            .map_err(|error| Error::Validation(error.to_string()))?;
        (self.validate)(arguments)
    }

    /// Compute a missing model input on a result reconstructed from the wire.
    pub fn model_input(&self, value: &ToolResultValue) -> Result<Option<ToolResultValue>, Error> {
        let convert = self.to_model_input.as_ref().or_else(|| {
            self.aggregator
                .as_ref()
                .and_then(|aggregate| aggregate.model_input.as_ref())
        });
        convert.map(|convert| convert(value)).transpose()
    }
}

/// Names resolve to the last registered executor, matching Python dictionary rules.
#[derive(Clone, Default)]
pub struct ToolResolver {
    tools: BTreeMap<String, AgentTool>,
}
impl ToolResolver {
    pub fn new(tools: Vec<AgentTool>) -> Self {
        Self {
            tools: tools
                .into_iter()
                .map(|tool| (tool.name().to_owned(), tool))
                .collect(),
        }
    }
    pub fn get(&self, name: &str) -> Option<&AgentTool> {
        self.tools.get(name)
    }
    pub fn resolve(&self, part: ToolCallPart) -> Result<BoundToolCall, Error> {
        let tool = self.get(&part.tool_name).ok_or_else(|| {
            Error::Configuration(format!(
                "No agent executor registered for tool {:?}",
                part.tool_name
            ))
        })?;
        Ok(BoundToolCall::new(part, tool.clone()))
    }
}

/// One model call bound to a host executor, including any cached replay result.
#[derive(Clone)]
pub struct BoundToolCall {
    pub part: ToolCallPart,
    pub tool: AgentTool,
}
impl BoundToolCall {
    pub fn new(part: ToolCallPart, tool: AgentTool) -> Self {
        Self { part, tool }
    }
    #[expect(
        clippy::misnamed_getters,
        reason = "A bound call identifies the tool invocation, not the internal message part"
    )]
    pub fn id(&self) -> &str {
        &self.part.tool_call_id
    }
    pub fn name(&self) -> &str {
        &self.part.tool_name
    }
    pub fn arguments(&self) -> Result<Value, Error> {
        let value = if self.part.tool_args.is_empty() {
            json!({})
        } else {
            serde_json::from_str(&self.part.tool_args)?
        };
        self.tool.validate_arguments(&value)?;
        Ok(value)
    }

    pub async fn execute(&self, context: ToolContext) -> Result<ToolCallResult, Error> {
        if let Some(result) = &self.part.cached_result {
            return Ok(tool_result(result.clone()));
        }
        let arguments = match self.arguments() {
            Ok(arguments) => arguments,
            Err(error) if self.tool.requires_approval() || context.middleware.is_empty() => {
                return Ok(error_tool_result(error, self.id(), self.name()));
            }
            Err(_) => json!({}),
        };
        self.execute_with_arguments(context, arguments).await
    }

    /// Host overrides merge into the best available arguments and validate eagerly.
    pub async fn execute_with_overrides(
        &self,
        context: ToolContext,
        overrides: Value,
    ) -> Result<ToolCallResult, Error> {
        if let Some(result) = &self.part.cached_result {
            return Ok(tool_result(result.clone()));
        }
        let mut arguments = self.arguments().unwrap_or_else(|_| json!({}));
        let object = arguments
            .as_object_mut()
            .ok_or_else(|| Error::Validation("tool arguments must be an object".into()))?;
        let overrides = overrides
            .as_object()
            .ok_or_else(|| Error::Validation("tool overrides must be an object".into()))?;
        object.extend(overrides.clone());
        self.tool.validate_arguments(&arguments)?;
        self.execute_with_arguments(context, arguments).await
    }

    /// Execute middleware-supplied replacement arguments. Model errors become results.
    pub async fn execute_with_arguments(
        &self,
        mut context: ToolContext,
        arguments: Value,
    ) -> Result<ToolCallResult, Error> {
        if let Some(result) = &self.part.cached_result {
            return Ok(tool_result(result.clone()));
        }
        context.tool_call_id = self.id().to_owned();
        context.tool_name = self.name().to_owned();
        context.bound_call = Some(self.clone());
        if self.tool.requires_approval() {
            if let Err(error) = self.tool.validate_arguments(&arguments) {
                return Ok(error_tool_result(error, self.id(), self.name()));
            }
            let metadata = BTreeMap::from([
                ("tool".into(), Value::String(self.name().to_owned())),
                ("kwargs".into(), arguments.clone()),
            ]);
            match context
                .hook::<ToolApproval>(format!("approve_{}", self.id()), metadata)
                .await
            {
                Ok(approval) if approval.granted => {}
                Ok(approval) => {
                    return Ok(tool_result(ToolResultPart::error(
                        self.id(),
                        self.name(),
                        json!(format!(
                            "Rejected: {}",
                            approval.reason.as_deref().unwrap_or("None")
                        )),
                    )));
                }
                Err(HookError::Deferred(hook)) => {
                    return Ok(deferred_tool_result(&hook, self.id(), self.name()));
                }
                Err(error) => return Ok(error_tool_result(error, self.id(), self.name())),
            }
        }
        let tool = self.tool.clone();
        let stack = context.middleware.clone();
        let real: ToolNext = Arc::new(move |call| {
            let tool = tool.clone();
            let mut context = context.clone();
            Box::pin(async move {
                context.tool_call_id = call.tool_call_id.clone();
                context.tool_name = call.tool_name.clone();
                let result = async {
                    tool.validate_arguments(&call.kwargs)?;
                    let mut output = (tool.executor)(call.kwargs, context).await?;
                    if let Some(convert) = &tool.to_model_input {
                        output.model_input = Some(convert(&output.value)?);
                    }
                    let mut part =
                        ToolResultPart::new(&call.tool_call_id, &call.tool_name, output.value);
                    if let Some(value) = output.model_input {
                        part.set_model_input(value);
                    }
                    Ok::<_, ToolError>(tool_result(part))
                }
                .await;
                Ok(match result {
                    Ok(result) => result,
                    Err(ToolError::Hook(HookError::Deferred(hook))) => {
                        deferred_tool_result(&hook, &call.tool_call_id, &call.tool_name)
                    }
                    Err(ToolError::Execution(error)) => {
                        error_tool_result(error, &call.tool_call_id, &call.tool_name)
                    }
                    Err(ToolError::Hook(error)) => {
                        error_tool_result(error, &call.tool_call_id, &call.tool_name)
                    }
                    Err(ToolError::Runtime(error)) => {
                        retained_error_tool_result(error, &call.tool_call_id, &call.tool_name)
                    }
                })
            })
        });
        middleware::tool_chain(&stack, real)(middleware::ToolContext {
            tool_call_id: self.id().to_owned(),
            tool_name: self.name().to_owned(),
            kwargs: arguments,
        })
        .await
    }
}

/// Wrap one core result part in the corresponding tool message and runtime event.
pub fn tool_result(part: ToolResultPart) -> ToolCallResult {
    ToolCallResult {
        replay: false,
        message: Message::new(MessageRole::Tool, vec![part.clone().into()]),
        results: vec![part],
        exception: None,
    }
}

/// Retain the original error while exposing its message as an error tool result.
pub fn error_tool_result(
    error: impl std::error::Error + Send + Sync + 'static,
    tool_call_id: &str,
    tool_name: &str,
) -> ToolCallResult {
    retained_error_tool_result(RuntimeError::new(error), tool_call_id, tool_name)
}

fn retained_error_tool_result(
    error: RuntimeError,
    tool_call_id: &str,
    tool_name: &str,
) -> ToolCallResult {
    let mut result = tool_result(ToolResultPart::error(
        tool_call_id,
        tool_name,
        json!(error.to_string()),
    ));
    result.exception = Some(error);
    result
}

/// A paired placeholder allows a deferred call to be replayed on a later run.
pub fn deferred_tool_result(
    hook: &HookPart,
    tool_call_id: &str,
    tool_name: &str,
) -> ToolCallResult {
    let mut part = ToolResultPart::error(
        tool_call_id,
        tool_name,
        json!(format!("Deferred on hook {:?}", hook.hook_id)),
    );
    part.is_hook_deferred = true;
    tool_result(part)
}

/// Forward a labeled stream through the runtime and return its model-facing value.
pub async fn yield_from<Item, S>(
    runtime: &Runtime,
    source: S,
    aggregate: impl Into<Aggregate>,
    tool_call_id: Option<String>,
    tool_name: Option<String>,
    label: Value,
) -> Result<ToolResultValue, Error>
where
    Item: Serialize,
    S: Stream<Item = Result<Item, Error>>,
{
    let output = aggregate_from(
        runtime,
        source,
        aggregate.into(),
        tool_call_id,
        tool_name,
        label,
    )
    .await?;
    Ok(output.model_input.unwrap_or(output.value))
}

async fn aggregate_from<Item, S, E>(
    runtime: &Runtime,
    source: S,
    aggregate: Aggregate,
    tool_call_id: Option<String>,
    tool_name: Option<String>,
    label: Value,
) -> Result<ToolOutput, E>
where
    Item: Serialize,
    S: Stream<Item = Result<Item, E>>,
    E: From<Error>,
{
    futures::pin_mut!(source);
    let mut reducer = aggregate.factory.create();
    while let Some(item) = source.next().await {
        let value = serde_json::to_value(item?).map_err(Error::from)?;
        aggregate.validate(&value)?;
        reducer.feed(value.clone());
        runtime
            .put_event(Event::PartialToolCallResult(PartialToolCallResult {
                replay: false,
                tool_call_id: tool_call_id.clone(),
                tool_name: tool_name.clone(),
                label: label.clone(),
                value,
                aggregator_factory: Some(aggregate.factory.clone()),
            }))
            .await?;
    }
    Ok(ToolOutput {
        value: aggregate.result(reducer.snapshot())?,
        model_input: Some(reducer.get_model_input().into()),
    })
}

/// Cancellable handle for one runner-owned future.
#[derive(Clone, Debug)]
pub struct ToolTask {
    id: Arc<()>,
    abort: AbortHandle,
}
impl ToolTask {
    /// Request cancellation. The runner omits cancelled futures on its next poll.
    pub fn cancel(&self) {
        self.abort.abort();
    }
}
struct RunningTool {
    id: Arc<()>,
    abort: AbortHandle,
    future: Abortable<BoxFuture<'static, Result<ToolCallResult, Error>>>,
}

/// Own concurrent tool futures and the completed results for their tool message.
/// Futures advance while the runner's event stream is polled. Dropping the runner
/// cancels unfinished work in insertion order and never detaches background tasks.
#[derive(Default)]
pub struct ToolRunner {
    tasks: Vec<RunningTool>,
    ready: VecDeque<(Arc<()>, AbortHandle, Result<ToolCallResult, Error>)>,
    results: Vec<ToolCallResult>,
}

impl ToolRunner {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn schedule(
        &mut self,
        future: impl Future<Output = Result<ToolCallResult, Error>> + Send + 'static,
    ) -> ToolTask {
        let (abort, registration) = AbortHandle::new_pair();
        let id = Arc::new(());
        self.tasks.push(RunningTool {
            id: id.clone(),
            abort: abort.clone(),
            future: Abortable::new(Box::pin(future), registration),
        });
        ToolTask { id, abort }
    }
    /// Cancel and drop a speculative call immediately without producing a result.
    pub fn discard(&mut self, task: &ToolTask) {
        task.cancel();
        if let Some(index) = self
            .tasks
            .iter()
            .position(|entry| Arc::ptr_eq(&entry.id, &task.id))
        {
            self.tasks.remove(index);
        }
        self.ready.retain(|(id, _, _)| !Arc::ptr_eq(id, &task.id));
    }
    pub fn add_result(&mut self, result: ToolCallResult) {
        self.schedule(async move { Ok(result) });
    }
    /// A fresh borrow resumes iteration after an earlier borrow was dropped.
    pub fn events(&mut self) -> &mut Self {
        self
    }
    pub fn get_tool_message(&self) -> Option<Message> {
        if self.results.is_empty() {
            return None;
        }
        Some(Message::new(
            MessageRole::Tool,
            self.results
                .iter()
                .flat_map(|result| result.message.parts.iter().cloned())
                .collect(),
        ))
    }
}

impl Stream for ToolRunner {
    type Item = Result<ToolCallResult, Error>;
    fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // Poll every task, including when an earlier task is already complete,
        // so each future receives the wake registration needed for concurrency.
        let mut index = 0;
        while index < self.tasks.len() {
            match Pin::new(&mut self.tasks[index].future).poll(context) {
                Poll::Pending => index += 1,
                Poll::Ready(outcome) => {
                    let task = self.tasks.remove(index);
                    if let Ok(result) = outcome {
                        self.ready.push_back((task.id, task.abort, result));
                    }
                }
            }
        }
        while let Some((_, abort, result)) = self.ready.pop_front() {
            if abort.is_aborted() {
                continue;
            }
            if let Ok(result) = &result {
                self.results.push(result.clone());
            }
            return Poll::Ready(Some(result));
        }
        if self.tasks.is_empty() {
            Poll::Ready(None)
        } else {
            Poll::Pending
        }
    }
}

impl Drop for ToolRunner {
    fn drop(&mut self) {
        for task in self.tasks.drain(..) {
            drop(task);
        }
    }
}
