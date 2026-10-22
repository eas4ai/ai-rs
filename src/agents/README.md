# Agents

`Agent` streams a model response, executes each function tool as its call finishes,
adds the assistant and tool messages to history, then repeats until the assistant
returns a final answer. Tools run concurrently. Provider tools stay declarations;
they have no local executor.

```rust
use suprnova_ai::{Agent, AgentTool, Model, user_message};
use futures::StreamExt;
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Deserialize, JsonSchema)]
struct Double { x: i32 }

# async fn example(model: Model) -> Result<(), suprnova_ai::Error> {
let double = AgentTool::new::<Double, i32, _, _>(
    "double", "Double a number", |input, _context| async move {
        Ok(input.x * 2)
    },
)?;
let mut run = Agent::new(vec![double]).run(model, vec![user_message("Double 7")]);
while let Some(event) = run.next().await {
    event?;
}
let answer = run.text()?;
println!("{answer}");
# Ok(())
# }
```

Inputs use `serde::Deserialize` and `schemars::JsonSchema`. The builder derives
the function declaration from that input type. Use serde defaults for optional
arguments. Async closures receive an explicit `ToolContext` with the runtime,
hook registry, middleware stack, tool name and call ID. During execution it also
contains the original bound call and an optional owning agent. Standalone calls
have no owner. `new_with_hooks` also
accepts hook errors. `new_special` preserves rich content and message results.
`with_model_input` converts a rich result to the value sent to the model.

For streaming tools, use `AgentTool::stream` and `Aggregate::last()` for status,
`Aggregate::concat("")` for text, or `Aggregate::messages()` for sub-agent events.
The consumer receives each yield as `PartialToolCallResult` with its aggregator
factory. The tool result stores the final snapshot and its model input separately.
An explicit model-input converter and a streaming aggregator are mutually exclusive.
Factories create a fresh reducer for each execution. Custom reducers use the core
`AggregatorFactory` and `ValueAggregator` contracts.

Set `require_approval(true)` on a tool to suspend on `approve_<tool_call_id>`.
Resolve through `run.hook_registry().resolve_hook(label, json_value)`, cancel
through its async `cancel_hook`, or defer with `defer_hook`. Invalid arguments
return an error result before approval. A deferred run finishes with an error
placeholder marked `is_hook_deferred`; pass its history into the next run and
pre-register the resolution on a registry in `RunOptions`. Completed sibling
results are cached and skipped on replay. Replay events stay out of consumer
output. `blocked()` and `deferred_hooks()` reflect delivered events.

`RunOptions` carries model parameters, structured output schema, hooks,
middleware and the loop buffer. The first middleware is outermost. The stack
wraps agent runs, model streams, generation, tool execution and hooks. Context
containers own their values. A wrapper can replace messages or arguments without
mutating the caller's input. Changing a tool call's correlation ID causes a history
integrity error.

`Agent::with_loop` accepts a custom loop returning an `EventStream`. Its
`RunContext` exposes `model_stream`, `generate`, `default_loop`, history updates,
and nested run options. Use `ToolContext::nested_options` or
`RunContext::nested_options` when starting sub-agents so they share hooks and
middleware. Explicit contexts replace Python context variables and subclass
configuration. Class defaults become tools passed to the builder; custom loops
become closures.

The default loop buffer lets the loop run ahead without a size limit, as Python
does. `Some(0)` runs in lockstep; positive capacities bound consumer buffering.
The runtime side queue is unbounded so consumer hook cancellation cannot deadlock
behind queued events. Applications with large streams should use bounded loop
buffering and avoid queuing unbounded side work. Dropping a buffered run requests
cancellation; cleanup finishes when its task is polled. Dropping a lockstep run
drops its source and pending tools immediately. Tool errors become error results with retained original
error handles; provider and middleware errors end the run. Provider errors do not
trigger automatic retries. As in Python, `Retry` events supplied by an execution
wrapper reset aggregation and run tracking.

MCP connections and telemetry span recording are outside this module.
