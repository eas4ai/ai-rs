Prefix: AGT

# Agent loop control

When an agent run stops and why, what it does with a tool call it cannot
run, how many tools run at once, and what usage it reports. A step is one
model response in the run together with the tool calls it asked for; a
response replayed from an earlier run is not a step. A run is done when the
model's last response is a final answer: an assistant message with no tool
calls. These rules apply to the default loop; a custom loop
(`Agent::with_loop`) that does not run the default loop decides for itself
when to stop, and its `stop_reason()` is `None`.

[AGT-001] After each step, including a step that ends with a final answer,
the default loop MUST run the checks in `RunOptions::step_checks` in order.
Each check is async, receives the run's progress (the step count, the
history and the total usage) and answers `Continue`, `Steer(message)` or
`Stop(reason)`; the first answer other than `Continue` decides, and later
checks do not run. On `Stop`, the run MUST stop with that step's tool
results in the history. On `Steer`, the loop MUST add the message, which
must have the user role, to the history and call the model again, even
after a final answer. When every check answers `Continue`, the run goes on
until it is done. `StepCheck::step_count(n)` and
`StepCheck::tool_called(names)` MUST be provided; they answer `Stop` with
the reasons `step count <n> reached` and `tool <name> was called`. The
default list MUST be empty.
Falsifier: With the default options, a run whose scripted model asks for a tool in each of 25 steps and then gives a final answer stops before that answer; a `step_count(2)` run makes a third model call; a `tool_called(["submit"])` run makes another model call after the step that called `submit`; a check that steers once after the final answer does not lead to one more model call whose request ends with the steering message; a check runs after an earlier one answered `Stop`; a check does not receive the current step count, history and total usage; or a steering message with another role does not end the run with `Error::Validation`.
Mechanism: core
Rationale: Python sets no limit, and the TypeScript tool loop agent stops at 20 steps (`packages/ai/src/agent/tool-loop-agent.ts:132`). Checks that can steer as well as stop let a progress monitor or a done gate (next-feature item `workflow-done-gate`) act between steps; a long job sets its budget with a check.
Status: Agreed 2026-10-22

[AGT-002] When the model calls a tool that has no executor, the run MUST add
an error tool result for that call with the text `Model tried to call
unavailable tool '<name>'. Available tools: <names>.`, where `<names>` lists
the executors' names, sorted, separated by `, `. It MUST deliver that result
as a `ToolCallResult` event and continue, so the next model request holds it.
Falsifier: A run whose model calls `missing`, with executors `beta` and `alpha`, ends with an error, or delivers no error `ToolCallResult` for that call, or sends a next model request without a tool result for that call whose text is `Model tried to call unavailable tool 'missing'. Available tools: alpha, beta.`
Mechanism: core
Rationale: Python and this port end the run with a configuration error (`src/agents/agent.rs:375`, `src/agents/tools.rs:364-372`); the TypeScript SDK returns the error to the model, which can then correct the call.
Status: Agreed 2026-10-22

[AGT-003] When `RunOptions::tool_concurrency` is set to N, at most N tool
executions of one step MUST run at once; the others MUST start in call order
as earlier ones finish, and every call still gets its result. When it is
unset, the default, every tool call of a step runs at once.
Falsifier: With N = 2 and a step of five tool calls whose executors record how many run at once, more than 2 run at once, fewer than five results reach the history, or the calls start out of call order; or with it unset, the five calls do not all run at once.
Mechanism: core
Status: Agreed 2026-10-22

[AGT-004] `AgentStream::usage()` MUST return the sum of the usage of every
model response the run has delivered, or `None` when none of them reported
usage.
Falsifier: After a two-step run whose responses report 10 input and 5 output tokens, then 20 input and 7 output tokens, `usage()` is not 30 input and 12 output tokens; or after a run whose responses report no usage, `usage()` is `Some`.
Mechanism: core
Rationale: The sum covers this agent's own model responses; a sub-agent run reports its own usage.
Status: Agreed 2026-10-22

[AGT-005] Once an agent run's stream has ended without an error,
`AgentStream::stop_reason()` MUST return `Done` when the run is done,
`Stopped` with the reason of the check that stopped it, and `Deferred` when
a hook, such as a tool approval, was deferred. Before the stream ends, and
after it ends with an error, it MUST return `None`.
Falsifier: A run that ends with a final answer, a run that `step_count(1)` stops while the model still asks for tools, and a run whose approval hook is deferred do not report `Done`, `Stopped` with the reason `step count 1 reached`, and `Deferred`; or `stop_reason()` is `Some` before the stream has ended or after it ended with an error.
Mechanism: core
Status: Agreed 2026-10-22
