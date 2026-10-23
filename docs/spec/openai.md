Prefix: OAI

# OpenAI

OpenAI's own APIs, as the AI SDK for TypeScript 7.0.148 package
`@ai-sdk/openai` sends and reads them. `OpenAICompatibleProvider::new` and
models.dev metadata that names `@ai-sdk/openai` speak the Responses API with
OpenAI's rules (`OpenAIResponsesProtocol`). The history a call sends there
holds items: messages, reasoning, function calls and their outputs, and
`item_reference` items that name an item OpenAI stored from an earlier
response by its item id. Paths in rationale lines are under
`packages/openai/src/`.

[OAI-001] On the Responses API with OpenAI's rules, each tool call of an
assistant turn in the history, which the application runs, MUST be sent as a
`function_call` item with the call's `call_id`, `name` and `arguments`, and
with the `namespace` its OpenAI metadata records when it records one. It MUST
NOT be sent as an `item_reference` or with the OpenAI item id, whether `store`
is true, false or unset. Its result MUST follow as a `function_call_output`
with the same `call_id`.
Falsifier: A recorded body for a history of an assistant tool call `call_1` named `weather` with arguments `{"city":"SF"}`, whose OpenAI metadata holds the item id `fc_1` and its raw `function_call` item, followed by its result, holds an `item_reference` to `fc_1` or a `function_call` item with `"id": "fc_1"` under `store` true, false or unset, or lacks `{"type": "function_call", "call_id": "call_1", "name": "weather", "arguments": "{\"city\":\"SF\"}"}` followed by a `function_call_output` with `call_id` `call_1`; or the same history whose raw item holds `"namespace": "tools"` is sent without that `namespace`.
Mechanism: openai
Rationale: `responses/convert-to-openai-responses-input.ts` sends these calls in full (TypeScript change 61bcdb5) because OpenAI pairs an output with its call only by `call_id`, and referenced calls failed with "No tool call found for function call output", most often after parallel calls across steps; a live two-step `gpt-5-nano` call with one referenced call succeeded on 2026-10-23.
Status: Agreed 2026-10-23
