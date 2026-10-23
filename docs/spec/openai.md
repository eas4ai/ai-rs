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

[OAI-002] On the Responses API with OpenAI's rules, a model is a reasoning
model when its id is `o` and digits followed by `-` or the end, or `gpt-<n>`
with `n` at least 5, optionally followed by `.<m>` and `-<variant>`, unless
it has no `.<m>` and its variant starts with `chat`. When `store` is not
false, reasoning and assistant text with an OpenAI item id MUST each be sent
as one `item_reference` to that id, except text on a reasoning model when no
reasoning part with an OpenAI item id comes before it in the same assistant
turn; that text, and any text when `store` is false, MUST be sent as an
assistant message holding the text, without the item id. When `store` is
false, a call to a reasoning model MUST add `reasoning.encrypted_content` to
`include` once, and a reasoning part MUST be sent only when it has encrypted
content.
Falsifier: A recorded `gpt-5.4` body with `store` unset, for an assistant turn of text `T` with item id `msg_1` and no reasoning, holds an `item_reference` to `msg_1` or lacks `T`; the same turn with a reasoning part of item id `rs_1` before `T` sends other than one `item_reference` to `rs_1` and one to `msg_1`; a `gpt-4.1` turn of `T` alone is not one `item_reference` to `msg_1`; with `store` false, a `gpt-5.4` body lacks `reasoning.encrypted_content` in `include` or holds it twice, sends `T` with `"id": "msg_1"` or as a reference, or sends a reasoning part without encrypted content; or a `gpt-4.1` body with `store` false adds `reasoning.encrypted_content` to `include`.
Mechanism: openai
Rationale: `responses/convert-to-openai-responses-input.ts` sends stored text as a reference only after its reasoning (TypeScript change 216e2c9), since a reasoning model's stored message cannot be referenced without its reasoning item, and `responses/openai-responses-language-model.ts` asks for encrypted reasoning when `store` is false; OpenAI keeps stored items for a limited time, 30 days by default by community reports, so a chat that resumes later needs `store` false.
Status: Agreed 2026-10-23
