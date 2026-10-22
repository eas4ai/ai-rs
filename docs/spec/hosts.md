Prefix: HST

# Hosted chat providers

Vendor APIs that speak a protocol the crate already implements: the OpenAI
Chat Completions protocol (Cerebras, DeepInfra, Fireworks, Together AI and
Z.ai) and the Anthropic Messages protocol (MiniMax). Each one is the
existing adapter with a host profile: its endpoint, its key, and the request
and response differences the AI SDK for TypeScript 7.0.148 records for it.

| Host | Name and handles | Default base URL | Key variable |
|---|---|---|---|
| Cerebras | `cerebras`, `@ai-sdk/cerebras` | `https://api.cerebras.ai/v1` | `CEREBRAS_API_KEY` |
| DeepInfra | `deepinfra`, `@ai-sdk/deepinfra` | `https://api.deepinfra.com/v1/openai` | `DEEPINFRA_API_KEY` |
| Fireworks | `fireworks`, `@ai-sdk/fireworks` | `https://api.fireworks.ai/inference/v1` | `FIREWORKS_API_KEY` |
| Together AI | `togetherai`, `@ai-sdk/togetherai` | `https://api.together.xyz/v1` | `TOGETHER_API_KEY` |
| Z.ai | `zai`, `@ai-sdk/zai` | `https://api.z.ai/api/paas/v4` | `ZAI_API_KEY` |
| MiniMax | `minimax`, `@ai-sdk/minimax` | `https://api.minimax.io/anthropic` | `MINIMAX_API_KEY` |

[HST-001] For each host in the table, `providers::hosts::<name>::register`
MUST register, under the host's name and handles, a provider named after the
host. Its model calls MUST stream the host's protocol from the configured
base URL, or the default base URL when none is configured: Chat Completions
at `<base>/chat/completions` with `authorization: Bearer <key>`, and for
MiniMax, Messages at `<base>/v1/messages` with `x-api-key: <key>` and
`anthropic-version: 2023-06-01`. The key is the configuration's API key
value, or the `env` value named by the host's key variable. Registering
every host, OpenAI and Anthropic into one registry MUST succeed.
Falsifier: For any host, a recorded request goes to another URL, uses another method or credential header, or omits the key; the provider is missing under one of its handles or has another name; or registering all six hosts with `openai::register`, `openai::register_factories` and `anthropic::register` into one registry fails.
Mechanism: hosts
Status: Agreed 2026-10-22

[HST-002] A Chat Completions request MUST send `stream_options:
{"include_usage": true}` for OpenAI, for an OpenAI-compatible provider
without a host profile, for Fireworks and for Together AI, and MUST NOT send
`stream_options` for Cerebras, DeepInfra or Z.ai.
Falsifier: The recorded body for Cerebras, DeepInfra or Z.ai has a `stream_options` field; or the body for OpenAI's Chat Completions protocol, for a provider built with `OpenAICompatibleProvider::from_modelsdev_provider` for an id without a host profile, for Fireworks or for Together AI lacks `stream_options.include_usage` set to true.
Mechanism: hosts
Rationale: The TypeScript SDK sends it only where a host opts in (`openai-compatible-chat-language-model.ts` `includeUsage`), because some servers reject the field; custom OpenAI-compatible servers keep it, as the crate does today, since many report streamed usage only when asked and billing reads that usage.
Status: Agreed 2026-10-22

[HST-003] Chat Completions requests MUST follow each host's field rules:
Cerebras sends the output token limit as `max_completion_tokens` and an
assistant turn's reasoning as `reasoning`; DeepInfra, Fireworks, Together AI
and Z.ai send `max_tokens` and `reasoning_content`; Fireworks sends the
reasoning efforts `minimal` as `low` and `xhigh` and `max` as `high`. An
output schema MUST be sent as `response_format` of type `json_schema` for
Cerebras, DeepInfra and Fireworks, and for Together AI only for the model
`deepseek-ai/DeepSeek-V4-Flash-0731`; for other Together AI models and for
Z.ai it MUST be sent as `{"type": "json_object"}`.
Falsifier: A recorded body for a host with an output token limit of 100, an assistant turn with reasoning, a reasoning effort and an output schema has the limit under another field, the reasoning under another field, a Fireworks effort of `minimal`, `xhigh` or `max`, or a `response_format` other than the one stated for that host and model.
Mechanism: hosts
Status: Agreed 2026-10-22

[HST-004] For Z.ai, a call with tool choice `none` MUST send neither `tools`
nor `tool_choice`, and a call with tool choice `required`, a named tool or a
tool selection, or with a frequency penalty, presence penalty or seed, MUST
fail with `Error::Validation` before any request is sent. A Z.ai finish
reason MUST map `sensitive` to `content_filter`,
`model_context_window_exceeded` to `length` and `network_error` to `error`.
Falsifier: A Z.ai call with tool choice `none` sends `tools` or `tool_choice`; a Z.ai call with tool choice `required`, a named tool or a selection, or with a frequency penalty, presence penalty or seed, sends a request or fails with another error kind; or a Z.ai stream ending with finish reason `sensitive`, `model_context_window_exceeded` or `network_error` reports a finish reason other than `content_filter`, `length` or `error`.
Mechanism: hosts
Rationale: The TypeScript SDK drops these with a warning; this crate reports unsupported parameters as validation errors, as its OpenAI and Anthropic adapters do.
Status: Agreed 2026-10-22

[HST-005] A Chat Completions error response MUST give `ProviderError` the
provider's message whether the body is `{"error": {"message": ...}}`,
`{"error": "<text>"}` or `{"message": ...}`, with the error kind from the
HTTP status.
Falsifier: For an HTTP 400 response with any of the three bodies, the error message is not the body's message text, or the kind is not `BadRequest`.
Mechanism: hosts
Rationale: OpenAI and DeepInfra use the first shape, Fireworks the first two, and Cerebras and Z.ai the third (`cerebras-provider.ts`, `fireworks-provider.ts`, `zai-error.ts`).
Status: Agreed 2026-10-22

[HST-006] When a DeepInfra response reports more reasoning tokens than
completion tokens, the usage MUST report the sum as output tokens. When a
Cerebras call has an output schema and its response streams text and then
tool calls, the call MUST drop the tool calls that follow the text and, when
the response ends with finish reason `tool_calls`, report finish reason
`stop`.
Falsifier: A DeepInfra stream whose usage has `completion_tokens` 10 and `completion_tokens_details.reasoning_tokens` 30 reports output tokens other than 40; or a Cerebras stream under an output schema that streams text and a tool call and finishes with `tool_calls` delivers a tool call event or a finish reason other than `stop`.
Mechanism: hosts
Status: Agreed 2026-10-22

[HST-007] For a MiniMax model whose id names version 3.1 or later
(`MiniMax-M3.1`, `minimax-m4`), a reasoning effort MUST be sent as
`output_config.effort`, with `minimal` sent as `low`, and disabled reasoning
MUST be sent as effort `low` instead of disabled thinking. Other MiniMax
models MUST keep the Anthropic adapter's reasoning mapping.
Falsifier: A recorded MiniMax-M3.1 body for disabled reasoning has a `thinking` field or lacks `output_config.effort` of `low`; for effort `minimal` it has an effort other than `low`; for effort `high` it lacks `output_config.effort` of `high`; or a MiniMax-M2 body differs from what the Anthropic adapter sends for the same request.
Mechanism: hosts
Rationale: MiniMax 3.1 and later always reason (`minimax-chat-language-model.ts`).
Status: Agreed 2026-10-22
