Prefix: HST

# Hosted chat providers

Vendor APIs that speak a protocol the crate already implements: the OpenAI
Chat Completions protocol (Cerebras, DeepInfra, DeepSeek, Fireworks, Mistral,
Moonshot AI, Together AI and Z.ai) and the Anthropic Messages protocol
(MiniMax). Each one is the existing adapter with a host profile: its
endpoint, its key, and the request and response differences the AI SDK for
TypeScript 7.0.148 records for it.

| Host | Name and handles | Default base URL | Key variable |
|---|---|---|---|
| Cerebras | `cerebras`, `@ai-sdk/cerebras` | `https://api.cerebras.ai/v1` | `CEREBRAS_API_KEY` |
| DeepInfra | `deepinfra`, `@ai-sdk/deepinfra` | `https://api.deepinfra.com/v1/openai` | `DEEPINFRA_API_KEY` |
| Fireworks | `fireworks`, `@ai-sdk/fireworks` | `https://api.fireworks.ai/inference/v1` | `FIREWORKS_API_KEY` |
| Together AI | `togetherai`, `@ai-sdk/togetherai` | `https://api.together.xyz/v1` | `TOGETHER_API_KEY` |
| Z.ai | `zai`, `@ai-sdk/zai` | `https://api.z.ai/api/paas/v4` | `ZAI_API_KEY` |
| MiniMax | `minimax`, `@ai-sdk/minimax` | `https://api.minimax.io/anthropic` | `MINIMAX_API_KEY` |
| DeepSeek | `deepseek`, `@ai-sdk/deepseek` | `https://api.deepseek.com` | `DEEPSEEK_API_KEY` |
| Moonshot AI | `moonshotai`, `@ai-sdk/moonshotai` | `https://api.moonshot.ai/v1` | `MOONSHOT_API_KEY` |
| Mistral | `mistral`, `@ai-sdk/mistral` | `https://api.mistral.ai/v1` | `MISTRAL_API_KEY` |

[HST-001] For each host in the table, `providers::hosts::<name>::register`
MUST register, under the host's name and handles, a provider named after the
host. Its model calls MUST stream the host's protocol from the configured
base URL, or the default base URL when none is configured: Chat Completions
at `<base>/chat/completions` with `authorization: Bearer <key>`, and for
MiniMax, Messages at `<base>/v1/messages` with `x-api-key: <key>` and
`anthropic-version: 2023-06-01`. The key is the configuration's API key
value, or the `env` value named by the host's key variable. Registering
every host, OpenAI and Anthropic into one registry MUST succeed.
Falsifier: For any host, a recorded request goes to another URL, uses another method or credential header, or omits the key; the provider is missing under one of its handles or has another name; or registering every host in the table with `openai::register`, `openai::register_factories` and `anthropic::register` into one registry fails.
Mechanism: hosts
Status: Agreed 2026-10-22

[HST-002] A Chat Completions request MUST send `stream_options:
{"include_usage": true}` for OpenAI, for an OpenAI-compatible provider
without a host profile, for DeepSeek, Fireworks, Moonshot AI and Together
AI, and MUST NOT send `stream_options` for Cerebras, DeepInfra, Mistral or
Z.ai.
Falsifier: The recorded body for Cerebras, DeepInfra, Mistral or Z.ai has a `stream_options` field; or the body for OpenAI's Chat Completions protocol, for a provider built with `OpenAICompatibleProvider::from_modelsdev_provider` for an id without a host profile, for DeepSeek, Fireworks, Moonshot AI or Together AI lacks `stream_options.include_usage` set to true.
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

## DeepSeek, Moonshot AI and Mistral

These three hosts differ from Chat Completions in more than field names, so
their profiles also change how messages, schemas and stream content are
written and read.

[HST-008] A DeepSeek, Moonshot AI or Mistral call MUST fail with
`Error::Validation` before any request is sent when it sets a request
setting that the host's TypeScript provider never sends: a service tier,
metadata, a cache retention or a tool selection (`ToolChoice::Selection`)
for all three; a safety identifier for DeepSeek and Mistral; a cache key
for DeepSeek; and parallel tool calls for DeepSeek and Moonshot AI.
Falsifier: For each of the three hosts, a call that sets one of the settings named for it sends a request or fails with another error kind; or a Moonshot AI call with a cache key and a safety identifier does not send `prompt_cache_key` and `safety_identifier`, or a Mistral call with a cache key, one tool and parallel tool calls set to false does not send `prompt_cache_key` and `parallel_tool_calls: false`.
Mechanism: hosts
Rationale: The six earlier hosts still send these settings unchanged; a new host rejects what its API is not known to accept, as HST-004 does for Z.ai.
Status: Agreed 2026-10-22

A DeepSeek model is a legacy model when its id has `deepseek-chat`,
`deepseek-reasoner`, or `deepseek-v1` to `deepseek-v3` with an optional
`.<digits>` minor version, at its start or after a `/` or `.`, followed by
the end of the id or by `-`, `/`, `:` or `.`. Every other DeepSeek model is
a V4 model (`is-deepseek-v4-model.ts`).

[HST-009] A DeepSeek request MUST send the output token limit as
`max_tokens` and an assistant turn's reasoning as `reasoning_content`. A
reasoning effort MUST be sent as `thinking: {"type": "enabled"}` with
`reasoning_effort`, where `minimal` and `low` are sent as `low`, `medium`
and `high` as `high`, and `xhigh` and `max` as `max`; disabled reasoning
MUST be sent as `thinking: {"type": "disabled"}` with no
`reasoning_effort`. An output schema MUST be sent as `response_format`
`{"type": "json_object"}`, with a system message first in `messages` whose
text is `Return JSON that conforms to the following schema: ` followed by
the schema as compact JSON. A call with a frequency penalty, presence
penalty or seed MUST fail with `Error::Validation` before any request is
sent. So MUST a call with a temperature or top-p while thinking is on, which
it is unless reasoning is disabled, when an effort is set or the model is
`deepseek-reasoner` or a V4 model; the error message MUST say that
disabling reasoning allows the setting.
Falsifier: A recorded DeepSeek body for an output token limit of 100, an assistant turn with reasoning and effort `medium` has the limit or the reasoning under another field, lacks `thinking` of type `enabled`, or has a `reasoning_effort` other than `high`; effort `minimal` or `xhigh` sends an effort other than `low` or `max`; disabled reasoning lacks `thinking` of type `disabled` or sends `reasoning_effort`; an output schema sends another `response_format` or another first message; a call with a penalty or a seed, or with a temperature for `deepseek-v4-flash` without disabled reasoning or for `deepseek-reasoner`, sends a request or fails with another error kind or, for the temperature, with a message that does not mention reasoning; or a temperature of 0.2 for `deepseek-v4-flash` with disabled reasoning, or for `deepseek-chat` with no effort, is not sent as `temperature: 0.2`.
Mechanism: hosts
Rationale: The TypeScript SDK drops these settings with a warning; V4 models reason by default and ignore temperature and top-p while they do.
Status: Agreed 2026-10-22

[HST-010] For a DeepSeek V4 model, every assistant message MUST carry
`reasoning_content`, the empty string when the turn has no reasoning; for a
legacy model, the reasoning of the assistant messages before the last user
message MUST NOT be sent. A user file that is not an image, or an image
whose media type is not `image/gif`, `image/jpeg`, `image/jpg`, `image/png`
or `image/webp`, MUST fail with `Error::Validation` before any request is
sent. A tool result whose output is a list of content parts with no image
MUST be sent as a string: the JSON text of that list. DeepSeek usage MUST
report `prompt_cache_hit_tokens` as cache-read tokens, or
`prompt_tokens_details.cached_tokens` when it is absent, and the finish
reason `insufficient_system_resource` MUST map to `error`.
Falsifier: A recorded `deepseek-v4-pro` body for a history with an assistant turn without reasoning lacks `"reasoning_content": ""` on that turn; a `deepseek-chat` body sends the reasoning of an assistant turn before the last user message, or drops the reasoning of one after it; a user PDF or `image/bmp` file sends a request or fails with another error kind; a tool result of two text content parts is sent as an array; or a DeepSeek stream whose usage has `prompt_cache_hit_tokens` 60 and `prompt_tokens_details.cached_tokens` 80 reports cache-read tokens other than 60, one with only `cached_tokens` 80 reports other than 80, or one that ends with finish reason `insufficient_system_resource` reports another finish reason than `error`.
Mechanism: hosts
Rationale: V4 rejects a thinking-mode request whose assistant turns lack `reasoning_content` (`convert-to-deepseek-chat-messages.ts`).
Status: Agreed 2026-10-22

A Moonshot AI model's family comes from its id: `kimi-k3`; `kimi-k2.7` for
`kimi-k2.7-code` and `kimi-k2.7-code-highspeed`; `kimi-k2.6`; `kimi-k2.5`;
`moonshot-v1` for ids that start with `moonshot-v1-`; and unknown for every
other id. The first four are the Kimi families (`moonshotai-chat-options.ts`).

[HST-011] A Moonshot AI request MUST send the output token limit as
`max_completion_tokens` and an assistant turn's reasoning as
`reasoning_content`. Reasoning MUST follow the model's family: for
`kimi-k3` and unknown models, an effort is sent as `reasoning_effort`, with
`minimal` and `low` as `low`, `medium` and `high` as `high`, and `xhigh` and
`max` as `max`, and no `thinking`; for `kimi-k2.6` and `kimi-k2.5`, an
effort is sent as `thinking: {"type": "enabled"}` and disabled reasoning as
`thinking: {"type": "disabled"}`, with no `reasoning_effort`; for
`kimi-k2.7`, an effort sends neither field. A call MUST fail with
`Error::Validation` before any request is sent when it disables reasoning
for `kimi-k3`, `kimi-k2.7` or an unknown model; sets an effort for
`moonshot-v1`; sets a seed; sets a temperature, top-p, frequency penalty or
presence penalty for a Kimi family; or sets tool choice `required` for
`kimi-k2.6`, `kimi-k2.7-code` or `kimi-k2.7-code-highspeed`.
Falsifier: A recorded Moonshot AI body has the output token limit under a field other than `max_completion_tokens`; a `kimi-k3` body for effort `xhigh` lacks `reasoning_effort: "max"` or has `thinking`; a `kimi-k2.6` body for effort `high` lacks `thinking` of type `enabled` or has `reasoning_effort`; a `kimi-k2.5` body for disabled reasoning lacks `thinking` of type `disabled`; a `kimi-k2.7-code` body for effort `high` has `thinking` or `reasoning_effort`; a call that the requirement says fails sends a request or fails with another error kind; or a `moonshot-v1-8k` call with temperature 0.5 does not send `temperature: 0.5`.
Mechanism: hosts
Rationale: The TypeScript SDK drops each of these settings with a warning (`moonshotai-chat-language-model.ts`).
Status: Agreed 2026-10-22

[HST-012] Moonshot AI tool parameter schemas and output schemas MUST be
sent as given except for three changes: a root `$schema` is removed; in
every subschema, a tuple `items` array is moved to the end of
`prefixItems`; and a string `type` next to `anyOf` moves into each `anyOf`
branch that has no `type`. An output schema MUST be sent as
`response_format` of type `json_schema` with `strict: true`. A schema whose
root is not an object with `type: "object"` MUST fail with
`Error::Validation` before any request is sent.
Falsifier: A recorded Moonshot AI body for a tool and an output schema whose schemas each have a root `$schema`, an optional property and a property with a tuple `items` array keeps the `$schema` or the `items` array, adds the optional property to `required`, or adds `additionalProperties`; a property `{"type": "string", "anyOf": [{"minLength": 1}, {"type": "null"}]}` is not sent as `{"anyOf": [{"type": "string", "minLength": 1}, {"type": "null"}]}`; the output schema's `response_format` lacks `strict: true`; or a tool whose parameter schema has the root type `string` sends a request or fails with another error kind.
Mechanism: hosts
Rationale: Moonshot validates schemas as MFJS, not as OpenAI strict JSON schema (`normalize-json-schema-for-mfjs.ts`), so the crate's strict rewrite, which makes every property required, is not applied.
Status: Agreed 2026-10-22

[HST-013] A Moonshot AI user image of type `image/jpeg`, `image/png`,
`image/gif`, `image/webp`, `image/bmp`, `image/heic` or `image/heif` MUST be
sent as `image_url`, and a video of type `video/mp4`, `video/mpeg`,
`video/mov`, `video/avi`, `video/x-flv`, `video/mpg`, `video/webm`,
`video/wmv` or `video/3gpp` as `video_url`, each as a data URL. An image or
video given as an `http` or `https` URL MUST be downloaded and sent as a
data URL when the provider was built with a download transport, and MUST
fail with `Error::Validation` before any request is sent when it was not. A `text/*`
file MUST be sent as text. Any other user file, such as a PDF or audio,
MUST fail with `Error::Validation` before any request is sent. An assistant
turn with tool calls and no text MUST send `"content": null`, and a tool
result MUST be sent as a string: its text, or the JSON text of any other
output. Usage MUST be read from a chunk's top-level `usage`, or from
`choices[0].usage` when no chunk has a top-level one, and MUST report
`usage.cached_tokens` as cache-read tokens, or
`prompt_tokens_details.cached_tokens` when it is absent.
Falsifier: A recorded Moonshot AI body sends an inline `video/mp4` file as anything other than a `video_url` data URL; an `https` image URL is not fetched through the download transport and sent as a data URL, or without a download transport sends a request or fails with another error kind; a `text/plain` file is not sent as text; a PDF or `audio/mpeg` file sends a request or fails with another error kind; an assistant turn with only a tool call lacks `"content": null`; a tool result with JSON output is sent as an array or object; or a stream whose only usage is `choices[0].usage` with 12 prompt and 5 completion tokens reports other token counts, one that also has a top-level usage of 99 and 33, before or after it, reports other than 99 and 33, or a top-level `usage.cached_tokens` of 10 reports cache-read tokens other than 10.
Mechanism: hosts
Rationale: Moonshot does not fetch external URLs (`moonshotai-chat-language-model.ts` `supportedUrls`), so the TypeScript SDK downloads them first.
Status: Agreed 2026-10-22

The Mistral models without reasoning settings are `codestral-2508`,
`codestral-latest`, `ministral-14b-2512`, `ministral-14b-latest`,
`ministral-3b-2512`, `ministral-3b-latest`, `ministral-8b-2512`,
`ministral-8b-latest`, `mistral-code-fim-latest`, `mistral-code-latest`,
`mistral-large-latest`, `mistral-large-2512`, `voxtral-small-2507` and
`voxtral-small-latest` (`mistral-chat-language-model.ts`).

[HST-014] A Mistral request MUST send the output token limit as
`max_tokens` and a seed as `random_seed`. Disabled reasoning MUST be sent as
`reasoning_effort: "none"` and every effort as `reasoning_effort: "high"`,
except that for a model without reasoning settings, an effort or disabled
reasoning MUST fail with `Error::Validation` before any request is sent. An
output schema MUST be sent as `response_format` of type `json_schema` with
`strict: false` and the schema as given. Tool choice `required` MUST be sent
as `"any"`, and a named tool as `"any"` with `tools` holding only that tool;
a call with no tools MUST send neither `tool_choice` nor
`parallel_tool_calls`.
Falsifier: A recorded Mistral body for an output token limit of 100 and seed 7 has the limit under another field, has a `seed` field or lacks `random_seed: 7`; effort `low` or `max` sends a `reasoning_effort` other than `high`, or disabled reasoning one other than `none`; an effort or disabled reasoning for `mistral-large-latest` sends a request or fails with another error kind; an output schema with an optional property is sent with `strict` true, with that property in `required`, or with `additionalProperties` added; tool choice `required` sends a `tool_choice` other than `any`; a choice of tool `b` among tools `a` and `b` sends another `tool_choice` or a `tools` list other than `b` alone; or a call with no tools and tool choice `auto` sends `tool_choice` or `parallel_tool_calls`.
Mechanism: hosts
Rationale: Mistral reasons at one level, so the TypeScript SDK sends `high` for every effort (`mistral-chat-language-model.ts`).
Status: Agreed 2026-10-22

[HST-015] A Mistral assistant turn with reasoning MUST be sent with
`content` as an array that holds, in the turn's order, a `{"type":
"thinking", "thinking": [{"type": "text", "text": <reasoning>}], "closed":
true}` part for each reasoning part and a `{"type": "text", "text": <text>}`
part for each text part, and with no reasoning field; a turn with only tool
calls MUST send `"content": ""`. When the last message is an assistant
message, it alone MUST be sent with `"prefix": true`. A tool result MUST be
sent with the tool's `name` and a string `content`: its text, or the JSON
text of any other output. A user image MUST be sent as `{"type":
"image_url", "image_url": <URL>}` with the URL or data URL as a string, and
a PDF as `{"type": "document_url", "document_url": <URL>}`, holding an
`https` URL as given or the file as a data URL. Any other user file MUST
fail with `Error::Validation` before any request is sent.
Falsifier: A recorded Mistral body for a history with an assistant turn of reasoning `R` then text `T`, an assistant turn with only a tool call, a JSON tool result of tool `t` and a final assistant message differs from the stated shapes, sends `prefix` on another message or lacks it on the last; a user image, a PDF as bytes or a PDF as an `https` URL is sent in another shape; or an `audio/mpeg` or `text/plain` file sends a request or fails with another error kind.
Mechanism: hosts
Rationale: These are the shapes `convert-to-mistral-chat-messages.ts` sends; the TypeScript SDK returns the continued text after a prefix as the server sends it, and so does the crate.
Status: Agreed 2026-10-22

[HST-016] A Mistral stream MUST read `delta.content` given as a string or
as an array of parts: `text` parts as text, the text inside `thinking`
parts as reasoning, and other parts, such as `reference`, as nothing. A text
block MUST end before a reasoning block starts, and each run of reasoning
MUST be a new reasoning block. The finish reasons `model_length` and `error`
MUST map to `length` and `error`, and usage MUST report `num_cached_tokens`
as cache-read tokens when it is present.
Falsifier: A Mistral stream whose `content` arrays carry thinking `A`, text `B`, a `reference` part, thinking `C` and text `D` delivers anything other than reasoning `A`, text `B`, reasoning `C` and text `D` as four blocks in that order, each ended before the next starts; a string `content` is lost; finish reason `model_length` or `error` maps to another reason; or usage with `num_cached_tokens` 5 reports cache-read tokens other than 5.
Mechanism: hosts
Status: Agreed 2026-10-22

## Every Chat Completions provider

[HST-017] A Chat Completions stream MUST key each tool call delta by its
`index`, or, when it has none, by its position in the chunk's `tool_calls`
array, and a delta whose `id` differs from the id of the call under its key
MUST start a new call. A tool call `index` that is present but is not a
non-negative integer MUST end the call with an `Error::Provider` of kind
`Response`.
Falsifier: A stream whose first chunk opens two calls without an index, with ids `weather_0` and `time_1`, and whose next chunk carries argument deltas without an index or id for both, does not deliver two calls, `weather_0` with arguments `{"location":"San Francisco"}` and `time_1` with `{"zone":"UTC"}`; a stream with two chunks that each carry one complete call without an index, with ids `a` and `b`, does not deliver two calls; or a call index `"not-a-number"` does not end the call with a `Response` provider error.
Mechanism: hosts
Rationale: Mistral and Moonshot AI send tool calls without an index (`mistral-tool-call.chunks.txt`, `moonshotai-stream-indexless-tool-calls.chunks.txt`), and the crate merged them into one call.
Status: Agreed 2026-10-22

| Error `code` or `type` | Status |
|---|---|
| `rate_limit_exceeded`, `rate_limit_error` | 429 |
| `server_error`, `api_error`, `internal_server_error` | 500 |
| `overloaded_error`, `service_unavailable` | 503 |
| `timeout`, `timeout_error` | 504 |
| `authentication_error`, `invalid_api_key` | 401 |
| `permission_error` | 403 |
| `not_found_error`, `model_not_found` | 404 |
| `bad_request`, `context_length_exceeded`, `invalid_request_error` | 400 |

[HST-018] For DeepSeek, Moonshot AI and Mistral, a stream event that
carries an error, as an `error` object or, for Mistral, as an object whose
`object` is `error`, MUST end the call with an `Error::Provider` that keeps
the error's message, `code` and `type`. Its kind MUST come from the first
of these rules that applies: a `code` or `type` of `insufficient_quota`
gives `RateLimit`; a `code` that is an HTTP status from 400 to 599, as a
number or a three-digit string, gives that status's kind
(`ProviderErrorKind::from_status`); a `code`, else a `type`, in the table
above gives the kind of the table's status; otherwise the kind is
`Response`. The error MUST be retryable when its status is 408, 409, 429 or
500 and above, unless it is `insufficient_quota`.
Falsifier: A DeepSeek stream error `{"error": {"message": "m", "type": "rate_limit_error", "code": "rate_limit_exceeded"}}` does not end the call with an error whose message is `m`, code `rate_limit_exceeded`, type `rate_limit_error` and kind `RateLimit`, or it is not retryable; a code `insufficient_quota` gives another kind or is retryable; a code of 503 or `"503"` gives a kind other than `ServiceUnavailable`; a Moonshot AI error of type `server_error` gives a kind other than `InternalServer` or is not retryable; a Mistral event `{"object": "error", "message": "m", "type": "invalid_request_error", "code": "3230"}` is ignored, or gives a message other than `m` or a kind other than `BadRequest`; or an error of type `unknown_error` gives a kind other than `Response` or is retryable.
Mechanism: hosts
Rationale: The table and rules are `deepseek-chat-language-model.ts`'s; an error that arrives before the first event is retried under MOD-005 when it is retryable.
Status: Agreed 2026-10-22
