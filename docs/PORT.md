# Inference core port

The source is AI SDK for Python 0.8.0 at
`https://github.com/vercel-labs/ai-python`. `LICENSE` is copied unchanged.
`NOTICE` records the port and upstream copyright.

## Module map

| Python module | Rust module | Responsibility |
| --- | --- | --- |
| `ai.__init__` | `ai` | Core API re-exports |
| `ai.types.events` | `types::events` | Event vocabulary, hydration, replay, run state |
| `ai.types.messages` | `types::messages` | Messages, parts, special tool results, IDs |
| `ai.types.builders` | `types::builders` | Message and part constructors |
| `ai.types.media` | `types::media` | Detection, media types, data URLs, base64 |
| `ai.types.usage` | `types::usage` | Token usage and addition |
| `ai.types.tools` | `types::tools` | Function/provider declarations and tool configuration |
| `ai.models.core.model` | `models`, `models::core::model` | Model references and parsing |
| `ai.models.core.api` | `models`, `models::core::api` | Streams, generation, probing |
| `ai.models.core.params` | `models::params`, `models::core::params` | Request parameters and sentinels |
| `ai.models.core.helpers.files` | `models::files`, `models::core::helpers::files` | Bounded public-network downloads |
| `ai.providers.base` | `providers::base` | Provider/protocol traits, configuration, registry |
| Operation payloads used by `providers.base` | `providers::operations` | Typed prompts, params, and result containers |
| `ai.providers._optional` | Cargo features | Optional components resolve at compile time |
| `ai.errors` | `errors` | Framework errors, provider categories, HTTP context |
| `ai.util` | `util`, `futures`, Tokio | Stream composition and task/resource ownership |
| Upstream SDK HTTP clients | `transport` | Application-supplied HTTP transport and SSE framing |
| `ai.ui.ai_sdk` | `ui::ai_sdk` | Inbound UI requests, persisted UI histories, approvals, identity metadata and outbound SSE |

## Wire format

Serialize the `Event` and `Part` enums for Python-compatible discriminators.
The Python 0.8.0 source uses `kind`, including `stream_start`, `tool_end`,
`builtin_tool_result`, and `file`. The brief called this field `type`; using
`type` would break the requested Python roundtrip. Nested special tool results
use their own Python `type` values, `content` and `messages`.

Messages retain IDs, turn IDs, replay state, usage, metadata, cached tool results,
and hooks. A tool result distinguishes an absent `model_input` from explicit
JSON null. File bytes serialize to base64 text, as Python does. Deserialization
retains that text because the wire format does not distinguish bytes from text.
Event replay flags remain runtime control state and do not serialize.
Partial-result events retain cloneable aggregator factories. Tool-result events
retain cloneable original errors and their source chains. Both runtime handles
stay out of serialized events. An object-safe JSON reducer facade adapts the
core's generic `Aggregator` trait for partial-result factories.
`ValueAggregator::snapshot_result` defaults to ordinary JSON. Reducers that
produce typed message bundles override it to retain the distinction between
special results and ordinary JSON with similar fields.

A yielded event contains a snapshot of its message at that point in the stream.
Rust owns snapshots. Python shares mutable message references. The final handle
contains the final aggregate. Usage and provider metadata follow the source's
latest-value rules. A stream that reaches EOF without `StreamEnd` returns a
retryable incomplete-response error and retains its partial message. Dropping a
handle cancels consumption without manufacturing an incomplete-response error.

## Rust differences

All inference runs asynchronously. There is no synchronous runtime wrapper.
A handle implements `futures::Stream`; callers drain it or drop it. There is no
Python async context manager or implicit task-local agent context.

`serde` replaces Pydantic serialization. `schemars` derives JSON Schema for typed
Rust inputs and outputs. `ToolDefinition` adds typed schema construction and
argument/result validation. The Python-shaped `Tool`, `ToolSpec`, and `ToolConfig`
carry function and provider declarations through inference. Tool execution and
approval decisions belong to the later agent module.

The validator supports `type`, `required`, `enum`, `const`, `properties`, `items`,
`additionalProperties`, local JSON Pointer references, schema combinations, numeric bounds,
length bounds, tuples, and unique array items. It rejects unsupported validation
keywords and remote references. Format strings remain annotations. Reference and
schema nesting are bounded at 128 levels. Numeric comparisons use decimal
arithmetic over the stored JSON numbers. Decimal representations are limited to
8192 bytes and 4096 exponent places to bound arithmetic. Named `$anchor` references
are unsupported. It does not claim full JSON Schema
conformance. SuprChat's lockfile had no `jsonschema` dependency, so this port uses
a local validator for the requested subset.

`ProviderValue<T>` preserves the difference between provider defaults and explicit
values, including disabled reasoning. Request options own their values. Provider
parameters use explicit string keys and JSON values instead of Python type keys.
Callers must use stable keys understood by their adapters.

Providers receive immutable configuration snapshots by shared reference. They do
not inspect environment variables. Applications supply the `env` map if they want
the Python configuration naming and precedence rules. There is no bundled
models.dev catalog. Applications register provider instances or construct them
from explicit models.dev metadata. Concrete adapter packages can supply catalog
integration later. Unknown or unregistered handles return `UnsupportedProvider`.

Trait objects use explicit snapshot restoration factories in `ProviderRegistry`.
`Model::to_value` and `Model::from_value` preserve the Python model/configuration
shape without a global deserialization registry. Custom providers and protocols
must include their settings in `to_value` and register corresponding factories.
Snapshots can contain API keys, like the Python configuration objects; store them
as credential material. Debug formatting of provider configuration redacts values.

Deterministic IDs use `with_id_generator` scoped to the polled future. Tokio does
not inherit task-local values into spawned tasks. Scope each spawned task
explicitly. Default IDs retain Python's prefix and 12-hex-digit format.

Pure media helpers accept bytes and text explicitly. Extension inference uses
`mime_guess`; platform-specific Python mimetypes
registrations are not available. Media downloads enforce public-address checks,
revalidate every redirect, pin destinations, limit bodies to the requested size,
and bound DNS resolution. A custom `DownloadTransport` must honor the documented
pinning and redirect contract. The optional implementation creates a fresh client
per hop, keeps the original host for HTTP/TLS, disables environment proxies, and
disables automatic redirects. A per-download cookie jar preserves domain, path, and secure-cookie scope across
redirects. Body size limits apply to decoded bytes yielded by the transport.
Custom download transports must decode supported content encodings before yielding
chunks. The bundled client supports gzip, deflate, Brotli, and Zstandard.

`util::merge` uses `futures::stream::select_all`. It does not restart exhausted
streams. Python queues, dynamic task groups, and exception-group teardown belong
to the later agent runtime. Rust `Drop` handles stream and resource cancellation.

## UI adapter

`ui::ai_sdk::to_messages` returns core history and extracted approval responses.
`apply_approvals` accepts an explicit `ApprovalResolver` supplied by the agent
runtime or application. `ToolResultValidator` supplies optional tool output
validation and coercion; the adapter does not depend on an ambient agent context
or Python runtime annotations.

`UIMessage` deserialization validates known part fields, accepts Python field
names and browser aliases, retains extra application fields, and skips unknown
part types. Parts use JSON values because tool and data type names are dynamic.
Core messages and events remain the crate's existing typed structures.

`to_ui_messages` collapses assistant, tool, and internal records into bubbles and
records the source IDs in `metadata.aiPython.sourceMessages`. Special content and
message-bundle results carry explicit `toolResultKinds` hints. The inbound path
uses those hints rather than inferring special results from arbitrary JSON.

`to_stream` emits typed UI events; `to_sse` encodes the same events as SSE and
appends `data: [DONE]` on successful exhaustion. Both consume fallible core event
streams lazily. A source error terminates the stream without a successful finish
or done marker. Applications set `UI_MESSAGE_STREAM_HEADERS` on the HTTP response.
The encoder preserves camelCase aliases, omitted top-level nulls, nested nulls,
Python JSON spacing, finite-float notation, ASCII escaping, and dataclass field
order. JSON values retain insertion order through Serde's `preserve_order` feature and use `float_roundtrip`
when parsing decimal numbers. Core `Metadata` retains its existing sorted map
representation.

An empty assistant `turn_id` uses the message ID and consumes the first record.
The Python implementation otherwise stalls on this input. Other turn grouping
rules follow the source.

## Test map

| Python tests | Rust integration tests | Coverage |
| --- | --- | --- |
| `tests/types/test_events.py` | `tests/types_events.rs` | Discriminators, omitted messages, hydration, multiple messages, retry, replay, run state |
| `tests/types/test_messages.py` | `tests/types_messages.rs` | Text/reasoning, media parts, output, special results, model-input presence, deterministic IDs |
| `tests/types/test_builders.py` | `tests/types_builders.rs` | Message parts, tool-message merges, constructors, errors |
| `tests/types/test_media.py` | `tests/types_media.rs` | Signature detection, URLs, inference, base64 and invalid data |
| `tests/types/test_tools.py` | `tests/types_tools.rs` | Function/provider shapes, opaque payloads, required fields; additional typed-schema validation |
| `tests/types/test_exclude_if_gc.py` | `tests/types_messages.rs`, `tests/types_events.rs` | Serialization omission rules; Python GC internals do not apply to Rust ownership |
| `tests/models/core/test_api.py` | `tests/models_core.rs` | Aggregation, option forwarding, generation fallback, protocol selection, replay, incomplete streams, probe |
| `tests/models/core/helpers/test_files.py` | `tests/models_files.rs`, `tests/transport_reqwest.rs` | Address policy, DNS/redirect validation, pinning, size/error handling, cleanup, original HTTP host |
| Base/configuration cases in `tests/models/test_resolution.py` and `tests/providers/*/test_provider.py` | `tests/models_core.rs`, `tests/providers_base.rs` | Explicit configuration, templates, aliases, duplicate registry entries, model resolution/restoration, unsupported operations |
| `ai.errors` contract | `tests/errors.rs` | Status mapping, retryability, context and error sources |
| Canonical Python wire fixtures | `tests/python_compat.rs`, `tests/fixtures/python-0.8.0.json` | Actual upstream serialization of all 21 events, 5 message roles, 3 tools, and usage |
| New transport boundary | `tests/transport.rs`, `tests/transport_reqwest.rs` | SSE chunk boundaries, limits, EOF, errors, HTTP request/response and redirects |
| `tests/ui/ai_sdk/test_inbound_messages.py` | `tests/ui_inbound_messages.rs` | Tool boundaries, approvals, special results, optional tool validation, UI-only parts and built-in tools |
| `tests/ui/ai_sdk/test_approvals.py` | `tests/ui_approvals.rs` | Approval extraction, hook identity, metadata and explicit registry resolution |
| `tests/ui/ai_sdk/test_outbound_messages.py` | `tests/ui_outbound_messages.rs` | Bubbles, tool deduplication, files, provider metadata and persisted identity roundtrips |
| `tests/ui/ai_sdk/test_outbound_stream.py` | `tests/ui_outbound_stream.rs` | Stream boundaries, retries, tools, typed partial results, hooks, files and SSE encoding |
| Captured Python UI outputs | `tests/ui_python_compat.rs`, `tests/fixtures/ui-python-0.8.0.json` | Requests, histories, event streams and exact JSON encoding from the unmodified Python tests |
| UI transport integration | `tests/ui_transport.rs`, `tests/ui_aggregator.rs` | Scripted `HttpTransport`, byte-split SSE, hydration, lazy conversion, error termination and typed reducer compatibility |

Tests combine related Python parameter cases into Rust table-driven tests. They do
not reproduce Python test counts. Python garbage-collector closure behavior,
Pydantic class construction, agent `Context`, and telemetry span tests are deferred
with their corresponding runtime components. Provider adapter-specific HTTP paths,
OIDC, SDK imports, models.dev catalog lookups, and vendor errors are deferred with
the adapters. No live provider credentials are needed by the Rust suite.

Regenerate wire fixtures with `scripts/generate_python_fixtures.py`, an explicit
path to the upstream checkout, and an isolated Python environment with Pydantic
2.13 or newer. The Rust tests read the committed fixture and do not need Python.

Regenerate UI fixtures with `scripts/generate_ui_fixtures.py UPSTREAM DESTINATION`
in an isolated environment containing the upstream runtime dependencies, pytest,
pytest-asyncio and async-solipsism. This runs all upstream `tests/ui` tests and
records their adapter calls. Runtime-only aggregator factories are checked by the
Rust streaming tests because they cannot be stored in JSON fixtures.

## Later modules

- `providers::openai`: implement `ProviderProtocol`, emit normalized events, use
  `HttpTransport` and `transport::sse`, map HTTP errors, list/probe without inference,
  and register handles plus restoration factories. Preserve provider tool config.
- `providers::anthropic`: use the same contracts. Preserve reasoning metadata,
  built-in tool events, multipart results, and provider stop-reason normalization.
- Gateway: implement the gateway wire protocol, authentication and catalog support.
  Model parsing and gateway aliases already exist in the core.
- `agents`: execute `Tool` declarations, validate arguments, maintain explicit
  context/history, dispatch hooks, and implement retry/replay policy. The core
  already carries result kinds, cached results, hook parts, agent events, replay,
  hydration, and pending-run state.
- Telemetry and dedicated operation front ends: wrap the existing model/provider
  contracts. The base exposes typed dedicated-operation methods and unsupported
  defaults; this cut does not implement any provider's remote operation.
- Provider history repair, testing helpers, MCP, durable workflow integrations,
  vendor convenience factories, and environment-loading examples remain separate.

The modules listed as deferred are absent. There are no placeholder implementations.
