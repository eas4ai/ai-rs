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

Tests combine related Python parameter cases into Rust table-driven tests. They do
not reproduce Python test counts. Python garbage-collector closure behavior,
Pydantic class construction, agent `Context`, and telemetry span tests are deferred
with their corresponding runtime components. Provider adapter-specific HTTP paths,
OIDC, SDK imports, models.dev catalog lookups, and vendor errors are deferred with
the adapters. No live provider credentials are needed by the Rust suite.

Regenerate wire fixtures with `scripts/generate_python_fixtures.py`, an explicit
path to the upstream checkout, and an isolated Python environment with Pydantic
2.13 or newer. The Rust tests read the committed fixture and do not need Python.

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
- `ui`: convert the core `Event`/`Message` shapes to the UI protocol. Use explicit
  event-message omission and hydration for persisted or transported streams.
- Telemetry and dedicated operation front ends: wrap the existing model/provider
  contracts. The base exposes typed dedicated-operation methods and unsupported
  defaults; this cut does not implement any provider's remote operation.
- Provider history repair, testing helpers, MCP, durable workflow integrations,
  vendor convenience factories, and environment-loading examples remain separate.

The deferred modules are absent. There are no placeholder implementations.
