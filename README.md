# ai

Rust port of the inference core in AI SDK for Python 0.8.0.

The crate provides messages, events, tool schemas, stream aggregation, provider
contracts, streaming HTTP transport, the OpenAI-compatible and Anthropic provider
adapters (`providers::openai`, `providers::anthropic`), the agent loop with tools,
approval hooks and middleware (`agents`), and the AI SDK UI message stream adapter
(`ui::ai_sdk`). The provider registry starts empty; register the adapters you use.
The Vercel AI Gateway, OIDC, telemetry and the MCP client are not ported.

`ui::ai_sdk` converts Vercel AI SDK v6 UI requests into core messages and converts
core histories and event streams back to UI messages and SSE. It preserves the
Python 0.8.0 adapter protocol and message identity metadata.

```rust
use std::sync::Arc;
use ai::{Model, ProviderConfig, ProviderProtocol, StreamOptions, stream};
use ai::providers::{ConfiguredProvider, EventStream};
use ai::types::{Event, Message};
use ai::types::events::{StreamStart, StreamEnd, TextStart, TextDelta};
use futures::StreamExt;

struct ExampleProtocol;

#[async_trait::async_trait]
impl ProviderProtocol for ExampleProtocol {
    fn protocol_class_id(&self) -> &str { "example" }

    fn stream(
        &self,
        _model: &Model,
        _messages: Vec<Message>,
        _options: StreamOptions,
        _provider: &dyn ai::Provider,
    ) -> EventStream {
        futures::stream::iter([
            Ok(Event::StreamStart(StreamStart::default())),
            Ok(Event::TextStart(TextStart::default())),
            Ok(Event::TextDelta(TextDelta::new("Hello"))),
            Ok(Event::StreamEnd(StreamEnd::default())),
        ]).boxed()
    }
}

# async fn example() -> Result<(), ai::Error> {
let provider = Arc::new(ConfiguredProvider::new(
    "example", ProviderConfig::default(), Arc::new(ExampleProtocol), None,
));
let model = Model::new("example-model", provider);
let mut response = stream(&model, vec![ai::user_message("Hello")], StreamOptions::default());
while let Some(event) = response.next().await {
    event?;
}
assert_eq!(response.text(), "Hello");
# Ok(())
# }
```

Use `StreamOptions::with_output::<T>()` with a `schemars::JsonSchema` type to send
an output schema. Read the result with `response.output::<T>()` for a
`serde::Deserialize` type. `generate` and `experimental_generate` return an
assistant message. They use native provider generation when available and fall
back to streaming when that operation is unsupported.

Provider configuration is explicit. The library does not read process environment
variables. Model IDs use `provider:model`; IDs without a colon use the registered
`gateway` provider. The aliases `gateway`, `ai-gateway`, and `vercel` share a
registry entry.

The default feature set contains no HTTP client. Implement `HttpTransport` to
use an existing client, or enable the optional `reqwest` feature. Media downloads
use a separate `DownloadTransport` contract that pins validated DNS addresses.

Run `./scripts/check.sh` for the pinned toolchain build, test, lint, and documentation checks.

For UI requests, parse `UIMessage` with Serde and pass the core event stream to
`to_sse`. Set the response headers from `UI_MESSAGE_STREAM_HEADERS`. The SSE
adapter consumes events lazily and propagates source errors.

```rust
use ai::ui::ai_sdk::{UIMessage, to_messages, to_sse, UI_MESSAGE_STREAM_HEADERS};

# fn example(model: &ai::Model, request: Vec<UIMessage>) -> Result<(), ai::Error> {
let (messages, approvals) = to_messages(&request, None)?;
// Apply approvals through a run-scoped ApprovalResolver before polling the run.
let body = to_sse(ai::stream(model, messages, ai::StreamOptions::default()));
let headers = UI_MESSAGE_STREAM_HEADERS;
# let _ = (approvals, body, headers);
# Ok(())
# }
```

See [docs/PORT.md](docs/PORT.md) for compatibility, test coverage, and deferred work.
